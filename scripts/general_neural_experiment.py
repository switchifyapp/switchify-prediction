#!/usr/bin/env python3
"""Fetch pinned model files, then benchmark two offline Rust processes."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import time

from aac_experiment import sha

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'general-neural-source-manifest.json'


def fetch(directory):
    directory.mkdir(parents=True, exist_ok=True)
    sources = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for name, source in sources['files'].items():
        path = directory / name
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                downloaded = Path(temporary) / 'source'
                subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '600',
                                '-o', str(downloaded), source['url']], check=True)
                if sha(downloaded) != source['sha256']:
                    raise ValueError(f'Corrupt download: {name}')
                downloaded.replace(path)
        if sha(path) != source['sha256']:
            raise ValueError(f'Corrupt cached source: {name}')


def windows_peak(process):
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('peak_working_set', 'working_set',
             'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    function = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    function.restype = ctypes.c_int
    if not function(int(process._handle), ctypes.byref(counters), counters.cb):
        return None
    return counters.peak_working_set


def run(command, env):
    peak = None
    process = subprocess.Popen(command, env=env, cwd=ROOT)
    while process.poll() is None:
        if os.name == 'nt':
            current = windows_peak(process)
            if current is not None:
                peak = max(peak or 0, current)
        time.sleep(0.05)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return peak


def decision(reports):
    gains = []
    for domain, cells in reports['baseline']['cells'].items():
        for prefix in ('0', '1', '2'):
            base, neural = cells[prefix], reports['neural']['cells'][domain][prefix]
            if base['queries'] != neural['queries']:
                raise ValueError('Mismatched query counts')
            gains.append((neural['top5_hits'] - base['top5_hits']) / base['queries'])
    mean = sum(gains) / len(gains)
    return {'mean_early_top5_gain': mean, 'worst_early_cell_gain': min(gains),
            'accuracy_passed': mean > 0 and min(gains) >= -0.01,
            'latency_passed': reports['neural']['warm_p95_ms'] < 20,
            'promote': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/general-neural')
    parser.add_argument('--fetch-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if any((out / f'{name}.json').exists() for name in ('baseline', 'neural', 'results')):
        raise ValueError('Previous results exist; use a fresh output directory')
    fetch(out / 'model')
    if args.fetch_only:
        return
    exe = ROOT / 'target/neural/release' / ('switchify-neural-experiment.exe' if os.name == 'nt' else 'switchify-neural-experiment')
    env = dict(os.environ, RAYON_NUM_THREADS='4')
    report = {'protocol': 'general-neural-v1', 'hardware': f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs',
              'rayon_threads': 4, 'dtype': 'F32', 'device': 'CPU', 'executable_sha256': sha(exe),
              'fixture_sha256': sha(ROOT / 'experiments/neural/fixtures.json'),
              'reports': {}, 'memory_method': 'Windows peak working set sampled from OS high-water counter; unavailable on other platforms. Includes loading and scoring, not model payload alone.'}
    for mode in ('baseline', 'neural'):
        dest = out / f'{mode}.json'
        command = [str(exe), '--baseline', str(args.baseline.resolve()), '--model', str(out / 'model'),
                   '--manifest', str(MANIFEST), '--fixtures', str(ROOT / 'experiments/neural/fixtures.json'),
                   '--training', str(ROOT / 'data/aac-oanc/prepared/candidate.txt'),
                   '--mode', mode, '--output', str(dest)]
        peak = run(command, env)
        result = json.loads(dest.read_text(encoding='utf-8'))
        result['peak_process_working_set_bytes'] = peak
        report['reports'][mode] = result
    report['decision'] = decision(report['reports'])
    (out / 'results.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report['decision'], indent=2))


if __name__ == '__main__':
    main()
