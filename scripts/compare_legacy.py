#!/usr/bin/env python3
"""Run frozen slot-fill comparison. Never distributes or trains on the legacy data."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

LEGACY_SHA = 'dedd65d263bde8315e7e5ed7d2c8e04f17c33598a68506c9d70661a6d1f57318'
BASELINE_SHA = '222253417d0a7a705823ffb7e599a3bcf5d5d3daf4a9d76161ac6b3e555aeaad'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def peak_rss(pid):
    if platform.system() == 'Windows':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [(x, ctypes.c_size_t) for x in ['peak', 'working', 'a', 'b', 'c', 'd', 'e', 'f']]
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        psapi = ctypes.WinDLL('psapi')
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        handle = kernel.OpenProcess(0x410, False, pid)
        if not handle:
            return None
        try:
            data = Counters(); data.cb = ctypes.sizeof(data)
            if psapi.GetProcessMemoryInfo(handle, ctypes.byref(data), data.cb):
                return data.peak
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            return int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)]).strip()) * 1024
        except (ValueError, subprocess.CalledProcessError):
            return None
    return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--partitions', type=Path, default=Path('data/aac-oanc/prepared'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sha(args.legacy) != LEGACY_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError('Comparison requires pinned legacy and released baseline databases')
    expected = json.loads(Path('production-quality.json').read_text())
    reports = []
    for split in ['dev', 'test']:
        for domain in ['aac', 'general', 'conversation']:
            path = args.partitions / f'{domain}-{split}.txt'
            if sha(path) != expected[split][domain]['candidate']['evaluation_sha256']:
                raise ValueError(f'Partition mismatch: {path}')
            batch = {}
            for mode in ['original', 'newer', 'combined']:
                command = [str(args.exe.resolve()), 'compare-legacy', '--mode', mode, '--baseline', str(args.baseline), '--legacy', str(args.legacy), '--input', str(path)]
                # Output is small; communicate drains the pipe while polling memory.
                import tempfile
                with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                    process = subprocess.Popen(command, stdout=out, stderr=err)
                    peak = None
                    while process.poll() is None:
                        sample = peak_rss(process.pid)
                        if sample is not None: peak = max(peak or 0, sample)
                        time.sleep(0.05)
                    out.seek(0); err.seek(0)
                    if process.returncode: raise RuntimeError(err.read().decode())
                    report = json.loads(out.read())
                report.update(domain=domain, split=split, peak_process_rss_bytes=peak, memory_method='Windows process peak working set' if platform.system() == 'Windows' else 'sampled RSS; lower bound', hardware=platform.platform() + ' ' + platform.processor())
                batch[mode] = report; reports.append(report)
                print(domain, split, mode, report['warm_p95_ms'], flush=True)
            if batch['combined']['position_violations']:
                raise ValueError('Modern positions changed')
            for new, combined in zip(batch['newer']['accuracy'], batch['combined']['accuracy']):
                if combined['top5'] < new['top5']: raise ValueError('Top-five accuracy regressed')
    if sha(args.legacy) != LEGACY_SHA or sha(args.baseline) != BASELINE_SHA:
        raise ValueError('Source database changed')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'protocol': 'legacy-slot-fill-v1', 'reports': reports}, indent=2) + '\n')

if __name__ == '__main__':
    main()
