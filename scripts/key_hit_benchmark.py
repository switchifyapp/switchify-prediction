#!/usr/bin/env python3
"""Build and aggregate the three pinned key-hit development benchmarks."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

from aac_experiment import sha
from quality import json_run

ROOT = Path(__file__).resolve().parent.parent
SUITES = {name: f'data/aac-oanc/prepared/{name}-dev.txt'
          for name in ['aac', 'general', 'conversation']}


def verify_inputs(root, production):
    paths = {}
    for name, relative in SUITES.items():
        path = root / relative
        if sha(path) != production['partitions'][name + '-dev']['sha256']:
            raise ValueError(f'Frozen development checksum mismatch: {relative}')
        paths[name] = path
    return paths


def check_result(result, expected_evaluation, expected_baseline):
    if result.get('protocol') != 'key-hits-v2':
        raise ValueError('Unexpected benchmark protocol or stale executable')
    if result.get('evaluation_sha256') != expected_evaluation:
        raise ValueError('Benchmark used the wrong evaluation file')
    if result.get('baseline', {}).get('logical_sha256') != expected_baseline:
        raise ValueError('Benchmark used the wrong production model')
    if [m['prefix_graphemes'] for m in result.get('metrics', [])] != [2, 3, 4]:
        raise ValueError('Missing benchmark prefix measurements')


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New empty output directory')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Choose an empty output directory')
    production = json.loads((ROOT / 'production-model.json').read_text(encoding='utf-8'))
    inputs = verify_inputs(ROOT, production)
    subprocess.run(['cargo', 'build', '--release', '--locked', '--example', 'key_hit_benchmark'],
                   cwd=ROOT, check=True)
    exe = ROOT / 'target/release/examples' / ('key_hit_benchmark.exe' if os.name == 'nt' else 'key_hit_benchmark')
    sources = ['src/key_hits.rs', 'examples/key_hit_benchmark.rs',
               'scripts/key_hit_benchmark.py', 'production-model.json']
    report = {
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
        'executable_sha256': sha(exe), 'baseline_file_sha256': sha(args.baseline.resolve()),
        'source_hashes': {name: sha(ROOT / name) for name in sources},
        'hardware': f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs',
        'python_version': platform.python_version(),
        'rust_version': subprocess.check_output(['rustc', '--version'], text=True).strip(),
        'complete': False, 'suites': {},
    }
    out.mkdir(parents=True, exist_ok=True)
    write_report(out / 'report.json', report)
    for name, path in inputs.items():
        print(f'Benchmarking {name}', flush=True)
        result = json_run(exe, args.baseline.resolve(), path)
        check_result(result, production['partitions'][name + '-dev']['sha256'], production['logical_sha256'])
        report['suites'][name] = result
        write_report(out / 'report.json', report)
    if sha(args.baseline.resolve()) != report['baseline_file_sha256']:
        raise ValueError('Production baseline changed during the benchmark')
    report['complete'] = True
    write_report(out / 'report.json', report)
    print(f'Complete: {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
