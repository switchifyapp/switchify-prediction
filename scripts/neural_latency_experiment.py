#!/usr/bin/env python3
"""Run frozen F32/Q8 context-cache experiments without changing production."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess

from aac_experiment import sha
from general_neural_experiment import ROOT, MANIFEST, fetch, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--only-batched', action='store_true', help='Run the batched follow-up modes only')
    parser.add_argument('--binary-dir', type=Path, default=ROOT / 'target/neural/release')
    parser.add_argument('--modes', nargs='+', choices=('baseline', 'f32', 'q8', 'f32-batched', 'q8-batched'))
    parser.add_argument('--build-label', default='portable-default')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    model = args.model.resolve()
    fetch(model)  # Verify all pinned source bytes before local conversion.
    suffix = '.exe' if os.name == 'nt' else ''
    binary = args.binary_dir.resolve() / ('switchify-neural-experiment' + suffix)
    converter = binary.with_name('quantize' + suffix)
    env = dict(os.environ, RAYON_NUM_THREADS='4')
    for kind in ('f32', 'q8'):
        subprocess.run([str(converter), str(model), str(out / f'{kind}.gguf'), kind], env=env, check=True)
    common = [str(binary), '--baseline', str(args.baseline.resolve()), '--model', str(model),
              '--manifest', str(MANIFEST), '--fixtures', str(ROOT / 'experiments/neural/fixtures.json'),
              '--training', str(ROOT / 'data/aac-oanc/prepared/candidate.txt')]
    # F32 conversion must reproduce original logits before any quantized score is trusted.
    control = out / 'f32.gguf'
    subprocess.run(common + ['--mode', 'neural', '--gguf', str(control), '--gguf-sha256', sha(control),
                            '--validate-conversion', '--per-cell', '1', '--output', str(out / 'parity.json')],
                   env=env, check=True)
    result = {'protocol': 'neural-latency-v1', 'hardware': platform.platform(),
              'logical_cpus': os.cpu_count(), 'rayon_threads': 4,
              'executable_sha256': sha(binary), 'converter_sha256': sha(converter),
              'build_label': args.build_label,
              'reports': {}, 'promote': False,
              'cache_workload': 'Each frozen query starts with no cached context and is then repeated once. Only context KV/logits are cached, never candidate scores. This is a best-case cache-hit probe, not a typing trace.',
              'memory_method': 'Windows peak process working set, including loading; unavailable on other platforms.'}
    modes = args.modes or (('f32-batched', 'q8-batched') if args.only_batched else ('baseline', 'f32', 'q8', 'f32-batched', 'q8-batched'))
    for mode in modes:
        dest = out / f'{mode}.json'
        command = common + ['--mode', 'baseline' if mode == 'baseline' else 'neural', '--output', str(dest)]
        if mode != 'baseline':
            command += ['--cache-probe']
        if mode.endswith('-batched'):
            command += ['--batch-candidates']
        if mode.startswith('q8'):
            quantized = out / 'q8.gguf'
            command += ['--gguf', str(quantized), '--gguf-sha256', sha(quantized)]
        peak = run(command, env)
        report = json.loads(dest.read_text(encoding='utf-8'))
        report['peak_process_working_set_bytes'] = peak
        result['reports'][mode] = report
    (out / 'results.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())


if __name__ == '__main__':
    main()
