#!/usr/bin/env python3
"""Compare the predeclared corpus mixture on development, then frozen test sets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_run(exe, *args):
    return json.loads(subprocess.check_output([str(exe), *map(str, args)]))


def development_passes(result, policy):
    gates = policy['acceptance']
    general = result['general']
    conversation = result['conversation']
    gain = conversation['candidate']['accuracy'][2]['top5'] - conversation['baseline']['accuracy'][2]['top5']
    loss = general['baseline']['accuracy'][2]['top5'] - general['candidate']['accuracy'][2]['top5']
    saving_gain = conversation['candidate']['selection_proxy']['savings_fraction'] - conversation['baseline']['selection_proxy']['savings_fraction']
    return {'conversation_top5_gain': gain,
            'general_top5_loss': loss,
            'selection_savings_gain': saving_gain,
            'quality_passed': gain >= gates['conversation_top5_prefix2_gain_min'] and loss <= gates['general_top5_prefix2_loss_max'] and saving_gain >= gates['conversation_selection_savings_gain_min'],
            'latency_target_met': all(v['candidate']['warm_p95_ms'] < gates['warm_p95_target_ms'] for v in result.values())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development-only', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    os.chdir(ROOT)
    exe = ROOT / 'target/release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
    prepared = ROOT / 'data/prepared'
    partitions = json_run(exe, 'prepare')
    policy = json.loads((ROOT / 'quality-policy.json').read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hardware = f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs'
    if platform.system() == 'Darwin':
        hardware += '; ' + subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    elif platform.system() == 'Linux':
        hardware += '; ' + next((x.split(':', 1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown CPU')
    models = {}
    for name in ['baseline', 'candidate']:
        path = prepared / (name + '.sqlite')
        if path.exists():
            path.unlink()
        json_run(exe, 'build', '--input', prepared / (name + '.txt'), '--output', path)
        models[name] = path
    report = {'protocol': partitions['protocol'], 'policy': policy, 'partitions': partitions,
              'algorithm': 'Unchanged fixed interpolation 0.1/0.3/0.6; personal learning disabled.',
              'limitations': 'Taskmaster is human-written simulated task dialogue, not actual AAC usage. General regression test was used in v1. Conversational test is kept separate from development. Selection savings are a modeled proxy, not observed switch use.',
              'development': {}}
    for suite in ['general', 'conversation']:
        report['development'][suite] = {}
        for model, path in models.items():
            report['development'][suite][model] = json_run(exe, 'score', '--baseline', path, '--input', prepared / f'{suite}-dev.txt', '--hardware', hardware)
    report['decision'] = development_passes(report['development'], policy)
    (out / 'development.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['decision'], indent=2), flush=True)
    if args.development_only:
        return
    if not report['decision']['quality_passed']:
        raise SystemExit('Development quality gate failed. Test set was not evaluated and candidate will not ship.')
    report['test'] = {}
    for suite in ['general', 'conversation']:
        report['test'][suite] = {}
        for model, path in models.items():
            report['test'][suite][model] = json_run(exe, 'score', '--baseline', path, '--input', prepared / f'{suite}-test.txt', '--hardware', hardware)
    (out / 'quality-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Frozen test evaluation complete; no tuning uses these results.', flush=True)


if __name__ == '__main__':
    main()
