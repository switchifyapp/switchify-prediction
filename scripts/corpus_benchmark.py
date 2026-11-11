#!/usr/bin/env python3
"""Compare existing training sources without changing the prediction engine."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from aac_experiment import sha
from quality import json_run

ROOT = Path(__file__).resolve().parent.parent


def ranked(lines):
    return sorted(set(lines), key=lambda s: (hashlib.sha256(s.encode()).hexdigest(), s))


def decompose(world, mixed, world_weight):
    """Recover Taskmaster including sentences shared with the general source."""
    counts = Counter(mixed)
    counts.subtract(Counter({s: world_weight for s in world}))
    if any(n < 0 or n > 1 for n in counts.values()):
        raise ValueError('Prepared mixture does not match the declared source weights')
    return {s for s, n in counts.items() if n == 1}


def mix(sources, weights):
    return [s for name, weight in weights.items()
            for _ in range(weight) for s in sorted(sources[name])]


def recipes(sources, full_oanc, policy):
    weights = policy['production_weights']
    result = {'production': mix(sources, weights)}
    result.update({name: sorted(sources[name]) for name in policy['sources']})
    for omitted in policy['sources']:
        result['without-' + omitted] = mix(sources, {n: w for n, w in weights.items() if n != omitted})
    result['full-oanc-blend'] = mix(dict(sources, oanc=full_oanc), weights)
    if list(result) != policy['models']:
        raise ValueError('Model list does not match frozen policy')
    return result


def reject_leakage(training, evaluations):
    reserved = set().union(*evaluations.values())
    for name, sentences in training.items():
        if set(sentences) & reserved:
            raise ValueError(f'Training/evaluation overlap: {name}')
    values = list(evaluations.values())
    if any(a & b for i, a in enumerate(values) for b in values[i + 1:]):
        raise ValueError('Evaluation partitions overlap')


def verify_prepared(prepared, info, production, evaluation_names):
    for name, item in info['files'].items():
        if sha(prepared / (name + '.txt')) != item['sha256']:
            raise ValueError(f'Prepared input checksum mismatch: {name}')
    # Local preparation metadata is not the authority for frozen evaluation.
    for name in evaluation_names:
        if sha(prepared / (name + '.txt')) != production['partitions'][name]['sha256']:
            raise ValueError(f'Frozen evaluation checksum mismatch: {name}')


def prepare(policy, work):
    """Use checksum-pinned prepared inputs; never rewrite production preparation."""
    prepared = ROOT / 'data/aac-oanc/prepared'
    production = json.loads((ROOT / 'production-model.json').read_text())
    info = json.loads((prepared / 'partitions.json').read_text())
    read = lambda name: (prepared / name).read_text(encoding='utf-8').splitlines()
    verify_prepared(prepared, info, production,
                    [f'{s}-{p}' for s in policy['suites'] for p in ['dev', 'test']])
    if sha(prepared / 'candidate.txt') != production['training_sha256']:
        raise ValueError('Prepared training is not the production corpus')
    world_path = ROOT / 'data/prepared/baseline.txt'
    source_manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    if sha(world_path) != source_manifest['prepared']['files']['baseline.txt']['sha256']:
        raise ValueError('General source checksum mismatch')
    world = set(world_path.read_text(encoding='utf-8').splitlines())
    current = read('baseline.txt')
    taskmaster = decompose(world, current, policy['production_weights']['tatoeba'])
    evaluations = {f'{s}-{p}': set(read(f'{s}-{p}.txt'))
                   for s in policy['suites'] for p in ['dev', 'test']}
    existing = {name: lines for name, lines in evaluations.items() if not name.startswith('aac-')}
    aac = {p: set(read(f'normalized-aac-{p}.txt')) for p in ['train', 'dev', 'test']}
    oanc = set(read('normalized-spoken.txt'))
    aac_train = aac['train'] - set().union(*existing.values()) - aac['dev'] - aac['test']
    eligible = ranked(s for s in oanc - set(current) - aac_train
                      - set().union(*existing.values()) - aac['dev'] - aac['test']
                      if policy['oanc_min_words'] <= len(s.split()) <= policy['oanc_max_words'])
    sources = {'tatoeba': world, 'taskmaster': taskmaster, 'aac': aac_train,
               'oanc': set(eligible[:policy['oanc_sample_sentences']])}
    training = recipes(sources, set(eligible), policy)
    if Counter(training['production']) != Counter(read('candidate.txt')):
        raise ValueError('Source decomposition does not reconstruct production counts')
    reject_leakage(training, evaluations)
    if any(not values for values in training.values()):
        raise ValueError('Empty training corpus')
    work.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, values in training.items():
        path = work / (name + '.txt')
        path.write_bytes(('\n'.join(values) + '\n').encode())
        files[name] = {'sha256': sha(path), 'lines': len(values),
                       'unique_lines': len(set(values)), 'tokens': sum(len(s.split()) for s in values)}
    for name, values in evaluations.items():
        path = work / (name + '.txt')
        path.write_bytes(('\n'.join(sorted(values)) + '\n').encode())
        files[name] = {'sha256': sha(path), 'lines': len(values)}
    inputs = [prepared / (n + '.txt') for n in info['files']]
    inputs += [prepared / f'normalized-aac-{p}.txt' for p in ['train', 'dev', 'test']]
    inputs += [prepared / 'normalized-spoken.txt', world_path,
               ROOT / 'source-manifest.json', ROOT / 'aac-source-manifest.json',
               ROOT / 'production-model.json', ROOT / 'corpus-benchmark-policy.json',
               Path(__file__), ROOT / 'scripts/quality.py']
    return {'source_unique_lines': {k: len(v) for k, v in sources.items()},
            'full_oanc_unique_lines': len(eligible), 'files': files,
            'input_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            'production_counts_reconstructed': True, 'training_evaluation_overlap': 0}


def write_report(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/corpus-benchmark')
    parser.add_argument('--partition', choices=['dev', 'test'], default='dev')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output must be empty; choose a fresh directory for each run')
    out.mkdir(parents=True, exist_ok=True)
    policy = json.loads((ROOT / 'corpus-benchmark-policy.json').read_text())
    prepared = prepare(policy, out / 'prepared')
    exe = ROOT / 'target/release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
    report = {'policy': policy, 'prepared': prepared, 'partition': args.partition,
              'algorithm': 'Unchanged unigram/bigram/trigram interpolation 0.1/0.3/0.6; personal learning disabled.',
              'limitations': 'Unequal corpus sizes. Isolated sources measure standalone behavior; ablations measure contribution at current weights, including changes in relative source proportions. Three suites are known regression sets, not unseen users. General and Taskmaster suites resemble their respective training domains. AAC is imagined communication. Selection savings are modeled, not observed switch use. No automatic winner or production promotion.',
              'models': {}, 'complete': False}
    write_report(out / 'report.json', report)
    if args.prepare_only:
        print(json.dumps(prepared, indent=2))
        return
    report['executable_sha256'] = sha(exe)
    report['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    report['git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    report['hardware'] = f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs'
    (out / 'models').mkdir()
    for name in policy['models']:
        print(f'Building {name}', flush=True)
        model = out / 'models' / (name + '.ngrams')
        built = json_run(exe, 'build', '--input', out / 'prepared' / (name + '.txt'),
                         '--output', model, '--format', 'ngrams')
        validated = json_run(exe, 'validate', '--database', model)
        if built != validated:
            raise ValueError(f'Model validation differs: {name}')
        if name == 'production':
            production = json.loads((ROOT / 'production-model.json').read_text())
            if built['logical_sha256'] != production['logical_sha256']:
                raise ValueError('Rebuilt production model differs from shipped model')
        result = {'build': built, 'sha256': sha(model), 'bytes': model.stat().st_size, 'scores': {}}
        report['models'][name] = result
        for suite in policy['suites']:
            print(f'Scoring {name}/{suite}/{args.partition}', flush=True)
            result['scores'][suite] = json_run(exe, 'score', '--baseline', model, '--input',
                                               out / 'prepared' / f'{suite}-{args.partition}.txt',
                                               '--hardware', report['hardware'])
            write_report(out / 'report.json', report)
    report['complete'] = True
    for name in ['corpus-benchmark-policy.json', 'source-manifest.json', 'aac-source-manifest.json', 'ATTRIBUTION.md']:
        shutil.copyfile(ROOT / name, out / name)
    shutil.copytree(ROOT / 'docs/corpus-notices', out / 'corpus-notices')
    write_report(out / 'report.json', report)
    print(f'Complete: {out / "report.json"}', flush=True)


if __name__ == '__main__':
    main()
