#!/usr/bin/env python3
"""Frozen local data experiment; never modifies the production model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from quality import json_run
from aac_experiment import sha

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'taskmaster2-source-manifest.json'
POLICY = {
    'version': 1,
    'speaker': 'USER',
    'min_words': 2,
    'max_words': 30,
    'max_sentences_per_domain': 3000,
    'weights': [1, 3],
    'selection_prefixes': [0, 1, 2],
    'max_domain_top5_loss': 0.01,
    'warm_p95_target_ms': 20,
    'selection': 'Highest mean top5 over three domains and prefixes 0,1,2 among candidates with positive mean gain, no per-domain/prefix loss over 1 percentage point, and all dev p95 below 20 ms. Tie: lower weight. If none qualify, evaluate the highest-mean candidate as diagnostic only. Never promote automatically.',
}
DOMAINS = ('aac', 'general', 'conversation')


def write_json(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def write_lines(path, lines):
    path.write_bytes(('\n'.join(lines) + '\n').encode())


def user_turns(dialogues):
    return [u['text'] for d in dialogues for u in d['utterances'] if u['speaker'] == 'USER']


def comm2_text(text):
    result = []
    ids = set()
    for line in text.splitlines():
        identifier, separator, sentence = line.partition('\t')
        if not separator or not identifier.startswith('comm2_') or identifier in ids or not sentence.strip():
            raise ValueError('Invalid or duplicate COMM2 record')
        ids.add(identifier)
        result.append(sentence)
    if not result:
        raise ValueError('Empty COMM2 data')
    return result


def partition(current, existing, comm2, domains, policy=POLICY):
    training = set(current)
    held_out = set().union(*existing.values())
    if training & held_out:
        raise ValueError('Existing training overlaps evaluation')
    # Reserve all COMM2 sentences before selecting any added training data.
    reserved = training | held_out | comm2
    selected, stats = [], {}
    for name, sentences in sorted(domains.items()):
        eligible = {s for s in sentences if policy['min_words'] <= len(s.split()) <= policy['max_words']}
        remaining = eligible - reserved
        chosen = sorted(remaining, key=lambda s: (hashlib.sha256(s.encode()).hexdigest(), s))[:policy['max_sentences_per_domain']]
        stats[name] = {'normalized_unique': len(sentences), 'eligible': len(eligible),
                       'excluded_reserved': len(eligible & reserved), 'selected': len(chosen)}
        selected.extend(chosen)
        reserved.update(chosen)
    fresh = sorted(comm2 - training - held_out)
    if not selected or not fresh:
        raise ValueError('Empty training supplement or COMM2 evaluation')
    stats['comm2'] = {'normalized_unique': len(comm2), 'overlap_training': len(comm2 & training),
                      'overlap_existing_eval': len(comm2 & held_out), 'retained': len(fresh)}
    return sorted(selected), fresh, stats


def select(development, policy=POLICY):
    candidates = []
    for weight in policy['weights']:
        name = f'tm2-{weight}x'
        gains = [development[d][name]['accuracy'][p]['top5'] - development[d]['baseline']['accuracy'][p]['top5']
                 for d in DOMAINS for p in policy['selection_prefixes']]
        mean = sum(gains) / len(gains)
        eligible = mean > 0 and min(gains) >= -policy['max_domain_top5_loss'] and all(
            development[d][name]['warm_p95_ms'] < policy['warm_p95_target_ms'] for d in DOMAINS)
        candidates.append({'name': name, 'weight': weight, 'mean_top5_gain': mean,
                           'worst_top5_gain': min(gains), 'eligible': eligible})
    pool = [c for c in candidates if c['eligible']] or candidates
    winner = max(pool, key=lambda c: (c['mean_top5_gain'], -c['weight']))
    return {'selected': winner['name'], 'diagnostic_only': not winner['eligible'], 'candidates': candidates}


def fetch(out):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for name, entry in manifest['files'].items():
        path = out / name
        if not path.exists():
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                downloaded = Path(tmp) / 'download'
                subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '600', '-o', str(downloaded), entry['url']], check=True)
                if sha(downloaded) != entry['sha256']:
                    raise ValueError(f'Checksum mismatch: {name}')
                downloaded.replace(path)
        if sha(path) != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path, help='Released en-aac-oanc-v1 database')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/taskmaster2')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'results.json').exists() or any(out.glob('tm2-*x.sqlite')):
        raise ValueError('Output contains a previous run; use a fresh output directory')
    exe = ROOT / 'target/release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
    production = json.loads((ROOT / 'production-model.json').read_text())
    baseline = args.baseline.resolve()
    expected_baseline = '222253417d0a7a705823ffb7e599a3bcf5d5d3daf4a9d76161ac6b3e555aeaad'
    if sha(baseline) != expected_baseline:
        raise ValueError('Baseline does not match released database')
    prepared = ROOT / 'data/aac-oanc/prepared'
    current_path = prepared / 'candidate.txt'
    if sha(current_path) != production['training_sha256']:
        raise ValueError('Production training checksum mismatch; run aac_experiment.py --prepare-only')
    current = current_path.read_text(encoding='utf-8').splitlines()
    existing = {}
    for domain in DOMAINS:
        for split in ('dev', 'test'):
            name = f'{domain}-{split}'
            path = prepared / f'{name}.txt'
            if sha(path) != production['partitions'][name]['sha256']:
                raise ValueError(f'Evaluation checksum mismatch: {name}')
            existing[name] = set(path.read_text(encoding='utf-8').splitlines())
    manifest = fetch(out)
    write_json(out / 'frozen-protocol.json', POLICY)

    def normalize(name, lines):
        raw, normalized = out / f'{name}-raw.txt', out / f'{name}-normalized.txt'
        write_lines(raw, lines)
        subprocess.run([str(exe), 'normalize', '--input', str(raw), '--output', str(normalized)], check=True, stdout=subprocess.DEVNULL)
        return set(normalized.read_text(encoding='utf-8').splitlines())

    comm2 = normalize('comm2', comm2_text((out / 'comm2.txt').read_text(encoding='utf-8')))
    domains = {name[:-5]: normalize(name[:-5], user_turns(json.loads((out / name).read_text(encoding='utf-8'))))
               for name in manifest['files'] if name.endswith('.json')}
    selected, fresh, stats = partition(current, existing, comm2, domains)
    write_lines(out / 'comm2-eval.txt', fresh)
    write_lines(out / 'tm2-selected.txt', selected)
    for weight in POLICY['weights']:
        write_lines(out / f'tm2-{weight}x.txt', current + selected * weight)
    report = {'protocol': POLICY, 'source_manifest': manifest, 'production_training_sha256': sha(current_path),
              'predictor_executable_sha256': sha(exe),
              'baseline_sha256': expected_baseline, 'partitions': stats,
              'prepared_sha256': {p.name: sha(p) for p in [out / 'comm2-eval.txt', out / 'tm2-selected.txt'] + [out / f'tm2-{w}x.txt' for w in POLICY['weights']]},
              'hardware': f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs',
              'development': {}, 'test': {}}
    write_json(out / 'preparation.json', report)
    print(json.dumps(stats), flush=True)
    if args.prepare_only:
        return
    models = {'baseline': baseline}
    for weight in POLICY['weights']:
        name = f'tm2-{weight}x'
        models[name] = out / f'{name}.sqlite'
        json_run(exe, 'build', '--input', out / f'{name}.txt', '--output', models[name])
    for domain in DOMAINS:
        report['development'][domain] = {}
        for name, db in models.items():
            print(f'Scoring development {domain} {name}', flush=True)
            report['development'][domain][name] = json_run(exe, 'score', '--baseline', db, '--input', prepared / f'{domain}-dev.txt', '--hardware', report['hardware'])
            write_json(out / 'results.json', report)
    report['decision'] = select(report['development'])
    # Persist the selection before evaluating any test target.
    write_json(out / 'development-decision.json', report['decision'])
    print(json.dumps(report['decision']), flush=True)
    for domain in (*DOMAINS, 'comm2'):
        report['test'][domain] = {}
        path = out / 'comm2-eval.txt' if domain == 'comm2' else prepared / f'{domain}-test.txt'
        for name in ('baseline', report['decision']['selected']):
            print(f'Scoring test {domain} {name}', flush=True)
            report['test'][domain][name] = json_run(exe, 'score', '--baseline', models[name], '--input', path, '--hardware', report['hardware'])
            write_json(out / 'results.json', report)
    if sha(baseline) != expected_baseline or sha(current_path) != production['training_sha256']:
        raise ValueError('Baseline or training source changed during experiment')
    print('Complete. No production files changed.', flush=True)


if __name__ == '__main__':
    main()
