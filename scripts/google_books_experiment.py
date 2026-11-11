#!/usr/bin/env python3
"""A fixed, bounded Google Books count import; never promotes a model."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import subprocess
import urllib.request
import zipfile

from aac_experiment import sha
from corpus_benchmark import verify_prepared, write_report
from quality import json_run

ROOT = Path(__file__).resolve().parent.parent
MAX_COUNT = 922337203685477580
WORD = re.compile(r"[a-z]+(?:'[a-z]+)*\Z")


def normalize_ngram(raw, order, max_length):
    try:
        words = raw.decode('utf-8').replace('\u2019', "'").lower().split(' ')
    except UnicodeDecodeError:
        return None
    if len(words) != order or any(len(w) > max_length or not WORD.fullmatch(w) for w in words):
        return None
    return tuple(words)


def aggregate_rows(rows, order, policy, vocabulary=None):
    counts = Counter()
    stats = {'rows': 0, 'year_rows': 0, 'accepted_groups': 0}
    low = str(policy['publication_year_min']).encode()
    high = str(policy['publication_year_max']).encode()
    previous = None
    total = 0

    def emit(raw, count):
        if not count:
            return
        gram = normalize_ngram(raw, order, policy['max_word_characters'])
        if gram is not None and (vocabulary is None or all(w in vocabulary for w in gram)):
            counts[gram] += count
            stats['accepted_groups'] += 1

    for row in rows:
        stats['rows'] += 1
        fields = row.rstrip(b'\r\n').split(b'\t')
        if len(fields) != 5:
            raise ValueError('Expected five columns in Google Books version 1')
        raw, year, matches, _pages, _volumes = fields
        if not year.isdigit():
            raise ValueError('Invalid Google publication year')
        if raw != previous:
            emit(previous, total)
            previous, total = raw, 0
        if len(year) == 4 and low <= year <= high:
            value = int(matches)
            if value <= 0:
                raise ValueError('Nonpositive Google count')
            total += value
            stats['year_rows'] += 1
    emit(previous, total)
    return counts, stats


def aggregate_file(task):
    path, item, policy, vocabulary = task
    if sha(path) != item['sha256']:
        raise ValueError(f'Source checksum mismatch: {path.name}')
    with zipfile.ZipFile(path) as archive:
        expected = path.name[:-4]
        if archive.namelist() != [expected]:
            raise ValueError('Unexpected Google archive member')
        with archive.open(expected) as rows:
            counts, stats = aggregate_rows(rows, item['order'], policy, vocabulary)
    return path.name, counts, stats


def prune(counts, limit, minimum=1):
    ordered = sorted(((gram, count) for gram, count in counts.items() if count >= minimum),
                     key=lambda pair: (-pair[1], pair[0]))
    return dict(ordered[:limit])


def coefficients(baseline, google, percent):
    if not 0 < percent < 100 or baseline <= 0 or google <= 0:
        raise ValueError('Invalid mixture mass')
    left, right = (100 - percent) * google, percent * baseline
    common = math.gcd(left, right)
    return left // common, right // common


def combine(baseline, google, percent, unigrams_only=False):
    base_mass = sum(v for k, v in baseline.items() if len(k) == 1)
    google_mass = sum(v for k, v in google.items() if len(k) == 1)
    left, right = coefficients(base_mass, google_mass, percent)
    mixed = Counter({k: v * left for k, v in baseline.items()})
    mixed.update({k: v * right for k, v in google.items() if not unigrams_only or len(k) == 1})
    return mixed, {'baseline_coefficient': left, 'google_coefficient': right,
                   'google_unigram_share_percent': percent, 'unigram_control': unigrams_only}


def read_counts(path):
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        return {(*json.loads(context), word): count for context, word, count in db.execute(
            'SELECT context,word,count FROM counts ORDER BY context,word')}


def write_counts(template, destination, counts, provenance):
    if destination.exists() or template.resolve() == destination.resolve():
        raise ValueError('Output must be a new file separate from production')
    unigrams = {k[0]: v for k, v in counts.items() if len(k) == 1}
    for gram, count in counts.items():
        if not 1 <= len(gram) <= 3 or not 0 < count <= MAX_COUNT:
            raise ValueError('Invalid or overflowing count')
        if any(word not in unigrams for word in gram) or count > unigrams[gram[-1]]:
            raise ValueError('Contextual counts inconsistent with unigrams')
    with closing(sqlite3.connect(template.resolve().as_uri() + '?mode=ro', uri=True)) as source:
        with closing(sqlite3.connect(destination)) as db:
            source.backup(db)
            with db:
                db.execute('DELETE FROM counts')
                db.execute('DELETE FROM vocabulary')
                db.execute('DELETE FROM imports')
                db.executemany('INSERT INTO vocabulary(word) VALUES (?)', ((w,) for w in sorted(unigrams)))
                db.executemany('INSERT INTO counts(context,word,count) VALUES (?,?,?)',
                               ((json.dumps(k[:-1], ensure_ascii=False, separators=(',', ':')), k[-1], v)
                                for k, v in sorted(counts.items())))
                db.execute("UPDATE metadata SET value=? WHERE key='provenance'",
                           (json.dumps(provenance, sort_keys=True),))


def fetch_sources(manifest, folder):
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, item in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('Invalid archive name')
        path = folder / name
        if not path.exists():
            temporary = path.with_suffix('.part')
            with urllib.request.urlopen(item['url'], timeout=120) as response, temporary.open('wb') as out:
                shutil.copyfileobj(response, out, 4 * 1024 * 1024)
            if temporary.stat().st_size != item['bytes'] or sha(temporary) != item['sha256']:
                raise ValueError(f'Download verification failed: {name}')
            temporary.replace(path)
        paths[name] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/google-books-pilot')
    parser.add_argument('--baseline', type=Path, required=True, help='Shipped production SQLite model')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use one through eight workers')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    policy_path = ROOT / 'google-books-policy.json'
    manifest_path = ROOT / 'google-books-source-manifest.json'
    policy = json.loads(policy_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    for order, shards in policy['shards'].items():
        found = sorted(i['shard'] for i in manifest['files'].values() if i['order'] == int(order))
        if found != sorted(shards):
            raise ValueError('Source shards differ from frozen policy')
    exe = ROOT / 'target/release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
    baseline = args.baseline.resolve()
    baseline_sha = sha(baseline)
    production = json.loads((ROOT / 'production-model.json').read_text())
    validation = json_run(exe, 'validate', '--database', baseline)
    if validation['logical_sha256'] != production['logical_sha256']:
        raise ValueError('Baseline is not the shipped production model')
    prepared = ROOT / 'data/aac-oanc/prepared'
    evaluations = [f'{s}-{p}' for s in policy['suites'] for p in ['dev', 'test']]
    info = json.loads((prepared / 'partitions.json').read_text())
    verify_prepared(prepared, info, production, evaluations)
    report = {'policy': policy, 'partition': 'dev', 'complete': False, 'models': {},
              'source_files': {}, 'baseline_validation': validation, 'baseline_sha256': baseline_sha,
              'policy_sha256': sha(policy_path), 'source_manifest_sha256': sha(manifest_path),
              'script_sha256': sha(Path(__file__)), 'executable_sha256': sha(exe),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
              'evaluation_hashes': {n: sha(prepared / (n + '.txt')) for n in evaluations},
              'algorithm': 'Unchanged 0.1/0.3/0.6 interpolated counts, personal learning disabled.',
              'limitations': policy['limitations'] + ' Aggregated Google counts cannot establish or remove full-sentence overlap with evaluations. These are known regression suites. Exact integer mixture scaling is for offline probability comparison; it changes absolute count units and must not ship with existing personal-learning weights without a separate calibration experiment.'}
    write_report(out / 'report.json', report)
    paths = fetch_sources(manifest, ROOT / 'data/google-books-1m-2009')
    google = {}
    vocabulary = None
    for order in [1, 2, 3]:
        aggregate = Counter()
        tasks = [(paths[n], i, policy, vocabulary) for n, i in manifest['files'].items() if i['order'] == order]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for name, counts, stats in pool.map(aggregate_file, tasks):
                aggregate.update(counts)
                report['source_files'][name] = dict(manifest['files'][name], **stats)
                print(f'Parsed {name}: {stats["rows"]:,} rows', flush=True)
                write_report(out / 'report.json', report)
        retained = prune(aggregate, policy['vocabulary_limit'] if order == 1 else policy['contextual_limit_per_order'],
                         1 if order == 1 else policy['contextual_min_count'])
        if order == 1:
            vocabulary = {g[0] for g in retained}
        report.setdefault('retention', {})[str(order)] = {
            'eligible_ngrams': len(aggregate), 'eligible_mass': sum(aggregate.values()),
            'retained_ngrams': len(retained), 'retained_mass': sum(retained.values())}
        google.update(retained)
        del aggregate
    if not all(r['retained_ngrams'] for r in report['retention'].values()):
        raise ValueError('Empty Google n-gram order')
    (out / 'models').mkdir()
    write_counts(baseline, out / 'google.sqlite', google, {'protocol': policy['protocol'], 'source': 'google-only'})
    report['google_validation'] = json_run(exe, 'validate', '--database', out / 'google.sqlite')
    write_report(out / 'report.json', report)
    if args.prepare_only:
        return
    base = read_counts(baseline)
    names = ['production', 'google-only'] + [f'google-{p}' for p in policy['google_unigram_share_percent']]
    names += [f'google-{policy["unigram_control_share_percent"]}-unigrams']
    hardware = f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs'
    for name in names:
        sqlite = baseline if name == 'production' else out / 'google.sqlite'
        recipe = {'source': name}
        if name.startswith('google-') and name != 'google-only':
            percent = int(name.split('-')[1])
            counts, recipe = combine(base, google, percent, name.endswith('-unigrams'))
            sqlite = out / (name + '.sqlite')
            write_counts(baseline, sqlite, counts, dict(protocol=policy['protocol'], **recipe))
            del counts
        compact = out / 'models' / (name + '.ngrams')
        validated = json_run(exe, 'convert', '--input', sqlite, '--output', compact)
        report['models'][name] = {'recipe': recipe, 'validation': validated,
                                 'sha256': sha(compact), 'bytes': compact.stat().st_size, 'scores': {}}
        for suite in policy['suites']:
            print(f'Scoring {name}/{suite}/dev', flush=True)
            report['models'][name]['scores'][suite] = json_run(
                exe, 'score', '--baseline', compact, '--input', prepared / f'{suite}-dev.txt', '--hardware', hardware)
            write_report(out / 'report.json', report)
    if sha(baseline) != baseline_sha:
        raise ValueError('Production baseline changed')
    report['complete'] = True
    report['production_baseline_unchanged'] = True
    for name in ['google-books-policy.json', 'google-books-source-manifest.json', 'ATTRIBUTION.md',
                 'source-manifest.json', 'aac-source-manifest.json']:
        shutil.copyfile(ROOT / name, out / name)
    shutil.copytree(ROOT / 'docs/corpus-notices', out / 'corpus-notices')
    write_report(out / 'report.json', report)
    print('Google development benchmark complete', flush=True)


if __name__ == '__main__':
    main()
