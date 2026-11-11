#!/usr/bin/env python3
"""Frozen broad-unigram experiment. Never modifies a production model."""
import argparse
from contextlib import closing
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
import urllib.request
import zipfile

from quality import json_run

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ('aac', 'general', 'conversation')


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def source_files(manifest, cache, output):
    """Keep complete pinned archives; extract only explicitly selected files."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, spec in manifest['sources'].items():
        path = cache / spec['filename']
        if not path.exists() or sha(path) != spec['sha256']:
            with tempfile.TemporaryDirectory(dir=cache) as temp:
                downloaded = Path(temp) / spec['filename']
                with urllib.request.urlopen(spec['url'], timeout=120) as response:
                    with downloaded.open('wb') as target:
                        shutil.copyfileobj(response, target)
                if sha(downloaded) != spec['sha256']:
                    raise ValueError(f'Source checksum mismatch: {name}')
                downloaded.replace(path)
        paths[name] = path
        shutil.copyfile(path, output / spec['filename'])
    dependencies = output / 'reader'
    package = dependencies / 'msgpack'
    package.mkdir(parents=True)
    with tarfile.open(paths['msgpack']) as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if (member.isfile() and len(parts) == 3 and parts[:2] == ('msgpack-1.1.2', 'msgpack')
                    and parts[2].endswith('.py')):
                (package / parts[2]).write_bytes(archive.extractfile(member).read())
    # Import only the pinned pure-Python reader, not a platform-specific extension.
    sys.path.insert(0, str(dependencies))
    from msgpack.fallback import unpackb
    spec = manifest['sources']['wordfreq']
    notices = output / 'wordfreq-notices'
    notices.mkdir()
    with zipfile.ZipFile(paths['wordfreq']) as archive:
        packed = archive.read(spec['data_member'])
        for name in archive.namelist():
            parts = PurePosixPath(name).parts
            if len(parts) == 2 and parts[0] == 'wordfreq-3.1.1.dist-info':
                (notices / parts[1]).write_bytes(archive.read(name))
    bins = unpackb(gzip.decompress(packed), raw=False)
    if not bins or bins[0] != {'format': 'cB', 'version': 1}:
        raise ValueError('Unexpected wordfreq bin format')
    return bins[1:], {'data_member_sha256': hashlib.sha256(packed).hexdigest(),
                      'wheel_sha256': sha(paths['wordfreq']),
                      'reader_sha256': sha(paths['msgpack'])}


def normalized_word(raw, policy):
    """Conservative single-word subset, checked against the Rust normalizer."""
    word = unicodedata.normalize('NFC', raw.replace('\u2019', "'").lower())
    if not policy['minimum_word_codepoints'] <= len(word) <= policy['maximum_word_codepoints']:
        return None
    for i, char in enumerate(word):
        if char.isalpha() or (unicodedata.category(char).startswith('M') and i > 0):
            continue
        if char == "'" and i > 0 and i + 1 < len(word) and word[i + 1].isalpha():
            continue
        return None
    return word


def frequencies(bins, policy):
    result = {}
    for index, bucket in enumerate(bins):
        if not isinstance(bucket, list) or any(not isinstance(word, str) for word in bucket):
            raise ValueError('Invalid frequency bucket')
        for raw in bucket:
            word = normalized_word(raw, policy)
            if word is None:
                continue
            if word not in result and len(result) >= policy['vocabulary_limit']:
                return result
            result[word] = result.get(word, 0.0) + 10 ** (-index / 100)
    if not result:
        raise ValueError('No usable source words')
    return result


def allocate(frequencies, budget):
    """Exact mass, stable largest-remainder allocation; no minimum pseudocount."""
    if not frequencies or budget < 0 or any(not math.isfinite(v) or v <= 0 for v in frequencies.values()):
        raise ValueError('Invalid allocation')
    words = sorted(frequencies)
    total = math.fsum(frequencies[word] for word in words)
    exact = {word: budget * frequencies[word] / total for word in words}
    result = {word: math.floor(exact[word]) for word in words}
    remaining = budget - sum(result.values())
    if not 0 <= remaining <= len(words):
        raise ValueError('Invalid rounding remainder')
    ranked = sorted(words, key=lambda word: (-(exact[word] - result[word]), word))
    for word in ranked[:remaining]:
        result[word] += 1
    return {word: count for word, count in result.items() if count}


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def contextual_hash(db):
    digest = hashlib.sha256()
    for row in db.execute("SELECT context, word, count FROM counts WHERE context != '[]' ORDER BY context, word"):
        digest.update((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
    return digest.hexdigest()


def supplement(baseline, destination, added, provenance):
    if destination.exists() or baseline.resolve() == destination.resolve():
        raise ValueError('Candidate output must be new and separate')
    with closing(readonly(baseline)) as source:
        before = contextual_hash(source)
        with closing(sqlite3.connect(destination)) as db, db:
            source.backup(db)
            db.executemany('INSERT OR IGNORE INTO vocabulary(word) VALUES (?)',
                           ((word,) for word in sorted(added)))
            db.executemany("INSERT INTO counts(context,word,count) VALUES ('[]',?,?) "
                           'ON CONFLICT(context,word) DO UPDATE SET count=counts.count+excluded.count',
                           sorted(added.items()))
            db.execute("UPDATE metadata SET value=? WHERE key='provenance'", (json.dumps(provenance, sort_keys=True),))
            if contextual_hash(db) != before:
                raise ValueError('Contextual counts changed')
    return before


def assessment(baseline, candidate, policy):
    deltas = [candidate[domain]['accuracy'][prefix]['top5'] - baseline[domain]['accuracy'][prefix]['top5']
              for domain in DOMAINS for prefix in range(3)]
    saving = [candidate[domain]['selection_proxy']['savings_fraction'] - baseline[domain]['selection_proxy']['savings_fraction']
              for domain in DOMAINS]
    mean = math.fsum(deltas) / len(deltas)
    aac_delta = candidate['aac']['accuracy'][2]['top5'] - baseline['aac']['accuracy'][2]['top5']
    gates = policy['acceptance']
    checks = {'mean_early_top5': mean >= gates['mean_early_top5_gain_min'],
              'worst_early_top5': min(deltas) >= -gates['worst_early_top5_loss_max'],
              'aac_prefix2': aac_delta >= -gates['aac_prefix2_loss_max'],
              'selection_savings': min(saving) >= -gates['worst_selection_savings_loss_max']}
    return {'mean_early_top5_delta': mean, 'worst_early_top5_delta': min(deltas),
            'aac_prefix2_delta': aac_delta, 'worst_selection_savings_delta': min(saving),
            'checks': checks, 'quality_passed': all(checks.values()),
            'latency_target_met': all(candidate[domain]['warm_p95_ms'] < gates['warm_p95_target_ms'] for domain in DOMAINS)}


def select_candidate(assessments, models):
    eligible = [name for name in assessments if assessments[name]['quality_passed']]
    pool = eligible or list(assessments)
    selected = min(pool, key=lambda name: (-assessments[name]['mean_early_top5_delta'],
                                          models[name]['mass_fraction'], models[name]['mode']))
    return {'candidate': selected, 'diagnostic_only': not bool(eligible), 'eligible': sorted(eligible)}


def score(exe, model, partitions, part, hardware):
    result = {}
    for domain in DOMAINS:
        print(f'Scoring {model.stem}: {domain}-{part}', flush=True)
        result[domain] = json_run(exe, 'score', '--baseline', model,
                                  '--input', partitions / f'{domain}-{part}.txt', '--hardware', hardware)
    return result


def verified_partitions(directory, production):
    hashes = {}
    for domain in DOMAINS:
        for part in ('dev', 'test'):
            name = f'{domain}-{part}'
            actual = sha(directory / (name + '.txt'))
            if actual != production['partitions'][name]['sha256']:
                raise ValueError(f'Frozen partition mismatch: {name}')
            hashes[name] = actual
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True, help='Shipped SQLite model, not a personal database')
    parser.add_argument('--cli', type=Path, default=ROOT / 'target/release' / ('switchify-prediction.exe' if sys.platform == 'win32' else 'switchify-prediction'))
    parser.add_argument('--source-cache', type=Path, default=ROOT / 'data/frequency-sources')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/broad-frequency-v1')
    args = parser.parse_args()
    args.baseline = args.baseline.resolve()
    args.cli = args.cli.resolve()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    policy_path = ROOT / 'frequency-quality-policy.json'
    manifest_path = ROOT / 'frequency-source-manifest.json'
    policy, manifest = read_json(policy_path), read_json(manifest_path)
    production = read_json(ROOT / 'production-model.json')
    partitions = ROOT / 'data/aac-oanc/prepared'
    partition_hashes = verified_partitions(partitions, production)
    initial_hash = sha(args.baseline)
    baseline_validation = json_run(args.cli, 'validate', '--database', args.baseline, '--production')
    if baseline_validation['logical_sha256'] != production['logical_sha256']:
        raise ValueError('Unexpected baseline')
    args.output.mkdir(parents=True)
    output = args.output.resolve()
    write_json(output / 'policy.json', policy)
    write_json(output / 'source-manifest.json', manifest)
    bins, source_info = source_files(manifest, args.source_cache, output)
    source_frequencies = frequencies(bins, policy)
    words = sorted(source_frequencies)
    raw_path, normalized_path = output / 'source-words.txt', output / 'normalized-words.txt'
    raw_path.write_text('\n'.join(words) + '\n', encoding='utf-8')
    subprocess.run([str(args.cli), 'normalize', '--input', str(raw_path), '--output', str(normalized_path)], check=True)
    if normalized_path.read_text(encoding='utf-8').splitlines() != words:
        raise ValueError('Source tokenization differs from predictor')
    with closing(readonly(args.baseline)) as db:
        baseline_counts = dict(db.execute("SELECT word,count FROM counts WHERE context='[]'"))
        baseline_contexts = contextual_hash(db)
    total = sum(baseline_counts.values())
    missing = {word: freq for word, freq in source_frequencies.items() if word not in baseline_counts}
    report = {'protocol': policy['protocol'], 'policy': policy,
              'implementation_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'experiment_script_sha256': sha(Path(__file__)), 'cli_sha256': sha(args.cli),
              'policy_sha256': sha(policy_path), 'source_manifest_sha256': sha(manifest_path),
              'source': source_info, 'partitions': partition_hashes,
              'baseline': {'sha256': initial_hash, 'validation': baseline_validation,
                           'unigram_mass': total, 'contextual_sha256': baseline_contexts},
              'coverage_audit': {'source_words': len(words), 'baseline_words': len(baseline_counts),
                                 'source_words_missing_from_baseline': len(missing),
                                 'missing_source_frequency_mass': math.fsum(missing.values())},
              'models': {}, 'development': {}, 'assessments': {},
              'algorithm_changed': False, 'production_model_changed': False,
              'personal_learning': False,
              'limitations': 'Known regression partitions, external source overlap cannot be excluded, no actual switch-user evaluation, no personal-learning validation, local experimental models are not for redistribution.'}
    shutil.copyfile(args.baseline, output / 'baseline.sqlite')
    json_run(args.cli, 'convert', '--input', output / 'baseline.sqlite', '--output', output / 'baseline.ngrams')
    for mode in policy['modes']:
        pool = missing if mode == 'missing' else source_frequencies
        for fraction in policy['mass_fractions']:
            name = f'{mode}-{round(fraction * 100):02d}'
            added = allocate(pool, math.floor(total * fraction + 0.5))
            path = output / (name + '.sqlite')
            provenance = {'protocol': policy['protocol'], 'parent_logical_sha256': production['logical_sha256'],
                          'policy_sha256': report['policy_sha256'], 'source_manifest_sha256': report['source_manifest_sha256'],
                          'source_data': source_info, 'mode': mode, 'mass_fraction': fraction,
                          'modification': 'Added unigram pseudocounts only; original bigrams and trigrams unchanged.',
                          'source_terms': manifest['sources']['wordfreq']['terms']}
            supplement(args.baseline, path, added, provenance)
            validation = json_run(args.cli, 'validate', '--database', path)
            compact = output / (name + '.ngrams')
            converted = json_run(args.cli, 'convert', '--input', path, '--output', compact)
            if validation['logical_sha256'] != converted['logical_sha256']:
                raise ValueError('Conversion changed candidate counts')
            report['models'][name] = {'mode': mode, 'mass_fraction': fraction, 'added_mass': sum(added.values()),
                                      'supplemented_words': len(added), 'new_words': len(set(added) - baseline_counts.keys()),
                                      'validation': validation, 'ngrams_bytes': compact.stat().st_size,
                                      'ngrams_sha256': sha(compact)}
    hardware = f'{platform.platform()}; {platform.machine()}; {__import__("os").cpu_count()} logical CPUs'
    report['hardware'] = hardware
    report['development']['baseline'] = score(args.cli, output / 'baseline.ngrams', partitions, 'dev', hardware)
    for name in report['models']:
        result = score(args.cli, output / (name + '.ngrams'), partitions, 'dev', hardware)
        report['development'][name] = result
        report['assessments'][name] = assessment(report['development']['baseline'], result, policy)
        write_json(output / 'development-progress.json', report)
    selection = select_candidate(report['assessments'], report['models'])
    selected = selection['candidate']
    selection.update({'policy_sha256': report['policy_sha256'], 'source_manifest_sha256': report['source_manifest_sha256'],
                      'model_fingerprints': {name: model['validation']['logical_sha256'] for name, model in report['models'].items()},
                      'development_sha256': hashlib.sha256(json.dumps(report['development'], sort_keys=True).encode()).hexdigest()})
    write_json(output / 'selection.json', selection)
    report['selection'] = selection
    report['selection_file_sha256'] = sha(output / 'selection.json')
    # No test metric has been read before this persisted selection.
    report['test'] = {name: score(args.cli, output / (name + '.ngrams'), partitions, 'test', hardware)
                      for name in ('baseline', selected)}
    report['test_assessment'] = assessment(report['test']['baseline'], report['test'][selected], policy)
    report['recommend_separate_promotion_review'] = not selection['diagnostic_only'] and report['test_assessment']['quality_passed']
    # Rebuild the selected candidate independently, including its source attribution.
    chosen = report['models'][selected]
    provenance = read_json_value(output / (selected + '.sqlite'), 'provenance')
    added = allocate(missing if chosen['mode'] == 'missing' else source_frequencies, chosen['added_mass'])
    repeat = output / 'repeat.sqlite'
    supplement(args.baseline, repeat, added, provenance)
    repeated = json_run(args.cli, 'validate', '--database', repeat)
    report['deterministic_logical_counts'] = repeated['logical_sha256'] == chosen['validation']['logical_sha256']
    report['original_baseline_unchanged'] = sha(args.baseline) == initial_hash
    if not report['deterministic_logical_counts'] or not report['original_baseline_unchanged']:
        raise ValueError('Reproducibility or baseline immutability failed')
    write_json(output / 'report.json', report)
    print(json.dumps({'selection': selection, 'test_assessment': report['test_assessment'],
                      'recommend_separate_promotion_review': report['recommend_separate_promotion_review']}, indent=2))


def read_json_value(path, key):
    with closing(readonly(path)) as db:
        return json.loads(db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()[0])


if __name__ == '__main__':
    main()
