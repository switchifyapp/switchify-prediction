#!/usr/bin/env python3
"""One frozen AAC/OANC experiment. Never replaces the default public baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from quality import json_run

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data/aac-oanc'
POLICY = ROOT / 'aac-quality-policy.json'
MANIFEST = ROOT / 'aac-source-manifest.json'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def fetch_sources():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for name, item in manifest['files'].items():
        notice = ROOT / 'docs/corpus-notices' / name
        target = DATA / name
        if notice.exists():
            if sha(notice) != item['sha256']:
                raise ValueError(f'Notice checksum mismatch: {name}')
            shutil.copyfile(notice, target)
        if not target.exists() or sha(target) != item['sha256']:
            # Download into a temporary sibling: never trust a partial download.
            with tempfile.TemporaryDirectory(dir=DATA) as temp:
                downloaded = Path(temp) / name
                subprocess.run(['curl', '-fsSL', '--retry', '3', '--max-time', '600',
                                '-o', str(downloaded), item['url']], check=True)
                if sha(downloaded) != item['sha256']:
                    raise ValueError(f'Source checksum mismatch: {name}')
                downloaded.replace(target)
        print(f'Verified {name}', flush=True)
    return manifest


def utterances(text, annotation):
    """Read only the GrAF sentence/utterance spans; never join speaker turns."""
    root = ET.fromstring(annotation)
    spans = []
    for region in root.findall('{http://www.xces.org/ns/GrAF/1.0/}region'):
        start, end = map(int, region.attrib['anchors'].split())
        if not 0 <= start < end <= len(text):
            raise ValueError('Invalid OANC utterance offset')
        spans.append((start, end))
    if not spans:
        raise ValueError('Missing OANC utterance spans')
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('Overlapping OANC utterance spans')
    # XML annotations use Java UTF-16 character offsets; do not silently mis-slice
    # a source with non-BMP characters. This pinned corpus contains none.
    if any(ord(c) > 0xffff for c in text):
        raise ValueError('OANC UTF-16 offsets require BMP text')
    return [' '.join(text[a:b].split()) for a, b in spans]


def extract_oanc(archive, output):
    files = []
    total = 0
    with zipfile.ZipFile(archive) as source, output.open('w', encoding='utf-8', newline='\n') as target:
        for name in sorted(source.namelist()):
            if not name.startswith('OANC-GrAF/data/spoken/') or not name.endswith('.txt'):
                continue
            text = source.read(name).decode('utf-8')
            spans = utterances(text, source.read(name[:-4] + '-s.xml'))
            for span in spans:
                target.write(span + '\n')
            total += len(spans)
            files.append(name)
    if not files:
        raise ValueError('No spoken OANC files')
    return {'files': len(files), 'utterances': total,
            'member_list_sha256': hashlib.sha256(('\n'.join(files) + '\n').encode()).hexdigest()}


def normalized(exe, path, out):
    subprocess.run([str(exe), 'normalize', '--input', str(path), '--output', str(out)], check=True)
    return set(out.read_text(encoding='utf-8').splitlines()) - {''}


def partitions(current, existing_eval, aac, oanc, policy):
    """Freeze current counts; remove held-out overlap before selecting new text."""
    old_train = set(current)
    old_eval = set().union(*existing_eval.values())
    aac_reserved = aac['dev'] | aac['test']
    aac_train = aac['train'] - old_eval - aac_reserved
    # Existing published training wins: remove those sentences from fresh AAC
    # evaluation, ensuring both compared models see exactly the same test set.
    aac_dev = aac['dev'] - old_train - old_eval - aac['train']
    aac_test = aac['test'] - old_train - old_eval - aac['train'] - aac['dev']
    eligible = [s for s in oanc - old_eval - aac_reserved - old_train - aac_train
                if policy['oanc_min_words'] <= len(s.split()) <= policy['oanc_max_words']]
    eligible.sort(key=lambda s: (hashlib.sha256(s.encode()).hexdigest(), s))
    selected = eligible[:policy['oanc_max_sentences']]
    weights = policy['weights']
    candidate = current * weights['current'] + sorted(aac_train) * weights['aac'] + selected * weights['oanc']
    result = dict(existing_eval, **{'aac-dev': sorted(aac_dev), 'aac-test': sorted(aac_test),
                                  'baseline': current, 'candidate': candidate})
    training = set(candidate)
    eval_sets = [set(result[k]) for k in result if k not in ['baseline', 'candidate']]
    if any(training & s for s in eval_sets):
        raise ValueError('Training/evaluation leakage')
    if any(a & b for i, a in enumerate(eval_sets) for b in eval_sets[i + 1:]):
        raise ValueError('Evaluation partition overlap')
    if any(not v for v in result.values()):
        raise ValueError('Empty partition')
    return result, {'aac_train_unique': len(aac_train), 'oanc_eligible_unique': len(eligible),
                    'oanc_selected_unique': len(selected), 'aac_original_normalized': {k: len(v) for k, v in aac.items()},
                    'training_evaluation_overlap': 0, 'evaluation_partition_overlap': 0}


def decision(results, policy):
    gates = policy['acceptance']
    delta = lambda suite: results[suite]['candidate']['accuracy'][2]['top5'] - results[suite]['baseline']['accuracy'][2]['top5']
    saving = results['aac']['candidate']['selection_proxy']['savings_fraction'] - results['aac']['baseline']['selection_proxy']['savings_fraction']
    checks = {'aac_top5_gain': delta('aac') >= gates['aac_top5_prefix2_gain_min'],
              'general_preserved': -delta('general') <= gates['general_top5_prefix2_loss_max'],
              'conversation_preserved': -delta('conversation') <= gates['conversation_top5_prefix2_loss_max'],
              'aac_selection_proxy_preserved': saving >= gates['aac_selection_savings_gain_min']}
    return {'top5_prefix2_delta': {s: delta(s) for s in results}, 'aac_selection_savings_delta': saving,
            'checks': checks, 'quality_passed': all(checks.values()),
            'latency_target_met': all(r['candidate']['warm_p95_ms'] < gates['warm_p95_target_ms'] for r in results.values())}


def prepare(exe, work, policy):
    subprocess.run([sys.executable, str(ROOT / 'scripts/fetch_corpus.py')], check=True)
    original = json_run(exe, 'prepare')
    current = (ROOT / 'data/prepared/candidate.txt').read_text(encoding='utf-8').splitlines()
    existing = {f'{s}-{p}': set((ROOT / f'data/prepared/{s}-{p}.txt').read_text(encoding='utf-8').splitlines())
                for s in ['general', 'conversation'] for p in ['dev', 'test']}
    aac = {p: normalized(exe, DATA / f'aac-{p}.txt', work / f'normalized-aac-{p}.txt') for p in ['train', 'dev', 'test']}
    extraction = extract_oanc(DATA / 'OANC_GrAF.zip', work / 'spoken.txt')
    oanc = normalized(exe, work / 'spoken.txt', work / 'normalized-spoken.txt')
    prepared, stats = partitions(current, existing, aac, oanc, policy)
    info = {}
    for name, lines in prepared.items():
        path = work / (name + '.txt')
        path.write_bytes(('\n'.join(sorted(lines) if isinstance(lines, set) else lines) + '\n').encode())
        info[name] = {'sha256': sha(path), 'sentences': len(lines), 'unique_sentences': len(set(lines))}
    return {'files': info, 'source_manifest_sha256': sha(MANIFEST), 'policy_sha256': sha(POLICY),
            'existing_partitions': original, 'oanc_extraction': extraction, **stats}


def hardware():
    value = f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs'
    if platform.system() == 'Darwin':
        value += '; ' + subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    elif platform.system() == 'Linux':
        value += '; ' + next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')), 'unknown CPU')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/aac-experiment')
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = fetch_sources()
    policy = json.loads(POLICY.read_text(encoding='utf-8'))
    exe = ROOT / 'target/release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
    work = DATA / 'prepared'
    work.mkdir(exist_ok=True)
    partition_info = prepare(exe, work, policy)
    if args.prepare_only:
        write_json(work / 'partitions.json', partition_info)
        print(json.dumps(partition_info, indent=2))
        return
    # Stage a complete run: never pair a new database with stale report files.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        stage = Path(temp)
        models = {}
        validations = {}
        for name in ['baseline', 'candidate']:
            path = stage / (name + '.sqlite')
            validations[name] = json_run(exe, 'build', '--input', work / (name + '.txt'), '--output', path)
            models[name] = path
        # Independent repeated build verifies deterministic logical contents.
        repeat = stage / 'repeat.sqlite'
        repeated = json_run(exe, 'build', '--input', work / 'candidate.txt', '--output', repeat)
        if repeated['logical_sha256'] != validations['candidate']['logical_sha256']:
            raise ValueError('Non-deterministic candidate counts')
        repeat.unlink()
        report = {'protocol': 'aac-oanc-v1', 'policy': policy, 'partitions': partition_info,
                  'validation': validations, 'deterministic_logical_contents': True,
                  'algorithm': 'Unchanged 0.1/0.3/0.6 interpolated n-grams, no personal learning.',
                  'limitations': 'AAC is imagined crowd-worker communication, not real AAC use. OANC is older American speech. General and Taskmaster tests are previously observed regression sets. AAC test is first-use held-out. Selection savings are a modeled proxy, not observed switch use.'}
        for part in ['dev', 'test']:
            report[part] = {}
            for suite in ['aac', 'general', 'conversation']:
                report[part][suite] = {}
                for name, path in models.items():
                    print(f'Scoring {part}/{suite}/{name}', flush=True)
                    report[part][suite][name] = json_run(exe, 'score', '--baseline', path, '--input', work / f'{suite}-{part}.txt', '--hardware', hardware())
            report[part + '_decision'] = decision(report[part], policy)
            print(json.dumps(report[part + '_decision']), flush=True)
        report['recommend_promotion'] = all(report[p + '_decision']['quality_passed'] for p in ['dev', 'test'])
        report['default_baseline_changed'] = False
        if os.name != 'nt':
            import resource
            rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            report['peak_child_process_rss_bytes'] = rss if platform.system() == 'Darwin' else rss * 1024
            report['memory_measurement'] = 'Maximum child-process RSS across preparation, builds and scoring; not predictor-only memory.'
        write_json(stage / 'report.json', report)
        write_json(stage / 'partitions.json', partition_info)
        write_json(stage / 'aac-source-manifest.json', manifest)
        for name in ['aac-quality-policy.json', 'source-manifest.json', 'ATTRIBUTION.md', 'LICENSE']:
            shutil.copyfile(ROOT / name, stage / name)
        shutil.copytree(ROOT / 'docs/corpus-notices', stage / 'corpus-notices')
        (stage / 'README.txt').write_text('Experimental AAC/OANC model, not the default baseline. Read report.json for acceptance and measured regressions. candidate.sqlite contains only training counts; baseline.sqlite is the existing conversational comparison model. These databases use public corpus data only. Retain ATTRIBUTION.md, both source manifests and corpus-notices when redistributing.\n', encoding='utf-8')
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        (stage / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(stage).as_posix()}\n' for p in files), encoding='utf-8')
        args.output.mkdir(exist_ok=True)
        # Explicit stage contains public corpus models, reports and notices only.
        for path in files + [stage / 'SHA256SUMS']:
            dest = args.output / path.relative_to(stage)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        print(json.dumps({'output': str(args.output), 'recommend_promotion': report['recommend_promotion']}), flush=True)


if __name__ == '__main__':
    main()
