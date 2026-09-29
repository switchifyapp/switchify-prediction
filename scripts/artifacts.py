#!/usr/bin/env python3
"""Build the pinned production model only after frozen quality/identity gates pass."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from aac_experiment import decision, sha, write_json
from quality import json_run

ROOT = Path(__file__).resolve().parent.parent
NOTICE_NAMES = ['aac-notice.html', 'aac-readme.txt', 'oanc-notice.html',
                'oanc-download.html', 'oanc-legacy-license.txt']


def verify_quality(report, manifest, expected_quality):
    """Recompute decisions; do not trust serialized success flags in a report."""
    policy = manifest['sources']['aac-quality-policy.json']
    if report['policy'] != policy or report['protocol'] != manifest['quality_protocol']:
        raise ValueError('Quality protocol/policy differs from the promoted model')
    if report['partitions']['files'] != manifest['partitions']:
        raise ValueError('Prepared corpus or evaluation partitions changed')
    if report['validation']['candidate']['logical_sha256'] != manifest['logical_sha256']:
        raise ValueError('Evaluated model differs from production fingerprint')
    if not report['deterministic_logical_contents']:
        raise ValueError('Repeated model builds differ')
    for name in ['training_evaluation_overlap', 'evaluation_partition_overlap']:
        if report['partitions'][name] != 0:
            raise ValueError('Held-out partition overlap')
    if report['partitions']['source_manifest_sha256'] != manifest['source_files']['aac-source-manifest.json']:
        raise ValueError('Evaluation source manifest changed')
    if report['partitions']['policy_sha256'] != manifest['source_files']['aac-quality-policy.json']:
        raise ValueError('Evaluation policy changed')
    for part in ['dev', 'test']:
        if not decision(report[part], policy)['quality_passed']:
            raise ValueError(f'{part} quality gate failed; production artifact refused')
        for suite in ['aac', 'general', 'conversation']:
            for model in ['baseline', 'candidate']:
                actual = report[part][suite][model]
                expected = expected_quality[part][suite][model]
                for field in ['accuracy', 'selection_proxy', 'evaluation_sha256', 'evaluated_sentences']:
                    if actual[field] != expected[field]:
                        raise ValueError(f'Frozen quality results changed: {part}/{suite}/{model}/{field}')


def main():
    os.chdir(ROOT)
    exe = ROOT / 'target/release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
    out = ROOT / 'artifacts/production'
    out.parent.mkdir(exist_ok=True)
    manifest = json.loads((ROOT / 'production-model.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_files'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Production source/policy fingerprint mismatch: {name}')
    expected_quality = json.loads((ROOT / 'production-quality.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='switchify-production-') as temp:
        stage = Path(temp) / 'package'
        stage.mkdir()
        experiment = Path(temp) / 'evaluation'
        subprocess.run([sys.executable, str(ROOT / 'scripts/aac_experiment.py'), '--output', str(experiment)], check=True)
        report = json.loads((experiment / 'report.json').read_text(encoding='utf-8'))
        verify_quality(report, manifest, expected_quality)
        if sha(ROOT / manifest['training_path']) != manifest['training_sha256']:
            raise ValueError('Production training text mismatch')
        validation = json_run(exe, 'build', '--output', stage / 'english.sqlite')
        verified = json_run(exe, 'validate', '--database', stage / 'english.sqlite', '--production')
        if validation != verified or validation['logical_sha256'] != report['validation']['candidate']['logical_sha256']:
            raise ValueError('Published model differs from the evaluated candidate')
        # All six candidate score reports must reference the same evaluated file.
        evaluated_hash = sha(experiment / 'candidate.sqlite')
        if any(report[p][s]['candidate']['database_sha256'] != evaluated_hash
               for p in ['dev', 'test'] for s in ['aac', 'general', 'conversation']):
            raise ValueError('Score report/database mismatch')
        report['default_baseline_changed'] = True
        report['artifact_role'] = 'production baseline'
        report['production'] = {'model_id': manifest['model_id'], 'database_sha256': sha(stage / 'english.sqlite'),
                                'logical_sha256': validation['logical_sha256'], 'default_baseline_changed': True,
                                'note': 'Production metadata embeds full provenance; counts exactly match the evaluated candidate.'}
        write_json(stage / 'quality-report.json', report)
        write_json(stage / 'validation.json', validation)
        shutil.copyfile(experiment / 'partitions.json', stage / 'partitions.json')
        for name in ['production-model.json', 'production-quality.json', 'source-manifest.json',
                     'aac-source-manifest.json', 'quality-policy.json', 'aac-quality-policy.json', 'ATTRIBUTION.md', 'LICENSE']:
            shutil.copyfile(ROOT / name, stage / name)
        (stage / 'corpus-notices').mkdir()
        for name in NOTICE_NAMES:
            shutil.copyfile(ROOT / 'docs/corpus-notices' / name, stage / 'corpus-notices' / name)
        shutil.copyfile(ROOT / 'scripts/verify_bundle.py', stage / 'verify_bundle.py')
        (stage / 'README.txt').write_text(
            'Switchify English production model en-aac-oanc-v1, schema v1.\n'
            'Run: python verify_bundle.py\n'
            'Then: switchify-prediction validate --database english.sqlite --production\n'
            'Reuse a long-lived library Predictor; opening/learning belongs off the UI thread.\n'
            'Retain ATTRIBUTION.md, manifests and corpus-notices when redistributing.\n'
            'The model uses public training corpora only; held-out evaluation text is excluded.\n'
            'quality-report.json describes the identical evaluated counts with separate evaluation-file hashes.\n', encoding='utf-8')
        files = sorted(p for p in stage.rglob('*') if p.is_file())
        (stage / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(stage).as_posix()}\n' for p in files), encoding='utf-8')
        subprocess.run([sys.executable, str(stage / 'verify_bundle.py')], cwd=stage, check=True)
        # Output is a dedicated generated directory, never an arbitrary caller path.
        # CI uploads only after this complete staging/verification succeeds.
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(stage, out)
        bundle = ROOT / 'artifacts/model-bundle'
        bundle.mkdir(exist_ok=True)
        archive = Path(shutil.make_archive(str(bundle / ('switchify-english-' + manifest['model_id'])), 'zip', stage))
        (bundle / (archive.name + '.sha256')).write_text(f'{sha(archive)}  {archive.name}\n', encoding='utf-8')
        print(json.dumps({'artifacts': str(out), 'model': manifest['model_id'], 'quality_passed': True}), flush=True)


if __name__ == '__main__':
    main()
