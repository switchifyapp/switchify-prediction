#!/usr/bin/env python3
"""Build the gated conversational baseline and publish only named public artifacts."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
exe = ROOT / 'target' / 'release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
out = ROOT / 'artifacts'
out.mkdir(exist_ok=True)
subprocess.run([sys.executable, 'scripts/fetch_corpus.py'], check=True)
with tempfile.TemporaryDirectory(prefix='switchify-public-artifacts-') as temp:
    stage = Path(temp)
    subprocess.run([sys.executable, 'scripts/quality.py', '--output', str(stage)], check=True)
    subprocess.run([str(exe), 'build', '--output', str(stage / 'english.sqlite')], check=True)
    validation = subprocess.check_output([str(exe), 'validate', '--database', str(stage / 'english.sqlite')])
    # Same count model evaluated by quality.py; shipping metadata includes full provenance.
    evaluated = json.loads(subprocess.check_output([str(exe), 'validate', '--database', 'data/prepared/candidate.sqlite']))
    if json.loads(validation)['logical_sha256'] != evaluated['logical_sha256']:
        raise SystemExit('Published model differs from the evaluated candidate')
    (stage / 'validation.json').write_bytes(validation)
    for name in ['source-manifest.json', 'quality-policy.json', 'ATTRIBUTION.md', 'LICENSE']:
        shutil.copyfile(ROOT / name, stage / name)
    shutil.copyfile(ROOT / 'data/prepared/partitions.json', stage / 'partitions.json')
    (stage / 'README.txt').write_text('english.sqlite is the development-accepted conversational baseline. Its counts exactly match the candidate evaluated in quality-report.json. Held-out development and test sentences are excluded. The comparison baseline uses only WorldAlphabets training sentences. No personal writing is included. Retain ATTRIBUTION.md when redistributing.\n')
    files = sorted(p for p in stage.iterdir() if p.is_file())
    (stage / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
    for file in stage.iterdir():
        shutil.copyfile(file, out / file.name)
    # Retire this script's old report so it cannot be mistaken for the new model.
    (out / 'evaluation.json').unlink(missing_ok=True)
    report = json.loads((stage / 'quality-report.json').read_text())
    print(json.dumps({'artifacts': str(out), 'development_decision': report['decision']}, indent=2))
