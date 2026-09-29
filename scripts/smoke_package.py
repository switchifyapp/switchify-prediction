#!/usr/bin/env python3
"""Exercise downloaded production binaries and database outside the checkout."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

from aac_experiment import sha
from verify_bundle import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        with zipfile.ZipFile(args.cli) as archive:
            archive.extractall(root)
        with zipfile.ZipFile(args.model) as archive:
            archive.extractall(root / 'model')
        cli = next(p for p in root.iterdir() if p.name.startswith('switchify-prediction-'))
        verify(cli)
        verify(root / 'model')
        exe = cli / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
        if os.name != 'nt':
            exe.chmod(0o755)
        baseline = root / 'model/english.sqlite'
        initial = sha(baseline)
        def run(*argv, text=None):
            return json.loads(subprocess.check_output([str(exe), *map(str, argv)], input=text, text=True, cwd=root))
        run('validate', '--database', baseline, '--production')
        predictions = run('predict', '--baseline', baseline, '--before', 'I need', '--prefix', 'he')
        if predictions[0]['word'] != 'help':
            raise ValueError('Packaged model prediction mismatch')
        personal = root / 'personal.sqlite'
        run('learn', '--baseline', baseline, '--personal', personal, text='I need zyzzyvate. ' * 3)
        result = run('predict', '--baseline', baseline, '--personal', personal, '--before', 'I need', '--prefix', 'zyzz')
        if result[0]['word'] != 'zyzzyvate':
            raise ValueError('Packaged personal learning failed')
        run('reset-personal', '--baseline', baseline, '--personal', personal)
        if sha(baseline) != initial:
            raise ValueError('Personal learning modified baseline')
    print('Downloaded production CLI/model smoke test passed outside the repository.')


if __name__ == '__main__':
    main()
