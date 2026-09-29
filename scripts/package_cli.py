#!/usr/bin/env python3
"""Package only the compiled CLI and explicit support files, then smoke test it."""
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile

from aac_experiment import sha, write_json

ROOT = Path(__file__).resolve().parent.parent


def main():
    compiler = subprocess.check_output(['rustc', '-vV'], text=True)
    host = next(line.split(': ', 1)[1] for line in compiler.splitlines() if line.startswith('host: '))
    executable = 'switchify-prediction' + ('.exe' if os.name == 'nt' else '')
    version = subprocess.check_output([str(ROOT / 'target/release' / executable), '--version'], text=True).strip().split()[-1]
    out = ROOT / 'artifacts/cli'
    out.mkdir(parents=True, exist_ok=True)
    name = f'switchify-prediction-{version}-{host}'
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp) / name
        stage.mkdir()
        shutil.copy2(ROOT / 'target/release' / executable, stage / executable)
        for path in ['LICENSE', 'Cargo.lock']:
            shutil.copyfile(ROOT / path, stage / path)
        shutil.copyfile(ROOT / 'scripts/verify_bundle.py', stage / 'verify_bundle.py')
        write_json(stage / 'BUILD.json', {'version': version, 'target': host, 'rustc': compiler,
                                         'platform': platform.platform(), 'libc': platform.libc_ver(),
                                         'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                         'model_id': 'en-aac-oanc-v1', 'signed': False})
        (stage / 'README.txt').write_text(
            'Switchify Prediction CLI. Bundled SQLite; no database server required.\n'
            'Run python verify_bundle.py to verify these files.\n'
            'Download and extract the separate English model into another directory.\n'
            'Run switchify-prediction validate --database /path/to/english.sqlite --production\n'
            'Run switchify-prediction predict --baseline /path/to/english.sqlite --before "I need" --prefix he\n'
            'These standalone CLI binaries are unsigned; they are not signed/notarized app installers.\n'
            'Personal databases are local and unencrypted. Do not put them in a distributed bundle.\n', encoding='utf-8')
        files = sorted(stage.iterdir())
        (stage / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files), encoding='utf-8')
        subprocess.run([str(stage / executable), '--version'], cwd=temp, check=True)
        archive = Path(shutil.make_archive(str(out / name), 'zip', temp, name))
        (out / (name + '.sha256')).write_text(f'{sha(archive)}  {archive.name}\n', encoding='utf-8')
    print(archive)


if __name__ == '__main__':
    main()
