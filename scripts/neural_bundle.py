#!/usr/bin/env python3
"""Assemble a local pinned model bundle. Never uploads or downloads assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def verify(path, pin):
    if path.stat().st_size != pin['bytes'] or hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != pin['sha256']:
        raise ValueError(f'Pinned file mismatch: {path.name}')


def assemble(source, gguf, output):
    manifest = json.loads((ROOT / 'neural/model-bundle.json').read_bytes())
    pins = json.loads((ROOT / 'neural/source-manifest.json').read_bytes())
    for name, pin in pins['files'].items():
        verify(source / name, pin)
    paths = {'model.gguf': gguf, 'MODEL_CARD.md': source / 'README.md',
             'MODEL_LICENSE.txt': ROOT / 'neural/MODEL_LICENSE.txt',
             'config.json': source / 'config.json', 'tokenizer.json': source / 'tokenizer.json'}
    for name, path in paths.items():
        verify(path, manifest['files'][name])
    if output.exists():
        raise ValueError('Output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / 'bundle'
        stage.mkdir()
        for name, path in paths.items():
            shutil.copyfile(path, stage / name)
            verify(stage / name, manifest['files'][name])
        shutil.copyfile(ROOT / 'neural/model-bundle.json', stage / 'model-bundle.json')
        shutil.copyfile(ROOT / 'neural/source-manifest.json', stage / 'source-manifest.json')
        shutil.copyfile(ROOT / 'neural/worker/src/bin/quantize.rs', stage / 'quantize.rs')
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--gguf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assemble(args.source, args.gguf, args.output)


if __name__ == '__main__':
    main()
