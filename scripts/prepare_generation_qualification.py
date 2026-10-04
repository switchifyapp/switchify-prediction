#!/usr/bin/env python3
"""Fetch checksum-pinned qualification inputs at build time, then convert locally."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile

from neural_bundle import ROOT, assemble, verify


def fetch(path, pin):
    if path.exists():
        verify(path, pin)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        downloaded = Path(temp) / 'input'
        with urllib.request.urlopen(pin['url'], timeout=300) as response, downloaded.open('wb') as out:
            remaining = pin['bytes'] + 1
            while remaining:
                chunk = response.read(min(65536, remaining))
                if not chunk:
                    break
                out.write(chunk)
                remaining -= len(chunk)
        verify(downloaded, pin)
        downloaded.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quantize', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.output / 'source'
    pins = json.loads((ROOT / 'neural/source-manifest.json').read_bytes())
    for name, pin in pins['files'].items():
        fetch(source / name, pin)
    model = args.output / 'model.gguf'
    if not model.exists():
        subprocess.run([str(args.quantize.resolve()), str(source.resolve()), str(model.resolve())], check=True)
    manifest = json.loads((ROOT / 'neural/model-bundle.json').read_bytes())
    verify(model, manifest['files']['model.gguf'])
    bundle = args.output / 'bundle'
    if not bundle.exists():
        assemble(source, model, bundle)
    else:
        for name, pin in manifest['files'].items():
            verify(bundle / name, pin)
        shutil.copyfile(ROOT / 'neural/model-bundle.json', bundle / 'model-bundle.json')
    baseline = json.loads((ROOT / 'neural/fixtures/baseline-pin.json').read_bytes())
    archive = args.output / 'baseline.zip'
    fetch(archive, baseline['archive'])
    database = args.output / 'english.sqlite'
    with zipfile.ZipFile(archive) as zipped:
        names = [n for n in zipped.namelist() if n.endswith('/english.sqlite') or n == 'english.sqlite']
        if len(names) != 1 or zipped.getinfo(names[0]).file_size != baseline['database']['bytes']:
            raise ValueError('Invalid baseline archive')
        with zipped.open(names[0]) as input_file, database.open('wb') as output_file:
            shutil.copyfileobj(input_file, output_file)
    verify(database, baseline['database'])


if __name__ == '__main__':
    main()
