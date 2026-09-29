#!/usr/bin/env python3
"""Fetch only checksum-pinned public sources; never discover personal data."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent


def fetch():
    manifest = json.loads((ROOT / 'source-manifest.json').read_text())
    sources = [(manifest['corpus'], 'en.txt'),
               (manifest['upstream_manifest'], 'SOURCES.json')]
    sources += [(item, 'taskmaster/' + name)
                for name, item in manifest['taskmaster']['files'].items()]
    for item, name in sources:
        target = ROOT / 'data' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = target.read_bytes() if target.exists() else b''
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            with urllib.request.urlopen(item['url'], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise SystemExit(f'Checksum mismatch: {name}')
            target.write_bytes(data)
        print(f'Verified {name}')


if __name__ == '__main__':
    fetch()
