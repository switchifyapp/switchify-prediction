#!/usr/bin/env python3
"""Fetch only checksum-pinned public source files; no personal data discovery."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
manifest = json.loads((ROOT / 'source-manifest.json').read_text())
(ROOT / 'data').mkdir(exist_ok=True)
for key, name in [('corpus', 'en.txt'), ('upstream_manifest', 'SOURCES.json')]:
    item = manifest[key]
    target = ROOT / 'data' / name
    data = target.read_bytes() if target.exists() else b''
    if hashlib.sha256(data).hexdigest() != item['sha256']:
        with urllib.request.urlopen(item['url'], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise SystemExit(f'Checksum mismatch: {name}')
        target.write_bytes(data)
    print(f'Verified {name}')
