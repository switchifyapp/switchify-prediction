#!/usr/bin/env python3
"""Collect notices from exact Cargo.lock-resolved crates for one binary target."""
import json
from pathlib import Path
import subprocess

# An added licence expression requires an explicit redistribution review.
PERMITTED = {
    'MIT', 'MIT OR Apache-2.0', 'Apache-2.0 OR MIT', 'MIT/Apache-2.0',
    'Zlib', 'Zlib OR Apache-2.0 OR MIT', 'Unlicense OR MIT',
    'Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT',
    '(MIT OR Apache-2.0) AND Unicode-3.0',
}


def crate_notices(package):
    if package['license'] not in PERMITTED:
        raise ValueError(f"Review new dependency licence: {package['name']} {package['license']}")
    root = Path(package['manifest_path']).parent
    files = {p for p in root.iterdir() if p.is_file() and
             p.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYRIGHT', 'COPYING', 'NOTICE'))}
    if package.get('license_file'):
        files.add(root / package['license_file'])
    if not files:
        raise ValueError(f"Dependency has no redistributable notice: {package['name']}")
    sections = [f"## {package['name']} {package['version']}\n\n"
                f"Declared licence: {package['license']}\n\n"
                f"Source: https://crates.io/crates/{package['name']}/{package['version']}\n"]
    for path in sorted(files, key=lambda p: p.name):
        sections.append(f'### {path.name}\n\n' + path.read_text(encoding='utf-8').strip() + '\n')
    if package['name'] == 'libsqlite3-sys':
        source = (root / 'sqlite3/sqlite3.c').read_text(encoding='utf-8')
        # Preserve the upstream public-domain blessing verbatim from bundled code.
        start = source.index('** The author disclaims copyright')
        start = source.rfind('/*', 0, start)
        end = source.index('*/', start) + 2
        sections.append('### Bundled SQLite public-domain notice\n\n' + source[start:end] + '\n')
    return '\n'.join(sections)


def generate(root, target):
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--locked', '--format-version', '1', '--filter-platform', target], cwd=root))
    packages = sorted((p for p in metadata['packages'] if p['source'] is not None),
                      key=lambda p: (p['name'], p['version']))
    if not packages:
        raise ValueError('No dependency metadata')
    return ('# Third-party notices\n\n'
            f'Target: {target}. Generated from Cargo.lock. Includes resolved build dependencies\n'
            'as well as runtime crates. Alternative licences are retained as supplied; this\n'
            'notice does not relicense those components. Rust library notices accompany this file.\n\n'
            + '\n'.join(crate_notices(p) for p in packages))
