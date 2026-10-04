#!/usr/bin/env python3
"""Package explicitly built workers and notices, without model assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import tomllib

import dependency_notices

ROOT = Path(__file__).resolve().parent.parent
# All additions have a permissive redistribution choice. Keep original notices.
NEURAL_LICENSES = {'Apache-2.0', 'Apache-2.0 / MIT', 'Apache-2.0/MIT',
                   'Apache-2.0 OR BSL-1.0', 'Apache-2.0 OR MIT OR Zlib',
                   'BSD-2-Clause OR Apache-2.0 OR MIT', 'Unicode-3.0',
                   'Unlicense/MIT', 'MIT OR Apache-2.0 OR LGPL-2.1-or-later'}


def notices(target):
    previous = dependency_notices.PERMITTED
    dependency_notices.PERMITTED = previous | NEURAL_LICENSES
    try:
        metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--locked', '--format-version', '1',
                                                      '--filter-platform', target], cwd=ROOT / 'neural'))
        sections = ['# Neural third-party notices\n\nExact lockfile dependencies, including build tools.\n']
        sources = json.loads((ROOT / 'neural/notices/sources.json').read_bytes())
        for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
            if not package['source']:
                continue
            extra = {('candle-nn','0.11.0'):['candle-LICENSE-MIT','candle-LICENSE-APACHE'],
                     ('candle-transformers','0.11.0'):['candle-LICENSE-MIT','candle-LICENSE-APACHE'],
                     ('pulp-wasm-simd-flag','0.1.1'):['pulp-LICENSE']}.get((package['name'],package['version']))
            if extra:
                sections.append(f"## {package['name']} {package['version']}\n\nDeclared license: {package['license']}\n")
                for name in extra:
                    path = ROOT / 'neural/notices' / name
                    if sha(path) != sources[name]['sha256']:
                        raise ValueError('Notice checksum mismatch')
                    sections.append(sources[name]['url']+'\n\n'+path.read_text(encoding='utf-8'))
            else:
                sections.append(dependency_notices.crate_notices(package))
            if package['name'] == 'onig_sys':
                path = Path(package['manifest_path']).parent / 'oniguruma/COPYING'
                sections.append('### Bundled Oniguruma\n\n'+path.read_text(encoding='utf-8'))
        return '\n'.join(sections)
    finally:
        dependency_notices.PERMITTED = previous


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--portable', type=Path, required=True, help='Release binary directory')
    parser.add_argument('--accelerated', type=Path, help='AVX2 release binary directory')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/neural-cli')
    args = parser.parse_args()
    rustc = subprocess.check_output(['rustc', '-vV'], text=True)
    target = next(line.split(': ',1)[1] for line in rustc.splitlines() if line.startswith('host: '))
    ext = '.exe' if os.name == 'nt' else ''
    version = tomllib.loads((ROOT / 'neural/Cargo.toml').read_text())['package']['version']
    name = 'switchify-prediction-neural-' + version + '-' + target
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp) / name
        stage.mkdir()
        for exe in ['switchify-prediction-neural', 'switchify-smol-worker']:
            shutil.copy2(args.portable / (exe+ext), stage / (exe+ext))
        if args.accelerated:
            if not target.startswith('x86_64-'):
                raise ValueError('Accelerated worker requires x86_64')
            shutil.copy2(args.accelerated / ('switchify-smol-worker'+ext), stage / ('switchify-smol-worker-avx2'+ext))
        for source, destination in [('LICENSE','LICENSE'), ('neural/Cargo.lock','Cargo.lock'),
                                    ('neural/README.md','README.md'), ('neural/SECURITY.md','SECURITY.md'),
                                    ('neural/qualification.json','QUALIFICATION.json'),
                                    ('docs/smol-production-results.md','QUALIFICATION.md'),
                                    ('scripts/verify_bundle.py','verify_bundle.py')]:
            shutil.copyfile(ROOT / source, stage / destination)
        (stage / 'THIRD_PARTY_NOTICES.md').write_text(notices(target), encoding='utf-8')
        sysroot = Path(subprocess.check_output(['rustc','--print','sysroot'], text=True).strip())
        shutil.copyfile(sysroot / 'share/doc/rust/COPYRIGHT-library.html', stage / 'RUST_LIBRARY_COPYRIGHT.html')
        (stage / 'BUILD.json').write_text(json.dumps({'version':version,'target':target, 'rustc':rustc,
            'platform':platform.platform(), 'libc':platform.libc_ver(), 'signed':False,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'model_id':'smollm2-135m-q8-v1', 'worker_protocol':2, 'generation_policy':json.loads((ROOT / 'neural/model-bundle.json').read_bytes())['policy']['generation'], 'accelerated_requires':['avx2','fma','f16c'] if args.accelerated else [],
            'production_qualified':json.loads((ROOT / 'neural/qualification.json').read_bytes())['production_qualified'],
            'model_assets_included':False}, indent=2)+'\n',encoding='utf-8')
        (stage / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(stage.iterdir())), encoding='utf-8')
        subprocess.run([str(stage / ('switchify-prediction-neural'+ext)), '--version'], check=True)
        archive = Path(shutil.make_archive(str(args.output / name), 'zip', temp, name))
        (args.output / (name+'.sha256')).write_text(f'{sha(archive)}  {archive.name}\n',encoding='utf-8')
    print(archive)


if __name__ == '__main__':
    main()
