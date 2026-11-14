#!/usr/bin/env python3
"""Build the TextSlinger runtime that Switchify PC bundles, for this host.

The runtime is standalone CPython with numpy, regex, KenLM and textslinger,
plus the binary n-gram model. Every download is checked against
textslinger/runtime.json. Build tools: uv, git, a C++ compiler.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PINS = ROOT / 'textslinger/runtime.json'
PATCH = ROOT / 'textslinger/lazy-torch-imports.patch'
# The bridge never imports these parts of the standard library.
TRIM_STDLIB = ['test', 'tkinter', 'idlelib', 'ensurepip', 'lib2to3', 'turtledemo', 'pydoc_data',
               'unittest/test', 'ctypes/test', 'sqlite3/test', 'distutils/tests', 'site-packages']
# A fixed timestamp keeps the archive's bytes independent of build time.
ZIP_TIME = (2026, 1, 1, 0, 0, 0)


def host_target(system=None, machine=None):
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == 'Windows' and machine in ('amd64', 'x86_64'):
        return 'x86_64-pc-windows-msvc'
    if system == 'Darwin' and machine in ('arm64', 'aarch64'):
        return 'aarch64-apple-darwin'
    if system == 'Darwin' and machine == 'x86_64':
        return 'x86_64-apple-darwin'
    raise SystemExit(f'No TextSlinger runtime for {system} {machine}')


def layout(target):
    """Paths inside the runtime, relative to its root."""
    if target.endswith('windows-msvc'):
        return {'python': 'python.exe', 'stdlib': 'Lib', 'site': 'Lib/site-packages'}
    return {'python': 'bin/python3.10', 'stdlib': 'lib/python3.10', 'site': 'lib/python3.10/site-packages'}


def requirements(pins, target):
    """Hash-checked requirement lines for the runtime's packages on target."""
    lines = []
    for name, package in sorted(pins['packages'].items()):
        hashes = package['sha256']
        digest = hashes.get(target) or hashes.get('any')
        if not digest:
            raise SystemExit(f'No pinned {name} for {target}')
        lines.append(f"{name}=={package['version']} --hash=sha256:{digest}")
    return lines


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path, digest, size=None):
    """Download url to path unless a verified copy exists. Fails on any mismatch."""
    if path.exists() and sha256(path) == digest:
        return path
    partial = path.with_name(path.name + '.partial')
    with urllib.request.urlopen(url, timeout=300) as response, open(partial, 'wb') as file:
        received = 0
        while chunk := response.read(1 << 20):
            received += len(chunk)
            if size is not None and received > size:
                raise SystemExit(f'{url} is larger than pinned')
            file.write(chunk)
    if (size is not None and received != size) or sha256(partial) != digest:
        partial.unlink()
        raise SystemExit(f'{url} does not match its pinned hash')
    partial.replace(path)
    return path


def run(command, **kwargs):
    print('+', ' '.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def safe_extract(archive, destination):
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter='data')


def build_kenlm(pins, target, uv, base_python, work):
    """Build the KenLM wheel from its hash-checked source with the model's order."""
    kenlm = pins['kenlm']
    source = download(kenlm['url'], work / 'kenlm-0.2.0.tar.gz', kenlm['sha256'])
    wheels = work / 'kenlm-wheel'
    shutil.rmtree(wheels, ignore_errors=True)
    venv = work / 'build-venv'
    shutil.rmtree(venv, ignore_errors=True)
    run([uv, 'venv', '--python', base_python, venv])
    venv_python = venv / ('Scripts/python.exe' if target.endswith('windows-msvc') else 'bin/python')
    tools = work / 'build-tools.txt'
    tools.write_text('\n'.join(pins['build_tools']) + '\n')
    run([uv, 'pip', 'install', '--python', venv_python, '--require-hashes', '-r', tools])
    env = dict(os.environ, MAX_ORDER=str(kenlm['max_order']), CMAKE_POLICY_VERSION_MINIMUM='3.5')
    env['PATH'] = str(venv_python.parent) + os.pathsep + env['PATH']
    if target.endswith('apple-darwin'):
        env['MACOSX_DEPLOYMENT_TARGET'] = pins['macos_deployment_target']
    run([venv_python, '-m', 'pip', 'wheel', '--no-cache-dir', '--no-build-isolation', '--no-deps',
         '-w', wheels, source], env=env)
    built = sorted(wheels.glob('kenlm-0.2.0-*.whl'))
    if len(built) != 1:
        raise SystemExit('The KenLM wheel was not built')
    licenses = work / 'kenlm-licenses'
    shutil.rmtree(licenses, ignore_errors=True)
    licenses.mkdir()
    with tarfile.open(source) as tar:
        for name in ('LICENSE', 'COPYING', 'COPYING.3', 'COPYING.LESSER.3'):
            member = tar.extractfile(f'kenlm-0.2.0/{name}')
            (licenses / name).write_bytes(member.read())
    return built[0], licenses


def trim(runtime, target):
    paths = layout(target)
    for name in TRIM_STDLIB:
        shutil.rmtree(runtime / paths['stdlib'] / name, ignore_errors=True)
    if target.endswith('windows-msvc'):
        for name in ('include', 'libs', 'Scripts', 'tcl'):
            shutil.rmtree(runtime / name, ignore_errors=True)
        (runtime / 'pythonw.exe').unlink(missing_ok=True)
        for pattern in ('_tkinter.pyd', 'tcl*.dll', 'tk*.dll', '_test*.pyd'):
            for path in (runtime / 'DLLs').glob(pattern):
                path.unlink()
    else:
        for name in ('include', 'share', 'lib/pkgconfig', 'lib/python3.10/config-3.10-darwin'):
            shutil.rmtree(runtime / name, ignore_errors=True)
        for pattern in ('tcl8*', 'tk8*', 'itcl*', 'thread*', 'libtcl*', 'libtk*'):
            for path in (runtime / 'lib').glob(pattern):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        for path in (runtime / 'lib/python3.10/lib-dynload').glob('_test*'):
            path.unlink()
        for path in (runtime / 'lib/python3.10/lib-dynload').glob('_tkinter*'):
            path.unlink()
        # Keep the real interpreter only; zip archives do not hold symlinks.
        for path in (runtime / 'bin').iterdir():
            if path.name != 'python3.10':
                path.unlink()
    (runtime / paths['site']).mkdir(parents=True, exist_ok=True)


def apply_patch(site, runtime):
    # git apply inside a repository resolves paths from its root and skips
    # ignored files, so stop it from finding the repository above the runtime.
    env = dict(os.environ, GIT_CEILING_DIRECTORIES=str(runtime.resolve()))
    # The wheel's files use LF; a CRLF checkout of the patch would not apply.
    patch = PATCH.read_bytes().replace(b'\r\n', b'\n')
    print('+ git apply', PATCH.name, flush=True)
    subprocess.run(['git', 'apply', '--whitespace=nowarn', '-'], input=patch, cwd=site, env=env, check=True)
    if '_precision_to_torch' not in (site / 'textslinger/helpers.py').read_text():
        raise SystemExit('The lazy import patch was not applied')


def clean(runtime, site):
    for path in list(site.iterdir()):
        if path.is_file() and (path.suffix in ('.lib', '.exp', '.whl') or path.name == '.lock'):
            path.unlink()
    shutil.rmtree(site / 'bin', ignore_errors=True)
    for path in sorted(runtime.rglob('__pycache__'), reverse=True):
        shutil.rmtree(path, ignore_errors=True)


def write_notices(pins, runtime, target, kenlm_licenses):
    notices = runtime / 'notices'
    notices.mkdir()
    shutil.copytree(kenlm_licenses, notices / 'kenlm')
    python_license = runtime / ('LICENSE.txt' if target.endswith('windows-msvc') else 'lib/python3.10/LICENSE.txt')
    shutil.copyfile(python_license, notices / 'python-LICENSE.txt')
    model = pins['model']
    packages = pins['packages']
    (notices / 'NOTICE.md').write_text(f"""# TextSlinger runtime notices

Built from switchify-prediction for Switchify PC. Licence texts for the Python
packages are in their `*.dist-info` folders under `{layout(target)['site']}`.

| Component | Version | Licence | Source |
|---|---|---|---|
| CPython (python-build-standalone {pins['python']['release']}) | {pins['python']['version']} | PSF-2.0 and bundled licences, see `python-LICENSE.txt` | https://github.com/astral-sh/python-build-standalone |
| textslinger | {packages['textslinger']['version']} | MIT | https://github.com/kdv123/textslinger |
| numpy | {packages['numpy']['version']} | BSD-3-Clause and bundled licences | https://pypi.org/project/numpy/ |
| regex | {packages['regex']['version']} | Apache-2.0 | https://pypi.org/project/regex/ |
| KenLM | {pins['kenlm']['version']} | LGPL-2.1-or-later, with exceptions listed in `kenlm/LICENSE` | {pins['kenlm']['url']} |
| {model['name']} | | {model['permission']} | {model['source']}, {model['url']} |

KenLM is built from the unmodified source above with `MAX_ORDER={pins['kenlm']['max_order']}`, as a
separate library that you may replace with your own build.

textslinger {packages['textslinger']['version']} includes a small patch so that it imports PyTorch only for
the models that need it; see https://github.com/kdv123/textslinger/pull/2.
""")


def smoke_test(runtime, target, pins):
    """The runtime must predict with the model and never import PyTorch."""
    check = """
import sys
from textslinger import NGramLanguageModel
model = NGramLanguageModel(sys.argv[1], model_alphabet=tuple(sys.argv[2]))
words = [w.word for w in model.predict_words(left_context="please open the ",
    word_spelling_characters=tuple("abcdefghijklmnopqrstuvwxyz'"),
    word_ending_characters=(" ",), nbest=5).top(5)]
assert "door" in words, words
assert not any(name.split(".")[0] == "torch" for name in sys.modules), "torch was imported"
print("smoke test:", words)
"""
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / 'check.py'
        script.write_text(check)
        run([runtime / layout(target)['python'], '-I', script,
             runtime / 'model' / pins['model']['name'], pins['model']['alphabet']])


def write_zip(runtime, archive):
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(p for p in runtime.rglob('*') if p.is_file()):
            info = zipfile.ZipInfo(path.relative_to(runtime).as_posix(), ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            # Keep executables executable for the macOS interpreter.
            mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            with open(path, 'rb') as file:
                zf.writestr(info, file.read())


def package(output, work, uv='uv', target=None):
    pins = json.loads(PINS.read_text())
    target = target or host_target()
    python = pins['python']['targets'][target]
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    archive = download(python['url'], work / f'python-{target}.tar.gz', python['sha256'], python['bytes'])
    base = work / 'base'
    shutil.rmtree(base, ignore_errors=True)
    safe_extract(archive, base)
    base_python = base / 'python' / layout(target)['python']

    kenlm_wheel, kenlm_licenses = build_kenlm(pins, target, uv, base_python, work)

    runtime = work / 'runtime'
    shutil.rmtree(runtime, ignore_errors=True)
    shutil.copytree(base / 'python', runtime, symlinks=True)
    trim(runtime, target)
    site = runtime / layout(target)['site']
    reqs = work / 'requirements.txt'
    reqs.write_text('\n'.join(requirements(pins, target)) + '\n')
    run([uv, 'pip', 'install', '--python', base_python, '--no-deps', '--require-hashes',
         '--target', site, '-r', reqs])
    run([uv, 'pip', 'install', '--python', base_python, '--no-deps', '--target', site, kenlm_wheel])
    apply_patch(site, runtime)
    clean(runtime, site)

    model = pins['model']
    model_dir = runtime / 'model'
    model_dir.mkdir()
    fetch_model(model, work, model_dir / model['name'])
    write_notices(pins, runtime, target, kenlm_licenses)
    (runtime / 'manifest.json').write_text(json.dumps({
        'format_version': 1,
        'target': target,
        'python': layout(target)['python'],
        'model': f"model/{model['name']}",
        'model_alphabet': model['alphabet'],
        'versions': {
            'python': pins['python']['version'],
            'textslinger': pins['packages']['textslinger']['version'],
            'kenlm': pins['kenlm']['version'],
            'numpy': pins['packages']['numpy']['version'],
        },
    }, indent=2) + '\n')
    smoke_test(runtime, target, pins)
    clean(runtime, site)

    name = f'textslinger-runtime-{target}.zip'
    zipped = output / name
    write_zip(runtime, zipped)
    (output / f'{name}.sha256').write_text(f'{sha256(zipped)}  {name}\n')
    print(f'{zipped} ({zipped.stat().st_size / 1e6:.1f} MB)')
    return zipped


def fetch_model(model, work, destination):
    """Download the gzip model once, then decompress and check the model's hash."""
    cached = work / model['name']
    if not (cached.exists() and sha256(cached) == model['sha256']):
        compressed = work / (model['name'] + '.gz')
        with urllib.request.urlopen(model['url'], timeout=300) as response, open(compressed, 'wb') as file:
            received = 0
            while chunk := response.read(1 << 20):
                received += len(chunk)
                if received > model['gzip_bytes']:
                    raise SystemExit('The model download is larger than pinned')
                file.write(chunk)
        if received != model['gzip_bytes']:
            raise SystemExit('The model download is not the pinned size')
        with gzip.open(compressed) as source, open(cached, 'wb') as target:
            shutil.copyfileobj(source, target)
        compressed.unlink()
        if sha256(cached) != model['sha256']:
            cached.unlink()
            raise SystemExit('The model does not match its pinned hash')
    shutil.copyfile(cached, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/textslinger')
    parser.add_argument('--work', type=Path, default=ROOT / 'target/textslinger')
    parser.add_argument('--uv', default=shutil.which('uv') or 'uv')
    args = parser.parse_args()
    package(args.output.resolve(), args.work.resolve(), args.uv)


if __name__ == '__main__':
    main()
