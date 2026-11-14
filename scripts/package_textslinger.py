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
PYTHON_LICENSES = ROOT / 'textslinger/licenses/python-build-standalone'
# The bridge never imports these parts of the standard library.
TRIM_STDLIB = ['test', 'tkinter', 'idlelib', 'ensurepip', 'lib2to3', 'turtledemo', 'pydoc_data',
               'unittest/test', 'ctypes/test', 'sqlite3/test', 'distutils/tests', 'site-packages']
# A fixed timestamp keeps archive metadata independent of build time. Compiled
# binaries are not byte-for-byte reproducible.
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


class Mismatch(Exception):
    """A download whose size or hash differs from its pin."""


def fetch_once(url, partial, digest, size):
    with urllib.request.urlopen(url, timeout=300) as response, open(partial, 'wb') as file:
        received = 0
        while chunk := response.read(1 << 20):
            received += len(chunk)
            if size is not None and received > size:
                raise Mismatch(f'{url} is larger than pinned')
            file.write(chunk)
    if (size is not None and received != size) or sha256(partial) != digest:
        raise Mismatch(f'{url} does not match its pinned size and hash')


def download(urls, path, digest, size=None, attempts=3, delay=10.0):
    """Download to path unless a verified copy exists.

    Tries each URL in order, retrying network errors, and accepts only bytes
    that match the pinned size and hash. A mismatching source is not retried.
    """
    import time
    if path.exists() and sha256(path) == digest:
        return path
    partial = path.with_name(path.name + '.partial')
    failures = []
    for url in [urls] if isinstance(urls, str) else urls:
        for attempt in range(attempts):
            try:
                fetch_once(url, partial, digest, size)
                partial.replace(path)
                return path
            except Mismatch as error:
                failures.append(str(error))
                break
            except OSError as error:  # URLError, HTTPError and timeouts.
                failures.append(f'{url}: {error}')
                if attempt + 1 < attempts:
                    print(f'retrying {url} after: {error}', flush=True)
                    time.sleep(delay * (attempt + 1))
            finally:
                partial.unlink(missing_ok=True)
    raise SystemExit('Download failed:\n  ' + '\n  '.join(failures))


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
    # KenLM's setup.py links zlib, bzip2 and xz whenever a test compile with
    # g++ finds them, which picks up whatever the build host has installed
    # (Homebrew's xz on Intel Macs). A failing g++ first on PATH turns those
    # checks off; the binary model needs none of them.
    shim = work / 'no-gxx'
    shutil.rmtree(shim, ignore_errors=True)
    shim.mkdir()
    (shim / 'g++').write_bytes(b'#!/bin/sh\nexit 1\n')
    (shim / 'g++').chmod(0o755)
    env = dict(os.environ, MAX_ORDER=str(kenlm['max_order']), CMAKE_POLICY_VERSION_MINIMUM='3.5')
    env['PATH'] = os.pathsep.join([str(shim), str(venv_python.parent), env['PATH']])
    if target.endswith('windows-msvc'):
        # Link the C++ runtime statically, so the extension needs no
        # MSVCP140.dll from a Visual C++ Redistributable.
        env['_CL_'] = '/MT'
    run([venv_python, '-m', 'pip', 'wheel', '--no-cache-dir', '--no-build-isolation', '--no-deps',
         '-w', wheels, source], env=env)
    built = sorted(wheels.glob('kenlm-0.2.0-*.whl'))
    if len(built) != 1:
        raise SystemExit('The KenLM wheel was not built')
    licenses = work / 'kenlm-licenses'
    shutil.rmtree(licenses, ignore_errors=True)
    licenses.mkdir()
    with tarfile.open(source) as tar:
        def member(name):
            return tar.extractfile(f'kenlm-0.2.0/{name}').read()
        for name in ('LICENSE', 'COPYING', 'COPYING.3', 'COPYING.LESSER.3'):
            (licenses / name).write_bytes(member(name))
        # Code compiled into the extension under its own licences.
        (licenses / 'double-conversion-LICENSE').write_bytes(member('util/double-conversion/LICENSE'))
        itoa = member('util/integer_to_string.cc').decode('utf-8')
        notice = itoa[itoa.index('/*') + 2:itoa.index('*/')].strip()
        (licenses / 'integer_to_string-NOTICE').write_text(notice + '\n', encoding='utf-8')
        # util/string_piece is Google's BSD-3-Clause code, in // comments.
        lines = member('util/string_piece.hh').decode('utf-8').splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith('// Copyright'))
        last = next(i for i in range(first, len(lines)) if 'SUCH DAMAGE.' in lines[i])
        notice = '\n'.join(line[3:] if line.startswith('// ') else line[2:] for line in lines[first:last + 1])
        (licenses / 'string_piece-LICENSE').write_text(notice + '\n', encoding='utf-8')
    # LGPL: ship the exact source the library was built from.
    shutil.copyfile(source, licenses / source.name)
    return built[0], licenses


def trim(runtime, target):
    paths = layout(target)
    for name in TRIM_STDLIB:
        shutil.rmtree(runtime / paths['stdlib'] / name, ignore_errors=True)
    if target.endswith('windows-msvc'):
        for name in ('include', 'libs', 'Scripts', 'tcl'):
            shutil.rmtree(runtime / name, ignore_errors=True)
        (runtime / 'pythonw.exe').unlink(missing_ok=True)
        for pattern in ('_tkinter.pyd', 'tcl*.dll', 'tk*.dll', '_test*.pyd', '_ctypes_test.pyd'):
            for path in (runtime / 'DLLs').glob(pattern):
                path.unlink()
        # Debug symbols are large and not needed to run.
        for path in runtime.rglob('*.pdb'):
            path.unlink()
    else:
        for name in ('include', 'share', 'lib/pkgconfig', 'lib/python3.10/config-3.10-darwin'):
            shutil.rmtree(runtime / name, ignore_errors=True)
        for pattern in ('tcl[89]*', 'tk[89]*', 'itcl*', 'thread*', 'libtcl*', 'libtk*'):
            for path in (runtime / 'lib').glob(pattern):
                shutil.rmtree(path) if path.is_dir() else path.unlink()
        # The interpreter links Python statically and extensions resolve its
        # symbols from it, so the shared library is unused. The dependency
        # check fails the build if anything still loads it.
        (runtime / 'lib/libpython3.10.dylib').unlink(missing_ok=True)
        for pattern in ('_test*', '_ctypes_test*', '_tkinter*'):
            for path in (runtime / 'lib/python3.10/lib-dynload').glob(pattern):
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
        if path.is_file() and (path.suffix in ('.exp', '.whl') or path.name == '.lock'):
            path.unlink()
    # Static libraries for building extensions, and numpy's own test suite.
    for path in list(site.rglob('*.lib')):
        path.unlink()
    for path in sorted(site.glob('numpy/**/tests'), reverse=True):
        shutil.rmtree(path, ignore_errors=True)
    # KenLM's setup.py also builds a standalone library that the Python
    # extension does not load.
    for name in ('kenlm.dll', 'libkenlm.dylib'):
        (site / name).unlink(missing_ok=True)
    shutil.rmtree(site / 'bin', ignore_errors=True)
    # Installer records that hold build paths and times.
    for info in site.glob('*.dist-info'):
        for name in ('direct_url.json', 'uv_cache.json', 'REQUESTED'):
            (info / name).unlink(missing_ok=True)
    for path in sorted(runtime.rglob('__pycache__'), reverse=True):
        shutil.rmtree(path, ignore_errors=True)
    for info in site.glob('*.dist-info'):
        prune_record(info / 'RECORD', site)


def prune_record(record, site):
    """Drop RECORD entries for files that were removed from the runtime."""
    import csv
    import io
    if not record.exists():
        return
    rows = [row for row in csv.reader(io.StringIO(record.read_text(encoding='utf-8')))
            if row and (site / row[0]).exists()]
    output = io.StringIO()
    csv.writer(output, lineterminator='\n').writerows(rows)
    record.write_text(output.getvalue(), encoding='utf-8')


def write_notices(pins, runtime, target, kenlm_licenses):
    notices = runtime / 'notices'
    notices.mkdir()
    shutil.copytree(kenlm_licenses, notices / 'kenlm')
    shutil.copytree(PYTHON_LICENSES, notices / 'python')
    model = pins['model']
    packages = pins['packages']
    (notices / 'NOTICE.md').write_text(f"""# TextSlinger runtime notices

Built from switchify-prediction for Switchify PC. Licence texts for the Python
packages are in their `*.dist-info` folders under `{layout(target)['site']}`.

| Component | Version | Licence | Source |
|---|---|---|---|
| CPython (python-build-standalone {pins['python']['release']}) | {pins['python']['version']} | PSF-2.0; libraries built into it (OpenSSL, libffi, SQLite, xz, zlib, bzip2, expat, mpdecimal and others) keep their own licences, in `python/` | https://github.com/astral-sh/python-build-standalone |
| textslinger | {packages['textslinger']['version']} | MIT | https://github.com/kdv123/textslinger |
| numpy | {packages['numpy']['version']} | BSD-3-Clause and bundled licences | https://pypi.org/project/numpy/ |
| regex | {packages['regex']['version']} | Apache-2.0 | https://pypi.org/project/regex/ |
| KenLM | {pins['kenlm']['version']} | LGPL-2.1-or-later; bundled double-conversion (BSD-3-Clause), string_piece (BSD-3-Clause) and integer_to_string (MIT) notices are in `kenlm/`; murmur_hash is public domain | {pins['kenlm']['url']} |
| {model['name']} | | {model['permission']} | {model['source']}, {model['url']} |

KenLM is built from the unmodified source above with `MAX_ORDER={pins['kenlm']['max_order']}`, as the
separate `kenlm` Python extension module, which you may replace with your own build. That
source is included as `kenlm/kenlm-{pins['kenlm']['version']}.tar.gz`.

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


# DLLs every supported Windows provides (msvcrt.dll is the system C runtime,
# not a redistributable). Anything else must ship in the runtime.
WINDOWS_SYSTEM_DLLS = {
    'advapi32.dll', 'bcrypt.dll', 'cabinet.dll', 'comctl32.dll', 'comdlg32.dll', 'crypt32.dll',
    'dbghelp.dll', 'gdi32.dll', 'imm32.dll', 'iphlpapi.dll', 'kernel32.dll', 'msi.dll',
    'msvcrt.dll', 'mswsock.dll', 'normaliz.dll', 'ntdll.dll', 'ole32.dll', 'oleaut32.dll', 'psapi.dll',
    'rpcrt4.dll', 'secur32.dll', 'setupapi.dll', 'shell32.dll', 'shlwapi.dll', 'sspicli.dll',
    'user32.dll', 'userenv.dll', 'version.dll', 'winhttp.dll', 'winmm.dll', 'wintrust.dll',
    'ws2_32.dll',
}


def pe_imports(data):
    """DLL names a PE file imports, including delay-loaded ones."""
    import struct
    if data[:2] != b'MZ':
        return []
    pe = struct.unpack_from('<I', data, 0x3C)[0]
    if data[pe:pe + 4] != b'PE\0\0':
        return []
    sections, optional_size = struct.unpack_from('<H12xH', data, pe + 6)
    optional = pe + 24
    magic = struct.unpack_from('<H', data, optional)[0]
    directories = optional + (112 if magic == 0x20B else 96)
    table = optional + optional_size
    ranges = [struct.unpack_from('<8xIIII', data, table + 40 * i) for i in range(sections)]

    def offset(rva):
        for size, address, raw_size, raw in ranges:
            if address <= rva < address + max(size, raw_size):
                return raw + rva - address
        raise ValueError(f'RVA {rva:#x} outside every section')

    def name(rva):
        start = offset(rva)
        return data[start:data.index(b'\0', start)].decode('ascii').lower()

    names = []
    for index, entry_size, name_at in ((1, 20, 12), (13, 32, 4)):
        rva, size = struct.unpack_from('<II', data, directories + 8 * index)
        if not rva:
            continue
        position = offset(rva)
        while True:
            entry = data[position:position + entry_size]
            if len(entry) < entry_size or not any(entry):
                break
            names.append(name(struct.unpack_from('<I', entry, name_at)[0]))
            position += entry_size
    return names


def windows_search_sets(files, runtime):
    """For each file, the DLL names the loader can find for it in the runtime.

    Python loads extensions searching their own folder and the interpreter's;
    wheels add their *.libs folder with os.add_dll_directory.
    """
    names = {}
    for path in files:
        names.setdefault(path.parent, set()).add(path.name.lower())
    shared = set(names.get(runtime, set()))
    for folder, contents in names.items():
        if folder.name.endswith('.libs'):
            shared |= contents
    return {path: shared | names[path.parent] for path in files}


def disallowed_windows(imports, present):
    return sorted({dll for dll in imports if dll not in present and dll not in WINDOWS_SYSTEM_DLLS
                   and not dll.startswith(('api-ms-win-', 'ext-ms-'))})


MACHO_LOAD_COMMANDS = ('LC_LOAD_DYLIB', 'LC_LOAD_WEAK_DYLIB', 'LC_REEXPORT_DYLIB',
                       'LC_LOAD_UPWARD_DYLIB', 'LC_LAZY_LOAD_DYLIB')
MACHO_MAGICS = (b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xca\xfe\xba\xbe', b'\xca\xfe\xba\xbf')


def parse_otool(output):
    """Dylib load paths and rpaths from `otool -l` output, for every architecture."""
    dependencies, rpaths, command = [], [], None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith('cmd '):
            command = line.split()[1]
        elif command in MACHO_LOAD_COMMANDS and line.startswith('name '):
            dependencies.append(line[5:].rsplit(' (offset', 1)[0])
        elif command == 'LC_RPATH' and line.startswith('path '):
            rpaths.append(line[5:].rsplit(' (offset', 1)[0])
    return dependencies, rpaths


def unresolved_macos(dependencies, rpaths, path, runtime):
    """Dependencies that are neither the OS's nor a file inside the runtime.

    Bundle-relative paths must resolve to a file in the runtime; an @rpath
    dependency resolves through the file's own rpaths.
    """
    executable_dir = runtime / 'bin'

    def expand(prefix_path):
        expanded = prefix_path.replace('@loader_path', str(path.parent)).replace('@executable_path', str(executable_dir))
        return Path(os.path.normpath(expanded))

    def inside(candidate):
        return candidate.is_file() and runtime.resolve() in candidate.resolve().parents

    problems = set()
    for dependency in dependencies:
        if dependency.startswith(('/usr/lib/', '/System/Library/')):
            continue
        if dependency.startswith(('@loader_path/', '@executable_path/')):
            if inside(expand(dependency)):
                continue
        elif dependency.startswith('@rpath/'):
            name = dependency[len('@rpath/'):]
            if any(inside(expand(rpath) / name) for rpath in rpaths):
                continue
        problems.add(dependency)
    return sorted(problems)


def check_native_dependencies(runtime, target):
    """Fail if a native file needs a library the runtime and the OS do not provide.

    The smoke test runs on the build host, where extra libraries can satisfy a
    dependency that a user's machine lacks.
    """
    problems = {}
    files = [p for p in runtime.rglob('*') if p.is_file()]
    if target.endswith('windows-msvc'):
        searchable = windows_search_sets(files, runtime)
        for path in files:
            if path.suffix.lower() in ('.exe', '.dll', '.pyd'):
                if missing := disallowed_windows(pe_imports(path.read_bytes()), searchable[path]):
                    problems[path.relative_to(runtime).as_posix()] = missing
    else:
        for path in files:
            with open(path, 'rb') as file:
                magic = file.read(4)
            if magic in MACHO_MAGICS:
                output = subprocess.run(['otool', '-l', str(path)], check=True,
                                        capture_output=True, text=True).stdout
                if missing := unresolved_macos(*parse_otool(output), path, runtime):
                    problems[path.relative_to(runtime).as_posix()] = missing
    if problems:
        details = '\n'.join(f'  {path}: {", ".join(names)}' for path, names in sorted(problems.items()))
        raise SystemExit(f'Native files depend on libraries outside the runtime:\n{details}')
    print('native dependencies: only the runtime and the operating system', flush=True)


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
    check_native_dependencies(runtime, target)

    name = f'textslinger-runtime-{target}.zip'
    zipped = output / name
    write_zip(runtime, zipped)
    (output / f'{name}.sha256').write_text(f'{sha256(zipped)}  {name}\n')
    print(f'{zipped} ({zipped.stat().st_size / 1e6:.1f} MB)')
    return zipped


def fetch_model(model, work, destination, attempts=3, delay=10.0):
    """Download the gzip model once, then decompress and check the model's hash."""
    cached = work / model['name']
    if not (cached.exists() and sha256(cached) == model['sha256']):
        compressed = work / (model['name'] + '.gz')
        # Our own copy first, then the original source; both must match the pin.
        download(model['download_urls'], compressed, model['gzip_sha256'], model['gzip_bytes'],
                 attempts=attempts, delay=delay)
        with gzip.open(compressed) as source, open(cached, 'wb') as target:
            written = 0
            while chunk := source.read(1 << 20):
                written += len(chunk)
                if written > model['bytes']:
                    raise SystemExit('The model is larger than pinned')
                target.write(chunk)
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
