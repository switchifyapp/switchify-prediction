import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

import package_textslinger as pkg

TARGETS = ['x86_64-pc-windows-msvc', 'aarch64-apple-darwin', 'x86_64-apple-darwin']


class Pins(unittest.TestCase):
    def setUp(self):
        self.pins = json.loads(pkg.PINS.read_text())

    def test_every_target_has_python_and_package_pins(self):
        self.assertEqual(sorted(self.pins['python']['targets']), sorted(TARGETS))
        for target, python in self.pins['python']['targets'].items():
            self.assertIn(target, python['url'])
            self.assertIn(self.pins['python']['release'], python['url'])
            self.assertRegex(python['sha256'], r'^[0-9a-f]{64}$')
            lines = pkg.requirements(self.pins, target)
            self.assertEqual([line.split('==')[0] for line in lines], ['numpy', 'regex', 'textslinger'])
            for line in lines:
                self.assertRegex(line, r'==[^ ]+ --hash=sha256:[0-9a-f]{64}$')

    def test_build_tools_and_sources_are_hash_pinned(self):
        for line in self.pins['build_tools']:
            self.assertRegex(line, r'^[a-z]+==[0-9.]+ --hash=sha256:[0-9a-f]{64}$')
        self.assertRegex(self.pins['kenlm']['sha256'], r'^[0-9a-f]{64}$')
        self.assertRegex(self.pins['model']['sha256'], r'^[0-9a-f]{64}$')
        self.assertEqual(self.pins['kenlm']['max_order'], 12)

    def test_model_alphabet_has_no_duplicates(self):
        alphabet = self.pins['model']['alphabet']
        self.assertEqual(len(set(alphabet)), len(alphabet))
        self.assertIn(' ', alphabet)

    def test_unknown_target_has_no_requirements(self):
        with self.assertRaises(SystemExit):
            pkg.requirements(self.pins, 'aarch64-pc-windows-msvc')

    def test_patch_targets_the_pinned_release(self):
        patch = pkg.PATCH.read_text()
        self.assertIn('a/textslinger/helpers.py', patch)
        self.assertIn('a/textslinger/__init__.py', patch)
        self.assertNotIn('\r\n', patch)
        self.assertEqual(self.pins['packages']['textslinger']['version'], '0.4.0')


class Hosts(unittest.TestCase):
    def test_host_targets(self):
        self.assertEqual(pkg.host_target('Windows', 'AMD64'), 'x86_64-pc-windows-msvc')
        self.assertEqual(pkg.host_target('Darwin', 'arm64'), 'aarch64-apple-darwin')
        self.assertEqual(pkg.host_target('Darwin', 'x86_64'), 'x86_64-apple-darwin')
        with self.assertRaises(SystemExit):
            pkg.host_target('Windows', 'ARM64')
        with self.assertRaises(SystemExit):
            pkg.host_target('Linux', 'x86_64')


def touch(root, *paths):
    for path in paths:
        target = Path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('x')


class Trim(unittest.TestCase):
    def test_windows_keeps_the_interpreter_and_drops_unused_parts(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, 'python.exe', 'pythonw.exe', 'python310.dll', 'Lib/os.py', 'Lib/test/x.py',
                  'Lib/tkinter/x.py', 'Lib/site-packages/README.txt', 'DLLs/_ctypes.pyd',
                  'DLLs/_tkinter.pyd', 'DLLs/tcl86t.dll', 'DLLs/_testcapi.pyd', 'tcl/x', 'include/x.h')
            pkg.trim(Path(root), 'x86_64-pc-windows-msvc')
            kept = {p.relative_to(root).as_posix() for p in Path(root).rglob('*') if p.is_file()}
            self.assertEqual(kept, {'python.exe', 'python310.dll', 'Lib/os.py', 'DLLs/_ctypes.pyd'})
            self.assertTrue(Path(root, 'Lib/site-packages').is_dir())

    def test_macos_keeps_only_the_real_interpreter(self):
        with tempfile.TemporaryDirectory() as root:
            touch(root, 'bin/python3.10', 'bin/pip3', 'bin/idle3.10', 'lib/libpython3.10.dylib',
                  'lib/libtcl8.6.dylib', 'lib/tcl8.6/init.tcl', 'lib/python3.10/os.py',
                  'lib/python3.10/config-3.10-darwin/libpython3.10.a', 'lib/python3.10/test/x.py',
                  'lib/python3.10/lib-dynload/_ctypes.so', 'lib/python3.10/lib-dynload/_tkinter.so',
                  'lib/python3.10/lib-dynload/_testcapi.so', 'share/man/x', 'include/python3.10/x.h')
            if os.name != 'nt':  # Symlinks need privileges on Windows.
                os.symlink('python3.10', Path(root, 'bin/python3'))
            pkg.trim(Path(root), 'aarch64-apple-darwin')
            kept = {p.relative_to(root).as_posix() for p in Path(root).rglob('*') if p.is_file()}
            self.assertEqual(kept, {'bin/python3.10', 'lib/libpython3.10.dylib', 'lib/python3.10/os.py',
                                    'lib/python3.10/lib-dynload/_ctypes.so'})


class Zip(unittest.TestCase):
    def test_archive_is_deterministic_and_keeps_executables(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory, 'runtime')
            touch(runtime, 'bin/python3.10', 'lib/python3.10/os.py')
            os.chmod(runtime / 'bin/python3.10', 0o755)
            os.chmod(runtime / 'lib/python3.10/os.py', 0o644)
            first, second = Path(directory, 'a.zip'), Path(directory, 'b.zip')
            pkg.write_zip(runtime, first)
            pkg.write_zip(runtime, second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as zf:
                self.assertEqual(zf.namelist(), ['bin/python3.10', 'lib/python3.10/os.py'])
                modes = {i.filename: (i.external_attr >> 16) for i in zf.infolist()}
            self.assertTrue(stat.S_ISREG(modes['bin/python3.10']))
            if os.name != 'nt':
                self.assertEqual(modes['bin/python3.10'] & 0o777, 0o755)
                self.assertEqual(modes['lib/python3.10/os.py'] & 0o777, 0o644)


if __name__ == '__main__':
    unittest.main()
