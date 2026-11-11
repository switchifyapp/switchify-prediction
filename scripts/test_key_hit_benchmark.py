from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from key_hit_benchmark import SUITES, build_example, check_result, example_executable, verify_inputs


class KeyHitBenchmarkTests(unittest.TestCase):
    def test_published_report_keeps_hash_provenance_without_local_absolute_paths(self):
        report = json.loads((Path(__file__).resolve().parent.parent /
                             'docs/key-hit-results.json').read_text(encoding='utf-8'))
        self.assertRegex(report['executable_sha256'], r'^[0-9a-f]{64}$')

        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    check(key)
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            elif isinstance(value, str):
                self.assertFalse(PurePosixPath(value).is_absolute() or
                                 PureWindowsPath(value).is_absolute(), value)
        check(report)

    def test_cargo_reported_executable_respects_both_target_directory_overrides(self):
        for mode in ['environment', 'configuration']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / 'examples').mkdir()
                (root / 'Cargo.toml').write_text(
                    '[package]\nname="key-hit-path-fixture"\nversion="0.1.0"\nedition="2024"\n',
                    encoding='utf-8')
                (root / 'Cargo.lock').write_text(
                    'version = 4\n\n[[package]]\nname = "key-hit-path-fixture"\nversion = "0.1.0"\n',
                    encoding='utf-8')
                (root / 'examples/key_hit_benchmark.rs').write_text(
                    'fn main() { println!("fresh fixture"); }\n', encoding='utf-8')
                binary_name = 'key_hit_benchmark.exe' if os.name == 'nt' else 'key_hit_benchmark'
                stale = root / 'target/release/examples' / binary_name
                stale.parent.mkdir(parents=True)
                stale.write_bytes(b'stale default-target binary')
                target = root / 'custom-output-\u00e9'
                with patch('key_hit_benchmark.ROOT', root), patch.dict(os.environ):
                    os.environ.pop('CARGO_TARGET_DIR', None)
                    if mode == 'environment':
                        os.environ['CARGO_TARGET_DIR'] = str(target)
                    else:
                        (root / '.cargo').mkdir()
                        (root / '.cargo/config.toml').write_text(
                            f'[build]\ntarget-dir = "{target.name}"\n', encoding='utf-8')
                    executable = build_example()
                self.assertEqual(executable, (target / 'release/examples' / binary_name).resolve())
                self.assertEqual(subprocess.check_output([str(executable)], text=True).strip(), 'fresh fixture')
                self.assertEqual(stale.read_bytes(), b'stale default-target binary')

    def test_missing_or_ambiguous_cargo_artifacts_never_fall_back(self):
        artifact = {'reason': 'compiler-artifact', 'target': {'name': 'key_hit_benchmark', 'kind': ['example']},
                    'executable': 'custom/release/examples/key_hit_benchmark'}
        self.assertEqual(example_executable([artifact], Path.cwd()),
                         (Path.cwd() / artifact['executable']).resolve())
        for messages in [[], [artifact, artifact], [dict(artifact, executable=None)],
                         [dict(artifact, target={'name': 'other', 'kind': ['example']})]]:
            with self.assertRaises(ValueError):
                example_executable(messages, Path.cwd())

    def test_pinned_inputs_cannot_silently_use_the_other_general_partition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            production = {'partitions': {}}
            for name, relative in SUITES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                contents = (name + ' frozen\n').encode()
                path.write_bytes(contents)
                production['partitions'][name + '-dev'] = {'sha256': hashlib.sha256(contents).hexdigest()}
            other = root / 'data/prepared/general-dev.txt'
            other.parent.mkdir(parents=True)
            other.write_bytes(b'other general partition\n')
            paths = verify_inputs(root, production)
            self.assertEqual(paths['general'], root / SUITES['general'])
            paths['general'].write_bytes(other.read_bytes())
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                verify_inputs(root, production)

    def test_aggregation_rejects_stale_or_mixed_measurements(self):
        result = {'protocol': 'key-hits-v2', 'evaluation_sha256': 'evaluation',
                  'baseline': {'logical_sha256': 'production'},
                  'metrics': [{'prefix_graphemes': n} for n in [2, 3, 4]]}
        check_result(result, 'evaluation', 'production')
        for field, value in [('protocol', 'key-hits-v1'), ('evaluation_sha256', 'other'),
                             ('baseline', {'logical_sha256': 'other'}), ('metrics', [])]:
            broken = deepcopy(result)
            broken[field] = value
            with self.assertRaises(ValueError):
                check_result(broken, 'evaluation', 'production')


if __name__ == '__main__':
    unittest.main()
