import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from artifacts import verify_quality
from verify_bundle import verify

ROOT = Path(__file__).resolve().parent.parent


class ProductionGates(unittest.TestCase):
    def fixture(self):
        manifest = json.loads((ROOT / 'production-model.json').read_text(encoding='utf-8'))
        quality = json.loads((ROOT / 'production-quality.json').read_text(encoding='utf-8'))
        report = copy.deepcopy(quality)
        for part in ['dev', 'test']:
            for suite in report[part].values():
                for model in suite.values():
                    model['warm_p95_ms'] = 1.0
        report.update(policy=manifest['sources']['aac-quality-policy.json'], protocol=manifest['quality_protocol'],
                      partitions={'files': manifest['partitions'], 'training_evaluation_overlap': 0,
                                  'evaluation_partition_overlap': 0,
                                  'source_manifest_sha256': manifest['source_files']['aac-source-manifest.json'],
                                  'policy_sha256': manifest['source_files']['aac-quality-policy.json']},
                      validation={'candidate': {'logical_sha256': manifest['logical_sha256']}},
                      deterministic_logical_contents=True)
        return report, manifest, quality

    def test_accepted_frozen_report(self):
        verify_quality(*self.fixture())

    def test_forged_success_and_wrong_models_fail_closed(self):
        report, manifest, quality = self.fixture()
        report['recommend_promotion'] = True
        report['test']['aac']['candidate']['accuracy'][2]['top5'] = .1
        with self.assertRaises(ValueError):
            verify_quality(report, manifest, quality)
        for field, value in [('logical_sha256', 'bad')]:
            report, manifest, quality = self.fixture()
            report['validation']['candidate'][field] = value
            with self.assertRaises(ValueError):
                verify_quality(report, manifest, quality)
        report, manifest, quality = self.fixture()
        report['partitions']['training_evaluation_overlap'] = 1
        with self.assertRaises(ValueError):
            verify_quality(report, manifest, quality)

    def test_changed_prefix_results_rejected_even_if_gates_pass(self):
        report, manifest, quality = self.fixture()
        report['dev']['aac']['candidate']['accuracy'][0]['top1'] = .99
        with self.assertRaises(ValueError):
            verify_quality(report, manifest, quality)

    def test_bundle_tampering_and_escaping_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'model').write_bytes(b'example')
            sha = hashlib.sha256(b'example').hexdigest()
            (root / 'SHA256SUMS').write_text(f'{sha}  model\n', encoding='utf-8')
            self.assertEqual(verify(root), 1)
            (root / 'model').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify(root)
            for name in ['../model', '/model', r'..\model', 'C:/model']:
                (root / 'SHA256SUMS').write_text(f'{sha}  {name}\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    verify(root)

class DependencyNotices(unittest.TestCase):
    def test_licences_are_preserved_and_missing_or_unreviewed_licences_stop_packaging(self):
        from dependency_notices import crate_notices
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = {'name': 'fixture', 'version': '1.0.0', 'license': 'MIT',
                       'manifest_path': str(root / 'Cargo.toml'), 'license_file': None}
            with self.assertRaises(ValueError):
                crate_notices(package)
            (root / 'LICENSE').write_text('Copyright fixture author\nPermission notice', encoding='utf-8')
            self.assertIn('Copyright fixture author\nPermission notice', crate_notices(package))
            package['license'] = 'Unreviewed-License'
            with self.assertRaises(ValueError):
                crate_notices(package)


class CratePackagePrivacy(unittest.TestCase):
    def test_generated_or_private_support_files_are_not_packaged(self):
        import subprocess
        generated = ROOT / 'artifacts'
        generated.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='synthetic-package-test-', dir=generated) as temp:
            for name in ['README.md', 'LICENSE', 'source-manifest.json']:
                (Path(temp) / name).write_text('synthetic private fixture', encoding='utf-8')
            listed = subprocess.check_output(
                ['cargo', 'package', '--list', '--locked', '--allow-dirty'], cwd=ROOT, text=True).splitlines()
            allowed = {'.cargo_vcs_info.json', 'Cargo.lock', 'Cargo.toml', 'Cargo.toml.orig',
                       'LICENSE', 'README.md', 'production-model.json', 'quality-policy.json',
                       'source-manifest.json', 'src/corpus.rs', 'src/evaluation.rs', 'src/lib.rs',
                       'src/main.rs', 'src/production.rs', 'src/experimental.rs'}
            self.assertEqual({name.replace('\\', '/') for name in listed}, allowed)


if __name__ == '__main__':
    unittest.main()
