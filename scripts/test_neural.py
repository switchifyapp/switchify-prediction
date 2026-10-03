import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from neural_bundle import verify

ROOT = Path(__file__).resolve().parent.parent


class NeuralBundleTests(unittest.TestCase):
    def test_corrupt_file_and_wrong_length_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'test'
            path.write_bytes(b'good')
            pin = {'bytes':4, 'sha256':hashlib.sha256(b'good').hexdigest()}
            verify(path, pin)
            for value in [b'evil', b'', b'longer']:
                path.write_bytes(value)
                with self.assertRaises(ValueError):
                    verify(path, pin)

    def test_pins_and_preserved_model_license(self):
        manifest = json.loads((ROOT / 'neural/model-bundle.json').read_bytes())
        source = json.loads((ROOT / 'neural/source-manifest.json').read_bytes())
        self.assertEqual(manifest['revision'], source['revision'])
        verify(ROOT / 'neural/MODEL_LICENSE.txt', manifest['files']['MODEL_LICENSE.txt'])
        self.assertEqual(manifest['files']['tokenizer.json']['sha256'], source['files']['tokenizer.json']['sha256'])
        self.assertEqual(manifest['policy']['max_results'], 5)

    def test_frozen_fixture_partitions_are_disjoint(self):
        extra = json.loads((ROOT / 'neural/fixtures/general-writing.json').read_bytes())
        old = json.loads((ROOT / 'neural/fixtures/regression.json').read_bytes())
        texts = [s for group in extra.values() for s in group]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertFalse(set(texts).intersection(s for group in old.values() for s in group))
        self.assertTrue(all(len(group) == 12 for group in extra.values()))


if __name__ == '__main__':
    unittest.main()
