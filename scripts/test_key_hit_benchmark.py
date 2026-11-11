from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

from key_hit_benchmark import SUITES, check_result, verify_inputs


class KeyHitBenchmarkTests(unittest.TestCase):
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
