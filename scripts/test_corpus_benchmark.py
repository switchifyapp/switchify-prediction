import unittest
import hashlib
from pathlib import Path
import tempfile

from corpus_benchmark import decompose, mix, ranked, recipes, reject_leakage, verify_prepared


class CorpusBenchmarkTests(unittest.TestCase):
    def test_changed_preparation_cannot_redefine_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'aac-test.txt'
            original = b'original evaluation\n'
            path.write_bytes(original)
            expected = hashlib.sha256(original).hexdigest()
            production = {'partitions': {'aac-test': {'sha256': expected}}}
            info = {'files': {'aac-test': {'sha256': expected}}}
            verify_prepared(root, info, production, ['aac-test'])
            path.write_bytes(b'changed evaluation\n')
            info['files']['aac-test']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'Frozen evaluation checksum'):
                verify_prepared(root, info, production, ['aac-test'])

    def test_decomposition_preserves_shared_source_sentences(self):
        world = {'general text', 'shared text'}
        mixed = ['general text'] * 3 + ['shared text'] * 4 + ['task text']
        self.assertEqual(decompose(world, mixed, 3), {'shared text', 'task text'})
        for bad in [mixed + ['task text'], ['general text']]:
            with self.assertRaises(ValueError):
                decompose(world, bad, 3)

    def test_ablation_preserves_other_sources_and_shared_counts(self):
        sources = {'tatoeba': {'shared', 'world'}, 'taskmaster': {'shared', 'task'},
                   'aac': {'aac'}, 'oanc': {'spoken'}}
        policy = {'sources': list(sources),
                  'production_weights': {'tatoeba': 3, 'taskmaster': 1, 'aac': 10, 'oanc': 1},
                  'models': ['production', *sources, *('without-' + n for n in sources), 'full-oanc-blend']}
        result = recipes(sources, {'spoken', 'extra'}, policy)
        self.assertEqual(result['production'].count('shared'), 4)
        self.assertEqual(result['without-tatoeba'].count('shared'), 1)
        self.assertEqual(result['without-taskmaster'].count('shared'), 3)
        self.assertEqual(result['without-oanc'].count('aac'), 10)
        self.assertEqual(result['aac'], ['aac'])
        self.assertEqual(result['full-oanc-blend'].count('extra'), 1)
        self.assertNotIn('extra', result['production'])

    def test_both_held_out_partitions_are_excluded(self):
        evaluations = {'dev': {'development'}, 'test': {'testing'}}
        reject_leakage({'clean': ['training']}, evaluations)
        for sentence in ['development', 'testing']:
            with self.assertRaisesRegex(ValueError, 'Training/evaluation'):
                reject_leakage({'leaking': [sentence]}, evaluations)
        with self.assertRaisesRegex(ValueError, 'partitions overlap'):
            reject_leakage({'clean': ['training']}, {'dev': {'shared'}, 'test': {'shared'}})

    def test_preparation_is_independent_of_input_order(self):
        self.assertEqual(ranked(['z', 'a', 'a', 'b']), ranked(['b', 'a', 'z']))
        self.assertEqual(mix({'x': {'a', 'b'}}, {'x': 2}), ['a', 'b', 'a', 'b'])


if __name__ == '__main__':
    unittest.main()
