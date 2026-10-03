import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from taskmaster2_experiment import DOMAINS, POLICY, comm2_text, fetch, partition, select, user_turns


class Taskmaster2Tests(unittest.TestCase):
    def test_corrupt_cached_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'files': {'source.json': {'url': 'unused', 'sha256': '0' * 64}}}))
            (root / 'source.json').write_bytes(b'corrupt')
            with patch('taskmaster2_experiment.MANIFEST', manifest):
                with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                    fetch(root)

    def test_speakers_and_comm2_ids(self):
        self.assertEqual(user_turns([{'utterances': [
            {'speaker': 'USER', 'text': 'user text'},
            {'speaker': 'ASSISTANT', 'text': 'assistant text'}]}]), ['user text'])
        self.assertEqual(comm2_text('comm2_1\tHello there!\ncomm2_2\tGoodbye.'), ['Hello there!', 'Goodbye.'])
        for bad in ['', 'missing separator', 'wrong_id\thello', 'comm2_1\t', 'comm2_1\ta\ncomm2_1\tb']:
            with self.assertRaises(ValueError):
                comm2_text(bad)

    def test_partition_isolation_and_determinism(self):
        policy = dict(POLICY, max_sentences_per_domain=2)
        current = ['old training'] * 3
        existing = {'dev': {'dev sentence'}, 'test': {'test sentence'}}
        comm2 = {'fresh comm', 'old training', 'dev sentence'}
        domains = {'b': {'shared sentence', 'second sentence', 'fresh comm'},
                   'a': {'shared sentence', 'first sentence', 'dev sentence', 'test sentence', 'old training', 'one'}}
        selected, fresh, stats = partition(current, existing, comm2, domains, policy)
        self.assertEqual(set(selected), {'shared sentence', 'first sentence', 'second sentence'})
        self.assertEqual(fresh, ['fresh comm'])
        self.assertEqual(stats['b']['selected'], 1)
        self.assertEqual((selected, fresh, stats), partition(current, existing, comm2, dict(reversed(list(domains.items()))), policy))
        self.assertEqual(stats['comm2']['overlap_training'], 1)
        with self.assertRaises(ValueError):
            partition(['dev sentence'], existing, comm2, domains)
        with self.assertRaises(ValueError):
            partition(current, existing, {'old training'}, domains)

    def development(self):
        return {d: {name: {'accuracy': [{'top5': score} for _ in range(5)], 'warm_p95_ms': 5}
                    for name, score in [('baseline', .5), ('tm2-1x', .52), ('tm2-3x', .53)]} for d in DOMAINS}

    def test_selection_guards_and_tie(self):
        dev = self.development()
        self.assertEqual(select(dev)['selected'], 'tm2-3x')
        dev['aac']['tm2-3x']['accuracy'][0]['top5'] = .48
        self.assertEqual(select(dev)['selected'], 'tm2-1x')
        for d in DOMAINS:
            dev[d]['tm2-3x'] = copy.deepcopy(dev[d]['tm2-1x'])
        self.assertEqual(select(dev)['selected'], 'tm2-1x')
        dev['aac']['tm2-1x']['warm_p95_ms'] = 20
        self.assertEqual(select(dev)['selected'], 'tm2-3x')
        dev['aac']['tm2-3x']['warm_p95_ms'] = 20
        self.assertTrue(select(dev)['diagnostic_only'])


if __name__ == '__main__':
    unittest.main()
