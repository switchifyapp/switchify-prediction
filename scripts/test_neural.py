import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from neural_bundle import verify
from neural_evaluate import warmup
from generation_evaluate import generated

ROOT = Path(__file__).resolve().parent.parent


class NeuralBundleTests(unittest.TestCase):
    def test_generation_comparison_resets_context_before_measuring(self):
        import io
        from types import SimpleNamespace
        replies = iter([{'type':'reset'}, {'type':'immediate', 'result':{'refinement_requested':True, 'request_id':8}},
                        {'type':'refined', 'result':{'request_id':8}}])
        client = SimpleNamespace(process=SimpleNamespace(stdin=io.StringIO()), receive=lambda: next(replies))
        generated(client, 'synthetic', 'sy')
        commands = [json.loads(line) for line in client.process.stdin.getvalue().splitlines()]
        self.assertEqual([c['command'] for c in commands], ['reset', 'generate'])

    def test_typing_measurement_can_keep_the_context(self):
        import io
        from types import SimpleNamespace
        replies = iter([{'type':'immediate', 'result':{'refinement_requested':True, 'request_id':9}},
                        {'type':'refined', 'result':{'request_id':9}}])
        client = SimpleNamespace(process=SimpleNamespace(stdin=io.StringIO()), receive=lambda: next(replies))
        generated(client, 'synthetic', 'sy', reset=False)
        commands = [json.loads(line) for line in client.process.stdin.getvalue().splitlines()]
        self.assertEqual([c['command'] for c in commands], ['generate'])

    @unittest.skipUnless(importlib.util.find_spec('regex'), 'benchmark dependency')
    def test_long_contexts_keep_earlier_texts_within_the_history(self):
        from generation_evaluate import typed_workload
        from neural_evaluate import natural_workload, workload
        queries = workload()
        plain = natural_workload(queries)
        long = natural_workload(queries, 512)
        self.assertTrue(all(q[4].endswith(p[4]) and len(q[4]) <= 512 for q, p in zip(long, plain)))
        self.assertGreater(sum(len(q[4]) > 400 for q in long), len(long) // 2)
        steps = typed_workload(512, 0)
        self.assertIsNone(steps[-1])
        self.assertTrue(all(len(s[0]) <= 512 for s in steps if s))
        self.assertEqual(steps[:3], [('', ''), ('', steps[1][1]), ('', steps[2][1])])
        groups = steps.count(None)
        bounded = typed_workload(512, 100)
        self.assertEqual(bounded.count(None), groups)
        self.assertEqual(len(bounded) - groups, -(-100 // groups) * groups)

    def test_warmup_cannot_silently_turn_into_statistical_only_measurement(self):
        class FakeClient:
            def __init__(self, status, requested, refined):
                self.status, self.requested, self.refined = status, requested, refined
            def query(self, before, prefix):
                return {'result':{'status':self.status, 'refinement_requested':self.requested}}, self.refined, 0, 0
        query = [None, None, None, None, '', '']
        for client in [FakeClient('Ready', True, None), FakeClient({'Unavailable':'Timeout'}, False, None)]:
            with self.assertRaisesRegex(RuntimeError, 'warm-up failed'):
                warmup(client, [query])
        warmup(FakeClient('Ready', True, {'result':{}}), [query])
        warmup(FakeClient('Ready', False, None), [query])

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
