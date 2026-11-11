import copy
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from frequency_experiment import (allocate, assessment, contextual_hash, frequencies,
                                  normalized_word, select_candidate, sha, supplement,
                                  verified_partitions)


class FrequencyExperimentTests(unittest.TestCase):
    def test_exact_budget_and_stable_ties_without_promoting_every_word(self):
        self.assertEqual(allocate({'z': 1, 'a': 1, 'b': 1}, 2), {'a': 1, 'b': 1})
        weights = {'a': 100, 'b': 3, 'c': 2}
        self.assertEqual(allocate(weights, 1000), allocate(dict(reversed(list(weights.items()))), 1000))
        self.assertEqual(sum(allocate(weights, 1000).values()), 1000)
        self.assertEqual(allocate(weights, 0), {})
        for bad in ({}, {'a': 0}, {'a': float('nan')}, {'a': float('inf')}):
            with self.assertRaises(ValueError):
                allocate(bad, 5)

    def test_normalized_single_words_and_frequency_cutoff(self):
        policy = {'minimum_word_codepoints': 2, 'maximum_word_codepoints': 32, 'vocabulary_limit': 2}
        self.assertEqual(normalized_word('CAFÉ', policy), 'café')
        self.assertEqual(normalized_word('can’t', policy), "can't")
        for word in ('two words', 'c3po', 'word-word', "'word", "word'", '\u0301a', 'a'):
            self.assertIsNone(normalized_word(word, policy))
        # Spelling variants are merged; a later new word cannot change the prefix.
        result = frequencies([['HELLO', 'hello'], ['world'], ['excluded']], policy)
        self.assertEqual(set(result), {'hello', 'world'})
        self.assertEqual(result['hello'], 2)
        with self.assertRaises(ValueError):
            frequencies([[4]], policy)

    def test_supplement_only_adds_unigrams_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, candidate = root / 'original.sqlite', root / 'candidate.sqlite'
            with closing(sqlite3.connect(original)) as db, db:
                db.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);'
                                 'CREATE TABLE vocabulary(word TEXT PRIMARY KEY);'
                                 'CREATE TABLE counts(context TEXT,word TEXT,count INTEGER,PRIMARY KEY(context,word));')
                db.execute("INSERT INTO metadata VALUES ('provenance','{}')")
                db.executemany('INSERT INTO vocabulary VALUES (?)', [('hello',), ('world',)])
                db.executemany('INSERT INTO counts VALUES (?,?,?)',
                               [('[]', 'hello', 10), ('[]', 'world', 4), ('["hello"]', 'world', 3)])
                before = contextual_hash(db)
            digest = sha(original)
            self.assertEqual(supplement(original, candidate, {'hello': 5, 'new': 2}, {'source': 'synthetic'}), before)
            self.assertEqual(sha(original), digest)
            with closing(sqlite3.connect(candidate)) as db:
                self.assertEqual(contextual_hash(db), before)
                self.assertEqual(dict(db.execute("SELECT word,count FROM counts WHERE context='[]'")),
                                 {'hello': 15, 'world': 4, 'new': 2})
                self.assertEqual(db.execute("SELECT value FROM metadata WHERE key='provenance'").fetchone()[0],
                                 '{"source": "synthetic"}')
            with self.assertRaises(ValueError):
                supplement(original, candidate, {}, {})
            with self.assertRaises(ValueError):
                supplement(original, original, {}, {})

    def test_selection_rejects_aac_regression_and_ignores_timing_for_choice(self):
        policy = {'acceptance': {'mean_early_top5_gain_min': .002, 'worst_early_top5_loss_max': .005,
                                'aac_prefix2_loss_max': 0, 'worst_selection_savings_loss_max': .001,
                                'warm_p95_target_ms': 20}}
        baseline = {domain: {'accuracy': [{'top5': .5}] * 5,
                             'selection_proxy': {'savings_fraction': .5}, 'warm_p95_ms': 1}
                    for domain in ('aac', 'general', 'conversation')}
        improved = copy.deepcopy(baseline)
        for domain in improved:
            improved[domain]['accuracy'] = [{'top5': .51}] * 5
        good = assessment(baseline, improved, policy)
        self.assertTrue(good['quality_passed'])
        improved['aac']['warm_p95_ms'] = 100
        slow = assessment(baseline, improved, policy)
        self.assertTrue(slow['quality_passed'])
        self.assertFalse(slow['latency_target_met'])
        improved['aac']['accuracy'][2] = {'top5': .499}
        rejected = assessment(baseline, improved, policy)
        self.assertFalse(rejected['quality_passed'])
        models = {'small': {'mass_fraction': .01, 'mode': 'missing'},
                  'large': {'mass_fraction': .20, 'mode': 'all'}}
        self.assertEqual(select_candidate({'small': slow, 'large': good}, models)['candidate'], 'small')
        self.assertTrue(select_candidate({'small': rejected}, models)['diagnostic_only'])

    def test_changed_frozen_partition_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            partitions = {}
            for domain in ('aac', 'general', 'conversation'):
                for part in ('dev', 'test'):
                    name = f'{domain}-{part}'
                    path = root / (name + '.txt')
                    path.write_text(name, encoding='utf-8')
                    partitions[name] = {'sha256': sha(path)}
            self.assertEqual(len(verified_partitions(root, {'partitions': partitions})), 6)
            (root / 'aac-test.txt').write_text('changed', encoding='utf-8')
            with self.assertRaises(ValueError):
                verified_partitions(root, {'partitions': partitions})


if __name__ == '__main__':
    unittest.main()
