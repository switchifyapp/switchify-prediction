import unittest
from futo_experiment import normalize_word, summarize


class FutoTests(unittest.TestCase):
    def test_words_and_apostrophes(self):
        self.assertEqual(normalize_word('  CAFÉ’S '), "café's")
        self.assertEqual(normalize_word('cafe\u0301'), 'café')
        for word in ('', 'two words', 'word.', '123', "'bad", "bad'", '\u0301bad'):
            self.assertIsNone(normalize_word(word))

    def test_three_slot_correction_and_exact_prefix_scores_are_separate(self):
        queries = [dict(domain='messages', prefix_chars=1, prefix='w', target='word')]
        report = summarize(queries, ['load\t10', '0\t1\tprime', '1\t2\tthe\tword\tWorld'])
        cell = report['cells']['messages']['1']
        self.assertEqual(cell['native_top1_hits'], 0)
        self.assertEqual(cell['exact_prefix_top1_hits'], 1)
        self.assertEqual(report['filtered_non_prefix_suggestions'], 1)
        self.assertEqual(report['warm_p95_ms'], 2)
        with self.assertRaises(ValueError):
            summarize(queries, ['load\t10'])


if __name__ == '__main__':
    unittest.main()
