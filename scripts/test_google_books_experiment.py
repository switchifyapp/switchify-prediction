from fractions import Fraction
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from google_books_experiment import MAX_COUNT, aggregate_rows, combine, normalize_ngram, read_counts, write_counts


class GoogleBooksTests(unittest.TestCase):
    policy = {'publication_year_min': 1980, 'publication_year_max': 2008, 'max_word_characters': 32}

    def test_year_filter_and_occurrence_column_with_case_aggregation(self):
        rows = [b'Hello\t1979\t99\t999\t999\n', b'Hello\t1980\t3\t999\t999\n',
                b'Hello\t2008\t7\t999\t999\n', b'hello\t2009\t88\t999\t999\n',
                b'hello\t2000\t5\t999\t999\n', b'hello.\t2000\t55\t999\t999\n']
        counts, stats = aggregate_rows(rows, 1, self.policy)
        self.assertEqual(counts, {('hello',): 15})
        self.assertEqual(stats['rows'], 6)
        self.assertEqual(stats['year_rows'], 4)
        with self.assertRaises(ValueError):
            aggregate_rows([b'bad\t2000\t1\t1\n'], 1, self.policy)

    def test_normalization_rejects_tags_boundaries_and_unavailable_words(self):
        self.assertEqual(normalize_ngram('I can’t'.encode(), 2, 32), ('i', "can't"))
        for raw in [b'hello. world', b'_START_ hello', b'hello  world', b'abc123 world', b'cafe\xc3\xa9 world']:
            self.assertIsNone(normalize_ngram(raw, 2, 32))
        counts, _ = aggregate_rows([b'known missing\t2000\t50\t1\t1\n'], 2, self.policy, {'known'})
        self.assertFalse(counts)

    def test_exact_mixture_keeps_rare_contexts_without_rounding(self):
        base = {('a',): 10, ('b',): 10, ('a', 'b'): 2}
        google = {('a',): 1000000, ('b',): 1000000, ('a', 'b'): 1}
        mixed, recipe = combine(base, google, 5)
        left, right = recipe['baseline_coefficient'], recipe['google_coefficient']
        self.assertEqual(Fraction(right * 2000000, left * 20 + right * 2000000), Fraction(5, 100))
        self.assertEqual(mixed[('a', 'b')], 2 * left + right)
        control, _ = combine(base, google, 5, True)
        self.assertEqual(control[('a', 'b')], 2 * left)
        self.assertEqual(mixed[('b',)], control[('b',)])

    def test_sqlite_writer_is_separate_consistent_and_closes_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            template, output = root / 'base.sqlite', root / 'candidate.sqlite'
            with closing(sqlite3.connect(template)) as db, db:
                db.executescript("CREATE TABLE vocabulary(word TEXT); CREATE TABLE counts(context TEXT,word TEXT,count INTEGER); CREATE TABLE imports(hash TEXT); CREATE TABLE metadata(key TEXT,value TEXT); INSERT INTO metadata VALUES('provenance','old');")
            before = template.read_bytes()
            counts = {('a',): 10, ('b',): 20, ('a', 'b'): 4}
            write_counts(template, output, counts, {'test': True})
            self.assertEqual(read_counts(output), counts)
            self.assertEqual(template.read_bytes(), before)
            output.unlink()
            with self.assertRaisesRegex(ValueError, 'overflowing'):
                write_counts(template, output, {('a',): MAX_COUNT + 1}, {})
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                write_counts(template, output, {('a',): 1, ('a', 'b'): 1}, {})


if __name__ == '__main__':
    unittest.main()
