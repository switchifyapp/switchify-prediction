import copy
import unittest
from compare_legacy import validate_batch

class ComparisonAcceptance(unittest.TestCase):
    def fixture(self):
        rows = [dict(prefix_chars=i, queries=10, top1=0.2, top5=0.6) for i in range(5)]
        return dict(newer=dict(accuracy=rows), combined=dict(accuracy=copy.deepcopy(rows), position_violations=0))

    def test_unchanged_and_improved_results_pass(self):
        value = self.fixture()
        validate_batch(value)
        value['combined']['accuracy'][2]['top5'] = 0.7
        validate_batch(value)

    def test_position_changes_and_regressions_fail(self):
        value = self.fixture()
        value['combined']['position_violations'] = 1
        with self.assertRaises(ValueError): validate_batch(value)
        for metric in ['top1', 'top5']:
            value = self.fixture()
            value['combined']['accuracy'][2][metric] = 0.1
            with self.assertRaises(ValueError): validate_batch(value)

    def test_incomparable_or_incomplete_reports_fail(self):
        for key in ['queries', 'prefix_chars']:
            value = self.fixture()
            value['combined']['accuracy'][2][key] = 99
            with self.assertRaises(ValueError): validate_batch(value)
        value = self.fixture()
        value['combined']['accuracy'].pop()
        with self.assertRaises(ValueError): validate_batch(value)
