import copy
import unittest

from general_neural_experiment import decision


class GeneralNeuralTests(unittest.TestCase):
    def test_general_domain_gates(self):
        base = {'cells': {d: {str(p): {'queries': 100, 'top5_hits': 50} for p in range(5)}
                          for d in ['messages', 'email', 'documents', 'search']}, 'warm_p95_ms': 5}
        neural = copy.deepcopy(base)
        for cells in neural['cells'].values():
            for cell in cells.values():
                cell['top5_hits'] = 52
        reports = {'baseline': base, 'neural': neural}
        self.assertTrue(decision(reports)['accuracy_passed'])
        neural['cells']['email']['2']['top5_hits'] = 48
        self.assertFalse(decision(reports)['accuracy_passed'])
        neural['warm_p95_ms'] = 20
        self.assertFalse(decision(reports)['latency_passed'])
        self.assertFalse(decision(reports)['promote'])
        neural['cells']['email']['2']['queries'] = 99
        with self.assertRaises(ValueError):
            decision(reports)


if __name__ == '__main__':
    unittest.main()
