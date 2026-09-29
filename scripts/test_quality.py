import copy
import unittest
from quality import development_passes


class GateTests(unittest.TestCase):
    def test_gate_uses_development_quality_not_timing_noise(self):
        policy = {'acceptance': {'conversation_top5_prefix2_gain_min': .05,
                                'general_top5_prefix2_loss_max': .02,
                                'conversation_selection_savings_gain_min': 0,
                                'warm_p95_target_ms': 20}}
        def model(accuracy, savings, latency=1):
            return {'accuracy': [{}, {}, {'top5': accuracy}],
                    'selection_proxy': {'savings_fraction': savings},
                    'warm_p95_ms': latency}
        data = {'general': {'baseline': model(.60, .1), 'candidate': model(.59, .1)},
                'conversation': {'baseline': model(.50, .1), 'candidate': model(.60, .2)}}
        self.assertTrue(development_passes(data, policy)['quality_passed'])
        rejected = copy.deepcopy(data)
        rejected['general']['candidate'] = model(.55, .1)
        self.assertFalse(development_passes(rejected, policy)['quality_passed'])
        rejected = copy.deepcopy(data)
        rejected['conversation']['candidate'] = model(.60, .09)
        self.assertFalse(development_passes(rejected, policy)['quality_passed'])
        data['conversation']['candidate']['warm_p95_ms'] = 100
        decision = development_passes(data, policy)
        self.assertTrue(decision['quality_passed'])
        self.assertFalse(decision['latency_target_met'])


if __name__ == '__main__':
    unittest.main()
