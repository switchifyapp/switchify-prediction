import unittest

from aac_experiment import decision, partitions, utterances


class AacExperimentTests(unittest.TestCase):
    def test_utterance_boundaries_and_offsets(self):
        xml = '<graph xmlns="http://www.xces.org/ns/GrAF/1.0/"><region anchors="0 5"/><region anchors="6 11"/></graph>'
        self.assertEqual(utterances('hello world', xml), ['hello', 'world'])
        for bad in [xml.replace('6 11', '4 11'), xml.replace('6 11', '6 12'), '<graph/>']:
            with self.assertRaises(ValueError):
                utterances('hello world', bad)

    def test_no_leakage_and_deterministic_sampling(self):
        current = ['old training'] * 3
        existing = {'general-dev': {'general dev'}, 'general-test': {'general test'},
                    'conversation-dev': {'task dev'}, 'conversation-test': {'task test'}}
        aac = {'train': {'aac training', 'aac overlap', 'general test'},
               'dev': {'aac dev', 'old training', 'aac overlap'},
               'test': {'aac test', 'aac dev', 'aac overlap', 'old training'}}
        oanc = {'spoken one', 'spoken two', 'spoken three', 'aac test', 'general test', 'task dev', 'old training'}
        policy = {'weights': {'current': 1, 'aac': 10, 'oanc': 1}, 'oanc_min_words': 2,
                  'oanc_max_words': 30, 'oanc_max_sentences': 2}
        data, stats = partitions(current, existing, aac, oanc, policy)
        self.assertEqual(data['aac-dev'], ['aac dev'])
        self.assertEqual(data['aac-test'], ['aac test'])
        self.assertEqual(data['candidate'].count('old training'), 3)
        self.assertEqual(data['candidate'].count('aac training'), 10)
        self.assertNotIn('aac overlap', data['candidate'])
        self.assertEqual(stats['oanc_selected_unique'], 2)
        self.assertEqual((data, stats), partitions(current, existing, aac, set(reversed(sorted(oanc))), policy))
        for name, sentences in data.items():
            if name.endswith(('-dev', '-test')):
                self.assertFalse(set(data['candidate']) & set(sentences))

    def test_leaking_existing_split_rejected(self):
        with self.assertRaises(ValueError):
            partitions(['leaked text'], {'general-dev': {'leaked text'}},
                       {'train': {'a'}, 'dev': {'b'}, 'test': {'c'}}, {'o'},
                       {'weights': {'current': 1, 'aac': 1, 'oanc': 1},
                        'oanc_min_words': 1, 'oanc_max_words': 30, 'oanc_max_sentences': 2})

    def test_regression_prevents_promotion_despite_aac_gain(self):
        def score(accuracy):
            return {'accuracy': [{'top5': accuracy}] * 5,
                    'selection_proxy': {'savings_fraction': .2}, 'warm_p95_ms': 1}
        result = {s: {'baseline': score(.7), 'candidate': score(.7)} for s in ['aac', 'general', 'conversation']}
        result['aac']['candidate'] = score(.8)
        policy = {'acceptance': {'aac_top5_prefix2_gain_min': .05, 'general_top5_prefix2_loss_max': .01,
                                'conversation_top5_prefix2_loss_max': .02, 'aac_selection_savings_gain_min': 0,
                                'warm_p95_target_ms': 20}}
        self.assertTrue(decision(result, policy)['quality_passed'])
        result['general']['candidate'] = score(.68)
        self.assertFalse(decision(result, policy)['quality_passed'])
        result['general']['candidate'] = score(.7)
        result['conversation']['candidate'] = score(.67)
        self.assertFalse(decision(result, policy)['quality_passed'])


if __name__ == '__main__':
    unittest.main()
