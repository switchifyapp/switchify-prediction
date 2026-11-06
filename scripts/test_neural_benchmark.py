import copy
import io
import unittest

from neural_benchmark import compare, read_exact, summary


class BenchmarkTests(unittest.TestCase):
    def test_truncated_frames_fail(self):
        self.assertEqual(read_exact(io.BytesIO(b'abc'), 3), b'abc')
        with self.assertRaises(EOFError):
            read_exact(io.BytesIO(b'ab'), 3)

    def test_summary_uses_nearest_rank_p95(self):
        self.assertEqual(summary(list(range(20))),
                         dict(samples=20, median=10, p95=18, maximum=19))

    def test_comparison_rejects_different_workloads_and_detects_output_changes(self):
        report = dict(format_version=1, platform='test', threads=4, runs=3, repeats=5,
                      warmup=1, capabilities={}, model_sha256='model', manifest_sha256='manifest',
                      workload_sha256='workload', output_hashes={'word': ['abc']},
                      startup_ms={'median': 100}, memory={'peak': {'median': 100}},
                      latency_ms={'word': {'median': 100}})
        current = copy.deepcopy(report)
        current['memory']['peak']['median'] = 50
        current['output_hashes']['word'] = ['def']
        result = compare(report, current)
        self.assertEqual(result['changed_output_cases'], ['word'])
        self.assertEqual(result['memory_median_change_percent']['peak'], -50)
        current['threads'] = 1
        with self.assertRaisesRegex(ValueError, 'threads'):
            compare(report, current)


if __name__ == '__main__':
    unittest.main()
