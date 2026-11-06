import copy
import io
import importlib.util
import json
import os
import subprocess
import sys
import unittest

from neural_benchmark import CASES, CANDIDATES, compare, read_exact, summary, workload_hash
from neural_throttle import WindowsCpuLimit, cpu_rate, creation_flags


class BenchmarkTests(unittest.TestCase):
    def test_fresh_only_is_a_distinct_workload(self):
        modes = ['Predict', 'Generate', 'GeneratePhrases']
        cases = [c[0] for c in CASES]
        import hashlib
        legacy = hashlib.sha256(json.dumps([CASES, CANDIDATES]).encode()).hexdigest()
        self.assertEqual(workload_hash(CANDIDATES, modes, cases, False), legacy)
        self.assertNotEqual(workload_hash(CANDIDATES, modes, cases, True), legacy)
        self.assertNotEqual(workload_hash(CANDIDATES, ['Predict'], ['long'], False),
                            workload_hash(CANDIDATES, ['Predict'], ['long'], True))
    def test_cpu_budget_is_scaled_to_the_whole_machine(self):
        self.assertEqual(cpu_rate(50, 24), 208)
        self.assertEqual(cpu_rate(100, 8), 1250)
        with self.assertRaises(ValueError):
            cpu_rate(0, 24)

    @unittest.skipUnless(os.name == 'nt' and importlib.util.find_spec('psutil') is not None,
                         'requires Windows and psutil')
    def test_cpu_limit_is_applied_before_suspended_process_runs(self):
        import psutil
        limit = WindowsCpuLimit(50)
        process = None
        try:
            code = ('import time,json; wall=time.perf_counter(); cpu=time.process_time(); '
                    'exec("while time.process_time()-cpu < 1.0: pass"); '
                    'print(json.dumps([time.perf_counter()-wall,time.process_time()-cpu]))')
            process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                       creationflags=creation_flags(True))
            info = psutil.Process(process.pid)
            limit.start(process, info, [psutil.Process().cpu_affinity()[0]])
            output, _ = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0)
            wall, cpu = json.loads(output)
            self.assertGreaterEqual(cpu, 1.0)
            self.assertLess(cpu / wall, .8, 'Hard cap should constrain a CPU-bound child')
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.communicate()
            limit.close()
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
        for key in ['Predict/long/hit', 'Predict/long/miss', 'Generate/long/hit']:
            report['latency_ms'][key] = {'median': 100}
            current['latency_ms'][key] = {'median': 1}
        result = compare(report, current)
        self.assertEqual(result['changed_output_cases'], ['word'])
        self.assertEqual(result['memory_median_change_percent']['peak'], -50)
        self.assertEqual(list(result['latency_case_notes']), ['Predict/long/hit'])
        self.assertIn('cache reuse', result['latency_case_notes']['Predict/long/hit'])
        current['threads'] = 1
        with self.assertRaisesRegex(ValueError, 'threads'):
            compare(report, current)
        current['threads'] = 4
        current['throttle'] = {'job_cpu_rate': 208}
        with self.assertRaisesRegex(ValueError, 'throttle'):
            compare(report, current)


if __name__ == '__main__':
    unittest.main()
