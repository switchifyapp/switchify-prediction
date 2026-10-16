import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import hardware_benchmark
from neural_evaluate import command, timing


TIMEOUT = '{"Unavailable": "Timeout"}'


def generation_report(timeout_count=0, slow=0, rerank=0):
    cell = {'queries': 4, 'six_slots_top6_hits': 3, 'statistical_six_top6_hits': 2,
            'current_top3_hits': 1, 'generated_words': 6}
    failures = {k: v for k, v in [('rerank', rerank), ('generation', timeout_count)] if v}
    return {'queries': 4, 'cells': {'a/b/0': cell}, 'failures': failures,
            'failure_statuses': {TIMEOUT: timeout_count} if timeout_count else {},
            'rerank_failure_statuses': {TIMEOUT: rerank} if rerank else {}, 'cold_load_ms': 1.0, 'peak_process_tree_rss_bytes': 2,
            'peak_process_tree_threads': 3, 'generated_sha256': 'x',
            'latency': {'generation_with_ipc': {'samples': 4 - timeout_count, 'median_ms': 1,
                                                'p95_ms': 2, 'max_ms': 3, 'at_least_deadline': slow}}}


class HardwareBenchmarkTests(unittest.TestCase):
    def test_deadline_counts_late_successes_and_timeouts(self):
        self.assertEqual(timing([1, 1999.9, 2000, 2500])['at_least_deadline'], 2)
        summary = hardware_benchmark.summarize(generation_report(timeout_count=1, slow=2), None)
        self.assertEqual(summary['generation']['deadline_misses'], 3)
        self.assertEqual(summary['generation']['timeouts'], 1)
        self.assertEqual(summary['generation']['failed_requests'], 1)
        self.assertEqual(summary['generation']['six_slot_top6_accuracy'], .75)
        self.assertEqual(summary['generation']['neural_slot_fill_rate'], .5)
        self.assertNotIn('refinement', summary)

    def test_rerank_failures_are_not_generation_deadline_misses(self):
        summary = hardware_benchmark.summarize(generation_report(timeout_count=1, slow=1, rerank=4), None)['generation']
        self.assertEqual((summary['failed_requests'], summary['timeouts'], summary['deadline_misses']), (1, 1, 2))
        self.assertEqual((summary['rerank_failed_requests'], summary['rerank_timeouts']), (4, 4))
        # Older reports combined both; all-timeout counts split exactly by request kind.
        old = generation_report(timeout_count=1, slow=1, rerank=4)
        old['failure_statuses'] = {TIMEOUT: 5}
        del old['rerank_failure_statuses']
        self.assertEqual(hardware_benchmark.summarize(old, None)['generation'], summary)
        old['failure_statuses'] = {TIMEOUT: 4, '{"Unavailable": "Worker"}': 1}
        with self.assertRaisesRegex(ValueError, 'cannot be split'):
            hardware_benchmark.summarize(old, None)

    def test_refinement_warmup_failure_keeps_the_generation_run(self):
        def generation(args):
            args.output.write_text(json.dumps(dict(generation_report(), capabilities={'accelerated': False})))
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for name in ['cli', 'worker']:
                (temp / name).write_bytes(b'synthetic')
            args = hardware_benchmark.parser().parse_args(
                ['--cli', str(temp / 'cli'), '--worker', str(temp / 'worker'), '--baseline', 'b', '--bundle', 'm',
                 '--output', str(temp / 'out.json'), '--label', 'synthetic', '--training', 't'])
            with mock.patch.object(hardware_benchmark, 'verify_inputs', return_value={}), \
                    mock.patch.object(hardware_benchmark, 'background_cpu_percent', return_value=0), \
                    mock.patch.object(hardware_benchmark, 'experiment_limits', return_value=None), \
                    mock.patch.object(hardware_benchmark, 'hardware', return_value={}), \
                    mock.patch.object(hardware_benchmark.generation_evaluate, 'evaluate', side_effect=generation), \
                    mock.patch.object(hardware_benchmark.neural_evaluate, 'evaluate',
                                      side_effect=RuntimeError('Neural warm-up failed; no measured run was produced')), \
                    mock.patch('builtins.print'):
                hardware_benchmark.run(args)
            report = json.loads((temp / 'out.json').read_bytes())
        self.assertEqual(report['summary']['generation']['queries'], 4)
        self.assertIn('warm-up failed', report['summary']['refinement']['error'])
        self.assertEqual(report['configuration']['experiment_build'], False)

    def test_intel_mac_avx_spelling(self):
        isa = hardware_benchmark.darwin_isa('FPU SSE4.1 SSE4.2 AVX1.0 F16C FMA', 'AVX2 BMI1')
        self.assertEqual(isa, dict.fromkeys(hardware_benchmark.ISA, True))
        self.assertFalse(hardware_benchmark.darwin_isa('SSE4.1', None)['avx'])

    def test_threads_are_passed_only_when_requested(self):
        args = SimpleNamespace(cli=Path('cli'), baseline=Path('b'), bundle=Path('m'), worker=Path('w'),
                               accelerated_worker=None, threads=None)
        self.assertNotIn('--threads', command(args))
        args.threads = 2
        self.assertEqual(command(args)[-2:], ['--threads', '2'])

    def test_executable_architecture_from_synthetic_headers(self):
        pe = bytearray(256)
        pe[:2], pe[60:64] = b'MZ', (128).to_bytes(4, 'little')
        pe[128:134] = b'PE\0\0' + (0xAA64).to_bytes(2, 'little')
        self.assertEqual(hardware_benchmark.binary_arch(bytes(pe)), 'aarch64')
        elf = b'\x7fELF' + bytes(14) + (62).to_bytes(2, 'little')
        self.assertEqual(hardware_benchmark.binary_arch(elf), 'x86_64')
        macho = b'\xcf\xfa\xed\xfe' + (0x0100000C).to_bytes(4, 'little')
        self.assertEqual(hardware_benchmark.binary_arch(macho), 'aarch64')
        self.assertEqual(hardware_benchmark.binary_arch(b'text'), 'unknown')

    def test_cpuinfo_flags_distinguish_missing_from_unknown(self):
        model, isa = hardware_benchmark.cpuinfo('model name\t: Synthetic CPU\nflags\t\t: sse4_1 sse4_2\n'
                                                'model name\t: Synthetic CPU\n')
        self.assertEqual(model, 'Synthetic CPU')
        self.assertEqual((isa['sse4_1'], isa['avx2']), (True, False))
        _, isa = hardware_benchmark.cpuinfo('Features\t: fp asimd\nCPU part\t: 0xd0b\n')
        self.assertIsNone(isa['avx2'])

    def test_production_worker_cannot_be_labelled_as_an_experiment(self):
        production = subprocess.CompletedProcess([], 1, '', '')
        with mock.patch.object(hardware_benchmark.subprocess, 'run', return_value=production):
            self.assertIsNone(hardware_benchmark.experiment_limits(Path('worker'), {}))
        # An experiment build is labelled as one even at the policy limits.
        policy = subprocess.CompletedProcess([], 0, '{"beam_width":8,"max_forward_evaluations":64}', '')
        with mock.patch.object(hardware_benchmark.subprocess, 'run', return_value=policy):
            self.assertEqual(hardware_benchmark.experiment_limits(Path('worker'), {})['beam_width'], 8)
        limits = {'beam_width': 4}
        for reply in [subprocess.CompletedProcess([], 1, '', ''),
                      subprocess.CompletedProcess([], 0, '{"beam_width":8}', '')]:
            with mock.patch.object(hardware_benchmark.subprocess, 'run', return_value=reply):
                with self.assertRaisesRegex(RuntimeError, 'not an experiment build'):
                    hardware_benchmark.experiment_limits(Path('worker'), limits)
        reply = subprocess.CompletedProcess([], 0, json.dumps({'beam_width': 4, 'max_forward_evaluations': 64}), '')
        with mock.patch.object(hardware_benchmark.subprocess, 'run', return_value=reply) as run:
            self.assertEqual(hardware_benchmark.experiment_limits(Path('worker'), limits)['beam_width'], 4)
        self.assertEqual(run.call_args.kwargs['env']['SWITCHIFY_SMOL_EXPERIMENT_BEAM_WIDTH'], '4')

    def test_substitute_model_or_database_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle = Path(temp)
            (bundle / 'model.gguf').write_bytes(b'synthetic')
            (bundle / 'english.sqlite').write_bytes(b'synthetic')
            args = SimpleNamespace(bundle=bundle, baseline=bundle / 'english.sqlite')
            with self.assertRaisesRegex(ValueError, 'pinned'):
                hardware_benchmark.verify_inputs(args)

    def test_cpu_restriction_and_limits_are_validated(self):
        parser = hardware_benchmark.parser()
        base = ['--cli', 'c', '--worker', 'w', '--baseline', 'b', '--bundle', 'm', '--output', 'o', '--label', 'l']
        self.assertEqual(parser.parse_args(base + ['--cpus', '13,12,13']).cpus, [12, 13])
        for extra in [['--cpus', '-1'], ['--beam-width', '9'], ['--max-forward-evaluations', '0'], ['--threads', '5']]:
            with self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                parser.parse_args(base + extra)


if __name__ == '__main__':
    unittest.main()
