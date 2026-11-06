#!/usr/bin/env python3
"""Benchmark a local neural worker with synthetic text, without a baseline DB.

Requires psutil. No downloads. Reports contain hashes, timings and memory,
not prompts. This measures worker IPC, not the desktop/controller or accuracy.
"""
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import queue
import struct
import subprocess
import threading
import time

from neural_throttle import WindowsCpuLimit, creation_flags


CASES = [
    ('empty', '', ''),
    ('message', 'Please send the', ''),
    ('prefix', 'I would like a', 'c'),
    ('long', 'I have been thinking about our plans for the weekend and would like to', 'go'),
]
CANDIDATES = ['receipt', 'order', 'tickets', 'confirmation', 'car', 'details', 'directions', 'address']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def summary(values):
    values = sorted(values)
    return dict(samples=len(values), median=values[len(values) // 2],
                p95=values[math.ceil(len(values) * .95) - 1], maximum=values[-1])


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        chunk = stream.read(size - len(result))
        if not chunk:
            raise EOFError('Worker closed its output')
        result.extend(chunk)
    return result


class Worker:
    def __init__(self, worker, bundle, threads, cpu_percent=None, affinity=None):
        import psutil
        self.stop = threading.Event()
        self.messages = queue.Queue()
        self.peak = 0
        self.limit = WindowsCpuLimit(cpu_percent) if cpu_percent is not None else None
        self.affinity = affinity
        self.process = None
        try:
            self.process = subprocess.Popen(
                [str(worker.resolve()), str(bundle.resolve())], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env={**os.environ, 'RAYON_NUM_THREADS': str(threads),
                     'CANDLE_NUM_THREADS': str(threads), 'TOKENIZERS_PARALLELISM': 'false'},
                creationflags=creation_flags(self.limit is not None))
            self.info = psutil.Process(self.process.pid)
            if self.limit:
                self.limit.start(self.process, self.info, affinity)
        except BaseException:
            if self.process:
                self.process.kill()
                self.process.wait()
                self.process.stdin.close()
                self.process.stdout.close()
            if self.limit:
                self.limit.close()
            raise

        def read():
            try:
                while True:
                    length, = struct.unpack('<I', read_exact(self.process.stdout, 4))
                    if not 0 < length <= 65536:
                        raise ValueError('Invalid worker frame')
                    self.messages.put(json.loads(read_exact(self.process.stdout, length)))
            except Exception as error:
                self.messages.put(error)

        def monitor():
            while not self.stop.is_set():
                try:
                    self.peak = max(self.peak, self.rss())
                except psutil.Error:
                    pass
                self.stop.wait(.005)

        self.reader = threading.Thread(target=read, daemon=True)
        self.monitor = threading.Thread(target=monitor, daemon=True)
        self.reader.start()
        self.monitor.start()

    def rss(self):
        return self.info.memory_info().rss

    def receive(self):
        reply = self.messages.get(timeout=35)
        if isinstance(reply, Exception):
            raise RuntimeError('Worker failed') from reply
        return reply

    def request(self, command):
        payload = json.dumps(command).encode()
        self.process.stdin.write(struct.pack('<I', len(payload)) + payload)
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.stop.set()
        self.monitor.join()
        self.reader.join(timeout=5)
        self.process.stdout.close()
        if self.limit:
            self.limit.close()


def command(mode, before, prefix):
    query = dict(id=1, session=1, before=before)
    if mode == 'Predict':
        query.update(candidates=CANDIDATES, limit=5)
    else:
        query.update(prefix=prefix, exclude=[], limit=3)
    return {mode: query}


def compare(previous, current):
    for key in ['format_version', 'platform', 'threads', 'runs', 'repeats', 'warmup',
                'capabilities', 'model_sha256', 'manifest_sha256', 'workload_sha256']:
        if previous[key] != current[key]:
            raise ValueError(f'Incompatible comparison: {key}')
    if previous.get('throttle') != current.get('throttle'):
        raise ValueError('Incompatible comparison: throttle')
    if previous.get('thread_environment') != current.get('thread_environment'):
        raise ValueError('Incompatible comparison: thread_environment')
    changed = [k for k in previous['output_hashes'].keys() | current['output_hashes'].keys()
               if previous['output_hashes'].get(k) != current['output_hashes'].get(k)]
    def change(old, new):
        return (new / old - 1) * 100 if old else None
    return dict(changed_output_cases=sorted(changed),
                startup_median_change_percent=change(previous['startup_ms']['median'], current['startup_ms']['median']),
                memory_median_change_percent={k: change(previous['memory'][k]['median'], v['median'])
                                              for k, v in current['memory'].items()},
                latency_median_change_percent={k: change(previous['latency_ms'][k]['median'], v['median'])
                                               for k, v in current['latency_ms'].items()})


def benchmark(args):
    times = collections.defaultdict(list)
    outputs = collections.defaultdict(set)
    memory = collections.defaultdict(list)
    startup = []
    cpu_usage = []
    deadline_exceeded = collections.Counter()
    result_counts = collections.defaultdict(collections.Counter)
    capabilities = None
    for run in range(args.runs):
        started = time.perf_counter()
        client = Worker(args.worker, args.bundle, args.threads, args.cpu_percent, args.cpu_affinity)
        try:
            ready = client.receive()
            if not isinstance(ready, dict) or ready.get('Ready', {}).get('version') != 3:
                raise RuntimeError('Expected protocol 3 worker')
            if capabilities is not None and ready != capabilities:
                raise RuntimeError('Worker capabilities changed')
            capabilities = ready
            startup.append((time.perf_counter() - started) * 1000)
            memory['ready_rss_bytes'].append(client.rss())
            request_started = time.perf_counter()
            cpu_started = client.info.cpu_times()
            for iteration in range(args.warmup + args.repeats):
                for mode in ['Predict', 'Generate', 'GeneratePhrases']:
                    for name, before, prefix in CASES:
                        if client.request('Reset') != 'Reset':
                            raise RuntimeError('Reset failed')
                        for cached in [False, True]:
                            started = time.perf_counter()
                            reply = client.request(command(mode, before, prefix))
                            elapsed = (time.perf_counter() - started) * 1000
                            expected = dict(Predict='Ranked', Generate='Generated',
                                            GeneratePhrases='GeneratedPhrases')[mode]
                            result = reply.get(expected, {})
                            if result.get('id') != 1 or result.get('cache_hit') != cached:
                                raise RuntimeError('Unexpected response or cache behavior')
                            if iteration >= args.warmup:
                                key = f'{mode}/{name}/{"hit" if cached else "miss"}'
                                times[key].append(elapsed)
                                deadline_exceeded[key] += elapsed > 2000
                                words = result.get('words', result.get('phrases'))
                                result_counts[key]['returned_results'] += len(words)
                                result_counts[key]['empty_responses'] += not words
                                outputs[key].add(hashlib.sha256(json.dumps(words).encode()).hexdigest())
            memory['warmed_rss_bytes'].append(client.rss())
            memory['sampled_peak_rss_bytes'].append(max(client.peak, client.rss()))
            cpu_ended = client.info.cpu_times()
            cpu_usage.append(100 * (cpu_ended.user + cpu_ended.system - cpu_started.user - cpu_started.system)
                             / (time.perf_counter() - request_started))
        finally:
            client.close()
        print(f'Completed process {run + 1}/{args.runs}', flush=True)
    report = dict(format_version=1, platform=platform.platform(), processor=platform.processor(),
                  logical_cpus=os.cpu_count(), threads=args.threads, runs=args.runs,
                  thread_environment=dict(RAYON_NUM_THREADS=str(args.threads),
                                          CANDLE_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM='false'),
                  repeats=args.repeats, warmup=args.warmup, capabilities=capabilities,
                  worker_sha256=sha(args.worker), model_sha256=sha(args.bundle / 'model.gguf'),
                  manifest_sha256=sha(args.bundle / 'model-bundle.json'),
                  workload_sha256=hashlib.sha256(json.dumps([CASES, CANDIDATES]).encode()).hexdigest(),
                  startup_ms=summary(startup), memory={k: summary(v) for k, v in memory.items()},
                  latency_ms={k: summary(v) for k, v in times.items()},
                  output_hashes={k: sorted(v) for k, v in outputs.items()},
                  requests_over_2000ms=dict(deadline_exceeded),
                  result_counts={k: dict(v) for k, v in result_counts.items()},
                  observed_cpu_percent_of_one_processor=summary(cpu_usage),
                  memory_method='Worker RSS sampled every 5 ms; brief peaks may be missed. '
                  'Startup uses a fresh process with OS file caches uncontrolled.')
    if args.cpu_percent is not None:
        from neural_throttle import cpu_rate
        rate = cpu_rate(args.cpu_percent, os.cpu_count())
        report['throttle'] = dict(method='Windows Job Object HARD_CAP; suspended startup',
                                  requested_percent_of_one_processor=args.cpu_percent,
                                  job_cpu_rate=rate, effective_percent_of_one_processor=rate * os.cpu_count() / 100,
                                  affinity=args.cpu_affinity)
    if args.compare:
        previous = json.loads(args.compare.read_bytes())
        report['comparison'] = dict(reference_sha256=sha(args.compare), **compare(previous, report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(f'Report: {args.output}')
    if args.compare and report['comparison']['changed_output_cases']:
        raise SystemExit('Prediction outputs changed; see comparison in report')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--threads', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--cpu-percent', type=float,
                        help='Windows hard CPU budget as percent of ONE logical processor, 0..100')
    parser.add_argument('--cpu-affinity', type=int, nargs='+',
                        help='Windows logical processor IDs; defaults to first allowed CPU when throttled')
    args = parser.parse_args()
    if min(args.runs, args.repeats) < 1 or args.warmup < 0:
        parser.error('runs/repeats must be positive and warmup nonnegative')
    if args.cpu_percent is not None:
        if os.name != 'nt' or not math.isfinite(args.cpu_percent) or not 0 < args.cpu_percent <= 100:
            parser.error('--cpu-percent requires Windows and a finite value greater than 0 and at most 100')
        import psutil
        allowed = psutil.Process().cpu_affinity()
        args.cpu_affinity = args.cpu_affinity or [allowed[0]]
        if len(set(args.cpu_affinity)) != len(args.cpu_affinity) or not set(args.cpu_affinity) <= set(allowed):
            parser.error('--cpu-affinity must name distinct allowed logical processors')
        if not 1 <= os.cpu_count() <= 64:
            parser.error('CPU caps currently support single-group machines with at most 64 logical processors')
    elif args.cpu_affinity is not None:
        parser.error('--cpu-affinity requires --cpu-percent')
    benchmark(args)


if __name__ == '__main__':
    main()
