#!/usr/bin/env python3
"""Reproduce the frozen SmolLM2 regression and progressive-prefix comparison.

Requires psutil and regex. Synthetic fixture text travels through stdin only.
Reports contain counts, hashes and timing, never captured application text.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import platform
import queue
import subprocess
import threading
import time
import unicodedata

import psutil
import regex

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def words(text):
    text = unicodedata.normalize('NFC', text.replace('’', "'").lower())
    return regex.findall(r"\p{L}[\p{L}\p{M}]*(?:'\p{L}[\p{L}\p{M}]*)*", text)


def key(text):
    return hashlib.sha256(text.encode()).hexdigest()


def workload():
    result = []
    regression = json.loads((ROOT / 'neural/fixtures/regression.json').read_bytes())
    for domain, texts in sorted(regression.items()):
        for n in range(5):
            cell, seen = [], set()
            for sentence in sorted({' '.join(words(text)) for text in texts}):
                tokens = sentence.split()
                for i, target in enumerate(tokens):
                    graphemes = regex.findall(r'\X', target)
                    if len(graphemes) <= n:
                        continue
                    before, prefix = ' '.join(tokens[:i]), ''.join(graphemes[:n])
                    if (before, prefix, target) in seen:
                        continue
                    seen.add((before, prefix, target))
                    cell.append((key(f'{domain}\n{sentence}\n{i}\n{n}'), 'regression', domain, n, before, prefix, target))
            assert len(cell) >= 64
            result.extend(sorted(cell)[:64])
    extra = json.loads((ROOT / 'neural/fixtures/general-writing.json').read_bytes())
    for part, start, end, count in [('development', 0, 6, 8), ('test', 6, 12, 16)]:
        for domain, texts in sorted(extra.items()):
            positions = []
            for text in texts[start:end]:
                tokens = words(text)
                sentence = ' '.join(tokens)
                for i, target in enumerate(tokens):
                    if len(regex.findall(r'\X', target)) > 4:
                        positions.append((key(f'{domain}\n{sentence}\n{i}'), tokens, i))
            assert len(positions) >= count
            for identifier, tokens, i in sorted(positions)[:count]:
                graphemes = regex.findall(r'\X', tokens[i])
                for n in range(5):
                    result.append((identifier + str(n), part, domain, n, ' '.join(tokens[:i]), ''.join(graphemes[:n]), tokens[i]))
    return result


def timing(values):
    values = sorted(values)
    if not values:
        return None
    return {'samples': len(values), 'median_ms': values[len(values)//2],
            'p95_ms': values[math.ceil(len(values)*.95)-1], 'max_ms': values[-1]}


class Client:
    def __init__(self, command):
        self.started = time.perf_counter()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding='utf-8')
        self.messages = queue.Queue()
        self.stop = threading.Event()
        self.peak = 0
        def read():
            try:
                for line in self.process.stdout:
                    self.messages.put(json.loads(line))
            finally:
                self.messages.put(None)
        def memory():
            parent = psutil.Process(self.process.pid)
            while not self.stop.is_set():
                try:
                    total = sum(p.memory_info().rss for p in [parent] + parent.children(recursive=True))
                    self.peak = max(self.peak, total)
                except psutil.Error:
                    pass
                self.stop.wait(.01)
        self.reader = threading.Thread(target=read, daemon=True)
        self.monitor = threading.Thread(target=memory, daemon=True)
        self.reader.start()
        self.monitor.start()
        self.ready = self.receive()
        if self.ready is None or self.ready['type'] != 'ready':
            self.close()
            raise RuntimeError('Worker did not become ready')
        self.cold_ms = (time.perf_counter()-self.started)*1000

    def receive(self):
        return self.messages.get(timeout=40)

    def query(self, before, prefix):
        start = time.perf_counter()
        self.process.stdin.write(json.dumps({'command':'predict', 'before':before, 'prefix':prefix,
                                            'session':1, 'min_chars':0})+'\n')
        self.process.stdin.flush()
        immediate = self.receive()
        assert immediate['type'] == 'immediate'
        ipc_immediate = (time.perf_counter()-start)*1000
        if not immediate['result']['refinement_requested']:
            return immediate, None, ipc_immediate, None
        refined = self.receive()
        elapsed = (time.perf_counter()-start)*1000
        if refined['type'] != 'refined':
            return immediate, None, ipc_immediate, elapsed
        assert refined['result']['request_id'] == immediate['result']['request_id']
        return immediate, refined, ipc_immediate, elapsed

    def retry(self):
        start = time.perf_counter()
        self.process.stdin.write('{"command":"retry"}\n')
        self.process.stdin.flush()
        while True:
            message = self.receive()
            if message is None:
                raise RuntimeError('Worker exited during explicit benchmark retry')
            if message.get('status') == 'Ready':
                return (time.perf_counter()-start)*1000
            if isinstance(message.get('status'), dict):
                raise RuntimeError('Worker retry failed')

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.stop.set()
        self.monitor.join()
        self.reader.join()
        self.process.stdout.close()
        self.process.stderr.close()


def evaluate(args):
    queries = workload()
    training = {' '.join(words(line)) for line in args.training.read_text(encoding='utf-8').splitlines()}
    for name in ['regression', 'general-writing']:
        fixtures = json.loads((ROOT / f'neural/fixtures/{name}.json').read_bytes())
        assert not training.intersection(' '.join(words(text)) for texts in fixtures.values() for text in texts)
    command = [str(args.cli.resolve()), 'stream', '--baseline', str(args.baseline.resolve()),
               '--bundle', str(args.bundle.resolve()), '--worker', str(args.worker.resolve())]
    if args.accelerated_worker:
        command += ['--accelerated-worker', str(args.accelerated_worker.resolve())]
    client = Client(command)
    times = collections.defaultdict(list)
    cells = collections.defaultdict(lambda: {'queries':0, 'baseline_top1':0, 'baseline_top5':0,
                                            'refined_top1':0, 'refined_top5':0, 'failures':0})
    digest = hashlib.sha256()
    failures = 0
    reload_ms = []
    try:
        for query in queries[:20]:
            client.query(query[4], query[5])
        for index, (_, part, domain, n, before, prefix, target) in enumerate(queries):
            immediate, refined, ipc_immediate, elapsed = client.query(before, prefix)
            baseline = immediate['result']['words']
            final = refined['result']['words'] if refined else baseline
            failed = refined is None and immediate['result']['refinement_requested']
            failures += int(failed)
            times['immediate'].append(immediate['elapsed_ms'])
            times['immediate_with_ipc'].append(ipc_immediate)
            if refined:
                times['refinement_with_ipc'].append(elapsed)
                times['context_hit' if refined['result']['cache_hit'] else 'context_miss'].append(elapsed)
            cell = cells[f'{part}/{domain}/{n}']
            cell['queries'] += 1
            cell['baseline_top1'] += int(bool(baseline) and baseline[0] == target)
            cell['baseline_top5'] += int(target in baseline)
            cell['refined_top1'] += int(bool(final) and final[0] == target)
            cell['refined_top5'] += int(target in final)
            cell['failures'] += int(failed)
            digest.update(json.dumps(final, ensure_ascii=False, separators=(',', ':')).encode()+b'\n')
            if failed:
                # The benchmark caller explicitly retries; the library never auto-reloads.
                reload_ms.append(client.retry())
            if index % 200 == 0:
                print(f'{index}/{len(queries)} completed', flush=True)
    finally:
        client.close()
    early_failures = [name for name, cell in cells.items() if int(name.rsplit('/',1)[1]) < 3
                      and (cell['baseline_top5']-cell['refined_top5'])*100/cell['queries'] > 1]
    total_baseline = sum(c['baseline_top5'] for c in cells.values())
    total_refined = sum(c['refined_top5'] for c in cells.values())
    report = {'platform':platform.platform(), 'processor':platform.processor(), 'logical_cpus':psutil.cpu_count(),
              'capabilities':client.ready['capabilities'], 'cold_load_ms':client.cold_ms,
              'process_tree_peak_rss_bytes':client.peak, 'memory_method':'10ms sum of parent and descendants RSS; shared pages may count twice',
              'queries':len(queries), 'failures':failures, 'timings':{k:timing(v) for k,v in times.items()},
              'explicit_retry_load_ms':reload_ms,
              'cells':dict(cells), 'prediction_sha256':digest.hexdigest(),
              'inputs':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:sha(p) for p in
                        [ROOT/'neural/evaluation-protocol.json', ROOT/'neural/fixtures/regression.json', ROOT/'neural/fixtures/general-writing.json',
                         args.baseline.resolve(), args.training.resolve(), args.cli.resolve(), args.worker.resolve()]},
              'gates':{'overall_top5':total_refined >= total_baseline, 'early_cell_regressions':early_failures,
                       'immediate_latency':timing(times['immediate'])['p95_ms'] < 20,
                       'refinement_latency':bool(times['refinement_with_ipc']) and timing(times['refinement_with_ipc'])['p95_ms'] < 150,
                       'minimum_successful_samples':len(times['refinement_with_ipc']) >= 1000, 'no_failures':failures == 0},
              'interpretation':'Frozen regression comparison, unknown pretraining overlap; no automatic promotion.'}
    if args.accelerated_worker:
        report['accelerated_worker_sha256'] = sha(args.accelerated_worker)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cli', 'worker', 'baseline', 'bundle', 'training', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--accelerated-worker', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists')
    evaluate(args)


if __name__ == '__main__':
    main()
