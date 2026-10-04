#!/usr/bin/env python3
"""Compare six reserved slots against reranking and six statistical suggestions.

Only frozen synthetic corpus text is submitted. Reports contain aggregate counts,
timing, hashes and process-tree RSS, never words or prompts. No desktop input.
Requires psutil and regex. No personal database is opened.
"""
import argparse
import collections
import json
from pathlib import Path
import platform
import sqlite3
import time

from neural_evaluate import Client, ROOT, sha, timing, workload


def generated(client, before, prefix):
    started = time.perf_counter()
    client.process.stdin.write(json.dumps(dict(command='generate', before=before,
                                              prefix=prefix, session=1)) + '\n')
    client.process.stdin.flush()
    immediate = client.receive()
    assert immediate['type'] == 'immediate'
    elapsed = (time.perf_counter() - started) * 1000
    if not immediate['result']['refinement_requested']:
        return immediate, None, elapsed, None
    reply = client.receive()
    total = (time.perf_counter() - started) * 1000
    if reply['type'] != 'refined':
        return immediate, None, elapsed, total
    assert reply['result']['request_id'] == immediate['result']['request_id']
    return immediate, reply, elapsed, total


def evaluate(args):
    queries = workload()
    # Cover every corpus partition and prefix length, even in a bounded run.
    if args.samples:
        queries = [queries[i * len(queries) // args.samples % len(queries)]
                   for i in range(args.samples)]
    db = sqlite3.connect(args.baseline.resolve().as_uri() + '?mode=ro', uri=True)
    vocabulary = {row[0] for row in db.execute('SELECT word FROM vocabulary')}
    db.close()
    command = [str(args.cli.resolve()), 'stream', '--baseline', str(args.baseline.resolve()),
               '--bundle', str(args.bundle.resolve()), '--worker', str(args.worker.resolve())]
    if args.accelerated_worker:
        command += ['--accelerated-worker', str(args.accelerated_worker.resolve())]
    client = Client(command)
    times = collections.defaultdict(list)
    cells = collections.defaultdict(collections.Counter)
    failures = collections.Counter()
    warmup_failures = 0
    try:
        for query in queries[:20]:
            _, reply, _, _ = generated(client, query[4], query[5])
            if reply is None:
                warmup_failures += 1
                client.retry()
        for index, (_, part, domain, n, before, prefix, target) in enumerate(queries):
            old, ranked, _, _ = client.query(before, prefix)
            old_words = ranked['result']['words'] if ranked else old['result']['words']
            if old['result']['refinement_requested'] and ranked is None:
                failures['rerank'] += 1
                client.retry()
            immediate, reply, latency, total = generated(client, before, prefix)
            instant = immediate['result']['words']
            stats = immediate['result']['statistical_six']
            assert instant == stats[:3]
            words = reply['result']['words'] if reply else []
            assert len(words) <= 3 and not set(words).intersection(instant)
            assert len(set(words)) == len(words)
            slots = instant + [None] * (3 - len(instant)) + words + [None] * (3 - len(words))
            times['immediate_with_ipc'].append(latency)
            if reply:
                times['generation_with_ipc'].append(total)
            else:
                failures['generation'] += 1
            cell = cells[f'{part}/{domain}/{n}']
            cell['queries'] += 1
            cell['generated_words'] += len(words)
            cell['filled_neural_queries'] += bool(words)
            cell['generated_oov_words'] += sum(w not in vocabulary for w in words)
            cell['oov_targets'] += target not in vocabulary
            cell['oov_hits'] += target not in vocabulary and target in words
            cell['generation_failures'] += reply is None
            for mode, values in [('current', old_words), ('statistical_six', stats), ('six_slots', slots)]:
                cell[mode + '_top3_hits'] += target in values[:3]
                cell[mode + '_top6_hits'] += target in values[:6]
            cell['regressions_vs_statistical_six'] += target in stats and target not in slots
            cell['gains_vs_statistical_six'] += target not in stats and target in slots
            cell['regressions_vs_current'] += target in old_words and target not in slots
            if reply is None:
                client.retry()  # Explicit benchmark recovery, not production policy.
            if (index + 1) % 100 == 0:
                print(json.dumps({'completed': index + 1, 'failures': dict(failures)}), flush=True)
    finally:
        client.close()
    for cell in cells.values():
        for mode in ['current', 'statistical_six', 'six_slots']:
            for k in [3, 6]:
                cell[f'{mode}_top{k}_accuracy'] = cell[f'{mode}_top{k}_hits'] / cell['queries']
        cell['neural_slot_fill_rate'] = cell['generated_words'] / (cell['queries'] * 3)
    report = dict(schema=1, policy=json.loads((ROOT / 'neural/model-bundle.json').read_bytes())['policy'],
                  platform=platform.platform(), queries=len(queries), warmup_queries=min(20, len(queries)),
                  warmup_failures=warmup_failures, failures=dict(failures),
                  latency={k: timing(v) for k, v in times.items()}, cold_load_ms=client.cold_ms,
                  peak_process_tree_rss_bytes=client.peak, capabilities=client.ready['capabilities'],
                  cells=dict(cells), baseline_sha256=sha(args.baseline), worker_sha256=sha(args.worker),
                  accelerated_worker_sha256=sha(args.accelerated_worker) if args.accelerated_worker else None,
                  fixture_hashes={n: sha(ROOT / 'neural/fixtures' / n) for n in ['regression.json', 'general-writing.json']},
                  note='Regression comparison only; fixture overlap with model pretraining is unknown. Current mode returns at most five words. Explicit benchmark retries counted; production never retries automatically.',
                  production_qualified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cli', 'baseline', 'bundle', 'worker', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--accelerated-worker', type=Path)
    parser.add_argument('--samples', type=int, default=0, help='0 uses the full frozen workload')
    args = parser.parse_args()
    if args.samples < 0:
        parser.error('samples must be nonnegative')
    evaluate(args)


if __name__ == '__main__':
    main()
