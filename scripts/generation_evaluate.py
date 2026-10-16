#!/usr/bin/env python3
"""Compare six reserved slots against reranking and six statistical suggestions,
or, with --phrases, two-word phrases from both engines.

Only frozen synthetic corpus text is submitted. Reports contain aggregate counts,
timing, hashes and process-tree RSS, never words or prompts. No desktop input.
Requires psutil and regex. No personal database is opened.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import time

from neural_evaluate import Client, ROOT, command, natural_workload, sha, timing, words, workload


def generated(client, before, prefix, command='generate', statuses=None):
    # Match the desktop's changed-context reset. The comparison mode must not
    # prefill generation's context cache before the measured request.
    client.process.stdin.write('{"command":"reset"}\n')
    client.process.stdin.flush()
    assert client.receive()['type'] == 'reset'
    started = time.perf_counter()
    client.process.stdin.write(json.dumps(dict(command=command, before=before,
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
        # Generation failures are counted apart from the comparison's rerank failures.
        (client.failure_statuses if statuses is None else statuses)[json.dumps(reply.get('status'))] += 1
        return immediate, None, elapsed, total
    assert reply['result']['request_id'] == immediate['result']['request_id']
    return immediate, reply, elapsed, total


def sample(queries, samples):
    # Cover every corpus partition and prefix length, even in a bounded run.
    if not samples:
        return queries
    return [queries[i * len(queries) // samples % len(queries)] for i in range(samples)]


def following():
    """The word after each fixture position in the same sentence, keyed as the
    workload keys queries. Phrases never cross sentence punctuation."""
    result = {}
    for name in ['regression.json', 'general-writing.json']:
        for texts in json.loads((ROOT / 'neural/fixtures' / name).read_bytes()).values():
            for text in texts:
                tokens = words(text)
                start = 0
                for sentence in re.split(r'[.!?\n\r]', text):
                    count = len(words(sentence))
                    for i in range(start, start + count - 1):
                        result.setdefault((' '.join(tokens[:i]), tokens[i]), tokens[i + 1])
                    start += count
                assert start == len(tokens)
    return result


def evaluate_phrases(args, queries):
    """Each query whose target has a following word in its sentence asks both
    engines for phrases. The target phrase is the target and that word.
    Sampling follows that filter, so a bounded run measures `samples` queries."""
    after = following()
    targets = [(q, after.get((q[4], q[6]))) for q in queries]
    targets = [(q, [q[6], nxt]) for q, nxt in targets if nxt]
    targets = sample(targets, args.samples)
    if args.natural_context:
        natural = natural_workload([q for q, _ in targets])
        targets = [(n, phrase) for n, (_, phrase) in zip(natural, targets)]
    client = Client(command(args), getattr(args, 'cpus', None))
    times = collections.defaultdict(list)
    cells = collections.defaultdict(collections.Counter)
    failures = collections.Counter()
    warmup_failures = 0
    # Ordered generated outputs: compares builds and settings without storing words.
    digest = hashlib.sha256()
    try:
        for query, _ in targets[:20]:
            _, reply, _, _ = generated(client, query[4], query[5], 'generate_phrases')
            if reply is None:
                warmup_failures += 1
                client.retry()
        client.failure_statuses.clear()  # Count measured queries only.
        for index, ((_, part, domain, n, before, prefix, _), target) in enumerate(targets):
            immediate, reply, latency, total = generated(client, before, prefix, 'generate_phrases')
            statistical = immediate['result']['phrases']
            neural = reply['result']['phrases'] if reply else []
            digest.update(json.dumps(neural, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
            for phrases in [statistical, neural]:
                assert len(phrases) <= 3 and all(len(p) == 2 for p in phrases)
                assert len({tuple(p) for p in phrases}) == len(phrases)
                assert all(p[0].startswith(prefix) for p in phrases)
            assert not {tuple(p) for p in neural} & {tuple(p) for p in statistical}
            times['immediate_with_ipc'].append(latency)
            if reply:
                times['generation_with_ipc'].append(total)
            else:
                failures['generation'] += 1
            cell = cells[f'{part}/{domain}/{n}']
            cell['queries'] += 1
            cell['statistical_phrases'] += len(statistical)
            cell['generated_phrases'] += len(neural)
            cell['filled_neural_queries'] += bool(neural)
            cell['generation_failures'] += reply is None
            cell['generation_context_hits'] += bool(reply and reply['result']['cache_hit'])
            for mode, phrases in [('statistical', statistical), ('neural', neural),
                                  ('six_slots', statistical + neural)]:
                cell[mode + '_phrase_hits'] += target in phrases
                cell[mode + '_first_word_hits'] += any(p[0] == target[0] for p in phrases)
            cell['gains_vs_statistical'] += target not in statistical and target in neural
            if reply is None:
                client.retry()  # Explicit benchmark recovery, not production policy.
            if (index + 1) % 100 == 0:
                print(json.dumps({'completed': index + 1, 'failures': dict(failures)}), flush=True)
    finally:
        client.close()
    for cell in cells.values():
        for mode in ['statistical', 'neural', 'six_slots']:
            cell[mode + '_phrase_accuracy'] = cell[mode + '_phrase_hits'] / cell['queries']
            cell[mode + '_first_word_accuracy'] = cell[mode + '_first_word_hits'] / cell['queries']
        cell['neural_slot_fill_rate'] = cell['generated_phrases'] / (cell['queries'] * 3)
    totals = collections.Counter()
    for cell in cells.values():
        totals.update({k: v for k, v in cell.items() if isinstance(v, int)})
    summary = {mode + '_phrase_accuracy': totals[mode + '_phrase_hits'] / max(totals['queries'], 1)
               for mode in ['statistical', 'neural', 'six_slots']}
    report = dict(schema=1, mode='phrases',
                  policy=json.loads((ROOT / 'neural/model-bundle.json').read_bytes())['policy'],
                  platform=platform.platform(), queries=len(targets), warmup_queries=min(20, len(targets)),
                  warmup_failures=warmup_failures, failures=dict(failures), summary=summary,
                  failure_statuses=dict(client.failure_statuses),
                  latency={k: timing(v) for k, v in times.items()}, cold_load_ms=client.cold_ms,
                  peak_process_tree_rss_bytes=client.peak, peak_process_tree_threads=client.peak_threads,
                  process_tree_cpu_affinity=client.affinity, generated_sha256=digest.hexdigest(),
                  capabilities=client.ready['capabilities'],
                  natural_context=args.natural_context,
                  cells=dict(cells), baseline_sha256=sha(args.baseline), worker_sha256=sha(args.worker),
                  accelerated_worker_sha256=sha(args.accelerated_worker) if args.accelerated_worker else None,
                  fixture_hashes={n: sha(ROOT / 'neural/fixtures' / n) for n in ['regression.json', 'general-writing.json']},
                  note='A hit needs both words. Fixture overlap with model pretraining is unknown. Explicit benchmark retries counted; production never retries automatically.',
                  production_qualified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def evaluate(args):
    queries = workload()
    if args.phrases:
        return evaluate_phrases(args, queries)
    if args.natural_context:
        queries = natural_workload(queries)
    queries = sample(queries, args.samples)
    db = sqlite3.connect(args.baseline.resolve().as_uri() + '?mode=ro', uri=True)
    vocabulary = {row[0] for row in db.execute('SELECT word FROM vocabulary')}
    db.close()
    client = Client(command(args), getattr(args, 'cpus', None))
    times = collections.defaultdict(list)
    cells = collections.defaultdict(collections.Counter)
    failures = collections.Counter()
    warmup_failures = 0
    # Ordered generated outputs: compares builds and settings without storing words.
    digest = hashlib.sha256()
    try:
        for query in queries[:20]:
            _, reply, _, _ = generated(client, query[4], query[5])
            if reply is None:
                warmup_failures += 1
                client.retry()
        client.failure_statuses.clear()  # Count measured queries only.
        generation_statuses = collections.Counter()
        for index, (_, part, domain, n, before, prefix, target) in enumerate(queries):
            old, ranked, _, _ = client.query(before, prefix)
            old_words = ranked['result']['words'] if ranked else old['result']['words']
            if old['result']['refinement_requested'] and ranked is None:
                failures['rerank'] += 1
                client.retry()
            immediate, reply, latency, total = generated(client, before, prefix, statuses=generation_statuses)
            instant = immediate['result']['words']
            stats = immediate['result']['statistical_six']
            assert instant == stats[:3]
            words = reply['result']['words'] if reply else []
            digest.update(json.dumps(words, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
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
            cell['generation_context_hits'] += bool(reply and reply['result']['cache_hit'])
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
                  failure_statuses=dict(generation_statuses),
                  rerank_failure_statuses=dict(client.failure_statuses),
                  latency={k: timing(v) for k, v in times.items()}, cold_load_ms=client.cold_ms,
                  peak_process_tree_rss_bytes=client.peak, peak_process_tree_threads=client.peak_threads,
                  process_tree_cpu_affinity=client.affinity, generated_sha256=digest.hexdigest(),
                  capabilities=client.ready['capabilities'],
                  natural_context=args.natural_context,
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
    parser.add_argument('--threads', type=int, choices=range(1, 5), help='worker threads; the CLI defaults to 4')
    parser.add_argument('--samples', type=int, default=0, help='0 uses the full frozen workload')
    parser.add_argument('--natural-context', action='store_true',
                        help='send preceding text with its original case and punctuation')
    parser.add_argument('--phrases', action='store_true',
                        help='measure two-word phrases instead of six word slots')
    args = parser.parse_args()
    if args.samples < 0:
        parser.error('samples must be nonnegative')
    evaluate(args)


if __name__ == '__main__':
    main()
