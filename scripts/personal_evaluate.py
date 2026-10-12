#!/usr/bin/env python3
"""Measure what learning adds to six-slot prediction, on a synthetic person.

A temporary personal database learns neural/fixtures/personal.json `learn` text
through the CLI. Held-out `test` sentences are then queried with and without
learning. The fixture is synthetic, written for this evaluation; results show
the mechanism, not real users' accuracy.
Reports contain counts and timing, never words. Requires psutil and regex.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

from neural_evaluate import Client, ROOT, timing, words

FIXTURE = ROOT / 'neural/fixtures/personal.json'


def queries(baseline_counts):
    import regex
    fixture = json.loads(FIXTURE.read_bytes())
    learned = {w for sentence in fixture['learn'] for w in words(sentence)}
    result = []
    for sentence in fixture['test']:
        text = sentence.replace('’', "'")
        for match in regex.finditer(r"\p{L}[\p{L}\p{M}]*(?:'\p{L}[\p{L}\p{M}]*)*", text):
            target = words(match.group())[0]
            graphemes = regex.findall(r'\X', target)
            for n in range(min(3, len(graphemes))):
                # Personal: learned, and rare or unknown in the baseline.
                own = target in learned and baseline_counts.get(target, 0) < 10
                result.append((text[:match.start()], ''.join(graphemes[:n]), target, own))
    return result


def run(args, personal, items):
    command = [str(args.cli), 'stream', '--baseline', str(args.baseline), '--bundle', str(args.bundle),
               '--worker', str(args.worker)]
    if args.accelerated_worker:
        command += ['--accelerated-worker', str(args.accelerated_worker)]
    if personal:
        command += ['--personal', str(personal)]
    client = Client(command)
    hits = dict(all=0, personal=0, generated=0)
    times = []
    try:
        for before, prefix, target, own in items:
            client.process.stdin.write('{"command":"reset"}\n')
            client.process.stdin.flush()
            assert client.receive()['type'] == 'reset'
            start = time.perf_counter()
            client.process.stdin.write(json.dumps(dict(command='generate', before=before,
                                                       prefix=prefix, session=1)) + '\n')
            client.process.stdin.flush()
            immediate = client.receive()
            instant = immediate['result']['words']
            generated = []
            if immediate['result']['refinement_requested']:
                reply = client.receive()
                if reply['type'] == 'refined':
                    generated = reply['result']['words']
                    times.append((time.perf_counter() - start) * 1000)
            hit = target in instant + generated
            hits['all'] += hit
            hits['personal'] += hit and own
            hits['generated'] += target in generated
    finally:
        client.close()
    count = len(items)
    own = sum(item[3] for item in items)
    return dict(top6=hits['all'] / count, personal_top6=hits['personal'] / own,
                generated_hits=hits['generated'] / count, latency=timing(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cli', 'learner', 'baseline', 'bundle', 'worker', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--accelerated-worker', type=Path)
    args = parser.parse_args()
    import sqlite3
    db = sqlite3.connect(args.baseline.resolve().as_uri() + '?mode=ro', uri=True)
    baseline_counts = dict(db.execute("SELECT word, count FROM counts WHERE context='[]'"))
    db.close()
    items = queries(baseline_counts)
    with tempfile.TemporaryDirectory() as directory:
        personal = Path(directory) / 'personal.sqlite'
        # Newlines separate context exactly as separate learn calls would.
        text = '\n'.join(json.loads(FIXTURE.read_bytes())['learn'])
        subprocess.run([str(args.learner), 'learn', '--baseline', str(args.baseline),
                        '--personal', str(personal)], input=text, text=True,
                       check=True, capture_output=True)
        report = dict(
            queries=len(items), personal_queries=sum(item[3] for item in items),
            without_learning=run(args, None, items),
            with_learning=run(args, personal, items),
            note='Synthetic fixture; shows the mechanism, not real-user accuracy.')
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
