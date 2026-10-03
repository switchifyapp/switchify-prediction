#!/usr/bin/env python3
"""Score the pinned native FUTO three-suggestion decoder on exported synthetic queries."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import unicodedata

from aac_experiment import sha
from general_neural_experiment import windows_peak

MODEL_SHA = '6545c1c9ef2d76e9bfb87ad4fcf2061889513af84fcf30d907412be7fcdedb7b'


def normalize_word(text):
    word = unicodedata.normalize('NFC', text.strip().replace('’', "'").lower())
    if not word:
        return None
    for i, c in enumerate(word):
        if c.isalpha() or (unicodedata.category(c).startswith('M') and i > 0):
            continue
        if c == "'" and i > 0 and i + 1 < len(word) and word[i + 1].isalpha():
            continue
        return None
    return word


def summarize(queries, lines):
    if len(lines) != len(queries) + 2 or not lines[0].startswith('load\t'):
        raise ValueError('Incomplete native output')
    cold_load = float(lines[0].split('\t')[1])
    prime = lines[1].split('\t')
    if not math.isfinite(cold_load) or cold_load < 0 or len(prime) < 3 or prime[:2] != ['0', 'ok']:
        raise ValueError('Invalid load or warm-up result')
    cells, latencies = {}, []
    filtered, unsupported = 0, 0
    # First query is an untimed prime, then each frozen query is measured once.
    for index, (q, line) in enumerate(zip(queries, lines[2:])):
        parts = line.split('\t')
        if len(parts) < 3 or parts[1] != 'ok':
            raise ValueError('Native inference did not report success')
        if int(parts[0]) != index + 1:
            raise ValueError('Native query order mismatch')
        ms = float(parts[2])
        if not math.isfinite(ms) or ms < 0:
            raise ValueError('Invalid timing')
        latencies.append(ms)
        native, exact = [], []
        for raw in parts[3:]:
            word = normalize_word(raw)
            if word and word not in native:
                native.append(word)
                if word.startswith(q['prefix']):
                    exact.append(word)
                else:
                    filtered += 1
        if any(c < 'a' or c > 'z' for c in q['prefix']):
            unsupported += 1
        cell = cells.setdefault(q['domain'], {}).setdefault(str(q['prefix_chars']),
            dict(queries=0, native_top1_hits=0, native_top3_hits=0,
                 exact_prefix_top1_hits=0, exact_prefix_top3_hits=0, empty_exact_results=0))
        cell['queries'] += 1
        cell['native_top1_hits'] += int(bool(native) and native[0] == q['target'])
        cell['native_top3_hits'] += int(q['target'] in native[:3])
        cell['exact_prefix_top1_hits'] += int(bool(exact) and exact[0] == q['target'])
        cell['exact_prefix_top3_hits'] += int(q['target'] in exact[:3])
        cell['empty_exact_results'] += int(not exact)
    latencies.sort()
    percentile = lambda p: latencies[math.ceil(len(latencies) * p) - 1]
    return dict(query_count=len(queries), successful_inference_queries=len(queries) - unsupported, cold_load_ms=cold_load,
                warm_median_ms=percentile(.5), warm_p95_ms=percentile(.95), warm_max_ms=latencies[-1],
                filtered_non_prefix_suggestions=filtered, unsupported_prefix_queries=unsupported,
                failed_queries=0, cells=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('executable', 'model', 'queries', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if sha(args.model) != MODEL_SHA:
        raise ValueError('FUTO model checksum mismatch')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    queries = json.loads(args.queries.read_text(encoding='utf-8'))
    if not queries:
        raise ValueError('Empty workload')
    for q in queries:
        if any(c in q['before'] + q['prefix'] for c in '\r\n\t'):
            raise ValueError('Unsafe fixture transport')
    fixture = out / 'queries.tsv'
    fixture.write_bytes(''.join(q['before'] + '\t' + q['prefix'] + '\n'
                               for q in [queries[0]] + queries).encode())
    peak = None
    with (out / 'native.tsv').open('wb') as stdout, (out / 'native.log').open('wb') as stderr:
        process = subprocess.Popen([str(args.executable.resolve()), str(args.model.resolve()), str(fixture)],
                                   stdout=stdout, stderr=stderr)
        deadline = time.monotonic() + 1800
        while process.poll() is None:
            if time.monotonic() > deadline:
                process.kill()
                process.wait()
                raise TimeoutError('FUTO benchmark exceeded 30 minutes')
            if os.name == 'nt':
                current = windows_peak(process)
                if current is not None:
                    peak = max(peak or 0, current)
            time.sleep(.05)
        if process.returncode:
            raise RuntimeError(f'Native FUTO exited {process.returncode}; inspect native.log')
    report = summarize(queries, (out / 'native.tsv').read_text(encoding='utf-8').splitlines())
    report.update(protocol='futo-native-v1', model_sha256=MODEL_SHA,
                  model_file_bytes=args.model.stat().st_size, executable_sha256=sha(args.executable),
                  queries_sha256=sha(args.queries), hardware=platform.platform(),
                  peak_process_working_set_bytes=peak, threads=1, results_limit=3,
                  promote=False, note='Native no-coordinate correction path, then exact-prefix filtering. No backfill. Not a five-slot reranker; not validated against an Android device.')
    (out / 'results.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())


if __name__ == '__main__':
    main()
