#!/usr/bin/env python3
"""Run the frozen generation and refinement benchmarks on one machine and record
its hardware, for the low-end and GPU experiments in docs/low-end-hardware.md.

The queries, fixtures and timing come unchanged from generation_evaluate.py and
neural_evaluate.py, so every machine measures the same workload. Reports contain
hardware facts, aggregate counts, timing, hashes and process-tree RSS and thread
counts, never words or prompts. No hostname or user name is recorded. Optional
CPU restriction and experiment limits are recorded as such: a restricted fast CPU
is a simulation, not a low-end measurement. No desktop input. Requires psutil and
regex.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import generation_evaluate
import neural_evaluate
from neural_evaluate import DEADLINE_MS, ROOT, sha

# Environment read only by workers built with the benchmark-only `experiment` feature.
EXPERIMENT = {'beam_width': 'SWITCHIFY_SMOL_EXPERIMENT_BEAM_WIDTH',
              'max_forward_evaluations': 'SWITCHIFY_SMOL_EXPERIMENT_EVALUATIONS'}
ISA = ['sse4_1', 'sse4_2', 'avx', 'avx2', 'fma', 'f16c']


def binary_arch(data):
    """The machine a PE, ELF or Mach-O executable header targets."""
    if data[:2] == b'MZ' and len(data) >= 64:
        offset = int.from_bytes(data[60:64], 'little')
        if data[offset:offset + 4] == b'PE\0\0':
            machine = int.from_bytes(data[offset + 4:offset + 6], 'little')
            return {0x8664: 'x86_64', 0xAA64: 'aarch64', 0x14C: 'x86'}.get(machine, hex(machine))
    if data[:4] == b'\x7fELF':
        machine = int.from_bytes(data[18:20], 'little')
        return {62: 'x86_64', 183: 'aarch64'}.get(machine, hex(machine))
    if data[:4] in (b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe'):
        if data[:4] == b'\xca\xfe\xba\xbe':
            return 'universal'
        cpu = int.from_bytes(data[4:8], 'little')
        return {0x01000007: 'x86_64', 0x0100000C: 'aarch64'}.get(cpu, hex(cpu))
    return 'unknown'


def cpuinfo(text):
    """Model name and ISA support from Linux /proc/cpuinfo text."""
    fields = {}
    for line in text.splitlines():
        name, _, value = line.partition(':')
        fields.setdefault(name.strip(), value.strip())
    flags = set((fields.get('flags') or fields.get('Features') or '').split())
    model = fields.get('model name') or fields.get('Hardware') or fields.get('CPU part')
    isa = {name: name in flags for name in ISA} if 'flags' in fields else dict.fromkeys(ISA)
    return model, isa


def sysctl(name):
    try:
        return subprocess.run(['sysctl', '-n', name], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def efficiency_classes(kernel):
    """Windows SYSTEM_CPU_SET_INFORMATION efficiency class per logical CPU. On hybrid
    CPUs a higher class is a faster core, so a CPU restriction can be interpreted."""
    import ctypes
    size = ctypes.c_ulong()
    kernel.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buffer = ctypes.create_string_buffer(size.value)
    if not kernel.GetSystemCpuSetInformation(buffer, size, ctypes.byref(size), None, 0):
        return None
    data, offset, result = buffer.raw[:size.value], 0, {}
    while offset + 19 <= len(data):
        length = int.from_bytes(data[offset:offset + 4], 'little')
        if length < 19:
            break
        result[data[offset + 14]] = data[offset + 18]  # LogicalProcessorIndex, EfficiencyClass
        offset += length
    return [result[i] for i in sorted(result)]


def hardware():
    import psutil
    system = platform.system()
    result = {'os': platform.platform(), 'system': system, 'python_machine': platform.machine(),
              'native_machine': None, 'translated': None, 'cpu_model': None,
              'isa': dict.fromkeys(ISA), 'isa_method': None,
              'physical_cores': psutil.cpu_count(logical=False), 'logical_cpus': psutil.cpu_count(),
              'ram_bytes': psutil.virtual_memory().total}
    if system == 'Windows':
        import ctypes
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            result['cpu_model'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.IsWow64Process2.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ushort)] * 2
        process, native = ctypes.c_ushort(), ctypes.c_ushort()
        # Reports the native machine even when this x64 process is emulated on ARM64.
        if kernel.IsWow64Process2(kernel.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)):
            result['native_machine'] = {0x8664: 'x86_64', 0xAA64: 'aarch64'}.get(native.value, hex(native.value))
        # PF_SSE4_1, PF_SSE4_2, PF_AVX and PF_AVX2. Windows exposes no FMA or F16C flag;
        # the CLI's accelerated capability reports the worker's own AVX2+FMA+F16C check.
        result['efficiency_class_by_cpu'] = efficiency_classes(kernel)
        if result['native_machine'] != 'aarch64':
            for name, feature in [('sse4_1', 37), ('sse4_2', 38), ('avx', 39), ('avx2', 40)]:
                result['isa'][name] = bool(kernel.IsProcessorFeaturePresent(feature))
            result['isa_method'] = 'IsProcessorFeaturePresent'
    elif system == 'Darwin':
        result['cpu_model'] = sysctl('machdep.cpu.brand_string')
        result['translated'] = sysctl('sysctl.proc_translated') == '1'
        result['native_machine'] = 'aarch64' if sysctl('hw.optional.arm64') == '1' else 'x86_64'
        if result['native_machine'] == 'x86_64':
            features = (sysctl('machdep.cpu.features') or '').lower().replace('.', '_').split()
            leaf7 = (sysctl('machdep.cpu.leaf7_features') or '').lower().split()
            for name in ISA:
                result['isa'][name] = name in features or name in leaf7
            result['isa_method'] = 'sysctl machdep.cpu'
        for level in ['0', '1']:
            cores = sysctl(f'hw.perflevel{level}.physicalcpu')
            if cores:
                result[f'perflevel{level}_physical_cores'] = int(cores)
    elif system == 'Linux':
        result['cpu_model'], result['isa'] = cpuinfo(Path('/proc/cpuinfo').read_text(encoding='utf-8', errors='replace'))
        result['native_machine'] = platform.machine()
        result['isa_method'] = '/proc/cpuinfo'
    try:
        frequency = psutil.cpu_freq()
        # Windows reports the base clock here, not the boost clock.
        result['reported_max_mhz'] = frequency.max if frequency else None
    except (NotImplementedError, OSError):
        result['reported_max_mhz'] = None
    battery = psutil.sensors_battery() if hasattr(psutil, 'sensors_battery') else None
    result['battery'] = {'percent': battery.percent, 'power_plugged': battery.power_plugged} if battery else None
    return result


def experiment_limits(worker, limits):
    """Confirm an experiment worker accepts these limits. A production worker has
    no such mode and exits without output, so it cannot be mislabelled."""
    if not limits:
        return None
    env = dict(os.environ, **{EXPERIMENT[k]: str(v) for k, v in limits.items()})
    reply = subprocess.run([str(worker), '--experiment-limits'], capture_output=True,
                           text=True, env=env, timeout=30)
    try:
        actual = json.loads(reply.stdout)
    except json.JSONDecodeError:
        actual = None
    if reply.returncode != 0 or not isinstance(actual, dict) or any(actual.get(k) != v for k, v in limits.items()):
        raise RuntimeError('Worker is not an experiment build or rejected the limits')
    return actual


def total(cells, field):
    return sum(c[field] for c in cells.values())


def summarize(generation, refinement):
    """The issue's per-machine metrics, from the unchanged benchmark reports."""
    def latency(timing, statuses):
        timeouts = sum(n for status, n in (statuses or {}).items() if 'Timeout' in status)
        return {'median_ms': timing and timing['median_ms'], 'p95_ms': timing and timing['p95_ms'],
                'max_ms': timing and timing['max_ms'], 'successful_samples': timing and timing['samples'],
                'timeouts': timeouts,
                # A reply timeout is a missed deadline, as is a late reply the CLI still displayed.
                'deadline_misses': timeouts + (timing['at_least_deadline'] if timing else 0)}
    result = {}
    if generation:
        cells, queries = generation['cells'], generation['queries']
        result['generation'] = dict(
            latency(generation['latency'].get('generation_with_ipc'), generation['failure_statuses']),
            queries=queries, failed_requests=sum(generation['failures'].values()),
            failures=generation['failures'], cold_startup_ms=generation['cold_load_ms'],
            peak_process_tree_rss_bytes=generation['peak_process_tree_rss_bytes'],
            peak_process_tree_threads=generation['peak_process_tree_threads'],
            six_slot_top6_accuracy=total(cells, 'six_slots_top6_hits') / queries,
            statistical_six_top6_accuracy=total(cells, 'statistical_six_top6_hits') / queries,
            current_top3_accuracy=total(cells, 'current_top3_hits') / queries,
            neural_slot_fill_rate=total(cells, 'generated_words') / (queries * 3),
            generated_sha256=generation['generated_sha256'])
    if refinement:
        cells, queries = refinement['cells'], refinement['queries']
        result['refinement'] = dict(
            latency(refinement['timings'].get('refinement_with_ipc'), refinement['failure_statuses']),
            queries=queries, failed_requests=refinement['failures'],
            cold_startup_ms=refinement['cold_load_ms'],
            peak_process_tree_rss_bytes=refinement['process_tree_peak_rss_bytes'],
            peak_process_tree_threads=refinement['process_tree_peak_threads'],
            refined_top5_accuracy=total(cells, 'refined_top5') / queries,
            statistical_top5_accuracy=total(cells, 'baseline_top5') / queries,
            prediction_sha256=refinement['prediction_sha256'])
    return result


def verify_inputs(args):
    """Never benchmark a substitute model or database."""
    pinned = json.loads((ROOT / 'neural/model-bundle.json').read_bytes())['files']['model.gguf']['sha256']
    baseline = json.loads((ROOT / 'neural/fixtures/baseline-pin.json').read_bytes())['database']['sha256']
    actual = {'model.gguf': sha(args.bundle / 'model.gguf'), 'english.sqlite': sha(args.baseline)}
    if actual != {'model.gguf': pinned, 'english.sqlite': baseline}:
        raise ValueError('Model bundle or baseline differs from the pinned files')
    return actual


def background_cpu_percent():
    import psutil
    return psutil.cpu_percent(interval=2)


def run(args):
    inputs = verify_inputs(args)
    # Other work on the machine skews every timing; measure only when it is quiet.
    background = background_cpu_percent()
    if background > args.max_background_cpu:
        raise SystemExit(f'System CPU use is {background:.0f}% before measuring; close other work and retry')
    limits = {k: v for k, v in [('beam_width', args.beam_width),
                                ('max_forward_evaluations', args.max_forward_evaluations)] if v}
    experiment = experiment_limits(args.worker.resolve(), limits)
    saved = {name: os.environ.get(name) for name in EXPERIMENT.values()}
    reports = {}
    try:
        for key, value in limits.items():
            os.environ[EXPERIMENT[key]] = str(value)
        with tempfile.TemporaryDirectory() as temp:
            common = dict(cli=args.cli, worker=args.worker, accelerated_worker=args.accelerated_worker,
                          baseline=args.baseline, bundle=args.bundle, threads=args.threads, cpus=args.cpus)
            output = Path(temp) / 'generation.json'
            generation_evaluate.evaluate(SimpleNamespace(**common, output=output, samples=args.samples,
                                                         natural_context=args.natural_context, phrases=False))
            reports['generation'] = json.loads(output.read_bytes())
            if args.training:
                output = Path(temp) / 'refinement.json'
                neural_evaluate.evaluate(SimpleNamespace(**common, output=output, training=args.training))
                reports['refinement'] = json.loads(output.read_bytes())
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    binaries = {'cli': args.cli, 'worker': args.worker, 'accelerated_worker': args.accelerated_worker}
    report = {
        'schema': 1, 'label': args.label, 'hardware': hardware(),
        'configuration': {
            'threads': args.threads or 4, 'samples': reports['generation']['queries'],
            'natural_context': args.natural_context, 'cpu_restriction': args.cpus,
            'simulated': bool(args.cpus), 'experiment_limits': experiment,
            'production_policy': not experiment,
            'accelerated_used': reports['generation']['capabilities']['accelerated'],
            'deadline_ms': DEADLINE_MS, 'refinement_benchmark': bool(args.training),
            'system_cpu_percent_before': background, 'system_cpu_percent_after': background_cpu_percent()},
        'summary': summarize(reports['generation'], reports.get('refinement')),
        'binaries': {k: {'sha256': sha(p), 'arch': binary_arch(p.read_bytes()[:4096])}
                     for k, p in binaries.items() if p},
        'inputs': dict(inputs, **{f'harness/{p.name}': sha(p) for p in
                                  [Path(__file__).resolve(), Path(generation_evaluate.__file__).resolve(),
                                   Path(neural_evaluate.__file__).resolve()]}),
        'reports': reports,
        'interpretation': 'Frozen regression fixtures with unknown pretraining overlap. A simulated or '
                          'experiment configuration is not production policy or a low-end measurement.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(report['summary'], indent=2), flush=True)


def cpus(text):
    values = sorted({int(v) for v in text.split(',')})
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError('expected comma-separated CPU numbers')
    return values


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ['cli', 'worker', 'baseline', 'bundle', 'output']:
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--label', required=True, help='short machine and mode name, without personal data')
    result.add_argument('--accelerated-worker', type=Path)
    result.add_argument('--training', type=Path, help='prepared candidate.txt; enables the refinement benchmark')
    result.add_argument('--threads', type=int, choices=range(1, 5), help='worker threads; the CLI defaults to 4')
    result.add_argument('--samples', type=int, default=0, help='generation queries; 0 uses the full frozen workload')
    result.add_argument('--natural-context', action='store_true')
    result.add_argument('--cpus', type=cpus, help='simulation only: restrict the CLI and worker to these logical CPUs')
    result.add_argument('--max-background-cpu', type=float, default=25,
                        help='refuse to start when other work uses more system CPU than this')
    result.add_argument('--beam-width', type=int, choices=range(1, 9), help='experiment worker only')
    result.add_argument('--max-forward-evaluations', type=int, choices=range(1, 65), help='experiment worker only')
    return result


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists')
    if args.samples < 0:
        raise SystemExit('samples must be nonnegative')
    run(args)


if __name__ == '__main__':
    sys.exit(main())
