#!/usr/bin/env python3
"""Evaluate held-out data, then build the full public baseline and artifact bundle."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
exe = ROOT / 'target' / 'release' / ('switchify-prediction.exe' if os.name == 'nt' else 'switchify-prediction')
out = ROOT / 'artifacts'
out.mkdir(exist_ok=True)
subprocess.run([sys.executable, 'scripts/fetch_corpus.py'], check=True)
hardware = f'{platform.platform()}; {platform.machine()}; {os.cpu_count()} logical CPUs'
if sys.platform == 'darwin':
    hardware += '; ' + subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
elif sys.platform.startswith('linux'):
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown CPU')
    hardware += '; ' + cpu
report = json.loads(subprocess.check_output([str(exe), 'evaluate', '--hardware', hardware]))
if os.name != 'nt':
    import resource
    # Child high-water RSS; evaluator is the substantial child at this point.
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    report['peak_process_rss_bytes'] = rss if sys.platform == 'darwin' else rss * 1024
    report['memory_measurement'] = 'Child-process high-water RSS including evaluation corpus, temporary build and predictor; not model-only memory.'
(out / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
output = out / 'english.sqlite'
# Only overwrite this script's generated baseline, never a caller-selected path.
if output.exists():
    output.unlink()
subprocess.run([str(exe), 'build', '--output', str(output)], check=True)
validation = subprocess.check_output([str(exe), 'validate', '--database', str(output)])
(out / 'validation.json').write_bytes(validation)
for name in ['source-manifest.json', 'ATTRIBUTION.md', 'LICENSE']:
    shutil.copyfile(ROOT / name, out / name)
(out / 'README.txt').write_text('english.sqlite uses the full pinned English corpus. evaluation.json evaluates a separate deduplicated 90/10 sentence split, not this full-corpus database. No personal writing is included.\n')
files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
(out / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in files))
print(json.dumps({'artifacts': str(out), 'warm_p95_ms': report['warm_p95_ms'], 'cold_load_ms': report['cold_load_ms']}, indent=2))
