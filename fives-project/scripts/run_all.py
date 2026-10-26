"""Run scripts 01-09 in order with the current interpreter; stop at the first failure.

Usage: python scripts/run_all.py [--start N]   (N = first script number to run, default 1)
Per-image feature caches and the saved ResNet embeddings are reused, so a re-run is much faster than the first run.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=1)
args = ap.parse_args()

scripts = sorted(p for p in HERE.glob("[0-9][0-9]_*.py") if args.start <= int(p.name[:2]) <= 9)
t0 = time.time()
for p in scripts:
    print(f"\n{'=' * 78}\n>>> {p.name}\n{'=' * 78}", flush=True)
    t = time.time()
    rc = subprocess.run([sys.executable, str(p)], cwd=HERE.parent).returncode
    print(f"<<< {p.name}: {'OK' if rc == 0 else f'FAILED (exit {rc})'} in {time.time() - t:.0f}s", flush=True)
    if rc != 0:
        sys.exit(f"Stopped: {p.name} failed.")
print(f"\nAll {len(scripts)} scripts finished in {(time.time() - t0) / 60:.1f} min")
