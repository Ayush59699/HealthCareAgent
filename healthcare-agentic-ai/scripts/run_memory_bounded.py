"""Run one local experiment with a conservative machine-wide memory watchdog.

Example: python scripts/run_memory_bounded.py -- scripts/inspect_hybrid_medical_retrieval.py
Uses decimal GB. Stops the child tree at 14 GB (1 GB below the user's 15 GB
ceiling). Polling is a safety margin, not an OS-wide allocation reservation;
other applications must not start memory-heavy work during the experiment.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def main(argv=None):
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stop-gb', type=float, default=14.0)
    parser.add_argument('--reserve-gb', type=float, default=1.5)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.stop_gb <= 14 or args.reserve_gb < 1:
        parser.error('Command, stop <=14 GB and startup reserve >=1 GB required')
    used = psutil.virtual_memory().used / 1e9
    if used + args.reserve_gb >= args.stop_gb:
        raise SystemExit(f'Insufficient safe headroom: {used:.2f} GB used; close applications first')
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
    child = subprocess.Popen([sys.executable, '-u', *command], env=env,
                             cwd=Path(__file__).resolve().parents[1])
    peak, tree_peak, stopped = used, 0., False
    try:
        while child.poll() is None:
            used = psutil.virtual_memory().used / 1e9
            peak = max(peak, used)
            try:
                parent = psutil.Process(child.pid)
                processes = [parent, *parent.children(recursive=True)]
                tree_peak = max(tree_peak, sum(p.memory_info().rss for p in processes if p.is_running()) / 1e9)
            except psutil.NoSuchProcess:
                processes = []
            if used >= args.stop_gb:
                stopped = True
                print(f'MEMORY STOP: system {used:.2f} GB', flush=True)
                for process in reversed(processes):
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                child.kill()
                break
            time.sleep(.05)
        child.wait()
    finally:
        if child.poll() is None:
            parent = psutil.Process(child.pid)
            for process in reversed([parent, *parent.children(recursive=True)]):
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
        print(f'Memory sampled peak: system={peak:.3f} GB; child-tree RSS={tree_peak:.3f} GB', flush=True)
    return 75 if stopped else child.returncode


if __name__ == '__main__':
    raise SystemExit(main())
