"""Run the bounded real Phase 2 validation sequentially and save evidence.

Uses real local embeddings/Qdrant; downloads only with explicit permission.
Audits the exact bounded count; does not delete pre-existing index contents.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--limit', type=int, default=1000)
    cli.add_argument('--queries', type=int, default=10)
    cli.add_argument('--top-k', type=int, default=5)
    cli.add_argument('--allow-download', action='store_true')
    cli.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/phase2-validation')
    args = cli.parse_args(argv)
    if min(args.limit, args.queries, args.top_k) < 1:
        cli.error('limit, queries and top-k must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {'status': 'running', 'steps': []}
    def save():
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    def run(name, command):
        print('Running ' + name, flush=True)
        with (output / (name + '.log')).open('w', encoding='utf8') as log:
            result = subprocess.run([sys.executable, *command], cwd=ROOT,
                env={**os.environ, 'PYTHONUTF8': '1'}, stdout=log, stderr=subprocess.STDOUT)
        summary['steps'].append({'name': name, 'returncode': result.returncode})
        save()
        if result.returncode:
            raise RuntimeError(name + ' failed; inspect ' + str(output / (name + '.log')))
    try:
        run('unit-tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        build = ['scripts/build_patient_index.py', '--limit', str(args.limit)]
        run('index-first', build + (['--allow-download'] if args.allow_download else []) +
            ['--report', str(output / 'index-first.json')])
        audit = ['scripts/audit_patient_index.py', '--expected-count', str(args.limit)]
        run('audit-first', audit + ['--report', str(output / 'audit-first.json')])
        run('retrieval', ['scripts/test_patient_rag.py', '--queries', str(args.queries),
            '--top-k', str(args.top_k), '--report', str(output / 'retrieval.json')])
        run('index-second', build + ['--report', str(output / 'index-second.json')])
        run('audit-second', audit + ['--report', str(output / 'audit-second.json')])
        first = json.loads((output / 'audit-first.json').read_text())
        second = json.loads((output / 'audit-second.json').read_text())
        if (first['indexed_count'] != second['indexed_count'] or
                first['payload_fingerprint'] != second['payload_fingerprint']):
            raise RuntimeError('Reindexing changed patient count or payloads')
        summary.update(status='passed', idempotency_verified=True,
            indexed_count=second['indexed_count'], validation_queries=args.queries, top_k=args.top_k)
        save()
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        save()
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
