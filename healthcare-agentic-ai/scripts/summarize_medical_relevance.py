"""Summarize manual relevance judgments, separately from raw ranking and diagnosis.

Reads local benchmark JSON only. No retrieval, embeddings, labels or API calls.
"""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluation.reviewed_retrieval import Benchmark, summarize


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--input', type=Path, default=PROJECT / 'evaluation/medical_relevance.json')
    cli.add_argument('--output', type=Path, help='Optional fresh JSON output; otherwise stdout only')
    args = cli.parse_args(argv)
    if args.output is not None and args.output.exists():
        cli.error('Output already exists; choose a fresh path. Input and reviews will not be overwritten.')
    try:
        benchmark = Benchmark.model_validate_json(args.input.read_text(encoding='utf8'))
        report = summarize(benchmark)
    except (ValueError, OSError):
        cli.error('Invalid/unreadable benchmark. Check the documented review fields and immutable pool; no metrics calculated.')
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf8') as handle:
            handle.write(serialized + '\n')
    print(serialized)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
