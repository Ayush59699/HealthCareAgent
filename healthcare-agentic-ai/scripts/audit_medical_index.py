"""Audit every medical payload/vector against reconstructed, hash-pinned sources."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_ingestion.indexing import audit_index


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA_ROOT / 'corpus.json')
    p.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    p.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/medlineplus-expansion/audit.json')
    p.add_argument('--max-words', type=int, default=180)
    p.add_argument('--overlap', type=int, default=30)
    args = p.parse_args()
    report = audit_index(args.manifest, args.storage_path, max_words=args.max_words, overlap=args.overlap)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
