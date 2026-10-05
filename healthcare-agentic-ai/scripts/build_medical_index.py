"""Reconstruct pinned sources and upsert medical-only Qdrant; reject stale chunks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_ingestion.indexing import build_index
from rag.medical_embeddings import MedicalEmbeddingModel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA_ROOT / 'corpus.json')
    p.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    p.add_argument('--report', type=Path, default=DATA_ROOT.parents[1] / 'outputs/phase3/index.json')
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--max-words', type=int, default=180)
    p.add_argument('--overlap', type=int, default=30)
    args = p.parse_args()
    embedding = MedicalEmbeddingModel(allow_download=args.allow_download)
    report = build_index(args.manifest, args.storage_path, embedding, max_words=args.max_words,
                         overlap=args.overlap, processed_dir=args.manifest.parent / 'processed')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
