"""Reconstruct pinned sources, chunk, embed with BGE, upsert medical-only Qdrant."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_ingestion.corpus import prepare_corpus, source_counts
from rag.medical_embeddings import MedicalEmbeddingModel
from rag.medical_vector_store import MedicalVectorStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA_ROOT / 'corpus.json')
    p.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    p.add_argument('--report', type=Path, default=DATA_ROOT.parents[1] / 'outputs/phase3/index.json')
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--max-words', type=int, default=180)
    p.add_argument('--overlap', type=int, default=30)
    args = p.parse_args()
    documents, chunks, serialized = prepare_corpus(args.manifest, args.max_words, args.overlap)
    embedding = MedicalEmbeddingModel(allow_download=args.allow_download)
    with MedicalVectorStore(args.storage_path, embedding.dimension, embedding_signature=embedding.signature) as store:
        expected = {c.chunk_id for c in chunks}
        existing = {str(point.id) for point in store.iter_points()}
        if existing - expected:
            raise ValueError('Corpus changed: use a new storage path to avoid stale chunks; no implicit deletion')
        for start in range(0, len(chunks), 16):
            batch = chunks[start:start + 16]
            store.upsert_chunks(batch, embedding.embed_texts([c.text for c in batch]))
        count = store.count()
        if count != len(chunks):
            raise ValueError('Stored count differs from expected corpus')
    processed = args.manifest.parent / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    (processed / 'chunks.jsonl').write_text(serialized, encoding='utf8')
    report = {'collection': 'medical_knowledge', 'embedding': embedding.signature,
              'sources': source_counts(documents, chunks), 'indexed_chunks': count,
              'processed_sha256': hashlib.sha256(serialized.encode()).hexdigest(),
              'truncated_texts': embedding.truncated_texts, 'max_words': args.max_words, 'overlap': args.overlap}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
