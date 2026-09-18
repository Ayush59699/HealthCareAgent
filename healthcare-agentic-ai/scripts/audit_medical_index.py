"""Audit every medical payload/vector against reconstructed, hash-pinned sources."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_ingestion.corpus import prepare_corpus, source_counts
from rag.medical_ingestion.models import validate_chunk
from rag.medical_embeddings import MedicalEmbeddingModel
from rag.medical_vector_store import MedicalVectorStore


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=DATA_ROOT / 'corpus.json')
    p.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    p.add_argument('--report', type=Path, default=DATA_ROOT.parents[1] / 'outputs/phase3/audit.json')
    p.add_argument('--max-words', type=int, default=180)
    p.add_argument('--overlap', type=int, default=30)
    args = p.parse_args()
    documents, chunks, _ = prepare_corpus(args.manifest, args.max_words, args.overlap)
    expected = {c.chunk_id: asdict(c) for c in chunks}
    embedding = MedicalEmbeddingModel()
    seen = set()
    with MedicalVectorStore(args.storage_path, embedding.dimension, embedding_signature=embedding.signature) as store:
        for point in store.iter_points(with_vectors=True):
            payload = validate_chunk(point.payload)
            if str(point.id) != payload['chunk_id'] or payload != expected.get(str(point.id)):
                raise ValueError('Stored payload differs from reconstructed source')
            vector = store._vector(point.vector)
            if abs(sum(v*v for v in vector) - 1) > 1e-4:
                raise ValueError('Vector is not normalized')
            seen.add(str(point.id))
    if seen != set(expected):
        raise ValueError('Stored IDs differ from corpus')
    report = {'passed': True, 'collection': 'medical_knowledge', 'audited_chunks': len(seen),
              'sources': source_counts(documents, chunks), 'checks': ['raw checksums', 'source reconstruction',
              'license allowlist', 'strict metadata schema', 'deterministic IDs', 'vector dimension/normalization', 'persistence reopening']}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
