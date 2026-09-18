"""Reproducible retrieval smoke checks, NOT clinical/relevance performance metrics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_retriever import MedicalKnowledgeRetriever

QUERIES = ['What is asthma and what causes wheezing?', 'What is diabetes and blood glucose?',
           'What is high blood pressure?']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--report', type=Path, default=DATA_ROOT.parents[1] / 'outputs/phase3/retrieval.json')
    args = p.parse_args()
    if args.top_k < 1:
        p.error('--top-k must be positive for validation')
    results = []
    with MedicalKnowledgeRetriever(storage_path=args.storage_path) as retriever:
        for query in QUERIES:
            hits = retriever.retrieve(query, args.top_k)
            if not hits or not all(h['text'] and h['provenance'] and h['license'] for h in hits):
                raise ValueError('Retrieval must return actual text and attribution')
            results.append({'query': query, 'hits': hits})
    report = {'passed': True, 'queries': len(results), 'top_k': args.top_k,
              'interpretation': 'Retrieval plumbing smoke test only; similarity is not medical accuracy or diagnostic confidence.',
              'results': results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({'passed': True, 'queries': len(results), 'top_k': args.top_k,
                      'top_titles': [r['hits'][0]['title'] for r in results]}, indent=2))


if __name__ == '__main__':
    main()
