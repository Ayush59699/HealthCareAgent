"""Topic-relevance smoke checks, not diagnostic accuracy or clinical validation."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_retriever import MedicalKnowledgeRetriever

# Predeclared topic expectations, not diagnosis labels or patient-case inputs.
TOPIC_CASES = [
    ('asthma', 'What is asthma and what causes wheezing?', {'Asthma', 'Asthma in Children'}),
    ('diabetes', 'What is diabetes and high blood glucose?', {'Diabetes', 'Diabetes Type 1', 'Diabetes Type 2'}),
    ('hypertension', 'What is hypertension or high blood pressure?', {'High Blood Pressure'}),
    ('anxiety', 'What are symptoms of anxiety?', {'Anxiety', 'Panic Disorder'}),
    ('depression', 'What are symptoms of depression?', {'Depression', 'Teen Depression'}),
    ('chest pain', 'What can cause chest pain?', {'Chest Pain', 'Angina'}),
    ('shortness of breath', 'What causes shortness of breath or difficulty breathing?', {'Breathing Problems', 'Asthma', 'Asthma in Children'}),
    ('palpitations', 'What causes palpitations, a racing or irregular heartbeat?', {'Arrhythmia', 'Atrial Fibrillation'}),
    ('migraine', 'What is migraine and what are its symptoms?', {'Migraine'}),
    ('pneumonia', 'What is pneumonia and what are its symptoms?', {'Pneumonia'}),
]
QUERIES = [case[1] for case in TOPIC_CASES]


def relevance_basis(hit, expected_titles):
    if hit['title'] in expected_titles:
        return 'document_title'
    # Retrieval units are source passages. A direct definition under the original
    # source heading can answer a topic query even on a related parent page.
    definitions = {f'what {verb} {title.casefold()}?' for title in expected_titles for verb in ('is', 'are')}
    if hit.get('section', '').casefold() in definitions:
        return 'source_definition_heading'
    return None


def check_hits(hits, expected_titles):
    if not hits or not all(h.get('text') and h.get('provenance') and h.get('license') for h in hits):
        return False
    return any(relevance_basis(h, expected_titles) for h in hits)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/medlineplus-expansion/retrieval.json')
    args = p.parse_args()
    if args.top_k < 1:
        p.error('--top-k must be positive for validation')
    results = []
    with MedicalKnowledgeRetriever(storage_path=args.storage_path) as retriever:
        for topic, query, expected_titles in TOPIC_CASES:
            hits = retriever.retrieve(query, args.top_k)
            results.append({'topic': topic, 'query': query, 'expected_titles': sorted(expected_titles),
                            'passed': check_hits(hits, expected_titles),
                            'parent_title_match': any(h['title'] in expected_titles for h in hits),
                            'relevance_basis': [relevance_basis(h, expected_titles) for h in hits], 'hits': hits})
    passed = all(r['passed'] for r in results)
    report = {'passed': passed, 'queries': len(results), 'top_k': args.top_k,
              'parent_title_matches': sum(r['parent_title_match'] for r in results),
              'interpretation': 'Cosine retrieval similarity, NOT disease probability. Topic smoke checks are not clinical validation or a relevance benchmark.',
              'results': results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps({'passed': passed, 'queries': len(results), 'top_k': args.top_k,
                      'results': [{'topic': r['topic'], 'passed': r['passed'], 'top_title': r['hits'][0]['title'] if r['hits'] else None, 'parent_title_match': r['parent_title_match']}
                                  for r in results]}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
