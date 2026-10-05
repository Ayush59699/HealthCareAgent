"""Label-free local retrieval probe; never loads evaluation labels."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.patient_parser import DDXPlusParser
from rag.medical_retriever import MedicalKnowledgeRetriever
r = list(DDXPlusParser().iter_patients('validate', limit=3, include_labels=False))
with MedicalKnowledgeRetriever() as medical:
    results = []
    for record in r:
        hits = medical.retrieve(record.patient.to_text(), top_k=10)
        results.append({'patient_id': record.patient_id, 'query': record.patient.to_text(), 'hits': hits})
    out = Path(__file__).resolve().parents[1] / 'outputs/enhance'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'before-retrieval.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    for result in results:
        print(result['patient_id'])
        for hit in result['hits']:
            print(round(hit['score'], 4), hit['title'])
