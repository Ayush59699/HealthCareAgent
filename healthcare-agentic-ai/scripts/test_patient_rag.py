"""Query validation features and audit every stored payload. Not diagnostic evaluation."""
import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import PatientRAGConfig
from rag.patient_parser import DDXPlusParser
from rag.patient_rag import PatientCaseRAG
from rag.patient_audit import audit_patient_index


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--queries', type=int, default=10)
    cli.add_argument('--top-k', type=int, default=5)
    cli.add_argument('--storage-path', type=Path)
    cli.add_argument('--report', type=Path)
    args = cli.parse_args(argv)
    if args.queries < 1 or args.top_k < 1:
        cli.error('queries and top-k must be positive')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    start = time.perf_counter()
    try:
        config = PatientRAGConfig()
        if args.storage_path:
            config = replace(config, storage_path=args.storage_path)
        parser = DDXPlusParser(args.data_dir)
        with PatientCaseRAG(config) as rag:
            count = rag.vector_store.count()
            if not count:
                raise ValueError('Index is empty. Run scripts/build_patient_index.py --limit 1000 first.')
            audit = audit_patient_index(rag.vector_store, parser)
            results = []
            for record in parser.iter_patients('validate', limit=args.queries, include_labels=False):
                query = record.patient.to_text()
                query_start = time.perf_counter()
                hits = rag.retrieve(query, top_k=args.top_k)
                seconds = time.perf_counter() - query_start
                if any(h['split'] != 'train' for h in hits):
                    raise ValueError('Split leakage detected')
                print('\nQUERY PATIENT ' + record.patient_id + '\n' + query)
                print(f'\nTOP {args.top_k} SIMILAR TRAINING CASES (cosine retrieval similarity scores, not probabilities)')
                for rank, hit in enumerate(hits, 1):
                    print(f"\n{rank}. patient_id: {hit['patient_id']}\nsplit: {hit['split']}\ncosine retrieval similarity score: {hit['score']:.8f}\nsource: {hit['source']}\ncase:\n{hit['text']}")
                results.append({'query_patient_id': record.patient_id, 'query_text': query,
                                'seconds': seconds, 'hits': hits})
            if len(results) != args.queries or any(len(r['hits']) != min(args.top_k, count) for r in results):
                raise ValueError('Unexpected query or retrieval count')
            report = {**audit, 'top_k': args.top_k, 'query_truncated_texts': rag.embedding_model.truncated_texts,
                'validation_queries': len(results), 'test_records_indexed': 0,
                'validation_records_indexed': 0, 'payload_fields': ['patient_id','split','text','source'],
                'model': rag.embedding_model.model_name, 'device': rag.embedding_model.device,
                'dimension': rag.embedding_model.dimension, 'total_seconds': time.perf_counter()-start,
                'retrieval_seconds': sum(r['seconds'] for r in results), 'results': results}
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
            print('\nAUDIT SUMMARY\n' + json.dumps({k:v for k,v in report.items() if k != 'results'}, indent=2))
        return 0
    except Exception as exc:
        logging.error('Retrieval validation failed: %s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
