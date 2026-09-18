"""Audit real index, reject held-out insertion, and verify persistence on reopen."""
import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import PatientRAGConfig
from rag.embeddings import EmbeddingModel
from rag.patient_parser import DDXPlusParser
from rag.patient_ingestion import patient_document
from rag.patient_audit import audit_patient_index
from rag.vector_store import QdrantVectorStore


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--storage-path', type=Path)
    cli.add_argument('--expected-count', type=int)
    cli.add_argument('--report', type=Path)
    args = cli.parse_args(argv)
    try:
        config = PatientRAGConfig()
        if args.storage_path:
            config = replace(config, storage_path=args.storage_path)
        model = EmbeddingModel(config.model_name, config.device, config.batch_size)
        parser = DDXPlusParser(args.data_dir)
        def open_store():
            return QdrantVectorStore(config.storage_path, model.dimension,
                config.collection_name, embedding_signature=model.signature)
        with open_store() as store:
            report = audit_patient_index(store, parser)
            if args.expected_count is not None and report['indexed_count'] != args.expected_count:
                raise ValueError('Unexpected index count')
            rejected = {}
            for split in ('validate', 'test'):
                record = next(parser.iter_patients(split, limit=1, include_labels=False))
                try:
                    patient_document(record)
                except ValueError:
                    pass
                else:
                    raise ValueError('Ingestion accepted held-out patient')
                payload = {'patient_id': record.patient_id, 'split': record.split,
                           'source': record.source, 'text': record.patient.to_text()}
                try:
                    store.upsert_patients([payload], [[1.] + [0.] * (model.dimension - 1)])
                except ValueError:
                    rejected[split] = True
                else:
                    raise ValueError('Vector store accepted held-out patient')
            report['held_out_insertion_rejected'] = rejected
            if store.count() != report['indexed_count']:
                raise ValueError('Leakage probes changed the count')
        with open_store() as reopened:
            after = audit_patient_index(reopened, parser)
            for key in ('indexed_count', 'payload_fingerprint', 'vector_dimensions'):
                if after[key] != report[key]:
                    raise ValueError('Reopened index differs: ' + key)
            report['reopen_count'] = after['indexed_count']
            report['persistence_verified'] = True
        report['embedding_signature'] = model.signature
        print(json.dumps(report, indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
        return 0
    except Exception as exc:
        logging.error('Index audit failed: %s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
