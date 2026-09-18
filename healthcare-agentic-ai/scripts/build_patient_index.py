"""Stream training patients into a persistent local semantic index."""
import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import PatientRAGConfig, inspect_dataset
from rag.patient_parser import DDXPlusParser
from rag.patient_rag import PatientCaseRAG


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--limit', type=int)
    cli.add_argument('--batch-size', type=int, default=16)
    cli.add_argument('--reset', action='store_true', help='Delete the selected collection before indexing')
    cli.add_argument('--allow-download', action='store_true', help='Explicitly allow download of selected embedding model')
    cli.add_argument('--storage-path', type=Path)
    cli.add_argument('--report', type=Path)
    args = cli.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    if args.batch_size < 1 or (args.limit is not None and args.limit < 0):
        cli.error('batch-size must be positive and limit nonnegative')
    start = time.perf_counter()
    try:
        inventory = inspect_dataset(args.data_dir)
        missing = [name for name, info in inventory['files'].items() if not info['exists']]
        if missing:
            raise ValueError('Missing DDXPlus files: ' + ', '.join(missing))
        config = replace(PatientRAGConfig(), batch_size=args.batch_size, allow_download=args.allow_download)
        if args.storage_path:
            config = replace(config, storage_path=args.storage_path)
        with PatientCaseRAG(config, parser=DDXPlusParser(args.data_dir)) as rag:
            report = rag.index(limit=args.limit, batch_size=args.batch_size, reset=args.reset)
            report.update(model=rag.embedding_model.model_name, device=rag.embedding_model.device,
                dimension=rag.embedding_model.dimension, collection=config.collection_name,
                storage_path=str(rag.vector_store.path), embedding_signature=rag.embedding_model.signature,
                distance='Cosine', mode='local persistent',
                total_seconds=time.perf_counter()-start)
            print(json.dumps(report, indent=2))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
        return 0
    except Exception as exc:
        logging.error('Indexing failed (completed batches, if any, remain; rerun safely): %s', exc)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
