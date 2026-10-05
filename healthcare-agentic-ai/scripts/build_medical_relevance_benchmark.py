"""Create an UNREVIEWED case-2 pool from saved retrieval and a copied local index.

No queries, embeddings, cloud calls, index writes, or relevance assignments.
Close other local Qdrant users before copying the existing index snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from evaluation.reviewed_retrieval import build_benchmark
from rag.focused_medical import state_from_patient
from rag.medical_ingestion.common import DATA_ROOT
from rag.patient_parser import DDXPlusParser


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--retrieval-report', type=Path, required=True,
                     help='Existing case-2 evaluate_medical_retrieval report containing both top-50 queries')
    cli.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    cli.add_argument('--data-dir', type=Path)
    cli.add_argument('--output', type=Path, default=PROJECT / 'evaluation/medical_relevance.json')
    args = cli.parse_args(argv)
    if args.output.exists():
        cli.error('Output already exists; choose a fresh path. Manual reviews must never be overwritten.')
    manifest_path = args.storage_path / 'medical-embedding.json'
    if not manifest_path.is_file():
        cli.error('Existing local medical index/manifest required; no index will be built.')
    raw_report = args.retrieval_report.read_bytes()
    report = json.loads(raw_report)
    manifest_bytes = manifest_path.read_bytes()
    signature = json.loads(manifest_bytes)
    if (signature != report['embedding_signature'] or
            hashlib.sha256(manifest_bytes).hexdigest() != report['manifest_sha256']):
        cli.error('Index manifest differs from the saved retrieval experiment.')
    records = DDXPlusParser(args.data_dir).iter_patients('validate', limit=2, include_labels=False)
    record = next((r for r in records if r.patient_id == 'ddxplus:validate:2'), None)
    if record is None:
        cli.error('Validation case 2 was not found.')
    state = state_from_patient(record.patient, record.patient_id)
    # Qdrant may manage local lock/metadata files on open, so never open production
    # storage at all. Read an isolated temporary copy, with no embedding model.
    from rag.medical_vector_store import MedicalVectorStore
    with tempfile.TemporaryDirectory(prefix='medical-review-snapshot-') as directory:
        copied = Path(directory) / 'qdrant'
        shutil.copytree(args.storage_path, copied)
        if (copied / 'medical-embedding.json').read_bytes() != manifest_bytes:
            cli.error('Index manifest changed during snapshot copy; close other index users and retry.')
        with MedicalVectorStore(copied, dimension=signature['dimension'], embedding_signature=signature) as store:
            if store.count() != report['medical_chunk_count']:
                cli.error('Index count differs from the saved retrieval experiment.')
            benchmark = build_benchmark(report, (p.payload for p in store.iter_points()), state,
                                        report_name=args.retrieval_report.name,
                                        report_sha256=hashlib.sha256(raw_report).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as handle:
        handle.write(benchmark.model_dump_json(indent=2) + '\n')
    print(f'Created {args.output}: {len(benchmark.candidates)} unique chunks; all uncertain and UNREVIEWED.')
    for title, ids in benchmark.requested_topics.items():
        print(f'Requested indexed topic {title}: {len(ids)} chunks (not a relevance judgment)')
    print('No relevance metrics or clinical/diagnostic conclusions have been calculated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
