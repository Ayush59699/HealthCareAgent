"""Read-only, label-free local top-5/10/20/50 probe. No LLM/API calls/downloads."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.focused_medical import state_from_patient
from rag.medical_ingestion.common import DATA_ROOT
from rag.patient_parser import DDXPlusParser
from rag.retrieval_experiment import DEPTHS, evaluate_depths


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=positive, default=2, help='One-based validation case (default 2)')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--storage-path', type=Path, default=DATA_ROOT / 'qdrant')
    parser.add_argument('--depths', type=positive, nargs='+', default=list(DEPTHS))
    parser.add_argument('--topic', action='append', default=[], help='Title substring to inspect; not a query or evaluation label')
    parser.add_argument('--compare-symptom-query', action='store_true', help='Also probe a symptom-only query; live query is unchanged')
    parser.add_argument('--report', type=Path, help='Fresh JSON report path; existing files are not overwritten')
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    report = args.report or Path(__file__).resolve().parents[1] / 'outputs/task6-continued' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-retrieval.json')
    if report.exists():
        raise ValueError('Report exists; choose a fresh path')
    manifest = args.storage_path / 'medical-embedding.json'
    # Fail before constructing a store (which otherwise creates a missing path).
    if not manifest.is_file():
        raise ValueError('Existing medical index and embedding manifest required; no index will be built')
    records = DDXPlusParser(args.data_dir).iter_patients('validate', limit=args.sample, include_labels=False)
    record = next((record for i, record in enumerate(records, 1) if i == args.sample), None)
    if record is None:
        raise ValueError('Validation sample not found')
    state = state_from_patient(record.patient, record.patient_id)
    # Heavy local dependencies load only after argument/index checks.
    from rag.medical_retriever import MedicalKnowledgeRetriever
    manifest_bytes = manifest.read_bytes()
    with MedicalKnowledgeRetriever(storage_path=args.storage_path, allow_download=False) as retriever:
        count = retriever.store.count()
        if not count:
            raise ValueError('Existing medical collection is empty')
        # Title availability is separate from retrieval ranking and does not
        # identify a correct diagnosis. It distinguishes corpus coverage gaps
        # from failures to rank an existing topic.
        catalog = Counter()
        if args.topic:
            from rag.medical_ingestion.models import validate_chunk
            for point in retriever.store.iter_points():
                title = validate_chunk(point.payload)['title']
                if any(topic.casefold() in title.casefold() for topic in args.topic):
                    catalog[title] += 1
        experiments = [evaluate_depths(state, retriever, depths=args.depths, topics=args.topic)]
        if args.compare_symptom_query:
            experiments.append(evaluate_depths(state, retriever, depths=args.depths,
                                               topics=args.topic, symptom_only=True))
        if retriever.store.count() != count or manifest.read_bytes() != manifest_bytes:
            raise ValueError('Index count or manifest changed during experiment')
        output = {'created_at': datetime.now(timezone.utc).isoformat(),
                  'medical_chunk_count': count, 'embedding_signature': retriever.embedding.signature,
                  'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                  'topic_catalog': [{'title': title, 'chunk_count': catalog[title]} for title in sorted(catalog)],
                  'labels_loaded': False, 'cloud_calls': 0, 'live_workflow_changed': False,
                  'experiments': experiments}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf8') as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    if args.topic:
        print('Matching indexed titles (not relevance judgments): ' + ', '.join(sorted(catalog)))
    for experiment in experiments:
        print(experiment['query_variant'] + ': ' + experiment['plan']['query'])
        for result in experiment['depths']:
            print(f"  top-{result['top_k']}: returned={result['returned_count']} retained={result['retained_count']}")
            for probe in result['topic_title_probes']:
                print(f"    {probe['topic']}: ranks={probe['ranks']}")
            print('    experimental lexical top-5: ' + ', '.join(row['hit']['title'] for row in result['experimental_top5']))
    print('Report: ' + str(report))
    print('Research heuristic only; title matches are not clinical relevance or diagnostic correctness.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
