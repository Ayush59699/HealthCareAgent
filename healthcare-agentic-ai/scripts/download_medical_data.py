"""Download a bounded development corpus from official APIs; no page scraping."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_ingestion.medlineplus import download_topics, parse_medlineplus, DEFAULT_TOPICS
from rag.medical_ingestion.pmc import download_article, parse_pmc
from rag.medical_ingestion.who import download_document, parse_who


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_ROOT)
    parser.add_argument('--topics', nargs='+', default=list(DEFAULT_TOPICS))
    parser.add_argument('--pmcid', nargs='*', default=['PMC2440804', 'PMC5451008'])
    parser.add_argument('--who-manifest', type=Path, help='Explicit reviewed JSON list; no WHO downloads by default')
    args = parser.parse_args()
    if len(args.pmcid) > 10 or len(args.topics) > 20:
        parser.error('Development corpus limited to 10 PMC articles / 20 topics')
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries, failures = [], []
    try:
        path, provenance = download_topics(root / 'raw/medlineplus')
        docs = list(parse_medlineplus(path, provenance, args.topics))
        found = sorted(d.title for d in docs)
        if set(found) != set(args.topics):
            raise ValueError('Missing requested topics; found: ' + repr(found))
        entries.append({'source': 'medlineplus', 'raw_path': path.relative_to(root).as_posix(), 'sha256': provenance['raw_sha256'], 'titles': found})
    except Exception as exc:
        failures.append({'source': 'medlineplus', 'error': str(exc)})
    for pmcid in args.pmcid:
        try:
            time.sleep(0.4)
            path, provenance = download_article(pmcid, root / 'raw/pmc')
            document = parse_pmc(path, provenance)
            if document.pmcid != pmcid:
                raise ValueError('Requested and returned PMCID differ')
            entries.append({'source': 'pmc', 'raw_path': path.relative_to(root).as_posix(), 'sha256': provenance['raw_sha256'], 'pmcid': pmcid})
        except Exception as exc:
            failures.append({'source': 'pmc', 'pmcid': pmcid, 'error': str(exc)})
    selections = json.loads(args.who_manifest.read_text(encoding='utf8')) if args.who_manifest else []
    if len(selections) > 5:
        parser.error('Development corpus limited to five reviewed WHO documents')
    for selection in selections:
        try:
            path, provenance = download_document(selection, root / 'raw/who')
            parse_who(path, provenance, selection)
            entries.append({'source': 'who', 'raw_path': path.relative_to(root).as_posix(), 'sha256': provenance['raw_sha256'], 'selection': selection})
        except Exception as exc:
            failures.append({'source': 'who', 'error': str(exc)})
    report = {'schema_version': 1, 'entries': entries, 'failures': failures,
              'who_status': 'Explicit selections supplied' if selections else 'No reviewed WHO selection; zero WHO documents downloaded/indexed'}
    (root / 'corpus.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
