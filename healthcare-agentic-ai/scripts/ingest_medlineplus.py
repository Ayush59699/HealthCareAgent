"""Download pinned MedlinePlus XML and safely replace the medical-only index.

Uses cached BGE by default. Existing authorized PMC selections are preserved.
Stop demo/retrieval/index processes before running (local Qdrant has exclusive locks).
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT
from rag.medical_ingestion.medlineplus import download_topics, parse_medlineplus, SOURCE_VERSION, XML_URL
from rag.medical_ingestion.pmc import download_article
from rag.medical_ingestion.corpus import load_documents
from rag.medical_ingestion.indexing import build_index, audit_index
from rag.medical_embeddings import MedicalEmbeddingModel
from rag.medical_vector_store import MedicalVectorStore, COLLECTION


def create_manifest(root, previous):
    path, provenance = download_topics(root / 'raw/medlineplus')
    stats = {}
    list(parse_medlineplus(path, provenance, stats=stats))
    entries = [{'source': 'medlineplus', 'raw_path': path.relative_to(root).as_posix(),
                'sha256': provenance['raw_sha256'], 'source_version': SOURCE_VERSION,
                'selection': 'all-english-summaries', 'parse_stats': stats}]
    if previous is not None:
        if previous.get('failures'):
            raise ValueError('Existing corpus has unresolved ingestion failures')
        if any(e['source'] not in {'medlineplus', 'pmc'} for e in previous['entries']):
            raise ValueError('Existing non-PMC sources require explicit review; refusing to remove them silently')
        entries.extend(e for e in previous['entries'] if e['source'] == 'pmc')
    else:
        # Same two explicitly authorized development articles, not a new PMC selection.
        for pmcid in ('PMC2440804', 'PMC5451008'):
            article, meta = download_article(pmcid, root / 'raw/pmc')
            entries.append({'source': 'pmc', 'raw_path': article.relative_to(root).as_posix(),
                            'sha256': meta['raw_sha256'], 'pmcid': pmcid})
    return {'schema_version': 2, 'entries': entries, 'failures': []}, provenance


def publish(root, stage, pending, storage, signature):
    """Rollback-capable directory replacement; never delete or rename patient stores."""
    if storage.exists():
        with MedicalVectorStore(storage, signature['dimension'], embedding_signature=signature) as store:
            collections = {c.name for c in store.client.get_collections().collections}
            if collections - {COLLECTION}:
                raise ValueError('Refusing directory replacement: storage contains a non-medical collection')
    backup = root / ('qdrant-backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir()
    moves = []
    try:
        for target in (storage, root / 'corpus.json', root / 'processed'):
            if target.exists():
                saved = backup / target.name
                target.rename(saved)
                moves.append((saved, target))
        published = []
        for source, target in ((stage / 'qdrant', storage), (stage / 'processed', root / 'processed'), (pending, root / 'corpus.json')):
            source.rename(target)
            published.append((target, source))
    except Exception:
        for target, source in reversed(locals().get('published', [])):
            target.rename(source)
        for saved, target in reversed(moves):
            saved.rename(target)
        raise
    return backup


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-dir', type=Path, default=DATA_ROOT)
    cli.add_argument('--allow-download', action='store_true', help='Only needed if the existing BGE model is not cached')
    cli.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/medlineplus-expansion/ingestion.json')
    args = cli.parse_args()
    root = args.data_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / '.medlineplus-ingest.lock'
    # Exclusive lock also prevents competing invocations from overwriting the candidate manifest.
    with lock.open('x', encoding='utf8') as handle:
        handle.write(datetime.now(timezone.utc).isoformat())
    stage = root / ('.medlineplus-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    pending = root / '.medlineplus-corpus.pending.json'
    report = {'passed': False, 'source_url': XML_URL}
    try:
        previous_path = root / 'corpus.json'
        previous = json.loads(previous_path.read_text(encoding='utf8')) if previous_path.exists() else None
        manifest, provenance = create_manifest(root, previous)
        pending.write_text(json.dumps(manifest, indent=2), encoding='utf8')
        load_documents(pending)  # Verify all preserved PMC raw bytes/licenses before model loading.
        embedding = MedicalEmbeddingModel(allow_download=args.allow_download)
        stage.mkdir()
        build = build_index(pending, stage / 'qdrant', embedding, processed_dir=stage / 'processed')
        audit = audit_index(pending, stage / 'qdrant')
        report.update(download=provenance, parse_stats=manifest['entries'][0]['parse_stats'], build=build, audit=audit)
        if not audit['passed']:
            raise ValueError('Staged medical audit failed; existing index is untouched')
        backup = publish(root, stage, pending, root / 'qdrant', embedding.signature)
        report.update(passed=True, backup_path=str(backup), storage_path=str(root / 'qdrant'))
    except Exception as exc:
        report['error'] = str(exc)
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding='utf8')
        lock.unlink(missing_ok=True)
        pending.unlink(missing_ok=True)
        if report['passed']:
            shutil.rmtree(stage, ignore_errors=True)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
