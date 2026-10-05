"""Offline new3 comparison; existing cached model and disposable byte-copy of store.

No benchmark labels, corpus re-embedding, production tuning or cloud calls.
Rows validate:2..6 are fixed before inspecting results, not chosen by diagnosis.
Run from project root: python scripts/evaluate_amg_query_adapter.py
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/new3'
sys.path.insert(0, str(ROOT))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inventory():
    files = set()
    for folder in ('application', 'orchestration', 'safety', 'rag', 'vendor/amg',
                   'data', 'evaluation', 'outputs/amg-live', 'outputs/new2'):
        files.update(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    files.update(p for p in (ROOT.parent / 'data').rglob('*') if p.is_file())
    # Credentials checked separately in memory; do not persist secret hashes.
    return {str(p.relative_to(ROOT.parent)): digest(p) for p in sorted(files) if p.name != '.env'}


def worker(copy_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from rag.amg.backend import load_upstream, AMGMedicalEvidence
    from rag.amg.alias_safety import CaseSafeAliases
    from rag.amg.query_adapter import build_query, patient_facts
    from rag.agents.models import PatientState
    from rag.agents.grounding import state_from_patient
    from rag.patient_parser import DDXPlusParser

    torch.set_num_threads(2)
    up = load_upstream()
    run = json.loads((OUT / 'sample_case_001.json').read_text(encoding='utf8'))
    snapshot = up.STORE / run['medical_retrieval']['upstream']['snapshot']
    manifest = json.loads((snapshot / 'manifest.json').read_text(encoding='utf8'))
    source_path = up.ROOT / 'knowledge/medlineplus' / manifest['config']['source_snapshot']
    source = json.loads((source_path / 'manifest.json').read_text(encoding='utf8'))
    assert digest(snapshot / 'chunks.jsonl') == manifest['chunks_sha256']
    assert digest(Path(up.__file__)) == manifest['config']['implementation_sha256']
    for name, sha in manifest['config']['artifacts'].items():
        assert digest(source_path / name) == sha
    assert manifest['status'] == source['status'] == 'complete'
    topics = [json.loads(s) for s in (source_path / 'topics.jsonl').read_text(encoding='utf8').splitlines()]
    chunks = {c['id']: c for c in map(json.loads, (snapshot / 'chunks.jsonl').read_text(encoding='utf8').splitlines())}
    collection = up.client_for(copy_path).get_collection(up.COLLECTION, embedding_function=None)
    assert collection.count() == manifest['chunks'] == len(chunks)
    assert collection.metadata['hnsw:space'] == 'l2'
    rows = []
    for offset in range(0, collection.count(), 256):
        data = collection.get(limit=256, offset=offset, include=['embeddings', 'documents', 'metadatas'])
        for id_, vector, text, metadata in zip(data['ids'], data['embeddings'], data['documents'], data['metadatas']):
            assert chunks[id_] == {'id': id_, 'text': text, 'metadata': metadata}
            rows.append((id_, vector, metadata))
    rows.sort(key=lambda row: row[0])
    vectors = np.asarray([r[1] for r in rows], dtype=np.float64)
    assert vectors.shape == (2289, 384) and np.isfinite(vectors).all()
    embedder = up.offline_embedder(source)
    assert embedder.identity == manifest['config']['embedding']
    raw = up.Retriever(collection, embedder, topics)
    guarded = up.Retriever(collection, embedder, topics)
    guarded.aliases = CaseSafeAliases(guarded.aliases)
    backend = AMGMedicalEvidence.__new__(AMGMedicalEvidence)
    backend.retriever = SimpleNamespace(embedder=embedder)
    tokens = lambda s: len(embedder.tokenizer.encode(s, add_special_tokens=True, verbose=False))
    # Requested report topics only; never passed to adapter, retrieval or patient parsing.
    target_titles = {'Anemia', 'Blood Thinners', 'Gastrointestinal Bleeding', 'Kidney Failure'}

    def measure(query, retriever, targets=False):
        result = retriever.search(query, top_k=5, max_distance=1.10)
        embedded = result['resolution']['embedding_query']
        vector = np.asarray(embedder.embed_documents([embedded])[0], dtype=np.float64)
        distances = np.einsum('ij,ij->i', vectors-vector, vectors-vector)
        order = sorted(range(len(rows)), key=lambda i: (distances[i], rows[i][0]))
        ranks = {i: rank for rank, i in enumerate(order, 1)}
        def item(i):
            return dict(id=rows[i][0], title=rows[i][2]['title'], rank=ranks[i],
                        distance=float(distances[i]), within_gate=bool(distances[i] <= 1.10))
        return dict(query_tokens=tokens(query), embedded_tokens=tokens(embedded), upstream=result,
                    exact_top_five=[item(i) for i in order[:5]],
                    requested_topic_chunks=[item(i) for i in order if targets and rows[i][2]['title'] in target_titles])

    states = [PatientState.model_validate(run['patient_state'])]
    for record in DDXPlusParser().iter_patients('validate', limit=6, include_labels=False):
        assert record.labels is None
        if record.patient_id != states[0].patient_id:
            states.append(state_from_patient(record.patient, record.patient_id))
    cases = []
    for state in states:
        before = state.model_dump_json()
        literal, omitted = backend.query_for(state)
        adapted = build_query(state, embedder.tokenizer, embedder.max_tokens)
        baseline = measure(literal, raw, targets=len(cases) == 0)
        safe_baseline = measure(literal, guarded, targets=len(cases) == 0)
        candidate = measure(adapted.query, guarded, targets=len(cases) == 0)
        assert state.model_dump_json() == before
        if not cases:
            assert literal == run['medical_retrieval']['query']
            saved = run['medical_retrieval']['upstream']
            assert baseline['upstream']['resolution'] == saved['resolution']
            assert [h['id'] for h in baseline['upstream']['results']] == [h['id'] for h in saved['results']]
            delta = max(abs(a['distance'] - b['distance']) for a, b in zip(baseline['upstream']['results'], saved['results']))
            assert delta < 1e-4
        facts = patient_facts(state)
        cases.append(dict(patient_id=state.patient_id, patient_state=state.model_dump(),
                          baseline=baseline, guarded_literal=safe_baseline, adapted=candidate,
                          baseline_omitted_count=omitted,
                          baseline_included=[f.text for f in facts if f.text in literal.split('; ')],
                          baseline_omitted=[f.text for f in facts if f.text not in literal.split('; ')],
                          representation=adapted.audit()))
        print(state.patient_id, tokens(literal), '->', adapted.token_count, 'omitted', omitted, '->',
              len(adapted.omitted), 'top5', [h['title'] for h in candidate['upstream']['results']], flush=True)
    probes = ['stuck in your bed all day long', 'All symptoms started yesterday', 'all',
              'What is all?', 'ALL', 'What is ALL?', 'all day with ALL', 'allergy']
    alias_checks = {q: dict(before=raw.aliases.resolve(q), after=guarded.aliases.resolve(q)) for q in probes}
    report = dict(scope='Offline retrieval only; no diagnostic/clinical outcome measurement',
                  rows_selected_before_results='Saved validate:1 plus consecutive validate:2..6, labels disabled',
                  model=embedder.identity, snapshot=run['medical_retrieval']['upstream']['snapshot'],
                  top_k=5, max_distance=1.10, metric='squared_l2', stored_chunks=len(rows),
                  saved_replay_max_distance_delta=delta, cases=cases, alias_checks=alias_checks)
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf8')


def main():
    import psutil
    if psutil.virtual_memory().available < 2_000_000_000:
        raise RuntimeError('Need 2 GB available memory for one cached CPU model')
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_files = ('outputs/amg-live/sample_case_001.json', 'outputs/new2/results.json',
                      'outputs/new2/query-as-embedded.txt')
    # Never overwrite a previously preserved baseline.
    for name in baseline_files:
        target = OUT / Path(name).name
        if not target.exists():
            shutil.copy2(ROOT / name, target)
        assert digest(ROOT / name) == digest(target)
    before = inventory()
    (OUT / 'protected-before.json').write_text(json.dumps(before, indent=2)+'\n', encoding='utf8')
    secrets = [p for p in (ROOT.parent / '.env', ROOT / 'vendor/amg/.env') if p.exists()]
    secret_before = {p: digest(p) for p in secrets}
    run = json.loads((OUT / 'sample_case_001.json').read_text(encoding='utf8'))
    source_db = ROOT / 'vendor/amg/knowledge/medlineplus_lab' / run['medical_retrieval']['upstream']['snapshot'] / 'chroma'
    temporary = Path(tempfile.mkdtemp(prefix='amg-new3-readonly-'))
    status = None
    try:
        copy_path = temporary / 'chroma'
        shutil.copytree(source_db, copy_path)
        assert {str(p.relative_to(source_db)): digest(p) for p in source_db.rglob('*') if p.is_file()} == {
            str(p.relative_to(copy_path)): digest(p) for p in copy_path.rglob('*') if p.is_file()}
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'ANONYMIZED_TELEMETRY': 'False', 'TOKENIZERS_PARALLELISM': 'false', 'PYTHONIOENCODING': 'utf-8'}
        status = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(copy_path)], env=env).returncode
    finally:
        shutil.rmtree(temporary)
        after = inventory()
        changes = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
        integrity = dict(protected_file_count=len(before), unchanged=not changes, changed_paths=changes,
                         credentials_unchanged=secret_before == {p: digest(p) for p in secrets},
                         temporary_database_removed=not temporary.exists(), worker_exit_code=status)
        (OUT / 'integrity.json').write_text(json.dumps(integrity, indent=2)+'\n', encoding='utf8')
        assert not changes and integrity['credentials_unchanged']
    if status:
        raise SystemExit(status)


if __name__ == '__main__':
    worker(Path(sys.argv[2])) if len(sys.argv) > 1 and sys.argv[1] == '--worker' else main()
