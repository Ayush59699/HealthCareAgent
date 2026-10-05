"""One fixed new4 compression candidate on saved validate:1 and validate:2..6.

Offline cached CPU embedding; Chroma opens only a disposable byte-copy. Prior
artifacts and production are protected. No labels, model tuning or promotion.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/new4'
sys.path.insert(0, str(ROOT))
from scripts.evaluate_amg_query_adapter import digest, inventory


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf8')


def protected_inventory():
    result = inventory()
    for folder in ('scripts', 'tests', 'docs', 'outputs/new3'):
        for path in (ROOT / folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.name != '.env':
                result[str(path.relative_to(ROOT.parent))] = digest(path)
    # This task explicitly adds tests to this existing module. All other existing
    # code and previous experiments are protected from edits.
    result.pop(str((ROOT / 'tests/test_amg_query_adapter.py').relative_to(ROOT.parent)), None)
    return result


def check_protected(output_name):
    before = json.loads((OUT / 'protected-before.json').read_text(encoding='utf8'))
    changed = [name for name, sha in before.items()
               if not (ROOT.parent / name).is_file() or digest(ROOT.parent / name) != sha]
    result = dict(protected_file_count=len(before), unchanged=not changed, changed_paths=changed)
    write_json(OUT / output_name, result)
    if changed:
        raise RuntimeError('Protected files changed; see integrity report')
    return result


def alias_diagnostics(query, resolver, search_result):
    """Observe existing alias behavior only; never modify resolution or search.

    Structural/ordinary-word flags are audit watch terms, not extra alias rules.
    All matches (including unflagged aliases) and their source titles are recorded.
    """
    from rag.amg.alias_safety import normalize
    proposed = resolver.resolve(query)
    actual = search_result['resolution']
    prefix = query + '\nSource topic: '
    def additions(text):
        return text[len(prefix):].split('; ') if text.startswith(prefix) else []
    applied = additions(actual['expanded_query'])
    watched = {'all', 'sex', 'age', 'presenting', 'symptoms', 'history', 'medications', 'context'}
    matches = []
    for match in proposed['matches']:
        titles = [resolver.titles[t] for t in match['topic_ids']]
        expands = len(titles) == 1 and normalize(titles[0]) != match['alias']
        matches.append({**match, 'titles': titles, 'proposes_source_title': expands,
                        'expansion_applied': expands and titles[0] in applied,
                        'ordinary_or_structural_watch_term': match['alias'] in watched})
    return dict(matches=matches, proposed_source_titles=additions(proposed['expanded_query']),
                applied_source_titles=applied, expansion_skipped=actual.get('expansion_skipped'),
                suppressed_ordinary_all_matches=proposed.get('suppressed_ordinary_all_matches', 0),
                watched_matches=[m for m in matches if m['ordinary_or_structural_watch_term']],
                query_preserved_as_expansion_prefix=actual['expanded_query'].startswith(query),
                embedding_query=actual['embedding_query'])


def worker(copy_path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from rag.amg.backend import load_upstream, AMGMedicalEvidence
    from rag.amg.alias_safety import CaseSafeAliases
    from rag.amg.baseline_compression import compress_baseline
    from rag.agents.models import PatientState
    from rag.agents.grounding import state_from_patient
    from rag.patient_parser import DDXPlusParser

    torch.set_num_threads(2)
    up = load_upstream()
    run = json.loads((ROOT / 'outputs/amg-live/sample_case_001.json').read_text(encoding='utf8'))
    prior = json.loads((ROOT / 'outputs/new3/comparison.json').read_text(encoding='utf8'))
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
    guarded.aliases = CaseSafeAliases(guarded.aliases)  # existing isolated guard, unchanged
    backend = AMGMedicalEvidence.__new__(AMGMedicalEvidence)
    backend.retriever = SimpleNamespace(embedder=embedder)
    tokens = lambda s: len(embedder.tokenizer.encode(s, add_special_tokens=True, verbose=False))
    # Reporting filter only: never supplied to compression or search.
    target_titles = {'Anemia', 'Blood Thinners', 'Gastrointestinal Bleeding', 'Kidney Failure'}

    def measure(query, retriever):
        result = retriever.search(query, top_k=5, max_distance=1.10)
        embedded = result['resolution']['embedding_query']
        vector = np.asarray(embedder.embed_documents([embedded])[0], dtype=np.float64)
        distances = np.einsum('ij,ij->i', vectors-vector, vectors-vector)
        order = sorted(range(len(rows)), key=lambda i: (distances[i], rows[i][0]))
        ranks = {i: rank for rank, i in enumerate(order, 1)}
        def item(i):
            return dict(id=rows[i][0], title=rows[i][2]['title'], rank=ranks[i],
                        distance=float(distances[i]), within_gate=bool(distances[i] <= 1.10))
        return dict(query=query, query_tokens=tokens(query), embedded_tokens=tokens(embedded),
                    character_count=len(query), embedded_character_count=len(embedded),
                    accepted_hits=len(result['results']), upstream=result,
                    aliases=alias_diagnostics(query, retriever.aliases, result),
                    exact_top_five=[item(i) for i in order[:5]],
                    requested_topic_chunks=[item(i) for i in order if rows[i][2]['title'] in target_titles])

    states = [PatientState.model_validate(run['patient_state'])]
    for record in DDXPlusParser().iter_patients('validate', limit=6, include_labels=False):
        assert record.labels is None
        if record.patient_id != states[0].patient_id:
            states.append(state_from_patient(record.patient, record.patient_id))
    assert [s.patient_id for s in states] == [f'ddxplus:validate:{i}' for i in range(1, 7)]
    cases = []
    for state, previous in zip(states, prior['cases']):
        assert state.model_dump() == previous['patient_state']
        original = state.model_dump_json()
        literal, omitted = backend.query_for(state)
        compressed = compress_baseline(state, literal, embedder.tokenizer, embedder.max_tokens)
        assert not compressed.baseline_facts_lost
        assert set(compressed.baseline_included) <= set(compressed.included)
        assert len(compressed.included) + len(compressed.omitted) == len(compressed.baseline_included) + omitted
        baseline = measure(literal, raw)
        control = measure(literal, guarded)
        candidate = measure(compressed.query, guarded)
        assert state.model_dump_json() == original
        assert literal == previous['baseline']['upstream']['query']
        assert baseline['upstream']['resolution'] == previous['baseline']['upstream']['resolution']
        assert [h['id'] for h in baseline['upstream']['results']] == [h['id'] for h in previous['baseline']['upstream']['results']]
        delta = max(abs(a['distance'] - b['distance']) for a, b in
                    zip(baseline['upstream']['results'], previous['baseline']['upstream']['results']))
        assert delta < 1e-4
        if not cases:
            assert literal == run['medical_retrieval']['query']
            assert baseline['upstream']['results'] == run['medical_retrieval']['upstream']['results']
        case = dict(patient_id=state.patient_id, patient_state=state.model_dump(), baseline=baseline,
                    guarded_literal_control=control, candidate=candidate, representation=compressed.audit(),
                    baseline_omitted_count=omitted, baseline_replay_max_distance_delta=delta,
                    lost_baseline_fact_count=len(compressed.baseline_facts_lost), state_unchanged=True)
        cases.append(case)
        folder = OUT / f'case-{len(cases)}'
        folder.mkdir(exist_ok=True)
        for name, data in (('baseline', baseline), ('candidate', candidate)):
            (folder / f'{name}-query.txt').write_text(data['query'] + '\n', encoding='utf8')
            (folder / f'{name}-embedded.txt').write_text(data['upstream']['resolution']['embedding_query'] + '\n', encoding='utf8')
            write_json(folder / f'{name}-aliases.json', data['aliases'])
        write_json(folder / 'facts.json', compressed.audit())
        print(state.patient_id, tokens(literal), '->', compressed.token_count,
              'lost', len(compressed.baseline_facts_lost), 'added', len(compressed.newly_added),
              'omitted', len(compressed.omitted), 'accepted', baseline['accepted_hits'], '->', candidate['accepted_hits'],
              [h['title'] for h in candidate['upstream']['results']], flush=True)
    probes = ['all', 'bed all day long', 'ALL', 'sex', 'age', 'Presenting:', 'Symptoms:', 'History:', 'Medications:', 'Context:']
    # Resolver-only diagnostics; not alternate candidate searches or query tuning.
    alias_probes = {q: dict(raw=raw.aliases.resolve(q), guarded=guarded.aliases.resolve(q)) for q in probes}
    write_json(OUT / 'comparison.json', dict(
        scope='One fixed baseline-preserving candidate, no clinical outcome/label evaluation',
        rows='Saved validate:1 and prespecified consecutive validate:2..6, same full states as new3',
        model=embedder.identity, snapshot=run['medical_retrieval']['upstream']['snapshot'],
        top_k=5, max_distance=1.10, metric='squared_l2', stored_chunks=len(rows),
        baseline_total_facts_lost=sum(c['lost_baseline_fact_count'] for c in cases),
        structural_metadata_added=False, alias_rules_changed=False, cases=cases, alias_probes=alias_probes))


def main():
    import psutil
    if psutil.virtual_memory().available < 2_000_000_000:
        raise RuntimeError('Need 2 GB available memory for one cached CPU model')
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'comparison.json').exists():
        raise RuntimeError('Comparison already exists; do not overwrite this experiment')
    if not (OUT / 'protected-before.json').exists():
        write_json(OUT / 'protected-before.json', protected_inventory())
    check_protected('pre-evaluation-integrity.json')
    paths = [Path(__file__), ROOT / 'rag/amg/baseline_compression.py', ROOT / 'rag/amg/query_adapter.py',
             ROOT / 'rag/amg/alias_safety.py', ROOT / 'rag/amg/backend.py']
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    write_json(OUT / 'experiment-code-sha256.json', code_hashes)
    secrets = [p for p in (ROOT.parent / '.env', ROOT / 'vendor/amg/.env') if p.exists()]
    secret_before = {p: digest(p) for p in secrets}  # in memory only
    run = json.loads((ROOT / 'outputs/amg-live/sample_case_001.json').read_text(encoding='utf8'))
    source_db = ROOT / 'vendor/amg/knowledge/medlineplus_lab' / run['medical_retrieval']['upstream']['snapshot'] / 'chroma'
    temporary = Path(tempfile.mkdtemp(prefix='amg-new4-readonly-'))
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
        integrity = check_protected('post-evaluation-protected.json')
        integrity.update(credentials_unchanged=secret_before == {p: digest(p) for p in secrets},
                         temporary_database_removed=not temporary.exists(), worker_exit_code=status,
                         experiment_code_unchanged=code_hashes == {str(p.relative_to(ROOT)): digest(p) for p in paths})
        write_json(OUT / 'integrity.json', integrity)
        if not integrity['credentials_unchanged'] or not integrity['experiment_code_unchanged']:
            raise RuntimeError('Credential or experiment-code integrity failure')
    if status:
        raise SystemExit(status)


if __name__ == '__main__':
    worker(Path(sys.argv[2])) if len(sys.argv) > 1 and sys.argv[1] == '--worker' else main()
