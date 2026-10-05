"""Thin adapter over the supplied, unchanged AMG MedlinePlus retriever.

No new retrieval/reranking, web search, graph expansion, LLM synthesis or ingestion.
The vendor code is byte-pinned by its own published snapshot manifest.
"""
import copy
import importlib
import json
from pathlib import Path
import sys
from .provenance import BACKEND, validate_amg_hit

VENDOR_ROOT = Path(__file__).resolve().parents[2] / 'vendor' / 'amg'


def load_upstream():
    # Upstream consists of standalone scripts with absolute sibling imports. Scope
    # path changes to loading; refuse collisions rather than silently import other code.
    names = ('medlineplus_ingest', 'medlineplus_retrieval_check', 'medlineplus_lab')
    for name in names:
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != VENDOR_ROOT:
            raise RuntimeError('Conflicting AMG module already loaded')
    sys.path.insert(0, str(VENDOR_ROOT))
    try:
        return importlib.import_module('medlineplus_lab')
    finally:
        sys.path.remove(str(VENDOR_ROOT))


class AMGMedicalEvidence:
    """One cached CPU MiniLM and one existing Chroma collection per application."""
    backend = BACKEND

    def __init__(self):
        upstream = load_upstream()
        self.retriever, self.manifest = upstream.open_experiment()
        self.max_distance = upstream.DEFAULT_MAX_DISTANCE  # unchanged tested policy
        pointer = json.loads((upstream.STORE / 'CURRENT.json').read_text(encoding='utf8'))
        self.snapshot = pointer['snapshot']
        self.audit = None
        # The existing loader verifies this artifact checksum. Retain only per-ID
        # hashes instead of a duplicate corpus; validate returned text AND metadata.
        path = upstream.STORE / self.snapshot / 'chunks.jsonl'
        from orchestration.evidence import fingerprint
        self._source_hashes = {}
        with path.open(encoding='utf8') as stream:
            for line in stream:
                chunk = json.loads(line)
                self._source_hashes[chunk['id']] = fingerprint(chunk)

    def count(self):
        return self.retriever.collection.count()

    def query_for(self, state):
        """Bound a single literal case query to AMG's finite embedding window.

        No predicted diagnoses, labels, synonym additions or new query expansion.
        Full patient facts still reach the agents; omitted retrieval facts are audited.
        """
        facts = list(dict.fromkeys(state.presenting_evidence + state.symptoms + state.antecedents))
        tokenizer = self.retriever.embedder.tokenizer
        limit = self.retriever.embedder.max_tokens
        parts, omitted = [], []
        for fact in facts:
            candidate = '; '.join(parts + [fact])
            if len(candidate) <= 4000 and len(tokenizer.encode(candidate, add_special_tokens=True, verbose=False)) <= limit:
                parts.append(fact)
            else:
                omitted.append(fact)
        return '; '.join(parts), len(omitted)

    def retrieve(self, query, top_k=5):
        from orchestration.evidence import fingerprint
        result = self.retriever.search(query, max_distance=self.max_distance, top_k=top_k)
        # Preserve the upstream audit, including rejected candidates, outside agent context.
        self.audit = {'backend': BACKEND, 'snapshot': self.snapshot, **copy.deepcopy(result)}
        accepted = []
        for hit in result['results']:
            source = {'id': hit['id'], 'text': hit['text'], 'metadata': hit['metadata']}
            if self._source_hashes.get(hit['id']) != fingerprint(source):
                raise ValueError('AMG returned evidence not matching the published source artifact')
            payload = {
                'backend': BACKEND, 'chunk_id': hit['id'], 'document_id': hit['metadata']['parent_id'],
                'source': 'medlineplus', 'title': hit['title'], 'url': hit['url'], 'text': hit['text'],
                'amg_metadata': copy.deepcopy(hit['metadata']),
                'retrieval': {'distance': hit['distance'], 'metric': 'squared_l2',
                              'acceptance': hit['acceptance'], 'max_distance': self.max_distance,
                              'snapshot': self.snapshot, 'collection': self.manifest['config']['collection']}}
            accepted.append({**validate_amg_hit(payload), 'score': -hit['distance']})
        return accepted

    def close(self):
        # Chroma PersistentClient has process-shared ownership and no public close;
        # do not use private stop/reset APIs that could invalidate another reader.
        self._source_hashes.clear()
        self.retriever = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
