"""Task 7.3 software contracts only; no clinical/relevance labels or cloud calls."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np

from rag.agents.models import PatientState
from rag.experimental_medical.index import (BlockSections, make_records, retrieval_windows,
    ExperimentalIndex, VERSION, canonical, digest, build_index, source_sections)
from rag.experimental_medical.queries import build_queries, observed_facts, semantic_groups
from rag.experimental_medical.retrieval import BM25, fuse, choose_final, HybridRetriever
from rag.experimental_medical.inspection import topic_trace
from tests.test_medical_rag import document


class FakeEmbedding:
    """Explicit test double; never a runtime embedding fallback."""
    signature = {'dimension': 3, 'model_name': 'unit-test-only'}
    def __init__(self):
        self.calls = []
    def embed_texts(self, texts):
        texts = list(texts)
        self.calls.extend(texts)
        return [[1., float('pain' in t), float('asthma' in t)] for t in texts]


def patient():
    return PatientState(patient_id='ddxplus:validate:999', age=10, sex='F',
        presenting_evidence=['Do you have shortness of breath? = Yes'],
        symptoms=['Do you have choking? = Yes', 'Do you have chest pain? = Yes',
                  'Do you have palpitations? = Yes', 'Do you have sweating? = Yes'],
        antecedents=['Have you had asthma? = Yes'], relevant_findings=[], missing_information=[], uncertainty_notes=[])


def records():
    return make_records([(replace(document(), document_id=f'pmc:PMC{i}', title=title, text=text),
                         [{'title': 'Symptoms', 'text': text}]) for i, (title, text) in enumerate([
                             ('Airways', 'Wheezing and breathing difficulty.'),
                             ('Circulation', 'Palpitations and sweating.'),
                             ('Unrelated', 'A kidney paper.'),
                             ('Background', 'Asthma background information.')], 1)], lambda x: len(x.split()), 100)


class ExperimentalMedicalTests(unittest.TestCase):
    def test_blocks_keep_source_list_and_paragraph_boundaries(self):
        p = BlockSections()
        p.feed('<h2>Symptoms</h2><p>These include:</p><ul><li>Wheezing</li><li>Chest pain</li></ul><script>BAD</script>')
        p.flush()
        self.assertEqual(p.sections, [{'title': 'Symptoms', 'text': 'These include:\nWheezing\nChest pain'}])

    def test_sentence_windows_and_full_parent_survive(self):
        text = 'First sentence is complete. Second sentence is complete. Third sentence is complete.'
        rs = make_records([(replace(document(), text=text), [{'title': 'Symptoms', 'text': text}])], lambda x: len(x.split()), 12)
        self.assertGreater(len(rs), 1)
        self.assertEqual({r['text'] for r in rs}, {text})
        self.assertTrue(all(r['window_text'].endswith('.') for r in rs))
        self.assertTrue(all(r['retrieval_text'].startswith('Test asthma\nSymptoms\n') for r in rs))
        self.assertEqual(len({r['parent_id'] for r in rs}), 1)
        self.assertEqual(len({r['chunk_id'] for r in rs}), len(rs))

    def test_oversize_atomic_span_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            list(retrieval_windows('one two three four five six', 'Header', lambda x: len(x.split()), 5))

    def test_reconstruction_does_not_invent_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'corpus.json'
            path.write_text('{"entries": []}')
            doc = document()
            self.assertEqual(list(source_sections(path, [doc]))[0][1][0]['text'], doc.text)
            bad = replace(doc, metadata={'sections': [{'title': 'Fabrication', 'text': 'Invented claim.'}]})
            with self.assertRaises(ValueError):
                list(source_sections(path, [bad]))

    def test_queries_label_free_and_source_referenced(self):
        state, embed = patient(), FakeEmbedding()
        before = state.model_dump()
        plan = build_queries(state, embed)
        self.assertEqual(state.model_dump(), before)
        self.assertTrue(plan['queries'])
        for q in plan['queries']:
            self.assertTrue(q['fact_ids'])
            self.assertNotIn(state.patient_id, q['text'])
            self.assertNotIn('diagnosis', q['text'])
        self.assertTrue(any('asthma' in q['text'] and q['kind'] == 'history' for q in plan['queries']))
        self.assertNotIn(state.patient_id, ' '.join(embed.calls))

    def test_negative_unknown_numeric_are_not_positive_queries(self):
        state = patient().model_copy(update={'symptoms': ['Fever? = No', 'Cough? = Unknown', 'Pain rating? = 9', 'Onset? = 2.5'],
                                           'presenting_evidence': [], 'antecedents': []})
        embed = FakeEmbedding()
        result = build_queries(state, embed)
        self.assertEqual(result['queries'], [])
        self.assertEqual(len(result['excluded_from_positive_queries']), 4)
        self.assertEqual(embed.calls, [])

    def test_raw_rows_and_extra_label_fields_rejected(self):
        with self.assertRaises(TypeError):
            build_queries({'PATHOLOGY': 'LABEL_SENTINEL'}, FakeEmbedding())
        with self.assertRaises(ValueError):
            PatientState.model_validate({**patient().model_dump(), 'PATHOLOGY': 'LABEL_SENTINEL'})

    def test_all_observations_accounted_for_not_first_twelve_only(self):
        state = patient().model_copy(update={'symptoms': [f'Observed distinctive symptomword{i} sensation = Yes' for i in range(20)]})
        facts, ignored = observed_facts(state)
        refs = {ref for f in facts for ref in f['fact_ids']} | {r['fact_id'] for r in ignored}
        self.assertTrue({f'patient:symptoms:{i}' for i in range(20)} <= refs)

    def test_semantic_clustering_is_deterministic_and_not_disease_specific(self):
        facts = [{'text': t} for t in ['x', 'y', 'z']]
        vectors = [[1., 0.], [.9, .1], [0., 1.]]
        self.assertEqual(semantic_groups(facts, vectors, 2), [[facts[0], facts[1]], [facts[2]]])

    def test_duplicate_observations_keep_all_original_fact_ids(self):
        state = patient().model_copy(update={'symptoms': ['Do you have shortness of breath? = Yes']})
        facts, _ = observed_facts(state)
        fact = next(f for f in facts if f['concept'] == 'shortness of breath')
        self.assertEqual(fact['fact_ids'], ['patient:presenting_evidence:0', 'patient:symptoms:0'])

    def test_bm25_rank_zero_overlap_and_repeat_determinism(self):
        bm = BM25(['asthma wheezing', 'kidney kidney', 'asthma'])
        self.assertEqual(bm.search('unknownword', 50), [])
        self.assertEqual(bm.search('asthma', 50)[0][0], 2)
        self.assertEqual(bm.search('asthma', 50), bm.search('asthma asthma', 50))
        with self.assertRaises(ValueError):
            bm.search('asthma', 0)

    def test_rrf_uses_ranks_not_incompatible_raw_scores(self):
        rs = records()
        ranks = [('q', 'dense', [(0, .9), (1, .8)]), ('q', 'bm25', [(1, 500), (0, 10)])]
        fused = fuse(ranks, rs)
        self.assertAlmostEqual(fused[0]['rrf_score'], 1/61 + 1/62)
        self.assertEqual(len(fused), 2)
        self.assertEqual(len(fused[0]['origins']), 2)
        with self.assertRaises(ValueError):
            fuse([('q', 'dense', [(0, .5), (0, .4)])], rs)

    def fixture(self):
        index = Mock()
        index.records = records()
        index.vectors = np.asarray([[1., 0., 0.]] * len(index.records))
        index.search.side_effect = lambda vector, depth: [(i, .8-i*.01) for i in range(len(index.records))]
        reranker = Mock()
        reranker.signature = {'model': 'test-double'}
        reranker.truncated_pairs = 0
        reranker.score.side_effect = lambda pairs: [float(i) for i in range(len(pairs))]
        return HybridRetriever(index, FakeEmbedding(), reranker), reranker

    def test_broad_pool_reranked_without_old_lexical_gate(self):
        retriever, reranker = self.fixture()
        with patch('rag.focused_medical.filter_candidates', side_effect=AssertionError('Old gate called')):
            result = retriever.retrieve(patient(), candidate_depth=100)
        self.assertEqual(len(result['candidate_pool']), 4)
        self.assertEqual(len(result['reranked']), 4)
        self.assertFalse(result['pre_reranking_lexical_filter'])
        self.assertTrue(any('kidney' in text for _, text in reranker.score.call_args.args[0]))
        self.assertLessEqual(len(result['final_passages']), 5)
        self.assertEqual(len({r['record']['source']['document_id'] for r in result['final_passages']}), len(result['final_passages']))

    def test_no_mutation_of_candidate_evidence(self):
        retriever, _ = self.fixture()
        before = copy.deepcopy(retriever.index.records)
        retriever.retrieve(patient())
        self.assertEqual(retriever.index.records, before)

    def test_failed_reranker_does_not_fall_back_or_deliver_evidence(self):
        retriever, reranker = self.fixture()
        reranker.score.side_effect = RuntimeError('model failed')
        with self.assertRaises(RuntimeError):
            retriever.retrieve(patient())
        reranker.score.side_effect = lambda pairs: [float('nan')] * len(pairs)
        with self.assertRaises(ValueError):
            retriever.retrieve(patient())

    def test_no_observations_and_zero_budget_are_empty(self):
        retriever, reranker = self.fixture()
        state = patient().model_copy(update={'symptoms': [], 'antecedents': [], 'presenting_evidence': []})
        self.assertEqual(retriever.retrieve(state)['final_passages'], [])
        self.assertEqual(retriever.retrieve(patient(), final_k=0)['final_passages'], [])
        reranker.score.assert_not_called()

    def test_depths_final_limits_and_unknown_plan_injection(self):
        retriever, _ = self.fixture()
        for depth in (5, 0, True, 50.0):
            with self.assertRaises(ValueError):
                retriever.retrieve(patient(), candidate_depth=depth)
        for final in (-1, 6, True):
            with self.assertRaises(ValueError):
                retriever.retrieve(patient(), final_k=final)
        with self.assertRaises(TypeError):
            retriever.retrieve(patient(), plan={'query': 'LABEL_SENTINEL'})

    def test_history_background_slot_does_not_assign_current_diagnosis(self):
        retriever, _ = self.fixture()
        result = retriever.retrieve(patient())
        self.assertTrue(any(r['selection_role'] == 'history' for r in result['final_passages']))
        self.assertNotIn('diagnosis', result)

    def test_missing_topic_not_called_genuine_corpus_gap(self):
        retriever, _ = self.fixture()
        exp = retriever.retrieve(patient())
        trace = topic_trace('Missing condition', retriever.index.records, {'broad_candidates': []}, exp)
        self.assertIn('unconfirmed', trace['failure_stage_or_outcome'])
        self.assertEqual(trace['catalog_windows'], 0)

    def test_index_hash_vector_and_source_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            rs = records()
            payload = b''.join(canonical(r) + b'\n' for r in rs)
            (path/'passages.jsonl').write_bytes(payload)
            np.save(path/'vectors.npy', np.asarray([[1., 0., 0.]] * len(rs)))
            config = {'version': VERSION, 'embedding': FakeEmbedding.signature, 'windows': len(rs),
                      'passages_sha256': digest(payload), 'vectors_sha256': digest((path/'vectors.npy').read_bytes())}
            (path/'index.json').write_bytes(canonical(config))
            index = ExperimentalIndex(path, FakeEmbedding.signature)
            self.assertEqual(len(index.search([1., 0., 0.], 50)), len(rs))
            with self.assertRaises(ValueError):
                ExperimentalIndex(path, {'dimension': 999})
            (path/'passages.jsonl').write_bytes(payload + b' ')
            with self.assertRaises(ValueError):
                ExperimentalIndex(path, FakeEmbedding.signature)

    def test_build_refuses_live_or_existing_destination_before_embeddings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            manifest = path/'corpus.json'
            manifest.write_text('{}')
            for output in (path, path/'qdrant'/'nested'):
                with self.assertRaises(ValueError):
                    build_index(manifest, output, FakeEmbedding())

    def test_cli_defaults_and_no_download_default(self):
        from scripts.inspect_hybrid_medical_retrieval import arguments
        args = arguments([])
        self.assertEqual(args.depths, [50, 100])
        self.assertEqual(args.samples, [1, 2, 3, 4, 5])
        self.assertFalse(args.allow_reranker_download)
        self.assertEqual(args.topic, [])


if __name__ == '__main__':
    unittest.main()
