"""Task 7.5 software contracts only; no medical relevance labels or model calls."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from orchestration.evidence import fingerprint
from rag.agents.models import PatientState
from rag.experimental_medical.qdrant_index import QdrantMedicalIndex
from rag.experimental_medical.retrieval import HybridRetriever, RERANKER_MODEL, RERANKER_REVISION
from rag.experimental_medical.ranking_ablation import (
    VARIANTS, ablate, make_snapshot, population_features, validate_frozen_experiment)
from rag.medical_ingestion.chunker import chunk_document
from tests.test_medical_rag import document
from tests.test_experimental_medical import FakeEmbedding, patient


def fixture():
    payloads = []
    for i, (title, text) in enumerate([
        ('Child Topic', 'Asthma can cause wheezing and difficulty breathing.'),
        ('Teen Topic', 'Chest pain and sweating need assessment.'),
        ('Background Topic', 'Long term clinical information.'),
        ('Other Topic', 'Different source information.')], 1):
        doc = replace(document(), document_id=f'pmc:PMC{i}', title=title, text=text)
        payloads.append(asdict(next(chunk_document(doc))))
    medical = Mock()
    medical.embedding = FakeEmbedding()
    medical.embedding.dimension = 3
    medical.store.collection_name = 'medical_knowledge'
    medical.store.signature = medical.embedding.signature
    medical.store.iter_points.return_value = [SimpleNamespace(payload=p, vector=[1., 0., 0.]) for p in payloads]
    medical.store.search.return_value = [{**p, 'score': .9 - i*.01} for i, p in enumerate(payloads)]
    index = QdrantMedicalIndex(medical)
    reranker = Mock(signature={'model': RERANKER_MODEL, 'revision': RERANKER_REVISION,
        'max_length': 512, 'score': 'raw cross-encoder logit; not clinical confidence'}, truncated_pairs=0)
    reranker.score.side_effect = lambda pairs: [3.] * len(pairs)
    state = patient()
    experiment = HybridRetriever(index, medical.embedding, reranker).retrieve(state, candidate_depth=50)
    experiment['candidate_records'] = index.records
    experiment['evidence_snapshot'] = make_snapshot(state, experiment['final_passages']).model_dump(mode='json')
    return state, experiment, {p['chunk_id']: p for p in payloads}


class RankingAblationTests(unittest.TestCase):
    def test_frozen_cache_revalidates_and_baseline_reproduces(self):
        state, experiment, payloads = fixture()
        records = validate_frozen_experiment(state, experiment, payloads)
        result = ablate(state, experiment, records, 'task74')
        self.assertEqual(result['evidence_snapshot'], experiment['evidence_snapshot'])
        self.assertEqual(result['selected_ids'], [r['chunk_id'] for r in experiment['final_passages']])

    def test_every_ablation_preserves_candidates_scores_queries_and_sources(self):
        state, experiment, payloads = fixture()
        records = validate_frozen_experiment(state, experiment, payloads)
        before = fingerprint(experiment)
        for variant in VARIANTS:
            result = ablate(state, experiment, records, variant)
            self.assertEqual(fingerprint(experiment), before)
            self.assertEqual(result['statistics']['candidates'], len(experiment['reranked']))
            self.assertEqual(result['frozen_scores_fingerprint'], fingerprint(experiment['reranked']))
            self.assertEqual(len(result['candidate_audit']), len(experiment['reranked']))

    def test_symptom_weight_does_not_change_raw_logit_gate(self):
        state, experiment, payloads = fixture()
        records = validate_frozen_experiment(state, experiment, payloads)
        for row in experiment['reranked']:
            row['symptoms_score'] = -2.
        result = ablate(state, experiment, records, 'symptoms_weight_2')
        self.assertEqual(result['statistics']['positive_symptom_scores'], 0)
        self.assertEqual(result['statistics']['selected_current'], 0)
        for row in result['candidate_audit']:
            role = row['role_details']['symptoms']
            self.assertEqual(role['ablation_ranking_score'], role['task74_ranking_score'] - 2.)
            self.assertEqual(role['raw_reranker_score'], -2.)

    def test_background_cap_does_not_fill_current_slots(self):
        state, experiment, payloads = fixture()
        records = validate_frozen_experiment(state, experiment, payloads)
        for row in experiment['reranked']:
            row['symptoms_score'] = -1.
        result = ablate(state, experiment, records, 'background_cap_1')
        self.assertEqual(result['statistics']['selected'], 1)
        self.assertEqual(result['statistics']['selected_background'], 1)
        self.assertTrue(any(r['selection_or_removal_stage'] == 'role_allocation' for r in result['candidate_audit']))

    def test_population_scope_explicit_only_and_unknown_not_mismatch(self):
        state, experiment, _ = fixture()
        record = deepcopy(experiment['candidate_records'][0])
        record['source']['title'] = 'Teen Topic'
        self.assertTrue(population_features(state, record)['mismatch'])
        self.assertEqual(population_features(state.model_copy(update={'age': 15}), record)['status'], 'match')
        self.assertEqual(population_features(state.model_copy(update={'age': None}), record)['status'], 'unknown')
        record['source']['title'] = 'General Topic'
        record['text'] = 'Teen and adult material is mentioned incidentally.'
        self.assertEqual(population_features(state, record)['status'], 'unknown')
        record['source']['title'] = 'Older Adult Topic'
        self.assertTrue(population_features(state.model_copy(update={'age': 55}), record)['mismatch'])
        self.assertFalse(population_features(state.model_copy(update={'age': 68}), record)['mismatch'])

    def test_population_penalty_is_soft_and_does_not_rewrite_model_scores(self):
        state, experiment, payloads = fixture()
        records = validate_frozen_experiment(state, experiment, payloads)
        result = ablate(state, experiment, records, 'population_penalty_0.5')
        mismatched = [r for r in result['candidate_audit'] if r['population_features']['mismatch']]
        self.assertTrue(mismatched)
        for row in mismatched:
            for details in row['role_details'].values():
                self.assertEqual(details['ablation_ranking_score'], details['task74_ranking_score'] - .5)
                self.assertEqual(details['raw_reranker_score'], 3.)
                self.assertTrue(details['positive_logit_eligible'])

    def test_history_novelty_uses_query_and_section_not_condition_boost(self):
        state, experiment, payloads = fixture()
        records = validate_frozen_experiment(state, experiment, payloads)
        for row in experiment['reranked']:
            row['symptoms_score'] = -1.
        result = ablate(state, experiment, records, 'history_novelty')
        # Fixture has one observed history query and separate documents.
        self.assertEqual(result['statistics']['selected_background'], 1)
        self.assertTrue(any('redundant_history_query' in r['final_selection_reason'] for r in result['candidate_audit']))

    def test_audit_for_rejected_candidates_has_all_roles_ranks_and_stages(self):
        state, experiment, payloads = fixture()
        records = validate_frozen_experiment(state, experiment, payloads)
        for row in experiment['reranked']:
            row['symptoms_score'] = row['history_score'] = -3.
        result = ablate(state, experiment, records, 'history_novelty')
        self.assertEqual(result['selected_ids'], [])
        for row in result['candidate_audit']:
            self.assertEqual(row['selection_or_removal_stage'], 'reranker_eligibility')
            for field in ('title', 'section', 'source', 'chunk_id', 'rrf_score', 'reranker_score',
                          'population_features', 'final_selection_reason', 'selector_events'):
                self.assertIn(field, row)
            for details in row['role_details'].values():
                for field in ('originating_query', 'query_role', 'dense_rank', 'bm25_rank', 'concept_coverage'):
                    self.assertIn(field, details)

    def test_depth_and_unknown_variant_cannot_be_changed(self):
        state, experiment, payloads = fixture()
        for depth in (100, 50.0, True):
            bad = deepcopy(experiment)
            bad['candidate_depth'] = depth
            with self.assertRaises(ValueError):
                validate_frozen_experiment(state, bad, payloads)
        with self.assertRaises(ValueError):
            ablate(state, experiment, {}, 'new_tuned_variant')

    def test_altered_fusion_query_payload_score_and_snapshot_fail_closed(self):
        state, experiment, payloads = fixture()
        for change in ('fusion', 'query', 'payload', 'score', 'snapshot'):
            bad = deepcopy(experiment)
            if change == 'fusion':
                bad['fused_candidates'][0]['rrf_score'] += .1
            elif change == 'query':
                bad['plan']['queries'][0]['text'] += ' INJECTED_LABEL'
            elif change == 'payload':
                bad['candidate_records'][0]['text'] = 'Altered text'
            elif change == 'score':
                bad['reranked'][0]['history_score'] = float('nan')
            else:
                bad['evidence_snapshot']['snapshot_id'] = '0'*64
            with self.assertRaises(ValueError, msg=change):
                validate_frozen_experiment(state, bad, payloads)

    def test_hidden_fields_and_missing_current_source_rejected(self):
        state, experiment, payloads = fixture()
        with self.assertRaises(ValueError):
            PatientState.model_validate({**state.model_dump(), 'PATHOLOGY': 'HIDDEN'})
        with self.assertRaises(TypeError):
            validate_frozen_experiment({'PATHOLOGY': 'HIDDEN'}, experiment, payloads)
        with self.assertRaises(ValueError):
            validate_frozen_experiment(state, experiment, {})

    def test_cli_frozen_defaults_and_no_model_or_depth_flags(self):
        from scripts.ablate_medical_ranking import arguments
        self.assertEqual(arguments([]).samples, [1, 2, 3])
        with self.assertRaises(SystemExit):
            arguments(['--candidate-depth', '100'])

    def test_memory_stop_records_current_phase_without_starting_workers(self):
        from scripts.ablate_medical_ranking import MemoryMonitor, MemorySafetyStop
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            monitor = MemoryMonitor(output)
            with patch.object(monitor.psutil, 'virtual_memory', return_value=SimpleNamespace(used=13.6e9)):
                with self.assertRaises(MemorySafetyStop):
                    monitor.begin('case:test:variant')
            result = json.loads((output / 'memory-stop.json').read_text())
            self.assertEqual(result['phase'], 'case:test:variant')
            self.assertEqual(result['system_gb'], 13.6)


if __name__ == '__main__':
    unittest.main()
