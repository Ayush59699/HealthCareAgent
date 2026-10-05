"""Local depth/reranking probes must remain independent of diagnostic routing."""
import copy
import unittest
from unittest.mock import Mock
from rag.retrieval_experiment import evaluate_depths, rerank_candidates
from rag.focused_medical import query_plan, state_from_patient
from tests.test_focused_medical import hit
from tests.test_patient_rag import record


class RetrievalExperimentTests(unittest.TestCase):
    def state(self):
        return state_from_patient(record().patient, record().patient_id)

    def test_four_depths_same_label_free_query_and_no_mutation(self):
        hits = [hit('Fever', 'Fever and cough clinical information.')]
        original = copy.deepcopy(hits)
        rag = Mock()
        rag.retrieve.return_value = hits
        state = self.state()
        report = evaluate_depths(state, rag, topics=['Fever', 'Panic'])
        self.assertEqual([c.kwargs['top_k'] for c in rag.retrieve.call_args_list], [5, 10, 20, 50])
        self.assertEqual({c.args[0] for c in rag.retrieve.call_args_list}, {query_plan(state)['query']})
        self.assertEqual(report['depths'][0]['topic_title_probes'], [
            {'topic': 'Fever', 'ranks': [1]}, {'topic': 'Panic', 'ranks': []}])
        self.assertEqual(report['depths'][0]['candidates'][0]['text'], hits[0]['text'])
        self.assertEqual(original, hits)
        self.assertNotIn(state.patient_id, rag.retrieve.call_args.args[0])
        self.assertNotIn('Panic', rag.retrieve.call_args.args[0])

    def test_reranker_prefers_symptoms_to_history_preserves_source(self):
        plan = {'symptom_concepts': ['shortness of breath', 'choking'], 'history_concepts': ['depression']}
        weak = hit('Depression', 'Mood history.', .95)
        strong = hit('Clinical symptoms', 'Difficulty breathing and choking.', .6)
        bad = hit('Transplant', 'Kidney transplant.', .99)
        ranked, audit = rerank_candidates(plan, [weak, strong, bad])
        self.assertEqual([r['hit'] for r in ranked], [strong, weak])
        self.assertIs(ranked[0]['hit'], strong)
        self.assertEqual(ranked[0]['hit']['score'], .6)
        self.assertEqual(len(audit), 3)
        self.assertFalse(next(a for a in audit if a['chunk_id'] == bad['chunk_id'])['retained'])

    def test_reranker_bounded_deterministic_and_validates_discarded_hits(self):
        plan = query_plan(self.state())
        hits = [hit('Fever ' + str(i), 'Fever clinical information ' + str(i)) for i in range(8)]
        ranked, _ = rerank_candidates(plan, hits)
        self.assertEqual(len(ranked), 5)
        self.assertEqual(ranked, rerank_candidates(plan, list(reversed(hits)))[0])
        bad = hit('Transplant', 'Kidney transplant.')
        bad['text'] = 'Tampered payload'
        with self.assertRaises(ValueError):
            rerank_candidates(plan, [bad])

    def test_symptom_query_is_explicit_alternative_not_live_change(self):
        state = self.state()
        before = query_plan(state)
        rag = Mock()
        rag.retrieve.return_value = []
        report = evaluate_depths(state, rag, symptom_only=True)
        self.assertEqual(report['query_variant'], 'symptom_only')
        self.assertNotEqual(report['plan']['query'], before['query'])
        self.assertEqual(query_plan(state), before)

    def test_invalid_depths_and_no_observations_fail_before_retrieval(self):
        rag = Mock()
        for depths in ([], [0], [True], [5, 5], [-1], [2.0]):
            with self.assertRaises(ValueError):
                evaluate_depths(self.state(), rag, depths=depths)
        empty = self.state().model_copy(update={'symptoms': [], 'presenting_evidence': []})
        with self.assertRaises(ValueError):
            evaluate_depths(empty, rag)
        rag.retrieve.assert_not_called()

    def test_missing_topic_is_not_reported_as_clinical_failure(self):
        rag = Mock()
        rag.retrieve.return_value = []
        report = evaluate_depths(self.state(), rag, topics=['Unknown topic'])
        self.assertEqual(report['depths'][0]['topic_title_probes'][0]['ranks'], [])
        self.assertEqual(report['depths'][0]['experimental_top5'], [])
        self.assertNotIn('recall', report)

    def test_cli_import_has_no_probe_side_effect_and_defaults_are_fixed(self):
        from scripts.evaluate_medical_retrieval import arguments
        args = arguments([])
        self.assertEqual(args.depths, [5, 10, 20, 50])
        self.assertFalse(args.compare_symptom_query)
        self.assertEqual(args.sample, 2)


if __name__ == '__main__':
    unittest.main()
