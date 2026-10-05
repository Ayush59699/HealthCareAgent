"""Focused Task 7.9 deterministic doubles only: no models, corpus or clinical labels."""
from copy import deepcopy
import hashlib
from pathlib import Path
import unittest
from unittest.mock import Mock

from rag.experimental_medical.task79_selection_analysis import (
    adapt_saved, original_functions, trace_selector, replay, analyze, join_review, fingerprint)
from scripts.experiment_task79_selection import require_all_controls, Monitor, MemoryStop

ROOT = Path(__file__).resolve().parents[1]


def fixture(entries=None):
    entries = entries or [
        ('H', 'Child Mental Health', -3., 9., 'docH', 'delta background topic'),
        ('F', 'Flu', 10., -3., 'docF', 'fever chills nasal secretions'),
        ('T', 'Fainting', 9., -3., 'docT', 'dizziness loss consciousness'),
        ('D', 'Example D', 8., -3., 'docD', 'alpha beta gamma'),
        ('L', 'Example L', 7., -3., 'docL', 'epsilon zeta theta'),
        ('P', 'Panic Disorder', 6., -3., 'docP', 'fear choking stomach'),
        ('A', 'Asthma in Children', -2., 8., 'docA', 'airways childhood breathlessness'),
        ('N', 'Asthma in Children', -2., -1., 'docA', 'definition wheezing coughing'),
        ('O', 'Older Adult Mental Health', -2., -1., 'docO', 'older smoking context')]
    queries = [{'query_id': role + ':0', 'role': role, 'retrieval_query': 'synthetic ' + role,
                'reranker_input': 'synthetic ' + role, 'fact_ids': ['synthetic:' + role]}
               for role in ('symptoms','history')]
    audit = []
    for i, (cid, title, symptoms, history, doc, text) in enumerate(entries, 1):
        roles = {role: {'originating_query_id': role + ':0', 'originating_query': 'synthetic ' + role,
            'reranker_input': 'synthetic ' + role, 'fact_ids': ['synthetic:' + role],
            'raw_reranker_score': score, 'task74_ranking_score': score,
            'positive_logit_eligible': score > 0} for role, score in (('symptoms',symptoms),('history',history))}
        audit.append({'chunk_id': cid, 'source_id': 'medical:' + cid, 'source': 'synthetic', 'title': title,
            'section': 'definition' if cid == 'N' else 'symptoms', 'url': 'https://example.invalid/' + cid,
            'document_id': doc, 'evidence_text': text, 'evidence_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'rrf_score': 1./(60+i), 'fusion_rank': i, 'reranked_rank': i,
            'role_details': roles, 'selected': False, 'selected_role': None, 'selector_events': []})
    saved = {'candidate_depth': 50, 'variant': 'experimental', 'pair_count': len(audit)*2,
             'candidate_audit': audit, 'formatted_queries': queries, 'selected_ids': []}
    choose, tok = original_functions(ROOT)
    rows, records, q = adapt_saved(saved)
    exact = choose(rows, records, q, 5)
    traced = trace_selector(rows, records, q, tok)
    saved['selected_ids'] = [r['chunk_id'] for r in exact]
    selected = {r['chunk_id']: r['selection_role'] for r in exact}
    for a in audit:
        cid = a['chunk_id']
        a.update(selected=cid in selected, selected_role=selected.get(cid), selector_events=[
            {k: e[k] for k in ('phase','role','effective_ranking_score','reason')}
            for e in traced['events'] if e['chunk_id'] == cid])
    return saved


class Task79Tests(unittest.TestCase):
    def setUp(self):
        self.choose, self.tokens = original_functions(ROOT)
        self.saved = fixture()

    def both(self):
        old = replay(self.saved, self.choose, self.tokens)
        new = replay(self.saved, self.choose, self.tokens, True)
        return old, new, analyze(self.saved, old, new)

    def test_exact_control_ids_order_roles_and_events(self):
        result = replay(self.saved, self.choose, self.tokens)
        self.assertEqual(result['selected_ids'], ['F','T','D','L','H'])
        self.assertEqual(result['chronological_selected_ids'], ['H','F','T','D','L'])
        self.assertEqual(result['selected_roles']['H'], 'history')

    def test_changed_control_ids_stop_before_counterfactual(self):
        self.saved['selected_ids'].reverse()
        with self.assertRaisesRegex(ValueError, 'IDs/order'):
            replay(self.saved, self.choose, self.tokens)

    def test_changed_saved_event_fails_closed(self):
        self.saved['candidate_audit'][0]['selector_events'][0]['reason'] = 'invented'
        with self.assertRaisesRegex(ValueError, 'event trace'):
            replay(self.saved, self.choose, self.tokens)

    def test_all_six_real_controls_required(self):
        controls = [{'case': case, 'score_condition': model, 'passed': True}
                    for case in (1,2,3) for model in ('control','experimental')]
        require_all_controls(controls)
        with self.assertRaises(ValueError):
            require_all_controls(controls[:-1])
        controls[-1]['passed'] = False
        with self.assertRaises(ValueError):
            require_all_controls(controls)

    def test_final_budget_only_keeps_history_one_then_current_then_fill(self):
        old, new, analysis = self.both()
        self.assertEqual(new['selected_ids'], ['F','T','D','L','P','H','A'])
        self.assertEqual(new['chronological_selected_ids'][0], 'H')
        events = [e for e in new['events'] if e['chunk_id'] == 'A']
        self.assertEqual(events[0]['budget_kind'], 'history_priority_quota_1')
        self.assertEqual(events[-1]['reason'], 'selected_history_fill')
        self.assertEqual(analysis['summary']['additional_candidates'], 2)

    def test_panic_eligible_rank_and_actual_budget_occupants(self):
        old, new, analysis = self.both()
        panic = next(a for a in analysis['candidate_analysis'] if a['chunk_id'] == 'P')
        current = panic['roles'][0]
        self.assertTrue(current['eligible'])
        self.assertEqual(current['ranking_position'], 5)
        self.assertFalse(panic['selected'])
        self.assertTrue(panic['counterfactual_selected'])
        self.assertEqual(current['terminal_rule'], 'final_evidence_count_5')
        self.assertEqual(current['saved_pass_events'][0]['selected_before'], ['H','F','T','D','L'])

    def test_flu_and_fainting_positive_controls(self):
        old, new, analysis = self.both()
        for cid in ('F','T'):
            self.assertIn(cid, old['selected_ids'])
            self.assertIn(cid, new['selected_ids'])

    def test_asthma_gate_versus_budget_and_older_gate(self):
        _, _, analysis = self.both()
        by_id = {a['chunk_id']: a for a in analysis['candidate_analysis']}
        self.assertTrue(by_id['A']['eligible'])
        self.assertFalse(by_id['A']['selected'])
        self.assertTrue(by_id['A']['counterfactual_selected'])
        for cid in ('N','O'):
            self.assertFalse(by_id[cid]['eligible'])
            self.assertFalse(by_id[cid]['counterfactual_selected'])

    def test_rejection_counts_are_candidate_not_pass_counts(self):
        _, _, analysis = self.both()
        s = analysis['summary']
        self.assertEqual(s['eligible_candidates'], 7)
        self.assertEqual(s['selected_candidates'], 5)
        self.assertEqual(s['eligible_not_selected_candidates'], 2)
        self.assertEqual(s['rejection_reason_counts'], {'final_evidence_count_5':2})
        self.assertEqual(s['confirmed_final_budget_only_recoveries'], 2)

    def test_document_cap_two_unchanged_without_final_budget(self):
        self.saved = fixture([(str(i), 'Same document', 10.-i, -1., 'same', text)
            for i, text in enumerate(('alpha beta', 'gamma delta', 'epsilon zeta'))])
        _, _, analysis = self.both()
        self.assertEqual(analysis['summary']['counterfactual_selected_candidates'], 2)
        self.assertEqual(analysis['summary']['counterfactual_rejection_reason_counts'], {'document_limit_2':1})

    def test_jaccard_threshold_and_exact_blocker(self):
        self.saved = fixture([('a','A',3.,-1.,'one','alpha beta gamma delta'),
                              ('b','B',2.,-1.,'two','alpha beta gamma delta epsilon')])
        _, _, analysis = self.both()
        b = analysis['candidate_analysis'][1]
        self.assertEqual(b['counterfactual_terminal_rules'], ['near_duplicate_text'])
        event = b['roles'][0]['counterfactual_pass_events'][0]
        self.assertEqual(event['nonbudget_blockers_at_this_prefix'], [{'chunk_id':'a','jaccard':.8}])

    def test_parent_constraint_and_no_double_select_across_roles(self):
        rows, records, queries = adapt_saved(self.saved)
        records['P']['parent_id'] = records['F']['parent_id']
        result = trace_selector(rows, records, queries, self.tokens, True)
        self.assertNotIn('P', result['selected_ids'])
        event = next(e for e in result['events'] if e['chunk_id']=='P' and e['role']=='symptoms')
        self.assertEqual(event['reason'], 'parent_already_selected')
        self.assertEqual(event['nonbudget_blockers_at_this_prefix'], ['F'])
        self.assertEqual(len(result['selected_ids']), len(set(result['selected_ids'])))

    def test_budget_masks_nonbudget_failure_not_all_budget_hits_recover(self):
        entries = [('h','H',-1.,10.,'h','history reserved')]
        entries += [(str(i),'S',10.-i,-1.,str(i),'alpha beta gamma' if i in (0,4) else 'word'+chr(100+i)+' separate'+chr(100+i)) for i in range(5)]
        self.saved = fixture(entries)
        old, new, analysis = self.both()
        last = next(a for a in analysis['candidate_analysis'] if a['chunk_id']=='4')
        self.assertEqual(last['terminal_rules'], ['final_evidence_count_5'])
        self.assertFalse(last['budget_only_at_saved_prefix'])
        self.assertFalse(last['counterfactual_selected'])
        self.assertEqual(last['counterfactual_terminal_rules'], ['near_duplicate_text'])

    def test_saved_eligibility_or_source_inconsistency_stops(self):
        for mutate in (lambda s: s['candidate_audit'][0]['role_details']['history'].update(positive_logit_eligible=False),
                       lambda s: s['candidate_audit'][0].update(evidence_text='changed'),
                       lambda s: s['candidate_audit'].reverse()):
            saved = deepcopy(self.saved)
            mutate(saved)
            with self.assertRaises(ValueError):
                adapt_saved(saved)

    def test_no_mutation_of_frozen_data(self):
        before = fingerprint(self.saved)
        self.both()
        self.assertEqual(fingerprint(self.saved), before)

    def test_abc_is_only_posthoc_join_and_checks_text_score_identity(self):
        _, _, analysis = self.both()
        p = next(a for a in analysis['candidate_analysis'] if a['chunk_id']=='P')
        review = {'case':2, 'chunk_id':'P', 'role':'symptoms', 'review_id':'synthetic', 'class':'A',
            'eligibility_transition':'nonpositive_to_positive', 'experimental_raw_logit':6.,
            'experimental_query':'synthetic symptoms', 'evidence_text_sha256':p['evidence_text_sha256']}
        joined = join_review(analysis, [review], 2)
        self.assertEqual(joined['counts']['A']['eligible_not_selected_for_role'], 1)
        self.assertEqual(joined['counts']['A']['newly_selected_for_role'], 1)
        review['experimental_query'] = 'different'
        with self.assertRaises(ValueError):
            join_review(analysis, [review], 2)

    def test_memory_stop_aborts_selector_iteration(self):
        sample = Mock(side_effect=MemoryStop('synthetic stop'))
        with self.assertRaises(MemoryStop):
            replay(self.saved, self.choose, self.tokens, sample=sample)


if __name__ == '__main__':
    unittest.main()
