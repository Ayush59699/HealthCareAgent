"""Offline software-contract tests. Stub text is not runtime medical evidence."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag.who_lookup import WHOLookup, LookupResult, DocumentMatch, Source
from rag.who_provenance import excerpt_hit, sha256
from rag.combined_medical_evidence import CombinedEvidenceService, CombinedMedicalEvidence, who_query
from rag.who_amg_experiment import create_experimental_workflow, separated_input, SOURCE_INSTRUCTIONS
from rag.agents.grounding import state_from_patient, evidence_from_hits, context
from orchestration.evidence import EvidenceSnapshot
from orchestration.phase6.grounding import grounding_report
from application.decision import build_final_decision
from application.workflow import create_workflow
from tests.amg_helpers import configure_medical, configure_provider, amg_hit
from tests.safety_helpers import setup, SafetyScenarioLLM, semantic, critique


def match(kind='fact_sheet', body='Software fixture only. Not medical guidance.\n'):
    prefix = '/news-room/fact-sheets/detail/' if kind == 'fact_sheet' else '/news-room/questions-and-answers/item/'
    header = 'WHO FACT SHEET' if kind == 'fact_sheet' else 'WHO QUESTIONS AND ANSWERS'
    source = Source('fixture', 'Fixture', 'https://www.who.int' + prefix + 'fixture', kind, 'fixture.txt')
    text = f'{header}\nTitle: Fixture\nURL: {source.url}\nRetrieved: 2026-01-01\n\n====\n{body}'
    return DocumentMatch(source, 'exact', ('fixture',), text)


def lookup_stub(documents=None):
    documents = (match(), match('question_answer')) if documents is None else documents
    who = Mock()
    who.lookup.side_effect = lambda query, **kwargs: LookupResult(query, 'MATCH' if documents else 'NO_MATCH', documents, 'Fixture')
    return who


class CombinedEvidenceTests(unittest.TestCase):
    def fixture(self, *, mode='who_plus_amg', documents=None, provider=None, empty_amg=False):
        _, case, llm, patients, medical = setup(provider)
        configure_medical(medical)
        configure_provider(llm)
        if empty_amg:
            medical.retrieve.return_value = []
        who = lookup_stub(documents)
        workflow = create_experimental_workflow(llm, patients, medical, mode=mode, who=who)
        return workflow, case, llm, patients, medical, who

    def retrieve(self, fixture):
        workflow, case, *_ = fixture
        return workflow.evidence.retrieve(case.patient, state_from_patient(case.patient, case.patient_id))

    def test_four_availability_states(self):
        for docs, empty, expected in ((None, False, 'BOTH_AVAILABLE'), (None, True, 'WHO_ONLY'),
                                     ((), False, 'AMG_ONLY'), ((), True, 'NONE_AVAILABLE')):
            with self.subTest(expected=expected):
                fixture = self.fixture(documents=docs, empty_amg=empty)
                snapshot = self.retrieve(fixture)
                combined = fixture[0].evidence.combined
                self.assertEqual(combined.status, expected)
                self.assertEqual(bool(snapshot.medical_knowledge), expected != 'NONE_AVAILABLE')
                self.assertEqual(CombinedMedicalEvidence.model_validate_json(combined.model_dump_json()), combined)

    def test_provenance_and_exact_source_spans(self):
        fixture = self.fixture()
        snapshot = self.retrieve(fixture)
        restored = EvidenceSnapshot.model_validate_json(snapshot.model_dump_json())
        med = restored.agent_evidence()[1]
        for item in med[:2]:
            meta = item.metadata
            original = match(meta['document_type'])
            self.assertEqual(item.text, original.text[meta['char_start']:meta['char_end']])
            self.assertEqual(meta['document_sha256'], sha256(original.text))
            self.assertEqual(meta['filename'], original.source.filename)
            self.assertEqual(meta['url'], original.source.url)
            self.assertEqual(meta['matched_terms'], list(original.matched_terms))
        expected = evidence_from_hits([amg_hit()], 'medical_knowledge')[0]
        self.assertEqual(med[-1], expected)
        combined = fixture[0].evidence.combined
        self.assertEqual(len(combined.who_evidence.fact_sheets), 1)
        self.assertEqual(len(combined.who_evidence.questions_answers), 1)
        self.assertEqual(len(combined.amg_evidence.accepted_passages), 1)

    def test_query_is_literal_label_free_and_does_not_use_amg_or_hypotheses(self):
        fixture = self.fixture()
        self.retrieve(fixture)
        state = state_from_patient(fixture[1].patient, fixture[1].patient_id)
        expected = '; '.join(dict.fromkeys(state.presenting_evidence + state.symptoms + state.antecedents))
        fixture[-1].lookup.assert_called_once_with(expected, max_results=3)
        self.assertNotIn('Fixture', expected)
        self.assertNotEqual(expected, fixture[4].query_for.return_value[0])
        fixture[4].retrieve.assert_called_once_with(fixture[4].query_for.return_value[0], top_k=5)

    def test_label_bearing_record_rejected_before_any_retrieval(self):
        workflow, case, llm, patients, medical, who = self.fixture()
        with self.assertRaises(TypeError):
            workflow.evidence.retrieve(case, state_from_patient(case.patient, case.patient_id))
        for mock in (patients.retrieve, medical.retrieve, who.lookup):
            mock.assert_not_called()

    def test_patient_state_tampering_rejected(self):
        fixture = self.fixture()
        state = state_from_patient(fixture[1].patient, fixture[1].patient_id)
        state.antecedents.append('Hidden invented diagnosis')
        with self.assertRaises(ValueError):
            fixture[0].evidence.retrieve(fixture[1].patient, state)
        fixture[4].retrieve.assert_not_called()

    def test_query_bounds_keep_whole_facts_and_audit_omissions(self):
        state = Mock(presenting_evidence=['x' * 4097], symptoms=['literal symptom'], antecedents=['literal history'])
        query, omitted = who_query(state)
        self.assertEqual(query, 'literal symptom; literal history')
        self.assertEqual(omitted, ('patient:presenting_evidence:0',))

    def test_who_only_does_not_call_amg(self):
        fixture = self.fixture(mode='who_only')
        self.retrieve(fixture)
        fixture[4].retrieve.assert_not_called()
        fixture[4].query_for.assert_not_called()
        fixture[3].retrieve.assert_called_once()
        self.assertEqual(fixture[0].evidence.combined.status, 'WHO_ONLY')

    def test_amg_only_exact_snapshot_and_no_lookup(self):
        fixture = self.fixture(mode='amg_only')
        snapshot = self.retrieve(fixture)
        workflow, case, llm, patients, medical, who = fixture
        default = create_workflow(llm, patients, medical)
        expected = default.evidence.retrieve(case.patient, state_from_patient(case.patient, case.patient_id))
        self.assertEqual(snapshot, expected)
        who.lookup.assert_not_called()

    def test_rejected_amg_candidates_audit_only(self):
        fixture = self.fixture()
        fixture[4].audit = {'candidates': [{'id': 'rejected-only', 'acceptance': 'rejected'}]}
        snapshot = self.retrieve(fixture)
        self.assertNotIn('rejected-only', snapshot.model_dump_json())
        self.assertIn('rejected-only', fixture[0].evidence.combined.provenance_json)

    def test_rejected_amg_hit_still_fails(self):
        fixture = self.fixture()
        fixture[4].retrieve.return_value[0]['retrieval']['acceptance'] = 'rejected'
        with self.assertRaises(ValueError):
            self.retrieve(fixture)
        self.assertIsNone(fixture[0].evidence.combined)

    def test_who_tampering_rejected(self):
        hit = excerpt_hit(match())
        for field, value in (('text', 'Changed'), ('url', 'https://evil.example'), ('filename', '../evil.txt'),
                             ('char_start', 999), ('chunk_id', 'fabricated'), ('score', .9), ('PATHOLOGY', 'label')):
            altered = copy.deepcopy(hit)
            altered[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                evidence_from_hits([altered], 'medical_knowledge')

    def test_excerpt_is_bounded_not_summarized(self):
        source = match(body='Original test line.\n' * 1000)
        hit = excerpt_hit(source)
        self.assertLessEqual(len(hit['text']), 4000)
        self.assertLess(hit['char_end'], len(source.text))
        self.assertEqual(hit['text'], source.text[hit['char_start']:hit['char_end']])

    def test_selected_files_only_in_large_metadata_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            folder = root / 'who_fact_sheets'
            folder.mkdir()
            for i in range(569):
                topic = f'unique{i}'
                records.append({'title': topic, 'topic': topic, 'url': 'https://www.who.int/news-room/fact-sheets/detail/' + topic,
                                'file': topic + '.txt'})
            (folder / 'index.json').write_text(json.dumps(records), encoding='utf8')
            (root / 'who_questions_answers').mkdir()
            (root / 'who_questions_answers/index.json').write_text('[]', encoding='utf8')
            wanted = records[0]
            text = f"WHO FACT SHEET\nTitle: {wanted['title']}\nURL: {wanted['url']}\nRetrieved: date\n\nSoftware fixture only."
            (folder / wanted['file']).write_text(text, encoding='utf8')
            from rag.who_lookup import bounded_read
            with patch('rag.who_lookup.bounded_read', wraps=bounded_read) as reads:
                who = WHOLookup(root)
                result = who.lookup('unique0', max_results=3)
            self.assertEqual(who.document_count, 569)
            self.assertEqual(len(result.documents), 1)
            self.assertEqual([c.args[0].name for c in reads.call_args_list], ['index.json', 'index.json', 'unique0.txt'])

    def test_source_grouped_input_does_not_mutate_original(self):
        fixture = self.fixture()
        cases, medical = self.retrieve(fixture).agent_evidence()
        state = state_from_patient(fixture[1].patient, fixture[1].patient_id)
        original = context(state, cases, medical)
        saved = copy.deepcopy(original)
        result = separated_input(original)
        self.assertEqual(original, saved)
        groups = result['MEDICAL_KNOWLEDGE_EVIDENCE']
        self.assertEqual(len(groups['WHO']['fact_sheets']), 1)
        self.assertEqual(len(groups['WHO']['questions_answers']), 1)
        self.assertEqual(groups['AMG']['accepted_passages'], [medical[-1].model_dump()])

    def test_both_empty_abstains_no_diagnostic_or_fake_safety(self):
        workflow, case, llm, *_ = self.fixture(documents=(), empty_amg=True)
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'abstained')
        self.assertEqual(state.safety_skip_reason, 'no_medical_evidence')
        self.assertFalse(state.diagnostics)
        self.assertEqual(build_final_decision(state).status, 'HUMAN_REVIEW')

    def test_identical_who_citations_reach_critic_safety_and_final_handoff(self):
        fixture = self.fixture(mode='who_only')
        workflow, case, llm, *_ = fixture
        ref = 'medical:' + excerpt_hit(match())['chunk_id']
        llm.diagnoses[0].primary_hypothesis.rationale.evidence_refs = [ref]
        llm.diagnoses[0].medical_knowledge_evidence = [ref]
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'final')
        decision = build_final_decision(state)
        self.assertEqual(decision.grounding_result.medical_entailment, 'not_established')
        self.assertIn('reference_present_not_verified', decision.grounding_result.model_dump_json())
        for call in llm.calls:
            schema = call['text']['format']['schema']['title']
            if schema in ('DiagnosticResult', 'ClinicalCritique', 'SemanticSafetyResult'):
                self.assertIn(ref, call['input'][0]['content'])
            if schema in ('DiagnosticResult', 'ClinicalCritique'):
                self.assertIn('fact_sheets', call['input'][0]['content'])
                self.assertIn('accepted_passages', call['input'][0]['content'])
                self.assertIn(SOURCE_INSTRUCTIONS, call['instructions'])
        self.assertFalse(decision.human_approval)

    def test_block_and_human_review_remain_authoritative(self):
        flagged = critique()
        flagged.safety_flags = ['Software fixture block']
        for provider, expected in ((SafetyScenarioLLM(critiques=[flagged]), 'BLOCK'),
                                   (SafetyScenarioLLM(safety_results=[semantic('safety_ambiguity')]), 'HUMAN_REVIEW')):
            with self.subTest(expected=expected):
                workflow, case, *_ = self.fixture(provider=provider)
                state = workflow.run(case.patient, case.patient_id)
                decision = build_final_decision(state)
                self.assertEqual(decision.status, expected)
                self.assertIsNone(decision.diagnostic_proposal)
                self.assertEqual(decision.review_state, 'pending')

    def test_retrieval_failure_does_not_reuse_previous_evidence(self):
        fixture = self.fixture()
        self.retrieve(fixture)
        fixture[-1].lookup.side_effect = RuntimeError('sensitive detail')
        workflow, case, *_ = fixture
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'failed')
        self.assertIsNone(workflow.evidence.combined)
        self.assertNotIn('sensitive detail', state.model_dump_json())

    def test_same_experimental_prompts_across_modes(self):
        prompts = []
        for mode in ('amg_only', 'who_only', 'who_plus_amg'):
            workflow, case, llm, *_ = self.fixture(mode=mode)
            if mode == 'who_only':
                ref = 'medical:' + excerpt_hit(match())['chunk_id']
                llm.diagnoses[0].primary_hypothesis.rationale.evidence_refs = [ref]
                llm.diagnoses[0].medical_knowledge_evidence = [ref]
            workflow.run(case.patient, case.patient_id)
            prompts.append({c['text']['format']['schema']['title']: c['instructions'] for c in llm.calls})
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[1], prompts[2])

    def test_revision_reuses_combined_snapshot(self):
        fixture = self.fixture(provider=SafetyScenarioLLM(critiques=[critique('revision_required')]))
        workflow, case, _, patients, medical, who = fixture
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'unresolved')
        patients.retrieve.assert_called_once()
        medical.retrieve.assert_called_once()
        who.lookup.assert_called_once()
        self.assertTrue(all(a.ticket.evidence_snapshot_id == state.evidence.snapshot_id for a in state.safety_assessments))

    def test_unknown_who_citation_cannot_pass_grounding(self):
        workflow, case, llm, *_ = self.fixture()
        llm.diagnoses[0].primary_hypothesis.rationale.evidence_refs = ['medical:who:not-supplied']
        llm.diagnoses[0].medical_knowledge_evidence = ['medical:who:not-supplied']
        state = workflow.run(case.patient, case.patient_id)
        self.assertEqual(state.status, 'failed')
        self.assertEqual(build_final_decision(state).status, 'BLOCK')

    def test_percent_encoded_url_matches_existing_lookup_contract(self):
        document = match()
        source = replace(document.source, topic='caf' + chr(233), url=document.source.url.replace('fixture', 'caf%C3%A9'))
        text = document.text.replace(document.source.url, source.url)
        item = replace(document, source=source, text=text)
        self.assertEqual(excerpt_hit(item)['topic'], source.topic)

    def test_offline_verifier_checks_source_bytes_and_final_handoff(self):
        from scripts.verify_who_amg_run import verify
        workflow, case, *_ = self.fixture()
        state = workflow.run(case.patient, case.patient_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'sample_state.json'
            path.write_text(state.model_dump_json(), encoding='utf8')
            (root / 'sample_combined.json').write_text(workflow.evidence.combined.model_dump_json(), encoding='utf8')
            decision_path = root / 'sample_decision.json'
            decision_path.write_text(build_final_decision(state).model_dump_json(), encoding='utf8')
            for folder, kind in (('who_fact_sheets', 'fact_sheet'), ('who_questions_answers', 'question_answer')):
                (root / folder).mkdir()
                (root / folder / 'fixture.txt').write_text(match(kind).text, encoding='utf8', newline='')
            hit = amg_hit()
            native = root / 'knowledge/medlineplus_lab' / hit['retrieval']['snapshot']
            native.mkdir(parents=True)
            (native / 'chunks.jsonl').write_text(json.dumps({'id': hit['chunk_id'], 'text': hit['text'],
                'metadata': hit['amg_metadata']}) + '\n', encoding='utf8')
            with patch('scripts.verify_who_amg_run.VENDOR_ROOT', root):
                result = verify(path, data_dir=root)
                self.assertEqual(result['sources_verified'], {'WHO': 2, 'AMG': 1})
                (root / 'who_fact_sheets/fixture.txt').write_text('corrupt', encoding='utf8')
                with self.assertRaises(ValueError):
                    verify(path, data_dir=root)

    def test_invalid_modes_rejected(self):
        with self.assertRaises(ValueError):
            CombinedEvidenceService(Mock(), Mock(), mode='automatic_allow')
        with self.assertRaises(ValueError):
            CombinedEvidenceService(Mock(), mode='who_plus_amg')


if __name__ == '__main__':
    unittest.main()
