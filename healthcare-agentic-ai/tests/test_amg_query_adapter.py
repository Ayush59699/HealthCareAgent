"""Focused, offline controlled-query tests; no stored model/data needed."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from rag.agents.models import PatientState
from rag.amg.alias_safety import CaseSafeAliases
from rag.amg.backend import AMGMedicalEvidence, load_upstream
from rag.amg.query_adapter import build_query, compact_question, patient_facts


class Tokenizer:
    def encode(self, text, **kwargs):
        return [0] + text.split() + [1]


def state(**kwargs):
    return PatientState(**{**dict(patient_id='ddxplus:validate:1', age=40, sex='F',
        presenting_evidence=[], symptoms=[], antecedents=[], relevant_findings=[],
        missing_information=[], uncertainty_notes=[]), **kwargs})


class QueryAdapterTests(unittest.TestCase):
    def test_shortening_is_syntactic_not_diagnostic(self):
        self.assertEqual(compact_question('Do you have a cough?'), 'a cough')
        self.assertEqual(compact_question('Have you ever had a fracture?'), 'ever had a fracture')
        self.assertEqual(compact_question('Have you recently had chills?'), 'recently had chills')
        self.assertEqual(compact_question('What color is the rash?'), 'What color is the rash')

    def test_negation_uncertainty_answers_and_qualifiers_are_literal(self):
        facts = ['Do you have a cough? = No', 'Do you feel cold? = Unknown',
                 'Have you recently had chills? = Not applicable',
                 'Do you have swelling when standing but not sitting? = Yes',
                 'How severe is the itching? = 1']
        result = build_query(state(symptoms=facts), Tokenizer())
        for fact in facts:
            answer = fact.partition(' = ')[2]
            self.assertIn(' = ' + answer, result.query)
        self.assertIn('recently had chills', result.query)
        self.assertIn('when standing but not sitting', result.query)
        self.assertIn('How severe is the itching = 1', result.query)
        self.assertNotIn('mild', result.query)
        self.assertEqual(result.omitted, ())

    def test_roles_and_full_patient_are_preserved(self):
        patient = state(presenting_evidence=['Do you have a cough? = Yes'],
            symptoms=['Do you have a cough? = Yes', 'Do you feel cold? = Yes'],
            antecedents=['Have you ever had a fracture? = Yes',
                         'Do you have any family members with asthma? = Yes',
                         'Are you taking a new medication? = No'])
        before = patient.model_dump_json()
        result = build_query(patient, Tokenizer())
        self.assertEqual(before, patient.model_dump_json())
        for role in ('Presenting:', 'Symptoms:', 'History:', 'Medications:', 'Context:'):
            self.assertIn(role, result.query)
        self.assertIn('any family members with asthma = Yes', result.query)
        self.assertIn('ever had a fracture = Yes', result.query)
        self.assertIn('taking a new medication = No', result.query)
        initial = result.included[0]
        self.assertEqual(initial.refs, ('patient:presenting_evidence:0', 'patient:symptoms:0'))
        self.assertEqual(result.query.count('a cough'), 1)

    def test_presenting_medication_fact_keeps_presenting_role(self):
        patient = state(presenting_evidence=['Are you taking a medication? = Yes'])
        self.assertEqual(patient_facts(patient)[0].role, 'Presenting')

    def test_only_identical_original_questions_share_a_heading(self):
        patient = state(symptoms=['Do you feel pain somewhere? = left arm',
                                 'Do you feel pain somewhere? = right arm',
                                 'Do you have pain somewhere? = No'])
        result = build_query(patient, Tokenizer())
        self.assertIn('pain somewhere = left arm | right arm', result.query)
        self.assertIn('pain somewhere = No', result.query)
        self.assertEqual(result.query.count('pain somewhere'), 2)
        self.assertEqual(len([f for f in result.included if f.role == 'Symptoms']), 3)

    def test_no_separator_and_unknown_wording_are_not_inferred(self):
        patient = state(symptoms=['Unstructured observation, uncertain?', 'Unfamiliar question? = 0'])
        result = build_query(patient, Tokenizer())
        self.assertIn('Unstructured observation, uncertain?', result.query)
        self.assertIn('Unfamiliar question = 0', result.query)

    def test_unanswered_question_does_not_swallow_explicit_answer(self):
        patient = state(symptoms=['Do you feel cold?', 'Do you feel cold? = No'])
        result = build_query(patient, Tokenizer())
        self.assertIn('Do you feel cold?', result.query)
        self.assertIn('cold = No', result.query)

    def test_round_robin_covers_history_before_late_symptoms(self):
        patient = state(presenting_evidence=['first'], symptoms=['s1', 's2', 's3', 's4'],
                        antecedents=['h1', 'h2', 'Are you taking medication? = Yes'])
        result = build_query(patient, Tokenizer(), max_tokens=13)
        self.assertLessEqual(result.token_count, 13)
        self.assertIn('History: h1', result.query)
        self.assertTrue(result.omitted)
        self.assertEqual(set(result.included) | set(result.omitted), set(patient_facts(patient)))
        self.assertFalse(set(result.included) & set(result.omitted))

    def test_no_partial_fact_and_later_short_fact_can_fit(self):
        long_fact = 'Do you have ' + 'long ' * 300 + '? = No'
        result = build_query(state(symptoms=[long_fact, 'Do you feel cold? = Yes']), Tokenizer(), 30)
        self.assertIn(long_fact, [f.text for f in result.omitted])
        self.assertNotIn('long', result.query)
        self.assertIn('cold = Yes', result.query)

    def test_character_limit_is_independent_of_token_limit(self):
        result = build_query(state(symptoms=['x' * 4001]), Tokenizer())
        self.assertLessEqual(len(result.query), 4000)
        self.assertEqual(result.omitted[0].text, 'x' * 4001)

    def test_budget_counts_special_tokens_and_is_deterministic(self):
        patient = state(symptoms=['one', 'two', 'three'])
        result = build_query(patient, Tokenizer(), 10)
        self.assertEqual(result.token_count, len(Tokenizer().encode(result.query)))
        self.assertEqual(result.audit(), build_query(patient, Tokenizer(), 10).audit())
        self.assertEqual(build_query(patient, Tokenizer(), 2).query, '')

    def test_unknown_demographics_not_assumed(self):
        result = build_query(state(age=None, sex=None), Tokenizer())
        self.assertIn('age = unknown', result.query)
        self.assertIn('sex = unknown', result.query)

    def test_raw_rows_and_labels_rejected(self):
        for patient in ({'PATHOLOGY': 'not allowed'}, SimpleNamespace(symptoms=[])):
            with self.assertRaises(TypeError):
                build_query(patient, Tokenizer())
        for budget in (True, 0, 257, 3.5):
            with self.assertRaises(ValueError):
                build_query(state(), Tokenizer(), budget)

    def test_production_query_path_remains_literal(self):
        patient = state(symptoms=['Do you have a cough? = Yes'])
        backend = AMGMedicalEvidence.__new__(AMGMedicalEvidence)
        backend.retriever = SimpleNamespace(embedder=SimpleNamespace(tokenizer=Tokenizer(), max_tokens=256))
        self.assertEqual(backend.query_for(patient), ('Do you have a cough? = Yes', 0))


class AliasSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = load_upstream()

    def setUp(self):
        self.raw = self.up.AliasIndex([
            dict(id='1', title='Acute Lymphocytic Leukemia', aliases=['ALL'], see_references=[]),
            dict(id='2', title='Sample Topic', aliases=['ZZZ'], see_references=[]),
            dict(id='3', title='All Topics', aliases=[], see_references=[]),
        ])
        self.safe = CaseSafeAliases(self.raw)

    def test_exact_demonstrated_bed_all_day_failure(self):
        query = 'Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? = Yes'
        self.assertIn('Acute Lymphocytic Leukemia', self.raw.resolve(query)['expanded_query'])
        result = self.safe.resolve(query)
        self.assertEqual(result['expanded_query'], query)
        self.assertFalse(result['matches'])
        self.assertFalse(result['unambiguous_lookup'])
        self.assertEqual(result['suppressed_ordinary_all_matches'], 1)

    def test_lowercase_or_sentence_initial_all_cannot_bypass_gate(self):
        for query in ('all', 'All', 'What is all?', 'Define All', 'all day', 'All symptoms started yesterday'):
            with self.subTest(query=query):
                result = self.safe.resolve(query)
                self.assertFalse(result['exact_topic_ids'])
                self.assertFalse(result['unambiguous_lookup'])
                self.assertEqual(result['expanded_query'], query)

    def test_explicit_uppercase_acronym_still_supported(self):
        for query in ('ALL', 'What is ALL?', 'history of ALL'):
            self.assertEqual(self.safe.resolve(query), self.raw.resolve(query))

    def test_other_aliases_unicode_boundaries_longest_matches_unchanged(self):
        for query in ('zzz', 'What is ZZZ?', 'allergy', 'small', 'All Topics', 'Acute Lymphocytic Leukemia'):
            self.assertEqual(self.safe.resolve(query), self.raw.resolve(query))
        self.assertEqual(self.safe.resolve('ＡＬＬ'), self.raw.resolve('ＡＬＬ'))
        self.assertEqual(self.safe.resolve('ａｌｌ')['expanded_query'], 'ａｌｌ')

    def test_other_expansions_survive_and_queries_do_not_mutate_resolver(self):
        before = copy.deepcopy(self.raw.index)
        result = self.safe.resolve('all day with zzz')
        self.assertEqual(result['expanded_query'], 'all day with zzz\nSource topic: Sample Topic')
        self.assertEqual(self.raw.index, before)
        self.assertEqual(self.safe.resolve('ALL'), self.raw.resolve('ALL'))
        self.assertEqual(self.safe.resolve('all')['expanded_query'], 'all')

    def test_mixed_all_and_explicit_ALL_retains_one_valid_acronym(self):
        result = self.safe.resolve('all day with ALL')
        self.assertEqual(len([m for m in result['matches'] if m['alias'] == 'all']), 1)
        self.assertIn('Source topic: Acute Lymphocytic Leukemia', result['expanded_query'])

    def test_uppercase_word_consumed_by_longer_alias_is_not_an_acronym(self):
        result = self.safe.resolve('ALL TOPICS all day')
        self.assertNotIn('Acute Lymphocytic Leukemia', result['expanded_query'])
        self.assertEqual(result['matches'], [dict(alias='all topics', topic_ids=['3'])])

    def test_real_search_embeds_guarded_text_and_gate_still_abstains(self):
        collection = Mock()
        collection.count.return_value = 1
        collection.query.return_value = dict(ids=[['chunk']], documents=[['source']],
            metadatas=[[dict(title='Acute Lymphocytic Leukemia', url='https://medlineplus.gov/', topic_id='1')]],
            distances=[[1.11]])
        embedder = SimpleNamespace(tokenizer=Tokenizer(), max_tokens=256, embed_documents=Mock(return_value=[[0.0]]))
        retriever = self.up.Retriever(collection, embedder, [])
        retriever.aliases = self.safe
        result = retriever.search('all', top_k=5, max_distance=1.10)
        embedder.embed_documents.assert_called_once_with(['all'])
        self.assertEqual(result['status'], 'insufficient_evidence')
        self.assertEqual(result['results'], [])
        self.assertNotIn('where', collection.query.call_args.kwargs)
        self.assertEqual(result['resolution']['embedding_query'], 'all')

    def test_raw_query_and_model_budget_are_not_rewritten_by_guard(self):
        embedder = SimpleNamespace(tokenizer=Tokenizer(), max_tokens=5)
        retriever = self.up.Retriever(Mock(), embedder, [])
        retriever.aliases = self.safe
        with self.assertRaises(ValueError):
            retriever.search('one two three four five all')


class BaselineCompressionTests(unittest.TestCase):
    def compress(self, patient, budget=256, tokenizer=None):
        from rag.amg.baseline_compression import compress_baseline
        tokenizer = tokenizer or Tokenizer()
        backend = AMGMedicalEvidence.__new__(AMGMedicalEvidence)
        backend.retriever = SimpleNamespace(embedder=SimpleNamespace(tokenizer=tokenizer, max_tokens=budget))
        baseline, _ = backend.query_for(patient)
        return compress_baseline(patient, baseline, tokenizer, budget)

    def test_every_baseline_fact_is_mandatory_before_additions(self):
        patient = state(presenting_evidence=['Do you have a cough? = Yes'],
                        symptoms=['Do you feel cold? = No', 'Do you have chest pain? = Unknown'],
                        antecedents=['Have you ever had a fracture? = Yes'])
        result = self.compress(patient, 27)
        self.assertTrue(result.newly_added)
        self.assertFalse(result.baseline_facts_lost)
        self.assertTrue(set(result.baseline_included) <= set(result.included))
        self.assertIn('chest pain = Unknown', result.query)
        self.assertIn('ever had a fracture = Yes', result.query)

    def test_zero_loss_for_many_budgets_without_role_balancing(self):
        patient = state(symptoms=['Do you have a cough? = Yes', 'Do you feel cold? = No',
                                 'Have you recently had a rash? = Yes', 'Do you feel a racing heart? = Yes'],
                        antecedents=['Have you ever had a fracture? = Yes',
                                     'Are you taking a new medication? = Yes'])
        for budget in range(2, 80):
            with self.subTest(budget=budget):
                result = self.compress(patient, budget)
                self.assertEqual(result.baseline_facts_lost, ())
                self.assertTrue(set(result.baseline_included) <= set(result.included))
                self.assertLessEqual(result.token_count, budget)
                self.assertFalse(set(result.included) & set(result.omitted))
                self.assertEqual(len(result.included) + len(result.omitted), 6)

    def test_deterministic_and_complete_state_immutable(self):
        patient = state(symptoms=['Do you have a cough? = No'], relevant_findings=['Do you have a cough? = No'],
                        missing_information=['antecedents'], uncertainty_notes=['Unknown findings'])
        before = patient.model_dump_json()
        first = self.compress(patient)
        self.assertEqual(first.audit(), self.compress(patient).audit())
        self.assertEqual(before, patient.model_dump_json())

    def test_exact_answers_negation_qualifiers_and_scales(self):
        patient = state(symptoms=['Do you feel slightly dizzy or lightheaded? = Yes',
                                 'Have you recently had chills? = No',
                                 'Do you have swelling when standing but not sitting? = Unknown',
                                 'How severe is the itching? = 0', 'Unfamiliar question? = Not applicable'],
                        antecedents=['Have you ever had a fracture? = Yes',
                                     'Do you have any family members who have asthma? = No'])
        result = self.compress(patient)
        for text in ('slightly dizzy or lightheaded = Yes', 'recently had chills = No',
                     'when standing but not sitting = Unknown', 'How severe is the itching = 0',
                     'Unfamiliar question = Not applicable', 'ever had a fracture = Yes',
                     'any family members who have asthma = No'):
            self.assertIn(text, result.query)
        self.assertFalse(result.omitted)
        self.assertNotIn('mild', result.query)

    def test_duplicates_keep_all_original_references(self):
        fact = 'Do you have a cough? = Yes'
        result = self.compress(state(presenting_evidence=[fact], symptoms=[fact, fact]))
        self.assertEqual(len(result.included), 1)
        self.assertEqual(result.included[0].refs,
                         ('patient:presenting_evidence:0', 'patient:symptoms:0', 'patient:symptoms:1'))
        self.assertEqual(result.query.count('a cough'), 1)

    def test_group_only_identical_original_questions_and_keep_unknown(self):
        patient = state(symptoms=['Do you feel pain somewhere? = left arm',
                                 'Do you feel pain somewhere? = right arm',
                                 'Do you have pain somewhere? = No', 'Do you feel pain somewhere?'])
        result = self.compress(patient)
        self.assertIn('pain somewhere = left arm | right arm', result.query)
        self.assertIn('pain somewhere = No', result.query)
        self.assertIn('Do you feel pain somewhere?', result.query)
        self.assertEqual(len(result.baseline_included), 4)

    def test_no_generated_headers_or_demographics(self):
        patient = state(symptoms=['Do you have a cough? = Yes'],
                        antecedents=['Are you taking a new medication? = No'])
        result = self.compress(patient)
        for word in ('Presenting:', 'Symptoms:', 'History:', 'Medications:', 'Context:', 'age =', 'sex ='):
            self.assertNotIn(word, result.query)
        self.assertEqual(len(result.included), 2)
        self.assertEqual(result.included[1].role, 'Medications')  # audit only

    def test_do_not_delete_patient_words_that_resemble_metadata(self):
        result = self.compress(state(symptoms=['Do you have symptoms after sex? = No']))
        self.assertIn('symptoms after sex = No', result.query)

    def test_exact_baseline_validation_rejects_foreign_or_truncated_queries(self):
        from rag.amg.baseline_compression import compress_baseline
        patient = state(symptoms=['Do you have a cough? = No'])
        for query in ('', 'a cough = No', 'Do you have a cough? = Yes'):
            with self.assertRaises(ValueError):
                compress_baseline(patient, query, Tokenizer())

    def test_baseline_inventory_does_not_split_inside_patient_facts(self):
        fact = 'Do you have symptoms in the morning; not at night? = No'
        result = self.compress(state(symptoms=[fact, 'unstructured; literal = uncertain']))
        self.assertEqual([f.text for f in result.baseline_included], [fact, 'unstructured; literal = uncertain'])
        self.assertIn('in the morning; not at night = No', result.query)

    def test_omitted_antecedents_first_then_other_facts_no_evictions(self):
        patient = state(symptoms=['Do you have a cough? = Yes',
                                 'Do you have trouble with some ordinary activities? = No'],
                        antecedents=['Do you feel cold? = Yes'])
        result = self.compress(patient, 12)
        self.assertEqual([f.text for f in result.baseline_included], [patient.symptoms[0]])
        self.assertEqual([f.text for f in result.newly_added], [patient.antecedents[0]])
        self.assertEqual([f.text for f in result.omitted], [patient.symptoms[1]])

    def test_tokenizer_expansion_falls_back_to_exact_baseline(self):
        class ExpandingTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                return list(range(100)) if text.startswith('a cough') else super().encode(text, **kwargs)
        result = self.compress(state(symptoms=['Do you have a cough? = Yes']), 20, ExpandingTokenizer())
        self.assertEqual(result.query, result.baseline_query)
        self.assertTrue(result.fallback_reason)
        self.assertFalse(result.newly_added)
        self.assertFalse(result.baseline_facts_lost)

    def test_character_budget_keeps_whole_facts(self):
        result = self.compress(state(symptoms=['x' * 3990, 'y' * 4001]))
        self.assertEqual(result.query, 'x' * 3990)
        self.assertEqual(len(result.omitted), 1)
        self.assertFalse(result.baseline_facts_lost)

    def test_empty_input_no_metadata_only_retrieval(self):
        result = self.compress(state())
        self.assertEqual(result.query, '')
        self.assertEqual(result.token_count, 2)
        self.assertFalse(result.included)

    def test_strict_types_and_existing_window(self):
        from rag.amg.baseline_compression import compress_baseline
        with self.assertRaises(TypeError):
            compress_baseline({'PATHOLOGY': 'not allowed'}, '', Tokenizer())
        with self.assertRaises(TypeError):
            compress_baseline(state(), None, Tokenizer())
        for budget in (True, 0, 257, 3.2):
            with self.assertRaises(ValueError):
                compress_baseline(state(), '', Tokenizer(), budget)


class CompressionAliasDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.raw = load_upstream().AliasIndex([
            dict(id='1', title='Acute Lymphocytic Leukemia', aliases=['ALL'], see_references=[]),
            dict(id='2', title='Sexual Health', aliases=['sex'], see_references=[]),
            dict(id='3', title='Medicines', aliases=['medications'], see_references=[]),
        ])
        self.resolver = CaseSafeAliases(self.raw)

    def test_diagnostics_records_proposed_applied_and_suppressed_not_new_rules(self):
        from scripts.evaluate_amg_baseline_compression import alias_diagnostics
        query = 'sex all day'
        resolution = self.resolver.resolve(query)
        result = {'resolution': {**resolution, 'embedding_query': resolution['expanded_query']}}
        before = copy.deepcopy(result)
        diagnostics = alias_diagnostics(query, self.resolver, result)
        self.assertEqual(result, before)
        self.assertEqual(diagnostics['proposed_source_titles'], ['Sexual Health'])
        self.assertEqual(diagnostics['applied_source_titles'], ['Sexual Health'])
        self.assertEqual(diagnostics['suppressed_ordinary_all_matches'], 1)
        self.assertEqual(diagnostics['watched_matches'][0]['alias'], 'sex')

    def test_skipped_expansion_not_reported_as_embedded(self):
        from scripts.evaluate_amg_baseline_compression import alias_diagnostics
        query = 'medications'
        resolution = {**self.resolver.resolve(query), 'expanded_query': query,
                      'embedding_query': query, 'expansion_skipped': 'Would exceed embedding window'}
        diagnostics = alias_diagnostics(query, self.resolver, {'resolution': resolution})
        self.assertEqual(diagnostics['proposed_source_titles'], ['Medicines'])
        self.assertEqual(diagnostics['applied_source_titles'], [])
        self.assertFalse(diagnostics['matches'][0]['expansion_applied'])

    def test_compressed_clinical_query_has_no_generated_metadata_aliases(self):
        from scripts.evaluate_amg_baseline_compression import alias_diagnostics
        result = BaselineCompressionTests().compress(state(symptoms=['Do you have a cough? = Yes']))
        resolution = self.resolver.resolve(result.query)
        diagnostics = alias_diagnostics(result.query, self.resolver,
            {'resolution': {**resolution, 'embedding_query': resolution['expanded_query']}})
        self.assertEqual(diagnostics['watched_matches'], [])
        self.assertEqual(diagnostics['applied_source_titles'], [])


if __name__ == '__main__':
    unittest.main()
