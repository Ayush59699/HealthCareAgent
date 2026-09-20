"""Post-inference evaluation only. Never imported by agents or inference pipeline."""
from .agents.models import PipelineResult
from .agents.grounding import claims, facts
from .models import EvaluationLabels


def evaluate_case(output: PipelineResult, labels: EvaluationLabels, *, top_k: int = 5) -> dict:
    if type(top_k) is not int or top_k < 1:
        raise ValueError('Evaluation top_k must be positive')
    diagnosis, critic = output.diagnosis, output.critique
    succeeded = output.status == 'success'
    normalize = lambda text: ' '.join(text.casefold().split())
    truth = labels.ground_truth_pathology
    predictions = []
    if diagnosis:
        for item in ([diagnosis.primary_hypothesis] if diagnosis.primary_hypothesis else []) + diagnosis.differential_diagnoses:
            if normalize(item.condition) not in predictions:
                predictions.append(normalize(item.condition))
    references = [ref for claim in claims(diagnosis) for ref in claim.evidence_refs] if diagnosis else []
    allowed = {e.source_id for e in output.patient_case_evidence + output.medical_knowledge_evidence}
    if output.patient_state:
        allowed |= set(facts(output.patient_state))
    valid = sum(ref in allowed for ref in references)
    measurable = bool(truth and predictions)
    correctness_note = None
    if diagnosis is None:
        correctness_note = 'Diagnostic Agent did not produce an accepted DiagnosticResult; diagnostic correctness was therefore not measurable.'
    elif not predictions:
        correctness_note = 'Accepted abstention; diagnosis correctness is not measurable.'
    return {'diagnostic_correctness_measurable': measurable,
            'accepted_primary_match': (normalize(diagnosis.primary_hypothesis.condition) == normalize(truth)) if measurable and diagnosis.primary_hypothesis else None,
            'accepted_top_k_match': normalize(truth) in predictions[:top_k] if measurable else None,
            'correctness_note': correctness_note,
            'valid_diagnostic_result': diagnosis is not None, 'valid_critique': critic is not None,
            'patient_id': output.patient_id, 'pipeline_success': output.status == 'success',
            'ground_truth': truth, 'top_k': top_k,
            'primary_match': (succeeded and bool(diagnosis and diagnosis.primary_hypothesis) and normalize(diagnosis.primary_hypothesis.condition) == normalize(truth)) if truth and diagnosis is not None else None,
            'top_k_match': (succeeded and normalize(truth) in predictions[:top_k]) if truth and diagnosis is not None else None,
            'citation_count': len(references), 'valid_citation_count': valid,
            'citation_provenance_coverage': valid / len(references) if references else None,
            'hallucinated_evidence_count': len(references) - valid,
            'model_reported_unsupported_claim_count': len(diagnosis.unsupported_claims) if diagnosis else None,
            'critic_unsupported_point_count': len(critic.unsupported_points) if critic else None,
            'critic_hallucination_flag_count': len(critic.hallucination_flags) if critic else None,
            'critic_flag_count': sum(len(getattr(critic, field)) for field in ('unsupported_points', 'contradictions', 'missing_evidence', 'hallucination_flags', 'safety_flags')) if critic else None}


def summarize(outputs: list[PipelineResult], cases: list[dict]) -> dict:
    labeled = [case for case in cases if case['primary_match'] is not None]
    valid = sum(sum(o.structured_validity.values()) for o in outputs)
    attempted = sum(len(o.structured_validity) for o in outputs)
    citations = sum(c['citation_count'] for c in cases)
    measurable = [c for c in cases if c.get('diagnostic_correctness_measurable')]
    return {'diagnostically_measurable_cases': len(measurable),
            'accepted_primary_accuracy': (sum(c['accepted_primary_match'] is True for c in measurable) / len(measurable)) if measurable else None,
            'accepted_top_k_accuracy': (sum(c['accepted_top_k_match'] is True for c in measurable) / len(measurable)) if measurable else None,
            'sample_size': len(outputs), 'labeled_sample_size': len(labeled),
            'pipeline_success_rate': sum(o.status == 'success' for o in outputs) / len(outputs) if outputs else None,
            'structured_validity': {'valid_agent_outputs': valid, 'attempted_agents': attempted,
                                    'rate': valid / attempted if attempted else None},
            'retrieval_success': {kind: {'nonempty': sum(o.retrieval_success.get(kind, False) for o in outputs),
                'attempted': sum(kind in o.retrieval_success for o in outputs),
                'rate': (sum(o.retrieval_success.get(kind, False) for o in outputs) / sum(kind in o.retrieval_success for o in outputs)) if any(kind in o.retrieval_success for o in outputs) else None} for kind in ('patient_case', 'medical_knowledge')},
            'primary_accuracy': sum(c['primary_match'] for c in labeled) / len(labeled) if labeled else None,
            'top_k_accuracy': sum(c['top_k_match'] for c in labeled) / len(labeled) if labeled else None,
            'citation_provenance_coverage': sum(c['valid_citation_count'] for c in cases) / citations if citations else None,
            'hallucinated_evidence_count': sum(c['hallucinated_evidence_count'] for c in cases),
            'critic_outputs_scored': sum(o.critique is not None for o in outputs),
            'diagnostic_outputs_scored': sum(o.diagnosis is not None for o in outputs),
            'critic_flag_count': sum(c['critic_flag_count'] or 0 for c in cases),
            'model_reported_unsupported_claim_count': sum(c['model_reported_unsupported_claim_count'] or 0 for c in cases),
            'limitations': ['Synthetic DDXPlus; small samples are not clinical validation.',
                'Exact case/whitespace-normalized label match only; no synonym or LLM judge.',
                'Legacy primary_accuracy/top_k_accuracy measure end-to-end yield among accepted diagnostic outputs (critic failures/abstentions count as misses); missing diagnoses are excluded. Use accepted_* metrics for diagnostic correctness independent of critic completion.',
                'Retrieval success means nonempty, not relevant.',
                'Citation coverage measures ID existence, not entailment. Rejected outputs are not scored as accepted claims.',
                'Unsupported and critic counts are model reports, not adjudicated hallucination rates.',
                'Similarity is not diagnostic probability; self-assessment is not calibrated clinical confidence.']}
