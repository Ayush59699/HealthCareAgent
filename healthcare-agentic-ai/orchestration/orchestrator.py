"""Application-owned deterministic orchestration around the unchanged Phase 4 gates."""
import copy
import math
import time
from uuid import uuid4
from rag.models import PatientRepresentation
from rag.agents.clinical import PatientAgent, DiagnosticAgent, ClinicalCritic
from rag.agents.models import PatientState, DiagnosticResult, ClinicalCritique, Failure
from rag.agents.grounding import patient_input, validate_patient, validate_references, context
from rag.llm.provider import StructuredLLM
from .contracts import StageTicket, StageResponse, require_current, DiagnosticRevisionInput
from .evidence import EvidenceService, EvidenceSnapshot, fingerprint
from .events import WorkflowEvent, timestamp
from .policy import WorkflowPolicy, WorkflowStop, SAFE_PROVIDER_CODES, SAFE_WORKFLOW_CODES
from .routing import route, require_transition
from .state import WorkflowState, DiagnosticVersion, CriticVersion, ValidationOutcome, InvocationTelemetry


def safe_observations(observations):
    """Only allowlisted scalar telemetry, never raw output, errors or arbitrary strings."""
    cleaned = []
    numeric = {'attempt', 'request_bytes', 'request_seconds', 'input_tokens', 'output_tokens',
               'total_tokens', 'http_status'}
    boolean = {'api_success', 'accepted', 'client_context_truncated', 'temperature_sent'}
    for item in observations:
        result = {}
        for key in numeric:
            value = item.get(key)
            if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                result[key] = value
        for key in boolean:
            if type(item.get(key)) is bool:
                result[key] = item[key]
        if item.get('failure_code') in SAFE_PROVIDER_CODES:
            result['failure_code'] = item['failure_code']
        if item.get('response_status') in {'completed', 'incomplete', 'failed'}:
            result['response_status'] = item['response_status']
        cleaned.append(result)
    return cleaned


class Orchestrator:
    def __init__(self, llm: StructuredLLM, patient_rag, medical_rag, *, top_k=1,
                 policy: WorkflowPolicy | None = None, clock=time.perf_counter):
        self.patient = PatientAgent(llm)
        self.diagnostic = DiagnosticAgent(llm)
        self.critic = ClinicalCritic(llm)
        self.evidence = EvidenceService(patient_rag, medical_rag, top_k=top_k)
        self.policy = policy or WorkflowPolicy()
        self.clock = clock
        config = getattr(llm, 'config', None)
        self.attempt_reserve = getattr(config, 'max_retries', self.policy.provider_attempt_limit - 1) + 1
        self.byte_reserve = getattr(config, 'max_input_bytes', 100_000) * self.attempt_reserve
        if type(self.attempt_reserve) is not int or not 1 <= self.attempt_reserve <= 3:
            raise ValueError('Provider must have bounded structured repair attempts')

    @staticmethod
    def _invoke(ticket, operation):
        # Synchronous adapter owns the envelope, never the LLM. Future executors
        # must return the original ticket; every completion is checked on receipt.
        return StageResponse(ticket=ticket, generation=operation())

    def run(self, patient: PatientRepresentation, patient_id: str) -> WorkflowState:
        supplied = patient_input(patient, patient_id)  # Reject labels before calls/state creation.
        started = self.clock()
        state = WorkflowState(run_id=str(uuid4()), patient_id=patient_id,
            original_patient=copy.deepcopy(patient), started_at=timestamp(), policy=self.policy)

        def commit(**changes):
            nonlocal state
            state = state.model_copy(update=changes)

        def event(name, reason, *, stage=None, valid=None, version=None):
            previous = state.current_stage
            target = stage or previous
            if target != previous:
                require_transition(previous, target)
            entry = WorkflowEvent(sequence=len(state.transition_history) + 1, timestamp=timestamp(),
                event=name, from_stage=previous, stage=target, revision_number=state.revision_count,
                reason=reason, validation_result=valid, run_id=state.run_id, patient_id=patient_id,
                diagnostic_version=version,
                evidence_snapshot_id=state.evidence.snapshot_id if state.evidence else None)
            commit(current_stage=target, transition_history=state.transition_history + (entry,))

        def budget(*, invocation=False):
            if self.clock() - started >= self.policy.max_seconds:
                raise WorkflowStop('time_budget_exhausted')
            if state.total_requests > self.policy.max_requests or state.total_request_bytes > self.policy.max_request_bytes:
                raise WorkflowStop('request_budget_exhausted')
            if invocation and (state.total_requests + self.attempt_reserve > self.policy.max_requests or
                               state.total_request_bytes + self.byte_reserve > self.policy.max_request_bytes):
                raise WorkflowStop('request_budget_exhausted')

        def timed(name, operation):
            before = self.clock()
            try:
                return operation()
            finally:
                durations = dict(state.stage_seconds)
                durations[name] = durations.get(name, 0.0) + max(0.0, self.clock() - before)
                commit(stage_seconds=durations)

        def ticket(version=None, diagnostic_hash=None):
            return StageTicket(run_id=state.run_id, patient_id=patient_id, stage=state.current_stage,
                revision_number=state.revision_count, diagnostic_version=version,
                evidence_snapshot_id=state.evidence.snapshot_id if state.evidence else None,
                diagnostic_fingerprint=diagnostic_hash)

        def validation(schema, grounding, reason):
            entry = ValidationOutcome(stage=state.current_stage, revision_number=state.revision_count,
                                      schema_valid=schema, grounding_valid=grounding, reason=reason)
            commit(validations=state.validations + (entry,))

        def generate(expected, operation, schema):
            budget(invocation=True)
            before = self.clock()
            response = timed(expected.stage, lambda: self._invoke(expected, operation))
            generation = response.generation
            observations = safe_observations(generation.telemetry)
            attempts = generation.attempts
            if type(attempts) is not int or not 0 <= attempts <= self.attempt_reserve:
                raise WorkflowStop('invalid_provider_telemetry')
            byte_count = sum(int(o.get('request_bytes', 0)) for o in observations)
            invocation = InvocationTelemetry(ticket=expected, attempts=attempts,
                provider_repairs=max(0, attempts - 1), seconds=max(0.0, self.clock() - before),
                request_bytes=byte_count, observations=observations)
            commit(invocations=state.invocations + (invocation,), total_requests=state.total_requests + attempts,
                   provider_repairs=state.provider_repairs + invocation.provider_repairs,
                   total_request_bytes=state.total_request_bytes + byte_count)
            require_current(expected, response.ticket)
            if generation.failure or generation.parsed is None:
                code = generation.failure.code if generation.failure else 'invalid_output'
                code = code if code in SAFE_PROVIDER_CODES else 'provider_failure'
                validation(False, None, code)
                raise WorkflowStop(code)
            budget()
            try:
                # Revalidate even model_construct/fake-provider objects, not merely isinstance.
                if type(generation.parsed) is not schema:
                    raise ValueError('Wrong output contract')
                parsed = schema.model_validate(generation.parsed.model_dump(warnings=False))
            except (ValueError, TypeError):
                validation(False, None, 'schema_validation_failure')
                raise WorkflowStop('schema_validation_failure') from None
            validation(True, None, 'schema_passed')
            return parsed

        def ground(output):
            try:
                cases, medical = state.evidence.agent_evidence()
                context(state.patient_state, cases, medical)
                validate_references(output, state.patient_state, cases, medical)
            except (ValueError, TypeError):
                validation(True, False, 'grounding_failure')
                raise WorkflowStop('grounding_failure') from None
            validation(True, True, 'grounding_passed')

        def finish(stage, outcome, reason):
            status, name = {'FINAL': ('final', 'finalized'), 'ABSTENTION': ('abstained', 'abstained'),
                'UNRESOLVED': ('unresolved', 'unresolved'), 'BLOCKED': ('blocked', 'terminated'),
                'TERMINAL_FAILURE': ('failed', 'terminated')}[stage]
            event(name, reason, stage=stage, valid=False if status == 'failed' else None,
                  version=state.diagnostics[-1].version if state.diagnostics else None)
            commit(status=status, outcome=outcome, completed_at=timestamp(), seconds=max(0.0, self.clock() - started))

        event('workflow_started', 'label_free_input_accepted')
        try:
            event('patient_started', 'initial_patient', stage='PATIENT')
            patient_ticket = ticket()
            accepted = generate(patient_ticket, lambda: self.patient.run(copy.deepcopy(patient), patient_id), PatientState)
            try:
                validate_patient(accepted, supplied)
            except (ValueError, TypeError):
                validation(True, False, 'patient_validation_failure')
                raise WorkflowStop('patient_validation_failure') from None
            validation(True, True, 'patient_validated')
            commit(patient_state=accepted)
            event('patient_agent_completed', 'patient_validated', valid=True)
            budget()
            event('evidence_started', 'retrieve_once', stage='EVIDENCE')
            try:
                snapshot = timed('EVIDENCE', lambda: self.evidence.retrieve(copy.deepcopy(patient)))
                snapshot = EvidenceSnapshot.model_validate(snapshot.model_dump())
            except Exception:
                raise WorkflowStop('invalid_evidence_or_retrieval_failure') from None
            commit(evidence=snapshot)
            event('evidence_retrieved', 'provenance_validated', valid=True)
            budget()
            if not snapshot.medical_knowledge:
                finish('ABSTENTION', 'insufficient_evidence', 'no_medical_evidence')
                return state.model_copy(deep=True)

            for revision in range(self.policy.max_revisions + 1):
                version = revision + 1
                if revision == 0:
                    event('diagnostic_started', 'initial_diagnosis', stage='DIAGNOSTIC', version=version)
                else:
                    commit(revision_count=revision)
                    event('diagnostic_revision_started', 'validated_critic_feedback', stage='DIAGNOSTIC_REVISION', version=version)
                revision_input = None
                previous_hash = None
                if revision:
                    previous, review = state.diagnostics[-1], state.critiques[-1]
                    # Reject stale feedback at consumption as well as at completion.
                    require_current(ticket(previous.version, previous.fingerprint).model_copy(update={
                        'stage': 'CRITIC', 'revision_number': revision - 1}), review.ticket)
                    if review.diagnostic_version != previous.version or review.evidence_snapshot_id != snapshot.snapshot_id:
                        raise WorkflowStop('stale_result')
                    previous_hash = previous.fingerprint
                    revision_input = DiagnosticRevisionInput(previous_diagnostic=previous.result.model_copy(deep=True),
                        critique=review.result.model_copy(deep=True), revision_number=revision)
                diagnostic_ticket = ticket(version, previous_hash)
                cases, medical = snapshot.agent_evidence()
                kwargs = {'revision': revision_input} if revision_input else {}
                diagnosis = generate(diagnostic_ticket, lambda: self.diagnostic.run(
                    state.patient_state.model_copy(deep=True), cases, medical, **kwargs), DiagnosticResult)
                event('diagnostic_validated', 'schema_passed', valid=True, version=version)
                event('grounding_started', 'mandatory_grounding_gate', stage='GROUNDING', version=version)
                timed('GROUNDING', lambda: ground(diagnosis))
                digest = fingerprint(diagnosis.model_dump())
                prior_hashes = [d.fingerprint for d in state.diagnostics]
                commit(diagnostics=state.diagnostics + (DiagnosticVersion(ticket=diagnostic_ticket,
                    version=version, fingerprint=digest, result=diagnosis),))
                event('grounding_passed', 'diagnosis_committed', valid=True, version=version)
                if digest in prior_hashes:
                    finish('UNRESOLVED', 'unresolved_critic',
                           'unchanged_diagnostic' if digest == prior_hashes[-1] else 'repeated_diagnostic')
                    break
                budget()
                event('critic_started', 'review_exact_version', stage='CRITIC', version=version)
                critic_ticket = ticket(version, digest)
                cases, medical = snapshot.agent_evidence()
                critique = generate(critic_ticket, lambda: self.critic.run(state.patient_state.model_copy(deep=True),
                    diagnosis.model_copy(deep=True), cases, medical), ClinicalCritique)
                ground(critique)
                commit(critiques=state.critiques + (CriticVersion(ticket=critic_ticket, diagnostic_version=version,
                    evidence_snapshot_id=snapshot.snapshot_id, result=critique),))
                event('critic_completed', 'critique_validated', valid=True, version=version)
                budget()
                event('route_started', 'deterministic_policy', stage='ROUTE', version=version)
                decision = route(diagnosis, critique, revision, self.policy.max_revisions)
                if decision.next_stage != 'DIAGNOSTIC_REVISION':
                    finish(decision.next_stage, decision.outcome, decision.reason)
                    break
                event('revision_requested', decision.reason, version=version)
        except Exception as exc:
            # Never persist raw exceptions, model output, transport text or repair prompts.
            code = exc.code if isinstance(exc, WorkflowStop) and exc.code in SAFE_WORKFLOW_CODES else 'stage_failure'
            commit(failure=Failure(code=code, message='Workflow terminated safely; inspect local contracts and configuration.',
                                   attempts=state.invocations[-1].attempts if state.invocations else 0))
            finish('TERMINAL_FAILURE', 'technical_failure', code)
        return state.model_copy(deep=True)
