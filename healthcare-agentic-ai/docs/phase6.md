# Phase 6 — dedicated output-safety validation

**Synthetic-data research only. CONTINUE is not clinical clearance.** Semantic
review uses the same generation backend and is not independent clinical
adjudication. No clinical accuracy, safety, or effectiveness claim is made.
`HUMAN_REVIEW` is a terminal withheld-output marker: no review is scheduled,
performed, queued, or implemented. No HITL, resume/override API, GUI, Graph RAG,
feedback memory, prescribing, execution tools, or new diagnostic agent is added.

## Compatibility boundary

The Phase 5 runtime, output contracts, prompts, gates, tests, runner and defaults
are unchanged. Phase 6 has a separate `orchestration/phase6/` package and
`scripts/run_phase6.py`. It reuses the existing three agents, evidence services,
provider, diagnostic/critic version records, revision input and grounding helpers.

The Phase 5 run loop has nested helpers rather than an extension hook. Phase 6
therefore narrowly adapts the loop instead of refactoring Phase 5. Parity tests
compare original provider requests, routing outcomes, evidence, clinical outputs
and baseline audit ordering when safety permits continuation. Maintain these tests
when changing either implementation; shared-engine extraction is a separate task.

`Phase6WorkflowState` is a separate strict, frozen contract, not a subclass that
can accidentally pass as a Phase 5 state. It preserves baseline fields and their
meanings, with `phase6-workflow-v1` / `phase6-routing-v1` identifiers, new stage
literals, safety records and coverage. Historical Phase 5 artifacts are not
migrated or granted safety clearance. Consumers must dispatch by schema version.

## Flow and priorities

```text
INITIAL -> PATIENT -> EVIDENCE -> DIAGNOSTIC -> GROUNDING -> CRITIC
                                                           |
                                                   SAFETY_VALIDATION
                                                           |
                                                         ROUTE
                         CONTINUE -------------------------+--> HUMAN_REVIEW
                            |                              `--> BLOCKED
                       unchanged Phase 5 route
                 FINAL / ABSTENTION / UNRESOLVED / DIAGNOSTIC_REVISION
                                                      |
                                              GROUNDING -> CRITIC
                                                           |
                                                   SAFETY_VALIDATION
```

Any nonterminal stage may fail technically. Direct `CRITIC -> ROUTE` is forbidden
in Phase 6. The Orchestrator owns state, invocation tickets, application policy
and every transition. The Safety Validator cannot write state or choose a route.

Priority:

1. Technical, schema, provenance, identity, grounding or budget failure terminates
   as `failed / technical_failure`. No fallback safety pass is manufactured.
2. Any validated critic safety flag causes `blocked / safety_blocked`. Deterministic
   assessment records the block and **does not call the semantic validator**.
3. A completed safety BLOCK also terminates as `blocked / safety_blocked`.
4. HUMAN_REVIEW terminates as `human_review_required / human_review_required`.
5. CONTINUE invokes the unchanged Phase 5 `route()`; it does not automatically
   finalize. Critic insufficiency, revisions, unsupported claims and abstention
   retain their original priorities and meanings.

No medical evidence still abstains before diagnosis. Invalid diagnostic/critic
outputs still fail at their original gates. Exact repeated diagnostic content
still stops before another critic or safety call. These early terminal paths are
explicitly **not assessed**. A diagnostic abstention that reaches the critic is
reviewed, because its generated prose can still contain unsafe content.

Up to two clinical revisions remain permitted. Each new non-repeated version
requires its own safety assessment. Safety findings are not added to evidence,
critic input or `DiagnosticRevisionInput`; no new safety-revision loop exists.

## Contracts and ownership

- `SafetyInput`: detached validated PatientState, current DiagnosticResult, matching
  ClinicalCritique, separated evidence lists, and fixed policy/instructions. No
  labels, raw records, full state, histories, credentials, or execution authority.
- `SafetyTicket`: application run/patient/stage/revision/version IDs, diagnostic
  fingerprint, evidence snapshot ID, critic fingerprint, policy version and full
  input fingerprint. It is not sent to or authored by the LLM.
- `SemanticSafetyResult`: exactly ten categories, each with `no_issue_identified`,
  `issue_identified` or `uncertain`, bounded explanations and anchored findings.
  Explicit limitations are mandatory. No route, severity or confidence fields.
- `ContentAnchor`: typed diagnostic/critique target, RFC 6901 JSON pointer to a
  string leaf, and an exact nonempty excerpt. Uncertainty without a specific
  unsafe passage may have no anchor rather than manufacturing one.
- `SafetyAssessment`: application ticket, deterministic checks, semantic status,
  normalized findings, derived decision/reasons, policy/prompt version and time.

All provider objects are independently revalidated at the application boundary,
including fake or `model_construct()` objects. A separate validator checks safety
anchors, category coverage, duplicate findings, exact excerpts and allowed source
references; existing diagnostic/critic `validate_references()` is unchanged.

Before assessment commitment and routing, patient copy, evidence provenance,
diagnostic/critic grounding, current critique association and all fingerprints
are checked again. Policy is recomputed at consumption to reject forged decisions.
A prior-version assessment cannot authorize a revised result. Handoffs and returned
state are detached; nested legacy model mutability is not shared with the service.
JSON output is an audit artifact, not authorization to resume or skip gates.

## Versioned safety policy

Policy: `phase6-safety-v1`; prompt: `phase6-safety-prompt-v1`.

Deterministic checks cover contracts, identities, evidence integrity, existing
critic flags, required semantic coverage and policy aggregation. They do not
pretend to infer clinical entailment using text overlap or keyword lists.

A non-abstaining hypothesis lacking **any structured medical reference** in its
rationale/supporting/contradicting claims requests HUMAN_REVIEW. This is a new,
explicit Phase 6 restriction, not a modification of Phase 5 grounding and not a
claim that the hypothesis is false. Missing demographics alone do not block.

Semantic categories and policy for an **identified** issue:

| Category | Disposition |
| --- | --- |
| invented_patient_facts | HUMAN_REVIEW |
| unsupported_certainty | HUMAN_REVIEW |
| similarity_as_probability | HUMAN_REVIEW |
| analogy_as_outcome | HUMAN_REVIEW |
| source_misrepresentation | HUMAN_REVIEW |
| prohibited_action | BLOCK |
| unsafe_delay | BLOCK |
| diagnostic_critic_contradiction | HUMAN_REVIEW |
| instruction_following | BLOCK |
| safety_ambiguity | HUMAN_REVIEW |

**Uncertainty in any category always requests HUMAN_REVIEW**, not clearance and
not a categorical clinical judgment. Any BLOCK takes precedence over review.
Empty/unknown/duplicated categories or invented anchors are invalid output, not
uncertainty. Identified issues require anchored findings. Valid concerning results
are never retried merely to obtain a more permissive answer.

The prompt reviews all generated prose, including uncertainty, missing information,
reasoning summaries and critic recommendations. Negation, quotations, hypothetical
examples and medicine names in retrieved material are not automatically treated
as endorsed action instructions. Retrieved text remains untrusted data.

Reference validation establishes ID existence, not entailment. Exact quotation
establishes traceability, not correctness of the semantic interpretation. The tiny
corpus is not a comprehensive triage, dosing, contraindication or guideline engine.
No new clinical thresholds, emergency rules or outside sources are invented.

## Provider, budgets and telemetry

The same stateless OpenAI-compatible GPT-5.6-Sol provider is reused. No new model,
API settings, tools or dependency is added. Required structured output/grounding
repairs remain bounded; refusals, incomplete output and transport failures do not
retry. Temperature remains omitted with deployment-default sampling, not a claimed
deterministic model response. Input context is sent to the configured cloud service.

Retained defaults: 21 requests, 7,000,000 cumulative request bytes and 900 seconds.
Phase 6 permits up to ten logical invocations (1 patient + 3 diagnoses + 3 critiques
+ 3 semantic assessments). The current two-attempt provider default permits at
most 20 requests for that path; a three-attempt provider needs up to 30. Smaller
budgets may deliberately stop before all revisions. Limits are never automatically
increased, and safety is never skipped for lack of budget.

Reservations use the configured provider attempt and byte limits. Phase 6 also
requires well-formed per-attempt request-byte telemetry; incomplete or invalid
accounting fails closed. This strengthens only Phase 6, not Phase 5. Rejected repair
attempts consume request/byte totals and are not clinical revisions.

Deadlines remain cooperative between synchronous calls, not hard cancellation.
Time is checked after deterministic work, provider returns and before terminal
routing acceptance. An in-flight call may exceed the deadline; its output is not
accepted afterward. Safety durations include input reconstruction, deterministic
checks, semantic calls, result validation and assessment policy work. Common
routing/final serialization overhead is not a provider duration.

## Outputs and CLI

From the project directory, with the existing indexes and dedicated GPT_SOL setup:

```text
python scripts/run_phase6.py --queries 1 --top-k 1
python scripts/run_phase6.py --queries 1 --top-k 1 --max-requests 30
python -m unittest discover -s tests -t . -p "test_safety*.py" -v
python -m unittest discover -s tests -t . -p "test_phase6*.py" -v
```

From this Windows parent workspace use `.venv\Scripts\python.exe` and prefix
script/test paths with `healthcare-agentic-ai/`. These CLI examples perform cloud
calls when run; offline tests use fakes and require no API call.

The runner uses only label-free validation records and existing indexes. It has
no evaluator import. A new `outputs/phase6/<timestamp>/` contains atomic per-case
state and `phase6_run_report.json`; an existing output directory is not overwritten.

Reports include current diagnostic/safety identity, latest-version coverage, skip
reason, semantic status, decision, category counts, safety requests/repairs/time,
policy/prompt versions and existing terminal status/outcome. Coverage refers to the current diagnostic attempt: a failed revision may leave an
older committed diagnostic in history, but its previous CONTINUE is not reported
as permission for the failed/current attempt. Match version identifiers explicitly.
`eligible_research_output` is true only for `final` plus matching CONTINUE, and
still does **not** mean clinical approval. Every report says clinical approval is
false and human review is not scheduled.

Exit zero means technical research completion, including blocked, abstained,
unresolved and human-review-required cases. Inspect status/outcome and `withheld`.
Setup failure, technical failure or fewer cases than requested exits nonzero.
Accepted clinical text and detailed findings remain sensitive research artifacts;
console summaries/audit events omit model prose, rejected output and raw errors.

## Remaining limits

No clinical effectiveness evidence or independent human adjudication exists.
Model-generated clean category declarations are not proof of complete assessment.
Same-backend review can share the diagnostic model's blind spots. Malicious input
is constrained by contracts, not proven immune to semantic prompt injection.
A safety block can be a false positive. No automated human workflow exists to
resolve it. Live Phase 6 generation is unvalidated unless separately documented.
See [validation](phase6-validation.md) for actual offline execution evidence.
