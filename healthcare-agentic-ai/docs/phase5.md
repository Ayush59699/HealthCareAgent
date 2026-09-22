# Phase 5 — deterministic research orchestration

**Research only. This is not clinical approval, clinical advice, or evidence of
clinical effectiveness. Human-in-the-loop review is NOT implemented.** A Critic
assessment is model-generated feedback, not independent clinical adjudication.

## Reconciliation with Phase 4

Before implementation, the existing agents, Pydantic contracts, grounding helpers,
provider repairs, RAG interfaces, runner, evaluation boundary and tests were
inspected. Phase 4 already supplied isolated agents, exact-copy PatientState
validation, evidence/provenance validation, strict schema parsing, grounding and
bounded provider repair. These were reused rather than redesigned.

`rag/phase4.py` and `scripts/run_phase4.py` remain byte-for-byte unchanged as the
sequential control. The only existing runtime file changed is
`rag/agents/clinical.py`: the **same DiagnosticAgent** accepts an optional, typed
revision input. Without that argument its Phase 4 payload and prompt are unchanged.
The provider, grounding rules, retrievers, stores, embeddings, corpus and evaluator
were not modified. No dependencies or agent framework were added.

## Architecture

```text
Orchestrator owns a run-local WorkflowState
  INITIAL -> PATIENT -> EVIDENCE (both existing RAGs, once)
          -> DIAGNOSTIC -> GROUNDING -> CRITIC -> ROUTE
                                                   |-- FINAL
                                                   |-- ABSTENTION
                                                   |-- BLOCKED
                                                   |-- UNRESOLVED
                                                   `-- DIAGNOSTIC_REVISION
                                                       -> GROUNDING -> CRITIC -> ROUTE
Any nonterminal stage -> TERMINAL_FAILURE on technical/contract/budget failure
```

- `orchestration/state.py`: versioned state, accepted output histories, validation
  outcomes and per-invocation telemetry.
- `evidence.py`: thin adapter around existing retrievers and `evidence_from_hits`.
- `contracts.py`: application-authored invocation tickets and completion envelopes.
- `events.py`: stages and structured audit events.
- `routing.py`: pure routing decisions and explicit transition allowlist.
- `policy.py`: two-revision policy, cumulative request/byte/time limits.
- `orchestrator.py`: sole state writer, restricted handoffs and mandatory gates.
- `rag/agents/revision.py`: the existing Diagnostic Agent's structured revision input.
- `scripts/run_phase5.py`: separate runner using the existing GPT-5.6-Sol provider.

There is no retrieval-revision loop, shared chat history, new clinical agent,
feedback memory, HITL, prescribing, tool execution or fallback model. Future stage
executors can use typed tickets/completions and add explicit policy transitions;
no placeholders for future agents have been added.

## WorkflowState and ownership

`phase5-workflow-v1` records a UUID run ID, parser patient ID, original label-free
`PatientRepresentation`, validated PatientState, frozen evidence, versioned
accepted diagnoses and critiques, current stage, transitions, revision count,
policy, validation results, sanitized failure and terminal outcome. It also
records timestamps, stage durations, request counts/bytes, provider repairs and
allowlisted telemetry. Policy version is `phase5-routing-v1`.

State is local to `run()`, not shared on the Orchestrator instance. Top-level fields
are frozen; only the Orchestrator replaces state. Existing mutable agent models
are retained for compatibility, but every handoff is a deep copy and the returned
terminal state is detached. Agents never receive WorkflowState or its histories.
There is no resume API; persisted JSON is an audit artifact, not an authorization
to restart or skip stages.

Only an exact `PatientRepresentation` is accepted at entry, via the existing
`patient_input` boundary. Whole PatientRecord objects and raw rows are rejected
before state creation or calls. The runner always uses `include_labels=False` and
never imports the evaluator or loads EvaluationLabels. Hypothesis names are model
outputs, not ground-truth labels. Agent-returned objects are independently
revalidated before commitment, including injected/fake provider objects.

## Frozen evidence

Each retriever is called once, with the existing deterministic patient text and
same top-k. Patient cases and medical knowledge remain separate. Original text,
source IDs, source, title, similarity and complete provenance are retained.
Similarity is never treated as diagnostic probability.

`phase5-evidence-v1` uses immutable tuples of frozen evidence items. Nested metadata
is stored as lossless JSON text to avoid the shallow immutability of a frozen model
containing dictionaries. `agent_evidence()` reconstructs fresh typed objects and
checks the original provenance validators and content hash on each handoff.
The snapshot ID is SHA-256 of canonical JSON for both separated evidence lists.
It fingerprints evidence content, not the entire underlying index.

Critic feedback and prior diagnoses are **never retrieved evidence**. They appear
only in the structured revision input. No new query is issued after critique.
Missing medical evidence causes explicit insufficient-evidence abstention before
Diagnostic Agent invocation; missing patient analogies alone do not block a case
with medical evidence. Nonempty retrieval is not proof of sufficient relevance.

## Revision and version contracts

Initial invocation receives PatientState and the two frozen evidence inventories.
Revision invocation additionally receives exactly:

- previous accepted DiagnosticResult;
- validated ClinicalCritique for that result;
- revision number (1 or 2);
- fixed constraints: feedback is not evidence, no new patient facts/sources,
  preserve abstention and grounding, return a complete result rather than a patch.

Every diagnostic output passes strict schema validation and the **unchanged**
`validate_references` gate before it enters history. The provider also runs its
existing validators during structured generation. The application gate is not
optional even if a provider claims its result was valid. Critiques are schema and
reference validated before routing. Rejected content and raw response text are
not saved to workflow state.

Diagnostic versions are 1 (initial), 2 and 3. Every invocation has a ticket with
run/patient/stage/revision, target diagnostic version and evidence snapshot ID.
Critic tickets additionally identify the exact diagnostic content hash. Revision
tickets identify the preceding diagnostic hash. Tickets are created by the
application, not requested from the LLM. Completion tickets must match exactly;
revision consumption checks the stored critique association again. A delayed or
stale completion cannot be committed. A failed/repeated revision may leave a
latest diagnostic with no corresponding critique: consumers must match version
IDs, never blindly zip the last diagnosis and last critique.

## Deterministic routing (priority order)

| Condition | Outcome |
| --- | --- |
| Provider refusal/incomplete/error, exhausted repairs, invalid schema/provenance/grounding, stale completion or budget failure | `failed / technical_failure` |
| No medical retrieval | `abstained / insufficient_evidence`, no manufactured diagnosis |
| Any validated Critic safety flags | `blocked / safety_blocked`, even if otherwise supported |
| Critic says insufficient evidence | `abstained / insufficient_evidence` |
| Revision requested, unsupported points, contradictions, hallucination flags, recommended revisions, or diagnostic unsupported claims | Revise if budget remains; otherwise `unresolved / max_revisions_reached` |
| Otherwise supported result has null primary and empty differential | `abstained / diagnostic_agent_abstention` |
| Otherwise supported result | `final / accepted_with_limitations` (research status only) |

Missing-evidence notes alone can remain limitations on an otherwise supported
result; explicit insufficient-evidence assessment takes the abstention route.
The LLM cannot pick transitions, change limits, bypass grounding or override
safety flags. Grounding failure terminates after the existing bounded provider
repair contract; it is not routed to the Critic or treated as a clinical revision.

Abstention preserves Phase 4: null primary plus empty differential requires empty
structured patient and medical evidence inventories. A valid abstention can be
reviewed and revised, but no hypothesis is manufactured to satisfy a Critic.

## Limits, progress and telemetry

`max_revisions = 2`: initial diagnosis plus **at most two clinical revisions**.
With the three existing roles this is at most seven agent invocations. Provider
schema/grounding repairs are separately counted (`attempts`, `provider_repairs`)
and do not increment clinical revisions. Transport failures, refusals and
incomplete output retain the provider's no-retry behavior.

Exact repeated diagnostic content hashes stop before another Critic call:
adjacent duplicates use reason `unchanged_diagnostic`, older duplicates use
`repeated_diagnostic`; both are `unresolved / unresolved_critic`. This is exact
structured-content detection, not a semantic-equivalence or clinical-progress
metric. Paraphrases remain bounded by the two-revision limit.

Default cumulative limits are 21 provider requests, 7,000,000 request bytes and
900 seconds per case. Before each invocation the Orchestrator conservatively
reserves **all possible provider repair attempts** and their configured maximum
input bytes. This can stop before every last unit is consumed. Completed requests,
including rejected/repair attempts, count toward totals. Providers injected through
the protocol must honor their declared bounded attempts and report telemetry;
for providers without public config the reserve is three attempts / 100,000 bytes
each. The production provider remains unchanged.

Elapsed time is checked before calls, after synchronous stages and before routing.
This is a **cooperative workflow deadline**, not cancellation of an in-flight
synchronous request. A single ongoing retrieval/provider call may exceed it;
provider per-request timeout remains independently enforced. No further agent is
called and no over-deadline output is accepted after that call returns. Durable
execution, hard process cancellation and parallel scheduling are not implemented.

Audit events include workflow/patient/evidence starts and completions, diagnostic
start/schema validation, grounding start/pass, Critic completion, route and
revision decisions, finalization, abstention, unresolved and termination. Events
contain ordered sequence numbers, UTC timestamp, source/target stages, revision,
run/patient/version/evidence IDs, machine reasons and validation results. Arbitrary
exception messages, rejected output and provider prose are omitted. Failure codes
and telemetry keys/scalars are allowlisted. Existing patient/evidence content in
accepted state is still sensitive research data; protect output files accordingly.

## Running and validation

From the project directory, with the same prebuilt indexes, cached BGE embeddings
and dedicated GPT_SOL settings used by Phase 4:

```text
python scripts/run_phase4.py --queries 1 --top-k 1   # unchanged baseline
python scripts/run_phase5.py --queries 1 --top-k 1 --max-requests 14 --max-seconds 600
python -m unittest discover -s tests -t . -p "test_orchestration*.py" -v
python -m unittest discover -s tests -t . -v
```

The Phase 5 runner writes atomic per-case WorkflowState JSON and a summary report
to a new `outputs/phase5/<timestamp>/` directory. It exits nonzero for setup or
technical failure/missing requested cases. A valid abstained, unresolved or blocked
run is technical completion (exit zero), **not approval**; inspect status/outcome.
Temperature handling is unchanged: the deployment rejects the parameter, so
sampling is deployment-default. Deterministic **routing** does not imply
deterministic LLM output.

See [Phase 5 validation report](phase5-validation.md) for tests, smoke result,
source/index integrity and limitations. These tests establish software contracts,
not diagnostic accuracy, clinical safety or comparative effectiveness.
