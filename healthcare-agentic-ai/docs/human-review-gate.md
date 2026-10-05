# Diagnostic audit and local Human Review Gate

## Part 1 — saved Diagnostic audit (completed before implementation)

Source: `outputs/phase6/20261005T133733712888Z/sample_case_001.json`, run `6b4f4216-7fe1-496d-b3c2-eeb2be33d0cf`, Diagnostic version 1. No live call. Native source/grounding/Safety/final-decision verification passed offline; exact Diagnostic fields and the two flags are retained in `outputs/human-review-gate/part1-exact-fields.json`.

### Flag 1: prominence of the possible bleeding-related concern

The Diagnostic places **“Current or recurrent symptomatic anemia”** first and **“Gastrointestinal blood loss with secondary anemia”** in the differential. Its exact bleeding rationale is:

> Uncertain model inference: black coal-like stools and new oral anticoagulant use are patient observations, not proof of gastrointestinal bleeding; their coexistence with pallor, disabling fatigue, and near-faintness permits a cautious bleeding-related anemia hypothesis.

The summary explicitly says:

> Reported black stools and new oral anticoagulant use raise an unconfirmed bleeding-related alternative, while chronic kidney failure provides another unconfirmed anemia context.

The Critic says the hierarchy/summary **“may underemphasize its safety significance.”** This is a prioritization judgment, not a missing-fact or fabricated-citation finding. Its Safety anchor is Critic `/safety_flags/0`, not an exact Diagnostic claim pointer. The related exact Diagnostic fields above are mapped from its wording, not from hidden model reasoning.

A possible summary revision, without asserting established GI bleeding:

> Safety-relevant uncertainty: reported black stools and oral anticoagulant use coexist with near-faintness, disabling fatigue and pallor. A possible bleeding-related process remains unconfirmed; no supplied medical passage establishes that mechanism.

### Flag 2: certainty of the primary hypothesis label

The exact label is **“Current or recurrent symptomatic anemia.”** The rationale ends **“This supports anemia only as an unconfirmed hypothesis.”** Uncertainty explicitly states **“No laboratory result confirms current anemia or establishes its severity or type.”**

The WHO Signs and symptoms passage supports nonspecific symptom overlap, not a current episode, recurrence, severity or cause. The patient reports previous anemia but supplies no current blood count. The bare label could sound too established; the complete output is already explicitly non-definitive. The Critic's concern is defensible, not proof that the output diagnosed confirmed anemia.

Smallest clearer label:

> Possible current anemia (unconfirmed)

### Reported versus examined pallor

The exact supporting claim says:

> Paler-than-usual skin is observed, and the supplied WHO excerpt includes pale skin among manifestations of severe anemia.

Input is a patient answer (`Is your skin much paler than usual? = Yes`), not an examination. Smallest accurate revision, preserving the existing patient and WHO references:

> The patient reports paler-than-usual skin; the WHO excerpt lists pale skin among possible manifestations of severe anemia, but this report does not establish anemia or its severity.

This wording problem is a related Critic contradiction, not the literal second Safety-flag anchor.

### Prompt/rule decision

**No Diagnostic prompt or rule changed.** Existing prompts already require non-definitive hypotheses, distinguish observations from model inference, prohibit invented examination/laboratory facts, and require explicit uncertainty. These saved-output wording improvements do not prove a new rule is necessary. The implementation below is **only the Human Review Gate**. Saved Diagnostic output, Critic/Safety findings and source evidence are not rewritten.

The two patient-only causal hypotheses still lack external medical support; human review is not a way to manufacture citations or to claim that a Safety BLOCK was incorrect. The detailed prior source audit remains in `docs/ddx-block-cause-audit.md`.


## Part 2 — implemented Human Review Gate

### Integration boundary

The new gate is local and synchronous, **after the completed AI/Safety decision**. Default batch (`run_phase6.py`), manual (`run_manual_who_amg.py`) and live demo entrypoints call it whenever the AI handoff requires review. The orchestrator, Diagnostic, Grounding, Critic, Safety, WHO/AMG and evidence gates are unchanged. Core/programmatic workflow calls still return their immutable AI result; terminal/UI callers invoke the new post-processing gate.

Original AI workflow and final-decision files are written before the batch/manual review starts. Human outcomes are recorded separately. This preserves both the Safety result and the exact version the human saw, instead of rewriting an AI BLOCK into an apparent AI approval.

### Interaction

In a local terminal, the gate displays:

- Case/run ID and current diagnostic version.
- The complete final AI proposal, structured claims/citations and saved public `reasoning_summary` (not private model reasoning).
- Patient facts, relevant findings, exact patient-reference inventory and original label-free input.
- **All** supplied medical passages, including uncited WHO/AMG material, plus the separate analogous-case evidence and provenance.
- Critic findings, Safety findings, the immutable AI final result, missing information, uncertainty, unsupported claims and missing-reference warnings.

The review display is complete JSON with escaped terminal controls, not the demo's shortened excerpt renderer. No evidence text is truncated or reinterpreted. A complete review context and original workflow snapshot are also saved locally.

It then asks exactly:

`Do you want to review this case? REVIEW / DECLINE:`

REVIEW displays the complete information again and asks:

`Explicit human decision: APPROVE / REJECT / REQUEST_MORE_EVIDENCE:`

There is no default choice. Case-insensitive spellings of those explicit words are accepted; `yes`, `continue`, approval at the first question, and free-form alternatives are not approvals. Invalid input is retried with a bounded ten-attempt limit per step; exhausted or blank input remains pending.

| Human action | Human-review record | Local case disposition | Original AI/Safety artifacts |
|---|---|---|---|
| DECLINE | Decline recorded; `review_state=pending`, no approval | BLOCKED | Unchanged |
| REVIEW without a second action | Pending; no approval | BLOCKED | Unchanged |
| REVIEW → APPROVE | Explicit approval recorded; `review_state=approved` | **Still BLOCKED/restricted** | Unchanged; never converted to ALLOW |
| REVIEW → REJECT | Rejection recorded; `review_state=rejected` | BLOCKED | Unchanged |
| REVIEW → REQUEST_MORE_EVIDENCE | Request recorded; `review_state=pending` | Unresolved/pending | Unchanged; no retrieval or rerun |
| Blank input, EOF, interruption, no terminal | No approval; pending event recorded | BLOCKED | Unchanged |
| State/source changes during review | `INVALIDATED`, no approval | BLOCKED/pending | No edited source data is accepted as the reviewed version |

**Approval is a recorded local human opinion, not a Safety override, clinical release, or claim of clinician authentication.** The original AI artifact may still say `review_state=pending` and `human_approval=false`: those fields describe the frozen AI disposition. The separate human record describes the actual human action. Callers must not conflate the two or infer approval from AI `ALLOW`/Safety `CONTINUE`.

A technical failure/abstention without a current validated Diagnostic proposal and assessed Safety result can be inspected, declined, rejected or marked for more evidence, but APPROVE is unavailable. Missing versions are recorded as `null`, never invented.

### No response, storage failures and local-only behavior

Both stdin and stdout must be interactive terminals. Redirected/piped input cannot approve; it records `NON_INTERACTIVE` and stays BLOCKED/pending. The saved-run command exits with status 2 for pending nonresponse/interruption/noninteractive/invalidation, 1 for load/persistence errors, and 0 for completed human choices or an AI result that does not require review. Status 0 is **not** clinical approval.

The gate persists an OFFERED/pending event before requesting input. If the process is killed while awaiting a response, the existing AI restriction and pending record remain. Partial/corrupt event files fail verification and cannot be interpreted as approval. Storage failure never produces a default human decision. No more-evidence request invokes a retriever, provider or rerun; subsequent cases in an explicitly requested batch are separate cases, not fulfillment of that request.

### Separate, version-bound audit

Each invocation creates a new directory `human_reviews/review-<session-id>/`; existing sessions/files are not overwritten:

- `workflow_snapshot.json`: complete original workflow audit, including prior versions, evidence, Critic/Safety records and invocation metadata.
- `review_context.json`: complete displayed review material.
- `event-0001.json`, `event-0002.json`, …: exclusive-created, fsynced event records with a previous-record SHA-256 chain.

Records include human action, UTC timestamp, local reviewer account label, case/run ID, reviewed Diagnostic version/fingerprint, workflow/context/evidence fingerprints and exact Safety-input fingerprint. Approval requires an explicit APPROVE event after REVIEW and successful full display. Records are frozen typed models; the verifier checks identity and event order as well as the hash chain.

A saved-case review also checks the original state file and companion AI final-decision file, where present, against their pre-review hashes before/after input. Mutated states or versions cannot silently inherit approval. A later review creates a distinct session and does not erase an earlier decline/rejection/approval.

These are **local integrity/audit records, not digitally signed clinical authorizations**. Local account identity is explicitly marked unverified. Someone with write access to all files can tamper with local history; cryptographic clinician authentication, access control, scheduling and clinical execution are not implemented by this task.

### Files added

- `application/human_review.py` — full local review context, interaction, typed records, persistence and offline session verification.
- `scripts/review_case.py` — review an existing saved result without models, retrieval or cloud calls; saved-input bound is 32 MiB per file.
- `tests/test_human_review.py` — 33 offline regression tests using synthetic fixtures.
- `docs/human-review-gate.md` — Part 1 audit and this implementation report.
- Workspace-root `HUMAN_REVIEW_RUN.txt` — plain-text command for the saved run, with no live pipeline rerun.
- Audit/probe/test artifacts under `outputs/human-review-gate/`.

### Existing files changed

- `scripts/run_phase6.py` — invoke the gate after saving the AI result; report the separate human record; clarify the report disclaimer.
- `scripts/run_manual_who_amg.py` — invoke the gate after saving/verifying the AI run; report the separate human record.
- `demo/terminal.py` — invoke the gate after displaying the AI final decision; display the separate human outcome.
- `README.md` — explain local review and the AI-versus-human record distinction.
- Project/workspace `okf.yaml` — synchronized context.

**No modifications** to Diagnostic prompts/behavior, WHO selection/validation/corpus, AMG, retrieval, Grounding, Critic, Safety, final-decision models/policy, evidence gates, benchmark data or saved source-run artifacts. Existing unrelated dirty work was preserved.

## Tests and offline verification

- **33 new offline tests** cover BLOCK offering; DECLINE; explicit two-step APPROVE; REJECT; REQUEST_MORE_EVIDENCE; EOF/blank/interrupt; invalid choices; noninteractive/piped input; hidden output; case/run/version/timestamp identity; full evidence and uncited-source display; original workflow/Safety preservation; multiple-session history; mutation/invalidation; no approval without an assessed proposal; no review for ALLOW; durable pending state; persistence failure; audit tampering; saved-case CLI and post-Safety batch integration.
- **113 focused tests passed**.
- **801 full project tests ran: 800 passed, one existing opt-in cloud skip**.
- The existing source verifier checked the requested saved run offline: two WHO passages, five AMG passages, one Diagnostic version, one Critic version, one Safety assessment and BLOCK/pending handoff.
- The new saved-case CLI was exercised against that run in this tool session, which has **no interactive terminal**. It correctly persisted OFFERED → NON_INTERACTIVE, BLOCK/pending, no approval. `verify_review_session` verified the resulting session against the unchanged source state. No human approval was supplied or simulated for the real saved run. APPROVE branches were exercised only with synthetic test fixtures.
- **No live cloud calls were made.** No new medical data, retrieval or model download was introduced.

Full tests used the workspace interpreter with the existing system site-packages appended for the previously documented BeautifulSoup environment gap. No dependency installation or dependency-file change was needed.

Logs and proofs: `focused-tests.log`, `regression-tests.log`, `saved-run-verification.json`, `saved-review.log`, `saved-review-verification.json`, `before-hashes.json`, and `integrity.json` under `outputs/human-review-gate/`.

## Run the local gate yourself — no cloud call

From the workspace root, use `HUMAN_REVIEW_RUN.txt` or enter:

```text
cd healthcare-agentic-ai
..\.venv\Scripts\python.exe scripts\review_case.py outputs\phase6\20261005T133733712888Z\sample_case_001.json
```

Use a real terminal, then type REVIEW or DECLINE. After REVIEW, type an explicit decision. Do not pipe an approval into the command. `--review-dir <folder>` selects an alternative audit directory; each session still gets a unique subdirectory. The command does not alter the existing result or rerun the AI pipeline.
