# Treatment / management layer

## Integration decision

The supported composition is now:

`Patient -> patient-case RAG + CHECK4 WHO + AMG -> Diagnostic -> Treatment -> Grounding -> Critic -> Safety -> Final Decision / Human Review`

Inspection found that Grounding, Critic, Safety fingerprints, final output restriction, and local Human Review already share a versioned `DiagnosticResult`. The smallest coherent handoff is therefore a **nested `diagnostic.treatment`**, not an unreviewed side-channel or a second final decision. Diagnostic fields and their evidence inventories are not rewritten. Management has its own exact citation inventories; final supporting evidence includes both inventories.

`application.workflow.create_workflow` enables a separate `TreatmentAgent` call by default. Historical/direct Phase 6 and experimental compositions retain their prior behavior unless `enable_treatment=True` is explicitly supplied. The supported demo uses the new composition and displays the treatment call separately.

## Generation and contracts

- Uses the same injected `StructuredLLM` / `OpenAIProvider` as the other agents: existing workspace `.env` loading, `GPT_SOL_API_KEY`, `GPT_SOL_ENDPOINT`, and the configured GPT-5-family deployment (`gpt-5.6-sol`). No credentials, new clients, model fallback, or environment changes were added.
- The treatment request contains the complete validated patient state (including findings, missingness, uncertainty), exact patient citation inventory, diagnostic hypotheses and reasoning, and the original source-separated WHO / AMG passages and analogous cases. No new retrieval occurs.
- `TreatmentPlan`: `status` (`proposed` or `deferred`), typed `actions`, exact case/medical citation inventories, nonempty `missing_information`, and nonempty `uncertainty`.
- Action categories: evaluation, monitoring, referral, supportive care, and treatment consideration. Each action contains a single `EvidenceClaim` for the proposal and brief justification.
- Proposed actions must cite both supplied patient facts and supplied medical evidence. Unknown citations, wrong inventories, fabricated URLs, blank proposals, and recognized patient-reference/polarity mismatches are rejected. These mechanical checks **do not establish clinical entailment**.
- Insufficient support means an explicit deferred plan with no actions and an explanation, not a fabricated recommendation. The prompt prohibits prescribing, dosing, starting/stopping medication, procedural instructions, false reassurance, or unsafe delay. Conditional clinician-facing research considerations are not clinical advice.
- The diagnostic prompt must return `treatment=null`; the independent management call owns that field. Legacy saved records may omit it. Null treatment is omitted from local serialization to preserve legacy diagnostic/workflow fingerprints; live JSON schemas still require every property.

## Gates, identity, and failure handling

The existing diagnosis validation runs before management consumes its input. The combined proposal then passes the mandatory Grounding gate. Management claims appear in the deterministic Grounding report, and the Critic explicitly reviews their relevance, prerequisites, uncertainty and safety. The existing Safety input includes the entire nested plan; existing JSON-pointer anchors can identify `/treatment/actions/0/proposal/statement` without changing Safety contracts or policy.

`TREATMENT` has its own Phase-6-only stage, immutable invocation ticket bound to the pre-management diagnostic fingerprint, transition events, validation records, timing, and normal provider repair/request/byte/time accounting. Each diagnostic revision gets a fresh management call over the same evidence snapshot. The committed combined fingerprint binds the plan to Critic and Safety; subsequent mutation fails identity checks.

Malformed or failed management generation terminates safely before Critic/Safety rather than continuing with a missing plan. Empty medical retrieval still takes the existing abstention path without calling Treatment. No budgets or safety thresholds were relaxed.

Final ALLOW output carries `diagnostic_proposal.treatment` for research inspection only. BLOCK and HUMAN_REVIEW withhold the entire proposal, including treatment. The existing local Human Review context automatically contains the complete nested plan and all evidence, with no change to approval or release rules. A deferred plan is not clinical reassurance or automatic approval; existing Critic/Safety/routing decisions still apply.

Retrieval algorithms, WHO selection/content review, AMG, Safety policy, and Human Review implementation are unchanged. Grounding and final evidence projection received only the extensions needed to cover management claims/citations.

## Verification

Run from `healthcare-agentic-ai`:

```text
python -m unittest tests.test_treatment tests.test_final_decision tests.test_human_review tests.test_check5_pipeline
python -m unittest discover -s tests -t .
```

Results in the available Python environment:

- **97 targeted tests passed**, including **17 new treatment tests**. Covers source/patient preservation, call order, source-separated WHO/AMG input, management-only citation retention, deferred output, independent Critic/Safety/Human Review handoffs, nested safety anchors, BLOCK/HUMAN_REVIEW withholding, serialization, fingerprints and tampering, validation even when an injected provider skips callbacks, request budgets/repairs, revisions, and no repeated retrieval.
- Full discovery: **778 entries: 775 passed, one opt-in cloud test skipped, two import errors**. The two WHO crawler test modules cannot import `bs4` (`beautifulsoup4` is absent in this environment). No test assertion failures. These dependencies were not installed or changed by this implementation.
- `git diff --check` passed.
- Tests use synthetic responses through the real structured provider parsing/validation path. No live cloud request, new medical retrieval, clinical accuracy evaluation, or treatment-safety validation is claimed.
