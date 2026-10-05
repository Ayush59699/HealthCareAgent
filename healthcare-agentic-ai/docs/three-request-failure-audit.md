# Three-request DDXPlus failure: exact saved-response audit

## Run identified

`outputs/phase6/20261005T131137964626Z/sample_case_001.json`

Run ID `d4cfeeeb-cc11-4333-a34d-54fe4bd196de`; patient `ddxplus:validate:1`. This is the newer three-request batch failure, not the earlier CHECK5 four-request debug run or the later successful five-call manual workflow.

| Request | Operation | Provider response | Application validation |
|---|---|---|---|
| 1 | Patient Agent | API success, completed | Accepted; exact patient copy validated |
| 2 | WHO catalog selection | API success, completed | Accepted |
| 3 | WHO full-content relevance review (`ContentRelevance`) | API success, completed | Rejected: `agent_validation_failure` / `nonliteral_patient_quote` |

One attempt per request, no repairs. No missing-data setup failure, timeout, refusal, incomplete response or client context truncation is recorded. Diagnostic was not reached; AMG retrieval had not run.

## What the model returned

The retained structured response is copied losslessly at the field/value level to:

`outputs/three-request-audit/returned-content-review.json`

```json
{
  "verdict": "relevant",
  "reason": "distinctive_positive_cluster",
  "positive_fact_quotes": [
    "Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? = Yes",
    "Do you feel slightly dizzy or lightheaded? = Yes",
    "Do you feel pain somewhere? = back of head",
    "Is your skin much paler than usual? = Yes"
  ],
  "content_quotes": [
    "Anaemia causes symptoms such as fatigue, reduced physical work capacity, and shortness of breath.",
    "- dizziness or feeling light-headed",
    "- headache,”,"
  ]
}
```

This is the **saved parsed object**, not the original HTTP envelope or original JSON formatting. Those raw response bytes were not persisted. The failed path also did not persist the accepted catalog ID list or the exact content-request document/payload; those cannot be reconstructed with certainty from this trace.

## Root cause: patient representation mismatch, not invalid JSON/schema

Offline validation of this object against `ContentRelevance` **passes**. Provider code parses/validates the schema before calling the custom validator (`rag/llm/provider.py:158–160`). It emits `agent_validation_failure`, rather than `parsing_failure`, when the schema is valid but the custom validator rejects it (`:168–174`).

The failing comparison is `rag/who_selector_relevance.py:65–69`:

```python
literals = [str(e.value) for group in (patient.symptoms, patient.antecedents,
                                      patient.initial_evidence) for e in group]
if any(not any(q in literal for literal in literals) for q in value.positive_fact_quotes):
    raise ValueError('Patient support is not literal input')
```

But the actual model input uses `Evidence.text`, i.e. `question = value` (`rag/models.py:14–15,26–34`). **All four returned patient quotes exactly equal rendered facts supplied to the model. None matches a bare `Evidence.value`.** Their matching bare values are `Yes`, `Yes`, `back of head`, and `Yes`.

For example:

- Supplied and quoted: `Do you feel slightly dizzy or lightheaded? = Yes`
- Validator searches only: `Yes`
- Result: deterministic rejection despite the exact fact being present in the input.

The CHECK4 prompt asks for literal values, which worked with manual prose-valued observations but is a poor contract for DDXPlus question/answer facts. Merely asking the model to quote `Yes` is not an adequate fix: it loses clinical meaning and fails the existing minimum four-character quote rule. The representation/validation contract needs alignment, not missing dataset files.

The orchestration trace marks `schema_valid: false` for this failure because its generic failure path does not distinguish custom grounding rejection from parsing failure. The provider failure classification, retained typed object and offline replay establish the more specific cause.

## A second issue visible in the response

The returned source quote `- headache,”,` is not the local WHO Anaemia source line, which is exactly `- headache` (`data/who_fact_sheets/anaemia.txt:56`). The other two source quotes occur literally in that local document.

This is a **candidate-source comparison**, not proof of the exact document transmitted: its identity/request payload is absent from the failed audit. The observed rejection occurred at the earlier patient-quote check, so source-quote validation was not reached. If this local Anaemia document was the supplied source, fixing patient representation alone would expose a second exact-source-quote failure. Do not normalize away or silently accept the added punctuation.

## Next correction indicated (not implemented by this audit)

Align patient grounding with canonical full question-and-answer facts or explicit fact references, preserving negation/uncertainty and rejecting fabricated/recombined facts. Do not accept bare `Yes` as independent clinical support. Retain strict source quotation/provenance checks. Persist the chosen document identity and request schema with failed content reviews so the exact source request can be audited without another cloud call.

## Verification and scope

`outputs/three-request-audit/replay.py` revalidates the saved workflow and returned schema, checks all four quotes against both representations, and deterministically reproduces `Patient support is not literal input`. `audit.json` records the saved-run hash, request telemetry, quote comparisons and source-comparison limitations.

**No new cloud request, production-code change, data edit or full regression run.** Only this audit/report and OKF entries were added. The existing saved trace was not modified.
