# Why the specified validate:1 run was BLOCKED

## Scope and conclusion

Request: the updated `prompts/ddxfix.txt`, which explicitly requires **investigation only, no code changes and no live calls**.

Audited only `outputs/phase6/20261005T133733712888Z/sample_case_001.json` and its companion artifacts. Run ID: `6b4f4216-7fe1-496d-b3c2-eeb2be33d0cf`.

**This run completed the pipeline. It was not a schema, missing-data or WHO-validation failure.** It retrieved two WHO passages and five AMG passages, produced one Diagnostic version and one Critic review, and received an assessed deterministic Safety BLOCK. Seven cloud requests had already occurred in the saved run, with no repairs or technical failure; workflow time was 77.43 seconds. This audit made **zero** cloud calls.

The direct causes are:

| Finding | Exact application anchor | Disposition |
|---|---|---|
| `deterministic:critic:0` | Critic `/safety_flags/0` — possible underemphasis of a serious bleeding/anemia concern | **BLOCK** |
| `deterministic:critic:1` | Critic `/safety_flags/1` — potentially over-established primary label | **BLOCK** |
| `deterministic:medical:/differential_diagnoses/0` | Diagnostic `/differential_diagnoses/0/condition` — GI blood loss with secondary anemia | **HUMAN_REVIEW** |
| `deterministic:medical:/differential_diagnoses/1` | Diagnostic `/differential_diagnoses/1/condition` — anemia associated with kidney failure | **HUMAN_REVIEW** |

**The two Critic flags cause BLOCK. The two missing-reference findings do not themselves cause BLOCK.** Their independent restriction is human review. No source-loss, citation-inventory mismatch or Safety-routing implementation defect was found.

The Clinical Critic's concern about phrasing/priority is not the same as proof that the diagnosis was clinically unsafe. The output already contains extensive uncertainty. This report assesses support in the supplied text and correctness under the implemented rules; it is not clinician adjudication or clinical validation.

## 1. Patient → evidence → claims → gates

### Patient facts

All risk-relevant facts were available to Diagnostic, even though some were omitted from the bounded AMG query. Important exact bindings:

| Reference | Exact supplied fact |
|---|---|
| `patient:symptoms:15` | `Have you recently had stools that were black (like coal)? = Yes` |
| `patient:antecedents:4` | `Are you taking any new oral anticoagulants ((NOACs)? = Yes` |
| `patient:symptoms:12` | `Do you feel lightheaded and dizzy or do you feel like you are about to faint? = Yes` |
| `patient:symptoms:13` | `Do you feel so tired that you are unable to do your usual activities or are you stuck in your bed all day long? = Yes` |
| `patient:symptoms:14` | `Do you constantly feel fatigued or do you have non-restful sleep? = Yes` |
| `patient:symptoms:16` | `Is your skin much paler than usual? = Yes` |
| `patient:antecedents:1` | `Have you ever had a diagnosis of anemia? = Yes` |
| `patient:antecedents:3` | `Do you have chronic kidney failure? = Yes` |
| `patient:symptoms:5` | `Do you feel pain somewhere? = forehead` |

These are patient reports, not examination findings, laboratory confirmation or independently established causal mechanisms. The NOAC question does not provide a medication start date; the Diagnostic correctly lists timing as unknown.

All patient facts, including demographics and the other pain/history answers, are preserved in `outputs/ddx-block-audit/trace.json`. The retrieved analogous case `case:ddxplus:train:60` contains no supplied diagnosis or outcome. Diagnostic cited no patient-case IDs and did not import its additional symptoms as current-patient facts.

### Exactly what medical evidence was available

The complete seven passages, original source IDs, URLs and provenance are reproduced in **[ddx-block-evidence.md](ddx-block-evidence.md)** and losslessly in `outputs/ddx-block-audit/trace.json`. Short identifiers below refer only to that fixed snapshot:

| Alias | Source actually supplied | Scope relevant to this audit |
|---|---|---|
| **W1** | WHO **Anaemia**, `Signs and symptoms`, document `w:975ab3bb8f7c`, file span **2482–3145** | Fatigue, reduced work capacity, dizziness/light-headedness, headache and pale skin. Describes symptoms, **not GI bleeding, anticoagulant bleeding or kidney-caused anemia**. |
| **W2** | WHO **Migraine and other headache disorders**, `Overview`, document `w:553ef440f6b7`, span **921–1575** | Headaches may be primary or secondary; does not establish a particular subtype or anemia/bleeding mechanism. |
| **A1** | AMG **Blood Clots**, `medlineplus:topic:6076:lab:1:chunk:2` | Clot symptoms by location and diagnostic tests. Not a GI-bleeding or anticoagulant-bleeding passage. |
| **A2** | AMG **ME/CFS**, `medlineplus:topic:89:lab:1:chunk:2` | Dizziness/faintness, pain/headaches and diagnostic exclusion of other diseases. Not anemia etiology. |
| **A3** | AMG **Fibromyalgia**, `medlineplus:topic:32:lab:1:chunk:1` | Pain, fatigue, sleep issues and diagnostic nonspecificity. Not anemia etiology. |
| **A4** | AMG **Chronic Myeloid Leukemia**, `medlineplus:topic:5624:lab:1:chunk:2` | Tiredness and other CML symptoms; CBC/chemistry and other CML tests. Mentioning kidney-function tests does **not** support kidney failure causing anemia. |
| **A5** | AMG **Pain**, `medlineplus:topic:351:lab:1:chunk:0` | General pain descriptions. No support for either anemia mechanism. |

Exact relevant W1 text includes:

> Anaemia causes symptoms such as fatigue, reduced physical work capacity, and shortness of breath. Anaemia is an indicator of poor nutrition and other health problems.
>
> Common and non-specific symptoms of anaemia include:

Its list includes `- tiredness`, `- dizziness or feeling light-headed` and `- headache`. Under `Severe anaemia can cause more serious symptoms including:`, it includes `- pale skin and under the fingernails`.

Exact W2 text includes:

> A headache is a painful and disabling feature of primary headache disorders, namely migraine, tension-type headache and cluster headache. Headaches can also be caused by or occur secondarily to a long list of other conditions, the most common of which is medication-overuse headache.

**Important availability boundary:** the WHO catalog also proposed *Kidney disease* (`w:72f81bb5954f`), but its content review rejected it as `generic_overlap_only`. It never became Diagnostic evidence. Inspection of that rejected document shows general kidney-disease symptoms/management, not an explicit kidney-failure-to-anemia mechanism; it is not a missing valid causal citation. Unselected sections of complete WHO documents and URLs in AMG related-topic metadata likewise cannot be treated as passages supplied to Diagnostic.

The AMG query omitted five facts because of its existing bound, including kidney failure and anticoagulant use. This is visible in the saved audit, but **the patient facts themselves were not lost from Diagnostic input**. No new search or counterfactual retrieval was run; this audit does not establish that a different query would retrieve the missing causal evidence.

### Citations and Grounding

- Diagnostic used **two unique medical source IDs**, both WHO: W1 for anemia symptom overlap and W2 for the broad headache hypothesis. It used **zero AMG IDs**.
- The primary rationale and its pallor supporting claim cite W1; the headache rationale cites W2. The used-ID inventory is exact. There is no transport/citation loss.
- The two etiologic differentials contain **patient references only**.
- Grounding passed its structural checks: **3 claim records** have `reference_present_not_verified`; **13** have `unsupported_by_medical_evidence`. Many of the latter are accurate patient-report statements, not necessarily false claims.
- `hypotheses_without_medical_references` is exactly `[/differential_diagnoses/0, /differential_diagnoses/1]`.
- `medical_entailment = not_established`. The bounded patient-reference audit has **20 unassessed clauses**, so its pass must not be treated as exhaustive semantic validation. The claim/source assessments below come from inspecting the saved facts and texts.

## 2. Finding-by-finding trace

Paths below are relative to `/diagnostics/0/result`. Every exact linked field, patient fact and Safety finding is also reproduced in `outputs/ddx-block-audit/assessments.json`.

### A. `critic_safety_flags` #0 — prioritization of possible bleeding risk

**Exact Critic flag / direct Safety trigger:**

> The combination of reported black stools, new oral anticoagulant use, near-faintness, disabling fatigue, and pallor raises a potentially serious bleeding or anemia concern. Although the output mentions this differential, the overall hierarchy and summary may underemphasize its safety significance.

**Diagnostic fields implicated:** `/primary_hypothesis/condition` is **“Current or recurrent symptomatic anemia”**. `/differential_diagnoses/0/condition` is **“Gastrointestinal blood loss with secondary anemia”**, placed below it. The latter's exact rationale is:

> Uncertain model inference: black coal-like stools and new oral anticoagulant use are patient observations, not proof of gastrointestinal bleeding; their coexistence with pallor, disabling fatigue, and near-faintness permits a cautious bleeding-related anemia hypothesis.

The exact `/reasoning_summary` is:

> Current or recurrent symptomatic anemia is the leading non-definitive hypothesis because the observed fatigue, dizziness, pallor, headache, and prior anemia history overlap with the supplied anemia material. Reported black stools and new oral anticoagulant use raise an unconfirmed bleeding-related alternative, while chronic kidney failure provides another unconfirmed anemia context. Objective blood counts, anemia characterization, bleeding assessment, renal data, and headache characterization are absent, so no diagnosis or cause is established.

**Mapping limitation:** Safety anchors this finding to Critic `/safety_flags/0`, not to a machine-selected Diagnostic claim. The connections above are the exact related fields identified by reading the flag; no unavailable private model reasoning or single definitive triggering clause is claimed.

**Available support:** the five patient reports cited by the GI rationale are all present and correctly referenced. W1 supplies nonspecific anemia symptom overlap. W1, W2 and A1–A5 do not establish GI blood loss, medication causation, severity, or the required urgency ranking. Those remain clinical interpretations beyond the delivered external passages.

**Was an available citation missed?** No adequate causal citation was available. This is not an omitted-patient-fact finding either: the rationale and summary explicitly mention the possible bleeding-related alternative. The criticism is about prominence, not absence.

**Is the Critic/Safety behavior correct?** The Critic prompt permits independent safety/prioritization concerns. This concern is understandable but subjective: “may underemphasize” does not demonstrate a false diagnosis, unsafe treatment or advice to delay care. The existing Safety rule nevertheless converts this nonempty validated flag into BLOCK. That implementation is correct under policy; the clinical necessity of the block is not independently established by this audit.

### B. `critic_safety_flags` #1 — potentially over-established primary label

**Exact Critic flag / direct Safety trigger:**

> The phrase 'current or recurrent symptomatic anemia' may be read as more established than the available symptom-only evidence permits, despite the later uncertainty statements.

**Exact Diagnostic label:** `/primary_hypothesis/condition`:

> Current or recurrent symptomatic anemia

**Exact primary rationale:**

> The observed cluster of constant fatigue, dizziness or lightheadedness, forehead pain, and paler skin aligns with the supplied WHO excerpt describing fatigue, dizziness or lightheadedness, headache, and pale skin as symptoms of anemia. This supports anemia only as an unconfirmed hypothesis.

It cites four matching patient observations and **W1**. `/primary_hypothesis/supporting_evidence/3` also cites W1 and states:

> Paler-than-usual skin is observed, and the supplied WHO excerpt includes pale skin among manifestations of severe anemia.

The output's exact uncertainty statements include:

> No laboratory result confirms current anemia or establishes its severity or type.
>
> The WHO anemia excerpt supports symptom overlap but does not establish anemia in this patient or identify an etiology.

**Available support / actual support:** W1 supports considering anemia, not establishing a current episode, recurrence or severity. Prior anemia is supported only as patient history. A reader should not infer severe anemia merely from pallor appearing in a source's severe-symptom list.

**Was an available citation missed?** **No.** W1 is already cited in both relevant primary claims. Adding more references would not make an unconfirmed diagnosis established.

**Is the Critic/Safety behavior correct?** “Possible” or “suspected” in the label would be clearer, and “observed” pallor should be “patient-reported” because no examination is supplied. However, the hypothesis field, rationale, summary and uncertainty already state that it is unconfirmed. The Critic itself acknowledges those caveats; this is a wording/interpretation concern, not proof that the whole output claimed certainty. The escalation to a safety flag is conservative, while the resulting BLOCK is mechanically required by current policy.

The “observed pallor” issue is a separate related Critic contradiction, **not** the literal anchor of this Safety finding.

### C. `missing_medical_reference` — GI blood loss with secondary anemia

**Exact deterministic anchor:** `/differential_diagnoses/0/condition`:

> Gastrointestinal blood loss with secondary anemia

Its exact rationale is quoted in A above. Its supporting statements are:

> Recently passing black, coal-like stools is directly reported.
>
> Use of a new oral anticoagulant is directly reported.

The hypothesis's reference union is exactly:

`patient:symptoms:15`, `patient:antecedents:4`, `patient:symptoms:16`, `patient:symptoms:13`, `patient:symptoms:12`.

**Available support / actual support:** these observations are supported by the patient's answers. The proposed blood-loss mechanism is **not supported by a supplied external passage**. W1 supports only the anemia/symptom component; A1's clot-symptom discussion is not evidence for GI bleeding or anticoagulant-related blood loss. The Diagnostic explicitly discloses the inference and does not assert the mechanism as confirmed.

**Did Diagnostic fail to cite genuinely available evidence?** No source in the snapshot establishes the causal chain. A narrowly scoped W1 citation could support a separate anemia-symptom statement within this hypothesis, but would not substantiate GI blood loss or drug causation. Adding it solely to suppress the missing-reference finding would confuse citation presence with causal support.

**Are the Critic and Safety correct?** Yes on this evidence-bound issue. The Critic correctly calls the mechanism externally unsupported while recognizing its cautious framing. `safety/policy.py` checks all rationale/supporting/contradicting references for each hypothesis. This one has no medical reference, so it independently requires **HUMAN_REVIEW**. It is not the cause of BLOCK.

Grounding can still pass: Phase 6 explicitly permits a meaningful patient-only hypothesis that begins `Uncertain model inference:`, uses relevant patient references, and supplies uncertainty and missingness. Permitting a disclosed inference is **not** equivalent to permitting autonomous final release.

### D. `missing_medical_reference` — anemia associated with kidney failure

**Exact deterministic anchor:** `/differential_diagnoses/1/condition`:

> Anemia associated with chronic kidney failure

**Exact rationale:**

> Uncertain model inference: chronic kidney failure, a previous anemia diagnosis, fatigue, and pallor are patient observations, not proof that kidney disease is causing current anemia.

Supporting statements:

> Chronic kidney failure is reported as an antecedent.
>
> A previous anemia diagnosis is reported.

Contradicting/context statement:

> The concurrent black-stool observation is not explained by this kidney-associated hypothesis on the supplied evidence and raises a competing possibility.

The reference union is exactly:

`patient:antecedents:3`, `patient:antecedents:1`, `patient:symptoms:14`, `patient:symptoms:16`, `patient:symptoms:15`.

**Available support / actual support:** kidney failure and previous anemia are reported; current anemia and kidney causation are not established. W1 supplies symptom overlap, not a renal mechanism. A4's reference to kidney-function tests concerns CML evaluation, not kidney-caused anemia. The rejected Kidney disease document was not supplied to Diagnostic and cannot be retroactively treated as a missing citation.

**Did Diagnostic miss an available citation?** No appropriate renal-causation citation was available. General W1 symptom support must not be presented as renal-causation support. The label may suggest an association, but its rationale correctly limits the evidence to coexistence and explicitly denies proof of causation.

**Are the Critic and Safety correct?** Yes: the Critic correctly separates coexistence from demonstrated causation. This hypothesis likewise has no medical IDs in any structured claim, correctly producing **HUMAN_REVIEW**, not BLOCK.

## 3. Why Grounding passed but Safety blocked, and why no revision occurred

Relevant existing rules were inspected and replayed without changes:

- `orchestration/phase6/diagnostic.py:13–23,54–77`: permits disclosed patient-only inference under the stated constraints.
- `orchestration/phase6/grounding.py:33–64`: validates references/provenance and labels citation presence without claiming entailment.
- `rag/agents/prompts.py:55–68` and `orchestration/phase6/diagnostic.py:101–110`: Critic reviews overconfidence, unsupported causality, contradictions and safety concerns independently.
- `safety/policy.py:15–35`: each Critic safety flag becomes BLOCK; each entirely uncited hypothesis becomes HUMAN_REVIEW.
- `safety/policy.py:38–66`: any BLOCK finding wins over HUMAN_REVIEW; semantic Safety generation must be skipped when Critic flags already block.
- `orchestration/phase6/routing.py:27–42`: Safety/flag restrictions take precedence over the ordinary Critic revision route.

The saved Critic requests `revision_required`, but the two flags prevent automatic revision. The final transition is `ROUTE → BLOCKED`, reason `critic_safety_flags`. Safety was **assessed** with `semantic_status = skipped_critic_block`; it was not missing or failed. No separate Safety cloud call was needed.

The final artifact correctly withholds the proposal and records **human review pending**, not approved or scheduled. There is no mismatch between the saved findings, deterministic policy, route and final decision.

## 4. Smallest exact improvements — proposals only

**No Safety/Grounding code correction is justified by this trace.** Changing all Critic flags from BLOCK to a weaker outcome would be a policy change, not repairing a demonstrated implementation bug.

The smallest concrete output corrections are:

1. `/primary_hypothesis/condition`: replace `Current or recurrent symptomatic anemia` with **`Possible current anemia (unconfirmed)`**. Keep previous anemia as reported history rather than implying confirmed recurrence.
2. `/primary_hypothesis/supporting_evidence/3/statement`: replace the examination-like wording with **`The patient reports paler-than-usual skin; the WHO excerpt lists pale skin among possible manifestations of severe anemia, but this report does not establish anemia or its severity.`** Preserve the existing patient and W1 references.
3. Make the **already disclosed** possible bleeding-related concern prominent in the summary's safety framing, without asserting bleeding, drug causation or severity as established. This addresses the Critic's stated concern; it does not guarantee a different future Critic decision.

These are proposed claim/label revisions, **not edits to saved output or prompts**, and no counterfactual run was performed. The headache differential's systemic observations could also be described as favoring a secondary explanation rather than literally contradicting the existence of headache, as the Critic requested; that is not the direct cause of any of the four findings audited here.

There is **no honest citation-only repair for the two etiologic hypotheses in this snapshot**. Keep them explicitly unsupported and subject to review unless relevant source passages become available through a separately authorized change. Do not add unrelated citations, borrow unseen source sections or remove clinically relevant uncertainty merely to improve a reference count. Even if future wording changes remove both Critic flags, the missing-reference rules still require at least human review; an unrun semantic Safety evaluation could impose further restrictions.

## 5. Verification, artifacts and limitations

The existing offline verifier passed for **two WHO passages, five AMG passages, one Diagnostic version, one Critic version, one Safety assessment and the final BLOCK/pending handoff**. It rechecked WHO full-source checksums and exact spans, selected/candidate source identities, AMG native text/metadata, combined snapshot identity, quote validation and request accounting.

Additional deterministic replay exactly reproduced all four Safety findings and the complete Grounding result. Saved evidence snapshot ID:

`225dd72d4d8c47c00c2d17e87b415bb3235c257ace4e69a0294369ce99b9a45c`

Diagnostic fingerprint:

`a8c3a9fa313e83bf6c294d3308e85c9a8fc301ad65400b2fbe32c6f3fc4051bb`

Safety input fingerprint:

`61e7c1f4fcd9ae70e45c718c82fb6d088edef4b8676dcc83eabd0987bc1cdbd1`

Artifacts in `outputs/ddx-block-audit/`:

- `verification.json`: existing verifier results.
- `trace.json`: every patient fact, exact medical passage/metadata, all Diagnostic claims/citations, Grounding, Critic, Safety and transitions.
- `assessments.json`: each finding mapped to exact Diagnostic fields and patient facts, plus the five requested assessments and proposed changes.
- `before-hashes.json` and `integrity.json`: pre/post protection of existing source, corpus and specified-run artifacts.

The batch runner does not retain raw Diagnostic/Critic request bodies. Source-group presentation was reconstructed from the committed snapshot and unchanged input delegate; no independently captured wire payload is claimed. The saved citations, version/snapshot identities and exact Safety-input fingerprint were verified.

**No code, tests, prompts, retrieval, source files or saved-run artifacts were changed. No live calls, new retrieval, benchmark-label access or additional tuning experiment occurred.** Only this report, its full-evidence appendix, JSON audit artifacts and the relevant OKF context entries were added/updated. No clinical correctness, necessity of BLOCK, or effectiveness claim is made beyond the explicit evidence/rule assessment above.
