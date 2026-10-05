# CHECK2: controlled multi-case evaluation of the current system

## Recommendation

**RETRIEVAL INVESTIGATION NEEDED**

CHECK1's lack of cited medical evidence recurs in this fixed, label-free sample: **6 of 7 technically completed cases have zero diagnostic medical citations**. Only one completed case cites any accepted medical passages (two passages). This supports investigating evidence availability and utilization, not immediately modifying retrieval. It does **not** establish that every unused passage is irrelevant or that retrieval alone causes the failures.

There is also a separate, material reliability problem: **3 of 10 cases fail Diagnostic Agent validation after the existing bounded repair attempt**. Investigate that contract boundary alongside retrieval; do not count those cases as successful diagnoses or as proof of irrelevant retrieval. No production change was required or made for this evaluation.

## Execution and recovery

All ten live cases had finished and saved their artifacts before the CLI interruption. Recovery preserved those traces and final decisions, replayed their verification offline, and completed this report and post-run tests. No live cases were repeated, replaced, or selected after seeing results; recovery made no additional cloud calls.

Artifacts are in [`../outputs/multi-case-evaluation/`](../outputs/multi-case-evaluation/):

- `selection.json`: selection method, label-free pool and frozen ordered IDs.
- `live/sample_case_001.json` through `sample_case_010.json`: complete persisted production workflow states, including patient state, evidence text/metadata, AMG retrieval audit, accepted diagnostics, grounding validations, critic, safety, transitions, timings, failures and invocation telemetry.
- `live/final_decision_001.json` through `final_decision_010.json`: final release/review decisions, all independently recomputed during verification.
- `live/phase6_run_report.json`: runtime configuration and per-case summaries/resources.
- `metrics.json`: per-case measurements and aggregate; `verification-replay.log`: offline replay output.
- `execution.log`, `memory-observations.json`, before/after test logs and integrity inventories.

**Trace limitation:** the existing production provider intentionally sanitizes validation failures and does not persist rejected raw model outputs or detailed validator exceptions in workflow states. The three failed cases retain both request observations, token counts, repair counts and failure codes, but the exact rejected claim cannot be reconstructed. No instrumentation or production logging was changed to recover that unavailable information.

## Method and frozen configuration

The runner is [`../scripts/evaluate_current_system.py`](../scripts/evaluate_current_system.py). It uses the current `application.workflow.create_workflow`, not an experimental retriever or alternate pipeline:

Patient Agent → Patient Case RAG → frozen AMG → Diagnostic Agent → Grounding → Clinical Critic → Safety → Final Decision.

Selection uses `DDXPlusParser.iter_patients('validate', limit=100, include_labels=False)`. The first validation row anchors CHECK1; the remaining nine maximize minimum Jaccard distance over exact recorded question/value facts, sex and age decade, with ties broken by row order. Selection was persisted before retrieval. Ordered validation rows: **1, 7, 74, 85, 63, 76, 43, 25, 47, 22**.

This is a **purposive feature-diversity sample**, not a random population sample or a diagnosis-stratified benchmark. It represents varied observed patient features within the first 100 validation rows, not known disease prevalence. No hidden diagnosis labels were used for selection, runtime reasoning, retrieval, prompts, scoring or routing. No cases or passages were manually assigned clinical/relevance labels; existing benchmark labels were not modified. Critic statements quoted below are model outputs, not our clinical adjudications.

Frozen runtime:

- Existing patient index: **1,000 cases**, patient top-k **1**.
- Existing AMG index: **2,289 chunks**, medical top-k **5**, unchanged squared-L2 gate **1.10**.
- Backend `amg-medlineplus-v1`, snapshot `4303e38d47f31ed7666f177e`.
- Existing generation deployment `gpt-5.6-sol`; temperature not sent (deployment default). This is one stochastic run per case, not a repeatability experiment.
- Existing embeddings/vector spaces, corpus, indexes, ranking, aliases, query construction, gate, safety and human-review policy retained. No multi-query retrieval, candidate-generation experiment, training, model download or added medical data.
- Sequential execution, cached CPU embedding models, a 2 GB available-memory preflight/check before cases. Recorded process RSS observations remained below 0.9 GB; minimum sampled available system RAM exceeded 4.8 GB. These are observations, not a continuous peak-memory guarantee.

## Measurement definitions

**Retrieved passages** means accepted AMG source chunks supplied to the agents, not rejected audit-only candidates. Distinct chunk IDs can have the same title. Counts summed across cases count retrieval events, not globally unique corpus chunks.

**Cited/used** means an accepted medical source ID appears in an accepted Diagnostic Agent hypothesis claim, as recomputed by the existing grounding report. Merely listing an evidence inventory, or a Critic citing a source, is not diagnostic use. Latest-version and any-version counts are identical here: each completed case has one accepted diagnostic version, and failed cases have none.

Citation use is only an **evidence-utilization proxy**. It proves neither medical usefulness, entailment, nor clinical correctness. Conversely, non-use does not establish irrelevance. For failed cases, no accepted diagnostic use is observable; use inside rejected, unsaved responses is unknown.

**Technical completion** means a terminal workflow with no technical failure, not clinical success or ALLOW. All seven technically completed cases reached deterministic Safety. The existing critic-block policy then skipped semantic Safety calls; this is an intentional production short circuit, not a missing execution stage. The three validation failures stopped before Grounding/Critic/Safety and received fail-closed final decisions.

## Aggregate answers (A–I)

| Question | Finding |
|---|---|
| **A. Completed successfully?** | **7/10 technically completed** the pipeline; **3/10 technical failures**. All ten attempts saved workflow and final-decision artifacts. No clinical-success claim. |
| **B. At least one medically useful passage?** | **Not independently measurable under this run's no-label/no-manual-adjudication constraints.** All 10 retrieved accepted passages. The observable utilization proxy is **1/10 attempted cases**, or **1/7 completed cases**, citing any medical passage. Do not report this as a proven usefulness rate. |
| **C. Retrieved passages unused?** | **43/45 (95.6%)** have no accepted diagnostic citation. Of these, **29/31** are unused in the seven completed cases; **14/14** belong to failed cases with no accepted diagnostic output. Rejected-response utilization is unknown. |
| **D. Zero cited medical evidence?** | **9/10 attempts** have zero observed accepted citations: **6/7 completed cases**, plus all three failed cases. |
| **E. Final decisions?** | **ALLOW 0, HUMAN_REVIEW 0, BLOCK 10.** Seven are Safety BLOCK; three are fail-closed technical failures without a Safety assessment. All ten separately require pending human review and withhold proposal release. Pending review does not change final BLOCK into HUMAN_REVIEW. |
| **F. Blocked because evidence was missing?** | All **7/7 assessed cases** have `missing_medical_reference` findings: **25 findings**, each **HUMAN_REVIEW**, none directly BLOCK. All seven also have critic safety flags: **25 `critic_safety_flags` BLOCK findings**. Thus **0 direct missing-reference BLOCK cases**, **7 critic-flag BLOCK cases with co-occurring medical-reference gaps**, and **3 technical BLOCK cases**. Critic flags also concern acuity, missing patient observations and over-specific reasoning; a missing-medical-evidence-only causal block count is not identifiable. |
| **G. Is CHECK1 isolated?** | **No, its no-citation/evidence-gap pattern recurs in this sample.** Excluding the anchor, **5/6 completed cases** have no diagnostic medical citations; the other three non-anchor cases fail validation. This is evidence of recurrence, not a population-wide prevalence estimate. |
| **H. Dominant problem?** | Among completed cases, the dominant observable limitation is **medical evidence availability/use**, with all seven having unsupported hypotheses. Structural grounding passes all seven, source provenance verifies, and Safety follows its policy. A retrieval-only cause versus diagnostic non-use cannot be separated here. Across all attempts, Diagnostic Agent contract reliability is an additional **30% technical-failure problem**. Critic concerns about reasoning and patient-information gaps are also present; they are not adjudicated clinical error rates. |
| **I. Investigate retrieval or something else first?** | **Investigate retrieval/evidence availability and diagnostic utilization next**, without promoting any retrieval change. Also investigate the three Diagnostic Agent validation failures as a separate reliability workstream. There is no evidence here to weaken grounding, critic, safety or human review to increase ALLOW. |

Across accepted diagnoses, there are **25 hypotheses without medical references**. Grounding classifies **113 claim entries** as `unsupported_by_medical_evidence` and **7** as `reference_present_not_verified`; medical entailment remains `not_established` in all seven completed cases. Structural grounding passing is not evidence of clinical correctness.

### Per-case summary

Patient IDs below all have prefix `ddxplus:validate:`. “No refs” counts hypotheses in the accepted diagnostic output; “—” means no accepted output, not zero hypotheses. All final decisions are BLOCK.

| Trace | Row | Accepted AMG | Cited chunks | Unused | No refs | Grounding | Critic assessment | Safety | Completed | Seconds | Cloud calls |
|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|
| 001 | 1 | 5 | 0 | 5 | 4 | passed | revision_required | BLOCK | yes | 44.37 | 3 |
| 002 | 7 | 5 | 0 | 5 | — | not reached | not reached | not assessed | no | 39.17 | 3 |
| 003 | 74 | 2 | 0 | 2 | 4 | passed | supported_with_limitations | BLOCK | yes | 48.30 | 3 |
| 004 | 85 | 5 | 2 | 3 | 3 | passed | revision_required | BLOCK | yes | 75.85 | 3 |
| 005 | 63 | 5 | 0 | 5 | 3 | passed | revision_required | BLOCK | yes | 42.89 | 3 |
| 006 | 76 | 5 | 0 | 5 | 2 | passed | revision_required | BLOCK | yes | 43.59 | 3 |
| 007 | 43 | 5 | 0 | 5 | 5 | passed | revision_required | BLOCK | yes | 46.25 | 3 |
| 008 | 25 | 5 | 0 | 5 | — | not reached | not reached | not assessed | no | 48.96 | 3 |
| 009 | 47 | 4 | 0 | 4 | 4 | passed | revision_required | BLOCK | yes | 52.08 | 3 |
| 010 | 22 | 4 | 0 | 4 | — | not reached | not reached | not assessed | no | 44.29 | 3 |

Total measured workflow latency: **485.74 seconds**, mean **48.57 seconds/case**, range **39.17–75.85 seconds**. These state timings exclude shared model/index initialization and final artifact writing; per-case wall timings including final-decision construction are also saved in `phase6_run_report.json`.

**30 cloud requests:** 10 Patient, 13 Diagnostic (including three bounded repairs), 7 Critic, 0 semantic Safety. There were no diagnostic revision cycles after Critic because the existing Safety policy blocked. All three failed cases have `agent_validation_failure` after two Diagnostic requests, each reported as API-successful, schema-valid but rejected by agent validation. Do not infer an API outage or a particular grounding defect from the sanitized error. No workflow entered the explicit `abstained` state; BLOCK/failure and withheld release are reported separately rather than called abstentions.

## Verification, tests and integrity

- **Before evaluation:** project suite **575 tests**, **574 passed**, **1 opt-in cloud skip**; upstream suite **56 passed**. Existing logs survived the interruption.
- **Recovery baseline:** the five new evaluation-bookkeeping tests passed independently (`tests-recovery-before.log`).
- **After evaluation:** project suite **580 tests**, **579 passed**, **1 opt-in cloud skip**; upstream suite **56 passed**. No failures. The five added tests cover deterministic diversity selection, rejection of labeled records, insufficient sample size, unique claim citation counting and retrieval-without-citations.
- Offline replay independently matched **45/45 accepted passage events** to native source artifacts, validated all **7 accepted diagnostic versions**, **7 Critic versions**, **7 Safety-input fingerprints**, and recomputed **10/10 final decisions**. No additional retrieval/model call was required.
- The pre-run integrity inventory covers **242 pre-existing files**. Post-run comparison finds **241 unchanged**; the sole changed inventoried file is `healthcare-agentic-ai/okf.yaml`, intentionally synchronized with this evaluation. Protected production source, upstream AMG source, benchmark files, dataset archives, snapshot source artifacts, inventoried HNSW files and patient SQLite store are unchanged.
- **Integrity boundary:** the original inventory does not contain the AMG Chroma SQLite bookkeeping file or every possible local cache/environment file. It does not prove byte identity for un-inventoried files. Prior documented Chroma read bookkeeping is not claimed absent. No index rebuild, ingestion or production-store write command was performed by this runner or recovery.

Reproduction (from `healthcare-agentic-ai`; use the existing dependency environments):

```text
python scripts/evaluate_current_system.py --summarize
python -m unittest discover -s tests -t .
# From vendor/amg, using the existing upstream-capable Python:
python -m unittest discover -s tests
```

`--summarize` only replays saved artifacts. `--run` is deliberately protected against overwriting an existing live directory; it was not invoked again during recovery. Artifacts contain label-free patient and internal reviewer material and should remain local research/audit outputs, not released clinical advice.

## Interpretation and next investigation boundary

The successful source/fingerprint checks argue against citation loss, fabricated provenance, or a final-routing mismatch in these saved cases. They do not validate medical relevance. Critic requests for condition-specific evidence recur in all seven accepted outputs, but these remain model assessments. The one case with citations still has three hypotheses without medical references and is blocked on multiple safety flags. Therefore obtaining any citation is not sufficient for ALLOW.

Next, investigate the unchanged retrieval-to-diagnostic interface: what evidence was available versus what accepted hypotheses could cite, with controlled label-free audit and separate attribution of retrieval omission versus diagnostic non-use. Investigate sanitized Diagnostic validation failures separately under an explicitly authorized observability task if deeper failure details are needed. This evaluation neither implements those investigations nor authorizes changes to query construction, aliases, ranking, corpus or safety policy.

The sample is small and purposive, three outputs are censored by technical failure, deployment sampling is uncontrolled, and no clinical entailment/adjudication was performed. **The defensible conclusion is a recurrent evidence-use problem warranting investigation, not a proven clinical accuracy rate or a proven universal retrieval defect.**

## Per-case retrieved titles and exact critic safety flags

The following appendix is transcribed from `metrics.json`, without relevance labels or clinical adjudication. Duplicate titles denote distinct accepted chunks. Full critic assessments, medical IDs, findings and grounding claim paths remain in the JSON traces.

### 001: ddxplus:validate:1

Accepted titles (retrieval order):
1. Blood Clots
2. Myalgic Encephalomyelitis/Chronic Fatigue Syndrome
3. Fibromyalgia
4. Chronic Myeloid Leukemia
5. Pain

Diagnostic passage used: false. Cited chunks: 0.

Critic safety flags (verbatim model output):
- The combination of reported black coal-like stools, new oral-anticoagulant use, near-fainting, disabling fatigue, and pallor represents a potentially serious bleeding or anemia pattern, but the output does not explicitly identify this constellation as a major safety concern.
- The output lists missing hemodynamic and laboratory information but does not clearly distinguish these as critical to assessing severity rather than routine diagnostic uncertainty.
- Head pain involving the temple and multiple head regions remains insufficiently characterized; the undefined onset-speed scale and absent neurologic and visual assessment limit safety evaluation.

No technical error. Deterministic Safety BLOCK; semantic Safety skipped by critic-block policy.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 002: ddxplus:validate:7

Accepted titles (retrieval order):
1. Hay Fever
2. Allergy
3. Asthma
4. Asthma
5. Food Allergy

Diagnostic passage used: false. Cited chunks: 0.

Critic not reached; no observed critic flags (not evidence of safety).

Error: `agent_validation_failure`, 2 Diagnostic attempts; Safety not assessed; final fail-closed BLOCK.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 003: ddxplus:validate:74

Accepted titles (retrieval order):
1. Pain
2. Thoracic Outlet Syndrome

Diagnostic passage used: false. Cited chunks: 0.

Critic safety flags (verbatim model output):
- The reported combination of severe rest chest pain, recent progression, dyspnea, nausea, sweating, and extensive cardiovascular antecedents represents a potentially high-acuity presentation, but no diagnosis is confirmed by objective data.
- Retaining "stable angina" without stronger emphasis on its conflict with the reported unstable pattern could create inappropriate reassurance.
- Recent international travel status should not be used to materially lower concern for pulmonary embolism when broader thromboembolic risk information is missing.
- The absence of objective data must not be interpreted as negative ECG, biomarkers, imaging, examination, or vital signs.

No technical error. Deterministic Safety BLOCK; semantic Safety skipped by critic-block policy.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 004: ddxplus:validate:85

Accepted titles (retrieval order):
1. Pain
2. Heart Attack
3. Fibromyalgia
4. Blood Clots
5. Non-Drug Pain Management

Diagnostic passage used: true. Cited chunks: 2.

Cited source IDs:
- `medical:medlineplus:topic:32:lab:1:chunk:1`
- `medical:medlineplus:topic:5:lab:1:chunk:2`

Critic safety flags (verbatim model output):
- Potentially serious causes of chest pain remain unresolved in a 64-year-old with recent surgery, fever, cough, and multiple cardiopulmonary comorbidities. The tentative viral ranking may create premature closure.
- Pulmonary embolism is included but under-grounded: recent surgery is cited without supporting medical evidence, while key pulmonary-clot manifestations described in the retrieved source remain unknown.
- Acute coronary syndrome is appropriately included, but atypical sharp or nonradiating pain does not exclude it; the output states this limitation and should preserve it prominently.
- The multifocal peeling rash with fever is not characterized sufficiently to assess whether it represents an independent or systemic process. Treating it as a generic viral manifestation could obscure other etiologies.
- The absence of recorded dyspnea, pleuritic worsening, abnormal vital signs, or other warning findings must not be interpreted as their absence; the output generally acknowledges this but the diagnostic ranking remains too specific.

No technical error. Deterministic Safety BLOCK; semantic Safety skipped by critic-block policy.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 005: ddxplus:validate:63

Accepted titles (retrieval order):
1. Encephalitis
2. Myalgic Encephalomyelitis/Chronic Fatigue Syndrome
3. Trigeminal Neuralgia
4. Headache
5. Bell's Palsy

Diagnostic passage used: false. Cited chunks: 0.

Critic safety flags (verbatim model output):
- Tongue, jaw, and neck motor symptoms may have airway or swallowing relevance, but the output does not explicitly identify the absence of respiratory, bulbar, and secretion-handling information as a major safety limitation.
- The absence of treatment or prescribing instructions is appropriate, but the leading label could still create premature diagnostic closure because medication causality and the movement phenotype remain unverified.
- Potentially serious neurologic alternatives cannot be meaningfully assessed without consciousness, examination, temporal-pattern, and systemic-symptom data.

No technical error. Deterministic Safety BLOCK; semantic Safety skipped by critic-block policy.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 006: ddxplus:validate:76

Accepted titles (retrieval order):
1. Hypothyroidism
2. Mpox
3. Blood Clots
4. Chronic Lymphocytic Leukemia
5. Thyroid Tests

Diagnostic passage used: false. Cited chunks: 0.

Critic safety flags (verbatim model output):
- Wheezing and significant shortness of breath with multisystem symptoms represent potentially high-acuity respiratory involvement, while objective physiologic severity is unknown.
- Labeling scombroid-like illness as the leading hypothesis may create premature closure and underemphasize the possible systemic allergic-reaction differential.
- The output acknowledges respiratory involvement but should more clearly distinguish diagnostic uncertainty from uncertainty about acuity; absence of documented hypotension or airway swelling cannot be treated as evidence that they were absent.

No technical error. Deterministic Safety BLOCK; semantic Safety skipped by critic-block policy.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 007: ddxplus:validate:43

Accepted titles (retrieval order):
1. Temporomandibular Disorders
2. Pain
3. Fibromyalgia
4. Sinusitis
5. Pain

Diagnostic passage used: false. Cited chunks: 0.

Critic safety flags (verbatim model output):
- Reported coughing up blood is a potentially important safety-relevant finding. Although the output acknowledges it, making a routine upper respiratory process primary may still promote premature closure because the amount, recurrence, source, associated cardiopulmonary features, and objective severity are unknown.
- The output does not provide a distinct safety assessment separating minor blood-streaking from potentially significant bleeding; the available facts do not permit that distinction.
- The structural or neoplastic alternative is appropriately tentative, but placing it against an allegedly contradictory infectious context could lead to unjustified down-ranking.

No technical error. Deterministic Safety BLOCK; semantic Safety skipped by critic-block policy.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 008: ddxplus:validate:25

Accepted titles (retrieval order):
1. Chest Pain
2. Heart Attack
3. Angina
4. Arrhythmia
5. Pericardial Disorders

Diagnostic passage used: false. Cited chunks: 0.

Critic not reached; no observed critic flags (not evidence of safety).

Error: `agent_validation_failure`, 2 Diagnostic attempts; Safety not assessed; final fail-closed BLOCK.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 009: ddxplus:validate:47

Accepted titles (retrieval order):
1. Anal Disorders
2. Anal Cancer
3. Abdominal Pain
4. Blood Clots

Diagnostic passage used: false. Cited chunks: 0.

Critic safety flags (verbatim model output):
- Testicular torsion is a potentially time-sensitive diagnostic consideration, but it is bundled with "another acute testicular disorder" and not clearly separated as a must-not-miss uncertainty. The available facts neither establish nor exclude it.
- A low numeric pain score should not be used to reduce concern because its scale is undefined and the retrieved abdominal-pain source states that mild pain does not necessarily mean a non-serious problem.
- Continued passage of stool or gas should not be presented as excluding bowel pathology; the output partly avoids this error but still uses the finding diagnostically without direct supporting medical evidence.
- The unsupported ranking of inguinal hernia as primary could anchor interpretation away from acute scrotal or other abdominal processes.

No technical error. Deterministic Safety BLOCK; semantic Safety skipped by critic-block policy.
Explicit workflow abstention: no. Final BLOCK; human review pending.

### 010: ddxplus:validate:22

Accepted titles (retrieval order):
1. Alcohol Use Disorder (AUD)
2. Alcohol Use Disorder (AUD)
3. Abdominal Pain
4. Nausea and Vomiting

Diagnostic passage used: false. Cited chunks: 0.

Critic not reached; no observed critic flags (not evidence of safety).

Error: `agent_validation_failure`, 2 Diagnostic attempts; Safety not assessed; final fail-closed BLOCK.
Explicit workflow abstention: no. Final BLOCK; human review pending.
