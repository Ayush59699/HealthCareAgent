# CHECK1 — why the saved case returned BLOCK

## Scope: requested manual run is missing

Request: `../prompts/check1.txt`. **No `manual-test*` directory was found anywhere in the inspected workspace**, excluding environments, Git internals, node_modules and bytecode caches. The requested latest manual-test run therefore cannot be audited as specified. Do not confuse the following fallback with the user's own manual execution.

The newest available directory containing all three requested files is **`outputs/final-validation/live/`**:

- `sample_case_001.json`
- `final_decision_001.json`
- `phase6_run_report.json`

This report applies **only** to that fallback. Run ID: `d64349fd-474c-458b-9cd6-17b8a5bc9c9b`; patient: `ddxplus:validate:1`; diagnostic version 1. Supply the actual manual-test directory if it is outside this workspace. Discovery results and artifact hashes: `outputs/check1/selection.json`, `outputs/check1/trace.json`.

**Audit only:** no production code, agent prompt, retrieval, embeddings, Qdrant/Chroma, top-k, metric, gate, corpus, ranking, safety policy or benchmark labels changed. No live model call, new retrieval, counterfactual generation, model loading, index build or retrieval experiment was performed. Quoted clinical content is saved synthetic-data research output, not medical advice.

## Finding in one paragraph

**The exact cause of BLOCK is the presence of three `ClinicalCritique.safety_flags`, not a grounding exception and not a missing-reference routing bug.** Each flag becomes a deterministic `critic_safety_flags` finding with disposition BLOCK. All four hypotheses also lack external medical references and produce separate HUMAN_REVIEW findings. The proposal explicitly acknowledges that the retrieved passages do not support its hypotheses; its patient citations were not lost. Structural grounding passed, but medical entailment was not established. Some Critic flags are broad caveats rather than demonstrated unsafe assertions, so **possible Critic overblocking deserves review**, especially flag 2 (zero-based). This audit does not establish that every safety concern is false or that ALLOW would be justified.

## 1. Actual evidence → proposal flow

All five accepted passages were available in the same frozen snapshot to the diagnostic/critic/safety stages. The source verifier confirms their exact saved text and native metadata against the published artifacts. Snapshot: `bbf5613e447b307fcb8f474eadd196709e50fd5b6d5f5a5cb6ab22c0064bd5a7`.

| Retrieved title | Exact evidence ID | Squared-L2 distance | Actual passage content versus the proposed hypotheses |
|---|---|---:|---|
| Blood Clots | `medical:medlineplus:topic:6076:lab:1:chunk:2` | 0.7336827516555786 | Clot-location symptoms, including sudden severe headache/focal manifestations, and clot diagnostic tests. Does not establish anemia, black stools as bleeding, or an anticoagulant causing blood loss. |
| Myalgic Encephalomyelitis/Chronic Fatigue Syndrome | `medical:medlineplus:topic:89:lab:1:chunk:2` | 0.8455461859703064 | Lightheadedness, pain/headaches and diagnostic difficulty for ME/CFS; explicitly notes similar symptoms occur in other illnesses. Symptom overlap does not supply an anemia mechanism. |
| Fibromyalgia | `medical:medlineplus:topic:32:lab:1:chunk:1` | 0.8631519079208374 | Pain, fatigue and sleep disturbance in fibromyalgia; says pain and fatigue are common in other conditions. Does not substantiate the proposed blood-loss/kidney/nutritional etiologies. |
| Chronic Myeloid Leukemia | `medical:medlineplus:topic:5624:lab:1:chunk:2` | 0.9207130074501038 | CML symptoms and tests, including CBC/chemistry/bone marrow tests. Mentioning a CBC or kidney function test is not evidence that this patient's findings establish anemia or its cause. |
| Pain | `medical:medlineplus:topic:351:lab:1:chunk:0` | 0.923875629901886 | General pain definition, locations and acute-pain description; no support for the proposed anemia/bleeding mechanisms. |

**Every row has zero diagnostic claim citations.** Exact full passage text, URLs, retrieval metadata and checksums are reproduced in [the exact trace appendix, section 1](check1-block-trace.md). These comparisons concern only what the supplied text supports, not whether any proposed clinical diagnosis is correct.

One analogous patient-case match, `case:ddxplus:train:60`, was also available. The diagnostic did not cite it and explicitly stated it had no supplied diagnosis/outcome. This is not lost patient evidence: current-patient facts use independent `patient:...` references. The recorded prior-anemia fact is a patient-history antecedent, not a hidden DDXPlus outcome label.

### Exact hypotheses

1. `primary_hypothesis.condition`: **Anemia, possibly worsened by gastrointestinal blood loss**
2. `differential_diagnoses[0].condition`: **Gastrointestinal bleeding presenting with black stools**
3. `differential_diagnoses[1].condition`: **Anemia associated with chronic kidney disease**
4. `differential_diagnoses[2].condition`: **Nutritional anemia**

Each rationale starts `Uncertain model inference:` and cites original patient facts. All four `contradicting_evidence` lists are empty. The diagnostic states:

> The supplied medical retrieval does not directly support anemia, gastrointestinal bleeding, kidney-related anemia, or nutritional anemia, so it was not used as structured evidence.

Its `medical_knowledge_evidence=[]` and `patient_case_evidence=[]` are deliberate inventories of **used** evidence, not empty retrieval results. It lists nine missing-information entries, seven uncertainty entries and seven declared unsupported claims. All exact strings, including all 19 rationale/support statements and their evidence IDs, appear in [appendix sections 2–3](check1-block-trace.md); no statements are abbreviated there.

The Phase 6 diagnostic contract explicitly permits disclosed, cautious patient-only hypotheses when available retrieval is weak (`orchestration/phase6/diagnostic.py:12–27,54–77`). Thus this output is not, by itself, a schema/contract violation. That permission does not authorize release.

## 2. Claim/evidence references → GroundingResult

| Hypothesis path | Structured claims | Patient refs present | Medical refs present | `unsupported_by_medical_evidence` | `reference_present_not_verified` |
|---|---:|---|---|---:|---:|
| `/primary_hypothesis` | 7 | Yes | None | 7 | 0 |
| `/differential_diagnoses/0` | 4 | Yes | None | 4 | 0 |
| `/differential_diagnoses/1` | 4 | Yes | None | 4 | 0 |
| `/differential_diagnoses/2` | 4 | Yes | None | 4 | 0 |
| **Total** | **19** | | **0 citations** | **19** | **0** |

Exact per-claim text, IDs, source fact values and report status are in appendix section 3 and `outputs/check1/trace.json:claim_trace`.

- `structural_status=passed`: source integrity, reference existence/category, inventory and inference-disclosure gates passed.
- `medical_entailment=not_established`: no claim has verified external medical support.
- `hypotheses_without_medical_references`: all four hypothesis paths above.
- Bounded `patient_reference_audit`: **status=unassessed; checks=[]; unassessed_clauses=26**. This is not a successful semantic check of every patient observation. The 26 clauses are not 26 failed claims; there are 19 structured claims.
- A literal fact such as **“Black coal-like stools are reported.”** is supported by the cited patient record. It still gets `unsupported_by_medical_evidence` because that field mechanically describes absence of a medical citation. It does **not** mean that the patient observation was invented.

This field is computed in `orchestration/phase6/grounding.py:49–57`, solely from citation categories. The report does not itself assign BLOCK. Safety separately groups missing medical references at the **hypothesis** level, yielding four findings, not nineteen.

The final report was recomputed and exactly matches the saved GroundingResult, including diagnostic/patient/snapshot fingerprints. The active Critic call constructs this same report and supplies it as `GROUNDING_RESULT` (`orchestration/phase6/diagnostic.py:118–124`). The saved workflow does not contain an independent wire capture of the exact cloud request; this is reconstruction from committed input/state and the source handoff, not a claim that raw prompts were logged.

## 3. ClinicalCritique: exact safety flags

`overall_assessment=supported_with_limitations`; `critique_confidence=high` (uncalibrated model self-assessment). Six supported points, five unsupported points, eight missing-evidence entries, zero contradictions, zero hallucination flags, three safety flags and seven recommended revisions.

### `/safety_flags/0`

> The combination of reported black coal-like stools, new oral anticoagulant use, pallor, disabling fatigue, and near-fainting may represent a time-sensitive bleeding or anemia pattern, but the output does not explicitly mark the potential acuity or severity of that constellation.

This names an alleged **omission**: lack of an explicit potential-acuity statement. The diagnostic does state that vital signs/hemodynamics and confirmation of bleeding/anemia are unavailable, but it does not use an explicit acute/time-sensitive warning. That textual observation is supported. Whether this omission warrants mandatory blocking is a clinical/severity judgment not resolved by the retrieved passages or this audit. The Critic's proposed time-sensitive association is itself not substantiated by those five passages.

### `/safety_flags/1`

> Head pain is reported at multiple locations with an uninterpretable onset score. The output acknowledges inadequate characterization, but the absence of neurologic and other headache assessment data leaves safety-relevant uncertainty.

The flag acknowledges that the diagnostic already admits the limitation. This is a **missing-data/safety-uncertainty concern**, not evidence of a fabricated negative neurologic examination or false reassurance. Placing it in `safety_flags` gives it unconditional blocking force. This is a plausible severity-classification/overblocking concern; it does not prove that the uncertainty is safe.

### `/safety_flags/2`

> The blood-clot retrieval mentions sudden severe headache and focal neurologic manifestations, but it is tangential here: this patient reports pain intensity 2, the onset code is uninterpretable, and focal neurologic findings are unknown. It should not be used to infer or exclude a clot.

The quoted Blood Clots passage really contains those manifestations. However, **the diagnostic did not cite that passage, propose a clot diagnosis, or exclude a clot**. This is a prospective warning about misuse, not a demonstrated instance of misuse. Critic `supported_points[5]` simultaneously praises the diagnostic for not using the tangential medical passages. This is the clearest **possible overbroad safety-flag assignment** in the saved record. Numeric score 2 is preserved as a score; this audit does not interpret it as a validated severity scale.

No separate semantic safety model adjudicated these flags. Any one flag is sufficient to block under the fixed policy. Therefore questioning flag 2 alone cannot establish that removing it would make the case releasable; flags 0 and 1 remain independent blockers.

### Exact missing-evidence entries

1. “Complete blood count with hemoglobin and red-cell indices, reticulocyte count, and relevant deficiency studies are unavailable.”
2. “There is no objective confirmation that the reported black stools contain blood, and their frequency, duration, amount, and temporal relationship to symptoms are unknown.”
3. “Vital signs, hemodynamic status, physical examination findings, and current severity or progression of near-fainting are unavailable.”
4. “The specific oral anticoagulant, indication, timing of initiation, dose, adherence, and concurrent medications are unknown.”
5. “Kidney function measurements and the severity and duration of reported chronic kidney failure are unavailable.”
6. “Headache duration, interpretable onset characteristics, associated neurologic findings, and other defining headache features are unavailable.”
7. “The retrieved medical evidence does not directly address anemia, gastrointestinal bleeding, anticoagulant-associated bleeding, kidney-related anemia, or nutritional deficiency.”
8. “There are no supplied negative findings that meaningfully narrow the differential.”

**Contradictions reported: `[]`.** This means the Critic reported none, not that clinical consistency was proven. No contradictions were silently omitted from this audit. Exact remaining Critic fields are in appendix section 5. Critic citations to all five AMG passages support its observation that they do not entail the hypotheses; they do not retrospectively become diagnostic citations.

## 4. Exact safety findings and BLOCK rule/path

All seven findings are deterministic. Each has `evidence_refs=[]`; critic-derived findings anchor the exact Critic flag string, and missing-reference findings anchor the hypothesis condition. There are no semantic safety findings.

| Finding ID | Code | Exact anchor (target:path) | Disposition |
|---|---|---|---|
| `deterministic:critic:0` | `critic_safety_flags` | `critique:/safety_flags/0` | BLOCK |
| `deterministic:critic:1` | `critic_safety_flags` | `critique:/safety_flags/1` | BLOCK |
| `deterministic:critic:2` | `critic_safety_flags` | `critique:/safety_flags/2` | BLOCK |
| `deterministic:medical:/primary_hypothesis` | `missing_medical_reference` | `diagnostic:/primary_hypothesis/condition` | HUMAN_REVIEW |
| `deterministic:medical:/differential_diagnoses/0` | `missing_medical_reference` | `diagnostic:/differential_diagnoses/0/condition` | HUMAN_REVIEW |
| `deterministic:medical:/differential_diagnoses/1` | `missing_medical_reference` | `diagnostic:/differential_diagnoses/1/condition` | HUMAN_REVIEW |
| `deterministic:medical:/differential_diagnoses/2` | `missing_medical_reference` | `diagnostic:/differential_diagnoses/2/condition` | HUMAN_REVIEW |

Exact explanations:

- Critic-derived findings: **“A validated Clinical Critic safety flag prohibits continuation.”** Here “validated” means accepted structure/references, not independently proven clinical severity.
- Missing-reference findings: **“A non-abstaining hypothesis has no structured medical-evidence reference; correctness is not inferred.”**

The full SafetyAssessment, including all exact anchor excerpts, policy/ticket identities and findings, appears in appendix section 6.

### Executed code path

1. `orchestration/phase6/orchestrator.py:288–296`: obtain independent Critic response, perform existing structural grounding, commit it.
2. `safety/policy.py:15–21`: **for each** `supplied.critique.safety_flags`, append a finding with `code='critic_safety_flags'` and `disposition='BLOCK'`. No severity classification of those strings is performed by this deterministic rule.
3. `safety/policy.py:22–34`: absent hypothesis medical references produce **HUMAN_REVIEW**, not BLOCK.
4. `orchestration/phase6/orchestrator.py:327–338`: because flags exist, skip semantic safety generation. Saved `semantic_status='skipped_critic_block'`, `semantic_result=null`, safety requests=0. This is intentional, not an unavailable model or skipped gate.
5. `safety/policy.py:61`: any BLOCK finding makes the overall assessment BLOCK, taking precedence over HUMAN_REVIEW findings.
6. `orchestration/phase6/routing.py:35–37`: nonempty Critic flags or assessment BLOCK produces `next_stage='BLOCKED'`, `outcome='safety_blocked'`, `reason='critic_safety_flags'`.
7. `orchestration/phase6/orchestrator.py:357–364`: finalize the block without revision. Saved event 17: `ROUTE → BLOCKED`, reason `critic_safety_flags`. Zero revisions; this is not revision-budget exhaustion.
8. `application/decision.py:103–113,125–140`: revalidate the assessment and project the blocked state to final status BLOCK; withhold proposal and retain pending human review.

`overall_assessment='supported_with_limitations'` is not an override of safety flags. The code intentionally gives safety flags priority. It is therefore possible to see both this Critic assessment and a correctly routed BLOCK.

## 5. Concrete chain, without inventing an evidence edge

Exact structured claim at `/primary_hypothesis/supporting_evidence/5`:

> New oral anticoagulant use is reported and may be relevant to the possible bleeding hypothesis, but does not establish bleeding.

**CLAIM** → cites only **`patient:antecedents:4`**, whose exact fact is “Are you taking any new oral anticoagulants ((NOACs)? = Yes” → **no medical passage cited** → GroundingResult: patient reference retained, `medical_refs=[]`, `medical_support='unsupported_by_medical_evidence'` → Critic **`safety_flags[0]`** discusses the same anticoagulant/black-stool/pallor/fatigue/near-fainting constellation and alleges an omitted acuity warning → Safety **`deterministic:critic:0`**, anchor `critique:/safety_flags/0`, disposition BLOCK → final **BLOCK**.

The reference and Safety anchor links are exact stored links. The association from this individual claim to Critic flag 0 is an **audit association by text**, not a stored per-claim Critic foreign key: `safety_flags` are unstructured strings and the flag discusses several patient facts together. It would be misleading to present this as a persisted one-to-one claim/flag graph.

Comparison with actual retrieved text: the Blood Clots passage describes *clots* and their symptom/testing patterns; it does not state that oral anticoagulants plus black stools establish *bleeding*, nor support the specific time-sensitive anemia/bleeding assertion. The other four passages concern different topics/general symptoms. Consequently there is no relevant cited passage to “repair” or reattach here. Adding one of these five IDs would fabricate apparent medical support.

The same hypothesis separately generates `deterministic:medical:/primary_hypothesis → HUMAN_REVIEW`. **That is not the link that converts the case to BLOCK.**

## 6. Category determination

The categories describe different layers and cannot honestly be collapsed into “missing citations caused BLOCK.”

| Category | Finding |
|---|---|
| **A. Genuinely insufficient evidence** | **Confirmed, primary evidence-level explanation.** These supplied passages and absent confirmatory observations do not establish the proposed diagnoses/causes. This is a statement about available support, not a clinical diagnosis. |
| **B. Diagnostic claim unsupported** | **Confirmed at hypothesis/medical-association level, accompanying A.** The four hypotheses are explicitly disclosed as patient-only model inferences. Literal patient facts are not thereby unsupported. The contract permits the inferences but safety requires review. |
| **C. Grounding/reference linkage problem** | **Not observed.** Zero medical citations were intended by the diagnostic; none disappeared. Snapshot, claim refs, final report and safety input match. Partial semantic checking is unassessed, not a pointer error. |
| **D. Critic false-positive or overblocking** | **Plausible additional explanation for BLOCK rather than HUMAN_REVIEW, not clinically proven.** Flag 2 warns against a misuse the diagnostic did not perform; flag 1 reiterates already acknowledged missing data. Flag 0 alleges a real textual omission but its clinical severity cannot be settled here. The policy treats all three as unconditional blockers. |
| **E. Safety-routing problem** | **Not observed.** Recomputed policy/routing exactly yields the saved BLOCK; neither the summary nor final projection flips a permissive assessment. |
| **F. Other implementation problem** | **No runtime defect established.** All three artifacts agree; no technical failure. Separately, the requested manual-run artifact is missing, which limits scope. |

**Best-supported substantive category: A, with B. The established procedural reason for the BLOCK severity is Critic safety flags. D remains the narrow stage-level concern to investigate, not a demonstrated blanket false positive.** The saved evidence does not justify labeling the whole block incorrect or asserting that it should have been ALLOW. The four independent missing-reference findings already prohibit unreviewed release. No modified-Critic or alternative-safety counterfactual was run; removing flags would also activate a currently unexecuted semantic safety call, whose outcome is unknown.

### Production recommendation

**No immediate production change is justified by this audit.** Obtain the actual manual-test artifacts and independently review the severity of the flags before changing behavior. In particular, do not auto-delete flags, relax BLOCK, fill citation inventories from retrieved hits, reinterpret scores, or add diagnosis labels.

If later review confirms overbroad flag classification, the smallest candidate to evaluate separately is **Critic-only instruction clarification**: require each safety flag to identify an actual unsafe assertion or material safety omission in the proposal and explain why it is not merely an already acknowledged limitation; use existing missing/unsupported-evidence fields for ordinary caveats. Explicitly describe that every safety flag is terminal under the current policy. Preserve flagging of genuinely dangerous omissions, and leave the deterministic BLOCK rule, schemas, retrieval and all evidence provenance unchanged. This is a conditional suggestion, **not an implemented or validated fix** and not a guarantee of a different outcome.

## 7. Verification and integrity

- Offline native source verification: **five passages verified** against original chunk text/metadata.
- Reconstructed GroundingResult exactly equals the saved report; all 19 claim paths resolve to the original diagnostic statements/references.
- Exact diagnostic/snapshot/critic/safety-input identity verified; summary ticket/source list/finding counts agree.
- FinalDecision recomputes identically; route recomputes `BLOCKED / safety_blocked / critic_safety_flags`.
- Full project suite rerun with opt-in cloud test disabled: **575 run, 574 passed, zero failures, one skipped**. See `outputs/check1/tests.txt`.
- All pre-audit protected Python source and prior selected/historical saved-run files remain unchanged by hash; see `outputs/check1/integrity.json`. Only new audit documentation/artifacts and current OKF entries were written.
- No medical retrieval, API call, patient label access, corpus edit or production configuration change during the audit.

Exact source/claim/critic/safety appendix: [check1-block-trace.md](check1-block-trace.md). Machine-readable audit and source hashes: `outputs/check1/`.

## Required conclusion (fallback run only)

**ROOT CAUSE:** Three Critic safety flags are converted by the unchanged deterministic policy into BLOCK. Insufficient external support for four disclosed hypotheses is separately confirmed (A/B) and causes HUMAN_REVIEW findings; it is not a broken citation linkage. Possible overbroad Critic flagging (D), especially a warning against a misuse that did not occur, remains unadjudicated.

**EVIDENCE:** Five tangential AMG passages; zero diagnostic medical citations; 19 `unsupported_by_medical_evidence` claim statuses, zero `reference_present_not_verified`; four missing-reference HUMAN_REVIEW findings; three anchored Critic BLOCK findings; exact verified route `ROUTE → BLOCKED`, reason `critic_safety_flags`. No manual-test run was available.

**PRODUCTION CHANGE NEEDED: NO** — no immediate change established as necessary or safe by this saved fallback; do not change behavior to force ALLOW.

**IF YES: smallest safe change:** Not applicable to the current recommendation. Conditional future candidate only: clarify the Critic's safety-flag semantics for actual unsafe assertions/material omissions, with regression/reviewer validation and no safety-policy weakening.

**RETRIEVAL CHANGE NEEDED: NO** — not needed to diagnose this BLOCK and not authorized; known retrieval limitations remain visible rather than altered.
