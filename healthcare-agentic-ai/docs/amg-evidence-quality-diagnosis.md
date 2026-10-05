# Why five AMG passages did not medically support the diagnosis

**Scope:** `prompts/new1.txt`; saved `outputs/amg-live/sample_case_001.json`, patient `ddxplus:validate:1`. Read-only source-support/workflow review, not clinical adjudication. No new retrieval, model inference, cloud calls, threshold changes or benchmark labels.

## Finding

**Five accepted passages were five distance-gated neighbors, not support for the four hypotheses.** They covered pain/fatigue and other conditions rather than anemia/bleeding mechanisms. The Diagnostic Agent recognized this, followed the permitted **“Uncertain model inference:”** route, and left its medical-reference inventory empty instead of citing irrelevant passages.

**Important correction:** the four `missing_medical_reference` findings came from **deterministic Safety**, not the Clinical Critic. They require **HUMAN_REVIEW**. The actual **BLOCK** came from three **critic safety flags**. No provenance-mapping defect was found.

## 1. Patient facts → query: material context loss

The complete patient context includes pallor, fatigue, dizziness/near-fainting, black stools, prior anemia, kidney failure, anticoagulant use, poor diet, underweight status and headache observations. Prior anemia is a supplied historical fact, not a hidden diagnosis label.

Cached, snapshot-pinned **tokenizer-only** replay reproduced the saved query exactly:

| Measurement | Result |
|---|---:|
| Unique facts; included / omitted | 24; **19 / 5** |
| Full unique-fact query, including special tokens | 339 tokens |
| Actual query, including special tokens | **254 / 256 tokens** |
| Actual characters | 988 / 4,000 |
| First 11 pain-related symptom entries, counted separately | 121 tokens |

The greedy presenting-facts → symptoms → antecedents order (`rag/amg/backend.py:55–71`) omitted:

- `patient:antecedents:2`: family history of anemia.
- `patient:antecedents:3`: chronic kidney failure.
- `patient:antecedents:4`: new oral anticoagulant use.
- `patient:antecedents:5`: recent travel to South East Asia.
- `patient:antecedents:6`: BMI below 18.5 / underweight.

**Material impact:** retrieval lost renal, medication and nutritional context subsequently used by diagnosis. Repetitive pain questions occupied nearly half the budget. However, **prior anemia, black stools, pallor, fatigue and dizziness remained**. Truncation is therefore not the whole explanation. All facts still reached the agents. Without a counterfactual retrieval, improvement in ranking or final outcome cannot be quantified.

The resolver recognized `anemia`, but `unambiguous_lookup=false`: a long case is not an exact source-topic lookup. It also matched ordinary “all” in “bed all day long” to ALL, proposing “Acute Lymphocytic Leukemia”. That 264-token expansion was **skipped**; it did not affect the actual embedding query or explain the returned **Chronic Myeloid Leukemia** passage. This incidental ambiguity was not changed.

## 2. Query → passages: what was actually supported

Retrieval occurred **once before diagnosis**, not per hypothesis. All five candidates passed the unchanged 1.10 distance gate; none was rejected. Absent-topic ranks/distances are not recorded.

IDs below share the prefix **`medical:medlineplus:topic:`**.

| ID suffix; source title | Squared L2 | Actual support boundary |
|---|---:|---|
| `6076:lab:1:chunk:2`; Blood Clots | 0.7337 | Clot-location symptoms, including sudden severe headache with neurologic features. **Not anemia, GI blood loss or anticoagulant bleeding.** |
| `89:lab:1:chunk:2`; ME/CFS | 0.8455 | Dizziness/faintness, pain/headaches and diagnostic uncertainty. **Symptom overlap, not anemia/bleeding causation.** |
| `32:lab:1:chunk:1`; Fibromyalgia | 0.8632 | Pain, fatigue, sleep problems and headaches; explicitly notes nonspecificity. **Not anemia mechanisms.** |
| `5624:lab:1:chunk:2`; Chronic Myeloid Leukemia | 0.9207 | CML symptoms/investigations. Its CBC/kidney-function-test references **do not establish this patient's anemia or renal-anemia mechanism**. |
| `351:lab:1:chunk:0`; Pain | 0.9239 | General pain definitions/patterns. **No direct support for the proposed hypotheses.** |

The critic cited ME/CFS and fibromyalgia only to support **nonspecificity**, not the anemia hypotheses.

## 3. Diagnostic claims → references → critic concerns

Patient IDs below use the prefix `patient:`. **None of the four hypotheses has direct medical support from the five passages.**

| Hypothesis | Actual fact references | Critic's substantive concern |
|---|---|---|
| Current/recurrent symptomatic anemia, potentially blood-loss related | Pallor/fatigue/dizziness: `symptoms:16,13,14,11,12`; prior anemia: `antecedents:1`; black stools/anticoagulant: `symptoms:15`, `antecedents:4`. | Unconfirmed without blood count/objective findings; “symptomatic anemia” may imply unproven causation. |
| GI blood loss | `symptoms:15,16,11,12`; `antecedents:4`. | Bleeding/source and relative ranking unverified. |
| Anemia associated with chronic kidney failure | `antecedents:3,1`; `symptoms:16,14,13`. | Renal severity and causal attribution unestablished. |
| Nutritional anemia | `antecedents:0,6,1,2`; `symptoms:16`. | Deficiency unconfirmed; family history does not specifically support a nutritional subtype, despite the caveat. |

The observations are fact-grounded; the disease links remain tentative inference. Every rationale starts **“Uncertain model inference:”**. Both diagnostic evidence inventories are empty. The agent expressly states: **“No supplied medical-knowledge passage directly supports anemia or gastrointestinal bleeding; the retrieved material is largely irrelevant to the leading possibilities.”** It also declines to use the similar training case without an outcome as proof.

This follows `orchestration/phase6/diagnostic.py:12–27,54–77`: disclosed patient-only hypotheses are allowed with weak/irrelevant evidence, but are not necessarily releasable. The empty-retrieval abstention rule did not trigger because five passages existed.

## 4. Grounding → Critic → Safety: why BLOCK occurred

**Provenance passed:** all five IDs, original texts and full metadata matched the published artifact. Diagnostic/critic snapshot identities and the reconstructed SafetyInput fingerprint matched the saved tickets. Empty diagnostic medical references reflect deliberate non-use, not dropped evidence.

**Grounding passed:** references existed, inventories matched actual usage, and inference disclosures were present. This is not medical entailment. The separate saved patient-reference consistency audit was `unassessed` with 10 unassessed clauses—not a comprehensive semantic pass.

**Critic:** overall `supported_with_limitations`; it accepted tentative consideration but raised three safety flags:

1. Insufficiently prominent safety significance of black stools + anticoagulants + near-fainting/pallor/fatigue.
2. Headache insufficiently addressed as an unresolved problem.
3. Warning against inferring thrombosis from headache alone.

**Safety:** `safety/policy.py:15–34,61–65` assigns **BLOCK** to each critic safety flag and **HUMAN_REVIEW** to each of the four hypotheses without medical references. BLOCK wins. Semantic safety generation was skipped and no revision occurred.

**Critic overreach caveat:** the diagnosis never proposed thrombosis or cited Blood Clots; flag 3 is precautionary rather than an observed citation misuse. Flag 2 overstates attribution: the diagnosis explicitly says headache cannot be attributed to anemia from these facts. These are possible critic-quality issues, not mapping bugs. Even without those flags, flag 1 still blocks under existing policy and four missing-reference findings remain. No policy was weakened.

## 5. Bottleneck classification and action

The corpus is **not devoid of relevant evidence**. Read-only inspection of the checksum-verified chunks artifact found available, unreturned passages:

- `medlineplus:topic:139:lab:1:chunk:0`, **Anemia**: blood-loss/nutritional causes, tiredness/dizziness/headache, diagnostic blood tests.
- `medlineplus:topic:1308:lab:1:chunk:0`, **Gastrointestinal Bleeding**: “Black or tarry stool”.
- `medlineplus:topic:4861:lab:1:chunk:1`, **Blood Thinners**: bleeding, red/black stools, dizziness/weakness.
- `medlineplus:topic:302:lab:1:chunk:0`, **Kidney Failure**: insufficient red-cell production.

This inventory was **not a new search or evidence injection**. These sources offer general context, not confirmation of this patient's diagnosis.

**Primary evidence bottleneck:** case-query representation/retrieval coverage produced an insufficiently relevant evidence set despite relevant local material. **Secondary:** permitted patient-only reasoning, unsupported ranking/wording and conservative/partly precautionary critic flags. **Not demonstrated:** corrupted provenance, lost citations, total corpus insufficiency or a threshold implementation failure. Increasing top-k or relaxing thresholds is not justified by this saved run alone.

**No clear runtime implementation bug requiring correction was established; code remains unchanged.** Only this report and OKF navigation were added/updated, plus local audit/test outputs.

### Verification

- Existing saved-run verifier: **5 sources, 1 diagnostic, 1 critic, 1 safety input verified**.
- **48 targeted tests passed**: AMG integration, safety rules, reference consistency and orchestration grounding; `outputs/new1-contract-tests.txt`.
- Tokenizer/source details: `outputs/new1-case-audit.json`; exact query replay and **166 protected Python/saved-run files unchanged** during inspection.
- Saved case SHA-256: `73f2b18abf18e148e06020b4dfa98a79cc18f64d51773e8a81ed16ac5cb39bf8`.

**Answer:** AMG returned broadly similar passages instead of support for the leading hypotheses. The agent honestly declined to cite them and used its allowed inference path. Missing references consequently remained visible to Safety; critic safety flags—not passage count or a provenance defect—caused the final BLOCK.
