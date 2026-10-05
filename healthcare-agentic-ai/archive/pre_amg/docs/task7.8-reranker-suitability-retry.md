# Task 7.8 — authorized retry completed: mixed result, no promotion

## Finding

**The same Task 7.8 comparison completed successfully after the user authorized a
retry.** All 2,868 frozen candidate-role pairs were scored with each model, using
verbatim original queries and the unchanged positive-logit gate and selector.

Replacing the reranker recovered eligibility for **60 pairs** (49 current-finding,
11 historical), but **10 previously positive historical pairs became nonpositive**.
It did **not** improve the requested candidates without degrading existing positive
controls:

- **Panic Disorder:** current pair became eligible, but lost during selection.
- **Flu:** current pair became eligible and was selected.
- **Anemia:** historical definition became eligible but lost during selection; a
  short diagnosis/treatment continuation became eligible and was selected.
- **CKD/kidney:** no recovery in case 1's 40 kidney-related role pairs.
- **Fainting:** remained positive and selected.
- **Pediatric asthma symptoms:** remained positive historically, but lost selection.
- **Asthma definition:** historical pair became nonpositive and lost selection.
- **Older Adult Mental Health:** historical pair became nonpositive and lost selection.

This demonstrates that the control's learned scoring/zero-gate combination is an
eligibility bottleneck for **specific frozen pairs**. It does **not** establish that
the control is generally unsuitable, that model suitability is the sole bottleneck,
or that MedCPT is a safe drop-in improvement. Selection remains a bottleneck for
newly eligible Panic Disorder and anemia-definition passages. Different raw-logit
scales and training objectives remain a confound for interpreting fixed-zero-gate
transitions as relevance improvements.

**No production promotion, clinical-accuracy claim or diagnostic-correctness claim.
No subsequent model experiment was run.**

## Fixed conditions and scope

- Control: `cross-encoder/ms-marco-MiniLM-L-6-v2`, revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`; cached locally.
- Sole alternative: `ncbi/MedCPT-Cross-Encoder`, recorded revision
  `71caf65d4927987813984f54c284405a13fcca49`; scalar BERT-base biomedical relevance
  cross-encoder, approximately 109.483M parameters, 437,998,062-byte weights.
- Conservative incremental memory estimate: **2.125996124 GB**, checked before
  scoring/download and again before loading. Preflight passed on this retry.
- One model resident at a time. All three controls passed before the alternative
  download/load/scoring; the control was unloaded first.
- CPU only, batch size 2, two CPU/BLAS threads, one case at a time, no workers.
- Same Task 7.5 depth-50 pools, validated against Task 7.7 closeout: **593 / 416 /
  425 candidates**, respectively; **1,186 / 832 / 850 role pairs**.
- Original query bytes, supporting-query assignments, roles, fact IDs, candidate
  IDs, evidence/retrieval text, provenance, dense/BM25 ranks, RRF ordering and
  non-neural features remained fixed. Only raw neural scores and their existing
  derived ranking fields changed.
- Unchanged raw-logit **> 0** gate and Task 7.4 selector, including diversity and
  passage budgets. No score normalization, threshold adjustment or fallback.
- No truncation: maximum control/alternative pair lengths were **422/384**,
  **331/297**, and **334/302**. Different tokenizers did not change input text.
- No benchmark contents, hidden DDXPlus labels, query generation, fresh retrieval,
  reindexing, embedding generation, full-corpus loading or downstream agents.
  Source chunks were streamed for frozen-ID integrity only.
- The only remote activity fetched public model metadata and pinned model files;
  no patient queries/passages were sent to a remote model.

### Retry-specific implementation changes

The user authorized retrying the same experiment, not a new model or condition.
Before launch, the metadata API lookup was explicitly pinned to the **already
recorded** MedCPT revision and the associated deterministic test fixture was updated.
The tokenizer allowlist now includes its 74-byte `added_tokens.json`. The retry
launcher disables optional Xet transfers in favor of sequential HTTP downloads.
These do not change the model, thresholds, input pairs or selection behavior.

The prior blocked attempt, historical artifacts and prior report remain intact.
No production defaults, downstream agents, existing guards or historical manifests
were modified. The Task-7.7-only config exception was **not** extended: this runner
strictly authenticated the completed Task 7.7 current-file closeout instead.

## Baseline reproduction — completed first

All controls reproduced before **any** MedCPT scoring. The existing absolute
numerical tolerance remained `1e-4`; this is not an eligibility threshold.

| Case | Max drift vs Task 7.5 | Max drift vs Task 7.6 / 7.7 originals | Eligibility changes | IDs and snapshot |
| --- | ---: | ---: | ---: | --- |
| 1 | 0.000002861023 | 0 / 0 | 0 | Identical |
| 2 | 0.000003337860 | 0 / 0 | 0 | Identical |
| 3 | 0.000002384186 | 0 / 0 | 0 | Identical |

Baseline reports reuse the existing helper: within the `task77` result, inherited
`task76_*` field names refer to the supplied Task 7.7 original reference.

## Every frozen pair: eligibility transitions

Current means the frozen `symptoms` role; historical means `history`.

| Case | Role | Nonpositive → positive | Positive → nonpositive | Positive → positive | Nonpositive → nonpositive |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Current | 12 | 0 | 1 | 580 |
| 1 | Historical | 8 | 0 | 0 | 585 |
| 2 | Current | 7 | 0 | 0 | 409 |
| 2 | Historical | 3 | 9 | 6 | 398 |
| 3 | Current | 30 | 0 | 0 | 395 |
| 3 | Historical | 0 | 1 | 0 | 424 |
| **Total** | **Current** | **49** | **0** | **1** | **1,384** |
| **Total** | **Historical** | **11** | **10** | **6** | **1,407** |

Total: **60 gained eligibility, 10 lost eligibility, 7 stayed positive and 2,791
stayed nonpositive**. Positive pairs increased from **17 to 67**, not a clinical
quality or relevance-precision metric. Of seven originally selected passages across
the three cases, only Fainting remained selected; there were 14 newly selected
passages and six selection losses. New selection is not always new eligibility.

## Required passage inspection

Scores below are raw logits, rounded to six decimal places, not probabilities.
Each case's `important-candidates.json` retains every requested group present in
that pool, its source text, both role pairs and role-specific selection status.
A role may be rejected even when its candidate is selected under the other role;
use the role status, not only the candidate-level final reason.

| Case | Passage / chunk prefix | Role | Control → MedCPT | Consequence |
| --- | --- | --- | ---: | --- |
| 1 | Anemia basic definition `ccd2e542` | Current | −10.431719 → −5.935638 | Still rejected |
| 1 | Anemia basic definition `ccd2e542` | Historical | −2.941430 → 3.352530 | Newly eligible; historical rank 3, loses to unchanged passage budget |
| 1 | Anemia continuation `806cbd50` | Historical | −1.477160 → 10.517439 | Newly eligible; historical rank 1, selected |
| 1 | CKD basic information `6892998d` | Current | −11.150599 → −14.837740 | Still rejected |
| 1 | CKD basic information `6892998d` | Historical | −0.613731 → −8.686604 | Still rejected |
| 1 | Fainting `3bc50c81` | Current | 3.206330 → 13.436291 | Remains eligible and selected |
| 2 | Panic Disorder symptoms `58a8e084` | Current | −2.388788 → 1.050864 | Newly eligible; current rank 6, loses to unchanged passage budget |
| 2 | Panic Disorder symptoms `58a8e084` | Historical | −4.392323 → −11.389576 | Still rejected |
| 2 | Pediatric asthma symptoms `9da28175` | Historical | 7.808829 → 5.471161 | Still eligible; historical rank 5, loses selection to passage budget |
| 2 | Pediatric asthma definition `c6c80e91` | Historical | 4.450912 → −1.026542 | Becomes ineligible; loses selection |
| 2 | Generic asthma symptoms `10632734` | Historical | 1.791513 → −3.235619 | Becomes ineligible; previously not selected |
| 2 | Child Mental Health continuation `9218536a` | Historical | 1.873995 → 8.842468 | Already eligible; newly selected, not eligibility recovery |
| 3 | Flu symptoms `5ec42dd3` | Current | −1.432960 → 6.997891 | Newly eligible; current rank 4, selected |
| 3 | Older Adult Mental Health `03f15d58` | Historical | 0.242057 → −9.959917 | Becomes ineligible; loses selection |

Additional scope checks:

- Case 1 anemia-title group: 14 role pairs, four newly positive, ten still
  nonpositive. The recovered historical pairs include the two Anemia chunks plus
  Aplastic Anemia diagnosis and treatment sections; only the Anemia continuation
  is selected. This does not imply any patient has aplastic anemia.
- Case 1 CKD/kidney group: **all 40 role pairs remain nonpositive**, including both
  CKD chunks and other kidney-related sections.
- Case 2 asthma-title group: 28 role pairs; five stay positive, three become
  nonpositive, twenty remain nonpositive. **All asthma current-role pairs remain
  negative and no asthma passage is finally selected.** Pediatric symptoms remain
  eligible but are displaced; the definition fails the gate. Those are different
  failure mechanisms.
- Case 2's nine positive-to-nonpositive historical transitions cover pediatric
  asthma definition and causes, generic asthma symptoms, two Depression chunks,
  Anxiety, Teen Depression, Postpartum Depression and Traumatic Brain Injury.
  Losing a passage from this diagnostic audit is not itself a measured clinical
  harm; the requested positive-control regression is nevertheless present.
- Both Flu roles and both roles of both Older Adult Mental Health chunks are
  audited. Only the Flu current pair recovers; the previously selected older-adult
  background pair fails the gate rather than merely losing a selection slot.

### Source-text interpretation, without diagnoses or hidden labels

**Case 1:** Fainting directly discusses dizziness/lightheadedness and remains the
leading current passage. The anemia definition mentions oxygen carriage, blood
loss, fatigue and dizziness. Its historical pair now passes the gate but loses to
a short continuation mentioning diagnosis and blood tests; the frozen historical
query itself says `adult diagnosis anemia; family who diagnosed anemia symptoms
and clinical information`. This is background-text recovery, not recovery of the
anemia definition for the current multi-finding query. Newly selected opioid
passages overlap pale skin/loss of consciousness but introduce exposure-specific
context not established by those query terms. Their selection is not evidence of
opioid exposure or improved diagnostic correctness.

**Case 2:** The exact Panic Disorder current query is `feeling detached from body
or surroundings; upper abdominal pain; choking; fear of dying`. The passage
explicitly mentions fear of death, choking, stomach pain, breathing problems and
other panic-attack symptoms. Its newly positive score demonstrates recovery of a
text-compatible frozen pair, yet the unchanged selector drops it below the final
budget. New current selections match sweating, breathlessness, palpitations or
numbness in other frozen queries; the disease-specific titles do not establish
those diseases. Pediatric asthma symptoms remain useful background by source-text
inspection but lose their slot, while the general definition now fails eligibility.
The newly selected Child Mental Health text is a short diagnosis/treatment
continuation that was already positive, not new acute-finding recovery.

**Case 3:** The Flu symptoms passage explicitly contains fever/chills, nasal
symptoms, aches and fatigue, compatible with the fixed current findings. Its
recovery is real at the eligibility and selection levels. However, selected cold
medication and fatigue-management passages are not equivalent to symptom evidence,
and the Mpox title must not be read as a diagnosis. The loss of Older Adult Mental
Health is historical-background suppression, not a demonstrated safety improvement.

## Final evidence and timings

The following lists preserve actual selected-ID order (unlike the generated compact
summary, which lists selected candidates in audit order). Times include token
checks, scoring, selection and audit, excluding integrity, model loading/download.

| Case | Model | Positive current / historical | Selected evidence in order | Time (s) |
| --- | --- | ---: | --- | ---: |
| 1 | Control | 1 / 0 | Fainting | 148.785 |
| 1 | MedCPT | 13 / 8 | Fainting; Opioid Overdose signs; Dehydration symptoms; Opioids/OUD risks; Anemia continuation | 691.469 |
| 2 | Control | 0 / 15 | Pediatric asthma symptoms; asthma definition; Depression symptoms; Anxiety symptoms; Teen Depression symptoms | 101.050 |
| 2 | MedCPT | 7 / 9 | Sarcoidosis symptoms; Peripheral Nerve Disorders symptoms; Lyme Disease symptoms continuation; Arrhythmia symptoms; Child Mental Health continuation | 439.206 |
| 3 | Control | 0 / 1 | Older Adult Mental Health | 105.213 |
| 3 | MedCPT | 30 / 0 | Common Cold symptoms; Cold and Cough Medicines types; Fatigue management; Flu symptoms; Mpox symptoms | 451.683 |

Pure scoring time totals were approximately **263.376 s control / 1,495.081 s
MedCPT** (~5.68× slower on this CPU). Runtime is another tradeoff, not a quality
metric.

## Memory, integrity and validation

- Experiment watchdog exited **0**, status **completed**.
- Unchanged outer stop **14 GB**, startup reserve **1.5 GB**, synchronous stop
  **13.5 GB**. No guards bypassed or relaxed; no memory stop occurred on this retry.
- Experiment sampled peak: **11.034 GB system / 0.986 GB child-tree RSS** externally;
  **11.033018368 GB system / 0.976429056 GB process/tree RSS** synchronously.
- Weight file size and SHA-256 matched the pinned public metadata:
  `61d5ccd48869e03500544525fc231641d7daa9ba267b202c82724750038dc1e0`.
- **60 deterministic tests passed**, including **15 Task 7.8 tests**; not 75 distinct
  tests. No live model, network or hidden-label work was performed by the tests.
- The 284-path Task 7.7 closeout was authenticated. The expanded **333-path** retry
  freeze matched after completion and again during independent artifact closeout.
- A separate **read-only, memory-bounded validation**, not another experiment,
  checked **all 2,868 pair transitions**, verbatim query hashes, source text,
  provenance, role assignments, raw scores/deltas, >0 eligibility, selected-role
  flags, token-budget records, selected IDs and evidence snapshot schemas. It
  rehashed all 333 protected files. **Passed.** No models, new scores, network calls
  or hidden labels were involved.
- Read-only validation peak: **10.785 GB system / 0.104 GB child-tree RSS** externally;
  synchronous peak **10.455957504 GB / 0.095580160 GB**.

## Artifacts

Full run: `outputs/task7.8/controlled-medcpt-retry1/`.

- `configuration.json`, `model-preflight.json`, `comparison.json`, `comparison.md`
- `frozen-files-before.json`, `frozen-files-after.json`, `tests.log`
- Each `validate-1/`, `validate-2/`, `validate-3/`: `control.json`,
  `baseline-control.json`, `experimental.json`, `pair-comparison.json`,
  `important-candidates.json`, `summary.json`
- `closeout/artifact-validation.json`, `closeout/frozen-files-closeout.json`,
  `closeout/validate-*-source-audit.json`, `closeout/required-passages.txt`
- Pinned local model files in `model/`

Launch/validation logs and exit record: `outputs/task7.8/retry1-launch.log`,
`retry1-exit.json`, `retry1-validation.log`. The authorized retry launcher, read-only
validator and passage summarizer are also preserved under `outputs/task7.8/`.
Original blocked attempt: `outputs/task7.8/controlled-medcpt/` and its original report.

## Exactly one recommended next step

**Obtain an independent source-text relevance review of these saved paired outputs,
without new scoring or hidden labels, before authorizing any production change.**
The demonstrated eligibility recoveries coexist with positive-control degradation
and selection displacement; automatic promotion is not warranted.
