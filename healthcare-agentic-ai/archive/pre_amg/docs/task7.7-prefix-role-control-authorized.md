# Task 7.7 — prefix-only role control: completed, negative result

## Outcome

**Outcome C — degradation of previously recovered background evidence**, with the
**Outcome B finding that prefix-only wording does not recover the demonstrated
current-finding eligibility bottleneck**. No automatic promotion.

Across all three frozen cases, **nonpositive → positive transitions were exactly
zero**, for both current and historical roles. Scores changed materially, but
12 previously positive historical pairs became nonpositive. No previously missed
candidate became eligible. Newly selected passages in case 2 were already eligible
under the original input; selection changes are not eligibility recovery.

These are label-free retrieval diagnostics, not clinical accuracy, diagnostic
correctness, relevance-benchmark results, reduced BLOCK rate, or generalization.
No downstream diagnostic/safety pipeline was run on these cases.

## Authorized integrity reconciliation — no protected file was edited

Task 7.7.1 authorized exactly one historical mismatch, `rag/config.py`:

- Historical SHA-256: `1247b27604da291857636b313284da08f36f59da4a287b6a67e841efbae60a24`
- Authorized/current SHA-256: `d9567f1c69a8e3e754e6cea2b765d2121add300145a248f00e3f932e3e51da58`
- Exact difference: `OKF_ENABLED = True` and a following blank line, after the imports.
- Removing only those bytes **in memory**, preserving CRLF, reproduces the historical
  hash. Git blob `63b030c3b852c596ee183d8938905d441df31662`, represented with CRLF,
  also reproduces that hash. Earlier blocked-run documentation records this as a
  user-requested addition.

The exception is byte-pinned and Task-7.7-only, not a blanket exemption for this
file. The authorization and its evidence were reported before scoring. All six
historical Task 7.5/7.6 manifests remain unchanged; they were not regenerated.
Every other historical path remained a strict match: **125 of 126 paths**, plus
this one authorized exception. Initial expanded preflight preserved 277 paths.
The expanded experiment freeze and post-run closeout verified **284 paths unchanged**,
including old runs, implementations, contracts, sources/indexes and opaque benchmark
artifacts. Benchmark contents were never parsed, inspected or imported.

Initial preflight: `outputs/task7.7/authorized-preflight-20261003T171618030868Z/`.
Original blocked runs and their documents were preserved rather than overwritten.

## Fixed experiment

- Frozen depth-50 pools: validate:1 **593**, validate:2 **416**, validate:3 **425**.
- Fixed query additions only: `Current findings: ` and `Historical/background: `.
  The trailing space separates the marker from the byte-identical original query.
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`, pinned revision
  `233902d25c440f23af6f7d6e94d2946bac0bee0a`; cached weights only, CPU,
  batch size 2, two PyTorch/BLAS threads, maximum pair length 512.
- Original supporting-query assignments, roles, fact IDs, candidate/RRF ordering,
  dense/BM25 ranks, RRF scores, source payloads, evidence text and demographic/concept
  features remained frozen. Only raw neural scores and their existing derived
  ranking fields were recomputed.
- Task 7.6 original pair planner, token-budget check, batch scoring and score
  application were reused. Task 7.4 selection, document limits, diversity,
  raw-logit **> 0** eligibility and snapshot/provenance contracts remained unchanged.
- One case at a time, original then prefix-only; no workers, parallel cases, model
  precision changes, downloads, query generation, retrieval or RRF generation.
- PatientState was verified against the unchanged parser's
  `iter_patients('validate', limit=3, include_labels=False)` and frozen records.
  Current source payloads were streamed only for integrity verification; no full
  corpus/index was loaded into memory.
- Both inputs were token-checked before scoring each case. Maximum original/prefix
  pair lengths were **422/425**, **331/334**, and **334/337**. No truncation occurred.
- **2,868 frozen candidate-role pairs**, each scored once per input: **5,736 total
  pair inputs**, with no additional prefix/model/threshold variants.

## Fresh baseline reproduction

Every original-input control passed **before** its prefix input was scored.
The established Task 7.6 absolute tolerance remained **1e-4**.

| Case | Max absolute drift vs Task 7.5 | Max drift vs Task 7.6 original | Eligibility changes | Final IDs and original snapshot |
| --- | ---: | ---: | ---: | --- |
| 1 | 0.000002861023 | 0 | 0 | Identical |
| 2 | 0.000003337860 | 0 | 0 | Identical |
| 3 | 0.000002384186 | 0 | 0 | Identical |

Full evidence IDs, snapshot hashes, pair audits and original/prefixed queries are
in the per-case artifacts and the generated detailed comparison.

## Required comparison

Times include input formatting/token checks, scoring, selection and audit; they
exclude initial integrity, model loading, patient/source verification and the
separate both-format preflight. Memory is sampled total-system decimal GB,
including other applications, not an OS high-water mark.

| Case | Input | Positive current/history | Nonpositive → positive current/history | Final evidence | Time (s) | Phase peak RAM (GB) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 1 | Original | 1/0 | Control | Fainting | 121.334 | 9.867 |
| 1 | Prefix-only | 1/0 | 0/0 | Fainting | 121.380 | 9.846 |
| 2 | Original | 0/15 | Control | Pediatric asthma symptoms; asthma definition; depression symptoms; anxiety symptoms; teen depression symptoms | 91.536 | 10.077 |
| 2 | Prefix-only | 0/4 | 0/0 | Pediatric asthma symptoms; pediatric asthma diagnosis background; Child Mental Health | 92.320 | 10.122 |
| 3 | Original | 0/1 | Control | Older Adult Mental Health | 97.634 | 10.231 |
| 3 | Prefix-only | 0/0 | 0/0 | None | 92.508 | 10.265 |

### All four eligibility transitions

| Case | Role | Nonpositive → positive | Positive → nonpositive | Positive → positive | Nonpositive → nonpositive |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Current | 0 | 0 | 1 | 592 |
| 1 | History | 0 | 0 | 0 | 593 |
| 2 | Current | 0 | 0 | 0 | 416 |
| 2 | History | 0 | 11 | 4 | 401 |
| 3 | Current | 0 | 0 | 0 | 425 |
| 3 | History | 0 | 1 | 0 | 424 |
| **Total** | **Current** | **0** | **0** | **1** | **1,433** |
| **Total** | **History** | **0** | **12** | **4** | **1,418** |

The most important number is **zero newly eligible pairs**, not the number of
final evidence passages. Mean logit deltas for current/history were +0.127565/
−0.316072 (case 1), −0.153066/−1.040677 (case 2), and −0.000763/−0.344509 (case 3).
There were both score increases and decreases; neither establishes clinical usefulness.

## Explicit passage audit

All requested titles were present. `important-candidates.json` includes **every
frozen section/chunk and both role pairs** for those titles, not just selected rows.
The following highlights include the requested basic definition and symptom passages.
Numbers are raw logits, rounded to six decimal places. "Current/history" is the
frozen query role, not a diagnosis or a relevance label.

| Case | Passage | Role | Original → prefix | Eligibility/selection consequence |
| --- | --- | --- | ---: | --- |
| 1 | Anemia — basic definition (`ccd2e542…`) | Current | −10.431719 → −10.585915 | Still ineligible |
| 1 | Anemia — basic definition | History | −2.941430 → −3.605015 | Still ineligible |
| 1 | Chronic Kidney Disease — basic information (`6892998d…`) | Current | −11.150599 → −11.061041 | Increased slightly, still ineligible |
| 1 | Chronic Kidney Disease — basic information | History | −0.613731 → −2.586525 | Still ineligible |
| 1 | Fainting | Current | 3.206330 → 1.997521 | Still eligible and selected |
| 2 | Asthma in Children — symptoms | History | 7.808829 → 4.476871 | Still eligible and selected |
| 2 | Asthma in Children — What is asthma? | History | 4.450912 → 0.739586 | Still eligible; no longer selected due to unchanged document-limit-2 rule |
| 2 | Asthma in Children — How is asthma in children diagnosed? | History | 3.791914 → 1.517336 | Already eligible; newly selected, not recovered eligibility |
| 2 | Panic Disorder — symptoms | Current | −2.388788 → −3.599778 | Still ineligible |
| 2 | Depression — formerly selected symptoms chunk (`6ac5a619…`) | History | 2.670391 → −0.221322 | Became ineligible; lost selection |
| 2 | Depression — second symptoms chunk (`ab272626…`) | History | 1.945513 → −0.524835 | Became ineligible |
| 2 | Anxiety — symptoms | History | 2.426752 → −0.869996 | Became ineligible; lost selection |
| 2 | Teen Depression — symptoms | History | 2.319069 → −0.951082 | Became ineligible; lost selection |
| 3 | Flu — symptoms | Current | −1.432960 → −3.282535 | Still ineligible; no acute-symptom recovery |
| 3 | Older Adult Mental Health — formerly selected chunk (`03f15d58…`) | History | 0.242057 → −2.879394 | Became ineligible; case now has no selected evidence |

### What the source text does and does not show

- **Case 1:** The anemia definition mentions oxygen carriage, fatigue and dizziness;
  CKD basic information describes kidney damage and filtration. These pre-existing
  background passages were not recovered. Fainting directly describes dizziness
  and lightheadedness and remains the selected current-finding passage, despite
  a lower score. No diagnosis is inferred from any of these titles.
- **Case 2:** Pediatric asthma symptom background, including breathing problems,
  remains selected, so **pediatric asthma background is partially preserved**.
  The generic definition remains eligible but is displaced by the already-positive
  pediatric diagnostic-information section under the unchanged two-passages-per-
  document limit. All asthma current-role scores remain negative. Panic Disorder's
  symptom passage explicitly mentions racing heart, sweating, breathing difficulty,
  choking, chest/stomach symptoms and fear of death; the original multi-finding
  pair remains nonpositive and becomes more negative with the prefix. This is a
  query–passage compatibility observation, not evidence of a patient diagnosis.
  Lost depression/anxiety/teen-depression background should not be equated with
  a measured clinical loss; the teen-specific age scope is also distinct from
  generic child wording. The new Child Mental Health selection was already
  positive (history **1.873995 → 1.554139**) and is a short continuation chunk
  about diagnosis/treatment and family history, not newly recovered acute evidence.
- **Case 3:** The Flu passage contains fever/chills, nasal symptoms, aches and
  fatigue, but its fixed current-query pair stays negative. Older Adult Mental
  Health is broad background, not recovered acute-symptom evidence; its loss
  demonstrates historical eligibility suppression, not improved clinical safety.
  **An empty final list is not a failure of the safety system**, which was not run.

## Memory, tests and independent artifact validation

- Unchanged outer stop: **14 GB**; startup reserve: **1.5 GB**.
- Unchanged synchronous stop: **13.5 GB**. No guard was relaxed or bypassed.
- Experiment sampled system peak: **10.265133056 GB** synchronously;
  outer watchdog reported **10.264 GB** on its separate sampling schedule.
- Experiment sampled process-tree RSS: **0.671 GB** outer peak;
  synchronous process/tree peak: **0.656703488 GB**.
- Model-load time: **56.165 s**. Phase, current case and scoring progress were logged.
- Later read-only artifact-validation system peak: **10.324045824 GB**
  (the highest recorded across these task stages); child-tree RSS **0.145 GB**.
- **18 focused tests passed**. **149 tests passed** in the combined relevant
  regression suite, including those 18; these are not 167 distinct tests.
- Regression modules: prefix control, reranker input experiment, ranking ablation,
  Task 7.4 retrieval, experimental/focused medical retrieval, abstention contract,
  orchestration grounding, reference entailment and safety contracts. All used the
  existing runner and deterministic doubles. Provider log messages in these tests
  come from synthetic responses, not cloud calls. No benchmark modules or live
  cloud/integration tests ran.
- Post-run verification independently checked **all 2,868 pair transitions**,
  exact prefix bytes, IDs, source-text hashes, query/fact assignments, ranks,
  raw scores/deltas, the >0 gate, token budgets and snapshot/provenance integrity.
  It also rehashed all **284 protected paths**, which still matched the run freeze.

## Artifacts and implementation

New run directory: `outputs/task7.7/prefix-role-control-authorized-20261003/`.

- `configuration.json`, `comparison.json`, `comparison.md`
- `protected-file-exception.json`, `frozen-files-before.json`, `frozen-files-after.json`
- Each `validate-1/`, `validate-2/`, `validate-3/`: `baseline.json`, `prefix_only.json`,
  `pair-comparison.json`, `summary.json`, `baseline-control.json`,
  `token-preflight.json`, `important-candidates.json`
- `closeout/artifact-validation.json`, `closeout/frozen-files-closeout.json`
- Runner/test logs, source-passage audit and final interpretation report

New opt-in implementation: `rag/experimental_medical/prefix_integrity.py`,
`rag/experimental_medical/reranker_prefix_control.py`,
`scripts/experiment_reranker_prefix_control.py`,
`tests/test_reranker_prefix_control.py`. Production behavior and Task 7.4/7.5/7.6
implementations were not edited. OKF context was synchronized.

## Required closing report

- **What changed:** Added isolated Task 7.7 implementation, deterministic tests,
  an exact authorized integrity exception and new run/report artifacts; only the
  two fixed prefixes differed in neural inputs.
- **What remained frozen:** Config's authorized current bytes, historical manifests,
  original blocked runs, pools, source/provenance, query assignments, retrieval
  features, model/settings, raw-logit gate, selectors, contracts and production defaults.
- **Nonpositive → positive transitions:** **None: 0 current and 0 historical**.
- **Previously useful candidates became eligible:** **No**; neither Panic Disorder
  nor Flu symptom background nor the specified anemia/CKD background was recovered.
- **Pediatric asthma background preserved:** **Partially**—symptoms remained selected;
  the still-positive definition lost its slot to already-positive diagnostic background.
- **Case-3 acute-symptom evidence recovered:** **No**. Flu remained ineligible;
  the former background passage became ineligible and final evidence was empty.
- **Peak memory:** Experiment **10.265 GB** system / **0.671 GB** child-tree RSS;
  highest system sample including post-run validation **10.324 GB**.
- **Tests passed:** **18 focused; 149 combined regression tests**, plus complete
  post-run validation of 2,868 pair transitions and 284 protected paths.
- **Protected files unchanged:** **Yes during this task**, with only the explicitly
  authorized, pre-existing config difference from historical manifests; no other mismatch.
- **Exactly one recommendation:** Investigate the pinned reranker's suitability for
  multi-finding medical query–passage relevance, focusing on the frozen original
  Panic Disorder and Flu pairs that remain nonpositive despite explicit symptom
  overlap, without changing labels, thresholds or production behavior.

Task 7.7 is complete. No next experiment was implemented or run.
