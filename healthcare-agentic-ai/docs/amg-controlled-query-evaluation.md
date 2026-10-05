# Controlled AMG query representation — implementation and evaluation

Request: `prompts/new3.txt`. **Decision: do not promote. Production is unchanged.**

The deterministic candidate improves fact coverage and Anemia retrieval on the saved case, but drops an important previously included finding, worsens GI Bleeding retrieval, and regresses on other cases. The ordinary `all` guard works in the experimental path; this does **not** mean the unchanged production resolver has been fixed.

## 1. Implementation and boundaries

New, opt-in modules, imported only by the comparison script/tests:

- `rag/amg/query_adapter.py`: `build_query(PatientState, tokenizer, max_tokens=256)` returns the query, token count, and complete included/omitted fact inventory with original patient references.
- `rag/amg/alias_safety.py`: `CaseSafeAliases` wraps the supplied resolver without modifying vendor source, source aliases, snapshots or their hash pins.
- `scripts/evaluate_amg_query_adapter.py`: reproducible offline paired comparison on the saved case and five consecutive existing validation cases.

The representation:

1. Uses only label-free patient facts. No LLM, clinical dictionary, synonym substitution, predicted diagnosis or patient-specific topic rules.
2. Removes limited interrogative scaffolding, e.g. `Do you have a cough? = No` becomes `a cough = No`. History retains `ever`, `recently`, family relationships, timing and other qualifiers. Unknown forms retain their wording; punctuation is trimmed only on structured question/answer facts.
3. Groups answers to **identical original questions** under one heading, e.g. multiple locations remain separate literal answers joined by ` | `. Different original questions are not merged merely because their compact headings match.
4. Emits Presenting, Symptoms, History, Medications and Context sections. Medication routing is limited to explicit `Are you [currently] taking...` wording; it is not a medical-topic classifier. Presenting facts retain their presenting role. Age/sex are context; absent values are explicitly unknown.
5. Packs whole facts round-robin across roles, then renders in original order within each role. Duplicate facts share their original references. Numeric scales, No/Unknown answers and qualifiers are not interpreted. Facts that cannot fit are audited rather than clipped.
6. Enforces the actual cached tokenizer's **256-token** and existing **4,000-character** limits. Role headers consume this same budget; no hidden additional window is provided.

`PatientState` is never mutated. The complete record remains available downstream. Application composition, `AMGMedicalEvidence.query_for`, evidence service, Diagnostic prompts, Grounding, Clinical Critic, Safety, model, corpus, metric, index, top-k and gate were not changed. No new production switch/default was introduced.

## 2. Baseline and evaluation controls

Before implementation, preserved byte copies of the saved run, prior investigation results and original query under `outputs/new3/`, with `baseline-sha256.json`. The original `outputs/amg-live/` and `outputs/new2/` artifacts remain unchanged.

- Replayed the saved query through unchanged AMG: **all five IDs and distances match exactly; maximum distance delta 0.0**.
- Used the same cached CPU MiniLM, all **2,289 stored vectors**, existing Chroma collection, squared L2, **top-k 5** and **1.10** gate.
- Chroma opened only a disposable byte-copy, removed after its child process exited. No original database was opened by Chroma, even for bookkeeping.
- Checked source hashes, vendor implementation pin, embedding identity, collection metric/count, and every copied chunk's source text/metadata.
- Compared three paths: original literal query/original resolver, original literal query/guarded resolver, and candidate query/guarded resolver. On the saved case, the alias-only control leaves embedding/results unchanged: the old false expansion already exceeded the window.
- Requested-topic ranks are **exhaustive global chunk ranks** over stored vectors for the actual embedded query, not diagnosis ranks. Topic names appear only in reporting selection, never as inputs to the adapter/search.
- Selected `validate:2..6` consecutively before examining results; labels disabled. This is a small qualitative regression check, not a new benchmark or clinical efficacy study.

## 3. Saved-case tokens and information coverage

| Measure | Baseline | Candidate |
|---|---:|---:|
| Query tokens, including special tokens | 254 | **251** |
| Actually embedded tokens | 254 | **251** |
| Unique clinical facts included | 19 / 24 | **23 / 24** |
| Clinical facts omitted | 5 | **1** |
| Additional demographic context facts | 0 | 2 |

The candidate preserves:

- Presenting pale-skin finding, with the original affirmative answer.
- All eleven pain-related facts: presence, tugging/cramp answers, four head locations, intensity `2`, radiation `nowhere`, location precision `3`, and onset speed `5`. No scale values are interpreted as clinical severity.
- Both dizziness/near-fainting facts and both fatigue/sleep/activity-limitation facts, with all their qualifiers and answers.
- Poor diet and prior diagnosis history; family history remains explicitly family history, not the patient's diagnosis.
- All five history/context facts previously omitted: family anemia, chronic kidney failure, oral anticoagulants, travel to South East Asia in the last four weeks, and BMI/underweight wording.
- Age 55 and sex F, as context, not as a clinical conclusion.

**New omission:** `Have you recently had stools that were black (like coal)? = Yes` was included in the baseline but does not fit after role-balanced packing. This is a material coverage regression despite the higher total count. The full downstream patient still retains it, but medical retrieval loses its influence. Exact query and per-fact references/answers/roles are in `comparison.json`, first case's `representation`.

No previously unrecorded diagnosis was added: diagnosis names in history came from the patient's original facts. No facts were paraphrased into GI Bleeding or other inferred topics.

## 4. Saved-case before/after retrieval

All ten entries below passed the unchanged distance gate. Lower squared L2 is closer, not confidence.

| Rank | Baseline top five | Distance | Candidate top five | Distance |
|---|---|---:|---|---:|
| 1 | Blood Clots | 0.733683 | ME/CFS | 0.805678 |
| 2 | ME/CFS | 0.845546 | Anemia | 0.807179 |
| 3 | Fibromyalgia | 0.863152 | Blood Clots | 0.841594 |
| 4 | Chronic Myeloid Leukemia | 0.920713 | Dehydration | 0.894008 |
| 5 | Pain | 0.923876 | Hypoglycemia | 0.924136 |

Requested-topic chunk details (IDs below omit the common `medlineplus:topic:` prefix):

| Topic / chunk suffix | Before rank | Before distance | After rank | After distance |
|---|---:|---:|---:|---:|
| Anemia `139:lab:1:chunk:0` | 21 | 1.052496 | **2** | **0.807179** |
| Blood Thinners, warnings `4861:lab:1:chunk:1` | 258 | 1.346185 | 51 | 1.164306 |
| Blood Thinners, definition `4861:lab:1:chunk:0` | 974 | 1.615433 | 241 | 1.378993 |
| GI Bleeding, stool passage `1308:lab:1:chunk:0` | 290 | 1.364999 | **305** | **1.410254** |
| GI Bleeding, investigations `1308:lab:1:chunk:1` | 1010 | 1.623815 | 1061 | 1.644161 |
| Kidney Failure `302:lab:1:chunk:0` | 1780 | 1.836119 | 553 | 1.502438 |

Only Anemia enters the candidate set. Blood Thinners and Kidney Failure improve in global rank/distance but remain outside both the top five and the gate. GI Bleeding worsens. Restoring a fact or recognizing an alias does not guarantee retrieving its topic. The candidate still fills nearly the whole window; its legitimate source-title additions do not fit and are skipped by unchanged AMG behavior.

## 5. Ordinary `all` guard and its limits

The wrapper removes only `all` alias matches without an eligible uppercase `ALL` token. It recomputes the existing source-title additions after filtering, never edits the patient sentence or adds aliases. Lowercase `all`/sentence-initial `All` also cannot obtain an exact-source gate bypass. Explicit `ALL`/`What is ALL?` retain the upstream behavior.

Regression coverage includes the exact **“stuck in your bed all day long”** failure, lowercase stand-alone/definition queries, mixed ordinary `all` plus explicit `ALL`, Unicode normalization, token boundaries, other aliases, and longest-phrase handling. An uppercase token consumed by a longer alias, e.g. `ALL TOPICS all day`, cannot license the unrelated ordinary `all`. A real upstream `Retriever.search` with a fake vector store verifies that guarded `all` embeds unchanged and a 1.11-distance hit still abstains at 1.10.

The saved candidate suppresses one false `all` match. No false leukemia expansion remains in its resolved/embedded query. **This fix is experimental only.** The original production path is intentionally unchanged under the task's non-promotion rule. Uppercase `ALL` is treated as intentional acronym spelling, not semantically disambiguated by an LLM; uppercase emphasis remains a limitation.

**Additional regression discovered:** literal demographic `sex = F/M` is recognized by unchanged AMG as `sex → Sexual Health`. On shorter cases 5 and 6, this source-title expansion fits and is embedded. This is unintended topic expansion caused by context formatting, not a patient symptom. `Medications` headers also match a source alias (the saved candidate skips all additions for budget reasons). These are additional reasons not to promote; the narrow `all` fix does not claim to solve every alias ambiguity. No broader alias rewrite was performed or tested.

## 6. Five additional cases: mixed results, not a clean improvement

Clinical omission counts exclude age/sex; each candidate additionally includes those two context facts.

| Validation row | Query tokens before → after | Embedded tokens before → after | Omitted clinical facts before → after | Accepted hits before → after |
|---|---:|---:|---:|---:|
| 2 | 256 → 251 | 256 → 251 | 9 → 3 | 5 → 5 |
| 3 | 255 → 255 | 255 → 255 | 9 → 4 | 5 → 5 |
| 4 | 242 → 241 | 251 → 241 | 3 → 0 | 5 → 5 |
| 5 | 219 → 160 | 225 → 169 | 0 → 0 | **5 → 3** |
| 6 | 170 → 147 | 175 → 155 | 0 → 0 | 5 → 5 |

- **Row 2:** Heart Attack / Arrhythmia / Chest Pain / Angina / Heart Disease in Women change to Post-Traumatic Stress Disorder / Anxiety / Panic Disorder / Mental Health / Stress. History coverage improves, but a previously included racing/irregular-heartbeat/palpitations fact is lost. No correctness claim is possible from topic names alone; losing that fact is a concrete regression.
- **Row 3:** Fibromyalgia / Blood Clots / Temporomandibular Disorders / Pain / Mpox change to Mpox / Fibromyalgia / Shingles / Blood Clots / Pain. Five more clinical facts fit, with no loss of baseline-included facts, but four remain omitted. This is mixed retrieval, not verified clinical improvement.
- **Row 4:** All facts now fit. Blood Clots remains first and gets closer; Anemia stays second but worsens **0.771755 → 0.880277**. ME/CFS, CML and CLL complete the candidate list. More coverage does not uniformly improve distance.
- **Row 5:** All facts remain, including chest/upper-abdominal pain, nausea and throwing up blood/coffee-bean-like material. Baseline accepted Heart Attack / Pain / Fibromyalgia / GERD / Abdominal Pain. Candidate accepts Fibromyalgia and two Alcohol Use Disorder chunks; PTSD **1.111351** and Pain **1.131552** fail the gate. Accepted evidence narrows to three chunks/two topics. The incidental Sexual Health expansion also activates.
- **Row 6:** Respiratory evidence remains, but Breathing Problems worsens **0.564340 → 0.937471**. Candidate top five are Breathing Problems / COPD / Lung Diseases / Asthma / Asthma. The former child-specific asthma passage is displaced, and incidental Sexual Health expansion activates. No clinical outcome was assessed.

Raw candidates, accepted passages, native IDs, distances, alias resolutions, embedded queries, full states and complete fact inventories for all cases are retained in `comparison.json`. These observations were not converted into benchmark/relevance labels.

## 7. Verification

- **23 focused tests: OK.** Grammar/role preservation, explicit answers and negation, undefined numeric values, duplicate references, repeated headings, budget/character limits, whole-fact omission, deterministic packing, label/raw-row rejection, state immutability, unchanged production query, and narrow alias/gate behavior.
- **529 project tests: OK, one opt-in cloud test skipped.** This includes the 23 new tests; they are not an additional 23 beyond 529.
- **56 standalone upstream tests: OK**, using the existing system Python environment with upstream dependencies.
- Final evaluation: **259 protected files unchanged**, credentials unchanged (no credential hashes published), temporary database removed, worker exit 0.
- Independent comparison to the prior investigation's **256-file inventory: zero changes**, including original production code, saved run, corpus, stores and evaluation data. Existing uncommitted work was preserved.
- No cloud calls, model download, index rebuild, new labels, changed gate/top-k, downstream prompt modifications or safety bypasses. The tokenizer warning about an over-budget proposed expansion is from unchanged AMG's budget check; it falls back before embedding.

Commands from `healthcare-agentic-ai/`:

```text
python scripts/evaluate_amg_query_adapter.py
python -m unittest tests.test_amg_query_adapter -v
python -m unittest discover -s tests -t .
# From vendor/amg/, using the installed upstream-dependency Python:
python -m unittest discover -s tests
```

Artifacts: `outputs/new3/comparison.json`, preserved baseline files and hashes, `evaluation.log`, `focused-tests.txt`, `project-tests.txt`, `upstream-tests.txt`, `integrity.json`, and `post-test-integrity.json`.

## 8. Production recommendation

**Leave the production path unchanged.** This candidate is a reproducible, tested experiment, not a promoted query replacement. Higher fact counts and one better requested topic are insufficient: black-stool and palpitations omissions, lower evidence diversity, and context-triggered alias expansion are meaningful regressions. No clinical improvement is claimed.

A future candidate would need to prevent newly losing baseline-included presenting/symptom facts and keep structural/demographic terms from being interpreted as medical aliases, then undergo another controlled evaluation. No such further tuning, broader alias change or downstream workaround is part of this implementation.
