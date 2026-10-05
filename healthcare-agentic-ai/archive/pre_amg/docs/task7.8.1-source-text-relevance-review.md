# Task 7.8.1 — read-only source-text relevance review

## Result

**Completed the review of all 60 nonpositive → positive and all 10 positive →
nonpositive Task 7.8 pairs.** Also reviewed every saved pair for the specified
Flu, Panic Disorder, Fainting, asthma and Older Adult Mental Health titles, and
all selected evidence under either model: **117 distinct case–chunk–role pairs**.

The gains contain useful evidence **and** ambiguous/weak matches. The losses
include substantively relevant asthma, anxiety, depression and head-trauma
background, but not every lost positive was useful:

- **Flu recovery is substantively useful** for the saved febrile/nasal query.
- **Panic Disorder recovery is substantively useful**, but its failure to reach
  final evidence is now a **later selection-budget problem**, not MedCPT rejecting
  that pair.
- **Pediatric asthma symptom/background/definition regressions are substantive
  for the explicit asthma-history query.** Symptom evidence remains eligible but
  is displaced; definition and other background pairs fail eligibility.
- **Older Adult Mental Health is a weaker, ambiguous loss**, not a demonstrated
  loss of strong acute-finding evidence. Its actual historical query is about
  cigarette smoking, not a reported psychiatric problem or increased smoking.
- Some new positive pairs are clearly weak: CPR instructions for head-location
  words, prediabetes skin signs for a neck-location-only query, and **veterinary
  advice for a human fatigue/appetite query**. None of the seven class-C gained
  pairs was ultimately selected.

No diagnosis is assigned or inferred. These are qualitative source-text judgments,
not clinical accuracy, diagnostic correctness, benchmark metrics or new model scores.

## Method and boundaries

Only saved artifacts under `outputs/task7.8/` were used. For judgment, each packet
provided the **exact original query**, title/section and full saved evidence text.
The judgment packets omitted numerical model scores; saved scores, transitions
and selection events were retained separately for traceability and aggregation.

This is a **single AI review performed separately from neural scoring**, not an
independent clinician adjudication or a fully blinded study. The reviewer had
access to prior experimental findings. No claim of inter-rater agreement or
objective ground-truth relevance is made. Borderline A/B decisions are explicit
in the per-pair rationale and should not be treated as calibrated labels.

Rubric:

- **A — Clearly useful/relevant medical evidence:** directly addresses the named
  medical-information topic or provides meaningful correspondence to the supplied
  findings. It need not explain every query term. A disease-specific symptom list
  can be useful *descriptively* without establishing that disease.
- **B — Possibly useful / ambiguous:** partial correspondence, unprovided exposure,
  duration/subtype/age context, fragmentary information, or only conditional
  background/management usefulness.
- **C — Clearly weak or irrelevant evidence:** anatomical/lexical coincidence,
  wrong subject or population, an unrelated topic, or unsupported specialized
  procedure context with no meaningful connection to this pair.

Relevance and evidence use are separate axes. An A can be historical background or
symptom-management information, not evidence of the cause of current findings.
Conversely, a positive neural score is not an A judgment. The review does not borrow
findings from another case or substitute another supporting query to rescue a pair.
An unmentioned feature is **unknown**, not proof that it is absent.

No reranker, embedding model, new pairs, threshold changes, retrieval changes,
benchmark labels, hidden DDXPlus labels, patient-file lookup or downstream agents
were used. Helpers only extracted text, joined explicit reviewer judgments,
counted them and checked coverage/integrity. Existing Task 7.8 artifacts and
manifests were not edited. New files are confined to a separate Task 7.8.1 review
folder, this report and a new OKF sidecar; existing OKF manifests remain untouched.

## Counts — denominators kept separate

The review unit is **case + chunk ID + frozen role/query assignment**, not a unique
document, disease or patient. Different chunks of one title and different roles
are distinct pairs. The supplementary controls are not additional eligibility flips.

| Reviewed set | Pairs | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| All nonpositive → positive changes | **60** | **22** | **31** | **7** |
| All positive → nonpositive changes | **10** | **7** | **2** | **1** |
| **All eligibility changes** | **70** | **29** | **33** | **8** |
| Supplementary unchanged-eligibility/control pairs | 47 | 8 | 17 | 22 |
| **Entire reviewed set** | **117** | **37** | **50** | **30** |

| Case | Gained eligibility: A / B / C | Lost eligibility: A / B / C |
| --- | ---: | ---: |
| 1 | 3 / 15 / 2 (20 pairs) | 0 / 0 / 0 |
| 2 | 8 / 2 / 0 (10 pairs) | 7 / 1 / 1 (9 pairs) |
| 3 | 11 / 14 / 5 (30 pairs) | 0 / 1 / 0 (1 pair) |

Full requested-title coverage, including unchanged pairs:

| Title | Reviewed role pairs | Eligibility-changing pairs |
| --- | ---: | ---: |
| Flu | 6 | 1 |
| Panic Disorder | 4 | 1 |
| Fainting | 4 | 0 |
| Older Adult Mental Health | 8 | 1 |
| Asthma | 12 | 1 |
| Asthma in Children | 18 | 2 |

The gained Bird Flu and H1N1 Flu pairs were also reviewed within the 60 gains.
Coverage above uses exact saved title names; it does not imply every influenza-
related background passage in the entire corpus was inspected.

## Important A/B/C examples with actual queries

### A: Flu symptoms — R092, case 3, current role

- Query: `sweating; fever; chills shivers; nasal congestion clear runny nose`
- Source: **Flu — What are the symptoms of the flu?**
- Evidence explicitly lists **“Fever or feeling feverish/chills”** and
  **“Runny or stuffy nose”**, with other symptoms and a cold/flu comparison.
- Judgment: **A, observed-finding evidence**. This is substantive correspondence,
  not merely title matching. It supports useful symptom information, **not an
  influenza diagnosis**.
- Saved outcome: **−1.432960 → 6.997891**, newly eligible and selected, current
  ranking position 4. Its smoking-history pair R093 remains C and rejected:
  useful evidence for one query is not automatically useful for another.

### A: Panic Disorder symptoms — R039, case 2, current role

- Query: `feeling detached from body or surroundings; upper abdominal pain; choking; fear of dying`
- Source: **Panic Disorder — What are the symptoms of panic disorder?**
- Evidence includes **“a fear of death”**, **“the feeling that they are choking”**
  and **“Stomach pain or nausea”**, plus associated physical signs.
- Judgment: **A, observed-finding evidence**; no diagnosis is inferred.
- Saved outcome: **−2.388788 → 1.050864**, newly eligible, current ranking position
  **6**, not selected. The saved selector event says **`pass_budget_reached`**.
  Under the control this was an eligibility rejection; under MedCPT the remaining
  failure is **later selection**, without changing the gate or budget.
- The same title is B for case 3's sweating/fever/nasal query (R115): sweating/chills
  alone are partial correspondence, while its salient febrile/nasal context is
  not addressed. Case 2's child-anxiety historical pair is B (R040), and the
  case-3 smoking historical pair is C (R116).

### A: Fainting control — R010, case 1, current role

- Query: `skin much paler than usual; cramping pain; slightly dizzy lightheaded; lightheaded dizzy about faint`
- Source: **Fainting — no section**.
- Evidence directly describes feeling dizzy/lightheaded just before fainting,
  associated physical signs and general information about fainting.
- Judgment: **A, observed-finding evidence**. It remained positive and selected
  (**3.206330 → 13.436291**). Its travel-history pair R011 is C; travel relevance
  cannot be inferred from this passage.

### B: Selected opioid passages — R019/R025, case 1

- Query: the same pallor/cramping/lightheaded/near-fainting query above.
- Sources: **Opioid Overdose — What are the signs of an opioid overdose?** and
  **Opioids and Opioid Use Disorder (OUD) — What are the side effects and risks of opioids?**
- Evidence includes pale skin and loss of consciousness, but also opioid-specific
  signs/context not supplied by the query.
- Judgment: **B, partial-finding correspondence**, not established exposure-related
  evidence. Near-fainting is not documented unconsciousness. Both became eligible
  and selected; this is not proof of opioid use or diagnostic improvement.

### B: Anemia continuation vs A definition — R005/R016, case 1

- Query: `adult diagnosis anemia; family who diagnosed anemia symptoms and clinical information`
- Source for both: **Anemia — no section**, different chunks.
- R005 starts **“short of breath or have a headache”** and gives a short testing/
  treatment statement. It is **B, related background**, newly selected.
- R016 explains oxygen carriage, causes including inherited conditions, and
  fatigue/dizziness. It is **A, relevant background**, newly eligible but not
  selected: historical rank 3, `pass_budget_reached`.
- The model recovered a useful definition at eligibility level but favored a
  sparse continuation for the final history slot. No new diagnosis is assigned.

### C: Newly positive mismatches

| Review ID / case | Exact saved query | Source title / section | Why C |
| --- | --- | --- | --- |
| R014 / 1 | `tugging; back head; top head; forehead` | CPR / How is CPR done? | Head positioning in resuscitation is a word match, not evidence about these head-location terms; no arrest context is supplied. |
| R024 / 1 | `adult diagnosis anemia; family who diagnosed anemia symptoms and clinical information` | Hemochromatosis / no section, continuation | This fragment describes testing and iron-removal treatment without the anemia connection present in another chunk; unrelated treatment cannot be inferred from the query. |
| R088 / 3 | `back neck; side neck; where affected region back neck; where affected region side neck` | Temporomandibular Disorders / How are temporomandibular disorders (TMDs) diagnosed? | Neck examination appears inside a jaw-disorder workup; no jaw symptoms are supplied. |
| R095 / 3 | Same neck-location query | Atherosclerosis / What is atherosclerosis? | Neck arteries and the back of the brain are anatomical matches, not evidence for a vascular disorder in this query. |
| R104 / 3 | Same neck-location query | Prediabetes / What are the symptoms of prediabetes? | The passage concerns darkened skin/growths on the neck, not just a named location; those findings are unprovided. |
| R106 / 3 | Same neck-location query | Rehabilitation / Who needs rehabilitation? | Location alone does not establish injury, disability or chronic pain requiring rehabilitation. |
| R107 / 3 | `heavy; exhausting; tired unable usual activities stuck bed; loss appetite get full more quickly` | Pet Health / no section | Appetite loss and tiredness describe a pet needing veterinary care, not human evidence. |

These are **all seven class-C eligibility gains**, not examples selected from an
unreviewed remainder. All remained unselected under the saved MedCPT selection.
The class-C eligibility loss was **Postpartum Depression** for
`child depression symptoms and clinical information` (R038): postpartum/infant
context was not established by that query.

## What useful evidence was newly recovered?

The **22 A-class newly eligible pairs** are:

- **Case 1:** Dehydration symptoms (R001), Fatigue definition (R012), Anemia
  definition/background (R016).
- **Case 2:** Child Mental Health warning signs (R027), Panic Disorder symptoms
  (R039), Peripheral Nerve Disorders symptoms (R042), Lyme Disease symptoms
  continuation (R046), Arrhythmia symptoms (R052), TBI diagnosis information for
  the named head-trauma history (R055), Anxiety diagnosis background (R066),
  Sarcoidosis symptoms (R068).
- **Case 3:** two Tuberculosis symptom chunks (R082/R097), Common Cold symptoms
  (R085), Sweat information (R089), Sinusitis overview (R091), Flu symptoms (R092),
  Fatigue definition (R094), two Cold and Cough Medicines sections (R103/R105),
  Mpox symptoms (R108), Bird Flu human symptoms (R111).

A-class disease-specific lists here explicitly overlap the query findings; they
do **not** establish infection, exposure, subtype or a diagnosis. Some cover only
the febrile or nasal portion of a multi-finding query. The two medication passages
are useful *management information*, not evidence identifying a cause. Many of
these A pairs did not survive selection; 31 other gains were B and seven were C.

## What previously recovered evidence was lost?

### All 10 positive → nonpositive pairs

| ID | Case | Source / section | Class | Text-based assessment |
| --- | ---: | --- | --- | --- |
| R029 | 2 | Depression / symptoms continuation | A | Substantive symptom list for explicit child-depression topic, despite a mid-sentence start. |
| R036 | 2 | Anxiety / symptoms | A | Directly relevant anxiety background; palpitations/breathlessness also overlap saved current findings. |
| R037 | 2 | Teen Depression / symptoms | B | Relevant mood content, but teen-specific scope is not guaranteed by “child.” |
| R038 | 2 | Postpartum Depression / symptoms | C | Unprovided postpartum/infant context; not a substantive loss for this query. |
| R041 | 2 | Depression / symptoms, main chunk | A | Direct child-depression background with explicit age-specific discussion. |
| R044 | 2 | Asthma / symptoms | A | Direct asthma symptom information for named asthma history. |
| R051 | 2 | Asthma in Children / causes | A | Explicit child-asthma trigger/causation background. |
| R059 | 2 | Asthma in Children / definition | A | Direct definition and airway mechanism for named history. |
| R067 | 2 | Traumatic Brain Injury / symptoms | A | Head-trauma symptom/warning information for the explicit head-trauma query. |
| R084 | 3 | Older Adult Mental Health / warning-sign continuation | B | Limited smoking/age connection; not direct acute-finding evidence. |

### Six passages actually lost from final evidence

| Pair | Class | Why selection was lost |
| --- | --- | --- |
| Pediatric asthma symptoms, R033 | A | **Still positive** (7.808829 → 5.471161), but history rank 5 and `pass_budget_reached`; not an eligibility loss. |
| Pediatric asthma definition, R059 | A | Became nonpositive (4.450912 → −1.026542). |
| Anxiety symptoms, R036 | A | Became nonpositive (2.426752 → −8.361932). |
| Depression main symptoms, R041 | A | Became nonpositive (2.670391 → −3.389512). |
| Teen Depression symptoms, R037 | B | Became nonpositive (2.319069 → −0.100669). |
| Older Adult Mental Health, R084 | B | Became nonpositive (0.242057 → −9.959917). |

The symptom asthma passage must not be included in the ten eligibility losses.
Conversely, several A eligibility losses were not previously in final evidence.

### Are the asthma regressions substantive?

**Yes, for the explicitly queried background topic.** The exact query is
`child asthma symptoms and clinical information`. Pediatric asthma symptoms (A,
R033) describe breathlessness, chest tightness, wheezing and warning signs; the
basic definition (A, R059) explains airway inflammation/narrowing. Generic symptoms
(R044) and pediatric causes (R051) are also relevant background. None establishes
the cause of current symptoms, but their loss is not merely removal of unrelated
material. MedCPT loses the definition at the gate and the still-positive pediatric
symptom passage at the selection budget; these are substantively different losses.

The current-role asthma pairs have a different query: `sweating; shortness of
breath; palpitations; numbness and tingling`. Symptom/definition evidence is only
B for that particular full cluster, whereas many trigger/risk/procedure passages
are C there. The same source can be strong history evidence and weaker current
multi-finding evidence without contradiction.

### Is the Older Adult Mental Health regression substantive?

**Not established as a strong evidence loss from the saved query text; classify B.**
The historical query is `older adult smoke cigarettes symptoms and clinical
information`. The lost passage lists mental-health warning signs, including
**“Smoking, drinking, or using drugs more than usual.”** The query reports cigarette
smoking, not an increase in use or a mental-health change. Matching age and that
conditional smoking item provides limited background plausibility, not direct
smoking-related medical evidence or acute-finding evidence.

The current-role query concerns fatigue/appetite and has some broad energy/eating
correspondence (B, R083), but that was not the role selected by the control. The
other introductory Older Adult Mental Health chunk has no smoking connection
(C for its history pair, R102). This review therefore qualifies the earlier
experimental description: **a positive-control selection was lost, but substantive
clinical relevance of that particular control was not demonstrated.** This is not
a claim that suppressing it improves clinical safety.

## Selected evidence: observed findings versus background

All **21 distinct selected case–chunk–role pairs** across both models were reviewed.
Fainting occurs in both models, giving seven control and fifteen MedCPT selection
occurrences, not 22 distinct pairs.

| Selection set | Occurrences | A | B | C |
| --- | ---: | ---: | ---: | ---: |
| Control | 7 | 5 | 2 | 0 |
| MedCPT | 15 | 10 | 5 | 0 |
| Newly selected under MedCPT | 14 | 9 | 5 | 0 |
| Lost from selection under MedCPT | 6 | 4 | 2 | 0 |

These small, purposively reviewed sets are **not comparative precision estimates**.
A count includes useful background and management; it is not a count of diagnoses.

| Case | MedCPT selected source | Class | Evidence use |
| --- | --- | --- | --- |
| 1 | Fainting | A | Direct near-faint/lightheaded finding information; retained control. |
| 1 | Dehydration symptoms | A | Direct dizziness/fainting symptom correspondence, not a cause assignment. |
| 1 | Opioid Overdose signs | B | Partial signs with unprovided exposure context. |
| 1 | Opioids/OUD risks | B | Partial signs and drug-risk background, not established opioid relevance. |
| 1 | Anemia continuation | B | Sparse named-history diagnosis/testing background, not current-finding explanation. |
| 2 | Peripheral Nerve Disorders symptoms | A | Explicit sensory/autonomic correspondence to tingling, sweating and heartbeat changes. |
| 2 | Lyme Disease symptoms continuation | A | Explicit palpitations, breathlessness and tingling correspondence; no infection inference. |
| 2 | Arrhythmia symptoms | A | Explicit palpitations/breathlessness/sweating correspondence; no diagnosis. |
| 2 | Sarcoidosis symptoms | A | Explicit multi-finding correspondence; broad disease list is not a patient diagnosis. |
| 2 | Child Mental Health continuation | B | Family-history assessment/treatment background; already eligible, not a new eligibility gain. |
| 3 | Common Cold symptoms | A | Clear nasal-symptom correspondence, not a complete explanation of the febrile cluster. |
| 3 | Flu symptoms | A | Substantive febrile/nasal symptom evidence. |
| 3 | Mpox symptoms | A | Explicit fever/chills/nasal correspondence; no rash/exposure/diagnosis inferred. |
| 3 | Cold and Cough Medicines types | A | Directly related symptom-management information; not causal/diagnostic evidence or a prescription. |
| 3 | Fatigue management | B | General assessment/self-management information; several-week duration is unprovided, not acute-cause evidence. |

MedCPT's selected set thus comprises **nine direct observed-finding passages, two
partial-finding passages, two related-background passages and two management
passages** under this rubric. The five control history selections in case 2 are
background for named asthma/depression/anxiety topics, with breathlessness and
palpitations overlap in some passages; they are not diagnoses of the current
presentation. The case-3 control is the ambiguous smoking/mental-health background
already discussed. All selected pairs' full rationales are in the ledger.

## Integrity and deliverables

New review artifacts: `outputs/task7.8.1/`.

- `case-1-text-packet.md`, `case-2-text-packet.md`, `case-3-text-packet.md`: exact
  saved queries and complete evidence text inspected, without numerical scores.
- `judgments.tsv`: all 117 explicit reviewer A/B/C judgments, evidence uses and
  individual source-text rationales; not programmatically inferred from scores.
- `reviewed-pairs.json`, `review-ledger.md`: complete join to original pair IDs,
  queries, source text/provenance, saved transitions, scores and selection events.
- `selected-evidence-review.json`: every selected role pair under either model.
- `review-summary.json`, `review-validation.json`: counts and coverage checks.
- `task78-input-hashes-before.json`, `task78-input-hashes-after.json`: matching
  SHA-256 digests for **all 64 existing non-bytecode Task 7.8 artifact files**, including
  the historical attempt, completed run and pinned weights. No file changed.
- Read-only extraction/join helpers and their logs are kept in the new review
  directory. Both ran through the unchanged memory-bounded wrapper, with a
  13.5 GB synchronous check; no reranker inference was invoked. Highest recorded
  system sample was **12.276 GB** during extraction; outer child-tree RSS peaked
  at **0.042 GB**. Review compilation peaked at **11.700 GB / 0.037 GB** externally.

Validation independently checked packet membership against the saved comparisons:
all 70 eligibility changes, all specified title pairs and all selected role pairs
are covered without duplicates; queries and source hashes match; class counts are
computed from the explicit judgments. This validates **coverage and preservation**,
not the correctness of subjective relevance judgments. No production tests or
reranker experiments were run for this analysis.

A separate `docs/task7.8.1.okf` indexes this review without editing existing manifests.

**Review complete. No further experiment is proposed or executed.**
