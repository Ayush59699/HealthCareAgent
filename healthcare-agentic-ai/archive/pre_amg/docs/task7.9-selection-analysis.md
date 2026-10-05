# Task 7.9 — controlled selection-stage analysis: completed

## Executive result

**Yes: all six saved selections reproduced exactly**—validate cases 1–3 under
both saved MiniLM and saved MedCPT scores. Final IDs, order, selected roles and
**every recorded Task 7.8 selector event** matched before the counterfactual ran.
No reranker was loaded or scored.

**The saved results demonstrate a selection-stage bottleneck for already-eligible,
previously reviewed useful evidence.** Under saved MedCPT scores:

- **67 candidates were eligible; 15 selected; 52 eligible but unselected.**
- All 52 encountered the final-count check as their terminal positive-role rule.
- Removing **only** the final count recovered **48 additional candidates**; four
  still failed the unchanged two-per-document cap.
- Of the **22 Task 7.8.1 A-rated newly-positive pairs**, **9 were selected and 13
  were eligible but unselected**. All 13 become selectable in the counterfactual.
- Panic Disorder becomes selectable; Flu and Fainting remain selected.
- Pediatric asthma symptoms **still fail selection after budget removal**, because
  higher-ranked treatment and diagnostic-evaluation sections consume that document's
  two slots. The asthma definition and Older Adult Mental Health remain ineligible.
- The same budget removal also selects **all seven existing C-rated newly-positive
  pairs**, previously unselected. Thus count recovery does not establish that
  removing the evidence limit is safe, desirable or a production improvement.

No clinical correctness judgment, diagnosis assignment, new relevance judgments,
production change or subsequent experiment was made.

## Frozen execution and reconstruction

Inputs were only the saved Task 7.8 `control.json`/`experimental.json` audits and
completed Task 7.8.1 review artifacts. Pool sizes stayed **593 / 416 / 425**, with
all candidates, original queries, fact/query assignments, source text/hashes,
provenance, evidence IDs, fusion/reranked order and numeric scores unchanged.
There was no Task 7.5 checkpoint loading, corpus loading, patient parsing, Qdrant,
BM25/RRF regeneration, embedding generation, download or neural scoring.

The original pure `choose_final` and `tokens` functions, plus their original STOP
set, were extracted **without changing their ASTs** from source authenticated
against the saved Task 7.8 protected-source manifest. Whole retrieval modules were
not imported, preventing model/retrieval initialization. An isolated diagnostic
trace was checked against that original function under both conditions; no
production selector was altered to obtain rejection reasons.

The saved audits omit `parent_id`. Reconstruction uses the **already-enforced
Task 7.8 invariant**, not a document-level guess: its authenticated
`validate_frozen_case` checked every record's `parent_id == payload['chunk_id']`
and `text == evidence_text` before scoring. The lookup view therefore uses the
same chunk ID as the parent. A missing invariant or source-hash mismatch is a
stop condition. Private chunk-key lookup indices are not new retrieval ranks or
a regenerated pool. No source/document provenance was substituted.

### The one counterfactual

The unchanged selector operates as follows:

1. If history queries exist, take up to **one** eligible history passage first.
2. Take current-finding passages until total selected count reaches **five**.
3. Backfill history until the same total reaches **five**.
4. Present current passages first, then history, with a stable final role sort.

Within each pass, order is descending saved role ranking score, descending RRF
score, then chunk ID. Checks occur in this order:

1. Pass/count limit.
2. Raw role logit > 0 (verified consistent with the saved eligibility flag).
3. Parent not already selected.
4. Fewer than **two** passages from the source document.
5. No selected passage with normalized-token Jaccard similarity **>= 0.8**.

Only the final count in steps 2/3 of the pass schedule was removed. The original
function received `final_k = infinity` in the isolated analysis; the diagnostic
trace used equivalent unbounded final-pass limits. **The history-priority quota
of one remained one**, as did both role passes, ordering, scores, eligibility,
parent constraint, document cap and Jaccard rule. This is one fixed counterfactual
applied to both saved score sets, not multiple selector alternatives.

### Reason accounting and the meaning of “solely budget”

A first-failing check is not proof that all later checks would pass. Task 7.8's
`pass_budget_reached` also covers the initial history quota, so this analysis
separately records `history_priority_quota_1` and `final_evidence_count_5`.

For every candidate/role, the trace records the exact saved pass event, rank,
score, eligibility, chronological selected prefix, actual blocking IDs, and the
first nonbudget failure at that prefix. Those extra checks are observational:
they never decide which item is selected.

Three distinct counts are reported:

- **Final-budget first:** actual terminal positive-role rejection is the final
  count. Later constraints may be masked.
- **Budget-only at saved prefix:** ignoring that count at the *same* recorded
  selected prefix leaves no parent/doc/Jaccard failure. This is a local test,
  not another selector policy or a guarantee of full-run survival.
- **Confirmed budget-only recovery:** the locally budget-only candidate actually
  survives the sole full counterfactual, with every other rule unchanged.

Reasons are counted once per **eligible unselected candidate**, not once per
pass. Role-specific traces remain available. Every eligible candidate in these
saved cases has exactly one positive role, so candidate and positive-role counts
happen to coincide. That is checked data behavior, not an assumed general rule.
Selected history passages may receive a later parent-duplicate event during
history backfill; they are **not** counted as rejected evidence. Ineligible pairs
can also encounter a budget-first event; their saved negative eligibility still
makes them eligibility failures, not recoverable selection losses.

## Per-case results

“Control” below means the saved MiniLM scores; “MedCPT” means the saved Task 7.8
experimental scores. Both received their own exact saved-selector replay.

| Case | Saved scores | Eligible | Selected | Eligible unselected | Final-budget-first / doc-cap-first | Budget-only at saved prefix | Confirmed additional survivors | No-final-budget selected | Still rejected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | MiniLM | 1 | 1 | 0 | 0 / 0 | 0 | 0 | 1 | 0 |
| 1 | MedCPT | 21 | 5 | 16 | 16 / 0 | 16 | 16 | 21 | 0 |
| 2 | MiniLM | 15 | 5 | 10 | 9 / 1 | 5 | 5 | 10 | 5 |
| 2 | MedCPT | 16 | 5 | 11 | 11 / 0 | 11 | 8 | 13 | 3 |
| 3 | MiniLM | 1 | 1 | 0 | 0 / 0 | 0 | 0 | 1 | 0 |
| 3 | MedCPT | 30 | 5 | 25 | 25 / 0 | 25 | 24 | 29 | 1 |

All still-rejected eligible candidates in the counterfactual fail the document
cap. No previously selected ID was lost in these particular counterfactuals;
history positions can move because additional current evidence is presented first.
Monotonic retention is an observed result here, not guaranteed for all inputs.

### Across-case totals by frozen score set

| Metric | MiniLM | MedCPT |
| --- | ---: | ---: |
| Eligible candidates / positive-role pairs | 17 | 67 |
| Selected | 7 | 15 |
| Eligible unselected | 10 | 52 |
| Actual final-budget-first rejections | 9 | 52 |
| Actual document-cap-first rejections | 1 | 0 |
| Budget-only at saved prefix | 5 | 52 |
| **Confirmed budget-only recoveries / additional selected** | **5** | **48** |
| Counterfactual selected | 12 | 63 |
| Counterfactual still rejected by document cap | 5 | 4 |
| Terminal parent/Jaccard/other exclusions of unselected eligible candidates | 0 | 0 |

For bookkeeping across both score conditions there are **84 eligible case–condition
candidate appearances**, 22 selected, 62 unselected, **53 additional survivors**
and nine still rejected after budget removal. These are **not unique candidates
across models** and should not be used as a pooled retrieval-quality metric.

Under MiniLM, four of nine budget-first rejections already mask a document-cap
failure at the saved prefix; only five recover. Under MedCPT, all 52 locally pass
nonbudget checks, but four encounter newly filled document caps as the unrestricted
pass advances. It would therefore be wrong to claim that all 52 were excluded
*solely* by the final cap in the full-run sense. **48 is the confirmed count.**

Role priority and saved ranking determine which passages consume the finite slots.
They are not separately invented rejection gates. This single counterfactual does
not identify an optimal allocation or isolate the causal effect of removing a
history reservation, changing query order, or changing document caps.

## Panic Disorder: a demonstrated later selection loss

Case 2, `58a8e084-1882-557e-8aec-6fb5719992bb`, source document `medlineplus:601`,
**What are the symptoms of panic disorder?**

- Frozen role/query: current `symptoms:0`,
  `feeling detached from body or surroundings; upper abdominal pain; choking; fear of dying`.
- Saved MedCPT raw score: **1.050863624**; saved ranking score: **1.294864941**.
- Exact current-role ranking position: **6**.
- Saved eligibility: true. Saved selected: false. Counterfactual selected: true,
  final current position **6**.
- Actual limited-run rejection: **final evidence-count budget full**.
- At that point no parent, document-cap or Jaccard check would block it.

The recorded chronological occupants of all five slots were:

1. Child Mental Health continuation `9218536a…` — reserved history-priority slot.
2. Sarcoidosis symptoms `30b020f3…` — current rank 1.
3. Peripheral Nerve Disorders symptoms `92ebd8da…` — current rank 2.
4. Lyme Disease symptoms continuation `1b599c7b…` — current rank 3.
5. Arrhythmia symptoms `177b755e…` — current rank 4.

These are the **actual displacing set**. There is no defensible unique one-for-one
“culprit”: an additional higher-ranked Heart Attack passage at current rank 5
also precedes Panic once the budget is removed. The counterfactual selects that
passage and then Panic while retaining the history reservation. No passage-deletion
or role-reallocation experiment was run.

The historical Panic pair remains negative. Existing Task 7.8.1 class A for the
newly-positive current pair is joined unchanged; no diagnosis is inferred.

## Flu: positive selection control

Case 3, `5ec42dd3-89ad-5b20-b546-bb0495e351de`, source document `medlineplus:299`,
**What are the symptoms of the flu?**

- Role/query: current `symptoms:0`,
  `sweating; fever; chills shivers; nasal congestion clear runny nose`.
- Saved MedCPT raw score: **6.997891426**; ranking score: **7.099205906**; rank **4**.
- No historical candidate is eligible in this case, so the history-priority pass
  consumes no slot.
- Common Cold symptoms, Cold and Cough Medicines types, and Fatigue management
  occupy positions 1–3. Flu passes eligibility, parent/doc/Jaccard checks and is
  selected fourth; Mpox symptoms fills slot 5.
- Flu remains fourth when the final budget is removed. Its parent/document does
  not create a rejection of another eligible candidate here.

Flu contributes one slot to the full budget, after which **25** eligible candidates
are excluded. Pet Health is the first later eligible item (current rank 6), and
becomes selectable without the final cap. This identifies capacity competition,
not a uniquely attributable “Flu displaced Pet Health” intervention; Flu was not
removed or rescored. The saved MiniLM Flu pair remains ineligible under either
selection budget. Existing Task 7.8.1 class A is not rejudged.

## Asthma: eligibility, global budget and within-document ordering

Case 2 historical asthma rows illustrate all three mechanisms. Every asthma
current-role score in this case remains negative under saved MedCPT scores.

| Passage / chunk prefix | MedCPT history raw score | History rank | Saved eligibility/selection | No-final-budget outcome |
| --- | ---: | ---: | --- | --- |
| Pediatric treatment continuation `3b3d7cab` | 8.622674 | 2 | Eligible; final-budget rejection | Selected during history backfill |
| Pediatric diagnostic evaluation `72fa5e8b` | 7.181234 | 3 | Eligible; final-budget rejection | Selected during history backfill |
| Pediatric treatment main chunk `4081c3f4` | 5.653145 | 4 | Eligible; final-budget rejection | Still rejected: document cap 2 |
| Pediatric symptoms `9da28175` | 5.471161 | 5 | Eligible; final-budget rejection | Still rejected: document cap 2 |
| Pediatric impact/background `b82d53fc` | 0.047579 | 9 | Eligible; final-budget rejection | Still rejected: document cap 2 |
| Pediatric definition `c6c80e91` | −1.026542 | 11 | Ineligible; not selected | Still ineligible |
| Generic asthma symptoms `10632734` | −3.235619 | 15 | Ineligible; not selected | Still ineligible |

The **exact two document-cap blockers** for the three remaining eligible pediatric
passages are the treatment continuation `3b3d7cab-7175-5537-b044-67fba3c17259` and
diagnostic-evaluation passage `72fa5e8b-af32-55b0-a830-9bb092ca9b3c`. They are from
the same document and are ranked ahead in history. This is not a Jaccard or
one-per-parent rejection; no new query/topic novelty rule was introduced.

Under saved MiniLM, pediatric symptoms and definition remain selected in their
original order. Diagnostic evaluation is already blocked by those two document
slots before the final evidence budget fills. Five other eligible pediatric
background rows remain document-cap rejects in the unlimited MiniLM replay.

Thus the prior A-reviewed pediatric symptom passage remains a selection loss even
after global budget removal. **Recovering all 22 A-rated newly-positive pairs does
not mean recovering every A-rated eligible passage:** this asthma symptom pair
was *already positive* and is outside that newly-positive subset. The definition's
loss is still eligibility, which a budget-only counterfactual cannot repair.

## Fainting and Older Adult Mental Health

**Fainting**, case 1, `3bc50c81-77f4-539e-a7d3-305beb00fdcb`:

- Exact current rank **1** under both score sets.
- MiniLM raw/ranking **3.206330299 / 3.321220248**;
  MedCPT **13.436290741 / 13.551180689**.
- Positive and selected first in final evidence under both saved and counterfactual
  selections. No diversity rejection; in MedCPT the history reservation occurred
  chronologically first but the original final sort presents Fainting first.

**Older Adult Mental Health**, case 3, `03f15d58-71a7-50a6-9f25-4a68fc2bda7e`:

- Frozen history query: `older adult smoke cigarettes symptoms and clinical information`.
- MiniLM history raw/ranking **0.242057472 / 0.517898482**, history rank **1**:
  selected in history priority, retained under either budget.
- MedCPT history raw/ranking **−9.959917068 / −9.684076059**, history rank **10**;
  current raw **−10.040582657** is also negative. **No role is eligible.**
- It remains unselected without the final cap. Its regression is therefore
  **eligibility**, not recoverable budget, parent, Jaccard or history-priority loss.
  A budget-first event later in the limited run does not override that fact.

No clinical correctness assessment of these passages was performed in Task 7.9.

## Existing Task 7.8.1 classifications: posthoc cross-reference only

A/B/C values were read unchanged from the completed review **after selection**.
They never entered the original selector, trace, ranking, eligibility or
counterfactual decision. No new clinical/relevance labels were created.

| Saved review class, newly-positive pairs | Total | Selected with saved budget | Eligible unselected | Actual rejection rule | Selected without final budget | Newly selected | Still rejected |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| A, clearly useful | **22** | **9** | **13** | Final count: 13 | **22** | **13** | 0 |
| B, ambiguous | 31 | 4 | 27 | Final count: 27 | 30 | 26 | Document cap: 1 |
| C, weak | 7 | 0 | 7 | Final count: 7 | 7 | 7 | 0 |

The remaining B pair is case-3 **Fatigue / What causes fatigue?**,
`1abaf0de-9a2d-56b7-90bc-7a68d8168550`, current rank **15**, raw **2.654592514**.
Fatigue management `84dc515d…` and definition `f34c73ca…` occupy that document's
two slots in the unrestricted sequence, so the cause continuation still fails.

Of the **48** additional MedCPT candidates, **46** are newly-positive reviewed
pairs (13 A + 26 B + 7 C). The other two are already-positive asthma treatment/
evaluation passages, not new eligibility recoveries. All seven saved C examples,
including CPR, veterinary Pet Health and neck-location mismatches, would now be
selected. This is an engineering tradeoff observation using the old review, not
a new classification or an argument to remove the cap.

## Exact IDs/order and all additional candidates

`outputs/task7.9/selection-analysis/selection-ledger.md` is the complete,
human-readable companion to this report. For **each case and score condition** it
contains:

- Every saved final evidence ID in exact order, with title/section.
- Every counterfactual final evidence ID in exact order, with title/section.
- Every additional eligible candidate, in counterfactual order.
- Every eligible candidate's document ID, role/rank/raw score, old/new selection
  positions and actual terminal rules, retaining the saved audit order.

The machine-readable equivalents are in `comparison.json` and each
`validate-*/{control,experimental}-selection-analysis.json`. They also retain every
candidate's query origin, source/hash, evidence ID, ranking metadata, eligibility,
complete phase traces, selected prefixes and exact parent/doc/Jaccard blocker IDs.
The required ineligible controls are included in `*-important-candidates.json`.

Additional-survivor counts are **0/5/0 under MiniLM** and **16/8/24 under MedCPT**.
For example, the eight case-2 MedCPT additions, in counterfactual order, are Heart
Attack symptoms, Panic Disorder symptoms, Heat Illness, pediatric asthma treatment
continuation, pediatric asthma diagnostic evaluation, Child Mental Health warning
signs, TBI diagnostic evaluation and Anxiety diagnostic evaluation. This title
summary does not replace the full ID/section ledger.

## Validation, memory and protected files

- **17 focused Task 7.9 deterministic tests passed.** No other test suite was run.
- Tests cover exact output/event replay and fail-closed mismatch handling; all-six-
  controls prerequisite; the fixed counterfactual and retained history quota;
  actual reason counts; Panic, Flu, asthma, Fainting and Older Adult Mental Health
  behaviors using synthetic fixtures; unchanged parent/doc/Jaccard constraints;
  budget masking; review joins; saved-data nonmutation and memory-stop propagation.
  The six actual saved-case controls separately passed in the guarded analysis.
- **227 protected artifact/source files** matched before and after analysis, and
  again in read-only closeout. Scope includes all existing Task 7.8/7.8.1 artifacts,
  their byte-pinned anchors, relevant historical reports/manifests, protected
  production/experiment source and the three new analysis/test files.
- Historical source hashes were checked against the saved Task 7.8 manifest;
  artifact hashes against the saved Task 7.8.1 preservation manifest. No historical
  manifest was regenerated or edited. External corpus, patient and benchmark
  content was **not opened** for verification; this is not a claim of revalidating
  all 333 paths from the earlier run. The exact 227-path scope is recorded in
  `protected-files-before.json` and `protected-files-after.json`.
- A separate **read-only closeout**, with no selector or model reruns, verified
  all six result sets, candidate ordering, copied scores/queries/eligibility,
  selection IDs/counts, actual-reason counts, all 60 old review joins and protected
  hashes. Status: **passed**.
- Analysis peak: **11.667 GB system / 0.061 GB outer child-tree RSS**;
  synchronous peaks **11.667312640 GB / 0.054087680 GB**.
- Closeout peak: **9.484 GB system / 0.058 GB outer child-tree RSS**.
- Unchanged watchdog **14 GB**, startup reserve **1.5 GB**, synchronous stop
  **13.5 GB**; no stop, bypass, guard relaxation or automatic retry occurred.
- One case at a time, no parallel workers, no model/corpus loading or production
  changes. The only saved labels read were the explicitly permitted completed
  review's A/B/C judgments; no benchmark, proposal, pathology, DDX or hidden labels.

New isolated files:

- `rag/experimental_medical/task79_selection_analysis.py`
- `scripts/experiment_task79_selection.py`
- `tests/test_task79_selection_analysis.py`
- This report and `docs/task7.9.okf`

Artifacts: `outputs/task7.9/selection-analysis/`, including the exact control
reproduction reports, analysis/important-candidate files, review cross-references,
selection ledger, focused test log, protected-file digests and closeout validation.
Launch log: `outputs/task7.9/selection-analysis-launch.log`; read-only validation
log: `outputs/task7.9/closeout.log`.

## Conclusion and exactly one engineering recommendation

The final evidence budget is the predominant observed blocker for saved MedCPT
eligible omissions: **48/52** recover when only it is removed, including **13/13**
previously unselected A-rated newly-positive pairs. Document-cap/section ordering
still blocks pediatric symptom background, and eligibility still blocks the asthma
definition and Older Adult Mental Health. No Jaccard or parent constraint caused
a terminal loss of an unselected eligible candidate in these frozen cases.

This does not establish that all exclusions are unnecessary, that unlimited
output is safe, or that any diagnosis is correct. Production remains unchanged.
Task 7.9 ends here; no next selector/reranker experiment was executed.

**One engineering recommendation:** Prioritize a separately authorized,
**budget-preserving review of evidence allocation and within-document section
ordering**, using these saved rejection traces as regression cases rather than
removing the evidence cap.
