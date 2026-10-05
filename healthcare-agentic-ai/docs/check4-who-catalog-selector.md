# CHECK4 — conservative WHO catalog-selector evaluation

## Decision

**Keep isolated for now. Consistent adequacy for WHO+AMG integration is not yet established.** This is an encouraging precision-oriented pilot, not a negative result: all four final document selections have source-supported topical relevance in the review below. However, eight authored cases, five abstentions, three relatively explicit positive presentations, no selected Q&A documents, and one run per case cannot establish general reliability or recall.

No production behavior was changed and no experimental WHO+AMG wiring was added. Diagnostic, Grounding, Critic, Safety, AMG, Patient RAG and the production WHO lookup were not called or modified. No patient diagnosis or differential was generated. Disease names below are **WHO document titles**, not predictions.

## Implementation

- `rag/who_catalog_selector.py`: isolated v2 catalog prompt; schema **hard maximum of three** unique, known IDs, revalidated before opening files. Whole existing catalog: **569 documents**, not a new index. No lexical search, keyword ranking or embeddings.
- `rag/who_selector_relevance.py`: `select_verified` is the **complete CHECK4 selector**. First shortlist 0–3 IDs, then load and review each candidate's **full existing text sequentially**. Only candidates with an affirmative content-based relevance judgment survive. No refill or alternate search if a candidate is rejected.
- Both prompts explicitly distinguish positive presenting features from negated, uncertain, historical and demographic information; include `no SOB`, mixed-polarity sentences and unknown tests; forbid selecting disease topics by inventing possible causes or missing facts. Generic symptoms/word overlap alone are insufficient. Model knowledge may interpret language but cannot replace source support.
- Content responses contain only constrained verdict/reason codes and short literal patient/source quotes. No free-form diagnosis, advice or differential fields. Relevant verdicts require both patient and body quotes; their exact presence is validated. **Quote checks verify provenance, not medical entailment or semantic negation correctness.** The semantic decision remains fallible LLM judgment.
- All candidate IDs are validated before any document opens. Missing/tampered source headers, unknown IDs, malformed judgments and provider errors fail closed. A content-call failure withholds all final selections, including previously accepted candidates; failures are distinct from valid empty results.
- Existing catalog/reader bounds remain: at most 1,000 metadata entries, 90,000 catalog JSON bytes, 4 MiB per document. No document cache, corpus-text scan, models or vector store. One document/request at a time; the existing provider request byte cap rejects oversize input, **never truncates**. Full candidate text is written to the audit directory, including rejected candidates. Maximum four calls per case; no repairs/API retries, parallel calls or quota-filling.

`WHOCatalogSelector.select` and the old CHECK3 evaluation script expose the catalog-only proposal stage. **Do not treat those proposals as CHECK4 final selections.** Use `select_verified` or the new CHECK4 runner. The generic candidate observed below demonstrates why this distinction matters.

## Fixed evaluation design

Eight new, hand-authored `PatientRepresentation` cases; no benchmark rows, labels, hidden diagnoses or expected diagnostic answers. All facts, categories, catalog and prompts were persisted **before the first cloud call** in `outputs/check4-selector/live/plan.json` and `catalog.json`. No case/prompt tuning or retries followed inspection of results. Case 2 resembles the earlier nonspecific presentation but is **not** an exact CHECK3 replay or a controlled before/after comparison.

Catalog SHA-256: `248d948e5d797ba7db2e1668b7a3a2a290294ffd8a13268e169249ad88d7a239`.

### Case-by-case results

Full literal inputs, selected IDs/titles/counts, source metadata, hashes, content-review quotes and request telemetry are in each case's `input.json` and `report.json`. Every loaded candidate has a complete UTF-8 `w-<id>.txt` copy in the same directory. Zero-selection cases without candidates open no document files.

| Case | Patient facts | Final documents (ID) | Count | Source-based relevance assessment / generic matches |
|---|---|---|---:|---|
| 01 — nonspecific | F32; mild fatigue and intermittent mild headache for two days; no fever, SOB or known chronic illness | None | 0 | Conservative abstention. **Catalog proposed** *Migraine and other headache disorders* (`w:553ef440f6b7`); content gate rejected generic overlap. Recurrent/pattern-specific disorder information is not clearly supported by this sparse presentation. No generic final selection. |
| 02 — multiple | F24; fatigue/dizziness ~3 weeks, occasional headache, reduced appetite, worse dizziness standing; no SOB or known chronic illness; pregnancy/menstrual history unknown; no tests | None | 0 | Avoids speculative disease-topic selection. No generic or negation-driven proposal observed. This does not prove exhaustive recall. |
| 03 — explicit negations | M40; mild cough for two days **but no fever/no SOB**; no chronic illness, weight loss, night sweats or known tuberculosis exposure | None | 0 | Positive cough alone is nonspecific. No document selected from the negated findings, including mixed polarity and abbreviation. |
| 04 — sparse | Age/sex unknown; “Feels unwell”; duration, other symptoms, history and tests unknown | None | 0 | Insufficient information; appropriate abstention under strict rubric. |
| 05 — no clear topic | F29; brief left-earlobe itching immediately after removing an earring; no hearing change, pain, discharge, rash or other symptoms | None | 0 | No clearly relevant topic identified; did not drift to generic hearing/ear-disease material. Not an exhaustive all-body relevance proof. |
| 06 — watery stools | M35; six loose watery stools since yesterday; thirst and reduced urination; no blood/cough; travel/exposures unknown | *Diarrhoeal disease* (`w:74482c6e742d`) | 1 | Relevant: body directly describes ≥3 loose/liquid stools per day, acute watery duration and fluid loss. Not just symptom-word overlap. Much epidemiology/treatment material is child-focused, so not every passage is applicable to this adult. No causal organism or severity inferred. |
| 07 — animal bite | F28; stray dog bit hand this morning and broke skin; dog/patient vaccination status unknown; no fever | *Animal bites* (`w:4d98794ba49b`); *Rabies* (`w:387f65717b32`) | 2 | Both relevant to the **observed exposure event**, not a rabies diagnosis. Dedicated dog-bite and broken-skin exposure sections supply support. Some topical overlap. Unknown vaccination does not establish infection, regional risk or treatment indications. |
| 08 — hot water | M46; hot water on forearm one hour ago; painful red skin/two blisters; no smoke exposure, breathing difficulty or other affected areas | *Burns* (`w:c29d73ce108e`) | 1 | Directly relevant body definition and hot-liquid mechanism, not incidental lexical overlap. No severity, depth or treatment decision generated. |

### Relevance review method

`outputs/check4-selector/relevance-review.json` records a **coding-assistant offline assessment** for every case and selected document, with source line references, limitations and generic-match flags. The assessment inspected the actual saved source sections, rather than accepting the content model's affirmative verdict as ground truth. It is **not independent clinician adjudication** or a clinical gold standard.

- Final selected documents: **4**, in **3/8 cases**; observed maximum **2**, schema maximum **3**.
- Abstentions: **5/8**. All eight cases technically completed.
- Final selections judged topically relevant by this limited review: **4/4**; no clearly irrelevant/generic final selections observed. **Do not interpret this as a validated precision estimate.**
- Catalog candidates: **5**. **One generic proposal occurred** (case 1), rejected by the content gate. No negation-driven selection observed in this sample; general negation handling remains unproven.
- No WHO Q&A document selected despite both collections being available. No all-corpus relevance labeling or recall estimate, and no clinical outcome measurements.

## Calls, memory boundaries and verification

**13 sequential cloud requests**: eight catalog calls plus five full-content calls. No retry, repair, technical failure or client context truncation. Total measured provider-request time **37.15 seconds** (not whole-program wall time), **166,181 input tokens**, **1,753 output tokens**; maximum request size **69,857 bytes**. Effective sampling was the deployment default, as in the existing provider; temperature was not sent. Reproducibility is therefore not a deterministic-output claim.

Only five candidate files were loaded, 8,031–14,570 characters each (one at a time). Complete source bytes/provenance are saved; in-memory results retain metadata and bounded quotes, not full documents. This is bounded application behavior, not an OS-enforced total process memory limit. No cloud requests from any downstream medical agents were made.

Offline verifier: `scripts/verify_who_selector_check4.py`. It rechecks all eight fixed patient identities, frozen catalog/prompts, every candidate and source byte, literal support quotes, final selected subset/count, summary and successful request telemetry. It verified **five source events and four final selections**; results in `outputs/check4-selector/verification.json`. This verifier does not independently establish semantic relevance.

Tests:
- **45 focused tests passed**: 24 existing selector contracts (updated for v2/max-three), 16 new relevance-gate/evaluation contracts and five new saved-audit tests.
- **728 project tests ran: 727 passed, one existing opt-in cloud skip.**
- **56 upstream tests passed.**
- The workspace virtualenv lacks BeautifulSoup, while system Python lacks pypdf. Initial full-suite attempts failed on these pre-existing environment gaps and their logs are retained. The successful full run used the workspace interpreter with the existing system site-packages directory appended for missing dependencies; **no installation or dependency-file change**. Focused tests and live runs use the workspace environment normally.
- Pre/post integrity inventory: 821 pre-existing files checked, 818 unchanged; only the isolated selector, its existing tests and project OKF changed (root OKF also updated separately). All 577 WHO assets remain byte-identical. See `outputs/check4-selector/integrity.json`. Production source remains byte-identical. Existing dirty changes from prior tasks were preserved, not reverted.

## Why not integrate yet?

The stronger prompt **alone** still admits a generic candidate. Keep the content gate mandatory in any later experiment. Also, the current WHO+AMG experiment uses a bounded body-prefix handoff and lexical-selection provenance. Neither can simply be reused and called an LLM selection:

- *Animal bites* starts with extensive snake-bite content; the accepted dog-bite support occurs around body character **4,765** and **5,381**, beyond a 4,000-character prefix. Whole-document relevance does not guarantee that the passage subsequently sent to Diagnostic is relevant.
- LLM selector provenance must not be presented as a lexical lookup result.
- More heterogeneous, less explicit positive presentations, repeated negation/paraphrase trials, Q&A coverage, abstention/recall review and assessment of the actual bounded downstream evidence are still needed. All such follow-up must remain label-free and isolated unless separately authorized.

**Recommendation: promising enough for further controlled validation, not yet enough to claim consistent adequacy or wire it into WHO+AMG. Production remains unchanged.**

## Reproduce

From `healthcare-agentic-ai`, using the existing environment with project dependencies:

```text
python scripts/evaluate_who_selector_check4.py --output-dir outputs/check4-preview-new
python scripts/evaluate_who_selector_check4.py --live --output-dir outputs/check4-live-new
python scripts/verify_who_selector_check4.py outputs/check4-selector/live
python -m unittest tests.test_who_catalog_selector tests.test_who_selector_relevance tests.test_who_selector_audit
```

Each output directory must be new; existing runs are never overwritten. `--live` uses the existing workspace GPT_SOL configuration and can make at most 32 sequential requests for the eight cases. Omit `--live` for a no-network preparation run. The saved audited live run is already complete; no rerun is needed to verify it.
