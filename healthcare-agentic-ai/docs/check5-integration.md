# CHECK5 — default CHECK4 WHO selector + AMG integration

## Delivered

The application factory now implements:

**Patient → Patient Agent → CHECK4 LLM WHO catalog selector/content gate + AMG → source-separated Combined Medical Evidence → Diagnostic → Grounding → Clinical Critic → Safety → Final Decision.**

The existing Patient Case RAG remains separate. WHO selection uses the original label-free patient representation only, after the Patient Agent's exact-copy validation; it never sees AMG output, retrieved case outcomes, diagnoses or benchmark labels. All calls are sequential.

This is a direct integration authorized by CHECK5, **not a new selector experiment or variant**. `rag/who_catalog_selector.py` and `rag/who_selector_relevance.py` remain byte-identical to CHECK4. Production default behavior intentionally changed as requested; the prior CHECK4 recommendation to keep it isolated is historical, not a claim that reliability has subsequently been established.

### Selection and section handoff

- Same existing **569-document catalog**; at most **three WHO documents**, with zero selection valid.
- The exact CHECK4 catalog prompt, full-document content review, schemas, positive/negated-fact rules and literal-quote validator are reused. No lexical matching to select documents, alias expansion, new embeddings or generated medical summaries.
- Selector requests use CHECK4's existing **1,800 output-token / zero-repair** limits through a provider sharing the existing cloud client. Diagnostic/Critic provider settings remain unchanged. No new credential or model configuration.
- Only validated, positively reviewed documents become medical evidence. Each selected file is reopened sequentially and its source/catalog identity and SHA-256 rechecked against the reviewed version.
- Verified source quotes locate their **complete original source sections**, including heading/conditional context. This is exact span extraction from an already selected document, not keyword retrieval or topic ranking. Adjacent/overlapping sections coalesce; unheaded material uses complete paragraphs.
- The new handoff is **not a 4,000-character body prefix**. There is a bounded aggregate limit of **12,000 source characters per selected document** (at most 36,000 across three). Oversized sections and ambiguous quote occurrences fail closed; there is no silent clipping, evidence replacement or synthesized text. Existing provider input-byte limits still apply.
- A document can produce multiple source excerpts/IDs, but there are still at most three selected documents. Combined-contract validation enforces document/character bounds and rejects overlapping spans or mixing legacy lexical and new LLM WHO evidence.

WHO metadata retains the catalog `w:…` document ID, URL, title, type, filename, retrieval timestamp, catalog/document/excerpt SHA-256 and original character spans. Citation IDs remain content-and-span-bound `medical:who:…` IDs. WHO fact sheets, WHO Q&A and AMG passages remain distinct input sections.

**Compatibility detail:** the historical `backend = who-local-lookup-v1` key is retained solely to use the existing Grounding provenance dispatch without editing Grounding. It does **not** describe how new documents were selected. New payloads explicitly declare `selection = llm_catalog_verified_sections`, `selection_method = who-catalog-selector-v2`, `match_type = llm_catalog`, an empty `matched_terms` list and a non-confidence score note. No lexical matches are fabricated.

### Accounting and failure behavior

Every catalog/content call goes through existing EVIDENCE tickets, telemetry validation and global request/byte/time budgets. Calls are not hidden inside retrieval accounting. CHECK4 content/selection calls do not retry; other agents retain their existing bounded repair configuration. The orchestrator's reservation remains conservative when the selector's retry allowance is lower.

A failed WHO selector/content/source contract terminates the workflow safely **before Diagnostic**, without a lexical fallback or bypass of the gate. In this implementation it also stops before AMG retrieval; this is not an AMG retrieval failure. Valid empty WHO selection still allows unchanged AMG retrieval, while both sources empty retain existing abstention and pending review. One immutable combined snapshot is reused across revisions; selector and retrieval calls are not repeated for revisions.

Safe validation error codes and rejected structured content-review quotes are retained in failed retrieval audits. They are audit-only, never evidence. No provider private reasoning is saved. The earliest debug attempt predates this extra error-code logging and is explicitly described below.

## What stayed unchanged

Byte-verified against the **existing dirty workspace at CHECK5 start**, not just Git HEAD:

- AMG adapter/service/vendor Python, retrieval query/aliases, top-k defaults, distance gate, embeddings and source content.
- Patient Case RAG.
- CHECK4 selector and content-gate source, prompts and schemas.
- `rag/agents/grounding.py`, Phase 6 grounding/diagnostic/reference validators, Critic agent code, Safety code/policy and final-decision/human-review rules.
- WHO lexical lookup implementation/crawlers and all **577 existing WHO assets** (catalogs, documents and related collection files); no corpus writes or indexing.
- Dependencies and credentials. No benchmark label or hidden diagnosis was loaded or injected.

The existing source-separated input delegate is reused for Diagnostic/Critic presentation. Their clinical validation logic and policy are unchanged. The historical lexical WHO experiments/standalone lookup remain available as historical code; **the default application factory, live demo, batch and manual application runner do not use them**.

The integrity inventory covers 907 pre-existing project files: **895 unchanged, 12 intentionally changed, none deleted**. Root `okf.yaml` was also updated outside that project inventory. See `outputs/check5/integrity.json`. AMG Chroma SQLite runtime bookkeeping was not inventoried; no index rebuild, parameter change or byte-identity claim for that bookkeeping is made.

## Completed real end-to-end run

Artifact directory: **`outputs/check5/end-to-end/`**.

Input: `data/manual_check5_hot_water.txt`, an **assistant-authored synthetic observation-only vignette**, not a real patient or a DDXPlus row. Age 46, male; hot water spilled on forearm one hour earlier; painful red affected skin and two blisters; no breathing difficulty, no smoke exposure, no other affected areas reported. No diagnosis/expected answer was supplied. The adapter uses a content-bound manual ID, never a fabricated benchmark ID:

`manual:4ba11c113174a2e968f6889fd4ad4d412bf104049bce49afe7a257c294a4c3f7`

| Measurement | Actual result |
|---|---|
| Workflow time | **49.20 seconds**, excluding resource setup |
| Cloud calls | **5**, sequential; **zero repairs** |
| Call order | Patient → WHO catalog → WHO content review → Diagnostic → Critic |
| WHO selected | **Burns**, document **`w:c29d73ce108e`** |
| WHO source | `https://www.who.int/news-room/fact-sheets/detail/burns` |
| Document SHA-256 | `53dbe9c10198641ee5b5bc5666204bb544dd6eac0f0d03f48310c6df392bcd87` |
| Actual WHO content supplied | Complete **Overview** section, original character span **474–854**, 380 characters; not a blind prefix |
| AMG | One unchanged accepted **Heat Illness** passage, `medical:medlineplus:topic:280:lab:1:chunk:1`, distance **1.0625395775** at unchanged **1.10** gate |
| Combined status | **BOTH_AVAILABLE** |
| Diagnostic medical IDs cited | **1 WHO**, **0 AMG** |
| Grounding | Structural checks passed; medical entailment **not established**; manual patient-reference audit has **9 unassessed clauses** |
| Safety | Assessed, **BLOCK**, `skipped_critic_block` |
| Final decision | **BLOCK**, human review **pending**, proposal withheld |
| Technical failure | **None** |

Safety did execute. No separate Safety cloud call was needed because the unchanged deterministic rule blocks on Critic safety flags. Those flags concern unknown severity-related information, limits of negative respiratory/exposure findings, and a weak differential that could distract from the reported mechanism. This report does not adjudicate whether those flags are clinically justified.

The WHO source covers the reported hot-liquid mechanism. The AMG Heat Illness passage is not assumed useful merely because the retriever accepted it; Diagnostic did not cite it. Citation presence is not clinical entailment or improved diagnosis. No clinical approval or simulated human review was produced.

Actual Diagnostic/Critic source inventories were captured and fingerprint-checked against the immutable snapshot. `case_state.json`, `case_combined.json`, `case_decision.json`, `agent_input_audit.json`, `report.json` and `verification.json` retain the evidence, source identities and gate results. The offline verifier independently rechecked one WHO source, one AMG source, one diagnostic version, one critique, one Safety assessment, selector accounting and the final decision.

### Verification of content beyond the old prefix

The completed hot-water run's relevant section happens to occur early in its document. **It alone does not demonstrate a late-section handoff.** That boundary is verified by:

1. Default-workflow tests whose only supporting WHO section starts after character 4,000; the exact late text reaches Diagnostic, Critic and Safety unchanged.
2. An offline replay of the already-saved, valid CHECK4 dog-bite content judgments, with **no new selector call, retrieval experiment or tuning** (`outputs/check5/section-replay.json`). The **Animal bites** support quotes occur at body offsets **4,765 and 5,381**. The new adapter preserves the complete Dog bites section at original file span **3334–6397**. The saved Rabies judgments yield exact source sections at **1465–3824** and **5900–8919**. Source checksums, quote inclusion and unchanged Grounding context validation passed.

This offline replay validates the handoff implementation, not a successful live dog-bite selector run.

## All live attempts — including failures

**Six attempts / 24 cloud requests total**, all retained in `outputs/check5/`; no concurrent calls or concealed retry loop. These are integration/debug traces, **not a new tuned selector evaluation or a representative performance sample**.

| Directory | Input | Requests | Outcome |
|---|---|---:|---|
| `live/` | Manual dog-bite observations | 4 | Content-review agent validation failure; BLOCK/pending; specific failed-quote code not persisted in this earliest trace |
| `live-complete/` | Same dog-bite input | 4 | **Failed despite directory name**; two nonliteral-source-quote validation errors; BLOCK/pending |
| `batch-live/` | Existing `ddxplus:validate:1`, `include_labels=False` | 4 | Two nonliteral-patient-quote validation errors; BLOCK/pending; Diagnostic not reached |
| `live-diagnostic-trace/` | Same dog-bite input | 4 | Source-quote mismatch confirmed in structured audit; BLOCK/pending |
| `live-final/` | Same dog-bite input, final CHECK4 provider limits | 3 | One nonliteral-source-quote error, no repair; BLOCK/pending |
| `end-to-end/` | Hot-water observations, final implementation | 5 | All gates completed, no technical failure; Safety BLOCK/pending |

The first four debug traces inherited the application's ordinary 8,192-token/one-repair provider settings. Integration was corrected to reuse CHECK4's existing 1,800-token/zero-repair limits, without changing selector prompts, schemas, matching or medical content. The final two traces use those limits.

One concrete failure: the model quoted `Dog bites account for tens of millions of injuries annually.` but the source continues with **a semicolon**, not that period. The unchanged exact-quote gate correctly rejected it; the integration did not normalize or silently accept the mismatch. The repeated dog-bite failures and DDXPlus patient-quote failure remain known usability limitations. **A completed hot-water example is not proof of consistent selector reliability.** No fix to CHECK4 semantics was authorized or introduced here.

## Tests

- Before changes: **728 project tests**, 727 passed, one existing opt-in cloud skip.
- Final focused suite: **143 passed**, including **24 new CHECK5 tests** (19 workflow/section tests and five audit/provider tests).
- Final full project suite: **752 tests**, **751 passed**, one existing opt-in cloud skip.
- Upstream AMG suite: **56 passed**.
- Completed live run reverified offline: `outputs/check5/end-to-end-reverification.json`.

Coverage includes max-three IDs, source-separated inputs, late-section delivery, exact WHO/AMG provenance, both-empty/WHO-only/AMG-only/both-available, quote rejection, source mutation, bounds/ambiguous anchors, labels rejected before calls, selector budget accounting, retrieve-once revision behavior, unchanged BLOCK/HUMAN_REVIEW routes and saved-audit tampering.

Full-suite command used the workspace Python with the **existing system site-packages appended** for BeautifulSoup, as in CHECK4; no packages were installed. Focused tests and live runs used the workspace environment normally. Existing AMG-focused test fixtures now explicitly return a valid empty CHECK4 selection; their safety assertions remain intact. The one prior call-count assertion changed from one to two for Patient plus WHO selection, not by skipping tests.

Logs: `baseline-tests.log`, `initial-regression-tests.log`, `integration-tests-first.log`, `focused-tests.log`, `regression-tests.log`, `upstream-tests.log`. Intermediate failures were retained; the final totals above are from the final implementation.

## Exact source/documentation changes

All paths below are relative to the workspace root.

### Modified existing files

| File | Change |
|---|---|
| `healthcare-agentic-ai/application/workflow.py` | Default CHECK4 WHO + AMG service and existing source-separated input delegate |
| `healthcare-agentic-ai/rag/combined_medical_evidence.py` | Validate new WHO document-count, section-size and overlap/source-selection constraints |
| `healthcare-agentic-ai/rag/who_provenance.py` | Additive dispatch to truthful LLM-selected section payload; legacy payload remains supported |
| `healthcare-agentic-ai/orchestration/phase6/orchestrator.py` | Account/budget EVIDENCE-stage cloud calls; preserve failed retrieval audits; no safety/routing policy changes |
| `healthcare-agentic-ai/scripts/run_phase6.py` | Default backend report identifier and persisted combined-evidence artifacts |
| `healthcare-agentic-ai/scripts/run_manual_who_amg.py` | Use default factory; no-cloud preview parses input only, never lexical fallback |
| `healthcare-agentic-ai/scripts/verify_who_amg_run.py` | Verify new section/catalog/selection/accounting contracts and batch artifact filenames |
| `healthcare-agentic-ai/demo/terminal.py` | Show WHO selection/section audit and nested AMG audit; avoid old audit-key crash; clarify local-only probe |
| `healthcare-agentic-ai/tests/amg_helpers.py` | Explicit valid-empty WHO response in AMG-focused fixtures |
| `healthcare-agentic-ai/tests/test_final_decision.py` | Update no-evidence expected calls to Patient + WHO selection |
| `healthcare-agentic-ai/README.md` | Current architecture, default run/verification instructions and limitations |
| `healthcare-agentic-ai/okf.yaml` | Synchronize current composition and CHECK5 results |
| `okf.yaml` | Synchronize workspace composition and CHECK5 results |

### Added files

| File | Purpose |
|---|---|
| `healthcare-agentic-ai/rag/selector_evidence_service.py` | Sequential default composition using unchanged CHECK4 selection/content gate and AMG |
| `healthcare-agentic-ai/rag/who_sections.py` | Exact quote-anchored source sections and strict new WHO payload |
| `healthcare-agentic-ai/tests/test_check5_pipeline.py` | 19 new integration tests |
| `healthcare-agentic-ai/tests/test_check5_audit.py` | Five new audit/provider tests |
| `healthcare-agentic-ai/data/manual_check5.txt` | Observation-only dog-bite input used in retained failed debug attempts |
| `healthcare-agentic-ai/data/manual_check5_hot_water.txt` | Observation-only completed end-to-end example |
| `healthcare-agentic-ai/docs/check5-integration.md` | This implementation/change/run report |
| `AHS_RUN.txt` | Requested plain-text Windows commands, no `start`, comments or Markdown fences |

Generated audit files, logs, edit/replay helpers and all six run directories are confined to `healthcare-agentic-ai/outputs/check5/`. `files-changed.json` inventories the deliverables and every generated audit file. No pre-existing files were deleted or unrelated dirty work reverted.

## Run it

From the workspace root, run the exact plain-text commands in **`AHS_RUN.txt`**. It uses the existing workspace virtualenv and the completed manual hot-water example. A new timestamped output directory is created automatically. Model output is nondeterministic; a future quote-validation failure remains possible and must not be bypassed.

From `healthcare-agentic-ai`, the default batch route is also available:

```text
python scripts/run_phase6.py --queries 1
python scripts/verify_who_amg_run.py outputs/check5/end-to-end/case_state.json
```

For your own observation-only input, copy the five-section manual input format and run `scripts/run_manual_who_amg.py <case-file> --live`. Without `--live`, it only parses the input. Existing GPT_SOL settings, cached models, patient index, AMG snapshots and WHO corpus are required; no indexing, credential changes or model download is performed by these commands.
