# CHECK3: live WHO + AMG run on the supplied vignette

## Bottom line

The experimental combined workflow **ran successfully as software**, but **did not produce a validated or releasable diagnosis**. Both evidence sources reached the Diagnostic Agent and independent Critic. WHO contributed no Diagnostic hypothesis citations; AMG contributed two. Safety returned **BLOCK**, the diagnostic proposal was withheld, and human review remains pending.

Input: `../prompts/check3.txt` (the actual workspace location, rather than `/prompt/check3.txt`). It describes a 24-year-old woman with approximately three weeks of fatigue and dizziness, positional dizziness, occasional headaches and reduced exercise tolerance; no fever, chest pain or shortness of breath; no known chronic illness or current medication; no available laboratory results.

No expected diagnosis was supplied or injected. No DDXPlus row was substituted for this vignette.

## Execution

```text
python scripts/run_manual_who_amg.py ../prompts/check3.txt --live --output-dir outputs/check3/live
```

Artifacts: `outputs/check3/live/`:

- `input.json`: source checksum, content-bound manual identity, exact line-to-field mapping and label-free input.
- `case_combined.json`: source-separated evidence and full retrieval audit.
- `case_state.json`: versioned Patient/Diagnostic/Critic/Safety workflow state.
- `agent_input_audit.json`: actual source-section inventories and evidence fingerprints seen by Diagnostic and Critic.
- `case_decision.json`: withheld final proposal, grounding, critic, safety and review state.
- `verification.json`: downloaded WHO/native AMG source-byte and existing gate/handoff verification.
- `report.json`: availability, queries, citations, checks and outcome.

This was one live combined run, **not** an A/B/C comparison. Runtime was **42.94 seconds**, with **three sequential cloud requests**: Patient, Diagnostic, Critic. There were no provider repairs or technical failures. Safety ran its deterministic checks and correctly skipped semantic generation after the Critic's blocking flags. No parallel calls were made.

## What was retrieved and actually used

| Source | Returned to agents | Diagnostic citations | Finding |
|---|---:|---:|---|
| WHO | 3 documents: one Fact Sheet, two Q&A | 0 | Available, but not useful support for the leading hypotheses in this run |
| AMG | 3 accepted passages, all from the same ME/CFS topic | 2 distinct passages | Used for limited symptom overlap and a duration mismatch, not confirmation |
| Combined | 6 individually identified records | 2 AMG, 0 WHO | BOTH_AVAILABLE describes availability only |

### WHO

The unchanged local lookup returned:

1. **Headache disorders: How common are headaches?** — Q&A, partial match on `headaches`.
2. **Radiation: The known health effects of ultraviolet radiation** — Q&A, partial match on `known`.
3. **Chronic obstructive pulmonary disease (COPD)** — Fact Sheet, partial match on `chronic`.

The words `known` and `chronic` occurred in **“No known chronic illness.”** The existing lookup is lexical and not negation-aware: these matches are **not evidence that this patient has COPD or ultraviolet-radiation exposure**. This is an observed evidence-quality limitation. No new search rule, stopword, query tuning or forced topic was introduced after seeing it.

The Diagnostic Agent explicitly stated that the COPD, general-headache and ultraviolet-radiation excerpts did not support its leading hypotheses. It did not cite any WHO document in a structured hypothesis claim. The Critic could identify and cite all three WHO records when explaining their lack of diagnostic support. Thus WHO plumbing worked, including both collections, but useful WHO contribution to the diagnostic differential was not demonstrated.

### AMG

The unchanged top-five retrieval produced:

| Topic / chunk | Squared-L2 distance | Disposition at unchanged 1.10 gate |
|---|---:|---|
| ME/CFS / chunk 2 | 0.932189 | Accepted |
| ME/CFS / chunk 1 | 1.051373 | Accepted |
| ME/CFS / chunk 3 | 1.082877 | Accepted |
| Legionnaires' Disease | 1.104783 | Rejected; audit only |
| Fatigue | 1.107495 | Rejected; audit only |

These distances are retrieval measures, not diagnostic confidence. Rejected records never entered the Diagnostic input. Three accepted chunks from one topic are not three independent corroborating sources.

The model cited ME/CFS chunks 1 and 2 for symptom overlap and, importantly, the retrieved six-month duration requirement. It recognized that the vignette describes only about **three weeks**, so the retrieved material does **not** establish ME/CFS.

## Diagnostic and review results

For software audit only, the withheld Diagnostic output proposed:

- Anemia, including possible iron-deficiency anemia — explicitly unconfirmed patient-only inference.
- An orthostatic process — explicitly unconfirmed patient-only inference.
- ME/CFS — marked poorly supported, with the duration mismatch acknowledged.

**These are not diagnoses endorsed by this report or released by the system.** No supplied medical passage supported the anemia or orthostatic hypotheses. The Critic objected to making anemia the leading explanation and to insufficient support for the iron-deficiency subtype, limited differential breadth, and missing objective/acuity information.

Grounding:

- Schema/source/reference integrity: passed.
- Seven claims: `unsupported_by_medical_evidence`.
- Four claims: `reference_present_not_verified`.
- Two hypotheses had no medical references.
- `medical_entailment=not_established`.
- The existing DDXPlus-oriented patient-reference semantic heuristic reported **unassessed**, with 14 unassessed clauses and no applicable checks. Exact input copying, ID existence and source integrity still passed. **Do not interpret structural grounding as verified semantic correctness for this manual vignette.** The independent Critic reviewed the actual text.

Critic: `revision_required`, with three safety flags concerning unavailable acuity/red-flag information and potentially premature diagnostic anchoring.

Safety: **BLOCK**, reasons `critic_safety_flags` and `missing_medical_reference`. The latter independently requests HUMAN_REVIEW; the former dominates with BLOCK. `semantic_status=skipped_critic_block` is the existing fail-closed policy, not a skipped Safety stage.

Final: **BLOCK**, `diagnostic_proposal=null`, review `pending`, no human approval or automatic scheduling. No retrieval availability or citation count overrode the safety result.

## Verification and changes needed for this run

The original entrypoint accepted only benchmark row identities. A small manual-input compatibility layer was added rather than labeling this vignette as `ddxplus:validate:1` or altering any benchmark data:

- New `rag/manual_patient.py`: strict bounded parser for this explicit five-section format. Source statements, negations, duration and missing laboratory information remain literal values. It rejects malformed/missing sections and unknown nested fields. It is not a general free-text clinical parser or LLM fact extractor.
- New `scripts/run_manual_who_amg.py`: one-case opt-in runner using the same combined evidence factory, agents, safety and handoff. Without `--live`, it performs retrieval only. Records the actual Diagnostic/Critic source inventories.
- New `tests/test_manual_patient.py`: nine offline input/identity/integration tests.
- Minimal change in `rag/agents/grounding.py`: patient identity admission now also accepts `manual:<SHA-256 of the exact label-free inference payload>`. An arbitrary manual ID or one bound to different patient facts is rejected. Existing DDXPlus identity handling, evidence/reference validation and all clinical/safety rules are unchanged.
- Added this report and synchronized the project/workspace OKF manifests.

The new parser uses section names as the existing Evidence question and the exact supplied sentence as its value, e.g. `Symptoms = No fever`. Both retrieval queries contain the same literal patient facts here, with no omitted query facts. This input-format adaptation is disclosed; it is not a claim of equivalence to DDXPlus-encoded questions. Existing agents retain their research-oriented prompts and limitations.

WHO/AMG retrievers, their parameters, corpora, indexes, WHO lookup/crawlers and downloaded documents were not edited. No diagnosis, critic, safety or abstention policy was changed to make this case pass.

Verification established:

- All three WHO records matched the selected downloaded source bytes/spans/checksums.
- All three accepted AMG records matched native published source chunks and metadata.
- The actual grouped evidence delivered to Diagnostic and Critic exactly matched the immutable snapshot fingerprints.
- The Diagnostic/Critic references, Safety assessment and final decision revalidated from saved artifacts.

Tests:

- Manual-input + combined-evidence + AMG integration: **47 passed** (including nine new manual-input tests).
- Full regression: **683 run, 682 passed, one existing opt-in cloud skip**. Same existing Python environment workaround as the preceding experiment (project Python plus already-installed system packages); no dependency installation/change.
- `git diff --check` passed.

Logs: `outputs/check3-tests.log`, `outputs/check3-regression.log`, `outputs/check3-live.log`.

## Conclusion / next step

**Both components are connected and provenance-preserving; they did not jointly establish a diagnosis for this case.** WHO's partial keyword matches were weak, and AMG's accepted evidence was concentrated on one poorly fitting chronic-fatigue topic. The critic and safety stages withheld the result rather than treating these retrieval hits as confirmation.

Keep the run as a negative/limited-usefulness example. If further work is authorized, investigate negated/generic-word WHO matches, medical evidence coverage for nonspecific presentations, and semantic reference checking for manual inputs. Do not weaken safety, force WHO citations, or change retrieval thresholds simply to obtain an ALLOW result. No such tuning was performed here.
