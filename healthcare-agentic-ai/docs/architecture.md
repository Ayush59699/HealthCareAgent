# Active architecture and contracts

## Composition

`application.workflow.create_workflow` is the supported composition root. Both batch and demo entrypoints use it. It injects `AMGEvidenceService` into the established Phase 6 orchestrator rather than selecting the old focused medical retriever. The versioned workflow engine is reused because it already validates stage tickets, immutable evidence, provider accounting, revisions and safety dispositions. Its historical constructors remain available for regression compatibility.

1. `DDXPlusParser.iter_patients(..., include_labels=False)` yields synthetic patient representations with no hidden outcomes.
2. Patient Agent copies supplied observations exactly. Existing validation rejects invented facts and treating missing findings as absent.
3. Patient Case RAG uses its existing BGE/Qdrant training index and excludes case outcomes. This is an analogy source, not medical evidence.
4. AMG medical evidence adapter opens the **existing** `medlineplus_lab` published snapshot. It calls the supplied `Retriever.search` unchanged. No graph construction, external searches, LLM extraction, source synthesis, new reranking, new index or new benchmark is introduced.
5. A single immutable snapshot binds both evidence categories to a content fingerprint. Retrieval occurs once, not once per revision.
6. Diagnostic Agent and Clinical Critic receive the full patient facts and accepted evidence only. Existing reference inventory and inference-disclosure rules apply. AMG text is context, never an automatically accepted diagnosis.
7. Grounding validates citation existence, category, native provenance and bounded patient-reference consistency. This does **not** prove clinical entailment.
8. Safety reconstructs and fingerprints its entire input, including the exact evidence snapshot. Critic safety flags can deterministically block without a redundant semantic safety API request. Otherwise semantic safety is required. Final, abstention, unresolved, block and human-review outcomes retain existing contracts.

The supported AMG workflow abstains before diagnosis if AMG returns no accepted passages. This is deliberately stricter than the historical focused path, which allowed patient-only hypotheses after empty retrieval. When passages exist, the existing inference-disclosure and downstream safety rules still guard hypotheses not backed by medical citations.

## Native evidence and trust boundary

The legacy medical payload validator expects deterministic UUIDs and old chunking versions. Forcing AMG into that shape would invent provenance. The new source-specific contract instead preserves:

- `medical:medlineplus:topic:<id>:lab:1:chunk:<index>` as citation ID;
- original title, URL, complete passage text and native metadata;
- source archive/member/checksum, parent ID, source spans, chunk checksum and chunker version;
- collection and snapshot identity, raw squared-L2 distance and acceptance route.

The loader checks published artifact hashes and embedding identity before retrieval. The adapter retains only ID-to-hash mappings, not a duplicate corpus, and verifies each returned passage's **text and full native metadata** against the published chunks artifact. Agent evidence is revalidated on snapshot creation, thawing, grounding and safety handoffs. Duplicate IDs, changed text/URLs, inconsistent distances and rejected candidates fail closed.

The existing `similarity` field is populated with **negative squared-L2 distance** for higher-is-closer compatibility. It is never a probability; original distance and acceptance are retained. An exact alias source lookup is reported separately and may bypass the exploratory distance gate. This does not mean the passage supports a particular diagnosis.

Rejected AMG candidates and resolution details remain in `medical_retrieval.upstream` audit data, outside all agent contexts. AMG synthesis and web results never enter the evidence snapshot.

## Query and resource bounds

AMG accepts at most 256 embedding tokens and 4,000 query characters. The adapter assembles one literal query from presenting facts, symptoms and antecedents, including complete facts while they fit. It deduplicates identical strings and audits the count omitted. It does not introduce disease predictions or manual retrieval synonyms. **Agents receive all original facts**, including those omitted from retrieval.

One BGE and one MiniLM model are used because their existing vector spaces and normalization differ. Replacing one with the other would invalidate an index. Retrieval/cases run sequentially; no cross-encoder, parallel graph or duplicate corpus is loaded. Chroma exposes no public close method for its process-shared persistent client, so the adapter never uses private reset/stop calls that could invalidate other readers. Patient Qdrant and cloud clients retain context-managed cleanup. Startup rejects missing patient indexes and less than 2 GB available system memory. The check is not a hard RAM cap; upstream CPU threads are bounded to four.

## Source organization

The supplied folder was moved intact to `vendor/amg`, including local data, hash-pinned stores and its unchanged retrieval implementation. Only its unrelated coding-agent script was archived. No import of its graph/answer pipeline is needed. A narrowly scoped sibling-import loader accommodates its original standalone scripts and rejects conflicting module names. The vendored snapshot hashes still match, so no rebuild was needed.

Previous documentation and output artifacts live in `archive/pre_amg`. Legacy medical implementation modules/tests stay available for historical compatibility, not application routing. Restoring old experiment paths may be necessary to reproduce historical byte/path integrity manifests; archived manifests were not rewritten to masquerade as current experiment runs.


## Explicit grounding and terminal handoff (FINAL completion)

`orchestration/phase6/grounding.py` now produces `GroundingResult` using the existing
hard gates: diagnostic/patient/snapshot fingerprints, per-claim JSON paths and separated
current-patient, analogous-case and medical references. Missing medical support remains
explicit. Citation presence is not semantic entailment. The existing independent Critic
receives this report under `GROUNDING_RESULT` and must still evaluate the actual evidence.
No retrieval or model call is added by the report.

Both supported CLIs call `application.decision.build_final_decision` on the completed
Phase6WorkflowState. This revalidates patient/evidence identity and the exact safety
assessment; ALLOW also requires the existing critic routing to resolve to FINAL. It
cannot promote CONTINUE alone. BLOCK and HUMAN_REVIEW withhold the diagnostic proposal,
keep a genuine pending human-review state and never record approval. Failure and early
abstention do not fabricate a safety assessment. The full original workflow is kept
separately as the local reviewer audit; its fingerprint binds the final artifact.

No duplicate agent schemas were introduced: PatientState, RetrievedEvidence,
DiagnosticResult, ClinicalCritique and SafetyAssessment are reused. Only the previously
missing GroundingResult and FinalDecision contracts were added. Historical Phase 6
state schema and safety policy stay unchanged. `scripts/verify_amg_run.py --final-decision`
can validate both saved files offline. See [validation and limitations](final-system-validation.md).
