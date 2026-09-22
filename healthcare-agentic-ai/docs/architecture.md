# Reference review and incremental architecture

## Reference inspected

Repository: https://github.com/YuxingLu613/DoctorRAG
Historical local clone location: `../../DoctorRAG-reference` from this document's
directory. That clone is not present in the current workspace; the review below
records the earlier inspection, not a fresh verification of the external source.
The clone is not required to run Phase 2.
Commit: `6cb9706b962535efc5e3c68b1f3e6ea7d275aac6`.
Review is static source analysis, not a reproduction of its experiments.
No reference code was imported, executed, or copied into this implementation.
No LICENSE file was present in the inspected tree; public availability alone does
not establish permission to redistribute its code or medical corpora.

## Whole-project map

- `README.md` and overview image describe knowledge + patient analogy and Med-TextGrad.
- `Utils/MedQA_Declarative_Sentence_EN.py` and `_CN.py` use remote LLMs to
  transform question/answer examples into statements. These are generated
  derivatives, not independently verified medical references.
- `Utils/MedQA_Concept_Tagging_EN.py` and `_CN.py` assign ICD-10 labels via
  remote LLM calls with normalization/fallbacks.
- `Utils/Knowledge_Base_Faiss.py`, despite its name, generates SentenceTransformer
  embeddings and NumPy/JSON artifacts; it does not itself build the FAISS indexes
  expected by the main scripts. It loads a full dataframe and accumulates vectors.
- `Scripts/DoctorRAG/DDXPlus_EN_DD.py` and `_FR_DD.py` load separate knowledge and
  patient FAISS indexes and pickled metadata. They classify the query into an
  ICD-10 chapter, filter knowledge, retrieve neighbors, and prompt a remote LLM.
  Their English/French scripts restrict predictions to ten hard-coded conditions
  and fall back to the first condition on an unrecognized answer. The English
  script splits evidence strings but does not decode the release dictionaries.
- `DialMed_DD.py`, `DialMed_TG.py`, and `Muzhi_DD.py` adapt retrieval to dialogue,
  disease, and medication/treatment tasks; use remote generation and sometimes
  permissive LLM-judged label matching. These are not citation-grounding tests.
- `NEJM_QA.py` retrieves similar questions and knowledge, supports a no-RAG
  baseline, and scores answer letters. Its configured artifact extensions and
  loaders are inconsistent (e.g. FAISS reader for a `.npy` path).
- `COD_DD.py` handles conversational generation with patient retrieval and supplied
  knowledge, with lexical/embedding text metrics. Its prompt builds a case string
  but does not include that string in the returned prompt.
- `RJUA_DD.py`, `RJUA_TG.py`, `RJUA_TR.py` retrieve training cases for diagnosis,
  text generation, and treatment advice. DD/TR contain visibly malformed
  indentation; they are not suitable as drop-in modules.
- `Scripts/Med-TextGrad/Med_TextGrad.py` repeatedly generates answers, runs
  knowledge/patient critics, derives textual edits, and updates the prompt.
  `Pairwise-Rater.py` compares completeness/relevance/safety via another LLM.
  This entails many remote calls and is not clinician oversight.
- `Datasets/`, `Knowledge_Base/`, and `Patient_Base/` mostly contain placeholders,
  not the data/index artifacts required to reproduce the pipelines. `Outputs/`
  contains saved example generations/refinement/evaluation JSON, not a reusable
  knowledge corpus or proof of performance on this local dataset.
- `requirements.txt` includes cloud SDKs, FAISS, dataframe/numeric libraries,
  embedding/transformer stacks, and text evaluation libraries. Some script imports
  are not declared there. There is no reusable package API or test suite in the clone.

## Concepts reused, not code

1. Separate retrieval of general medical knowledge and analogous patient cases.
2. A compact textual patient representation for semantic retrieval.
3. Configurable top-k retrieval with source metadata.
4. Explicitly separate knowledge and historical-case context before generation.
5. Later compare no-RAG vs knowledge-only vs dual-retrieval experiments.
6. Later provide a boundary for independent clinical critique and human review.

## Intentionally not copied

Cloud/OpenAI/DeepSeek API calls; FAISS/pickle artifacts; multilingual experiments;
LLM-generated ICD-10 routing and hard-coded diagnostic options; arbitrary diagnosis
fallbacks; treatment automation; raw-code prompting; presumed corpus licenses;
full-dataset in-memory loading; automatic Med-TextGrad loops; text overlap or
LLM ratings presented as clinical reliability. Similarity is not diagnostic
probability, and precision is not a per-patient confidence score.

## Minimal architecture and data boundary

Implemented now:

```
Release dictionaries + streamed split CSV in ZIP
        -> DDXPlusParser
             -> PatientRepresentation -> to_inference_dict() / to_text()
             -> EvaluationLabels (explicit opt-in, separate export)
TRAIN representation -> allowlisted document -> local normalized BGE embedding
                     -> persistent Qdrant -> patient-case semantic retrieval
VALIDATE representation -> same local embedder -> query only (never insertion)
```

`PatientRecord` attaches split-qualified IDs and source archive/member/row
provenance. `PatientRepresentation` has no label fields. Only its allowlisted
serialization is intended for embeddings or prompts. Do not serialize the entire
record with `dataclasses.asdict` for inference: an evaluation-enabled record can
contain labels. Condition dictionaries are used for label normalization, never
to add diagnoses to clinical features. Initial categorical questions without an
answer remain unanswered. Unlisted findings are not invented as negatives.

Phase 2 modules now implemented: `patient_ingestion.py`, `embeddings.py`,
`vector_store.py`, `patient_retriever.py`, `patient_rag.py`, and `patient_audit.py`.
The full payload/vector audit reconstructs label-free source features. A separate
CLI (`scripts/audit_patient_index.py`) tests held-out rejection and persistence. Model manifests reject mismatches;
UUID5 point IDs give idempotent upserts. Context truncation is counted and logged.

## Phase 3 and Phase 4 additions

Phase 3 adds `medical_ingestion/`, `medical_embeddings.py`,
`medical_vector_store.py` and `medical_retriever.py`, preserving a separate
`medical_knowledge` collection and source/license provenance. See
[medical knowledge](medical_knowledge.md) for the actual 5-document/87-chunk corpus.

Phase 4 reuses both existing retrievers without replacing parser, embeddings or
stores. Its explicit flow is:

```text
PatientRepresentation (no labels)
    → isolated Patient Agent → exact-copy validated PatientState
    → PatientCaseRAG + MedicalKnowledgeRetriever (query only)
    → isolated Diagnostic Agent + source-preserving evidence → DiagnosticResult
    → isolated Clinical Critic + structured handoffs → ClinicalCritique
    → saved PipelineResult
    → separate evaluator (validation labels loaded only now)
```

`rag/llm/provider.py` uses GPT-5.6-Sol through the OpenAI/Azure-compatible
Responses API for all three roles. `rag/agents/` owns strict Pydantic contracts,
versioned prompts and deterministic grounding guards; `rag/phase4.py` sequences
calls without shared chat history; `rag/phase4_evaluation.py` is post-inference
only. `scripts/run_phase4.py` saves per-case outputs and an evaluation report with
atomic JSON replacement. Context-budget and validation failures stop closed.

Optional reranking, richer evidence fusion, LangGraph and feedback memory
are not implemented. Phase 5 uses explicit Python state and deterministic routing.
See [Phase 4](phase4.md) for APIs, configuration, tests and limits and
[migration validation](phase4-validation.md) for actual validation outcomes.

## Safety boundaries and future evaluation plan

- Only training features enter the patient collection. Validation/test are query
  and evaluation inputs, never retrievable patient documents. Detect duplicate
  features across splits separately before reporting retrieval performance.
- Conservative policy: PATHOLOGY and DIFFERENTIAL_DIAGNOSIS remain evaluation-only
  for every split, including train. Historical outcome will read "not supplied"
  unless a later explicitly approved, independently sourced outcome policy exists.
- DDXPlus is synthetic; condition associations are dataset metadata, not clinical
  guidelines, drug-safety documents, or independently validated evidence.
- External references need actual source/title/section/type/disease/year and usage
  rights. The admitted Phase 3 sources carry these fields where available; none are invented. Missing metadata stays unknown.
- Citation ID validation checks source existence, not medical entailment. Clinical
  grounding/hallucination need claim-level annotations and human adjudication;
  do not manufacture scores from string overlap or absent reference annotations.
- Report actual retrieval scores and insufficiency, never arbitrary percentages.
- Keep diagnosis accuracy/top-k recall separate from relevance/case quality,
  grounding, hallucination, and citation correctness. Store denominators, failures,
  corpus/model versions, split, and mode A/B/C. No performance claims until measured.
- Human clinician review is required. No autonomous diagnosis, prescribing, or
  patient-facing action tools are planned in this phase.

## Dependency decisions

Phase 1 adds **zero external packages**. Standard-library csv/io/zipfile stream
archives; ast.literal_eval safely reads Python literal lists without executing
code; json loads dictionaries; dataclasses/typing separate contracts; pathlib/os
configure paths; argparse/logging implement CLI diagnostics; unittest/tempfile
provide offline regression tests.

Phase 2 uses `sentence-transformers` to provide the requested
`BAAI/bge-small-en-v1.5` embeddings (general English, not claimed clinically
validated); its PyTorch/transformers dependencies are relatively large, so use
CPU embeddings and small batches on the 16 GB RAM / 4 GB VRAM target.
`qdrant-client` provides local disk-backed collections without a GPU; larger
collections may use a local Qdrant server. Both dependencies are installed and used
in Phase 2. Local Qdrant keeps vectors in memory despite disk persistence; only
bounded indexing has been validated. A narrow local-client compatibility workaround
closes SQLite before collection deletion on Windows (qdrant-client 1.19).
Phase 4 declares the OpenAI SDK for cloud generation and Pydantic 2 for strict
schemas. No orchestration framework was added. The coding agent is separate from
the healthcare runtime; generation has no tool access or shared conversation.
RAG, DDXPlus, embeddings and Qdrant remain local, while selected prompt context
is sent to the cloud. The deployment rejects temperature, so requested 0 is
recorded but omitted from API calls; effective sampling is deployment-default.


## Phase 5 composition

`orchestration/` owns versioned WorkflowState, a retrieve-once evidence snapshot,
application-authored stage tickets, validation/audit events and explicit routing.
The existing Diagnostic Agent supports a restricted revision input; all other
Phase 4 components and grounding rules are reused unchanged. Every accepted
diagnosis passes schema and grounding before review, with at most two clinical
revisions, separate provider repair accounting, progress checks and budgets.
Safety flags block completion and abstention never becomes clinical approval.
`scripts/run_phase5.py` is separate from the preserved Phase 4 baseline runner.
See [Phase 5 architecture](phase5.md) and [validation](phase5-validation.md).
