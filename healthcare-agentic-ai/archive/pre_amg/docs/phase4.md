# Phase 4 — GPT-5.6-Sol healthcare generation

Research prototype using **synthetic DDXPlus**, not clinical advice or validated
clinical decision support. Phase 5, HITL, prescribing and treatment execution are
not implemented. See [historical migration validation](phase4-validation.md) and
[the abstention-contract follow-up](abstention-contract.md) for actual results.
The latter completed one live case with a valid abstention; the critic requested
revisions, so technical completion is not clinical endorsement.

## Architecture

```text
DDXPlus validation features (no ground-truth labels)
    -> Patient / EHR Agent -> exact-copy validated PatientState
    -> Patient Case RAG + Medical Knowledge RAG (local query only)
    -> Diagnostic Agent -> grounded DiagnosticResult
    -> Clinical Critic -> ClinicalCritique
    -> post-inference evaluator (labels loaded only now)

All three agents -> OpenAIProvider -> GPT-5.6-Sol / Responses API
```

`rag/llm/provider.py` contains the only generation implementation, `OpenAIProvider`,
and the small `StructuredLLM` protocol. It uses `OpenAI.responses.create` against
the Azure AI Foundry OpenAI-compatible v1 endpoint. No Chat Completions requests,
coding tools, shared response history, alternative provider or fallback exists.
The coding agent and its `AZURE_OPENAI_*` credentials are separate and never imported.

Local components remain unchanged: DDXPlus parsing/split separation, BGE-small
embeddings, Qdrant patient collection, medical corpus and medical collection.
The patient collection contains training **features only**, not diagnosis labels.
Retrieval returns original text and provenance. No cloud embedding or retrieval
service was added. Only generation is cloud-backed.

**Privacy boundary:** supplied patient features, retrieved excerpts and structured
handoffs are sent to the configured endpoint. Local files do not imply local-only
processing. `store=False` requests no Responses storage; it is not a guarantee
about all provider retention/logging policies. Real patient data would require
appropriate approvals, contracts and privacy safeguards.

## Configuration

Install from the parent workspace with `uv pip install -r healthcare-agentic-ai/requirements.txt`.
The OpenAI SDK is declared in the healthcare requirements; the parent coding-agent
project already depends on it. Normal unit tests require no key or network.

Reuse the workspace's dedicated deployment variables, shown without secrets in
[.env.example](../.env.example):

| Variable | Default / purpose |
| --- | --- |
| `GPT_SOL_API_KEY` | Required deployment credential; never serialized in reports |
| `GPT_SOL_ENDPOINT` | `https://meeting-summarizer-ai.services.ai.azure.com/openai/v1` |
| `LLM_PROVIDER` | `openai`; other providers rejected |
| `OPENAI_MODEL` | `gpt-5.6-sol`; other models rejected |
| `OPENAI_TIMEOUT` | 300 seconds per API request |
| `OPENAI_TEMPERATURE` | Requested experiment value 0; see compatibility note below |
| `OPENAI_MAX_OUTPUT_TOKENS` | 8192, Responses output cap |
| `OPENAI_MAX_INPUT_BYTES` | 100000, application request-byte cap |

The CLI loads only these allowlisted variables from the **parent workspace `.env`**;
existing process variables take precedence, including explicitly empty values.
It never selects the coding agent's deployment. Library callers can export settings
or explicitly call `load_generation_env()` before constructing `OpenAIProvider`.
No credentials are loaded and no network requests occur on module import.

**Verified deployment limitation:** the service rejects `temperature` with HTTP 400,
"Unsupported parameter". The adapter therefore omits it rather than blindly passing
0. Reports distinguish requested temperature 0 from effective
`deployment_default`; deterministic temperature-0 sampling is **not established**.
This is not a switch to another model. Do not claim the strict temperature-0
experiment requirement was achieved.

## Validation and failure behavior

Each request supplies the original role prompt and isolated input plus strict
Pydantic JSON schema in Responses `text.format`. No previous response ID or tools
are sent. Completed assistant text is parsed as strict JSON, then existing exact-copy,
source inventory, provenance, citation and abstention guards run unchanged.
Refusals, incomplete output, unexpected output items, malformed envelopes, fenced
JSON, extra fields, wrong types and invalid references fail closed.

One structured-output repair is permitted by default, with no rejected raw text
or exception details sent back. API retries are disabled. The full request byte
cap includes the schema and any repair on each attempt; evidence is never silently
truncated. This byte cap is not a claim about the model's context window. The
existing BGE embedding truncation policy remains unchanged and is reported separately.

Failure codes distinguish `api_failure`, `connection_failure`, `timeout`,
`provider_failure`, `refusal`, `incomplete_output`, `parsing_failure`,
`agent_validation_failure` and `context_budget`. Pipeline retrieval exceptions
remain stage failures identified by `stage= retrieval`; runner setup and evaluation
failures are reported separately. A missing accepted diagnosis is **not** scored as
a diagnostic mismatch. A critic failure after an accepted diagnosis is still
reported separately from the evaluator's legacy end-to-end yield metrics.

The `phase4-v2-abstention` prompt bundle explicitly requires empty patient/medical
inventories when the Diagnostic Agent supplies neither a primary hypothesis nor a
differential. Prose citations do not count as structured hypothesis claims. This
clarifies the existing exact inventory invariant without changing the schema or
grounding validator, and never requires manufacturing a diagnosis to cite evidence.

## Commands

From `healthcare-agentic-ai/`, with the workspace environment active:

```text
python -m unittest discover -s tests -v
python scripts/run_phase4.py --queries 1 --top-k 1
```

From the parent workspace on Windows:

```text
.venv\Scripts\python.exe -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
.venv\Scripts\python.exe healthcare-agentic-ai\scripts\run_phase4.py --queries 1 --top-k 1
```

`--top-k 1` applies to both local retrievers. Evaluation `--diagnostic-top-k` defaults
to 5. Optional `--data-dir`, `--patient-storage-path`, `--medical-storage-path`, and
`--output-dir` resolve relative to the current directory. Outputs must use a fresh
directory; the runner never resets or rebuilds an index.

The patient store defaults to `healthcare-agentic-ai/data/qdrant/patient_cases`.
The existing medical default is **workspace `data/medical_knowledge/qdrant`**, not
inside the healthcare directory. Run one process at a time against local Qdrant.

There is one explicit paid integration test, running one real validation case.
Enable it only when deliberately requested, rather than in ordinary CI:

```powershell
$env:RUN_GPT_INTEGRATION = '1'
.venv\Scripts\python.exe -m unittest discover -s healthcare-agentic-ai/tests/integration -t healthcare-agentic-ai -v
Remove-Item Env:RUN_GPT_INTEGRATION
```

Per-case artifacts include accepted structured outputs, retrieved evidence and
provenance, stage timing, skipped stages, attempts and sanitized failures. The run
report includes evaluation, usage token counts, API success and request latency.
No key, raw rejected generation, provider exception body or reasoning trace is saved.
Similarity is not diagnostic probability; citation existence is not entailment;
critic self-assessment is not calibrated clinical confidence.
