# Patient Case RAG â€” Phase 2

Research-only retrieval infrastructure using **synthetic DDXPlus patients**.
No clinical validity, diagnostic correctness, or safety for clinical use is claimed.

## Implemented

- Streaming DDXPlus ZIP/CSV parser and label-free `PatientRepresentation`.
- Training-only ingestion, local `BAAI/bge-small-en-v1.5` embeddings (CPU default),
  normalized vectors and persistent local Qdrant with cosine distance.
- Patient-case semantic retrieval using validation features as queries.
- Strict payload allowlist: `patient_id`, `split`, `text`, `source`.
- Deterministic UUID point IDs and idempotent upserts.
- Embedding manifest checks (model, dimension, normalization, context length,
  representation version and prompt), full payload/vector audits, source-text
  reconstruction, held-out insertion rejection, persistence checks and unit tests.

**Phase 3 implemented:** separate medical-knowledge XML/PDF/text ingestion, license/provenance checks, BGE embeddings and the `medical_knowledge` Qdrant collection. The actual development corpus contains 5 documents / 87 chunks (MedlinePlus and PMC); the WHO adapter is review-gated with zero WHO documents indexed. See [Phase 3 sources, commands and validation](docs/medical_knowledge.md).

**Not implemented:** comprehensive clinical guidelines RAG, LLM generation,
Gemma/Qwen integration, clinical critic, multi-agent orchestration, LangGraph,
human-in-the-loop feedback, feedback memory, or autonomous clinical actions.
No autonomous diagnosis, prescribing, treatment execution or patient-facing advice.

## Environment and commands

Python 3.10+; the existing workspace `.venv` can be reused. Dependencies are declared
in `requirements.txt`: sentence-transformers (including PyTorch) and qdrant-client.
No LangChain, LangGraph, cloud embeddings, API key, GPU, Docker or LLM is required.
The current parent workspace uses **uv** (its venv does not bundle pip): from that
workspace use `uv pip install -r healthcare-agentic-ai/requirements.txt` if needed,
and `uv pip check` to verify compatibility. The pip command below is for pip-based
environments.

From `healthcare-agentic-ai/`, using your environment's Python:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/prepare_ddxplus.py --inspect
python scripts/build_patient_index.py --limit 1000 --allow-download --report outputs/phase2/index-first.json
python scripts/test_patient_rag.py --queries 10 --top-k 5 --report outputs/phase2/retrieval.json
python scripts/audit_patient_index.py --expected-count 1000 --report outputs/phase2/audit.json
python scripts/build_patient_index.py --limit 1000 --report outputs/phase2/index-second.json
```

On Windows, the existing parent environment's executable is
`..\.venv\Scripts\python.exe`; on POSIX it is `../.venv/bin/python`.
From the parent workspace, tests can also run as:
`python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v`.
Scripts insert the project import path themselves and can run from either directory.
From the parent workspace, prefix script paths with `healthcare-agentic-ai/`.
Explicit relative paths such as `--report` and `--storage-path` are resolved from
your current working directory, unlike the project-anchored defaults.

To run all checks sequentially and retain logs plus machine-readable reports:
`python scripts/validate_phase2.py --allow-download`.
The validator upserts without resetting the collection and audits exactly `--limit`
records (default 1,000); a collection containing additional records will fail that
count check. Use a separate `QDRANT_PATH` for an independent bounded run rather
than deleting an index you need. The default output directory is
`outputs/phase2-validation` under the project; reruns overwrite its reports/logs.
Use `--output-dir outputs/phase2-validation-rerun` to preserve earlier evidence.

The first embedding download is opt-in; later runs use cached local files.
If Hugging Face Xet stalls, retry with `HF_HUB_DISABLE_XET=1` (PowerShell:
`$env:HF_HUB_DISABLE_XET = "1"`). This changes download transport, not the model.
A missing model fails clearly, with no substitute embedding implementation.
Run one process at a time against a local Qdrant directory (exclusive lock).
Omitting `--limit` streams the full training split in batches, but embedded Qdrant
keeps vectors in memory: full-corpus capacity has not been validated. Start bounded.
`--reset` explicitly deletes the selected collection; a mismatched manifest is
rejected before reset, so use a new collection/path for a changed embedding setup.

## Data and configuration

Required original files are `release_evidences.json`, `release_conditions.json`,
and `release_{train,validate,test}_patients.zip`. ZIPs contain extensionless CSVs;
`.csv` members are supported too. No full extraction or dataset copy is necessary.

Dataset path precedence: `--data-dir`, `DDXPLUS_DATA_DIR`, project `data/ddxplus`
when metadata exists there, otherwise parent workspace `data/ddxplus`.
Default Qdrant storage: `healthcare-agentic-ai/data/qdrant/patient_cases`;
collection: `ddxplus_patient_cases`. These defaults are anchored to the project.

Environment overrides: `EMBEDDING_MODEL`, `EMBEDDING_DEVICE` (default `cpu`),
`EMBEDDING_BATCH_SIZE`, `QDRANT_PATH`, `QDRANT_COLLECTION`.
CLI `--storage-path` overrides Qdrant storage; indexing `--batch-size` defaults to 16.

## Research data boundary

Only TRAIN records are indexed. Validation/test are query/evaluation inputs only.
`PATHOLOGY` and `DIFFERENTIAL_DIAGNOSIS` are never read to build features or stored
in vector payloads. Never serialize a whole label-containing `PatientRecord`
with `dataclasses.asdict()` for embeddings. Evaluation labels are separate and
opt-in (`include_labels=True`); no diagnosis-based metrics are implemented yet.

The audit checks every stored field, split, vector dimension and deterministic ID,
and compares payloads to regenerated label-free TRAIN source features. Disease
words can legitimately occur in antecedent questions; leakage protection relies
on feature/label separation, not deleting medical words. Arbitrary raw text passed
to the low-level retrieval API must already be label-free.

Patient text preserves original decoded questions/answers, demographics and initial
findings. Unlisted findings are not assumed absent. Unknown codes or malformed
rows fail closed. Texts exceeding the model context are **truncated for embedding**;
counts are logged/reported and original full text remains in Qdrant. No invented
medical summary compensates for truncation.

Scores are **cosine retrieval similarity scores**, not diagnostic probabilities or
clinical confidence. Working retrieval does not establish retrieval quality.
Recall/MRR or diagnosis-label agreement need a separate, leakage-safe offline
protocol; cross-split feature duplication also needs study before quality claims.

## Python API

```python
from rag.patient_parser import DDXPlusParser
from rag.patient_rag import PatientCaseRAG

query = next(DDXPlusParser().iter_patients('validate', limit=1)).patient
with PatientCaseRAG() as rag:
    hits = rag.retrieve(query, top_k=5)
    for hit in hits:
        print(hit['patient_id'], hit['score'], hit['source'], hit['text'])
```

See [validation](docs/validation.md) for actual execution results and artifacts,
and [architecture](docs/architecture.md) for the reference review and future plans.
Phase 3 is retrieval-only and separate from patient cases. See
[medical knowledge documentation](docs/medical_knowledge.md) for exact sources,
licensing, reproduction commands and actual validation evidence.
