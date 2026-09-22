# Healthcare Agentic AI & Mini-CC Execution Guide

This document provides complete, step-by-step instructions and commands to set up, test, index, and run the **Healthcare Agentic AI** pipeline (Phases 1–5) and the **Mini Terminal Coding Agent** (`codeagent.py`).

---

## Technical Overview

1. **Mini Terminal Coding Agent (`codeagent.py`)**:
   - A command-line coding assistant powered by Azure OpenAI / OpenAI Responses API.
   - Capable of reading/writing workspace files and executing terminal commands.

2. **Healthcare Agentic AI System (`healthcare-agentic-ai/`)**:
   - **Phase 1 & 2**: DDXPlus synthetic patient parser, local CPU embeddings (`BAAI/bge-small-en-v1.5`), and embedded Qdrant vector database (`ddxplus_patient_cases`).
   - **Phase 3**: Medical Knowledge ingestion engine (PMC/MedlinePlus text & XML into `medical_knowledge` Qdrant collection).
   - **Phase 4**: Multi-agent pipeline (`Patient Agent` -> `Dual RAG` -> `Diagnostic Agent` -> `Clinical Critic`) using `gpt-5.6-sol`.
   - **Phase 5**: Stateful deterministic orchestrator enforcing safety gates, versioned critiques, maximum 2 revisions, and explicit abstention/safety-blocked handling.

---

## 1. Prerequisites & Environment Setup

### Required Tools
- **Python**: 3.10+ (Project verified on Python 3.13)
- **Package Manager**: `uv` (recommended) or standard `python -m venv` / `pip`

### Step 1.1: Environment File Configuration (`.env`)
Create or edit `.env` in the project root (`mini-cc/.env`):

```ini
# Azure OpenAI Credentials (Used by codeagent.py & Healthcare Agents)
AZURE_OPENAI_API_KEY="your-azure-api-key"
AZURE_OPENAI_DEPLOYMENT="gpt-6-astra"
AZURE_OPENAI_ENDPOINT="https://your-endpoint.services.ai.azure.com/openai/v1"

# LLM Settings for Phase 4 & Phase 5 Agents
GPT_SOL_API_KEY="your-azure-api-key"
GPT_SOL_ENDPOINT="https://your-endpoint.services.ai.azure.com/openai/v1"
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-5.6-sol
OPENAI_TIMEOUT=300
OPENAI_TEMPERATURE=0
OPENAI_MAX_OUTPUT_TOKENS=8192
OPENAI_MAX_INPUT_BYTES=100000
```

---

## 2. Running the Mini Terminal Coding Agent (`codeagent.py`)

To run the interactive CLI agent:

```bash
# Using uv:
uv run ./codeagent.py

# Using active virtual environment:
python codeagent.py
```

*Type your coding request at the `USER:` prompt, or type `exit` to quit.*

---

## 3. Healthcare Agentic AI Pipeline Execution (Phases 1 to 5)

All commands below can be run from the root project directory (`mini-cc/`).

---

### Step 3.1: Run Unit & Integration Tests (Offline, No LLM required)

Verify that the local environment, parsers, RAG modules, and agent schemas pass all 185 unit tests:

```bash
uv run python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v
```

---

### Step 3.2: Dataset Inspection & Preparation (Phase 1)

Inspect DDXPlus synthetic patient dataset structure:

```bash
uv run python healthcare-agentic-ai/scripts/prepare_ddxplus.py --inspect
```

---

### Step 3.3: Ingest & Index Patient Cases into Vector Store (Phase 2)

Downloads embeddings model (`BAAI/bge-small-en-v1.5`) on first run and builds the Qdrant local patient vector store (indexing a bounded subset of 1,000 training records for fast verification):

```bash
# RUN THIS ONCE ONLY TO BUILD THE VECTOR STORE   (1,000 cases)
uv run python healthcare-agentic-ai/scripts/build_patient_index.py --limit 1000 --allow-download --report healthcare-agentic-ai/outputs/phase2/index-first.json
```

```bash
# Test Patient RAG retrieval (query with 10 validation cases)
uv run python healthcare-agentic-ai/scripts/test_patient_rag.py --queries 10 --top-k 5 --report healthcare-agentic-ai/outputs/phase2/retrieval.json

# Audit index data integrity & leakage prevention checks
uv run python healthcare-agentic-ai/scripts/audit_patient_index.py --expected-count 1000 --report healthcare-agentic-ai/outputs/phase2/audit.json
```

---

### Step 3.4: Ingest & Index Medical Knowledge (Phase 3)

Ingest medical documents (MedlinePlus & PubMed Central) into the `medical_knowledge` Qdrant collection:

```bash
# Download reference medical knowledge data
uv run python healthcare-agentic-ai/scripts/download_medical_data.py

# Index medical knowledge documents into vector database
uv run python healthcare-agentic-ai/scripts/build_medical_index.py

# Test Medical RAG search query
uv run python healthcare-agentic-ai/scripts/test_medical_rag.py

# Audit Medical Knowledge vector store
uv run python healthcare-agentic-ai/scripts/audit_medical_index.py
```

---

### Step 3.5: Run Phase 4 Pipeline (Sequential Multi-Agent Execution)

*Requires active LLM API credentials in `.env` (`GPT_SOL_API_KEY` & `GPT_SOL_ENDPOINT`).*

Runs patient features through Patient Agent -> RAG -> Diagnostic Agent -> Clinical Critic:

```bash
# Run 1 validation patient case through Phase 4
uv run python healthcare-agentic-ai/scripts/run_phase4.py --queries 1 --top-k 1
```

---

### Step 3.6: Run Phase 5 Pipeline (Stateful Orchestrator with Safety Blocking)

*Requires active LLM API credentials in `.env`.*

Runs full stateful research orchestration with grounding gates, revision loops (max 2), and mandatory safety-stop handling:

```bash
# Run Phase 5 orchestrator on 1 validation patient case
uv run python healthcare-agentic-ai/scripts/run_phase5.py --queries 1 --top-k 1 --max-requests 14 --max-seconds 600
```

---

## 4. Summary Quick-Reference Table

| Task | Command | Notes |
|---|---|---|
| **Run CLI Agent** | `uv run ./codeagent.py` | Interactive terminal agent |
| **Run Unit Tests** | `uv run python -m unittest discover -s healthcare-agentic-ai/tests -t healthcare-agentic-ai -v` | Offline, tests all 185 tests |
| **Index Patients** | `uv run python healthcare-agentic-ai/scripts/build_patient_index.py --limit 1000 --allow-download` | Local CPU embedding + Qdrant |
| **Index Medical Knowledge** | `uv run python healthcare-agentic-ai/scripts/build_medical_index.py` | Indexes MedlinePlus/PMC data |
| **Run Phase 4 Pipeline** | `uv run python healthcare-agentic-ai/scripts/run_phase4.py --queries 1 --top-k 1` | Requires Azure LLM API Key |
| **Run Phase 5 Orchestrator** | `uv run python healthcare-agentic-ai/scripts/run_phase5.py --queries 1 --top-k 1 --max-requests 14` | Full stateful agent execution |

---

> **Disclaimer**: Research-only system using synthetic DDXPlus datasets. No clinical endorsement or real-world medical diagnostic use.
