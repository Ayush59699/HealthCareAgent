# MULTIMODAL AGENTIC REASONING ARCHITECTURE FOR HEALTHCARE
## Enterprise System Blueprint: EHR Harmonization, Medical Image Reasoning, Multi-Agent Clinical Inference, Deterministic Validation, and Multi-Level Doctor-in-the-Loop Feedback

---

## EXECUTIVE SUMMARY & ARCHITECTURAL VISION

The objective is to transform the lightweight prototype in `mini-cc` (a terminal-based single-loop ReAct coding assistant) into an enterprise-grade, multimodal, multi-agent clinical reasoning platform. 

In clinical medicine, unstructured patient narratives, heterogeneous Electronic Health Records (EHR), multi-spectral medical imaging (CT, MRI, X-Ray, Histopathology, Ultrasound), and longitudinal laboratory telemetry must converge into actionable, evidence-backed clinical intelligence. Unlike coding agents where hallucinations yield syntax errors easily caught by compilers, clinical hallucinations or missed contraindications carry severe patient safety risks.

Therefore, this architecture is designed around **five core pillars**:
1. **Multimodal Ingestion & Data Harmonization**: Deterministic normalization of disparate clinical data into FHIR (Fast Healthcare Interoperability Resources R4/R5) and OMOP Common Data Model (CDM) with standard ontologies (SNOMED CT, LOINC, RxNorm, ICD-10-CM).
2. **Deep Medical Image Reasoning**: Combining deep specialized medical vision models (MONAI, Med-SAM, BiomedCLIP) with high-capacity Vision-Language Models (VLMs) for visual grounding, lesion segmentation, and cross-referencing with clinical context.
3. **Multi-Agent Deliberative Reasoning**: Decoupled, specialized agents operating over a shared state graph with explicit clinical deduction, guideline retrieval (GraphRAG over PubMed, UpToDate, clinical practice guidelines), and differential diagnosis synthesis.
4. **Deterministic Validation & Clinical Guardrails**: Autonomous verification layers executing pharmacological checks (drug-drug interactions, organ-clearance dosing, allergy cross-reactivity), hallucination detection, and guideline compliance before clinical presentation.
5. **Multi-Level Doctor-in-the-Loop (HITL) & Continuous Learning**: Interactive decision gates at ingestion, intermediate reasoning, imaging interpretation, and treatment recommendation, capturing granular clinician feedback (accept, modify, critique, bounding box correction) to fuel Direct Preference Optimization (DPO) and audit logging.

---

## 1. CURRENT PROJECT ANALYSIS (`mini-cc`)

### 1.1 Codebase Audit
The existing repository consists of three sequential evolutionary scripts:
* **`step1.py`**: A barebones OpenAI-compatible conversational script connecting to an Azure OpenAI endpoint (`https://meeting-summarizer-ai.services.ai.azure.com/openai/v1`) using a static model deployment (`gpt-5` / hardcoded API key). It runs a simple synchronous terminal prompt loop without memory management or tooling.
* **`step2.py`**: Introduces structured tool calling via OpenAI tool schemas (`safe_read_file`), an execution turn counter (`MAX_TURNS = 10`), and environment variable retrieval (`OPENAI_BASE_URL`, `AZURE_OPENAI_ENDPOINT`).
* **`step3.py`**: A fully functional autonomous terminal agent ("Mini Claude Code"). It defines a ReAct loop with 4 tools:
  * `list_file`: Directory enumeration via `os.listdir`.
  * `read_file`: Unbounded file content reader via Python IO.
  * `write_file`: File writer.
  * `run_command`: Subprocess executor (`subprocess.run(shell=True)`).
  * Backend model: Configured via `.env` pointing to Azure OpenAI deployment `gpt-6-astra`.

### 1.2 Structural Strengths
* **Native Tool-Calling Loop**: `step3.py` correctly implements the assistant message tool call protocol, JSON argument parsing, tool dispatch via dictionary lookup, and tool-result message injection back to context.
* **Pluggable Foundation**: The separation of `TOOLS` mapping and `TOOL_SCHEMAS` provides a clean hook to replace file/shell tools with clinical tools (FHIR queries, DICOM viewers, guideline vector search).

### 1.3 Architectural Gaps for Healthcare Applications
| Dimension | Current `mini-cc` State | Required Healthcare State |
| :--- | :--- | :--- |
| **Agent Topology** | Monolithic single-agent while-loop. | Hierarchical / DAG Multi-Agent State Graph (Orchestrator + Specialists + Validators). |
| **Data Modalities** | Pure text / raw code strings. | Multimodal: DICOM (16-bit radiologic volumes), FHIR JSON, HL7 v2 messages, tabular labs, ECG signals, clinical notes. |
| **Reasoning Depth** | Single LLM zero-shot / ReAct tool loop. | Multi-agent chain-of-thought, differential generation, debate/critique, and evidence-grounded GraphRAG. |
| **Safety & Validation** | Subprocess shell execution with no safety filters. | Multi-tier deterministic verification: Drug-Drug Interaction (DDI) engines, guideline checks, allergy alerts, dosing calculators. |
| **Human Interaction** | Terminal `input("USER: ")` at start only. | Multi-level Doctor-in-the-Loop (HITL) with interrupt points, interactive differential adjustments, and image annotation. |
| **Regulatory & Privacy** | Plain environment keys, raw file dumps, zero PHI scrubbing. | HIPAA/GDPR compliance, zero-data-retention BAA endpoints, local PHI de-identification (Presidio), strict audit trail. |

---

## 2. HIGH-LEVEL AGENTIC ARCHITECTURE

```
                                    +-------------------------------------------------+
                                    |          CLINICIAN / DOCTOR WORKSTATION         |
                                    |    (Next.js / WebDICOM / FHIR UI / Review)      |
                                    +-----------------------+-------------------------+
                                                            |  Interactive Review &
                                                            |  Multi-Level Feedback
                                                            v
+-----------------------------------------------------------------------------------------------------------------------+
|                                              ORCHESTRATION STATE GRAPH                                                |
|                                                                                                                       |
|   +--------------------------+         +--------------------------+         +-------------------------------------+   |
|   | 1. INGESTION & TRIAGE    |         | 2. DATA HARMONIZATION    |         | 3. MULTIMODAL MEDICAL IMAGE         |   |
|   |    AGENT                 |         |    AGENT                 |         |    REASONING AGENT                  |   |
|   | - Patient Intake         | ------> | - FHIR R4/R5 Extraction  | ------> | - DICOM Header & Voxel Parser       |   |
|   | - Chief Complaint Class. |         | - OMOP CDM Mapping       |         | - Med-SAM Segmentation              |   |
|   | - Urgency & Acuity Score |         | - SNOMED / LOINC / RxNorm|         | - MONAI 3D Volumetric Feature Extr. |   |
|   | - Local De-ID (Presidio) |         | - Longitudinal Chronology|         | - VLM Lesion Grounding & Draft Rep. |   |
|   +--------------------------+         +--------------------------+         +-------------------------------------+   |
|                                                                                                |                      |
|                                                                                                v                      |
|   +---------------------------------------------------------------------------------------------------------------+   |
|   |                                     [DOCTOR CHECKPOINT 1: DATA & IMAGE AUDIT]                                 |   |
|   |                     Clinician validates segmented ROI, confirms extraction, or overrides findings.           |   |
|   +---------------------------------------------------------------------------------------------------------------+   |
|                                                            | Approved State                                           |
|                                                            v                                                          |
|   +------------------------------------+         +----------------------------------------------------------------+   |
|   | 4. MULTIMODAL SYNTHESIS &          |         | 5. CLINICAL DIFFERENTIAL & DIAGNOSTIC                          |   |
|   |    CROSS-MODAL FUSION AGENT        | ------> |    REASONING AGENT                                             |   |
|   | - Correlates Labs + Imaging + EHR  |         | - Pathophysiological Multi-Hop Deduction                       |   |
|   | - Resolves Discrepancies           |         | - Evidence Retrieval (GraphRAG: UpToDate/PubMed/Guidelines)    |   |
|   | - Constructs Unified Patient State |         | - Ranked Differential Diagnoses (with Prior & Posterior Lik.)  |   |
|   +------------------------------------+         +----------------------------------------------------------------+   |
|                                                                                                |                      |
|                                                                                                v                      |
|   +---------------------------------------------------------------------------------------------------------------+   |
|   |                                    [DOCTOR CHECKPOINT 2: DIFFERENTIAL REVIEW]                                 |   |
|   |                     Clinician reviews candidate diagnoses, orders additional tests, or eliminates candidates |   |
|   +---------------------------------------------------------------------------------------------------------------+   |
|                                                            | Confirmed Differential                                   |
|                                                            v                                                          |
|   +------------------------------------+         +----------------------------------------------------------------+   |
|   | 6. CLINICAL SAFETY &               |         | 7. DOCTOR-IN-THE-LOOP FEEDBACK &                               |   |
|   |    VALIDATION AGENT                | ------> |    CONTINUOUS LEARNING AGENT                                   |   |
|   | - Deterministic Drug-Drug Checking |         | - Structured Decision Approval (Sign-Off)                      |   |
|   | - Organ Clearance & Dosing Audit   |         | - Clinical Rationale Capture (Critique / Override)            |   |
|   | - Guideline Compliance Verification|         | - Direct Preference Optimization (DPO) Dataset Generation      |   |
|   | - Hallucination & Factuality Guard |         | - Immutable FHIR Provenance & Audit Log Generation             |   |
|   +------------------------------------+         +----------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 3. SPECIALIZED AGENT CATALOG

### Agent 1: Clinical Ingestion & Triage Agent
* **Role**: First-line gatekeeper handling multimodal ingest (patient portal messages, emergency department triage notes, paramedic summaries, vital sign streams).
* **Key Responsibilities**:
  * Strips Direct Identifiers (PII/PHI) using an on-premise de-identification tool (Microsoft Presidio / Philter) before downstream reasoning.
  * Assesses clinical acuity using standardized triage systems (e.g., Emergency Severity Index - ESI 1 to 5).
  * Categorizes chief complaints into clinical domains (Cardiology, Pulmonology, Neurology, Oncology, etc.).
* **Inputs**: Unstructured text, audio dictations (Whisper-Med transcription), raw vitals (HR, BP, SpO2, RR, Temp, GCS).
* **Tools**:
  * `deidentify_phi_stream`: Local NER model tagging 18 HIPAA identifiers.
  * `calculate_esi_score`: Algorithmic evaluation of hemodynamic stability and expected resource utilization.
  * `dispatch_triage_alert`: High-priority webhook to on-call clinical teams if ESI is 1 or 2 (immediate life threat).
* **Execution Tier**: Local / Private Sovereign Cloud for low latency and zero data leakage.

### Agent 2: EHR & Data Harmonization Agent
* **Role**: Transforms dirty, siloed healthcare records into structured, semantically interoperable clinical timelines.
* **Key Responsibilities**:
  * Ingests legacy HL7 v2.x feeds (ADT, ORU, MDM), CDA/C-CDA XML documents, and unstructured physician progress notes.
  * Maps medical terminology to standard international ontologies:
    * Conditions/Diagnoses $\to$ **SNOMED CT** & **ICD-10-CM**
    * Laboratory tests and observations $\to$ **LOINC**
    * Medications, dosages, and ingredients $\to$ **RxNorm**
    * Anatomical sites $\to$ **FMA (Foundational Model of Anatomy)**
  * Produces valid **FHIR R4/R5** bundles (`Patient`, `Condition`, `Observation`, `MedicationRequest`, `DiagnosticReport`, `AllergyIntolerance`).
* **Tools**:
  * `fhir_client_query`: RESTful query to EHR FHIR server (Epic on FHIR, Cerner Millennium, or HAPI FHIR).
  * `omop_vocabulary_lookup`: Semantic search against Athena / UMLS Knowledge Base.
  * `temporal_chronology_builder`: Assembles a longitudinal timeline of disease progression, prior surgeries, and medication changes.
* **Output**: A normalized JSON Graph of the patient's longitudinal trajectory.

### Agent 3: Multimodal Medical Image Reasoning Agent
* **Role**: Ingests, parses, segments, and clinically interprets radiologic and pathology imaging data.
* **Key Responsibilities**:
  * Direct DICOM parsing: extracts metadata (slice thickness, acquisition plane, contrast timing, window width/level, radiation dose).
  * Coordinates specialized computer vision sub-models:
    * **Med-SAM (Segment Anything in Medical Images)**: Zero-shot promptable segmentation of organs, nodules, fractures, and lesions.
    * **MONAI (Medical Open Network for AI)**: 3D volumetric segmentation for CT/MRI (e.g., brain tumors, liver lesions, pulmonary embolisms).
    * **BioMedCLIP / RadImageNet**: Medical zero-shot classification and image-text retrieval.
  * Generates visual grounding bounding boxes / masks and drafts structured radiologic findings (e.g., RECIST criteria for oncology, Lung-RADS, BI-RADS).
* **Tools**:
  * `dicom_voxel_loader`: Loads multi-frame DICOM files, applies Hounsfield Unit (HU) windowing.
  * `run_medsam_inference`: Takes point/box prompts and outputs high-resolution segmentation masks.
  * `extract_radiomic_features`: Calculates lesion volume, sphericity, heterogeneity, and density metrics.
  * `vlm_visual_qa`: Sends high-resolution anatomical crops to a medical Vision-Language Model with explicit spatial reasoning prompts.
* **Output**: Structured imaging report containing findings, quantitative measurements, segmentation overlays, and differential impressions.

### Agent 4: Cross-Modal Synthesis & Fusion Agent
* **Role**: Connects the dots across siloed clinical modalities.
* **Key Responsibilities**:
  * Reconciles clinical note assertions with objective diagnostic telemetry (e.g., patient complains of dyspnea $\leftrightarrow$ chest X-ray reveals bilateral pleural effusion $\leftrightarrow$ lab shows elevated NT-proBNP at 4,200 pg/mL).
  * Flags clinical discrepancies: e.g., physician note states "no history of kidney disease", but longitudinal serum creatinine indicates an eGFR drop to 28 mL/min/1.73m² (Stage 4 CKD).
* **Tools**:
  * `cross_modal_aligner`: Bipartite graph matching between documented symptoms, laboratory trends, and imaging findings.
  * `flag_clinical_conflict`: Surfaces contradictions for doctor review.
* **Output**: Synthesized Unified Clinical State Vector.

### Agent 5: Clinical Differential & Diagnostic Reasoning Agent
* **Role**: Primary deliberative reasoning engine simulating expert clinical grand rounds.
* **Key Responsibilities**:
  * Formulates a comprehensive, ranked differential diagnosis using probabilistic Bayesian reasoning (combining pre-test probabilities with likelihood ratios of present/absent symptoms and test results).
  * Executes **GraphRAG** queries across medical knowledge graphs and peer-reviewed literature (PubMed, UpToDate, Cochrane Reviews, clinical practice guidelines from AHA, ACC, ASCO, IDSA).
  * Employs structured Chain-of-Thought (CoT) with step-by-step pathophysiological mechanisms, avoiding premature diagnostic closure.
* **Tools**:
  * `graph_rag_medical_query`: Graph-traversal query over Neo4j containing UMLS relations + PubMed knowledge embeddings.
  * `guideline_evidence_retriever`: Retrieval of exact clinical society guidelines with grading of recommendations (Class I, Level of Evidence A, etc.).
  * `bayesian_risk_calculator`: Computes validated clinical decision rules (e.g., Wells' Score for PE, CHA₂DS₂-VASc for stroke risk, HEART score for acute coronary syndrome).
* **Output**: Ranked differential table: Candidate Condition, Supporting Evidence, Contradicting Evidence, Probability Category (Definitive, Highly Probable, Possible, Unlikely), and Recommended Next Best Action.

### Agent 6: Clinical Validation & Safety Guardrail Agent
* **Role**: Strict, non-negotiable adversarial gatekeeper ensuring patient safety and regulatory compliance.
* **Key Responsibilities**:
  * **Pharmacological Validation**:
    * Drug-Drug Interactions (DDI): Severity scoring (Contraindicated, Major, Moderate) via First Databank / RxNorm APIs.
    * Drug-Allergy Checking: Cross-reactivity detection (e.g., penicillin allergy vs cephalosporins).
    * Organ-Clearance Dosing: Validates dosage against eGFR (Cockcroft-Gault) and hepatic function (Child-Pugh).
  * **Clinical Guideline Compliance**: Checks whether suggested interventions violate standard of care (e.g., prescribing beta-blockers in acute decompensated heart failure with cardiogenic shock).
  * **Hallucination Detection**: Token-level attribution verifying that every clinical claim maps to an explicit record in the patient's EHR or validated retrieval corpus.
* **Tools**:
  * `rxnorm_interaction_check`: Queries the NLM Drug Interaction API.
  * `check_renal_dose_adjustment`: Calculates renal clearance limits for suggested medications.
  * `factuality_verifier`: Natural Language Inference (NLI) model testing hypothesis (agent statement) against premise (patient EHR / lab values).
* **Output**: Safety Clearance Certification or Hard Block Alert with specific contraindication citations.

### Agent 7: Doctor-in-the-Loop (HITL) & Feedback Integration Agent
* **Role**: Facilitates seamless doctor collaboration, capturing clinical intent and feedback at multiple levels.
* **Key Responsibilities**:
  * Manages state graph interrupts: suspends execution at defined clinical gates until a licensed clinician provides authenticated approval.
  * Supports 4 distinct levels of clinical feedback:
    1. *Level 1 (Data Verification)*: Approving/rejecting parsed EHR data and image segmentation boundaries.
    2. *Level 2 (Differential Refinement)*: Adding suspected rare diseases or removing ruled-out diagnoses from the candidate list.
    3. *Level 3 (Intervention Modification)*: Tweaking recommended medications, dosages, or diagnostic workups.
    4. *Level 4 (Free-Text Clinical Critique)*: Providing qualitative rationale ("Patient has severe gout; avoid thiazides despite hypertension").
  * Serializes feedback into structured datasets for model alignment (DPO, RLHF, and automated prompt calibration).
* **Tools**:
  * `emit_doctor_review_request`: Dispatches state payload to clinician workstation via WebSocket.
  * `capture_clinician_override`: Records delta between agent recommendation and doctor decision, tagged with doctor NPI/license and rationale.
  * `log_fhir_provenance`: Writes a tamper-evident audit record complying with 21 CFR Part 11 and HIPAA Audit Rule.

---

## 4. LOCAL VS. CLOUD (GPT-ASTRA / AZURE) VS. HYBRID ANALYSIS

One of the most critical architectural decisions is where reasoning and data processing occur. Healthcare systems have stringent requirements around Protected Health Information (PHI), latency, uptime, and reasoning quality.

### 4.1 Comparative Evaluation

| Attribute | Pure Local / On-Premise | Cloud LLMs (GPT-Astra, Azure OpenAI, Claude 3.5) | Recommended Hybrid Architecture |
| :--- | :--- | :--- | :--- |
| **PHI / HIPAA Compliance** | **Optimal**: Zero data leaves the hospital firewall; no Business Associate Agreement (BAA) needed with 3rd parties. | **Requires BAA**: Data encrypted in transit/rest; zero-data-retention agreement required with Microsoft/AWS/GCP. | **Optimal**: Sensitive data de-identified locally; high-res raw images remain on-prem; de-identified embeddings go to cloud. |
| **Complex Medical Reasoning** | **Moderate**: 8B–70B open weights (Llama-3-Med42, BioMistral) perform well on factual recall, but struggle with 10+ turn complex cross-modal deductions. | **Superior**: Frontier models (GPT-Astra, Gemini 2.0 Pro / MedLM, Claude 3.5/3.7) show state-of-the-art diagnostic reasoning and structured JSON fidelity. | **Superior**: Hybrid topology leverages local specialized models for deterministic tasks, routing synthesized context to frontier models for differential diagnosis. |
| **Medical Image Processing** | **Superior**: Local GPU nodes (NVIDIA RTX 6000 / A100 / H100) run MONAI and 3D volumetric DICOM segmentation in seconds without uploading 2GB CT volumes over WAN. | **Poor / High Latency**: Uploading 3D DICOM series (500–2,000 slices, 500MB–2GB) to cloud APIs is bandwidth-prohibitive and introduces massive latency. | **Superior**: Raw volumetric DICOMs processed locally by MONAI/Med-SAM; 2D key slices, feature vectors, and segmentation masks forwarded to cloud VLM. |
| **Operational Cost** | High initial CapEx for GPU infrastructure; low variable token cost. | Zero CapEx; variable OpEx based on token and vision processing volume. | Balanced: Edge hardware for image filtering reduces cloud token/image ingestion costs by up to 70%. |
| **Latency & Resilience** | Deterministic local latency; runs during internet outages. | Dependent on cloud network latency and provider rate limits. | Local triage and validation function autonomously even during wide-area network degradation. |

### 4.2 The 4-Tier Hybrid Topology (The Gold Standard)

```
[ HOSPITAL PRIVATE LAN / EDGE ]
  │
  ├── Tier 1: Local Ingestion & Edge Perception
  │     ├── Microsoft Presidio / Philter (Local PHI De-identification)
  │     ├── Orthanc / dcm4chee DICOM PACS Server
  │     └── MONAI + Med-SAM on local NVIDIA GPUs (3D CT/MRI segmentation & radiomics)
  │
  ├── Tier 2: Sovereign Harmonization & Rule Enforcement
  │     ├── HAPI FHIR Server (Local Patient Records)
  │     ├── Local BioMistral / Llama-3-8B-Instruct (via vLLM) for initial extraction
  │     └── SQLite / PostgreSQL cache for local clinical ontologies (SNOMED, RxNorm)
  │
  ▼  (Secure De-Identified TLS Tunnel with BAA)
[ ENTERPRISE CLOUD COMPUTE (Azure / AWS Sovereign Health Cloud) ]
  │
  └── Tier 3: High-Order Multimodal Reasoning Engine
        ├── Azure OpenAI `gpt-6-astra` / Claude 3.5 Sonnet / Med-Gemini
        ├── Complex Multi-Hop Differential Reasoning
        ├── Cross-Modal Clinical Synthesis
        └── GraphRAG over 35 Million PubMed abstracts & clinical guidelines
  │
  ▼  (Structured Differential & Recommendation Payload)
[ HOSPITAL WORKSTATION & VALIDATION GATE ]
  │
  └── Tier 4: Deterministic Guardrails & Clinician Approval
        ├── Local Deterministic Pharmacology Engine (DDI, eGFR dosing)
        ├── Doctor-in-the-Loop Interactive Review Workstation
        └── Cryptographic Provenance & FHIR Audit Log Generation
```

---

## 5. CUTTING-EDGE TECHNOLOGIES & LIBRARIES (2026/2027 STACK)

### 5.1 Multi-Agent Orchestration & Workflow State
* **LangGraph (Recommended Core)**:
  * Stateful multi-agent graph architecture with cyclicity, conditional branching, and checkpointing.
  * Native `interrupt()` support: essential for pausing execution at clinical checkpoints until the doctor enters approval or corrections.
  * "Time travel" capability: enables clinicians to modify an earlier diagnostic assumption and re-run all downstream reasoning branches to compare outcomes.
* **DSPy (Declarative Self-Improving Prompts)**:
  * Replaces fragile prompt engineering with programmatic teleprompters. Optimizes medical reasoning pipelines against clinical validation benchmarks automatically.
* **LlamaIndex Workflows**:
  * Advanced event-driven orchestration tailored for complex document extraction and recursive indexing over hospital records.

### 5.2 Medical Multimodal Imaging & Computer Vision
* **MONAI (Medical Open Network for AI)**:
  * Industry-standard PyTorch-based framework for healthcare imaging. Provides pretrained pipelines for 3D spleen/liver/lung segmentation, tumor boundary tracking, and multi-parametric MRI registration.
* **Med-SAM (Segment Anything in Medical Images)** & **SAM-Med2D/3D**:
  * Foundation models for zero-shot anatomical segmentation via bounding box or point prompts.
* **BioMedCLIP & CXR-Foundation**:
  * Vision-Language contrastive models pretrained on millions of PubMed images and chest radiographs for zero-shot medical classification.
* **CornerstoneJS / OHIF Viewer**:
  * Web-based zero-footprint medical image viewer integration for interactive doctor visualization of agent segmentations.

### 5.3 Health Interoperability & Clinical Ontologies
* **HL7 FHIR (Release 4 & 5)**:
  * The global standard for healthcare data exchange. Python library: `fhir.resources` (Pydantic models for every FHIR resource).
* **UMLS (Unified Medical Language System) & Athena (OHDSI)**:
  * Unified mapping layer cross-referencing SNOMED CT, LOINC, RxNorm, MeSH, and ICD-10.
* **GraphRAG with Neo4j / Memgraph**:
  * Knowledge Graph Retrieval-Augmented Generation. Enables the reasoning agent to traverse pathophysiological relationships (e.g., `Drug A` $\to$ `inhibits Enzyme B` $\to$ `decreases clearance of Drug C`).

### 5.4 Safety, Guardrails & Explainability
* **NeMo Guardrails (NVIDIA)**:
  * Programmable Colang safety rails enforcing strict dialogue boundaries, preventing off-topic drift, and triggering fallback protocols when clinical uncertainty exceeds thresholds.
* **Outlines / Guidance**:
  * Guaranteed structured generation (Pydantic models / regex CFGs) eliminating JSON syntax parsing errors during tool calls.
* **Microsoft Presidio**:
  * Open-source, production-grade de-identification SDK using transformer-based Named Entity Recognition to detect and mask PHI.
* **Arize Phoenix / Langfuse / OpenTelemetry**:
  * Full-lifecycle agent observability. Traces every tool call, latency breakdown, token cost, and clinician interaction for clinical auditability.

---

## 6. DOCTOR-IN-THE-LOOP (HITL) MULTI-LEVEL FEEDBACK FRAMEWORK

Healthcare agents must **never** operate as autonomous, unverified black boxes. They must operate as **Centaur Systems**—cognitive co-pilots augmenting licensed clinical experts.

### 6.1 Feedback Architecture Levels

```
                     LEVEL 1: DATA ACCURACY FEEDBACK
  [Ingestion / EHR Extraction] ──> Clinician validates extracted allergy, dosage, or timeline.
                                    Action: Quick Accept / Inline Edit / Delete Record.

                     LEVEL 2: IMAGING & SEGMENTATION FEEDBACK
  [Medical Image Agent] ─────────> Clinician reviews segmentation overlay in OHIF/Cornerstone.
                                    Action: Nudge mask boundary / Re-classify lesion type.

                     LEVEL 3: DIFFERENTIAL DIAGNOSIS FEEDBACK
  [Reasoning Agent] ─────────────> Clinician reviews ranked differential candidate list.
                                    Action: Re-order ranks / Eliminate candidate / Add rare rule-out.

                     LEVEL 4: MANAGEMENT & TREATMENT FEEDBACK
  [Validation & Safety Agent] ───> Clinician reviews drug order, therapy, or discharge instructions.
                                    Action: Override warning with clinical rationale / Approve order.
```

### 6.2 Data Model for Doctor Feedback (JSON Schema)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ClinicalFeedbackPayload",
  "type": "object",
  "properties": {
    "feedback_id": { "type": "string", "format": "uuid" },
    "session_id": { "type": "string", "format": "uuid" },
    "patient_fhir_id": { "type": "string" },
    "doctor_id": { "type": "string", "description": "National Provider Identifier (NPI)" },
    "checkpoint_level": { 
      "type": "string", 
      "enum": ["DATA_EXTRACTION", "IMAGE_SEGMENTATION", "DIFFERENTIAL_RANKING", "TREATMENT_PLAN"] 
    },
    "agent_proposal": { "type": "object" },
    "doctor_action": { 
      "type": "string", 
      "enum": ["FULL_ACCEPT", "REJECT", "MODIFY_WITH_OVERRIDE", "ADD_CRITIQUE"] 
    },
    "doctor_modifications": {
      "type": "object",
      "properties": {
        "modified_fields": { "type": "array", "items": { "type": "string" } },
        "revised_value": { "type": "object" },
        "clinical_justification": { "type": "string" }
      }
    },
    "feedback_metadata": {
      "timestamp": { "type": "string", "format": "date-time" },
      "time_spent_seconds": { "type": "number" },
      "interaction_type": { "type": "string", "enum": ["CLICK", "SLIDER", "FREE_TEXT", "ROI_DRAW"] }
    }
  },
  "required": ["feedback_id", "session_id", "doctor_id", "checkpoint_level", "doctor_action", "feedback_metadata"]
}
```

### 6.3 Continuous Learning Pipeline (Closing the Loop)
1. **Real-Time Context Updating**: When a doctor modifies a diagnostic ranking at Checkpoint 2, the State Graph immediately updates its working memory and invalidates any downstream treatment proposals generated under the previous diagnosis.
2. **Preference Optimization (DPO)**: Triplet pairs $(x, y_w, y_l)$ are generated where:
   * $x$ = Patient clinical scenario and imaging data.
   * $y_w$ = Doctor-approved differential / treatment.
   * $y_l$ = Agent-generated proposal rejected or edited by the doctor.
   These triplets update specialized local adapter weights (LoRA) on a bi-weekly cycle.
3. **Automated Error Taxonomy Mining**: An asynchronous batch agent clusters clinician overrides weekly to surface systemic biases (e.g., agent consistently under-estimating pulmonary embolism in post-op orthopedic patients) to update DSPy assertion constraints.

---

## 7. MIGRATION ROADMAP: FROM `mini-cc` TO MULTIMODAL HEALTHCARE PLATFORM

Here is the step-by-step path to evolve the existing files in `mini-cc` into this target state:

```
mini-cc (Current)                      Target Architecture (Evolved)
├── step1.py (Basic API)       ───>    src/core/llm_router.py (Multi-tier routing: Azure OpenAI Astra / Local vLLM)
├── step2.py (Safe Tool Loop)  ───>    src/agents/base_agent.py (Type-safe, Pydantic-validated tool executions)
├── step3.py (Mini Claude Code)───>    src/orchestrator/state_graph.py (LangGraph DAG with HITL interrupts)
└── .env (Raw API keys)        ───>    src/config/security.py (Azure Key Vault / HSM / Zero PHI Retention)
                               +──>    src/agents/ingestion_agent.py (Presidio PHI + Triage)
                               +──>    src/agents/ehr_harmonizer.py (FHIR R4/R5 + OMOP)
                               +──>    src/agents/medical_image_agent.py (DICOM + MONAI + Med-SAM)
                               +──>    src/agents/differential_agent.py (Pathophysiology + GraphRAG)
                               +──>    src/agents/validation_agent.py (DDI + Dosing + Hallucination Check)
                               +──>    src/feedback/doctor_hitl.py (Webhooks + DPO Logger + OHIF Bridge)
```

### Phase 1: Foundation & Harmonization (Weeks 1–3)
* Upgrade `pyproject.toml` dependencies:
  * `langgraph>=0.2.0`, `pydantic>=2.7.0`, `fhir.resources>=7.1.0`
  * `pydicom>=2.4.0`, `monai>=1.3.0`, `torch>=2.3.0`
  * `presidio-analyzer`, `presidio-anonymizer`
* Replace `step3.py`'s basic `TOOLS` dictionary with type-safe Pydantic tools:
  * `extract_fhir_resources`
  * `query_rxnorm_ddi`
  * `inspect_dicom_metadata`
* Implement local PHI de-identification before any string is dispatched to `AZURE_OPENAI_ENDPOINT`.

### Phase 2: Multimodal Image & GraphRAG Integration (Weeks 4–6)
* Build the `MedicalImageAgent`:
  * Add DICOM reader tool supporting 16-bit CT/MRI arrays.
  * Integrate Med-SAM for zero-shot lesion segmentation.
  * Implement visual grounding prompts feeding key 2D slices to Azure `gpt-6-astra` VLM.
* Deploy a medical GraphRAG instance:
  * Ingest SNOMED CT and clinical practice guidelines into Neo4j.
  * Replace naive file search with semantic multi-hop knowledge retrieval.

### Phase 3: Deterministic Validation & Clinical Safety Rails (Weeks 7–8)
* Construct the `ClinicalValidationAgent`:
  * Implement automated renal dosing check (Cockcroft-Gault calculator tool).
  * Build a deterministic drug interaction matrix using RxNorm REST APIs.
  * Implement token-level citation checking (ensuring every recommended action references an identified finding or guideline).

### Phase 4: Doctor-in-the-Loop Webstation & Active Feedback (Weeks 9–10)
* Connect LangGraph State Graph with an interactive UI (Next.js + Tailwind + CornerstoneJS):
  * State Graph halts execution at `CHECKPOINT_DIFFERENTIAL` using LangGraph `interrupt()`.
  * Doctor reviews differential, adds/removes diagnoses, clicks "Approve".
  * Graph resumes to generate validated management plan.
* Build the Feedback Serialization pipeline saving doctor modifications into Parquet/JSONL datasets for ongoing preference fine-tuning.

---

## 8. SUMMARY MATRIX: AGENTS, MODELS, AND TOOLS

| Agent Name | Recommended Model | Primary Role | Core Tools / Integrations | Fallback / Safety Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Ingestion & Triage** | Local BioMistral-7B / Llama-3-8B | Patient intake, PHI stripping, urgency scoring | Presidio, Whisper-Med, ESI Scoring Rule | Revert to Human Triage Nurse if ambiguity > 0.3 |
| **EHR Harmonizer** | Azure `gpt-6-astra` (Structured JSON mode) | Normalizing dirty records into FHIR R4/R5 | `fhir.resources`, Athena OHDSI, LOINC lookup | Strict Pydantic schema validation; invalid JSON rejected |
| **Medical Image Reasoning** | MONAI + Med-SAM (Local) + `gpt-6-astra` (VLM) | Lesion segmentation, DICOM interpretation, visual grounding | PyDICOM, SimpleITK, Med-SAM box prompt, OHIF | Flag image quality artifacts; require radiologist review |
| **Clinical Differential Reasoning** | Azure `gpt-6-astra` / Claude 3.5 Sonnet | Multi-hop clinical deduction & differential ranking | GraphRAG (Neo4j), UpToDate / PubMed Retriever, Wells/HEART rule | Explicit counter-argument check; rule out life-threats first |
| **Safety & Validation** | Deterministic Python Engine + NeMo Guardrails | DDI check, organ dosing, allergy verification, fact check | RxNorm API, Cockcroft-Gault, NLI Fact Checker | **HARD BLOCK**: Prohibits clinician dispatch if DDI is severe |
| **Doctor Feedback & Learning** | LangGraph State Manager | Workflow pauses, feedback capture, DPO pipeline | LangGraph `interrupt()`, FHIR Provenance, DPO Logger | State snapshotting; 100% auditable under 21 CFR Part 11 |

---
*Documented and compiled in `context/base` for enterprise architecture execution.*
