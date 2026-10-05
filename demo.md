# Terminal research demonstration

Run from the parent workspace:

```text
uv run python healthcare-agentic-ai/scripts/demo.py
uv run python healthcare-agentic-ai/scripts/demo.py --sample 2 --detailed --architecture
```

The menu offers the first three **label-free synthetic DDXPlus validation cases**,
a detailed-mode toggle, and quit/back controls. Manual patient entry remains
unavailable: invented input must not be assigned a dataset ID as false provenance.

## Local and live checks

```text
uv run python healthcare-agentic-ai/scripts/demo.py --help
uv run python healthcare-agentic-ai/scripts/demo.py --list-samples
uv run python healthcare-agentic-ai/scripts/demo.py --sample 2 --validate-only --detailed
uv run python healthcare-agentic-ai/scripts/demo.py --sample 2 --medical-top-k 5 --detailed
```

`--list-samples` uses only the parser. `--validate-only` opens existing indexes and
cached BGE models, builds an exact label-free PatientState copy locally, and checks
focused retrieval without GPT requests or a diagnostic claim. The live workflow
instead obtains PatientState from the existing Patient Agent and verifies its
exact correspondence to the parser input before query construction.

Missing/empty indexes or invalid manifests fail. No rebuild/download/fallback model
is requested. A valid search with no retained medical evidence is explicitly marked
`insufficient_or_irrelevant`; it is not a setup failure or fabricated evidence.

Paths: `--data-dir`, `--patient-storage-path`, `--medical-storage-path`. The medical
index defaults to the parent workspace's `data/medical_knowledge/qdrant`. Close
other programs holding the embedded Qdrant stores. The live run uses dedicated
`GPT_SOL_*` configuration and the unchanged GPT-5.6-Sol Responses provider. Synthetic
patient and retained evidence context is sent to that configured cloud deployment;
credentials and transport details are not displayed.

## Current enhanced Phase 6 behavior

- Patient-case search: full label-free representation, `--top-k 1` by default.
- Medical search: focused PatientState concepts, independent `--medical-top-k 5`.
- All candidates undergo provenance validation, then deterministic topic filtering.
  Original scores, IDs and content are retained. The candidate audit explicitly
  shows discarded chunks; only retained chunks enter agent evidence contexts.
- Weak medical RAG no longer mechanically forces abstention. Explicitly uncertain,
  patient-fact-linked hypotheses may be proposed. Insufficient patient information
  still requires explained abstention. No invented evidence is permitted.
- Grounding, Clinical Critic and Safety Validator remain mandatory. Patient-only
  hypotheses still trigger the unchanged `missing_medical_reference → HUMAN_REVIEW`
  policy. Critic safety flags and identified prohibited actions still BLOCK.
- No treatment, prescribing, HITL, new agent, cloud retrieval or new dependency.

The detailed [implementation, limits and validation report](healthcare-agentic-ai/docs/enhanced-medical-retrieval.md)
explains the score probe, lexical filtering rule, inference contract and exact files.

## Presentation and exits

Live call notifications are provisional. The final **audited walkthrough** presents
committed PatientState, retrieval candidates and retained evidence, diagnostic
versions/references, grounding checks, critiques, safety findings and final routing.
It is not falsely presented as a live stream of accepted events. Rejected provider
output, raw prompts, credentials and free-form `reasoning_summary` are omitted.
Untrusted terminal prose is bounded and control characters are stripped.

The demo subclasses Phase6Orchestrator only for call notifications; its run loop is
inherited. Both CLI paths explicitly enable `enhanced_medical=True`. Direct library
callers default to the historical baseline for compatibility and may opt in. Phase
4/5 and the safety decision policy remain unchanged.

`--max-requests` defaults to 21 and `--max-seconds` to 900. The deadline is cooperative,
not hard cancellation of an in-flight request. Ctrl+C/EOF exits 130; setup/workflow
technical failure exits 1. Completed abstention, BLOCK and HUMAN_REVIEW exit 0 but
are prominently marked **withheld**, not successful diagnoses.

## Actual validation

Final full suite: **307 tests, 306 passed, 1 opt-in cloud test skipped**, including
24 focused-retrieval/inference regressions and the original 20 demo tests. Syntax,
help, sample discovery and existing-index local validation passed.

A real before/after comparison on `ddxplus:validate:2` used the existing **1,000
training cases / 87 medical chunks** without loading evaluation labels:

- Before: full narrative, top-1 BK transplant paper at **0.6704**; diagnostic
  abstention, critic revision request, safety **BLOCK**; 4 requests, 58.55 seconds.
- Enhanced: focused top-5 search retained only an asthma chunk at **0.6534**;
  irrelevant diabetes, intestinal-inflammation and BK chunks were filtered out.
  The agent proposed tentative hypotheses but the critic found substantive errors;
  safety still **BLOCKED**. 3 requests, 76.97 seconds.
- A separate real terminal demo also produced withheld hypotheses and **BLOCK**:
  3 requests, no repairs, 62.82 seconds. Its transcript is at
  `healthcare-agentic-ai/outputs/enhance/enhance-demo-live.log`.

The semantic safety generation step was correctly skipped after critic blocking in
these live runs; offline tests exercise its review/block paths. Irrelevant chunks
still appear among raw candidates. One retained topic does not establish adequate
medical coverage or support every hypothesis. The critic identified errors that
reference-existence grounding alone cannot detect.

**This improves retrieval and reasoning behavior in the research prototype; it does
not establish clinical validity.** Similarity is not disease probability, hypotheses
are not diagnoses, and HUMAN_REVIEW does not schedule or implement human review.
