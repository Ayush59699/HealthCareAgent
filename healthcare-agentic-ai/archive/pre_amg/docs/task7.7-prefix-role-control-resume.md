# Task 7.7 resume — stopped at protected-file integrity

## Status

The unchanged `scripts/run_memory_bounded.py` accepted startup. The required integrity check then failed, so no experiment implementation, model loading, baseline scoring, prefix scoring, or regression tests proceeded. No guard or experiment setting was changed. No automatic retry was performed.

## Integrity findings

Opaque, streaming SHA-256 verification checked 126 distinct protected paths against all six before/after manifests from Task 7.5 frozen/validated ranking ablations and Task 7.6 role-explicit inputs.

- 125 protected paths matched every historical reference.
- `healthcare-agentic-ai/rag/config.py` mismatched all six references.
- Expected SHA-256: `1247b27604da291857636b313284da08f36f59da4a287b6a67e841efbae60a24`
- Actual SHA-256: `d9567f1c69a8e3e754e6cea2b765d2121add300145a248f00e3f932e3e51da58`

The previous blocked-state documentation mentions an earlier user-requested `OKF_ENABLED` config addition. That is context, not proof of the complete cause of this mismatch. No config changes were reverted, accepted as a new baseline, or silently exempted.

Before/after hashes also cover the original Task 7.7 blocked-run artifacts and the Task 7.6 output tree. **No checked files changed during this preflight.** This does not mean the historical baseline is intact: the config mismatch already existed when verification began.

## Memory and execution

- Outer stop: unchanged 14 GB; startup reserve: unchanged 1.5 GB.
- Synchronous stop: unchanged existing 13.5 GB mechanism.
- Synchronous sampled system peak: 9.644883968 GB.
- Synchronous child-process RSS peak: 0.031248384 GB.
- Outer-runner sampled system peak: 9.644 GB (rounded).
- Outer-runner sampled child-tree RSS peak: 0.037 GB (rounded).
- Phase: `preflight-protected-artifact-integrity`; no current case; scoring progress: 0 pairs.
- Integrity child returned failure (2); see `integrity-runner.log` and `integrity-report.json`.
- No labels or benchmark contents inspected; no retrieval, corpus parsing, index API access, model downloads, cloud calls, or downstream agents. Protected data were only streamed as opaque bytes for hashing.

## Required closing report

- **What changed:** Added an isolated resume integrity helper, logs, before/after hashes, and stopped-status reports in a new output directory; updated OKF context. No experiment behavior was changed.
- **What remained frozen:** Existing production and experimental implementations, memory runner and limits, candidate pools, sources/indexes, evidence contracts, and original blocked-run artifacts were not modified.
- **Nonpositive → positive transitions:** Not measured; this is not a zero-transition result.
- **Previously useful candidates became eligible:** Not measured.
- **Pediatric asthma background preserved:** Not measured.
- **Case-3 acute-symptom evidence recovered:** Not measured.
- **Peak memory:** Approximately 9.645 GB system; outer sampled child-tree RSS 0.037 GB.
- **Tests passed:** No experiment/regression tests run. Integrity preflight failed for one of 126 protected paths; before/after preservation check passed.
- **Protected files unchanged:** Unchanged during this attempt; historical integrity is blocked by `rag/config.py`.
- **Exactly one recommendation:** Investigate and reconcile the protected `rag/config.py` mismatch with explicit authorization before resuming the unchanged Task 7.7 experiment.

No Outcome A/B/C/D, clinical accuracy, diagnostic correctness, BLOCK-rate, or generalization conclusion is supported.
