# Task 7.7 — prefix-only role control: stopped at memory preflight

## Status

**Blocked; not implemented or experimentally completed.** The mandatory existing
memory-bounded runner refused startup before the protected-artifact integrity
check could run. No automatic retry or relaxed memory limit was used.

## Project context reviewed

- Phases 1–2 establish streamed, label-isolated patient parsing and local patient
  retrieval; held-out patients are query inputs, not indexed cases.
- Phase 3 adds separately sourced medical knowledge with provenance.
- Phase 4 adds isolated patient, diagnostic, and clinical-critic agents.
- Phase 5 adds deterministic orchestration, bounded revisions, and immutable
  evidence snapshots.
- Phase 6 adds separately versioned safety validation without treating continuation
  as clinical clearance. Subsequent work tightens reference/abstention contracts
  and investigates focused retrieval.
- Task 7.4 adds opt-in hybrid retrieval with local reranking, retaining production
  defaults and evidence contracts.
- Task 7.5 freezes depth-50 pools for selection experiments.
- Task 7.6 tests a bundled role-explicit query rewrite. Its documented negative
  result motivates Task 7.7's prefix-only control; it is not a Task 7.7 result.

## Requested experiment, not yet executed

Use the existing 593 / 416 / 425 candidate pools for validate:1–3. Preserve every
original query byte after either `Current findings: ` or
`Historical/background: `. Preserve supporting-query assignments, source text,
channel and fusion data, ranking features, the strict raw-logit > 0 gate, selector,
and evidence contracts. Use only the pinned cached local cross-encoder, CPU,
batch size two, two threads, and maximum pair length 512.

Each case requires a freshly scored original-input control passing Task 7.6's
1e-4 numerical tolerance with unchanged eligibility and evidence snapshot before
prefix scoring. No fresh retrieval, query generation, fusion, benchmark labels,
cloud calls, or downstream clinical agents are permitted.

## Attempt and stopping evidence

The attempted command used `scripts/run_memory_bounded.py` to launch an opaque
SHA-256 verification of the files listed in the existing Task 7.5/7.6 protected
manifests. The child was not launched. The runner reported:

```text
Insufficient safe headroom: 12.68 GB used; close applications first
```

The unchanged startup check requires used system memory plus a 1.5 GB reserve to
be below 14 GB. At the reported reading, 12.68 + 1.5 = 14.18 GB. The separate
13.5 GB synchronous experiment stop was not reached because no experiment child
started. The reported 12.68 GB is a rounded startup reading, **not a measured run
peak**. Process-tree RSS was not measured.

Stopped-run artifacts are under `outputs/task7.7/prefix-role-control/`:
`configuration.json`, `memory-stop.json`, `experiment-stopped.json`,
`comparison.json`, and `comparison.md`. There are no fabricated case results,
transition tables, baseline hashes, or before/after manifests. A future authorized
attempt must use a new output directory, preserving this stopped attempt.

Protected-file integrity remains unverified. Existing uncommitted work was left
in place, including the earlier user-requested `OKF_ENABLED` config addition.
Any historical-manifest mismatch must be investigated rather than silently
accepted when the integrity check can safely run.

## Required closing report

- **What changed:** Added stopped-run documentation/artifacts and an OKF status
  entry only. No Task 7.7 experiment implementation was added.
- **What remained frozen:** This attempt did not modify production code, Task
  7.4/7.5/7.6 implementations or artifacts, corpus/indexes, agents, or contracts.
- **Nonpositive → positive transitions:** Unknown; no pairs were scored. This is
  not a measured zero-transition result.
- **Previously useful candidates became eligible:** Not measured.
- **Pediatric asthma background preserved:** Not measured.
- **Case-3 acute-symptom evidence recovered:** Not measured.
- **Peak memory:** Unavailable; startup system-memory reading was 12.68 GB.
- **Tests passed:** None run; the mandatory runner stopped before launching its child.
- **Protected files unchanged:** No protected files were written by this attempt;
  cryptographic before/after verification was blocked and is not claimed.
- **Exactly one next recommendation:** Investigate and free system-memory headroom
  before resuming this same fixed Task 7.7 experiment, without relaxing memory limits.

No Outcome A/B/C/D classification, clinical accuracy, diagnostic correctness,
BLOCK-rate, or generalization claim is supported by this stopped attempt.
