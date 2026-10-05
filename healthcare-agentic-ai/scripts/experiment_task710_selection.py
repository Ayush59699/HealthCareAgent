"""Task 7.10 frozen-input preflight. Run only through run_memory_bounded.py.

No model, retrieval, corpus, patient-file or benchmark imports. Fail closed.
"""
import argparse
import gc
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiment_task79_selection import Monitor, MemoryStop, digest, load, load_case, verify, write, require_all_controls
from rag.experimental_medical.task79_selection_analysis import original_functions, replay

TASK79 = ROOT / 'outputs/task7.9/selection-analysis'
MANIFEST_SHA256 = 'ddfbe46f184eaeb14c9b488009d68e38cec926a7e4e587bfc5bbe5ed72f54656'


def authenticate(monitor):
    manifest = TASK79 / 'protected-files-after.json'
    if digest(manifest, monitor) != MANIFEST_SHA256:
        raise ValueError('Protected Task 7.9 manifest mismatch')
    expected = load(manifest, monitor)
    verify(expected, monitor)
    # Freeze Task 7.9 outputs too; never update its historical manifest.
    for directory in (TASK79, ROOT / 'docs'):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and (directory == TASK79 or p.name.startswith('task7.9')):
                expected[str(p.resolve())] = digest(p, monitor)
    expected[str(manifest.resolve())] = MANIFEST_SHA256
    return expected


def exact_control(saved, prior, analysis, choose, tokenize, sample=lambda: None):
    result = replay(saved, choose, tokenize, sample=sample)
    audit = {a['chunk_id']: a for a in saved['candidate_audit']}
    if (result['selected_ids'] != prior['selected_ids']
            or result['selected_roles'] != prior['selected_roles']
            or result['input_fingerprint'] != prior['input_fingerprint']
            or result['selected_ids'] != analysis['summary']['saved_selected_ids']
            or result['chronological_selected_ids'] != analysis['saved_chronological_selection']):
        raise ValueError('Task 7.9 saved control mismatch')
    by_id = {a['chunk_id']: a for a in analysis['candidate_analysis']}
    if set(by_id) != set(audit):
        raise ValueError('Task 7.9 candidate IDs mismatch')
    for cid, a in audit.items():
        old = by_id[cid]
        for key in ('document_id', 'source_id', 'source', 'title', 'section', 'url', 'evidence_text_sha256', 'fusion_rank', 'rrf_score', 'reranked_rank'):
            if a[key] != old[key]:
                raise ValueError('Task 7.9 metadata mismatch: ' + key)
        for role in old['roles']:
            events = [e for e in result['events'] if e['chunk_id'] == cid and e['role'] == role['role']]
            if events != role['saved_pass_events']:
                raise ValueError('Task 7.9 rejection/selection events mismatch')
    selected = [{k: audit[cid][k] for k in ('chunk_id', 'document_id', 'title', 'section')} | {'role': result['selected_roles'][cid]} for cid in result['selected_ids']]
    if len(selected) != saved['statistics']['selected'] or len(selected) != analysis['summary']['selected_candidates']:
        raise ValueError('Task 7.9 count mismatch')
    return {**prior, 'passed': True, 'selected': selected, 'final_count': len(selected),
            'exact_task79_events_documents_count': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if any(os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('Use unchanged run_memory_bounded.py')
    output = args.output.resolve()
    if ROOT / 'outputs/task7.10' not in output.parents:
        raise ValueError('New Task 7.10 output directory required')
    output.mkdir(parents=True, exist_ok=False)
    monitor = Monitor(output)
    report = {'task': '7.10', 'status': 'started', 'controls': [], 'prototype_executed': False}
    try:
        monitor.begin('authenticate-task79-protected-files')
        before = authenticate(monitor)
        write(output / 'protected-files-before.json', before)
        prior = load(TASK79 / 'control-reproduction.json', monitor)
        choose, tokenize = original_functions(ROOT)
        for case in (1, 2, 3):
            monitor.begin(f'validate:{case}:exact-task79-controls')
            pair = load_case(case, monitor)
            for condition, saved in pair.items():
                old = next(r for r in prior if r['case'] == case and r['score_condition'] == condition)
                analysis = load(TASK79 / f'validate-{case}/{condition}-selection-analysis.json', monitor)
                result = exact_control(saved, old, analysis, choose, tokenize, monitor.sample)
                report['controls'].append(result)
            del pair, saved, analysis
            gc.collect()
        require_all_controls(report['controls'])
        report['all_controls_passed'] = True
        monitor.begin('verify-protected-files-after-control-preflight')
        after = verify(before, monitor)
        write(output / 'protected-files-after.json', after)
        report.update(status='control_preflight_completed', memory=monitor.totals(), protected_file_count=len(after), protected_files_unchanged=True)
        write(output / 'report.json', report)
        print('All six Task 7.9 controls reproduced exactly.', flush=True)
        return 0
    except Exception as exc:
        report.update(status='stopped', stop={'phase': monitor.phase, 'type': type(exc).__name__, 'reason': str(exc)}, memory=monitor.totals())
        write(output / 'report.json', report)
        print('STOP: ' + str(exc), flush=True)
        return 75 if isinstance(exc, MemoryStop) else 1


if __name__ == '__main__':
    raise SystemExit(main())
