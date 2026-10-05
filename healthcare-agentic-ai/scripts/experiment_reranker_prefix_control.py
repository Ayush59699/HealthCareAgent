"""Opt-in Task 7.7 prefix-only control. Launch with run_memory_bounded.py.

Exactly validate:1,2,3; one cached case at a time, original then fixed prefix.
All limits and model settings are fixed, not exposed as tuning arguments.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ablate_medical_ranking import MemoryMonitor, MemorySafetyStop, current_payloads, write_json
from rag.experimental_medical.prefix_integrity import (
    digest, verify_historical, capture_additional, verify_unchanged)

ROOT = Path(__file__).resolve().parents[1]


class RunMemoryMonitor(MemoryMonitor):
    """Retain original synchronous stop; additionally track whole-run/tree peaks."""
    def __init__(self, output):
        self.run_system_peak = self.tree_peak = self.run_process_peak = 0.
        super().__init__(output)

    def sample(self):
        super().sample()  # unchanged 13.5 GB fail-closed check
        self.run_system_peak = max(self.run_system_peak, self.system_peak)
        self.run_process_peak = max(self.run_process_peak, self.process_peak)
        processes = [self.process, *self.process.children(recursive=True)]
        rss = 0
        for process in processes:
            try:
                rss += process.memory_info().rss
            except self.psutil.NoSuchProcess:
                pass
        self.tree_peak = max(self.tree_peak, rss / 1e9)

    def totals(self):
        self.sample()
        return {'system_peak_gb': self.run_system_peak,
                'process_rss_peak_gb': self.run_process_peak, 'process_tree_rss_peak_gb': self.tree_peak,
                'method': 'synchronous samples; outer watchdog separately samples every 0.05 seconds'}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    parser.add_argument('--authorized-preflight', type=Path, required=True,
                        help='Completed authorized preflight directory; must still match')
    return parser.parse_args(argv)


def markdown_report(report):
    lines = ['# Task 7.7 — prefix-only role control', '',
        'Only the two fixed role prefixes were prepended. No prefix/threshold/model tuning.',
        'Single authorized historical exception: `rag/config.py`, the exact user-requested `OKF_ENABLED = True` addition; current bytes remained frozen.',
        'Historical manifests, prior blocked runs, candidates, assignments, evidence, retrieval features, selectors and evidence contracts are unchanged.', '',
        '| Case | Input | Positive current/history | Nonpositive→positive current/history | Final evidence | Time (s) | Peak system RAM (GB) |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for case in report['cases']:
        transitions = case['eligibility_transitions']
        for variant in case['variants']:
            stats = variant['statistics']
            names = '; '.join(r['title'] + ' / ' + (r['section'] or '(no section)') for r in variant['selected']) or '(none)'
            flips = 'control' if variant['format'] == 'original' else f"{transitions['symptoms']['nonpositive_to_positive']}/{transitions['history']['nonpositive_to_positive']}"
            lines.append(f"| {case['patient_id']} | {variant['format']} | {stats['positive_symptom_scores']}/{stats['positive_history_scores']} | {flips} | {names} | {variant['timings']['total_seconds']:.3f} | {variant['memory']['system_peak_gb']:.3f} |")
    for case in report['cases']:
        lines += ['', f"## {case['patient_id']}", '',
                  '**Original baseline control against Tasks 7.5 and 7.6**',
                  '```json', json.dumps(case['baseline_control'], indent=2), '```',
                  '**Eligibility transitions (every scored role pair)**',
                  '```json', json.dumps(case['eligibility_transitions'], indent=2), '```',
                  '**Frozen queries → exact prefixed inputs**']
        for q in case['formatted_queries']:
            lines += [f"- {q['query_id']} ({q['role']}): `{q['retrieval_query']}` → `{q['reranker_input']}`"]
        lines += ['', 'All specified titles/sections, including rejected passages, are recorded in `important-candidates.json`; every pair is traced in `pair-comparison.json`.']
    lines += ['', '## Boundaries',
        'Higher logits or nonempty evidence do not establish better medical evidence. Empty evidence is not failure of a safety system. Disease-specific background is not a patient diagnosis.',
        'No benchmark labels, downstream clinical agents, cloud calls, model downloads, fresh retrieval/query generation/fusion, or tuning were used.',
        'See the accompanying final interpretation report for the qualitative audit, outcome classification, tests and exactly one next recommendation.']
    return '\n'.join(lines) + '\n'


def run_case(sample, patient, reranker, monitor, output):
    from rag.focused_medical import state_from_patient
    from rag.experimental_medical.reranker_prefix_control import (
        FORMATS, pair_plan, check_token_budget, validate_frozen_case, run_format,
        baseline_control, require_baseline, compare_pairs, important_audit)
    if patient.labels is not None:
        raise ValueError('Hidden labels not permitted')
    directory = output / f'validate-{sample}'
    directory.mkdir(exist_ok=False)
    monitor.begin(f'validate:{sample}:load-and-verify-frozen-case')
    source75 = ROOT / 'outputs/task7.5/validated-ranking-ablation' / f'validate-{sample}'
    source76 = ROOT / 'outputs/task7.6/role-explicit-inputs' / f'validate-{sample}'
    checkpoint = source75 / 'frozen-depth-50.json'
    sha = digest(checkpoint, monitor.sample)
    summary75 = json.loads((source75 / 'summary.json').read_text(encoding='utf8'))
    summary76 = json.loads((source76 / 'summary.json').read_text(encoding='utf8'))
    if sha != summary75['checkpoint_sha256'] or sha != summary76['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash differs from Task 7.5/7.6')
    if digest(source76 / 'frozen-depth-50.json', monitor.sample) != sha:
        raise ValueError('Task 7.6 frozen checkpoint differs from Task 7.5')
    cached = json.loads(checkpoint.read_text(encoding='utf8'))
    experiment = cached['experiment']
    state = state_from_patient(patient.patient, patient.patient_id)
    if state.patient_id != summary75['patient_id'] or state.model_dump() != summary76['patient_state']:
        raise ValueError('Case identity/state changed')
    payloads = current_payloads({r['chunk_id'] for r in experiment['candidate_records']}, monitor)
    records = validate_frozen_case(state, cached, payloads, monitor.sample)
    if reranker.signature != experiment['reranker']:
        raise ValueError('Local reranker signature mismatch')
    previous_original = json.loads((source76 / 'original.json').read_text(encoding='utf8'))
    case = {'patient_id': state.patient_id, 'patient_state': state.model_dump(),
            'checkpoint_sha256': sha, 'candidate_count': len(experiment['candidate_pool']), 'variants': []}
    # Both exact inputs must fit without truncation before any fresh scoring.
    monitor.begin(f'validate:{sample}:token-preflight-both-formats')
    token_checks = {}
    for variant in FORMATS:
        pairs, audit, _ = pair_plan(state, experiment, records, variant)
        lengths = check_token_budget(reranker, pairs, audit, monitor.sample)
        token_checks[variant] = {'pairs': len(pairs), 'maximum_pair_tokens': max(lengths, default=0), 'over_512': 0}
    write_json(directory / 'token-preflight.json', token_checks)
    del pairs, audit, lengths
    original = None
    for variant in FORMATS:
        if variant == 'prefix_only':
            require_baseline(case['baseline_control'])  # no prefix call on a failed control
        monitor.begin(f'validate:{sample}:{variant}:scoring')
        def progress(done, total):
            if done == 0 or done % 100 == 0 or done == total:
                data = {'case': state.patient_id, 'phase': monitor.phase,
                        'completed_pairs': done, 'total_pairs': total, **monitor.totals()}
                write_json(output / 'scoring-progress.json', data)
                print(f"{monitor.phase}: {done}/{total}; system peak={data['system_peak_gb']:.3f} GB; tree RSS peak={data['process_tree_rss_peak_gb']:.3f} GB", flush=True)
        progress(0, experiment['reranker_pair_count'])
        result = run_format(state, experiment, records, reranker, variant, monitor.sample, progress)
        result['memory'] = monitor.measurement()
        result['checkpoint_sha256'] = sha
        filename = 'baseline.json' if variant == 'original' else 'prefix_only.json'
        write_json(directory / filename, result)
        if variant == 'original':
            control = baseline_control(experiment, result, previous_original)
            case['baseline_control'] = control
            write_json(directory / 'baseline-control.json', control)
            require_baseline(control)
            original = result
        else:
            comparison = compare_pairs(original, result, experiment)
            write_json(directory / 'pair-comparison.json', comparison)
            write_json(directory / 'important-candidates.json', important_audit(state.patient_id, comparison))
            case.update({k: comparison[k] for k in ('eligibility_transitions', 'selected_gained', 'selected_lost')})
            case['formatted_queries'] = result['formatted_queries']
        by_id = {r['chunk_id']: r for r in result['candidate_audit']}
        selected = [{k: by_id[cid][k] for k in ('chunk_id', 'title', 'section', 'selected_role', 'final_selection_reason')}
                    for cid in result['selected_ids']]
        case['variants'].append({'format': variant, 'statistics': result['statistics'],
            'timings': result['timings'], 'memory': result['memory'], 'selected': selected,
            'selected_ids': result['selected_ids'], 'snapshot_hash': result['evidence_snapshot']['snapshot_id'],
            'pair_count': result['pair_count'], 'max_pair_tokens': result['max_pair_tokens']})
        print(f"{state.patient_id} {variant}: {result['statistics']}", flush=True)
    write_json(directory / 'summary.json', case)
    return case  # all pool/pair/model-output temporaries released on return


def main(argv=None):
    args = arguments(argv)
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Refusing to overwrite an existing output directory')
    # The outer watchdog must supply the established two-thread environment.
    if any(os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('Launch through unchanged scripts/run_memory_bounded.py')
    output.mkdir(parents=True, exist_ok=False)
    monitor = RunMemoryMonitor(output)
    report = {'task': '7.7', 'status': 'started', 'cases': [], 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'labels_loaded': False, 'benchmark_contents_inspected': False, 'retrieval_calls': 0,
              'query_generation_calls': 0, 'rrf_generation_calls': 0, 'cloud_calls': 0,
              'model_downloads': 0, 'embedding_loads': 0, 'downstream_agent_calls': 0,
              'outer_stop_gb': 14.0, 'startup_reserve_gb': 1.5, 'synchronous_stop_gb': 13.5}
    before = None
    try:
        monitor.begin('verify-authorized-preflight-still-unchanged')
        initial = json.loads((args.authorized_preflight / 'frozen-files-before.json').read_text(encoding='utf8'))
        verify_unchanged(initial, monitor.sample)
        monitor.begin('complete-protected-integrity-with-one-authorized-exception')
        before, integrity = verify_historical(ROOT, monitor.sample)
        before = capture_additional(ROOT, output, before, monitor.sample)
        write_json(output / 'protected-file-exception.json', integrity)
        write_json(output / 'frozen-files-before.json', before)
        print('AUTHORIZED EXCEPTION ONLY: rag/config.py; exact intentional OKF_ENABLED = True addition, authorized by Task 7.7.1. Historical manifests untouched; all other paths strict.', flush=True)
        from rag.experimental_medical.reranker_prefix_control import VERSION, FORMATS, PREFIXES
        from rag.experimental_medical.retrieval import LocalCrossEncoder
        from rag.patient_parser import DDXPlusParser
        report.update(version=VERSION, formats=FORMATS, prefixes=PREFIXES, batch_size=2, threads=2,
                      candidate_depth=50, maximum_pair_length=512, positive_logit_gate='raw role logit > 0',
                      selector='unchanged task74', protected_integrity=integrity)
        write_json(output / 'configuration.json', report)
        monitor.begin('load-pinned-cached-local-cross-encoder-cpu')
        start = perf_counter()
        import torch
        torch.set_num_threads(2)
        reranker = LocalCrossEncoder(allow_download=False, batch_size=2)
        report.update(model=reranker.signature, model_load_seconds=perf_counter()-start,
                      model_load_memory=monitor.measurement())
        write_json(output / 'configuration.json', report)
        parser = DDXPlusParser()
        for sample, patient in enumerate(parser.iter_patients('validate', limit=3, include_labels=False), 1):
            monitor.begin(f'validate:{sample}:integrity-before-case')
            verify_unchanged(before, monitor.sample)
            report['cases'].append(run_case(sample, patient, reranker, monitor, output))
            gc.collect()
            write_json(output / 'comparison.json', report)
        if len(report['cases']) != 3:
            raise ValueError('Exactly three existing frozen cases required')
        del reranker, parser, patient
        gc.collect()
        monitor.begin('verify-protected-files-after-completion')
        after = verify_unchanged(before, monitor.sample)
        verify_historical(ROOT, monitor.sample)
        write_json(output / 'frozen-files-after.json', after)
        report.update(status='completed', frozen_files_unchanged=True, protected_file_count=len(before),
                      memory=monitor.totals())
        write_json(output / 'comparison.json', report)
        (output / 'comparison.md').write_text(markdown_report(report), encoding='utf8')
        monitor.begin('completed')
        print('Report: ' + str(output / 'comparison.md'), flush=True)
        return 0
    except Exception as exc:
        stopped = {'phase': monitor.phase, 'reason': str(exc), 'error_type': type(exc).__name__,
                   'completed_cases': len(report['cases']), 'system_peak_gb': monitor.run_system_peak,
                   'process_tree_rss_peak_gb': monitor.tree_peak}
        write_json(output / 'experiment-stopped.json', stopped)
        report.update(status='stopped', stop=stopped)
        write_json(output / 'comparison.json', report)
        # Never keep hashing/scoring after a synchronous memory stop.
        if not isinstance(exc, MemorySafetyStop) and before is not None:
            try:
                after = verify_unchanged(before, monitor.sample)
                write_json(output / 'frozen-files-after.json', after)
            except Exception as integrity_exc:
                write_json(output / 'post-stop-integrity-error.json', {'reason': str(integrity_exc)})
        print(json.dumps(stopped), flush=True)
        return 75 if isinstance(exc, MemorySafetyStop) else 2


if __name__ == '__main__':
    raise SystemExit(main())
