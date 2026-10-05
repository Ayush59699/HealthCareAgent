"""Task 7.8: ONE isolated same-pair biomedical reranker comparison.

Launch only through unchanged scripts/run_memory_bounded.py. No tuning flags.
All three controls must reproduce before any experimental model scoring.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.ablate_medical_ranking import current_payloads, write_json, MemorySafetyStop
from scripts.experiment_reranker_prefix_control import RunMemoryMonitor
from rag.experimental_medical.prefix_integrity import digest, verify_unchanged, capture_additional

# Anchor to the COMPLETED Task 7.7 closeout, not its Task-7.7-only historical
# exception. Every current file (including config) must match strictly.
CLOSEOUT = ROOT / 'outputs/task7.7/prefix-role-control-authorized-20261003/closeout/frozen-files-closeout.json'
CLOSEOUT_SHA256 = '50f4d50b1ac2d9ad2dc3ff93345576d4f712070ff9d2b5c3a0e644e18808c017'


def load_case(sample, monitor):
    from rag.patient_parser import DDXPlusParser
    from rag.focused_medical import state_from_patient
    from rag.experimental_medical.reranker_prefix_control import validate_frozen_case
    source = ROOT / 'outputs/task7.5/validated-ranking-ablation' / f'validate-{sample}'
    checkpoint = source / 'frozen-depth-50.json'
    sha = digest(checkpoint, monitor.sample)
    cached = json.loads(checkpoint.read_text(encoding='utf8'))
    for directory in ('outputs/task7.5/validated-ranking-ablation', 'outputs/task7.6/role-explicit-inputs'):
        previous = ROOT / directory / f'validate-{sample}'
        summary = json.loads((previous / 'summary.json').read_text(encoding='utf8'))
        if summary['checkpoint_sha256'] != sha or digest(previous / 'frozen-depth-50.json', monitor.sample) != sha:
            raise ValueError('Frozen checkpoint differs from historical references')
    patient = next(p for i, p in enumerate(DDXPlusParser().iter_patients('validate', limit=3, include_labels=False), 1) if i == sample)
    if patient.labels is not None:
        raise ValueError('Hidden labels forbidden')
    state = state_from_patient(patient.patient, patient.patient_id)
    experiment = cached['experiment']
    payloads = current_payloads({r['chunk_id'] for r in experiment['candidate_records']}, monitor)
    records = validate_frozen_case(state, cached, payloads, monitor.sample)
    return state, experiment, records, sha


def baseline_checks(experiment, result, sample):
    from rag.experimental_medical.reranker_prefix_control import baseline_control
    checks = {}
    for task, directory, name in (
        ('task76', 'outputs/task7.6/role-explicit-inputs', 'original.json'),
        ('task77', 'outputs/task7.7/prefix-role-control-authorized-20261003', 'baseline.json')):
        old = json.loads((ROOT / directory / f'validate-{sample}' / name).read_text(encoding='utf8'))
        checks[task] = baseline_control(experiment, result, old)
    checks['passed'] = all(c['passed'] for c in checks.values())
    return checks


def require_all_controls(output):
    for sample in (1, 2, 3):
        checks = json.loads((output / f'validate-{sample}' / 'baseline-control.json').read_text(encoding='utf8'))
        if not checks['passed']:
            raise ValueError('Baseline failed; experimental scoring forbidden')


def important_audit(comparison, control):
    from rag.experimental_medical.suitability_audit import important_audit as audit
    return audit(comparison, control)


def markdown(report):
    lines = ['# Task 7.8 â€” controlled reranker suitability', '',
             f"Status: **{report['status']}**", '',
             'Same verbatim frozen pairs, raw logit > 0, unchanged Task 7.4 selection.', '',
             '| Case | Role | Nonpositive â†’ positive | Positive â†’ nonpositive | Positive â†’ positive | Nonpositive â†’ nonpositive |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for case in report['cases']:
        for role, counts in case['eligibility_transitions'].items():
            lines.append(f"| {case['patient_id']} | {role} | " + ' | '.join(str(counts[k]) for k in (
                'nonpositive_to_positive', 'positive_to_nonpositive', 'positive_to_positive', 'nonpositive_to_nonpositive')) + ' |')
        lines += ['', f"## {case['patient_id']}", 'Control selected: ' + '; '.join(case['control_selected']),
                  'Experimental selected: ' + ('; '.join(case['experimental_selected']) or '(none)'),
                  'Inspect every requested passage/role and its rejection or selection reason in `important-candidates.json`.']
    lines += ['', '## Interpretation limits',
        'The alternative uses a differently trained raw logit scale. More positive logits alone do not prove better relevance or that the control is unsuitable. Review text compatibility, positive-control preservation and selection displacement separately.',
        'No benchmark/hidden labels or downstream diagnostic agents used. No clinical accuracy or diagnostic correctness claim. No automatic promotion.',
        'A final label-free qualitative interpretation belongs in docs/task7.8-reranker-suitability.md.', '',
        '## Exactly one recommended next step',
        'Review this single fixed-gate comparison before authorizing any subsequent change; do not promote automatically.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory only')
    args = parser.parse_args(argv)
    if any(os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('Launch through unchanged scripts/run_memory_bounded.py')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    monitor = RunMemoryMonitor(output)
    report = {'task': '7.8', 'status': 'started', 'cases': [], 'controls_completed': [],
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'cpu_only': True, 'batch_size': 2, 'threads': 2, 'candidate_depth': 50,
              'positive_gate': 'raw role logit > 0', 'query_format': 'verbatim original',
              'outer_stop_gb': 14., 'startup_reserve_gb': 1.5, 'synchronous_stop_gb': 13.5,
              'labels_loaded': False, 'benchmark_contents_inspected': False,
              'retrieval_calls': 0, 'embedding_generation_calls': 0, 'downstream_agent_calls': 0,
              'control_scored_pairs': 0, 'experimental_scored_pairs': 0}
    before = None
    try:
        monitor.begin('strict-task77-closeout-integrity')
        if digest(CLOSEOUT, monitor.sample) != CLOSEOUT_SHA256:
            raise ValueError('Task 7.7 closeout manifest anchor changed')
        before = json.loads(CLOSEOUT.read_text(encoding='utf8'))
        verify_unchanged(before, monitor.sample)
        before[str(CLOSEOUT)] = CLOSEOUT_SHA256
        before = capture_additional(ROOT, output, before, monitor.sample)
        write_json(output / 'frozen-files-before.json', before)
        monitor.begin('deterministic-tests-no-model-or-network')
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            'tests.test_reranker_suitability', 'tests.test_reranker_prefix_control',
            'tests.test_reranker_input_experiment', 'tests.test_ranking_ablation'])
        with (output / 'tests.log').open('w', encoding='utf8') as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report['tests'] = {'run': tests.testsRun, 'failures': len(tests.failures), 'errors': len(tests.errors), 'passed': tests.wasSuccessful()}
        if not tests.wasSuccessful():
            raise ValueError('Deterministic regression tests failed')
        monitor.begin('inspect-one-biomedical-model-metadata-no-weights')
        from rag.experimental_medical.reranker_suitability import (
            inspect_model, require_headroom, download_model, BiomedicalCrossEncoder,
            run_condition, compare_pairs)
        metadata = inspect_model(monitor.sample)
        write_json(output / 'model-preflight.json', metadata)
        require_headroom(monitor.psutil.virtual_memory().used / 1e9, metadata['memory_estimate'])
        report['experimental_model'] = metadata
        write_json(output / 'configuration.json', report)
        # All baselines first. Never hold the two models or multiple cases together.
        monitor.begin('load-cached-pinned-control-cpu')
        import torch
        torch.set_num_threads(2)
        from rag.experimental_medical.retrieval import LocalCrossEncoder
        reranker = LocalCrossEncoder(allow_download=False, batch_size=2)
        for sample in (1, 2, 3):
            monitor.begin(f'validate:{sample}:control')
            directory = output / f'validate-{sample}'
            directory.mkdir()
            verify_unchanged(before, monitor.sample)
            state, experiment, records, sha = load_case(sample, monitor)
            result = run_condition(state, experiment, records, reranker, 'control', monitor.sample)
            result.update(checkpoint_sha256=sha, memory=monitor.measurement())
            write_json(directory / 'control.json', result)
            checks = baseline_checks(experiment, result, sample)
            write_json(directory / 'baseline-control.json', checks)
            if not checks['passed']:
                raise ValueError('Fresh baseline failed against Task 7.5/7.6/7.7')
            report['controls_completed'].append(sample)
            report['control_scored_pairs'] += result['pair_count']
            write_json(output / 'comparison.json', report)
            del state, experiment, records, result
            gc.collect()
        del reranker
        gc.collect()
        require_all_controls(output)
        monitor.begin('safe-sequential-model-download')
        directory = download_model(metadata, output / 'model', monitor)
        monitor.begin('load-pinned-local-biomedical-model-cpu')
        require_headroom(monitor.psutil.virtual_memory().used / 1e9, metadata['memory_estimate'])
        reranker = BiomedicalCrossEncoder(directory, metadata)
        for sample in (1, 2, 3):
            monitor.begin(f'validate:{sample}:experimental')
            verify_unchanged(before, monitor.sample)
            require_all_controls(output)
            directory = output / f'validate-{sample}'
            state, experiment, records, sha = load_case(sample, monitor)
            def progress(done, total):
                if done % 100 == 0 or done == total:
                    write_json(output / 'scoring-progress.json', {'phase': monitor.phase, 'completed_pairs': done, 'total_pairs': total, **monitor.totals()})
            result = run_condition(state, experiment, records, reranker, 'experimental', monitor.sample, progress)
            result.update(checkpoint_sha256=sha, memory=monitor.measurement())
            write_json(directory / 'experimental.json', result)
            control = json.loads((directory / 'control.json').read_text(encoding='utf8'))
            comparison = compare_pairs(control, result, experiment)
            write_json(directory / 'pair-comparison.json', comparison)
            write_json(directory / 'important-candidates.json', important_audit(comparison, control))
            names = lambda r: [a['title'] + ' / ' + (a['section'] or '(no section)') for a in r['candidate_audit'] if a['selected']]
            case = {'patient_id': state.patient_id, 'checkpoint_sha256': sha,
                    'eligibility_transitions': comparison['eligibility_transitions'],
                    'control_selected': names(control), 'experimental_selected': names(result),
                    'selected_gained': comparison['selected_gained'], 'selected_lost': comparison['selected_lost']}
            write_json(directory / 'summary.json', case)
            report['cases'].append(case)
            report['experimental_scored_pairs'] += result['pair_count']
            write_json(output / 'comparison.json', report)
            del state, experiment, records, result, control, comparison
            gc.collect()
        del reranker
        gc.collect()
        monitor.begin('verify-all-protected-files-after-run')
        write_json(output / 'frozen-files-after.json', verify_unchanged(before, monitor.sample))
        if report['control_scored_pairs'] != 2868 or report['experimental_scored_pairs'] != 2868:
            raise ValueError('Incomplete frozen pair coverage')
        report.update(status='completed', memory=monitor.totals(), frozen_files_unchanged=True)
        write_json(output / 'comparison.json', report)
        (output / 'comparison.md').write_text(markdown(report), encoding='utf8')
        return 0
    except Exception as exc:
        report.update(status='stopped', stop={'phase': monitor.phase, 'type': type(exc).__name__, 'reason': str(exc)},
                      memory={'system_peak_gb': monitor.run_system_peak, 'process_tree_rss_peak_gb': monitor.tree_peak})
        write_json(output / 'experiment-stopped.json', report['stop'])
        write_json(output / 'comparison.json', report)
        # No more work after a memory stop. Other failures are still fail-closed.
        print(f'Task 7.8 stopped: {exc}', flush=True)
        return 75 if isinstance(exc, MemorySafetyStop) else 1


if __name__ == '__main__':
    raise SystemExit(main())
