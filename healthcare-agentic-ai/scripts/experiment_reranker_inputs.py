"""Task 7.6: sequential same-model reranker-input experiment, frozen depth 50.

Run through scripts/run_memory_bounded.py. Only query-side formatting changes.
No retrieval, index writes, downloads, cloud inference or production activation.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ablate_medical_ranking import (MemoryMonitor, MemorySafetyStop,
    current_payloads, protected_hashes, write_json)
from rag.experimental_medical.inspection import file_digest


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, choices=[1, 2, 3], nargs='+', default=[1, 2, 3])
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/task7.5/validated-ranking-ablation')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, help='NEW output directory; refuses overwrite')
    return parser.parse_args(argv)


def frozen_hashes(artifacts):
    hashes = protected_hashes()
    root = Path(__file__).resolve().parents[1]
    extra = [root / 'rag/experimental_medical/ranking_ablation.py',
             root / 'scripts/ablate_medical_ranking.py', Path(__file__).resolve()]
    extra += [p for p in artifacts.rglob('*') if p.is_file()]
    for path in extra:
        hashes[str(path.resolve())] = file_digest(path)
    return hashes


def markdown_report(report):
    lines = ['# Task 7.6 — role-explicit reranker-input experiment', '',
        'Same frozen depth-50 pools, pair assignments, passage text, model, batch size, feature weights and selector.',
        'Only query-side reranker wording changes. Fresh original-input control is checked against Task 7.5.',
        'Time includes formatting/token checks, model scoring, selection and audit; it excludes model load and frozen retrieval stages.',
        'RAM is sampled total-system GB, with process RSS in parentheses. Nonempty output is not clinical success.', '',
        '| Case | Input format | Positive current/history scores | Final current/history passages | Final evidence | Time (s) | Peak RAM (GB) |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for case in report['cases']:
        for variant in case['variants']:
            stats, memory = variant['statistics'], variant['memory']
            evidence = '; '.join(s['title'] + ' / ' + (s['section'] or '(no section)') for s in variant['selected']) or '(none)'
            lines.append(f"| {case['patient_id']} | {variant['format']} | {stats['positive_symptom_scores']}/{stats['positive_history_scores']} | {stats['selected_current']}/{stats['selected_background']} | {evidence} | {variant['timings']['total_seconds']:.3f} | {memory['system_peak_gb']:.3f} ({memory['process_rss_peak_gb']:.3f}) |")
    for case in report['cases']:
        lines += ['', f"## {case['patient_id']}", '', '**Unchanged observed facts**',
            '```json', json.dumps(case['patient_state'], indent=2), '```', '',
            '**Frozen pool statistics**', str(case['candidate_statistics']), '',
            '**Fresh original vs cached score control**', str(case['baseline_drift']), '',
            '**Only changed input: query-side wording**']
        for query in case['formatted_queries']:
            lines += [f"- {query['query_id']} ({query['role']}):", '  - Original: ' + query['retrieval_query'],
                      '  - Formatted: ' + query['reranker_input'], '  - Facts: ' + ', '.join(query['fact_ids'])]
        lines += ['', '**Eligibility transitions**', str(case['eligibility_transitions']), '',
            'Full candidate audit, including rejected passages, actual pair inputs, raw logits, source references and unchanged evidence snapshots, is in this case\'s JSON files.',
            'These are retrieval diagnostics, not relevance labels or diagnoses.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = arguments(argv)
    if len(set(args.samples)) != len(args.samples):
        raise ValueError('Distinct existing samples required')
    output = args.output or Path(__file__).resolve().parents[1] / 'outputs/task7.6' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists():
        raise ValueError('Output already exists; choose a NEW directory')
    for sample in args.samples:
        for name in ('frozen-depth-50.json', 'summary.json'):
            if not (args.artifacts / f'validate-{sample}' / name).is_file():
                raise ValueError('Verified Task 7.5 case artifacts required')
    if not (args.artifacts / 'frozen-files-after.json').is_file():
        raise ValueError('Task 7.5 protected-file manifest required')
    output.mkdir(parents=True, exist_ok=False)
    monitor = MemoryMonitor(output)
    try:
        monitor.begin('verify-task75-protected-files-and-capture-freeze')
        before = frozen_hashes(args.artifacts)
        with (args.artifacts / 'frozen-files-after.json').open(encoding='utf8') as stream:
            previous_freeze = json.load(stream)
        if any(before.get(path) != digest for path, digest in previous_freeze.items()):
            raise ValueError('Task 7.5 frozen sources/indexes changed; no input-only experiment allowed')
        write_json(output / 'frozen-files-before.json', before)
        from rag.experimental_medical.reranker_input_experiment import (
            VERSION, FORMATS, CURRENT_TEMPLATE, HISTORY_TEMPLATE, pair_plan,
            check_token_budget, run_format, baseline_drift, compare_formats)
        from rag.experimental_medical.ranking_ablation import validate_frozen_experiment
        from rag.experimental_medical.retrieval import LocalCrossEncoder
        from rag.patient_parser import DDXPlusParser
        from rag.focused_medical import state_from_patient
        monitor.begin('load-cached-local-cross-encoder-only')
        load_start = perf_counter()
        import torch
        torch.set_num_threads(2)
        reranker = LocalCrossEncoder(allow_download=False, batch_size=2)
        load_seconds = perf_counter() - load_start
        load_memory = monitor.measurement()
        report = {'task': '7.6', 'version': VERSION, 'candidate_depth': 50,
            'formats': FORMATS, 'templates': {'current': CURRENT_TEMPLATE, 'history': HISTORY_TEMPLATE},
            'model': reranker.signature, 'batch_size': 2, 'threads': 2,
            'model_load_seconds': load_seconds, 'model_load_memory': load_memory,
            'labels_loaded': False, 'benchmark_used': False, 'cloud_calls': 0,
            'embedding_loads': 0, 'retrieval_search_calls': 0, 'model_downloads': 0,
            'positive_logit_gate': 'unchanged: raw role logit > 0',
            'selector': 'unchanged Task 7.4; no Task 7.5 selection variants enabled',
            'execution': 'one case at a time; original then role-explicit; same supporting queries and passage inputs',
            'cases': []}
        write_json(output / 'configuration.json', report)
        for sample, patient in enumerate(DDXPlusParser(args.data_dir).iter_patients('validate', limit=max(args.samples), include_labels=False), 1):
            if sample not in args.samples:
                continue
            monitor.begin(f'validate:{sample}:load-and-verify-one-frozen-pool')
            directory = output / f'validate-{sample}'
            directory.mkdir()
            source = args.artifacts / f'validate-{sample}'
            checkpoint = source / 'frozen-depth-50.json'
            digest = file_digest(checkpoint)
            with (source / 'summary.json').open(encoding='utf8') as stream:
                previous_summary = json.load(stream)
            if digest != previous_summary['checkpoint_sha256']:
                raise ValueError('Frozen Task 7.5 checkpoint hash mismatch')
            with checkpoint.open(encoding='utf8') as stream:
                cached = json.load(stream)
            state = state_from_patient(patient.patient, patient.patient_id)
            if state.model_dump() != cached['patient_state'] or state.patient_id != previous_summary['patient_id']:
                raise ValueError('Frozen state differs from existing label-free parser case')
            experiment = cached['experiment']
            payloads = current_payloads({r['chunk_id'] for r in experiment['candidate_records']}, monitor)
            records = validate_frozen_experiment(state, experiment, payloads)
            if reranker.signature != experiment['reranker']:
                raise ValueError('Loaded reranker signature differs from frozen experiment')
            shutil.copyfile(checkpoint, directory / 'frozen-depth-50.json')
            if file_digest(directory / 'frozen-depth-50.json') != digest:
                raise ValueError('Frozen checkpoint copy mismatch')
            case_report = {'patient_id': state.patient_id, 'patient_state': state.model_dump(),
                'checkpoint_sha256': digest, 'candidate_statistics': experiment['statistics'],
                'original_task74_retrieval_seconds': experiment['timings']['total_seconds'], 'variants': []}
            # Preflight BOTH formats before scoring: no passage truncation is
            # allowed to become a hidden second experimental variable.
            monitor.begin(f'validate:{sample}:token-preflight-both-formats')
            token_checks = {}
            for variant in FORMATS:
                pairs, audit, _ = pair_plan(state, experiment, records, variant)
                lengths = check_token_budget(reranker, pairs, audit, monitor.sample)
                token_checks[variant] = {'pairs': len(pairs), 'maximum_pair_tokens': max(lengths, default=0), 'over_512': 0}
            write_json(directory / 'token-preflight.json', token_checks)
            del pairs, audit, lengths
            original_result = None
            for variant in FORMATS:
                monitor.begin(f'validate:{sample}:{variant}:scoring')
                def progress(done, total):
                    if done % 100 == 0 or done == total:
                        write_json(output / 'scoring-progress.json', {'phase': monitor.phase, 'completed_pairs': done,
                            'total_pairs': total, **monitor.measurement()})
                result = run_format(state, experiment, records, reranker, variant, monitor.sample, progress)
                result['memory'] = monitor.measurement()
                result['checkpoint_sha256'] = digest
                if variant == 'original':
                    control = baseline_drift(experiment, result)
                    case_report['baseline_drift'] = control
                    write_json(directory / 'baseline-drift.json', control)
                    write_json(directory / 'original.json', result)
                    if not control['passed']:
                        raise ValueError('Fresh original-input control drifted; no formatted comparison permitted')
                    original_result = result
                else:
                    comparison = compare_formats(original_result, result)
                    write_json(directory / 'role_explicit.json', result)
                    write_json(directory / 'pair-comparison.json', comparison)
                    case_report['eligibility_transitions'] = comparison['eligibility_transitions']
                    case_report['selected_gained'] = comparison['selected_gained']
                    case_report['selected_lost'] = comparison['selected_lost']
                    case_report['formatted_queries'] = result['formatted_queries']
                    del comparison
                by_id = {r['chunk_id']: r for r in result['candidate_audit']}
                selected = [{k: by_id[cid][k] for k in ('chunk_id', 'title', 'section', 'selected_role', 'final_selection_reason')}
                            for cid in result['selected_ids']]
                case_report['variants'].append({'format': variant, 'statistics': result['statistics'],
                    'timings': result['timings'], 'memory': result['memory'], 'selected': selected,
                    'pair_count': result['pair_count'], 'max_pair_tokens': result['max_pair_tokens']})
                print(f"  {variant}: current/history eligible={result['statistics']['positive_symptom_scores']}/{result['statistics']['positive_history_scores']}; final=" + '; '.join(s['title'] for s in selected), flush=True)
                print(f"  {result['timings']['total_seconds']:.2f}s; system={result['memory']['system_peak_gb']:.3f} GB; RSS={result['memory']['process_rss_peak_gb']:.3f} GB", flush=True)
                del result, by_id
                gc.collect()
            write_json(directory / 'summary.json', case_report)
            report['cases'].append(case_report)
            del cached, experiment, records, payloads, original_result, case_report, previous_summary
            gc.collect()
        if len(report['cases']) != len(args.samples):
            raise ValueError('Missing requested label-free case')
        monitor.begin('verify-all-frozen-files-unchanged')
        after = frozen_hashes(args.artifacts)
        write_json(output / 'frozen-files-after.json', after)
        if before != after:
            raise ValueError('Frozen input, source, index, agent or default changed')
        report['frozen_files_unchanged'] = True
        report['frozen_file_count'] = len(before)
        write_json(output / 'comparison.json', report)
        (output / 'comparison.md').write_text(markdown_report(report), encoding='utf8')
        monitor.begin('completed')
        print('Report: ' + str(output / 'comparison.md'), flush=True)
        return 0
    except MemorySafetyStop as exc:
        print(str(exc), flush=True)
        return 75
    except Exception as exc:
        write_json(output / 'experiment-stopped.json', {'phase': monitor.phase, 'error_type': type(exc).__name__, 'reason': str(exc)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
