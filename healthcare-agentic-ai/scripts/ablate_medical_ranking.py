"""Task 7.5 sequential, label-free frozen-pool ranking ablations.

Run via scripts/run_memory_bounded.py. Reuses ONLY task-7.4 depth-50 checkpoints;
never loads models, opens a writable Qdrant store, searches, or reads labels.
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

from rag.medical_ingestion.common import DATA_ROOT


class MemorySafetyStop(RuntimeError):
    pass


class MemoryMonitor:
    """Synchronous samples only: no additional workers/threads/experiments."""
    def __init__(self, output):
        import psutil
        self.psutil, self.process = psutil, psutil.Process()
        self.output, self.phase = output, 'startup'
        self.system_peak, self.process_peak = 0., 0.

    def sample(self):
        system = self.psutil.virtual_memory().used / 1e9
        rss = self.process.memory_info().rss / 1e9
        self.system_peak, self.process_peak = max(system, self.system_peak), max(rss, self.process_peak)
        if system >= 13.5:  # stop before the unchanged outer watchdog's 14 GB stop
            write_json(self.output / 'memory-stop.json', {'phase': self.phase,
                'system_gb': system, 'process_rss_gb': rss,
                'reason': 'approaching unchanged watchdog 14 GB stop / user 15 GB ceiling'})
            raise MemorySafetyStop(f'Memory stop at {self.phase}: system={system:.3f} GB; RSS={rss:.3f} GB')

    def begin(self, phase):
        self.phase = phase
        self.system_peak, self.process_peak = 0., 0.
        self.sample()
        write_json(self.output / 'current-phase.json', {'phase': phase,
            'system_gb': self.system_peak, 'process_rss_gb': self.process_peak})
        print(phase, flush=True)

    def measurement(self):
        self.sample()
        return {'system_peak_gb': self.system_peak, 'process_rss_peak_gb': self.process_peak,
                'method': 'synchronous samples during every selector/audit iteration; decimal GB, not an OS high-water mark'}


def write_json(path, value):
    with Path(path).open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, choices=[1, 2, 3], nargs='+', default=[1, 2, 3])
    parser.add_argument('--artifacts', type=Path, default=Path(__file__).resolve().parents[1] / 'outputs/task7.4/qdrant-validation')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, help='New output directory; refuses overwrite')
    return parser.parse_args(argv)


def protected_hashes():
    """Content-only checks, excluding benchmark files and newly added ablations."""
    from rag.experimental_medical.inspection import file_digest
    root = Path(__file__).resolve().parents[1]
    sources = []
    for directory in ('rag', 'orchestration', 'safety', 'demo'):
        sources.extend(p for p in (root / directory).rglob('*.py') if p.name != 'ranking_ablation.py')
    sources += [root / 'scripts/run_memory_bounded.py', root / 'scripts/run_phase6.py',
                root / 'scripts/build_medical_index.py', root / 'scripts/build_experimental_medical_index.py']
    sources += [DATA_ROOT / 'corpus.json']
    for name in ('raw', 'processed', 'qdrant', 'experimental-task74'):
        sources.extend(p for p in (DATA_ROOT / name).rglob('*') if p.is_file())
    return {str(p.resolve()): file_digest(p) for p in sorted(set(sources))}


def current_payloads(ids, monitor):
    """Read approved source-only chunks, not raw patient/benchmark tables."""
    result = {}
    with (DATA_ROOT / 'processed/chunks.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            payload = json.loads(line)
            if payload['chunk_id'] in ids:
                if payload['chunk_id'] in result:
                    raise ValueError('Duplicate current corpus chunk')
                result[payload['chunk_id']] = payload
            monitor.sample()
    return result


def markdown_summary(report):
    lines = ['# Task 7.5 — frozen-pool ranking ablations', '',
        'Sequential depth-50 replay, no neural model/search reruns. Findings are not relevance labels or diagnoses.',
        'Time is selector + exhaustive audit + unchanged snapshot validation only; original upstream latency is reported separately.',
        'Peak RAM is sampled total-system GB, with process RSS in parentheses. Other applications contribute to system usage.', '',
        '| Case | Ranking variant | Final evidence | Useful candidate missed? | Main selection issue | Time (s) | Peak RAM (GB) |',
        '| ---- | --------------- | -------------- | ------------------------ | -------------------- | -------: | ------------: |']
    for case in report['cases']:
        for result in case['variants']:
            counts = result['statistics']
            names = '; '.join(r['title'] + ' / ' + (r['section'] or '(no section)') for r in result['selected']) or '(none)'
            issue = f"positive current/history scores: {counts['positive_symptom_scores']}/{counts['positive_history_scores']}; final current/history: {counts['selected_current']}/{counts['selected_background']}"
            memory = result['memory']
            lines.append(f"| {case['patient_id']} | {result['variant']} | {names} | See qualitative case analysis; nonempty is not success | {issue} | {result['replay_seconds']:.4f} | {memory['system_peak_gb']:.3f} ({memory['process_rss_peak_gb']:.3f}) |")
    for case in report['cases']:
        lines += ['', f"## {case['patient_id']} stage trace", '', '**Observed findings (unchanged)**',
                  '```json', json.dumps(case['patient_state'], indent=2), '```', '', '**Queries (identical in all five variants)**']
        lines += ['- ' + q['query_id'] + ': ' + q['text'] for q in case['queries']]
        lines += ['', '**Frozen candidate generation / RRF / learned reranker**',
                  str(case['candidate_statistics']),
                  f"Original upstream+selection time: {case['original_task74_seconds']:.3f} seconds (not remeasured).",
                  f"Source checkpoint SHA-256: `{case['checkpoint_sha256']}`.",
                  f"All original payloads and scores: `{case['directory']}/frozen-depth-50.json`.",
                  'Each variant JSON includes every candidate, role-specific query/ranks/scores, population features, selector events and removal stage.', '']
    lines += ['## Interpretation boundary',
        'No relevance annotations, hidden diagnoses, benchmark metrics or automatic production promotion.',
        'See docs/task7.5-ranking-ablation.md for qualitative passage inspection and exactly one next engineering recommendation.']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = arguments(argv)
    if len(set(args.samples)) != len(args.samples):
        raise ValueError('Distinct existing sample IDs required')
    output = args.output or Path(__file__).resolve().parents[1] / 'outputs/task7.5' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')
    for sample in args.samples:
        if not (args.artifacts / f'validate-{sample}-depth-50.json').is_file():
            raise ValueError('Required Task 7.4 depth-50 checkpoint unavailable')
    output.mkdir(parents=True, exist_ok=False)
    monitor = MemoryMonitor(output)
    try:
        monitor.begin('hash-frozen-sources-and-indexes')
        before = protected_hashes()
        write_json(output / 'frozen-files-before.json', before)
        from rag.experimental_medical.ranking_ablation import VARIANTS, DESCRIPTIONS, ablate, validate_frozen_experiment
        from rag.experimental_medical.inspection import file_digest
        from rag.patient_parser import DDXPlusParser
        from rag.focused_medical import state_from_patient
        from orchestration.evidence import fingerprint
        implementation = {str(p): file_digest(p) for p in (Path(__file__), Path(__file__).resolve().parents[1] / 'rag/experimental_medical/ranking_ablation.py')}
        report = {'implementation_sha256': implementation, 'task': '7.5', 'candidate_depth': 50, 'labels_loaded': False,
            'benchmark_used': False, 'cloud_calls': 0, 'model_loads': 0, 'search_calls': 0,
            'execution': 'one case at a time; variants sequential; verified task74 score replay',
            'variants': DESCRIPTIONS, 'cases': []}
        for i, patient in enumerate(DDXPlusParser(args.data_dir).iter_patients('validate', limit=max(args.samples), include_labels=False), 1):
            if i not in args.samples:
                continue
            monitor.begin(f'validate:{i}:load-one-frozen-case')
            checkpoint = args.artifacts / f'validate-{i}-depth-50.json'
            case_dir = output / f'validate-{i}'
            case_dir.mkdir()
            source_hash = file_digest(checkpoint)
            with checkpoint.open(encoding='utf8') as stream:
                cached = json.load(stream)
            state = state_from_patient(patient.patient, patient.patient_id)
            if cached['patient_state'] != state.model_dump():
                raise ValueError('Checkpoint state differs from label-free parser record')
            experiment = cached['experiment']
            ids = {r['chunk_id'] for r in experiment['candidate_records']}
            payloads = current_payloads(ids, monitor)
            monitor.begin(f'validate:{i}:verify-query-fusion-source-and-snapshot')
            records = validate_frozen_experiment(state, experiment, payloads)
            if file_digest(checkpoint) != source_hash:
                raise ValueError('Checkpoint changed during loading')
            shutil.copyfile(checkpoint, case_dir / 'frozen-depth-50.json')
            if file_digest(case_dir / 'frozen-depth-50.json') != source_hash:
                raise ValueError('Checkpoint copy integrity failure')
            immutable = fingerprint(experiment)
            case_report = {'patient_id': state.patient_id, 'patient_state': state.model_dump(),
                'queries': experiment['plan']['queries'], 'candidate_statistics': experiment['statistics'],
                'original_task74_seconds': experiment['timings']['total_seconds'],
                'checkpoint_sha256': source_hash, 'directory': case_dir.name, 'variants': []}
            for variant in VARIANTS:
                monitor.begin(f'validate:{i}:{variant}')
                started = perf_counter()
                result = ablate(state, experiment, records, variant, monitor.sample)
                elapsed = perf_counter() - started
                result.update(replay_seconds=elapsed, memory=monitor.measurement(), checkpoint_sha256=source_hash)
                if fingerprint(experiment) != immutable:
                    raise ValueError('Ablation mutated a frozen input')
                write_json(case_dir / (variant + '.json'), result)
                by_id = {r['chunk_id']: r for r in result['candidate_audit']}
                selected = [{k: by_id[chunk_id][k] for k in ('chunk_id', 'title', 'section', 'selected_role', 'selection_or_removal_stage', 'final_selection_reason')}
                            for chunk_id in result['selected_ids']]
                case_report['variants'].append({'variant': variant, 'replay_seconds': elapsed,
                    'memory': result['memory'], 'statistics': result['statistics'], 'selected': selected})
                print('  final: ' + '; '.join(r['title'] for r in selected), flush=True)
                print(f"  replay={elapsed:.4f}s; system={result['memory']['system_peak_gb']:.3f} GB; RSS={result['memory']['process_rss_peak_gb']:.3f} GB", flush=True)
                del result, by_id
                gc.collect()
            report['cases'].append(case_report)
            write_json(case_dir / 'summary.json', case_report)
            del cached, experiment, records, payloads, case_report
            gc.collect()
        if len(report['cases']) != len(args.samples):
            raise ValueError('Missing requested label-free validation case')
        monitor.begin('verify-frozen-sources-and-indexes-unchanged')
        after = protected_hashes()
        write_json(output / 'frozen-files-after.json', after)
        if before != after:
            raise ValueError('A frozen source/index/default changed during ablation')
        report['frozen_files_unchanged'] = True
        write_json(output / 'comparison.json', report)
        (output / 'comparison.md').write_text(markdown_summary(report), encoding='utf8')
        monitor.begin('completed')
        print('Comparison: ' + str(output / 'comparison.md'), flush=True)
        return 0
    except MemorySafetyStop as exc:
        print(str(exc), flush=True)
        return 75


if __name__ == '__main__':
    raise SystemExit(main())
