"""Task 7.9: saved-score analysis only. Launch with unchanged run_memory_bounded.py."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'outputs/task7.8/controlled-medcpt-retry1'
REVIEW = ROOT / 'outputs/task7.8.1'
ANCHORS = {
    'outputs/task7.8/controlled-medcpt-retry1/frozen-files-after.json': '0696d69fddd40c4704d2e882db93210c4778080cd702e2e21eebe2500c286908',
    'outputs/task7.8.1/task78-input-hashes-after.json': '08ae445b8f13d79b1f319a63a7f743e29434210c236114c89f180fb69d8f1c0b',
    'outputs/task7.8.1/judgments.tsv': '2e87c0d925bf3bfe612acf5582735ee99378895bd92cbd08eb5d8d0b31670742',
    'outputs/task7.8.1/reviewed-pairs.json': 'fdbe78d8a43180d36b14462a2cc333d3605d32dc4b32bbaa642621daf9fee573'}


def write(path, value):
    with Path(path).open('w', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


class MemoryStop(RuntimeError):
    pass


class Monitor:
    def __init__(self, output):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.output = output
        self.phase = 'startup'
        self.system_peak = self.tree_peak = 0.

    def sample(self):
        system = self.psutil.virtual_memory().used / 1e9
        rss = self.process.memory_info().rss / 1e9
        for p in self.process.children(recursive=True):
            try:
                rss += p.memory_info().rss / 1e9
            except self.psutil.NoSuchProcess:
                pass
        self.system_peak, self.tree_peak = max(self.system_peak, system), max(self.tree_peak, rss)
        if system >= 13.5:
            state = {'phase': self.phase, 'operation': self.phase, 'system_gb': system, 'tree_rss_gb': rss}
            write(self.output / 'memory-stop.json', state)
            raise MemoryStop(str(state))

    def begin(self, phase):
        self.phase = phase
        self.sample()
        write(self.output / 'current-phase.json', {'phase': phase, **self.totals()})
        print(phase, flush=True)

    def totals(self):
        return {'system_peak_gb': self.system_peak, 'process_tree_rss_peak_gb': self.tree_peak}


def digest(path, monitor):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            monitor.sample()
            block = stream.read(1024*1024)
            if not block:
                return h.hexdigest()
            h.update(block)


def load(path, monitor):
    monitor.sample()
    result = json.loads(Path(path).read_text(encoding='utf8'))
    monitor.sample()
    return result


def protect(monitor):
    """No external corpus/benchmark contents: artifact and protected-source hashes only."""
    expected = {str(ROOT / name): value for name, value in ANCHORS.items()}
    for name, value in expected.items():
        if digest(name, monitor) != value:
            raise ValueError('Protected anchor mismatch: ' + name)
    task78 = load(RUN / 'frozen-files-after.json', monitor)
    code_scopes = {str(ROOT / s) for s in ('rag', 'scripts', 'tests', 'demo', 'orchestration', 'safety')}
    for name, value in task78.items():
        p = Path(name)
        if p.suffix == '.py' and any(str(parent) in code_scopes for parent in p.parents):
            expected[name] = value
    for name, value in load(REVIEW / 'task78-input-hashes-after.json', monitor).items():
        expected[str(ROOT / name)] = value
    for name, value in expected.items():
        if digest(name, monitor) != value:
            raise ValueError('Protected historical artifact/source mismatch: ' + name)
    paths = set()
    for directory in (ROOT / 'outputs/task7.8', REVIEW):
        paths.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for directory in ('rag', 'scripts', 'tests', 'demo', 'orchestration', 'safety'):
        paths.update((ROOT / directory).rglob('*.py'))
    paths.update((ROOT / 'docs').glob('task7.8*'))
    paths.add(ROOT.parent / 'okf.yaml')
    for p in sorted(paths):
        name = str(p.resolve())
        if name not in expected:
            expected[name] = digest(p, monitor)
    return expected


def verify(expected, monitor):
    for name, value in expected.items():
        if not Path(name).is_file() or digest(name, monitor) != value:
            raise ValueError('Protected file changed: ' + name)
    return dict(expected)


def load_case(case, monitor):
    from rag.experimental_medical.task79_selection_analysis import adapt_saved, POOLS
    folder = RUN / f'validate-{case}'
    pair = {name: load(folder / f'{name}.json', monitor) for name in ('control', 'experimental')}
    for saved in pair.values():
        adapt_saved(saved, POOLS[case])
        if saved['statistics']['candidates'] != POOLS[case] or saved['pair_count'] != POOLS[case]*2:
            raise ValueError('Saved case count inconsistent')
    a, b = pair['control'], pair['experimental']
    for key in ('frozen_pool_fingerprint', 'original_scores_fingerprint', 'pair_order_fingerprint', 'checkpoint_sha256', 'formatted_queries'):
        if a[key] != b[key]:
            raise ValueError('Task 7.8 score conditions are not the same frozen inputs: ' + key)
    by_a = {r['chunk_id']: r for r in a['candidate_audit']}
    by_b = {r['chunk_id']: r for r in b['candidate_audit']}
    if set(by_a) != set(by_b):
        raise ValueError('Frozen IDs differ between conditions')
    for cid in by_a:
        for key in ('source_id', 'source', 'document_id', 'title', 'section', 'url', 'evidence_text',
                    'evidence_text_sha256', 'fusion_rank', 'rrf_score', 'all_channel_origins'):
            if by_a[cid][key] != by_b[cid][key]:
                raise ValueError('Frozen non-neural candidate metadata differs: ' + key)
    return pair


def require_all_controls(controls):
    expected = {(c, m) for c in (1,2,3) for m in ('control', 'experimental')}
    if {(r['case'], r['score_condition']) for r in controls if r['passed']} != expected or len(controls) != 6:
        raise ValueError('All six exact saved-selection controls must pass before counterfactual analysis')


def totals(cases):
    result = {}
    for model in ('control', 'experimental'):
        group = [c['summary'] for c in cases if c['score_condition'] == model]
        result[model] = {key: sum(s[key] for s in group) for key in (
            'eligible_candidates', 'eligible_role_pairs', 'selected_candidates', 'eligible_not_selected_candidates',
            'budget_only_at_saved_prefix', 'confirmed_final_budget_only_recoveries',
            'counterfactual_selected_candidates', 'additional_candidates', 'counterfactual_still_rejected_candidates')}
        for key in ('rejection_reason_counts', 'counterfactual_rejection_reason_counts'):
            counts = Counter()
            for s in group:
                counts.update(s[key])
            result[model][key] = dict(counts)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New Task 7.9 directory only')
    args = parser.parse_args(argv)
    if any(os.environ.get(k) != '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        raise ValueError('Launch using unchanged run_memory_bounded.py')
    output = args.output.resolve()
    if ROOT / 'outputs/task7.9' not in output.parents:
        raise ValueError('Output must be a new directory under outputs/task7.9')
    output.mkdir(parents=True, exist_ok=False)
    monitor = Monitor(output)
    report = {'task': '7.9', 'status': 'started', 'controls': [], 'cases': [],
        'model_loads': 0, 'new_scores': 0, 'retrieval_calls': 0, 'labels_read': False,
        'counterfactual': 'remove final count only; history-priority quota 1 unchanged',
        'memory_guards': {'outer_stop_gb': 14, 'startup_reserve_gb': 1.5, 'synchronous_stop_gb': 13.5}}
    try:
        monitor.begin('authenticate-saved-artifacts-and-protected-selector-source')
        before = protect(monitor)
        write(output / 'protected-files-before.json', before)
        from rag.experimental_medical.task79_selection_analysis import original_functions, replay, analyze, join_review
        choose_final, tokenize = original_functions(ROOT)
        monitor.begin('focused-task79-synthetic-tests')
        suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_task79_selection_analysis')
        with (output / 'tests.log').open('w', encoding='utf8') as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        report['tests'] = {'run': tests.testsRun, 'passed': tests.wasSuccessful(),
                           'failures': len(tests.failures), 'errors': len(tests.errors)}
        if not tests.wasSuccessful():
            raise ValueError('Focused tests failed; no analysis')
        for case in (1,2,3):
            monitor.begin(f'validate:{case}:exact-controls-both-saved-score-sets')
            pair = load_case(case, monitor)
            folder = output / f'validate-{case}'
            folder.mkdir()
            for model, saved in pair.items():
                baseline = replay(saved, choose_final, tokenize, sample=monitor.sample)
                item = {'case': case, 'score_condition': model, 'passed': True,
                        'selected_ids': baseline['selected_ids'], 'selected_roles': baseline['selected_roles'],
                        'all_saved_selector_events_exact': True, 'input_fingerprint': baseline['input_fingerprint']}
                report['controls'].append(item)
                write(folder / f'{model}-control-reproduction.json', item)
            del pair, saved, baseline
            gc.collect()
        require_all_controls(report['controls'])
        report['all_controls_passed'] = True
        write(output / 'control-reproduction.json', report['controls'])
        monitor.begin('verify-protected-files-before-single-counterfactual')
        verify(before, monitor)
        review_rows = load(REVIEW / 'reviewed-pairs.json', monitor)
        abc_all = []
        for case in (1,2,3):
            monitor.begin(f'validate:{case}:one-final-budget-only-counterfactual')
            pair = load_case(case, monitor)
            folder = output / f'validate-{case}'
            for model, saved in pair.items():
                baseline = replay(saved, choose_final, tokenize, sample=monitor.sample)
                unlimited = replay(saved, choose_final, tokenize, remove_final_budget=True, sample=monitor.sample)
                analysis = analyze(saved, baseline, unlimited)
                write(folder / f'{model}-selection-analysis.json', analysis)
                important = [r for r in analysis['candidate_analysis'] if r['title'] in
                    ('Panic Disorder', 'Flu', 'Fainting', 'Older Adult Mental Health') or 'asthma' in r['title'].casefold()]
                write(folder / f'{model}-important-candidates.json', important)
                if model == 'experimental':
                    joined = join_review(analysis, review_rows, case)
                    write(folder / 'task781-review-cross-reference.json', joined)
                    abc_all.extend(joined['pairs'])
                report['cases'].append({'case': case, 'score_condition': model, 'summary': analysis['summary']})
            write(output / 'comparison.json', report)
            del pair, saved, baseline, unlimited, analysis, important
            gc.collect()
        if Counter(r['class'] for r in abc_all) != Counter({'A':22, 'B':31, 'C':7}):
            raise ValueError('Saved review coverage differs from all 60 new positives')
        report['review_cross_reference'] = {}
        for label in ('A','B','C'):
            rows = [r for r in abc_all if r['class'] == label]
            report['review_cross_reference'][label] = {'total': len(rows),
                'selected_for_role': sum(r['selected_for_role'] for r in rows),
                'eligible_not_selected_for_role': sum(not r['selected_for_role'] for r in rows),
                'counterfactual_selected_for_role': sum(r['counterfactual_selected_for_role'] for r in rows),
                'newly_selected_for_role': sum(r['counterfactual_selected_for_role'] and not r['selected_for_role'] for r in rows),
                'rejection_reason_counts': dict(Counter(r['rejection_rule'] for r in rows if not r['selected_for_role'])),
                'counterfactual_rejection_reason_counts': dict(Counter(r['counterfactual_rejection_rule'] for r in rows if not r['counterfactual_selected_for_role']))}
        report['totals'] = totals(report['cases'])
        monitor.begin('verify-all-protected-files-after-analysis')
        after = verify(before, monitor)
        write(output / 'protected-files-after.json', after)
        report.update(status='completed', protected_file_count=len(after), protected_files_unchanged=True, memory=monitor.totals())
        write(output / 'comparison.json', report)
        print(json.dumps({'status': report['status'], 'totals': report['totals'], 'review': report['review_cross_reference'],
                          'memory': report['memory'], 'tests': report['tests'], 'protected_files': len(after)}, indent=2), flush=True)
        return 0
    except Exception as exc:
        report.update(status='stopped', stop={'phase': monitor.phase, 'operation': monitor.phase,
            'error_type': type(exc).__name__, 'reason': str(exc)}, memory=monitor.totals())
        write(output / 'comparison.json', report)
        write(output / 'analysis-stopped.json', report['stop'])
        print('STOP: ' + str(exc), flush=True)
        return 75 if isinstance(exc, MemoryStop) else 1


if __name__ == '__main__':
    raise SystemExit(main())
