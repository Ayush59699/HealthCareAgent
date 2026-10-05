"""Offline verification of a complete CHECK4 run; no cloud or relevance adjudication."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.who_catalog_selector import WHOCatalogSelector, WHOSelection, PROMPT, VERSION, digest
from rag.who_selector_relevance import ContentRelevance, REVIEW_PROMPT, validate_content_review
from rag.who_lookup import DEFAULT_DATA_DIR
from rag.manual_patient import manual_patient_id
from scripts.evaluate_who_selector_check4 import CASES, case_patient, write


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory, data_dir=DEFAULT_DATA_DIR):
    directory = Path(directory)
    def read(path):
        return json.loads(path.read_text(encoding='utf8'))
    selector = WHOCatalogSelector(data_dir)
    plan = read(directory / 'plan.json')
    summary = read(directory / 'summary.json')
    catalog = read(directory / 'catalog.json')
    require(plan['live'] and plan['experiment'] == VERSION, 'Not a live current-version run')
    require(plan['catalog_prompt'] == PROMPT and plan['content_prompt'] == REVIEW_PROMPT, 'Prompt mismatch')
    require(catalog == selector.catalog_payload() and plan['catalog_sha256'] == selector.catalog_sha256,
            'Catalog/source metadata changed')
    require(plan['catalog_documents'] == len(catalog['catalog']), 'Catalog count mismatch')
    require(len(plan['cases']) == len(CASES) == len(summary['cases']), 'Case count mismatch')
    require(digest(plan['cases']) == plan['cases_sha256'], 'Case plan hash mismatch')
    telemetry, source_events, selected = [], 0, 0
    summary_rows = []
    for case, record in zip(CASES, plan['cases']):
        patient = case_patient(case)
        require(record['case_id'] == case[0] and record['patient_facts'] == patient.to_inference_dict()
                and record['patient_id'] == manual_patient_id(patient), 'Fixed case changed')
        folder = directory / case[0]
        require(read(folder / 'input.json') == record, 'Input differs from plan')
        report = read(folder / 'report.json')
        require(all(report[k] == v for k, v in record.items()), 'Report input mismatch')
        require(report['status'] in ('no_selection', 'selected_and_verified'), 'Run did not complete')
        require(report['max_selected'] == 3 and report['pipeline_run'] is False, 'Experiment boundary mismatch')
        stage = report['catalog_stage']
        require(stage['failure'] is None, 'Catalog failure')
        proposal = selector.validate_selection(WHOSelection.model_validate(stage['parsed']))
        telemetry.extend(stage['telemetry'])
        require(len(report['candidates']) == len(proposal.selected_ids), 'Missing candidate audit')
        expected_ids, expected_docs, expected_files = [], [], set()
        for document, row in zip(selector.iter_selected(proposal), report['candidates']):
            require(row['document'] == document.audit(), 'Source provenance changed')
            filename = document.document_id.replace(':', '-') + '.txt'
            expected_files.add(filename)
            raw = (folder / filename).read_bytes()
            require(raw == document.text.encode('utf8') and hashlib.sha256(raw).hexdigest() == document.document_sha256,
                    'Saved full content differs from source')
            review = row['content_review']
            require(review['failure'] is None, 'Content review failure')
            verdict = validate_content_review(ContentRelevance.model_validate(review['parsed']), patient, document)
            telemetry.extend(review['telemetry'])
            if verdict.verdict == 'relevant':
                expected_ids.append(document.document_id)
                expected_docs.append(document.audit())
            source_events += 1
        require({p.name for p in folder.glob('*.txt')} == expected_files, 'Missing/extra loaded files')
        require(report['selected_ids'] == expected_ids and report['selected_documents'] == expected_docs
                and report['selection_count'] == len(expected_ids), 'Final selection mismatch')
        require(report['status'] == ('selected_and_verified' if expected_ids else 'no_selection'), 'Status mismatch')
        summary_rows.append({'case_id': case[0], 'status': report['status'],
                             'selection_count': len(expected_ids), 'selected_ids': expected_ids})
        selected += len(expected_ids)
    require(summary['cases'] == summary_rows and summary['live'] and summary['pipeline_run'] is False,
            'Summary mismatch')
    require(all(t['accepted'] and t['api_success'] and not t['client_context_truncated'] for t in telemetry),
            'Request failure/truncation')
    return {'verified': True, 'cases': len(CASES), 'catalog_documents': len(catalog['catalog']),
            'source_events': source_events, 'final_selected_documents': selected,
            'abstaining_cases': sum(c['selection_count'] == 0 for c in summary_rows),
            'cloud_requests': len(telemetry), 'request_seconds': sum(t['request_seconds'] for t in telemetry),
            'input_tokens': sum(t.get('input_tokens', 0) for t in telemetry),
            'output_tokens': sum(t.get('output_tokens', 0) for t in telemetry),
            'max_request_bytes': max(t['request_bytes'] for t in telemetry),
            'boundary': 'Provenance/contract verification only, not independent semantic relevance validation.'}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('run_dir', type=Path)
    cli.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    cli.add_argument('--output', type=Path)
    args = cli.parse_args(argv)
    result = verify(args.run_dir, args.data_dir)
    if args.output:
        write(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
