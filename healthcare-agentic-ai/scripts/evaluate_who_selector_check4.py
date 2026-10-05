"""Fixed eight-case, label-free CHECK4 selector evaluation; opt-in sequential cloud.

No benchmark reader, lexical search, AMG or downstream agents. No post-result
case selection, retries, refill or content truncation. Prepared runs make no calls.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.models import Evidence, PatientRepresentation
from rag.manual_patient import manual_patient_id
from rag.who_catalog_selector import WHOCatalogSelector, PROMPT, VERSION, digest
from rag.who_selector_relevance import select_verified, REVIEW_PROMPT
from rag.who_lookup import DEFAULT_DATA_DIR

# Fixed before retrieval; hand-authored observations, not disease/benchmark labels.
CASES = (
    ('01-nonspecific', ('nonspecific symptoms',), 32, 'F',
     ('Mild fatigue and intermittent mild headache for two days.',),
     ('No fever. No shortness of breath. No known chronic illness.',)),
    ('02-multiple', ('multiple symptoms',), 24, 'F',
     ('Fatigue and dizziness for approximately three weeks.',
      'Occasional headaches and reduced appetite.', 'Dizziness is worse on standing.'),
     ('No shortness of breath. No known chronic illness.',
      'Pregnancy status and menstrual history are not supplied. No test results are available.')),
    ('03-negations', ('explicit negations', 'mixed polarity',), 40, 'M',
     ('A mild cough for two days, but no fever and no SOB.',),
     ('No chronic illness. No weight loss. No night sweats. No known tuberculosis exposure.',)),
    ('04-sparse', ('sparse information',), None, None,
     ('Feels unwell.',), ('Duration, other symptoms, history and tests are not supplied.',)),
    ('05-no-clear-topic', ('no clearly relevant WHO topic',), 29, 'F',
     ('Brief itching confined to the left earlobe immediately after removing an earring.',),
     ('No hearing change, ear pain, discharge, rash or other symptoms reported.',)),
    ('06-watery-stools', ('multiple positive presenting features',), 35, 'M',
     ('Six loose watery stools since yesterday.', 'Thirst and reduced urination today.'),
     ('No blood in the stools. No cough. Travel and exposure history not supplied.',)),
    ('07-animal-bite', ('specific presenting event',), 28, 'F',
     ('Bitten on the hand by a stray dog this morning; the skin is broken.',),
     ('Dog vaccination status is unknown. No fever. Patient vaccination history not supplied.',)),
    ('08-hot-water', ('specific presenting event',), 46, 'M',
     ('Hot water spilled on the forearm one hour ago.',
      'The affected skin is painful and red with two blisters.'),
     ('No smoke exposure. No breathing difficulty. No other affected body areas reported.',)),
)


def case_patient(case):
    _, _, age, sex, positive, context = case
    def evidence(values, antecedent):
        return tuple(Evidence(f'manual:check4:{i}', 'History' if antecedent else 'Presenting observation',
                              text, antecedent) for i, text in enumerate(values))
    return PatientRepresentation(age, sex, evidence(positive, False), evidence(context, True))


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temp.replace(path)


def evaluate(output, selector, provider=None):
    output.mkdir(parents=True, exist_ok=False)
    records = [{'case_id': c[0], 'categories': c[1], 'patient_id': manual_patient_id(case_patient(c)),
                'patient_facts': case_patient(c).to_inference_dict()} for c in CASES]
    plan = {'experiment': VERSION, 'case_origin': 'Fixed hand-authored label-free observations; no diagnoses or benchmark data.',
            'live': provider is not None, 'cases': records, 'cases_sha256': digest(records),
            'catalog_sha256': selector.catalog_sha256, 'catalog_documents': len(selector.catalog_payload()['catalog']),
            'max_selected': 3, 'sequential': True, 'production_integration': False,
            'catalog_prompt': PROMPT, 'content_prompt': REVIEW_PROMPT,
            'policy': 'No lexical selection; one catalog call then at most three content calls. No retries or refill.',
            'limitations': 'Semantic model judgment plus exact-quote checks is not independent clinical relevance validation.'}
    write(output / 'plan.json', plan)  # Freeze ALL cases/prompts before first call.
    write(output / 'catalog.json', selector.catalog_payload())
    summary = {'live': provider is not None, 'pipeline_run': False, 'cases': []}
    for case, record in zip(CASES, records):
        directory = output / case[0]
        directory.mkdir()
        report = dict(record)
        write(directory / 'input.json', record)
        if provider is None:
            report['status'] = 'prepared_not_run'
        else:
            def save_document(document):
                # Same UTF-8 content as source; colon-free name for Windows.
                path = directory / (document.document_id.replace(':', '-') + '.txt')
                path.write_bytes(document.text.encode('utf8'))
                if hashlib.sha256(path.read_bytes()).hexdigest() != document.document_sha256:
                    raise ValueError('Persisted source checksum mismatch')
            try:
                report.update(select_verified(selector, provider, case_patient(case), save_document))
            except Exception as exc:
                report.update(status='experiment_failure', error_type=type(exc).__name__,
                              selected_ids=[], selected_documents=[], selection_count=0)
            print(case[0], report['status'], [d['title'] for d in report.get('selected_documents', [])], flush=True)
        write(directory / 'report.json', report)
        summary['cases'].append({'case_id': case[0], 'status': report['status'],
                                 'selection_count': report.get('selection_count'),
                                 'selected_ids': report.get('selected_ids', [])})
        write(output / 'summary.json', summary)
    return summary


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--live', action='store_true')
    cli.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    cli.add_argument('--output-dir', type=Path, default=DEFAULT_DATA_DIR.parent / 'outputs/check4-selector' /
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args(argv)
    selector = WHOCatalogSelector(args.data_dir)
    if not args.live:
        evaluate(args.output_dir, selector)
        return 0
    from rag.config import OpenAIConfig, load_generation_env
    from rag.llm.provider import OpenAIProvider
    load_generation_env()
    # Existing request byte cap prevents oversized content calls, never truncates.
    config = replace(OpenAIConfig(), max_retries=0, max_output_tokens=1800)
    with OpenAIProvider(config) as provider:
        summary = evaluate(args.output_dir, selector, provider)
    return int(any(c['status'] not in ('selected_and_verified', 'no_selection') for c in summary['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
