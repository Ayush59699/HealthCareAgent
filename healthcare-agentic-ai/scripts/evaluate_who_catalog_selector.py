"""Selector-only comparison on a manual vignette; never run downstream agents."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Literal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.agents.models import StrictModel
from rag.models import Evidence, PatientRepresentation
from rag.manual_patient import manual_patient_id, read_manual_patient, MAX_CASE_BYTES
from rag.who_catalog_selector import WHOCatalogSelector, PROMPT, VERSION, digest
from rag.who_lookup import WHOLookup, DEFAULT_DATA_DIR, bounded_read


class SavedFeatures(StrictModel):
    age: int | None
    sex: Literal['M', 'F'] | None
    symptoms: list[str]
    antecedents: list[str]
    initial_evidence: list[str]


def load_saved_case(path):
    """Recover the prior CHECK3 input, not its predictions or selected documents.

    The task file was replaced by new instructions. Its original patient payload
    survives in the prior run's input.json. Verify its content-bound identity.
    """
    content = bounded_read(Path(path), MAX_CASE_BYTES)
    saved = json.loads(content)
    values = SavedFeatures.model_validate(saved['label_free_input'])
    def entries(items, antecedent):
        result = []
        for i, item in enumerate(items):
            if ' = ' not in item:
                raise ValueError('Unsupported saved manual evidence format')
            question, value = item.split(' = ', 1)
            result.append(Evidence(f'manual:restored:{i}', question, value, antecedent))
        return tuple(result)
    patient = PatientRepresentation(values.age, values.sex, entries(values.symptoms, False),
        entries(values.antecedents, True), entries(values.initial_evidence, False))
    if patient.to_inference_dict() != values.model_dump() or manual_patient_id(patient) != saved['patient_id']:
        raise ValueError('Saved manual case identity mismatch')
    return patient, {'origin': 'Exact label-free CHECK3 payload restored from prior input audit; no predictions read.',
        'input_audit_file': str(path), 'input_audit_sha256': hashlib.sha256(content.encode('utf8')).hexdigest(),
        'patient_id': saved['patient_id'], 'original_source_sha256': saved['source_sha256']}


def lexical_baseline(patient, data_dir):
    # Reuse the OLD query constructor exactly; do not implement a new lexical search.
    from rag.combined_medical_evidence import who_query
    features = patient.to_inference_dict()
    state = SimpleNamespace(presenting_evidence=features['initial_evidence'],
                            symptoms=features['symptoms'], antecedents=features['antecedents'])
    query, omitted = who_query(state)
    result = WHOLookup(data_dir).search(query, max_results=3)  # metadata only
    return {'query': query, 'omitted_fact_refs': omitted, 'status': result.status,
            'errors': result.errors, 'max_results': 3,
            'documents': [{'source': asdict(d.source), 'match_type': d.match_type,
                           'matched_terms': d.matched_terms} for d in result.documents]}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    temporary.replace(path)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group(required=True)
    source.add_argument('--case-file', type=Path, help='Sectioned manual vignette, not task instructions')
    source.add_argument('--saved-case', type=Path, help='Prior manual-run input.json, never a diagnosis file')
    cli.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    cli.add_argument('--live', action='store_true', help='One selector cloud request; no downstream agents')
    cli.add_argument('--output-dir', type=Path, default=DEFAULT_DATA_DIR.parent / 'outputs/check3-selector' /
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    args = cli.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'experiment': VERSION, 'live': args.live, 'pipeline_run': False, 'selected_documents': [],
              'limitation': 'Catalog selection plausibility only; no diagnosis, entailment or clinical usefulness validation.'}
    try:
        patient, origin = load_saved_case(args.saved_case) if args.saved_case else read_manual_patient(args.case_file)
        selector = WHOCatalogSelector(args.data_dir)
        payload = {'patient_case': patient.to_inference_dict(), **selector.catalog_payload()}
        write(args.output_dir / 'selector_request.json', {'instructions': PROMPT, 'payload': payload})
        report.update(origin=origin, catalog_sha256=selector.catalog_sha256,
                      catalog_documents=len(payload['catalog']), selector_payload_sha256=digest(payload))
        # Baseline results are never part of the selector's prompt/patient payload.
        report['old_lexical'] = lexical_baseline(patient, args.data_dir)
        if not args.live:
            report['status'] = 'prepared_not_run'
        else:
            from rag.config import OpenAIConfig, load_generation_env
            from rag.llm.provider import OpenAIProvider
            load_generation_env()
            config = replace(OpenAIConfig(), max_retries=0, max_output_tokens=512)
            report['provider_limits'] = {'max_retries': 0, 'max_output_tokens': 512,
                                         'max_input_bytes': config.max_input_bytes}
            with OpenAIProvider(config) as provider:
                generation = selector.select(provider, patient)
            report['requests'] = generation.attempts
            report['telemetry'] = generation.telemetry
            if generation.failure is not None:
                report.update(status='selector_failure', failure=generation.failure.model_dump())
                write(args.output_dir / 'report.json', report)
                return 1
            report['selected_ids'] = generation.parsed.selected_ids
            # Only after EVERY ID is validated do we open selected files. Retain
            # metadata/checksums only; full texts are yielded one at a time, uncached.
            for document in selector.iter_selected(generation.parsed):
                report['selected_documents'].append(document.audit())
            chosen = {(d['document_type'], d['url']) for d in report['selected_documents']}
            old_generic = [d['source'] for d in report['old_lexical']['documents']
                           if set(d['matched_terms']) & {'known', 'chronic'}]
            retained = [s['title'] for s in old_generic if (s['document_type'], s['url']) in chosen]
            report['check3_comparison'] = {'old_known_chronic_titles': [s['title'] for s in old_generic],
                'retained_known_chronic_titles': retained,
                'known_chronic_false_matches_absent_in_this_run': bool(old_generic) and not retained,
                'note': 'Case-specific membership comparison, not proof of general negation handling or clinical relevance.'}
            report['status'] = 'selected_and_loaded' if chosen else 'no_selection'
        write(args.output_dir / 'report.json', report)
        print('Selector status:', report['status'], '| Catalog entries:', report['catalog_documents'])
        print('Old lexical WHO:', [d['source']['title'] for d in report['old_lexical']['documents']])
        print('LLM-selected WHO:', [d['title'] for d in report['selected_documents']])
        print('Diagnostic/Critic/Safety pipeline: NOT RUN')
        return 0
    except Exception as exc:
        report.update(status='experiment_failure', error_type=type(exc).__name__,
                      message='Input, catalog, source or provider contract failed; sensitive details omitted.')
        write(args.output_dir / 'report.json', report)
        print(report['message'], file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
