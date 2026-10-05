"""Bounded live comparison on label-free validate:2; uses existing indexes only."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import load_generation_env, OpenAIConfig
from rag.patient_parser import DDXPlusParser
from rag.patient_rag import PatientCaseRAG
from rag.medical_retriever import MedicalKnowledgeRetriever
from rag.llm.provider import OpenAIProvider
from orchestration.phase6 import Phase6Orchestrator
from scripts.run_phase6 import case_summary

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--enhanced', action='store_true')
    args = cli.parse_args()
    out = Path(__file__).resolve().parents[1] / 'outputs/enhance'
    out.mkdir(parents=True, exist_ok=True)
    name = 'after' if args.enhanced else 'before'
    load_generation_env()
    record = list(DDXPlusParser().iter_patients('validate', limit=2, include_labels=False))[1]
    with ExitStack() as stack:
        patients = stack.enter_context(PatientCaseRAG())
        medical = stack.enter_context(MedicalKnowledgeRetriever())
        provider = stack.enter_context(OpenAIProvider(OpenAIConfig()))
        options = {'enhanced_medical': True} if args.enhanced else {}
        workflow = Phase6Orchestrator(provider, patients, medical, **options)
        state = workflow.run(record.patient, record.patient_id)
        (out / (name + '-state.json')).write_text(state.model_dump_json(indent=2), encoding='utf8')
        report = case_summary(state)
        (out / (name + '-summary.json')).write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps(report, indent=2))
        return int(state.status == 'failed')

if __name__ == '__main__':
    raise SystemExit(main())
