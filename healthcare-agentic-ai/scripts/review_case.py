"""Review a saved AI/Safety result in a local terminal. No cloud or retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from application.decision import build_final_decision, FinalDecision
from application.human_review import offer_human_review
from orchestration.phase6.state import Phase6WorkflowState

MAX_STATE_BYTES = 32 * 1024 * 1024


def bounded_bytes(path):
    with path.open('rb') as stream:
        value = stream.read(MAX_STATE_BYTES + 1)
    if len(value) > MAX_STATE_BYTES:
        raise ValueError('Saved review input exceeds size bound')
    return value


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('state', type=Path, help='sample_case_001.json or case_state.json')
    cli.add_argument('--review-dir', type=Path, help='New sessions created here; existing files never overwritten')
    args = cli.parse_args(argv)
    try:
        raw = bounded_bytes(args.state)
        state = Phase6WorkflowState.model_validate_json(raw)
        final = build_final_decision(state)
        decision_path = (args.state.with_name('final_decision_' + args.state.stem.removeprefix('sample_case_') + '.json')
            if args.state.name.startswith('sample_case_') else
            args.state.with_name(args.state.name.removesuffix('_state.json') + '_decision.json'))
        protected = {args.state: hashlib.sha256(raw).hexdigest()}
        if decision_path.is_file():
            saved = bounded_bytes(decision_path)
            if FinalDecision.model_validate_json(saved) != final:
                raise ValueError('Saved AI final decision mismatch')
            protected[decision_path] = hashlib.sha256(saved).hexdigest()
        def unchanged():
            return all(hashlib.sha256(bounded_bytes(path)).hexdigest() == expected
                       for path, expected in protected.items())
        record = offer_human_review(state, args.review_dir or args.state.parent / 'human_reviews',
                                    integrity_check=unchanged)
        if record is None:
            print('AI result does not require human review. No human approval recorded.')
            return 0
        print(json.dumps(record.model_dump(mode='json'), indent=2))
        print('Local review event recorded separately. Original AI/Safety result and release restrictions are unchanged.')
        return 2 if record.event in ('NO_RESPONSE', 'INTERRUPTED', 'NON_INTERACTIVE', 'INVALIDATED') else 0
    except (OSError, ValueError, TypeError):
        print('Review could not be safely loaded or persisted. No approval may be inferred; case remains restricted.', file=sys.stderr)
        return 1
    except (EOFError, KeyboardInterrupt):
        print('Review interrupted; no approval may be inferred. Case remains restricted.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
