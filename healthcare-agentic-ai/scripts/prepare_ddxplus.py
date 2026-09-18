"""Inspect or stream DDXPlus into label-free JSONL; labels require a separate file."""
import argparse
from contextlib import ExitStack
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag import DDXPlusParser, DDXPlusError
from rag.config import inspect_dataset


def main(argv=None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-dir", type=Path)
    cli.add_argument("--inspect", action="store_true")
    cli.add_argument("--split", choices=("train", "validate", "test"), default="train")
    cli.add_argument("--limit", type=int, help="Default: 5 preview rows; all rows when --output is specified")
    cli.add_argument("--output", type=Path, help="Label-free features JSONL; refuses to overwrite")
    cli.add_argument("--labels-output", type=Path, help="Separate evaluation-only labels JSONL; never index this file")
    args = cli.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.limit is not None and args.limit < 0:
        cli.error("--limit must be nonnegative")
    if args.labels_output and not args.output:
        cli.error("--labels-output requires --output")
    outputs = [p.resolve() for p in (args.output, args.labels_output) if p is not None]
    if len(outputs) != len(set(outputs)):
        cli.error("Features and evaluation labels must have different output paths")
    created = []
    try:
        if args.inspect:
            inventory = inspect_dataset(args.data_dir)
            print(json.dumps(inventory, indent=2))
            return 0 if all(v["exists"] for v in inventory["files"].values()) else 1
        parser = DDXPlusParser(args.data_dir)
        limit = args.limit if args.limit is not None else (None if args.output else 5)
        if args.labels_output:
            logging.warning("Evaluation labels are exported separately; do not pass them to retrieval or generation")
        count = 0
        with ExitStack() as stack:
            handles = []
            for path in outputs:
                path.parent.mkdir(parents=True, exist_ok=True)
                handles.append(stack.enter_context(path.open("x", encoding="utf-8", newline="\n")))
                created.append(path)
            for record in parser.iter_patients(args.split, limit=limit, include_labels=bool(args.labels_output)):
                features = {"patient_id": record.patient_id, "split": record.split,
                            "source": record.source, "patient": record.to_inference_dict()}
                line = json.dumps(features, ensure_ascii=True)
                if handles:
                    handles[0].write(line + "\n")
                else:
                    print(line)
                if args.labels_output:
                    handles[1].write(json.dumps({"patient_id": record.patient_id,
                        "evaluation_only": True, **record.labels.to_dict()}, ensure_ascii=True) + "\n")
                count += 1
                if count % 10000 == 0:
                    logging.info("Processed %d rows", count)
        logging.info("Successfully processed %d %s rows", count, args.split)
        return 0
    except (DDXPlusError, OSError) as exc:
        logging.error("Preparation failed: %s", exc)
        # Remove only files created by this invocation, never pre-existing output.
        for path in created:
            path.unlink(missing_ok=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
