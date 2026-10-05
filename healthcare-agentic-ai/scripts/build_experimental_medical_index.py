"""Build a NEW task-7.4 local index from the existing hash-pinned medical corpus."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.medical_ingestion.common import DATA_ROOT


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=DATA_ROOT / 'corpus.json')
    parser.add_argument('--output', type=Path, default=DATA_ROOT / 'experimental-task74')
    parser.add_argument('--threads', type=int, default=2)
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.output.exists():
        raise ValueError('Experimental output already exists; choose a new directory')
    if not args.manifest.is_file() or args.threads < 1:
        raise ValueError('Existing corpus manifest and positive thread count required')
    import torch
    torch.set_num_threads(args.threads)
    from rag.medical_embeddings import MedicalEmbeddingModel
    from rag.experimental_medical.index import build_index
    result = build_index(args.manifest, args.output, MedicalEmbeddingModel(allow_download=False, batch_size=2))
    print(json.dumps(result, indent=2))
    print('Separate index: ' + str(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
