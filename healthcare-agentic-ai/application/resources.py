"""Fail before creating missing indexes; bounded CPU-only research runtime."""
import os


def prepare_runtime(patient_config):
    path = patient_config.storage_path
    if not path.is_dir() or not (path / 'meta.json').is_file():
        raise ValueError('Prebuilt patient index required; run scripts/build_patient_index.py explicitly')
    if patient_config.device != 'cpu':
        raise ValueError('The integrated laptop runtime requires CPU embeddings')
    import psutil
    if psutil.virtual_memory().available < 2_000_000_000:
        raise RuntimeError('Less than 2 GB memory headroom; close applications before loading models')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
    import torch
    torch.set_num_threads(min(2, os.cpu_count() or 1))
