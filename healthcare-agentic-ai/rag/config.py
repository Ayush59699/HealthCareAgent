"""Path configuration; no model loading or network activity at import time."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = {split: f"release_{split}_patients.zip" for split in ("train", "validate", "test")}
REQUIRED_FILES = ("release_evidences.json", "release_conditions.json", *SPLIT_FILES.values())


def resolve_data_dir(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    if os.environ.get("DDXPLUS_DATA_DIR"):
        return Path(os.environ["DDXPLUS_DATA_DIR"]).expanduser().resolve()
    local = PROJECT_ROOT / "data" / "ddxplus"
    if (local / "release_evidences.json").is_file():
        return local
    return PROJECT_ROOT.parent / "data" / "ddxplus"


def inspect_dataset(path: str | Path | None = None) -> dict:
    directory = resolve_data_dir(path)
    return {"data_dir": str(directory), "files": {
        name: {"exists": (directory / name).is_file(),
               "bytes": (directory / name).stat().st_size if (directory / name).is_file() else None}
        for name in REQUIRED_FILES
    }}


# Phase 2 settings; read at construction time so environment overrides are usable.
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PatientRAGConfig:
    model_name: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "cpu"))
    batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
    storage_path: Path = field(default_factory=lambda: Path(os.getenv("QDRANT_PATH", str(PROJECT_ROOT / "data/qdrant/patient_cases"))).expanduser().resolve())
    collection_name: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "ddxplus_patient_cases"))
    allow_download: bool = False

    def __post_init__(self):
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not self.model_name.strip() or not self.device.strip() or not self.collection_name.strip():
            raise ValueError("Model, device and collection name must be nonempty")
