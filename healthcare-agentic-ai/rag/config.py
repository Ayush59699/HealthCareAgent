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


# Generation credentials are deliberately NOT dataclass fields: reports use asdict.
# Reuse the dedicated deployment convention, never the coding agent's credentials.
def load_generation_env(path: Path | None = None) -> None:
    """CLI-only .env loading; existing process environment always takes precedence."""
    path = path or PROJECT_ROOT.parent / '.env'
    allowed = {'GPT_SOL_ENDPOINT', 'GPT_SOL_API_KEY', 'OPENAI_MODEL', 'LLM_PROVIDER',
               'OPENAI_TIMEOUT', 'OPENAI_TEMPERATURE', 'OPENAI_MAX_OUTPUT_TOKENS',
               'OPENAI_MAX_INPUT_BYTES'}
    if path.is_file():
        for line in path.read_text(encoding='utf8').splitlines():
            if line.strip().startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            if key in allowed and key not in os.environ:
                os.environ[key] = value.strip().strip("\"'")


@dataclass(frozen=True)
class OpenAIConfig:
    provider: str = field(default_factory=lambda: os.getenv('LLM_PROVIDER', 'openai'))
    base_url: str = field(default_factory=lambda: os.getenv('GPT_SOL_ENDPOINT', 'https://meeting-summarizer-ai.services.ai.azure.com/openai/v1'))
    model: str = field(default_factory=lambda: os.getenv('OPENAI_MODEL', 'gpt-5.6-sol'))
    temperature: float = field(default_factory=lambda: float(os.getenv('OPENAI_TEMPERATURE', '0')))
    timeout: float = field(default_factory=lambda: float(os.getenv('OPENAI_TIMEOUT', '300')))
    max_retries: int = 1  # Structured-output repair only; SDK/API retries disabled.
    max_input_bytes: int = field(default_factory=lambda: int(os.getenv('OPENAI_MAX_INPUT_BYTES', '100000')))
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv('OPENAI_MAX_OUTPUT_TOKENS', '8192')))

    def __post_init__(self):
        import math
        from urllib.parse import urlsplit
        url = urlsplit(self.base_url)
        if (url.scheme != 'https' or not url.hostname or url.username or url.password
                or url.query or url.fragment or url.path.rstrip('/') not in {'/openai/v1', '/v1'}
                or url.hostname in {'localhost', '127.0.0.1', '::1'}):
            raise ValueError('Generation requires a cloud HTTPS v1 endpoint without URL credentials')
        if url.port is not None and not 1 <= url.port <= 65535:
            raise ValueError('Invalid API port')
        if self.provider != 'openai' or self.model != 'gpt-5.6-sol':
            raise ValueError('Only the gpt-5.6-sol OpenAI-compatible deployment is supported')
        if self.temperature != 0:
            raise ValueError('This experiment requires temperature 0')
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('API timeout must be positive and finite')
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 2:
            raise ValueError('At most two structured-output retries are permitted')
        if any(type(v) is not int or v < 1 for v in (self.max_input_bytes, self.max_output_tokens)):
            raise ValueError('Input and output limits must be positive integers')
