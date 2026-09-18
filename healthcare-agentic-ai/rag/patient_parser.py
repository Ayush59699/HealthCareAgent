"""Deterministic English DDXPlus decoding with bounded-memory ZIP streaming."""
import ast
import csv
import io
import json
import logging
import math
import re
import zipfile
from itertools import islice
from pathlib import Path
from typing import Iterator, Mapping, Any

from .config import SPLIT_FILES, resolve_data_dir
from .models import Evidence, PatientRepresentation, PatientRecord, EvaluationLabels, DifferentialDiagnosis

logger = logging.getLogger(__name__)
CODE_PATTERN = re.compile(r"\b[EV]_\d+\b")


class DDXPlusError(ValueError):
    """Invalid dataset schema, evidence, or patient row."""


def _list(value: Any, field: str) -> list:
    if value is None or value == "":
        raise DDXPlusError(f"{field} is missing; use [] for an explicitly empty list")
    if isinstance(value, str):
        if len(value) > 100_000:
            raise DDXPlusError(f"{field} exceeds the supported field size")
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError, RecursionError) as exc:
            raise DDXPlusError(f"{field} must be a literal list") from exc
    if not isinstance(value, list):
        raise DDXPlusError(f"{field} must be a list")
    return value


def _english(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or CODE_PATTERN.search(value):
        raise DDXPlusError(f"Missing or coded English text in {field}")
    return " ".join(value.split())


class DDXPlusParser:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = resolve_data_dir(data_dir)
        self.evidences = self._load_mapping("release_evidences.json")
        self.conditions = self._load_mapping("release_conditions.json")
        for code, definition in self.evidences.items():
            _english(definition.get("question_en"), f"question for {code}")
            if definition.get("data_type") not in {"B", "C", "M"}:
                raise DDXPlusError(f"Unsupported evidence type for {code}")
            if not isinstance(definition.get("is_antecedent"), bool):
                raise DDXPlusError(f"Missing antecedent flag for {code}")
            if not isinstance(definition.get("value_meaning", {}), dict) or not isinstance(definition.get("possible-values", []), list):
                raise DDXPlusError(f"Invalid value mapping for {code}")
        self.condition_names = {
            code: _english(definition.get("cond-name-eng") or definition.get("condition_name"), "condition name")
            for code, definition in self.conditions.items()
        }
        logger.info("Loaded %d evidence definitions and %d conditions", len(self.evidences), len(self.conditions))

    def _load_mapping(self, filename: str) -> dict:
        path = self.data_dir / filename
        try:
            with path.open(encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            raise DDXPlusError(f"Cannot load {path}: {exc}") from exc
        if not isinstance(data, dict) or not data or not all(isinstance(v, dict) for v in data.values()):
            raise DDXPlusError(f"{filename} must contain a nonempty object of definitions")
        return data

    def decode_evidence(self, token: str, *, initial: bool = False) -> Evidence:
        if not isinstance(token, str) or not token.strip():
            raise DDXPlusError("Evidence token must be a nonempty string")
        parts = token.strip().split("_@_")
        if len(parts) > 2:
            raise DDXPlusError("Malformed evidence/value token")
        code = parts[0]
        if code not in self.evidences:
            raise DDXPlusError(f"Unknown evidence code: {code}")
        definition = self.evidences[code]
        question = _english(definition["question_en"], "question")
        raw = parts[1] if len(parts) == 2 else None
        kind = definition["data_type"]
        if kind == "B":
            if raw is None:
                value = "Yes"
            elif raw.lower() in {"1", "true", "yes"}:
                value = "Yes"
            elif raw.lower() in {"0", "false", "no"}:
                value = "No"
            else:
                raise DDXPlusError(f"Invalid binary value for {code}")
        elif raw is None:
            if not initial:
                raise DDXPlusError(f"Categorical evidence {code} requires a value")
            value = None
        else:
            allowed = {str(v) for v in definition.get("possible-values", [])}
            meanings = definition.get("value_meaning", {})
            if allowed and raw not in allowed:
                raise DDXPlusError(f"Value outside declared domain for {code}")
            if raw in meanings:
                entry = meanings[raw]
                value = _english(entry.get("en") if isinstance(entry, dict) else entry, "value meaning")
                value = {"N": "No", "Y": "Yes", "NA": "Not applicable"}.get(value, value)
            elif raw in allowed and re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
                value = raw  # Preserve numeric scale; do not invent units or severity.
            else:
                raise DDXPlusError(f"No English value mapping for {code}")
        return Evidence(code, question, value, definition["is_antecedent"], raw)

    def _condition(self, value: Any) -> str:
        if not isinstance(value, str):
            raise DDXPlusError("Condition label must be a string")
        value = value.strip()
        if value in self.condition_names:
            return self.condition_names[value]
        if value in self.condition_names.values():
            return value
        raise DDXPlusError("Unknown evaluation condition label")

    def parse_labels(self, row: Mapping[str, Any]) -> EvaluationLabels:
        pathology = row.get("PATHOLOGY")
        pathology = self._condition(pathology) if pathology not in (None, "") else None
        differential = []
        for entry in _list(row.get("DIFFERENTIAL_DIAGNOSIS", []), "DIFFERENTIAL_DIAGNOSIS"):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise DDXPlusError("Differential entry must be [condition, probability]")
            try:
                if isinstance(entry[1], bool):
                    raise ValueError
                probability = float(entry[1])
            except (TypeError, ValueError) as exc:
                raise DDXPlusError("Invalid differential probability") from exc
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise DDXPlusError("Differential probability must be finite and between 0 and 1")
            differential.append(DifferentialDiagnosis(self._condition(entry[0]), probability))
        return EvaluationLabels(pathology, tuple(differential))

    def parse_patient(self, row: Mapping[str, Any]) -> PatientRepresentation:
        """Reads clinical columns only; never reads PATHOLOGY or DIFFERENTIAL_DIAGNOSIS."""
        raw_age = row.get("AGE")
        age = None
        if raw_age not in (None, ""):
            if isinstance(raw_age, bool) or not re.fullmatch(r"\d+", str(raw_age).strip()):
                raise DDXPlusError("AGE must be a nonnegative integer or missing")
            age = int(raw_age)
        raw_sex = row.get("SEX")
        sex = None if raw_sex in (None, "") else str(raw_sex).strip().upper()
        if sex not in (None, "M", "F"):
            raise DDXPlusError("SEX must be M, F, or missing")
        decoded = []
        seen = set()
        values_by_code: dict[str, set[str | None]] = {}
        for token in _list(row.get("EVIDENCES"), "EVIDENCES"):
            evidence = self.decode_evidence(token)
            key = (evidence.code, evidence.value)
            values = values_by_code.setdefault(evidence.code, set())
            values.add(evidence.value)
            if self.evidences[evidence.code]["data_type"] != "M" and len(values) > 1:
                raise DDXPlusError("Conflicting values for a single-valued evidence")
            if key not in seen:
                seen.add(key)
                decoded.append(evidence)
        initial_items = ()
        raw_initial = row.get("INITIAL_EVIDENCE")
        if raw_initial not in (None, ""):
            initial = self.decode_evidence(raw_initial, initial=True)
            matches = tuple(e for e in decoded if e.code == initial.code)
            if initial.value_code is not None:
                if matches and initial not in matches:
                    raise DDXPlusError("INITIAL_EVIDENCE conflicts with recorded evidence")
                initial_items = (initial,)
            else:
                if self.evidences[initial.code]["data_type"] == "B" and matches and matches[0].value != "Yes":
                    raise DDXPlusError("Initial positive evidence conflicts with a negative finding")
                initial_items = matches or (initial,)
        return PatientRepresentation(age, sex,
            tuple(e for e in decoded if not e.is_antecedent),
            tuple(e for e in decoded if e.is_antecedent), initial_items)

    def iter_patients(self, split: str = "train", *, limit: int | None = None,
                      include_labels: bool = False) -> Iterator[PatientRecord]:
        """Read only the requested split, one row at a time; fail closed on bad rows."""
        if split not in SPLIT_FILES:
            raise DDXPlusError("split must be train, validate, or test")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise DDXPlusError("limit must be a nonnegative integer or None")
        path = self.data_dir / SPLIT_FILES[split]
        logger.info("Streaming %s (limit=%s, labels=%s)", path.name, limit, include_labels)
        try:
            with zipfile.ZipFile(path) as archive:
                expected = f"release_{split}_patients"
                members = [m for m in archive.infolist() if not m.is_dir() and Path(m.filename).name in {expected, expected + '.csv'}]
                if len(members) != 1:
                    raise DDXPlusError(f"{path.name} must contain exactly one {expected}[.csv] member")
                with archive.open(members[0]) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
                    reader = csv.DictReader(text)
                    required = {"AGE", "SEX", "EVIDENCES", "INITIAL_EVIDENCE"}
                    if include_labels:
                        required |= {"PATHOLOGY", "DIFFERENTIAL_DIAGNOSIS"}
                    if not required.issubset(reader.fieldnames or []):
                        raise DDXPlusError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
                    for number, row in enumerate(islice(reader, limit), 1):
                        try:
                            if None in row or any(v is None for v in row.values()):
                                raise DDXPlusError("CSV row has an incorrect number of fields")
                            patient = self.parse_patient(row)
                            labels = self.parse_labels(row) if include_labels else None
                        except DDXPlusError as exc:
                            raise DDXPlusError(f"{split} row {number}: {exc}") from exc
                        yield PatientRecord(f"ddxplus:{split}:{number}", split,
                            f"{path.name}!{members[0].filename}#row={number}", patient, labels)
        except (OSError, zipfile.BadZipFile, UnicodeError, csv.Error, EOFError) as exc:
            raise DDXPlusError(f"Cannot read {path}: {exc}") from exc
