"""Clinical features and evaluation labels deliberately live in separate objects."""
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Evidence:
    code: str
    question: str
    value: str | None
    is_antecedent: bool
    value_code: str | None = None

    @property
    def text(self) -> str:
        return f"{self.question} = {self.value}" if self.value is not None else f"{self.question} (answer not supplied)"


@dataclass(frozen=True)
class PatientRepresentation:
    age: int | None
    sex: str | None
    symptoms: tuple[Evidence, ...]
    antecedents: tuple[Evidence, ...]
    initial_evidence: tuple[Evidence, ...] = ()

    def to_inference_dict(self) -> dict:
        """Allowlisted, human-readable payload: no codes, provenance, or labels."""
        return {
            "age": self.age,
            "sex": self.sex,
            "symptoms": [e.text for e in self.symptoms],
            "antecedents": [e.text for e in self.antecedents],
            "initial_evidence": [e.text for e in self.initial_evidence],
        }

    def to_text(self) -> str:
        sex = {"M": "Male", "F": "Female"}.get(self.sex, "Not supplied")
        lines = ["Patient:", f"Age: {self.age if self.age is not None else 'Not supplied'}", f"Sex: {sex}"]
        for title, items in (("Symptoms / clinical evidence", self.symptoms), ("Antecedents", self.antecedents), ("Initial evidence", self.initial_evidence)):
            lines.extend(["", title + ":"])
            lines.extend(["- " + e.text for e in items] or ["- Not supplied"])
        lines.append("\nUnlisted findings are not assumed absent.")
        return "\n".join(lines)


@dataclass(frozen=True)
class DifferentialDiagnosis:
    condition: str
    probability: float


@dataclass(frozen=True)
class EvaluationLabels:
    ground_truth_pathology: str | None
    differential_diagnosis: tuple[DifferentialDiagnosis, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    split: str
    source: str
    patient: PatientRepresentation
    labels: EvaluationLabels | None = field(default=None, repr=False)

    def to_inference_dict(self) -> dict:
        return self.patient.to_inference_dict()

    def to_text(self) -> str:
        return self.patient.to_text()
