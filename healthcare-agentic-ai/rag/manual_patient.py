"""Strict local vignette input; never masquerades as a DDXPlus benchmark row."""
import hashlib
import json
from pathlib import Path
import re
from .models import Evidence, PatientRepresentation

MAX_CASE_BYTES = 64 * 1024
SECTIONS = ('Patient', 'Chief complaint', 'Symptoms', 'History', 'Available information')


def manual_patient_id(patient):
    if type(patient) is not PatientRepresentation:
        raise TypeError('Expected label-free PatientRepresentation')
    canonical = json.dumps(patient.to_inference_dict(), sort_keys=True, ensure_ascii=False,
                           separators=(',', ':'), allow_nan=False)
    return 'manual:' + hashlib.sha256(canonical.encode('utf8')).hexdigest()


def read_manual_patient(path):
    """Parse the explicit check3 section format, retaining literal values/negations.

    No LLM extraction, diagnosis labels, inferred findings, or dataset writes.
    Unknown sections/lines fail closed rather than silently dropping information.
    """
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(MAX_CASE_BYTES + 1)
    if len(raw) > MAX_CASE_BYTES:
        raise ValueError('Manual case exceeds input limit')
    sections, current = {}, None
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.endswith(':') and line[:-1] in SECTIONS:
            current = line[:-1]
            if current in sections:
                raise ValueError('Duplicate manual case section')
            sections[current] = []
        elif current is not None:
            if ':' in line and current != 'Patient':
                raise ValueError('Unsupported nested field in manual case')
            sections[current].append((number, line))
        else:
            raise ValueError('Expected explicit manual case sections')
    if tuple(sections) != SECTIONS or any(not values for values in sections.values()):
        raise ValueError('Expected all five nonempty manual case sections in order')
    demographic = sections['Patient']
    if (len(demographic) != 2 or not re.fullmatch(r'Age: (0|[1-9]\d{0,2})', demographic[0][1])
            or demographic[1][1] not in ('Sex: Female', 'Sex: Male')):
        raise ValueError('Expected explicit age and sex')
    age = int(demographic[0][1].split(': ')[1])
    if age > 130:
        raise ValueError('Age outside supported range')
    sex = {'Sex: Female': 'F', 'Sex: Male': 'M'}[demographic[1][1]]
    mapping = []

    def evidence(section, field, antecedent=False, offset=0):
        entries = []
        for i, (number, line) in enumerate(sections[section]):
            if section in ('Symptoms', 'History', 'Available information'):
                if not line.startswith('- ') or not line[2:].strip():
                    raise ValueError('Expected explicit nonempty bullet')
                literal = line[2:]
            else:
                literal = line
            # Section heading is the question; the original text is the answer.
            # E.g. Symptoms = No fever preserves the negative, not a fever diagnosis.
            item = Evidence(f'manual:{section}:{i}', section, literal, antecedent)
            entries.append(item)
            mapping.append({'source_line': number, 'literal': literal,
                            'patient_field': field, 'index': offset + i, 'rendered': item.text})
        return tuple(entries)

    symptoms = evidence('Symptoms', 'symptoms')
    history = evidence('History', 'antecedents', True)
    chief = evidence('Chief complaint', 'initial_evidence')
    available = evidence('Available information', 'initial_evidence', offset=len(chief))
    patient = PatientRepresentation(age, sex, symptoms, history, chief + available)
    return patient, {'source_file': path.name, 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'origin': 'User-supplied vignette; not a DDXPlus row; no expected diagnosis supplied.',
        'patient_id': manual_patient_id(patient), 'demographics': {'age': age, 'sex': sex},
        'literal_mapping': mapping, 'label_free_input': patient.to_inference_dict()}
