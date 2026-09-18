"""Training-only, allowlisted documents. No evaluation labels cross this boundary."""
import re
from itertools import islice
from .models import PatientRecord
from .patient_parser import DDXPlusParser

PAYLOAD_FIELDS = frozenset({'patient_id', 'split', 'text', 'source'})


def validate_document(document: dict) -> dict:
    if set(document) != PAYLOAD_FIELDS:
        raise ValueError('Patient payload must contain exactly patient_id, split, text, source; labels are forbidden')
    if not all(isinstance(v, str) and v.strip() for v in document.values()):
        raise ValueError('Patient payload values must be nonempty strings')
    if document['split'] != 'train' or not re.fullmatch(r'ddxplus:train:[1-9]\d*', document['patient_id']):
        raise ValueError('Only training patients may be indexed')
    if not re.fullmatch(r'release_train_patients.zip!.*release_train_patients(?:\.csv)?#row=[1-9]\d*', document['source']):
        raise ValueError('Patient source must identify the training archive')
    if document['source'].rsplit('#row=', 1)[1] != document['patient_id'].rsplit(':', 1)[1]:
        raise ValueError('Patient ID and source row must agree')
    return dict(document)


def patient_document(record: PatientRecord) -> dict:
    # Deliberately never serialize record.labels or the whole record.
    return validate_document({'patient_id': record.patient_id, 'split': record.split,
                              'text': record.patient.to_text(), 'source': record.source})


def iter_training_documents(parser: DDXPlusParser, limit: int | None = None):
    for record in parser.iter_patients('train', limit=limit, include_labels=False):
        yield patient_document(record)


def batches(items, batch_size: int):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch
