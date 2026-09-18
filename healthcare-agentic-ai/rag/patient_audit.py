"""Read-only, bounded-batch audit of actual Qdrant vectors and source features."""
import hashlib
import json
from .patient_ingestion import patient_document, validate_document


def audit_patient_index(store, parser=None):
    """Fail closed on contamination, wrong IDs/vectors, or altered source text.

    Source comparison streams TRAIN without labels and retains only IDs/digests,
    not full source texts. Clinical antecedents can legitimately name diseases;
    label independence is established by exact feature reconstruction, not words.
    """
    if not store.collection_exists():
        raise ValueError('Patient collection does not exist')
    store._validate_collection()
    params = store.client.get_collection(store.collection_name).config.params.vectors
    expected = {}
    fields = set()
    dimensions = set()
    offset = None
    count = 0
    while True:
        points, offset = store.client.scroll(store.collection_name, limit=128,
            offset=offset, with_payload=True, with_vectors=True)
        for point in points:
            payload = validate_document(point.payload or {})
            if str(point.id) != store.point_id(payload['patient_id']):
                raise ValueError('Stored point ID does not match deterministic patient ID')
            store._vector(point.vector)
            dimensions.add(len(point.vector))
            fields.update(payload)
            if payload['patient_id'] in expected:
                raise ValueError('Duplicate patient ID')
            expected[payload['patient_id']] = _digest(payload)
            count += 1
        if offset is None:
            break
    if count != store.count():
        raise ValueError('Scroll/count mismatch')
    fingerprint = _digest(expected)
    compared = 0
    if parser is not None and expected:
        last_row = max(int(pid.rsplit(':', 1)[1]) for pid in expected)
        for record in parser.iter_patients('train', limit=last_row, include_labels=False):
            digest = expected.pop(record.patient_id, None)
            if digest is not None:
                if digest != _digest(patient_document(record)):
                    raise ValueError('Stored payload differs from label-free training source')
                compared += 1
        if expected:
            raise ValueError('Stored patients are missing from training source')
    return {'collection': store.collection_name, 'storage_path': str(store.path),
        'collection_exists': True, 'indexed_count': count, 'payloads_audited': count,
        'vector_dimensions': sorted(dimensions), 'configured_dimension': params.size,
        'distance': params.distance.value, 'payload_fields': sorted(fields),
        'split_counts': {'train': count, 'validate': 0, 'test': 0},
        'source_payloads_verified': compared, 'deterministic_ids_verified': count,
        'payload_fingerprint': fingerprint}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
