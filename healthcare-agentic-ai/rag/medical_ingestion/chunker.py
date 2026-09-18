"""Deterministic bounded word windows with explicit overlap/version."""
from dataclasses import asdict
from .models import MedicalDocument, MedicalChunk, chunk_id, clean_text, validate_chunk


def chunk_document(document, max_words=180, overlap=30):
    if not isinstance(document, MedicalDocument) or isinstance(document, MedicalChunk):
        raise ValueError('Expected an approved medical source document, not a patient record/chunk')
    if type(max_words) is not int or type(overlap) is not int or not 0 <= overlap < max_words:
        raise ValueError('Require max_words > overlap >= 0')
    words = clean_text(document.text).split()
    version = f'words-v1:{max_words}:{overlap}'
    for index, start in enumerate(range(0, len(words), max_words - overlap)):
        text = ' '.join(words[start:start + max_words])
        payload = {**asdict(document), 'text': text, 'chunk_index': index,
                   'chunking_version': version, 'chunk_id': chunk_id(document.document_id, index, text, version)}
        yield MedicalChunk(**validate_chunk(payload))
        if start + max_words >= len(words):
            break
