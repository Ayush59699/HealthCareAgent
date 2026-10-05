"""Deterministic 180-word windows, 30 overlap; prefer provided summary sections."""
from dataclasses import asdict
from .models import MedicalDocument, MedicalChunk, chunk_id, clean_text, validate_chunk


def chunk_document(document, max_words=180, overlap=30):
    if not isinstance(document, MedicalDocument) or isinstance(document, MedicalChunk):
        raise ValueError('Expected an approved medical source document, not a patient record/chunk')
    if type(max_words) is not int or type(overlap) is not int or not 0 <= overlap < max_words:
        raise ValueError('Require max_words > overlap >= 0')
    sections = document.metadata.get('sections')
    version = f"{'sections-v2' if sections else 'words-v1'}:{max_words}:{overlap}"
    passages = []
    for section in sections or [{'title': '', 'text': document.text}]:
        title = section['title']
        text = clean_text((title + ' ' if title else '') + section['text'])
        # Fold a short adjacent section into a meaningful passage, not tiny chunks.
        if passages and min(len(passages[-1][1].split()), len(text.split())) < 40 and len(passages[-1][1].split()) + len(text.split()) <= max_words:
            previous_title, previous_text = passages.pop()
            passages.append((' / '.join(filter(None, (previous_title, title))), previous_text + ' ' + text))
        else:
            passages.append((title, text))
    index, seen = 0, set()
    for section, passage in passages:
        words = passage.split()
        for start in range(0, len(words), max_words - overlap):
            text = ' '.join(words[start:start + max_words])
            if text not in seen:
                payload = {**asdict(document), 'text': text, 'section': section,
                           'metadata': {k: v for k, v in document.metadata.items() if k != 'sections'},
                           'chunk_index': index, 'chunking_version': version,
                           'chunk_id': chunk_id(document.document_id, index, text, version)}
                yield MedicalChunk(**validate_chunk(payload))
                seen.add(text)
                index += 1
            if start + max_words >= len(words):
                break
