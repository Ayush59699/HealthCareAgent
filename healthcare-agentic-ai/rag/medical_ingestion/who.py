"""Explicit reviewed WHO PDF/text manifest, never website bulk downloads."""
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from .common import download
from .models import MedicalDocument, clean_text, reusable_license

WHO_HOSTS = {'www.who.int', 'iris.who.int', 'cdn.who.int', 'apps.who.int'}


def extract_text(path):
    path = Path(path)
    if path.suffix.lower() == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ValueError('Encrypted PDF is not supported')
        text = ' '.join(page.extract_text() or '' for page in reader.pages)
    elif path.suffix.lower() == '.txt':
        text = path.read_text(encoding='utf8')
    else:
        raise ValueError('Only selected PDF/UTF-8 text documents are supported')
    text = clean_text(text)
    if not text:
        raise ValueError('No extractable text; scanned documents need a separate reviewed OCR process')
    return text


def validate_selection(selection):
    required = {'id', 'title', 'url', 'download_url', 'sha256', 'license_url', 'license_evidence', 'reviewed_by', 'terms_url', 'format'}
    if not required <= selection.keys() or not all(isinstance(selection[k], str) and selection[k].strip() for k in required):
        raise ValueError('WHO selection requires per-document license review and pinned checksum')
    if not reusable_license(selection['license_url']):
        raise ValueError('WHO license excluded by initial CC0/BY/BY-SA policy (including NC)')
    for key in ('url', 'download_url', 'terms_url'):
        p = urlparse(selection[key])
        if p.scheme != 'https' or p.hostname not in WHO_HOSTS:
            raise ValueError('WHO selection must reference official HTTPS sources')
    if selection['format'] not in ('pdf', 'txt'):
        raise ValueError('Unsupported WHO format')


def download_document(selection, raw_dir):
    validate_selection(selection)
    # Hash-derived filename prevents manifest path traversal.
    name = hashlib.sha256(selection['id'].encode()).hexdigest()[:20]
    path = Path(raw_dir) / (name + '.' + selection['format'])
    provenance = download(selection['download_url'], path, WHO_HOSTS)
    if provenance['raw_sha256'] != selection['sha256']:
        raise ValueError('WHO selected document checksum mismatch')
    return path, provenance


def parse_who(path, provenance, selection):
    validate_selection(selection)
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != selection['sha256'] or provenance['raw_sha256'] != selection['sha256']:
        raise ValueError('WHO document differs from reviewed bytes')
    text = extract_text(path)
    if clean_text(selection['license_evidence']).casefold() not in text.casefold():
        raise ValueError('Reviewed license evidence not found in document')
    return MedicalDocument(document_id='who:' + selection['id'], title=selection['title'], text=text,
        source='who', url=selection['url'], license=reusable_license(selection['license_url']),
        publication_date=selection.get('publication_date', ''), document_type=selection.get('document_type', 'report'),
        provenance={**provenance, **{k: selection[k] for k in ('license_url', 'license_evidence', 'reviewed_by', 'terms_url')}})
