"""Strict source-only medical documents, separate from patient/evaluation data."""
from dataclasses import asdict, dataclass, fields
import hashlib
import re
import unicodedata
from uuid import NAMESPACE_URL, uuid5

SOURCES = {'medlineplus', 'pmc', 'who'}
PROVENANCE_FIELDS = {'download_url', 'retrieved_at', 'raw_path', 'raw_sha256',
                     'license_url', 'license_evidence', 'terms_url', 'reviewed_by'}


def clean_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def reusable_license(url):
    """Exact CC URL allowlist; noncommercial and no-derivatives are excluded."""
    from urllib.parse import urlparse
    p = urlparse(url.strip())
    if p.scheme not in ('http', 'https') or p.hostname not in ('creativecommons.org', 'www.creativecommons.org'):
        return None
    path = p.path.rstrip('/')
    if path == '/publicdomain/zero/1.0':
        return 'CC0-1.0'
    match = re.fullmatch(r'/licenses/(by|by-sa)/(1\.0|2\.0|2\.5|3\.0|4\.0)(/igo)?', path)
    if match:
        return 'CC-' + match[1].upper() + '-' + match[2] + ('-IGO' if match[3] else '')
    return None


@dataclass(frozen=True)
class MedicalDocument:
    document_id: str
    text: str
    title: str
    source: str
    url: str
    license: str
    provenance: dict
    doi: str = ''
    pmid: str = ''
    pmcid: str = ''
    publication_date: str = ''
    document_type: str = ''

    def __post_init__(self):
        for name in ('document_id', 'text', 'title', 'url', 'license'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'Missing {name}')
        for name in ('doi', 'pmid', 'pmcid', 'publication_date', 'document_type'):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'Invalid metadata type: {name}')
        if self.source not in SOURCES or not self.document_id.startswith(self.source + ':'):
            raise ValueError('Only approved medical source documents are accepted')
        if not isinstance(self.provenance, dict) or set(self.provenance) - PROVENANCE_FIELDS:
            raise ValueError('Unexpected provenance fields')
        required = {'download_url', 'retrieved_at', 'raw_path', 'raw_sha256', 'license_url', 'license_evidence'}
        if not required <= self.provenance.keys() or not all(isinstance(v, str) and v.strip() for v in self.provenance.values()):
            raise ValueError('Missing source/license provenance')
        if not re.fullmatch('[0-9a-f]{64}', self.provenance['raw_sha256']):
            raise ValueError('Invalid raw SHA256')
        if self.source == 'medlineplus':
            if self.license != 'NLM-public-domain-summary' or self.provenance['license_url'] != 'https://medlineplus.gov/copyright.html':
                raise ValueError('Only NLM topic summaries are supported')
        elif reusable_license(self.provenance['license_url']) != self.license:
            raise ValueError('License is not explicitly reusable under the corpus policy')
        if any(marker in self.text.lower() for marker in ('ddxplus:', 'differential_diagnosis', 'ground_truth_pathology', 'pathology:', 'pathology=')):
            raise ValueError('Patient dataset/diagnosis-label serialization rejected')


@dataclass(frozen=True)
class MedicalChunk(MedicalDocument):
    chunk_id: str = ''
    chunk_index: int = 0
    chunking_version: str = 'words-v1:180:30'


def chunk_id(document_id, index, text, version):
    digest = hashlib.sha256(text.encode('utf8')).hexdigest()
    return str(uuid5(NAMESPACE_URL, f'medical-knowledge/{document_id}/{version}/{index}/{digest}'))


def validate_chunk(payload):
    if not isinstance(payload, dict) or set(payload) != {f.name for f in fields(MedicalChunk)}:
        raise ValueError('Medical chunk payload must use the exact allowlist (no patient/label fields)')
    chunk = MedicalChunk(**payload)
    if type(chunk.chunk_index) is not int or chunk.chunk_index < 0:
        raise ValueError('Invalid chunk index')
    if not re.fullmatch(r'words-v1:\d+:\d+', chunk.chunking_version):
        raise ValueError('Unknown chunking version')
    if chunk.chunk_id != chunk_id(chunk.document_id, chunk.chunk_index, chunk.text, chunk.chunking_version):
        raise ValueError('Non-deterministic chunk ID or altered text')
    return asdict(chunk)
