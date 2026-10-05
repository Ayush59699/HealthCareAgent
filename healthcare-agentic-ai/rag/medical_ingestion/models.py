"""Strict source-only medical documents, separate from patient/evaluation data."""
from dataclasses import asdict, dataclass, fields, field
import hashlib
import re
import unicodedata
from uuid import NAMESPACE_URL, uuid5

SOURCES = {'medlineplus', 'pmc', 'who'}
PROVENANCE_FIELDS = {'download_url', 'retrieved_at', 'raw_path', 'raw_sha256',
                     'license_url', 'license_evidence', 'terms_url', 'reviewed_by',
                     'file_size_bytes', 'dataset_filename'}


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


def validate_metadata(metadata):
    # Source-only allowlist: expansion must not open an arbitrary label payload.
    string_lists = {'also_called', 'see_references'}
    record_lists = {'groups', 'related_topics', 'primary_institutes', 'mesh_headings', 'language_mapped_topics'}
    attribute_fields = {
        'topic_attributes': {'id', 'title', 'url', 'language', 'date-created', 'meta-desc', 'date-updated'},
        'dataset_attributes': {'total', 'date-generated'},
    }
    allowed = string_lists | record_lists | set(attribute_fields) | {'sections'}
    if not isinstance(metadata, dict) or set(metadata) - allowed:
        raise ValueError('Unexpected medical metadata fields')
    def record(value, keys):
        if not isinstance(value, dict) or set(value) - keys or not all(isinstance(v, str) for v in value.values()):
            raise ValueError('Invalid structured medical metadata')
    for key, value in metadata.items():
        if key in attribute_fields:
            record(value, attribute_fields[key])
        elif not isinstance(value, list):
            raise ValueError('Expected medical metadata list')
        elif key in string_lists:
            if not all(isinstance(v, str) for v in value):
                raise ValueError('Expected source strings')
        else:
            for item in value:
                record(item, {'title', 'text'} if key == 'sections' else {'id', 'title', 'url', 'language'})
                if key == 'sections' and (set(item) != {'title', 'text'} or not item['text'].strip()):
                    raise ValueError('Invalid medical section')


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
    language: str = ''
    source_version: str = ''
    section: str = ''
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ('document_id', 'text', 'title', 'url', 'license'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'Missing {name}')
        for name in ('doi', 'pmid', 'pmcid', 'publication_date', 'document_type', 'language', 'source_version', 'section'):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f'Invalid metadata type: {name}')
        validate_metadata(self.metadata)
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
    allowed = {f.name for f in fields(MedicalChunk)}
    legacy = allowed - {'language', 'source_version', 'section', 'metadata'}
    if not isinstance(payload, dict) or set(payload) not in (allowed, legacy):
        raise ValueError('Medical chunk payload must use the exact allowlist (no patient/label fields)')
    chunk = MedicalChunk(**payload)
    if type(chunk.chunk_index) is not int or chunk.chunk_index < 0:
        raise ValueError('Invalid chunk index')
    if not re.fullmatch(r'(words-v1|sections-v2):\d+:\d+', chunk.chunking_version):
        raise ValueError('Unknown chunking version')
    if chunk.chunking_version.startswith('sections-v2:'):
        if set(payload) != allowed or (chunk.source == 'medlineplus' and (chunk.language != 'en' or not chunk.source_version)):
            raise ValueError('Section-aware MedlinePlus chunks require expanded provenance metadata')
    if chunk.chunk_id != chunk_id(chunk.document_id, chunk.chunk_index, chunk.text, chunk.chunking_version):
        raise ValueError('Non-deterministic chunk ID or altered text')
    # Historical indexes remain readable, without admitting arbitrary extra keys.
    return {k: v for k, v in asdict(chunk).items() if k in payload}
