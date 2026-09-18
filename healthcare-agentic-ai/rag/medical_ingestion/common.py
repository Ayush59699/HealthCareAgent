"""Bounded official downloads, immutable raw evidence and safe extraction."""
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from defusedxml import ElementTree as ET
from .models import clean_text

DATA_ROOT = Path(__file__).resolve().parents[3] / 'data' / 'medical_knowledge'


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0
    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.hidden += 1
        if tag in ('p', 'br', 'li', 'div'):
            self.parts.append(' ')
    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)
        self.parts.append(' ')
    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def html_text(value):
    parser = TextOnly()
    parser.feed(value)
    return clean_text(''.join(parser.parts))


def element_text(element):
    return clean_text(' '.join(element.itertext())) if element is not None else ''


def xml_root(path):
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    return root


def download(url, destination, allowed_hosts, max_bytes=80_000_000):
    """Only HTTPS official endpoints, bounded bytes/time; never HTML scraping."""
    def check(value):
        p = urlparse(value)
        if p.scheme != 'https' or p.hostname not in allowed_hosts or p.username or p.password:
            raise ValueError('Download URL must use an approved official HTTPS host')
    check(url)
    destination = Path(destination)
    sidecar = destination.with_suffix(destination.suffix + '.provenance.json')
    if destination.exists() and sidecar.exists():
        metadata = json.loads(sidecar.read_text(encoding='utf8'))
        if metadata['download_url'] != url or metadata['raw_sha256'] != hashlib.sha256(destination.read_bytes()).hexdigest():
            raise ValueError('Cached raw file provenance mismatch')
        return metadata
    class OfficialRedirect(__import__('urllib.request', fromlist=['HTTPRedirectHandler']).HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)
    from urllib.request import build_opener
    request = Request(url, headers={'User-Agent': 'MedicalKnowledgeResearch/0.1 (bounded retrieval-only corpus)'})
    with build_opener(OfficialRedirect()).open(request, timeout=90) as response:
        check(response.url)
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError('Download exceeds size limit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {'download_url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'raw_path': str(destination.resolve()), 'raw_sha256': hashlib.sha256(data).hexdigest()}
    destination.write_bytes(data)
    sidecar.write_text(json.dumps(metadata, indent=2), encoding='utf8')
    return metadata


def verified_provenance(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(path.suffix + '.provenance.json').read_text(encoding='utf8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['raw_sha256']:
        raise ValueError('Raw data checksum mismatch')
    return metadata
