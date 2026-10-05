"""Official topic XML: index NLM summaries only, never linked third-party pages."""
from html.parser import HTMLParser
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from .common import download, xml_root, html_text, element_text
from .models import MedicalDocument, clean_text

XML_URL = 'https://medlineplus.gov/xml/mplus_topics_2026-09-22.xml'
SOURCE_VERSION = '2026-09-22'
TERMS_URL = 'https://medlineplus.gov/copyright.html'
DEFAULT_TOPICS = ('Asthma', 'Diabetes', 'High Blood Pressure')


def download_topics(raw_dir):
    path = Path(raw_dir) / ('mplus_topics_' + SOURCE_VERSION + '.xml')
    return path, download(XML_URL, path, {'medlineplus.gov'}, exact_url=True)


class SummarySections(HTMLParser):
    """Keep source headings/paragraphs; strip HTML and non-content elements."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.sections, self.parts, self.heading_parts = [], [], []
        self.title = ''
        self.heading = False
        self.hidden = 0

    def flush(self):
        text = clean_text(''.join(self.parts))
        if text:
            self.sections.append({'title': self.title, 'text': text})
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav'):
            self.hidden += 1
        if self.hidden:
            return
        if re.fullmatch('h[1-6]', tag):
            self.flush()
            self.heading, self.heading_parts = True, []
        elif tag in ('p', 'br', 'li', 'div', 'ul', 'ol'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav'):
            self.hidden = max(0, self.hidden - 1)
            return
        if self.hidden:
            return
        if re.fullmatch('h[1-6]', tag):
            self.title = clean_text(''.join(self.heading_parts))
            self.heading = False
        elif tag in ('p', 'li', 'div', 'ul', 'ol'):
            self.parts.append('\n')

    def handle_data(self, data):
        if not self.hidden:
            (self.heading_parts if self.heading else self.parts).append(data)


def summary_sections(summary):
    # Actual snapshot uses escaped HTML; support real nested XML markup as well.
    value = summary.text or ''
    if len(summary):
        value += ''.join(ET.tostring(c, encoding='unicode') for c in summary)
    parser = SummarySections()
    parser.feed(value)
    parser.close()
    parser.flush()
    return parser.sections


def structured(nodes):
    return [{**{k: v for k, v in n.attrib.items() if k in {'id', 'url', 'language'}},
             'title': element_text(n)} for n in nodes]


def parse_medlineplus(path, provenance, titles=None, *, stats=None):
    """All English summaries by default. Missing summaries are reported, not invented.

    Explicit title selections retain the historical words-v1 normalization for
    reproducibility of the original development corpus.
    """
    selected = set(titles) if titles is not None else None
    root = xml_root(path)
    if root.tag != 'health-topics':
        raise ValueError('Expected official MedlinePlus health-topics XML')
    counts = stats if stats is not None else {}
    counts.update(total_records=0, english_records=0, parsed_documents=0,
                  excluded_non_english=0, excluded_unselected=0, missing_summary=[])
    version = re.search(r'mplus_topics_(\d{4}-\d{2}-\d{2})\.xml$', str(path))
    seen = set()
    for topic in root.iter('health-topic'):
        counts['total_records'] += 1
        if topic.get('language') not in ('English', 'en'):
            counts['excluded_non_english'] += 1
            continue
        counts['english_records'] += 1
        if selected is not None and topic.get('title') not in selected:
            counts['excluded_unselected'] += 1
            continue
        if not all(topic.get(k, '').strip() for k in ('id', 'title', 'url')):
            raise ValueError('English topic missing required id/title/url')
        if topic.get('id') in seen:
            raise ValueError('Duplicate MedlinePlus topic ID: ' + topic.get('id'))
        seen.add(topic.get('id'))
        summary = topic.find('full-summary')
        sections = summary_sections(summary) if summary is not None else []
        if not sections:
            counts['missing_summary'].append({'id': topic.get('id'), 'title': topic.get('title')})
            continue
        metadata = {
            'topic_attributes': dict(topic.attrib), 'dataset_attributes': dict(root.attrib),
            'also_called': [element_text(n) for n in topic.findall('also-called')],
            'see_references': [element_text(n) for n in topic.findall('see-reference')],
            'groups': structured(topic.findall('group')),
            'related_topics': structured(topic.findall('related-topic')),
            'primary_institutes': structured(topic.findall('primary-institute')),
            'mesh_headings': structured(topic.findall('mesh-heading/descriptor')),
            'language_mapped_topics': structured(topic.findall('language-mapped-topic')),
        }
        if selected is None:
            metadata['sections'] = sections
            text = '\n\n'.join((s['title'] + '\n' if s['title'] else '') + s['text'] for s in sections)
        else:
            text = html_text(element_text(summary))
        counts['parsed_documents'] += 1
        yield MedicalDocument(document_id='medlineplus:' + topic.attrib['id'], text=text,
            title=topic.attrib['title'], source='medlineplus', url=topic.attrib['url'],
            language='en', source_version=version[1] if version else '', metadata=metadata,
            publication_date=topic.get('date-created', ''), document_type='health-topic-summary',
            license='NLM-public-domain-summary', provenance={**provenance,
                'license_url': TERMS_URL, 'terms_url': TERMS_URL,
                'license_evidence': 'NLM-produced health topic summaries only; linked external content, images, encyclopedia and drug monographs excluded. See MedlinePlus copyright policy.'})
    if root.get('total') is not None and int(root.get('total')) != counts['total_records']:
        raise ValueError('MedlinePlus root total differs from parsed record count')
    if not counts['parsed_documents']:
        raise ValueError('No valid English MedlinePlus summaries')
