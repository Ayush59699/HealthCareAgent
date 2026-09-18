"""Official MedlinePlus health topic XML; only NLM full-summary text is indexed."""
from pathlib import Path
from .common import download, xml_root, html_text, element_text
from .models import MedicalDocument

XML_URL = 'https://medlineplus.gov/xml/mplus_topics_2026-09-18.xml'
TERMS_URL = 'https://medlineplus.gov/copyright.html'
DEFAULT_TOPICS = ('Asthma', 'Diabetes', 'High Blood Pressure')


def download_topics(raw_dir):
    path = Path(raw_dir) / 'mplus_topics_2026-09-18.xml'
    return path, download(XML_URL, path, {'medlineplus.gov'})


def parse_medlineplus(path, provenance, titles=DEFAULT_TOPICS):
    selected = set(titles)
    for topic in xml_root(path).iter('health-topic'):
        if topic.get('language', 'English') != 'English' or topic.get('title') not in selected:
            continue
        summary = topic.find('full-summary')
        if summary is None:
            continue
        # XML contains escaped HTML in current dumps; real child markup is also supported.
        text = html_text(element_text(summary))
        if not text:
            continue
        yield MedicalDocument(document_id='medlineplus:' + topic.attrib['id'], text=text,
            title=topic.attrib['title'], source='medlineplus', url=topic.attrib['url'],
            publication_date=topic.get('date-created', ''), document_type='health-topic-summary',
            license='NLM-public-domain-summary', provenance={**provenance,
                'license_url': TERMS_URL, 'terms_url': TERMS_URL,
                'license_evidence': 'NLM-produced health topic summaries only; linked external content, images, encyclopedia and drug monographs excluded. See MedlinePlus copyright policy.'})
