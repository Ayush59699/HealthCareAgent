"""PMC official OAI-PMH JATS; pmc-open set and explicit article license required."""
from pathlib import Path
import re
from .common import download, xml_root, element_text
from .models import MedicalDocument, reusable_license

HOSTS = {'www.ncbi.nlm.nih.gov', 'pmc.ncbi.nlm.nih.gov'}


def download_article(pmcid, raw_dir):
    if not re.fullmatch(r'PMC\d+', pmcid):
        raise ValueError('Expected PMCID')
    directory = Path(raw_dir)
    path = directory / (pmcid + '.xml')
    url = 'https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:' + pmcid[3:] + '&metadataPrefix=pmc'
    provenance = download(url, path, HOSTS, 15_000_000)
    root = xml_root(path)
    if not any(element_text(n) == 'pmc-open' for n in root.iter('setSpec')):
        raise ValueError('OAI record is not in the PMC Open Access set')
    parse_pmc(path, provenance)  # Fail closed before admitting to corpus.
    return path, provenance


def parse_pmc(path, provenance):
    root = xml_root(path)
    if root.tag != 'article' and not any(element_text(n) == 'pmc-open' for n in root.iter('setSpec')):
        raise ValueError('OAI record is not in the PMC Open Access set')
    article = root if root.tag == 'article' else root.find('.//article')
    if article is None:
        raise ValueError('No authorized JATS article in response')
    meta = article.find('./front/article-meta')
    if meta is None:
        raise ValueError('Missing article metadata')
    licenses = meta.findall('./permissions/license')
    accepted = []
    for node in licenses:
        urls = [value for key, value in node.attrib.items() if key.rsplit('}', 1)[-1] == 'href']
        urls += [value for child in node.iter() for key, value in child.attrib.items() if key.rsplit('}', 1)[-1] == 'href']
        urls += [element_text(child) for child in node.iter('license_ref')]
        for url in urls:
            name = reusable_license(url)
            if name:
                accepted.append((name, url, element_text(node)))
    if not accepted:
        raise ValueError('No explicitly reusable article license')
    name, license_url, evidence = accepted[0]
    ids = {n.get('pub-id-type'): element_text(n) for n in meta.findall('article-id')}
    pmcid = ids.get('pmc', ids.get('pmcid', ''))
    if pmcid and not pmcid.startswith('PMC'):
        pmcid = 'PMC' + pmcid
    if not re.fullmatch(r'PMC\d+', pmcid):
        raise ValueError('Missing PMCID')
    # Exclude references, tables and figures (potential third-party material).
    sections = meta.findall('abstract') + article.findall('body')
    paragraphs = []
    def collect(node):
        if node.tag in ('fig', 'table-wrap', 'ref-list', 'supplementary-material'):
            return
        if node.tag in ('p', 'title'):
            paragraphs.append(element_text(node))
        else:
            for child in node:
                collect(child)
    for section in sections:
        collect(section)
    dates = meta.findall('pub-date')
    date = '-'.join(element_text(dates[0].find(k)) for k in ('year', 'month', 'day') if element_text(dates[0].find(k))) if dates else ''
    return MedicalDocument(document_id='pmc:' + pmcid, text=' '.join(paragraphs),
        title=element_text(meta.find('title-group/article-title')), source='pmc',
        url='https://pmc.ncbi.nlm.nih.gov/articles/' + pmcid + '/', pmcid=pmcid,
        pmid=ids.get('pmid', ''), doi=ids.get('doi', ''), publication_date=date,
        document_type=article.get('article-type', 'journal-article'), license=name,
        provenance={**provenance, 'license_url': license_url, 'license_evidence': evidence or license_url})
