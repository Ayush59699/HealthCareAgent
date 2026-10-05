"""AMG evidence contract: native IDs/text/metadata, never legacy UUID fabrication."""
import copy
import hashlib
import json
import math
import re
from urllib.parse import urlsplit

BACKEND = 'amg-medlineplus-v1'
_FIELDS = {'backend', 'chunk_id', 'document_id', 'source', 'title', 'url', 'text',
           'amg_metadata', 'retrieval'}
_METADATA = set('source record_type topic_id canonical_id title url language date date_created date_generated source_archive source_member source_sha256 parser_version summary_status aliases groups mesh_terms related_topics see_references primary_institutes content_sha256 chunk_id parent_id token_count chunker_version source_spans excluded_summary_lines split_sentence chunk_sha256'.split())


def validate_amg_hit(payload):
    if type(payload) is not dict or set(payload) != _FIELDS or payload['backend'] != BACKEND:
        raise ValueError('Invalid AMG evidence envelope')
    m, r = payload['amg_metadata'], payload['retrieval']
    if type(m) is not dict or set(m) != _METADATA:
        raise ValueError('Unexpected AMG source metadata')
    if (m['source'] != 'MedlinePlus' or m['language'] != 'English' or
            m['record_type'] != 'health_topic' or payload['source'] != 'medlineplus'):
        raise ValueError('Only supplied English MedlinePlus topics are accepted')
    if not isinstance(m['topic_id'], str) or not m['topic_id'].isdigit():
        raise ValueError('Invalid topic ID')
    parent = 'medlineplus:topic:' + m['topic_id']
    if (payload['document_id'] != parent or m['canonical_id'] != parent or m['parent_id'] != parent or
            type(m['chunk_id']) is not int or m['chunk_id'] < 0 or m['chunker_version'] != '1' or
            payload['chunk_id'] != f"{parent}:lab:1:chunk:{m['chunk_id']}"):
        raise ValueError('AMG native source identity mismatch')
    if payload['title'] != m['title'] or payload['url'] != m['url']:
        raise ValueError('AMG source binding mismatch')
    url = urlsplit(payload['url'])
    if url.scheme != 'https' or url.hostname != 'medlineplus.gov' or url.username or url.password:
        raise ValueError('Invalid MedlinePlus URL')
    text = payload['text']
    if not isinstance(text, str) or not text.strip() or hashlib.sha256(text.encode()).hexdigest() != m['chunk_sha256']:
        raise ValueError('AMG source text checksum mismatch')
    for key in ('chunk_sha256', 'content_sha256', 'source_sha256'):
        if not isinstance(m[key], str) or not re.fullmatch('[0-9a-f]{64}', m[key]):
            raise ValueError('Invalid source checksum')
    if type(m['token_count']) is not int or not 1 <= m['token_count'] <= 256:
        raise ValueError('Invalid source token budget')
    spans = json.loads(m['source_spans'])
    if not spans or any(type(s) is not dict or set(s) != {'start', 'end', 'split_sentence'} or
                        type(s['start']) is not int or type(s['end']) is not int or
                        not 0 <= s['start'] < s['end'] or type(s['split_sentence']) is not bool for s in spans):
        raise ValueError('Invalid source spans')
    if (type(r) is not dict or set(r) != {'distance', 'metric', 'acceptance', 'max_distance', 'snapshot', 'collection'} or
            r['metric'] != 'squared_l2' or r['collection'] != 'medlineplus_en_experimental' or
            not isinstance(r['snapshot'], str) or not re.fullmatch('[0-9a-f]{24}', r['snapshot'])):
        raise ValueError('Invalid AMG retrieval provenance')
    for key in ('distance', 'max_distance'):
        if type(r[key]) not in (float, int) or not math.isfinite(r[key]) or r[key] < 0:
            raise ValueError('Invalid AMG distance')
    if r['acceptance'] not in {'distance_gate', 'exact_source_lookup'}:
        raise ValueError('Rejected candidates cannot enter evidence')
    if r['acceptance'] == 'distance_gate' and r['distance'] > r['max_distance']:
        raise ValueError('Evidence did not pass AMG gate')
    return copy.deepcopy(payload)
