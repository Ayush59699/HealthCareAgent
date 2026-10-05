"""Narrow, opt-in ALL ambiguity guard around the byte-pinned vendor resolver.

Ordinary 'all'/'All' never grants an acronym expansion or lexical gate bypass.
Explicit uppercase ALL retains source-backed acronym behavior. Other aliases and
vendor code/snapshot pins are untouched; this wrapper holds no per-query state.
"""
import re
import unicodedata


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


class CaseSafeAliases:
    def __init__(self, upstream):
        self.upstream = upstream
        self.titles = upstream.titles

    def resolve(self, query):
        result = self.upstream.resolve(query)
        # Match vendor NFKC/token boundaries while retaining original casing.
        # An uppercase token already consumed by a longer phrase is NOT an
        # explicit acronym: "ALL TOPICS all day" must not expand ordinary all.
        tokens = re.findall(r'\w+', unicodedata.normalize('NFKC', query))
        key = normalize(query)
        longer_spans = [span.span() for alias in {m['alias'] for m in result['matches']
                        if m['alias'] != 'all' and 'all' in m['alias'].split()}
                        for span in re.finditer(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', key)]
        explicit, offset = 0, 0
        for token in tokens:
            if token == 'ALL' and not any(start <= offset < end for start, end in longer_spans):
                explicit += 1
            offset += len(token.casefold()) + 1
        matches = []
        removed = 0
        for match in result['matches']:
            if match['alias'] == 'all':
                if explicit:
                    explicit -= 1
                else:
                    removed += 1
                    continue
            matches.append(match)
        if not removed:
            return result
        # Reconstruct only the vendor's source-title additions after filtering.
        # No string substitution in the patient sentence, and no new aliases.
        additions = sorted({self.titles[m['topic_ids'][0]] for m in matches
                            if len(m['topic_ids']) == 1 and
                            normalize(self.titles[m['topic_ids'][0]]) != m['alias']})
        expanded = query + ('\nSource topic: ' + '; '.join(additions) if additions else '')
        concept = re.sub(r'^(?:what is|what are|define|explain|tell me about)\s+', '', normalize(query))
        exact = [] if concept == 'all' else result['exact_topic_ids']
        return {**result, 'matches': matches, 'expanded_query': expanded,
                'exact_topic_ids': exact, 'unambiguous_lookup': len(exact) == 1,
                'suppressed_ordinary_all_matches': removed}
