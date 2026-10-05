"""Experimental baseline-first compression; never imported by production.

One fixed strategy: reuse new3's syntactic compression, without headers or added
metadata; preserve ALL baseline facts, then try omitted antecedents in source
order followed by remaining omitted clinical facts. No topic priorities/labels.
"""
from dataclasses import asdict, dataclass

from .query_adapter import QueryFact, patient_facts


@dataclass(frozen=True)
class BaselineCompression:
    query: str
    token_count: int
    baseline_query: str
    baseline_token_count: int
    compressed_baseline_query: str
    compressed_baseline_token_count: int
    remaining_tokens_after_baseline: int
    baseline_included: tuple[QueryFact, ...]
    included: tuple[QueryFact, ...]
    newly_added: tuple[QueryFact, ...]
    omitted: tuple[QueryFact, ...]
    baseline_facts_lost: tuple[QueryFact, ...]
    fallback_reason: str | None
    version: str = 'baseline-preserving-v1'

    def audit(self):
        return {**asdict(self), 'character_count': len(self.query),
                'baseline_character_count': len(self.baseline_query)}


def render_clinical_facts(facts):
    """Flatten exact-question groups, with no invented role/demographic words.

    Unstructured facts are literal. Original question identity (not its compact
    spelling) is the grouping key; unanswered questions cannot swallow answers.
    """
    groups = {}
    for fact in facts:
        key = (fact.text.partition(' = ')[0], fact.answer is None)
        groups.setdefault(key, []).append(fact)
    entries = []
    for group in groups.values():
        first = group[0]
        entries.append(first.text if first.answer is None else
                       first.question + ' = ' + ' | '.join(f.answer for f in group))
    return '; '.join(entries)


def compress_baseline(state, baseline_query, tokenizer, max_tokens=256):
    """Validate exact production query and pack additions without any evictions.

    Recover baseline membership by replaying its whole-fact greedy selection,
    never substring matching or splitting on separators contained in patient text.
    Reject stale/foreign queries. If mandatory compression exceeds either budget,
    return the exact baseline, with no additions, rather than losing any fact.
    """
    if type(max_tokens) is not int or not 2 <= max_tokens <= 256:
        raise ValueError('Budget must be between 2 and the existing 256-token window')
    if not isinstance(baseline_query, str):
        raise TypeError('Expected the exact production query as text')
    # new3's inventory preserves original facts/references. Exclude its generated
    # demographics: only the three production clinical collections are eligible.
    facts = tuple(f for f in patient_facts(state)
                  if any(ref.startswith(('patient:presenting_evidence:', 'patient:symptoms:',
                                         'patient:antecedents:')) for ref in f.refs))
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, verbose=False))
    fits = lambda text: len(text) <= 4000 and count(text) <= max_tokens
    mandatory = []
    for fact in facts:
        if fits('; '.join(f.text for f in mandatory + [fact])):
            mandatory.append(fact)
    expected = '; '.join(f.text for f in mandatory)
    if baseline_query != expected:
        raise ValueError('Baseline is not the exact production query for this state/budget')
    selected = list(mandatory)
    compressed = render_clinical_facts(mandatory)
    fallback = None
    if not fits(compressed):
        query = baseline_query
        fallback = 'Mandatory compressed baseline exceeds query budget; exact baseline retained'
    else:
        query = compressed
        missing = [f for f in facts if f not in mandatory]
        # Source roles are audit/ordering only. No diagnosis or drug-name matching.
        ordered = [f for f in missing if any(r.startswith('patient:antecedents:') for r in f.refs)]
        ordered += [f for f in missing if f not in ordered]
        for fact in ordered:
            candidate = render_clinical_facts(selected + [fact])
            if fits(candidate):
                selected.append(fact)
                query = candidate
    lost = tuple(f for f in mandatory if f not in selected)
    if lost or not fits(query):
        raise RuntimeError('Baseline-preservation or query-budget invariant failed')
    return BaselineCompression(query=query, token_count=count(query),
        baseline_query=baseline_query, baseline_token_count=count(baseline_query),
        compressed_baseline_query=compressed, compressed_baseline_token_count=count(compressed),
        remaining_tokens_after_baseline=max_tokens - count(baseline_query if fallback else compressed),
        baseline_included=tuple(mandatory), included=tuple(selected),
        newly_added=tuple(f for f in selected if f not in mandatory),
        omitted=tuple(f for f in facts if f not in selected), baseline_facts_lost=lost,
        fallback_reason=fallback)
