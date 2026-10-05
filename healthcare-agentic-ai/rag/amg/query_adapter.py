"""Experimental, deterministic retrieval-only representation (not enabled by default).

No medical dictionary, predicted topics, label access, model or query expansion.
Only question scaffolding is shortened; answers and clinical qualifiers are literal.
Unrecognized question wording is retained. Every included/omitted fact is traceable.
"""
from dataclasses import asdict, dataclass
import re

from rag.agents.models import PatientState

ROLES = ('Presenting', 'Symptoms', 'History', 'Medications', 'Context')


def compact_question(question):
    """Conservative syntactic rewrites; never remove negation/time/family clauses."""
    question = question.strip().rstrip('?:').strip()
    replacements = (
        (r'^Do you have\s+', ''),
        (r'^Do you feel\s+', ''),
        (r'^Have you ever had\s+', 'ever had '),
        (r'^Have you recently had\s+', 'recently had '),
        (r'^Have you\s+', ''),
        (r'^Are you\s+', ''),
        (r'^Is your\s+', ''),
        (r'^Do you\s+', ''),
        (r'^Characterize your\s+', ''),
    )
    for pattern, replacement in replacements:
        rewritten, count = re.subn(pattern, replacement, question, count=1, flags=re.I)
        if count:
            return rewritten or question
    return question


@dataclass(frozen=True)
class QueryFact:
    text: str
    refs: tuple[str, ...]
    role: str
    question: str
    answer: str | None


@dataclass(frozen=True)
class QueryRepresentation:
    query: str
    token_count: int
    included: tuple[QueryFact, ...]
    omitted: tuple[QueryFact, ...]
    version: str = 'controlled-role-v1'

    def audit(self):
        return asdict(self)


def patient_facts(state):
    if type(state) is not PatientState:
        raise TypeError('Expected label-free PatientState, not a raw row or labeled record')
    inventory = {}
    for field, role in (('presenting_evidence', 'Presenting'), ('symptoms', 'Symptoms'),
                        ('antecedents', 'History')):
        for index, text in enumerate(getattr(state, field)):
            ref = f'patient:{field}:{index}'
            if text in inventory:
                inventory[text]['refs'].append(ref)
                continue
            question, separator, answer = text.partition(' = ')
            # Explicit medication-taking wording only, not a drug/topic dictionary.
            assigned = 'Medications' if role != 'Presenting' and re.match(r'^Are you (?:currently )?taking\b', question, re.I) else role
            inventory[text] = dict(text=text, refs=[ref], role=assigned,
                                   question=compact_question(question) if separator else text,
                                   answer=answer if separator else None)
    facts = [QueryFact(**{**item, 'refs': tuple(item['refs'])}) for item in inventory.values()]
    for field in ('age', 'sex'):
        value = getattr(state, field)
        facts.append(QueryFact(f'{field} = {value if value is not None else "unknown"}',
                               (f'patient:{field}',), 'Context', field,
                               str(value) if value is not None else 'unknown'))
    return facts


def render(facts):
    sections = []
    for role in ROLES:
        groups = {}
        for fact in facts:
            if fact.role == role:
                # Merge repeated EXACT original questions only. Do not collapse distinct
                # questions that happen to compact to the same wording.
                key = (fact.text.partition(' = ')[0], fact.answer is None)
                groups.setdefault(key, []).append(fact)
        entries = []
        for group in groups.values():
            first = group[0]
            entries.append(first.question if first.answer is None else
                           first.question + ' = ' + ' | '.join(f.answer for f in group))
        if entries:
            sections.append(role + ': ' + '; '.join(entries))
    return '\n'.join(sections)


def build_query(state, tokenizer, max_tokens=256):
    """Whole-fact packing, round-robin across roles; no clinical importance scores.

    Duplicates share references. Multi-answer question headings are emitted once.
    A fact that will not fit is audited, never clipped. Downstream state is untouched.
    """
    if type(max_tokens) is not int or max_tokens < 2 or max_tokens > 256:
        raise ValueError('Token budget must be between 2 and the existing 256-token window')
    facts = patient_facts(state)
    queues = [[f for f in facts if f.role == role] for role in ROLES]
    selected = []
    count = lambda text: len(tokenizer.encode(text, add_special_tokens=True, verbose=False))
    for index in range(max(map(len, queues), default=0)):
        for queue in queues:
            if index >= len(queue):
                continue
            fact = queue[index]
            # Output retains source order within each role, not selection order.
            candidate = render([f for f in facts if f in selected or f == fact])
            if len(candidate) <= 4000 and count(candidate) <= max_tokens:
                selected.append(fact)
    included = tuple(f for f in facts if f in selected)
    query = render(included)
    return QueryRepresentation(query, count(query), included,
                               tuple(f for f in facts if f not in selected))
