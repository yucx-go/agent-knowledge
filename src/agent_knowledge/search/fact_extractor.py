"""Extract incidental personal facts from conversational text.

LongMemEval and similar benchmarks reveal a common conversational pattern:
users embed personal life events as parenthetical asides inside otherwise
unrelated discussions. Examples:

    "Can you suggest cocktail recipes? By the way, I've been using Spotify lately."
    "I'm looking for a contract template — I just signed my first client today."
    "Need fertilizer advice. By the way, I planted 12 tomato saplings two weeks ago."

These insertions are the answers to memory questions like
"what music service do I use" or "what kitchen appliance did I buy",
but BM25 over the whole session can't surface them — the surrounding
text dilutes the keyword signal.

This module extracts those short fact-bearing sentences as standalone
units, which the search engine then indexes as child documents pointing
back to the parent source. Short focused docs let BM25 score them well
when the query happens to match.
"""

from __future__ import annotations

import re

# Insertion markers — phrases that introduce incidental personal facts.
# Each pattern captures the rest of the clause until end-of-sentence.
_INCIDENTAL_PATTERNS: list[re.Pattern] = [
    # Explicit insertion marker
    re.compile(r"\bby the way[,]?\s+([^.!?\n]{8,300})", re.IGNORECASE),

    # First-person past actions with specific verbs (avoids false positives
    # on "I just want to..." style meta-utterances)
    re.compile(
        r"\bI\s+(?:just|recently|finally)\s+"
        r"(?:got|bought|purchased|attended|completed|finished|had|signed|"
        r"started|tried|made|visited|saw|came back from|returned from|"
        r"adopted|moved|joined|launched|earned|achieved|received|"
        r"booked|ordered|installed|cooked|baked|read|watched|met|"
        r"played|ran|biked|hiked|swam|wrote|published)\s+"
        r"([^.!?\n]{4,250})",
        re.IGNORECASE,
    ),

    # Ongoing states — "I've been X-ing"
    re.compile(
        r"\bI(?:'ve| have)\s+been\s+"
        r"(?:using|trying|practicing|learning|reading|working on|going to|"
        r"thinking about|playing|watching|taking|studying|dating|seeing|"
        r"living in|staying at|eating|drinking|riding|driving|building|"
        r"writing|exploring|collecting)\s+"
        r"([^.!?\n]{4,250})",
        re.IGNORECASE,
    ),

    # Time-anchored mentions — these are the temporal-reasoning gold pattern
    re.compile(
        r"\b(?:today|yesterday|last (?:week|month|year|night|weekend|monday|"
        r"tuesday|wednesday|thursday|friday|saturday|sunday)|"
        r"this (?:morning|afternoon|evening|week|month)|"
        r"(?:two|three|four|five|six|several|a few)\s+(?:days?|weeks?|months?)\s+ago|"
        r"a (?:week|month|day)\s+ago)\s*[,]?\s*"
        r"I\s+([^.!?\n]{4,250})",
        re.IGNORECASE,
    ),

    # "I [verb-past] X (today|yesterday|last week|N weeks ago)"
    re.compile(
        r"\bI\s+"
        r"(?:got|bought|attended|completed|finished|signed|started|tried|"
        r"made|visited|saw|cooked|baked|ran|biked|read|watched)\s+"
        r"([^.!?\n]{4,200}?"
        r"(?:today|yesterday|last (?:week|month|night|weekend)|"
        r"(?:two|three|four|several|a few)\s+(?:days?|weeks?|months?)\s+ago|"
        r"a (?:week|month|day)\s+ago))",
        re.IGNORECASE,
    ),
]

# Minimum extracted fact length to be useful for indexing
_MIN_FACT_CHARS = 12
# Maximum fact length — longer than this is probably the rest of the paragraph
_MAX_FACT_CHARS = 280


def extract_incidental_facts(text: str) -> list[str]:
    """Extract incidental personal facts from text.

    Returns a list of short, fact-bearing sentence fragments suitable
    for indexing as standalone documents. Each fact is a self-contained
    snippet — no overlap deduplication is required because the search
    engine deduplicates results by parent source.

    Args:
        text: Raw conversational text (typically a session transcript).

    Returns:
        List of extracted facts, each a string. Empty list if no facts found.
    """
    if not text:
        return []

    seen: set[str] = set()
    facts: list[str] = []

    for pattern in _INCIDENTAL_PATTERNS:
        for m in pattern.finditer(text):
            # Use full match so the marker (e.g. "by the way") stays in
            # the indexed text — helpful for queries that mention it.
            fact = m.group(0).strip()
            # Strip trailing punctuation/whitespace
            fact = fact.rstrip(",;:")
            if len(fact) < _MIN_FACT_CHARS or len(fact) > _MAX_FACT_CHARS:
                continue
            key = fact.lower()
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)

    return facts
