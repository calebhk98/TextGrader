"""How often a quotation is attributed, and how plainly.

The legacy version searched a fixed +/-100 character window around every
quotation for a speech verb.  That window has no idea where the *next* or
*previous* quotation starts, so a "said" that belongs to a neighbouring line
of dialogue, or to an unrelated sentence three lines above, can get credited
to the wrong quote whenever two quotations sit within 200 characters of each
other, which is the ordinary case in a fast dialogue exchange.

The search region here is bounded by two real things instead of an arbitrary
character count: the adjacent quotation's own span (attribution can only come
from the narration between this quote and its neighbour, never past it into
someone else's) and a paragraph break within that gap (a new paragraph
almost always means a new beat, and prose that puts the tag in a following
paragraph is rare enough that stopping there is the safer default).
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import FAST, finding, rate

FAMILY = "dialogue"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

BASIC_TAGS = frozenset({"said", "asked"})
ELABORATE_TAGS = frozenset({
    "replied", "whispered", "shouted", "murmured", "cried", "called", "answered", "exclaimed",
})
ALL_TAGS = BASIC_TAGS | ELABORATE_TAGS
TAG_RE = re.compile(r"\b(" + "|".join(sorted(ALL_TAGS)) + r")\b", re.IGNORECASE)

SNIPPET_CHARS = 120


def _bounded_context(text: str, start: int, end: int, prev_end: int,
                     next_start: int) -> tuple[str, str]:
    """Narration before and after a quote, cut at the neighbouring quote and
    at the nearest paragraph break inside that gap, whichever comes first."""

    before = text[prev_end:start]
    cut = before.rfind("\n\n")
    if cut != -1:
        before = before[cut + 2:]

    after = text[end:next_start]
    cut = after.find("\n\n")
    if cut != -1:
        after = after[:cut]

    return before, after


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    spans = analysis.quotation_spans
    text = analysis.text
    total = len(spans)

    if total == 0:
        warning = "no quoted dialogue found in this text"
        return [
            finding("style.dialogue_tag_density", "Dialogue tag density", None, "% of quotations",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("style.elaborate_tag_rate", "Elaborate dialogue tags", None, "%",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
        ]

    tags: list[str] = []
    evidence: list[dict[str, Any]] = []
    for index, (start, end, content) in enumerate(spans):
        prev_end = spans[index - 1][1] if index > 0 else 0
        next_start = spans[index + 1][0] if index + 1 < total else len(text)
        before, after = _bounded_context(text, start, end, prev_end, next_start)
        match = TAG_RE.search(before) or TAG_RE.search(after)
        if not match:
            continue
        tag = match.group(1).lower()
        tags.append(tag)
        if len(evidence) < 25:
            evidence.append({"tag": tag, "quote_index": index, "quote": content[:SNIPPET_CHARS]})

    elaborate_count = sum(1 for tag in tags if tag not in BASIC_TAGS)
    elaborate_evidence = [row for row in evidence if row["tag"] not in BASIC_TAGS]

    return [
        finding("style.dialogue_tag_density", "Dialogue tag density",
                rate(len(tags), total), "% of quotations", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE, evidence=evidence),
        finding("style.elaborate_tag_rate", "Elaborate dialogue tags",
                rate(elaborate_count, len(tags)) if tags else None, "%", family=FAMILY,
                sample_size=len(tags), min_sample=MIN_SAMPLE, evidence=elaborate_evidence,
                warning=None if tags else "no dialogue tags were found to classify"),
    ]
