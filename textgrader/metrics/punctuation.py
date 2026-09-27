"""Rate of a fixed set of punctuation marks per 1,000 words.

This module deliberately covers a small, stable set of marks (the same set
it always has: semicolon, colon, parenthesis, ellipsis, exclamation,
question mark, em dash, en dash).  A separate module,
``punctuation_profile.py``, owns full coverage of the punctuation inventory
and its per-sentence framing; this one stays narrow so that ``vector()``, its
externally-imported helper, keeps returning exactly the keys other code
already depends on.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import FAST, finding, rate
from .common import tokens as tokenize

FAMILY = "punctuation"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

MARKS = {
    ";": "semicolon", ":": "colon", "(": "parenthesis", "…": "ellipsis",
    "!": "exclamation", "?": "question", "—": "em_dash", "–": "en_dash",
}

#: The typewriter spellings of two of those marks.  Plain-text editions,
#: Project Gutenberg's included, write an em dash as ``--`` and an ellipsis
#: as ``...``; counting only the Unicode characters read every such book as
#: using neither, so a manuscript with ordinary dash use sat at the 100th
#: percentile of a corpus that used dashes at the same rate.  A run of two or
#: more hyphens is one dash and a run of three or more periods one ellipsis,
#: matching ``punctuation_profile``.
_ASCII_FORMS = {"em_dash": re.compile(r"-{2,}"), "ellipsis": re.compile(r"\.{3,}")}


def mark_counts(text: str) -> dict[str, int]:
    """How many times each mark in :data:`MARKS` occurs, keyed by mark name."""

    counts = Counter(text)
    found = {name: counts[mark] for mark, name in MARKS.items()}
    for name, pattern in _ASCII_FORMS.items():
        found[name] += len(pattern.findall(text))
    return found


def vector(text: str) -> dict[str, float]:
    """Punctuation-mark rate per 1,000 words, keyed by mark name.

    Kept as a standalone function of raw text, not of a
    :class:`~textgrader.document.DocumentAnalysis`, because other modules
    import and call it directly on their own text (a speaker's lines, a
    transcript turn) that never went through the shared pipeline.
    """

    total = len(tokenize(text))
    if not total:
        return {}
    return {name: 1000 * count / total for name, count in mark_counts(text).items()}


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    total = analysis.word_count
    if total == 0:
        warning = "no words to measure punctuation rates over"
        return [finding(f"style.punctuation_{name}", f"Punctuation: {name.replace('_', ' ')}",
                        None, "per 1,000 words", family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE, warning=warning)
                for name in MARKS.values()]

    counts = mark_counts(analysis.text)
    return [finding(f"style.punctuation_{name}", f"Punctuation: {name.replace('_', ' ')}",
                    rate(count, total, 1000.0), "per 1,000 words", family=FAMILY,
                    sample_size=total, min_sample=MIN_SAMPLE)
            for name, count in counts.items()]
