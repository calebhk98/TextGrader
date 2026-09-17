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
    counts = Counter(text)
    return {name: 1000 * counts[mark] / total for mark, name in MARKS.items()}


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    total = analysis.word_count
    if total == 0:
        warning = "no words to measure punctuation rates over"
        return [finding(f"style.punctuation_{name}", f"Punctuation: {name.replace('_', ' ')}",
                        None, "per 1,000 words", family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE, warning=warning)
                for name in MARKS.values()]

    counts = Counter(analysis.text)
    return [finding(f"style.punctuation_{name}", f"Punctuation: {name.replace('_', ' ')}",
                    rate(counts[mark], total, 1000.0), "per 1,000 words", family=FAMILY,
                    sample_size=total, min_sample=MIN_SAMPLE)
            for mark, name in MARKS.items()]
