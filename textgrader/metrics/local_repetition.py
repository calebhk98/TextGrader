"""How often a content word recurs shortly after itself.

The legacy version chopped the token stream into non-overlapping windows (50,
100, 250 tokens) and counted repeats inside each chunk.  That has an arbitrary
boundary: two occurrences of the same word at positions 49 and 51 land in
different windows and the repeat disappears, purely because of where the
chunk happened to be cut.  A sliding window with a stride (a quarter of the
window width, i.e. 12/25/62 tokens for the defaults) would narrow that gap but
not remove it.

Instead this module computes the one thing a window is trying to
approximate: for every content word, the exact token distance back to its
previous occurrence.  That has no boundary at all, no chunk size to tune, and
it subsumes every window size in a single O(tokens) pass with a
last-seen-position dict, because "was this word seen within the last w
tokens" is just "is its distance to its previous occurrence <= w".  The
configured window sizes are kept, not as separate scans, but as named
thresholds applied to that one distance list, which is why the per-window
rows below are cheap and always agree with each other by construction.

"Content word" means the same thing it did before: not a closed-class
function word, and longer than three characters, so "the", "and", "was" do
not dominate the count as trivially-frequent noise.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import MODERATE, finding, option, rate

FAMILY = "repetition"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 500
UNIT_SENSITIVE = False

DEFAULT_WINDOWS = (50, 100, 250)

COMMON = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "was", "it", "he", "she", "they", "i", "you",
})


def _is_content_word(token: str) -> bool:
    return len(token) > 3 and token not in COMMON


def _reuse_gaps(tokens: Sequence[str]) -> list[tuple[int, str, int | None]]:
    """``(position, word, gap_to_previous_occurrence)`` for every content word.

    ``gap`` is ``None`` on a word's first occurrence, since there is nothing
    to reuse yet.
    """

    last_seen: dict[str, int] = {}
    out: list[tuple[int, str, int | None]] = []
    for index, token in enumerate(tokens):
        if not _is_content_word(token):
            continue
        gap = index - last_seen[token] if token in last_seen else None
        out.append((index, token, gap))
        last_seen[token] = index
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    tokens = analysis.tokens
    total = len(tokens)
    windows = sorted({int(w) for w in option(config, "windows", list(DEFAULT_WINDOWS)) if int(w) > 0})

    if total == 0 or not windows:
        warning = "no tokens to measure local repetition over"
        return [
            finding("style.local_lexical_repetition", "Local lexical repetition", None, "%",
                    family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE, warning=warning),
            finding("style.local_repetition_gap",
                    "Distance to the previous occurrence of a repeated content word",
                    None, "tokens", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    entries = _reuse_gaps(tokens)
    gaps = [gap for _, _, gap in entries if gap is not None]

    per_window: list[dict[str, Any]] = []
    rate_sum = 0.0
    for width in windows:
        repeated = sum(1 for gap in gaps if gap <= width)
        window_rate = rate(repeated, total) or 0.0
        per_window.append({"window": width, "repeat_count": repeated,
                           "repeat_rate_percent": window_rate})
        rate_sum += window_rate
    headline = rate_sum / len(windows)

    gap_summary = summarize(gaps) if gaps else {"count": 0}
    content_total = len(entries)

    return [
        finding("style.local_lexical_repetition", "Local lexical repetition", headline, "%",
                family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE, details=per_window,
                distribution={"content_word_tokens": content_total},
                evidence=[{"window": row["window"], "repeat_count": row["repeat_count"]}
                          for row in per_window],
                warning=None if content_total else
                "no content words (length > 3, not a closed-class word) found"),
        finding("style.local_repetition_gap",
                "Distance to the previous occurrence of a repeated content word",
                gap_summary.get("median"), "tokens", family=FAMILY,
                sample_size=gap_summary.get("count", 0), distribution=gap_summary,
                min_sample=MIN_SAMPLE, evidence=[{"first_gaps": gaps[:20]}] if gaps else None,
                warning=None if gaps else "no content word repeats anywhere in this text"),
    ]
