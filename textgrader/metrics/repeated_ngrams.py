"""Repeated word sequences, scored by how much they repeat, not how many types.

The legacy version counted repeated n-gram *types*: a three-word phrase used
twice and one used twenty times both counted as exactly one hit, so a novel
with one habitual tic ("she couldn't help but") scored the same as one where
every third paragraph opens on it.  That is a measurement of vocabulary, not
of repetition.

This module keeps the type count (``style.repeated_ngrams``) for corpus-profile
compatibility, since existing profiles are keyed on it, but the headline
severity measures are now about occurrences:

``style.repeated_ngram_excess_rate``
    total *excess* occurrences (each repeat beyond the first one counts),
    normalized per 10,000 tokens so a 400,000-word novel is not penalized
    just for having more tokens than a 5,000-word chapter.
``style.repeated_ngram_max_count``
    how often the single worst-offending n-gram recurs, also per 10,000
    tokens, so a phrase repeated twenty times in a short passage is
    distinguishable from the same twenty repeats spread across a whole book.
``style.repeated_ngram_token_share``
    the share of all tokens that fall inside *some* repeated n-gram's span,
    which is what a reader actually notices: how much of the text is made of
    reused material, independent of how many distinct phrases that comes from.
``style.repeated_ngram_gap``
    the distribution of token distances between one occurrence of a repeated
    n-gram and its next occurrence.  A phrase that repeats every few
    sentences reads very differently from one that repeats twice, ten
    thousand words apart, even at the same total count.

Finding every repeated n-gram is a single pass per size: build ``{gram:
[positions]}`` with a dict, which is O(tokens) per configured size and never
compares one window against another directly, so the whole scan stays linear
in the token count even though the "positions" step below marks up to
``n`` characters per repeat (n is a small constant, at most 6 by default).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import MODERATE, finding, option, rate

FAMILY = "repetition"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 1000
UNIT_SENSITIVE = False

DEFAULT_SIZES = (3, 4, 5, 6)
DEFAULT_MIN_COUNT = 2
DEFAULT_MAX_REPORTED = 40

_METRIC_NAMES = {
    "style.repeated_ngrams": "Repeated n-gram types",
    "style.repeated_ngram_excess_rate": "Repeated n-gram excess rate",
    "style.repeated_ngram_max_count": "Most-repeated n-gram rate",
    "style.repeated_ngram_token_share": "Token share covered by a repeated n-gram",
    "style.repeated_ngram_gap": "Gap between repeats of the same n-gram",
}


def _repeated(tokens: Sequence[str], sizes: Sequence[int],
              min_count: int) -> list[tuple[int, tuple[str, ...], list[int]]]:
    """``(n, gram, positions)`` for every n-gram occurring at least ``min_count``
    times, for each requested size.  One dict pass per size: O(tokens)."""

    rows: list[tuple[int, tuple[str, ...], list[int]]] = []
    for n in sizes:
        if n <= 0 or n > len(tokens):
            continue
        seen: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in range(len(tokens) - n + 1):
            seen[tuple(tokens[i:i + n])].append(i)
        for gram, positions in seen.items():
            if len(positions) >= min_count:
                rows.append((n, gram, positions))
    return rows


def _empty(total: int, warning: str) -> list[dict[str, Any]]:
    return [finding(metric_id, name, None, None, family=FAMILY, sample_size=total,
                    min_sample=MIN_SAMPLE, warning=warning)
            for metric_id, name in _METRIC_NAMES.items()]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    tokens = analysis.tokens
    total = len(tokens)
    sizes = sorted({int(n) for n in option(config, "sizes", list(DEFAULT_SIZES)) if int(n) > 0})
    min_count = max(2, int(option(config, "min_count", DEFAULT_MIN_COUNT)))
    max_reported = max(1, int(option(config, "max_reported", DEFAULT_MAX_REPORTED)))

    if total == 0 or not sizes:
        return _empty(total, "no tokens to search for repeated n-grams")

    rows = _repeated(tokens, sizes, min_count)
    if not rows:
        out = _empty(total, "no n-gram repeated at least min_count times")
        out[0] = finding("style.repeated_ngrams", _METRIC_NAMES["style.repeated_ngrams"], 0,
                         "types", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                         unit_sensitive=True)
        return out

    counts = [len(positions) for _, _, positions in rows]
    type_count = len(rows)
    total_excess = sum(count - 1 for count in counts)
    worst_index = max(range(len(rows)), key=lambda i: counts[i])
    max_count = counts[worst_index]

    covered = bytearray(total)
    gaps: list[int] = []
    for n, _, positions in rows:
        for position in positions:
            end = min(position + n, total)
            for offset in range(position, end):
                covered[offset] = 1
        for a, b in zip(positions, positions[1:]):
            gaps.append(max(0, b - a - n))

    token_share = rate(sum(covered), total)
    excess_rate = 10_000 * total_excess / total
    max_count_rate = 10_000 * max_count / total

    rows_sorted = sorted(zip(rows, counts), key=lambda item: -item[1])[:max_reported]
    evidence = [{"n": n, "text": " ".join(gram), "count": count,
                "first_position": positions[0]}
                for (n, gram, positions), count in rows_sorted]

    gap_summary = summarize(gaps) if gaps else {"count": 0}
    worst_n, worst_gram, worst_positions = rows[worst_index]

    return [
        finding("style.repeated_ngrams", _METRIC_NAMES["style.repeated_ngrams"], type_count,
                "types", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                unit_sensitive=True, evidence=evidence),
        finding("style.repeated_ngram_excess_rate", _METRIC_NAMES["style.repeated_ngram_excess_rate"],
                excess_rate, "excess occurrences per 10,000 tokens", family=FAMILY,
                sample_size=total, min_sample=MIN_SAMPLE,
                distribution={"total_excess": total_excess, "type_count": type_count}),
        finding("style.repeated_ngram_max_count", _METRIC_NAMES["style.repeated_ngram_max_count"],
                max_count_rate, "occurrences per 10,000 tokens", family=FAMILY,
                sample_size=total, min_sample=MIN_SAMPLE,
                distribution={"raw_count": max_count, "n": worst_n},
                evidence=[{"n": worst_n, "text": " ".join(worst_gram), "count": max_count,
                          "positions": worst_positions[:20]}]),
        finding("style.repeated_ngram_token_share", _METRIC_NAMES["style.repeated_ngram_token_share"],
                token_share, "percent", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE),
        finding("style.repeated_ngram_gap", _METRIC_NAMES["style.repeated_ngram_gap"],
                gap_summary.get("median"), "tokens", family=FAMILY,
                unit_sensitive=True,
                sample_size=gap_summary.get("count", 0), distribution=gap_summary,
                min_sample=MIN_SAMPLE, evidence=[{"first_gaps": gaps[:20]}] if gaps else None,
                warning=None if gaps else "every repeated n-gram occurred exactly once, no gaps to measure"),
    ]
