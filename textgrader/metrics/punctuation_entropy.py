"""Shannon entropy of the punctuation mix: how much a text leans on one mark.

Two texts can have identical per-1,000-word rates for every mark in
``punctuation_profile`` while one spreads its punctuation across many kinds
and the other reaches for a comma (or a dash) almost every time it needs
anything at all. Entropy over the mark-frequency vector separates them: a low
value means a narrow punctuation vocabulary, regardless of how much
punctuation there is in absolute terms.

Entropy is computed directly from the proportion vector of mark counts
(``-sum(p * log2(p))``) rather than through ``stats.shannon_entropy``, which
is built to bin a numeric *sample* (sentence lengths, distances) rather than
sum a set of category counts that are already a full frequency table. Feeding
mark names through that function would require inventing an arbitrary integer
per category first; going straight from counts to proportions is both simpler
and exact.

A second, more actionable entropy is computed over just the sentence-final
mark: period, question mark, exclamation mark or ellipsis. This is a
different signal from the mark mix as a whole. A novel can have healthy
overall punctuation variety, entirely inside its sentences, while ending
every single sentence on a period; conversely, all-mark entropy can look
modest while sentence endings alternate constantly between "?" and "!".
Sentence-final classification is done by trimming closing quotes/brackets
from the end of each sentence and reading its last character(s), not by
another regex sentence split.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import MODERATE, finding, rate
from .punctuation_profile import MARK_NAMES, marks_for

FAMILY = "punctuation"
# Measured well under a second by itself; declared MODERATE to match the
# registry row, which groups it with the rest of the punctuation family.
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# Characters a sentence may trail off with (a closing quote or bracket) before
# its actual terminal mark. Kept separate from MARK_RE so this module never
# has to re-run a sentence-boundary regex over the whole text.
_TRAILERS = "\"'”’)]"


def _entropy_from_counts(counts: Mapping[str, int]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def _final_mark(sentence: str) -> str | None:
    trimmed = sentence.rstrip(_TRAILERS)
    if trimmed.endswith("...") or trimmed.endswith("…"):
        return "ellipsis"
    if not trimmed:
        return None
    last = trimmed[-1]
    if last == ".":
        return "period"
    if last == "?":
        return "question"
    if last == "!":
        return "exclamation"
    return None


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    if not words_total:
        warning = "no words in text"
        return [
            finding("punct.mark_entropy", "Punctuation-mark entropy", None, "bits",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("punct.mark_entropy_normalized", "Punctuation-mark entropy, normalized",
                    None, "ratio", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("punct.top_mark_share", "Share of punctuation held by the top mark",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("punct.sentence_final_entropy", "Sentence-final mark entropy", None,
                    "bits", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    counts = marks_for(analysis)
    total_marks = sum(counts.values())
    out: list[dict[str, Any]] = []

    entropy = _entropy_from_counts(counts)
    distinct = sum(1 for name in MARK_NAMES if counts.get(name))
    max_entropy = math.log2(distinct) if distinct > 1 else 0.0
    normalized = (entropy / max_entropy) if entropy is not None and max_entropy > 0 else (
        0.0 if entropy is not None else None)
    top_mark, top_count = (counts.most_common(1)[0] if counts else (None, 0))
    top_share = rate(top_count, total_marks, 100.0)

    out.append(finding(
        "punct.mark_entropy", "Punctuation-mark entropy", entropy, "bits", family=FAMILY,
        sample_size=total_marks, min_sample=MIN_SAMPLE,
        distribution={"distinct_marks": distinct, "possible_marks": len(MARK_NAMES)},
        evidence=[{"mark": name, "count": count} for name, count in counts.most_common(10)],
        warning=None if total_marks else "no punctuation marks found"))
    out.append(finding(
        "punct.mark_entropy_normalized", "Punctuation-mark entropy, normalized",
        normalized, "ratio", family=FAMILY, sample_size=total_marks, min_sample=MIN_SAMPLE,
        distribution={"distinct_marks": distinct, "raw_entropy_bits": entropy},
        warning=None if distinct > 1 else
        "fewer than two distinct marks were used; normalization is undefined so 0.0 was used"))
    out.append(finding(
        "punct.top_mark_share", "Share of all punctuation held by the single most common mark",
        top_share, "percent", family=FAMILY, sample_size=total_marks, min_sample=MIN_SAMPLE,
        evidence=[{"mark": top_mark, "count": top_count}] if top_mark else None,
        warning=None if total_marks else "no punctuation marks found"))

    final_counts = Counter(mark for mark in (
        _final_mark(sentence) for sentence in analysis.sentences) if mark is not None)
    final_total = sum(final_counts.values())
    final_entropy = _entropy_from_counts(final_counts)
    out.append(finding(
        "punct.sentence_final_entropy", "Sentence-final mark entropy (., ?, !, ...)",
        final_entropy, "bits", family=FAMILY, sample_size=final_total, min_sample=MIN_SAMPLE,
        distribution={"distinct_final_marks": len(final_counts),
                      "sentences_without_a_recognized_ending": analysis.sentence_count - final_total},
        evidence=[{"mark": name, "count": count} for name, count in final_counts.most_common()],
        warning=None if final_total else
        "no sentence ended in ., ?, ! or ..."))
    return out
