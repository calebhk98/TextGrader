"""Paragraph-level rhythm: the questions the sentence-rhythm modules ask, one
level up.

Sentence rhythm can look varied while paragraphs are metronomic, or the other
way around: a novel can alternate long and short sentences freely inside
paragraphs that are all, without exception, four sentences long.

Paragraph length is therefore measured two ways, and each gets the same
treatment rather than one being the headline and the other an afterthought:

``words``
    sheer paragraph size.  This is what a reader sees as a block on the page,
    and what changes when sentences get longer without being regrouped.
``sentences``
    how the prose is chunked, independent of how long the sentences are.  A
    book that always writes four-sentence paragraphs has a flat sentence-count
    series even when its word counts vary widely.

Both publish their full distribution (quantiles, dispersion, entropy, lag-1
autocorrelation, bimodality) plus a first-class coefficient of variation and
lag-1 autocorrelation, and both get named buckets, so "how many paragraphs are
one sentence long" does not have to be reconstructed from quantiles.

Paragraphs are a coarser unit than sentences, so this module's minimum sample
is lower than the sentence-rhythm modules' default.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import histogram, summarize
from .common import FAST, finding

FAMILY = "paragraph_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

# A short beat, a normal paragraph, a long one, and a page-length block, which
# is where paragraphing itself becomes the story rather than an incidental fact.
WORD_HISTOGRAM_EDGES = (20, 50, 100, 200)
# One sentence, a pair, a normal paragraph, a long one.
SENTENCE_HISTOGRAM_EDGES = (1, 2, 5, 10)

#: ``(key, id_stem, unit, label, headline)`` for the two series, so neither
#: can quietly acquire a measurement the other lacks.  Words per paragraph
#: range widely and keep the median.  Sentences per paragraph are small whole
#: numbers whose median was 2 on 36 of the 50 reference books, so it headlines
#: the mean.
SERIES = (
    ("words", "paragraph_words", "words", "paragraph length in words", "median"),
    ("sentences", "paragraph_sentences", "sentences", "paragraph length in sentences", "mean"),
)

# The word-length autocorrelation shipped under this id before the sentence
# series became first class. It stays, so corpus profiles keep comparing.
LEGACY_WORD_AUTOCORRELATION = "rhythm.paragraph_length_autocorrelation_lag1"


def _series_findings(stem: str, unit: str, label: str, values: Sequence[int],
                     edges: Sequence[int], headline: str = "median") -> list[dict[str, Any]]:
    summary = summarize(values)
    total = len(values)
    buckets = histogram(values, list(edges))
    autocorrelation = summary.get("lag1_autocorrelation")
    return [
        finding(f"rhythm.{stem}", label.capitalize(), summary.get(headline), unit,
                family=FAMILY, sample_size=total, distribution={**summary, "headline": headline},
                min_sample=MIN_SAMPLE,
                details=[{"bucket": name, "count": count} for name, count in buckets.items()]),
        finding(f"rhythm.{stem}_cv", f"Coefficient of variation of {label}",
                summary.get("cv"), "percent", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE),
        finding(f"rhythm.{stem}_autocorrelation_lag1",
                f"Lag-1 autocorrelation of {label}", autocorrelation, "correlation",
                family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                warning=None if autocorrelation is not None else
                "needs more than two paragraphs with varying length"),
    ]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    series = {"words": analysis.paragraph_lengths,
              "sentences": analysis.paragraph_sentence_counts}
    edges = {"words": WORD_HISTOGRAM_EDGES, "sentences": SENTENCE_HISTOGRAM_EDGES}
    total = len(series["words"])
    out: list[dict[str, Any]] = []
    if total == 0:
        warning = "no paragraphs to measure"
        for _, stem, unit, label, _ in SERIES:
            out.extend([
                finding(f"rhythm.{stem}", label.capitalize(), None, unit, family=FAMILY,
                        sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
                finding(f"rhythm.{stem}_cv", f"Coefficient of variation of {label}",
                        None, "percent", family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE, warning=warning),
                finding(f"rhythm.{stem}_autocorrelation_lag1",
                        f"Lag-1 autocorrelation of {label}", None, "correlation",
                        family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                        warning=warning),
            ])
        out.append(finding(LEGACY_WORD_AUTOCORRELATION,
                           "Lag-1 autocorrelation of paragraph length in words (alias)",
                           None, "correlation", family=FAMILY, sample_size=0,
                           min_sample=MIN_SAMPLE, warning=warning))
        return out

    for key, stem, unit, label, headline in SERIES:
        out.extend(_series_findings(stem, unit, label, series[key], edges[key], headline))
    word_autocorrelation = next(
        item["value"] for item in out
        if item["metric_id"] == "rhythm.paragraph_words_autocorrelation_lag1")
    out.append(finding(
        LEGACY_WORD_AUTOCORRELATION,
        "Lag-1 autocorrelation of paragraph length in words (alias)",
        word_autocorrelation, "correlation", family=FAMILY, sample_size=total,
        min_sample=MIN_SAMPLE,
        warning="deprecated alias for rhythm.paragraph_words_autocorrelation_lag1, kept "
                "so existing corpus profiles keep comparing"))
    return out
