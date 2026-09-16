"""Paragraph-level rhythm: the same questions :mod:`rhythm_autocorrelation`
and the sentence-level rhythm modules ask, one level up.

Sentence rhythm can look varied while paragraphs are metronomic (or the other
way around): a novel can alternate long and short sentences freely inside
paragraphs that are all, without exception, four sentences long.  Paragraph
length is measured two ways because they catch different failures - word
count reflects sheer paragraph size, sentence count reflects how paragraphs
are chunked regardless of sentence length - and both get the full
distribution shape (quantiles, dispersion, lag-1 autocorrelation) rather than
a mean, plus named buckets for the word-length histogram so a reader does not
have to reconstruct "how many paragraphs are short" from quantiles alone.

Paragraphs are a coarser unit than sentences, so the minimum sample this
module accepts is lower than the sentence-rhythm modules' default.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import histogram, summarize
from .common import FAST, finding

FAMILY = "paragraph_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

# Buckets for paragraph word length: a short beat, a normal paragraph, a long
# one, and a page-length block, which is where paragraphing itself becomes
# the story rather than an incidental fact about it.
WORD_HISTOGRAM_EDGES = (20, 50, 100, 200)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    word_lengths = analysis.paragraph_lengths
    sentence_counts = analysis.paragraph_sentence_counts
    total = len(word_lengths)
    if total == 0:
        warning = "no paragraphs to measure"
        return [
            finding("rhythm.paragraph_words", "Paragraph length in words",
                    None, "words", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.paragraph_sentences", "Paragraph length in sentences",
                    None, "sentences", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.paragraph_words_cv", "Coefficient of variation of paragraph word length",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.paragraph_length_autocorrelation_lag1",
                    "Lag-1 autocorrelation of paragraph word length",
                    None, "correlation", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    word_summary = summarize(word_lengths)
    sentence_summary = summarize(sentence_counts)
    buckets = histogram(word_lengths, list(WORD_HISTOGRAM_EDGES))

    out = [
        finding("rhythm.paragraph_words", "Paragraph length in words",
                word_summary.get("median"), "words", family=FAMILY, sample_size=total,
                distribution=word_summary, min_sample=MIN_SAMPLE,
                details=[{"bucket": name, "count": count} for name, count in buckets.items()]),
        finding("rhythm.paragraph_sentences", "Paragraph length in sentences",
                sentence_summary.get("median"), "sentences", family=FAMILY, sample_size=total,
                distribution=sentence_summary, min_sample=MIN_SAMPLE),
        finding("rhythm.paragraph_words_cv", "Coefficient of variation of paragraph word length",
                word_summary.get("cv"), "percent", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE),
        finding("rhythm.paragraph_length_autocorrelation_lag1",
                "Lag-1 autocorrelation of paragraph word length",
                word_summary.get("lag1_autocorrelation"), "correlation", family=FAMILY,
                sample_size=total, min_sample=MIN_SAMPLE,
                warning=None if word_summary.get("lag1_autocorrelation") is not None else
                "needs more than two paragraphs with varying length"),
    ]
    return out
