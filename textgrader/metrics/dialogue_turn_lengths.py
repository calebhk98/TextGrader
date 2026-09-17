"""The shape of spoken turn length, in words and in sentences.

A mean turn length hides exactly what a scene needs to show: fast one-line
exchanges buried inside a chapter that also contains long speeches average
out to "medium", which describes neither.  This module publishes the
distribution instead, with named buckets on word length (terse, short,
conversational, speech-length) so a reader does not have to reconstruct the
shape from quantiles alone, plus the share of turns that are exactly one
sentence, which is the plainest available signal of clipped, naturalistic
dialogue versus turns that run on.

Sentence counts come from ``analysis.derive(turn, "turn").sentence_count``,
a view over just that turn's text, rather than a private regex: this reuses
the same ``Segmenter`` instance the rest of the pipeline uses (``derive``
passes it through), so a turn's sentence count agrees with every other
sentence-based metric in the report instead of being segmented by a second,
possibly different, rule.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from .. import text as textlib
from ..stats import histogram, summarize
from .common import MODERATE, finding, rate

FAMILY = "dialogue"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

# Terse / short / conversational / speech-length, in words.
BUCKET_EDGES = (4, 10, 30)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    turns = analysis.turns
    if not turns:
        warning = "no spoken turns found"
        return [
            finding("dialogue.turn_words", "Spoken turn length", None, "words", family=FAMILY,
                    sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.turn_sentences", "Spoken turn length", None, "sentences",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.single_sentence_turn_share", "Share of turns that are one sentence",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    word_counts = [len(textlib.words(turn)) for turn in turns]
    sentence_counts = [analysis.derive(turn, "turn").sentence_count for turn in turns]
    single_sentence = sum(1 for count in sentence_counts if count == 1)

    word_summary = summarize(word_counts)
    word_summary["buckets"] = histogram(word_counts, list(BUCKET_EDGES))
    sentence_summary = summarize(sentence_counts)

    return [
        finding("dialogue.turn_words", "Spoken turn length", word_summary.get("median"),
                "words", family=FAMILY, sample_size=len(word_counts), distribution=word_summary,
                min_sample=MIN_SAMPLE, evidence=[{"first_turn_word_counts": word_counts[:20]}]),
        finding("dialogue.turn_sentences", "Spoken turn length", sentence_summary.get("median"),
                "sentences", family=FAMILY, sample_size=len(sentence_counts),
                distribution=sentence_summary, min_sample=MIN_SAMPLE,
                evidence=[{"first_turn_sentence_counts": sentence_counts[:20]}]),
        finding("dialogue.single_sentence_turn_share", "Share of turns that are exactly one sentence",
                rate(single_sentence, len(turns)), "percent", family=FAMILY, sample_size=len(turns),
                min_sample=MIN_SAMPLE),
    ]
