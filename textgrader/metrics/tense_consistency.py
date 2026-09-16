"""How often narration changes tense from one sentence to the next.

The legacy version reported a raw count of sentence-to-sentence tense
changes.  A raw count grows with the document: a 100,000-word novel
necessarily contains more sentence boundaries than a 5,000-word chapter, so it
racks up more "drift" for being longer, independent of whether it is actually
less tense-consistent.  ``nlp.tense_drift`` is kept as that count, for
compatibility, but ``nlp.tense_drift_rate`` (changes per 100 sentences) is
the number that can be compared across documents of different length.

The count and rate are also computed on narration alone
(``nlp.narration_tense_drift_rate``), because dialogue legitimately carries
present tense inside a past-tense novel ("Are you coming?" she asked) and
that is not the narrator's tense drifting; folding dialogue in inflates a
narrator's real consistency problems with an artifact of what characters say.

Each sentence's tense is its most common verb/auxiliary ``Tense`` morphology
feature; a sentence with no tensed verb (a fragment, an exclamation) has no
tense and is excluded from both the transition count and the "dominant
tense" share, rather than being forced into a made-up class.

Narration sentences are picked out of the ONE shared parse via
``analysis.spacy_sents_by_channel()`` rather than by parsing
``analysis.narration`` separately: parsing is by a wide margin the most
expensive step in the tool, so doubling it for one extra rate would cost far
more than the number is worth.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, rate, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

_METRIC_NAMES = {
    "nlp.tense_drift": "Sentence-to-sentence tense changes",
    "nlp.tense_drift_rate": "Tense drift rate",
    "nlp.narration_tense_drift_rate": "Narration tense drift rate",
    "nlp.dominant_tense": "Dominant tense",
    "nlp.dominant_tense_share": "Share of sentences carrying the dominant tense",
}


def _tense_label(sent: Any) -> str | None:
    tenses = Counter(
        verb.morph.get("Tense")[0] for verb in sent
        if verb.pos_ in ("VERB", "AUX") and verb.morph.get("Tense"))
    return tenses.most_common(1)[0][0] if tenses else None


def _labels_by_channel(analysis: DocumentAnalysis) -> tuple[list[str | None], list[str | None]]:
    """``(all_sentence_labels, narration_sentence_labels)`` from one shared parse."""

    full: list[str | None] = []
    narration: list[str | None] = []
    for sent, channel, _ in analysis.spacy_sents_by_channel():
        label = _tense_label(sent)
        full.append(label)
        if channel == "narration":
            narration.append(label)
    return full, narration


def _drift(labels: Iterable[str | None]) -> tuple[int, int]:
    """``(changes, eligible_pairs)`` over consecutive sentences that both
    carried a tense; a sentence with no tensed verb cannot evidence a change
    either way and is skipped rather than snapping to a neighbour's tense."""

    evidenced = [label for label in labels if label is not None]
    changes = sum(1 for a, b in zip(evidenced, evidenced[1:]) if a != b)
    return changes, len(evidenced)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, name, analysis.nlp_unavailable, family=FAMILY)
                for metric_id, name in _METRIC_NAMES.items()]

    labels, narration_labels = _labels_by_channel(analysis)
    total_sentences = len(labels)
    changes, evidenced = _drift(labels)
    narration_changes, narration_evidenced = _drift(narration_labels)
    narration_total = len(narration_labels)

    tense_counts = Counter(label for label in labels if label is not None)
    dominant_tense, dominant_count = (tense_counts.most_common(1)[0]
                                      if tense_counts else (None, 0))

    no_data = "no sentence carried a tensed verb or auxiliary"

    out = [
        finding("nlp.tense_drift", _METRIC_NAMES["nlp.tense_drift"], changes, "changes",
                family=FAMILY, sample_size=total_sentences, min_sample=MIN_SAMPLE,
                unit_sensitive=True,
                distribution={"eligible_pairs": evidenced, "total_sentences": total_sentences},
                warning=None if evidenced >= 2 else no_data),
        finding("nlp.tense_drift_rate", _METRIC_NAMES["nlp.tense_drift_rate"],
                rate(changes, total_sentences) if total_sentences else None,
                "changes per 100 sentences", family=FAMILY, sample_size=total_sentences,
                min_sample=MIN_SAMPLE, warning=None if evidenced >= 2 else no_data),
        finding("nlp.narration_tense_drift_rate", _METRIC_NAMES["nlp.narration_tense_drift_rate"],
                rate(narration_changes, narration_total) if narration_total else None,
                "changes per 100 narration sentences", family=FAMILY, channel="narration",
                sample_size=narration_total, min_sample=MIN_SAMPLE,
                warning=None if narration_evidenced >= 2 else
                "no narration sentence carried a tensed verb or auxiliary"),
        finding("nlp.dominant_tense", _METRIC_NAMES["nlp.dominant_tense"], dominant_tense, None,
                family=FAMILY, sample_size=evidenced,
                min_sample=MIN_SAMPLE, warning=None if dominant_tense else no_data),
        finding("nlp.dominant_tense_share", _METRIC_NAMES["nlp.dominant_tense_share"],
                rate(dominant_count, evidenced) if evidenced else None, "%", family=FAMILY,
                sample_size=evidenced, min_sample=MIN_SAMPLE,
                distribution={"dominant_tense": dominant_tense, "tense_counts": dict(tense_counts)},
                warning=None if dominant_tense else no_data),
    ]
    return out
