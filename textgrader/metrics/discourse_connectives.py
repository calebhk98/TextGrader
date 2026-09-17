"""Sentences that open on a discourse connective ("However,", "Indeed,", ...).

A connective at the front of a sentence is the plainest signal a writer gives
that this sentence is doing logical work against the last one: contrast,
concession, addition, consequence.  Overusing the device reads as an essay
wearing a narrator's voice; ignoring it entirely can read as sentences that
never talk to each other.  Neither is measured here as good or bad, only as a
rate and a mix.

The vocabulary below is a closed-class list curated once from general English
usage.  It is a fixed dictionary this module checks a text against, not a
claim about what any particular manuscript does or should do; a text that
never opens on "Nevertheless" has not necessarily failed to transition, and a
text that leans on "However" has not necessarily overused it until the rate
and the entropy below are read together.

Detection matches only the literal opening word(s) of a sentence.  It cannot
always tell a connective use from a different grammatical role at the same
position ("Instead of leaving, she stayed" is not the connective "Instead,");
that ambiguity is not resolved here, only accepted.

Reported in narration as well as overall, because dialogue supplies its own
sentence openings (a character saying "However..." is a speech habit, not the
narrator's connective use) and folding the two together would blur which one
is happening.

Measured on a 400,000-word novel, this module's own counting logic costs
about 0.3s once sentence segmentation is cached. If this is the first metric
in a run to touch ``analysis.narration.sentences``, that first touch pays the
shared, one-time cost of re-segmenting the narration text (about 4-5s with
pysbd on that same novel); every later metric that reuses
``analysis.narration`` gets it for free. That cost lives in
:mod:`textgrader.document`, not here, and is paid at most once per document.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from .. import text as textlib
from ..document import DocumentAnalysis
from .common import FAST, finding, rate

FAMILY = "discourse"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# Canonical spelling -> lower-cased word sequence matched at the sentence start.
# Multi-word entries ("In fact") match only that exact opening pair of words.
CONNECTIVES: dict[str, tuple[str, ...]] = {
    "However": ("however",),
    "Indeed": ("indeed",),
    "Moreover": ("moreover",),
    "Furthermore": ("furthermore",),
    "Ultimately": ("ultimately",),
    "Nevertheless": ("nevertheless",),
    "Nonetheless": ("nonetheless",),
    "Additionally": ("additionally",),
    "Consequently": ("consequently",),
    "Therefore": ("therefore",),
    "Thus": ("thus",),
    "Hence": ("hence",),
    "Meanwhile": ("meanwhile",),
    "Instead": ("instead",),
    "Overall": ("overall",),
    "Similarly": ("similarly",),
    "Conversely": ("conversely",),
    "Regardless": ("regardless",),
    "In fact": ("in", "fact"),
    "In addition": ("in", "addition"),
    "That said": ("that", "said"),
}


def _connective_counts(view: DocumentAnalysis) -> tuple[Counter, int]:
    """Count, once per sentence, which connective it opens on, if any."""

    counts: Counter[str] = Counter()
    total = 0
    for sentence in view.sentences:
        total += 1
        lowered = [word.lower() for word in textlib.words(sentence)]
        if not lowered:
            continue
        for name, phrase in CONNECTIVES.items():
            if lowered[:len(phrase)] == list(phrase):
                counts[name] += 1
                break
    return counts, total


def _entropy(counts: Mapping[str, int]) -> float | None:
    """Shannon entropy, in bits, of a categorical count distribution."""

    total = sum(counts.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def _evidence(counts: Counter, total: int, limit: int = 25) -> list[dict[str, Any]]:
    return [{"connective": name, "count": count,
             "rate_per_100_sentences": rate(count, total, 100.0)}
            for name, count in counts.most_common(limit)]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    counts, total_sentences = _connective_counts(analysis)
    matched = sum(counts.values())
    out = [finding(
        "discourse.sentence_initial_connective_rate",
        "Sentences opening on a discourse connective",
        rate(matched, total_sentences, 100.0), "per 100 sentences", family=FAMILY,
        sample_size=total_sentences, min_sample=MIN_SAMPLE,
        evidence=_evidence(counts, total_sentences),
        warning=None if total_sentences else "no sentences to measure")]

    narration = analysis.narration
    n_counts, n_total = _connective_counts(narration)
    n_matched = sum(n_counts.values())
    out.append(finding(
        "discourse.sentence_initial_connective_rate_narration",
        "Sentences opening on a discourse connective (narration only)",
        rate(n_matched, n_total, 100.0), "per 100 sentences", family=FAMILY,
        sample_size=n_total, channel="narration", min_sample=MIN_SAMPLE,
        evidence=_evidence(n_counts, n_total),
        warning=None if n_total else "no narration sentences to measure"))

    entropy = _entropy(counts)
    out.append(finding(
        "discourse.connective_entropy",
        "Entropy of the sentence-initial connective mix (low = one connective dominates)",
        entropy, "bits", family=FAMILY, sample_size=matched, min_sample=10,
        distribution={"distinct_connectives": len(counts), "matched_sentences": matched,
                      "max_possible_bits": math.log2(len(counts)) if len(counts) > 1 else 0.0},
        evidence=_evidence(counts, total_sentences),
        warning=None if matched else "no sentence-initial connectives found"))
    return out
