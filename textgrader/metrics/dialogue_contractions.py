"""How often spoken dialogue uses a contraction, not just an apostrophe.

The legacy version counted any token containing ``'`` or ``'`` as a
contraction, which makes a possessive like "John's coat" indistinguishable
from a contraction like "John's leaving".  That is not a small error: a
manuscript full of possessive nouns in dialogue (character names, objects)
would report a contraction rate that has almost nothing to do with how
colloquial the speech actually sounds.

This module instead recognizes contractions by their closed set of suffixes
(``n't``, ``'ll``, ``'re``, ``'ve``, ``'d``, ``'m``, which between them cover
every regular English contraction) plus a curated list of whole-word
contractions that do not fit that pattern (``can't`` also matches the suffix
list, but irregular ones like ``won't``, ``shan't``, ``ain't``, ``let's``,
``o'clock`` need to be named explicitly).

Bare ``'s`` is deliberately excluded from the contraction count.  It is
genuinely ambiguous without a parse: "it's raining" is a contraction, "the
dog's bowl" is a possessive, and a suffix rule cannot tell them apart.
Guessing would misclassify a large, silent share of dialogue, so that mass is
instead reported separately as ``style.dialogue_ambiguous_s_rate``, with a
warning restating why it could not be resolved either way.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from .common import FAST, finding, rate

FAMILY = "dialogue"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 100
UNIT_SENSITIVE = False

# Every regular English contraction ends in one of these; matching the suffix
# is enough because the apostrophe only appears in this position for a
# contraction, never for a possessive or a plural.
CONTRACTION_SUFFIXES = ("n't", "'ll", "'re", "'ve", "'d", "'m")

# Contractions that do not reduce to one of the suffixes above and would
# otherwise be missed (or, for "can't"/"won't", are worth naming explicitly
# since their stem is irregular).
WHOLE_WORD_CONTRACTIONS = frozenset({
    "can't", "won't", "shan't", "ain't", "let's", "o'clock",
    "y'all", "ma'am", "'tis", "'twas", "'twill", "g'day", "d'you", "d'ye",
})

AMBIGUOUS_SUFFIX = "'s"

BARE_S_WARNING = (
    "Bare 's tokens are excluded from the contraction count because they are "
    "ambiguous between a copula contraction (\"it's raining\") and a "
    "possessive (\"the dog's bowl\") without a parse; see "
    "style.dialogue_ambiguous_s_rate for that excluded share."
)


def _classify(token: str) -> str:
    if token in WHOLE_WORD_CONTRACTIONS or token.endswith(CONTRACTION_SUFFIXES):
        return "contraction"
    if token.endswith(AMBIGUOUS_SUFFIX):
        return "ambiguous_s"
    return "other"


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    dialogue = analysis.dialogue
    tokens: Sequence[str] = dialogue.tokens
    total = len(tokens)
    dialogue_warning = "; ".join(analysis.dialogue_split.warnings) or None

    if total == 0:
        warning = "no dialogue found in this text" + (f"; {dialogue_warning}" if dialogue_warning else "")
        return [
            finding("style.dialogue_contraction_rate", "Dialogue contraction rate", None, "%",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, channel="dialogue",
                    warning=warning),
            finding("style.dialogue_ambiguous_s_rate", "Ambiguous bare 's rate in dialogue",
                    None, "%", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    channel="dialogue", warning=warning),
        ]

    counts = {"contraction": 0, "ambiguous_s": 0, "other": 0}
    examples: dict[str, list[str]] = {"contraction": [], "ambiguous_s": []}
    for token in tokens:
        label = _classify(token)
        counts[label] += 1
        if label in examples and len(examples[label]) < 15:
            examples[label].append(token)

    contraction_warning = dialogue_warning
    ambiguous_warning = BARE_S_WARNING + (f" {dialogue_warning}" if dialogue_warning else "")

    return [
        finding("style.dialogue_contraction_rate", "Dialogue contraction rate",
                rate(counts["contraction"], total), "%", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE, channel="dialogue",
                evidence=[{"token": token} for token in examples["contraction"]],
                warning=contraction_warning),
        finding("style.dialogue_ambiguous_s_rate", "Ambiguous bare 's rate in dialogue",
                rate(counts["ambiguous_s"], total), "%", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE, channel="dialogue",
                evidence=[{"token": token} for token in examples["ambiguous_s"]],
                warning=ambiguous_warning),
    ]
