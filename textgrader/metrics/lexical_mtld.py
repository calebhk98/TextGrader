"""MTLD: how many words go by before the vocabulary "resets".

Type-token ratio collapses toward zero as a text gets longer, which makes it
useless for comparing a 500-word scene to a 50,000-word chapter.  MTLD (Measure
of Textual Lexical Diversity, McCarthy & Jarvis 2010) fixes that by walking the
token stream and counting how many words it takes, on average, for the running
TTR to fall to a fixed threshold; that "factor length" does not shrink as more
text is added; it just keeps recurring.  The final partial factor is scored by
how far its TTR got toward the threshold rather than thrown away, and the whole
walk is run forward and backward and averaged, because a single direction is
biased by whatever happens to open or close the document.

This is a diversity proxy, not a vocabulary size or a style fingerprint: two
texts with identical MTLD can differ completely in which words they reuse. It
also says nothing about whether reuse is a deliberate motif or a filler word;
:mod:`lexical_repetition_distance` and :mod:`lexical_lemma_repetition` answer
that question by *where* words recur, not just how often the vocabulary turns
over.

``lexicalrichness`` is the reference implementation and is preferred when it
imports cleanly.  In this environment it does not (see ``REQUIRES``), so this
module also carries a dependency-free implementation of the same bidirectional
algorithm; either way the finding's ``warning`` field says which one produced
the number, because "MTLD 62.3" is not the same claim from two different code
paths.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..optional import require
from .common import MODERATE, finding, option

FAMILY = "lexical"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("lexicalrichness",)
MIN_SAMPLE = 50
UNIT_SENSITIVE = False


def _factor_pass(tokens: Sequence[str], threshold: float) -> float | None:
    """One directional MTLD walk; see the module docstring for the algorithm.

    Returns ``None`` when the walk never completes a single factor, which
    means the running TTR never fell to ``threshold`` (the vocabulary is at or
    near ceiling for however many tokens there are) and MTLD is undefined in
    that direction rather than merely large.
    """

    factors = 0.0
    types: set[str] = set()
    count = 0
    ttr = 1.0
    for token in tokens:
        types.add(token)
        count += 1
        ttr = len(types) / count
        if ttr <= threshold:
            factors += 1.0
            types, count, ttr = set(), 0, 1.0
    if count:
        # The trailing partial factor counts for however close it came to the
        # threshold, so a document that ends mid-factor is not just discarded.
        factors += (1.0 - ttr) / (1.0 - threshold)
    if factors <= 0:
        return None
    return len(tokens) / factors


def _mtld_fallback(tokens: Sequence[str], threshold: float) -> float | None:
    forward = _factor_pass(tokens, threshold)
    backward = _factor_pass(list(reversed(tokens)), threshold)
    if forward is None or backward is None:
        return None
    return (forward + backward) / 2.0


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    threshold = float(option(config, "threshold", 0.72))
    if not 0.0 < threshold < 1.0:
        threshold = 0.72  # an out-of-range config value would divide by zero below

    tokens = analysis.tokens
    if len(tokens) < 2:
        return [finding(
            "lexical.mtld", "MTLD (lexical diversity)", None, "words/factor",
            family=FAMILY, sample_size=len(tokens), min_sample=MIN_SAMPLE,
            warning=f"needs at least a couple dozen words; this text has {len(tokens)}")]

    module, reason = require("lexicalrichness")
    value: float | None = None
    note: str
    if module is not None:
        try:
            lex = module.LexicalRichness(" ".join(analysis.words))
            value = float(lex.mtld(threshold=threshold))
            note = "value computed by the lexicalrichness package"
        except Exception as exc:  # pragma: no cover - third-party failure mode
            reason = f"lexicalrichness.mtld raised {type(exc).__name__}: {exc}"
            module = None
    if module is None:
        value = _mtld_fallback(tokens, threshold)
        note = (f"lexicalrichness unavailable ({reason}); value computed by this module's "
                f"dependency-free bidirectional MTLD implementation")
    if value is None:
        note = (f"{note}; MTLD is undefined because the type-token ratio never fell to the "
                f"{threshold:g} threshold in either direction ({len(tokens)} tokens, "
                f"vocabulary too diverse for this sample size)")

    return [finding(
        "lexical.mtld", "MTLD (lexical diversity)", value, "words/factor",
        family=FAMILY, sample_size=len(tokens), min_sample=MIN_SAMPLE, warning=note)]
