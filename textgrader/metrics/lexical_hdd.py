"""HD-D: expected vocabulary diversity in a small random draw from this text.

HD-D (McCarthy & Jarvis 2007's implementation of Hypergeometric Distribution
Diversity, the vocd-D idea done exactly rather than by curve-fitting) asks a
concrete question: if you drew a random sample of ``n`` tokens from this text,
how many distinct types would you expect to see?  Divided by ``n``, that
expectation is length-resistant in the same spirit as MTLD but arrived at
differently, so the two are worth having side by side rather than as
alternates.

For a type that occurs ``count`` times in an ``N``-token text, the chance it is
completely absent from a random ``n``-draw (without replacement) is the
hypergeometric ``C(N - count, n) / C(N, n)``; one minus that is the chance it
contributes to the draw, and HD-D sums ``(1 - absence probability) / n`` over
every type. This module computes that directly with ``math.comb`` rather than
simulating draws, so the result is exact, not a Monte Carlo estimate.

``lexicalrichness`` is preferred when it imports; here it does not (see
``REQUIRES``), so a dependency-free implementation of the same formula is used
instead, and the finding says which one ran.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..optional import require
from .common import MODERATE, finding, option

FAMILY = "lexical"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("lexicalrichness",)
MIN_SAMPLE = 50
UNIT_SENSITIVE = False


def _hdd_fallback(tokens: list[str], n: int) -> float | None:
    total = len(tokens)
    if total < n:
        return None
    counts = Counter(tokens)
    # Many distinct words share the same occurrence count in a novel-length
    # text (thousands of hapax legomena, for instance), and the absence
    # probability depends only on that count, not on which word it is. Grouping
    # by count turns "one comb() per distinct word" into "one comb() per
    # distinct count", which is what keeps this MODERATE rather than a real
    # bottleneck on a 400,000-word file.
    types_by_count = Counter(counts.values())
    denominator = math.comb(total, n)
    hdd = 0.0
    for count, num_types in types_by_count.items():
        if total - count < n:
            absence = 0.0  # too few remaining tokens to ever miss this type
        else:
            absence = math.comb(total - count, n) / denominator
        hdd += num_types * (1.0 - absence) / n
    return hdd


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    sample = int(option(config, "sample", 42))
    tokens = analysis.tokens
    if len(tokens) < sample or sample < 1:
        return [finding(
            "lexical.hdd", "HD-D (lexical diversity)", None, "types per draw",
            family=FAMILY, sample_size=len(tokens), min_sample=MIN_SAMPLE,
            warning=(f"needs at least the sample size ({sample}) in tokens; "
                    f"this text has {len(tokens)}"))]

    module, reason = require("lexicalrichness")
    value: float | None = None
    note: str
    if module is not None:
        try:
            lex = module.LexicalRichness(" ".join(analysis.words))
            value = float(lex.hdd(draws=sample))
            note = "value computed by the lexicalrichness package"
        except Exception as exc:  # pragma: no cover - third-party failure mode
            reason = f"lexicalrichness.hdd raised {type(exc).__name__}: {exc}"
            module = None
    if module is None:
        value = _hdd_fallback(tokens, sample)
        note = (f"lexicalrichness unavailable ({reason}); value computed by this module's "
                f"dependency-free hypergeometric HD-D implementation")

    return [finding(
        "lexical.hdd", "HD-D (lexical diversity)", value, "types per draw",
        family=FAMILY, sample_size=len(tokens), min_sample=MIN_SAMPLE, warning=note)]
