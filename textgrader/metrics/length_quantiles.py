"""Sentence and paragraph length quantiles, with the full shape at the median.

Five quantiles already say more than a mean does, but a reader who only sees
five numbers still cannot tell a smooth distribution from a bimodal one (a
text that alternates one-line and thirty-word paragraphs can share the same
p10/p50/p90 as one that is evenly midsized throughout).  The p50 finding for
each unit therefore also carries the complete :func:`~textgrader.stats.summarize`
shape (dispersion, entropy, bimodality, two-group split), so the one number a
report is most likely to headline with is also the one that can answer "is
this really one population".
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import quantile, summarize
from .common import FAST, finding

FAMILY = "sentence_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

QUANTILES = (("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90))


def _unit_findings(kind: str, values: Sequence[int]) -> list[dict[str, Any]]:
    total = len(values)
    summary = summarize(values) if total else None
    out = []
    for label, q in QUANTILES:
        out.append(finding(
            f"style.{kind}_words_{label}", f"{kind.title()} length {label}",
            quantile(values, q) if total else None, "words", family=FAMILY,
            sample_size=total, min_sample=MIN_SAMPLE,
            distribution=summary if label == "p50" else None,
            warning=None if total else f"no {kind}s to measure"))
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    out = _unit_findings("sentence", analysis.sentence_lengths)
    out.extend(_unit_findings("paragraph", analysis.paragraph_lengths))
    return out
