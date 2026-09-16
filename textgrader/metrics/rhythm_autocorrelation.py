"""Serial dependence in sentence length.

Mean sentence length says nothing about order.  ``14 14 14 14 14 14`` and
``4 7 13 29 9 22`` share a mean of fourteen and are not the same prose.  The
lag-1 autocorrelation separates them and separates both from a third failure
mode that an average hides completely:

* near zero - successive lengths are independent, which is what unforced prose
  usually looks like;
* strongly positive - long stretches of similar lengths, the "short, short,
  short, short" behaviour, or its long-winded opposite;
* strongly negative - mechanical alternation, a long sentence dutifully
  followed by a short one.

Reported for narration as well as for the whole text, because a scene of
one-line dialogue exchanges produces a rhythm that is a fact about the
dialogue, not about the narrative voice.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import autocorrelation, summarize
from .common import FAST, finding, option

FAMILY = "sentence_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False


def _channel_findings(view: DocumentAnalysis, lags: Sequence[int], label: str,
                      suffix: str) -> list[dict[str, Any]]:
    lengths = view.sentence_lengths
    out: list[dict[str, Any]] = []
    for lag in lags:
        value = autocorrelation(lengths, lag)
        out.append(finding(
            f"rhythm.sentence_length_autocorrelation_lag{lag}{suffix}",
            f"Sentence-length autocorrelation at lag {lag}{label}",
            value, "correlation", family=FAMILY, sample_size=len(lengths),
            channel=view.channel, min_sample=MIN_SAMPLE,
            warning=None if value is not None else
            f"needs more than {lag + 1} sentences; this text has {len(lengths)}"))
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    lags = [int(lag) for lag in option(config, "lags", [1, 2, 3]) if int(lag) >= 1]
    out = _channel_findings(analysis, lags, "", "")
    lengths = analysis.sentence_lengths
    out.append(finding(
        "rhythm.sentence_length_shape", "Sentence length",
        summarize(lengths).get("median"), "words", family=FAMILY,
        sample_size=len(lengths), distribution=summarize(lengths),
        min_sample=MIN_SAMPLE,
        evidence=[{"first_sentences": lengths[:20]}] if lengths else None))
    narration = analysis.narration
    if narration.sentence_count >= MIN_SAMPLE and narration.sentence_count < analysis.sentence_count:
        out.extend(_channel_findings(narration, lags[:1], " (narration only)", "_narration"))
    return out
