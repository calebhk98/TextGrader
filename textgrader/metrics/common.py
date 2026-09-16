"""The contract every metric module implements, and the helpers they share.

A metric module exposes module-level metadata and one function::

    FAMILY = "sentence_rhythm"      # which group of style this belongs to
    REQUIRES = ()                   # optional package names it needs, if any
    COST = "fast"                   # fast | moderate | parse
    MIN_SAMPLE = 30                 # units of text below which it says so
    UNIT_SENSITIVE = False          # True if the value scales with text length

    def measure(analysis, config=None, profile=None):
        return [finding(...), ...]

``analysis`` is a :class:`textgrader.document.DocumentAnalysis`: the text has
already been cleaned, tokenized, split into sentences and paragraphs, and had
its dialogue separated from its narration.  A metric must never re-derive any
of those, because the whole point of the shared pipeline is that two numbers in
one report describe the same document.

``COST`` tells the runner and the reader what a metric costs on a book-length
text (a few hundred thousand words):

``fast``
    linear over tokens or sentences; under half a second.
``moderate``
    superlinear or repeated passes; up to a few seconds.
``parse``
    needs the spaCy parse, which is tens of seconds for a novel.  Everything
    marked ``parse`` shares one parse, so enabling five of them costs one.
``model``
    encodes the text with a sentence-embedding model: tens of seconds for a
    novel, shared between the metrics that want the same units.  Without the
    model these still run, on a lexical fallback that is a weaker measurement
    and says so in every finding.

``MIN_SAMPLE`` is the manuscript-side sample-size rule.  A passive-voice rate
from three sentences and one from three thousand do not deserve equal standing,
so a finding whose ``sample_size`` is below the metric's minimum is reported
with ``action="insufficient_data"`` and never becomes a corpus outlier.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

from .. import stats as stats_module
from ..stats import quantile, summarize

FAST, MODERATE, PARSE, MODEL = "fast", "moderate", "parse", "model"


def finding(metric_id: str, name: str, value: Any = None, unit: str | None = None, *,
            family: str | None = None, sample_size: int | None = None,
            distribution: Mapping[str, Any] | None = None,
            evidence: Sequence[Mapping[str, Any]] | None = None,
            details: Sequence[Mapping[str, Any]] | None = None,
            warning: str | None = None, channel: str = "full",
            min_sample: int | None = None,
            unit_sensitive: bool | None = None) -> dict[str, Any]:
    """One measurement, in the shape :mod:`grade` turns into a result."""

    return {
        "metric_id": metric_id, "name": name, "value": value, "unit": unit,
        "family": family, "sample_size": sample_size,
        "distribution": dict(distribution) if distribution else None,
        "evidence": [dict(item) for item in (evidence or [])],
        "details": [dict(item) for item in (details or [])],
        "warning": warning, "channel": channel, "min_sample": min_sample,
        "unit_sensitive": unit_sensitive,
    }


def unavailable(metric_id: str, name: str, reason: str, *, family: str | None = None,
                unit: str | None = None, channel: str = "full") -> dict[str, Any]:
    """A measurement that could not be made, with the reason kept visible."""

    return finding(metric_id, name, None, unit, family=family, warning=reason, channel=channel)


def shape(metric_id: str, name: str, values: Sequence[float], unit: str, *,
          family: str, prefix: str | None = None, channel: str = "full",
          evidence: Sequence[Mapping[str, Any]] | None = None,
          min_sample: int | None = None) -> list[dict[str, Any]]:
    """Publish a sample as its median plus its full shape.

    The headline value is the median rather than the mean, because the mean of
    a bimodal or long-tailed sample describes neither mode.  The distribution
    carries the quantiles, dispersion, entropy, lag-1 autocorrelation and
    two-group split, so a reader that cares about rhythm does not have to guess
    it from one number.
    """

    summary = summarize(values)
    return [finding(metric_id, name, summary.get("median"), unit, family=family,
                    sample_size=summary.get("count"), distribution=summary,
                    evidence=evidence, channel=channel, min_sample=min_sample)]


# -------------------------------------------------------------- small helpers

def rate(count: float, total: float, scale: float = 100.0) -> float | None:
    return scale * count / total if total else None


def rates(counter: Mapping[Any, int], total: float, scale: float = 100.0) -> dict[Any, float]:
    return {key: scale * value / total for key, value in counter.items()} if total else {}


def cosine_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float | None:
    keys = set(a) | set(b)
    dot = sum(a.get(key, 0.0) * b.get(key, 0.0) for key in keys)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return 1 - dot / (norm_a * norm_b) if norm_a and norm_b else None


def top(counter: Counter, limit: int = 20, key: str = "item") -> list[dict[str, Any]]:
    return [{key: item, "count": count} for item, count in counter.most_common(limit)]


def option(config: Mapping[str, Any] | None, name: str, default: Any) -> Any:
    """Read one metric option, ignoring the ``enabled`` switch itself."""

    value = (config or {}).get(name, default)
    return default if value is None else value


# Backwards-compatible names used by the first generation of metric modules.
def tokens(text: str) -> list[str]:
    from ..text import words
    return [word.lower().replace("’", "'") for word in words(text)]


def lengths(items: Iterable[str]) -> list[int]:
    from ..text import words
    return [len(words(item)) for item in items if words(item)]


def result(metric_id, name, value, unit=None, details=None, warning=None, **extra):
    """Legacy finding constructor retained for external metric modules."""

    return finding(metric_id, name, value, unit, details=details, warning=warning, **extra)


__all__ = ["FAST", "MODERATE", "PARSE", "MODEL", "finding", "unavailable", "shape", "rate",
           "rates", "cosine_distance", "top", "option", "tokens", "lengths",
           "result", "quantile", "summarize", "statistics", "stats_module"]
