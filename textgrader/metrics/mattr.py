"""Moving-average type-token ratio: lexical diversity that resists length.

A plain type-token ratio falls as a text gets longer, purely because a longer
sample keeps re-using words it has already used; that makes it useless for
comparing a chapter with a novel.  MATTR fixes that by sliding a fixed-size
window across the token stream and averaging the window-by-window ratio
instead, so the number reflects a per-window vocabulary rate rather than a
running total.

The legacy implementation called its own ``value()`` helper twice per run
(once to report the metric, once again inside the same expression to check it
was not ``None``), which doubled the cost of an already ``O(n * window)``
scan that rebuilt a fresh ``set`` at every position.  Both problems are fixed
here: the value is computed once, and the window slides with a running
``Counter`` that adds the incoming token and removes the outgoing one, which
makes the whole pass ``O(tokens)``.

A single mean MATTR score hides exactly the failure this metric exists to
catch: a text that is lexically rich for a while and then flattens into a
repetitive stretch can have the same mean as one that is evenly diverse
throughout.  ``style.mattr_window_shape`` publishes the full distribution of
per-window ratios (quantiles, dispersion, bimodality) so a flat stretch shows
up as a low-percentile tail rather than disappearing into the average.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import MODERATE, finding, option

FAMILY = "lexical"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

DEFAULT_WINDOW = 100


def _window_ttrs(tokens: Sequence[str], window: int) -> list[float]:
    """Type-token ratio of every window of ``window`` tokens, computed once
    each via a running count rather than rebuilding a set per position."""

    total = len(tokens)
    if total == 0:
        return []
    if total <= window:
        return [len(set(tokens)) / total]
    counts: Counter[str] = Counter(tokens[:window])
    ratios = [len(counts) / window]
    for index in range(window, total):
        incoming, outgoing = tokens[index], tokens[index - window]
        counts[incoming] += 1
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        ratios.append(len(counts) / window)
    return ratios


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    tokens = analysis.tokens
    total = len(tokens)
    window = max(1, int(option(config, "window", DEFAULT_WINDOW)))

    if total == 0:
        warning = "no tokens to measure lexical diversity over"
        return [
            finding("style.mattr", "Moving-average type-token ratio", None, "%",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("style.mattr_window_shape", "Distribution of per-window type-token ratios",
                    None, "%", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
        ]

    ratios = _window_ttrs(tokens, window)
    percentages = [ratio * 100 for ratio in ratios]
    mean_ttr = statistics.fmean(percentages)
    summary = summarize(percentages)

    return [
        finding("style.mattr", "Moving-average type-token ratio", mean_ttr, "%",
                family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                distribution={"window": window, "windows_counted": len(ratios)},
                warning=None if total >= window else
                f"text has only {total} tokens, below the configured window of {window}; "
                f"the whole text was used as one window"),
        finding("style.mattr_window_shape", "Distribution of per-window type-token ratios",
                summary.get("median"), "%", family=FAMILY, sample_size=len(ratios),
                distribution=summary, min_sample=MIN_SAMPLE,
                evidence=[{"first_window_ttrs": [round(value, 2) for value in percentages[:20]]}]),
    ]
