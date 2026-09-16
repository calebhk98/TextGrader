"""Gradual style drift: the trend a per-section z-score cannot see.

``drift_chapter_zscores`` finds a section that looks unlike its neighbours. It
is the wrong tool for a book that drifts smoothly -- sentences that lengthen
by a word or two every chapter, dialogue that thins out chapter by chapter --
because every section there is close to the sections next to it; nothing is
ever an outlier against "the rest of the book", yet chapter 1 and chapter 20
can read like different writers.

This module reuses the section splitting and feature extraction from
``drift_chapter_zscores`` (:func:`get_sections`, :func:`section_features`,
:data:`FEATURE_NAMES`) rather than repeating them, and asks two different
questions of the same per-section feature table, in reading order:

* a rank (Spearman-style) correlation of each feature against section index.
  Rank correlation is implemented directly here -- average ranks broken for
  ties, then the ordinary Pearson correlation of the two rank sequences,
  which is the exact definition of Spearman's rho -- because the brief asks
  for a from-scratch implementation. SciPy's ``spearmanr`` would agree with
  it on ranks without ties and is not needed for that reason; it is not used.
* the standardized difference (a pooled-variance Cohen's-d-style gap) between
  the first third and the last third of the book, which answers "how much did
  it actually move" in a way a correlation coefficient alone does not: a
  correlation of 0.9 across a two-word swing is a real trend and a small one.

The feature with the largest absolute rank correlation is reported as the
headline trend, on the view that "which measure is drifting" is more useful
to an editor than an omnibus drift index that mixes eight different scales.

Performance note: shares the same cost as ``drift_chapter_zscores`` and for
the same reason (each section's sentence lengths force pysbd to resegment
that section from scratch). Measured on a 400,000-word book with no headings
(156 windows): about 7.0 seconds, almost entirely inside that resegmentation.
See ``drift_chapter_zscores`` for the full explanation.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from .common import MODERATE, finding, option
from .drift_chapter_zscores import FEATURE_NAMES, get_sections, section_features

FAMILY = "book_drift"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
# Needs enough sections that "first third" and "last third" are not the same
# one or two sections, and that a rank correlation is not just noise.
MIN_SAMPLE = 6
UNIT_SENSITIVE = False


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks (1-based), tied values sharing the mean of their ranks."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    return covariance / math.sqrt(var_a * var_b)


def _rank_correlation(indices: Sequence[int], values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return _pearson(_ranks(list(indices)), _ranks(list(values)))


def _standardized_gap(first: Sequence[float], last: Sequence[float]) -> float | None:
    if len(first) < 2 or len(last) < 2:
        return None
    var_first, var_last = statistics.pvariance(first), statistics.pvariance(last)
    pooled = math.sqrt((var_first + var_last) / 2)
    if pooled <= 0:
        return None
    return (statistics.fmean(last) - statistics.fmean(first)) / pooled


def _insufficient(method: str | None, count: int) -> list[dict[str, Any]]:
    warning = (f"only {count} section(s) available (method={method}); need at least "
               f"{MIN_SAMPLE} to measure a trend across the book" if method else
               "text has no measurable sections")
    return [
        finding("drift.strongest_trend_correlation", "Strongest section-index rank correlation",
                None, "correlation", family=FAMILY, sample_size=count, min_sample=MIN_SAMPLE,
                warning=warning),
        finding("drift.trend_feature", "Feature with the strongest trend across the book",
                None, "feature", family=FAMILY, sample_size=count, min_sample=MIN_SAMPLE,
                warning=warning),
        finding("drift.first_last_third_gap",
                "Standardized gap between the book's first and last third",
                None, "standardized difference", family=FAMILY, sample_size=count,
                min_sample=MIN_SAMPLE, warning=warning),
    ]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    window_words = int(option(config, "window_words", 2500))
    sections, method = get_sections(analysis, window_words)
    if len(sections) < MIN_SAMPLE:
        return _insufficient(method, len(sections))

    features = [section_features(view) for _, view in sections]
    n = len(sections)
    third = max(1, n // 3)

    correlations: dict[str, float | None] = {}
    gaps: dict[str, float | None] = {}
    for feature in FEATURE_NAMES:
        indices = [i for i in range(n) if features[i][feature] is not None]
        values = [features[i][feature] for i in indices]
        correlations[feature] = _rank_correlation(indices, values)

        first_values = [features[i][feature] for i in range(0, third)
                        if features[i][feature] is not None]
        last_values = [features[i][feature] for i in range(n - third, n)
                       if features[i][feature] is not None]
        gaps[feature] = _standardized_gap(first_values, last_values)

    scored = [(feature, corr) for feature, corr in correlations.items() if corr is not None]
    if not scored:
        return _insufficient(method, n)
    top_feature, top_correlation = max(scored, key=lambda item: abs(item[1]))
    top_gap = gaps.get(top_feature)

    evidence = [
        {"feature": feature, "rank_correlation": correlations[feature],
         "first_third_vs_last_third_gap": gaps[feature]}
        for feature in FEATURE_NAMES
    ]
    evidence.sort(key=lambda row: abs(row["rank_correlation"] or 0.0), reverse=True)

    return [
        finding("drift.strongest_trend_correlation",
                f"Strongest section-index rank correlation (method={method})",
                top_correlation, "correlation", family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE, evidence=evidence,
                distribution={"feature": top_feature, "method": method}),
        finding("drift.trend_feature",
                f"Feature with the strongest trend across the book (method={method})",
                top_feature, "feature", family=FAMILY, sample_size=n, min_sample=MIN_SAMPLE,
                distribution={"rank_correlation": top_correlation}),
        finding("drift.first_last_third_gap",
                f"Standardized gap between the first and last third of the book on "
                f"{top_feature} (method={method})",
                top_gap, "standardized difference", family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE,
                distribution={"feature": top_feature, "first_third_sections": third,
                              "last_third_sections": third},
                warning=None if top_gap is not None else
                "the top-trend feature had no variation within its first or last third"),
    ]
