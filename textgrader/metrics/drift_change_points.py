"""Where a book's style changes abruptly, as opposed to gradually or locally.

``drift_chapter_zscores`` asks "which section is unlike the rest"; that
question is symmetric and does not care about order. ``drift_rolling`` asks
"is there a smooth trend end to end". Neither answers "the voice is
consistent through chapter 12 and then something changes at chapter 13 and
stays changed" -- a step function, not an outlier or a slope. That is a
genuine change-point detection problem, and ``ruptures`` (PELT: pruned exact
linear time) is the right tool for it rather than a hand-rolled one, when it
is available.

Sections and the per-section feature vector are reused from
``drift_chapter_zscores`` (:func:`get_sections`, :func:`section_features`,
:data:`FEATURE_NAMES`) rather than recomputed here. Each feature column is
standardized (zero mean, unit variance across the book's own sections) before
detection, so a feature measured in words cannot dominate one measured as a
percentage; a section missing a feature (too short to have a defined value)
is imputed at that column's mean, i.e. treated as unremarkable on that axis
rather than dropped, which would misalign section indices.

``ruptures.Pelt`` (an ``l2`` cost, i.e. it looks for shifts in the mean of the
standardized feature vector) is used through :func:`optional.require`, which
this repo requires for every third-party import. ``penalty`` is a *multiplier*
on the BIC term ``features * log(sections)`` (see :func:`bic_penalty`), not a
raw penalty: PELT's penalty is the whole false-positive control, and a constant
one admits spurious change points at a rate that grows with the book. Each
finding records the detector that ran and the penalty it used, because a count
from PELT and a count from the fallback are different measurements. When ``ruptures`` is not
installed, or its call fails for any reason, this degrades to a plain
dependency-free scan: try every possible single split point, keep the one
whose before/after mean feature vectors are furthest apart in Euclidean
distance. That fallback is a strictly weaker method -- it can only ever find
one change point, where PELT can find several -- and every finding produced
under it says so in its warning rather than presenting the number as
equivalent.

Performance note: shares the same cost as ``drift_chapter_zscores`` and for
the same reason (each section's sentence lengths force pysbd to resegment
that section from scratch); PELT itself is fast on a feature matrix this
small. Measured on a 400,000-word book with no headings (156 windows): about
7.5 seconds, almost entirely inside that resegmentation. See
``drift_chapter_zscores`` for the full explanation.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..optional import require
from .common import MODERATE, finding, option
from .drift_chapter_zscores import FEATURE_NAMES, get_sections, section_features

FAMILY = "book_drift"
COST = MODERATE
REQUIRES = ("ruptures",)
MIN_SAMPLE = 4
UNIT_SENSITIVE = False


def _standardize(matrix: Sequence[Sequence[float | None]]) -> list[list[float]]:
    if not matrix:
        return []
    columns = list(zip(*matrix))
    standardized_columns = []
    for column in columns:
        present = [value for value in column if value is not None]
        mean = statistics.fmean(present) if present else 0.0
        std = statistics.pstdev(present) if len(present) > 1 else 0.0
        standardized_columns.append([
            ((value - mean) / std if std > 0 else 0.0) if value is not None else 0.0
            for value in column
        ])
    return [list(row) for row in zip(*standardized_columns)]


def _segment_shift(standardized: Sequence[Sequence[float]], split: int) -> float:
    before, after = standardized[:split], standardized[split:]
    mean_before = [statistics.fmean(column) for column in zip(*before)]
    mean_after = [statistics.fmean(column) for column in zip(*after)]
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(mean_before, mean_after)))


def _fallback_scan(standardized: Sequence[Sequence[float]]) -> list[int]:
    """A single-split maximum mean-shift scan: the dependency-free fallback."""

    n = len(standardized)
    best_split, best_shift = None, 0.0
    for split in range(1, n):
        shift = _segment_shift(standardized, split)
        if shift > best_shift:
            best_split, best_shift = split, shift
    return [best_split] if best_split else []


def bic_penalty(sections: int, features: int, multiplier: float) -> float:
    """The PELT penalty for ``sections`` observations of ``features`` columns.

    PELT adds a segment only when doing so improves the fit by more than this,
    so the penalty is the entire false-positive control and it cannot be a
    constant. The cost here is ``l2`` on standardized columns, i.e. a Gaussian
    mean-shift model, whose BIC term is ``features * log(sections)`` per added
    segment: a longer book offers more places to split and every extra feature
    column adds another way for noise to look like a shift, so a penalty that
    ignores both admits false change points at a rate that grows with the book.

    Measured on pure standardized noise with this module's own eight features,
    the old constant 3.0 produced about one spurious change point every five
    or six sections at every length tested -- 21 of them in a section count
    typical of a 300,000-word novel. The same noise under this penalty
    produces none, while an injected step shift is still found within a couple
    of sections of where it was planted.
    """

    if sections < 2 or features < 1:
        return multiplier
    return multiplier * features * math.log(sections)


def _detect(standardized: Sequence[Sequence[float]],
            multiplier: float) -> tuple[list[int], str | None, str, float]:
    """Breakpoints, a warning, the detector that ran, and the penalty it used."""

    sections = len(standardized)
    features = len(standardized[0]) if standardized else 0
    penalty = bic_penalty(sections, features, multiplier)
    ruptures, reason = require("ruptures")
    if ruptures is not None:
        try:
            numpy, numpy_reason = require("numpy")
            signal = numpy.array(standardized) if numpy is not None else standardized
            algo = ruptures.Pelt(model="l2").fit(signal)
            result = algo.predict(pen=penalty)
            breakpoints = [point for point in result if point < sections]
            return breakpoints, None, "pelt", penalty
        except Exception as exc:  # pragma: no cover - library/runtime guard
            reason = f"ruptures.Pelt failed ({type(exc).__name__}: {exc})"
    warning = (f"{reason}; used a dependency-free single-split maximum mean-shift scan "
              f"instead of full PELT change-point detection, so at most one change "
              f"point could be found")
    return _fallback_scan(standardized), warning, "single_split_scan", penalty


def _insufficient(method: str | None, count: int) -> list[dict[str, Any]]:
    warning = (f"only {count} section(s) available (method={method}); need at least "
               f"{MIN_SAMPLE} to look for a change point" if method else
               "text has no measurable sections")
    return [
        finding("drift.change_point_count", "Number of style change points", None,
                "change points", family=FAMILY, sample_size=count, min_sample=MIN_SAMPLE,
                unit_sensitive=True, warning=warning),
        finding("drift.change_point_rate", "Style change points per 100 sections", None,
                "change points per 100 sections", family=FAMILY, sample_size=count,
                min_sample=MIN_SAMPLE, warning=warning),
        finding("drift.largest_change_magnitude", "Size of the largest style shift", None,
                "standardized distance", family=FAMILY, sample_size=count,
                min_sample=MIN_SAMPLE, warning=warning),
    ]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    window_words = int(option(config, "window_words", 2500))
    penalty_multiplier = float(option(config, "penalty", 2.0))
    sections, method = get_sections(analysis, window_words)
    if len(sections) < MIN_SAMPLE:
        return _insufficient(method, len(sections))

    features = [section_features(view) for _, view in sections]
    matrix = [[row[feature] for feature in FEATURE_NAMES] for row in features]
    standardized = _standardize(matrix)
    breakpoints, degrade_warning, detector, penalty_used = _detect(
        standardized, penalty_multiplier)
    settings = {"detector": detector, "penalty_multiplier": penalty_multiplier,
                "penalty_effective": penalty_used, "sections": len(sections),
                "features": len(FEATURE_NAMES)}

    shifts = [(split, _segment_shift(standardized, split)) for split in breakpoints]
    evidence = [
        {"section_index": split, "title": sections[split][0], "shift": shift}
        for split, shift in shifts
    ]
    largest = max(shifts, key=lambda item: item[1]) if shifts else None

    return [
        finding("drift.change_point_count",
                f"Number of style change points (method={method})",
                len(breakpoints), "change points", family=FAMILY, sample_size=len(sections),
                # A longer book is cut into more sections and so offers more
                # places to change. Measured over thirty published novels this
                # count correlated with word count at r = +0.97, which makes it
                # a length measurement; the rate below is the comparable one.
                unit_sensitive=True,
                min_sample=MIN_SAMPLE, evidence=evidence[:25], distribution=settings,
                warning=degrade_warning),
        finding("drift.change_point_rate",
                f"Style change points per 100 sections (method={method})",
                100 * len(breakpoints) / len(sections) if sections else None,
                "change points per 100 sections", family=FAMILY,
                sample_size=len(sections), min_sample=MIN_SAMPLE,
                distribution=settings, warning=degrade_warning),
        finding("drift.largest_change_magnitude",
                f"Size of the largest style shift (method={method})",
                largest[1] if largest else 0.0, "standardized distance", family=FAMILY,
                sample_size=len(sections), min_sample=MIN_SAMPLE,
                distribution={**settings, "section_index": largest[0],
                              "title": sections[largest[0]][0]}
                if largest else settings,
                warning=degrade_warning if degrade_warning else
                (None if largest else "no change point found")),
    ]
