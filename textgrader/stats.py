"""Distribution summaries and robust corpus comparison.

Two ideas drive this module.

The first is that a mean is rarely the interesting fact about a text.  Two
chapters can share a mean sentence length of fourteen words while one is a
metronome and the other alternates four-word beats with thirty-word
paragraphs-in-a-sentence.  :func:`summarize` therefore returns the shape of a
sample - quantiles, dispersion, entropy, lag-1 autocorrelation, run lengths and
a two-group split - and metrics are expected to publish that shape rather than
a single number.

The second is that "different from the corpus" needs a defensible threshold.
:func:`compare` uses median/MAD where the corpus has variation, falls back to
the interquartile range when MAD collapses to zero, falls back again to the
empirical percentile, and refuses to call anything an outlier when the corpus
is too small or has no variation at all.  A discrete metric measured over five
books does not get the authority of a continuous one measured over fifty.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .optional import require

# A corpus smaller than this cannot support an outlier claim at all.
MIN_CORPUS_SAMPLE = 8
# Below this a claim is made but marked low confidence.
CONFIDENT_CORPUS_SAMPLE = 20
# Robust-distance threshold.  3.5 is the conventional modified-z cutoff.
OUTLIER_DISTANCE = 3.5
# A one-sided scale needs enough observations on that side to be a scale at all.
MIN_SIDE_SAMPLE = 4
#: Severity is reported for ranking, so it is saturated rather than left
#: unbounded.  Zero-inflated rates (ellipses per 1,000 words: half the corpus at
#: zero, one book at 9.7) leave a robust deviation near zero, and the tail then
#: scores at tens or hundreds of "sigma".  That the book is extreme is true and
#: worth flagging; that it is 77 times more extreme than a metric scoring 4.0 is
#: not a claim any of this supports, and it puts one broken ruler at the top of
#: every agent's to-do list.  ``robust_distance`` keeps the exact value.
SEVERITY_CAP = 10.0
# 1 / 0.6745: scales the median absolute deviation to a normal-consistent sigma.
MAD_TO_SIGMA = 1.4826
# IQR of a normal distribution is 1.349 sigma.
IQR_TO_SIGMA = 1.349


def _clean(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            out.append(float(value))
    return out


def quantile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation quantile, matching numpy's default method."""

    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * min(max(q, 0.0), 1.0)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def shannon_entropy(values: Sequence[float], bins: int = 0) -> float | None:
    """Entropy in bits of a sample.

    Integer-valued samples (sentence lengths, sentences per paragraph) are
    counted exactly.  Continuous samples are binned with the Freedman-Diaconis
    rule unless ``bins`` is given, because entropy of a continuous sample is
    otherwise just a function of how many distinct floats happened to occur.
    """

    numbers = _clean(values)
    if not numbers:
        return None
    integral = all(float(value).is_integer() for value in numbers)
    if integral and not bins:
        counts = Counter(int(value) for value in numbers)
    else:
        count = bins or _freedman_diaconis_bins(numbers)
        low, high = min(numbers), max(numbers)
        if high <= low:
            return 0.0
        width = (high - low) / count
        counts = Counter(min(int((value - low) / width), count - 1) for value in numbers)
    total = sum(counts.values())
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n) + 0.0


def _freedman_diaconis_bins(numbers: Sequence[float]) -> int:
    spread = (quantile(numbers, 0.75) or 0.0) - (quantile(numbers, 0.25) or 0.0)
    if spread <= 0:
        return max(1, min(64, int(math.sqrt(len(numbers))) or 1))
    width = 2 * spread / (len(numbers) ** (1 / 3))
    extent = max(numbers) - min(numbers)
    if width <= 0 or extent <= 0:
        return 1
    return max(1, min(256, int(math.ceil(extent / width))))


def autocorrelation(values: Sequence[float], lag: int = 1) -> float | None:
    """Sample autocorrelation at ``lag``.

    Near zero means successive values are independent; a strongly negative
    lag-1 value is the signature of deliberate alternation, and a strongly
    positive one of long same-length stretches.
    """

    numbers = _clean(values)
    if len(numbers) <= lag + 1:
        return None
    mean = statistics.fmean(numbers)
    denominator = sum((value - mean) ** 2 for value in numbers)
    if denominator <= 0:
        return None
    numerator = sum((numbers[i] - mean) * (numbers[i + lag] - mean)
                    for i in range(len(numbers) - lag))
    return numerator / denominator


def double_mad(numbers: Sequence[float], median: float) -> tuple[float, float]:
    """Median absolute deviation computed separately on each side of the median.

    Many style rates are floored at zero and right-skewed: second-person
    pronouns in narration, parentheses, em dashes, colons.  Half the corpus sits
    near zero, which makes the pooled MAD tiny, which makes every book in the
    long upper tail a 3.5-sigma outlier.  Leave-one-out validation over thirty
    published novels showed exactly that: the seven worst-behaved metrics all
    had skew above 1.2 and every single one of their flags was "high".

    Scaling each side by its own spread removes the artefact.  On a symmetric
    sample this is not a different estimator: the half-sample deviation median
    equals the pooled one, so nothing well behaved changes.
    """

    lower = [value for value in numbers if value <= median]
    upper = [value for value in numbers if value >= median]
    low = statistics.median([abs(value - median) for value in lower]) if lower else 0.0
    high = statistics.median([abs(value - median) for value in upper]) if upper else 0.0
    # A side with almost nothing in it cannot supply a scale; fall back to the
    # other side rather than inventing a very small one.
    if len(lower) < MIN_SIDE_SAMPLE:
        low = high or low
    if len(upper) < MIN_SIDE_SAMPLE:
        high = low or high
    return low, high


def _skewness(numbers: Sequence[float]) -> float | None:
    n = len(numbers)
    if n < 3:
        return None
    mean = statistics.fmean(numbers)
    sd = statistics.pstdev(numbers)
    if sd <= 0:
        return 0.0
    m3 = sum((value - mean) ** 3 for value in numbers) / n
    g1 = m3 / sd ** 3
    return math.sqrt(n * (n - 1)) / (n - 2) * g1


def _kurtosis_excess(numbers: Sequence[float]) -> float | None:
    n = len(numbers)
    if n < 4:
        return None
    mean = statistics.fmean(numbers)
    sd = statistics.pstdev(numbers)
    if sd <= 0:
        return 0.0
    m4 = sum((value - mean) ** 4 for value in numbers) / n
    g2 = m4 / sd ** 4 - 3
    return ((n - 1) * ((n + 1) * g2 + 6)) / ((n - 2) * (n - 3))


def bimodality_coefficient(values: Sequence[float]) -> float | None:
    """SAS bimodality coefficient; above about 0.555 suggests two groups.

    This is a moment-based screening statistic, not a test.  It says the sample
    is shaped less like one hump than a normal distribution would be; it cannot
    say where the groups are.  :func:`two_group_split` answers that.
    """

    numbers = _clean(values)
    n = len(numbers)
    if n < 4:
        return None
    skew = _skewness(numbers)
    kurtosis = _kurtosis_excess(numbers)
    if skew is None or kurtosis is None:
        return None
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denominator = kurtosis + correction
    if denominator <= 0:
        return None
    return (skew ** 2 + 1) / denominator


def two_group_split(values: Sequence[float], iterations: int = 40) -> dict[str, Any] | None:
    """One-dimensional two-means split, reported with its separation.

    ``separation`` is the between-group variance as a share of total variance.
    A value near zero means the split is arbitrary; near one means the sample
    really is two populations.  Combined with :func:`bimodality_coefficient`
    this gives a metric an honest way to say "this text has two modes" instead
    of reporting an average that describes neither.
    """

    numbers = sorted(_clean(values))
    if len(numbers) < 6 or numbers[0] == numbers[-1]:
        return None
    low, high = numbers[0], numbers[-1]
    centre_low, centre_high = low + (high - low) / 4, high - (high - low) / 4
    members_low: list[float] = []
    members_high: list[float] = []
    for _ in range(iterations):
        boundary = (centre_low + centre_high) / 2
        members_low = [value for value in numbers if value <= boundary]
        members_high = [value for value in numbers if value > boundary]
        if not members_low or not members_high:
            return None
        new_low, new_high = statistics.fmean(members_low), statistics.fmean(members_high)
        if abs(new_low - centre_low) < 1e-12 and abs(new_high - centre_high) < 1e-12:
            centre_low, centre_high = new_low, new_high
            break
        centre_low, centre_high = new_low, new_high
    grand_mean = statistics.fmean(numbers)
    total = sum((value - grand_mean) ** 2 for value in numbers)
    between = (len(members_low) * (centre_low - grand_mean) ** 2
               + len(members_high) * (centre_high - grand_mean) ** 2)
    return {
        "low_centre": centre_low, "high_centre": centre_high,
        "low_count": len(members_low), "high_count": len(members_high),
        "low_share": len(members_low) / len(numbers),
        "separation": between / total if total else 0.0,
    }


def run_lengths(labels: Sequence[Any]) -> dict[Any, list[int]]:
    """Lengths of each maximal run of identical labels, keyed by label."""

    out: dict[Any, list[int]] = {}
    current: Any = object()
    length = 0
    for label in labels:
        if label == current:
            length += 1
        else:
            if length:
                out.setdefault(current, []).append(length)
            current, length = label, 1
    if length:
        out.setdefault(current, []).append(length)
    return out


def band_labels(values: Sequence[float], cuts: Sequence[float], names: Sequence[str]) -> list[str]:
    """Label each value by which band it falls in.  ``cuts`` are upper bounds."""

    labels = []
    for value in values:
        for cut, name in zip(cuts, names):
            if value <= cut:
                labels.append(name)
                break
        else:
            labels.append(names[-1])
    return labels


def histogram(values: Sequence[float], edges: Sequence[float]) -> dict[str, int]:
    """Count values into half-open buckets described by ``edges``.

    Buckets are reported in interval notation so an agent reading the JSON does
    not have to reconstruct the edges: ``"<=3"``, ``"(3,8]"``, ..., ``">40"``.
    """

    numbers = _clean(values)
    counts: dict[str, int] = {}
    previous: float | None = None
    for edge in edges:
        name = f"<={edge:g}" if previous is None else f"({previous:g},{edge:g}]"
        counts[name] = sum(1 for value in numbers
                           if (previous is None or value > previous) and value <= edge)
        previous = edge
    counts[f">{previous:g}"] = sum(1 for value in numbers if previous is not None and value > previous)
    return counts


def summarize(values: Iterable[Any], *, name: str = "", shape: bool = True) -> dict[str, Any]:
    """Describe a sample by its shape rather than by one central number.

    ``shape`` may be turned off for very large samples where entropy, the
    two-group split and autocorrelation are not wanted; the quantiles cost a
    single sort either way.
    """

    numbers = _clean(values)
    summary: dict[str, Any] = {"count": len(numbers)}
    if name:
        summary["name"] = name
    if not numbers:
        return summary
    ordered = sorted(numbers)
    mean = statistics.fmean(ordered)
    sd = statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    median = statistics.median(ordered)
    summary.update({
        "mean": mean, "median": median, "std": sd,
        "cv": 100 * sd / mean if mean else None,
        "min": ordered[0], "max": ordered[-1],
        "p10": quantile(ordered, .10), "p25": quantile(ordered, .25),
        "p75": quantile(ordered, .75), "p90": quantile(ordered, .90),
        "iqr": (quantile(ordered, .75) or 0.0) - (quantile(ordered, .25) or 0.0),
        "mad": statistics.median([abs(value - median) for value in ordered]),
    })
    if shape:
        summary["entropy"] = shannon_entropy(numbers)
        summary["lag1_autocorrelation"] = autocorrelation(numbers, 1)
        summary["skew"] = _skewness(numbers)
        summary["bimodality"] = bimodality_coefficient(numbers)
        split = two_group_split(numbers)
        if split:
            summary["two_group_split"] = split
    return summary


def wasserstein(sample: Sequence[float], reference: Sequence[float]) -> tuple[float | None, str | None]:
    """First Wasserstein (earth-mover) distance between two empirical samples.

    SciPy is used when present.  The pure-Python fallback is the exact same
    quantity for one-dimensional unweighted samples: the integral of the
    absolute difference of the two empirical CDFs.
    """

    left, right = _clean(sample), _clean(reference)
    if not left or not right:
        return None, "both samples must be non-empty"
    stats_module, reason = require("scipy.stats")
    if stats_module is not None:
        try:
            return float(stats_module.wasserstein_distance(left, right)), None
        except Exception as exc:  # pragma: no cover - scipy input guard
            reason = f"scipy.stats.wasserstein_distance failed: {exc}"
    left, right = sorted(left), sorted(right)
    merged = sorted(set(left) | set(right))
    total = 0.0
    for index in range(len(merged) - 1):
        point, nxt = merged[index], merged[index + 1]
        cdf_left = sum(1 for value in left if value <= point) / len(left)
        cdf_right = sum(1 for value in right if value <= point) / len(right)
        total += abs(cdf_left - cdf_right) * (nxt - point)
    return total, reason


# ---------------------------------------------------------------- item shapes

#: Quantile grid stored for a pooled corpus distribution.  101 points is the
#: whole shape to a percentile, in a kilobyte, and is what makes it possible to
#: keep the population rather than a per-document summary of it.
QUANTILE_GRID = 101


def quantile_curve(values: Iterable[Any], points: int = QUANTILE_GRID) -> list[float]:
    """The quantile function of a sample, sampled on an even grid.

    This is the sample's shape, not a summary of it: ``curve[0]`` is the
    minimum, ``curve[-1]`` the maximum, ``curve[50]`` the median on the default
    grid.  Storing this instead of a mean is what lets a 3,000-word chapter be
    held against half a million sentences of published prose, because the unit
    of observation on both sides is the sentence rather than the document.
    """

    ordered = sorted(_clean(values))
    if not ordered:
        return []
    return [quantile(ordered, index / (points - 1)) for index in range(points)]


def curve_distance(left: Sequence[float], right: Sequence[float]) -> float | None:
    """First Wasserstein distance between two samples, from their quantile curves.

    For one-dimensional samples the earth-mover distance is the mean absolute
    gap between the two quantile functions, so two curves on the same grid give
    it directly and neither original sample has to be kept.
    """

    if not left or not right or len(left) != len(right):
        return None
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def curve_quantile(curve: Sequence[float], q: float) -> float | None:
    """Read one quantile off a stored curve."""

    if not curve:
        return None
    position = (len(curve) - 1) * min(max(q, 0.0), 1.0)
    low = math.floor(position)
    high = min(low + 1, len(curve) - 1)
    return curve[low] + (curve[high] - curve[low]) * (position - low)


def band_shares(values: Iterable[Any], edges: Sequence[float]) -> list[float]:
    """Share of ``values`` falling in each band, as percentages.

    With ``edges`` taken from a corpus's own quantiles, a text that matched the
    corpus exactly would put an equal share in every band, so the deviation
    from flat is readable without knowing anything about the metric's units.
    """

    numbers = _clean(values)
    if not numbers:
        return []
    counts = [0] * (len(edges) + 1)
    for value in numbers:
        placed = False
        for index, edge in enumerate(edges):
            if value <= edge:
                counts[index] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    return [100.0 * count / len(numbers) for count in counts]


@dataclass
class Comparison:
    """The outcome of holding one measurement against a corpus distribution."""

    value: float | None = None
    corpus_count: int = 0
    corpus_median: float | None = None
    corpus_p10: float | None = None
    corpus_p90: float | None = None
    percentile: float | None = None
    robust_distance: float | None = None
    #: The robust scale the distance was divided by, in the metric's own unit.
    scale: float | None = None
    severity: float | None = None
    direction: str = "unknown"
    method: str = "none"
    outlier: bool | None = None
    confidence: str = "none"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_count": self.corpus_count, "corpus_median": self.corpus_median,
            "corpus_p10": self.corpus_p10, "corpus_p90": self.corpus_p90,
            "percentile": self.percentile, "robust_distance": self.robust_distance,
            "scale": self.scale, "severity": self.severity, "direction": self.direction,
            "method": self.method, "outlier": self.outlier,
            "confidence": self.confidence, "notes": list(self.notes),
        }


def compare(value: float | None, reference: Sequence[float], *,
            min_corpus: int = MIN_CORPUS_SAMPLE,
            threshold: float = OUTLIER_DISTANCE) -> Comparison:
    """Locate ``value`` in a corpus distribution, honestly.

    The estimator is chosen by what the corpus can actually support:

    ``median/double-MAD``
        the default, when the corpus has variation on the side the value falls.
        Each side of the median is scaled by its own spread, so a distribution
        floored at zero does not treat its whole upper tail as outlying.
    ``median/MAD``
        when one side has too few observations to supply a scale.
    ``median/IQR``
        when MAD is zero but the tails still differ.  A discrete metric over a
        small corpus - "3, 3, 3, 3, 5, 9" - has MAD 0 and would otherwise make
        every non-3 an infinitely distant outlier.
    ``empirical percentile``
        when even the IQR is zero but the sample is not constant.
    ``insufficient variation``
        when every observation is identical.  No outlier claim is made.

    A corpus below ``min_corpus`` observations yields ``outlier=None`` and a
    note, so five books never carry the authority of fifty.
    """

    numbers = sorted(_clean(reference))
    result = Comparison(value=value, corpus_count=len(numbers))
    if value is None or not math.isfinite(float(value)):
        result.notes.append("no measurement to compare")
        return result
    value = float(value)
    if not numbers:
        result.notes.append("no corpus distribution available")
        return result
    median = statistics.median(numbers)
    result.corpus_median = median
    result.corpus_p10 = quantile(numbers, .10)
    result.corpus_p90 = quantile(numbers, .90)
    result.percentile = 100 * (sum(1 for item in numbers if item < value)
                               + .5 * sum(1 for item in numbers if item == value)) / len(numbers)
    result.direction = "high" if value > median else "low" if value < median else "typical"

    mad = statistics.median([abs(item - median) for item in numbers])
    low_mad, high_mad = double_mad(numbers, median)
    sided = high_mad if value > median else low_mad
    iqr = (quantile(numbers, .75) or 0.0) - (quantile(numbers, .25) or 0.0)
    if sided > 0:
        result.method = "median/double-MAD"
        result.robust_distance = (value - median) / (MAD_TO_SIGMA * sided)
        result.severity = abs(result.robust_distance)
        result.scale = sided
        if mad > 0 and abs(sided - mad) / mad > 0.25:
            result.notes.append(
                f"the distribution is asymmetric, so the {'upper' if value > median else 'lower'} "
                f"side was scaled by its own spread ({sided:.4g}) rather than the pooled "
                f"MAD ({mad:.4g})")
    elif mad > 0:
        result.method = "median/MAD"
        result.robust_distance = (value - median) / (MAD_TO_SIGMA * mad)
        result.severity = abs(result.robust_distance)
        result.scale = mad
        result.notes.append("one side of the median had no spread; the pooled MAD was used")
    elif iqr > 0:
        result.method = "median/IQR"
        result.scale = iqr / IQR_TO_SIGMA
        result.robust_distance = (value - median) / result.scale
        result.severity = abs(result.robust_distance)
        result.notes.append("MAD was zero on both sides; interquartile range used instead")
    elif numbers[0] != numbers[-1]:
        result.method = "empirical percentile"
        # Distance in "tail steps": how far outside the observed range.
        span = numbers[-1] - numbers[0]
        if value > numbers[-1]:
            result.severity = threshold * (1 + (value - numbers[-1]) / span)
        elif value < numbers[0]:
            result.severity = threshold * (1 + (numbers[0] - value) / span)
        else:
            result.severity = abs(result.percentile - 50) / 50 * (threshold - 0.01)
        result.notes.append("MAD and IQR were both zero; empirical range used instead")
    else:
        result.method = "insufficient variation"
        result.severity = None
        result.outlier = None
        result.confidence = "none"
        result.notes.append(
            f"every corpus observation equals {numbers[0]:g}; no outlier claim is possible")
        return result

    if result.severity is not None and result.severity > SEVERITY_CAP:
        result.notes.append(
            f"severity saturated at {SEVERITY_CAP:g}; the exact robust distance is "
            f"{result.robust_distance:.1f} but the corpus spread is too small for the "
            f"difference between that and {SEVERITY_CAP:g} to mean anything"
            if result.robust_distance is not None else
            f"severity saturated at {SEVERITY_CAP:g}")
        result.severity = SEVERITY_CAP
    if len(numbers) < min_corpus:
        result.outlier = None
        result.confidence = "none"
        result.notes.append(
            f"corpus has {len(numbers)} observations, below the {min_corpus} needed "
            f"to call an outlier")
        return result
    result.outlier = bool(result.severity is not None and result.severity > threshold)
    result.confidence = "high" if len(numbers) >= CONFIDENT_CORPUS_SAMPLE else "low"
    if result.confidence == "low":
        result.notes.append(
            f"corpus has {len(numbers)} observations; fewer than {CONFIDENT_CORPUS_SAMPLE} "
            f"makes the distribution tails poorly determined")
    return result
