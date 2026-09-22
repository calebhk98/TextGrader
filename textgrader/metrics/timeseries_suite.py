"""General time-series features over the named sequences in :mod:`textgrader.sequences`.

Two documents can share an identical mean, variance and even a full
distribution for some measurement -- sentence length, punctuation density,
parse depth -- while arranging those values completely differently over the
course of the book: one lengthens its sentences steadily from the first page
to the last, another alternates rigidly between short and long, a third
repeats a fixed pattern like a template.  Ordinary distributional statistics
cannot see any of that, because they throw away order on purpose.  This
module asks classical time-series questions (autocorrelation, trend,
stationarity, spectral shape, run structure, scaling behaviour, change
points, drift-detector events) of every sequence :mod:`textgrader.sequences`
exposes, and reports each `(sequence, feature)` pair as one finding under a
stable id.

**What this is not.**  ``rhythm_autocorrelation``, ``rhythm_deltas``,
``rhythm_runs`` and ``rhythm_entropy`` already give sentence length its own
hand-tuned lag-1..3 autocorrelation, delta distribution, run-length bands and
entropy; ``drift_rolling``, ``drift_chapter_zscores`` and
``drift_change_points`` already give a book's per-section feature vector its
own trend correlation, per-section deviation and PELT change-point count.
This module does not replace any of that and produces different, explicitly
different, ids (``rhythm.timeseries_sentence_words_acf`` next to
``rhythm.sentence_length_autocorrelation_lag1``, for instance): where the
underlying arithmetic genuinely overlaps (autocorrelation, change points)
the numbers can be expected to agree, or nearly so, because they are
measuring the same sentence-length series two different ways, and that
agreement is worth having as a cross-check rather than a reason to delete
one.  Where this module differs on purpose is breadth: it runs the same
feature battery over *every* sequence in the registry, sentence length among
many, so a book that drifts in dependency distance rather than sentence
length is not invisible just because nobody wrote a bespoke
``dependency_distance_autocorrelation`` module.

**Guarding the combinatorics.**  Sequences (16 in the registry) times
features (15 groups) is 240 possible findings before a single lag or box
size is counted -- exactly the explosion the task's own brief warns against.
Three things bound it: (1) every feature group is reported as *one* finding
per sequence, with its secondary numbers (all lags, all box sizes, segment
slopes, ...) folded into that finding's ``distribution`` rather than exploded
into their own ids; (2) the default configuration selects three
dependency-free, always-available sequences and five feature groups that are
informative even on a single chapter (15 findings by default, see
``DEFAULT_SEQUENCES``/``DEFAULT_FEATURE_GROUPS`` below for the reasoning);
(3) ``max_findings`` (default 200) is a hard stop -- a user who selects
every sequence and every feature group gets the first 200 combinations, in
the order they configured, plus one finding that says so, rather than a
silent multi-thousand-row report.

**Every measurement is individually selectable.**  ``sequences`` and
``feature_groups`` are both explicit lists in this metric's configuration
(mirrored in ``config.json``): a user who wants "just sentence length and
just the autocorrelation group" sets ``sequences: ["sentence_words"]`` and
``feature_groups: ["acf"]`` and gets exactly one finding.  Every other
tunable a feature needs (lag lists, PACF depth, rolling-window size,
change-point penalty, Page-Hinkley sensitivity, permutation-entropy order,
detrending) is its own configuration key, read through
``common.option()``, with the same default used whether or not the suite's
switch is on.

**Default selection.**  ``sentence_words``, ``paragraph_words`` and
``sentence_punctuation`` need no optional package and are defined on any
text with at least a few sentences, so the default report is identical
across every environment this runs in.  ``dispersion``, ``acf``, ``trend``,
``turning_points`` and ``runs`` are the feature groups with the lowest
minimum length and the most direct reading: "how spread out is this",
"do neighbours resemble each other", "is there a slope", "how choppy is
it", "how long are its streaks".  The nonlinear/spectral/scaling groups
(Hurst, DFA, spectral entropy, permutation entropy) and the detector groups
(change points, Page-Hinkley) need materially more data to mean anything and
are opt-in.

**Sample-size honesty.**  Every feature has its own minimum length
(``DEFAULT_MIN_LENGTHS``, overridable per feature through
``min_lengths``), enforced before the feature runs, not after: below it, the
finding reports ``value=None`` with a warning naming the shortfall, which
``grade.py`` reports as unavailable rather than as a number a corpus outlier
check could ever pick up.  This matters more here than for a single
hand-written metric, because "Hurst exponent from 11 points" is exactly the
kind of nonsense number the task's own brief calls out by name.  Every
finding's ``distribution`` also records the sequence's name, unit, sample
unit, whatever window/lag/model settings produced it, whether it was
detrended first, and (where a library did the work) that library's version
-- the settings a corpus profile has to match before two numbers can be
compared, per the task's corpus/profile requirement.

Headline values are deliberately chosen to be scale-free (a correlation, a
share, an exponent, a rate per 100 points) rather than a raw count, so no
finding here needs a separate normalized sibling; raw counts that exist
purely as evidence (peak count, event count, change-point locations) live
inside ``distribution`` instead of becoming their own id.  ``hurst``,
``dfa``, ``permutation_entropy``, ``spectral`` and ``runs`` are marked
``sample_size_sensitive``: each has a documented, systematic dependency of
its *expected* value on how many points it was estimated from (the
classical result that the longest run in ``n`` random draws grows only as
``log2(n)``, for instance), not just a documented reduction in noise, so
comparing them across texts of very different sequence length is comparing
different quantities even after normalization.

**Deferred.**  ``catch22``/``catch24`` (needs ``pycatch22``) and a
configurable ``tsfresh`` feature set are named in the task's acceptance
criteria but are not implemented here: neither package is installed in this
environment, both are large enough that their success path could not be
exercised or validated before committing, and the task's own scope note
("nothing that needs an untestable package") rules them out for this pass.
Wavelet energy by scale (``PyWavelets``) and recurrence-quantification
features (``PyRQA``) are left out for the identical reason. A streaming
``river`` ADWIN detector is skipped in favour of a small dependency-free
Page-Hinkley implementation (``page_hinkley``, below), which needs no
optional package and is directly testable against a synthetic step-shift
sequence. Sentiment/emotion scoring and topic-probability sequences are not
in this module at all; see ``textgrader/sequences.py``'s docstring for why
they were left out of the sequence registry itself, one level down.

Deterministic settings throughout: no random seeds are needed because every
feature here is a closed-form or exact-recursion computation, not a fit with
random initialization (a genuine, tested exception is the ``ruptures``-backed
``change_points`` feature's PELT search, which is itself deterministic).
"""

from __future__ import annotations

import itertools
import math
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence as TypingSequence

from .. import sequences as seq
from ..document import DocumentAnalysis
from ..optional import require
from ..stats import autocorrelation, run_lengths, summarize
from .common import MODERATE, finding, option

FAMILY = "sentence_rhythm"  # fallback only; every finding sets its own real family
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
# A floor under every feature's own, higher, minimum -- see DEFAULT_MIN_LENGTHS.
MIN_SAMPLE = 5
UNIT_SENSITIVE = False

# ------------------------------------------------------------------- selection

#: No optional package, always defined on any text with a handful of
#: sentences and two paragraphs -- see the module docstring for why these
#: three are the default rather than an arbitrary sample of the registry.
DEFAULT_SEQUENCES = ("sentence_words", "paragraph_words", "sentence_punctuation")

#: The five feature groups with the lowest minimum length and the most
#: direct, least assumption-laden reading. See module docstring.
DEFAULT_FEATURE_GROUPS = ("dispersion", "acf", "trend", "turning_points", "runs")

FEATURE_NAMES = (
    "dispersion", "rolling_dispersion", "acf", "pacf", "trend", "piecewise_trend",
    "stationarity", "turning_points", "runs", "spectral", "hurst", "dfa",
    "permutation_entropy", "change_points", "page_hinkley",
)

FEATURE_LABELS = {
    "dispersion": "Dispersion", "rolling_dispersion": "Rolling dispersion",
    "acf": "Autocorrelation", "pacf": "Partial autocorrelation", "trend": "Linear trend",
    "piecewise_trend": "Piecewise trend gap", "stationarity": "Stationarity",
    "turning_points": "Turning-point rate", "runs": "Longest run",
    "spectral": "Spectral shape", "hurst": "Hurst exponent", "dfa": "DFA scaling exponent",
    "permutation_entropy": "Permutation entropy", "change_points": "Change-point rate",
    "page_hinkley": "Page-Hinkley drift-event rate",
}

FEATURE_UNITS = {
    "dispersion": None, "rolling_dispersion": None, "acf": "correlation",
    "pacf": "correlation", "trend": "correlation", "piecewise_trend": "correlation difference",
    "stationarity": "p-value", "turning_points": "%", "runs": "%", "spectral": "ratio",
    "hurst": "exponent", "dfa": "exponent", "permutation_entropy": "ratio",
    "change_points": "change points per 100 points", "page_hinkley": "events per 100 points",
}

# Below this many points a feature is refused rather than reported as a
# number a reader (or a corpus outlier check) could mistake for a real
# estimate.  Overridable per feature via the ``min_lengths`` option.
DEFAULT_MIN_LENGTHS = {
    "dispersion": 5, "rolling_dispersion": 12, "acf": 15, "pacf": 20, "trend": 6,
    "piecewise_trend": 10, "stationarity": 30, "turning_points": 6, "runs": 6,
    "spectral": 16, "hurst": 40, "dfa": 40, "permutation_entropy": 20,
    "change_points": 10, "page_hinkley": 10,
}

# Features whose expected value has a documented, systematic dependence on
# sequence length even after normalization (see module docstring).
SAMPLE_SIZE_SENSITIVE_FEATURES = frozenset({"hurst", "dfa", "permutation_entropy", "spectral", "runs"})

# Features that describe the raw shape of the values themselves; detrending
# them first would be circular (piecewise_trend/trend) or pointless
# (dispersion already IS a dispersion measure, trend or no).
_NEVER_DETREND = frozenset({"dispersion", "rolling_dispersion", "trend", "piecewise_trend"})

# Sequences that take their own extra settings beyond the shared defaults.
_WINDOW_SEQUENCES = frozenset({"window_dialogue_fraction", "window_pronoun_rate"})
_RARITY_SEQUENCES = frozenset({"sentence_content_rarity"})
_EMBEDDING_SEQUENCES = frozenset({"sentence_similarity_prev", "sentence_distance_centroid"})


@dataclass(frozen=True)
class _Outcome:
    value: float | None
    distribution: Mapping[str, Any] = field(default_factory=dict)
    warning: str | None = None
    sample_size_sensitive: bool = False


def _lib_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


# -------------------------------------------------------------- small maths

def _ols(xs: TypingSequence[float], ys: TypingSequence[float]) -> tuple[float | None, float | None, float | None]:
    """``(slope, intercept, r)`` of ``ys`` regressed on ``xs``, or ``(None, None, None)``."""

    n = len(xs)
    if n < 2:
        return None, None, None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0:
        return None, None, None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    var_y = sum((y - mean_y) ** 2 for y in ys)
    r = cov / math.sqrt(var_x * var_y) if var_y > 0 else None
    return slope, intercept, r


def _residuals(values: TypingSequence[float]) -> list[float]:
    xs = list(range(len(values)))
    slope, intercept, _ = _ols(xs, values)
    if slope is None:
        return list(values)
    return [v - (intercept + slope * x) for x, v in zip(xs, values)]


def _durbin_levinson_pacf(values: TypingSequence[float], max_lag: int) -> list[float | None]:
    """Partial autocorrelation at lags ``1..max_lag`` via Durbin-Levinson recursion."""

    n = len(values)
    mean = statistics.fmean(values)
    centered = [v - mean for v in values]
    c = [sum(centered[t] * centered[t + k] for t in range(n - k)) / n for k in range(max_lag + 1)]
    if c[0] <= 0:
        return [None] * max_lag
    phi: dict[tuple[int, int], float] = {}
    pacf: list[float | None] = []
    phi[(1, 1)] = c[1] / c[0]
    pacf.append(phi[(1, 1)])
    for k in range(2, max_lag + 1):
        numerator = c[k] - sum(phi[(k - 1, j)] * c[k - j] for j in range(1, k))
        denominator = c[0] - sum(phi[(k - 1, j)] * c[j] for j in range(1, k))
        if denominator == 0:
            pacf.append(None)
            continue
        phi[(k, k)] = numerator / denominator
        for j in range(1, k):
            phi[(k, j)] = phi[(k - 1, j)] - phi[(k, k)] * phi[(k - 1, k - j)]
        pacf.append(phi[(k, k)])
    return pacf


# ------------------------------------------------------------------- features

def _feature_dispersion(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    summary = summarize(list(values))
    return _Outcome(summary.get("median"), distribution=summary)


def _feature_rolling_dispersion(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    window = max(3, int(cfg.get("rolling_window", 10)))
    if len(values) < window + 2:
        return _Outcome(None, warning=f"needs more than {window + 2} points for a "
                                      f"{window}-point rolling window")
    rolled = [statistics.pstdev(values[i:i + window]) for i in range(0, len(values) - window + 1)]
    summary = summarize(rolled)
    return _Outcome(summary.get("median"), distribution={"rolling_window": window, **summary})


def _feature_acf(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    lags = [int(lag) for lag in cfg.get("lags", [1, 2, 3]) if int(lag) >= 1]
    if not lags:
        return _Outcome(None, warning="no positive lag configured")
    acf_values = {f"lag{lag}": autocorrelation(values, lag) for lag in lags}
    headline = acf_values.get(f"lag{lags[0]}")
    return _Outcome(headline, distribution={"lags": lags, "acf": acf_values})


def _feature_pacf(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    max_lag = max(1, int(cfg.get("pacf_max_lag", 5)))
    max_lag = min(max_lag, len(values) // 3 or 1)
    pacf_values = _durbin_levinson_pacf(values, max_lag)
    headline = pacf_values[0] if pacf_values else None
    return _Outcome(headline, distribution={
        "max_lag": max_lag, "pacf": {f"lag{i + 1}": v for i, v in enumerate(pacf_values)}})


def _feature_trend(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    xs = list(range(len(values)))
    slope, intercept, r = _ols(xs, values)
    if r is None:
        return _Outcome(None, warning="every value is identical; no trend is defined")
    return _Outcome(r, distribution={"slope_per_step": slope, "intercept": intercept,
                                     "r_squared": r * r})


def _feature_piecewise_trend(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    n = len(values)
    segments = max(2, int(cfg.get("piecewise_segments", 2)))
    size = n // segments
    if size < 3:
        return _Outcome(None, warning=f"needs at least {segments * 3} points for "
                                      f"{segments} piecewise segments")
    correlations: list[float | None] = []
    for index in range(segments):
        start = index * size
        stop = n if index == segments - 1 else (index + 1) * size
        chunk = values[start:stop]
        _, _, r = _ols(list(range(len(chunk))), chunk)
        correlations.append(r)
    valid = [r for r in correlations if r is not None]
    gap = (valid[-1] - valid[0]) if len(valid) >= 2 else None
    return _Outcome(gap, distribution={"segment_correlations": correlations, "segments": segments},
                    warning=None if gap is not None else
                    "fewer than two segments had a defined trend to compare")


def _feature_stationarity(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    statsmodels_module, reason = require("statsmodels")
    if statsmodels_module is not None:
        try:
            from statsmodels.tsa.stattools import adfuller
            # A near-constant or exactly periodic series can make the
            # regression statsmodels fits internally rank-deficient; that is
            # a property of the input, not a bug, and it still returns a
            # usable (if less trustworthy) statistic, so the warning is
            # suppressed here rather than surfaced as if this module raised it.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                statistic, pvalue, used_lag, *_ = adfuller(
                    list(values), autolag="AIC", regression="c", result_object=False)
            return _Outcome(float(pvalue), distribution={
                "backend": "adfuller", "library": "statsmodels",
                "library_version": _lib_version(statsmodels_module),
                "adf_statistic": float(statistic), "lags_used": int(used_lag),
                "null_hypothesis": "the series has a unit root (is non-stationary); a small "
                                   "p-value rejects that, i.e. is evidence FOR stationarity"})
        except Exception as exc:  # pragma: no cover - library/runtime guard
            reason = f"statsmodels.tsa.stattools.adfuller failed ({type(exc).__name__}: {exc})"
    n = len(values)
    half = n // 2
    first, second = values[:half], values[half:]
    pooled_sd = statistics.pstdev(values) or 1.0
    proxy = abs(statistics.fmean(second) - statistics.fmean(first)) / pooled_sd
    warning = (f"{reason}; used a dependency-free half-split standardized mean-gap proxy "
              f"instead of the Augmented Dickey-Fuller test. This is a weaker, non-standard "
              f"indicator: it catches a level shift between the first and second half but says "
              f"nothing about a unit root, and a value near zero does not mean the series is "
              f"stationary, only that its two halves have similar means")
    return _Outcome(proxy, distribution={"backend": "half_split_proxy"}, warning=warning)


def _feature_turning_points(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    n = len(values)
    peaks = troughs = 0
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            peaks += 1
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            troughs += 1
    interior = n - 2
    share = 100.0 * (peaks + troughs) / interior if interior else None
    return _Outcome(share, distribution={"peaks": peaks, "troughs": troughs,
                                         "interior_points": interior})


def _feature_runs(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    median = statistics.median(values)
    labels = ["above" if v > median else "below" if v < median else "equal" for v in values]
    runs = run_lengths(labels)
    by_label = {label: (max(lengths) if lengths else 0) for label, lengths in runs.items()}
    longest = max(by_label.values(), default=0)
    share = 100.0 * longest / len(values) if values else None
    return _Outcome(share, distribution={"longest_run_points": longest, "by_label": by_label},
                    sample_size_sensitive=True)


def _feature_spectral(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    arr = numpy.asarray(values, dtype=float)
    arr = arr - arr.mean()
    if not numpy.any(numpy.abs(arr) > 1e-12):
        return _Outcome(0.0, distribution={"dominant_frequency_cycles_per_point": 0.0,
                                           "spectral_concentration": None},
                        warning="the sequence is constant once its mean is removed; spectral "
                                "entropy is undefined so 0.0 (a pure tone) was used",
                        sample_size_sensitive=True)
    spectrum = numpy.abs(numpy.fft.rfft(arr)) ** 2
    spectrum = spectrum[1:]  # drop the (near-zero, post mean-removal) DC term
    total = float(spectrum.sum())
    if total <= 0 or len(spectrum) < 2:
        return _Outcome(None, warning="too few frequency bins to summarize a spectrum",
                        sample_size_sensitive=True)
    p = spectrum / total
    raw_entropy = float(-numpy.sum(p * numpy.log2(numpy.where(p > 0, p, 1.0))))
    max_entropy = math.log2(len(p))
    normalized = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    dominant_index = int(numpy.argmax(spectrum))
    n = len(arr)
    return _Outcome(normalized, distribution={
        "raw_entropy_bits": raw_entropy,
        "dominant_frequency_cycles_per_point": (dominant_index + 1) / n,
        "spectral_concentration": float(p.max()),
        "library": "numpy", "library_version": _lib_version(numpy)},
        sample_size_sensitive=True)


def _rescaled_range_scales(values: TypingSequence[float], min_chunk: int) -> list[int]:
    sizes = []
    size = len(values)
    while size >= min_chunk:
        sizes.append(size)
        size //= 2
    return sorted(set(sizes))


def _feature_hurst(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    sizes = _rescaled_range_scales(values, min_chunk=8)
    log_sizes: list[float] = []
    log_rs: list[float] = []
    n = len(values)
    for size in sizes:
        chunks = [values[i:i + size] for i in range(0, n - n % size, size)] if size else []
        rs_values = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_c = statistics.fmean(chunk)
            cumulative = list(itertools.accumulate(v - mean_c for v in chunk))
            spread = max(cumulative) - min(cumulative)
            sd = statistics.pstdev(chunk)
            if sd > 0:
                rs_values.append(spread / sd)
        if rs_values:
            log_sizes.append(math.log(size))
            log_rs.append(math.log(statistics.fmean(rs_values)))
    if len(log_sizes) < 3:
        return _Outcome(None, warning="could not compute a rescaled range at enough distinct "
                                      "scales (need variation within at least three window sizes)",
                        sample_size_sensitive=True)
    slope, _, _ = _ols(log_sizes, log_rs)
    return _Outcome(slope, distribution={"scales": sizes}, sample_size_sensitive=True)


def _feature_dfa(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    n = len(values)
    mean_v = statistics.fmean(values)
    profile = list(itertools.accumulate(v - mean_v for v in values))
    min_box = max(4, int(cfg.get("dfa_min_box", 4)))
    sizes: list[int] = []
    box = min_box
    while box <= n // 4:
        sizes.append(box)
        box = int(box * 1.5) + 1
    sizes = sorted(set(sizes))
    log_sizes: list[float] = []
    log_f: list[float] = []
    for box in sizes:
        n_boxes = n // box
        if n_boxes < 2:
            continue
        fluctuations = []
        for i in range(n_boxes):
            segment = profile[i * box:(i + 1) * box]
            xs = list(range(box))
            slope, intercept, _ = _ols(xs, segment)
            if slope is None:
                continue
            residual_sq = sum((segment[j] - (intercept + slope * j)) ** 2 for j in range(box))
            fluctuations.append(math.sqrt(residual_sq / box))
        if fluctuations:
            rms = math.sqrt(sum(f * f for f in fluctuations) / len(fluctuations))
            if rms > 0:
                log_sizes.append(math.log(box))
                log_f.append(math.log(rms))
    if len(log_sizes) < 3:
        return _Outcome(None, warning="could not compute a detrended fluctuation at enough box "
                                      "sizes for this sequence length",
                        sample_size_sensitive=True)
    slope, _, _ = _ols(log_sizes, log_f)
    return _Outcome(slope, distribution={"box_sizes": sizes}, sample_size_sensitive=True)


def _feature_permutation_entropy(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    order = max(2, int(cfg.get("permutation_entropy_order", 3)))
    delay = max(1, int(cfg.get("permutation_entropy_delay", 1)))
    n = len(values)
    windows = n - (order - 1) * delay
    if windows < 5:
        return _Outcome(None, warning=f"needs at least {5 + (order - 1) * delay} points for "
                                      f"permutation-entropy order {order}",
                        sample_size_sensitive=True)
    patterns: Counter[tuple[int, ...]] = Counter()
    for i in range(windows):
        window = [values[i + j * delay] for j in range(order)]
        rank = tuple(sorted(range(order), key=lambda idx: (window[idx], idx)))
        patterns[rank] += 1
    total = sum(patterns.values())
    raw_entropy = -sum((c / total) * math.log2(c / total) for c in patterns.values())
    max_entropy = math.log2(math.factorial(order))
    normalized = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    return _Outcome(normalized, distribution={
        "order": order, "delay": delay, "distinct_patterns": len(patterns),
        "possible_patterns": math.factorial(order), "raw_entropy_bits": raw_entropy},
        sample_size_sensitive=True)


def _feature_change_points(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    n = len(values)
    penalty = float(cfg.get("change_point_penalty", 3.0))
    ruptures_module, reason = require("ruptures")
    numpy, numpy_reason = require("numpy")
    if ruptures_module is not None and numpy is not None:
        try:
            arr = numpy.asarray(values, dtype=float).reshape(-1, 1)
            std = float(arr.std())
            if std > 0:
                arr = (arr - arr.mean()) / std
            algo = ruptures_module.Pelt(model="l2").fit(arr)
            breakpoints = [point for point in algo.predict(pen=penalty) if point < n]
            rate = 100.0 * len(breakpoints) / n
            return _Outcome(rate, distribution={
                "backend": "ruptures.Pelt", "library_version": _lib_version(ruptures_module),
                "count": len(breakpoints), "locations": breakpoints[:25]})
        except Exception as exc:  # pragma: no cover - library/runtime guard
            reason = f"ruptures.Pelt failed ({type(exc).__name__}: {exc})"
    reason = reason or numpy_reason
    prefix = list(itertools.accumulate(values))
    total_sum = prefix[-1]
    sd_all = statistics.pstdev(values) or 1.0
    best_split, best_shift = None, 0.0
    for split in range(1, n):
        mean_before = prefix[split - 1] / split
        mean_after = (total_sum - prefix[split - 1]) / (n - split)
        shift = abs(mean_after - mean_before) / sd_all
        if shift > best_shift:
            best_split, best_shift = split, shift
    count = 1 if best_split else 0
    rate = 100.0 * count / n
    warning = (f"{reason}; used a dependency-free single-split maximum mean-shift scan instead "
              f"of full PELT change-point detection, so at most one change point could be found")
    return _Outcome(rate, distribution={"backend": "single_split_scan", "count": count,
                                        "locations": [best_split] if best_split else []},
                    warning=warning)


def _feature_page_hinkley(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    """Event count of a classical Page-Hinkley drift detector, run over the series.

    A dependency-free stand-in for ``river``'s streaming ADWIN/Page-Hinkley
    detectors: ``delta`` (a magnitude, scaled by the series' own SD) is the
    smallest mean shift worth flagging, and an event fires when the
    cumulative deviation from the running mean exceeds ``lambda`` (also
    scaled by SD) since the last reset. Both are configurable as multiples of
    the sequence's own standard deviation so one setting works across
    sequences measured in different units.
    """

    n = len(values)
    sd = statistics.pstdev(values) or 1.0
    delta = float(cfg.get("page_hinkley_delta", 0.005)) * sd
    lam = float(cfg.get("page_hinkley_lambda", 3.0)) * sd
    mean_est = values[0]
    count_seen = 1
    cumulative = 0.0
    min_cumulative = 0.0
    events: list[int] = []
    for i in range(1, n):
        count_seen += 1
        mean_est += (values[i] - mean_est) / count_seen
        cumulative += values[i] - mean_est - delta
        min_cumulative = min(min_cumulative, cumulative)
        if cumulative - min_cumulative > lam:
            events.append(i)
            mean_est, count_seen, cumulative, min_cumulative = values[i], 1, 0.0, 0.0
    rate = 100.0 * len(events) / n
    return _Outcome(rate, distribution={
        "event_count": len(events), "first_event_index": events[0] if events else None,
        "last_event_index": events[-1] if events else None,
        "delta": delta, "lambda_threshold": lam})


_FEATURES: dict[str, Callable[[TypingSequence[float], Mapping[str, Any]], _Outcome]] = {
    "dispersion": _feature_dispersion,
    "rolling_dispersion": _feature_rolling_dispersion,
    "acf": _feature_acf,
    "pacf": _feature_pacf,
    "trend": _feature_trend,
    "piecewise_trend": _feature_piecewise_trend,
    "stationarity": _feature_stationarity,
    "turning_points": _feature_turning_points,
    "runs": _feature_runs,
    "spectral": _feature_spectral,
    "hurst": _feature_hurst,
    "dfa": _feature_dfa,
    "permutation_entropy": _feature_permutation_entropy,
    "change_points": _feature_change_points,
    "page_hinkley": _feature_page_hinkley,
}

assert set(_FEATURES) == set(FEATURE_NAMES)


# ------------------------------------------------------------------- settings

def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "sequences": list(option(config, "sequences", DEFAULT_SEQUENCES)),
        "feature_groups": list(option(config, "feature_groups", DEFAULT_FEATURE_GROUPS)),
        "lags": [int(x) for x in option(config, "lags", [1, 2, 3])],
        "pacf_max_lag": int(option(config, "pacf_max_lag", 5)),
        "window_words": int(option(config, "window_words", 2000)),
        "rolling_window": int(option(config, "rolling_window", 10)),
        "change_point_penalty": float(option(config, "change_point_penalty", 3.0)),
        "page_hinkley_delta": float(option(config, "page_hinkley_delta", 0.005)),
        "page_hinkley_lambda": float(option(config, "page_hinkley_lambda", 3.0)),
        "permutation_entropy_order": int(option(config, "permutation_entropy_order", 3)),
        "permutation_entropy_delay": int(option(config, "permutation_entropy_delay", 1)),
        "dfa_min_box": int(option(config, "dfa_min_box", 4)),
        "piecewise_segments": int(option(config, "piecewise_segments", 2)),
        "detrend": bool(option(config, "detrend", False)),
        "embedding_model": option(config, "embedding_model", "all-MiniLM-L6-v2"),
        "language": option(config, "language", "en"),
        "min_lengths": dict(option(config, "min_lengths", {})),
        "max_findings": int(option(config, "max_findings", 200)),
    }


def _sequence_settings(name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if name in _WINDOW_SEQUENCES:
        return {"window_words": cfg["window_words"]}
    if name in _RARITY_SEQUENCES:
        return {"language": cfg["language"]}
    if name in _EMBEDDING_SEQUENCES:
        return {"model": cfg["embedding_model"]}
    return {}


def _join(existing: str | None, addition: str | None) -> str | None:
    if not addition:
        return existing
    return f"{existing}; {addition}" if existing else addition


def _metric_id(sequence_name: str, feature_name: str, family: str) -> str:
    prefix = "drift.timeseries_" if family == "book_drift" else "rhythm.timeseries_"
    return f"{prefix}{sequence_name}_{feature_name}"


def _measure_one(analysis: DocumentAnalysis, sequence_name: str, feature_name: str,
                 cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec = seq.SEQUENCES[sequence_name]
    metric_id = _metric_id(sequence_name, feature_name, spec.family)
    name = f"{FEATURE_LABELS[feature_name]} of {sequence_name.replace('_', ' ')}"

    sequence = seq.get_sequence(analysis, sequence_name, _sequence_settings(sequence_name, cfg))
    # dispersion/rolling_dispersion report in the sequence's own unit (words,
    # marks, bits, ...); every other feature's unit is scale-free and fixed.
    unit = FEATURE_UNITS[feature_name] or sequence.unit
    if not sequence.values:
        return finding(metric_id, name, None, unit, family=spec.family,
                       sample_size=0, warning=sequence.warning or "sequence has no values")

    min_length = int(cfg["min_lengths"].get(feature_name, DEFAULT_MIN_LENGTHS[feature_name]))
    if sequence.length < min_length:
        return finding(
            metric_id, name, None, unit, family=spec.family, sample_size=sequence.length,
            min_sample=min_length,
            warning=f"insufficient data: {sequence_name} has {sequence.length} "
                    f"{sequence.sample_unit} points, below the {min_length} the "
                    f"{feature_name} feature needs")

    working = list(sequence.values)
    detrended = cfg["detrend"] and feature_name not in _NEVER_DETREND
    if detrended:
        working = _residuals(working)

    outcome = _FEATURES[feature_name](working, cfg)
    distribution = dict(outcome.distribution)
    distribution.setdefault("sequence", sequence_name)
    distribution.setdefault("sequence_unit", sequence.unit)
    distribution.setdefault("sample_unit", sequence.sample_unit)
    distribution["detrended"] = detrended
    if sequence.settings:
        distribution["sequence_settings"] = dict(sequence.settings)

    return finding(
        metric_id, name, outcome.value, unit, family=spec.family, sample_size=sequence.length,
        min_sample=min_length, distribution=distribution,
        warning=_join(sequence.warning, outcome.warning),
        sample_size_sensitive=outcome.sample_size_sensitive or feature_name in SAMPLE_SIZE_SENSITIVE_FEATURES)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = _settings(config)

    sequences = [name for name in cfg["sequences"] if name in seq.SEQUENCES]
    features = [name for name in cfg["feature_groups"] if name in _FEATURES]
    unknown_sequences = [name for name in cfg["sequences"] if name not in seq.SEQUENCES]
    unknown_features = [name for name in cfg["feature_groups"] if name not in _FEATURES]

    if not sequences or not features:
        problems = []
        if not sequences:
            problems.append("no valid entry in 'sequences'" +
                            (f" (unknown: {unknown_sequences})" if unknown_sequences else ""))
        if not features:
            problems.append("no valid entry in 'feature_groups'" +
                            (f" (unknown: {unknown_features})" if unknown_features else ""))
        return [finding("rhythm.timeseries_config", "Time-series suite configuration",
                        None, None, family=FAMILY,
                        warning="; ".join(problems))]

    combos = [(sequence_name, feature_name) for sequence_name in sequences
             for feature_name in features]
    max_findings = max(1, cfg["max_findings"])
    truncated = len(combos) > max_findings
    combos = combos[:max_findings]

    out = [_measure_one(analysis, sequence_name, feature_name, cfg)
          for sequence_name, feature_name in combos]

    if unknown_sequences or unknown_features:
        note = ", ".join(filter(None, [
            f"unknown sequences ignored: {unknown_sequences}" if unknown_sequences else "",
            f"unknown feature_groups ignored: {unknown_features}" if unknown_features else "",
        ]))
        out.append(finding("rhythm.timeseries_config", "Time-series suite configuration",
                           len(combos), "combinations", family=FAMILY, warning=note))
    if truncated:
        out.append(finding(
            "rhythm.timeseries_truncated", "Time-series suite output truncated by max_findings",
            max_findings, "findings", family=FAMILY,
            warning=f"{len(sequences) * len(features)} (sequence, feature) combinations were "
                    f"selected but only max_findings={max_findings} were computed; raise "
                    f"metrics.timeseries_suite.max_findings, or narrow 'sequences' or "
                    f"'feature_groups', to see the rest"))
    return out
