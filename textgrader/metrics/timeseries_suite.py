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
features (43 groups: the original 15, catch22's 22, catch22's two catch24
extras, the two wavelet groups, the textdescriptives cross-check, and one
configurable ``tsfresh`` group added this pass) is 688 possible findings
before a single lag or box size is counted -- bigger again than the 640 the
previous pass left this docstring warning about, precisely because this pass
made two more feature families real. Five things bound it: (1) every feature
group is reported as *one* finding per sequence, with its secondary numbers
(all lags, all box sizes, segment slopes, per-scale wavelet energies, every
tsfresh sub-feature, ...) folded into that finding's ``distribution`` rather
than exploded into their own ids -- catch22 (and, by extension, the two
catch24 extras) is the one deliberate exception, because the task asked for
each of catch22's 22 features under its own stable id rather than folded
together, so it is 22 (+2) *feature groups*, not one group with 22 (+2)
numbers inside it; ``tsfresh`` is folded the *other* way on purpose --
``comprehensive`` alone can extract nearly 800 numbers from one sequence, and
turning each into its own id would make this suite's own combinatorics
problem the thing it most needed guarding against, so ``tsfresh`` is always
exactly one finding per sequence no matter which preset is selected, with
``tsfresh_max_features`` bounding how many of its sub-values are kept in that
one finding's ``distribution``; (2) the default configuration is **unchanged
by this pass on purpose**: it still selects the same three dependency-free,
always-available sequences and the same five low-assumption feature groups
it always has (15 findings by default, see ``DEFAULT_SEQUENCES``/
``DEFAULT_FEATURE_GROUPS`` below) -- catch22, catch24, the wavelet groups,
``tsfresh`` and the embedding-backed sequences are all opt-in, both because
catch22 alone (22 features) over even the three default sequences would
already be 66 findings, more than four times the current default, and
because two of those sequences are embedding-backed and must never turn on
by themselves (see "Why the embedding sequences stay out of every default"
below); (3) ``max_findings`` (default 200) is a hard stop -- a user who
selects every sequence and every feature group gets the first 200
combinations, in the order they configured, plus one finding that says so,
rather than a silent multi-thousand-row report; (4) a ``"catch22"``
shorthand in ``feature_groups`` expands to all 22 ``catch22_*`` names, and a
``"catch24"`` shorthand expands to those same 22 plus the two catch24-only
extras (24 total), each deduplicated against anything already listed, so a
user does not have to type 22 or 24 strings to turn the whole group on, but
either is still exactly as bounded by ``max_findings`` as if they had; (5)
``tsfresh`` itself is off by default and, even when selected, defaults to
its ``minimal`` preset (10 fixed, hand-named sub-features) rather than
``efficient``/``comprehensive`` -- see ``tsfresh_feature_set`` below.

**Why the embedding sequences stay out of every default.**  This suite is
``cost="moderate"`` with ``requires=()``, which is what lets the corpus
builder profile it over every reference book (see
``textgrader/corpus.py``'s ``needs_parse``/``needs_model`` gate). That gate
only checks ``"sentence_transformers" in requires``, so it cannot see that
two sequences in the registry this suite draws from
(``sentence_similarity_prev``, ``sentence_distance_centroid``) load a
sentence-embedding model. Adding ``sentence_transformers`` to ``REQUIRES``
would fix that for free but would also flip this suite's own
``needs_model`` to true and silently drop its other fourteen, cheap
sequences out of corpus profiling too -- a real regression this pass
chooses not to make quietly. (The more honest long-term fix is a finer-
grained gate -- one that asked "does this specific configuration need a
model" rather than "does this metric's ``requires`` tuple ever mention one"
-- but that is a change to ``textgrader/corpus.py``'s gating rule itself,
outside every file this pass is allowed to touch, so it is left as an
argument for whoever owns that module rather than made quietly here.)
Instead, ``DEFAULT_SEQUENCES`` simply never contains an embedding-backed
sequence, and no code path in this module imports or loads an embedding
model unless one of those two sequences is explicitly present in a caller's
``sequences`` list (``sequences.get_sequence`` only reaches
``semantic_adjacent``'s model loader for those two names, and only when
asked for by name) -- so corpus profiling, which always runs with
``MetricSpec.defaults``, never touches it.

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

**Deferred.**  ``pycatch22``, ``sentence-transformers``, ``PyWavelets`` and
(this pass) ``tsfresh`` were not installed when this suite was first
written, so ``catch22``, the two embedding-backed sequences, wavelet
features and any tsfresh channel were either missing or running only on a
fallback whose success path had never executed. All four are installed now
and every pass since has wired one more in and validated the real path, not
just the fallback -- see ``catch22_*`` below (22 canonical features, Lubba
et al. 2019, each under its own stable id per the task's naming rule),
``catch24_raw_mean``/``catch24_raw_variance`` (the two extra raw, non-
z-scored features ``catch22_all(catch24=True)`` adds on top of the same 22 --
see "catch22 / catch24" below for why these were reversed into this suite
rather than left out), ``wavelet_energy``/``wavelet_entropy`` (energy per
scale and the Shannon entropy of that distribution, ``PyWavelets``),
``tsfresh`` (a configurable, capped tsfresh feature set -- see "tsfresh"
below) and the sequence registry's own docstring for the embedding-sequence
validation.

``catch24`` was skipped in an earlier pass on the reasoning that its two
extra features (a raw mean and a raw variance) would only duplicate this
suite's own ``dispersion`` group under a different id. That reasoning was
overruled: this project's own philosophy is that a metric is a sensor, not
an opinion, and two channels agreeing -- even two that are *expected* to
agree closely, because they compute a textbook quantity two different ways
-- is a cross-check worth having, not a duplicate to delete. Both new ids
say so explicitly in their own ``distribution["relationship_note"]``, naming
exactly which existing finding they are expected to track (``dispersion``'s
``mean``/``std`` for the same sequence) and why the numbers should still be
computed independently rather than one copied from the other. See "catch22 /
catch24" below for the two features' exact identity and unit.

Recurrence-quantification features (``PyRQA``) are left out because the
package is still not installed here, so its success path still cannot be
exercised or validated, and this suite would rather ship nothing for it than
ship code nobody has run. A streaming ``river`` ADWIN detector is skipped in
favour of a small dependency-free Page-Hinkley implementation
(``page_hinkley``, below), which needs no optional package and is directly
testable against a synthetic step-shift sequence. Sentiment/emotion scoring
and topic-probability sequences are not in this module at all; see
``textgrader/sequences.py``'s docstring for why they were left out of the
sequence registry itself, one level down. A ``textdescriptives`` cross-check
is wired in for exactly the one sequence where it overlaps this suite's own
work end to end (``sentence_dependency_distance`` -- see
``textdescriptives_check`` below); ``textdescriptives``'
``readability``/``information_theory``/``quality`` components measure
whole-document properties this suite's per-sentence, ordered-sequence
contract does not cover; wiring in every component that merely *touches* a
related idea would not be a cross-check, only noise.

A per-book corpus-reference channel for sequence shape (caching this
suite's own summary of, say, ``sentence_words`` per book under
``profile_vector`` so a manuscript's dispersion or spectral shape could be
read as a corpus z-score, the way ``stylometry_suite``'s
``corpus_reference`` already does for function words) was considered and
deliberately not added this pass. Every other feature group here already
folds unboundedly into ``distribution`` rather than the corpus profile, this
suite's ``max_findings`` guard exists specifically because its own
combinatorics are already this pass's largest risk, and a genuinely useful
corpus-reference channel needs its own settings-comparability handling
(matching ``sequences``/``feature_groups``/``detrend`` between the profile
that built a cached vector and the manuscript being compared against it, the
same care ``stylometry_suite`` already takes for function words) rather than
a same-pass bolt-on. Left as a real, scoped candidate for a future pass
rather than shipped half-considered here.

Deterministic settings throughout: no random seeds are needed because every
feature here is a closed-form or exact-recursion computation, not a fit with
random initialization. The ``ruptures``-backed ``change_points`` feature's
PELT search is the one genuine exception in spirit (it is a search, not a
closed form) and it is itself deterministic -- but until this pass installed
``ruptures``, that search had never actually run in this environment; only
its single-split fallback had. Running it for the first time found a real
bug: the fixed ``change_point_penalty=3.0`` this module shipped with was
tuned against nothing (the fallback code path that used it never existed --
the fallback ignores ``change_point_penalty`` entirely), and on standardized
values it made PELT flag roughly 1-2% of points as change points in *pure
white noise*, worse the longer the sequence ran (2 spurious points at
n=200, 7 at n=500, 17 at n=1,000, 48 at n=3,000 on one fixed seed), because
a flat penalty does not grow with the number of candidate split points a
longer sequence offers PELT to overfit. ``change_point_penalty`` is now a
multiplier of ``log(n)`` (default 2.0, the standard BIC-style penalty for a
Gaussian mean-shift model on unit-variance data) rather than a flat score,
which measured near zero false positives across lengths from 80 to 1,200 in
a 30-seed check while still finding this module's own synthetic level-shift
test within a few points of where it was injected.
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

#: The original 15 feature groups, each a self-contained ``(values, cfg) ->
#: _Outcome`` function needing nothing but the sequence's own numbers.  The
#: three families this pass added (catch22, wavelets, the textdescriptives
#: cross-check) need more context -- the analysis, the sequence name, whether
#: detrending is on -- than that signature carries, so they are registered
#: separately in ``_EXTENDED_FEATURES`` below rather than forced into it.
#: ``FEATURE_NAMES`` (further down) is the union callers should read.
_BASE_FEATURE_NAMES = (
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
    """Change-point rate via ``ruptures``' PELT search, or a single-split fallback.

    ``change_point_penalty`` is a *multiplier of* ``log(n)``, not a flat PELT
    penalty. That is a deliberate fix, not the original design: this suite
    shipped for a long time with a flat ``pen=3.0`` that only the fallback's
    code path defined a default for (the fallback itself never reads
    ``change_point_penalty`` at all), because ``ruptures`` was not installed
    anywhere this suite had run. The first time it actually ran, a flat
    penalty let PELT overfit noise -- worse the longer the sequence, because
    a fixed score does not keep pace with how many candidate split points a
    longer sequence gives PELT to search: on pure white noise it fired on
    roughly 1.5% of points regardless of length (2 events at n=200, 7 at
    n=500, 17 at n=1,000, 48 at n=3,000, one fixed seed). Scaling the penalty
    by ``log(n)`` (the standard BIC-style penalty for a Gaussian mean-shift
    model on unit-variance data) is the classical fix and measured near zero
    false positives from n=80 to n=1,200 across 30 seeds in this module's own
    validation, while still finding an injected level shift within a few
    points of where it happened.
    """

    n = len(values)
    penalty_multiplier = float(cfg.get("change_point_penalty", 2.0))
    ruptures_module, reason = require("ruptures")
    numpy, numpy_reason = require("numpy")
    if ruptures_module is not None and numpy is not None:
        try:
            arr = numpy.asarray(values, dtype=float).reshape(-1, 1)
            std = float(arr.std())
            if std > 0:
                arr = (arr - arr.mean()) / std
            penalty = penalty_multiplier * math.log(max(n, 2))
            algo = ruptures_module.Pelt(model="l2").fit(arr)
            breakpoints = [point for point in algo.predict(pen=penalty) if point < n]
            rate = 100.0 * len(breakpoints) / n
            return _Outcome(rate, distribution={
                "backend": "ruptures.Pelt", "library_version": _lib_version(ruptures_module),
                "count": len(breakpoints), "locations": breakpoints[:25],
                "penalty": penalty, "penalty_multiplier": penalty_multiplier})
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

assert set(_FEATURES) == set(_BASE_FEATURE_NAMES)


# ------------------------------------------------------- catch22 / wavelets /
# ------------------------------------------------------- textdescriptives
#
# These three families need more than a sequence's own numbers: catch22 and
# the wavelet decomposition are each one shared, potentially expensive
# computation that several of this module's feature *names* read from (22
# catch22 features from one ``catch22_all`` call; energy and entropy from one
# wavelet decomposition), and the textdescriptives cross-check needs the
# document's shared spaCy parse, not just a sequence of floats. None of that
# fits the plain ``(values, cfg) -> _Outcome`` signature every feature above
# uses, so each gets a small ``ctx`` (analysis, sequence name, detrending
# flag, the raw config) instead, and is registered in ``_EXTENDED_FEATURES``
# rather than ``_FEATURES``. ``_measure_one`` dispatches to whichever dict
# has the requested name.


def _extra_settings_key(sequence_name: str, cfg: Mapping[str, Any]) -> str:
    extra = _sequence_settings(sequence_name, cfg)
    return ",".join(f"{key}={extra[key]}" for key in sorted(extra))


# ---- catch22 -----------------------------------------------------------

#: (library feature name, this suite's stable id suffix, unit, description).
#: Keyed by the *library's* name so a value from a future pycatch22 release
#: can be matched to our own id by name, not by position -- catch22_all
#: returns a positional list, and the task's own stable-id rule forbids
#: exposing either that library name or a bare position as a metric id, so
#: this table is the one place a name from either world is allowed to touch
#: the other. Every one of catch22's 22 features is scale- and offset-free by
#: construction (the library z-scores its input before computing all but a
#: couple of them), so none of these fall back to the sequence's own unit the
#: way ``dispersion`` does; each has its own fixed unit below.
_CATCH22_CATALOGUE: tuple[tuple[str, str, str, str], ...] = (
    ("DN_HistogramMode_5", "histogram_mode_5bin", "z-score",
     "Mode of a 5-bin histogram of the (z-scored) values"),
    ("DN_HistogramMode_10", "histogram_mode_10bin", "z-score",
     "Mode of a 10-bin histogram of the (z-scored) values"),
    ("CO_f1ecac", "acf_1e_decay_time", "lag steps",
     "First lag at which the autocorrelation function crosses 1/e"),
    ("CO_FirstMin_ac", "acf_first_minimum", "lag steps",
     "First minimum of the autocorrelation function"),
    ("CO_HistogramAMI_even_2_5", "auto_mutual_info_2bin_lag5", "bits",
     "Automutual information at lag 5, 2-bin histogram estimator"),
    ("CO_trev_1_num", "time_reversal_asymmetry", "ratio",
     "Time-reversal asymmetry statistic at lag 1"),
    ("MD_hrv_classic_pnn40", "large_step_fraction", "proportion",
     "Proportion of successive differences exceeding 0.4 standard deviations (pNN40)"),
    ("SB_BinaryStats_mean_longstretch1", "longest_above_mean_run", "points",
     "Longest run of consecutive values above the mean"),
    ("SB_TransitionMatrix_3ac_sumdiagcov", "transition_matrix_trace", "ratio",
     "Trace of the covariance of a 3-letter symbolized transition matrix"),
    ("PD_PeriodicityWang_th0_01", "periodicity_wang", "lag steps",
     "Wang et al.'s periodicity-detection lag"),
    ("CO_Embed2_Dist_tau_d_expfit_meandiff", "embed2_dist_expfit", "ratio",
     "Goodness of an exponential fit to 2D time-delay-embedding point distances"),
    ("IN_AutoMutualInfoStats_40_gaussian_fmmi", "auto_mutual_info_first_min", "lag steps",
     "First minimum of the automutual information function (Gaussian estimator)"),
    ("FC_LocalSimple_mean1_tauresrat", "forecast_ar1_error_ratio", "ratio",
     "Change in autocorrelation timescale after a 1-step local-mean forecast"),
    ("DN_OutlierInclude_p_001_mdrmd", "outlier_timing_positive", "fraction of series",
     "Timing of positive outliers relative to the series"),
    ("DN_OutlierInclude_n_001_mdrmd", "outlier_timing_negative", "fraction of series",
     "Timing of negative outliers relative to the series"),
    ("SP_Summaries_welch_rect_area_5_1", "spectral_low_freq_power", "power ratio",
     "Share of power in the lowest-frequency fifth of the Welch spectrum"),
    ("SB_BinaryStats_diff_longstretch0", "longest_decreasing_run", "points",
     "Longest run of consecutive decreases"),
    ("SB_MotifThree_quantile_hh", "motif3_entropy", "bits",
     "Shannon entropy of a 3-letter symbolic-word distribution"),
    ("SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1", "rs_range_low_scale_fit", "ratio",
     "Rescaled-range fluctuation-analysis fit, low-scale proportion"),
    ("SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "dfa_low_scale_fit", "ratio",
     "Detrended-fluctuation-analysis fit, low-scale proportion"),
    ("SP_Summaries_welch_rect_centroid", "spectral_centroid", "radians per sample",
     "Centroid frequency of the Welch power spectrum"),
    ("FC_LocalSimple_mean3_stderr", "forecast_ma3_error_stderr", "ratio",
     "Residual standard error of a 3-step local-mean forecast"),
)

_CATCH22_ALIAS = "catch22"
_CATCH22_FEATURE_NAMES = tuple(f"catch22_{stable}" for _, stable, _, _ in _CATCH22_CATALOGUE)
_CATCH22_LIBRARY_NAME_BY_FEATURE = {f"catch22_{stable}": library
                                    for library, stable, _, _ in _CATCH22_CATALOGUE}
#: Minimum catch22 needs before its features stop occasionally returning a
#: non-finite value on this suite's own tiny-document test fixtures (a
#: three-point series already produced one NaN in manual testing); 30 is the
#: same order of magnitude as this module's other nonlinear/scaling minimums
#: (``hurst``/``dfa`` at 40) rather than a value from the catch22 paper.
_CATCH22_MIN_LENGTH = 30


def _catch22_result(ctx: Mapping[str, Any]) -> tuple[dict[str, float] | None, str | None, Any]:
    """The one shared ``pycatch22.catch22_all`` call every catch22/catch24 feature reads from.

    Always requests ``catch24=True``: on top of the same 22 canonical
    features, that costs nothing extra (pycatch22 computes the raw mean and
    standard deviation as a side effect of z-scoring its input either way)
    and is what lets ``catch24_raw_mean``/``catch24_raw_variance`` below read
    from this exact cache instead of a second library call -- one book's
    call to this function, from any of the 24 features, is one call to the
    library, not 22 or 24.
    """

    analysis: DocumentAnalysis = ctx["analysis"]
    key = (f"timeseries_suite:catch22:{ctx['sequence_name']}:"
           f"{_extra_settings_key(ctx['sequence_name'], ctx['cfg'])}:detrend={ctx['detrended']}")

    def build() -> tuple[dict[str, float] | None, str | None, Any]:
        module, reason = require("pycatch22")
        if module is None:
            return None, reason, None
        try:
            result = module.catch22_all(list(ctx["working"]), catch24=True)
            return dict(zip(result["names"], result["values"])), None, module
        except Exception as exc:  # pragma: no cover - library/runtime guard
            return None, f"pycatch22.catch22_all failed ({type(exc).__name__}: {exc})", None

    return analysis.memo(key, build)


def _feature_catch22(feature_name: str, values: TypingSequence[float], cfg: Mapping[str, Any],
                     ctx: Mapping[str, Any] | None = None) -> _Outcome:
    if ctx is None:
        return _Outcome(None, warning="catch22 features need suite context and cannot run standalone")
    values_by_library_name, reason, module = _catch22_result(ctx)
    if values_by_library_name is None:
        return _Outcome(None, warning=reason)
    library_name = _CATCH22_LIBRARY_NAME_BY_FEATURE[feature_name]
    if library_name not in values_by_library_name:
        return _Outcome(None, warning=f"pycatch22 did not return {library_name!r}; the installed "
                                      f"pycatch22 version may not match what this suite expects")
    raw = values_by_library_name[library_name]
    if raw is None or not math.isfinite(raw):
        return _Outcome(None, warning="pycatch22 returned a non-finite value for this sequence, "
                                      "most often because it has zero variance")
    return _Outcome(float(raw), distribution={
        "catch22_feature": library_name, "backend": "pycatch22.catch22_all",
        "library": "pycatch22", "library_version": _lib_version(module)})


def _make_catch22_feature(feature_name: str):
    def feature(values: TypingSequence[float], cfg: Mapping[str, Any],
               ctx: Mapping[str, Any] | None = None) -> _Outcome:
        return _feature_catch22(feature_name, values, cfg, ctx)
    feature.__name__ = f"_feature_{feature_name}"
    return feature


# ---- catch24 (catch22's two raw, non-z-scored extras) -------------------
#
# ``catch22_all(catch24=True)`` adds exactly two features on top of the same
# 22: the raw arithmetic mean and the raw sample standard deviation of the
# *un*-z-scored input (every one of the 22 above is scale- and offset-free by
# construction; these two deliberately are not). An earlier pass of this
# suite left catch24 out on the reasoning that both numbers would only
# duplicate ``dispersion``'s own ``mean``/``std`` for the same sequence under
# a different id. That reasoning was overruled: correlated measurements are
# not a bug in this project's design, they are two independent instruments
# reading the same thing, and a metric here is a sensor, not an opinion. Each
# finding below names the sibling it is expected to agree with, in its own
# ``distribution["relationship_note"]``, rather than pretending the overlap
# does not exist.
#
# ``catch24_raw_variance`` is a genuine, if small, departure from a bare
# passthrough: pycatch22 reports ``DN_Spread_Std``, a sample standard
# deviation (``ddof=1``, matching ``textgrader.stats.summarize``'s own
# ``std``), not a variance -- squaring it here is what makes the stable id
# say what it actually holds, per the raw request for "raw mean and
# variance", rather than silently relabeling a standard deviation as one.
_CATCH24_ALIAS = "catch24"
_CATCH24_EXTRAS: tuple[tuple[str, str, str, str], ...] = (
    ("DN_Mean", "catch24_raw_mean", "Raw mean (not z-scored)",
     "Arithmetic mean of the raw sequence values, before any z-scoring -- "
     "closely related to this suite's own dispersion.mean for the same "
     "sequence (both are the textbook arithmetic mean); kept as its own id "
     "and computed independently rather than copied, per this project's "
     "sensor-not-opinion policy on correlated measurements."),
    ("DN_Spread_Std", "catch24_raw_variance", "Raw variance (not z-scored)",
     "Sample variance (the square of pycatch22's raw, non-z-scored sample "
     "standard deviation, ddof=1) of the sequence values -- closely related "
     "to the square of this suite's own dispersion.std for the same "
     "sequence (same ddof=1 convention); kept as its own id for the same "
     "reason as catch24_raw_mean above."),
)
_CATCH24_FEATURE_NAMES = tuple(stable for _, stable, _, _ in _CATCH24_EXTRAS)
_CATCH24_LIBRARY_NAME_BY_FEATURE = {stable: library for library, stable, _, _ in _CATCH24_EXTRAS}
_CATCH24_DESCRIPTION_BY_FEATURE = {stable: note for _, stable, _, note in _CATCH24_EXTRAS}
#: catch24's two extras are ordinary summary statistics (a mean, a variance),
#: not nonlinear/scaling estimates, so they need nothing like catch22's own
#: 30-point floor to be *meaningful* -- but they still run through the same
#: shared ``catch22_all`` call, which pycatch22 has been observed (see
#: ``_CATCH22_MIN_LENGTH``'s own note) to mishandle at the very smallest
#: inputs, so this stays at ``dispersion``'s own floor rather than at 2 or 3.
_CATCH24_MIN_LENGTH = 5


def _feature_catch24(feature_name: str, values: TypingSequence[float], cfg: Mapping[str, Any],
                     ctx: Mapping[str, Any] | None = None) -> _Outcome:
    if ctx is None:
        return _Outcome(None, warning="catch24 features need suite context and cannot run standalone")
    values_by_library_name, reason, module = _catch22_result(ctx)
    if values_by_library_name is None:
        return _Outcome(None, warning=reason)
    library_name = _CATCH24_LIBRARY_NAME_BY_FEATURE[feature_name]
    if library_name not in values_by_library_name:
        return _Outcome(None, warning=f"pycatch22 did not return {library_name!r} (a catch24 extra); "
                                      f"the installed pycatch22 version may not support catch24")
    raw = values_by_library_name[library_name]
    if raw is None or not math.isfinite(raw):
        return _Outcome(None, warning="pycatch22 returned a non-finite value for this sequence, "
                                      "most often because it has zero variance")
    if feature_name == "catch24_raw_variance":
        value = float(raw) ** 2
        backend_note = "pycatch22.catch22_all(catch24=True), DN_Spread_Std squared into a variance"
    else:
        value = float(raw)
        backend_note = "pycatch22.catch22_all(catch24=True)"
    return _Outcome(value, distribution={
        "catch22_feature": library_name, "backend": backend_note,
        "library": "pycatch22", "library_version": _lib_version(module),
        "relationship_note": _CATCH24_DESCRIPTION_BY_FEATURE[feature_name]})


def _make_catch24_feature(feature_name: str):
    def feature(values: TypingSequence[float], cfg: Mapping[str, Any],
               ctx: Mapping[str, Any] | None = None) -> _Outcome:
        return _feature_catch24(feature_name, values, cfg, ctx)
    feature.__name__ = f"_feature_{feature_name}"
    return feature


# ---- wavelets -----------------------------------------------------------

_WAVELET_MIN_LENGTH = 40


def _wavelet_result(ctx: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    analysis: DocumentAnalysis = ctx["analysis"]
    wavelet_name = str(ctx["cfg"].get("wavelet_name", "db4"))
    max_level_cfg = int(ctx["cfg"].get("wavelet_max_level", 5))
    key = (f"timeseries_suite:wavelet:{ctx['sequence_name']}:"
           f"{_extra_settings_key(ctx['sequence_name'], ctx['cfg'])}:detrend={ctx['detrended']}:"
           f"wavelet={wavelet_name}:max_level={max_level_cfg}")

    def build() -> tuple[dict[str, Any] | None, str | None]:
        pywt_module, reason = require("pywt")
        if pywt_module is None:
            return None, reason
        numpy_module, numpy_reason = require("numpy")
        if numpy_module is None:
            return None, numpy_reason
        try:
            wavelet_obj = pywt_module.Wavelet(wavelet_name)
        except Exception as exc:
            return None, f"unknown wavelet {wavelet_name!r} ({type(exc).__name__}: {exc})"
        n = len(ctx["working"])
        max_possible = pywt_module.dwt_max_level(n, wavelet_obj.dec_len)
        if max_possible < 1:
            return None, (f"needs more points for even a level-1 {wavelet_name!r} decomposition "
                          f"(has {n})")
        level = min(max_possible, max(1, max_level_cfg))
        try:
            arr = numpy_module.asarray(ctx["working"], dtype=float)
            coeffs = pywt_module.wavedec(arr, wavelet_name, level=level)
            energies = [float(numpy_module.sum(numpy_module.asarray(band, dtype=float) ** 2))
                       for band in coeffs]
        except Exception as exc:  # pragma: no cover - library/runtime guard
            return None, f"pywt.wavedec failed ({type(exc).__name__}: {exc})"
        band_labels = [f"cA{level}"] + [f"cD{lvl}" for lvl in range(level, 0, -1)]
        return {"energies": energies, "band_labels": band_labels, "level": level,
                "wavelet": wavelet_name, "library_version": _lib_version(pywt_module)}, None

    return analysis.memo(key, build)


def _feature_wavelet_energy(values: TypingSequence[float], cfg: Mapping[str, Any],
                            ctx: Mapping[str, Any] | None = None) -> _Outcome:
    if ctx is None:
        return _Outcome(None, warning="wavelet features need suite context and cannot run standalone")
    result, reason = _wavelet_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    energies = result["energies"]
    total = sum(energies)
    if total <= 0:
        return _Outcome(None, warning="the sequence has no variance at any wavelet scale; "
                                      "energy share is undefined", sample_size_sensitive=True)
    return _Outcome(100.0 * energies[0] / total, distribution={
        "energies": energies, "energy_share_percent": [100.0 * e / total for e in energies],
        "band_labels": result["band_labels"], "level": result["level"], "wavelet": result["wavelet"],
        "library": "PyWavelets", "library_version": result["library_version"]},
        sample_size_sensitive=True)


def _feature_wavelet_entropy(values: TypingSequence[float], cfg: Mapping[str, Any],
                             ctx: Mapping[str, Any] | None = None) -> _Outcome:
    if ctx is None:
        return _Outcome(None, warning="wavelet features need suite context and cannot run standalone")
    result, reason = _wavelet_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    energies = result["energies"]
    total = sum(energies)
    common = {"energies": energies, "band_labels": result["band_labels"], "level": result["level"],
             "wavelet": result["wavelet"], "library": "PyWavelets",
             "library_version": result["library_version"]}
    if total <= 0:
        return _Outcome(0.0, distribution=common, sample_size_sensitive=True,
                        warning="the sequence has no variance at any wavelet scale; wavelet "
                                "entropy is undefined so 0.0 (all energy in one band) was used")
    p = [e / total for e in energies if e > 0]
    raw_entropy = -sum(pi * math.log2(pi) for pi in p)
    max_entropy = math.log2(len(energies)) if len(energies) > 1 else 1.0
    normalized = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    return _Outcome(normalized, distribution={**common, "raw_entropy_bits": raw_entropy},
                    sample_size_sensitive=True)


# ---- textdescriptives cross-check ---------------------------------------

#: Sequences this suite already builds that a textdescriptives component
#: also measures, independently, off the same spaCy parse. Extend this (and
#: the dispatch in ``_feature_textdescriptives_check``) if a future sequence
#: gains a second textdescriptives overlap; today there is exactly one.
_TEXTDESCRIPTIVES_CROSS_CHECK_SEQUENCES = frozenset({"sentence_dependency_distance"})


def _feature_textdescriptives_check(values: TypingSequence[float], cfg: Mapping[str, Any],
                                    ctx: Mapping[str, Any] | None = None) -> _Outcome:
    if ctx is None:
        return _Outcome(None, warning="the textdescriptives cross-check needs suite context and "
                                      "cannot run standalone")
    sequence_name = ctx["sequence_name"]
    if sequence_name not in _TEXTDESCRIPTIVES_CROSS_CHECK_SEQUENCES:
        return _Outcome(None, warning=f"no textdescriptives cross-check is defined for the "
                                      f"{sequence_name!r} sequence (only "
                                      f"{sorted(_TEXTDESCRIPTIVES_CROSS_CHECK_SEQUENCES)} have one)")
    analysis: DocumentAnalysis = ctx["analysis"]
    if analysis.nlp_unavailable:
        return _Outcome(None, warning=analysis.nlp_unavailable)
    td_module, reason = require("textdescriptives")
    if td_module is None:
        return _Outcome(None, warning=reason)
    try:
        from textdescriptives.components.dependency_distance import DependencyDistance
        # Registers Doc/Span/Token extensions as getters; idempotent, and it
        # needs no pipeline component added, so it runs on the shared parse
        # this document already built rather than re-parsing.
        DependencyDistance(analysis.nlp)
        # Recomputed here, in the same loop as textdescriptives' own value,
        # rather than read from ``values``/``ctx["sequence"]``: this suite's
        # own sentence_dependency_distance sequence skips a sentence with no
        # non-ROOT token (a one-word sentence), and textdescriptives does
        # not, so pulling the two from separately-built lists produced a
        # spurious count mismatch on ordinary prose in manual testing. Pairing
        # them sentence-by-sentence in one pass guarantees they line up.
        ours: list[float] = []
        other: list[float] = []
        for _, doc in analysis.spacy_docs():
            for sent in doc.sents:
                if len(sent) == 0:
                    continue
                distances = [abs(token.i - token.head.i) for token in sent if token.dep_ != "ROOT"]
                if not distances:
                    continue
                ours.append(statistics.fmean(distances))
                other.append(float(sent._.dependency_distance["dependency_distance_mean"]))
    except Exception as exc:  # pragma: no cover - library/runtime guard
        return _Outcome(None, warning=f"textdescriptives dependency-distance cross-check failed "
                                      f"({type(exc).__name__}: {exc})")
    if len(ours) < 5:
        return _Outcome(None, warning="fewer than five sentences; the cross-check needs more to "
                                      "compare")
    _, _, r = _ols(ours, other)
    mean_abs_gap = statistics.fmean(abs(a - b) for a, b in zip(ours, other))
    return _Outcome(r, distribution={
        "backend": "textdescriptives.dependency_distance", "library": "textdescriptives",
        "library_version": _lib_version(td_module), "mean_absolute_gap": mean_abs_gap,
        "own_mean": statistics.fmean(ours), "textdescriptives_mean": statistics.fmean(other),
        "note": "textdescriptives' per-sentence mean includes the ROOT token (distance 0); this "
                "suite's own sentence_dependency_distance sequence excludes it, so a nonzero "
                "mean_absolute_gap here reflects that definitional difference, not disagreement "
                "about the underlying parse"})


# ---- tsfresh --------------------------------------------------------------
#
# ``tsfresh`` ships three fixed presets (``MinimalFCParameters``,
# ``EfficientFCParameters``, ``ComprehensiveFCParameters``), each a growing
# catalogue of calculators; some calculators are unparameterized (``mean``,
# ``length``) and some expand into many columns (``ar_coefficient`` alone
# produces one column per lag it was asked for). ``minimal`` is exactly 10
# fixed, unparameterized columns -- small and stable enough to hand-name
# every one of them the way catch22's 22 are hand-named above.
# ``efficient``/``comprehensive`` run to the high hundreds of columns
# (measured: 777 and 783 respectively over a 300-point synthetic series),
# far too many to hand-name individually without the table itself becoming a
# second thing to keep in sync with every future tsfresh release; those two
# keep tsfresh's own descriptive calculator name (never a plain integer or
# array position -- tsfresh does not have positional output at all, every
# column is already a semantic string like ``"abs_energy"`` or
# ``"number_peaks__n_5"``) with its ``"__"`` parameter separator flattened to
# a single ``"_"`` so nothing here repeats tsfresh's own internal formatting
# convention verbatim.
#
# Regardless of preset, this suite reports exactly **one** finding per
# sequence for the whole ``tsfresh`` feature group -- not one id per
# sub-feature the way catch22 does -- specifically so that selecting
# ``comprehensive`` cannot multiply this suite's already-guarded
# combinatorics by another 780. ``tsfresh_max_features`` caps how many of
# that one finding's sub-values are kept in its ``distribution`` (sorted by
# name for a deterministic, reproducible subset); the headline value is the
# percentage of the *full, uncapped* preset that came back finite, which
# stays informative (a genuine data-quality signal: how much of what was
# asked for could actually be computed on a sequence this length) regardless
# of how large the underlying preset is or how many of its values got
# truncated out of the reported subset.
_TSFRESH_PRESETS = {"minimal": "MinimalFCParameters", "efficient": "EfficientFCParameters",
                    "comprehensive": "ComprehensiveFCParameters"}
#: Ordinary summary statistics (min/max/rms) need very little data to be
#: defined, but several efficient/comprehensive calculators (FFT bins, AR
#: coefficients up to lag 10, agg_linear_trend chunks) are only meaningful
#: with materially more; 20 matches this module's acf/pacf floor rather than
#: a value from tsfresh's own documentation, which does not specify one.
_TSFRESH_MIN_LENGTH = 20
#: Hand-named because ``minimal`` is small and fixed (10 unparameterized
#: columns; see the section note above) -- the same reasoning that gives
#: catch22's 22 features their own table rather than tsfresh's raw names.
_TSFRESH_MINIMAL_LABELS = {
    "sum_values": "Sum", "median": "Median", "mean": "Mean", "length": "Length",
    "standard_deviation": "Standard deviation", "variance": "Variance",
    "root_mean_square": "Root mean square", "maximum": "Maximum",
    "absolute_maximum": "Maximum absolute value", "minimum": "Minimum",
}


def _tsfresh_result(ctx: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    analysis: DocumentAnalysis = ctx["analysis"]
    feature_set = str(ctx["cfg"].get("tsfresh_feature_set", "minimal"))
    key = (f"timeseries_suite:tsfresh:{ctx['sequence_name']}:"
           f"{_extra_settings_key(ctx['sequence_name'], ctx['cfg'])}:detrend={ctx['detrended']}:"
           f"feature_set={feature_set}")

    def build() -> tuple[dict[str, Any] | None, str | None]:
        preset_name = _TSFRESH_PRESETS.get(feature_set)
        if preset_name is None:
            return None, (f"unknown tsfresh_feature_set {feature_set!r}; use one of "
                          f"{sorted(_TSFRESH_PRESETS)}")
        tsfresh_module, reason = require("tsfresh")
        if tsfresh_module is None:
            return None, reason
        pandas_module, pandas_reason = require("pandas")
        if pandas_module is None:
            return None, pandas_reason
        try:
            from tsfresh.feature_extraction import extract_features
            from tsfresh.feature_extraction import settings as tsfresh_settings
            fc_parameters = getattr(tsfresh_settings, preset_name)()
            working_values = list(ctx["working"])
            frame = pandas_module.DataFrame({
                "id": [0] * len(working_values), "time": range(len(working_values)),
                "value": working_values})
            # tsfresh's own calculators routinely divide by zero or take the
            # log of zero on a constant or very short input and warn loudly
            # about it; that is the input's property, not a bug this suite's
            # own run introduced, and the resulting NaN is already handled
            # below by being reported as "not finite" rather than a number.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                extracted = extract_features(
                    frame, column_id="id", column_sort="time", column_value="value",
                    default_fc_parameters=fc_parameters, disable_progressbar=True, n_jobs=0)
        except Exception as exc:  # pragma: no cover - library/runtime guard
            return None, f"tsfresh.extract_features failed ({type(exc).__name__}: {exc})"
        row = extracted.iloc[0].to_dict()
        stable_values: dict[str, float | None] = {}
        for raw_name, raw_value in row.items():
            # tsfresh always names its one column "value__..."; the "value"
            # half is this function's own column choice, not part of the
            # feature's identity, so it is stripped rather than reported.
            suffix = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
            stable_name = suffix.replace("__", "_")
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                numeric = None
            stable_values[stable_name] = numeric if numeric is not None and math.isfinite(numeric) else None
        return {"values": stable_values, "feature_set": feature_set,
                "library_version": _lib_version(tsfresh_module)}, None

    return analysis.memo(key, build)


def _feature_tsfresh(values: TypingSequence[float], cfg: Mapping[str, Any],
                     ctx: Mapping[str, Any] | None = None) -> _Outcome:
    if ctx is None:
        return _Outcome(None, warning="the tsfresh feature set needs suite context and cannot run standalone")
    result, reason = _tsfresh_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    all_values = result["values"]
    total = len(all_values)
    if total == 0:
        return _Outcome(None, warning="tsfresh returned no columns for the selected feature set")
    finite_names = sorted(name for name, value in all_values.items() if value is not None)
    finite_share = 100.0 * len(finite_names) / total
    max_reported = max(1, int(cfg.get("tsfresh_max_features", 25)))
    reported_names = sorted(all_values)[:max_reported]
    reported = {name: all_values[name] for name in reported_names}
    labels = ({_TSFRESH_MINIMAL_LABELS[name] for name in reported_names if name in _TSFRESH_MINIMAL_LABELS}
             if result["feature_set"] == "minimal" else set())
    return _Outcome(finite_share, distribution={
        "feature_set": result["feature_set"], "requested_features": total,
        "reported_features": len(reported), "truncated_feature_count": max(0, total - len(reported)),
        "finite_feature_count": len(finite_names), "values": reported,
        "labels": sorted(labels) if labels else None,
        "backend": "tsfresh.extract_features", "library": "tsfresh",
        "library_version": result["library_version"]}, sample_size_sensitive=True)


_EXTENDED_FEATURES: dict[str, Callable[..., _Outcome]] = {
    **{feature_name: _make_catch22_feature(feature_name) for feature_name in _CATCH22_FEATURE_NAMES},
    **{feature_name: _make_catch24_feature(feature_name) for feature_name in _CATCH24_FEATURE_NAMES},
    "wavelet_energy": _feature_wavelet_energy,
    "wavelet_entropy": _feature_wavelet_entropy,
    "textdescriptives_check": _feature_textdescriptives_check,
    "tsfresh": _feature_tsfresh,
}

#: Every valid ``feature_groups`` entry: the original 15 plus the 22 catch22
#: features, the 2 catch24 extras, the 2 wavelet features, the 1
#: textdescriptives cross-check and the 1 (internally configurable) tsfresh
#: group. None of the 28 new ones are in ``DEFAULT_FEATURE_GROUPS`` -- see
#: the module docstring's "Guarding the combinatorics" section for why.
FEATURE_NAMES = _BASE_FEATURE_NAMES + tuple(_EXTENDED_FEATURES)

FEATURE_LABELS.update({
    **{name: f"catch22: {desc}" for name, (_, _, _, desc) in
       zip(_CATCH22_FEATURE_NAMES, _CATCH22_CATALOGUE)},
    **{name: f"catch24: {label}" for _, name, label, _ in _CATCH24_EXTRAS},
    "wavelet_energy": "Wavelet energy concentration",
    "wavelet_entropy": "Wavelet entropy",
    "textdescriptives_check": "Agreement with textdescriptives",
    "tsfresh": "tsfresh feature set (configurable preset)",
})

FEATURE_UNITS.update({
    **{f"catch22_{stable}": unit for _, stable, unit, _ in _CATCH22_CATALOGUE},
    "catch24_raw_mean": None,  # falls back to the sequence's own unit, like dispersion
    "catch24_raw_variance": "squared sequence unit",
    "wavelet_energy": "%", "wavelet_entropy": "ratio", "textdescriptives_check": "correlation",
    "tsfresh": "% of requested tsfresh features returned a finite value",
})

DEFAULT_MIN_LENGTHS.update({
    **{name: _CATCH22_MIN_LENGTH for name in _CATCH22_FEATURE_NAMES},
    **{name: _CATCH24_MIN_LENGTH for name in _CATCH24_FEATURE_NAMES},
    "wavelet_energy": _WAVELET_MIN_LENGTH, "wavelet_entropy": _WAVELET_MIN_LENGTH,
    "textdescriptives_check": 5, "tsfresh": _TSFRESH_MIN_LENGTH,
})

# Wavelet decomposition depth (and so the number of bands the energy/entropy
# is spread across) grows with sequence length, exactly the systematic,
# length-linked dependence the module docstring's "Sample-size honesty"
# section describes for hurst/dfa/spectral/runs/permutation_entropy. Several
# tsfresh calculators (FFT bins, AR coefficients, agg_linear_trend chunks)
# share that same dependence for the same reason.
SAMPLE_SIZE_SENSITIVE_FEATURES = SAMPLE_SIZE_SENSITIVE_FEATURES | frozenset(
    {"wavelet_energy", "wavelet_entropy", "tsfresh"})

# A raw-value agreement check against another library's implementation;
# detrending one side and not the other would make the comparison meaningless.
_NEVER_DETREND = _NEVER_DETREND | frozenset({"textdescriptives_check"})

# catch24_raw_mean would be forced to (numerically) zero by detrending -- an
# OLS residual series always has mean ~0 by construction -- which is exactly
# the circularity ``dispersion``/``rolling_dispersion`` are excluded from
# detrending to avoid above; catch24_raw_variance is excluded alongside it so
# the pair keeps behaving like the ``dispersion`` group it is designed to be
# compared against, rather than one member of the pair reacting to `detrend`
# and the other not.
_NEVER_DETREND = _NEVER_DETREND | frozenset(_CATCH24_FEATURE_NAMES)

assert set(_EXTENDED_FEATURES) == set(FEATURE_NAMES) - set(_BASE_FEATURE_NAMES)


# ------------------------------------------------------------------- settings

def _expand_feature_group_aliases(names: list[str]) -> list[str]:
    """Expand the ``"catch22"``/``"catch24"`` shorthands into their stable feature names.

    A convenience only: every catch22/catch24 feature is just as selectable
    by its own name (``"catch22_histogram_mode_5bin"``,
    ``"catch24_raw_mean"``, ...), and the expansion happens before
    ``max_findings`` truncation, so this changes nothing about what is
    computed or how it is bounded -- only how many characters a user has to
    type to ask for all of it. ``"catch24"`` expands to all 24 (the 22 plus
    both extras); ``"catch22"`` still expands to just the 22, unchanged.
    """

    expanded: list[str] = []
    for name in names:
        if name == _CATCH22_ALIAS:
            candidates = _CATCH22_FEATURE_NAMES
        elif name == _CATCH24_ALIAS:
            candidates = _CATCH22_FEATURE_NAMES + _CATCH24_FEATURE_NAMES
        else:
            candidates = (name,)
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return expanded


def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "sequences": list(option(config, "sequences", DEFAULT_SEQUENCES)),
        "feature_groups": _expand_feature_group_aliases(
            list(option(config, "feature_groups", DEFAULT_FEATURE_GROUPS))),
        "lags": [int(x) for x in option(config, "lags", [1, 2, 3])],
        "pacf_max_lag": int(option(config, "pacf_max_lag", 5)),
        "window_words": int(option(config, "window_words", 2000)),
        "rolling_window": int(option(config, "rolling_window", 10)),
        # A multiplier of log(n), not a flat PELT penalty -- see
        # _feature_change_points' docstring for why this changed from a flat
        # 3.0 the first time the ruptures-backed path actually ran.
        "change_point_penalty": float(option(config, "change_point_penalty", 2.0)),
        "page_hinkley_delta": float(option(config, "page_hinkley_delta", 0.005)),
        "page_hinkley_lambda": float(option(config, "page_hinkley_lambda", 3.0)),
        "permutation_entropy_order": int(option(config, "permutation_entropy_order", 3)),
        "permutation_entropy_delay": int(option(config, "permutation_entropy_delay", 1)),
        "dfa_min_box": int(option(config, "dfa_min_box", 4)),
        "piecewise_segments": int(option(config, "piecewise_segments", 2)),
        "detrend": bool(option(config, "detrend", False)),
        "embedding_model": option(config, "embedding_model", "all-MiniLM-L6-v2"),
        "language": option(config, "language", "en"),
        "wavelet_name": str(option(config, "wavelet_name", "db4")),
        "wavelet_max_level": int(option(config, "wavelet_max_level", 5)),
        # Which of tsfresh's three fixed presets the "tsfresh" feature group
        # runs ("minimal", "efficient" or "comprehensive"); see the "tsfresh"
        # section above for why the group itself stays one finding per
        # sequence regardless of which preset this picks.
        "tsfresh_feature_set": str(option(config, "tsfresh_feature_set", "minimal")),
        "tsfresh_max_features": int(option(config, "tsfresh_max_features", 25)),
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

    if feature_name in _EXTENDED_FEATURES:
        ctx = {"analysis": analysis, "sequence_name": sequence_name, "sequence": sequence,
              "cfg": cfg, "detrended": detrended, "working": working}
        outcome = _EXTENDED_FEATURES[feature_name](working, cfg, ctx)
    else:
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
    features = [name for name in cfg["feature_groups"]
               if name in _FEATURES or name in _EXTENDED_FEATURES]
    unknown_sequences = [name for name in cfg["sequences"] if name not in seq.SEQUENCES]
    unknown_features = [name for name in cfg["feature_groups"]
                        if name not in _FEATURES and name not in _EXTENDED_FEATURES]

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
