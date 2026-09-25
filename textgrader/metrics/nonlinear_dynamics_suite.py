"""Recurrence quantification analysis and nonlinear-dynamics measures over named sequences.

Every other suite that reads :mod:`textgrader.sequences` (``timeseries_suite``,
``signal_processing_suite``) treats a document's ordered numeric channels
(sentence length, punctuation density, parse depth, ...) as classical linear
time series: autocorrelation, trend, spectral shape, Hurst/DFA scaling. This
suite asks a different family of questions of the *same* channels, from
nonlinear dynamics: does the sequence revisit similar states in a structured
way (recurrence quantification analysis, RQA), and do its fluctuations show
signatures -- long-range dependence, low-dimensional determinism, entropy
rate -- that a linear model would not predict. This is explicitly
experimental (see the task this module implements): a metric here is a
sensor, not a verdict on whether a text is "good".

**Recurrence quantification analysis (the core, numpy-only, on by default).**
A sequence is embedded into ``embedding_dimension``-dimensional delay vectors
(Takens' embedding, delay ``time_delay``), every pair of embedded vectors is
compared by Euclidean distance, and any pair closer than a threshold is
called "recurrent". The resulting NxN recurrence matrix is not a spatial
picture here (no image is ever produced) but a bookkeeping device this module
reduces immediately to ten scalar statistics: recurrence rate, determinism
(the fraction of recurrence points that form diagonal lines, i.e. repeated
trajectory segments, rather than isolated points), average and longest
diagonal-line length, the Shannon entropy of the diagonal-line-length
distribution, laminarity and trapping time (the vertical-line analogues --
states the system gets "stuck" in), the longest vertical line, a
recurrence-time statistic (the mean gap, in points, between successive
visits to the same neighbourhood), and TREND (Zbilut & Webber's classical
regression of local recurrence rate against distance from the line of
identity -- the one RQA measure that is explicitly a nonstationarity/drift
indicator rather than a recurrence-density one). All ten share one matrix
computation per (sequence, settings) pair, cached exactly the way
``signal_processing_suite``'s ``_welch_result`` shares one Welch call across
six features -- selecting all ten costs one embedding and one distance
matrix, not ten.

**Matrix size is capped, always, deterministically.** A recurrence matrix is
O(n^2) in both time and memory. ``max_series_points`` (default 1,500) bounds
the *embedded* series length: a sequence within that bound is used whole; a
longer one (a whole novel's ``sentence_words``, ``syllable_stress``, ...) is
reduced to exactly ``max_series_points`` points by a deterministic, seeded
contiguous window -- ``numpy.random.default_rng(sampling_seed)`` picks one
starting offset from the sequence's own length, then a contiguous run of
``max_series_points`` values is read from there. Contiguous, not a random
scatter of individual points, because RQA's whole premise (and every
nonlinear/scaling estimator this module also reports) depends on the
*order* of values being real reading order; a random subset would silently
manufacture a different, meaningless dynamical system. The same seed and the
same input sequence always pick the same window (see
``test_repeated_runs_are_bit_identical``), and every finding's
``distribution`` records the exact strategy, seed and window used
(``sampling_strategy``, ``sampling_seed``, ``sampling_start_index``,
``original_length``, ``sampled_length``) -- the corpus/profile requirement
this task calls out by name: two numbers are comparable only when they used
matching embedding/threshold/downsampling settings. The one capped/sampled
series per (sequence, settings) is itself cached and shared by *every*
feature this suite computes for that sequence, RQA and library estimators
alike, so the cost of measuring a whole book never grows past this one fixed
window regardless of how long the book's own sequence is.

**Threshold selection is automatic, but the choice is always reported.**
``threshold_mode="target_rr"`` (the default) picks the distance threshold
that would put approximately ``target_recurrence_rate`` (default 5%) of all
pairwise distances below it -- the standard practical recipe in the RQA
literature, computed here as one empirical percentile of the pairwise
distance distribution rather than an iterative search, since a percentile of
a fixed, already-computed distance matrix gives the same answer in one pass.
``threshold_mode="std_fraction"`` instead sets the threshold to
``threshold_std_fraction`` (default 0.1) times the sampled series' own
standard deviation, a second classical recipe some RQA work prefers when
comparing texts of very different absolute scale. Either way, the finding's
``distribution`` reports ``threshold_mode``, ``threshold`` (the actual
distance value used) and, for ``target_rr``, the recurrence rate actually
achieved (which the discreteness of a finite sample keeps only approximately
at the target) -- so a reader never has to reconstruct the threshold from the
settings alone.

**Why ``target_rr`` is the default, and what that costs.** ``rqa_determinism``
and ``rqa_laminarity`` (and, to a lesser extent, ``rqa_trend``) are only
comparable between two books when both recurrence plots were built at the
*same recurrence density*: a book thresholded into a denser plot shows higher
determinism/laminarity than an otherwise-identical one thresholded into a
sparser plot, purely because a denser plot offers more chances for points to
line up into a diagonal or vertical run. Pinning the recurrence rate near
``target_recurrence_rate`` holds that density constant across every book,
which is the whole point of defaulting to this mode. The cost is real and is
not hidden: under ``target_rr``, ``rqa_recurrence_rate`` stops being a
measurement of the sequence and becomes a diagnostic that the threshold
search worked close to its target -- book-to-book differences in it are
threshold-search quantization, not signal, and its own finding says so in a
``distribution["set_by_threshold_mode"]`` flag and a warning naming
``rqa_threshold`` instead. ``rqa_threshold`` (the radius the search actually
had to choose, in units of the sampled series' own standard deviation) is
where the real cross-book information moves to: a series whose embedded
trajectory is more spread out, or noisier, needs a proportionally larger raw
radius to reach the same target density, and standardizing by the series' own
spread is what keeps that comparable across books of very different absolute
scale. Under ``threshold_mode="std_fraction"`` this inverts:
``rqa_recurrence_rate`` becomes the genuine measurement (a fixed radius
produces very different densities on different books) and ``rqa_threshold``
is simply the configured ``threshold_std_fraction`` restated in the same
units, which both findings' own ``aggregation`` text says explicitly.

**PyRQA:** ``pip install --dry-run pyrqa`` shows a clean, no-downgrade
install (``PyRQA-8.1.0`` plus ``pyopencl``, ``Mako``, ``pytools``,
``siphash24`` -- all pure-Python or manylinux wheels, no compiler needed; see
``requirements.txt``). Installed for real and exercised for real: importing
``pyrqa`` and even ``pyrqa.computation`` succeeds cleanly (neither module
touches OpenCL at import time), but *constructing* an OpenCL context --
which every actual ``RQAComputation.create(...)`` call in this pyrqa version
requires; there is no CPU-only computation path in 8.1.0 -- fails in this
container. Reproduced directly, with no synthetic data at all:

    >>> import pyopencl as cl
    >>> cl.get_platforms()
    pyopencl._cl.LogicError: clGetPlatformIDs failed: PLATFORM_NOT_FOUND_KHR

and running a real ``RQAComputation`` against a trivial sine-wave series
raises the identical ``pyopencl._cl.LogicError`` before a single RQA number
is produced -- there is no OpenCL platform in this container, full stop, not
merely a slow or absent GPU. Because of this, RQA above is implemented
directly in numpy as the one and only default backend, not as a fallback for
PyRQA: nothing in the default path ever imports ``pyrqa``. PyRQA itself is
wired in as an off-by-default cross-check feature (``pyrqa_crosscheck``,
absent from ``DEFAULT_FEATURE_GROUPS``): when enabled on a machine with a
real OpenCL platform (a real GPU, or POCL/oclgrind for CPU-only OpenCL), it
runs ``pyrqa.computation.RQAComputation`` with the same embedding/delay/
threshold/sampled-series settings as this suite's own RQA and reports
PyRQA's own recurrence rate, determinism, laminarity, average/longest
diagonal length, longest vertical line and trapping time as one
finding's ``distribution``, naming ``drift.nonlinear_<seq>_rqa_recurrence_rate``
as the id it cross-checks. In *this* environment it reports 'unavailable'
quoting the exact ``LogicError`` above, which is the honest, evidenced
answer this project's rules require rather than a silent skip.

**Nonlinear/scaling library estimators**, each kept as an *independent*
measurement from anything computed elsewhere in this codebase (never
re-emitted under a new name -- see "Do not duplicate" in the task and
`distribution["overlaps_existing_metric_id"]`/`related_existing_metric_ids`
below for exactly which existing id each one is a cross-check of, where one
exists):

* ``hurst_nolds`` -- ``nolds.hurst_rs`` (rescaled-range Hurst exponent).
  Overlaps ``timeseries_suite``'s own ``hurst`` feature for the same
  sequence (a different implementation of the classical R/S statistic;
  ``timeseries_suite`` computes it from scratch in pure Python, this is
  ``nolds``' independent implementation over the *same, capped/sampled*
  series this suite shares across its own features).
* ``dfa_nolds`` -- ``nolds.dfa`` (detrended fluctuation analysis exponent).
  Overlaps ``timeseries_suite``'s ``dfa``, same relationship as above.
* ``lyapunov_nolds`` -- ``nolds.lyap_r`` (Rosenstein et al.'s largest
  Lyapunov-exponent estimator). Not computed anywhere else in this codebase.
  Per the task's own instruction ("only when sample-size and method
  assumptions are met; otherwise skip"), this has its own, higher minimum
  length (``lyapunov_min_length``, default 200) and reports
  ``insufficient_data`` rather than a number below it, since Rosenstein's
  method needs enough points to build a real trajectory and find genuine
  (not sampling-noise) nearest neighbours.
* ``correlation_dimension_nolds`` -- ``nolds.corr_dim`` (Grassberger-Procaccia
  correlation dimension). The task's own "correlation/fractal dimension"
  item; not computed anywhere else.
* ``higuchi_fd_antropy`` / ``petrosian_fd_antropy`` -- ``antropy``'s Higuchi
  and Petrosian fractal-dimension estimators. Two more, algorithmically
  independent, fractal-dimension estimators alongside
  ``correlation_dimension_nolds`` -- kept as three separate findings on
  purpose (this project's "library disagreement is data" rule): Higuchi
  measures curve-length scaling, Petrosian is a cheap sign-change proxy,
  Grassberger-Procaccia is phase-space-based; the three routinely disagree
  even on the same series, and that disagreement is exactly what a reader
  comparing them should see.
* ``sample_entropy_antropy`` / ``approximate_entropy_antropy`` --
  ``antropy.sample_entropy``/``antropy.app_entropy``. Overlap
  ``style.randomness_sample_entropy``/``style.randomness_approximate_entropy``
  (``randomness_suite``'s own hand-written ApEn/SampEn, computed only over
  sentence-length values); this suite's version runs over whichever sequence
  is selected (a different channel in general) using ``antropy``'s
  independent, KD-tree-accelerated implementation, so the two numbers are a
  genuine cross-check when the sequence happens to be sentence length, and
  an extension to other channels otherwise.
* ``lz_complexity_antropy`` -- ``antropy.lziv_complexity``, run on the
  sequence's own values binarized at their median (above/below), normalized
  by the classical ``n / log_2(n)`` asymptotic count. Overlaps
  ``style.randomness_lz_complexity_normalized``, which parses the document's
  raw lowercased *character stream*, not a numeric sequence; the two measure
  Lempel-Ziv complexity of genuinely different symbol streams, so the
  overlap is conceptual (the same complexity notion) rather than the same
  computation on the same data.
* ``permutation_entropy_ordpy`` -- ``ordpy.permutation_entropy`` (Bandt &
  Pompe's ordinal-pattern entropy, ``ordpy``'s own independent
  implementation). Overlaps both ``timeseries_suite``'s per-sequence
  ``permutation_entropy`` feature and ``style.randomness_permutation_entropy``
  (over sentence lengths only) -- named in ``overlaps_existing_metric_id``
  and ``related_existing_metric_ids`` respectively.
* ``ordinal_complexity_ordpy`` -- ``ordpy.complexity_entropy``'s statistical
  complexity ``C`` (the Jensen-Shannon-divergence-based Lopez-Ruiz/Rosso
  complexity, from the same ordinal distribution the permutation-entropy
  feature above reads its entropy half from). Not computed anywhere else in
  this codebase: entropy alone cannot distinguish "highly ordered" from
  "highly random" (both can be low- or high-entropy for different reasons),
  which is exactly the gap the entropy-complexity plane is built to fill, so
  this is reported as its own finding rather than folded into the entropy
  one, with ``related_existing_metric_ids`` naming the permutation-entropy
  finding it is computed alongside.
* ``fuzzy_entropy_entropyhub`` / ``dispersion_entropy_entropyhub`` --
  ``EntropyHub.FuzzEn``/``EntropyHub.DispEn``. Both are genuinely different
  estimators from anything else in this codebase: fuzzy entropy replaces
  ApEn/SampEn's hard "within r or not" match test with a smooth fuzzy
  membership function (so it does not inherit their sharp
  boundary-sensitivity), and dispersion entropy maps values to a small
  symbol alphabet via a normal-CDF transform before measuring pattern
  diversity, a fundamentally different symbolization from ordinal ranking
  (permutation entropy) or amplitude thresholding (ApEn/SampEn).

**Determinism.** ``nolds``' own default fit method for its power-law
regressions (``fit="RANSAC"``) is genuinely randomized -- verified directly
in this environment: three back-to-back calls to ``nolds.hurst_rs`` on the
same input gave three different floats (0.570, 0.572, 0.572) with the
default. Every ``nolds`` call in this module passes ``fit="poly"``
(ordinary least-squares on the log-log scaling relationship) instead, which
gave bit-identical results across repeated calls on the same check. Every
other library used here (``antropy``, ``EntropyHub``, ``ordpy``) was checked
the same way and is deterministic with its own defaults, and this suite's own
sampling uses a seeded ``numpy.random.default_rng`` throughout, never the
global numpy random state.

``nolds`` 0.6.3 also fails to import at all under Python 3.11 with a real
bug unrelated to any of this: ``nolds/__init__.py`` eagerly loads two bundled
fixture datasets via ``importlib.resources.files(__name__)`` where
``__name__`` is ``"nolds.datasets"``, and that name resolves to a plain
*module* (``nolds/datasets.py``) rather than a package, which Python 3.11's
stricter ``importlib.resources`` rejects outright:
``TypeError: 'nolds.datasets' is not a package`` -- reproduced directly,
before a single ``nolds`` function is reachable. See
:func:`textgrader.optional.shim_nolds_resources` for the narrow,
verified-working patch this module applies before every ``nolds`` import
(the datasets nolds bundles for its own examples live in a same-named sibling
*directory*, so falling back to the resolved module's own directory on
exactly this ``TypeError`` finds them right where the eager import expects).

**Everything is off by default.** The whole suite is off by default
(``config.json``'s ``nonlinear_dynamics_suite.enabled: false``), and within
it, ``DEFAULT_FEATURE_GROUPS`` is only the ten RQA core features -- numpy-only,
no additional optional package, matching the "no optional package, cheapest,
most direct reading" default-selection convention every sibling sequence
suite in this codebase uses. Every library estimator and the PyRQA
cross-check are individually selectable through ``feature_groups`` but
excluded from the default list, both because they need their own optional
packages and because several (Lyapunov, correlation dimension) are the most
assumption-sensitive measures in the whole suite.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence as TypingSequence

from .. import sequences as seq
from ..document import DocumentAnalysis
from ..optional import require, shim_nolds_resources
from .common import MODERATE, finding, option

FAMILY = "book_drift"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 5
UNIT_SENSITIVE = False

ID_PREFIX = "drift.nonlinear_"

# ------------------------------------------------------------------- selection

#: No optional package beyond numpy, always defined on any text with a
#: handful of sentences and two paragraphs -- the same three every sibling
#: sequence-driven suite in this codebase defaults to, for the same reason.
DEFAULT_SEQUENCES = ("sentence_words", "paragraph_words", "sentence_punctuation")

#: The ten RQA core measures, sharing one recurrence-matrix computation.
_RQA_FEATURE_ORDER = (
    "rqa_recurrence_rate", "rqa_threshold", "rqa_determinism", "rqa_avg_diagonal_length",
    "rqa_longest_diagonal_line", "rqa_diagonal_entropy", "rqa_laminarity",
    "rqa_trapping_time", "rqa_longest_vertical_line", "rqa_recurrence_time",
    "rqa_trend",
)

#: Independent library estimators -- see the module docstring for what each
#: overlaps (or does not) among this codebase's existing findings.
_LIBRARY_FEATURE_ORDER = (
    "hurst_nolds", "dfa_nolds", "lyapunov_nolds", "correlation_dimension_nolds",
    "higuchi_fd_antropy", "petrosian_fd_antropy",
    "sample_entropy_antropy", "approximate_entropy_antropy", "lz_complexity_antropy",
    "permutation_entropy_ordpy", "ordinal_complexity_ordpy",
    "fuzzy_entropy_entropyhub", "dispersion_entropy_entropyhub",
)

_CROSSCHECK_FEATURE_ORDER = ("pyrqa_crosscheck",)

_ALL_FEATURE_ORDER = _RQA_FEATURE_ORDER + _LIBRARY_FEATURE_ORDER + _CROSSCHECK_FEATURE_ORDER

#: Only the RQA core: numpy-only, no other optional package, cheapest and
#: most direct reading -- see module docstring "Everything is off by default".
DEFAULT_FEATURE_GROUPS = _RQA_FEATURE_ORDER

FEATURE_LABELS = {
    "rqa_recurrence_rate": "RQA recurrence rate",
    "rqa_threshold": "RQA recurrence threshold",
    "rqa_determinism": "RQA determinism",
    "rqa_avg_diagonal_length": "RQA average diagonal-line length",
    "rqa_longest_diagonal_line": "RQA longest diagonal line",
    "rqa_diagonal_entropy": "RQA diagonal-line-length entropy",
    "rqa_laminarity": "RQA laminarity",
    "rqa_trapping_time": "RQA trapping time",
    "rqa_longest_vertical_line": "RQA longest vertical line",
    "rqa_recurrence_time": "RQA recurrence time",
    "rqa_trend": "RQA trend (nonstationarity)",
    "hurst_nolds": "Hurst exponent (nolds, rescaled range)",
    "dfa_nolds": "DFA scaling exponent (nolds)",
    "lyapunov_nolds": "Largest Lyapunov exponent (nolds, Rosenstein)",
    "correlation_dimension_nolds": "Correlation dimension (nolds, Grassberger-Procaccia)",
    "higuchi_fd_antropy": "Higuchi fractal dimension (antropy)",
    "petrosian_fd_antropy": "Petrosian fractal dimension (antropy)",
    "sample_entropy_antropy": "Sample entropy (antropy)",
    "approximate_entropy_antropy": "Approximate entropy (antropy)",
    "lz_complexity_antropy": "Lempel-Ziv complexity, median-binarized (antropy)",
    "permutation_entropy_ordpy": "Permutation entropy (ordpy)",
    "ordinal_complexity_ordpy": "Ordinal statistical complexity (ordpy)",
    "fuzzy_entropy_entropyhub": "Fuzzy entropy (EntropyHub)",
    "dispersion_entropy_entropyhub": "Dispersion entropy (EntropyHub)",
    "pyrqa_crosscheck": "RQA cross-check (PyRQA/OpenCL)",
}

FEATURE_UNITS = {
    "rqa_recurrence_rate": "%", "rqa_threshold": "series standard deviations",
    "rqa_determinism": "%", "rqa_avg_diagonal_length": "points",
    "rqa_longest_diagonal_line": "points", "rqa_diagonal_entropy": "bits",
    "rqa_laminarity": "%", "rqa_trapping_time": "points", "rqa_longest_vertical_line": "points",
    "rqa_recurrence_time": "points", "rqa_trend": "per 1,000 diagonal steps",
    "hurst_nolds": "exponent", "dfa_nolds": "exponent", "lyapunov_nolds": "bits/point",
    "correlation_dimension_nolds": "dimension",
    "higuchi_fd_antropy": "dimension", "petrosian_fd_antropy": "dimension",
    "sample_entropy_antropy": "nats", "approximate_entropy_antropy": "nats",
    "lz_complexity_antropy": "ratio",
    "permutation_entropy_ordpy": "ratio", "ordinal_complexity_ordpy": "ratio",
    "fuzzy_entropy_entropyhub": "nats", "dispersion_entropy_entropyhub": "nats",
    "pyrqa_crosscheck": "%",
}

# Below this many RAW (pre-cap) sequence points a feature is refused rather
# than reported as a number a corpus outlier check could mistake for a real
# estimate. Overridable per feature through the `min_lengths` option.
DEFAULT_MIN_LENGTHS: dict[str, int] = {
    **{name: 40 for name in _RQA_FEATURE_ORDER},
    "hurst_nolds": 40, "dfa_nolds": 40,
    "lyapunov_nolds": 200,  # see the task's own "only when assumptions are met" instruction
    "correlation_dimension_nolds": 50,
    "higuchi_fd_antropy": 20, "petrosian_fd_antropy": 20,
    "sample_entropy_antropy": 20, "approximate_entropy_antropy": 20,
    "lz_complexity_antropy": 50,
    "permutation_entropy_ordpy": 20, "ordinal_complexity_ordpy": 20,
    "fuzzy_entropy_entropyhub": 30, "dispersion_entropy_entropyhub": 30,
    "pyrqa_crosscheck": 40,
}

# Every measure here has a documented, systematic dependence of its expected
# value on how many points it was estimated from -- this is the entire
# reason a matrix-size/series-length cap exists in the first place.
SAMPLE_SIZE_SENSITIVE_FEATURES = frozenset(_ALL_FEATURE_ORDER)

# Sequences that take their own extra settings beyond this suite's shared
# defaults -- mirrors timeseries_suite's/signal_processing_suite's identical
# sets so any registry sequence is reachable when a caller names it.
_WINDOW_SEQUENCES = frozenset({"window_dialogue_fraction", "window_pronoun_rate", "window_topic_id"})
_RARITY_SEQUENCES = frozenset({"sentence_content_rarity"})
_EMBEDDING_SEQUENCES = frozenset({"sentence_similarity_prev", "sentence_distance_centroid"})
_TOPIC_SEQUENCES = frozenset({"window_topic_id"})


@dataclass(frozen=True)
class _Outcome:
    value: float | None
    distribution: Mapping[str, Any] = field(default_factory=dict)
    warning: str | None = None


def _lib_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _join(existing: str | None, addition: str | None) -> str | None:
    if not addition:
        return existing
    return f"{existing}; {addition}" if existing else addition


def _metric_id(sequence_name: str, feature_name: str) -> str:
    # Fixed prefix/family per this task's own spec -- unlike timeseries_suite
    # and signal_processing_suite, this suite does not branch on the
    # sequence's own family; every finding here is `book_drift`.
    return f"{ID_PREFIX}{sequence_name}_{feature_name}"


def _timeseries_metric_id(sequence_name: str, feature_name: str, family: str) -> str:
    """The id ``timeseries_suite`` would use for the same (sequence, feature).

    Duplicated rather than imported, same reasoning as
    ``signal_processing_suite._timeseries_metric_id``: this is pure string
    formatting restating a stable, documented naming rule, not a dependency
    worth taking on a sibling metric module's private helper.
    """

    prefix = "drift.timeseries_" if family == "book_drift" else "rhythm.timeseries_"
    return f"{prefix}{sequence_name}_{feature_name}"


# ------------------------------------------------------------------- settings

def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "sequences": list(option(config, "sequences", DEFAULT_SEQUENCES)),
        "feature_groups": list(option(config, "feature_groups", DEFAULT_FEATURE_GROUPS)),
        "embedding_dimension": max(1, int(option(config, "embedding_dimension", 2))),
        "time_delay": max(1, int(option(config, "time_delay", 1))),
        "theiler_window": max(1, int(option(config, "theiler_window", 1))),
        "min_diagonal_length": max(2, int(option(config, "min_diagonal_length", 2))),
        "min_vertical_length": max(2, int(option(config, "min_vertical_length", 2))),
        "threshold_mode": str(option(config, "threshold_mode", "target_rr")),
        "target_recurrence_rate": float(option(config, "target_recurrence_rate", 0.05)),
        "threshold_std_fraction": float(option(config, "threshold_std_fraction", 0.1)),
        # The one cap that bounds every feature's cost in this suite, RQA and
        # library estimators alike -- see module docstring.
        "max_series_points": max(20, int(option(config, "max_series_points", 1500))),
        "sampling_seed": int(option(config, "sampling_seed", 42)),
        "rqa_min_embedded_points": max(10, int(option(config, "rqa_min_embedded_points", 30))),
        "trend_edge_margin_fraction": float(option(config, "trend_edge_margin_fraction", 0.1)),
        "lyapunov_min_length": max(50, int(option(config, "lyapunov_min_length", 200))),
        "higuchi_kmax": max(2, int(option(config, "higuchi_kmax", 10))),
        "permutation_order": max(2, int(option(config, "permutation_order", 3))),
        "fuzzy_entropy_m": max(1, int(option(config, "fuzzy_entropy_m", 2))),
        "dispersion_entropy_c": max(2, int(option(config, "dispersion_entropy_c", 4))),
        "window_words": int(option(config, "window_words", 2000)),
        "language": option(config, "language", "en"),
        "embedding_model": option(config, "embedding_model", "all-MiniLM-L6-v2"),
        "topic_n_topics": int(option(config, "topic_n_topics", 4)),
        "topic_model": str(option(config, "topic_model", "nmf")),
        "topic_random_state": int(option(config, "topic_random_state", 42)),
        "topic_max_features": int(option(config, "topic_max_features", 2000)),
        "min_lengths": dict(option(config, "min_lengths", {})),
        "max_findings": int(option(config, "max_findings", 200)),
    }


def _sequence_settings(name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if name in _TOPIC_SEQUENCES:
        return {"window_words": cfg["window_words"], "n_topics": cfg["topic_n_topics"],
               "topic_model": cfg["topic_model"], "random_state": cfg["topic_random_state"],
               "max_features": cfg["topic_max_features"]}
    if name in _WINDOW_SEQUENCES:
        return {"window_words": cfg["window_words"]}
    if name in _RARITY_SEQUENCES:
        return {"language": cfg["language"]}
    if name in _EMBEDDING_SEQUENCES:
        return {"model": cfg["embedding_model"]}
    return {}


def _sequence_settings_key(name: str, cfg: Mapping[str, Any]) -> str:
    extra = _sequence_settings(name, cfg)
    return ",".join(f"{key}={extra[key]}" for key in sorted(extra))


# ------------------------------------------------------ shared capped series

def _capped_series(analysis: DocumentAnalysis, sequence_name: str, values: TypingSequence[float],
                   cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The one deterministic, seeded, length-capped series every feature below shares.

    See the module docstring's "Matrix size is capped, always, deterministically"
    section. Cached per (sequence, settings, cap, seed) so RQA and every
    library estimator pay for exactly one sampling decision per sequence.
    """

    max_points = cfg["max_series_points"]
    seed = cfg["sampling_seed"]
    key = (f"nonlinear_dynamics_suite:capped:{sequence_name}:"
           f"{_sequence_settings_key(sequence_name, cfg)}:max={max_points}:seed={seed}")

    def build() -> dict[str, Any]:
        numpy, reason = require("numpy")
        n = len(values)
        if numpy is None or n <= max_points:
            return {"values": list(values), "strategy": "full_sequence",
                    "original_length": n, "sampled_length": n, "seed": seed, "start_index": 0}
        rng = numpy.random.default_rng(seed)
        start = int(rng.integers(0, n - max_points + 1))
        sampled = list(values[start:start + max_points])
        return {"values": sampled, "strategy": "deterministic_seeded_contiguous_window",
                "original_length": n, "sampled_length": len(sampled), "seed": seed,
                "start_index": start}

    return analysis.memo(key, build)


def _capped_distribution(capped: Mapping[str, Any]) -> dict[str, Any]:
    return {"sampling_strategy": capped["strategy"], "sampling_seed": capped["seed"],
           "sampling_start_index": capped["start_index"], "original_length": capped["original_length"],
           "sampled_length": capped["sampled_length"]}


# ------------------------------------------------------------- RQA computation

def _true_run_lengths(numpy: Any, mask: Any) -> Any:
    """Lengths of every maximal run of ``True`` in a 1-D boolean array, vectorized."""

    if mask.size == 0:
        return numpy.array([], dtype=numpy.int64)
    padded = numpy.concatenate(([False], mask, [False]))
    diff = numpy.diff(padded.astype(numpy.int8))
    starts = numpy.flatnonzero(diff == 1)
    ends = numpy.flatnonzero(diff == -1)
    return ends - starts


def _rqa_result(ctx: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The one shared embedding + recurrence-matrix computation every ``rqa_*`` feature reads.

    Implemented directly in numpy (see module docstring's "PyRQA" section for
    why this is the default backend, not a fallback). Cached per (sequence,
    settings, embedding, threshold, cap) the same way
    ``signal_processing_suite._welch_result`` shares one Welch call across
    six features.
    """

    analysis: DocumentAnalysis = ctx["analysis"]
    cfg = ctx["cfg"]
    sequence_name = ctx["sequence_name"]
    m, tau = cfg["embedding_dimension"], cfg["time_delay"]
    key = (f"nonlinear_dynamics_suite:rqa:{sequence_name}:"
           f"{_sequence_settings_key(sequence_name, cfg)}:m={m}:tau={tau}:"
           f"theiler={cfg['theiler_window']}:mode={cfg['threshold_mode']}:"
           f"target_rr={cfg['target_recurrence_rate']}:std_frac={cfg['threshold_std_fraction']}:"
           f"max_points={cfg['max_series_points']}:seed={cfg['sampling_seed']}")

    def build() -> tuple[dict[str, Any] | None, str | None]:
        numpy, reason = require("numpy")
        if numpy is None:
            return None, reason
        capped = _capped_series(analysis, sequence_name, ctx["working"], cfg)
        values = numpy.asarray(capped["values"], dtype=float)
        n = len(values)
        embed_len = n - (m - 1) * tau
        if embed_len < cfg["rqa_min_embedded_points"]:
            return None, (f"only {max(embed_len, 0)} embedded points available (embedding_dimension="
                          f"{m}, time_delay={tau} over {n} sampled points), below the "
                          f"{cfg['rqa_min_embedded_points']} rqa_min_embedded_points needs")
        embedded = numpy.array([values[i:i + (m - 1) * tau + 1:tau] for i in range(embed_len)])
        matrix = embedded.shape[0]
        # Pairwise Euclidean distance matrix, plain numpy broadcasting -- no
        # optional dependency beyond numpy itself for the suite's default
        # (RQA-core) features. matrix is capped by max_series_points, so this
        # is bounded work even on a whole novel's sequence.
        diff = embedded[:, None, :] - embedded[None, :, :]
        dist = numpy.sqrt(numpy.sum(diff * diff, axis=-1))

        theiler = min(cfg["theiler_window"], matrix - 1) if matrix > 1 else 0
        idx = numpy.arange(matrix)
        theiler_mask = numpy.abs(idx[:, None] - idx[None, :]) < theiler

        mode = cfg["threshold_mode"]
        off_theiler = dist[~theiler_mask]
        # The series' own standard deviation -- computed unconditionally,
        # not only under threshold_mode="std_fraction", because it is also
        # what makes the *chosen* threshold comparable across books of
        # different absolute scale (see rqa_threshold below and the module
        # docstring's "Threshold selection" section).
        std_values = float(numpy.std(values))
        if mode == "std_fraction":
            threshold = cfg["threshold_std_fraction"] * std_values
        else:
            mode = "target_rr"
            pct = min(max(cfg["target_recurrence_rate"], 0.0), 1.0) * 100.0
            threshold = float(numpy.percentile(off_theiler, pct)) if off_theiler.size else 0.0
        threshold_std_units = threshold / std_values if std_values > 0 else None

        recurrence = (dist <= threshold) & ~theiler_mask
        achieved_rr = float(recurrence.sum()) / off_theiler.size if off_theiler.size else 0.0

        # Computed once here, eagerly, and shared by every rqa_* feature that
        # needs it (determinism/avg/longest/entropy all read the same
        # diagonal-line lengths; laminarity/trapping-time/longest-vertical all
        # read the same vertical-line lengths) -- the same one-computation-
        # many-findings discipline signal_processing_suite's _welch_result
        # uses for its six Welch-based features, extended here to the O(n)
        # line-length scans, which would otherwise each be paid for once per
        # feature instead of once per sequence.
        diagonal_lengths = _diagonal_line_lengths(numpy, recurrence, theiler)
        vertical_lengths = _vertical_line_lengths(numpy, recurrence)
        recurrence_time_gaps = _recurrence_time_gaps(numpy, recurrence)

        return {
            "matrix_size": matrix, "recurrence": recurrence, "theiler_mask": theiler_mask,
            "theiler_window": theiler, "threshold": threshold, "threshold_mode": mode,
            "std_values": std_values, "threshold_std_units": threshold_std_units,
            "achieved_recurrence_rate": achieved_rr, "capped": capped,
            "embedding_dimension": m, "time_delay": tau,
            "diagonal_lengths": diagonal_lengths, "vertical_lengths": vertical_lengths,
            "recurrence_time_gaps": recurrence_time_gaps,
        }, None

    return analysis.memo(key, build)


def _rqa_common_distribution(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "matrix_size": result["matrix_size"], "threshold": result["threshold"],
        "threshold_mode": result["threshold_mode"],
        "threshold_std_units": result["threshold_std_units"],
        "achieved_recurrence_rate_percent": 100.0 * result["achieved_recurrence_rate"],
        "embedding_dimension": result["embedding_dimension"], "time_delay": result["time_delay"],
        "theiler_window": result["theiler_window"],
        **_capped_distribution(result["capped"]),
        "library": "numpy",
    }


def _diagonal_line_lengths(numpy: Any, recurrence: Any, theiler: int) -> Any:
    matrix = recurrence.shape[0]
    lengths = []
    # Upper triangle only: the recurrence matrix is symmetric, so offsets
    # k and -k carry the same information (a mirrored copy of the same
    # diagonal), and using only k >= theiler avoids both double-counting and
    # the Theiler-excluded band around the line of identity.
    for k in range(theiler, matrix):
        diag = numpy.diagonal(recurrence, offset=k)
        if diag.size:
            lengths.append(_true_run_lengths(numpy, diag))
    return numpy.concatenate(lengths) if lengths else numpy.array([], dtype=numpy.int64)


def _vertical_line_lengths(numpy: Any, recurrence: Any) -> Any:
    matrix = recurrence.shape[0]
    lengths = [_true_run_lengths(numpy, recurrence[:, j]) for j in range(matrix)]
    return numpy.concatenate(lengths) if lengths else numpy.array([], dtype=numpy.int64)


def _recurrence_time_gaps(numpy: Any, recurrence: Any) -> Any:
    matrix = recurrence.shape[0]
    gaps = []
    for j in range(matrix):
        positions = numpy.flatnonzero(recurrence[:, j])
        if positions.size > 1:
            gaps.append(numpy.diff(positions))
    return numpy.concatenate(gaps) if gaps else numpy.array([], dtype=numpy.int64)


def _entropy_of_lengths(numpy: Any, lengths: Any) -> float | None:
    if lengths.size == 0:
        return None
    values, counts = numpy.unique(lengths, return_counts=True)
    total = counts.sum()
    p = counts / total
    return float(-numpy.sum(p * numpy.log2(p)))


def _feature_rqa_recurrence_rate(values, cfg, ctx) -> _Outcome:
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    common = _rqa_common_distribution(result)
    value = 100.0 * result["achieved_recurrence_rate"]
    if result["threshold_mode"] == "target_rr":
        # See the module docstring's "Threshold selection" section: under the
        # default mode this number is chosen, not measured -- it is pinned
        # near target_recurrence_rate by the very percentile search that
        # picks the threshold, so the actual cross-book signal moved into
        # the threshold itself (rqa_threshold, below), not into this rate.
        return _Outcome(value, distribution={
            **common, "set_by_threshold_mode": True,
            "aggregation": "share of off-Theiler-band embedded-vector pairs whose distance is "
                           "at or below the threshold -- fixed near target_recurrence_rate by "
                           "construction under threshold_mode='target_rr' (the default); this "
                           "is a diagnostic that the threshold search worked, not a measurement "
                           "of the sequence"},
            warning=f"this value is set by threshold_mode='target_rr' targeting "
                    f"{100.0 * cfg['target_recurrence_rate']:.1f}%, not measured from the "
                    f"sequence -- differences between books here are threshold-search "
                    f"quantization, not signal; compare rqa_threshold across books instead")
    return _Outcome(value, distribution={
        **common,
        "aggregation": "share of off-Theiler-band embedded-vector pairs whose distance is at "
                       "or below a threshold fixed as threshold_std_fraction times the "
                       "series' own standard deviation (threshold_mode='std_fraction') -- a "
                       "genuine measurement here, since nothing about the threshold search "
                       "targets a particular rate"})


def _feature_rqa_threshold(values, cfg, ctx) -> _Outcome:
    """The recurrence-distance threshold actually chosen, in units of the series' own SD.

    Under ``threshold_mode="target_rr"`` (the default), ``rqa_recurrence_rate``
    is pinned near ``target_recurrence_rate`` by construction (see that
    feature's own docstring/warning) -- the information about how tightly or
    loosely this sequence's embedded trajectory revisits itself moves into
    *how large a radius was needed* to hit that target, not into the
    resulting rate. This finding reports exactly that radius, standardized by
    the sampled series' own standard deviation so it is comparable across
    books of very different absolute scale (a book whose sentence-length
    values range 3-60 and one whose values range 0.1-0.9 should not be
    compared on raw distance units). A more spread-out or noisier series
    needs a proportionally larger raw threshold to reach the same target
    recurrence rate; the standardized value strips out pure scale
    differences and is what should actually be compared across a corpus.
    """

    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    common = _rqa_common_distribution(result)
    value = result["threshold_std_units"]
    if value is None:
        return _Outcome(None, distribution=common,
                        warning="the sampled series has zero standard deviation; a "
                                "standardized threshold is undefined")
    if result["threshold_mode"] == "target_rr":
        aggregation = ("the recurrence-distance threshold chosen to reach "
                       f"target_recurrence_rate={100.0 * cfg['target_recurrence_rate']:.1f}%, "
                       "expressed in units of the sampled series' own standard deviation -- "
                       "under threshold_mode='target_rr' (the default) this, not "
                       "rqa_recurrence_rate, is the real cross-book measurement")
    else:
        aggregation = ("the fixed recurrence-distance threshold (threshold_std_fraction times "
                       "the series' own standard deviation), restated here in the same "
                       "standardized units for a consistent id across both threshold_mode "
                       "settings; under threshold_mode='std_fraction' this is simply the "
                       "configured threshold_std_fraction value")
    return _Outcome(value, distribution={
        **common, "raw_threshold": result["threshold"], "std_of_sampled_values": result["std_values"],
        "aggregation": aggregation})


def _feature_rqa_determinism(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    lmin = cfg["min_diagonal_length"]
    lengths = result["diagonal_lengths"]
    total_points = float(lengths.sum())
    common = _rqa_common_distribution(result)
    if total_points <= 0:
        return _Outcome(None, distribution=common,
                        warning="no recurrence points found at this threshold; determinism is "
                                "undefined")
    on_lines = float(lengths[lengths >= lmin].sum())
    det = 100.0 * on_lines / total_points
    return _Outcome(det, distribution={
        **common, "min_diagonal_length": lmin, "diagonal_recurrence_points": total_points,
        "diagonal_line_count": int((lengths >= lmin).sum()),
        "aggregation": "share of upper-triangle recurrence points that belong to a diagonal "
                       f"line of at least {lmin} points"})


def _feature_rqa_avg_diagonal_length(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    lmin = cfg["min_diagonal_length"]
    lengths = result["diagonal_lengths"]
    qualifying = lengths[lengths >= lmin]
    common = _rqa_common_distribution(result)
    if qualifying.size == 0:
        return _Outcome(None, distribution=common,
                        warning=f"no diagonal line reached the minimum length ({lmin} points)")
    return _Outcome(float(qualifying.mean()), distribution={
        **common, "min_diagonal_length": lmin, "diagonal_line_count": int(qualifying.size),
        "aggregation": f"mean length (points) of diagonal lines at or above {lmin} points"})


def _feature_rqa_longest_diagonal_line(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    lengths = result["diagonal_lengths"]
    common = _rqa_common_distribution(result)
    if lengths.size == 0:
        return _Outcome(0.0, distribution=common,
                        warning="no diagonal line of any length was found (no recurrence)")
    return _Outcome(float(lengths.max()), distribution={
        **common,
        "aggregation": "length (points) of the single longest diagonal line found, excluding "
                       "the Theiler-excluded band around the line of identity"})


def _feature_rqa_diagonal_entropy(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    lmin = cfg["min_diagonal_length"]
    lengths = result["diagonal_lengths"]
    qualifying = lengths[lengths >= lmin]
    common = _rqa_common_distribution(result)
    entropy = _entropy_of_lengths(numpy, qualifying)
    if entropy is None:
        return _Outcome(0.0, distribution={**common, "min_diagonal_length": lmin},
                        warning=f"no diagonal line reached the minimum length ({lmin} points); "
                                "entropy over an empty distribution is undefined, reporting 0.0")
    return _Outcome(entropy, distribution={
        **common, "min_diagonal_length": lmin, "distinct_lengths": int(numpy.unique(qualifying).size),
        "aggregation": "Shannon entropy, in bits, of the diagonal-line-length distribution "
                       f"(lines at or above {lmin} points)"})


def _feature_rqa_laminarity(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    vmin = cfg["min_vertical_length"]
    lengths = result["vertical_lengths"]
    total_points = float(lengths.sum())
    common = _rqa_common_distribution(result)
    if total_points <= 0:
        return _Outcome(None, distribution=common,
                        warning="no recurrence points found at this threshold; laminarity is "
                                "undefined")
    on_lines = float(lengths[lengths >= vmin].sum())
    return _Outcome(100.0 * on_lines / total_points, distribution={
        **common, "min_vertical_length": vmin, "vertical_recurrence_points": total_points,
        "aggregation": "share of recurrence points that belong to a vertical line of at least "
                       f"{vmin} points"})


def _feature_rqa_trapping_time(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    vmin = cfg["min_vertical_length"]
    lengths = result["vertical_lengths"]
    qualifying = lengths[lengths >= vmin]
    common = _rqa_common_distribution(result)
    if qualifying.size == 0:
        return _Outcome(None, distribution=common,
                        warning=f"no vertical line reached the minimum length ({vmin} points)")
    return _Outcome(float(qualifying.mean()), distribution={
        **common, "min_vertical_length": vmin, "vertical_line_count": int(qualifying.size),
        "aggregation": f"mean length (points) of vertical lines at or above {vmin} points -- "
                       "how long the system stays trapped in a similar state"})


def _feature_rqa_longest_vertical_line(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    lengths = result["vertical_lengths"]
    common = _rqa_common_distribution(result)
    if lengths.size == 0:
        return _Outcome(0.0, distribution=common, warning="no vertical line was found")
    return _Outcome(float(lengths.max()), distribution={
        **common, "aggregation": "length (points) of the single longest vertical line found"})


def _feature_rqa_recurrence_time(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    gaps = result["recurrence_time_gaps"]
    common = _rqa_common_distribution(result)
    if gaps.size == 0:
        return _Outcome(None, distribution=common,
                        warning="no column had two or more recurrences; a recurrence time needs "
                                "at least two visits to the same neighbourhood")
    entropy = _entropy_of_lengths(numpy, gaps)
    return _Outcome(float(gaps.mean()), distribution={
        **common, "gap_count": int(gaps.size), "recurrence_time_entropy_bits": entropy,
        "aggregation": "mean gap (points), pooled over every column, between successive visits "
                       "to the same embedded-vector neighbourhood"})


def _feature_rqa_trend(values, cfg, ctx) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    result, reason = _rqa_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    recurrence, theiler = result["recurrence"], result["theiler_window"]
    matrix = result["matrix_size"]
    edge_margin = max(1, int(cfg["trend_edge_margin_fraction"] * matrix))
    max_k = matrix - edge_margin
    common = _rqa_common_distribution(result)
    if max_k <= theiler + 2:
        return _Outcome(None, distribution=common,
                        warning="matrix too small to exclude an edge margin and still have "
                                "enough diagonals for a trend regression")
    ks = numpy.arange(theiler, max_k)
    local_rr = numpy.array([float(numpy.diagonal(recurrence, offset=int(k)).mean()) for k in ks])
    if numpy.allclose(local_rr, local_rr[0]):
        return _Outcome(0.0, distribution={**common, "diagonals_used": int(ks.size)},
                        warning="local recurrence rate is identical at every distance from the "
                                "line of identity; trend/slope is exactly zero, not undefined")
    slope, _ = numpy.polyfit(ks, local_rr, 1)
    return _Outcome(float(slope * 1000.0), distribution={
        **common, "diagonals_used": int(ks.size), "edge_margin": edge_margin,
        "aggregation": "1000 x the OLS slope of local recurrence rate (mean of each diagonal "
                       "parallel to the line of identity) against distance from the line of "
                       "identity -- Zbilut & Webber's TREND: near zero for a stationary "
                       "sequence, systematically away from zero when recurrence structure "
                       "drifts over the course of the document"})


_RQA_FEATURES: dict[str, Callable[..., _Outcome]] = {
    "rqa_recurrence_rate": _feature_rqa_recurrence_rate,
    "rqa_threshold": _feature_rqa_threshold,
    "rqa_determinism": _feature_rqa_determinism,
    "rqa_avg_diagonal_length": _feature_rqa_avg_diagonal_length,
    "rqa_longest_diagonal_line": _feature_rqa_longest_diagonal_line,
    "rqa_diagonal_entropy": _feature_rqa_diagonal_entropy,
    "rqa_laminarity": _feature_rqa_laminarity,
    "rqa_trapping_time": _feature_rqa_trapping_time,
    "rqa_longest_vertical_line": _feature_rqa_longest_vertical_line,
    "rqa_recurrence_time": _feature_rqa_recurrence_time,
    "rqa_trend": _feature_rqa_trend,
}

assert set(_RQA_FEATURES) == set(_RQA_FEATURE_ORDER)


# ------------------------------------------------------------- PyRQA cross-check

def _feature_pyrqa_crosscheck(values, cfg, ctx) -> _Outcome:
    """PyRQA's own RQA computation over the same capped series, off by default.

    See the module docstring's "PyRQA" section: this always fails in a
    container with no OpenCL platform, and reports the real, quoted error
    rather than a silent skip. On a machine with a working OpenCL platform,
    this runs for real and cross-checks
    ``drift.nonlinear_<seq>_rqa_recurrence_rate``.
    """

    pyrqa_module, reason = require("pyrqa")
    if pyrqa_module is None:
        return _Outcome(None, warning=reason)
    analysis: DocumentAnalysis = ctx["analysis"]
    sequence_name = ctx["sequence_name"]
    capped = _capped_series(analysis, sequence_name, ctx["working"], cfg)
    series = capped["values"]
    if len(series) - (cfg["embedding_dimension"] - 1) * cfg["time_delay"] < cfg["rqa_min_embedded_points"]:
        return _Outcome(None, warning="too few sampled points to embed for a PyRQA cross-check")
    own_id = _metric_id(sequence_name, "rqa_recurrence_rate")
    try:
        from pyrqa.time_series import TimeSeries
        from pyrqa.settings import Settings
        from pyrqa.analysis_type import Classic
        from pyrqa.neighbourhood import FixedRadius
        from pyrqa.metric import EuclideanMetric
        from pyrqa.computation import RQAComputation
        import numpy as _np

        threshold = float(option(cfg, "pyrqa_threshold", None) or
                          (cfg["threshold_std_fraction"] * float(_np.std(series))))
        time_series = TimeSeries(series, embedding_dimension=cfg["embedding_dimension"],
                                 time_delay=cfg["time_delay"])
        settings = Settings(time_series, analysis_type=Classic,
                            neighbourhood=FixedRadius(threshold),
                            similarity_measure=EuclideanMetric,
                            theiler_corrector=cfg["theiler_window"])
        computation = RQAComputation.create(settings, verbose=False)
        result = computation.run()
    except Exception as exc:  # pragma: no cover - needs a real OpenCL platform
        return _Outcome(None, warning=f"pyrqa RQAComputation failed ({type(exc).__name__}: {exc})")
    return _Outcome(100.0 * float(result.recurrence_rate), distribution={
        "backend": "pyrqa", "library": "pyrqa", "library_version": _lib_version(pyrqa_module),
        "determinism_percent": 100.0 * float(result.determinism),
        "laminarity_percent": 100.0 * float(result.laminarity),
        "average_diagonal_line": float(result.average_diagonal_line),
        "longest_diagonal_line": float(result.longest_diagonal_line),
        "longest_vertical_line": float(result.longest_vertical_line),
        "trapping_time": float(result.trapping_time),
        "threshold": threshold, **_capped_distribution(capped),
        "overlaps_existing_metric_id": own_id,
        "overlap_note": "PyRQA's own OpenCL-backed RQA implementation, run over the identical "
                        "capped/sampled series, as a cross-check of this suite's numpy RQA core",
        "aggregation": "PyRQA's own recurrence-rate computation (percent)"})


# ------------------------------------------------------------- library estimators

def _require_nolds():
    """``require("nolds")``, with the import-time shim active for exactly that call.

    ``shim_nolds_resources`` is a context manager (see its own docstring for
    why -- a permanent, module-global patch is the exact mistake this
    project's own BookNLP shim already made and had to be rewritten to
    avoid), so it is entered and exited here, around ``require`` alone, every
    single time this is called. After the first successful (or failed) real
    import, ``require``'s own cache means every later call re-enters and
    exits the context manager without ``importlib.resources.files`` ever
    actually being touched -- a harmless no-op, not a second exposure window.
    """

    with shim_nolds_resources():
        return require("nolds")


def _feature_hurst_nolds(values, cfg, ctx) -> _Outcome:
    nolds_module, reason = _require_nolds()
    if nolds_module is None:
        return _Outcome(None, warning=reason)
    capped = ctx["capped"]
    overlap = _timeseries_metric_id(ctx["sequence_name"], "hurst", ctx["sequence_family"])
    try:
        value = float(nolds_module.hurst_rs(capped["values"], fit="poly"))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"nolds.hurst_rs failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "nolds", "library_version": _lib_version(nolds_module), "fit": "poly",
        **_capped_distribution(capped), "overlaps_existing_metric_id": overlap,
        "overlap_note": "timeseries_suite's own hurst feature computes the same rescaled-range "
                        "statistic from scratch in pure Python; nolds is an independent "
                        "implementation over this suite's own capped/sampled series",
        "aggregation": "rescaled-range (R/S) Hurst exponent, OLS fit"})


def _feature_dfa_nolds(values, cfg, ctx) -> _Outcome:
    nolds_module, reason = _require_nolds()
    if nolds_module is None:
        return _Outcome(None, warning=reason)
    capped = ctx["capped"]
    overlap = _timeseries_metric_id(ctx["sequence_name"], "dfa", ctx["sequence_family"])
    try:
        value = float(nolds_module.dfa(capped["values"], fit_exp="poly"))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"nolds.dfa failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "nolds", "library_version": _lib_version(nolds_module), "fit_exp": "poly",
        **_capped_distribution(capped), "overlaps_existing_metric_id": overlap,
        "overlap_note": "timeseries_suite's own dfa feature computes DFA from scratch in pure "
                        "Python; nolds is an independent implementation over this suite's own "
                        "capped/sampled series",
        "aggregation": "detrended fluctuation analysis scaling exponent, OLS fit"})


def _feature_lyapunov_nolds(values, cfg, ctx) -> _Outcome:
    nolds_module, reason = _require_nolds()
    if nolds_module is None:
        return _Outcome(None, warning=reason)
    capped = ctx["capped"]
    if capped["sampled_length"] < cfg["lyapunov_min_length"]:
        return _Outcome(None, distribution=_capped_distribution(capped),
                        warning=f"insufficient data for a Lyapunov-exponent estimate: "
                                f"{capped['sampled_length']} sampled points, below "
                                f"lyapunov_min_length={cfg['lyapunov_min_length']} -- Rosenstein's "
                                "method needs enough points to find genuine, not sampling-noise, "
                                "nearest neighbours")
    try:
        value = float(nolds_module.lyap_r(capped["values"], fit="poly"))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"nolds.lyap_r failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "nolds", "library_version": _lib_version(nolds_module), "fit": "poly",
        **_capped_distribution(capped),
        "aggregation": "largest Lyapunov exponent, Rosenstein et al.'s algorithm; positive "
                       "suggests sensitive dependence on initial conditions (chaos-like), "
                       "non-positive suggests contraction/stability -- reported only above the "
                       "task's own sample-size floor, never guessed below it"})


def _feature_correlation_dimension_nolds(values, cfg, ctx) -> _Outcome:
    nolds_module, reason = _require_nolds()
    if nolds_module is None:
        return _Outcome(None, warning=reason)
    capped = ctx["capped"]
    try:
        value = float(nolds_module.corr_dim(capped["values"], emb_dim=cfg["embedding_dimension"],
                                            fit="poly"))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"nolds.corr_dim failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "nolds", "library_version": _lib_version(nolds_module), "fit": "poly",
        "embedding_dimension": cfg["embedding_dimension"], **_capped_distribution(capped),
        "aggregation": "Grassberger-Procaccia correlation dimension, OLS fit of the correlation "
                       "sum's power-law scaling"})


def _feature_higuchi_fd_antropy(values, cfg, ctx) -> _Outcome:
    antropy_module, reason = require("antropy")
    if antropy_module is None:
        return _Outcome(None, warning=reason)
    numpy, numpy_reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=numpy_reason)
    capped = ctx["capped"]
    kmax = min(cfg["higuchi_kmax"], max(2, len(capped["values"]) // 4))
    try:
        value = float(antropy_module.higuchi_fd(numpy.asarray(capped["values"], dtype=float), kmax=kmax))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"antropy.higuchi_fd failed ({type(exc).__name__}: {exc})")
    if not math.isfinite(value):
        # A constant sequence, or one exactly periodic at a period that
        # divides evenly into one of the k sub-series lengths kmax tries, can
        # drive one of Higuchi's per-k curve lengths to exactly zero, which
        # makes the log-log fit's log(0) a non-finite slope -- an antropy
        # numerical edge case on close-to-degenerate input, not a crash to hide.
        return _Outcome(None, distribution={**_capped_distribution(capped), "kmax": kmax},
                        warning="antropy.higuchi_fd returned a non-finite value on this sequence "
                                "(a degenerate or exactly periodic input can drive one of its "
                                "sub-series curve lengths to exactly zero)")
    return _Outcome(value, distribution={
        "library": "antropy", "library_version": _lib_version(antropy_module), "kmax": kmax,
        **_capped_distribution(capped),
        "aggregation": "Higuchi fractal dimension: slope of curve length against sub-series "
                       "step size on a log-log plot"})


def _feature_petrosian_fd_antropy(values, cfg, ctx) -> _Outcome:
    antropy_module, reason = require("antropy")
    if antropy_module is None:
        return _Outcome(None, warning=reason)
    numpy, numpy_reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=numpy_reason)
    capped = ctx["capped"]
    try:
        value = float(antropy_module.petrosian_fd(numpy.asarray(capped["values"], dtype=float)))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"antropy.petrosian_fd failed ({type(exc).__name__}: {exc})")
    if not math.isfinite(value):
        return _Outcome(None, distribution=_capped_distribution(capped),
                        warning="antropy.petrosian_fd returned a non-finite value on this sequence")
    return _Outcome(value, distribution={
        "library": "antropy", "library_version": _lib_version(antropy_module),
        **_capped_distribution(capped),
        "aggregation": "Petrosian fractal dimension: a cheap sign-change-count proxy for "
                       "curve complexity"})


def _feature_sample_entropy_antropy(values, cfg, ctx) -> _Outcome:
    antropy_module, reason = require("antropy")
    if antropy_module is None:
        return _Outcome(None, warning=reason)
    numpy, numpy_reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=numpy_reason)
    capped = ctx["capped"]
    overlap = "style.randomness_sample_entropy"
    try:
        value = antropy_module.sample_entropy(numpy.asarray(capped["values"], dtype=float))
        value = None if value is None or not math.isfinite(value) else float(value)
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"antropy.sample_entropy failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "antropy", "library_version": _lib_version(antropy_module),
        **_capped_distribution(capped), "overlaps_existing_metric_id": overlap,
        "overlap_note": "randomness_suite's own sample_entropy is computed only over sentence "
                        "lengths, using a hand-written O(n^2) implementation; this is antropy's "
                        "independent, KD-tree-accelerated implementation over whichever sequence "
                        "is selected here",
        "aggregation": "sample entropy (order 2, tolerance 0.2 x std), natural log"},
        warning=None if value is not None else "antropy.sample_entropy returned an undefined value")


def _feature_approximate_entropy_antropy(values, cfg, ctx) -> _Outcome:
    antropy_module, reason = require("antropy")
    if antropy_module is None:
        return _Outcome(None, warning=reason)
    numpy, numpy_reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=numpy_reason)
    capped = ctx["capped"]
    overlap = "style.randomness_approximate_entropy"
    try:
        value = antropy_module.app_entropy(numpy.asarray(capped["values"], dtype=float))
        value = None if value is None or not math.isfinite(value) else float(value)
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"antropy.app_entropy failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "antropy", "library_version": _lib_version(antropy_module),
        **_capped_distribution(capped), "overlaps_existing_metric_id": overlap,
        "overlap_note": "randomness_suite's own approximate_entropy is computed only over "
                        "sentence lengths, using a hand-written implementation; this is "
                        "antropy's independent implementation over whichever sequence is "
                        "selected here",
        "aggregation": "approximate entropy (order 2, tolerance 0.2 x std), natural log"},
        warning=None if value is not None else "antropy.app_entropy returned an undefined value")


def _feature_lz_complexity_antropy(values, cfg, ctx) -> _Outcome:
    antropy_module, reason = require("antropy")
    if antropy_module is None:
        return _Outcome(None, warning=reason)
    numpy, numpy_reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=numpy_reason)
    capped = ctx["capped"]
    arr = numpy.asarray(capped["values"], dtype=float)
    median = float(numpy.median(arr))
    binary = (arr > median).astype(numpy.uint8)
    if binary.min() == binary.max():
        return _Outcome(None, distribution=_capped_distribution(capped),
                        warning="every sampled value fell on the same side of the median; "
                                "median-binarization collapsed to a single symbol, so LZ "
                                "complexity is undefined")
    try:
        value = float(antropy_module.lziv_complexity(binary, normalize=True))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"antropy.lziv_complexity failed ({type(exc).__name__}: {exc})")
    overlap = "style.randomness_lz_complexity_normalized"
    return _Outcome(value, distribution={
        "library": "antropy", "library_version": _lib_version(antropy_module),
        "binarization": "above/below own median", **_capped_distribution(capped),
        "overlaps_existing_metric_id": overlap,
        "overlap_note": "randomness_suite's own lz_complexity_normalized parses the document's "
                        "raw lowercased character stream, a different symbol alphabet; this "
                        "measures the same Lempel-Ziv notion of complexity but over the selected "
                        "numeric sequence, binarized at its own median",
        "aggregation": "Lempel-Ziv complexity of the median-binarized sequence, normalized by "
                       "n/log2(n)"})


def _feature_permutation_entropy_ordpy(values, cfg, ctx) -> _Outcome:
    ordpy_module, reason = require("ordpy")
    if ordpy_module is None:
        return _Outcome(None, warning=reason)
    capped = ctx["capped"]
    order = cfg["permutation_order"]
    if capped["sampled_length"] < order + 5:
        return _Outcome(None, distribution=_capped_distribution(capped),
                        warning=f"needs at least {order + 5} sampled points for permutation "
                                f"order {order}")
    try:
        value = float(ordpy_module.permutation_entropy(capped["values"], dx=order, base=2,
                                                        normalized=True))
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"ordpy.permutation_entropy failed ({type(exc).__name__}: {exc})")
    overlap = _timeseries_metric_id(ctx["sequence_name"], "permutation_entropy", ctx["sequence_family"])
    return _Outcome(value, distribution={
        "library": "ordpy", "library_version": _lib_version(ordpy_module), "order": order,
        **_capped_distribution(capped), "overlaps_existing_metric_id": overlap,
        "related_existing_metric_ids": ["style.randomness_permutation_entropy"],
        "overlap_note": "timeseries_suite's own permutation_entropy feature for this sequence "
                        "computes the identical Bandt-Pompe ordinal-pattern entropy from "
                        "scratch; ordpy is an independent implementation. "
                        "style.randomness_permutation_entropy is the same notion but computed "
                        "only over sentence lengths",
        "aggregation": "normalized Shannon entropy (bits) of the ordinal-pattern distribution"})


def _feature_ordinal_complexity_ordpy(values, cfg, ctx) -> _Outcome:
    ordpy_module, reason = require("ordpy")
    if ordpy_module is None:
        return _Outcome(None, warning=reason)
    capped = ctx["capped"]
    order = cfg["permutation_order"]
    if capped["sampled_length"] < order + 5:
        return _Outcome(None, distribution=_capped_distribution(capped),
                        warning=f"needs at least {order + 5} sampled points for permutation "
                                f"order {order}")
    try:
        entropy_h, complexity_c = ordpy_module.complexity_entropy(capped["values"], dx=order)
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"ordpy.complexity_entropy failed ({type(exc).__name__}: {exc})")
    related = _timeseries_metric_id(ctx["sequence_name"], "permutation_entropy", ctx["sequence_family"])
    return _Outcome(float(complexity_c), distribution={
        "library": "ordpy", "library_version": _lib_version(ordpy_module), "order": order,
        "permutation_entropy_component": float(entropy_h), **_capped_distribution(capped),
        "related_existing_metric_ids": [related, _metric_id(ctx["sequence_name"],
                                                             "permutation_entropy_ordpy")],
        "aggregation": "Lopez-Ruiz/Mendes/Rosso statistical complexity (Jensen-Shannon "
                       "divergence from the ordinal-pattern distribution's own entropy), from "
                       "the same ordinal distribution the permutation-entropy findings read "
                       "their entropy half from -- not itself an entropy, and not computed "
                       "anywhere else in this codebase"})


def _feature_fuzzy_entropy_entropyhub(values, cfg, ctx) -> _Outcome:
    entropyhub_module, reason = require("entropyhub")
    if entropyhub_module is None:
        return _Outcome(None, warning=reason)
    numpy, numpy_reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=numpy_reason)
    capped = ctx["capped"]
    m = cfg["fuzzy_entropy_m"]
    try:
        fuzz, _ps1, _ps2 = entropyhub_module.FuzzEn(numpy.asarray(capped["values"], dtype=float), m=m)
        value = float(fuzz[-1])
        value = None if not math.isfinite(value) else value
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"EntropyHub.FuzzEn failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "EntropyHub", "library_version": _lib_version(entropyhub_module), "m": m,
        **_capped_distribution(capped),
        "aggregation": f"fuzzy entropy at embedding dimension {m} (natural log), a smooth-"
                       "membership-function generalization of sample entropy's hard match test"},
        warning=None if value is not None else "EntropyHub.FuzzEn returned an undefined value")


def _feature_dispersion_entropy_entropyhub(values, cfg, ctx) -> _Outcome:
    entropyhub_module, reason = require("entropyhub")
    if entropyhub_module is None:
        return _Outcome(None, warning=reason)
    numpy, numpy_reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=numpy_reason)
    capped = ctx["capped"]
    c = cfg["dispersion_entropy_c"]
    try:
        disp, _ppi = entropyhub_module.DispEn(numpy.asarray(capped["values"], dtype=float),
                                              m=2, c=c)
        value = float(disp)
        value = None if not math.isfinite(value) else value
    except Exception as exc:  # pragma: no cover - numerical edge case
        return _Outcome(None, warning=f"EntropyHub.DispEn failed ({type(exc).__name__}: {exc})")
    return _Outcome(value, distribution={
        "library": "EntropyHub", "library_version": _lib_version(entropyhub_module),
        "symbols": c, **_capped_distribution(capped),
        "aggregation": f"dispersion entropy ({c} NCDF-mapped symbols, natural log) -- symbolizes "
                       "amplitude by cumulative-distribution rank rather than by ordinal rank "
                       "(permutation entropy) or a hard match radius (ApEn/SampEn)"},
        warning=None if value is not None else "EntropyHub.DispEn returned an undefined value")


_LIBRARY_FEATURES: dict[str, Callable[..., _Outcome]] = {
    "hurst_nolds": _feature_hurst_nolds,
    "dfa_nolds": _feature_dfa_nolds,
    "lyapunov_nolds": _feature_lyapunov_nolds,
    "correlation_dimension_nolds": _feature_correlation_dimension_nolds,
    "higuchi_fd_antropy": _feature_higuchi_fd_antropy,
    "petrosian_fd_antropy": _feature_petrosian_fd_antropy,
    "sample_entropy_antropy": _feature_sample_entropy_antropy,
    "approximate_entropy_antropy": _feature_approximate_entropy_antropy,
    "lz_complexity_antropy": _feature_lz_complexity_antropy,
    "permutation_entropy_ordpy": _feature_permutation_entropy_ordpy,
    "ordinal_complexity_ordpy": _feature_ordinal_complexity_ordpy,
    "fuzzy_entropy_entropyhub": _feature_fuzzy_entropy_entropyhub,
    "dispersion_entropy_entropyhub": _feature_dispersion_entropy_entropyhub,
}

assert set(_LIBRARY_FEATURES) == set(_LIBRARY_FEATURE_ORDER)

_EXTENDED_FEATURES: dict[str, Callable[..., _Outcome]] = {
    **_RQA_FEATURES,
    "pyrqa_crosscheck": _feature_pyrqa_crosscheck,
}


# --------------------------------------------------------------------- measuring

def _measure_one(analysis: DocumentAnalysis, sequence_name: str, feature_name: str,
                 cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec = seq.SEQUENCES[sequence_name]
    metric_id = _metric_id(sequence_name, feature_name)
    name = f"{FEATURE_LABELS[feature_name]} of {sequence_name.replace('_', ' ')}"
    unit = FEATURE_UNITS[feature_name]

    sequence = seq.get_sequence(analysis, sequence_name, _sequence_settings(sequence_name, cfg))
    if not sequence.values:
        return finding(metric_id, name, None, unit, family=FAMILY, sample_size=0,
                       warning=sequence.warning or "sequence has no values")

    min_length = int(cfg["min_lengths"].get(feature_name, DEFAULT_MIN_LENGTHS[feature_name]))
    if sequence.length < min_length:
        return finding(
            metric_id, name, None, unit, family=FAMILY, sample_size=sequence.length,
            min_sample=min_length,
            warning=f"insufficient data: {sequence_name} has {sequence.length} "
                    f"{sequence.sample_unit} points, below the {min_length} the "
                    f"{feature_name} feature needs")

    working = list(sequence.values)
    if feature_name in _EXTENDED_FEATURES:
        ctx = {"analysis": analysis, "sequence_name": sequence_name, "cfg": cfg, "working": working,
              "sequence_family": spec.family}
        outcome = _EXTENDED_FEATURES[feature_name](working, cfg, ctx)
    else:
        capped = _capped_series(analysis, sequence_name, working, cfg)
        ctx = {"analysis": analysis, "sequence_name": sequence_name, "cfg": cfg, "working": working,
              "capped": capped, "sequence_family": spec.family}
        outcome = _LIBRARY_FEATURES[feature_name](working, cfg, ctx)

    distribution = dict(outcome.distribution)
    distribution.setdefault("sequence", sequence_name)
    distribution.setdefault("sequence_unit", sequence.unit)
    distribution.setdefault("sample_unit", sequence.sample_unit)
    if sequence.settings:
        distribution["sequence_settings"] = dict(sequence.settings)

    return finding(
        metric_id, name, outcome.value, unit, family=FAMILY, sample_size=sequence.length,
        min_sample=min_length, distribution=distribution,
        warning=_join(sequence.warning, outcome.warning), sample_size_sensitive=True)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = _settings(config)

    sequences = [name for name in cfg["sequences"] if name in seq.SEQUENCES]
    features = [name for name in cfg["feature_groups"] if name in _ALL_FEATURE_ORDER]
    unknown_sequences = [name for name in cfg["sequences"] if name not in seq.SEQUENCES]
    unknown_features = [name for name in cfg["feature_groups"] if name not in _ALL_FEATURE_ORDER]

    if not sequences or not features:
        problems = []
        if not sequences:
            problems.append("no valid entry in 'sequences'" +
                            (f" (unknown: {unknown_sequences})" if unknown_sequences else ""))
        if not features:
            problems.append("no valid entry in 'feature_groups'" +
                            (f" (unknown: {unknown_features})" if unknown_features else ""))
        return [finding("drift.nonlinear_config", "Nonlinear-dynamics suite configuration",
                        None, None, family=FAMILY, warning="; ".join(problems))]

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
        out.append(finding("drift.nonlinear_config", "Nonlinear-dynamics suite configuration",
                           len(combos), "combinations", family=FAMILY, warning=note))
    if truncated:
        out.append(finding(
            "drift.nonlinear_truncated", "Nonlinear-dynamics suite output truncated by max_findings",
            max_findings, "findings", family=FAMILY,
            warning=f"{len(sequences) * len(features)} (sequence, feature) combinations were "
                    f"selected but only max_findings={max_findings} were computed; raise "
                    f"metrics.nonlinear_dynamics_suite.max_findings, or narrow 'sequences' or "
                    f"'feature_groups', to see the rest"))
    return out
