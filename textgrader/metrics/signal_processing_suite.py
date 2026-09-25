"""Generic signal-processing features over the named sequences in :mod:`textgrader.sequences`.

:mod:`textgrader.metrics.timeseries_suite` already asks classical time-series
questions -- autocorrelation, trend, stationarity, a plain FFT spectral shape,
catch22/catch24, wavelet energy/entropy, tsfresh, change-point/drift
detectors -- of every named sequence the registry exposes. This suite adds the
signal-processing tools that module does not already emit: Welch periodograms
(a windowed, segment-averaged spectrum, rather than one raw FFT of the whole
series), spectral centroid/bandwidth/flatness/roll-off, low/mid/high band
energy shares, time-domain peak count/prominence/distance, a zero-crossing
rate, a simple real-cepstrum peak summary, a multiscale (aggregated-variance)
scaling exponent, and -- new to this codebase -- cross-sequence features
(cross-correlation, cross-spectral phase, magnitude-squared coherence)
between two sequences that share a sample unit.

**Reuse, not reimplementation.** This module imports
:mod:`textgrader.sequences` exactly the way ``timeseries_suite`` does and
builds no sequence of its own: ``SEQUENCES``/``get_sequence`` is the one
registry both suites read from, so a sequence added there (a stress sequence,
a POS-code channel, anything) is available to this suite the moment a
caller's own ``sequences`` setting names it -- no code in this file changes.
See ``test_a_registry_sequence_this_module_never_named_is_usable_with_no_code_change``
for a synthetic proof: a throwaway sequence is registered into
``textgrader.sequences.SEQUENCES`` at test time (monkeypatched, never a real
registry edit) and this suite measures it correctly on the first try.

**Nothing here duplicates a timeseries_suite id**, but several of these
features are deliberately close cousins of one, computed a different way:
Welch's segment-averaged, windowed spectrum versus ``timeseries_suite``'s own
single raw periodogram (``spectral``), and this suite's ``spectral_centroid``
versus catch22's own Welch-based ``catch22_spectral_centroid`` (rectangular
window, radians/sample) and this suite's ``band_energy`` low-band share versus
catch22's ``catch22_spectral_low_freq_power`` (lowest fifth, not this suite's
configurable band edges). Per this project's "a metric is a sensor, not an
opinion" rule, every one of those is kept as its own finding rather than
folded into or replacing the existing one, and each names the existing id it
is a variant of in its own ``distribution["overlaps_existing_metric_id"]`` so
a reader (or a future analysis) knows the two numbers are expected to be
related, not independent surprises. ``multiscale_variance`` is a genuinely
different estimator from ``timeseries_suite``'s ``hurst``/``dfa`` (aggregated
block-mean variance versus rescaled range and detrended fluctuation), not a
variant of either, so it is not marked as an overlap -- but its own
``distribution["related_existing_metric_ids"]`` still names both, since all
three estimate a scaling/self-similarity exponent from the same kind of data
and a reader comparing them should know they are asking a related question.

**Time-domain versus frequency-domain peaks.** ``peaks`` (scipy's
``find_peaks`` over the raw sequence, with a scale-free prominence threshold)
is a different measurement from ``timeseries_suite``'s ``turning_points``:
turning points counts *every* local direction reversal (a strict local
maximum or minimum, however small), while ``peaks`` counts only prominence-
and distance-qualified peaks -- a text that reverses direction on almost every
sentence (which drives ``turning_points`` high) can still have very few
"real" peaks once a plateau/noise floor is required to separate them. The two
are complementary, not duplicates, so neither is marked as an overlap.

**One headline per id, every time**, per this project's rule that a metric's
headline statistic must not switch depending on the data: ``welch_spectral``
always reports normalized spectral entropy (never sometimes entropy and
sometimes dominant frequency); ``band_energy`` always reports the low-band
share (mid/high live in ``distribution``); ``cepstral_peak`` always reports
the peak-to-floor prominence ratio (the quefrency of that peak, in "points"
-- never seconds -- lives in ``distribution``).

**Units are honest.** Every frequency this suite reports is in cycles per
whatever one position in the sequence actually is -- cycles per sentence,
per paragraph, per window, per token -- read directly from the sequence's own
``sample_unit``, never Hz or seconds (there is no clock here, only reading
order). ``spectral_centroid``, ``spectral_bandwidth`` and ``spectral_rolloff``
compose their unit as ``f"cycles per {sequence.sample_unit}"`` at measurement
time rather than a fixed string, and every finding also records
``sample_unit`` directly in its ``distribution`` so a reader never has to
infer it from the unit string. Quefrency (``cepstral_peak``) is reported in
"points", the cepstral analogue of the same idea: a lag measured in sequence
positions, not time.

**Categorical/nominal sequences are labelled experimental, generically.**
:mod:`textgrader.sequences` already documents its one nominal-label sequence
(``window_topic_id``) as such directly in that sequence's own ``description``
string ("... (nominal label)."). Rather than hard-coding that one sequence's
name here, ``_measure_one`` checks for the substring ``"nominal"`` in
whatever sequence's description it is handed and appends an explicit warning
that spectral/amplitude-based interpretation of a nominal label is
experimental -- so a *future* categorical sequence (a POS-code channel, say)
that documents itself as nominal the same way is flagged automatically, with
no change needed here.

**Cross-sequence features run only when units genuinely match.**
``cross_correlation``, ``cross_spectrum`` and ``coherence`` take a ``pairs``
setting: a list of ``[sequence_a, sequence_b]`` name pairs. Before computing
anything, each pair's two sequences are checked for an identical
``sample_unit`` (sentence with sentence, paragraph with paragraph, window with
window) -- a sentence-level sequence can never be compared position-by-
position against a window-level one, since position ``i`` means something
different in each -- and a mismatched pair reports why, with no computation
attempted, rather than silently truncating to the shorter one and pretending
the alignment was meaningful.

**Guarding the combinatorics**, the same way ``timeseries_suite`` does:
every feature is one finding per sequence (or per pair), with secondary
numbers folded into ``distribution``; the default selection
(``DEFAULT_SEQUENCES`` x the eight default-on single-sequence features, plus
``DEFAULT_PAIRS`` x the two default-on pair features) is a deliberately small,
dependency-light slice (26 findings); every sequence, feature and pair is its
own switch; and ``max_findings`` (default 200) is a hard stop with its own
truncation notice, identical in spirit to ``timeseries_suite``'s own guard.

**What stays off by default and why.** ``cepstral_peak`` and
``multiscale_variance`` need materially more data to mean anything (see
``DEFAULT_MIN_LENGTHS``) and are opt-in, the same reasoning
``timeseries_suite`` uses for ``hurst``/``dfa``/``permutation_entropy``.
``cross_spectrum`` is off by default because its headline (phase) is a genuine
extra piece of information on top of ``coherence`` (which stays on) but is
also the least intuitive number in this suite, best turned on deliberately.
Every sequence that needs the shared spaCy parse, an embedding model, or a
fitted topic model (``sentence_parse_depth``, ``sentence_similarity_prev``,
``window_topic_id``, ...) is already absent from ``DEFAULT_SEQUENCES`` for the
same reason ``timeseries_suite`` keeps them out of its own default: this
suite's cost is ``"moderate"``, so ``MetricSpec.needs_parse``/``needs_model``
cannot see what an individual *sequence* pulls in, and nothing should turn on
a parse or a model during ordinary corpus profiling.

**Deferred.** The task's suggested source-sequence list ("POS-tag numeric
sequence", "stress sequence from prosody tools") names two channels that do
not exist in :mod:`textgrader.sequences` yet, and this suite is not allowed
to add them there (see this module's own docstring above on reuse). Task 16
is expected to expose a stress sequence
(``textgrader.prosody.stress_sequence``) that the orchestrator registers into
``textgrader.sequences.SEQUENCES`` after merging; once it is, adding
``"stress_sequence"`` to this suite's ``sequences`` setting is the *entire*
integration step -- see the module docstring's opening section and the
registry-agnostic test named above. ``librosa`` (the task's one "optional,
experimental" library suggestion) is not used: every feature this suite
needed librosa for -- Welch, peak-finding, coherence -- scipy's own ``signal``
module already provides directly on a plain numeric array, with no audio
sample-rate semantics to work around; adding a second, audio-flavoured
library on top would not add a genuinely new measurement, only a second name
for one this suite already has. Recurrence quantification (PyRQA) is Task
21's scope, not a gap here.
"""

from __future__ import annotations

import math
import statistics
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence as TypingSequence

from .. import sequences as seq
from ..document import DocumentAnalysis
from ..optional import require
from .common import MODERATE, finding, option

FAMILY = "sentence_rhythm"  # fallback only; every finding sets its own real family
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 5
UNIT_SENSITIVE = False


# ------------------------------------------------------------------- selection

#: No optional package beyond numpy/scipy (already effectively load-bearing
#: across this codebase), always defined on any text with a handful of
#: sentences and two paragraphs -- the same three ``timeseries_suite`` uses
#: as its own default, for the same reason (see that module's docstring).
DEFAULT_SEQUENCES = ("sentence_words", "paragraph_words", "sentence_punctuation")

#: One naturally aligned (both per-sentence) built-in pair, cheap and always
#: available, so the default configuration demonstrates real cross-sequence
#: coverage rather than shipping the family switched on with nothing to run.
DEFAULT_PAIRS = (("sentence_words", "sentence_punctuation"),)

#: Every single-sequence feature name, in the order findings are produced.
_FEATURE_ORDER = (
    "welch_spectral", "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "band_energy", "peaks", "zero_crossing_rate",
    "cepstral_peak", "multiscale_variance",
)

#: Every cross-sequence (pair) feature name, in the order findings are produced.
_PAIR_FEATURE_ORDER = ("cross_correlation", "cross_spectrum", "coherence")

_ALL_FEATURE_NAMES = frozenset(_FEATURE_ORDER) | frozenset(_PAIR_FEATURE_ORDER)

#: See "What stays off by default and why" in the module docstring.
DEFAULT_FEATURES: dict[str, bool] = {
    "welch_spectral": True, "spectral_centroid": True, "spectral_bandwidth": True,
    "spectral_flatness": True, "spectral_rolloff": True, "band_energy": True,
    "peaks": True, "zero_crossing_rate": True,
    "cepstral_peak": False, "multiscale_variance": False,
    "cross_correlation": True, "cross_spectrum": False, "coherence": True,
}

FEATURE_LABELS = {
    "welch_spectral": "Welch spectral entropy",
    "spectral_centroid": "Spectral centroid",
    "spectral_bandwidth": "Spectral bandwidth",
    "spectral_flatness": "Spectral flatness",
    "spectral_rolloff": "Spectral roll-off frequency",
    "band_energy": "Low-band energy share",
    "peaks": "Peak rate",
    "zero_crossing_rate": "Zero-crossing rate",
    "cepstral_peak": "Cepstral peak prominence",
    "multiscale_variance": "Multiscale variance scaling",
}

#: ``None`` means "compose dynamically from the sequence's own sample_unit"
#: (``f"cycles per {sample_unit}"``) -- see the module docstring's "Units are
#: honest" section.
FEATURE_UNITS: dict[str, str | None] = {
    "welch_spectral": "ratio",
    "spectral_centroid": None,
    "spectral_bandwidth": None,
    "spectral_flatness": "ratio",
    "spectral_rolloff": None,
    "band_energy": "%",
    "peaks": "peaks per 100 points",
    "zero_crossing_rate": "crossings per 100 points",
    "cepstral_peak": "ratio",
    "multiscale_variance": "exponent",
}

PAIR_FEATURE_LABELS = {
    "cross_correlation": "Cross-correlation",
    "cross_spectrum": "Cross-spectral phase",
    "coherence": "Magnitude-squared coherence",
}

PAIR_FEATURE_UNITS = {
    "cross_correlation": "correlation",
    "cross_spectrum": "radians",
    "coherence": "ratio",
}

# Below this many points a feature is refused rather than reported as a
# number a corpus outlier check could mistake for a real estimate.
DEFAULT_MIN_LENGTHS = {
    "welch_spectral": 16, "spectral_centroid": 16, "spectral_bandwidth": 16,
    "spectral_flatness": 16, "spectral_rolloff": 16, "band_energy": 16,
    "peaks": 10, "zero_crossing_rate": 5, "cepstral_peak": 32, "multiscale_variance": 16,
    "cross_correlation": 10, "cross_spectrum": 16, "coherence": 16,
}

# Features whose expected value has a documented, systematic dependence on
# sequence length even at fixed settings (more frequency bins, more scales,
# a wider quefrency search window as the sequence grows) -- see
# timeseries_suite's own docstring for why this is a separate flag from
# unit-sensitivity.
SAMPLE_SIZE_SENSITIVE_FEATURES = frozenset({
    "welch_spectral", "spectral_centroid", "spectral_bandwidth", "spectral_flatness",
    "spectral_rolloff", "band_energy", "cepstral_peak", "multiscale_variance",
    "cross_spectrum", "coherence",
})

# Sequences that take their own extra settings beyond this suite's shared
# defaults -- mirrors timeseries_suite's identical sets so that any registry
# sequence (not only this suite's own conservative DEFAULT_SEQUENCES) is
# reachable when a caller names it explicitly.
_WINDOW_SEQUENCES = frozenset({"window_dialogue_fraction", "window_pronoun_rate", "window_topic_id"})
_RARITY_SEQUENCES = frozenset({"sentence_content_rarity"})
_EMBEDDING_SEQUENCES = frozenset({"sentence_similarity_prev", "sentence_distance_centroid"})
_TOPIC_SEQUENCES = frozenset({"window_topic_id"})


@dataclass(frozen=True)
class _Outcome:
    value: float | None
    distribution: Mapping[str, Any] = field(default_factory=dict)
    warning: str | None = None
    sample_size_sensitive: bool = False


def _lib_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _join(existing: str | None, addition: str | None) -> str | None:
    if not addition:
        return existing
    return f"{existing}; {addition}" if existing else addition


def _timeseries_metric_id(sequence_name: str, feature_name: str, family: str) -> str:
    """The id ``timeseries_suite`` would use for the same (sequence, feature).

    Duplicated rather than imported: this is two lines of pure string
    formatting (see ``timeseries_suite._metric_id``), and importing a sibling
    metric module just to reach a private helper would be a heavier coupling
    than restating its (stable, documented) naming rule here.
    """

    prefix = "drift.timeseries_" if family == "book_drift" else "rhythm.timeseries_"
    return f"{prefix}{sequence_name}_{feature_name}"


def _metric_id(sequence_name: str, feature_name: str, family: str) -> str:
    prefix = "drift.signal_" if family == "book_drift" else "rhythm.signal_"
    return f"{prefix}{sequence_name}_{feature_name}"


def _pair_metric_id(name_a: str, name_b: str, feature_name: str, family: str) -> str:
    prefix = "drift.signal_" if family == "book_drift" else "rhythm.signal_"
    return f"{prefix}{name_a}_x_{name_b}_{feature_name}"


# ------------------------------------------------------------------- settings

def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "sequences": list(option(config, "sequences", DEFAULT_SEQUENCES)),
        "features": dict(option(config, "features", DEFAULT_FEATURES)),
        "pairs": [tuple(pair) for pair in option(config, "pairs", DEFAULT_PAIRS)],
        "welch_nperseg": int(option(config, "welch_nperseg", 256)),
        "welch_noverlap": option(config, "welch_noverlap", None),
        "welch_window": str(option(config, "welch_window", "hann")),
        "welch_detrend": option(config, "welch_detrend", "constant"),
        "band_edges": [float(x) for x in option(config, "band_edges", [1 / 3, 2 / 3])],
        "rolloff_percent": float(option(config, "rolloff_percent", 0.85)),
        "peak_prominence_sigma": float(option(config, "peak_prominence_sigma", 0.5)),
        "peak_min_distance": int(option(config, "peak_min_distance", 1)),
        "cepstral_min_quefrency": int(option(config, "cepstral_min_quefrency", 2)),
        "multiscale_min_blocks": int(option(config, "multiscale_min_blocks", 4)),
        "cross_correlation_max_lag": int(option(config, "cross_correlation_max_lag", 10)),
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


# ----------------------------------------------------------------- small maths

def _center(values: TypingSequence[float]) -> list[float]:
    mean = statistics.fmean(values)
    return [v - mean for v in values]


# -------------------------------------------------------- shared Welch result

def _welch_result(ctx: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """The one shared ``scipy.signal.welch`` call every Welch-based feature reads from.

    ``spectral_centroid``, ``spectral_bandwidth``, ``spectral_flatness``,
    ``spectral_rolloff``, ``band_energy`` and ``welch_spectral`` all want the
    same (frequencies, power) pair for one sequence under one set of Welch
    settings; caching it once per document (keyed on the sequence, its own
    settings and every Welch parameter) means selecting all six costs one
    Welch call, not six -- the same sharing discipline
    ``timeseries_suite._catch22_result``/``_wavelet_result`` use for catch22
    and wavelets.
    """

    analysis: DocumentAnalysis = ctx["analysis"]
    cfg = ctx["cfg"]
    sequence_name = ctx["sequence_name"]
    key = (f"signal_processing_suite:welch:{sequence_name}:"
           f"{_sequence_settings_key(sequence_name, cfg)}:nperseg={cfg['welch_nperseg']}:"
           f"noverlap={cfg['welch_noverlap']}:window={cfg['welch_window']}:"
           f"detrend={cfg['welch_detrend']}")

    def build() -> tuple[dict[str, Any] | None, str | None]:
        numpy, reason = require("numpy")
        if numpy is None:
            return None, reason
        scipy_signal, reason = require("scipy.signal")
        if scipy_signal is None:
            return None, reason
        arr = numpy.asarray(ctx["working"], dtype=float)
        n = len(arr)
        nperseg = max(4, min(int(cfg["welch_nperseg"]), n))
        noverlap_cfg = cfg["welch_noverlap"]
        noverlap = int(noverlap_cfg) if noverlap_cfg is not None else nperseg // 2
        noverlap = max(0, min(noverlap, nperseg - 1))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                freqs, psd = scipy_signal.welch(
                    arr, fs=1.0, window=cfg["welch_window"], nperseg=nperseg,
                    noverlap=noverlap, detrend=cfg["welch_detrend"])
        except Exception as exc:  # pragma: no cover - library/runtime guard
            return None, f"scipy.signal.welch failed ({type(exc).__name__}: {exc})"
        scipy_module, _ = require("scipy")
        return {"freqs": freqs, "psd": psd, "nperseg": nperseg, "noverlap": noverlap,
               "window": cfg["welch_window"], "detrend": cfg["welch_detrend"],
               "library_version": _lib_version(scipy_module)}, None

    return analysis.memo(key, build)


def _welch_common_distribution(result: dict[str, Any]) -> dict[str, Any]:
    return {"method": "welch", "window": result["window"], "nperseg": result["nperseg"],
           "noverlap": result["noverlap"], "detrend": result["detrend"],
           "library": "scipy", "library_version": result["library_version"]}


# --------------------------------------------------------- Welch-based features

def _feature_welch_spectral(values: TypingSequence[float], cfg: Mapping[str, Any],
                            ctx: Mapping[str, Any]) -> _Outcome:
    result, reason = _welch_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    freqs, psd = result["freqs"], result["psd"]
    if len(psd) < 3:
        return _Outcome(None, warning="too few frequency bins to summarize a Welch spectrum",
                        sample_size_sensitive=True)
    freqs_ac, psd_ac = freqs[1:], psd[1:]  # drop the DC bin, as timeseries_suite's own FFT feature does
    total = float(psd_ac.sum())
    common = _welch_common_distribution(result)
    overlap = _timeseries_metric_id(ctx["sequence_name"], "spectral", ctx["family"])
    if total <= 0:
        return _Outcome(0.0, distribution={
            **common, "dominant_frequency": 0.0, "spectral_concentration": None,
            "overlaps_existing_metric_id": overlap,
            "aggregation": "normalized Shannon entropy of the Welch power spectral density"},
            warning="the sequence is constant once its mean is removed; spectral entropy is "
                    "undefined so 0.0 (a pure tone) was used",
            sample_size_sensitive=True)
    numpy, _ = require("numpy")  # already confirmed available by the _welch_result call above
    p = psd_ac / total
    raw_entropy = float(-numpy.sum(p * numpy.log2(numpy.where(p > 0, p, 1.0))))
    max_entropy = math.log2(len(p))
    normalized = raw_entropy / max_entropy if max_entropy > 0 else 0.0
    dominant_index = int(numpy.argmax(psd_ac))
    return _Outcome(normalized, distribution={
        **common, "dominant_frequency": float(freqs_ac[dominant_index]),
        "spectral_concentration": float(p.max()), "raw_entropy_bits": raw_entropy,
        "overlaps_existing_metric_id": overlap,
        "aggregation": "normalized Shannon entropy of the Welch power spectral density "
                       "(one value per document)"},
        sample_size_sensitive=True)


def _feature_spectral_centroid(values: TypingSequence[float], cfg: Mapping[str, Any],
                               ctx: Mapping[str, Any]) -> _Outcome:
    result, reason = _welch_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    freqs, psd = result["freqs"], result["psd"]
    freqs_ac, psd_ac = freqs[1:], psd[1:]
    total = float(psd_ac.sum())
    common = _welch_common_distribution(result)
    if total <= 0 or len(psd_ac) < 2:
        return _Outcome(None, distribution=common,
                        warning="the sequence is constant once its mean is removed; a spectral "
                                "centroid is undefined", sample_size_sensitive=True)
    centroid = float((freqs_ac * psd_ac).sum() / total)
    overlap = _timeseries_metric_id(ctx["sequence_name"], "catch22_spectral_centroid", ctx["family"])
    return _Outcome(centroid, distribution={
        **common, "overlaps_existing_metric_id": overlap,
        "overlap_note": "catch22's own Welch-based spectral centroid uses a rectangular window "
                        "and reports radians/sample rather than cycles/sample; expected to track "
                        "this finding, not to match it exactly",
        "aggregation": "power-weighted mean frequency of the Welch power spectral density"},
        sample_size_sensitive=True)


def _feature_spectral_bandwidth(values: TypingSequence[float], cfg: Mapping[str, Any],
                                ctx: Mapping[str, Any]) -> _Outcome:
    result, reason = _welch_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    freqs, psd = result["freqs"], result["psd"]
    freqs_ac, psd_ac = freqs[1:], psd[1:]
    total = float(psd_ac.sum())
    common = _welch_common_distribution(result)
    if total <= 0 or len(psd_ac) < 2:
        return _Outcome(None, distribution=common,
                        warning="the sequence is constant once its mean is removed; a spectral "
                                "bandwidth is undefined", sample_size_sensitive=True)
    centroid = float((freqs_ac * psd_ac).sum() / total)
    variance = float((((freqs_ac - centroid) ** 2) * psd_ac).sum() / total)
    return _Outcome(math.sqrt(max(variance, 0.0)), distribution={
        **common, "centroid": centroid,
        "aggregation": "power-weighted standard deviation of frequency around the spectral "
                       "centroid"},
        sample_size_sensitive=True)


def _feature_spectral_flatness(values: TypingSequence[float], cfg: Mapping[str, Any],
                               ctx: Mapping[str, Any]) -> _Outcome:
    result, reason = _welch_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    psd_ac = [float(p) for p in result["psd"][1:] if p > 0]
    common = _welch_common_distribution(result)
    if len(psd_ac) < 2:
        return _Outcome(None, distribution=common,
                        warning="fewer than two positive-power frequency bins; spectral flatness "
                                "is undefined", sample_size_sensitive=True)
    log_mean = statistics.fmean(math.log(p) for p in psd_ac)
    geometric_mean = math.exp(log_mean)
    arithmetic_mean = statistics.fmean(psd_ac)
    flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else None
    return _Outcome(flatness, distribution={
        **common, "positive_power_bins": len(psd_ac),
        "aggregation": "ratio of the geometric to the arithmetic mean of the Welch power "
                       "spectrum's positive bins (0 = a single pure tone, 1 = perfectly flat/"
                       "white)"},
        sample_size_sensitive=True)


def _feature_spectral_rolloff(values: TypingSequence[float], cfg: Mapping[str, Any],
                              ctx: Mapping[str, Any]) -> _Outcome:
    result, reason = _welch_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    freqs, psd = result["freqs"], result["psd"]
    freqs_ac, psd_ac = freqs[1:], psd[1:]
    total = float(psd_ac.sum())
    common = _welch_common_distribution(result)
    percent = min(max(float(cfg["rolloff_percent"]), 0.01), 0.99)
    if total <= 0 or len(psd_ac) < 2:
        return _Outcome(None, distribution={**common, "rolloff_percent": percent},
                        warning="the sequence is constant once its mean is removed; a spectral "
                                "roll-off is undefined", sample_size_sensitive=True)
    target = percent * total
    cumulative = 0.0
    rolloff_freq = float(freqs_ac[-1])
    for f, p in zip(freqs_ac, psd_ac):
        cumulative += float(p)
        if cumulative >= target:
            rolloff_freq = float(f)
            break
    return _Outcome(rolloff_freq, distribution={
        **common, "rolloff_percent": percent,
        "aggregation": f"lowest frequency below which {percent:.0%} of the Welch spectrum's "
                       f"energy (excluding the DC bin) is contained"},
        sample_size_sensitive=True)


def _feature_band_energy(values: TypingSequence[float], cfg: Mapping[str, Any],
                         ctx: Mapping[str, Any]) -> _Outcome:
    result, reason = _welch_result(ctx)
    if result is None:
        return _Outcome(None, warning=reason)
    freqs, psd = result["freqs"], result["psd"]
    common = _welch_common_distribution(result)
    total = float(psd.sum())
    edges = sorted(min(max(float(x), 0.0), 1.0) for x in cfg["band_edges"])[:2]
    if len(edges) < 2:
        edges = [1 / 3, 2 / 3]
    nyquist = float(freqs[-1]) if len(freqs) else 0.5
    edge1, edge2 = edges[0] * nyquist, edges[1] * nyquist
    if total <= 0 or nyquist <= 0:
        return _Outcome(None, distribution={**common, "band_edges": edges},
                        warning="the sequence has no spectral energy to distribute across bands",
                        sample_size_sensitive=True)
    low = sum(float(p) for f, p in zip(freqs, psd) if f <= edge1)
    mid = sum(float(p) for f, p in zip(freqs, psd) if edge1 < f <= edge2)
    high = sum(float(p) for f, p in zip(freqs, psd) if f > edge2)
    shares = [100.0 * band / total for band in (low, mid, high)]
    overlap = _timeseries_metric_id(ctx["sequence_name"], "catch22_spectral_low_freq_power",
                                    ctx["family"])
    return _Outcome(shares[0], distribution={
        **common, "band_edges_fraction_of_nyquist": edges,
        "band_edges": [edge1, edge2], "low_share_percent": shares[0],
        "mid_share_percent": shares[1], "high_share_percent": shares[2],
        "overlaps_existing_metric_id": overlap,
        "overlap_note": "catch22's own low-frequency-power feature fixes the low band at the "
                        "lowest fifth of the Welch spectrum; this suite's band_edges default to "
                        "thirds and are independently configurable",
        "aggregation": "share of total Welch spectral power in the lowest configured band"},
        sample_size_sensitive=True)


# ----------------------------------------------------------- other single-sequence features

def _feature_peaks(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    scipy_signal, reason = require("scipy.signal")
    if scipy_signal is None:
        return _Outcome(None, warning=reason)
    arr = numpy.asarray(values, dtype=float)
    n = len(arr)
    std = float(arr.std())
    prominence = float(cfg["peak_prominence_sigma"]) * std
    distance = max(1, int(cfg["peak_min_distance"]))
    try:
        indices, properties = scipy_signal.find_peaks(arr, prominence=prominence, distance=distance)
    except Exception as exc:  # pragma: no cover - library/runtime guard
        return _Outcome(None, warning=f"scipy.signal.find_peaks failed ({type(exc).__name__}: {exc})")
    rate = 100.0 * len(indices) / n
    prominences = list(properties.get("prominences", []))
    mean_prominence = float(numpy.mean(prominences)) if len(prominences) else None
    distances = [int(b - a) for a, b in zip(indices[:-1], indices[1:])] if len(indices) > 1 else []
    mean_distance = float(numpy.mean(distances)) if distances else None
    return _Outcome(rate, distribution={
        "peak_count": int(len(indices)), "mean_prominence": mean_prominence,
        "mean_distance_points": mean_distance, "prominence_threshold": prominence,
        "min_distance_points": distance, "library": "scipy", "library_version": _lib_version(scipy_signal),
        "aggregation": "count of scipy.signal.find_peaks-qualified peaks per 100 sequence points, "
                       "over the raw (uncentered) sequence values"})


def _feature_zero_crossing_rate(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    centered = _center(values)
    n = len(centered)
    crossings = 0
    previous = centered[0]
    for value in centered[1:]:
        if previous == 0.0:
            previous = value
            continue
        if value == 0.0:
            continue
        if (previous > 0) != (value > 0):
            crossings += 1
        previous = value
    rate = 100.0 * crossings / (n - 1) if n > 1 else None
    return _Outcome(rate, distribution={
        "crossing_count": crossings, "opportunities": n - 1,
        "aggregation": "sign changes per 100 points, after subtracting the sequence's own mean"})


def _feature_cepstral_peak(values: TypingSequence[float], cfg: Mapping[str, Any]) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    arr = numpy.asarray(values, dtype=float)
    arr = arr - arr.mean()
    n = len(arr)
    spectrum = numpy.fft.rfft(arr)
    magnitude = numpy.abs(spectrum)
    log_magnitude = numpy.log(magnitude + 1e-12)
    cepstrum = numpy.fft.irfft(log_magnitude, n=n)
    min_q = max(1, int(cfg["cepstral_min_quefrency"]))
    max_q = n // 2
    if max_q <= min_q + 2:
        return _Outcome(None, warning=f"needs more than {2 * (min_q + 2)} points for a cepstral "
                                      f"search window past quefrency {min_q}")
    window = numpy.abs(cepstrum[min_q:max_q])
    floor = float(numpy.median(window))
    peak_index = int(numpy.argmax(window))
    peak_value = float(window[peak_index])
    peak_quefrency = peak_index + min_q
    prominence = peak_value / floor if floor > 0 else None
    return _Outcome(prominence, distribution={
        "peak_quefrency_points": peak_quefrency, "peak_value": peak_value, "floor_value": floor,
        "min_quefrency_points": min_q, "max_quefrency_points": max_q,
        "aggregation": "ratio of the real cepstrum's peak magnitude (searched past a minimum "
                       "quefrency) to the median magnitude of the same search window"},
        sample_size_sensitive=True)


def _feature_multiscale_variance(values: TypingSequence[float], cfg: Mapping[str, Any],
                                 ctx: Mapping[str, Any]) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    n = len(values)
    min_blocks = max(2, int(cfg["multiscale_min_blocks"]))
    scales: list[int] = []
    scale = 1
    while n // scale >= min_blocks:
        scales.append(scale)
        scale *= 2
    log_scales: list[float] = []
    log_vars: list[float] = []
    scale_variances: dict[int, float] = {}
    for scale in scales:
        n_blocks = n // scale
        block_means = [statistics.fmean(values[i * scale:(i + 1) * scale]) for i in range(n_blocks)]
        variance = statistics.pvariance(block_means) if len(block_means) > 1 else 0.0
        scale_variances[scale] = variance
        if variance > 0:
            log_scales.append(math.log(scale))
            log_vars.append(math.log(variance))
    if len(log_scales) < 3:
        return _Outcome(None, warning="could not compute a block-mean variance at enough distinct "
                                      "scales (need variation within at least three block sizes)",
                        sample_size_sensitive=True)
    slope, _ = numpy.polyfit(log_scales, log_vars, 1)
    sequence_name, family = ctx["sequence_name"], ctx["family"]
    return _Outcome(float(slope), distribution={
        "scales": scales, "variance_by_scale": scale_variances,
        "related_existing_metric_ids": [
            _timeseries_metric_id(sequence_name, "hurst", family),
            _timeseries_metric_id(sequence_name, "dfa", family),
        ],
        "aggregation": "OLS slope of log(block-mean variance) against log(block scale) -- the "
                       "classical aggregated-variance long-range-dependence estimator, computed "
                       "independently of this suite's own hurst/dfa (a different algorithm on the "
                       "same kind of question)"},
        sample_size_sensitive=True)


_FEATURES: dict[str, Callable[..., _Outcome]] = {
    "peaks": _feature_peaks,
    "zero_crossing_rate": _feature_zero_crossing_rate,
    "cepstral_peak": _feature_cepstral_peak,
}

_EXTENDED_FEATURES: dict[str, Callable[..., _Outcome]] = {
    "welch_spectral": _feature_welch_spectral,
    "spectral_centroid": _feature_spectral_centroid,
    "spectral_bandwidth": _feature_spectral_bandwidth,
    "spectral_flatness": _feature_spectral_flatness,
    "spectral_rolloff": _feature_spectral_rolloff,
    "band_energy": _feature_band_energy,
    "multiscale_variance": _feature_multiscale_variance,
}

assert set(_FEATURES) | set(_EXTENDED_FEATURES) == set(_FEATURE_ORDER)


# ------------------------------------------------------------------- pair features

def _feature_cross_correlation(values_a: TypingSequence[float], values_b: TypingSequence[float],
                               cfg: Mapping[str, Any]) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    n = len(values_a)
    a = numpy.asarray(values_a, dtype=float) - statistics.fmean(values_a)
    b = numpy.asarray(values_b, dtype=float) - statistics.fmean(values_b)
    denom = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if denom <= 0:
        return _Outcome(None, warning="one or both sequences are constant; cross-correlation is "
                                      "undefined")
    max_lag = max(1, min(int(cfg["cross_correlation_max_lag"]), n - 1))
    correlations: dict[int, float] = {}
    best_lag, best_value = 0, 0.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            numerator = float((a[:n - lag] * b[lag:]).sum()) if lag < n else 0.0
        else:
            numerator = float((a[-lag:] * b[:n + lag]).sum())
        value = numerator / denom
        correlations[lag] = value
        if abs(value) > abs(best_value):
            best_value, best_lag = value, lag
    return _Outcome(best_value, distribution={
        "best_lag": best_lag, "max_lag_searched": max_lag,
        "lag0_correlation": correlations.get(0),
        "aggregation": "maximum-magnitude normalized cross-correlation over lags "
                       f"[-{max_lag}, {max_lag}]"})


def _feature_coherence(values_a: TypingSequence[float], values_b: TypingSequence[float],
                       cfg: Mapping[str, Any]) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    scipy_signal, reason = require("scipy.signal")
    if scipy_signal is None:
        return _Outcome(None, warning=reason)
    n = len(values_a)
    a = numpy.asarray(values_a, dtype=float)
    b = numpy.asarray(values_b, dtype=float)
    if float(a.std()) <= 0 or float(b.std()) <= 0:
        return _Outcome(None, warning="one or both sequences are constant; coherence is undefined")
    # A single, non-averaged Welch segment makes coherence trivially 1.0 at
    # every frequency (no cross-quantity to average away) -- capping nperseg
    # at n // 4 guarantees at least a handful of segments to average across
    # for short sequences; for a real book (thousands of points) n // 4 is
    # far bigger than welch_nperseg, so this changes nothing there.
    nperseg = max(4, min(int(cfg["welch_nperseg"]), n // 4))
    noverlap = nperseg // 2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            freqs, coh = scipy_signal.coherence(a, b, fs=1.0, window=cfg["welch_window"],
                                                nperseg=nperseg, noverlap=noverlap)
    except Exception as exc:  # pragma: no cover - library/runtime guard
        return _Outcome(None, warning=f"scipy.signal.coherence failed ({type(exc).__name__}: {exc})")
    if len(coh) < 2 or not numpy.all(numpy.isfinite(coh)):
        return _Outcome(None, warning="too few frequency bins to summarize coherence, or scipy "
                                      "returned a non-finite value")
    coh_ac, freqs_ac = coh[1:], freqs[1:]
    mean_coherence = float(numpy.mean(coh_ac))
    peak_index = int(numpy.argmax(coh_ac))
    scipy_module, _ = require("scipy")
    return _Outcome(mean_coherence, distribution={
        "peak_coherence": float(coh_ac[peak_index]), "peak_frequency": float(freqs_ac[peak_index]),
        "nperseg": nperseg, "noverlap": noverlap, "window": cfg["welch_window"],
        "library": "scipy", "library_version": _lib_version(scipy_module),
        "aggregation": "mean magnitude-squared coherence across non-DC frequency bins (0 = "
                       "unrelated at that frequency, 1 = a perfect linear relationship)"})


def _feature_cross_spectrum(values_a: TypingSequence[float], values_b: TypingSequence[float],
                            cfg: Mapping[str, Any]) -> _Outcome:
    numpy, reason = require("numpy")
    if numpy is None:
        return _Outcome(None, warning=reason)
    scipy_signal, reason = require("scipy.signal")
    if scipy_signal is None:
        return _Outcome(None, warning=reason)
    n = len(values_a)
    a = numpy.asarray(values_a, dtype=float)
    b = numpy.asarray(values_b, dtype=float)
    if float(a.std()) <= 0 or float(b.std()) <= 0:
        return _Outcome(None, warning="one or both sequences are constant; a cross-spectrum is "
                                      "undefined")
    # See _feature_coherence's comment: capping nperseg at n // 4 keeps this
    # off the same single-segment degeneracy for short sequences.
    nperseg = max(4, min(int(cfg["welch_nperseg"]), n // 4))
    noverlap = nperseg // 2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            freqs, csd = scipy_signal.csd(a, b, fs=1.0, window=cfg["welch_window"],
                                          nperseg=nperseg, noverlap=noverlap)
            _, coh = scipy_signal.coherence(a, b, fs=1.0, window=cfg["welch_window"],
                                            nperseg=nperseg, noverlap=noverlap)
    except Exception as exc:  # pragma: no cover - library/runtime guard
        return _Outcome(None, warning=f"scipy.signal.csd failed ({type(exc).__name__}: {exc})")
    if len(csd) < 2 or not numpy.all(numpy.isfinite(coh)):
        return _Outcome(None, warning="too few frequency bins to summarize a cross-spectrum, or "
                                      "scipy returned a non-finite value")
    csd_ac, freqs_ac, coh_ac = csd[1:], freqs[1:], coh[1:]
    # Report phase at the frequency where the two sequences are most coherent
    # (a phase reading at a frequency they barely share is not meaningful).
    peak_index = int(numpy.argmax(coh_ac))
    phase = float(numpy.angle(csd_ac[peak_index]))
    scipy_module, _ = require("scipy")
    return _Outcome(phase, distribution={
        "peak_frequency": float(freqs_ac[peak_index]), "peak_coherence": float(coh_ac[peak_index]),
        "magnitude": float(numpy.abs(csd_ac[peak_index])), "nperseg": nperseg, "noverlap": noverlap,
        "window": cfg["welch_window"], "library": "scipy", "library_version": _lib_version(scipy_module),
        "aggregation": "phase (radians) of the cross-spectral density, read at the frequency "
                       "where the two sequences' magnitude-squared coherence peaks; positive "
                       "means the second sequence leads the first at that frequency"},
        sample_size_sensitive=True)


_PAIR_FEATURES: dict[str, Callable[..., _Outcome]] = {
    "cross_correlation": _feature_cross_correlation,
    "coherence": _feature_coherence,
    "cross_spectrum": _feature_cross_spectrum,
}

assert set(_PAIR_FEATURES) == set(_PAIR_FEATURE_ORDER)


# --------------------------------------------------------------------- measuring

def _resolve_unit(feature_name: str, sample_unit: str) -> str:
    declared = FEATURE_UNITS[feature_name]
    return declared if declared is not None else f"cycles per {sample_unit}"


def _nominal_warning(sequence: "seq.Sequence") -> str | None:
    if "nominal" in sequence.description.lower():
        return ("this sequence's own values are a nominal label (see its description); "
                "spectral/amplitude-based interpretation here is experimental and may not "
                "correspond to a real periodicity")
    return None


def _measure_one(analysis: DocumentAnalysis, sequence_name: str, feature_name: str,
                 cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec = seq.SEQUENCES[sequence_name]
    metric_id = _metric_id(sequence_name, feature_name, spec.family)
    name = f"{FEATURE_LABELS[feature_name]} of {sequence_name.replace('_', ' ')}"

    sequence = seq.get_sequence(analysis, sequence_name, _sequence_settings(sequence_name, cfg))
    unit = _resolve_unit(feature_name, sequence.sample_unit)
    if not sequence.values:
        return finding(metric_id, name, None, unit, family=spec.family, sample_size=0,
                       warning=sequence.warning or "sequence has no values")

    min_length = int(cfg["min_lengths"].get(feature_name, DEFAULT_MIN_LENGTHS[feature_name]))
    if sequence.length < min_length:
        return finding(
            metric_id, name, None, unit, family=spec.family, sample_size=sequence.length,
            min_sample=min_length,
            warning=f"insufficient data: {sequence_name} has {sequence.length} "
                    f"{sequence.sample_unit} points, below the {min_length} the "
                    f"{feature_name} feature needs")

    working = list(sequence.values)
    if feature_name in _EXTENDED_FEATURES:
        ctx = {"analysis": analysis, "sequence_name": sequence_name, "cfg": cfg,
              "working": working, "family": spec.family}
        outcome = _EXTENDED_FEATURES[feature_name](working, cfg, ctx)
    else:
        outcome = _FEATURES[feature_name](working, cfg)

    distribution = dict(outcome.distribution)
    distribution.setdefault("sequence", sequence_name)
    distribution.setdefault("sequence_unit", sequence.unit)
    distribution.setdefault("sample_unit", sequence.sample_unit)
    if sequence.settings:
        distribution["sequence_settings"] = dict(sequence.settings)

    return finding(
        metric_id, name, outcome.value, unit, family=spec.family, sample_size=sequence.length,
        min_sample=min_length, distribution=distribution,
        warning=_join(_join(sequence.warning, outcome.warning), _nominal_warning(sequence)),
        sample_size_sensitive=outcome.sample_size_sensitive or feature_name in SAMPLE_SIZE_SENSITIVE_FEATURES)


def _measure_pair_one(analysis: DocumentAnalysis, name_a: str, name_b: str, feature_name: str,
                      cfg: Mapping[str, Any]) -> dict[str, Any]:
    spec_a = seq.SEQUENCES.get(name_a)
    spec_b = seq.SEQUENCES.get(name_b)
    family = spec_a.family if spec_a is not None else "sentence_rhythm"
    metric_id = _pair_metric_id(name_a, name_b, feature_name, family)
    name = (f"{PAIR_FEATURE_LABELS[feature_name]} between {name_a.replace('_', ' ')} "
           f"and {name_b.replace('_', ' ')}")
    unit = PAIR_FEATURE_UNITS[feature_name]

    if spec_a is None or spec_b is None:
        missing = [n for n, s in ((name_a, spec_a), (name_b, spec_b)) if s is None]
        return finding(metric_id, name, None, unit, family=family,
                       warning=f"unknown sequence(s) in pair: {missing}")

    sequence_a = seq.get_sequence(analysis, name_a, _sequence_settings(name_a, cfg))
    sequence_b = seq.get_sequence(analysis, name_b, _sequence_settings(name_b, cfg))

    if sequence_a.sample_unit != sequence_b.sample_unit:
        return finding(metric_id, name, None, unit, family=family,
                       warning=f"cross-sequence features need matching sample units: {name_a} "
                               f"is per-{sequence_a.sample_unit}, {name_b} is per-"
                               f"{sequence_b.sample_unit}")

    combined_warning = _join(sequence_a.warning, sequence_b.warning)
    if not sequence_a.values or not sequence_b.values:
        return finding(metric_id, name, None, unit, family=family, sample_size=0,
                       warning=combined_warning or "one or both sequences have no values")

    aligned_length = min(sequence_a.length, sequence_b.length)
    min_length = int(cfg["min_lengths"].get(feature_name, DEFAULT_MIN_LENGTHS[feature_name]))
    if aligned_length < min_length:
        return finding(
            metric_id, name, None, unit, family=family, sample_size=aligned_length,
            min_sample=min_length,
            warning=f"insufficient data: aligned length is {aligned_length} "
                    f"{sequence_a.sample_unit} points, below the {min_length} the "
                    f"{feature_name} feature needs")

    values_a = list(sequence_a.values[:aligned_length])
    values_b = list(sequence_b.values[:aligned_length])
    outcome = _PAIR_FEATURES[feature_name](values_a, values_b, cfg)

    distribution = dict(outcome.distribution)
    distribution.setdefault("sequence_a", name_a)
    distribution.setdefault("sequence_b", name_b)
    distribution.setdefault("sample_unit", sequence_a.sample_unit)
    distribution.setdefault("aligned_length", aligned_length)
    if aligned_length < sequence_a.length or aligned_length < sequence_b.length:
        distribution["truncated_to_shorter_sequence"] = True

    return finding(
        metric_id, name, outcome.value, unit, family=family, sample_size=aligned_length,
        min_sample=min_length, distribution=distribution,
        warning=_join(combined_warning, outcome.warning),
        sample_size_sensitive=outcome.sample_size_sensitive or feature_name in SAMPLE_SIZE_SENSITIVE_FEATURES)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = _settings(config)

    sequences = [name for name in cfg["sequences"] if name in seq.SEQUENCES]
    unknown_sequences = [name for name in cfg["sequences"] if name not in seq.SEQUENCES]
    single_features = [name for name in _FEATURE_ORDER if cfg["features"].get(name, False)]
    pair_features = [name for name in _PAIR_FEATURE_ORDER if cfg["features"].get(name, False)]
    unknown_features = [name for name in cfg["features"] if name not in _ALL_FEATURE_NAMES]
    valid_pairs = [(a, b) for a, b in cfg["pairs"] if a in seq.SEQUENCES and b in seq.SEQUENCES]
    invalid_pairs = [(a, b) for a, b in cfg["pairs"] if a not in seq.SEQUENCES or b not in seq.SEQUENCES]

    combos = [(sequence_name, feature_name) for sequence_name in sequences
             for feature_name in single_features]
    pair_combos = [(a, b, feature_name) for a, b in valid_pairs for feature_name in pair_features]

    if not combos and not pair_combos:
        problems = []
        if not sequences:
            problems.append("no valid entry in 'sequences'" +
                            (f" (unknown: {unknown_sequences})" if unknown_sequences else ""))
        if not single_features and not pair_features:
            problems.append("no feature enabled in 'features'" +
                            (f" (unknown: {unknown_features})" if unknown_features else ""))
        if cfg["pairs"] and not valid_pairs:
            problems.append(f"no valid pair in 'pairs' (invalid: {invalid_pairs})")
        return [finding("rhythm.signal_config", "Signal-processing suite configuration",
                        None, None, family=FAMILY, warning="; ".join(problems) or "nothing selected")]

    max_findings = max(1, cfg["max_findings"])
    ordered = [("single", *combo) for combo in combos] + [("pair", *combo) for combo in pair_combos]
    truncated = len(ordered) > max_findings
    ordered = ordered[:max_findings]

    out: list[dict[str, Any]] = []
    for item in ordered:
        if item[0] == "single":
            _, sequence_name, feature_name = item
            out.append(_measure_one(analysis, sequence_name, feature_name, cfg))
        else:
            _, name_a, name_b, feature_name = item
            out.append(_measure_pair_one(analysis, name_a, name_b, feature_name, cfg))

    if unknown_sequences or unknown_features or invalid_pairs:
        note = ", ".join(filter(None, [
            f"unknown sequences ignored: {unknown_sequences}" if unknown_sequences else "",
            f"unknown feature names ignored: {unknown_features}" if unknown_features else "",
            f"invalid pairs ignored: {invalid_pairs}" if invalid_pairs else "",
        ]))
        out.append(finding("rhythm.signal_config", "Signal-processing suite configuration",
                           len(ordered), "combinations", family=FAMILY, warning=note))
    if truncated:
        out.append(finding(
            "rhythm.signal_truncated", "Signal-processing suite output truncated by max_findings",
            max_findings, "findings", family=FAMILY,
            warning=f"{len(combos) + len(pair_combos)} (sequence, feature) combinations were "
                    f"selected but only max_findings={max_findings} were computed; raise "
                    f"metrics.signal_processing_suite.max_findings, or narrow 'sequences', "
                    f"'pairs' or 'features', to see the rest"))
    return out
