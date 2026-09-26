"""Automatic bulk feature generation over every named sequence.

:mod:`textgrader.sequences` names, builds and caches the ordered numeric
channels a document exposes (sentence length, punctuation density, parse
depth, sentiment, ...).  :mod:`textgrader.metrics.timeseries_suite` already
asks a curated, hand-designed battery of classical time-series questions of
each one, folding every extractor's internals into a *single* finding per
``(sequence, feature_group)`` pair specifically to keep its own combinatorics
bounded (see that module's docstring).  This suite is the deliberately
different half of Task 23: instead of hand-picking which questions to ask, it
turns each library's own *catalogue* of features into individually addressable
findings -- "each extracted scalar is one finding" -- subject to allowlist/
denylist filtering, a hard output cap and a deterministic selection rule when
that cap is hit.

**Three modes**, matching the task's own definition:

``minimal`` (the default)
    A compact, hand-picked, dependency-free set of ten robust summary
    statistics per sequence (extra quantiles, IQR, MAD, skewness, kurtosis,
    coefficient of variation, range) -- none of which is already published as
    its own stable id anywhere else in this codebase (they exist only nested
    inside ``timeseries_suite``'s folded ``dispersion`` finding, never as a
    top-level id), so nothing here is a second copy of an existing number
    under a new name.

``standard``
    ``minimal`` plus catch22's 22 canonical features (Lubba et al. 2019, via
    ``pycatch22``) and tsfresh's ``MinimalFCParameters`` preset (10 features,
    via ``tsfresh``), each of catch22's/tsfresh's numbers reported under its
    *own* stable id -- unlike ``timeseries_suite``, which folds all of catch22
    into 22 of its own ids too (a hand-curated table) but folds the whole of
    tsfresh into one.  Per this task's own mode definition ("catch22 plus
    selected tsfresh features"), catch22 is not skipped here even though
    ``timeseries_suite`` can compute the identical 22 numbers when its own
    ``catch22`` feature-group alias is explicitly selected (never by
    default): every catch22/tsfresh finding here names the sibling id in
    ``distribution["overlaps_existing_metric_id"]`` so the relationship is
    visible rather than silently duplicated, which is the "keep it, but flag
    the overlap" branch of this task's own instructions.

``comprehensive``
    ``standard`` plus tsfresh's ``ComprehensiveFCParameters`` preset (750+
    features per sequence as of the tsfresh version this was measured
    against), each exploded into its own sanitized id, capped by
    ``max_findings`` with a deterministic (sorted-name) selection rule.  This
    is the one mode genuinely capable of profile bloat -- see "Profile size"
    below -- and is never selected by default.

**Stable, sanitized ids.**  Every finding's id is
``style.autofeature_<sequence>_<extractor>_<feature>``, where ``<extractor>``
is ``stats``, ``catch22``, ``tsfresh`` (the minimal preset) or
``tsfresh_comprehensive`` (comprehensive mode only), and ``<feature>`` is the
extractor's own feature name run through :func:`sanitize_feature_name`: lowercased,
every run of non-alphanumeric characters (tsfresh's ``__`` separators, ``"``,
``(``, ``)``, ``,``, spaces) collapsed to a single ``_``, trimmed, and -- for
the rare tsfresh comprehensive name over 80 characters -- truncated with a
short deterministic hash suffix.  :func:`dedupe_sanitized_ids` then resolves
any residual collision (two distinct raw names sanitizing to the same string)
by appending a short hash of the *raw* name, deterministically and in sorted
raw-name order, so the same raw-name set always produces the same id set no
matter how many times it runs.  ``tests/test_automatic_features.py`` proves
both properties against tsfresh's real, installed ``ComprehensiveFCParameters``
column names, not just a synthetic example.

**Profile size.**  A corpus profile (``textgrader.corpus.build_profile``)
never stores a finding's ``distribution`` -- only its scalar ``value``, once
per book, plus one pooled distribution (with every book's value inside it)
per metric id.  Profile growth from this suite is therefore governed entirely
by *how many metric ids* it emits, not by any one book's individual number.
Measured directly (see the module's own validation and this task's report):
the default configuration (``mode="minimal"``, the three dependency-free
default sequences) emits 30 metric ids, comprehensive mode with no cap would
emit thousands.  ``max_findings`` defaults to 60/150/400 for
minimal/standard/comprehensive respectively, keeping every mode's contribution
to a 50-book profile bounded and stated (see the task report for the measured
byte figures).

**Deferred / never emitted.**  catch24's two raw-mean/raw-variance extras are
skipped entirely here (not merely flagged): they would triple-duplicate an
existing number (``timeseries_suite``'s ``dispersion.mean``/``std`` AND its
own ``catch24_raw_mean``/``catch24_raw_variance``, computed from the exact
same ``pycatch22.catch22_all(catch24=True)`` call) with nothing this suite's
own ``stats`` extractor does not already offer a genuinely different view of
(the coefficient of variation).  Cross-sequence features (the task's own
"optional" bullet) are left for a future pass: this suite already reuses
``sequences.py``'s full registry with no code change per sequence, and adding
a pairwise cross-sequence extractor is additive, not a redesign, when wanted.

**Correlation: flagged where it is analytic, not yet where it would need a
corpus.**  ``related_features``/``overlaps_existing_metric_id`` above are
*known-by-construction* relationships (a shared formula term, a shared
library call under a different id) checkable from one document.  A genuine
empirical correlation between two catch22 features, or between two of
tsfresh's ~750 comprehensive columns, is a property of *many* documents' worth
of values, the same leave-one-out corpus statistics ``anomaly_suite`` computes
at grading time from a corpus profile's cached feature vectors -- and no
suite in this codebase, including this one, computes that inside a single
``measure()`` call over one document.  Building it here would mean this
suite's own corpus-profile section (feature schema, coverage, and now a
pairwise correlation matrix over however many hundreds of ids a mode selects)
rather than a same-pass bolt-on; a real, scoped candidate for a future pass,
left honestly undone rather than approximated.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import warnings
from typing import Any, Mapping, Sequence as TypingSequence

from .. import sequences as seq
from .. import stats as stats_module
from ..document import DocumentAnalysis
from ..optional import require
from .common import MODERATE, finding, option

FAMILY = "distribution_shape"  # fallback only; every real finding uses its sequence's own family
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 5
UNIT_SENSITIVE = False

#: Bumped whenever the sanitizing scheme, the stats catalogue or the id
#: format changes in a way that could shift ids -- recorded in every
#: finding's distribution alongside the mode and library versions, per this
#: task's "settings/version metadata" requirement.
AUTOMATIC_FEATURES_VERSION = 1

MODES = ("minimal", "standard", "comprehensive")
DEFAULT_MODE = "minimal"

#: No optional package, defined on any text with a handful of sentences and
#: two paragraphs -- the same three sequences timeseries_suite/
#: signal_processing_suite default to, for the same reason (see their
#: module docstrings): identical across every environment this runs in, and
#: none needs the shared spaCy parse or an embedding/topic model.
DEFAULT_SEQUENCES = ("sentence_words", "paragraph_words", "sentence_punctuation")

_DEFAULT_EXTRACTORS_BY_MODE: dict[str, tuple[str, ...]] = {
    "minimal": ("stats",),
    "standard": ("stats", "catch22", "tsfresh"),
    "comprehensive": ("stats", "catch22", "tsfresh", "tsfresh_comprehensive"),
}
EXTRACTOR_NAMES = ("stats", "catch22", "tsfresh", "tsfresh_comprehensive")

#: See "Profile size" in the module docstring: bounded and stated per mode.
_DEFAULT_MAX_FINDINGS = {"minimal": 60, "standard": 150, "comprehensive": 400}

#: Sequences longer than this are deterministically downsampled (evenly
#: strided, never randomly) before catch22/tsfresh run, which is where the
#: cost genuinely explodes on a book-length sequence -- see the module's own
#: docstring section on the ``syllable_stress`` benchmark and the task report
#: for the measured seconds this bounds.  The dependency-free ``stats``
#: extractor is left at full length: every one of its features is O(n log n)
#: or better and was measured well under a second even at ~250,000 points.
DEFAULT_MAX_SEQUENCE_LENGTH = 3000

_EXTRACTOR_MIN_LENGTH = {"stats": 5, "catch22": 30, "tsfresh": 20, "tsfresh_comprehensive": 30}

# Sequences that take their own extra settings beyond the shared defaults --
# copied from timeseries_suite's own split rather than imported, so this
# module's settings stay self-contained (see that module's docstring for why
# each sequence needs what it needs).
_WINDOW_SEQUENCES = frozenset({"window_dialogue_fraction", "window_pronoun_rate"})
_RARITY_SEQUENCES = frozenset({"sentence_content_rarity"})
_EMBEDDING_SEQUENCES = frozenset({"sentence_similarity_prev", "sentence_distance_centroid"})
_TOPIC_SEQUENCES = frozenset({"window_topic_id"})


def _lib_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


# ------------------------------------------------------------- id sanitizing

_SANITIZE_RE = re.compile(r"[^a-z0-9]+")
_MAX_ID_SEGMENT = 80


def sanitize_feature_name(raw: str) -> str:
    """A deterministic, filesystem/id-safe rendering of one extractor's raw feature name.

    tsfresh feature names carry ``__`` parameter separators, quoted string
    parameters (``attr_"rvalue"``) and commas; catch22's are already
    alphanumeric-plus-underscore but in ``CamelCase``.  Both are lowercased
    and every run of anything else collapsed to one ``_``, so
    ``'value__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"'``
    and ``'DN_HistogramMode_5'`` both become plain, stable, readable
    ``snake_case`` strings.  A name over 80 characters (comprehensive tsfresh
    can produce these) is truncated with an 8-character hash of the *full*
    raw name appended, so truncation itself cannot silently create a
    collision between two long names that only differ near the end.
    """

    name = raw.lower()
    name = _SANITIZE_RE.sub("_", name).strip("_")
    if not name:
        name = "feature"
    if len(name) > _MAX_ID_SEGMENT:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:_MAX_ID_SEGMENT - 9]}_{digest}"
    return name


def dedupe_sanitized_ids(raw_names: TypingSequence[str]) -> dict[str, str]:
    """``{raw_name: collision-free sanitized id}`` for one extractor's whole catalogue.

    Processed in sorted *raw*-name order so the mapping never depends on
    iteration order, and a raw name's id never changes just because some
    other, later-sorting raw name was also present.  A genuine collision (two
    distinct raw names sanitizing to the same string) is broken by appending
    a short hash of the raw name that lost the plain spelling; the loop
    widens that hash if it, in turn, collides, which cannot fail to terminate
    since a SHA-1 hex digest has 40 distinct prefix lengths to grow into
    before it would have to repeat.
    """

    mapping: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for raw in sorted(set(raw_names)):
        base = sanitize_feature_name(raw)
        candidate = base
        suffix_len = 6
        while candidate in claimed and claimed[candidate] != raw:
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:suffix_len]
            candidate = f"{base}_{digest}"
            suffix_len += 2
        claimed[candidate] = raw
        mapping[raw] = candidate
    return mapping


# ------------------------------------------------------------------ extractor: stats

#: Every one of these ten is a hand-picked, dependency-free, deterministic
#: statistic that is NOT already published as its own stable metric id
#: anywhere else in this codebase (p10/p90/IQR/MAD/skew/kurtosis live only
#: nested inside timeseries_suite's folded "dispersion" finding today) -- see
#: the module docstring's "minimal" mode description.
STATS_FEATURES = ("quantile_p05", "quantile_p10", "quantile_p90", "quantile_p95",
                  "iqr", "mad", "skewness", "kurtosis", "cv", "range")

#: Which stats features are reported in the sequence's own unit (words,
#: marks, ...) versus scale-free (skewness/kurtosis are standardized moments;
#: cv is a ratio) -- the same "unit, or scale-free" split
#: timeseries_suite's FEATURE_UNITS makes for dispersion/trend/etc.
_STATS_UNIT_IS_SEQUENCE_UNIT = frozenset({
    "quantile_p05", "quantile_p10", "quantile_p90", "quantile_p95", "iqr", "mad", "range"})

#: Hand-annotated analytic relationships within this suite's own stats group,
#: for the "flag correlated features, never delete" requirement. A single
#: document gives one scalar per feature, not a sample to correlate across,
#: so this records known FORMULA-level relationships (two quantiles bounding
#: the same tail, IQR being built from two of the quantiles here) rather than
#: an empirically-measured correlation, which would need a corpus of many
#: documents' vectors -- a natural extension, not implemented this pass.
_STATS_RELATED = {
    "quantile_p05": ("quantile_p10",), "quantile_p10": ("quantile_p05", "iqr"),
    "quantile_p90": ("quantile_p95", "iqr"), "quantile_p95": ("quantile_p90",),
    "iqr": ("quantile_p10", "quantile_p90", "mad"), "mad": ("iqr",),
    "skewness": ("kurtosis",), "kurtosis": ("skewness",),
    "cv": (), "range": ("quantile_p05", "quantile_p95"),
}

_STATS_SAMPLE_SIZE_SENSITIVE = frozenset({"skewness", "kurtosis"})


def _compute_stats(values: TypingSequence[float]) -> dict[str, tuple[float | None, str | None]]:
    ordered = sorted(values)
    n = len(ordered)
    out: dict[str, tuple[float | None, str | None]] = {}
    p05 = stats_module.quantile(ordered, 0.05)
    p10 = stats_module.quantile(ordered, 0.10)
    p25 = stats_module.quantile(ordered, 0.25)
    p75 = stats_module.quantile(ordered, 0.75)
    p90 = stats_module.quantile(ordered, 0.90)
    p95 = stats_module.quantile(ordered, 0.95)
    out["quantile_p05"] = (p05, None)
    out["quantile_p10"] = (p10, None)
    out["quantile_p90"] = (p90, None)
    out["quantile_p95"] = (p95, None)
    iqr = (p75 - p25) if p75 is not None and p25 is not None else None
    out["iqr"] = (iqr, None)
    median = statistics.median(ordered)
    mad = statistics.median([abs(v - median) for v in ordered]) if ordered else None
    out["mad"] = (mad, None)
    skew = stats_module._skewness(list(values))
    out["skewness"] = (skew, None if n >= 3 else "needs at least 3 points for skewness")
    kurtosis = stats_module._kurtosis_excess(list(values))
    out["kurtosis"] = (kurtosis, None if n >= 4 else "needs at least 4 points for kurtosis")
    mean = statistics.fmean(values) if values else 0.0
    sd = statistics.pstdev(values) if values else 0.0
    out["cv"] = ((sd / mean) if mean else None,
                None if mean else "the mean is zero; coefficient of variation is undefined")
    out["range"] = ((ordered[-1] - ordered[0]) if ordered else None, None)
    return out


# ---------------------------------------------------------------- extractor: catch22

#: The 22 canonical catch22 feature names (Lubba et al. 2019), exactly as
#: ``pycatch22.catch22_all`` returns them -- hardcoded so this suite's id set
#: is defined and stable even when pycatch22 is not installed (every id still
#: exists; it simply reports "unavailable"), the same convention every other
#: optional-package metric in this codebase follows.
CATCH22_RAW_NAMES = (
    "DN_HistogramMode_5", "DN_HistogramMode_10", "CO_f1ecac", "CO_FirstMin_ac",
    "CO_HistogramAMI_even_2_5", "CO_trev_1_num", "MD_hrv_classic_pnn40",
    "SB_BinaryStats_mean_longstretch1", "SB_TransitionMatrix_3ac_sumdiagcov",
    "PD_PeriodicityWang_th0_01", "CO_Embed2_Dist_tau_d_expfit_meandiff",
    "IN_AutoMutualInfoStats_40_gaussian_fmmi", "FC_LocalSimple_mean1_tauresrat",
    "DN_OutlierInclude_p_001_mdrmd", "DN_OutlierInclude_n_001_mdrmd",
    "SP_Summaries_welch_rect_area_5_1", "SB_BinaryStats_diff_longstretch0",
    "SB_MotifThree_quantile_hh", "SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1",
    "SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1", "SP_Summaries_welch_rect_centroid",
    "FC_LocalSimple_mean3_stderr",
)
assert len(CATCH22_RAW_NAMES) == 22

#: Purely a cross-reference so ``overlaps_existing_metric_id`` can name the
#: exact sibling finding ``timeseries_suite`` would compute for the same
#: pycatch22 raw feature -- copied from that module's own hand table (not
#: imported: it is a private, suite-internal structure there, and this is
#: only documentation metadata, never used to compute a value).
_CATCH22_TIMESERIES_STABLE_SUFFIX = dict(zip(CATCH22_RAW_NAMES, (
    "histogram_mode_5bin", "histogram_mode_10bin", "acf_1e_decay_time", "acf_first_minimum",
    "auto_mutual_info_2bin_lag5", "time_reversal_asymmetry", "large_step_fraction",
    "longest_above_mean_run", "transition_matrix_trace", "periodicity_wang",
    "embed2_dist_expfit", "auto_mutual_info_first_min", "forecast_ar1_error_ratio",
    "outlier_timing_positive", "outlier_timing_negative", "spectral_low_freq_power",
    "longest_decreasing_run", "motif3_entropy", "rs_range_low_scale_fit",
    "dfa_low_scale_fit", "spectral_centroid", "forecast_ma3_error_stderr",
)))


def _catch22_raw(analysis: DocumentAnalysis, sequence_name: str, working: TypingSequence[float],
                 detrended: bool) -> tuple[dict[str, float] | None, str | None, Any]:
    key = f"automatic_features:catch22:{sequence_name}:detrend={detrended}:n={len(working)}"

    def build() -> tuple[dict[str, float] | None, str | None, Any]:
        module, reason = require("pycatch22")
        if module is None:
            return None, reason, None
        try:
            result = module.catch22_all(list(working), catch24=False)
            raw = dict(zip(result["names"], result["values"]))
        except Exception as exc:  # pragma: no cover - library/runtime guard
            return None, f"pycatch22.catch22_all failed ({type(exc).__name__}: {exc})", None
        # NaN is never equal to itself, so leaving it in place would make two
        # otherwise-identical runs compare unequal -- this suite's own
        # determinism requirement -- and it is not a real finite measurement
        # either way (pycatch22 returns it for zero-variance/degenerate
        # input); normalized to None here, the same convention tsfresh's own
        # extraction already follows below.
        cleaned = {name: (float(value) if value is not None and math.isfinite(value) else None)
                  for name, value in raw.items()}
        return cleaned, None, module

    return analysis.memo(key, build)


# ----------------------------------------------------------------- extractor: tsfresh

_TSFRESH_PRESETS = {"minimal": "MinimalFCParameters", "comprehensive": "ComprehensiveFCParameters"}

#: tsfresh's own ``MinimalFCParameters`` preset: 10 fixed, unparameterized
#: column names, stable across tsfresh releases -- hardcoded for the same
#: reason ``CATCH22_RAW_NAMES`` is: the id set must exist even when tsfresh is
#: not installed.
TSFRESH_MINIMAL_RAW_NAMES = ("sum_values", "median", "mean", "length", "standard_deviation",
                             "variance", "root_mean_square", "maximum", "absolute_maximum",
                             "minimum")


def _tsfresh_raw(analysis: DocumentAnalysis, sequence_name: str, working: TypingSequence[float],
                 detrended: bool, preset: str) -> tuple[dict[str, float | None] | None, str | None, str | None]:
    key = f"automatic_features:tsfresh:{sequence_name}:detrend={detrended}:preset={preset}:n={len(working)}"

    def build() -> tuple[dict[str, float | None] | None, str | None, str | None]:
        preset_name = _TSFRESH_PRESETS.get(preset)
        if preset_name is None:
            return None, f"unknown tsfresh preset {preset!r}; use one of {sorted(_TSFRESH_PRESETS)}", None
        tsfresh_module, reason = require("tsfresh")
        if tsfresh_module is None:
            return None, reason, None
        pandas_module, pandas_reason = require("pandas")
        if pandas_module is None:
            return None, pandas_reason, None
        try:
            from tsfresh.feature_extraction import extract_features
            from tsfresh.feature_extraction import settings as tsfresh_settings
            fc_parameters = getattr(tsfresh_settings, preset_name)()
            values = list(working)
            frame = pandas_module.DataFrame({
                "id": [0] * len(values), "time": range(len(values)), "value": values})
            # Serial (n_jobs=0), never tsfresh's own multiprocessing: this
            # task's determinism requirement asks for bit-identical
            # comprehensive-mode output across runs with the same settings,
            # and tsfresh's process pool has been observed elsewhere in this
            # codebase to reorder floating-point summation across workers.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                extracted = extract_features(
                    frame, column_id="id", column_sort="time", column_value="value",
                    default_fc_parameters=fc_parameters, disable_progressbar=True, n_jobs=0)
        except Exception as exc:  # pragma: no cover - library/runtime guard
            return None, f"tsfresh.extract_features failed ({type(exc).__name__}: {exc})", None
        row = extracted.iloc[0].to_dict()
        out: dict[str, float | None] = {}
        for raw_name, raw_value in row.items():
            suffix = raw_name.split("__", 1)[1] if "__" in raw_name else raw_name
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and math.isfinite(numeric):
                # Rounded at the boundary so a library whose last bit can
                # wobble between otherwise-identical runs (observed with
                # some tsfresh calculators under BLAS threading) cannot break
                # comprehensive mode's bit-identical-output requirement.
                numeric = round(numeric, 10)
            else:
                numeric = None
            out[suffix] = numeric
        return out, None, _lib_version(tsfresh_module)

    return analysis.memo(key, build)


# --------------------------------------------------------------- shared machinery

def _sequence_settings(name: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    if name in _WINDOW_SEQUENCES:
        return {"window_words": cfg["window_words"]}
    if name in _RARITY_SEQUENCES:
        return {"language": cfg["language"]}
    if name in _EMBEDDING_SEQUENCES:
        return {"model": cfg["embedding_model"]}
    if name in _TOPIC_SEQUENCES:
        return {"window_words": cfg["window_words"], "n_topics": cfg["topic_n_topics"],
               "topic_model": cfg["topic_model"], "random_state": cfg["topic_random_state"],
               "max_features": cfg["topic_max_features"]}
    return {}


def _residuals(values: TypingSequence[float]) -> list[float]:
    n = len(values)
    xs = list(range(n))
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(values)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0:
        return list(values)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    slope = cov / var_x
    intercept = mean_y - slope * mean_x
    return [v - (intercept + slope * x) for x, v in zip(xs, values)]


def _bounded(values: TypingSequence[float], max_length: int) -> tuple[list[float], int]:
    """Deterministic, evenly-strided downsample -- never a random sample.

    Runs of ``catch22``/``tsfresh`` on a 250,000-point sequence
    (``syllable_stress`` over a whole novel) are the one place this suite's
    cost genuinely explodes; see the module docstring and the task report for
    the measured seconds this bounds.  The stride is recorded in every
    finding's ``distribution`` so a reader knows the number came from a
    subsampled series, not the full one.
    """

    n = len(values)
    if max_length <= 0 or n <= max_length:
        return list(values), 1
    stride = math.ceil(n / max_length)
    return list(values[::stride]), stride


def _passes(name: str, allow_re: "re.Pattern[str] | None", deny_re: "re.Pattern[str] | None") -> bool:
    if allow_re is not None and not allow_re.search(name):
        return False
    if deny_re is not None and deny_re.search(name):
        return False
    return True


def _metric_id(sequence_name: str, extractor: str, feature: str) -> str:
    return f"style.autofeature_{sequence_name}_{extractor}_{feature}"


# ------------------------------------------------------------------------- settings

def _settings(config: Mapping[str, Any] | None) -> dict[str, Any]:
    mode = str(option(config, "mode", DEFAULT_MODE)).lower()
    valid_mode = mode if mode in MODES else DEFAULT_MODE
    default_extractors = _DEFAULT_EXTRACTORS_BY_MODE[valid_mode]
    default_tsfresh_preset = "comprehensive" if valid_mode == "comprehensive" else "minimal"
    default_max_findings = _DEFAULT_MAX_FINDINGS[valid_mode]
    return {
        "mode": mode,
        "sequences": list(option(config, "sequences", DEFAULT_SEQUENCES)),
        "extractors": list(option(config, "extractors", default_extractors)),
        "tsfresh_feature_set": str(option(config, "tsfresh_feature_set", default_tsfresh_preset)),
        "allow_pattern": option(config, "allow_pattern", None),
        "deny_pattern": option(config, "deny_pattern", None),
        "max_findings": int(option(config, "max_findings", default_max_findings)),
        "max_sequence_length": int(option(config, "max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)),
        "min_lengths": dict(option(config, "min_lengths", {})),
        "detrend": bool(option(config, "detrend", False)),
        "window_words": int(option(config, "window_words", 2000)),
        "embedding_model": option(config, "embedding_model", "all-MiniLM-L6-v2"),
        "language": option(config, "language", "en"),
        "topic_n_topics": int(option(config, "topic_n_topics", 4)),
        "topic_model": str(option(config, "topic_model", "nmf")),
        "topic_random_state": int(option(config, "topic_random_state", 42)),
        "topic_max_features": int(option(config, "topic_max_features", 2000)),
        "include_full_feature_vector": bool(option(config, "include_full_feature_vector", False)),
        "full_feature_vector_max_features": int(
            option(config, "full_feature_vector_max_features", 2000)),
    }


# -------------------------------------------------------------- candidate generation

class _Candidate:
    __slots__ = ("sequence_name", "extractor", "raw_name", "feature_id")

    def __init__(self, sequence_name: str, extractor: str, raw_name: str, feature_id: str):
        self.sequence_name = sequence_name
        self.extractor = extractor
        self.raw_name = raw_name
        self.feature_id = feature_id


def _expected_raw_names(extractor: str) -> tuple[str, ...] | None:
    """The extractor's fixed catalogue, or ``None`` when it can only be known
    by actually calling the library (tsfresh's comprehensive preset)."""

    if extractor == "stats":
        return STATS_FEATURES
    if extractor == "catch22":
        return CATCH22_RAW_NAMES
    if extractor == "tsfresh":
        return TSFRESH_MINIMAL_RAW_NAMES
    return None


def _prepare_sequence(analysis: DocumentAnalysis, sequence_name: str,
                      cfg: Mapping[str, Any]) -> tuple[Any, list[float]]:
    sequence = seq.get_sequence(analysis, sequence_name, _sequence_settings(sequence_name, cfg))
    working = list(sequence.values)
    if cfg["detrend"] and len(working) >= 2:
        working = _residuals(working)
    return sequence, working


def _extractor_values(analysis: DocumentAnalysis, sequence_name: str, extractor: str,
                      working: list[float], detrended: bool,
                      cfg: Mapping[str, Any]) -> tuple[dict[str, float | None] | None, str | None,
                                                       dict[str, Any]]:
    """``(raw_name -> value, unavailable_reason, extra_distribution_fields)``."""

    if extractor == "stats":
        try:
            computed = _compute_stats(working)
        except Exception as exc:  # pragma: no cover - defensive guard
            return None, f"the stats extractor failed ({type(exc).__name__}: {exc})", {}
        values: dict[str, float | None] = {}
        notes: dict[str, str] = {}
        for stat_name, (value, note) in computed.items():
            if value is not None and not math.isfinite(value):
                note = _join(note, "non-finite value")
                value = None
            values[stat_name] = value
            if note:
                notes[stat_name] = note
        return values, None, {"notes": notes}
    bounded, stride = _bounded(working, cfg["max_sequence_length"])
    extra = {"downsample_stride": stride, "sequence_length_used": len(bounded)}
    if extractor == "catch22":
        values_by_name, reason, module = _catch22_raw(analysis, sequence_name, bounded, detrended)
        if values_by_name is None:
            return None, reason, extra
        extra["library"] = "pycatch22"
        extra["library_version"] = _lib_version(module)
        return values_by_name, None, extra
    if extractor in ("tsfresh", "tsfresh_comprehensive"):
        preset = "minimal" if extractor == "tsfresh" else cfg["tsfresh_feature_set"]
        values_by_name, reason, version = _tsfresh_raw(analysis, sequence_name, bounded,
                                                        detrended, preset)
        if values_by_name is None:
            return None, reason, extra
        extra["library"] = "tsfresh"
        extra["library_version"] = version
        extra["tsfresh_preset"] = preset
        return values_by_name, None, extra
    return None, f"unknown extractor {extractor!r}", extra


# ------------------------------------------------------------------------- measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = _settings(config)
    raw_mode = str(option(config, "mode", DEFAULT_MODE)).lower()
    if raw_mode not in MODES:
        return [finding("style.autofeature_config", "Automatic feature generation configuration",
                        None, None, family=FAMILY,
                        warning=f"unknown mode {raw_mode!r}; use one of {MODES}")]

    sequences = [name for name in dict.fromkeys(cfg["sequences"]) if name in seq.SEQUENCES]
    unknown_sequences = [name for name in cfg["sequences"] if name not in seq.SEQUENCES]
    extractors = [name for name in dict.fromkeys(cfg["extractors"]) if name in EXTRACTOR_NAMES]
    unknown_extractors = [name for name in cfg["extractors"] if name not in EXTRACTOR_NAMES]

    if not sequences or not extractors:
        problems = []
        if not sequences:
            problems.append("no valid entry in 'sequences'" +
                            (f" (unknown: {unknown_sequences})" if unknown_sequences else ""))
        if not extractors:
            problems.append("no valid entry in 'extractors'" +
                            (f" (unknown: {unknown_extractors})" if unknown_extractors else ""))
        return [finding("style.autofeature_config", "Automatic feature generation configuration",
                        None, None, family=FAMILY, warning="; ".join(problems))]

    allow_re = re.compile(cfg["allow_pattern"]) if cfg["allow_pattern"] else None
    deny_re = re.compile(cfg["deny_pattern"]) if cfg["deny_pattern"] else None

    candidates: list[_Candidate] = []
    # (sequence, extractor) -> resolved values/reason/extra, computed once and
    # reused both while building candidates and while building findings, so a
    # tsfresh/catch22 call never runs twice for the same pair.
    resolved: dict[tuple[str, str], tuple[dict[str, float | None] | None, str | None, dict[str, Any]]] = {}
    sequence_cache: dict[str, tuple[Any, list[float]]] = {}
    library_versions: dict[str, str] = {}
    full_vectors: dict[str, dict[str, float | None]] = {}

    for sequence_name in sequences:
        sequence_obj, working = _prepare_sequence(analysis, sequence_name, cfg)
        sequence_cache[sequence_name] = (sequence_obj, working)
        for extractor in extractors:
            min_length = int(cfg["min_lengths"].get(extractor, _EXTRACTOR_MIN_LENGTH[extractor]))
            expected = _expected_raw_names(extractor)
            if len(working) < min_length:
                # Below the floor: still enumerate the fixed catalogues (so
                # the id set is stable and each reports insufficient_data),
                # but never call a library on data too short to mean
                # anything -- and for the dynamic tsfresh_comprehensive
                # catalogue there is nothing to enumerate at all, so this
                # sequence/extractor pair is skipped rather than guessed.
                if expected is None:
                    continue
                resolved[(sequence_name, extractor)] = (
                    None, f"insufficient data: {sequence_name} has {len(working)} "
                          f"{sequence_obj.sample_unit} points, below the {min_length} the "
                          f"{extractor} extractor needs", {})
                names = expected
            else:
                values_by_name, reason, extra = _extractor_values(
                    analysis, sequence_name, extractor, working, cfg["detrend"], cfg)
                resolved[(sequence_name, extractor)] = (values_by_name, reason, extra)
                if extra.get("library_version"):
                    library_versions[extra["library"]] = extra["library_version"]
                if values_by_name is None:
                    if expected is None:
                        continue
                    names = expected
                else:
                    names = expected if expected is not None else tuple(sorted(values_by_name))
                    if (extractor == "tsfresh_comprehensive" and cfg["include_full_feature_vector"]
                            and values_by_name):
                        cap = cfg["full_feature_vector_max_features"]
                        id_map = dedupe_sanitized_ids(names)
                        full_vectors[sequence_name] = {
                            id_map[raw]: values_by_name[raw]
                            for raw in sorted(names)[:cap]}
            id_map = dedupe_sanitized_ids(names)
            for raw_name in sorted(names):
                feature_id = id_map[raw_name]
                if not _passes(feature_id, allow_re, deny_re):
                    continue
                candidates.append(_Candidate(sequence_name, extractor, raw_name, feature_id))

    max_findings = max(1, cfg["max_findings"])
    truncated = len(candidates) > max_findings
    kept = candidates[:max_findings]

    out: list[dict[str, Any]] = []
    for candidate in kept:
        out.append(_build_finding(analysis, candidate, sequence_cache, resolved, cfg))

    manifest_distribution = {
        "mode": cfg["mode"], "automatic_features_version": AUTOMATIC_FEATURES_VERSION,
        "sequences": sequences, "extractors": extractors,
        "tsfresh_feature_set": cfg["tsfresh_feature_set"],
        "max_findings": max_findings, "max_sequence_length": cfg["max_sequence_length"],
        "detrend": cfg["detrend"], "candidates_generated": len(candidates),
        "findings_emitted": len(kept), "library_versions": library_versions,
    }
    if full_vectors:
        manifest_distribution["full_feature_vectors"] = full_vectors
    out.append(finding("style.autofeature_manifest", "Automatic feature generation manifest",
                       len(kept), "findings", family=FAMILY, distribution=manifest_distribution))

    if unknown_sequences or unknown_extractors:
        note = "; ".join(filter(None, [
            f"unknown sequences ignored: {unknown_sequences}" if unknown_sequences else "",
            f"unknown extractors ignored: {unknown_extractors}" if unknown_extractors else ""]))
        out.append(finding("style.autofeature_config", "Automatic feature generation configuration",
                           len(kept), "findings", family=FAMILY, warning=note))
    if truncated:
        out.append(finding(
            "style.autofeature_truncated", "Automatic feature generation output truncated by max_findings",
            max_findings, "findings", family=FAMILY,
            warning=f"{len(candidates)} candidate features were selected but only "
                    f"max_findings={max_findings} were emitted, chosen deterministically in "
                    f"(sequence, extractor, sorted feature name) order; raise "
                    f"metrics.automatic_feature_generation.max_findings, or narrow 'sequences', "
                    f"'extractors' or the allow/deny patterns, to see the rest"))
    return out


def _build_finding(analysis: DocumentAnalysis, candidate: _Candidate,
                   sequence_cache: dict[str, tuple[Any, list[float]]],
                   resolved: dict[tuple[str, str], tuple[dict[str, float | None] | None, str | None,
                                                          dict[str, Any]]],
                   cfg: Mapping[str, Any]) -> dict[str, Any]:
    sequence_name, extractor, raw_name, feature_id = (
        candidate.sequence_name, candidate.extractor, candidate.raw_name, candidate.feature_id)
    spec = seq.SEQUENCES[sequence_name]
    sequence_obj, working = sequence_cache[sequence_name]
    metric_id = _metric_id(sequence_name, extractor, feature_id)
    name = f"Automatic {extractor} feature '{feature_id}' of {sequence_name.replace('_', ' ')}"
    # No single unit fits catch22's or tsfresh's whole catalogue (catch22
    # alone spans z-scores, lag steps, bits, proportions and points across
    # its 22 features; tsfresh's hundreds span even more), so those stay
    # unitless here rather than implying a precision this bulk,
    # per-catalogue-feature design does not have. "stats" features that ARE
    # in the sequence's own unit (the quantiles, IQR, MAD, range) report it;
    # the standardized ones (skewness, kurtosis, cv) do not.
    unit = (sequence_obj.unit if extractor == "stats"
                              and raw_name in _STATS_UNIT_IS_SEQUENCE_UNIT else None)

    if not sequence_obj.values:
        return finding(metric_id, name, None, unit, family=spec.family, sample_size=0,
                       warning=sequence_obj.warning or "sequence has no values")

    values_by_name, reason, extra = resolved.get((sequence_name, extractor), (None, "not computed", {}))
    distribution: dict[str, Any] = {
        "sequence": sequence_name, "extractor": extractor, "feature": feature_id,
        "raw_feature_name": raw_name, "mode": cfg["mode"],
        "automatic_features_version": AUTOMATIC_FEATURES_VERSION,
        "sequence_unit": sequence_obj.unit, "sample_unit": sequence_obj.sample_unit,
        "detrended": cfg["detrend"],
    }
    # "notes" carries every feature's own note keyed by raw name (read
    # separately below, per-feature) -- excluded here so one feature's
    # finding does not carry every sibling feature's note too.
    distribution.update({k: v for k, v in extra.items() if k != "notes"})

    min_length = int(cfg["min_lengths"].get(extractor, _EXTRACTOR_MIN_LENGTH[extractor]))
    if len(working) < min_length:
        return finding(metric_id, name, None, unit, family=spec.family, sample_size=len(working),
                       min_sample=min_length, distribution=distribution,
                       warning=_join(sequence_obj.warning, reason))

    if extractor == "stats":
        if values_by_name is None:
            return finding(metric_id, name, None, unit, family=spec.family,
                           sample_size=len(working), min_sample=min_length,
                           distribution=distribution, warning=_join(sequence_obj.warning, reason))
        outcome_value = values_by_name.get(raw_name)
        outcome_note = extra.get("notes", {}).get(raw_name)
        if raw_name in _STATS_RELATED and _STATS_RELATED[raw_name]:
            distribution["related_features"] = [
                _metric_id(sequence_name, "stats", sibling) for sibling in _STATS_RELATED[raw_name]]
            distribution["related_features_note"] = (
                "flagged as analytically related (shares a formula component or the same "
                "underlying tail), never deleted -- per this project's policy on correlated "
                "measurements")
        return finding(metric_id, name, outcome_value, unit, family=spec.family,
                       sample_size=len(working), min_sample=min_length, distribution=distribution,
                       warning=_join(sequence_obj.warning, outcome_note),
                       sample_size_sensitive=raw_name in _STATS_SAMPLE_SIZE_SENSITIVE)

    if values_by_name is None:
        return finding(metric_id, name, None, unit, family=spec.family, sample_size=len(working),
                       min_sample=min_length, distribution=distribution,
                       warning=_join(sequence_obj.warning, reason))

    raw_value = values_by_name.get(raw_name)
    if extractor == "catch22":
        overlap_suffix = _CATCH22_TIMESERIES_STABLE_SUFFIX.get(raw_name)
        if overlap_suffix:
            prefix = "drift.timeseries_" if spec.family == "book_drift" else "rhythm.timeseries_"
            distribution["overlaps_existing_metric_id"] = f"{prefix}{sequence_name}_catch22_{overlap_suffix}"
            distribution["overlap_note"] = (
                "timeseries_suite can compute this exact pycatch22 number under its own hand-"
                "picked id when its 'catch22' feature-group alias is explicitly selected (never "
                "by default); kept here too, per this suite's own bulk/per-catalogue-feature "
                "design, rather than silently duplicated")
        warning = None if raw_value is not None else (
            "pycatch22 returned a non-finite value for this sequence, most often because it "
            "has zero variance")
        return finding(metric_id, name, raw_value, unit, family=spec.family,
                       sample_size=len(working), min_sample=min_length, distribution=distribution,
                       warning=_join(sequence_obj.warning, warning), sample_size_sensitive=True)

    # tsfresh / tsfresh_comprehensive
    if extractor == "tsfresh" and raw_name in TSFRESH_MINIMAL_RAW_NAMES:
        prefix = "drift.timeseries_" if spec.family == "book_drift" else "rhythm.timeseries_"
        distribution["overlaps_existing_metric_id"] = f"{prefix}{sequence_name}_tsfresh"
        distribution["overlap_note"] = (
            "this exact number is already reachable inside timeseries_suite's own folded "
            "tsfresh finding for this sequence (its distribution['values'][raw_feature_name]) "
            "when tsfresh_feature_set='minimal' there; reported here as its own id instead, "
            "per this suite's per-catalogue-feature design")
    warning = None if raw_value is not None else (
        "the extractor returned a non-finite value for this sequence (division by zero, a "
        "log of zero, or too little variance are the usual causes)")
    return finding(metric_id, name, raw_value, unit, family=spec.family, sample_size=len(working),
                   min_sample=min_length, distribution=distribution,
                   warning=_join(sequence_obj.warning, warning), sample_size_sensitive=True)


def _join(existing: str | None, addition: str | None) -> str | None:
    if not addition:
        return existing
    return f"{existing}; {addition}" if existing else addition


__all__ = ["measure", "sanitize_feature_name", "dedupe_sanitized_ids", "STATS_FEATURES",
          "CATCH22_RAW_NAMES", "TSFRESH_MINIMAL_RAW_NAMES", "MODES", "DEFAULT_MODE",
          "DEFAULT_SEQUENCES", "EXTRACTOR_NAMES", "AUTOMATIC_FEATURES_VERSION"]
