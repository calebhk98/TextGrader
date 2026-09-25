"""Synthetic-sequence validation for recurrence quantification and nonlinear dynamics.

Mirrors ``tests/test_timeseries_suite.py``/``tests/test_signal_processing_suite.py``'s
approach: feature functions are exercised directly against numeric sequences
with a known or clearly directional answer (a constant series, a periodic
series, the logistic map at r=4 as a chaotic series, Gaussian noise, a
shuffled version of a periodic series, a too-short series, and a very long
series that must trigger the matrix-size cap) rather than only through a
novel-sized fixture.
"""

from __future__ import annotations

import math
import random
import time
import tracemalloc

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, nonlinear_dynamics_suite as nd
from textgrader import sequences as seq_module

SETTINGS = nd._settings({})


def _ctx(sequence_name, working, cfg=None, family="sentence_rhythm", analysis=None):
    """A minimal ``ctx`` for calling an ``_EXTENDED_FEATURES``/library function directly.

    RQA/capped-series results are cached via ``analysis.memo``, keyed on
    ``sequence_name`` + settings -- not on the document's actual text -- so a
    throwaway analysis and a synthetic ``working`` array exercise the real
    caching path exactly like a real sequence would (see the identical helper
    in ``test_signal_processing_suite.py``).
    """

    analysis = analysis if analysis is not None else DocumentAnalysis.from_text("Cache scope.")
    cfg = cfg if cfg is not None else SETTINGS
    return {"analysis": analysis, "sequence_name": sequence_name, "cfg": cfg, "working": working,
           "sequence_family": family, "capped": nd._capped_series(analysis, sequence_name, working, cfg)}


def _skip_without(*names):
    for name in names:
        if name == "nolds":
            # nolds needs textgrader.optional.shim_nolds_resources applied
            # BEFORE its first import anywhere in the process, or the real
            # importlib.resources bug it works around (see that shim's
            # docstring) poisons textgrader.optional's module-cache with a
            # permanent "unavailable" for the rest of the test session --
            # every real call site in this suite goes through
            # nd._require_nolds(), which shims first; this check must too.
            available = nd._require_nolds()[0] is not None
        else:
            available = optional.have(name)
        if not available:
            pytest.skip(f"{name} not installed in this environment")


# ------------------------------------------------------------ synthetic series

def _constant(n=200, value=5.0):
    return [value] * n


def _periodic(n=200, pattern=(1.0, 2.0, 3.0, 4.0, 3.0, 2.0)):
    pattern = list(pattern)
    return (pattern * (n // len(pattern) + 1))[:n]


def _logistic_map(n=200, x0=0.4, r=4.0):
    """The logistic map at r=4: the canonical fully-chaotic one-dimensional map."""

    values = []
    x = x0
    for _ in range(n):
        x = r * x * (1.0 - x)
        values.append(x)
    return values


def _noise(n=200, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


# ------------------------------------------------------------------- RQA core

def test_rqa_recurrence_rate_is_100_percent_for_a_constant_sequence():
    _skip_without("numpy")
    values = _constant(200)
    out = nd._feature_rqa_recurrence_rate(values, SETTINGS, _ctx("toy", values))
    assert out.value == pytest.approx(100.0)


def test_rqa_determinism_and_laminarity_are_maximal_for_a_constant_sequence():
    _skip_without("numpy")
    values = _constant(200)
    ctx = _ctx("toy", values)
    det = nd._feature_rqa_determinism(values, SETTINGS, ctx)
    lam = nd._feature_rqa_laminarity(values, SETTINGS, ctx)
    assert det.value == pytest.approx(100.0, abs=0.1)
    assert lam.value == pytest.approx(100.0, abs=0.1)


def test_rqa_trend_is_exactly_zero_for_a_constant_sequence():
    _skip_without("numpy")
    values = _constant(200)
    out = nd._feature_rqa_trend(values, SETTINGS, _ctx("toy", values))
    assert out.value == pytest.approx(0.0, abs=1e-9)


def test_rqa_determinism_orders_periodic_above_chaotic_above_noise():
    """The core separation this whole family exists to make.

    A periodic series revisits the same trajectory segment exactly, over and
    over: determinism should be very high. The logistic map at r=4 is fully
    chaotic but still deterministic (short segments recur approximately, by
    the map's own short-term predictability); Gaussian noise has no
    trajectory to revisit at all. The classical, well-established ordering is
    DET(periodic) > DET(chaotic) > DET(noise), not merely "different from
    each other".
    """

    _skip_without("numpy")
    periodic = _periodic(200)
    chaotic = _logistic_map(200)
    noise = _noise(200)
    det_periodic = nd._feature_rqa_determinism(periodic, SETTINGS, _ctx("periodic", periodic)).value
    det_chaotic = nd._feature_rqa_determinism(chaotic, SETTINGS, _ctx("chaotic", chaotic)).value
    det_noise = nd._feature_rqa_determinism(noise, SETTINGS, _ctx("noise", noise)).value
    assert det_periodic > det_chaotic > det_noise
    assert det_periodic > 95.0
    assert det_noise < 60.0


def test_rqa_longest_diagonal_line_orders_periodic_above_chaotic_above_noise():
    _skip_without("numpy")
    periodic = _periodic(200)
    chaotic = _logistic_map(200)
    noise = _noise(200)
    long_periodic = nd._feature_rqa_longest_diagonal_line(periodic, SETTINGS, _ctx("periodic", periodic)).value
    long_chaotic = nd._feature_rqa_longest_diagonal_line(chaotic, SETTINGS, _ctx("chaotic", chaotic)).value
    long_noise = nd._feature_rqa_longest_diagonal_line(noise, SETTINGS, _ctx("noise", noise)).value
    assert long_periodic > long_chaotic > long_noise


def test_rqa_determinism_drops_sharply_when_a_periodic_sequence_is_shuffled():
    """Shuffling preserves the value distribution but destroys temporal recurrence structure."""

    _skip_without("numpy")
    values = _periodic(200)
    shuffled = list(values)
    random.Random(7).shuffle(shuffled)
    assert sorted(values) == pytest.approx(sorted(shuffled))  # same values, different order

    det_original = nd._feature_rqa_determinism(values, SETTINGS, _ctx("periodic", values)).value
    det_shuffled = nd._feature_rqa_determinism(
        shuffled, SETTINGS, _ctx("periodic_shuffled", shuffled)).value
    assert det_original > 95.0
    assert det_shuffled < det_original - 30.0


def test_rqa_features_report_insufficient_data_for_a_too_short_sequence():
    _skip_without("numpy")
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    analysis = DocumentAnalysis.from_text("One sentence. Another. A third. A fourth. A fifth.")
    for feature_name in nd._RQA_FEATURE_ORDER:
        findings = nd.measure(analysis, config={
            "sequences": ["sentence_words"], "feature_groups": [feature_name]})
        item = findings[0]
        assert item["value"] is None, feature_name
        assert "insufficient data" in (item["warning"] or ""), feature_name


def test_rqa_matrix_size_and_runtime_and_memory_are_capped_for_a_very_long_sequence():
    """Proves the matrix-size cap holds, both runtime and memory, per the task's own test list."""

    _skip_without("numpy")
    rng = random.Random(0)
    huge = [rng.gauss(0.0, 1.0) for _ in range(250_000)]  # syllable_stress-scale
    analysis = DocumentAnalysis.from_text("cap test")
    ctx = _ctx("huge", huge, analysis=analysis)

    tracemalloc.start()
    started = time.time()
    out = nd._feature_rqa_recurrence_rate(huge, SETTINGS, ctx)
    elapsed = time.time() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert out.value is not None
    assert out.distribution["matrix_size"] <= SETTINGS["max_series_points"]
    assert out.distribution["sampling_strategy"] == "deterministic_seeded_contiguous_window"
    assert out.distribution["original_length"] == 250_000
    # Generous ceilings: a 1,500-point capped matrix is a few million cells,
    # not the 250,000^2 an uncapped one would have been.
    assert elapsed < 10.0
    assert peak < 300_000_000  # 300 MB


def test_rqa_sampling_is_deterministic_for_a_fixed_seed():
    _skip_without("numpy")
    rng = random.Random(1)
    huge = [rng.gauss(0.0, 1.0) for _ in range(10_000)]
    analysis_a = DocumentAnalysis.from_text("a")
    analysis_b = DocumentAnalysis.from_text("b")
    out_a = nd._feature_rqa_recurrence_rate(huge, SETTINGS, _ctx("huge", huge, analysis=analysis_a))
    out_b = nd._feature_rqa_recurrence_rate(huge, SETTINGS, _ctx("huge", huge, analysis=analysis_b))
    assert out_a.value == out_b.value
    assert out_a.distribution["sampling_start_index"] == out_b.distribution["sampling_start_index"]


def test_rqa_threshold_and_strategy_are_reported():
    _skip_without("numpy")
    values = _logistic_map(200)
    out = nd._feature_rqa_recurrence_rate(values, SETTINGS, _ctx("toy", values))
    assert out.distribution["threshold_mode"] == "target_rr"
    assert isinstance(out.distribution["threshold"], float)
    assert "achieved_recurrence_rate_percent" in out.distribution


def test_rqa_std_fraction_threshold_mode_is_also_reported():
    _skip_without("numpy")
    values = _logistic_map(200)
    cfg = {**SETTINGS, "threshold_mode": "std_fraction"}
    out = nd._feature_rqa_recurrence_rate(values, cfg, _ctx("toy", values, cfg=cfg))
    assert out.distribution["threshold_mode"] == "std_fraction"


# ------------------------------------------------- rqa_recurrence_rate / rqa_threshold

def test_recurrence_rate_is_pinned_and_flagged_under_target_rr():
    """Under the default threshold_mode, recurrence_rate is chosen, not measured."""

    _skip_without("numpy")
    chaotic = _logistic_map(300)
    noise = _noise(300)
    out_chaotic = nd._feature_rqa_recurrence_rate(chaotic, SETTINGS, _ctx("chaotic", chaotic))
    out_noise = nd._feature_rqa_recurrence_rate(noise, SETTINGS, _ctx("noise", noise))
    # Two structurally very different (but both continuous-valued, so the
    # percentile threshold search has no exact ties to contend with) series
    # still land within a fraction of a point of the same
    # target_recurrence_rate (5%) -- pinned by construction, not a real
    # difference between them.
    assert out_chaotic.value == pytest.approx(out_noise.value, abs=0.5)
    assert out_chaotic.value == pytest.approx(5.0, abs=0.5)
    for out in (out_chaotic, out_noise):
        assert out.distribution["set_by_threshold_mode"] is True
        assert out.warning and "rqa_threshold" in out.warning


def test_recurrence_rate_can_exceed_the_target_when_the_series_has_many_exact_ties():
    """An honest, documented edge case, not a bug: a series with many repeated exact
    values (real sentence-length/punctuation-count sequences included) can land well
    above target_recurrence_rate, because the percentile search has ties to spend."""

    _skip_without("numpy")
    periodic = _periodic(300)  # a short exact pattern repeated -- many tied distances
    out = nd._feature_rqa_recurrence_rate(periodic, SETTINGS, _ctx("periodic", periodic))
    assert out.distribution["set_by_threshold_mode"] is True
    # Still "pinned by construction" in the sense that it is not a measurement
    # of the sequence's dynamics, but the achieved rate is not tightly bound
    # to the target the way it is for a tie-free continuous series.
    assert out.value > 10.0


def test_recurrence_rate_is_a_real_measurement_under_std_fraction():
    _skip_without("numpy")
    cfg = {**SETTINGS, "threshold_mode": "std_fraction"}
    periodic = _periodic(300)
    noise = _noise(300)
    out_periodic = nd._feature_rqa_recurrence_rate(periodic, cfg, _ctx("periodic", periodic, cfg=cfg))
    out_noise = nd._feature_rqa_recurrence_rate(noise, cfg, _ctx("noise", noise, cfg=cfg))
    assert "set_by_threshold_mode" not in out_periodic.distribution
    assert out_periodic.warning is None
    # A fixed radius (as a share of each series' own SD) produces genuinely
    # different densities on structurally different series.
    assert out_periodic.value != pytest.approx(out_noise.value, abs=0.5)


def test_rqa_threshold_is_reported_as_its_own_finding_in_std_units():
    _skip_without("numpy")
    values = _logistic_map(200)
    out = nd._feature_rqa_threshold(values, SETTINGS, _ctx("toy", values))
    assert out.value is not None
    assert isinstance(out.distribution["raw_threshold"], float)
    assert isinstance(out.distribution["std_of_sampled_values"], float)
    assert out.value == pytest.approx(
        out.distribution["raw_threshold"] / out.distribution["std_of_sampled_values"])


def test_rqa_threshold_adapts_to_series_spread_while_recurrence_rate_does_not():
    """The reviewer's own required check: a more spread-out/noisier series needs a
    different (raw) threshold to reach the same target recurrence rate, while
    rqa_recurrence_rate itself stays pinned near the target regardless."""

    _skip_without("numpy")
    narrow = _noise(300, seed=1)          # SD ~= 1
    wide = [v * 5.0 for v in narrow]      # identical shape, SD ~= 5

    threshold_narrow = nd._feature_rqa_threshold(narrow, SETTINGS, _ctx("narrow", narrow))
    threshold_wide = nd._feature_rqa_threshold(wide, SETTINGS, _ctx("wide", wide))
    raw_narrow = threshold_narrow.distribution["raw_threshold"]
    raw_wide = threshold_wide.distribution["raw_threshold"]
    # The raw (unnormalized) radius needed to hit the same target recurrence
    # rate scales with the series' own spread -- a materially larger radius
    # for the more spread-out series.
    assert raw_wide > raw_narrow * 3.0
    # Standardized by each series' own SD, the two are comparable again (a
    # pure rescale of the same underlying process needs the same *relative*
    # radius) -- this is what makes rqa_threshold, not the raw distance,
    # the number worth comparing across books of different absolute scale.
    assert threshold_narrow.value == pytest.approx(threshold_wide.value, rel=0.05)

    rate_narrow = nd._feature_rqa_recurrence_rate(narrow, SETTINGS, _ctx("narrow2", narrow)).value
    rate_wide = nd._feature_rqa_recurrence_rate(wide, SETTINGS, _ctx("wide2", wide)).value
    assert rate_narrow == pytest.approx(rate_wide, abs=1.0)


def test_rqa_threshold_differs_for_noisier_series_of_the_same_scale():
    """Same standard deviation, different internal structure (more purely noisy
    vs. a periodic-plus-noise mix) -- the standardized threshold is sensitive
    to real structure, not only to raw scale, unlike rqa_recurrence_rate."""

    _skip_without("numpy")
    pure_noise = _noise(300, seed=2)
    periodic_component = _periodic(300, pattern=(1.0, -1.0))
    rng = random.Random(2)
    mixed = [p + rng.gauss(0.0, 0.1) for p in periodic_component]  # mostly periodic, little noise

    threshold_noise = nd._feature_rqa_threshold(pure_noise, SETTINGS, _ctx("noise", pure_noise))
    threshold_mixed = nd._feature_rqa_threshold(mixed, SETTINGS, _ctx("mixed", mixed))
    assert threshold_noise.value != pytest.approx(threshold_mixed.value, rel=0.1)


def test_rqa_diagonal_entropy_orders_periodic_above_chaotic_above_noise():
    """Diagonal-line-length entropy measures diversity of exact line lengths, not "disorder".

    A periodic sequence recurs at nearly every multiple of its period, and
    because a diagonal at offset k has length M-k, each of those recurring
    offsets contributes a *different* line length -- many distinct lengths,
    high entropy of the length distribution. Chaotic recurrence is shorter
    and more homogeneous (fewer distinct lengths); white noise's "lines" are
    almost all isolated points (length 1), the least diverse of all.
    Verified empirically before writing this assertion (see the module's own
    manual exploration): periodic ~=5.6 bits over ~49 distinct lengths,
    chaotic ~=2.1 bits over ~10, noise ~=0.9 bits over ~4.
    """

    _skip_without("numpy")
    periodic = _periodic(300)
    chaotic = _logistic_map(300)
    noise = _noise(300)
    entropy_periodic = nd._feature_rqa_diagonal_entropy(periodic, SETTINGS, _ctx("periodic", periodic)).value
    entropy_chaotic = nd._feature_rqa_diagonal_entropy(chaotic, SETTINGS, _ctx("chaotic", chaotic)).value
    entropy_noise = nd._feature_rqa_diagonal_entropy(noise, SETTINGS, _ctx("noise", noise)).value
    assert entropy_periodic > entropy_chaotic > entropy_noise


def test_rqa_features_share_one_recurrence_matrix_computation(monkeypatch):
    """All ten rqa_* features must pay for one embedding + distance matrix, not ten.

    ``numpy.percentile`` is called exactly once inside ``_rqa_result``'s
    ``build()`` closure (to pick the target-recurrence-rate threshold), so
    counting its calls while every rqa_* feature runs against the same
    analysis/sequence/settings directly measures whether the underlying O(n^2)
    computation is actually being shared, not merely whether ``analysis.memo``
    was asked the same question ten times.
    """

    _skip_without("numpy")
    import numpy

    calls = {"n": 0}
    original_percentile = numpy.percentile

    def counting_percentile(*args, **kwargs):
        calls["n"] += 1
        return original_percentile(*args, **kwargs)

    monkeypatch.setattr(numpy, "percentile", counting_percentile)
    analysis = DocumentAnalysis.from_text("share test")
    values = _logistic_map(200)
    shared_ctx = _ctx("toy", values, analysis=analysis)
    for feature_name in nd._RQA_FEATURE_ORDER:
        nd._EXTENDED_FEATURES[feature_name](values, SETTINGS, shared_ctx)
    assert calls["n"] == 1


# ------------------------------------------------------------- library estimators

def test_hurst_nolds_names_the_overlapping_timeseries_id():
    _skip_without("nolds")
    values = _logistic_map(200)
    out = nd._feature_hurst_nolds(values, SETTINGS, _ctx("sentence_words", values))
    assert out.distribution["overlaps_existing_metric_id"] == \
        "rhythm.timeseries_sentence_words_hurst"
    assert out.value is not None


def test_dfa_nolds_names_the_overlapping_timeseries_id():
    _skip_without("nolds")
    values = _logistic_map(200)
    out = nd._feature_dfa_nolds(values, SETTINGS, _ctx("sentence_words", values))
    assert out.distribution["overlaps_existing_metric_id"] == \
        "rhythm.timeseries_sentence_words_dfa"
    assert out.value is not None


def test_nolds_calls_are_bit_identical_across_repeated_calls():
    """nolds' own default fit='RANSAC' is randomized; this codebase forces fit='poly'."""

    _skip_without("nolds")
    values = _noise(300, seed=9)
    first = nd._feature_hurst_nolds(values, SETTINGS, _ctx("toy", values)).value
    second = nd._feature_hurst_nolds(values, SETTINGS, _ctx("toy2", values)).value
    third = nd._feature_dfa_nolds(values, SETTINGS, _ctx("toy3", values)).value
    fourth = nd._feature_dfa_nolds(values, SETTINGS, _ctx("toy4", values)).value
    assert first == second
    assert third == fourth


def test_lyapunov_nolds_reports_insufficient_data_below_its_own_floor():
    _skip_without("nolds")
    values = _logistic_map(50)
    out = nd._feature_lyapunov_nolds(values, SETTINGS, _ctx("toy", values))
    assert out.value is None
    assert "insufficient data" in out.warning


def test_lyapunov_nolds_runs_above_its_floor():
    _skip_without("nolds")
    values = _logistic_map(SETTINGS["lyapunov_min_length"] + 50)
    out = nd._feature_lyapunov_nolds(values, SETTINGS, _ctx("toy", values))
    assert out.value is not None


def test_correlation_dimension_nolds_returns_a_number():
    _skip_without("nolds")
    values = _logistic_map(300)
    out = nd._feature_correlation_dimension_nolds(values, SETTINGS, _ctx("toy", values))
    assert out.value is not None


def test_higuchi_fd_antropy_is_higher_for_noise_than_a_chaotic_map():
    # A constant or exactly periodic sequence can drive one of Higuchi's
    # per-k curve lengths to exactly zero (log(0) -> a non-finite fit slope,
    # an antropy numerical edge case handled as 'unavailable' rather than
    # NaN -- see test_higuchi_fd_antropy_reports_unavailable_for_a_degenerate_sequence).
    _skip_without("antropy", "numpy")
    noise = _noise(200)
    chaotic = _logistic_map(200)
    higuchi_noise = nd._feature_higuchi_fd_antropy(noise, SETTINGS, _ctx("noise", noise)).value
    higuchi_chaotic = nd._feature_higuchi_fd_antropy(chaotic, SETTINGS, _ctx("chaotic", chaotic)).value
    assert higuchi_noise > higuchi_chaotic


def test_higuchi_fd_antropy_reports_unavailable_for_a_degenerate_sequence():
    _skip_without("antropy", "numpy")
    const = _constant(200)
    out = nd._feature_higuchi_fd_antropy(const, SETTINGS, _ctx("const", const))
    assert out.value is None
    assert out.warning


def test_petrosian_fd_antropy_separates_noise_from_a_constant_sequence():
    _skip_without("antropy", "numpy")
    noise = _noise(200)
    const = _constant(200)
    petrosian_noise = nd._feature_petrosian_fd_antropy(noise, SETTINGS, _ctx("noise", noise)).value
    petrosian_const = nd._feature_petrosian_fd_antropy(const, SETTINGS, _ctx("const", const)).value
    assert petrosian_noise > petrosian_const


def test_sample_and_approximate_entropy_antropy_are_lower_for_periodic_than_noise():
    _skip_without("antropy", "numpy")
    periodic = _periodic(200)
    noise = _noise(200)
    se_periodic = nd._feature_sample_entropy_antropy(periodic, SETTINGS, _ctx("periodic", periodic)).value
    se_noise = nd._feature_sample_entropy_antropy(noise, SETTINGS, _ctx("noise", noise)).value
    ae_periodic = nd._feature_approximate_entropy_antropy(periodic, SETTINGS, _ctx("periodic2", periodic)).value
    ae_noise = nd._feature_approximate_entropy_antropy(noise, SETTINGS, _ctx("noise2", noise)).value
    assert se_periodic < se_noise
    assert ae_periodic < ae_noise


def test_sample_entropy_antropy_names_the_overlapping_randomness_id():
    _skip_without("antropy", "numpy")
    values = _noise(200)
    out = nd._feature_sample_entropy_antropy(values, SETTINGS, _ctx("toy", values))
    assert out.distribution["overlaps_existing_metric_id"] == "style.randomness_sample_entropy"


def test_approximate_entropy_antropy_names_the_overlapping_randomness_id():
    _skip_without("antropy", "numpy")
    values = _noise(200)
    out = nd._feature_approximate_entropy_antropy(values, SETTINGS, _ctx("toy", values))
    assert out.distribution["overlaps_existing_metric_id"] == "style.randomness_approximate_entropy"


def test_lz_complexity_antropy_is_lower_for_periodic_than_noise():
    _skip_without("antropy", "numpy")
    periodic = _periodic(300)
    noise = _noise(300)
    lz_periodic = nd._feature_lz_complexity_antropy(periodic, SETTINGS, _ctx("periodic", periodic)).value
    lz_noise = nd._feature_lz_complexity_antropy(noise, SETTINGS, _ctx("noise", noise)).value
    assert lz_periodic < lz_noise
    assert lz_periodic is not None and lz_noise is not None


def test_permutation_entropy_ordpy_is_lower_for_periodic_than_noise():
    _skip_without("ordpy")
    periodic = _periodic(200)
    noise = _noise(200)
    pe_periodic = nd._feature_permutation_entropy_ordpy(periodic, SETTINGS, _ctx("periodic", periodic)).value
    pe_noise = nd._feature_permutation_entropy_ordpy(noise, SETTINGS, _ctx("noise", noise)).value
    assert pe_periodic < pe_noise


def test_permutation_entropy_ordpy_names_overlapping_ids():
    _skip_without("ordpy")
    values = _noise(200)
    out = nd._feature_permutation_entropy_ordpy(values, SETTINGS, _ctx("sentence_words", values))
    assert out.distribution["overlaps_existing_metric_id"] == \
        "rhythm.timeseries_sentence_words_permutation_entropy"
    assert out.distribution["related_existing_metric_ids"] == ["style.randomness_permutation_entropy"]


def test_ordinal_complexity_ordpy_is_near_zero_for_noise_but_not_for_structured_sequences():
    """The entropy-complexity plane's defining fact: pure randomness has near-zero complexity.

    Lopez-Ruiz/Mendes/Rosso statistical complexity is built specifically to
    separate "highly ordered" and "highly random" from "structured but not
    fully predictable", and it is white noise -- not a periodic sequence --
    that sits at the near-zero-complexity end regardless of its (maximal)
    entropy. Verified empirically (dx=3): periodic C~=0.235, chaotic
    (logistic map) C~=0.172, i.i.d. Gaussian noise C~=0.0016 -- noise's
    complexity is over two orders of magnitude below either structured
    sequence's, which is the one part of this ordering robust to exactly
    which embedding order is used.
    """

    _skip_without("ordpy")
    periodic = _periodic(300)
    noise = _noise(300)
    chaotic = _logistic_map(300)
    c_periodic = nd._feature_ordinal_complexity_ordpy(periodic, SETTINGS, _ctx("periodic", periodic)).value
    c_noise = nd._feature_ordinal_complexity_ordpy(noise, SETTINGS, _ctx("noise", noise)).value
    c_chaotic = nd._feature_ordinal_complexity_ordpy(chaotic, SETTINGS, _ctx("chaotic", chaotic)).value
    assert c_periodic > c_noise * 10
    assert c_chaotic > c_noise * 10


def test_fuzzy_and_dispersion_entropy_entropyhub_are_lower_for_periodic_than_noise():
    _skip_without("entropyhub", "numpy")
    periodic = _periodic(200)
    noise = _noise(200)
    fuzz_periodic = nd._feature_fuzzy_entropy_entropyhub(periodic, SETTINGS, _ctx("periodic", periodic)).value
    fuzz_noise = nd._feature_fuzzy_entropy_entropyhub(noise, SETTINGS, _ctx("noise", noise)).value
    disp_periodic = nd._feature_dispersion_entropy_entropyhub(periodic, SETTINGS, _ctx("periodic2", periodic)).value
    disp_noise = nd._feature_dispersion_entropy_entropyhub(noise, SETTINGS, _ctx("noise2", noise)).value
    assert fuzz_periodic < fuzz_noise
    assert disp_periodic < disp_noise


def test_entropyhub_calls_are_bit_identical_across_repeated_calls():
    _skip_without("entropyhub", "numpy")
    values = _noise(200, seed=3)
    first = nd._feature_fuzzy_entropy_entropyhub(values, SETTINGS, _ctx("a", values)).value
    second = nd._feature_fuzzy_entropy_entropyhub(values, SETTINGS, _ctx("b", values)).value
    assert first == second


def test_pyrqa_crosscheck_reports_unavailable_or_a_real_cross_check():
    """Either this container has no OpenCL platform (reports the exact error), or it does."""

    _skip_without("pyrqa")
    values = _logistic_map(200)
    out = nd._feature_pyrqa_crosscheck(values, SETTINGS, _ctx("sentence_words", values))
    if out.value is None:
        assert out.warning and "pyrqa" in out.warning.lower()
    else:
        assert out.distribution["overlaps_existing_metric_id"] == \
            "drift.nonlinear_sentence_words_rqa_recurrence_rate"


# ------------------------------------------------------------------ the suite

def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text)


def _long_text(paragraphs: int = 80, seed: int = 5) -> str:
    rng = random.Random(seed)
    vocab = ("the quiet, room held a long silence; while she considered what had "
            "happened and whether anyone would notice however perhaps not "
            "because nobody asked her directly about any of it").split()
    blocks = []
    for _ in range(paragraphs):
        sentences = [" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 20)))
                    .capitalize() + rng.choice([".", ".", ".", "?", "!"])
                    for _ in range(rng.randint(1, 4))]
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


def test_default_configuration_produces_a_bounded_defensible_finding_count():
    analysis = _analysis(_long_text())
    findings = nd.measure(analysis, config={})
    expected = len(nd.DEFAULT_SEQUENCES) * len(nd.DEFAULT_FEATURE_GROUPS)
    assert len(findings) == expected
    ids = {item["metric_id"] for item in findings}
    assert "drift.nonlinear_sentence_words_rqa_recurrence_rate" in ids


def test_default_configuration_only_selects_rqa_core_features():
    assert set(nd.DEFAULT_FEATURE_GROUPS) == set(nd._RQA_FEATURE_ORDER)
    assert not set(nd.DEFAULT_FEATURE_GROUPS) & set(nd._LIBRARY_FEATURE_ORDER)
    assert "pyrqa_crosscheck" not in nd.DEFAULT_FEATURE_GROUPS


def test_every_finding_uses_the_book_drift_family_and_prefix():
    analysis = _analysis(_long_text())
    findings = nd.measure(analysis, config={"feature_groups": ["rqa_recurrence_rate"]})
    for item in findings:
        if item["metric_id"] in ("drift.nonlinear_config", "drift.nonlinear_truncated"):
            continue
        assert item["metric_id"].startswith("drift.nonlinear_")
        assert item["family"] == "book_drift"


def test_single_sequence_and_feature_selection_produces_exactly_one_finding():
    analysis = _analysis(_long_text())
    findings = nd.measure(analysis, config={
        "sequences": ["sentence_words"], "feature_groups": ["rqa_recurrence_rate"]})
    assert len(findings) == 1
    assert findings[0]["metric_id"] == "drift.nonlinear_sentence_words_rqa_recurrence_rate"


def test_insufficient_data_is_reported_for_a_too_short_sequence():
    analysis = _analysis("One sentence. Another one. A third.")
    findings = nd.measure(analysis, config={
        "sequences": ["sentence_words"], "feature_groups": ["rqa_recurrence_rate"]})
    item = findings[0]
    assert item["value"] is None
    assert "insufficient data" in item["warning"]


def test_unknown_sequence_and_feature_are_reported_but_do_not_stop_valid_work():
    analysis = _analysis(_long_text())
    findings = nd.measure(analysis, config={
        "sequences": ["sentence_words", "not_a_real_sequence"],
        "feature_groups": ["rqa_recurrence_rate", "not_a_real_feature"]})
    ids = {item["metric_id"] for item in findings}
    assert "drift.nonlinear_sentence_words_rqa_recurrence_rate" in ids
    config_notes = [item for item in findings if item["metric_id"] == "drift.nonlinear_config"]
    assert config_notes and "not_a_real_sequence" in config_notes[0]["warning"]


def test_nothing_selected_reports_one_configuration_finding():
    analysis = _analysis(_long_text())
    findings = nd.measure(analysis, config={"sequences": [], "feature_groups": []})
    assert len(findings) == 1
    assert findings[0]["metric_id"] == "drift.nonlinear_config"
    assert findings[0]["value"] is None


def test_max_findings_truncates_and_reports_it():
    analysis = _analysis(_long_text())
    findings = nd.measure(analysis, config={
        "sequences": list(seq_module.SEQUENCES), "feature_groups": list(nd._ALL_FEATURE_ORDER),
        "max_findings": 5})
    assert len([f for f in findings if f["metric_id"] != "drift.nonlinear_truncated"]) == 5
    assert any(item["metric_id"] == "drift.nonlinear_truncated" for item in findings)


def test_repeated_runs_are_bit_identical():
    """Corpus profiles must rebuild byte for byte -- see tests/test_corpus_profile.py."""

    analysis1 = _analysis(_long_text())
    analysis2 = _analysis(_long_text())
    config = {"feature_groups": list(nd._RQA_FEATURE_ORDER)}
    first = {item["metric_id"]: item["value"] for item in nd.measure(analysis1, config=config)}
    second = {item["metric_id"]: item["value"] for item in nd.measure(analysis2, config=config)}
    assert first == second


# ------------------------------------------------------------------ registration

def test_registered_in_the_metric_registry_off_by_default():
    assert "nonlinear_dynamics_suite" in REGISTRY
    spec = REGISTRY["nonlinear_dynamics_suite"]
    assert spec.cost == "moderate"
    assert spec.family == "book_drift"
    assert not spec.needs_parse
    assert not spec.needs_model


def test_off_by_default_in_a_grade_run(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(_long_text(), encoding="utf-8")
    report = grade.analyze(source, base_config)
    ids = {item.metric_id for item in report.results}
    assert not any(metric_id.startswith("drift.nonlinear_") for metric_id in ids)


def test_every_registered_metric_runs_without_raising(manuscript, base_config):
    from textgrader.results import StatusType
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "nonlinear_dynamics_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
    ids = {item.metric_id for item in report.results}
    assert any(metric_id.startswith("drift.nonlinear_") for metric_id in ids)


def test_every_feature_group_runs_without_raising_on_a_real_manuscript(manuscript, base_config):
    from textgrader.results import StatusType
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "nonlinear_dynamics_suite": {
                                             "enabled": True,
                                             "feature_groups": list(nd._ALL_FEATURE_ORDER)}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_degenerate_documents_never_raise(text, tmp_path, base_config):
    from textgrader.results import StatusType
    source = tmp_path / "tiny.txt"
    source.write_text(text, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "nonlinear_dynamics_suite": {
                                             "enabled": True,
                                             "feature_groups": list(nd._ALL_FEATURE_ORDER)}}}
    report = grade.analyze(source, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


def test_degrades_fully_with_every_optional_package_disabled(monkeypatch, manuscript, base_config):
    from textgrader.results import StatusType
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        config = {**base_config, "metrics": {**base_config["metrics"],
                                             "nonlinear_dynamics_suite": {
                                                 "enabled": True,
                                                 "feature_groups": list(nd._ALL_FEATURE_ORDER)}}}
        report = grade.analyze(manuscript, config)
        errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
        assert not errors, [(item.metric_id, item.error) for item in errors]
        unavailable = [item for item in report.results if item.metric_id.startswith("drift.nonlinear_")]
        assert unavailable  # RQA core needs only numpy, which "all" also disables
        assert all(item.value is None for item in unavailable
                  if item.metric_id not in ("drift.nonlinear_config", "drift.nonlinear_truncated"))
    finally:
        optional.reset_cache()


def test_nolds_import_works_via_the_shim():
    """Guards against the real nolds/importlib.resources bug regressing silently."""

    _skip_without("nolds")
    module, reason = nd._require_nolds()
    assert module is not None, reason


# ------------------------------------------------------ shim_nolds_resources scope
#
# The coordinator's own review flagged this by name: a permanent, module-wide
# monkeypatch of a stdlib function (or, in an earlier pass of this project,
# torch.nn.Module.load_state_dict for BookNLP) leaks into every later,
# unrelated caller unless it is scoped as a context manager that restores the
# original on every exit. shim_nolds_resources is that shape; these tests
# prove the restoration actually happens, both after a call that succeeds and
# after the context manager exits at all, and that an unrelated package's own
# resource lookups are never altered by the patch being briefly active.

def test_nolds_shim_restores_importlib_resources_files_afterward():
    import importlib.resources as resources

    original = resources.files
    nd._require_nolds()  # imports (or, on a later call, no-ops) inside the shim
    assert resources.files is original


def test_nolds_shim_restores_even_when_the_body_raises():
    import importlib.resources as resources

    original = resources.files

    class _Boom(Exception):
        pass

    with pytest.raises(_Boom):
        with optional.shim_nolds_resources():
            raise _Boom("something else failed while the shim was active")
    assert resources.files is original


def test_nolds_shim_does_not_alter_behavior_for_an_unrelated_package():
    """A real, unrelated package's own resources.files() call must be identical
    whether or not the nolds shim happened to run around it."""

    import importlib.resources as resources

    before = resources.files("json")
    before_listing = sorted(p.name for p in before.iterdir())

    with optional.shim_nolds_resources():
        during = resources.files("json")
        during_listing = sorted(p.name for p in during.iterdir())

    after = resources.files("json")
    after_listing = sorted(p.name for p in after.iterdir())

    assert before_listing == during_listing == after_listing
    assert resources.files is not None
    # Not the patched wrapper, in either its before or after state.
    assert resources.files.__name__ != "_tolerant_files"


def test_nolds_shim_never_touches_a_non_typeerror():
    """A completely different exception type must never be caught by this shim at all."""

    with optional.shim_nolds_resources():
        import importlib.resources as resources
        with pytest.raises(AttributeError):
            resources.files(object())  # not a string or module -- AttributeError, not TypeError


def test_nolds_shim_reraises_a_typeerror_that_is_not_its_own(monkeypatch):
    """Only the exact "is not a package" resolution failure is swallowed -- everything
    else (including a different-message TypeError) propagates unchanged."""

    import importlib.resources as resources

    def _fake_files(anchor=None):
        raise TypeError("a completely unrelated failure")

    monkeypatch.setattr(resources, "files", _fake_files)
    with optional.shim_nolds_resources():
        with pytest.raises(TypeError, match="a completely unrelated failure"):
            resources.files("whatever")


def test_nolds_shim_falls_back_only_for_the_exact_not_a_package_message(monkeypatch, tmp_path):
    """Unit-tests the fallback logic itself against nolds' real failure shape: a
    module (not a package) whose own __file__ gives the directory to fall back to."""

    import importlib.resources as resources
    import sys
    import types

    fake_file = tmp_path / "fake_pkg_module.py"
    fake_file.write_text("", encoding="utf-8")
    fake_module = types.ModuleType("fake_pkg_module")
    fake_module.__file__ = str(fake_file)
    monkeypatch.setitem(sys.modules, "fake_pkg_module", fake_module)

    def _fake_files(anchor=None):
        raise TypeError(f"{anchor!r} is not a package")

    monkeypatch.setattr(resources, "files", _fake_files)
    with optional.shim_nolds_resources():
        result = resources.files("fake_pkg_module")
    assert str(result) == str(tmp_path)


def test_nolds_shim_is_a_context_manager_not_a_one_way_patch():
    import inspect
    assert inspect.isgeneratorfunction(optional.shim_nolds_resources.__wrapped__)
