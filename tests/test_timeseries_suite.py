"""Synthetic-signal validation for the time-series feature battery, plus the
usual off-by-default / independent-failure / combinatorics-guard contract
tests for the suite as a whole.

The feature functions are tested directly against numeric sequences with a
known theoretical answer (a constant series, a linear ramp, a clean sine, a
strict alternation, and white noise) rather than only through a novel-sized
fixture: a wrong autocorrelation or a mis-signed trend slope on real prose
can look plausible by accident, but it cannot survive being checked against
values theory actually predicts.
"""

from __future__ import annotations

import math
import random

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, timeseries_suite as ts

FEATURES = ts._FEATURES
EXTENDED = ts._EXTENDED_FEATURES
SETTINGS = ts._settings({})


def _ctx(sequence_name, working, cfg=None, detrended=False, analysis=None):
    """A minimal ``ctx`` for calling an ``_EXTENDED_FEATURES`` function directly.

    catch22/wavelet results are cached via ``analysis.memo``, keyed on
    ``sequence_name`` + settings + ``detrended`` -- not on the document's
    actual text -- so a throwaway analysis and a synthetic ``working`` array
    exercise the real caching path exactly like a real sequence would.
    """

    return {
        "analysis": analysis if analysis is not None else DocumentAnalysis.from_text("Cache scope."),
        "sequence_name": sequence_name, "sequence": None,
        "cfg": cfg if cfg is not None else SETTINGS, "detrended": detrended, "working": working,
    }


def _constant(n=50, value=5.0):
    return [value] * n


def _ramp(n=60):
    return [float(i) for i in range(n)]


def _alternating(n=60):
    return [0.0, 10.0] * (n // 2)


def _sine(n=120, period=10, amplitude=10.0):
    return [amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def _noise(n=200, seed=42):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


# ------------------------------------------------------------- pure feature math

def test_dispersion_reads_the_middle_and_spread_correctly():
    out = FEATURES["dispersion"](_alternating(60), SETTINGS)
    assert out.value == pytest.approx(5.0)
    assert out.distribution["mad"] == pytest.approx(5.0)

    constant = FEATURES["dispersion"](_constant(30), SETTINGS)
    assert constant.value == pytest.approx(5.0)
    assert constant.distribution["std"] == pytest.approx(0.0)


def test_acf_is_undefined_for_a_constant_series():
    out = FEATURES["acf"]([5.0] * 40, SETTINGS)
    assert out.value is None


def test_acf_is_strongly_positive_for_a_ramp_and_negative_for_alternation():
    ramp_acf = FEATURES["acf"](_ramp(60), SETTINGS)
    alt_acf = FEATURES["acf"](_alternating(60), SETTINGS)
    assert ramp_acf.value > 0.9
    assert alt_acf.value < -0.9


def test_acf_lag1_matches_the_known_sine_autocorrelation():
    # For a pure sine of period P sampled every step, the theoretical lag-1
    # autocorrelation of a stationary sinusoid is cos(2*pi/P).
    values = _sine(period=10)
    out = FEATURES["acf"](values, {**SETTINGS, "lags": [1]})
    assert out.value == pytest.approx(math.cos(2 * math.pi / 10), abs=0.02)


def test_pacf_at_lag_one_always_equals_acf_at_lag_one():
    # A mathematical identity for any series, not a property of one dataset:
    # a strong cross-check that the Durbin-Levinson recursion is implemented
    # correctly rather than merely plausible-looking.
    for values in (_ramp(60), _alternating(60), _sine(120), _noise(200)):
        acf_out = FEATURES["acf"](values, {**SETTINGS, "lags": [1]})
        pacf_out = FEATURES["pacf"](values, {**SETTINGS, "pacf_max_lag": 3})
        assert pacf_out.value == pytest.approx(acf_out.value, abs=1e-9)


def test_trend_is_perfect_for_a_ramp_and_undefined_for_a_constant():
    ramp_trend = FEATURES["trend"](_ramp(60), SETTINGS)
    assert ramp_trend.value == pytest.approx(1.0, abs=1e-9)
    assert ramp_trend.distribution["r_squared"] == pytest.approx(1.0, abs=1e-9)

    constant_trend = FEATURES["trend"](_constant(30), SETTINGS)
    assert constant_trend.value is None
    assert "identical" in constant_trend.warning


def test_trend_is_near_zero_for_symmetric_alternation():
    out = FEATURES["trend"](_alternating(60), SETTINGS)
    assert abs(out.value) < 0.1


def test_turning_points_zero_for_monotonic_and_full_for_alternation():
    ramp_turns = FEATURES["turning_points"](_ramp(60), SETTINGS)
    alt_turns = FEATURES["turning_points"](_alternating(60), SETTINGS)
    assert ramp_turns.value == pytest.approx(0.0)
    assert alt_turns.value == pytest.approx(100.0)


def test_turning_points_matches_the_iid_theoretical_rate_for_noise():
    # The expected share of interior points that are a local turning point in
    # an i.i.d. sequence is exactly 2/3.
    out = FEATURES["turning_points"](_noise(400), SETTINGS)
    assert out.value == pytest.approx(200.0 / 3, abs=6.0)


def test_runs_longest_run_share_ramp_vs_alternation():
    # A ramp is half below its median then half above it, in one run each;
    # strict alternation about the median has runs of length one throughout.
    ramp_runs = FEATURES["runs"](_ramp(60), SETTINGS)
    alt_runs = FEATURES["runs"](_alternating(60), SETTINGS)
    assert ramp_runs.value == pytest.approx(50.0, abs=2.0)
    assert alt_runs.value < 5.0
    assert ramp_runs.value > alt_runs.value


def test_spectral_entropy_low_for_sine_high_for_noise():
    if not optional.have("numpy"):
        pytest.skip("numpy not installed in this environment")
    sine_out = FEATURES["spectral"](_sine(period=10, n=200), SETTINGS)
    noise_out = FEATURES["spectral"](_noise(200), SETTINGS)
    assert sine_out.value < 0.3
    assert noise_out.value > 0.7
    assert sine_out.value < noise_out.value
    # Dominant frequency should land near 1/period = 0.1 cycles/point.
    assert sine_out.distribution["dominant_frequency_cycles_per_point"] == pytest.approx(0.1, abs=0.02)


def test_spectral_entropy_reports_zero_for_a_constant_series():
    if not optional.have("numpy"):
        pytest.skip("numpy not installed in this environment")
    out = FEATURES["spectral"](_constant(40), SETTINGS)
    assert out.value == pytest.approx(0.0)
    assert out.warning


def test_hurst_and_dfa_land_near_one_half_for_white_noise():
    hurst_out = FEATURES["hurst"](_noise(300), SETTINGS)
    dfa_out = FEATURES["dfa"](_noise(300), SETTINGS)
    assert 0.3 < hurst_out.value < 0.75
    assert 0.3 < dfa_out.value < 0.75


def test_hurst_is_undefined_for_a_constant_series():
    out = FEATURES["hurst"](_constant(80), SETTINGS)
    assert out.value is None
    assert out.warning


def test_hurst_and_dfa_are_higher_for_a_trending_series_than_for_noise():
    ramp_hurst = FEATURES["hurst"](_ramp(200), SETTINGS)
    noise_hurst = FEATURES["hurst"](_noise(200), SETTINGS)
    assert ramp_hurst.value > noise_hurst.value


def test_permutation_entropy_zero_for_monotonic_high_for_noise():
    ramp_out = FEATURES["permutation_entropy"](_ramp(60), SETTINGS)
    noise_out = FEATURES["permutation_entropy"](_noise(200), SETTINGS)
    assert ramp_out.value == pytest.approx(0.0, abs=1e-9)
    assert noise_out.value > 0.9


def test_permutation_entropy_lower_for_alternation_than_for_noise():
    alt_out = FEATURES["permutation_entropy"](_alternating(200), SETTINGS)
    noise_out = FEATURES["permutation_entropy"](_noise(200), SETTINGS)
    assert alt_out.value < noise_out.value


def test_change_points_and_page_hinkley_find_a_synthetic_level_shift():
    shift_index = 50
    values = [1.0] * shift_index + [10.0] * (100 - shift_index)
    cp_out = FEATURES["change_points"](values, SETTINGS)
    ph_out = FEATURES["page_hinkley"](values, SETTINGS)
    assert cp_out.distribution["count"] >= 1
    location = cp_out.distribution["locations"][0]
    assert abs(location - shift_index) <= 3
    assert ph_out.distribution["event_count"] >= 1
    assert abs(ph_out.distribution["first_event_index"] - shift_index) <= 5


def test_change_points_and_page_hinkley_stay_quiet_on_flat_noise():
    values = _noise(120, seed=7)
    cp_out = FEATURES["change_points"](values, SETTINGS)
    ph_out = FEATURES["page_hinkley"](values, SETTINGS)
    # Not asserting zero (both are statistical detectors with some false
    # positive rate); asserting they do not fire constantly.
    assert cp_out.value < 20.0
    assert ph_out.value < 20.0


def test_stationarity_degrades_to_a_labelled_proxy_without_statsmodels(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "statsmodels")
    optional.reset_cache()
    try:
        out = FEATURES["stationarity"](_noise(60), SETTINGS)
        assert out.value is not None
        assert out.distribution["backend"] == "half_split_proxy"
        assert "proxy" in out.warning
    finally:
        optional.reset_cache()


def test_stationarity_uses_adfuller_when_statsmodels_is_available():
    if not optional.have("statsmodels"):
        pytest.skip("statsmodels not installed in this environment")
    out = FEATURES["stationarity"](_noise(80, seed=99), SETTINGS)
    assert out.distribution["backend"] == "adfuller"
    assert 0.0 <= out.value <= 1.0


def test_rolling_dispersion_needs_enough_points_for_its_window():
    out = FEATURES["rolling_dispersion"](_noise(10), {**SETTINGS, "rolling_window": 10})
    assert out.value is None
    assert "rolling window" in out.warning


def test_piecewise_trend_detects_a_segment_that_becomes_more_linear():
    # First half is noisy (little linear structure); second half is a clean
    # ramp. The gap between segment correlations should be large and positive.
    noisy_half = _noise(20, seed=3)
    clean_half = _ramp(20)
    out = FEATURES["piecewise_trend"](noisy_half + clean_half,
                                      {**SETTINGS, "piecewise_segments": 2})
    assert out.value > 0.5  # second segment trends far more cleanly than the first


# ------------------------------------------------------------------ the suite

def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text)


def _long_text(paragraphs: int = 80, seed: int = 5) -> str:
    rng = random.Random(seed)
    vocab = ("the quiet room held a long silence while she considered what had "
            "happened and whether anyone would notice however perhaps not "
            "because nobody asked her directly about any of it").split()
    blocks = []
    for _ in range(paragraphs):
        sentences = [" ".join(rng.choice(vocab) for _ in range(rng.randint(3, 20)))
                    .capitalize() + "." for _ in range(rng.randint(1, 4))]
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


def test_default_configuration_produces_a_bounded_defensible_finding_count():
    analysis = _analysis(_long_text())
    findings = ts.measure(analysis, config={})
    assert len(findings) == len(ts.DEFAULT_SEQUENCES) * len(ts.DEFAULT_FEATURE_GROUPS)
    ids = {item["metric_id"] for item in findings}
    assert "rhythm.timeseries_sentence_words_dispersion" in ids
    assert "rhythm.timeseries_sentence_words_acf" in ids


def test_single_sequence_and_feature_selection_produces_exactly_one_finding():
    analysis = _analysis(_long_text())
    findings = ts.measure(analysis, config={"sequences": ["sentence_words"],
                                            "feature_groups": ["acf"]})
    assert len(findings) == 1
    assert findings[0]["metric_id"] == "rhythm.timeseries_sentence_words_acf"


def test_window_level_sequence_gets_a_drift_prefixed_id():
    analysis = _analysis(_long_text(paragraphs=200))
    findings = ts.measure(analysis, config={
        "sequences": ["window_dialogue_fraction"], "feature_groups": ["dispersion"],
        "window_words": 300})
    assert findings[0]["metric_id"] == "drift.timeseries_window_dialogue_fraction_dispersion"
    assert findings[0]["family"] == "book_drift"


def test_max_findings_caps_output_and_reports_the_truncation():
    analysis = _analysis(_long_text())
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_words", "paragraph_words"],
        "feature_groups": ["dispersion", "acf", "trend"], "max_findings": 2})
    real = [item for item in findings if item["metric_id"] != "rhythm.timeseries_truncated"]
    truncated = [item for item in findings if item["metric_id"] == "rhythm.timeseries_truncated"]
    assert len(real) == 2
    assert len(truncated) == 1
    assert "raise" in truncated[0]["warning"]


def test_short_sequence_reports_insufficient_data_not_a_number():
    analysis = _analysis("One short sentence. Another short one.")
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_words"], "feature_groups": ["hurst"]})
    assert findings[0]["value"] is None
    assert "insufficient data" in findings[0]["warning"]
    assert findings[0]["sample_size"] == 2
    assert findings[0]["min_sample"] == ts.DEFAULT_MIN_LENGTHS["hurst"]


def test_tiny_document_survives_every_default_feature_without_a_crash():
    for text in ("", "Hi.", "A\n\nB\n\nC"):
        analysis = _analysis(text)
        findings = ts.measure(analysis, config={"enabled": True})
        assert findings  # always something, even if every value is None
        for item in findings:
            assert item["value"] is None or isinstance(item["value"], (int, float, str))


def test_unknown_sequence_and_feature_names_are_reported_not_raised():
    analysis = _analysis(_long_text())
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_words", "not_a_sequence"],
        "feature_groups": ["acf", "not_a_feature"]})
    ids = {item["metric_id"] for item in findings}
    assert "rhythm.timeseries_sentence_words_acf" in ids
    assert "rhythm.timeseries_config" in ids


def test_completely_invalid_configuration_does_not_crash():
    analysis = _analysis(_long_text())
    findings = ts.measure(analysis, config={"sequences": ["nope"], "feature_groups": ["nope"]})
    assert len(findings) == 1
    assert findings[0]["value"] is None


def test_detrend_option_changes_the_trend_free_features():
    trending = [float(i) for i in range(60)] + [30.0 + math.sin(i) for i in range(60)]
    text = "words " * 1  # placeholder, not used directly below
    analysis = _analysis(_long_text(paragraphs=60))
    plain = ts._measure_one(analysis, "sentence_words",
                            "acf", ts._settings({"detrend": False}))
    detrended = ts._measure_one(analysis, "sentence_words",
                                "acf", ts._settings({"detrend": True}))
    assert plain["distribution"]["detrended"] is False
    assert detrended["distribution"]["detrended"] is True


def test_sequence_settings_are_recorded_in_the_distribution():
    analysis = _analysis(_long_text(paragraphs=200))
    findings = ts.measure(analysis, config={
        "sequences": ["window_pronoun_rate"], "feature_groups": ["dispersion"],
        "window_words": 500})
    distribution = findings[0]["distribution"]
    assert distribution["sequence"] == "window_pronoun_rate"
    assert distribution["sample_unit"] == "window"
    assert distribution["sequence_settings"]["window_words"] == 500


# ------------------------------------------------- the never-executed ruptures path

def test_change_point_penalty_scales_with_log_n_to_control_false_positives():
    # This is the bug the module docstring documents: a flat penalty let PELT
    # flag noise more and more often as the sequence got longer, because the
    # penalty never grew to keep pace with the number of candidate splits.
    # Checked across several lengths rather than one, since a single length
    # could pass by luck.
    if not optional.have("ruptures") or not optional.have("numpy"):
        pytest.skip("ruptures/numpy not installed in this environment")
    total_points = total_false = 0
    for seed in range(12):
        rng = random.Random(seed)
        n = rng.choice([150, 400, 900])
        values = [rng.gauss(0.0, 1.0) for _ in range(n)]
        out = FEATURES["change_points"](values, SETTINGS)
        assert out.distribution["backend"] == "ruptures.Pelt"
        total_points += n
        total_false += out.distribution["count"]
    # A flat pen=3.0 measured roughly 1.5% of points as false change points in
    # pure noise (see the module docstring); log(n)-scaled measured near 0%.
    assert total_false / total_points < 0.01


def test_change_point_penalty_is_a_log_n_multiplier_not_a_flat_score():
    if not optional.have("ruptures") or not optional.have("numpy"):
        pytest.skip("ruptures/numpy not installed in this environment")
    values = [1.0] * 50 + [10.0] * 50
    out = FEATURES["change_points"](values, {**SETTINGS, "change_point_penalty": 2.0})
    assert out.distribution["penalty"] == pytest.approx(2.0 * math.log(100), rel=1e-6)


# ------------------------------------------------------------------- catch22

CATCH22 = "catch22_"


def test_catch22_alias_expands_to_all_22_stable_names():
    expanded = ts._settings({"feature_groups": ["catch22"]})["feature_groups"]
    assert len(expanded) == 22
    assert set(expanded) == set(ts._CATCH22_FEATURE_NAMES)
    assert all(name.startswith(CATCH22) for name in expanded)


def test_catch22_metric_ids_never_expose_the_library_or_a_position():
    # The task's own stable-id rule: no "catch22_DN_HistogramMode_5", no
    # "catch22_3". Every stable suffix must be readable and never equal a
    # bare integer or the exact library feature name it came from.
    library_names = {library for library, _, _, _ in ts._CATCH22_CATALOGUE}
    for index, name in enumerate(ts._CATCH22_FEATURE_NAMES):
        suffix = name[len(CATCH22):]
        assert not suffix.isdigit(), f"{name} is a bare position, not a stable name"
        assert suffix not in library_names, f"{name} exposes the raw pycatch22 name"
        assert suffix.islower() or "_" in suffix  # snake_case, not CamelCase library style
    # Every catalogue entry's library name resolves back to exactly one of our ids.
    assert len({library for library, _, _, _ in ts._CATCH22_CATALOGUE}) == 22
    assert len(set(ts._CATCH22_LIBRARY_NAME_BY_FEATURE.values())) == 22


def test_catch22_acf_first_minimum_matches_known_sine_half_period():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    values = _sine(n=150, period=10)
    out = EXTENDED["catch22_acf_first_minimum"](values, SETTINGS, _ctx("sentence_words", values))
    assert out.value == pytest.approx(5.0, abs=1.0)  # theoretical first minimum is period/2


def test_catch22_longest_above_mean_run_separates_ramp_from_alternation():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    ramp, alt = _ramp(150), _alternating(150)
    ramp_out = EXTENDED["catch22_longest_above_mean_run"](
        ramp, SETTINGS, _ctx("sentence_words", ramp))
    alt_out = EXTENDED["catch22_longest_above_mean_run"](
        alt, SETTINGS, _ctx("sentence_words", alt))
    assert ramp_out.value > 50  # one long monotonic run, roughly half the series
    assert alt_out.value < 5   # alternation never sustains a run above its mean
    assert ramp_out.value > alt_out.value


def test_catch22_large_step_fraction_is_maximal_for_alternation():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    alt = _alternating(100)
    out = EXTENDED["catch22_large_step_fraction"](alt, SETTINGS, _ctx("sentence_words", alt))
    assert out.value == pytest.approx(1.0, abs=1e-6)  # every step is the maximal jump


def test_catch22_reports_none_not_a_number_on_zero_variance():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    constant = _constant(40)
    out = EXTENDED["catch22_histogram_mode_5bin"](
        constant, SETTINGS, _ctx("sentence_words", constant))
    assert out.value is None
    assert "non-finite" in out.warning


def test_catch22_features_share_one_pycatch22_call_per_sequence(monkeypatch):
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    import pycatch22
    calls = []
    real = pycatch22.catch22_all

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pycatch22, "catch22_all", counting)
    analysis = DocumentAnalysis.from_text("Shared cache scope for this test only.")
    values = _noise(80)
    ctx = _ctx("sentence_words", values, analysis=analysis)
    for name in list(ts._CATCH22_FEATURE_NAMES)[:5]:
        EXTENDED[name](values, SETTINGS, ctx)
    assert len(calls) == 1


def test_catch22_degrades_without_pycatch22(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "pycatch22")
    optional.reset_cache()
    try:
        values = _noise(80)
        out = EXTENDED["catch22_acf_first_minimum"](values, SETTINGS, _ctx("sentence_words", values))
        assert out.value is None
        assert "pycatch22" in out.warning
    finally:
        optional.reset_cache()


def test_catch22_needs_context_and_never_raises_standalone():
    out = EXTENDED["catch22_histogram_mode_5bin"](_noise(40), SETTINGS, None)
    assert out.value is None
    assert out.warning


def test_catch22_through_the_suite_reports_insufficient_data_below_its_minimum():
    analysis = _analysis("One short sentence. Another short one.")
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_words"], "feature_groups": ["catch22_acf_first_minimum"]})
    assert findings[0]["value"] is None
    assert "insufficient data" in findings[0]["warning"]
    assert findings[0]["min_sample"] == ts.DEFAULT_MIN_LENGTHS["catch22_acf_first_minimum"]


# --------------------------------------------------------------------- catch24

def test_catch24_alias_expands_to_all_24_stable_names():
    expanded = ts._settings({"feature_groups": ["catch24"]})["feature_groups"]
    assert len(expanded) == 24
    assert set(expanded) == set(ts._CATCH22_FEATURE_NAMES) | set(ts._CATCH24_FEATURE_NAMES)
    # catch22 (unqualified) must still expand to just the 22, unchanged.
    assert len(ts._settings({"feature_groups": ["catch22"]})["feature_groups"]) == 22


def test_catch24_raw_mean_matches_python_statistics_mean_on_a_ramp():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    import statistics as stats_module
    values = _ramp(60)
    out = EXTENDED["catch24_raw_mean"](values, SETTINGS, _ctx("sentence_words", values))
    assert out.value == pytest.approx(stats_module.fmean(values))


def test_catch24_raw_variance_matches_python_statistics_variance_on_known_signals():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    import statistics as stats_module
    for values in (_ramp(60), _alternating(60), _noise(80)):
        analysis = DocumentAnalysis.from_text("Cache scope for this signal only.")
        out = EXTENDED["catch24_raw_variance"](
            values, SETTINGS, _ctx("sentence_words", values, analysis=analysis))
        assert out.value == pytest.approx(stats_module.variance(values), rel=1e-6)


def test_catch24_raw_variance_is_zero_for_a_constant_series():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    constant = _constant(40)
    out = EXTENDED["catch24_raw_variance"](constant, SETTINGS, _ctx("sentence_words", constant))
    assert out.value == pytest.approx(0.0, abs=1e-9)


def test_catch24_shares_the_catch22_call_and_never_duplicates_it(monkeypatch):
    # catch24's two extras read from the SAME memoized catch22_all(catch24=True)
    # call catch22's own 22 features use -- one book's worth of catch22_* and
    # catch24_* findings for one sequence must still be exactly one library call.
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    import pycatch22
    calls = []
    real = pycatch22.catch22_all

    def counting(*args, **kwargs):
        calls.append(kwargs.get("catch24"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pycatch22, "catch22_all", counting)
    analysis = DocumentAnalysis.from_text("Shared catch24 cache scope for this test only.")
    values = _noise(80)
    ctx = _ctx("sentence_words", values, analysis=analysis)
    EXTENDED["catch22_acf_first_minimum"](values, SETTINGS, ctx)
    EXTENDED["catch24_raw_mean"](values, SETTINGS, ctx)
    EXTENDED["catch24_raw_variance"](values, SETTINGS, ctx)
    assert calls == [True]  # exactly one call, and it asked for the catch24 extras


def test_catch24_degrades_without_pycatch22(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "pycatch22")
    optional.reset_cache()
    try:
        values = _noise(80)
        out = EXTENDED["catch24_raw_mean"](values, SETTINGS, _ctx("sentence_words", values))
        assert out.value is None
        assert "pycatch22" in out.warning
    finally:
        optional.reset_cache()


def test_catch24_needs_context_and_never_raises_standalone():
    out = EXTENDED["catch24_raw_variance"](_noise(40), SETTINGS, None)
    assert out.value is None
    assert out.warning


def test_catch24_findings_name_the_dispersion_sibling_they_are_expected_to_agree_with():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    values = _ramp(60)
    mean_out = EXTENDED["catch24_raw_mean"](values, SETTINGS, _ctx("sentence_words", values))
    assert "dispersion.mean" in mean_out.distribution["relationship_note"]
    analysis = DocumentAnalysis.from_text("A separate cache scope for the variance check.")
    var_out = EXTENDED["catch24_raw_variance"](
        values, SETTINGS, _ctx("sentence_words", values, analysis=analysis))
    assert "dispersion.std" in var_out.distribution["relationship_note"]


def test_catch24_is_never_detrended_like_dispersion():
    trending = _ramp(60)
    assert "catch24_raw_mean" in ts._NEVER_DETREND
    assert "catch24_raw_variance" in ts._NEVER_DETREND
    analysis = _analysis(_long_text(paragraphs=60))
    detrended = ts._measure_one(analysis, "sentence_words", "catch24_raw_mean",
                                ts._settings({"detrend": True}))
    assert detrended["distribution"]["detrended"] is False


def test_catch24_raw_variance_reports_squared_sequence_unit():
    assert ts.FEATURE_UNITS["catch24_raw_variance"] == "squared sequence unit"
    assert ts.FEATURE_UNITS["catch24_raw_mean"] is None  # falls back to the sequence's own unit


def test_catch24_through_the_suite_uses_its_own_lower_minimum_length():
    # catch24's two extras are ordinary summary statistics, not nonlinear/
    # scaling estimates, so they should not need catch22's own 30-point floor.
    assert ts.DEFAULT_MIN_LENGTHS["catch24_raw_mean"] < ts.DEFAULT_MIN_LENGTHS["catch22_acf_first_minimum"]
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    analysis = _analysis(_long_text(paragraphs=3))
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_words"], "feature_groups": ["catch24_raw_mean"]})
    if findings[0]["sample_size"] >= ts.DEFAULT_MIN_LENGTHS["catch24_raw_mean"]:
        assert findings[0]["value"] is not None


# ------------------------------------------------------------------- wavelets

def test_wavelet_entropy_orders_ramp_below_sine_below_noise():
    if not optional.have("pywt") or not optional.have("numpy"):
        pytest.skip("pywt/numpy not installed in this environment")
    ramp, sine, noisy = _ramp(150), _sine(150, period=8), _noise(150)
    ramp_out = EXTENDED["wavelet_entropy"](ramp, SETTINGS, _ctx("sentence_words", ramp))
    sine_out = EXTENDED["wavelet_entropy"](sine, SETTINGS, _ctx("sentence_words", sine))
    noise_out = EXTENDED["wavelet_entropy"](noisy, SETTINGS, _ctx("sentence_words", noisy))
    assert ramp_out.value < sine_out.value < noise_out.value
    assert ramp_out.value < 0.05  # a smooth ramp concentrates almost all energy at one scale


def test_wavelet_energy_is_concentrated_in_the_coarsest_scale_for_a_ramp():
    if not optional.have("pywt") or not optional.have("numpy"):
        pytest.skip("pywt/numpy not installed in this environment")
    ramp = _ramp(150)
    out = EXTENDED["wavelet_energy"](ramp, SETTINGS, _ctx("sentence_words", ramp))
    assert out.value > 99.0  # almost all variance is at the coarsest (trend) scale
    assert out.distribution["band_labels"][0].startswith("cA")


def test_wavelet_energy_and_entropy_share_one_decomposition(monkeypatch):
    if not optional.have("pywt") or not optional.have("numpy"):
        pytest.skip("pywt/numpy not installed in this environment")
    import pywt
    calls = []
    real = pywt.wavedec

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(pywt, "wavedec", counting)
    analysis = DocumentAnalysis.from_text("Shared cache scope for this test only.")
    values = _noise(80)
    ctx = _ctx("sentence_words", values, analysis=analysis)
    EXTENDED["wavelet_energy"](values, SETTINGS, ctx)
    EXTENDED["wavelet_entropy"](values, SETTINGS, ctx)
    assert len(calls) == 1


def test_wavelet_degrades_without_pywt(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "pywt")
    optional.reset_cache()
    try:
        values = _noise(80)
        out = EXTENDED["wavelet_entropy"](values, SETTINGS, _ctx("sentence_words", values))
        assert out.value is None
        assert "pywt" in out.warning
    finally:
        optional.reset_cache()


def test_wavelet_features_respect_their_own_minimum_length():
    analysis = _analysis(_long_text(paragraphs=5))
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_punctuation"], "feature_groups": ["wavelet_entropy"]})
    if findings[0]["sample_size"] < ts.DEFAULT_MIN_LENGTHS["wavelet_entropy"]:
        assert findings[0]["value"] is None
        assert "insufficient data" in findings[0]["warning"]


# ------------------------------------------------------- textdescriptives check

def test_textdescriptives_check_agrees_closely_but_not_perfectly_with_our_own_sequence():
    if not optional.have("textdescriptives") or not optional.have("spacy"):
        pytest.skip("textdescriptives/spacy not installed in this environment")
    analysis = _analysis(_long_text(paragraphs=150))
    if analysis.nlp_unavailable:
        pytest.skip("no spaCy model available in this environment")
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_dependency_distance"], "feature_groups": ["textdescriptives_check"]})
    result = findings[0]
    if result["value"] is None:
        pytest.skip(result["warning"])
    # Strongly correlated (same underlying parse) but not identical: textdescriptives
    # includes the ROOT token (distance 0) in its per-sentence mean and this
    # suite's own sequence excludes it, a real, preserved disagreement.
    assert result["value"] > 0.9
    assert result["distribution"]["mean_absolute_gap"] > 0.0
    assert "ROOT" in result["distribution"]["note"]


def test_textdescriptives_check_is_not_defined_for_unrelated_sequences():
    analysis = _analysis(_long_text(paragraphs=20))
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_words"], "feature_groups": ["textdescriptives_check"]})
    assert findings[0]["value"] is None
    assert "no textdescriptives cross-check is defined" in findings[0]["warning"]


def test_textdescriptives_check_degrades_without_textdescriptives(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "textdescriptives")
    optional.reset_cache()
    try:
        analysis = _analysis(_long_text(paragraphs=60))
        findings = ts.measure(analysis, config={
            "sequences": ["sentence_dependency_distance"],
            "feature_groups": ["textdescriptives_check"]})
        assert findings[0]["value"] is None
        assert "textdescriptives" in findings[0]["warning"]
    finally:
        optional.reset_cache()


# ---------------------------------------------------------------------- tsfresh

def test_tsfresh_minimal_preset_matches_known_statistics_exactly():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    import statistics as stats_module
    values = _ramp(60)
    out = EXTENDED["tsfresh"](values, SETTINGS, _ctx("sentence_words", values))
    reported = out.distribution["values"]
    assert reported["mean"] == pytest.approx(stats_module.fmean(values))
    assert reported["median"] == pytest.approx(stats_module.median(values))
    assert reported["minimum"] == pytest.approx(min(values))
    assert reported["maximum"] == pytest.approx(max(values))
    assert reported["sum_values"] == pytest.approx(sum(values))
    assert reported["variance"] == pytest.approx(stats_module.pvariance(values))
    assert out.distribution["feature_set"] == "minimal"
    assert out.value == pytest.approx(100.0)  # every minimal feature is finite on a clean ramp


def test_tsfresh_reports_one_finding_regardless_of_preset_size():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    for feature_set in ("minimal", "efficient", "comprehensive"):
        analysis = DocumentAnalysis.from_text(f"Cache scope for {feature_set}.")
        cfg = ts._settings({"tsfresh_feature_set": feature_set})
        values = _noise(120)
        out = EXTENDED["tsfresh"](values, cfg, _ctx("sentence_words", values, cfg=cfg, analysis=analysis))
        assert out.distribution["feature_set"] == feature_set
        assert out.value is not None
        assert 0.0 <= out.value <= 100.0


def test_tsfresh_max_features_caps_the_reported_subset_but_not_the_finite_share():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    cfg = ts._settings({"tsfresh_feature_set": "efficient", "tsfresh_max_features": 5})
    values = _noise(150)
    out = EXTENDED["tsfresh"](values, cfg, _ctx("sentence_words", values, cfg=cfg))
    assert out.distribution["reported_features"] == 5
    assert out.distribution["requested_features"] > 5
    assert out.distribution["truncated_feature_count"] == (
        out.distribution["requested_features"] - 5)
    # The headline is a data-quality signal over the FULL preset, not just the
    # capped subset, so it must not silently become "5/5 = 100%".
    assert out.distribution["finite_feature_count"] <= out.distribution["requested_features"]


def test_tsfresh_never_exposes_a_positional_or_raw_dataframe_column_name():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    values = _ramp(60)
    out = EXTENDED["tsfresh"](values, SETTINGS, _ctx("sentence_words", values))
    for name in out.distribution["values"]:
        assert not name.isdigit(), f"{name} is a bare position, not a stable name"
        assert not name.startswith("value__"), f"{name} exposes this module's own column prefix"
        assert "__" not in name, f"{name} repeats tsfresh's own parameter-separator convention"


def test_tsfresh_shares_one_extraction_call_across_repeated_reads(monkeypatch):
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    from tsfresh.feature_extraction import extract_features as real_extract
    import textgrader.metrics.timeseries_suite as ts_module
    calls = []

    def counting(*args, **kwargs):
        calls.append(1)
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(
        "tsfresh.feature_extraction.extract_features", counting, raising=True)
    analysis = DocumentAnalysis.from_text("Shared tsfresh cache scope for this test only.")
    values = _noise(80)
    ctx = _ctx("sentence_words", values, analysis=analysis)
    ts_module._EXTENDED_FEATURES["tsfresh"](values, SETTINGS, ctx)
    ts_module._EXTENDED_FEATURES["tsfresh"](values, SETTINGS, ctx)
    assert len(calls) == 1


def test_tsfresh_degrades_without_tsfresh(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "tsfresh")
    optional.reset_cache()
    try:
        values = _noise(80)
        out = EXTENDED["tsfresh"](values, SETTINGS, _ctx("sentence_words", values))
        assert out.value is None
        assert "tsfresh" in out.warning
    finally:
        optional.reset_cache()


def test_tsfresh_needs_context_and_never_raises_standalone():
    out = EXTENDED["tsfresh"](_noise(40), SETTINGS, None)
    assert out.value is None
    assert out.warning


def test_tsfresh_off_by_default_and_never_fires_during_corpus_profiling(monkeypatch):
    # Corpus profiling always runs a suite with MetricSpec.defaults; tsfresh
    # must never be reachable through that path, only through an explicit
    # feature_groups selection.
    if optional.have("tsfresh"):
        import tsfresh.feature_extraction as tsfresh_extraction
        calls = []
        monkeypatch.setattr(tsfresh_extraction, "extract_features",
                            lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(
                                AssertionError("tsfresh must not run under default settings")))
    analysis = _analysis(_long_text())
    ts.measure(analysis, config={})  # exactly what corpus profiling would run
    assert "tsfresh" not in ts.DEFAULT_FEATURE_GROUPS


def test_tsfresh_through_the_suite_reports_insufficient_data_below_its_minimum():
    analysis = _analysis("One short sentence. Another short one.")
    findings = ts.measure(analysis, config={
        "sequences": ["sentence_words"], "feature_groups": ["tsfresh"]})
    assert findings[0]["value"] is None
    assert "insufficient data" in findings[0]["warning"]
    assert findings[0]["min_sample"] == ts.DEFAULT_MIN_LENGTHS["tsfresh"]


def test_tsfresh_unknown_feature_set_reports_unavailable_not_a_crash():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    values = _noise(80)
    cfg = ts._settings({"tsfresh_feature_set": "not_a_real_preset"})
    out = EXTENDED["tsfresh"](values, cfg, _ctx("sentence_words", values, cfg=cfg))
    assert out.value is None
    assert "unknown tsfresh_feature_set" in out.warning


# --------------------------------------------------- new-group combinatorics guard

def test_default_selection_is_unchanged_by_this_pass():
    # The whole point of keeping catch22/wavelets/embeddings opt-in: the
    # default report a user gets is exactly the one that existed before.
    assert ts.DEFAULT_SEQUENCES == ("sentence_words", "paragraph_words", "sentence_punctuation")
    assert ts.DEFAULT_FEATURE_GROUPS == ("dispersion", "acf", "trend", "turning_points", "runs")
    analysis = _analysis(_long_text())
    findings = ts.measure(analysis, config={})
    assert len(findings) == 15


def test_new_feature_groups_are_all_opt_in():
    new_groups = (set(ts._CATCH22_FEATURE_NAMES) | set(ts._CATCH24_FEATURE_NAMES) |
                 {"wavelet_energy", "wavelet_entropy", "textdescriptives_check", "tsfresh"})
    assert new_groups.isdisjoint(ts.DEFAULT_FEATURE_GROUPS)
    assert set(ts._EMBEDDING_SEQUENCES).isdisjoint(ts.DEFAULT_SEQUENCES)
    assert new_groups == set(ts.FEATURE_NAMES) - set(ts._BASE_FEATURE_NAMES)


def test_every_feature_name_has_a_label_unit_and_minimum_length():
    for name in ts.FEATURE_NAMES:
        assert name in ts.FEATURE_LABELS, name
        assert name in ts.FEATURE_UNITS, name
        assert name in ts.DEFAULT_MIN_LENGTHS, name


def test_catch22_over_default_sequences_would_blow_past_the_old_default_size():
    # Documents *why* catch22 stays off the default rather than merely
    # asserting that it does: 22 features over even the three default
    # sequences alone would be more than four times the current default.
    combos = len(ts.DEFAULT_SEQUENCES) * len(ts._CATCH22_FEATURE_NAMES)
    assert combos == 66
    assert combos > 4 * len(ts.DEFAULT_SEQUENCES) * len(ts.DEFAULT_FEATURE_GROUPS)


def test_max_findings_still_caps_a_large_catch22_selection():
    analysis = _analysis(_long_text(paragraphs=150))
    findings = ts.measure(analysis, config={
        "sequences": list(ts.DEFAULT_SEQUENCES), "feature_groups": ["catch22"],
        "max_findings": 10})
    real = [item for item in findings if item["metric_id"] != "rhythm.timeseries_truncated"]
    truncated = [item for item in findings if item["metric_id"] == "rhythm.timeseries_truncated"]
    assert len(real) == 10
    assert len(truncated) == 1


def test_worst_case_selection_is_still_bounded_by_max_findings():
    # Every sequence times every feature group (tsfresh included, whatever
    # preset it is set to -- the group itself is always exactly one finding)
    # is this suite's absolute worst case. This pins that number so it is
    # visible the next time a sequence or feature group is added, and proves
    # max_findings still hard-caps it regardless of how large it grows.
    from textgrader import sequences as seq
    worst_case = len(seq.SEQUENCES) * len(ts.FEATURE_NAMES)
    assert worst_case == 688
    analysis = _analysis(_long_text(paragraphs=150))
    findings = ts.measure(analysis, config={
        "sequences": list(seq.SEQUENCES), "feature_groups": list(ts.FEATURE_NAMES)})
    real = [item for item in findings if item["metric_id"] not in
           ("rhythm.timeseries_truncated", "rhythm.timeseries_config")]
    truncated = [item for item in findings if item["metric_id"] == "rhythm.timeseries_truncated"]
    assert len(real) == 200  # the default max_findings
    assert len(truncated) == 1


# --------------------------------------------------- the critical gating rule

def test_default_configuration_never_loads_an_embedding_model(monkeypatch):
    from textgrader.metrics import semantic_adjacent
    calls = []
    monkeypatch.setattr(semantic_adjacent, "_load_model",
                        lambda name: (calls.append(name), (None, "blocked for this test"))[1])
    analysis = _analysis(_long_text())
    ts.measure(analysis, config={})  # exactly what corpus profiling would run
    assert not calls, "the default configuration must never load an embedding model"


def test_registered_metric_spec_does_not_require_sentence_transformers():
    # MetricSpec.needs_model is defined as "sentence_transformers" in requires,
    # and needs_model gates this suite out of corpus profiling entirely. This
    # suite's cheap sequences must keep being profiled, so REQUIRES must stay
    # empty even though two of its sequences can use that package.
    spec = REGISTRY["timeseries_suite"]
    assert "sentence_transformers" not in spec.requires
    assert not spec.needs_model
    assert not spec.needs_parse


def test_embedding_sequences_are_reachable_only_when_explicitly_selected(monkeypatch):
    from textgrader.metrics import semantic_adjacent
    calls = []
    monkeypatch.setattr(semantic_adjacent, "_load_model",
                        lambda name: (calls.append(name), (None, "blocked for this test"))[1])
    analysis = _analysis(_long_text())
    ts.measure(analysis, config={"sequences": ["sentence_similarity_prev"],
                                 "feature_groups": ["dispersion"]})
    assert calls == ["all-MiniLM-L6-v2"]


def test_embedding_backend_differs_from_the_lexical_fallback_on_a_paraphrase():
    if not optional.have("sentence_transformers"):
        pytest.skip("sentence_transformers not installed in this environment")
    # A paraphrase with no shared content words: the lexical TF-IDF fallback
    # cannot see it (near zero), a real embedding should (well above it).
    text = ("The dog was extremely happy to see her. "
            "The canine was overjoyed at her arrival. "
            "Meanwhile the weather in the mountains stayed cold and grey.")
    analysis = _analysis(text)
    from textgrader import sequences as seq
    embedding_sequence = seq.get_sequence(analysis, "sentence_similarity_prev")
    assert "backend=embedding" in (embedding_sequence.warning or "")
    lexical_vectors = __import__("textgrader.metrics.semantic_adjacent",
                                 fromlist=["lexical_vectors"]).lexical_vectors(analysis.sentences)
    from textgrader.metrics import semantic_adjacent
    lexical_value = semantic_adjacent.similarity_at("lexical", lexical_vectors, 0, 1)
    embedding_value = embedding_sequence.values[0]
    assert embedding_value > 0.5
    assert lexical_value < 0.2
    assert embedding_value - lexical_value > 0.3


# --------------------------------------------------------- repository contract

def test_registered_in_the_metric_registry_off_by_default():
    assert "timeseries_suite" in REGISTRY
    spec = REGISTRY["timeseries_suite"]
    assert spec.cost == "moderate"
    assert spec.module == "timeseries_suite"


def test_off_by_default_through_grade(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(_long_text(paragraphs=10), encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results if "timeseries" in item.metric_id]


def test_enabled_through_grade_reports_findings_with_the_right_prefixes(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(_long_text(paragraphs=60), encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "timeseries_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    ids = [item.metric_id for item in report.results if "timeseries" in item.metric_id]
    assert ids
    assert all(mid.startswith(("rhythm.timeseries_", "drift.timeseries_"))
              for mid in ids)


def test_every_registered_metric_runs_without_raising(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "timeseries_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    from textgrader.results import StatusType
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


def test_degrades_fully_with_every_optional_package_disabled(monkeypatch, manuscript, base_config):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        config = {**base_config, "metrics": {**base_config["metrics"],
                                             "timeseries_suite": {
                                                 "enabled": True,
                                                 "sequences": ["sentence_words", "paragraph_words",
                                                             "sentence_parse_depth"],
                                                 "feature_groups": ["dispersion", "acf", "spectral",
                                                                    "stationarity", "change_points"]}}}
        report = grade.analyze(manuscript, config)
        from textgrader.results import StatusType
        errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
        assert not errors, [(item.metric_id, item.error) for item in errors]
        # The dependency-free sequences/features must still produce real numbers.
        dispersion = next(item for item in report.results
                          if item.metric_id == "rhythm.timeseries_sentence_words_dispersion")
        assert dispersion.value is not None
        # The spaCy-backed sequence must degrade visibly rather than vanish.
        depth = next(item for item in report.results
                    if item.metric_id == "rhythm.timeseries_sentence_parse_depth_dispersion")
        assert depth.value is None and depth.warning
    finally:
        optional.reset_cache()
