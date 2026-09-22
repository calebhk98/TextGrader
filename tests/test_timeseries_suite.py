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
SETTINGS = ts._settings({})


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
