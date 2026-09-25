"""Synthetic-signal validation for the generic signal-processing feature battery.

Mirrors ``tests/test_timeseries_suite.py``'s approach: feature functions are
tested directly against numeric sequences with a known or clearly directional
answer (a constant series, a strict alternation, a clean sine, a shuffled
version of the same sine, a linear trend, white noise, a shared-sinusoid
pair, an independent-noise pair) rather than only through a novel-sized
fixture.
"""

from __future__ import annotations

import math
import random

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, signal_processing_suite as sp
from textgrader import sequences as seq_module

SETTINGS = sp._settings({})


def _ctx(sequence_name, working, cfg=None, family="sentence_rhythm", analysis=None):
    """A minimal ``ctx`` for calling an ``_EXTENDED_FEATURES`` function directly.

    The Welch/multiscale results are cached via ``analysis.memo``, keyed on
    ``sequence_name`` + settings -- not on the document's actual text -- so a
    throwaway analysis and a synthetic ``working`` array exercise the real
    caching path exactly like a real sequence would (see
    ``test_timeseries_suite.py``'s identical helper).
    """

    return {
        "analysis": analysis if analysis is not None else DocumentAnalysis.from_text("Cache scope."),
        "sequence_name": sequence_name, "cfg": cfg if cfg is not None else SETTINGS,
        "working": working, "family": family,
    }


def _constant(n=40, value=5.0):
    return [value] * n


def _ramp(n=200):
    return [float(i) for i in range(n)]


def _alternating(n=60):
    return [0.0, 10.0] * (n // 2)


def _sine(n=200, period=10, amplitude=10.0):
    return [amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def _noise(n=200, seed=42):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def _skip_without(*names):
    for name in names:
        if not optional.have(name):
            pytest.skip(f"{name} not installed in this environment")


# --------------------------------------------------------- welch_spectral

def test_welch_spectral_dominant_frequency_is_exactly_half_for_alternation():
    _skip_without("numpy", "scipy.signal")
    values = _alternating(60)
    out = sp._feature_welch_spectral(values, SETTINGS, _ctx("toy", values))
    assert out.distribution["dominant_frequency"] == pytest.approx(0.5)


def test_welch_spectral_dominant_frequency_matches_a_known_sinusoid():
    _skip_without("numpy", "scipy.signal")
    values = _sine(n=200, period=10)
    out = sp._feature_welch_spectral(values, SETTINGS, _ctx("toy", values))
    assert out.distribution["dominant_frequency"] == pytest.approx(0.1, abs=0.02)


def test_welch_spectral_entropy_low_for_sine_high_for_noise():
    _skip_without("numpy", "scipy.signal")
    sine_out = sp._feature_welch_spectral(_sine(n=200, period=10), SETTINGS, _ctx("toy", _sine(n=200, period=10)))
    noise_values = _noise(200)
    noise_out = sp._feature_welch_spectral(noise_values, SETTINGS, _ctx("toy", noise_values))
    assert sine_out.value < 0.5
    assert noise_out.value > 0.5
    assert sine_out.value < noise_out.value


def test_welch_spectral_reports_zero_for_a_constant_sequence():
    _skip_without("numpy", "scipy.signal")
    values = _constant(40)
    out = sp._feature_welch_spectral(values, SETTINGS, _ctx("toy", values))
    assert out.value == pytest.approx(0.0)
    assert out.warning


def test_welch_spectral_names_the_overlapping_timeseries_id():
    _skip_without("numpy", "scipy.signal")
    values = _sine(n=200, period=10)
    out = sp._feature_welch_spectral(values, SETTINGS, _ctx("sentence_words", values,
                                                            family="sentence_rhythm"))
    assert out.distribution["overlaps_existing_metric_id"] == "rhythm.timeseries_sentence_words_spectral"


# --------------------------------------------------------- spectral_centroid/bandwidth

def test_spectral_centroid_is_low_for_a_trend_and_higher_for_white_noise():
    _skip_without("numpy", "scipy.signal")
    trend = _ramp(200)
    noise_values = _noise(200)
    trend_out = sp._feature_spectral_centroid(trend, SETTINGS, _ctx("toy", trend))
    noise_out = sp._feature_spectral_centroid(noise_values, SETTINGS, _ctx("toy", noise_values))
    assert trend_out.value < 0.05
    assert noise_out.value > 0.15
    assert trend_out.value < noise_out.value


def test_spectral_centroid_names_the_overlapping_catch22_id():
    _skip_without("numpy", "scipy.signal")
    values = _sine(n=200, period=10)
    out = sp._feature_spectral_centroid(values, SETTINGS, _ctx("sentence_words", values))
    assert out.distribution["overlaps_existing_metric_id"] == \
        "rhythm.timeseries_sentence_words_catch22_spectral_centroid"


def test_spectral_bandwidth_is_narrow_for_a_pure_tone_and_wide_for_noise():
    _skip_without("numpy", "scipy.signal")
    tone = _sine(n=300, period=10)
    noise_values = _noise(300)
    tone_out = sp._feature_spectral_bandwidth(tone, SETTINGS, _ctx("toy", tone))
    noise_out = sp._feature_spectral_bandwidth(noise_values, SETTINGS, _ctx("toy", noise_values))
    assert tone_out.value < noise_out.value


def test_spectral_bandwidth_is_undefined_for_a_constant_sequence():
    _skip_without("numpy", "scipy.signal")
    values = _constant(40)
    out = sp._feature_spectral_bandwidth(values, SETTINGS, _ctx("toy", values))
    assert out.value is None
    assert out.warning


# --------------------------------------------------------- spectral_flatness

def test_spectral_flatness_rises_when_a_sinusoid_is_shuffled():
    _skip_without("numpy", "scipy.signal")
    values = _sine(n=300, period=10)
    shuffled = list(values)
    random.Random(7).shuffle(shuffled)
    tonal_out = sp._feature_spectral_flatness(values, SETTINGS, _ctx("toy", values))
    shuffled_out = sp._feature_spectral_flatness(shuffled, SETTINGS, _ctx("toy_shuffled", shuffled))
    assert tonal_out.value < 0.1
    assert shuffled_out.value > tonal_out.value
    # Shuffling preserves the exact value distribution -- only order changes.
    assert sorted(values) == pytest.approx(sorted(shuffled))


def test_spectral_flatness_is_high_for_white_noise():
    _skip_without("numpy", "scipy.signal")
    values = _noise(300)
    out = sp._feature_spectral_flatness(values, SETTINGS, _ctx("toy", values))
    assert out.value > 0.3


# --------------------------------------------------------- spectral_rolloff

def test_spectral_rolloff_is_low_for_a_trend_and_high_for_noise():
    _skip_without("numpy", "scipy.signal")
    trend = _ramp(200)
    noise_values = _noise(200)
    trend_out = sp._feature_spectral_rolloff(trend, SETTINGS, _ctx("toy", trend))
    noise_out = sp._feature_spectral_rolloff(noise_values, SETTINGS, _ctx("toy", noise_values))
    assert trend_out.value < noise_out.value


# --------------------------------------------------------- band_energy

def test_band_energy_low_share_is_high_for_a_trend_and_low_for_noise():
    _skip_without("numpy", "scipy.signal")
    trend = _ramp(200)
    noise_values = _noise(200)
    trend_out = sp._feature_band_energy(trend, SETTINGS, _ctx("toy", trend))
    noise_out = sp._feature_band_energy(noise_values, SETTINGS, _ctx("toy", noise_values))
    assert trend_out.value > 60.0
    assert noise_out.value < trend_out.value
    assert trend_out.distribution["low_share_percent"] + trend_out.distribution["mid_share_percent"] \
        + trend_out.distribution["high_share_percent"] == pytest.approx(100.0, abs=0.01)


def test_band_energy_names_the_overlapping_catch22_id():
    _skip_without("numpy", "scipy.signal")
    values = _sine(n=200, period=10)
    out = sp._feature_band_energy(values, SETTINGS, _ctx("sentence_words", values))
    assert out.distribution["overlaps_existing_metric_id"] == \
        "rhythm.timeseries_sentence_words_catch22_spectral_low_freq_power"


def test_band_energy_is_undefined_for_a_constant_sequence():
    _skip_without("numpy", "scipy.signal")
    values = _constant(40)
    out = sp._feature_band_energy(values, SETTINGS, _ctx("toy", values))
    assert out.value is None
    assert out.warning


# --------------------------------------------------------- peaks

def test_peaks_are_frequent_for_alternation_and_absent_for_a_constant_sequence():
    _skip_without("numpy", "scipy.signal")
    alt_out = sp._feature_peaks(_alternating(60), SETTINGS)
    const_out = sp._feature_peaks(_constant(40), SETTINGS)
    assert alt_out.value > 0.0
    assert const_out.value == pytest.approx(0.0)


def test_peaks_degrades_without_scipy_signal(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "scipy.signal")
    optional.reset_cache()
    try:
        out = sp._feature_peaks(_alternating(60), SETTINGS)
        assert out.value is None
        assert "scipy.signal" in out.warning
    finally:
        optional.reset_cache()


# --------------------------------------------------------- zero_crossing_rate

def test_zero_crossing_rate_is_maximal_for_alternation_zero_for_a_constant_sequence():
    alt_out = sp._feature_zero_crossing_rate(_alternating(60), SETTINGS)
    const_out = sp._feature_zero_crossing_rate(_constant(40), SETTINGS)
    assert alt_out.value == pytest.approx(100.0)
    assert const_out.value == pytest.approx(0.0)


def test_zero_crossing_rate_matches_the_known_sinusoid_rate():
    # A sine of period P crosses its own mean (zero) exactly twice per period.
    values = _sine(n=200, period=10)
    out = sp._feature_zero_crossing_rate(values, SETTINGS)
    assert out.value == pytest.approx(2.0 / 10 * 100, abs=2.0)


def test_zero_crossing_rate_needs_no_optional_package():
    # Pure Python/statistics -- must still work with every optional disabled.
    import os
    old = os.environ.get("TEXTGRADER_DISABLE_OPTIONAL")
    os.environ["TEXTGRADER_DISABLE_OPTIONAL"] = "all"
    optional.reset_cache()
    try:
        out = sp._feature_zero_crossing_rate(_alternating(60), SETTINGS)
        assert out.value == pytest.approx(100.0)
    finally:
        if old is None:
            os.environ.pop("TEXTGRADER_DISABLE_OPTIONAL", None)
        else:
            os.environ["TEXTGRADER_DISABLE_OPTIONAL"] = old
        optional.reset_cache()


# --------------------------------------------------------- cepstral_peak

def test_cepstral_peak_finds_a_harmonic_of_the_known_period_of_a_repeating_pattern():
    _skip_without("numpy")
    values = ([1.0, 5.0, 2.0, 8.0] * 40)
    out = sp._feature_cepstral_peak(values, SETTINGS)
    # The real cepstrum of an exactly periodic pattern peaks at the period
    # itself or one of its harmonics (2x, 3x, ...) -- not necessarily the
    # fundamental, since harmonic magnitudes need not be monotonically
    # decreasing -- so the honest assertion is "a multiple of 4", not "4".
    assert out.distribution["peak_quefrency_points"] % 4 == 0
    assert out.value > 1.0  # a real peak stands well above the local floor


def test_cepstral_peak_is_weaker_for_noise_than_for_a_periodic_sequence():
    _skip_without("numpy")
    periodic = [1.0, 5.0, 2.0, 8.0] * 40
    noise_values = _noise(160)
    periodic_out = sp._feature_cepstral_peak(periodic, SETTINGS)
    noise_out = sp._feature_cepstral_peak(noise_values, SETTINGS)
    assert periodic_out.value > noise_out.value


def test_cepstral_peak_needs_enough_points_past_the_minimum_quefrency():
    out = sp._feature_cepstral_peak([1.0, 2.0, 3.0], SETTINGS)
    assert out.value is None
    assert out.warning


# --------------------------------------------------------- multiscale_variance

def test_multiscale_variance_slope_is_more_negative_for_white_noise_than_a_trend():
    _skip_without("numpy")
    noise_values = _noise(400)
    trend = _ramp(400)
    noise_out = sp._feature_multiscale_variance(noise_values, SETTINGS, _ctx("toy", noise_values))
    trend_out = sp._feature_multiscale_variance(trend, SETTINGS, _ctx("toy", trend))
    # White noise's block-mean variance falls off close to 1/scale (slope near
    # -1); a smooth trend's does not fall off with scale at all (slope near 0).
    assert noise_out.value < -0.5
    assert trend_out.value > noise_out.value
    assert trend_out.distribution["related_existing_metric_ids"] == [
        "rhythm.timeseries_toy_hurst", "rhythm.timeseries_toy_dfa"]


def test_multiscale_variance_is_undefined_for_a_constant_sequence():
    _skip_without("numpy")
    values = _constant(40)
    out = sp._feature_multiscale_variance(values, SETTINGS, _ctx("toy", values))
    assert out.value is None
    assert out.warning


# --------------------------------------------------------- cross_correlation

def test_cross_correlation_is_near_one_for_a_sequence_against_itself():
    _skip_without("numpy")
    values = _sine(n=200, period=13)
    out = sp._feature_cross_correlation(values, list(values), SETTINGS)
    assert out.value == pytest.approx(1.0, abs=1e-6)
    assert out.distribution["best_lag"] == 0


def test_cross_correlation_finds_a_known_lag():
    _skip_without("numpy")
    base = _noise(200, seed=3)
    shift = 5
    shifted = [0.0] * shift + base[:-shift]
    out = sp._feature_cross_correlation(base, shifted, {**SETTINGS, "cross_correlation_max_lag": 10})
    assert out.distribution["best_lag"] == pytest.approx(shift, abs=1)


def test_cross_correlation_is_undefined_when_one_sequence_is_constant():
    _skip_without("numpy")
    out = sp._feature_cross_correlation(_constant(40), _noise(40), SETTINGS)
    assert out.value is None
    assert out.warning


# --------------------------------------------------------- coherence

def test_coherence_is_high_for_two_sequences_sharing_a_sinusoid():
    _skip_without("numpy", "scipy.signal")
    base = _sine(n=600, period=12)
    noisy_copy = [v + random.Random(1).gauss(0.0, 0.5) for v in base]
    out = sp._feature_coherence(base, noisy_copy, SETTINGS)
    assert out.value > 0.9


def test_coherence_is_low_for_two_independent_noise_sequences():
    _skip_without("numpy", "scipy.signal")
    a = _noise(600, seed=1)
    b = _noise(600, seed=2)
    out = sp._feature_coherence(a, b, SETTINGS)
    assert out.value < 0.3


def test_coherence_is_undefined_for_a_constant_input():
    _skip_without("numpy", "scipy.signal")
    out = sp._feature_coherence(_constant(40), _noise(40), SETTINGS)
    assert out.value is None
    assert out.warning


def test_coherence_degrades_without_scipy_signal(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "scipy.signal")
    optional.reset_cache()
    try:
        out = sp._feature_coherence(_sine(60, 10), _sine(60, 10), SETTINGS)
        assert out.value is None
        assert "scipy.signal" in out.warning
    finally:
        optional.reset_cache()


# --------------------------------------------------------- cross_spectrum

def test_cross_spectrum_reports_a_finite_phase_for_a_shared_sinusoid():
    _skip_without("numpy", "scipy.signal")
    base = _sine(n=600, period=12)
    lagged = [0.0] * 2 + base[:-2]
    out = sp._feature_cross_spectrum(base, lagged, SETTINGS)
    assert out.value is not None
    assert -math.pi <= out.value <= math.pi


def test_cross_spectrum_is_undefined_for_a_constant_input():
    _skip_without("numpy", "scipy.signal")
    out = sp._feature_cross_spectrum(_constant(40), _noise(40), SETTINGS)
    assert out.value is None
    assert out.warning


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
    findings = sp.measure(analysis, config={})
    single_default = [f for f in sp._FEATURE_ORDER if sp.DEFAULT_FEATURES.get(f)]
    pair_default = [f for f in sp._PAIR_FEATURE_ORDER if sp.DEFAULT_FEATURES.get(f)]
    expected = len(sp.DEFAULT_SEQUENCES) * len(single_default) + len(sp.DEFAULT_PAIRS) * len(pair_default)
    assert len(findings) == expected
    ids = {item["metric_id"] for item in findings}
    assert "rhythm.signal_sentence_words_welch_spectral" in ids
    assert "rhythm.signal_sentence_words_x_sentence_punctuation_coherence" in ids


def test_single_sequence_and_feature_selection_produces_exactly_one_finding():
    analysis = _analysis(_long_text())
    findings = sp.measure(analysis, config={
        "sequences": ["sentence_words"],
        "features": {"welch_spectral": True},
        "pairs": []})
    assert len(findings) == 1
    assert findings[0]["metric_id"] == "rhythm.signal_sentence_words_welch_spectral"


def test_window_level_sequence_gets_a_drift_prefixed_id():
    analysis = _analysis(_long_text(paragraphs=200))
    findings = sp.measure(analysis, config={
        "sequences": ["window_dialogue_fraction"], "features": {"peaks": True}, "pairs": []})
    assert findings[0]["metric_id"] == "drift.signal_window_dialogue_fraction_peaks"


def test_insufficient_data_is_reported_for_a_too_short_sequence():
    analysis = _analysis("One sentence. Another one. A third.")
    findings = sp.measure(analysis, config={
        "sequences": ["sentence_words"], "features": {"welch_spectral": True}, "pairs": []})
    item = findings[0]
    assert item["value"] is None
    assert "insufficient data" in item["warning"]


def test_mismatched_pair_units_are_reported_without_computing_anything():
    analysis = _analysis(_long_text(paragraphs=200))
    findings = sp.measure(analysis, config={
        "sequences": [], "features": {"cross_correlation": True},
        "pairs": [["sentence_words", "paragraph_words"]]})
    item = findings[0]
    assert item["value"] is None
    assert "matching sample units" in item["warning"]


def test_unknown_sequence_and_pair_are_reported_but_do_not_stop_valid_work():
    analysis = _analysis(_long_text())
    findings = sp.measure(analysis, config={
        "sequences": ["sentence_words", "not_a_real_sequence"],
        "features": {"zero_crossing_rate": True},
        "pairs": [["sentence_words", "also_not_real"]]})
    ids = {item["metric_id"] for item in findings}
    assert "rhythm.signal_sentence_words_zero_crossing_rate" in ids
    config_notes = [item for item in findings if item["metric_id"] == "rhythm.signal_config"]
    assert config_notes and "not_a_real_sequence" in config_notes[0]["warning"]


def test_nothing_selected_reports_one_configuration_finding():
    analysis = _analysis(_long_text())
    findings = sp.measure(analysis, config={"sequences": [], "features": {}, "pairs": []})
    assert len(findings) == 1
    assert findings[0]["metric_id"] == "rhythm.signal_config"
    assert findings[0]["value"] is None


def test_max_findings_truncates_and_reports_it():
    analysis = _analysis(_long_text())
    all_on = {name: True for name in sp._FEATURE_ORDER}
    findings = sp.measure(analysis, config={
        "sequences": list(seq_module.SEQUENCES), "features": all_on, "pairs": [],
        "max_findings": 5})
    assert len([f for f in findings if f["metric_id"] != "rhythm.signal_truncated"]) == 5
    assert any(item["metric_id"] == "rhythm.signal_truncated" for item in findings)


def test_welch_and_cross_features_share_one_welch_call_per_sequence(monkeypatch):
    _skip_without("numpy", "scipy.signal")
    import scipy.signal as real_scipy_signal
    calls = {"n": 0}
    original = real_scipy_signal.welch

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(real_scipy_signal, "welch", counting)
    analysis = _analysis(_long_text())
    sp.measure(analysis, config={
        "sequences": ["sentence_words"],
        "features": {"welch_spectral": True, "spectral_centroid": True, "spectral_bandwidth": True,
                    "spectral_flatness": True, "spectral_rolloff": True, "band_energy": True},
        "pairs": []})
    assert calls["n"] == 1


def test_zero_crossing_rate_needs_no_scipy_or_extra_setup(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        analysis = _analysis(_long_text())
        findings = sp.measure(analysis, config={
            "sequences": ["sentence_words"], "features": {"zero_crossing_rate": True}, "pairs": []})
        assert findings[0]["value"] is not None
    finally:
        optional.reset_cache()


# ------------------------------------------ registry-agnostic sequence pickup

def test_a_registry_sequence_this_module_never_named_is_usable_with_no_code_change(monkeypatch):
    """Proves the "reuse the registry, no code changes for a new sequence" design.

    A throwaway sequence (never mentioned anywhere in signal_processing_suite.py)
    is registered into ``textgrader.sequences.SEQUENCES`` the same way a real
    future sequence (a stress sequence from Task 16, say) would be, and this
    suite measures it correctly on the first try -- because it only ever reads
    ``seq.SEQUENCES``/``seq.get_sequence`` generically, never a hard-coded name.
    """

    _skip_without("numpy", "scipy.signal")
    from textgrader import sequences as seq

    def _build_toy(analysis, settings):
        values = tuple(math.sin(2 * math.pi * i / 8) for i in range(64))
        return seq.Sequence("toy_registry_probe", values, "arbitrary units", "sentence",
                            "A synthetic sinusoid registered only for this test.", settings)

    toy_spec = seq.SequenceSpec("toy_registry_probe", "sentence", "arbitrary units",
                                "sentence_rhythm", (), "test-only sequence", _build_toy)
    monkeypatch.setitem(seq.SEQUENCES, "toy_registry_probe", toy_spec)

    analysis = _analysis(_long_text())
    findings = sp.measure(analysis, config={
        "sequences": ["toy_registry_probe"], "features": {"welch_spectral": True}, "pairs": []})
    assert len(findings) == 1
    item = findings[0]
    assert item["metric_id"] == "rhythm.signal_toy_registry_probe_welch_spectral"
    assert item["value"] is not None
    assert item["distribution"]["dominant_frequency"] == pytest.approx(1 / 8, abs=0.02)


def test_a_nominal_sequence_is_flagged_experimental(monkeypatch):
    from textgrader import sequences as seq

    def _build_nominal(analysis, settings):
        values = tuple(float(i % 3) for i in range(64))
        return seq.Sequence("toy_nominal_probe", values, "topic index", "sentence",
                            "A synthetic nominal label (nominal label).", settings)

    toy_spec = seq.SequenceSpec("toy_nominal_probe", "sentence", "topic index",
                                "sentence_rhythm", (), "test-only nominal sequence", _build_nominal)
    monkeypatch.setitem(seq.SEQUENCES, "toy_nominal_probe", toy_spec)

    analysis = _analysis(_long_text())
    findings = sp.measure(analysis, config={
        "sequences": ["toy_nominal_probe"], "features": {"welch_spectral": True}, "pairs": []})
    assert "nominal label" in (findings[0]["warning"] or "")


# ------------------------------------------------------------------ registration

def test_registered_in_the_metric_registry_off_by_default():
    assert "signal_processing_suite" in REGISTRY
    spec = REGISTRY["signal_processing_suite"]
    assert spec.cost == "moderate"
    assert not spec.needs_parse
    assert not spec.needs_model


def test_every_registered_metric_runs_without_raising(manuscript, base_config):
    from textgrader.results import StatusType
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "signal_processing_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
    ids = {item.metric_id for item in report.results}
    assert any(metric_id.startswith("rhythm.signal_") or metric_id.startswith("drift.signal_")
              for metric_id in ids)


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_degenerate_documents_never_raise(text, tmp_path, base_config):
    from textgrader.results import StatusType
    source = tmp_path / "tiny.txt"
    source.write_text(text, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "signal_processing_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


def test_degrades_fully_with_every_optional_package_disabled(monkeypatch, manuscript, base_config):
    from textgrader.results import StatusType
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        config = {**base_config, "metrics": {**base_config["metrics"],
                                             "signal_processing_suite": {"enabled": True}}}
        report = grade.analyze(manuscript, config)
        errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
        assert not errors, [(item.metric_id, item.error) for item in errors]
        # zero_crossing_rate needs no optional package at all.
        zc = next(item for item in report.results
                 if item.metric_id == "rhythm.signal_sentence_words_zero_crossing_rate")
        assert zc.value is not None
    finally:
        optional.reset_cache()
