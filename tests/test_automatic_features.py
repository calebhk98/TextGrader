"""Tests for automatic_features: id sanitizing, synthetic-signal behaviour,
determinism, output/runtime caps, graceful degradation, corpus-profile size
and the four-book default-mode variation check the task asks for.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, automatic_features as af


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


# ------------------------------------------------------------------- id sanitizing

def test_sanitize_feature_name_strips_tsfresh_punctuation():
    raw = 'agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"'
    sanitized = af.sanitize_feature_name(raw)
    assert sanitized == "agg_linear_trend_attr_rvalue_chunk_len_10_f_agg_mean"
    assert "__" not in sanitized
    assert '"' not in sanitized


def test_sanitize_feature_name_handles_catch22_camel_case():
    assert af.sanitize_feature_name("DN_HistogramMode_5") == "dn_histogrammode_5"


def test_sanitize_feature_name_is_deterministic():
    raw = 'fft_coefficient__attr_"abs"__coeff_3'
    assert af.sanitize_feature_name(raw) == af.sanitize_feature_name(raw)


def test_sanitize_feature_name_truncates_long_names_with_a_stable_hash():
    raw = "x" * 200
    sanitized = af.sanitize_feature_name(raw)
    assert len(sanitized) <= 80
    assert af.sanitize_feature_name(raw) == sanitized  # stable across calls


def test_dedupe_sanitized_ids_is_stable_and_collision_free_on_a_synthetic_clash():
    # Two distinct raw names that sanitize to the identical string.
    raw_names = ["Feature-One", "Feature!One", "Feature_One"]
    mapping = af.dedupe_sanitized_ids(raw_names)
    assert len(set(mapping.values())) == len(raw_names)  # no collisions survive
    # Re-running with the same set gives the exact same mapping.
    assert af.dedupe_sanitized_ids(raw_names) == mapping
    # Sorted-first raw name keeps the plain spelling.
    assert mapping[sorted(raw_names)[0]] == "feature_one"


def test_dedupe_sanitized_ids_over_real_tsfresh_comprehensive_columns_is_collision_free():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    from tsfresh.feature_extraction import extract_features
    from tsfresh.feature_extraction import settings as tsfresh_settings
    import pandas as pd
    values = [math.sin(i / 3.0) + 0.01 * i for i in range(120)]
    frame = pd.DataFrame({"id": [0] * len(values), "time": range(len(values)), "value": values})
    fc_parameters = tsfresh_settings.ComprehensiveFCParameters()
    extracted = extract_features(frame, column_id="id", column_sort="time", column_value="value",
                                 default_fc_parameters=fc_parameters, disable_progressbar=True,
                                 n_jobs=0)
    raw_columns = [c.split("__", 1)[1] if "__" in c else c for c in extracted.columns]
    assert len(raw_columns) > 400  # sanity: this really is the big preset
    mapping = af.dedupe_sanitized_ids(raw_columns)
    assert len(set(mapping.values())) == len(set(raw_columns))
    # Every id is a legal, bounded, safe metric-id fragment.
    for sanitized in mapping.values():
        assert sanitized == sanitized.lower()
        assert all(ch.isalnum() or ch == "_" for ch in sanitized)
        assert len(sanitized) <= 80
    # Stable: running it again reproduces the identical mapping.
    assert af.dedupe_sanitized_ids(raw_columns) == mapping


# ---------------------------------------------------------------- stats extractor

def _constant(n=40, value=5.0):
    return [value] * n


def _trend(n=40):
    return [float(i) for i in range(n)]


def _alternating(n=40):
    return [0.0, 10.0] * (n // 2)


def _periodic(n=120, period=10, amplitude=10.0):
    return [amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


def _shuffled(n=200, seed=3):
    values = list(range(n))
    random.Random(seed).shuffle(values)
    return [float(v) for v in values]


def test_stats_quantiles_and_range_on_a_known_uniform_ramp():
    values = _trend(101)  # 0..100
    out = af._compute_stats(values)
    assert out["quantile_p05"][0] == pytest.approx(5.0, abs=0.5)
    assert out["quantile_p95"][0] == pytest.approx(95.0, abs=0.5)
    assert out["range"][0] == pytest.approx(100.0)
    assert out["iqr"][0] == pytest.approx(50.0, abs=1.0)


def test_stats_iqr_and_mad_are_zero_for_a_constant_series():
    out = af._compute_stats(_constant())
    assert out["iqr"][0] == pytest.approx(0.0)
    assert out["mad"][0] == pytest.approx(0.0)
    assert out["range"][0] == pytest.approx(0.0)


def test_stats_cv_is_undefined_when_mean_is_zero():
    out = af._compute_stats(_alternating(40, ))  # not zero mean; construct a zero-mean series
    values = [-5.0, 5.0] * 20
    out = af._compute_stats(values)
    assert out["cv"][0] is None
    assert "zero" in out["cv"][1]


def test_stats_skewness_zero_for_symmetric_alternation_and_kurtosis_defined():
    out = af._compute_stats(_alternating(60))
    assert out["skewness"][0] == pytest.approx(0.0, abs=1e-9)
    assert out["kurtosis"][0] is not None


def test_stats_skewness_and_kurtosis_need_a_minimum_of_points():
    out = af._compute_stats([1.0, 2.0])
    assert out["skewness"][0] is None
    assert "at least 3" in out["skewness"][1]
    out3 = af._compute_stats([1.0, 2.0, 3.0])
    assert out3["kurtosis"][0] is None
    assert "at least 4" in out3["kurtosis"][1]


def test_stats_on_periodic_and_shuffled_sequences_produce_finite_numbers():
    for values in (_periodic(), _shuffled()):
        out = af._compute_stats(values)
        for name in af.STATS_FEATURES:
            value, _ = out[name]
            assert value is None or math.isfinite(value)


# ---------------------------------------------------------------- through measure()

def test_minimal_mode_emits_exactly_stats_over_the_default_sequences():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={})
    real = [f for f in findings if f["metric_id"] != "style.autofeature_manifest"]
    assert len(real) == len(af.DEFAULT_SEQUENCES) * len(af.STATS_FEATURES)
    assert all("_stats_" in f["metric_id"] for f in real)
    assert all(f["metric_id"].startswith("style.autofeature_") for f in real)


def test_minimal_is_the_default_mode_and_matches_registry_defaults():
    assert af.DEFAULT_MODE == "minimal"
    spec = REGISTRY["automatic_feature_generation"]
    assert spec.defaults["mode"] == "minimal"
    # null means "derive from mode" -- see automatic_features.py's module
    # docstring and this test module's test_mode_alone_changes_behaviour_
    # even_when_other_keys_are_merged_in_from_defaults.
    assert spec.defaults["extractors"] is None


def test_mode_alone_changes_behaviour_even_when_other_keys_are_merged_in_from_defaults():
    # Reproduces exactly how corpus.py/grade.py build a metric's options:
    # dict(spec.defaults) first, then override only the keys a caller's own
    # config names. A caller who sets only "mode" must still get that mode's
    # own extractors/max_findings, not the ones spelled out in defaults --
    # this is what makes "extractors"/"tsfresh_feature_set"/"max_findings"
    # default to null (see the module docstring) rather than a fixed value.
    spec = REGISTRY["automatic_feature_generation"]
    analysis = _analysis(_long_text())
    minimal_options = dict(spec.defaults)
    minimal_options.update({k: v for k, v in {"mode": "minimal"}.items() if k != "enabled"})
    standard_options = dict(spec.defaults)
    standard_options.update({k: v for k, v in {"mode": "standard"}.items() if k != "enabled"})
    minimal_findings = af.measure(analysis, config=minimal_options)
    standard_findings = af.measure(analysis, config=standard_options)
    minimal_real = [f for f in minimal_findings if f["metric_id"] != "style.autofeature_manifest"]
    standard_real = [f for f in standard_findings if f["metric_id"] != "style.autofeature_manifest"]
    assert len(standard_real) > len(minimal_real)
    assert any("_catch22_" in f["metric_id"] for f in standard_real)
    assert not any("_catch22_" in f["metric_id"] for f in minimal_real)


def test_standard_mode_adds_catch22_and_tsfresh_minimal_ids():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"mode": "standard"})
    ids = {f["metric_id"] for f in findings}
    assert "style.autofeature_sentence_words_catch22_dn_histogrammode_5" in ids
    assert "style.autofeature_sentence_words_tsfresh_mean" in ids


def test_comprehensive_mode_is_a_strict_superset_of_standard_before_truncation():
    analysis = _analysis(_long_text())
    standard = {f["metric_id"] for f in af.measure(analysis, config={
        "mode": "standard", "max_findings": 100000})}
    comprehensive = {f["metric_id"] for f in af.measure(analysis, config={
        "mode": "comprehensive", "max_findings": 100000})}
    standard.discard("style.autofeature_manifest")
    comprehensive.discard("style.autofeature_manifest")
    assert standard <= comprehensive


def test_option_mismatch_between_modes_is_recorded_in_every_finding():
    analysis = _analysis(_long_text())
    minimal = af.measure(analysis, config={"mode": "minimal"})
    standard = af.measure(analysis, config={"mode": "standard"})
    real_minimal = next(f for f in minimal if f["metric_id"] != "style.autofeature_manifest")
    real_standard = next(f for f in standard if f["metric_id"] != "style.autofeature_manifest")
    assert real_minimal["distribution"]["mode"] == "minimal"
    assert real_standard["distribution"]["mode"] == "standard"
    manifest_minimal = next(f for f in minimal if f["metric_id"] == "style.autofeature_manifest")
    manifest_standard = next(f for f in standard if f["metric_id"] == "style.autofeature_manifest")
    assert manifest_minimal["distribution"]["mode"] != manifest_standard["distribution"]["mode"]


def test_unknown_mode_is_reported_not_raised():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"mode": "not_a_real_mode"})
    assert len(findings) == 1
    assert findings[0]["metric_id"] == "style.autofeature_config"
    assert "unknown mode" in findings[0]["warning"]


def test_unknown_sequence_and_extractor_are_reported_not_raised():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={
        "sequences": ["sentence_words", "not_a_sequence"],
        "extractors": ["stats", "not_an_extractor"]})
    ids = {f["metric_id"] for f in findings}
    assert any(mid.startswith("style.autofeature_sentence_words_stats_") for mid in ids)
    assert "style.autofeature_config" in ids


def test_completely_invalid_configuration_does_not_crash():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"sequences": ["nope"], "extractors": ["nope"]})
    assert len(findings) == 1
    assert findings[0]["value"] is None


# ---------------------------------------------------------- overlap-with-timeseries

def test_catch22_findings_name_the_overlapping_timeseries_suite_id():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"mode": "standard"})
    catch22_finding = next(f for f in findings
                           if f["metric_id"] == "style.autofeature_sentence_words_catch22_dn_histogrammode_5")
    assert catch22_finding["distribution"]["overlaps_existing_metric_id"] == (
        "rhythm.timeseries_sentence_words_catch22_histogram_mode_5bin")


def test_tsfresh_minimal_findings_name_the_overlapping_timeseries_suite_id():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"mode": "standard"})
    tsfresh_finding = next(f for f in findings
                          if f["metric_id"] == "style.autofeature_sentence_words_tsfresh_mean")
    assert tsfresh_finding["distribution"]["overlaps_existing_metric_id"] == (
        "rhythm.timeseries_sentence_words_tsfresh")


def test_stats_group_flags_related_features_without_deleting_any():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={})
    iqr = next(f for f in findings
              if f["metric_id"] == "style.autofeature_sentence_words_stats_iqr")
    assert "related_features" in iqr["distribution"]
    assert any("quantile_p10" in rid for rid in iqr["distribution"]["related_features"])
    # Every related id named is itself present as a real, undeleted finding.
    ids = {f["metric_id"] for f in findings}
    for related_id in iqr["distribution"]["related_features"]:
        assert related_id in ids


# --------------------------------------------------------------------- determinism

def test_comprehensive_mode_is_bit_identical_across_repeated_runs():
    if not optional.have("tsfresh") or not optional.have("pycatch22"):
        pytest.skip("tsfresh/pycatch22 not installed in this environment")
    text = _long_text(paragraphs=90, seed=7)

    def run():
        analysis = _analysis(text)
        findings = af.measure(analysis, config={"mode": "comprehensive"})
        return [(f["metric_id"], f["value"]) for f in findings]

    first, second = run(), run()
    assert first == second


def test_standard_mode_is_bit_identical_across_repeated_runs():
    text = _long_text(paragraphs=60, seed=9)

    def run():
        return [(f["metric_id"], f["value"]) for f in af.measure(_analysis(text), config={"mode": "standard"})]

    assert run() == run()


# ------------------------------------------------------------------------ caps

def test_max_findings_caps_output_deterministically():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"mode": "standard", "max_findings": 5})
    real = [f for f in findings if f["metric_id"] not in
           ("style.autofeature_manifest", "style.autofeature_truncated")]
    truncated = [f for f in findings if f["metric_id"] == "style.autofeature_truncated"]
    assert len(real) == 5
    assert len(truncated) == 1
    assert "raise" in truncated[0]["warning"]
    # Deterministic selection: repeating gives the identical five ids.
    again = af.measure(analysis, config={"mode": "standard", "max_findings": 5})
    real_again = [f["metric_id"] for f in again if f["metric_id"] not in
                 ("style.autofeature_manifest", "style.autofeature_truncated")]
    assert [f["metric_id"] for f in real] == real_again


def test_allow_pattern_restricts_output_to_matching_feature_ids():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"allow_pattern": "quantile"})
    real = [f for f in findings if f["metric_id"] != "style.autofeature_manifest"]
    assert real
    assert all("quantile" in f["metric_id"] for f in real)


def test_deny_pattern_excludes_matching_feature_ids():
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={"deny_pattern": "quantile"})
    real = [f for f in findings if f["metric_id"] != "style.autofeature_manifest"]
    assert real
    assert not any("quantile" in f["metric_id"] for f in real)


def test_min_length_reports_insufficient_data_not_a_crash():
    analysis = _analysis("One short sentence. Another short one.")
    findings = af.measure(analysis, config={"mode": "standard"})
    catch22_findings = [f for f in findings if "_catch22_" in f["metric_id"]]
    assert catch22_findings
    assert all(f["value"] is None for f in catch22_findings)
    assert all("insufficient data" in (f["warning"] or "") for f in catch22_findings)


# ------------------------------------------------------------- downsampling

def test_long_sequences_are_deterministically_downsampled_for_catch22_and_tsfresh():
    if not optional.have("pycatch22"):
        pytest.skip("pycatch22 not installed in this environment")
    text = _long_text(paragraphs=1200, seed=11)
    analysis = _analysis(text)
    findings = af.measure(analysis, config={
        "sequences": ["sentence_words"], "extractors": ["catch22"],
        "max_sequence_length": 50, "max_findings": 1000})
    real = [f for f in findings if f["metric_id"] != "style.autofeature_manifest"]
    assert real
    for f in real:
        if f["value"] is not None:
            assert f["distribution"]["sequence_length_used"] <= 50
            assert f["distribution"]["downsample_stride"] >= 1


# ---------------------------------------------------------- crash/NaN safety

def test_a_raising_stats_computation_does_not_crash_the_suite(monkeypatch):
    original = af._compute_stats
    calls = {"n": 0}

    def flaky(values):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("synthetic failure")
        return original(values)

    monkeypatch.setattr(af, "_compute_stats", flaky)
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={})
    # Did not raise, and the failing feature is reported as unavailable, not silently dropped.
    real = [f for f in findings if f["metric_id"] != "style.autofeature_manifest"]
    assert real
    failed = [f for f in real if f["value"] is None and f["warning"] and "failed" in f["warning"]]
    assert failed


def test_a_nan_producing_extractor_reports_unavailable_not_a_number(monkeypatch):
    def fake_stats(values):
        return {name: (float("nan"), None) for name in af.STATS_FEATURES}

    monkeypatch.setattr(af, "_compute_stats", fake_stats)
    analysis = _analysis(_long_text())
    findings = af.measure(analysis, config={})
    real = [f for f in findings if f["metric_id"] != "style.autofeature_manifest"]
    assert real
    for f in real:
        assert f["value"] is None or math.isfinite(f["value"])


# ------------------------------------------------------------ full feature vector

def test_full_feature_vector_is_attached_only_when_explicitly_requested():
    if not optional.have("tsfresh"):
        pytest.skip("tsfresh not installed in this environment")
    analysis = _analysis(_long_text())
    without = af.measure(analysis, config={
        "mode": "comprehensive", "sequences": ["sentence_words"], "max_findings": 5})
    manifest_without = next(f for f in without if f["metric_id"] == "style.autofeature_manifest")
    assert "full_feature_vectors" not in manifest_without["distribution"]

    with_vector = af.measure(analysis, config={
        "mode": "comprehensive", "sequences": ["sentence_words"], "max_findings": 5,
        "include_full_feature_vector": True, "full_feature_vector_max_features": 10})
    manifest_with = next(f for f in with_vector if f["metric_id"] == "style.autofeature_manifest")
    assert "full_feature_vectors" in manifest_with["distribution"]
    vector = manifest_with["distribution"]["full_feature_vectors"]["sentence_words"]
    assert 0 < len(vector) <= 10


# --------------------------------------------------------- variation over real books

# Real-book checks run only when TEXTGRADER_REAL_CORPUS names a folder of
# corpus .txt files (e.g. one built with corpus_builder); they are slow
# benchmarks rather than unit tests, so every other run skips them.
_REAL_CORPUS = os.environ.get("TEXTGRADER_REAL_CORPUS", "")
CORPUS_DIR = Path(_REAL_CORPUS) if _REAL_CORPUS else Path("/nonexistent-textgrader-real-corpus")


def _corpus_books(n=4):
    if not CORPUS_DIR.is_dir():
        return []
    files = sorted(CORPUS_DIR.glob("*.txt"))[:n]
    return files


def test_default_mode_findings_vary_across_four_real_books():
    files = _corpus_books(4)
    if len(files) < 4:
        pytest.skip("reference corpus not available in this environment")
    per_book_values: list[dict[str, float | None]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Cap to keep this test fast; the default sequences/features do not
        # need a whole novel to vary.
        analysis = _analysis(text[:400_000])
        findings = af.measure(analysis, config={})
        per_book_values.append({f["metric_id"]: f["value"] for f in findings
                                if f["metric_id"] != "style.autofeature_manifest"})
    all_ids = set(per_book_values[0])
    for row in per_book_values[1:]:
        all_ids &= set(row)
    identical = []
    for metric_id in sorted(all_ids):
        values = [row[metric_id] for row in per_book_values]
        if len(set(values)) == 1:
            identical.append(metric_id)
    fraction = len(identical) / len(all_ids) if all_ids else 1.0
    # Report what's identical (for the task write-up) and require that the
    # overwhelming majority of ids genuinely differ across four different books.
    print(f"identical across all 4 books: {identical} ({fraction:.1%} of {len(all_ids)})")
    assert fraction < 0.2, identical


# ------------------------------------------------------------------ profile size

def test_measured_profile_bytes_per_mode_for_five_books(tmp_path):
    files = _corpus_books(5)
    if len(files) < 5:
        pytest.skip("reference corpus not available in this environment")
    from textgrader import corpus as corpus_module

    inputs = [str(path) for path in files]

    def build(mode):
        profile = corpus_module.build_profile(
            inputs, metrics={"automatic_feature_generation": {"enabled": True, "mode": mode}},
            metric_selection="enabled")
        return len(json.dumps(profile, sort_keys=True))

    baseline = corpus_module.build_profile(
        inputs, metrics={"automatic_feature_generation": {"enabled": False}},
        metric_selection="enabled")
    baseline_bytes = len(json.dumps(baseline, sort_keys=True))
    results = {}
    for mode in ("minimal", "standard"):
        results[mode] = build(mode) - baseline_bytes
    # 50-book extrapolation: the per-book contribution (books[] rows and the
    # pooled distribution's "values" list) scales roughly linearly in book
    # count; the small, mode-independent per-id summary overhead does not.
    # Reported, not asserted tightly, since real prose lengths vary.
    scale = 50 / 5
    print("measured added bytes for 5 books:", results)
    print("extrapolated (x10, linear-in-books approximation) for 50 books:",
         {mode: int(delta * scale) for mode, delta in results.items()})
    assert results["minimal"] > 0
    assert results["standard"] > results["minimal"]
    # The default mode must stay a small fraction of the 10.9MB reference profile.
    assert results["minimal"] * scale < 2_000_000  # well under 2MB added for 50 books


# --------------------------------------------------------------- repository contract

def test_registered_in_the_metric_registry_off_by_default():
    assert "automatic_feature_generation" in REGISTRY
    spec = REGISTRY["automatic_feature_generation"]
    assert spec.module == "automatic_features"
    assert spec.cost == "moderate"
    assert not spec.needs_parse
    assert not spec.needs_model


def test_registry_defaults_and_config_json_stay_mirrored():
    spec = REGISTRY["automatic_feature_generation"]
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    configured = config["metrics"]["automatic_feature_generation"]
    for key in spec.defaults:
        assert configured[key] == spec.defaults[key], key


def test_off_by_default_through_grade(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(_long_text(paragraphs=10), encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results if "autofeature" in item.metric_id]


def test_enabled_through_grade_reports_findings_with_the_right_prefix(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(_long_text(paragraphs=60), encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "automatic_feature_generation": {"enabled": True}}}
    report = grade.analyze(source, config)
    ids = [item.metric_id for item in report.results if "autofeature" in item.metric_id]
    assert ids
    assert all(mid.startswith("style.autofeature_") for mid in ids)


def test_every_registered_metric_runs_without_raising(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "automatic_feature_generation": {
                                             "enabled": True, "mode": "standard"}}}
    report = grade.analyze(manuscript, config)
    from textgrader.results import StatusType
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


def test_degrades_fully_with_every_optional_package_disabled(monkeypatch, manuscript, base_config):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        config = {**base_config, "metrics": {**base_config["metrics"],
                                             "automatic_feature_generation": {
                                                 "enabled": True, "mode": "comprehensive"}}}
        report = grade.analyze(manuscript, config)
        from textgrader.results import StatusType
        errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
        assert not errors, [(item.metric_id, item.error) for item in errors]
        stats_results = [item for item in report.results
                         if "_stats_" in item.metric_id]
        assert stats_results
        assert all(item.value is not None for item in stats_results)
    finally:
        optional.reset_cache()


# ---------------------------------------------------------- performance (novel)

NOVEL_PATH = CORPUS_DIR / "gutenberg-514-little-women.txt"


def test_full_novel_runtime_is_reported_for_each_mode(capsys):
    if not NOVEL_PATH.is_file():
        pytest.skip("gutenberg-514-little-women.txt not available in this environment")
    import time
    import resource
    text = NOVEL_PATH.read_text(encoding="utf-8", errors="replace")
    for mode in ("minimal", "standard", "comprehensive"):
        # A fresh DocumentAnalysis per mode: the first measure() call on any
        # document always pays for tokenizing/splitting the whole novel once
        # (cached afterwards on analysis._shared), which would otherwise make
        # whichever mode runs first look artificially slow and every later
        # mode look artificially fast.
        analysis = _analysis(text)
        start = time.perf_counter()
        findings = af.measure(analysis, config={"mode": mode})
        elapsed = time.perf_counter() - start
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"automatic_features mode={mode}: {elapsed:.2f}s, {len(findings)} findings, "
             f"peak RSS so far {peak_kb / 1024:.0f} MB")
        assert elapsed < 180.0  # generous ceiling; the task report states the measured figure


def test_syllable_stress_comprehensive_mode_is_bounded_by_downsampling():
    # syllable_stress over a whole novel is roughly 200,000-250,000 points --
    # exactly the case the task warns tsfresh comprehensive can be very slow
    # on. This proves the deterministic downsample keeps it fast rather than
    # asserting a specific number of seconds (which would be flaky across
    # machines): every finding's sequence_length_used must stay at or below
    # max_sequence_length no matter how long the real sequence is.
    if not NOVEL_PATH.is_file():
        pytest.skip("gutenberg-514-little-women.txt not available in this environment")
    if not optional.have("pronouncing"):
        pytest.skip("pronouncing not installed in this environment")
    import time
    text = NOVEL_PATH.read_text(encoding="utf-8", errors="replace")
    analysis = _analysis(text)
    from textgrader import sequences as seq_module
    sequence = seq_module.get_sequence(analysis, "syllable_stress")
    if sequence.length < 50_000:
        pytest.skip("syllable_stress sequence unexpectedly short in this environment")
    start = time.perf_counter()
    findings = af.measure(analysis, config={
        "mode": "comprehensive", "sequences": ["syllable_stress"], "max_findings": 1000})
    elapsed = time.perf_counter() - start
    real = [f for f in findings if f["metric_id"] != "style.autofeature_manifest" and f["value"] is not None
           and "_stats_" not in f["metric_id"]]
    assert real
    for f in real:
        assert f["distribution"]["sequence_length_used"] <= af.DEFAULT_MAX_SEQUENCE_LENGTH
    print(f"syllable_stress ({sequence.length} points) comprehensive mode: {elapsed:.2f}s, "
         f"downsampled to {real[0]['distribution']['sequence_length_used']} points")
    assert elapsed < 60.0
