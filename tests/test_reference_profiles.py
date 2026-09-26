"""Task 24: multi-corpus, genre-aware reference profiles.

Covers the ``reference_profiles`` config block, ``grade.py``'s
``load_reference_profiles``/``apply_reference_profiles``, the
``reference_fit`` metric, and the ``corpus_builder`` dataset adapters that
fail gracefully without a licence/local copy.
"""

import json

import pytest

import grade
from textgrader.corpus import METRIC_DEFINITION_VERSION, build_profile, write_profile
from textgrader.document import TextProcessing
from textgrader.results import StatusType


def _write(tmp_path, name, profile):
    path = tmp_path / name
    write_profile(profile, path)
    return name


def _default_text_processing():
    # The exact fingerprint grade.py's own default DocumentAnalysis will
    # compute, so a synthetic profile matches unless a test deliberately
    # changes one field -- never a hand-typed guess at what "auto" resolves
    # to on this machine (see TextProcessing.fingerprint's segmenter_resolved).
    return TextProcessing.from_config(None).fingerprint()


def _synthetic_profile(values_by_key, **kwargs):
    """A hand-built, minimal profile: just enough shape for ``distribution()``
    and the Task 24 safeguards to accept it, with full control over exactly
    which numbers a metric is compared against."""

    profile = {
        "schema_version": 2, "corpus_name": kwargs.pop("corpus_name", "synthetic"),
        "comparison_unit": kwargs.pop("comparison_unit", "book"),
        "text_processing": kwargs.pop("text_processing", _default_text_processing()),
        "metric_definition_version": kwargs.pop("metric_definition_version",
                                                METRIC_DEFINITION_VERSION),
        "lexile_frequency_source": kwargs.pop("lexile_frequency_source", "none"),
        "metric_settings": kwargs.pop("metric_settings", {}),
        "book_count": len(next(iter(values_by_key.values()))),
        "books": [{"word_count": 5000} for _ in next(iter(values_by_key.values()))],
        "distributions": {key: {"values": values} for key, values in values_by_key.items()},
    }
    profile.update(kwargs)
    return profile


# --------------------------------------------------------------- backward compat

def test_report_without_reference_profiles_is_unchanged(manuscript, tmp_path, base_config,
                                                         corpus_dir):
    """A report built with no ``reference_profiles`` configured must be
    byte-for-byte identical to one built before this feature existed -- the
    non-negotiable backward-compatibility requirement."""

    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z")
    write_profile(profile, tmp_path / "profile.json")
    base = {**base_config, "corpus_profile": "profile.json"}
    without_key = grade.analyze(manuscript, base)
    with_empty_key = grade.analyze(manuscript, {**base, "reference_profiles": {}})
    assert json.dumps(without_key.to_dict(), sort_keys=True, default=str) == \
        json.dumps(with_empty_key.to_dict(), sort_keys=True, default=str)
    assert not any(item.metric_id.startswith("corpus.reference_profiles")
                  for item in without_key.results)
    assert not any((item.distribution or {}).get("reference_fits")
                  for item in without_key.results)


def test_reference_profiles_notes_key_is_ignored(manuscript, tmp_path, base_config, corpus_dir):
    """The documentation-only ``_notes`` key never becomes a profile alias."""

    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z")
    write_profile(profile, tmp_path / "profile.json")
    config = {**base_config, "corpus_profile": "profile.json",
             "reference_profiles": {"_notes": "see docs"}}
    report = grade.analyze(manuscript, config)
    assert not any(item.metric_id.startswith("corpus.reference_profiles")
                  for item in report.results)


# ------------------------------------------------------ two synthetic profiles

def test_two_profiles_give_the_same_metric_different_percentiles(manuscript, tmp_path,
                                                                  base_config):
    low = _synthetic_profile({"wps": [4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4, 5.6, 5.8]},
                             corpus_name="terse")
    high = _synthetic_profile({"wps": [20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0,
                                       24.5]}, corpus_name="ornate")
    config = {**base_config,
             "reference_profiles": {
                 "terse": {"path": _write(tmp_path, "terse.json", low), "label": "Terse corpus"},
                 "ornate": {"path": _write(tmp_path, "ornate.json", high),
                           "label": "Ornate corpus"},
             }}
    report = grade.analyze(manuscript, config)
    wps = next(item for item in report.results if item.metric_id == "prose.wps")
    fits = wps.distribution["reference_fits"]
    assert set(fits) == {"terse", "ornate"}
    assert fits["terse"]["percentile"] != fits["ornate"]["percentile"]
    # This manuscript's mean sentence length is well inside typical prose
    # (see conftest._prose), so it should read far higher against the corpus
    # of uniformly SHORT sentences than against the corpus of uniformly LONG
    # ones.
    assert fits["terse"]["percentile"] > fits["ornate"]["percentile"]


# ------------------------------------------------------------- safeguards

def test_incompatible_fingerprint_withholds_that_profile_only(manuscript, tmp_path, base_config):
    good = _synthetic_profile({"wps": list(range(8, 18))}, corpus_name="good")
    bad_processing = dict(_default_text_processing())
    bad_processing["segmenter"] = "builtin"
    bad = _synthetic_profile({"wps": list(range(8, 18))}, corpus_name="bad",
                             text_processing=bad_processing)
    config = {**base_config,
             "reference_profiles": {
                 "good": {"path": _write(tmp_path, "good.json", good)},
                 "bad": {"path": _write(tmp_path, "bad.json", bad)},
             }}
    report = grade.analyze(manuscript, config)
    notice = next(item for item in report.results
                 if item.metric_id == "corpus.reference_profiles.bad.text_processing")
    assert notice.status_type is StatusType.UNAVAILABLE
    assert "prepared differently" in notice.warning
    wps = next(item for item in report.results if item.metric_id == "prose.wps")
    fits = wps.distribution["reference_fits"]
    assert "good" in fits
    assert "bad" not in fits


def test_comparison_unit_mismatch_withholds_profile(manuscript, tmp_path, base_config):
    mismatched = _synthetic_profile({"wps": list(range(8, 18))}, comparison_unit="scene")
    config = {**base_config, "analysis": {**base_config["analysis"], "comparison_unit": "book"},
             "reference_profiles": {
                 "scenes": {"path": _write(tmp_path, "scenes.json", mismatched)}}}
    report = grade.analyze(manuscript, config)
    notice = next(item for item in report.results
                 if item.metric_id == "corpus.reference_profiles.scenes.comparison_unit")
    assert notice.status_type is StatusType.UNAVAILABLE
    assert "book" in notice.warning and "scene" in notice.warning
    wps = next(item for item in report.results if item.metric_id == "prose.wps")
    assert "scenes" not in (wps.distribution or {}).get("reference_fits", {})


def test_missing_reference_profile_fails_gracefully(manuscript, base_config):
    config = {**base_config,
             "reference_profiles": {"ghost": {"path": "does-not-exist.json"}}}
    report = grade.analyze(manuscript, config)
    notice = next(item for item in report.results if item.metric_id == "corpus.reference_profiles.ghost")
    assert notice.status_type is StatusType.UNAVAILABLE
    assert "not found" in notice.warning
    # The rest of the run is unaffected: a bad profile never crashes grading.
    assert any(item.metric_id == "prose.wps" for item in report.results)


def test_reference_profiles_needs_a_path(manuscript, base_config):
    config = {**base_config, "reference_profiles": {"broken": {"label": "no path given"}}}
    report = grade.analyze(manuscript, config)
    notice = next(item for item in report.results if item.metric_id == "corpus.reference_profiles.broken")
    assert notice.status_type is StatusType.UNAVAILABLE
    assert "path" in notice.warning


def test_multiple_profiles_coexist_in_one_report(manuscript, tmp_path, base_config):
    profiles = {
        alias: _synthetic_profile({"wps": [float(8 + index) for index in range(10)]},
                                  corpus_name=alias)
        for alias in ("news", "fiction", "poetry")
    }
    config = {**base_config,
             "reference_profiles": {
                 alias: {"path": _write(tmp_path, f"{alias}.json", profile),
                        "genre": alias}
                 for alias, profile in profiles.items()}}
    report = grade.analyze(manuscript, config)
    for alias in profiles:
        info = next(item for item in report.results
                   if item.metric_id == f"corpus.reference_profiles.{alias}")
        assert info.details[0]["genre"] == alias
    wps = next(item for item in report.results if item.metric_id == "prose.wps")
    fits = wps.distribution["reference_fits"]
    assert set(fits) == set(profiles)


# ---------------------------------------------------------------- reference_fit

def _with_reference_fit(base_config, **extra_metric_options):
    return {**base_config,
           "metrics": {**base_config["metrics"],
                      "reference_fit": {"enabled": True, **extra_metric_options}}}


def test_reference_fit_vector_against_two_profiles(manuscript, tmp_path, base_config,
                                                    corpus_dir):
    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z")
    write_profile(profile, tmp_path / "primary.json")
    config = _with_reference_fit(base_config, min_corpus_documents=8)
    config["corpus_profile"] = "primary.json"
    config["reference_profiles"] = {
        "twin": {"path": "primary.json", "label": "Same corpus again"},
    }
    report = grade.analyze(manuscript, config)
    ids = {item.metric_id: item for item in report.results}
    assert "style.reference_fit_primary_distance_mean_abs_z" in ids
    assert ids["style.reference_fit_primary_distance_mean_abs_z"].value is not None
    assert "style.reference_fit_twin_median_abs_z" in ids
    assert ids["style.reference_fit_primary_coverage"].value > 0
    assert "style.reference_fit_primary_loo_calibration" in ids
    # Two usable profiles: cross-profile findings must appear.
    assert "style.reference_fit_nearest_profile" in ids
    assert "style.reference_fit_second_nearest_profile" in ids
    assert "style.reference_fit_nearest_margin" in ids
    assert "style.reference_fit_disagreement" in ids
    assert "style.reference_fit_vector" in ids
    vector = ids["style.reference_fit_vector"].details
    assert {row["alias"] for row in vector} == {"primary", "twin"}
    # The SAME corpus profile compared against itself must read as an almost
    # exact fit -- the two aliases' distances should closely agree.
    primary_distance = ids["style.reference_fit_primary_distance_mean_abs_z"].value
    twin_distance = ids["style.reference_fit_twin_distance_mean_abs_z"].value
    assert primary_distance == pytest.approx(twin_distance, rel=1e-9)


def test_reference_fit_is_off_by_default(manuscript, base_config, corpus_dir, tmp_path):
    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z")
    write_profile(profile, tmp_path / "profile.json")
    report = grade.analyze(manuscript, {**base_config, "corpus_profile": "profile.json"})
    assert not any(item.metric_id.startswith("style.reference_fit_") for item in report.results)


def test_reference_fit_reports_unavailable_with_no_profile_at_all(manuscript, base_config):
    report = grade.analyze(manuscript, _with_reference_fit(base_config))
    vector = next(item for item in report.results if item.metric_id == "style.reference_fit_vector")
    assert vector.status_type is StatusType.UNAVAILABLE
    assert "no reference profile is configured" in vector.warning


def test_reference_fit_withholds_a_too_small_profile(manuscript, base_config, tmp_path):
    tiny = _synthetic_profile({"fk": [8.0, 9.0, 10.0]})
    tiny["books"] = [{"fk": value} for value in (8.0, 9.0, 10.0)]
    config = _with_reference_fit(base_config, min_corpus_documents=8)
    config["corpus_profile"] = _write(tmp_path, "tiny.json", tiny)
    report = grade.analyze(manuscript, config)
    coverage = next(item for item in report.results
                   if item.metric_id == "style.reference_fit_primary_coverage")
    assert coverage.value is None
    assert "reference book" in coverage.warning


# ------------------------------------------------------------- leave-one-out

def test_loo_calibration_on_a_small_hand_built_profile():
    from textgrader.metrics import reference_fit as rf

    books = [{"fk": value, "wps": value * 1.5} for value in
            (8.0, 8.2, 7.9, 8.1, 8.0, 7.8, 8.3, 7.9, 8.0, 8.1)]
    result = rf._loo_calibration("t", "Test profile", books, ("fk", "wps"), config=None)
    assert result["value"] is not None
    assert result["distribution"]["quality"] == "adequate"
    assert result["distribution"]["held_out_books"] >= 3


def test_loo_calibration_says_so_when_poorly_calibrated():
    from textgrader.metrics import reference_fit as rf

    # Wildly scattered books: a genuine member of this "corpus" does not
    # resemble the rest of it, so calibration must say so.
    values = [0.0, 50.0, 1.0, 49.0, 2.0, 48.0, 3.0, 47.0, 4.0, 46.0]
    books = [{"fk": value} for value in values]
    result = rf._loo_calibration("t", "Scattered profile", books, ("fk",), config=None)
    assert result["value"] is not None
    assert result["distribution"]["quality"] == "poor"
    assert "not tightly self-consistent" in result["warning"]


def test_loo_calibration_too_few_books_is_explicit():
    from textgrader.metrics import reference_fit as rf

    result = rf._loo_calibration("t", "Tiny", [{"fk": 8.0}, {"fk": 9.0}], ("fk",), config=None)
    assert result["value"] is None
    assert "leave-one-out" in result["warning"]


# ---------------------------------------------------------- dataset adapters

def test_licensed_corpus_recipe_fails_gracefully_without_local_files(tmp_path):
    from corpus_builder.dataset_adapters import AdapterError, require_local_licensed_corpus

    with pytest.raises(AdapterError) as excinfo:
        require_local_licensed_corpus("penn_treebank", tmp_path / "does-not-exist")
    assert "LDC" in str(excinfo.value)


def test_unknown_licensed_corpus_name_fails_gracefully(tmp_path):
    from corpus_builder.dataset_adapters import AdapterError, require_local_licensed_corpus

    with pytest.raises(AdapterError):
        require_local_licensed_corpus("not-a-real-corpus", tmp_path)


def test_missing_nltk_data_reports_setup_instructions(monkeypatch):
    """A Brown/Reuters corpus that has never been downloaded must fail with
    the exact remedy, never a bare traceback."""

    from corpus_builder import dataset_adapters as da

    nltk = pytest.importorskip("nltk")
    if "brown" in _downloaded_corpora(nltk):
        pytest.skip("the 'brown' NLTK corpus is already downloaded in this environment")
    with pytest.raises(da.AdapterError) as excinfo:
        da.build_brown_corpus("unused-output")
    assert "nltk.download" in str(excinfo.value)


def _downloaded_corpora(nltk):
    try:
        nltk.corpus.brown.categories()
        return {"brown"}
    except LookupError:
        return set()
