"""Every optional metric must be off by default, independent, and honest."""

import json

import pytest

import grade
from textgrader import optional
from textgrader.corpus import build_profile
from textgrader.metrics import REGISTRY
from textgrader.results import Action, StatusType

PREFIXES = {"style", "nlp", "rhythm", "syntax", "lexical", "semantic",
            "dialogue", "discourse", "pov", "punct", "drift", "repetition"}


def test_all_are_off_unless_enabled(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text("He walked home. He walked home.", encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results
                if item.metric_id.split(".")[0] in PREFIXES]


def test_enabled_metrics_are_structured_and_independent(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text("He walked to the old house. He walked to the old house.",
                      encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "repeated_ngrams": {"enabled": True},
                                         "mattr": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(source, config).results}
    assert "style.repeated_ngrams" in ids
    assert "style.mattr" in ids


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_metric_runs_without_raising(name, manuscript, base_config):
    """The registry is the public surface; nothing in it may crash a run."""

    config = {**base_config, "metrics": {**base_config["metrics"],
                                         name: {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results
              if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_every_registered_metric_survives_a_degenerate_document(text, tmp_path, base_config):
    source = tmp_path / "tiny.txt"
    source.write_text(text, encoding="utf-8")
    config = {**base_config,
              "metrics": {**base_config["metrics"],
                          **{name: {"enabled": True} for name in REGISTRY}}}
    report = grade.analyze(source, config)
    errors = [item for item in report.results
              if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


def test_profile_contains_comparison_distributions(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text("One short line. Another longer line follows it.", encoding="utf-8")
    profile = build_profile([source], built_at="2026-01-01T00:00:00Z")
    assert "style.mattr" in profile["distributions"]
    assert len(profile["feature_profiles"]["function_words"]) == 1
    assert profile["comparison_unit"] == "book"
    assert profile["text_processing"]["strip_gutenberg"] is True


def test_profile_records_the_options_it_used(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text("one two three four five six.", encoding="utf-8")
    profile = build_profile([source], metrics={"mattr": {"enabled": True, "window": 3}})
    assert profile["metric_settings"]["mattr"]["window"] == 3
    assert profile["distributions"]["style.mattr"]["values"][0] == pytest.approx(100.0)


def test_mismatched_metric_options_are_not_compared(tmp_path, base_config):
    source = tmp_path / "book.txt"
    source.write_text("one two three one two three. " * 60, encoding="utf-8")
    profile = build_profile([source], metrics={"mattr": {"window": 100}})
    (tmp_path / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    report = grade.analyze(source, {**base_config, "corpus_profile": "profile.json",
                                    "metrics": {**base_config["metrics"],
                                                "mattr": {"enabled": True, "window": 3}}})
    item = next(item for item in report.results if item.metric_id == "style.mattr")
    assert item.action is Action.INSUFFICIENT_DATA
    assert "different" in item.warning


def test_nlp_metric_without_a_model_is_a_visible_warning(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text("The door was opened.", encoding="utf-8")
    report = grade.analyze(source, {**base_config,
                                    "metrics": {**base_config["metrics"],
                                                "passive_voice": {"enabled": True}},
                                    "nlp": {"model": "certainly_missing_model"}})
    item = next(item for item in report.results if item.metric_id == "nlp.passive_voice")
    assert item.value is None
    assert item.warning and "certainly_missing_model" in item.warning


def test_a_missing_optional_package_degrades_one_metric(monkeypatch, manuscript, base_config):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        config = {**base_config,
                  "metrics": {**base_config["metrics"],
                              **{name: {"enabled": True} for name in REGISTRY}}}
        report = grade.analyze(manuscript, config)
        errors = [item for item in report.results
                  if item.status_type is StatusType.INTERNAL_ERROR]
        assert not errors, [(item.metric_id, item.error) for item in errors]
        # Something still got measured: the dependency-free core is unaffected.
        assert any(item.value is not None and item.metric_id.startswith("prose.")
                   for item in report.results)
    finally:
        optional.reset_cache()
