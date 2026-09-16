"""The structured-report contract: accounting, toggles and failure visibility."""

import json
import sys
from pathlib import Path

import pytest

import grade
from textgrader.results import Action, StatusType

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())


def test_shipped_defaults_favour_the_generic_metrics():
    # A public tool should not ship with one author's project reports enabled
    # and every generic measurement switched off.
    assert all(CONFIG["metrics"][name] is False for name in grade.BUNDLED_MEASURES)
    assert CONFIG["metrics"]["mattr"]["enabled"] is True
    assert CONFIG["metrics"]["repeated_ngrams"]["enabled"] is True


def test_new_metric_families_ship_off_but_present():
    for name in ("sentence_length_autocorrelation", "dependency_distance", "mtld",
                 "adjacent_sentence_similarity", "dialogue_channels",
                 "hedges_boosters", "narration_pov", "punctuation_profile",
                 "chapter_zscores"):
        assert CONFIG["metrics"][name]["enabled"] is False, name


def test_one_metrics_map_controls_both_kinds_of_measure(tmp_path, base_config):
    source = tmp_path / "story.md"
    source.write_text("There were 12 birds and 3 trees.", encoding="utf-8")
    off = grade.analyze(source, base_config)
    assert not any(item.metric_id.startswith("measure.") for item in off.results)
    on = grade.analyze(source, {**base_config, "metrics": {
        **base_config["metrics"], "number_report": True}})
    item = next(item for item in on.results if item.metric_id == "measure.number_report")
    assert item.family == "project_report"


def test_missing_profile_does_not_use_an_implicit_fallback(tmp_path, base_config):
    source = tmp_path / "story.md"
    source.write_text("One sentence. Another sentence follows.", encoding="utf-8")
    report = grade.analyze(source, {**base_config, "corpus_profile": "nowhere.json"})
    assert report.corpus_profile is None
    item = next(item for item in report.results if item.metric_id == "prose.fk")
    assert item.corpus is None


def test_zero_mad_corpus_does_not_make_every_difference_an_outlier(tmp_path, base_config):
    # A discrete metric over a small corpus has MAD 0. Calling any difference
    # an outlier there produced confident nonsense; the IQR, then the empirical
    # range, then an explicit refusal, are the fallbacks.
    profile = {"corpus_name": "flat", "comparison_unit": "book",
               "distributions": {"wps": {"values": [10.0] * 20}}}
    (tmp_path / "flat.json").write_text(json.dumps(profile), encoding="utf-8")
    source = tmp_path / "story.md"
    source.write_text("Ten words in this one sentence here right now yes. " * 60,
                      encoding="utf-8")
    report = grade.analyze(source, {**base_config, "corpus_profile": "flat.json"})
    item = next(item for item in report.results if item.metric_id == "prose.wps")
    assert item.status_type is not StatusType.CORPUS_OUTLIER
    assert item.corpus["method"] == "insufficient variation"


def test_crashed_child_is_counted_as_internal_error():
    result = grade.run_metric_process(
        "fixture.crash", [sys.executable, "-c", "raise RuntimeError('boom')"])
    assert len(result) == 1
    assert result[0].status_type is StatusType.INTERNAL_ERROR
    assert "exited" in result[0].error


def test_invalid_child_json_is_an_internal_error():
    result = grade.run_metric_process(
        "fixture.invalid", [sys.executable, "-c", "print('PASS words can change')"])
    assert result[0].status_type is StatusType.INTERNAL_ERROR


def test_json_ids_and_accounting_are_stable(tmp_path, base_config):
    source = tmp_path / "story.md"
    source.write_text("# Heading\n\nOne small sentence. Another longer one follows here.",
                      encoding="utf-8")
    payload = grade.analyze(source, base_config).to_dict()
    assert payload["summary"]["total"] == len(payload["results"])
    assert all(item["metric_id"] for item in payload["results"])
    assert json.loads(json.dumps(payload, sort_keys=True, default=str))


def test_project_rules_are_off_until_configured(tmp_path, base_config):
    source = tmp_path / "story.md"
    source.write_text("I noticed the door. It opened.", encoding="utf-8")
    empty = grade.analyze(source, {**base_config, "project_rules": {}})
    assert not any(item.action is Action.RULE_VIOLATION for item in empty.results)
    configured = grade.analyze(source, {**base_config, "project_rules": {
        "banned_phrases": [{"id": "noticed", "name": "Noticed", "pattern": r"\bI noticed\b"}]}})
    assert [item.metric_id for item in configured.results
            if item.action is Action.RULE_VIOLATION] == ["rule.noticed"]


def test_external_commands_require_explicit_trust(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text("One sentence.", encoding="utf-8")
    report = grade.analyze(source, {**base_config,
                                    "metric_commands": [{"id": "unsafe", "command": ["false"]}]})
    item = next(item for item in report.results
                if item.metric_id == "metric.external_commands")
    assert item.status_type is StatusType.UNAVAILABLE
