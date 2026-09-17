"""End-to-end behaviour of the grading contract an authoring agent depends on."""

import json

import pytest

import grade
from textgrader.corpus import build_profile, write_profile
from textgrader.metrics import REGISTRY
from textgrader.results import Action, StatusType


def _ids(report):
    return {item.metric_id for item in report.results}


def test_core_and_optional_metrics_analyze_the_same_text(tmp_path, base_config):
    # Core analysis strips Gutenberg boilerplate and headings; the optional
    # metrics used to receive the raw file, so the two described different
    # documents. The word counts must now agree exactly.
    source = tmp_path / "story.md"
    source.write_text(
        "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n# Chapter\n\n"
        + "Alpha beta gamma delta. " * 200
        + "\n\n*** END OF THIS PROJECT GUTENBERG EBOOK X ***\n", encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"], "mattr": {"enabled": True}}}
    report = grade.analyze(source, config)
    words = next(item for item in report.results if item.metric_id == "prose.words")
    assert words.value == report.document["analyzed_words"]
    assert "GUTENBERG" not in str(report.document)
    assert report.document["raw_words"] > words.value


def test_a_tiny_document_gets_no_confident_outliers(tmp_path, base_config, corpus_dir):
    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z")
    write_profile(profile, tmp_path / "profile.json")
    source = tmp_path / "tiny.md"
    source.write_text("Two words.", encoding="utf-8")
    config = {**base_config, "corpus_profile": "profile.json"}
    report = grade.analyze(source, config)
    assert not [item for item in report.results
                if item.status_type is StatusType.CORPUS_OUTLIER]
    assert any(item.action is Action.INSUFFICIENT_DATA for item in report.results)


def test_a_real_document_does_get_compared(manuscript, tmp_path, base_config, corpus_dir):
    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z")
    write_profile(profile, tmp_path / "profile.json")
    report = grade.analyze(manuscript, {**base_config, "corpus_profile": "profile.json"})
    compared = [item for item in report.results if item.corpus]
    assert compared
    assert all(item.corpus["corpus_count"] == 12 for item in compared)
    assert all(item.confidence in ("low", "high", "none") for item in compared)


def test_chapter_against_a_book_corpus_withholds_length_comparisons(
        manuscript, tmp_path, base_config, corpus_dir):
    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z",
                            comparison_unit="book")
    write_profile(profile, tmp_path / "profile.json")
    config = {**base_config, "corpus_profile": "profile.json",
              "analysis": {"comparison_unit": "chapter"}}
    report = grade.analyze(manuscript, config)
    words = next(item for item in report.results if item.metric_id == "prose.words")
    assert words.action is Action.INSUFFICIENT_DATA
    assert "scales with document length" in words.warning
    # A rate is unaffected: that is what expressing a measurement as a rate is for.
    rate = next(item for item in report.results if item.metric_id == "prose.u10")
    assert rate.action is not Action.INSUFFICIENT_DATA


def test_text_processing_reaches_the_corpus_builder(corpus_dir):
    stripped = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z")
    kept = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z",
                         text_processing={"strip_gutenberg": False})
    assert stripped["text_processing"]["strip_gutenberg"] is True
    assert kept["text_processing"]["strip_gutenberg"] is False


def test_mismatched_preprocessing_is_reported(manuscript, tmp_path, base_config, corpus_dir):
    profile = build_profile([corpus_dir], built_at="2026-01-01T00:00:00Z",
                            text_processing={"segmenter": "builtin"})
    write_profile(profile, tmp_path / "profile.json")
    report = grade.analyze(manuscript, {**base_config, "corpus_profile": "profile.json"})
    note = next(item for item in report.results if item.metric_id == "corpus.text_processing")
    assert "prepared differently" in note.warning


def test_every_optional_metric_is_off_unless_enabled(tmp_path, base_config):
    source = tmp_path / "story.md"
    source.write_text("He walked home. He walked home.", encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results
                if item.metric_id.split(".")[0] in
                {"style", "nlp", "rhythm", "syntax", "lexical", "semantic",
                 "dialogue", "discourse", "pov", "punct", "drift", "repetition"}]


def test_invalid_user_regex_is_reported_not_raised(manuscript, base_config):
    config = {**base_config,
              "project_rules": {"banned_phrases": [{"id": "bad", "pattern": "(unclosed"}]}}
    report = grade.analyze(manuscript, config)
    rule = next(item for item in report.results if item.metric_id == "rule.bad")
    assert rule.status_type is StatusType.UNAVAILABLE
    assert "invalid pattern" in rule.error


def test_a_configured_rule_is_a_violation_not_an_observation(tmp_path, base_config):
    source = tmp_path / "story.md"
    source.write_text("It was very very good.", encoding="utf-8")
    config = {**base_config,
              "project_rules": {"banned_phrases":
                                [{"id": "very", "name": "Filter", "pattern": r"\bvery\b"}]}}
    report = grade.analyze(source, config)
    rule = next(item for item in report.results if item.metric_id == "rule.very")
    assert rule.value == 2
    assert rule.action is Action.RULE_VIOLATION


def test_report_json_is_serializable_and_has_the_agent_fields(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    payload = json.loads(json.dumps(report.to_dict(), default=str))
    assert payload["schema_version"] >= 2
    assert payload["document"]["segmenter"]
    first = payload["results"][0]
    for field in ("family", "direction", "severity", "comparison_unit", "action",
                  "sample_size", "evidence", "channel", "distribution"):
        assert field in first
    assert "top_findings" in payload["summary"]
    assert "by_action" in payload["summary"]


def _bare_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"metrics": {}, "corpus_profile": None,
                                "analysis": {"comparison_unit": "book"}}), encoding="utf-8")
    return str(path)


def test_cli_switches_enable_metrics(tmp_path, capsys, manuscript):
    code = grade.main([str(manuscript), "--config", _bare_config(tmp_path),
                       "--enable", "mattr", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(item["metric_id"] == "style.mattr" for item in payload["results"])


def test_cli_enables_a_whole_family(tmp_path, capsys, manuscript):
    grade.main([str(manuscript), "--config", _bare_config(tmp_path),
                "--enable-family", "sentence_rhythm", "--json"])
    payload = json.loads(capsys.readouterr().out)
    families = {item["family"] for item in payload["results"]}
    assert "sentence_rhythm" in families


def test_cli_rejects_an_unknown_metric(manuscript):
    with pytest.raises(SystemExit):
        grade.main([str(manuscript), "--enable", "no_such_metric"])


def test_every_bundled_measure_names_exactly_one_target():
    # dialogue_study was handed the manuscript's PARENT DIRECTORY as a corpus,
    # and prose_check/verify_citations were handed the manuscript as if it were
    # a character sheet. The requirement that came out of that is that every
    # template NAMES WHAT IT PASSES, which is what this checks.
    #
    # It used to be checked by banning {manuscript_dir} and requiring
    # {manuscript} everywhere, which over-corrected: it also stopped register,
    # absolutes, tics and check_edits receiving the chapters directory they
    # compare chapters from, so they saw one unit of text and collapsed to a
    # single row. A closed vocabulary keeps the original guard - a template
    # cannot name something nobody resolves, and cannot silently name two
    # things - without forcing every report to take the same one.
    from textgrader.reports import TARGETS
    for name, template in grade.BUNDLED_MEASURES.items():
        named = [target for target in TARGETS if target in template]
        assert len(named) == 1, f"{name} names {named or 'no target'}"
    # dialogue_study is the one that was actually broken by a directory.
    assert "{manuscript}" in grade.BUNDLED_MEASURES["dialogue_study"]
    for name in ("prose_check", "verify_citations"):
        assert "--manuscript" in grade.BUNDLED_MEASURES[name], name


def test_per_chapter_reports_get_the_chapters_directory():
    # register judges each chapter against the book's own median. Handed the
    # concatenated manuscript it has one row, the median IS that row, nothing
    # can exceed it, and it prints a pass in the same words as a real one.
    for name in ("register", "absolutes", "quote_length", "check_edits"):
        assert grade.BUNDLED_MEASURES[name] == ("{chapters_dir}",), name


def test_a_report_asking_for_chapters_gets_them(tmp_path, monkeypatch):
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command

        class Result:
            returncode, stdout, stderr = 0, "", ""
        return Result()

    monkeypatch.setattr(grade.subprocess, "run", fake_run)
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "01.md").write_text("One.", encoding="utf-8")
    config = {"_config_dir": str(tmp_path), "_config_path": str(tmp_path / "mine.json"),
              "chapters_dir": "chapters"}
    grade.run_bundled_measure("register", tmp_path / "draft.md", config)
    assert str(chapters) in recorded["command"]


def test_a_missing_chapters_directory_falls_back_to_the_manuscript(tmp_path, monkeypatch):
    # A project that keeps no chapters directory still has to run. It gets the
    # single-unit output those reports collapse to, which is what every report
    # got before this vocabulary existed.
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["command"] = command

        class Result:
            returncode, stdout, stderr = 0, "", ""
        return Result()

    monkeypatch.setattr(grade.subprocess, "run", fake_run)
    manuscript = tmp_path / "draft.md"
    manuscript.write_text("One.", encoding="utf-8")
    config = {"_config_dir": str(tmp_path), "_config_path": str(tmp_path / "mine.json")}
    grade.run_bundled_measure("register", manuscript, config)
    assert str(manuscript) in recorded["command"]


def test_a_custom_config_reaches_the_bundled_reports(tmp_path, monkeypatch):
    from textgrader.project import CONFIG_ENV_VAR
    recorded = {}

    def fake_run(command, **kwargs):
        recorded["config"] = kwargs["env"][CONFIG_ENV_VAR]

        class Result:
            returncode, stdout, stderr = 0, "", ""
        return Result()

    monkeypatch.setattr(grade.subprocess, "run", fake_run)
    config = {"_config_dir": str(tmp_path), "_config_path": str(tmp_path / "mine.json")}
    grade.run_bundled_measure("register", tmp_path / "draft.md", config)
    assert recorded["config"] == str(tmp_path / "mine.json")
