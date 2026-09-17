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


def _speaker_findings(text, **processing):
    from textgrader.document import DocumentAnalysis, TextProcessing
    from textgrader.metrics import dialogue_speaker_style
    analysis = DocumentAnalysis.from_text(
        text, processing=TextProcessing(**processing), comparison_unit="book")
    return {item["metric_id"]: item for item in dialogue_speaker_style.measure(analysis)}


def test_speaker_metrics_ignore_a_stripped_transcript():
    """A metric must not measure text the canonical cleanup removed.

    Speaker attribution read the raw file, so on a manuscript with chat
    chapters it silently replaced prose dialogue with unpunctuated chat lines.
    Contractions there are written without apostrophes and questions without
    question marks, so both rates collapsed to zero and were then compared
    against a corpus of prose dialogue.
    """

    chat = "\n".join(f"{name}: cant find it anywhere and im out of ideas"
                     for name in ["alice"] * 12 + ["bob"] * 12)
    prose = " ".join('"I cannot find it anywhere," Alice said. '
                     '"Then where did you look?" Bob said.' for _ in range(12))
    text = f"{prose}\n\n{chat}\n"
    stripped = _speaker_findings(text)
    kept = _speaker_findings(text, strip_transcript=False)
    assert "transcript" not in (stripped["dialogue.speaker_question_rate"]["warning"] or "")
    assert kept["dialogue.identified_speaker_count"]["value"] == 2


def test_speaker_metrics_refuse_thin_attribution():
    """Rates from a tenth of the turns are not comparable with rates from all.

    Victorian prose tags almost every spoken turn; modern prose drops the tag
    once a two-hander is established. Ungated, these metrics rank books by how
    explicitly they attribute speech rather than by how characters sound.
    """

    tagged = "\n\n".join('"I cannot find it," Alice said.\n\n"Where did you look?" Bob said.'
                        for _ in range(10))
    # Separate paragraphs, because an untagged quotation immediately following
    # another with no sentence between them is one continued turn, not two.
    untagged = "\n\n".join('"Somewhere else entirely." He shrugged at that.'
                          for _ in range(200))
    found = _speaker_findings(f"{tagged}\n\n{untagged}")
    item = found["dialogue.speaker_question_rate"]
    assert item["value"] is None
    assert "could be attributed" in item["warning"]


def test_speaker_metrics_work_when_attribution_is_dense():
    dense = "\n\n".join('"I cannot find it anywhere," Alice said.\n\n'
                        '"Then where did you look for it?" Bob said.' for _ in range(12))
    found = _speaker_findings(dense)
    assert found["dialogue.identified_speaker_count"]["value"] == 2
    assert found["dialogue.speaker_question_rate"]["value"] is not None
