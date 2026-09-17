"""The shared pipeline is the thing every other number depends on."""

import pytest

from textgrader.document import (DocumentAnalysis, NlpSettings, TextProcessing,
                                 units_comparable)


def test_cleanup_happens_once_and_is_recorded():
    raw = "# Heading\n\nAlpha beta gamma. Delta epsilon.\n\nalice: chat line\n"
    analysis = DocumentAnalysis.from_text(raw)
    assert "Heading" not in analysis.text
    assert "chat line" not in analysis.text
    assert analysis.transcript_share > 0
    assert analysis.describe()["text_processing"]["strip_markdown_headings"] is True


def test_text_processing_switches_are_honoured():
    raw = "# Heading\n\nAlpha beta.\n"
    keep = DocumentAnalysis.from_text(
        raw, processing=TextProcessing(strip_markdown_headings=False))
    assert "Heading" in keep.text


def test_unknown_text_processing_option_is_refused():
    with pytest.raises(ValueError, match="unknown text_processing options"):
        TextProcessing.from_config({"strip_gutenburg": True})


def test_a_paragraph_before_a_scene_break_survives():
    # A Setext underline must be on the line below its heading. Matching \s{0,3}
    # let a blank line plus "---" swallow the paragraph above the scene break.
    raw = "One.\n\nThe paragraph before the break.\n\n---\n\nAfter.\n"
    assert "paragraph before the break" in DocumentAnalysis.from_text(raw).text


def test_removing_a_heading_does_not_merge_paragraphs():
    analysis = DocumentAnalysis.from_text("Para one.\n\n## Head\n\nPara two.\n")
    assert len(analysis.paragraphs) == 2


def test_dialogue_and_narration_are_separate_documents():
    raw = '"Hello there, how are you?" she said. He looked away and said nothing at all.'
    analysis = DocumentAnalysis.from_text(raw)
    assert "Hello there" in analysis.dialogue.text
    assert "Hello there" not in analysis.narration.text
    assert "looked away" in analysis.narration.text
    assert analysis.dialogue.channel == "dialogue"


def test_split_attribution_rejoins_into_one_turn():
    analysis = DocumentAnalysis.from_text('"A word here," she says, "and another."')
    assert len(analysis.turns) == 1


def test_separate_turns_are_not_joined():
    analysis = DocumentAnalysis.from_text('"First." Alice says. "Second," Bob says.')
    assert len(analysis.turns) == 2


def test_in_dialogue_offsets():
    raw = 'He said "yes" firmly.'
    analysis = DocumentAnalysis.from_text(raw)
    assert analysis.in_dialogue(raw.index("yes"))
    assert not analysis.in_dialogue(raw.index("firmly"))


def test_sections_split_on_headings():
    raw = "# One\n\nAlpha beta.\n\n## Two\n\nGamma delta.\n"
    titles = [title for title, _ in DocumentAnalysis.from_text(raw).sections]
    assert titles == ["One", "Two"]


def test_windows_cover_the_document(sample_text):
    analysis = DocumentAnalysis.from_text(sample_text)
    windows = analysis.windows(400)
    assert len(windows) > 1
    assert sum(window.word_count for window in windows) == analysis.word_count


def test_empty_document_is_safe():
    analysis = DocumentAnalysis.from_text("")
    assert analysis.sentences == [] and analysis.paragraphs == []
    assert analysis.dialogue.text == "" and analysis.turns == []
    assert analysis.describe()["analyzed_words"] == 0


def test_scale_dependent_metrics_refuse_unit_mismatch():
    assert not units_comparable("_words", "chapter", "book")
    assert units_comparable("_words", "book", "book")
    # An unspecified unit resolves to a whole document on both sides.
    assert units_comparable("_words", "unknown", "book")
    # Rates are what units exist for and are never refused.
    assert units_comparable("style.mattr", "chapter", "book")


def test_nlp_settings_reject_unknown_options():
    with pytest.raises(ValueError):
        NlpSettings.from_config({"modell": "en_core_web_sm"})


def test_only_size_metrics_are_gated_by_unit():
    """A rate does not care how much text produced it.

    A loose "ends with _words" test caught every ``*_per_1000_words`` rate and
    refused to compare it across units, which are the most carefully normalized
    measurements in the tool. Gating is by metric id now.
    """

    from textgrader.document import units_comparable
    for rate in ("style.mattr", "punct.comma_per_1000_words", "lexical.word_zipf",
                 "rhythm.sentence_length_entropy", "dialogue.turn_words"):
        assert units_comparable(rate, "chapter", "book"), rate
    for size in ("_words", "prose.words", "word_count", "drift.change_point_count"):
        assert not units_comparable(size, "chapter", "book"), size
        assert units_comparable(size, "book", "book"), size
