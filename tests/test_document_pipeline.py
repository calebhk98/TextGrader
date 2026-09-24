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


# ------------------------------------------------------------ sentence splitting
#
# pySBD refuses to split inside quotation marks, brackets and ``--`` pairs, so
# a paragraph that is one long quoted speech used to come back as a single
# sentence: 815 words in one paragraph of *Sense and Sensibility*.  The
# built-in splitter gets those right but ends a sentence at every "Mr.".
# These pin the behaviour the quote-aware segmenter must keep on both counts.

def _pysbd():
    from textgrader.optional import require
    return require("pysbd")[0]


# Keyed to optional.require, exactly as the segmenter is, so a run with
# TEXTGRADER_DISABLE_OPTIONAL skips these instead of failing on the built-in
# splitter's different counts.  Only these tests: the rest of this file must
# still run on a machine without pySBD.
needs_pysbd = pytest.mark.skipif(_pysbd() is None, reason="pySBD unavailable")


def _sentences(text, **settings):
    processing = TextProcessing.from_config({"segmenter": "auto", **settings})
    return DocumentAnalysis.from_text(text, processing=processing).sentences


@pytest.mark.parametrize("text, expected", [
    ("Mr. Darcy bowed. Dr. Smith and Mrs. Jennings arrived at St. Paul's. It was late.", 3),
    ("J. R. R. Tolkien wrote it. H. G. Wells did not.", 2),
    ("It cost $3.50 at 4 p.m. on the 4th. She paid 2.5 times more.", 2),
    ("“Come here. Sit down. We need to talk.” He sat.", 4),
    ("“Is it you?” she asked. “Yes!” he cried.", 2),
    ("“Stop!” he shouted. “Now!”", 2),
    ("He said, “I will go. You stay.” Then he left.", 3),
    ("She asked “why?” and left.", 1),
    ('"Come here. Sit down." He sat.', 3),
    ("“He told me, ‘Run.’ So I ran.”", 2),
    ("‘Come here. Sit down.’ He sat.", 3),
    ("I did n’t mean that. I ‘m glad to see you. I am.", 3),
    ("(See chapter 3. It explains.) Read it.", 3),
    ("He went--as he always did--to the store.", 1),
    ("It was late--too late. She knew it--and so did he. Then silence.", 3),
    ("It rained. The lamp was lit. Nobody came.", 3),
])
@needs_pysbd
def test_quote_aware_segmentation_counts(text, expected):
    assert len(_sentences(text)) == expected, _sentences(text)


@needs_pysbd
def test_a_long_quoted_speech_is_split_into_its_sentences():
    speech = "“" + " ".join(f"Sentence number {n} is here." for n in range(12)) + "”"
    sentences = _sentences(speech)
    assert len(sentences) == 12
    assert max(len(sentence.split()) for sentence in sentences) == 5


@needs_pysbd
def test_a_closing_quote_stays_with_the_sentence_it_closes():
    sentences = _sentences("“Come here. We need to talk.” He sat.")
    assert sentences[1] == "We need to talk.”"
    assert sentences[2] == "He sat."


@needs_pysbd
def test_unquoted_text_is_segmented_exactly_as_plain_pysbd():
    text = ("Mr. Darcy bowed at 4 p.m. on the 4th. J. R. R. Tolkien wrote it. "
            "The lamp was lit... then it went out. It rained.")
    plain = [s.strip() for s in _pysbd().Segmenter(language="en", clean=False).segment(text)]
    assert _sentences(text) == [s for s in plain if s]


def test_the_fingerprint_records_the_segmenter_that_actually_ran():
    # "auto" means pySBD where it is installed and the built-in splitter where
    # it is not.  A profile and a manuscript that both say "auto" but split
    # differently must not look comparable.
    processing = TextProcessing.from_config({"segmenter": "auto"})
    analysis = DocumentAnalysis.from_text("One. Two.", processing=processing)
    assert processing.fingerprint()["segmenter_resolved"] == analysis.segmenter
    builtin = TextProcessing.from_config({"segmenter": "builtin"})
    assert builtin.fingerprint()["segmenter_resolved"] == "builtin"
    assert processing.fingerprint() != builtin.fingerprint()


@needs_pysbd
def test_cleaning_keeps_plain_pysbd_and_says_why():
    analysis = DocumentAnalysis.from_text(
        "“Come here. Sit down.” He sat.",
        processing=TextProcessing.from_config({"segmenter": "auto", "segmenter_clean": True}))
    assert analysis.segmenter == "pysbd"
    assert any("segmenter_clean" in warning for warning in analysis.warnings)
