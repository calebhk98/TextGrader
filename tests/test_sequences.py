"""Direct tests of the named-sequence registry, independent of any metric.

These check the foundation the whole ``timeseries_suite`` is built on: that
each sequence reads the right thing off ``DocumentAnalysis``, that it
memoizes, and that it degrades to an empty, clearly-labelled sequence rather
than raising when an optional package is unavailable.
"""

import pytest

from textgrader import optional
from textgrader import sequences as seq
from textgrader.document import DocumentAnalysis, NlpSettings

TEXT = ("He walked to the old house. She ran fast down the road! Why did they "
        "stop there? It was already very late by then.\n\n"
        "A second, much longer paragraph follows, with several more words in "
        "it than the first one had, so the two paragraphs are not the same size.")


@pytest.fixture
def analysis():
    return DocumentAnalysis.from_text(TEXT)


def test_registry_lists_every_sequence_once():
    names = seq.list_sequences()
    assert len(names) == len(set(names))
    assert "sentence_words" in names
    assert "window_dialogue_fraction" in names


def test_sentence_words_matches_document_analysis(analysis):
    sequence = seq.get_sequence(analysis, "sentence_words")
    assert sequence.sample_unit == "sentence"
    assert sequence.unit == "words"
    assert list(sequence.values) == [float(n) for n in analysis.sentence_lengths]


def test_sentence_chars_counts_characters_not_words(analysis):
    sequence = seq.get_sequence(analysis, "sentence_chars")
    assert sequence.length == analysis.sentence_count
    assert all(value > 0 for value in sequence.values)
    # A sentence's character count is always at least its word count.
    words = seq.get_sequence(analysis, "sentence_words")
    assert all(c >= w for c, w in zip(sequence.values, words.values))


def test_paragraph_sequences_match_document_analysis(analysis):
    words = seq.get_sequence(analysis, "paragraph_words")
    sentences = seq.get_sequence(analysis, "paragraph_sentences")
    assert words.sample_unit == sentences.sample_unit == "paragraph"
    assert list(words.values) == [float(n) for n in analysis.paragraph_lengths]
    assert list(sentences.values) == [float(n) for n in analysis.paragraph_sentence_counts]
    assert words.length == 2  # the fixture text has two paragraphs


def test_sentence_punctuation_counts_marks():
    analysis = DocumentAnalysis.from_text("Wait, what? No! Really... okay then.")
    sequence = seq.get_sequence(analysis, "sentence_punctuation")
    # "Wait, what?" has a comma and a question mark: two marks.
    assert sequence.values[0] == 2.0


def test_sentence_char_entropy_is_zero_for_a_single_repeated_letter():
    analysis = DocumentAnalysis.from_text("aaaa aaaa aaaa. Bbbb bbbb bbbb.")
    sequence = seq.get_sequence(analysis, "sentence_char_entropy")
    assert sequence.values[0] == pytest.approx(0.0, abs=1e-9)


def test_get_sequence_memoizes_within_a_document(analysis):
    first = seq.get_sequence(analysis, "sentence_words")
    second = seq.get_sequence(analysis, "sentence_words")
    assert first is second


def test_get_sequence_keys_on_settings_separately(analysis):
    small = seq.get_sequence(analysis, "window_dialogue_fraction", {"window_words": 200})
    large = seq.get_sequence(analysis, "window_dialogue_fraction", {"window_words": 2000})
    assert small.settings["window_words"] == 200
    assert large.settings["window_words"] == 2000
    # Different settings must not share a cache slot even though both are
    # requests for the same named sequence.
    assert small is not large


def test_unknown_sequence_name_raises_a_clear_error(analysis):
    with pytest.raises(KeyError):
        seq.get_sequence(analysis, "not_a_real_sequence")


def test_window_sequences_use_analysis_windows(analysis):
    dialogue = seq.get_sequence(analysis, "window_dialogue_fraction", {"window_words": 200})
    pronoun = seq.get_sequence(analysis, "window_pronoun_rate", {"window_words": 200})
    assert dialogue.sample_unit == pronoun.sample_unit == "window"
    assert dialogue.length == len(analysis.windows(200))


def test_parse_sequences_degrade_without_spacy(monkeypatch, analysis):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "spacy")
    optional.reset_cache()
    try:
        # Force the pipeline to be re-resolved under the simulated absence.
        degraded = DocumentAnalysis.from_text(TEXT)
        for name in ("sentence_parse_depth", "sentence_dependency_distance",
                    "sentence_clause_count", "sentence_pos_entropy"):
            sequence = seq.get_sequence(degraded, name)
            assert sequence.values == ()
            assert sequence.warning
    finally:
        optional.reset_cache()


def test_entity_sequence_explains_ner_being_disabled(analysis):
    # The shared pipeline disables 'ner' by default (see NlpSettings), and
    # this sequence needs it; it must degrade with that specific reason
    # rather than silently returning zeros.
    sequence = seq.get_sequence(analysis, "sentence_entity_count")
    if analysis.nlp_unavailable:
        pytest.skip("no spaCy model available in this environment")
    assert sequence.values == ()
    assert "ner" in (sequence.warning or "")


def test_entity_sequence_works_once_ner_is_enabled():
    analysis = DocumentAnalysis.from_text(
        "Sarah walked to the market. Sarah bought bread there.",
        nlp_settings=NlpSettings(disable=()))
    if analysis.nlp_unavailable:
        pytest.skip("no spaCy model available in this environment")
    sequence = seq.get_sequence(analysis, "sentence_entity_count")
    assert sequence.warning is None
    assert sequence.length == analysis.sentence_count
    assert sum(sequence.values) >= 1  # "Sarah" should be recognized at least once


def test_content_rarity_degrades_without_wordfreq(monkeypatch, analysis):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "wordfreq")
    optional.reset_cache()
    try:
        sequence = seq.get_sequence(analysis, "sentence_content_rarity")
        assert sequence.values == ()
        assert "wordfreq" in (sequence.warning or "")
    finally:
        optional.reset_cache()


def test_content_rarity_scores_common_words_as_common(analysis):
    if not optional.have("wordfreq"):
        pytest.skip("wordfreq not installed in this environment")
    sequence = seq.get_sequence(analysis, "sentence_content_rarity")
    assert sequence.length == analysis.sentence_count
    # Zipf scores for ordinary English prose sit well above 1 (vanishingly rare).
    assert all(value > 1.0 for value in sequence.values)


def test_semantic_sequences_fall_back_to_a_labelled_lexical_backend(analysis):
    # sentence_transformers is not part of this task's guaranteed environment;
    # whether or not it happens to be installed, the sequence must either use
    # it cleanly or fall back with an explicit backend note - never silently.
    similarity = seq.get_sequence(analysis, "sentence_similarity_prev")
    centroid = seq.get_sequence(analysis, "sentence_distance_centroid")
    assert similarity.length == analysis.sentence_count - 1
    assert centroid.length == analysis.sentence_count
    assert similarity.warning and ("backend=" in similarity.warning)
    assert centroid.warning and ("backend=" in centroid.warning)


def test_embedding_sequences_need_a_minimum_of_sentences():
    tiny = DocumentAnalysis.from_text("Only one sentence here.")
    similarity = seq.get_sequence(tiny, "sentence_similarity_prev")
    centroid = seq.get_sequence(tiny, "sentence_distance_centroid")
    assert similarity.values == ()
    assert centroid.values == ()
