"""Plain descriptions for measurements whose names a writer cannot act on."""

from textgrader.plain_names import describe


def test_generated_families_name_the_sequence_and_the_direction():
    text = describe("rhythm.signal_sentence_words_zero_crossing_rate")
    assert "sentence length" in text and "alternate" in text
    assert "paragraph length" in describe("drift.nonlinear_paragraph_words_rqa_laminarity")
    assert describe("style.autofeature_sentence_words_stats_quantile_p05").startswith(
        "the 5th-percentile sentence length")
    assert "ks test" in describe("style.distribution_sentence_commas_ks_pvalue")


def test_one_off_obscure_measures_are_described():
    assert "doubled letters" in describe("style.randomness_char_run_length_entropy")
    assert "coleman liau" in describe("nlp.readability_coleman_liau_disagreement")


def test_self_explanatory_measures_are_left_alone():
    assert describe("prose.wps") is None
    assert describe("style.pov_first") is None
