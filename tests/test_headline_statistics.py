"""Headlines that must be one fixed statistic, and the version that says so.

A corpus comparison reads only a finding's headline.  Small whole-number
counts (run lengths, characters per word, commas per sentence, sentences per
paragraph) have a median that sits on the same value for nearly every book:
on the 50-book reference, sentences per paragraph had a median of 2 on 36
books and characters per narration word a median of 4 on all 50.  Those ids
headline the mean, and the metric definition version moved with them.
"""

import statistics

import grade
from textgrader.corpus import METRIC_DEFINITION_VERSION
from textgrader.document import DocumentAnalysis
from textgrader.metrics import paragraph_rhythm, punctuation_profile, rhythm_runs
from textgrader.results import Report


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


TEXT = "\n\n".join([
    "One. Two, three. Four.",
    "Five.",
    "Six, seven, eight. Nine. Ten. Eleven.",
    "Twelve.",
] * 6)


def test_sentences_per_paragraph_headlines_the_mean_and_words_keep_the_median():
    analysis = DocumentAnalysis.from_text(TEXT)
    found = _by_id(paragraph_rhythm.measure(analysis))
    counts = [len(paragraph) for paragraph in analysis.sentences_by_paragraph]
    assert found["rhythm.paragraph_sentences"]["value"] == statistics.fmean(counts)
    assert found["rhythm.paragraph_sentences"]["distribution"]["median"] == statistics.median(counts)
    words = [len(paragraph.split()) for paragraph in analysis.paragraphs]
    assert found["rhythm.paragraph_words"]["value"] == statistics.median(words)


def test_commas_per_sentence_headlines_the_mean():
    analysis = DocumentAnalysis.from_text(TEXT)
    found = _by_id(punctuation_profile.measure(analysis))
    commas = [sentence.count(",") for sentence in analysis.sentences]
    item = found["punct.commas_per_sentence_shape"]
    assert item["value"] == statistics.fmean(commas)
    assert item["distribution"]["median"] == statistics.median(commas) == 0


def test_run_lengths_headline_the_mean():
    analysis = DocumentAnalysis.from_text(TEXT)
    for item in rhythm_runs.measure(analysis):
        if item["metric_id"].startswith("rhythm.run_length_") and item["value"] is not None:
            assert item["value"] == item["distribution"]["mean"]


def test_a_profile_from_older_metric_definitions_is_flagged():
    old = Report()
    grade.check_definition_version({"metric_definition_version": "2"}, old)
    [warning] = [r for r in old.results if r.metric_id == "corpus.metric_definition_version"]
    assert "rebuild" in warning.warning
    current = Report()
    grade.check_definition_version({"metric_definition_version": METRIC_DEFINITION_VERSION},
                                   current)
    assert current.results == []
