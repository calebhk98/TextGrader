"""Everyday-vocabulary word rates and keyness."""

from collections import Counter

import pytest

from textgrader.corpus import WORD_RATE_PREFIX, _word_rate_tables, build_profile, without_source
from textgrader.document import DocumentAnalysis, TextProcessing
from textgrader.metrics import word_rates
from textgrader.plain_names import describe


def test_the_vocabulary_keeps_widely_used_words_and_drops_one_book_names():
    books = [Counter({"the": 50, "said": 5, "zorblax": 40}), Counter({"the": 40, "said": 8}),
             Counter({"the": 60, "said": 2, "house": 1})]
    rates, document_frequency = _word_rate_tables(books, sum(books, Counter()))
    # zorblax is the second most frequent word but occurs in 1 of 3 books.
    assert rates["vocabulary"] == ["the", "said"]
    assert rates["per_book"][0] == pytest.approx([1000 * 50 / 95, 1000 * 5 / 95])
    assert document_frequency["zorblax"] == 1 and document_frequency["the"] == 3


def _write_books(tmp_path):
    for index in range(4):
        (tmp_path / f"book{index}.txt").write_text(
            " ".join(f"She said the door was open number {n} and he walked to the house."
                     + (" Then he waited." if n % (index + 2) == 0 else "")
                     for n in range(150)), encoding="utf-8")


def test_the_profile_carries_word_rates_and_a_hold_out_drops_that_row(tmp_path):
    _write_books(tmp_path)
    profile = build_profile([tmp_path], corpus_name="t", built_at="2026-01-01T00:00:00Z",
                            metrics={}, metric_selection="enabled", jobs=1)
    rates = profile["word_rates"]
    assert len(rates["per_book"]) == profile["book_count"] == 4
    assert "said" in rates["vocabulary"]
    key = f"{WORD_RATE_PREFIX}then"
    assert len(profile["distributions"][key]["values"]) == 4
    held = without_source(profile, profile["books"][0]["source_id"])
    assert len(held["word_rates"]["per_book"]) == 3
    assert len(held["distributions"][key]["values"]) == 3
    assert held["distributions"][key]["values"] == sorted(
        row[rates["vocabulary"].index("then")] for row in rates["per_book"][1:])


def _analysis(text):
    return DocumentAnalysis.from_text(text, processing=TextProcessing(), comparison_unit="book")


def test_the_metric_reports_each_rate_and_the_overused_words(tmp_path):
    _write_books(tmp_path)
    profile = build_profile([tmp_path], corpus_name="t", built_at="2026-01-01T00:00:00Z",
                            metrics={}, metric_selection="enabled", jobs=1)
    text = " ".join("The spaceship hummed and the spaceship said nothing." for _ in range(80))
    findings = {item["metric_id"]: item for item in word_rates.measure(_analysis(text),
                                                                          profile=profile)}
    tokens = _analysis(text).tokens
    assert findings[f"{WORD_RATE_PREFIX}said"]["value"] == 1000 * tokens.count("said") / len(tokens)
    overused = findings["lexical.keyness_overused"]
    words = [row["word"] for row in overused["evidence"]]
    assert words[0] in {"spaceship", "hummed", "nothing"}
    assert "said" not in words or overused["evidence"][words.index("said")]["g2"] >= 10.83
    assert overused["evidence"][0]["corpus_observations_using_it"] == "0 of 4"


def test_without_a_word_rate_table_the_metric_says_how_to_get_one():
    [finding] = word_rates.measure(_analysis("A short text."), profile={})
    assert finding["value"] is None and "rebuild" in finding["warning"]


def test_word_rates_need_no_plain_name():
    # lexical.word_rate.said already says what it is; a means: line under
    # each of up to 300 rows would only lengthen the report.
    assert describe("lexical.word_rate.said") is None
