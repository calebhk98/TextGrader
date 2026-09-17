"""Comparing populations rather than per-document averages.

The point of this metric is that a small text and a large corpus compare
directly, because the unit of observation on both sides is the sentence, the
paragraph or the word, never the document. These tests pin that property.
"""

import pytest

from textgrader import stats
from textgrader.corpus import build_profile
from textgrader.document import DocumentAnalysis
from textgrader.metrics import distribution_shape


@pytest.fixture(scope="module")
def corpus_profile(tmp_path_factory, prose):
    directory = tmp_path_factory.mktemp("shape")
    for index in range(8):
        (directory / f"book{index:02d}.txt").write_text(prose(400 + index, 120),
                                                        encoding="utf-8")
    return build_profile([directory], built_at="2026-01-01T00:00:00Z")


def _findings(text, profile):
    analysis = DocumentAnalysis.from_text(text, comparison_unit="chapter")
    return {item["metric_id"]: item for item in
            distribution_shape.measure(analysis, profile=profile)}


def test_profile_stores_the_population_not_a_summary(corpus_profile):
    stored = corpus_profile["item_distributions"]
    assert stored["sentence_words"]["count"] > corpus_profile["book_count"] * 50
    assert len(stored["sentence_words"]["quantiles"]) == stats.QUANTILE_GRID
    # The curve is the shape: minimum, median and maximum all readable from it.
    curve = stored["sentence_words"]["quantiles"]
    assert curve[0] <= curve[50] <= curve[-1]


def test_a_chapter_sized_text_compares_with_a_book_sized_corpus(corpus_profile, prose):
    """The property the whole design exists for.

    A few hundred sentences against tens of thousands is not a unit mismatch,
    it is a smaller sample of the same population, and it should come back with
    a small distance rather than a refusal.
    """

    chapter = prose(999, 30)
    found = _findings(chapter, corpus_profile)
    item = found["shape.sentence_words_distance"]
    assert item["value"] is not None
    assert item["sample_size"] < 200
    assert item["distribution"]["corpus_items"] > 20 * item["sample_size"]
    # Same generator, so the distance is small even though the sizes differ by
    # two orders of magnitude.
    assert item["value"] < 4.0


def test_a_metronomic_text_is_far_from_the_corpus(corpus_profile):
    flat = "\n\n".join(" ".join(["word"] * 12) + "." for _ in range(200))
    found = _findings(flat, corpus_profile)
    assert found["shape.sentence_words_distance"]["value"] > 5.0
    # Everything lands in one band that is a fifth of the corpus by construction.
    assert found["shape.sentence_words_concentration"]["value"] > 40


def test_bands_come_from_the_corpus_and_sum_to_a_hundred(corpus_profile, prose):
    found = _findings(prose(1234, 40), corpus_profile)
    rows = found["shape.sentence_words_distance"]["evidence"]
    assert len(rows) == 5
    assert sum(row["text_share"] for row in rows) == pytest.approx(100.0, abs=0.5)
    assert all(row["corpus_share"] == pytest.approx(20.0) for row in rows)


def test_too_few_items_is_a_refusal_not_a_noisy_number(corpus_profile):
    found = _findings("One. Two. Three.", corpus_profile)
    item = found["shape.sentence_words_distance"]
    assert item["value"] is None
    assert "below the" in item["warning"]


def test_a_profile_without_pooled_items_says_so():
    found = _findings("A sentence here. And another one there.", {"book_count": 3})
    item = found["shape.sentence_words_distance"]
    assert item["value"] is None
    assert "rebuild" in item["warning"]
