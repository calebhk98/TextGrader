"""What the profile in data/ has to carry.

The profile shipped before this predated two format changes and silently cost
a metric family. It had no pooled item distributions, so the six shape.*
metrics reported "unavailable: rebuild it to compare distributions rather than
per-book averages" - one of the tool's genuine advantages, off by default for
anyone using the bundled profile. It also predated recorded text_processing,
so corpus.text_processing fired on every run with a warning the user could not
act on because they had not built the profile.

Both are honest failures rather than wrong numbers, which is the design
working. They are also both fixed by rebuilding, and nothing checked that the
rebuild had happened.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROFILE = json.loads((ROOT / "data/prose_reference.json").read_text(encoding="utf-8"))

#: The six metrics in textgrader/metrics/distribution_shape.py, which compare
#: populations of items rather than per-book averages.
POOLED = {"sentence_words", "paragraph_words", "paragraph_sentences",
          "word_characters", "sentence_commas", "turn_words"}


def test_the_shipped_profile_has_pooled_item_distributions():
    assert set(PROFILE.get("item_distributions") or {}) >= POOLED


def test_the_shipped_profile_records_how_its_texts_were_prepared():
    # Without this the corpus and the manuscript cannot be shown to have been
    # prepared the same way, and the check that says so fires on every run.
    recorded = PROFILE.get("text_processing")
    assert recorded
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    assert recorded == config["text_processing"]


def test_the_shipped_profile_is_large_enough_to_take_a_percentile_from():
    # build_corpus.py warns below 30 for the same reason.
    assert PROFILE.get("book_count", 0) >= 30
    assert len(PROFILE.get("books") or {}) >= 30


def test_the_shipped_profile_declares_its_comparison_unit():
    assert PROFILE.get("comparison_unit") == "book"


def test_the_shipped_profile_built_cleanly():
    assert not PROFILE.get("skipped_sources")
    assert not PROFILE.get("metric_errors")


def _book_names(profile):
    books = profile["books"]
    if isinstance(books, dict):
        return set(books)
    return {Path(row["source_filename"]).stem for row in books}


def test_both_shipped_references_describe_the_same_corpus():
    # absolutes.py keeps its own reference. Built from a different shelf, its
    # percentiles and the core ones describe different populations while
    # reading as one report.
    absolutes = json.loads((ROOT / "data/absolutes_reference.json").read_text(encoding="utf-8"))
    assert set(absolutes) == _book_names(PROFILE)


def test_a_benchmark_can_name_a_book_in_the_shipped_profile():
    """Guards a layout the benchmark code did not originally handle.

    The legacy profile is {book_name: {metric: value}}; the current one puts
    the rows in a LIST keyed by source_id/source_filename. Written against the
    legacy shape alone, benchmark_comparison raised AttributeError on the
    profile this repository now ships.
    """

    import grade
    name = sorted(_book_names(PROFILE))[0]
    got = grade.benchmark_comparison(PROFILE, {"fk": 5.0, "wps": 11.0}, name)
    assert got["error"] is None
    assert got["of"] >= 1


def test_a_benchmark_naming_nothing_lists_what_is_there():
    import grade
    got = grade.benchmark_comparison(PROFILE, {"fk": 5.0}, "not_a_book_in_here")
    assert "not_a_book_in_here" in got["error"]
    assert got["available"]


def test_the_shipped_profile_records_its_lexile_frequency_source():
    # "none", "bundled" and "wordfreq" are three different scales. A corpus
    # that does not say which one it used cannot be compared against safely,
    # and a Lexile the corpus never measured cannot be compared at all.
    assert "lexile_frequency_source" in PROFILE


def test_the_shipped_profile_carries_lexile_so_the_metric_is_comparable():
    # Enabling Lexile used to give a number with no percentile, because
    # textgrader/corpus.py never passed a frequency source and so the corpus
    # never measured it: the metric was measurable and never comparable.
    source = PROFILE["lexile_frequency_source"]
    assert source != "none"
    assert "lexile" in PROFILE["distributions"]
    assert any("lexile" in row for row in PROFILE["books"])
