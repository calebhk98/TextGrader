"""Reading-level targets that move across a book."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chapter_report
from textgrader.bands import BandError, compile_bands, judge, parse_range


def _bands(*entries, metric="fk"):
    metric, bands, problems = compile_bands({"metric": metric, "bands": list(entries)})
    assert problems == [], problems
    return bands


def test_a_range_is_inclusive_at_both_ends():
    assert parse_range("11-15") == {11, 12, 13, 14, 15}
    assert parse_range("7") == {7}
    assert parse_range(7) == {7}


def test_a_backwards_range_is_rejected():
    with pytest.raises(BandError):
        parse_range("15-11")


def test_the_band_average_is_judged_not_each_chapter():
    """The detail that made two bands permanently unfixable.

    An earlier version judged the average and ALSO failed the band whenever
    its own footnote listed a chapter below the floor. The average cleared,
    a chapter did not, and no revision could satisfy both.
    """

    bands = _bands({"chapters": "1-3", "floor": 5.5})
    got = judge(bands, {1: 5.0, 2: 5.4, 3: 6.9})[0]
    assert got["average"] == 5.77
    assert got["status"] == "within"
    # The chapters under the floor are still named, as a pointer.
    assert got["outside"] == [1, 2]


def test_a_band_average_under_its_floor_fails():
    got = judge(_bands({"chapters": "1-3", "floor": 7.0}), {1: 5.0, 2: 5.4, 3: 6.0})[0]
    assert got["status"] == "under"
    assert got["outside"] == [1, 2, 3]


def test_a_band_can_carry_a_ceiling():
    # The original had floors only, which is why nothing in it flagged three
    # chapters running at Flesch-Kincaid 11 in a book aimed at ninth graders.
    got = judge(_bands({"chapters": "1-3", "floor": 7.2, "ceiling": 9.5}),
                {1: 10.8, 2: 11.2, 3: 9.0})[0]
    assert got["status"] == "over"
    assert got["outside"] == [1, 2]


def test_a_flat_target_hides_what_bands_find():
    """The case for bands at all, in numbers.

    A book whose reading level is supposed to climb reads fine on a single
    whole-book average and badly on the band that matters.
    """

    values = {**{index: 5.5 for index in range(1, 11)},
              **{index: 8.9 for index in range(11, 15)}}
    whole_book = sum(values.values()) / len(values)
    assert whole_book < 7.0          # under a flat 7.0 target: looks fine
    late = judge(_bands({"chapters": "11-14", "floor": 7.2, "ceiling": 8.0}), values)[0]
    assert late["status"] == "over"  # the band says otherwise


def test_an_exempt_chapter_leaves_the_average_and_is_reported():
    # An exemption is a decision on the record, not a number going missing.
    bands = _bands({"chapters": "1-3", "floor": 7.0, "exempt": {"2": "a plain flashback"}})
    got = judge(bands, {1: 7.5, 2: 1.0, 3: 7.5})[0]
    assert got["status"] == "within"
    assert got["average"] == 7.5
    assert got["excused"] == [2]
    assert got["chapters"] == [1, 3]


def test_a_band_whose_chapters_are_all_exempt_says_so():
    bands = _bands({"chapters": "1-2", "floor": 7.0,
                    "exempt": {"1": "reason", "2": "reason"}})
    got = judge(bands, {1: 1.0, 2: 1.0})[0]
    assert got["status"] == "insufficient_data"
    assert got["average"] is None
    assert "exempt" in got["reason"]


def test_an_unmeasured_band_is_insufficient_data_not_a_failure():
    got = judge(_bands({"chapters": "40-45", "floor": 7.0}), {1: 5.0})[0]
    assert got["status"] == "insufficient_data"
    assert got["average"] is None


def test_overlapping_bands_are_reported():
    # A chapter in two bands is judged twice and reads as two findings for one
    # problem.
    _, _, problems = compile_bands({"metric": "fk", "bands": [
        {"chapters": "1-10", "floor": 5.0}, {"chapters": "8-15", "floor": 6.0}]})
    assert any("8" in problem and "both" in problem for problem in problems)


def test_a_band_with_neither_floor_nor_ceiling_is_reported():
    _, bands, problems = compile_bands({"metric": "fk",
                                        "bands": [{"chapters": "1-3"}]})
    assert bands == []
    assert any("neither floor nor ceiling" in problem for problem in problems)


def test_a_ceiling_below_its_floor_is_reported():
    _, bands, problems = compile_bands({"metric": "fk", "bands": [
        {"chapters": "1-3", "floor": 9.0, "ceiling": 5.0}]})
    assert bands == []
    assert any("ceiling below its floor" in problem for problem in problems)


def test_a_missing_metric_is_reported_rather_than_raised():
    _, _, problems = compile_bands({"bands": [{"chapters": "1-3", "floor": 5.0}]})
    assert any("metric" in problem for problem in problems)


def test_garbage_is_reported_rather_than_raised():
    for settings in ("nonsense", 7, None, {"metric": "fk"}, {"metric": "fk", "bands": []}):
        metric, bands, problems = compile_bands(settings)
        assert problems
        assert bands == []


def test_unnumbered_chapters_are_reported():
    rows = [{"number": None, "chapter": "prologue", "values": {"fk": 6.0}}]
    got = chapter_report.band_report(rows, {"metric": "fk",
                                            "bands": [{"chapters": "1-3", "floor": 5.0}]})
    assert any("starts with a number" in problem for problem in got["problems"])


def test_a_metric_no_chapter_produced_is_reported():
    rows = [{"number": 1, "chapter": "01_a", "values": {"wps": 12.0}}]
    got = chapter_report.band_report(rows, {"metric": "lexile",
                                            "bands": [{"chapters": "1-3", "floor": 5.0}]})
    assert any("lexile" in problem for problem in got["problems"])


def test_the_band_report_runs_end_to_end():
    rows = [{"number": index, "chapter": f"{index:02d}_ch", "values": {"fk": 5.0 + index}}
            for index in range(1, 7)]
    got = chapter_report.band_report(rows, {
        "metric": "fk",
        "bands": [{"chapters": "1-3", "floor": 5.0}, {"chapters": "4-6", "floor": 12.0}]})
    assert got["problems"] == []
    assert [band["status"] for band in got["bands"]] == ["within", "under"]


def test_chapter_bands_is_a_recognised_config_key():
    from textgrader.project import config_issues
    assert config_issues({"chapter_bands": {"metric": "fk", "bands": []}}) == []
