"""The aggregate: where the whole document sits, as one number.

A findings list answers "what is furthest out right now". It cannot answer
"did this revision help", because every metric can stay comfortably inside its
band while the whole moves.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grade
from textgrader.results import MetricResult, Polarity, Report


def _at(metric_id, percentile, polarity):
    return MetricResult(metric_id, metric_id, 1.0, "%", corpus={"percentile": percentile},
                        polarity=polarity)


def _report(*results):
    report = Report(source="test")
    report.results.extend(results)
    return report


def test_a_lower_is_better_metric_is_turned_the_right_way_up():
    # Sitting at the 30th percentile for "sentences under ten words" is the
    # same news as sitting at the 70th for "words per sentence". Averaging
    # them unoriented produces a number that means nothing.
    card = _report(_at("prose.wps", 70, Polarity.HIGHER),
                   _at("prose.u10", 30, Polarity.LOWER)).maturity()
    assert card["percentile"] == 70.0
    assert card["metric_count"] == 2


def test_neutral_metrics_are_excluded_not_defaulted():
    # More or fewer fronted subordinate clauses is a style choice. Giving it
    # an arbitrary direction would let a stylistic extreme move the aggregate.
    with_neutral = _report(_at("prose.fk", 50, Polarity.HIGHER),
                           _at("prose.front", 99, Polarity.NEUTRAL)).maturity()
    assert with_neutral["percentile"] == 50.0
    assert with_neutral["metric_count"] == 1


def test_a_metric_without_a_corpus_percentile_is_not_counted():
    card = _report(_at("prose.fk", 50, Polarity.HIGHER),
                   MetricResult("prose.lexile", "Lexile", None, "L",
                                polarity=Polarity.HIGHER)).maturity()
    assert card["metric_count"] == 1


def test_no_comparable_metric_gives_no_number_rather_than_zero():
    card = _report(_at("prose.front", 50, Polarity.NEUTRAL)).maturity()
    assert card["percentile"] is None
    assert card["metric_count"] == 0


def test_the_aggregate_moves_when_no_single_metric_flags():
    """The failure this exists to catch.

    A revision that pushes every metric a little toward the corpus floor,
    while leaving each one an inlier, produces no finding at all.
    """

    before = _report(*[_at(f"prose.{index}", 70, Polarity.HIGHER) for index in range(18)])
    after = _report(*[_at(f"prose.{index}", 55, Polarity.HIGHER) for index in range(18)])
    assert before.maturity()["percentile"] == 70.0
    assert after.maturity()["percentile"] == 55.0
    # Neither run has a single finding to look at.
    assert before.summary()["top_findings"] == after.summary()["top_findings"] == []


def test_polarity_defaults_to_neutral():
    # A metric is kept out of the aggregate until someone decides what its
    # direction means, rather than being swept in with an assumed one.
    assert MetricResult("x", "X").polarity is Polarity.NEUTRAL


def test_the_four_inverted_metrics_point_down():
    # These read inverted against a tool that orients them. Both tools measure
    # the same thing; one reports distributional position and the other
    # reports it oriented.
    for key in ("top100", "u10", "shortruns", "simple"):
        assert grade.METRIC_NAMES[key][3] is Polarity.LOWER, key


def test_eighteen_metrics_are_graded_and_the_style_choices_are_not():
    graded = [key for key, entry in grade.METRIC_NAMES.items()
              if entry[3] is not Polarity.NEUTRAL]
    assert len(graded) == 18
    for key in ("front", "and2", "andrate", "negative"):
        assert grade.METRIC_NAMES[key][3] is Polarity.NEUTRAL, key
    for key in ("_words", "_sentences", "_paragraphs", "_transcript"):
        assert grade.METRIC_NAMES[key][3] is Polarity.NEUTRAL, key


def test_every_core_metric_has_a_polarity_decision():
    for key, entry in grade.METRIC_NAMES.items():
        assert len(entry) == 4, key
        assert isinstance(entry[3], Polarity), key


PROFILE = {
    "peter_pan": {"fk": 6.0, "u10": 30.0, "wps": 16.0},
    "alice": {"fk": 5.0, "u10": 40.0, "wps": 14.0},
    "little_women": {"fk": 9.0, "u10": 20.0, "wps": 22.0},
    "treasure_island": {"fk": 8.0, "u10": 25.0, "wps": 20.0},
}


def test_a_benchmark_loss_is_ranked_by_corpus_standard_deviations():
    # A 5-point gap in one unit and a 0.5-point gap in another are not
    # comparable until both are in units of how much the corpus itself varies.
    got = grade.benchmark_comparison(PROFILE, {"fk": 5.0, "u10": 35.0, "wps": 15.0}, "peter_pan")
    assert got["error"] is None
    assert got["of"] == 3
    assert {gap["metric_id"] for gap in got["gaps"]} == {"prose.fk", "prose.u10", "prose.wps"}
    sizes = [gap["gap_sd"] for gap in got["gaps"]]
    assert sizes == sorted(sizes, reverse=True)


def test_beating_the_benchmark_is_not_a_gap():
    got = grade.benchmark_comparison(PROFILE, {"fk": 9.0, "u10": 10.0, "wps": 25.0}, "peter_pan")
    assert got["gaps"] == []
    assert got["lost"] == 0
    assert got["of"] == 3


def test_a_lower_is_better_metric_loses_by_being_higher():
    # u10 is "sentences under ten words": more of them is behind, not ahead.
    got = grade.benchmark_comparison(PROFILE, {"u10": 45.0}, "peter_pan")
    assert [gap["metric_id"] for gap in got["gaps"]] == ["prose.u10"]
    got = grade.benchmark_comparison(PROFILE, {"u10": 15.0}, "peter_pan")
    assert got["gaps"] == []


def test_an_unknown_benchmark_name_is_reported_not_silent():
    got = grade.benchmark_comparison(PROFILE, {"fk": 5.0}, "peter_panne")
    assert "peter_panne" in got["error"]
    assert "peter_pan" in got["available"]


def test_no_benchmark_configured_means_no_comparison():
    assert grade.benchmark_comparison(PROFILE, {"fk": 5.0}, None) is None
    assert grade.benchmark_comparison(None, {"fk": 5.0}, "peter_pan") is None


def test_neutral_metrics_are_left_out_of_the_benchmark_too():
    profile = {**PROFILE, "peter_pan": {**PROFILE["peter_pan"], "andrate": 4.0}}
    got = grade.benchmark_comparison(profile, {"andrate": 1.0}, "peter_pan")
    assert got["of"] == 0
    assert got["gaps"] == []


def test_maturity_reaches_the_json_payload(tmp_path, capsys, sample_text):
    manuscript = tmp_path / "draft.md"
    manuscript.write_text(sample_text, encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_profile": "", "metrics": {}}), encoding="utf-8")
    assert grade.main([str(manuscript), "--config", str(config), "--json"]) == 0
    maturity = json.loads(capsys.readouterr().out)["summary"]["maturity"]
    assert set(maturity) == {"percentile", "metric_count", "benchmark"}


def test_benchmark_is_a_recognised_config_key():
    # Without this it would be silently accepted and do nothing, which is the
    # bug the configuration check exists for.
    from textgrader.project import config_issues
    assert config_issues({"analysis": {"benchmark": "peter_pan"}}) == []
