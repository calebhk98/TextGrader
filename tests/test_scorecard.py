"""The census at the end of a run, and its denominator.

A severity-ranked findings list cannot tell a clean run from a broken one:
"2 to review" is the same sentence whether 91 things were measured or 65.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import grade
from textgrader.results import Action, MetricResult, Report, StatusType


def _report(*results):
    report = Report(source="test")
    report.results.extend(results)
    return report


def _ok(metric_id, family="prose"):
    return MetricResult(metric_id, metric_id, 1.0, "unit", family=family)


def _review(metric_id, severity=3.0, family="prose"):
    return MetricResult(metric_id, metric_id, 1.0, "unit", status="review",
                        status_type=StatusType.CORPUS_OUTLIER, severity=severity,
                        family=family)


def _errored(metric_id, family="prose"):
    return MetricResult(metric_id, metric_id, status="error",
                        status_type=StatusType.INTERNAL_ERROR, error="boom", family=family)


def _unavailable(metric_id, family="prose"):
    return MetricResult(metric_id, metric_id, status="unavailable",
                        status_type=StatusType.UNAVAILABLE, warning="no package",
                        family=family)


def test_the_denominator_is_reported():
    card = _report(_ok("a"), _ok("b"), _review("c")).scorecard()
    assert card["passing"] == 2
    assert card["failing"] == 1
    assert card["measured"] == 3
    assert card["passing_share"] == 66.7


def test_a_measure_that_did_not_happen_is_neither_a_pass_nor_a_failure():
    # This is the whole design. A metric that errored is not a pass, and a
    # metric that was unavailable is not a failure. Folding either into the
    # measured count is how a tool reports "23 of 23 passing (100%)" while
    # three of its measures are crashing.
    card = _report(_ok("a"), _errored("b"), _unavailable("c")).scorecard()
    assert card["measured"] == 1
    assert card["passing"] == 1
    assert card["failing"] == 0
    assert card["not_taken"] == 2
    assert card["passing_share"] == 100.0


def test_silent_measure_loss_moves_a_visible_number():
    """The failure this exists to catch, in the shape bug 1 produced.

    Turning the bundled Lexile source on used to raise NameError inside the
    core analysis, which grade.py caught at the whole-analysis level: 24 core
    metrics vanished and were replaced by one error result. The run still
    exited zero and still printed a summary.
    """

    healthy = _report(*[_ok(f"prose.{index}") for index in range(24)])
    broken = _report(_errored("prose.analysis"))
    assert healthy.scorecard()["measured"] == 24
    assert broken.scorecard()["measured"] == 0
    assert broken.scorecard()["not_taken"] == 1
    # Both runs report zero failures. Only the denominator and not_taken say
    # which of them actually measured anything.
    assert healthy.scorecard()["failing"] == broken.scorecard()["failing"] == 0


def test_failures_are_grouped_by_family():
    # Eleven findings in one family is a habit; eleven across eleven families
    # is noise.
    card = _report(_review("a", family="syntax"), _review("b", family="syntax"),
                   _review("c", family="lexical"), _ok("d", family="syntax")).scorecard()
    assert card["by_family"]["syntax"] == {"passing": 1, "failing": 2, "not_taken": 0}
    assert card["by_family"]["lexical"] == {"passing": 0, "failing": 1, "not_taken": 0}


def test_a_configuration_issue_is_counted_apart_from_the_measures():
    # A misspelt key is worth seeing, and it is not a measurement that failed
    # to happen. Putting it in not_taken would blunt that number's one job.
    configuration = MetricResult("config.banana", "Configuration", status="unavailable",
                                 status_type=StatusType.UNAVAILABLE,
                                 family="configuration", warning="unrecognised")
    card = _report(_ok("a"), configuration).scorecard()
    assert card["measured"] == 1
    assert card["not_taken"] == 0
    assert card["configuration_issues"] == 1


def test_every_action_is_classified():
    # A new Action cannot be added without deciding what it counts as.
    from textgrader.results import _FAILING, _NOT_TAKEN, _PASSING
    buckets = [_PASSING, _FAILING, _NOT_TAKEN]
    assert set().union(*buckets) == set(Action)
    for left in range(len(buckets)):
        for right in range(left + 1, len(buckets)):
            assert not buckets[left] & buckets[right]


def test_an_empty_run_does_not_divide_by_zero():
    card = _report().scorecard()
    assert card["measured"] == 0
    assert card["passing_share"] is None


def test_the_scorecard_reaches_the_json_payload(tmp_path, capsys, sample_text):
    manuscript = tmp_path / "draft.md"
    manuscript.write_text(sample_text, encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_profile": "", "metrics": {}}), encoding="utf-8")
    assert grade.main([str(manuscript), "--config", str(config), "--json"]) == 0
    card = json.loads(capsys.readouterr().out)["summary"]["scorecard"]
    assert card["measured"] > 0
    assert set(card) >= {"passing", "failing", "measured", "not_taken", "by_family"}


def test_a_run_with_a_bad_scorecard_still_exits_zero(tmp_path, capsys, sample_text):
    # A census is not a quality score and does not make a run fail.
    manuscript = tmp_path / "draft.md"
    manuscript.write_text(sample_text, encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_profile": "", "metrics": {}}), encoding="utf-8")
    assert grade.main([str(manuscript), "--config", str(config)]) == 0
    assert "SCORECARD" in capsys.readouterr().out


def test_not_taken_prints_even_at_zero(capsys):
    # A count that only appears when it is interesting is a count nobody
    # learns to look for.
    grade.print_scorecard(_report(_ok("a")).scorecard())
    assert "0 not taken" in capsys.readouterr().out
