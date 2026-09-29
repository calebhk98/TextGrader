"""The calibrated authorship score, and the grouped text output that prints it."""

import random

import grade
from textgrader import authorship
from textgrader.results import Action, MetricResult, Report
from textgrader.stats import compare


def test_leave_one_out_percentiles_exclude_the_observation_itself():
    assert authorship._leave_one_out([1.0, 2.0, 3.0, 4.0]) == [0.0, 100 / 3, 200 / 3, 100.0]
    # Ties count half, as in stats.compare; a missing value stays missing.
    assert authorship._leave_one_out([5.0, 5.0, None]) == [50.0, 50.0, None]


def _profile(books=40, families=(("rhythm", 3), ("vocab", 60)), seed=3):
    rng = random.Random(seed)
    rows = [{"source_id": f"b{i}"} for i in range(books)]
    keys = []
    for family, count in families:
        for index in range(count):
            key = f"{family}.m{index}"
            keys.append((key, family))
            for row in rows:
                row[key] = rng.gauss(0, 1)
    distributions = {key: {"values": sorted(row[key] for row in rows)} for key, _ in keys}
    return {"books": rows, "distributions": distributions}, keys


def _compared(profile, keys, value_for):
    rows = []
    for key, family in keys:
        percentile = compare(value_for(key), profile["distributions"][key]["values"]).percentile
        rows.append((key, key, family, percentile))
    return rows


def test_a_text_at_every_corpus_median_outscores_the_corpus_and_one_past_every_max_scores_zero():
    profile, keys = _profile()
    median = lambda key: sorted(profile["distributions"][key]["values"])[20]
    typical = authorship.score(_compared(profile, keys, median), profile)
    assert typical["score"] == 100.0 and typical["measurements"] == len(keys)
    extreme = authorship.score(_compared(profile, keys, lambda key: 99.0), profile)
    assert extreme["score"] == 0.0
    assert extreme["families"]["rhythm"]["score"] == 0.0


def test_a_large_family_is_one_vote():
    # Exactly typical on 60 vocabulary measures, past the range on all 3 rhythm
    # ones.  Averaged per measurement that is 3/63 atypical, far more typical
    # than any real chapter (whose percentiles spread out, averaging about
    # 0.5).  Averaged by family it is (0 + 1) / 2: the rhythm miss counts as
    # much as the whole vocabulary match, and the family score shows where.
    profile, keys = _profile()
    value = lambda key: 99.0 if key.startswith("rhythm") else sorted(
        profile["distributions"][key]["values"])[20]
    result = authorship.score(_compared(profile, keys, value), profile)
    assert 0.45 < result["atypicality"] < 0.55
    assert result["families"]["rhythm"]["score"] == 0.0
    assert result["families"]["vocab"]["score"] == 100.0


def test_a_genuine_held_out_observation_scores_like_the_corpus():
    # Score each corpus row against the others: about half should land above
    # 50, which is what "calibrated" promises.
    profile, keys = _profile(books=41)
    scores = []
    for held in range(0, 41, 4):
        row = profile["books"][held]
        rest = {"books": [b for i, b in enumerate(profile["books"]) if i != held]}
        rest["distributions"] = {key: {"values": sorted(b[key] for b in rest["books"])}
                                 for key, _ in keys}
        scores.append(authorship.score(_compared(rest, keys, lambda key: row[key]), rest)["score"])
    assert 20 < sum(scores) / len(scores) < 80


def test_a_measurement_without_per_observation_values_is_left_out_and_counted():
    profile, keys = _profile(families=(("rhythm", 3),))
    rows = _compared(profile, keys, lambda key: 0.0) + [("shape.pooled", "shape.pooled",
                                                         "shape", 90.0)]
    result = authorship.score(rows, profile)
    assert result["uncalibrated"] == 1 and "shape" not in result["families"]


def test_word_rates_are_calibrated_from_their_table():
    profile, keys = _profile(families=(("rhythm", 2),))
    profile["word_rates"] = {"vocabulary": ["said"],
                             "per_book": [[float(i)] for i in range(len(profile["books"]))]}
    rows = _compared(profile, keys, lambda key: 0.0) + [
        ("lexical.word_rate.said", "lexical.word_rate.said", "vocabulary", 50.0)]
    assert authorship.score(rows, profile)["families"]["vocabulary"]["measurements"] == 1


def test_too_small_a_corpus_says_so():
    profile, keys = _profile(books=5)
    assert authorship.score(_compared(profile, keys, lambda key: 0.0), profile)["score"] is None


def _result(metric_id, family, value, reference):
    item = MetricResult(metric_id, metric_id, value, "words", family=family)
    item.corpus = compare(value, reference).to_dict()
    item.severity = item.corpus["severity"]
    item.action = Action.REVIEW if item.corpus["outlier"] else Action.INFORMATIONAL
    return item


def test_grouped_output_counts_every_family_and_prints_a_row_for_each_failure(capsys):
    reference = [float(n) for n in range(10, 41)]
    report = Report(source="draft.md", results=[
        _result("rhythm.a", "sentence_rhythm", 25.0, reference),
        _result("rhythm.b", "sentence_rhythm", 90.0, reference),
        _result("rhythm.c", "sentence_rhythm", 38.5, reference),
        MetricResult("rhythm.d", "rhythm.d", None, family="sentence_rhythm",
                     action=Action.UNAVAILABLE, warning="too short"),
    ])
    grade.render(report)
    out = capsys.readouterr().out
    assert ("sentence_rhythm (1 agree, 2 outside p10-p90: 1 critical, 1 review; "
            "1 not compared)") in out
    assert "critical | rhythm.b | 90 words |" in out
    assert "| 10 | 13 | 17.5 | 25 | 32.5 | 37 | 40" in out
    assert "rhythm.a" not in out.split("sentence_rhythm (")[1].split("\n\n")[0]
    assert "not compared [unavailable], too short: rhythm.d -" in out
    grade.render(report, detailed=True)
    assert "rhythm.a" in capsys.readouterr().out
