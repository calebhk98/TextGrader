"""The statistics layer decides what counts as a finding, so it is tested hard."""

import random

import pytest

from textgrader import stats


def test_two_texts_with_one_mean_have_different_shapes():
    flat = stats.summarize([14, 14, 14, 14, 14, 14])
    varied = stats.summarize([4, 7, 13, 29, 9, 22])
    assert flat["mean"] == varied["mean"] == 14
    assert flat["std"] == 0 and varied["std"] > 8
    assert flat["entropy"] == 0 and varied["entropy"] > 2


def test_alternation_is_visible_in_lag1_autocorrelation():
    # The biased estimator caps at -(n-1)/n for a perfect alternation.
    assert stats.autocorrelation([2, 30, 2, 30, 2, 30, 2, 30]) < -0.85
    assert stats.autocorrelation([2, 2, 2, 30, 30, 30, 2, 2, 2]) > 0.2


def test_bimodal_sample_is_flagged_and_split():
    values = [3, 4, 3, 4, 5, 40, 41, 42, 40, 39, 4, 41]
    assert stats.bimodality_coefficient(values) > 0.555
    split = stats.two_group_split(values)
    assert split["separation"] > 0.9
    assert split["low_centre"] < 10 < split["high_centre"]


def test_zero_mad_falls_back_to_the_interquartile_range():
    # Discrete metric, small corpus: MAD is zero but the tails still differ.
    reference = [3, 3, 3, 3, 3, 4, 5, 6, 7, 9, 3, 3]
    result = stats.compare(9, reference)
    assert result.method == "median/IQR"
    assert result.robust_distance is not None
    assert any("MAD was zero" in note for note in result.notes)


def test_no_variation_makes_no_outlier_claim():
    result = stats.compare(9, [4] * 30)
    assert result.method == "insufficient variation"
    assert result.outlier is None
    assert result.severity is None


def test_small_corpus_withholds_the_outlier_claim():
    result = stats.compare(100, [1, 2, 3, 4, 5])
    assert result.outlier is None
    assert result.confidence == "none"
    assert any("below the" in note for note in result.notes)


def test_a_modest_corpus_is_marked_low_confidence():
    result = stats.compare(100, list(range(1, 13)))
    assert result.outlier is True
    assert result.confidence == "low"


def test_a_large_corpus_supports_a_confident_claim():
    result = stats.compare(100, list(range(1, 41)))
    assert result.confidence == "high"


def test_direction_and_percentile():
    result = stats.compare(5, list(range(1, 41)))
    assert result.direction == "low"
    assert 0 < result.percentile < 50


def test_wasserstein_matches_scipy_when_present():
    left, right = [1, 2, 3, 4, 5], [3, 4, 5, 6, 7]
    value, _ = stats.wasserstein(left, right)
    assert value == pytest.approx(2.0, abs=1e-9)


def test_wasserstein_fallback_agrees_with_scipy(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "scipy.stats")
    from textgrader import optional
    optional.reset_cache()
    try:
        value, reason = stats.wasserstein([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
        assert value == pytest.approx(2.0, abs=1e-6)
        assert reason is not None
    finally:
        optional.reset_cache()


def test_histogram_buckets_are_named_unambiguously():
    assert stats.histogram([1, 5, 9, 50], [3, 8, 40]) == {
        "<=3": 1, "(3,8]": 1, "(8,40]": 1, ">40": 1}


def test_run_lengths():
    assert stats.run_lengths(["s", "s", "l", "s", "s", "s"]) == {"s": [2, 3], "l": [1]}


def test_empty_and_degenerate_samples_are_safe():
    assert stats.summarize([])["count"] == 0
    assert stats.autocorrelation([1]) is None
    assert stats.two_group_split([1, 1, 1]) is None
    assert stats.compare(None, [1, 2, 3]).severity is None
    assert stats.compare(1, []).corpus_median is None


def test_double_mad_reduces_to_mad_on_a_symmetric_sample():
    """Scaling each side separately must not change a well-behaved metric."""

    import statistics as stdlib
    rng = random.Random(11)
    sample = [rng.gauss(0, 1) for _ in range(2000)]
    median = stdlib.median(sample)
    pooled = stdlib.median([abs(value - median) for value in sample])
    low, high = stats.double_mad(sample, median)
    assert low == pytest.approx(pooled, rel=0.05)
    assert high == pytest.approx(pooled, rel=0.05)


def test_a_floored_skewed_metric_does_not_flag_its_own_population():
    """The defect leave-one-out validation over thirty published novels found.

    Second-person pronouns in narration are floored at zero and right-skewed.
    Half the corpus sits near zero, so the pooled MAD is tiny and every book in
    the long upper tail scores past 3.5. These are the real measured values.
    """

    corpus = [0.0, 0.0, 0.0, 0.03, 0.04, 0.05, 0.05, 0.2, 0.3, 0.4, 0.4, 0.5,
              0.77, 0.92, 0.98, 1.28, 2.0, 2.07, 2.23, 2.34, 3.46, 4.95, 5.52,
              10.34, 10.62, 11.1, 15.3, 15.53, 15.97, 21.48]
    flagged = 0
    for value in corpus:
        rest = list(corpus)
        rest.remove(value)
        result = stats.compare(value, rest)
        assert result.method == "median/double-MAD"
        flagged += bool(result.outlier)
    # Every one of these books belongs to the population. Under the pooled MAD
    # seven of thirty were called outliers.
    assert flagged <= 1, f"{flagged} of {len(corpus)} members flagged as outliers"


def test_an_asymmetric_scale_is_reported_not_hidden():
    """A different scale on each side is a fact the reader gets told."""

    corpus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
              1.1, 1.2, 1.3, 1.4, 1.5, 3.0, 5.0, 9.0, 14.0, 22.0]
    result = stats.compare(14.0, corpus)
    assert result.method == "median/double-MAD"
    assert result.scale is not None and result.scale > 0
    assert any("asymmetric" in note for note in result.notes)


def test_a_median_tied_distribution_falls_through_to_the_range():
    """More than half the corpus at one value leaves no spread to scale by."""

    corpus = [0.0] * 15 + [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0]
    result = stats.compare(21.0, corpus)
    assert result.method == "median/IQR"
    assert any("zero on both sides" in note for note in result.notes)


def test_a_genuine_outlier_is_still_caught():
    rng = random.Random(4)
    corpus = [rng.gauss(50, 5) for _ in range(40)]
    assert stats.compare(120.0, corpus).outlier is True
    assert stats.compare(51.0, corpus).outlier is False


def test_severity_saturates_instead_of_running_away():
    """A zero-inflated rate must not put a broken ruler at the top of the list.

    Ellipses per 1,000 words, measured across thirty published novels: half the
    corpus uses none at all, one book uses 9.7. Flagging that book is right.
    Claiming it is seventy-seven times more extreme than a finding that scored
    4.0 is not something the data supports.
    """

    corpus = [0.0] * 15 + [0.008, 0.009, 0.01, 0.025, 0.034, 0.088, 0.169,
                           0.23, 0.237, 0.254, 0.338, 0.432, 1.679, 5.102]
    result = stats.compare(9.71, corpus)
    assert result.outlier is True
    assert result.severity == stats.SEVERITY_CAP
    assert result.robust_distance > stats.SEVERITY_CAP
    assert any("saturated" in note for note in result.notes)


def test_an_ordinary_finding_is_not_saturated():
    rng = random.Random(3)
    corpus = [rng.gauss(50, 10) for _ in range(30)]
    result = stats.compare(90.0, corpus)
    assert result.outlier is True
    assert 0 < result.severity < stats.SEVERITY_CAP
    assert not any("saturated" in note for note in result.notes)
