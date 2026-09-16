"""The statistics layer decides what counts as a finding, so it is tested hard."""

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
