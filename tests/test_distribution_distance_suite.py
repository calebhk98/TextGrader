"""A full battery of distances/tests against the corpus's pooled distributions.

Every channel must separate two inputs it should (a sample drawn from the
same generator as the corpus reference vs. one that is clearly different),
and every channel must report zero or its minimum on identical inputs. Both
are checked directly, at the channel-function level, with synthetic samples
whose true relationship (same distribution, mean shift, variance shift,
bimodal, tail-only contamination) is known -- rather than only indirectly
through prose, where controlling the exact shape of the sentence-length
distribution is much harder.
"""

import json
import math
import random
from pathlib import Path

import pytest

from textgrader import optional, stats
from textgrader.corpus import build_profile
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY
from textgrader.metrics import distribution_distance_suite as dds

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _require_scipy():
    """Skip a scipy-backed test under ``TEXTGRADER_DISABLE_OPTIONAL=all`` etc.

    Mirrors the pattern every other suite's tests use (see
    ``test_stylometry_suite.py``, ``test_randomness_suite.py``): the
    production code already degrades to ``unavailable()`` without scipy (see
    ``test_a_missing_scipy_degrades_only_the_scipy_backed_channels`` below),
    this just keeps a test that specifically exercises the scipy math from
    failing on ``None`` when scipy itself has been disabled for the run.
    """

    if not optional.have("scipy.stats"):
        pytest.skip("scipy not usable in this environment")


# --------------------------------------------------------------- synthetic data

def _normal_ints(seed, n, mean=14, spread=6, low=2, high=60):
    rng = random.Random(seed)
    return [float(max(low, min(high, round(rng.gauss(mean, spread))))) for _ in range(n)]


def _bimodal_ints(seed, n, low_centre=4, high_centre=24, spread=2):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        centre = low_centre if rng.random() < 0.5 else high_centre
        out.append(float(max(1, round(rng.gauss(centre, spread)))))
    return out


def _tail_contaminated(seed, n, mean=14, spread=6, fraction=0.03, tail_value=120):
    rng = random.Random(seed)
    base = _normal_ints(seed, n, mean, spread)
    for index in rng.sample(range(n), max(1, int(n * fraction))):
        base[index] = float(tail_value)
    return base


@pytest.fixture(scope="module")
def reference_curve():
    return stats.quantile_curve(_normal_ints(1, 3000))


@pytest.fixture(scope="module")
def synthetic_samples():
    return {
        "same": _normal_ints(2, 400),
        "mean_shift": _normal_ints(3, 400, mean=26),
        "variance_shift": _normal_ints(4, 400, spread=18),
        "bimodal": _bimodal_ints(5, 400),
        "tail": _tail_contaminated(6, 400),
    }


ALL_FEATURES = {key: True for key in dds.DEFAULT_FEATURES}


# ------------------------------------------------------ channel-level separation

def test_wasserstein_separates_and_is_zero_on_identical(reference_curve, synthetic_samples):
    _require_scipy()
    same = dds._wasserstein_channel("s", "S", "words", synthetic_samples["same"],
                                    reference_curve, 400)[0]["value"]
    shifted = dds._wasserstein_channel("s", "S", "words", synthetic_samples["mean_shift"],
                                       reference_curve, 400)[0]["value"]
    identical = dds._wasserstein_channel("s", "S", "words", list(reference_curve),
                                         reference_curve, len(reference_curve))[0]["value"]
    assert same < shifted
    assert identical == pytest.approx(0.0, abs=1e-9)


def test_energy_separates_and_is_zero_on_identical(reference_curve, synthetic_samples):
    _require_scipy()
    same = dds._energy_channel("s", "S", "words", synthetic_samples["same"],
                               reference_curve, 400)[0]["value"]
    shifted = dds._energy_channel("s", "S", "words", synthetic_samples["variance_shift"],
                                  reference_curve, 400)[0]["value"]
    identical = dds._energy_channel("s", "S", "words", list(reference_curve),
                                    reference_curve, len(reference_curve))[0]["value"]
    assert same < shifted
    assert identical == pytest.approx(0.0, abs=1e-9)


def test_ks_statistic_separates_and_is_zero_on_identical(reference_curve, synthetic_samples):
    _require_scipy()
    same = dds._ks_channel("s", "S", synthetic_samples["same"], reference_curve, 400)
    shifted = dds._ks_channel("s", "S", synthetic_samples["bimodal"], reference_curve, 400)
    identical = dds._ks_channel("s", "S", list(reference_curve), reference_curve,
                                len(reference_curve))
    assert same[0]["value"] < shifted[0]["value"]
    assert identical[0]["value"] == pytest.approx(0.0, abs=1e-9)
    # The statistic is the effect size; the p-value is its own, separately
    # flagged finding, never a stand-in for it (implementation detail #4).
    assert same[1]["metric_id"].endswith("_ks_pvalue")
    assert same[1]["sample_size_sensitive"] is True
    assert same[0]["sample_size_sensitive"] is not True


def test_cvm_and_anderson_darling_separate_and_are_stable_on_identical(
        reference_curve, synthetic_samples):
    _require_scipy()
    same_cvm = dds._cvm_channel("s", "S", synthetic_samples["same"], reference_curve, 400, 5)
    shifted_cvm = dds._cvm_channel("s", "S", synthetic_samples["mean_shift"], reference_curve,
                                   400, 5)
    assert same_cvm[0]["value"] < shifted_cvm[0]["value"]
    assert same_cvm[0]["value"] is not None and math.isfinite(same_cvm[0]["value"])

    same_ad = dds._anderson_darling_channel("s", "S", synthetic_samples["same"], reference_curve,
                                            400, 5)
    shifted_ad = dds._anderson_darling_channel("s", "S", synthetic_samples["mean_shift"],
                                               reference_curve, 400, 5)
    assert same_ad[0]["value"] < shifted_ad[0]["value"]
    assert same_ad[0]["value"] is not None and math.isfinite(same_ad[0]["value"])


def test_cvm_and_anderson_darling_refuse_below_min_distinct_values(reference_curve):
    """Ties inflate both statistics rather than reflecting a real difference.

    Two independent draws from the SAME narrow, heavily-tied set produced a
    Cramer-von Mises p-value of 1.5e-9 during development -- a false
    "highly significant" result between identical generators. Both channels
    must refuse below ``min_distinct_values`` rather than publish that.
    """

    degenerate = [12.0] * 200
    cvm = dds._cvm_channel("s", "S", degenerate, [12.0] * 101, 200, 5)
    ad = dds._anderson_darling_channel("s", "S", degenerate, [12.0] * 101, 200, 5)
    assert cvm[0]["value"] is None and "distinct" in cvm[0]["warning"]
    assert cvm[1]["value"] is None
    assert ad[0]["value"] is None and "distinct" in ad[0]["warning"]
    assert ad[1]["value"] is None


def test_histogram_family_separates_and_is_zero_on_identical(reference_curve, synthetic_samples):
    same = {f["metric_id"]: f["value"] for f in
           dds._histogram_channels("s", "S", synthetic_samples["same"], reference_curve, 400,
                                   ALL_FEATURES, 16, 0.5)}
    shifted = {f["metric_id"]: f["value"] for f in
              dds._histogram_channels("s", "S", synthetic_samples["mean_shift"], reference_curve,
                                      400, ALL_FEATURES, 16, 0.5)}
    identical = {f["metric_id"]: f["value"] for f in
                dds._histogram_channels("s", "S", list(reference_curve), reference_curve,
                                        len(reference_curve), ALL_FEATURES, 16, 0.5)}
    for suffix in ("jensen_shannon", "kl_forward", "kl_reverse", "kl_symmetric",
                  "hellinger", "bhattacharyya", "total_variation"):
        metric_id = f"style.distribution_s_{suffix}"
        assert same[metric_id] < shifted[metric_id], suffix
        assert identical[metric_id] == pytest.approx(0.0, abs=1e-6), suffix
        assert math.isfinite(same[metric_id]) and math.isfinite(shifted[metric_id])


def test_mmd_separates_and_is_near_zero_on_identical(reference_curve, synthetic_samples):
    same = dds._mmd_channel("s", "S", synthetic_samples["same"], reference_curve, 400, 500)[0]
    shifted = dds._mmd_channel("s", "S", synthetic_samples["bimodal"], reference_curve, 400,
                               500)[0]
    identical = dds._mmd_channel("s", "S", list(reference_curve), reference_curve,
                                 len(reference_curve), 500)[0]
    assert same["value"] < shifted["value"]
    # The unbiased estimator can be slightly negative on identical inputs
    # (documented in the module docstring); it must stay close to zero, not
    # be clamped or blow up.
    assert abs(identical["value"]) < 0.05


def test_quantile_vector_separates_and_is_zero_on_identical(reference_curve, synthetic_samples):
    same = dds._quantile_vector_channel("s", "S", "words", synthetic_samples["same"],
                                        reference_curve, 400,
                                        dds.DEFAULT_QUANTILE_VECTOR_POINTS)[0]["value"]
    shifted = dds._quantile_vector_channel("s", "S", "words", synthetic_samples["mean_shift"],
                                           reference_curve, 400,
                                           dds.DEFAULT_QUANTILE_VECTOR_POINTS)[0]["value"]
    identical = dds._quantile_vector_channel("s", "S", "words", list(reference_curve),
                                             reference_curve, len(reference_curve),
                                             dds.DEFAULT_QUANTILE_VECTOR_POINTS)[0]["value"]
    assert same < shifted
    assert identical == pytest.approx(0.0, abs=1e-9)


def test_tail_channels_separate_and_are_zero_on_identical(reference_curve, synthetic_samples):
    same = dds._tail_channel("s", "S", "words", synthetic_samples["same"], reference_curve, 400,
                             0.10)
    shifted = dds._tail_channel("s", "S", "words", synthetic_samples["variance_shift"],
                                reference_curve, 400, 0.10)
    identical = dds._tail_channel("s", "S", "words", list(reference_curve), reference_curve,
                                  len(reference_curve), 0.10)
    assert same[0]["value"] <= shifted[0]["value"]
    assert same[1]["value"] <= shifted[1]["value"]
    assert identical[0]["value"] == pytest.approx(0.0, abs=1e-9)
    assert identical[1]["value"] == pytest.approx(0.0, abs=1e-9)


def test_optimal_transport_separates_and_is_zero_on_identical(reference_curve, synthetic_samples):
    same = dds._optimal_transport_channel("s", "S", "words", synthetic_samples["same"],
                                          reference_curve, 400)[0]["value"]
    shifted = dds._optimal_transport_channel("s", "S", "words", synthetic_samples["mean_shift"],
                                             reference_curve, 400)[0]["value"]
    identical = dds._optimal_transport_channel("s", "S", "words", list(reference_curve),
                                               reference_curve, len(reference_curve))[0]["value"]
    assert same < shifted
    assert identical == pytest.approx(0.0, abs=1e-9)


def test_tail_only_contamination_is_caught_by_wasserstein_and_optimal_transport_but_not_ks(
        reference_curve, synthetic_samples):
    """Different distances are meant to disagree; this is the case the task
    brief names explicitly ("tail-only contamination... verify different
    distances rank these cases differently").

    97% of the contaminated sample is drawn from the exact same generator as
    the corpus reference, with a few points moved far into the tail. A
    distance built from moving mass (Wasserstein, quadratic optimal
    transport) must notice; the Kolmogorov-Smirnov statistic, which only
    tracks the single worst CDF gap, is not built to and should barely move.
    """

    _require_scipy()
    same_w = dds._wasserstein_channel("s", "S", "words", synthetic_samples["same"],
                                      reference_curve, 400)[0]["value"]
    tail_w = dds._wasserstein_channel("s", "S", "words", synthetic_samples["tail"],
                                      reference_curve, 400)[0]["value"]
    same_ot = dds._optimal_transport_channel("s", "S", "words", synthetic_samples["same"],
                                             reference_curve, 400)[0]["value"]
    tail_ot = dds._optimal_transport_channel("s", "S", "words", synthetic_samples["tail"],
                                             reference_curve, 400)[0]["value"]
    same_ks = dds._ks_channel("s", "S", synthetic_samples["same"], reference_curve, 400)[0]["value"]
    tail_ks = dds._ks_channel("s", "S", synthetic_samples["tail"], reference_curve, 400)[0]["value"]
    mean_shift_ks = dds._ks_channel("s", "S", synthetic_samples["mean_shift"], reference_curve,
                                    400)[0]["value"]

    assert tail_w > same_w * 3
    assert tail_ot > same_ot * 10
    # KS moved far less, proportionally, than it does for an across-the-board
    # mean shift of comparable overall size: the two channels disagree, and
    # that disagreement is the point.
    assert (tail_ks - same_ks) < (mean_shift_ks - same_ks) / 4


def test_distance_correlation_separates_association_from_independence():
    rng = random.Random(0)
    x = [rng.gauss(0, 1) for _ in range(200)]
    correlated = [2 * value + rng.gauss(0, 0.1) for value in x]
    independent = [rng.gauss(0, 1) for _ in range(200)]
    high = dds._distance_correlation(x, correlated)
    low = dds._distance_correlation(x, independent)
    assert high > 0.9
    assert low < 0.3
    assert high > low


def test_distance_correlation_handles_no_variation():
    assert dds._distance_correlation([5.0] * 20, list(range(20))) is None
    assert dds._distance_correlation([1.0, 2.0], [1.0, 2.0]) is None  # below n=4


# --------------------------------------------------------------- full pipeline

@pytest.fixture(scope="module")
def corpus_profile(tmp_path_factory, prose):
    directory = tmp_path_factory.mktemp("distdist")
    for index in range(10):
        (directory / f"book{index:02d}.txt").write_text(prose(400 + index, 150),
                                                        encoding="utf-8")
    return build_profile([directory], built_at="2026-01-01T00:00:00Z")


def _findings(text, profile, features=None):
    analysis = DocumentAnalysis.from_text(text, comparison_unit="chapter")
    config = {"features": {**ALL_FEATURES, **(features or {})}}
    return {item["metric_id"]: item for item in dds.measure(analysis, config=config,
                                                            profile=profile)}


def test_a_matching_text_is_closer_than_a_metronomic_one(corpus_profile, prose):
    matching = _findings(prose(999, 60), corpus_profile)
    flat = _findings("\n\n".join(" ".join(["word"] * 12) + "." for _ in range(200)),
                     corpus_profile)
    # Dependency-free channels: always exercised, even with scipy disabled.
    for suffix in ("jensen_shannon", "hellinger", "total_variation", "quantile_vector",
                  "optimal_transport"):
        metric_id = f"style.distribution_sentence_words_{suffix}"
        assert matching[metric_id]["value"] < flat[metric_id]["value"], suffix
    if optional.have("scipy.stats"):
        for suffix in ("wasserstein", "energy", "ks"):
            metric_id = f"style.distribution_sentence_words_{suffix}"
            assert matching[metric_id]["value"] < flat[metric_id]["value"], suffix


def test_every_metric_id_is_present_and_stable(corpus_profile, prose):
    found = _findings(prose(42, 80), corpus_profile)
    for source in dds.DEFAULT_SOURCES:
        for feature, suffixes in dds.FEATURE_SUFFIXES.items():
            for suffix in suffixes:
                assert f"style.distribution_{source}_{suffix}" in found
    assert "style.distribution_sentence_distance_correlation" in found
    assert "style.distribution_paragraph_distance_correlation" in found


def test_too_few_items_is_insufficient_data_not_a_noisy_number(corpus_profile):
    found = _findings("One. Two. Three.", corpus_profile)
    item = found["style.distribution_sentence_words_wasserstein"]
    assert item["value"] is None
    assert "below the" in item["warning"]


def test_a_profile_without_pooled_items_says_so():
    found = _findings("A sentence here. And another one there. A third one, yes.", {})
    item = found["style.distribution_sentence_words_wasserstein"]
    assert item["value"] is None
    assert "rebuild" in item["warning"]
    # distance_correlation needs no corpus at all, so it is unaffected.
    dcor = found["style.distribution_sentence_distance_correlation"]
    assert dcor["value"] is None  # too few paired sentences in this tiny text
    assert "paired" in dcor["warning"]


def test_no_profile_at_all_degrades_every_channel(prose):
    found = _findings(prose(7, 60), None)
    assert all(item["value"] is None for metric_id, item in found.items()
              if metric_id != "style.distribution_sentence_distance_correlation"
              and metric_id != "style.distribution_paragraph_distance_correlation")


def test_features_can_be_switched_off_independently(corpus_profile, prose):
    found = _findings(prose(5, 60), corpus_profile,
                      features={"wasserstein": False, "mmd": False})
    assert "style.distribution_sentence_words_wasserstein" not in found
    assert "style.distribution_sentence_words_mmd" not in found
    assert "style.distribution_sentence_words_energy" in found


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_survives_degenerate_documents(text, corpus_profile):
    found = _findings(text, corpus_profile)
    assert found  # never raises, never empty


# --------------------------------------------------------------- registry/config

def test_registered_and_off_by_default():
    spec = REGISTRY["distribution_distance_suite"]
    assert spec.family == "distribution_shape"
    assert spec.cost == "moderate"
    assert spec.needs_parse is False
    assert spec.needs_model is False
    assert "sentence_transformers" not in spec.requires
    assert "spacy" not in spec.requires
    assert set(spec.defaults["features"]) == set(dds.DEFAULT_FEATURES)

    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    entry = config["metrics"]["distribution_distance_suite"]
    assert entry["enabled"] is False
    for key, value in spec.defaults.items():
        assert entry[key] == value, key


def test_corpus_profiling_never_runs_this_suite_unless_enabled(tmp_path, prose):
    """The gating rule: an OFF suite must add zero cost to profiling.

    ``distribution_distance_suite`` is not ``needs_parse``/``needs_model``, so
    :mod:`textgrader.corpus` WOULD happily call it if it were enabled; the
    guarantee this test actually checks is the one that matters by default --
    a profile built from the project's default (suite disabled) config never
    invokes this module at all.
    """

    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "book.txt").write_text(prose(1, 40), encoding="utf-8")
    profile = build_profile([directory], built_at="2026-01-01T00:00:00Z",
                            metrics={"distribution_distance_suite": {"enabled": False}})
    assert not any(key.startswith("style.distribution_") for key in profile["distributions"])
    assert "distribution_distance_suite" not in profile.get("feature_profiles", {})


def test_a_missing_scipy_degrades_only_the_scipy_backed_channels(monkeypatch, corpus_profile,
                                                                 prose):
    from textgrader import optional
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "scipy.stats")
    optional.reset_cache()
    try:
        found = _findings(prose(3, 60), corpus_profile)
    finally:
        optional.reset_cache()
    assert found["style.distribution_sentence_words_wasserstein"]["value"] is None
    assert "scipy" in found["style.distribution_sentence_words_wasserstein"]["warning"]
    # Dependency-free channels are unaffected.
    assert found["style.distribution_sentence_words_hellinger"]["value"] is not None
    assert found["style.distribution_sentence_words_quantile_vector"]["value"] is not None
