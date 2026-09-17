"""Dependency-free does not mean approximate.

TextGrader must run with no third-party package installed, so several
quantities are implemented here that SciPy and NumPy also provide.  That is
only defensible if the two agree, so these tests pin our implementations
against the reference ones and are skipped rather than guessed at when the
reference is missing.

The second half of the file pins the opposite case: the places where the
dependency-free path is genuinely a different, weaker measurement, so that
nobody reads one as the other.
"""

import random

import pytest

from textgrader import optional, stats

numpy = pytest.importorskip("numpy")
scipy_stats = pytest.importorskip("scipy.stats")


@pytest.fixture(scope="module")
def samples():
    rng = random.Random(5)
    return {
        "small": [rng.lognormvariate(2, .7) for _ in range(30)],
        "large": [rng.lognormvariate(2, .7) for _ in range(2000)],
        "discrete": [rng.randint(1, 40) for _ in range(500)],
    }


@pytest.mark.parametrize("key", ["small", "large"])
def test_skew_matches_scipy(samples, key):
    assert stats._skewness(samples[key]) == pytest.approx(
        scipy_stats.skew(samples[key], bias=False), rel=1e-12)


@pytest.mark.parametrize("key", ["small", "large"])
def test_excess_kurtosis_matches_scipy(samples, key):
    assert stats._kurtosis_excess(samples[key]) == pytest.approx(
        scipy_stats.kurtosis(samples[key], bias=False), rel=1e-12)


@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_quantiles_match_numpy(samples, q):
    for key in ("small", "large", "discrete"):
        assert stats.quantile(samples[key], q) == pytest.approx(
            float(numpy.percentile(samples[key], q * 100)), rel=1e-12)


def test_entropy_matches_scipy(samples):
    from collections import Counter
    values = [int(item) for item in samples["large"]]
    counts = list(Counter(values).values())
    assert stats.shannon_entropy(values) == pytest.approx(
        float(scipy_stats.entropy(counts, base=2)), rel=1e-12)


def test_spearman_matches_scipy_including_ties():
    from textgrader.metrics import drift_rolling
    rng = random.Random(9)
    x = list(range(20))
    y = [rng.random() for _ in x]
    y[3] = y[4] = y[5] = 0.5
    assert drift_rolling._rank_correlation(x, y) == pytest.approx(
        float(scipy_stats.spearmanr(x, y).statistic), rel=1e-12)


def test_wasserstein_fallback_matches_scipy(monkeypatch, samples):
    reference, _ = stats.wasserstein(samples["small"], samples["discrete"])
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "scipy.stats")
    optional.reset_cache()
    try:
        fallback, note = stats.wasserstein(samples["small"], samples["discrete"])
    finally:
        optional.reset_cache()
    assert note is not None
    assert fallback == pytest.approx(reference, rel=1e-6)


# --------------------------------------------------------------------------
# Where the dependency-free path is NOT equivalent, and must say so.

PARAPHRASE = ("The dog was extremely happy to see her.\n\n"
              "The canine was overjoyed at her arrival.\n\n"
              "He opened the window and the rain came in.\n\n"
              "Snow settled on the roof of the parked car.\n\n")


def _adjacent_similarity(text):
    from textgrader.document import DocumentAnalysis
    from textgrader.metrics import semantic_adjacent
    findings = semantic_adjacent.measure(DocumentAnalysis.from_text(text))
    top = findings[0]
    best = max((row["similarity"] for row in top["evidence"]), default=None)
    return best, top["warning"], top["evidence"][0]["backend"] if top["evidence"] else None


def test_lexical_fallback_cannot_see_a_paraphrase(monkeypatch):
    """The one place where doing without a dependency changes the answer.

    Lexical overlap is blind to a paraphrase that swaps every content word,
    which is exactly what this metric family exists to catch, so the finding
    has to say so rather than report a confident zero.
    """

    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "sentence_transformers")
    optional.reset_cache()
    try:
        similarity, warning, backend = _adjacent_similarity(PARAPHRASE)
    finally:
        optional.reset_cache()
    assert backend == "lexical"
    assert similarity is not None and similarity < 0.2
    assert "NOT semantic similarity" in warning
    assert "sentence-transformers" in warning


@pytest.mark.skipif(not optional.have("sentence_transformers"),
                    reason="sentence-transformers is not installed")
def test_embedding_backend_does_see_the_paraphrase():
    similarity, warning, backend = _adjacent_similarity(PARAPHRASE)
    assert backend == "embedding"
    assert similarity > 0.6
    assert "backend=embedding" in warning
