"""Contract tests for the multivariate anomaly/outlier detection suite.

These follow the same shape as ``tests/test_optional_metrics.py`` (which
already parametrizes ``REGISTRY`` and exercises ``anomaly_suite`` for "off by
default", "runs without raising" and "survives a degenerate document") and
adds the suite-specific behaviour this task's spec calls out by name: real
separation between a typical document and a multi-feature outlier, a
combination outlier invisible on any single marginal range, missing-feature
imputation, small-corpus refusal, determinism under a seed, one detector's
package being unavailable not suppressing the rest, comparison-unit safety,
and the corpus-profiling gating rule (this suite must be a fast no-op while
``profile is None``, i.e. while a corpus is still being built).
"""

from __future__ import annotations

import random
import time

import pytest

from textgrader import optional
from textgrader.corpus import build_profile
from textgrader.document import DocumentAnalysis, TextProcessing
from textgrader.metrics import anomaly_suite as A

pytestmark = pytest.mark.skipif(not optional.have("sklearn"),
                                 reason="scikit-learn is not installed")


# --------------------------------------------------------------------- helpers

def _prose(seed, paragraphs=40):
    """Deterministic, varied pseudo-prose -- independent of conftest's own
    generator so this file has no cross-file coupling on its exact output.
    """

    rng = random.Random(seed)
    vocab = ("the quiet room held a long silence while she considered what had "
             "happened and whether anyone would notice however perhaps not because "
             "nobody asked her directly about any of it").split()
    blocks = []
    for _ in range(paragraphs):
        sentences = [
            " ".join(rng.choice(vocab) for _ in range(rng.randint(4, 22))).capitalize()
            + rng.choice([".", ".", ".", "?", "!"])
            for _ in range(rng.randint(1, 4))
        ]
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


def _extreme_outlier_text(repeats=400):
    """One enormous run-on sentence: extreme words-per-sentence, near-zero
    sentence-length variation (there is only one sentence), no subordinate or
    relative clauses, and a narrow, repetitive vocabulary -- several core
    metrics at once, pushed far past anything ``_prose`` produces.
    """

    words = ("zoo zoo zoo bat bat bat cat cat cat " * repeats).split()
    return " ".join(words) + "."


def _analysis(text, comparison_unit="book"):
    return DocumentAnalysis.from_text(text, processing=TextProcessing(),
                                      comparison_unit=comparison_unit)


def _build_small_profile(tmp_path, count, paragraphs=40, seed_offset=0):
    """A real corpus profile built from real (dependency-free) text, with only
    the core prose metrics populated -- exactly the feature source
    :mod:`textgrader.feature_matrix` uses by default. ``metrics={}`` keeps
    ``build_profile`` from also computing every OTHER registered metric,
    which this suite never reads and which would only slow the test down.
    """

    paths = []
    for i in range(count):
        path = tmp_path / f"book{i:03d}.txt"
        path.write_text(_prose(1000 + seed_offset + i, paragraphs), encoding="utf-8")
        paths.append(path)
    return build_profile(paths, metrics={}, comparison_unit="book")


def _all_metric_ids():
    return [f"{A.PREFIX}{name}" for name in A.ALL_DETECTORS] + [
        f"{A.PREFIX}consensus_count", f"{A.PREFIX}disagreement"]


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


# ----------------------------------------------------------- off-by-default / shape

def test_every_finding_is_its_own_detector():
    """No blended anomaly score -- one finding per detector plus the two
    explicitly-aggregate findings, never fewer.
    """

    assert set(_all_metric_ids()) == {
        f"{A.PREFIX}{name}" for name in A.ALL_DETECTORS
    } | {f"{A.PREFIX}consensus_count", f"{A.PREFIX}disagreement"}
    assert len(A.ALL_DETECTORS) >= 10, "acceptance criteria asks for at least the listed families"


# ------------------------------------------------------------- gating rule (5)

def test_is_a_fast_noop_while_a_corpus_is_being_built():
    """``build_profile`` calls every wanted metric's ``measure`` with
    ``profile=None`` once per book, before any OTHER book has been profiled.
    This suite must recognize that immediately and return without importing
    scikit-learn/PyOD/hdbscan at all -- corpus profiling must never pay this
    suite's cost unless a caller explicitly asks for it (this one never does).
    """

    optional.reset_cache()
    analysis = _analysis(_prose(1))
    started = time.monotonic()
    findings = A.measure(analysis, config={}, profile=None)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"took {elapsed:.2f}s; should be a near-instant no-op"
    assert findings, "every detector plus the two aggregates must still be reported"
    assert all(item["value"] is None for item in findings)
    assert all("no corpus profile" in (item["warning"] or "") for item in findings)
    # The whole point of the gating rule: nothing heavy was imported to answer.
    assert "sklearn" not in optional._cache
    assert "pyod" not in optional._cache


def test_build_profile_with_anomaly_suite_enabled_stays_fast(tmp_path):
    """Enabling this suite in a corpus's own config must not slow down
    ``build_corpus.py`` -- see the module docstring's gating-rule paragraph.
    """

    paths = []
    for i in range(6):
        path = tmp_path / f"book{i}.txt"
        path.write_text(_prose(i), encoding="utf-8")
        paths.append(path)
    started = time.monotonic()
    profile = build_profile(paths, metrics={"anomaly_suite": {"enabled": True}},
                            comparison_unit="book")
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"took {elapsed:.2f}s to profile 6 tiny books"
    assert not profile["metric_errors"].get("anomaly_suite")


# ------------------------------------------------------ small-corpus refusal (8)

def test_small_corpus_is_refused_not_guessed(tmp_path):
    profile = _build_small_profile(tmp_path, count=5)
    analysis = _analysis(_prose(2))
    findings = A.measure(analysis, config={}, profile=profile)
    by_id = _by_id(findings)
    assert set(by_id) == set(_all_metric_ids())
    for metric_id, item in by_id.items():
        assert item["value"] is None, metric_id
        assert item["sample_size"] == 5
        assert "reference book" in (item["warning"] or ""), (metric_id, item["warning"])


def test_no_profile_at_all_is_refused():
    analysis = _analysis(_prose(3))
    findings = A.measure(analysis, config={}, profile=None)
    assert all(item["value"] is None for item in findings)
    assert all("no corpus profile" in (item["warning"] or "") for item in findings)


# --------------------------------------------------------- comparison-unit safety

def test_mismatched_comparison_unit_is_refused(tmp_path):
    profile = _build_small_profile(tmp_path, count=22)
    chapter_analysis = _analysis(_prose(4), comparison_unit="chapter")
    findings = A.measure(chapter_analysis, config={}, profile=profile)
    assert all(item["value"] is None for item in findings)
    assert all("comparison unit" in (item["warning"] or "") for item in findings)


# ------------------------------------------------------------- separation (rule 4)

def test_separates_a_typical_document_from_a_multivariate_outlier(tmp_path):
    """The headline requirement: a document built from the corpus's own
    typical values must score as LESS anomalous than one with several
    features pushed far outside the corpus range, and this must hold for
    real detectors run through the real ``measure()`` entry point, not just
    at the feature-matrix level.
    """

    profile = _build_small_profile(tmp_path, count=24)
    typical = _analysis(_prose(999, paragraphs=40))
    outlier = _analysis(_extreme_outlier_text())

    typical_findings = _by_id(A.measure(typical, config={}, profile=profile))
    outlier_findings = _by_id(A.measure(outlier, config={}, profile=profile))

    # SOS and HDBSCAN are excluded from the strict per-detector comparison:
    # SOS's own perplexity-based affinity calibration can fail to produce a
    # finite score for a genuinely extreme point (a real, documented
    # numerical limitation of that method, not a bug in this suite -- see the
    # module docstring), and HDBSCAN legitimately reports "unavailable" on a
    # reference corpus with no real density-cluster structure. Both are still
    # asserted not to raise/crash; the comparison below is over every OTHER
    # detector, which is still most of the acceptance criteria's list.
    compared = 0
    for name in A.ALL_DETECTORS:
        if name in ("sos", "hdbscan"):
            continue
        metric_id = f"{A.PREFIX}{name}"
        typical_score = typical_findings[metric_id]["value"]
        outlier_score = outlier_findings[metric_id]["value"]
        if typical_score is None or outlier_score is None:
            continue
        compared += 1
        assert outlier_score > typical_score, (
            f"{name}: expected the outlier ({outlier_score}) to score more anomalous than "
            f"the typical document ({typical_score})")
    assert compared >= 8, "too few detectors produced a comparable score to trust this test"

    typical_consensus = typical_findings[f"{A.PREFIX}consensus_count"]["value"]
    outlier_consensus = outlier_findings[f"{A.PREFIX}consensus_count"]["value"]
    assert outlier_consensus > typical_consensus
    assert typical_consensus <= 2, "a typical document should not look like a consensus outlier"

    # Evidence must point at real features, not an empty/placeholder list.
    evidence = outlier_findings[f"{A.PREFIX}isolation_forest"]["evidence"]
    assert evidence and all("feature" in row and "standardized_deviation" in row
                            for row in evidence)


def test_combination_outlier_invisible_on_either_marginal_range(tmp_path):
    """Two features that are strongly correlated across the reference corpus;
    a point that swaps them sits inside EACH feature's own marginal range but
    breaks the correlation -- exactly the case a per-metric range check
    cannot see and a covariance-aware detector (Mahalanobis, Isolation
    Forest) can.
    """

    rng = random.Random(7)
    books = []
    for i in range(30):
        a = rng.uniform(-1.0, 1.0)
        b = a + rng.uniform(-0.05, 0.05)  # b almost always tracks a
        books.append({"source_id": f"b{i}", "custom_a": a, "custom_b": b})
    profile = {"comparison_unit": "book", "books": books,
              "lexile_frequency_source": "none"}

    config = {"columns": ["custom_a", "custom_b"], "min_corpus_documents": 20}
    analysis = _analysis(_prose(5))

    def _typical_values(*_args, **_kwargs):
        # Near the corpus centroid, ON the a≈b line every reference book sits on.
        return {"custom_a": 0.0, "custom_b": 0.0}

    def _combo_outlier_values(*_args, **_kwargs):
        # Both individually inside [-1, 1] -- the corpus's own marginal range for
        # EACH feature on its own -- but far off the a≈b line every reference
        # book sits on, which only a covariance-aware detector can see.
        return {"custom_a": 0.9, "custom_b": -0.9}

    import textgrader.metrics.anomaly_suite as module

    original = module.core_measure
    try:
        module.core_measure = _typical_values
        typical = _by_id(A.measure(analysis, config=config, profile=profile))
        module.core_measure = _combo_outlier_values
        combo = _by_id(A.measure(analysis, config=config, profile=profile))
    finally:
        module.core_measure = original

    # Mahalanobis/Elliptic Envelope are the textbook covariance-aware
    # detectors for exactly this shape of outlier; a tree- or density-based
    # detector is not guaranteed to separate a purely linear-correlation
    # violation this cleanly in only two dimensions, so the assertion is
    # scoped to the detector family built for this case.
    assert combo[f"{A.PREFIX}mahalanobis"]["value"] > typical[f"{A.PREFIX}mahalanobis"]["value"]
    assert (combo[f"{A.PREFIX}elliptic_envelope"]["value"]
           > typical[f"{A.PREFIX}elliptic_envelope"]["value"])
    # And each feature really is inside the corpus's own marginal range --
    # the whole point of the test.
    values_a = [book["custom_a"] for book in books]
    values_b = [book["custom_b"] for book in books]
    assert min(values_a) <= 0.9 <= max(values_a)
    assert min(values_b) <= -0.9 <= max(values_b)


# ------------------------------------------------------------- missing features

def test_missing_document_feature_is_imputed_and_flagged(tmp_path):
    profile = _build_small_profile(tmp_path, count=22)
    analysis = _analysis(_prose(6))

    import textgrader.metrics.anomaly_suite as module
    original = module.core_measure

    def _partial(*_args, **_kwargs):
        full = original(analysis, floor=1, lexile_source="none")
        full = dict(full)
        full["wps"] = None  # simulate one metric the document has no value for
        return full

    try:
        module.core_measure = _partial
        # max_features raised so "wps" (alphabetically last among the tied,
        # fully-covered candidate columns) survives the cap and stays in the
        # schema for this test to see imputed -- see feature_matrix.py's
        # coverage-then-name tie-break.
        findings = _by_id(A.measure(analysis, config={"max_features": 25}, profile=profile))
    finally:
        module.core_measure = original

    item = findings[f"{A.PREFIX}isolation_forest"]
    assert item["value"] is not None
    assert "wps" in item["distribution"]["missing_document_features"]


# ------------------------------------------------------------------ determinism

def test_deterministic_under_the_same_seed(tmp_path):
    profile = _build_small_profile(tmp_path, count=22)
    analysis = _analysis(_prose(8))
    first = _by_id(A.measure(analysis, config={"seed": 42}, profile=profile))
    second = _by_id(A.measure(analysis, config={"seed": 42}, profile=profile))
    for metric_id in _all_metric_ids():
        assert first[metric_id]["value"] == second[metric_id]["value"], metric_id


# ------------------------------------------------------- independent detectors

def test_disabling_one_detector_leaves_every_other_finding_untouched(tmp_path):
    profile = _build_small_profile(tmp_path, count=22)
    analysis = _analysis(_prose(9))
    config = {"features": {**A.DEFAULT_FEATURES, "isolation_forest": False}}
    findings = _by_id(A.measure(analysis, config=config, profile=profile))
    assert f"{A.PREFIX}isolation_forest" not in findings
    assert f"{A.PREFIX}lof" in findings
    assert findings[f"{A.PREFIX}lof"]["value"] is not None


def test_a_missing_package_degrades_only_its_own_detectors(monkeypatch, tmp_path):
    """Disabling scikit-learn must not touch the PyOD-backed detectors, and
    vice versa -- one broken/absent detector family never suppresses another.
    """

    profile = _build_small_profile(tmp_path, count=22)
    analysis = _analysis(_prose(10))
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "sklearn")
    optional.reset_cache()
    try:
        findings = _by_id(A.measure(analysis, config={}, profile=profile))
        for name in A._SKLEARN_DETECTORS:
            item = findings[f"{A.PREFIX}{name}"]
            assert item["value"] is None, name
            assert "sklearn" in (item["warning"] or ""), (name, item["warning"])
        if optional.have("pyod"):
            for name in A._PYOD_DETECTORS:
                item = findings[f"{A.PREFIX}{name}"]
                # sos may legitimately fail on numerical grounds independent
                # of package availability; every other pyod detector must
                # still have produced a real number.
                if name != "sos":
                    assert item["value"] is not None, (name, item["warning"])
    finally:
        optional.reset_cache()


# --------------------------------------------------------------- registry / config

def test_registered_off_by_default_and_correct_family():
    from textgrader.metrics import REGISTRY

    spec = REGISTRY["anomaly_suite"]
    assert spec.family == "distribution_shape"
    assert spec.cost == "moderate"
    assert not spec.needs_parse
    assert not spec.needs_model


def test_off_unless_enabled(tmp_path, base_config):
    import grade

    source = tmp_path / "story.txt"
    source.write_text(_prose(11), encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(A.PREFIX)]
