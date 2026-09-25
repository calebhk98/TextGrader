"""Contract tests for the metric-relationships meta-analysis layer (Task 22).

Two layers are tested: the dependency-free statistics in
:mod:`textgrader.relationships` directly (known values), and the suite in
:mod:`textgrader.metrics.metric_relationships`, exercised both as a unit
(``relationship_findings`` called directly against a synthetic profile, for
precise control over which metrics correlate) and end to end through
``grade.py``'s post-metric phase.
"""

from __future__ import annotations

import random

import pytest

import grade
from textgrader import relationships as rel
from textgrader.document import DocumentAnalysis, TextProcessing
from textgrader.metrics import REGISTRY, metric_relationships as mr

pytestmark = pytest.mark.filterwarnings("ignore")


def _prose(seed, paragraphs=30):
    rng = random.Random(seed)
    vocab = ("the quiet room held a long silence while she considered what had "
             "happened and whether anyone would notice however perhaps not because "
             "nobody asked her directly about any of it").split()
    blocks = []
    for _ in range(paragraphs):
        sentences = [" ".join(rng.choice(vocab) for _ in range(rng.randint(4, 20))).capitalize()
                    + "." for _ in range(rng.randint(1, 4))]
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


def _analysis(text="Hello there. This is a test.", source="doc.md", comparison_unit="book"):
    return DocumentAnalysis.from_text(text, processing=TextProcessing(),
                                      comparison_unit=comparison_unit, source=source)


def _profile(books, comparison_unit="book"):
    return {"books": books, "comparison_unit": comparison_unit}


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


BASE_CONFIG = dict(mr.DEFAULTS)


def _config(**overrides):
    config = {**BASE_CONFIG, **overrides}
    config.setdefault("min_corpus_documents", 8)
    return config


# ------------------------------------------------------- textgrader.relationships

def test_pearson_spearman_kendall_known_values():
    pairs = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 11.0)]
    assert rel.pearson(pairs) == pytest.approx(0.9958, abs=1e-3)
    assert rel.spearman(pairs) == pytest.approx(1.0)
    assert rel.kendall_tau(pairs) == pytest.approx(1.0)


def test_distance_correlation_is_zero_for_independence_and_positive_for_dependence():
    rng = random.Random(0)
    independent = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(200)]
    dependent = [(x, x * x) for x in [rng.gauss(0, 1) for _ in range(200)]]
    dcor_independent = rel.distance_correlation(independent)
    dcor_dependent = rel.distance_correlation(dependent)
    assert dcor_independent is not None and dcor_dependent is not None
    assert dcor_independent < 0.15
    # x and x^2 are Pearson-uncorrelated (for symmetric x) but not independent;
    # distance correlation, unlike Pearson, is expected to catch that.
    assert dcor_dependent > 0.3
    assert rel.pearson(dependent) is not None and abs(rel.pearson(dependent)) < 0.2


def test_mutual_information_nonnegative_and_higher_for_dependence():
    rng = random.Random(1)
    independent = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(300)]
    dependent = [(x, 2 * x + rng.gauss(0, 0.05)) for x in [rng.gauss(0, 1) for _ in range(300)]]
    mi_independent = rel.mutual_information(independent)
    mi_dependent = rel.mutual_information(dependent)
    assert mi_independent is not None and mi_dependent is not None
    assert mi_independent >= 0 and mi_dependent >= 0
    assert mi_dependent > mi_independent


def test_ols_fit_recovers_linear_coefficients():
    rows = [[1.0, x] for x in range(20)]
    y = [3.0 + 2.0 * x for x in range(20)]
    coefficients = rel.ols_fit(rows, y)
    assert coefficients[0] == pytest.approx(3.0, abs=1e-6)
    assert coefficients[1] == pytest.approx(2.0, abs=1e-6)


def test_residual_model_near_zero_on_line_and_large_off_line():
    rng = random.Random(2)
    books = [{"a": x, "b": 2 * x + 3 + rng.gauss(0, 0.05)} for x in
            (rng.uniform(1, 10) for _ in range(40))]
    model = rel.fit_residual_model(books, "a", ["b"])
    assert model is not None
    on_line = rel.standardized_residual(model, 5.0, [2 * 5 + 3])
    off_line = rel.standardized_residual(model, 5.0 + 20, [2 * 5 + 3])
    assert abs(on_line) < 1.0
    assert abs(off_line) > 50


def test_orthogonal_residual_is_order_independent():
    rng = random.Random(3)
    pairs = [(x, 1.5 * x + rng.gauss(0, 0.1)) for x in (rng.uniform(0, 10) for _ in range(40))]
    swapped = [(y, x) for x, y in pairs]
    forward = rel.fit_orthogonal_model(pairs)
    backward = rel.fit_orthogonal_model(swapped)
    point = (5.0, 25.0)  # far off the line in both directions
    forward_value = rel.orthogonal_residual(forward, point)
    backward_value = rel.orthogonal_residual(backward, (point[1], point[0]))
    assert abs(forward_value) == pytest.approx(abs(backward_value), rel=1e-6)


def test_bootstrap_and_loo_stability_are_low_for_a_stable_relationship():
    rng = random.Random(4)
    pairs = [(x, 2 * x + rng.gauss(0, 0.1)) for x in (rng.uniform(0, 10) for _ in range(50))]
    bootstrap_std = rel.bootstrap_correlation_stability(pairs, 200, seed=0)
    loo_std = rel.leave_one_out_stability(pairs)
    assert bootstrap_std is not None and bootstrap_std < 0.05
    assert loo_std is not None and loo_std < 0.05


def test_partial_correlation_removes_a_confound():
    rng = random.Random(5)
    z = [rng.gauss(0, 1) for _ in range(200)]
    a = [2 * value + rng.gauss(0, 0.01) for value in z]
    b = [3 * value + rng.gauss(0, 0.01) for value in z]
    raw = rel.pearson(list(zip(a, b)))
    partial = rel.partial_correlation(a, b, [z])
    assert raw > 0.9  # a and b look strongly correlated ...
    assert abs(partial) < 0.2  # ... but only because both follow the same z


def test_principal_components_share_sums_near_one():
    rng = random.Random(6)
    rows = [[rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(60)]
    result = rel.principal_components(rows, max_components=3)
    assert result is not None
    assert sum(result["explained_variance_share"]) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------- registry

def test_registered_off_by_default():
    assert "metric_relationships" in REGISTRY
    spec = REGISTRY["metric_relationships"]
    assert spec.family == "distribution_shape"
    assert spec.requires == ()  # pure Python: no optional package at all


def test_measure_hook_is_a_noop():
    analysis = _analysis()
    assert mr.measure(analysis, config={}, profile=None) == []
    assert mr.measure(analysis, config={}, profile={"books": [{"wps": 10.0}]}) == []


def test_disabled_by_default(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(_prose(1), encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results
               if item.metric_id.startswith("style.relationship_")]


def test_disabled_switch_adds_no_findings_and_costs_nothing(tmp_path, base_config, monkeypatch):
    """An explicit ``enabled: false`` must short-circuit before the suite
    module is even imported -- ``_relationship_results`` returns immediately.
    """

    calls = []
    original = grade.importlib.import_module

    def _tracking_import(name, *args, **kwargs):
        calls.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(grade.importlib, "import_module", _tracking_import)
    source = tmp_path / "story.txt"
    source.write_text(_prose(1), encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                        "metric_relationships": {"enabled": False}}}
    grade.analyze(source, config)
    assert not any("metric_relationships" in name for name in calls)


# ---------------------------------------------------------------- no profile

def test_no_profile_is_refused_not_guessed():
    findings = mr.relationship_findings(_analysis(), {"a": 1.0}, config=_config(), profile=None)
    assert findings
    assert all(item["value"] is None for item in findings)
    assert all("no corpus profile" in (item["warning"] or "") for item in findings)


def test_mismatched_comparison_unit_is_refused():
    books = [{"a": float(i), "b": float(i) * 2, "source_filename": f"b{i}.txt"} for i in range(20)]
    profile = _profile(books)
    chapter_doc = _analysis(comparison_unit="chapter")
    findings = mr.relationship_findings(chapter_doc, {"a": 1.0, "b": 2.0}, config=_config(),
                                        profile=profile)
    assert all(item["value"] is None for item in findings)
    assert all("comparison unit" in (item["warning"] or "") for item in findings)


def test_small_corpus_is_refused():
    books = [{"a": float(i), "b": float(i) * 2, "source_filename": f"b{i}.txt"} for i in range(5)]
    findings = mr.relationship_findings(_analysis(), {"a": 1.0, "b": 2.0}, config=_config(),
                                        profile=_profile(books))
    assert all(item["value"] is None for item in findings)
    assert any("leave-one-out" in (item["warning"] or "") for item in findings)


# --------------------------------------------------------- synthetic separation

def _books_with_relation(n, seed, target="a", predictor="b", slope=2.0, intercept=3.0,
                         noise=0.05, extra=None):
    rng = random.Random(seed)
    books = []
    for i in range(n):
        x = rng.uniform(1, 10)
        row = {predictor: x, target: slope * x + intercept + rng.gauss(0, noise),
              "source_filename": f"book{i:03d}.txt"}
        if extra:
            row.update(extra(rng, i))
        books.append(row)
    return books


def test_synthetic_perfectly_correlated_pair_gives_near_zero_residual():
    books = _books_with_relation(40, seed=10)
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={})
    findings = mr.relationship_findings(_analysis(), {"a": 2 * 5 + 3, "b": 5.0}, config=config,
                                        profile=profile)
    by_id = _by_id(findings)
    residual = by_id["style.relationship_residual_a_given_b"]
    assert residual["value"] is not None
    assert abs(residual["value"]) < 1.0


def test_synthetic_anomaly_gives_high_residual_only_for_the_anomalous_pair():
    """Two independent correlated relationships in the same corpus; the graded
    document breaks only one of them.  Only that pair's residual should fire.
    """

    rng = random.Random(11)
    books = []
    for i in range(40):
        x1 = rng.uniform(1, 10)
        x2 = rng.uniform(1, 10)
        books.append({
            "a": 2 * x1 + 3 + rng.gauss(0, 0.05), "b": x1,
            "c": -1 * x2 + 7 + rng.gauss(0, 0.05), "d": x2,
            "source_filename": f"book{i:03d}.txt",
        })
    profile = _profile(books)
    config = _config(pairs=[["a", "b"], ["c", "d"]], groups={})
    # "a given b" is broken (a is 30 higher than the line predicts); "c given d"
    # sits exactly on its own corpus line.
    document_values = {"a": 2 * 5 + 3 + 30, "b": 5.0, "c": -1 * 4 + 7, "d": 4.0}
    findings = mr.relationship_findings(_analysis(), document_values, config=config,
                                        profile=profile)
    by_id = _by_id(findings)
    broken = by_id["style.relationship_residual_a_given_b"]
    intact = by_id["style.relationship_residual_c_given_d"]
    assert abs(broken["value"]) > 20
    assert abs(intact["value"]) < 1.0


def test_words_per_paragraph_given_words_per_sentence_and_sentences_per_paragraph():
    """The task spec's own required multivariate example."""

    rng = random.Random(12)
    books = []
    for i in range(30):
        wps = rng.uniform(6, 20)
        spp = rng.uniform(2, 8)
        books.append({
            "wps": wps, "spp": spp, "wpp": wps * spp + rng.gauss(0, 0.5),
            "source_filename": f"book{i:03d}.txt",
        })
    profile = _profile(books)
    config = _config(pairs=[], groups={"wpp": ["wps", "spp"]})
    # On-relationship document: wpp should match wps*spp closely.
    on = mr.relationship_findings(_analysis(), {"wps": 10.0, "spp": 4.0, "wpp": 40.0},
                                  config=config, profile=profile)
    by_id_on = _by_id(on)
    residual_id = "style.relationship_residual_wpp_given_spp_wps"
    assert residual_id in by_id_on
    assert abs(by_id_on[residual_id]["value"]) < 1.5
    # Off-relationship document: same wps/spp, but a wpp far from wps*spp.
    off = mr.relationship_findings(_analysis(), {"wps": 10.0, "spp": 4.0, "wpp": 200.0},
                                   config=config, profile=profile)
    by_id_off = _by_id(off)
    assert abs(by_id_off[residual_id]["value"]) > 10


def test_missing_source_metric_disables_only_the_dependent_finding():
    books = _books_with_relation(30, seed=13, extra=lambda rng, i: {"c": rng.uniform(0, 1)})
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={})
    # "a" is missing from this document's own measured values (e.g. its
    # metric errored or was never enabled); "b" alone cannot fill in for it.
    findings = mr.relationship_findings(_analysis(), {"b": 5.0}, config=config, profile=profile)
    by_id = _by_id(findings)
    residual = by_id["style.relationship_residual_a_given_b"]
    assert residual["value"] is None
    assert "a" in residual["warning"]
    # The corpus fit itself is still visible (not itself "missing"): distribution
    # carries real corpus statistics even though this document has no value.
    assert residual["distribution"]["n"] == 30
    assert residual["distribution"]["pearson"] is not None


def test_both_inliers_but_unusual_combination_is_flagged():
    """The spec's key acceptance test: a document whose two raw metrics are
    EACH, individually, ordinary corpus values, but whose combination is not.
    """

    rng = random.Random(14)
    a_values, b_values, books = [], [], []
    for i in range(60):
        a = rng.gauss(0, 1)
        b = 0.9 * a + rng.gauss(0, 0.15)
        a_values.append(a)
        b_values.append(b)
        books.append({"a": a, "b": b, "source_filename": f"book{i:03d}.txt"})
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={})

    doc_a, doc_b = 0.3, -0.3  # both near the marginal median of a and b
    z_a = rel.robust_z(doc_a, a_values)
    z_b = rel.robust_z(doc_b, b_values)
    assert abs(z_a) < 1.0, "the point of this test: a alone must look ordinary"
    assert abs(z_b) < 1.0, "the point of this test: b alone must look ordinary"

    findings = mr.relationship_findings(_analysis(), {"a": doc_a, "b": doc_b}, config=config,
                                        profile=profile)
    by_id = _by_id(findings)
    residual = by_id["style.relationship_residual_a_given_b"]
    symmetric = by_id["style.relationship_symmetric_a__b"]
    assert abs(residual["value"]) > 3.0, "the joint combination must be flagged"
    assert abs(symmetric["value"]) > 3.0


def test_leave_one_out_excludes_the_graded_documents_own_corpus_row():
    """If the graded document IS one of the profile's own books, its row must
    not help fit the relationship it is then scored against.
    """

    rng = random.Random(15)
    books = _books_with_relation(30, seed=16)
    # Plant a deliberately extreme "self" row and grade a document with the
    # exact same source name and the exact same (extreme) values. If that row
    # were left in the fit, the line would be dragged toward it and its own
    # residual would shrink; excluded, the residual stays large.
    self_source = "self.txt"
    extreme_b, extreme_a = 5.0, 500.0
    books.append({"a": extreme_a, "b": extreme_b, "source_filename": self_source})
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={}, min_corpus_documents=8)

    analysis = _analysis(source=self_source)
    findings = mr.relationship_findings(analysis, {"a": extreme_a, "b": extreme_b}, config=config,
                                        profile=profile)
    by_id = _by_id(findings)
    residual = by_id["style.relationship_residual_a_given_b"]
    assert residual["distribution"]["n"] == 30, "the self row must not be counted in the fit"
    assert abs(residual["value"]) > 20, ("with the self row excluded, this document's own "
                                        "extreme point must still look extreme")

    # Sanity check: fitting WITH the self row included would have pulled the
    # line toward it and produced a much smaller residual for the same point.
    contaminated = mr.relationship_findings(_analysis(source="different.txt"),
                                            {"a": extreme_a, "b": extreme_b}, config=config,
                                            profile=profile)
    contaminated_residual = _by_id(contaminated)["style.relationship_residual_a_given_b"]
    assert contaminated_residual["distribution"]["n"] == 31
    assert abs(contaminated_residual["value"]) < abs(residual["value"])


def test_leave_one_out_matches_by_source_id_and_source_path_too():
    books = _books_with_relation(30, seed=17)
    books[0]["source_filename"] = "book000.txt"
    books[0]["source_path"] = "chapters/book000.txt"
    books[0]["source_id"] = "book000-abcdef"
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={})
    for source in ("chapters/book000.txt", "book000-abcdef"):
        findings = mr.relationship_findings(_analysis(source=source), {"a": 1.0, "b": 1.0},
                                            config=config, profile=profile)
        residual = _by_id(findings)["style.relationship_residual_a_given_b"]
        assert residual["distribution"]["n"] == 29


# ------------------------------------------------------------------- discovery

def test_discovery_finds_a_strong_pair_and_ignores_a_weak_one():
    rng = random.Random(18)
    books = []
    for i in range(40):
        x = rng.uniform(0, 10)
        books.append({
            "strong_a": x, "strong_b": 2 * x + rng.gauss(0, 0.1),
            "weak_a": rng.gauss(0, 1), "weak_b": rng.gauss(0, 1),
            "source_filename": f"book{i:03d}.txt",
        })
    profile = _profile(books)
    config = _config(pairs=[], groups={}, discover_min_abs_spearman=0.7,
                     max_candidate_metrics=10)
    findings = mr.relationship_findings(_analysis(), {"strong_a": 5.0, "strong_b": 10.0,
                                                      "weak_a": 0.1, "weak_b": 0.1},
                                        config=config, profile=profile)
    ids = set(_by_id(findings))
    assert any("strong_a" in metric_id and "strong_b" in metric_id for metric_id in ids)
    assert not any("weak_a" in metric_id and "weak_b" in metric_id for metric_id in ids)


def test_max_pairs_caps_discovered_pairs_deterministically():
    rng = random.Random(19)
    books = []
    for i in range(30):
        row = {"source_filename": f"book{i:03d}.txt"}
        x = rng.uniform(0, 10)
        for j in range(6):
            row[f"m{j}"] = (j + 1) * x + rng.gauss(0, 0.02 * (j + 1))
        books.append(row)
    profile = _profile(books)
    config = _config(pairs=[], groups={}, discover_min_abs_spearman=0.5, max_pairs=2,
                     max_candidate_metrics=10)
    document_values = {f"m{j}": float(j + 1) for j in range(6)}
    findings1 = mr.relationship_findings(_analysis(), document_values, config=config,
                                         profile=profile)
    findings2 = mr.relationship_findings(_analysis(), document_values, config=config,
                                         profile=profile)
    residual_ids_1 = sorted(item["metric_id"] for item in findings1
                            if item["metric_id"].startswith("style.relationship_residual_m"))
    residual_ids_2 = sorted(item["metric_id"] for item in findings2
                            if item["metric_id"].startswith("style.relationship_residual_m"))
    assert residual_ids_1 == residual_ids_2  # deterministic pair selection
    assert len(residual_ids_1) == 2  # capped at max_pairs


def test_configured_pairs_are_kept_even_below_the_discovery_threshold():
    rng = random.Random(20)
    books = [{"a": rng.gauss(0, 1), "b": rng.gauss(0, 1), "source_filename": f"b{i}.txt"}
            for i in range(30)]
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={}, discover_min_abs_spearman=0.99)
    findings = mr.relationship_findings(_analysis(), {"a": 0.5, "b": 0.5}, config=config,
                                        profile=profile)
    assert "style.relationship_residual_a_given_b" in _by_id(findings)


# ------------------------------------------------------------- aggregate/PCA

def test_aggregate_severity_findings_reflect_the_worst_relationship():
    rng = random.Random(21)
    books = _books_with_relation(30, seed=22)
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={})
    findings = mr.relationship_findings(_analysis(), {"a": 2 * 5 + 3 + 40, "b": 5.0},
                                        config=config, profile=profile)
    by_id = _by_id(findings)
    assert by_id["style.relationship_max_residual_severity"]["value"] > 5
    assert by_id["style.relationship_topk_residual_severity"]["value"] > 0
    above_p90 = by_id["style.relationship_residuals_above_p90"]
    assert above_p90["value"] >= 1


def test_pca_diagnostic_is_reported_and_marked_as_diagnostic_only():
    rng = random.Random(23)
    books = []
    for i in range(30):
        books.append({"a": rng.gauss(0, 1), "b": rng.gauss(0, 1), "c": rng.gauss(0, 1),
                      "source_filename": f"b{i}.txt"})
    profile = _profile(books)
    config = _config(pairs=[], groups={}, max_candidate_metrics=10)
    findings = mr.relationship_findings(_analysis(), {"a": 0.1, "b": 0.1, "c": 0.1},
                                        config=config, profile=profile)
    pca = _by_id(findings)["style.relationship_pca_diagnostic"]
    assert pca["value"] is not None
    assert 0.0 <= pca["value"] <= 1.0
    assert "diagnostic only" in pca["warning"]


# ------------------------------------------------------------------- caching

def test_fit_is_cached_per_profile_per_process():
    books = _books_with_relation(30, seed=24)
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={})
    first = mr._fit(profile, "unrelated.txt", config)
    second = mr._fit(profile, "unrelated.txt", config)
    assert first is second


def test_fit_cache_is_keyed_by_excluded_source():
    books = _books_with_relation(30, seed=25)
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={})
    fit_a = mr._fit(profile, books[0]["source_filename"], config)
    fit_b = mr._fit(profile, books[1]["source_filename"], config)
    assert fit_a is not fit_b
    assert fit_a["meta"]["corpus_books_used"] == 29
    assert fit_b["meta"]["corpus_books_used"] == 29


# ----------------------------------------------------------------- determinism

def test_determinism_same_inputs_same_output():
    books = _books_with_relation(30, seed=26)
    profile = _profile(books)
    config = _config(pairs=[["a", "b"]], groups={}, bootstrap_samples=50, seed=7)
    document_values = {"a": 2 * 6 + 3, "b": 6.0}
    first = mr.relationship_findings(_analysis(), document_values, config=config, profile=profile)
    second = mr.relationship_findings(_analysis(), document_values, config=config, profile=profile)
    assert first == second


# ---------------------------------------------------------------------- end to end

def test_end_to_end_through_grade_py(tmp_path, base_config):
    """Exercises the real post-metric phase wired into ``grade.py``'s
    ``_analyze``, with a real corpus profile built via ``build_profile``.
    """

    from textgrader.corpus import build_profile

    paths = []
    for i in range(20):
        path = tmp_path / f"book{i:03d}.txt"
        path.write_text(_prose(100 + i), encoding="utf-8")
        paths.append(path)
    profile = build_profile(paths, metrics={}, comparison_unit="book")
    import json
    (tmp_path / "profile.json").write_text(json.dumps(profile), encoding="utf-8")

    manuscript = tmp_path / "draft.md"
    manuscript.write_text(_prose(999), encoding="utf-8")
    config = {**base_config, "corpus_profile": "profile.json",
             "metrics": {**base_config["metrics"], "metric_relationships": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    relationship_results = [item for item in report.results
                            if item.metric_id.startswith("style.relationship_")]
    assert relationship_results
    errors = [item for item in report.results if item.status_type.value == "internal_error"]
    assert not errors, [(item.metric_id, item.error) for item in errors]
    # Every finding stays neutral -- see the task spec's cross-task rules.
    from textgrader.results import Polarity
    assert all(item.polarity is Polarity.NEUTRAL for item in relationship_results)
