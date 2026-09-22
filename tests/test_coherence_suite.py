"""The experimental discourse-coherence suite: off by default, independently
switchable, honest about sample size, and never confused by a missing
optional package.
"""

from __future__ import annotations

import time

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import coherence_suite
from textgrader.results import Action, StatusType

PREFIX = "discourse.coherence_"


def _ids(findings):
    return {item["metric_id"] for item in findings}


def _analysis(text, **kwargs):
    return DocumentAnalysis.from_text(text, comparison_unit="book", **kwargs)


def _paragraph_of(sentences):
    return " ".join(sentences)


# --------------------------------------------------------------- wiring

def test_off_by_default(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(PREFIX)]


def test_enabling_the_suite_alone_turns_on_every_default_feature(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "coherence_suite": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(manuscript, config).results}
    matched = {mid for mid in ids if mid.startswith(PREFIX)}
    # One id from each of the five feature groups should be present.
    assert "discourse.coherence_adjacent_sentence_overlap" in matched
    assert "discourse.coherence_local_semantic_cohesion" in matched
    assert "discourse.coherence_connective_family_rate" in matched
    assert "discourse.coherence_entity_new_given_ratio" in matched
    assert "discourse.coherence_sentence_order_percentile" in matched


def test_every_metric_id_uses_the_stable_prefix(sample_text):
    findings = coherence_suite.measure(_analysis(sample_text))
    ids = _ids(findings)
    assert ids
    assert all(mid.startswith(PREFIX) for mid in ids)


def test_features_are_independently_switchable(sample_text):
    analysis = _analysis(sample_text)
    lexical_only = coherence_suite.measure(analysis, config={
        "features": {"lexical": True, "semantic": False, "entity": False,
                    "connectives": False, "order_permutation": False}})
    ids = _ids(lexical_only)
    assert "discourse.coherence_adjacent_sentence_overlap" in ids
    assert not any(mid.startswith("discourse.coherence_entity_") for mid in ids)
    assert not any(mid.startswith("discourse.coherence_local_semantic") for mid in ids)
    assert not any("connective" in mid for mid in ids)
    assert not any("order_percentile" in mid for mid in ids)

    entity_only = coherence_suite.measure(_analysis(sample_text), config={
        "features": {"lexical": False, "semantic": False, "entity": True,
                    "connectives": False, "order_permutation": False}})
    entity_ids = _ids(entity_only)
    assert entity_ids == {
        "discourse.coherence_entity_new_given_ratio",
        "discourse.coherence_entity_reintroduction_distance",
        "discourse.coherence_entity_dangling_rate",
        "discourse.coherence_entity_grid_transition_entropy",
        "discourse.coherence_entity_graph_density",
    }


def test_config_and_registry_declare_the_same_option_surface():
    """The whole tunable surface must appear in both places, per the task spec."""

    import json
    from pathlib import Path
    from textgrader.metrics import REGISTRY

    spec = REGISTRY["coherence_suite"]
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    configured = dict(config["metrics"]["coherence_suite"])
    configured.pop("enabled")
    assert set(configured) == set(spec.defaults)
    assert configured["features"] == spec.defaults["features"]


# --------------------------------------------------------------- degradation

def test_missing_optional_packages_degrade_only_the_metrics_that_need_them(monkeypatch,
                                                                          sample_text):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        analysis = _analysis(sample_text)
        findings = {item["metric_id"]: item for item in coherence_suite.measure(analysis)}
    finally:
        optional.reset_cache()

    # Dependency-free groups still produce real numbers.
    assert findings["discourse.coherence_adjacent_sentence_overlap"]["value"] is not None
    assert findings["discourse.coherence_connective_family_rate"]["value"] is not None
    assert findings["discourse.coherence_sentence_order_percentile"]["value"] is not None

    # spaCy-dependent entity metrics say why they could not run, and do not crash.
    entity_finding = findings["discourse.coherence_entity_new_given_ratio"]
    assert entity_finding["value"] is None
    assert entity_finding["warning"]

    # The semantic group falls back to its lexical proxy rather than disappearing,
    # and says so, exactly like the rest of the semantic_repetition family.
    semantic_finding = findings["discourse.coherence_local_semantic_cohesion"]
    assert semantic_finding["value"] is not None
    assert "lexical" in semantic_finding["warning"]

    # The entity graph needs spaCy first; it must not be reached at all here,
    # but if it is ever asked to run without networkx it must degrade, not crash.
    graph_finding = findings["discourse.coherence_entity_graph_density"]
    assert graph_finding["value"] is None


def test_networkx_missing_degrades_only_the_graph_metric(monkeypatch, sample_text):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "networkx")
    optional.reset_cache()
    try:
        analysis = _analysis(sample_text)
        findings = {item["metric_id"]: item for item in coherence_suite.measure(analysis)}
    finally:
        optional.reset_cache()

    graph_finding = findings["discourse.coherence_entity_graph_density"]
    assert graph_finding["value"] is None
    assert "networkx" in graph_finding["warning"]
    # Its sibling entity metrics, which do not need networkx, are unaffected.
    assert findings["discourse.coherence_entity_new_given_ratio"]["value"] is not None


@pytest.mark.parametrize("text", ["", "Hi.", "A lone fragment", "A\n\nB\n\nC"])
def test_survives_degenerate_and_fragment_documents(text):
    analysis = _analysis(text)
    findings = coherence_suite.measure(analysis)
    assert findings
    for item in findings:
        assert item["metric_id"].startswith(PREFIX)


# ------------------------------------------------------------ sample honesty

def test_small_document_reports_insufficient_data_not_a_confident_number(tmp_path, base_config):
    text = "Alice walked in. Bob looked up. She smiled. He nodded."
    source = tmp_path / "tiny.txt"
    source.write_text(text, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "coherence_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    entity_ratio = next(item for item in report.results
                       if item.metric_id == "discourse.coherence_entity_new_given_ratio")
    if entity_ratio.value is None:
        assert entity_ratio.action is Action.UNAVAILABLE
    else:
        assert entity_ratio.action is Action.INSUFFICIENT_DATA


def test_every_finding_carries_its_own_sample_size_and_floor(sample_text):
    for item in coherence_suite.measure(_analysis(sample_text)):
        assert "sample_size" in item
        assert "min_sample" in item


# ------------------------------------------------------------- order tests

def test_topically_grouped_order_outranks_an_interleaved_shuffle_of_the_same_sentences():
    """The acceptance criterion in one paragraph: same sentences, same
    vocabulary, only the order differs, and the order-sensitive metric must
    tell the two apart."""

    topic_a = [f"Rivers carry silt and stone past the {word} valley." for word in
              ("northern", "eastern", "quiet", "narrow", "distant", "steep")]
    topic_b = [f"Parliament debated the {word} budget long into the night." for word in
              ("annual", "regional", "contested", "delayed", "shrinking", "urban")]
    grouped = _paragraph_of(topic_a + topic_b)
    interleaved = _paragraph_of([s for pair in zip(topic_a, topic_b) for s in pair])

    config = {"permutations": 200, "seed": 0, "order_min_sentences_per_paragraph": 4}
    grouped_findings = {item["metric_id"]: item
                       for item in coherence_suite.measure(_analysis(grouped), config=config)}
    interleaved_findings = {item["metric_id"]: item
                           for item in coherence_suite.measure(_analysis(interleaved),
                                                              config=config)}

    grouped_percentile = grouped_findings["discourse.coherence_sentence_order_percentile"]["value"]
    interleaved_percentile = interleaved_findings[
        "discourse.coherence_sentence_order_percentile"]["value"]
    assert grouped_percentile is not None and interleaved_percentile is not None
    assert grouped_percentile > interleaved_percentile
    assert grouped_percentile >= 90.0


def test_permutation_percentile_is_deterministic_under_a_fixed_seed(sample_text):
    analysis_a = _analysis(sample_text)
    analysis_b = _analysis(sample_text)
    config = {"seed": 7, "permutations": 40}
    first = {item["metric_id"]: item["value"]
            for item in coherence_suite.measure(analysis_a, config=config)
            if "order_percentile" in item["metric_id"]}
    second = {item["metric_id"]: item["value"]
             for item in coherence_suite.measure(analysis_b, config=config)
             if "order_percentile" in item["metric_id"]}
    assert first == second


def test_repeated_topic_but_shuffled_text_keeps_high_overlap_but_not_necessarily_high_order():
    """High lexical overlap must not be conflated with good ordering: both
    numbers are retained, and they may disagree."""

    sentences = [f"The garden was full of {word} light that afternoon." for word in
                ("golden", "pale", "warm", "soft", "faint", "bright", "dim", "clear")]
    analysis = _analysis(_paragraph_of(sentences))
    findings = {item["metric_id"]: item
               for item in coherence_suite.measure(analysis, config={"permutations": 100})}
    overlap = findings["discourse.coherence_adjacent_sentence_overlap"]["value"]
    assert overlap is not None and overlap > 0.2
    # The ordering metric is reported independently of how high overlap is;
    # a uniformly-similar paragraph like this one has no "correct" order, so
    # its percentile is not asserted against a direction, only that it exists
    # and is a real, distinct measurement.
    order = findings["discourse.coherence_sentence_order_percentile"]["value"]
    assert order is not None


# ------------------------------------------------------------ entity fixture

def test_entity_continuity_with_pronouns_and_repeated_named_entities():
    sentences = []
    for _ in range(6):
        sentences.append("Alice opened the heavy door and looked around the empty room.")
        sentences.append("She had not expected the room to be so cold.")
        sentences.append("Bob called her name from the hallway outside.")
        sentences.append("Alice answered him without turning around.")
    analysis = _analysis(_paragraph_of(sentences))
    if analysis.nlp_unavailable:
        pytest.skip("spaCy is not available in this environment")
    findings = {item["metric_id"]: item for item in coherence_suite.measure(
        analysis, config={"features": {"entity": True, "lexical": False, "semantic": False,
                                       "connectives": False, "order_permutation": False}})}
    reintroduction = findings["discourse.coherence_entity_reintroduction_distance"]
    assert reintroduction["value"] is not None
    assert reintroduction["sample_size"] > 0
    given_new = findings["discourse.coherence_entity_new_given_ratio"]
    assert given_new["value"] is not None
    # "Alice", "Bob" and "room" all recur; given mentions should dominate new ones.
    assert given_new["value"] < 1.0


# ---------------------------------------------------------------- settings

def test_findings_record_the_settings_that_produced_the_number(sample_text):
    findings = {item["metric_id"]: item for item in coherence_suite.measure(
        _analysis(sample_text), config={"seed": 3, "permutations": 20})}

    order = findings["discourse.coherence_paragraph_order_percentile"]
    if order["value"] is not None:
        assert order["distribution"]["seed"] == 3
        assert order["distribution"]["permutations"] == 20

    semantic = findings["discourse.coherence_local_semantic_cohesion"]
    assert semantic["distribution"]["model"]
    assert semantic["distribution"]["backend"] in ("embedding", "lexical")

    if not findings["discourse.coherence_entity_grid_transition_entropy"]["warning"]:
        entropy = findings["discourse.coherence_entity_grid_transition_entropy"]
        assert entropy["distribution"]["role_schema"]


# ------------------------------------------------------------------ runtime

def test_large_document_stays_within_a_runtime_budget(prose):
    text = "\n\n".join(prose(seed, paragraphs=40) for seed in range(6))
    analysis = _analysis(text)
    started = time.monotonic()
    findings = coherence_suite.measure(analysis, config={"permutations": 30})
    elapsed = time.monotonic() - started
    assert findings
    assert elapsed < 90.0, f"coherence_suite took {elapsed:.1f}s on a large document"


def test_every_registered_id_runs_without_raising_via_grade(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "coherence_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
