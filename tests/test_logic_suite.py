"""Contract tests for the experimental logic/consistency suite.

These mirror the synthetic cases the task spec calls for: a contradiction
pair, an entailment-shaped pair, a neutral pair, one test per connective
family, the pair cap holding under a pathological repeat, graceful
degradation without spaCy, and that a bare lexical-overlap or entity-key
absence is never promoted into a stronger claim than the module makes.
"""

from __future__ import annotations

import pytest

from textgrader import optional
from textgrader.document import DocumentAnalysis, NlpSettings
from textgrader.metrics import REGISTRY, logic_suite

SPACY_READY = optional.have("spacy")
requires_spacy = pytest.mark.skipif(not SPACY_READY, reason="spacy/en_core_web_sm not available")


def _findings(text, config=None, **doc_kwargs):
    analysis = DocumentAnalysis.from_text(text, comparison_unit="book", **doc_kwargs)
    return {item["metric_id"]: item for item in logic_suite.measure(analysis, config=config)}


def test_registered_and_off_by_default():
    assert "logic_suite" in REGISTRY
    spec = REGISTRY["logic_suite"]
    assert spec.family == "discourse"
    # Every option this module reads must be documented in MetricSpec.defaults.
    for name in ("features", "window_sentences", "max_pairs", "max_comparisons",
                "max_evidence", "proposition_cap", "connective_min_words",
                "repeated_assertion_min_words"):
        assert name in spec.defaults, name


def test_all_metric_ids_are_stable_and_prefixed():
    for group_ids in logic_suite.FEATURE_METRICS.values():
        for metric_id in group_ids:
            assert metric_id.startswith("discourse.logic_"), metric_id
            assert metric_id in logic_suite._METRIC_NAMES


# ------------------------------------------------------------ synthetic pairs

@requires_spacy
def test_synthetic_contradiction_pair_is_a_negation_flip_candidate():
    text = ("The museum is open every day this week. " * 2 +
           "By Friday evening, however, the museum is not open at all.")
    found = _findings(text)
    item = found["discourse.logic_negation_flip_candidates"]
    assert item["value"] is not None
    assert item["value"] > 0
    assert item["evidence"], "a contradiction candidate must carry evidence, not just a count"
    row = item["evidence"][0]
    assert row["type"] == "negation"
    assert "not" in row["sentence_b"]["text"].lower() or "not" in row["sentence_a"]["text"].lower()
    # The finding must never claim more than a candidate.
    assert "candidate" in item["warning"]


@requires_spacy
def test_synthetic_entailment_shaped_pair_scores_high_overlap():
    text = ("The rain flooded the narrow streets of the old town. Therefore the "
           "narrow streets of the old town were flooded by the rain.")
    found = _findings(text)
    item = found["discourse.logic_therefore_overlap"]
    assert item["value"] is not None
    assert item["value"] > 0.5


@requires_spacy
def test_neutral_unrelated_pair_scores_low_overlap():
    text = ("The gardener planted tulips along the fence. Therefore quantum "
           "particles obey the Pauli exclusion principle when spin aligns.")
    found = _findings(text)
    item = found["discourse.logic_therefore_overlap"]
    assert item["value"] is not None
    assert item["value"] < 0.3


def test_because_and_however_connectives_are_scored_independently():
    text = ("She stayed home because the storm had knocked out the power. "
           "However, her brother went out anyway.")
    found = _findings(text)
    because = found["discourse.logic_because_overlap"]
    contrast = found["discourse.logic_contrast_overlap"]
    assert because["sample_size"] >= 1
    assert contrast["sample_size"] >= 1


def test_conditional_clause_shape_flags_bare_idiom():
    text = ("If the bridge floods, the town closes the eastern road. "
           "If only. Unless the council intervenes, the ferry keeps running.")
    found = _findings(text)
    item = found["discourse.logic_conditional_clause_shape_rate"]
    assert item["sample_size"] == 3
    assert item["value"] is not None
    assert item["value"] < 100.0
    assert any("if only" in row["text"].lower() for row in item["evidence"])


def test_connective_chain_length_needs_a_real_chain():
    text = "Plain sentence one. Plain sentence two. Plain sentence three."
    found = _findings(text)
    item = found["discourse.logic_connective_chain_length"]
    assert item["value"] is None
    assert "no chain" in item["warning"] or "no sentence" in item["warning"]


# ---------------------------------------------------------- pair-cap boundedness

@requires_spacy
def test_pair_cap_is_obeyed_on_a_pathological_repeat():
    sentences = []
    for index in range(120):
        if index % 2 == 0:
            sentences.append("The council approved the plan.")
        else:
            sentences.append("The council did not approve the plan.")
    text = " ".join(sentences)
    found = _findings(text, config={"max_pairs": 5, "window_sentences": 200,
                                    "max_comparisons": 50_000})
    item = found["discourse.logic_negation_flip_candidates"]
    assert len(item["evidence"]) <= 5
    assert item["distribution"]["candidate_count"] <= 5
    assert item["distribution"]["settings"]["pairs_capped"] is True


# --------------------------------------------------------- graceful degradation

def test_missing_spacy_disables_only_the_proposition_group(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "spacy")
    optional.reset_cache()
    try:
        text = ("The museum is open. It is not open. Therefore the sign is wrong, "
               "however nobody read it. Every visitor never noticed.")
        found = _findings(text)
        for metric_id in logic_suite.FEATURE_METRICS["propositions"]:
            assert found[metric_id]["value"] is None
            assert found[metric_id]["warning"]
        for metric_id in (logic_suite.FEATURE_METRICS["negation_and_quantifiers"] +
                          logic_suite.FEATURE_METRICS["connective_relations"]):
            assert found[metric_id]["metric_id"] == metric_id  # present and computed
    finally:
        optional.reset_cache()


def test_a_disabled_feature_group_reports_why_not_silence():
    text = "The museum is open. It is not open. Therefore the sign is wrong."
    found = _findings(text, config={"features": {"propositions": False}})
    for metric_id in logic_suite.FEATURE_METRICS["propositions"]:
        item = found[metric_id]
        assert item["value"] is None
        assert "features.propositions=false" in item["warning"]


def test_degenerate_documents_do_not_crash():
    for text in ("", "Hi.", "A\n\nB\n\nC"):
        found = _findings(text)
        assert len(found) == sum(len(v) for v in logic_suite.FEATURE_METRICS.values())


def test_off_unless_enabled_via_grade(tmp_path, base_config):
    import grade
    source = tmp_path / "story.txt"
    source.write_text("The museum is open. It is not open.", encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results if item.metric_id.startswith("discourse.logic_")]


@requires_spacy
def test_enabled_via_grade_reports_expected_ids(tmp_path, base_config):
    import grade
    source = tmp_path / "story.txt"
    source.write_text(
        "The museum is open every single day. It is not open on Mondays, however. "
        "Therefore the sign outside is misleading. Because the sign is old, nobody "
        "trusts it. If the town replaces it, visitors will notice.",
        encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"], "logic_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    ids = {item.metric_id for item in report.results}
    for group_ids in logic_suite.FEATURE_METRICS.values():
        for metric_id in group_ids:
            assert metric_id in ids, metric_id


# ------------------------------------------------------- no unsupported claims

@requires_spacy
def test_missing_shared_property_is_not_a_conflict():
    """Two propositions about unrelated objects must never be flagged.

    Regression for the requirement that an absent lexical-resource edge (here,
    simply two different subjects) is never treated as a contradiction.
    """

    text = "The kettle is red. The umbrella is blue."
    found = _findings(text)
    item = found["discourse.logic_entity_attribute_conflict_candidates"]
    assert item["distribution"]["candidate_count"] == 0


@requires_spacy
def test_pronoun_subjects_are_excluded_from_cross_sentence_matching():
    text = "She was tired. She was not tired."
    analysis = DocumentAnalysis.from_text(text, comparison_unit="book")
    from textgrader import propositions as prop_lib
    extraction = prop_lib.extract(analysis, 1000)
    assert all(p.subject_key == "" for p in extraction.propositions if p.subject_is_pronoun)
    found = _findings(text)
    # No usable non-pronoun subject anywhere in this tiny document.
    item = found["discourse.logic_negation_flip_candidates"]
    assert item["distribution"]["candidate_count"] == 0
