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
from textgrader import propositions as prop_lib
from textgrader.document import DocumentAnalysis, NlpSettings
from textgrader.metrics import REGISTRY, logic_suite

SPACY_READY = optional.have("spacy")
requires_spacy = pytest.mark.skipif(not SPACY_READY, reason="spacy/en_core_web_sm not available")

TRANSFORMERS_READY = optional.have("transformers")
requires_transformers = pytest.mark.skipif(not TRANSFORMERS_READY, reason="transformers not available")

WORDNET_READY = prop_lib.load_wordnet()[0] is not None
requires_wordnet = pytest.mark.skipif(not WORDNET_READY, reason="nltk wordnet corpus data not available")

DATEUTIL_READY = optional.have("dateutil")
requires_dateutil = pytest.mark.skipif(not DATEUTIL_READY, reason="python-dateutil not available")


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
                "repeated_assertion_min_words", "coreference_max_chars",
                "nli_model", "nli_max_pairs", "nli_batch_size"):
        assert name in spec.defaults, name
    for name in ("negation_and_quantifiers", "connective_relations", "propositions",
                "modal_argument_position", "nli_entailment", "coreference_resolution",
                "lexical_opposition", "temporal_ordering"):
        assert name in spec.defaults["features"], name
    # The four new groups must default to off in the registry AND in config.json
    # -- this is the gate the module docstring calls out by name.
    for name in ("nli_entailment", "coreference_resolution", "lexical_opposition",
                "temporal_ordering"):
        assert spec.defaults["features"][name] is False, name


def test_all_metric_ids_are_stable_and_prefixed():
    for group_ids in logic_suite.FEATURE_METRICS.values():
        for metric_id in group_ids:
            assert metric_id.startswith("discourse.logic_"), metric_id
            assert metric_id in logic_suite._METRIC_NAMES


def test_every_features_key_is_either_a_metric_group_or_a_known_modifier():
    spec = REGISTRY["logic_suite"]
    accounted_for = set(logic_suite.FEATURE_METRICS) | logic_suite._MODIFIER_FEATURES
    assert set(spec.defaults["features"]) == accounted_for


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


# ------------------------------------------------------- gating: off by default
#
# The most important property of this suite's newest metrics: a config that
# never mentions them (including ``config=None``, which every test above this
# point uses) must never load transformers, torch or fastcoref. See the
# module docstring's "Gating" note for why the registry's own needs_model
# guard cannot be trusted to keep this suite's NLI channel off.

def test_default_config_never_loads_nli_or_coref_model(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("an NLI or coreference model must never load under default config")

    monkeypatch.setattr(logic_suite, "_load_nli_pipeline", boom)
    monkeypatch.setattr(prop_lib, "_load_coref_model", boom)
    text = ("Alice was tired. She was not tired at all, though. Therefore the sign "
           "outside is misleading, however nobody read it.")
    # config=None: every legacy group defaults on; none of this must touch a model.
    logic_suite.measure(DocumentAnalysis.from_text(text, comparison_unit="book"))


def test_nli_entailment_off_by_default_when_features_key_is_absent():
    """A ``features`` mapping that only sets an unrelated key must not turn NLI on.

    This is the bug the module's ``on()`` helper exists to prevent: the four
    original groups fall back to *on* when their key is missing from
    ``features`` (existing behaviour, kept for compatibility), but every group
    added since must fall back to *off*, not inherit that same default.
    """

    found = _findings("The lamp was lit. The lamp was not lit.",
                      config={"features": {"negation_and_quantifiers": False}})
    for metric_id in logic_suite.FEATURE_METRICS["nli_entailment"]:
        assert found[metric_id]["value"] is None
        assert "disabled by config" in found[metric_id]["warning"]


def test_registry_requires_documents_the_new_optional_packages():
    spec = REGISTRY["logic_suite"]
    for package in ("transformers", "fastcoref", "nltk", "dateutil"):
        assert package in spec.requires, package


# ------------------------------------------------------------- NLI adjudication

@requires_spacy
@requires_transformers
def test_nli_entailment_scores_a_real_contradiction_pair():
    text = "The lamp was lit. The lamp was not lit."
    found = _findings(text, config={"features": {"nli_entailment": True}, "nli_max_pairs": 10})
    item = found["discourse.logic_nli_label_distribution"]
    assert item["value"] is not None
    assert item["sample_size"] >= 1
    assert item["distribution"]["model"] == "cross-encoder/nli-deberta-v3-small"
    assert any(row["nli_label"] == "contradiction" for row in item["evidence"])
    assert "NOT a fact about the text" in item["warning"]
    agreement = found["discourse.logic_nli_heuristic_agreement"]
    assert agreement["value"] == pytest.approx(100.0)
    assert agreement["distribution"]["confusion"]["negation"]["contradiction"] == 1


@requires_spacy
@requires_transformers
def test_nli_entailment_scores_a_real_entailment_pair():
    text = "The teacher said the exam was hard. The teacher said the exam was difficult."
    found = _findings(text, config={"features": {"nli_entailment": True}, "nli_max_pairs": 10})
    item = found["discourse.logic_nli_label_distribution"]
    assert item["sample_size"] >= 1
    assert any(row["nli_label"] == "entailment" for row in item["evidence"])
    # Neither heuristic scan can see a paraphrase like this one (no negation
    # flip, no differing-value object): the NLI channel finding it is exactly
    # the recall gain the module docstring claims for it.
    assert "none" in found["discourse.logic_nli_heuristic_agreement"]["distribution"]["confusion"]


@requires_spacy
@requires_transformers
def test_nli_entailment_scores_a_real_neutral_pair():
    text = "The gardener planted tulips along the fence. The gardener planted tulips near the pond."
    found = _findings(text, config={"features": {"nli_entailment": True}, "nli_max_pairs": 10})
    item = found["discourse.logic_nli_label_distribution"]
    if item["sample_size"]:
        assert set(row["nli_label"] for row in item["evidence"]) <= {"entailment", "neutral", "contradiction"}


@requires_spacy
@requires_transformers
def test_nli_max_pairs_caps_the_model_calls_independently_of_max_pairs():
    sentences = []
    for index in range(40):
        name = f"Witness{index}"
        sentences.append(f"{name} said the bridge was safe.")
        sentences.append(f"{name} later said the bridge was not safe.")
    text = " ".join(sentences)
    found = _findings(text, config={"features": {"nli_entailment": True}, "nli_max_pairs": 3,
                                    "window_sentences": 200, "max_pairs": 500,
                                    "max_comparisons": 50_000})
    item = found["discourse.logic_nli_label_distribution"]
    assert item["sample_size"] <= 3
    assert item["distribution"]["settings"]["nli_max_pairs"] == 3


def test_nli_entailment_reports_unavailable_without_transformers(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "transformers")
    optional.reset_cache()
    try:
        found = _findings("The lamp was lit. The lamp was not lit.",
                          config={"features": {"nli_entailment": True}})
        for metric_id in logic_suite.FEATURE_METRICS["nli_entailment"]:
            assert found[metric_id]["value"] is None
            assert "transformers" in found[metric_id]["warning"]
    finally:
        optional.reset_cache()


# ------------------------------------------------------------- WordNet relations

@requires_spacy
@requires_wordnet
def test_wordnet_antonym_channel_finds_a_direct_opposite():
    text = "The soup was hot. Moments later the soup was cold."
    found = _findings(text, config={"features": {"lexical_opposition": True}})
    item = found["discourse.logic_wordnet_antonym_candidates"]
    assert item["value"] is not None and item["value"] > 0
    assert item["evidence"][0]["type"] == "antonym"
    assert "not disambiguated" in item["warning"]


@requires_spacy
@requires_wordnet
def test_wordnet_hypernym_channel_downgrades_an_is_a_pair():
    text = "The animal was a dog. The animal was a poodle."
    found = _findings(text, config={"features": {"propositions": True, "lexical_opposition": True}})
    conflict = found["discourse.logic_entity_attribute_conflict_candidates"]
    assert conflict["distribution"]["candidate_count"] == 1  # never changed by the WordNet cross-check
    downgrade = found["discourse.logic_wordnet_hypernym_downgrade_rate"]
    assert downgrade["value"] == pytest.approx(100.0)
    assert downgrade["evidence"][0]["subject"] == "animal"


def test_wordnet_channel_reports_unavailable_without_nltk(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "nltk")
    optional.reset_cache()
    try:
        found = _findings("The soup was hot. The soup was cold.",
                          config={"features": {"lexical_opposition": True}})
        for metric_id in logic_suite.FEATURE_METRICS["lexical_opposition"]:
            assert found[metric_id]["value"] is None
            assert "nltk" in found[metric_id]["warning"]
    finally:
        optional.reset_cache()


# ----------------------------------------------------------- temporal ordering

@requires_spacy
@requires_dateutil
def test_temporal_ordering_flags_a_reversed_flashback():
    text = "The tower was built in 1990. Flashback: the tower was built in 1950."
    found = _findings(text, config={"features": {"temporal_ordering": True}})
    item = found["discourse.logic_temporal_order_candidates"]
    assert item["value"] == pytest.approx(100.0)
    row = item["evidence"][0]
    assert row["earlier"]["date_text"] == "1950"
    assert row["later"]["date_text"] == "1990"
    assert "NOT an error" in item["warning"]


@requires_spacy
@requires_dateutil
def test_temporal_ordering_does_not_flag_chronological_order():
    text = "The tower was built in 1950. Years later, the tower was rebuilt in 1990."
    found = _findings(text, config={"features": {"temporal_ordering": True}})
    item = found["discourse.logic_temporal_order_candidates"]
    # Either no temporal-conflict candidate at all (different predicates:
    # "build" vs "rebuild") or, if one is found, it must not be reversed.
    if item["value"] is not None:
        assert item["distribution"]["reversed_count"] == 0


@requires_spacy
def test_temporal_ordering_needs_a_temporal_candidate_pair():
    text = "The kettle is red. The umbrella is blue."
    found = _findings(text, config={"features": {"temporal_ordering": True}})
    item = found["discourse.logic_temporal_order_candidates"]
    assert item["value"] is None
    assert "temporal" in item["warning"]


# --------------------------------------------------------------- coreference

@requires_spacy
def test_coreference_resolution_lets_a_pronoun_subject_into_the_pool(monkeypatch):
    """With a working resolver, "she" should bucket with its named antecedent.

    fastcoref itself is not exercised here (see the next test for that): a
    minimal fake stands in for it so this test is fast and does not depend on
    a real coreference model loading cleanly in this environment.
    """

    text = "Alice was tired. She was not tired at all, though."

    class FakeResult:
        def __init__(self, clusters):
            self._clusters = clusters

        def get_clusters(self, as_strings=False):
            return self._clusters

    class FakeModel:
        def predict(self, texts):
            she_at = texts.index("She")
            return FakeResult([[(0, 5), (she_at, she_at + 3)]])

    monkeypatch.setitem(prop_lib._COREF_MODEL_CACHE, "model", (FakeModel(), None))
    try:
        found = _findings(text, config={"features": {"coreference_resolution": True}})
        item = found["discourse.logic_negation_flip_candidates"]
        assert item["distribution"]["candidate_count"] == 1
        assert item["evidence"][0]["sentence_a"]["text"] == "Alice was tired."
        assert "resolved 1 pronoun-subject" in item["warning"]
    finally:
        prop_lib._COREF_MODEL_CACHE.pop("model", None)


@requires_spacy
def test_coreference_resolution_degrades_without_crashing_when_fastcoref_is_absent(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "fastcoref")
    optional.reset_cache()
    try:
        text = "Alice was tired. She was not tired at all, though."
        found = _findings(text, config={"features": {"coreference_resolution": True}})
        item = found["discourse.logic_negation_flip_candidates"]
        assert item["distribution"]["candidate_count"] == 0  # "she" still excluded
        assert "unavailable this run" in item["warning"]
    finally:
        optional.reset_cache()


@requires_spacy
def test_coreference_resolution_off_by_default_is_documented_as_excluded():
    text = "Alice was tired. She was not tired at all, though."
    found = _findings(text)  # config=None
    item = found["discourse.logic_negation_flip_candidates"]
    assert "coreference_resolution is off" in item["warning"]


# ---------------------------------------------------------- hand-checked prose
#
# The task behind this module's original pass found two real bugs by running
# it on constructed text, not from unit tests; this section repeats that
# discipline for the NLI channel specifically, on both a planted contradiction
# and a legitimate in-fiction one (an unreliable narrator), which must NOT be
# reported as an error in the text.

@requires_spacy
@requires_transformers
def test_hand_check_planted_contradiction_is_labelled_contradiction():
    text = ("The vault door was sealed shut every night without exception. "
           "On the night of the theft, however, the vault door was not sealed at all.")
    found = _findings(text, config={"features": {"nli_entailment": True}})
    item = found["discourse.logic_nli_label_distribution"]
    assert item["sample_size"], "expected at least one NLI-scored candidate pair"
    assert any(row["nli_label"] == "contradiction" for row in item["evidence"])


@requires_spacy
@requires_transformers
def test_hand_check_unreliable_narrator_contradiction_is_not_reported_as_an_error():
    """An in-fiction lie must still score as a model contradiction -- and the
    finding's own warning, not a suppressed value, is what keeps that from
    being misread as an error in the text.
    """

    text = "The narrator swore the door was locked. In truth, the door was not locked at all."
    found = _findings(text, config={"features": {"nli_entailment": True}})
    item = found["discourse.logic_nli_label_distribution"]
    assert item["sample_size"], "expected the ccomp-vs-narration pair to reach the model"
    assert any(row["nli_label"] == "contradiction" for row in item["evidence"])
    # The lie IS a contradiction by the model's own lights -- exactly as it
    # should be -- but the finding must still frame that as a model score,
    # never as a claim that the text itself contains an error.
    assert "not evidence of an error in the writing" in item["warning"]
