"""The experimental discourse-coherence suite: off by default, independently
switchable, honest about sample size, and never confused by a missing
optional package.
"""

from __future__ import annotations

import time

import pytest

import grade
from textgrader import coherence as coh
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
                    "coreference": False, "connectives": False, "order_permutation": False}})
    entity_ids = _ids(entity_only)
    channels = ("", "_narration", "_dialogue")
    stems = ("entity_new_given_ratio", "entity_reintroduction_distance", "entity_dangling_rate",
            "entity_grid_transition_entropy")
    expected = {f"discourse.coherence_{stem}{tag}" for stem in stems for tag in channels}
    expected.add("discourse.coherence_entity_graph_density")
    expected.add("discourse.coherence_entity_grid_transition_corpus_delta")
    assert entity_ids == expected
    # coreference is off by default even when entity is on, and it is a
    # wholly separate feature flag: no "_coref" id leaks in here.
    assert not any(mid.endswith("_coref") for mid in entity_ids)

    coref_only = coherence_suite.measure(_analysis(sample_text), config={
        "features": {"lexical": False, "semantic": False, "entity": False,
                    "coreference": True, "connectives": False, "order_permutation": False},
        "coreference_max_words": 200})
    coref_ids = _ids(coref_only)
    assert coref_ids == {f"discourse.coherence_{stem}_coref" for stem in stems} | {
        "discourse.coherence_entity_graph_density_coref"}


def test_config_and_registry_declare_the_same_option_surface():
    """The whole tunable surface must appear in both places, per the task spec."""

    import json
    from pathlib import Path
    from textgrader.metrics import REGISTRY

    spec = REGISTRY["coherence_suite"]
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    configured = dict(config["metrics"]["coherence_suite"])
    configured.pop("enabled")
    # "_"-prefixed keys are documentation for whoever reads config.json (what
    # a feature needs installed, what happens without it) -- grade.py's
    # metric_options() and options_match_profile() strip them before they
    # reach a metric or affect profile comparability, so they are not part of
    # the tunable option surface this test is otherwise checking parity on.
    for key in [key for key in configured if key.startswith("_")]:
        assert configured.pop(key)  # every note must actually say something
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


# --------------------------------------------------------- real coreference

def _alice_bob_text():
    sentences = []
    for _ in range(6):
        sentences.append("Alice opened the heavy door and looked around the empty room.")
        sentences.append("She had not expected the room to be so cold.")
        sentences.append("Bob called her name from the hallway outside.")
        sentences.append("Alice answered him without turning around.")
    return _paragraph_of(sentences)


def test_no_coreference_model_is_loaded_under_the_default_config(monkeypatch, manuscript,
                                                                 base_config):
    """The critical gating rule: coherence_suite's cost stays "parse", so
    corpus.py's needs_model guard (which only checks for
    "sentence_transformers") never sees a fastcoref-backed feature - nothing
    but this suite's own off-by-default feature flag protects a 300,000-word
    novel from an unwanted coreference pass, so that flag must actually work.
    """

    def _boom(model_name):
        raise AssertionError(f"fastcoref model {model_name!r} must not load by default")

    monkeypatch.setattr(coh, "_load_coref_model", _boom)
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "coherence_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)  # default features: coreference is off
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
    assert not any(item.metric_id.endswith("_coref") for item in report.results)


def test_coreference_backend_is_off_by_default_even_with_the_suite_enabled(sample_text):
    ids = _ids(coherence_suite.measure(_analysis(sample_text)))
    assert not any(mid.endswith("_coref") for mid in ids)


def test_coreference_resolves_pronouns_the_surface_backend_cannot():
    """The disagreement the module docstring promises: two implementations
    of "the same entity recurs", one blind to pronouns and one that resolves
    them, must actually produce different numbers on text built exactly to
    show the gap - if they agreed exactly here, the coreference backend would
    not really be doing anything.
    """

    analysis = _analysis(_alice_bob_text())
    if analysis.nlp_unavailable:
        pytest.skip("spaCy is not available in this environment")
    module, reason = optional.require("fastcoref")
    if module is None:
        pytest.skip(f"fastcoref is not available in this environment: {reason}")

    findings = {item["metric_id"]: item for item in coherence_suite.measure(
        analysis, config={"features": {"entity": True, "coreference": True, "lexical": False,
                                       "semantic": False, "connectives": False,
                                       "order_permutation": False},
                          "coreference_max_words": 2000})}

    surface = findings["discourse.coherence_entity_reintroduction_distance"]
    coref = findings["discourse.coherence_entity_reintroduction_distance_coref"]
    assert surface["value"] is not None and coref["value"] is not None
    # "She"/"her"/"him" immediately follow the name they refer to, so real
    # coreference sees much tighter continuity than lemma matching, which is
    # blind to every pronoun mention entirely.
    assert coref["value"] < surface["value"]
    assert coref["distribution"]["backend"] == "coreference"
    assert surface["distribution"]["backend"] == "surface_lemma"

    surface_new_given = findings["discourse.coherence_entity_new_given_ratio"]
    coref_new_given = findings["discourse.coherence_entity_new_given_ratio_coref"]
    assert (coref_new_given["distribution"]["given_mentions"]
           > surface_new_given["distribution"]["given_mentions"])

    graph = findings["discourse.coherence_entity_graph_density_coref"]
    if graph["value"] is not None:
        assert graph["distribution"]["backend"] == "coreference"
        assert graph["distribution"]["model"]


def test_coreference_window_is_bounded_and_recorded(sample_text):
    analysis = _analysis(sample_text)
    if analysis.nlp_unavailable:
        pytest.skip("spaCy is not available in this environment")
    module, reason = optional.require("fastcoref")
    if module is None:
        pytest.skip(f"fastcoref is not available in this environment: {reason}")
    findings = {item["metric_id"]: item for item in coherence_suite.measure(
        analysis, config={"features": {"coreference": True, "lexical": False, "semantic": False,
                                       "entity": False, "connectives": False,
                                       "order_permutation": False},
                          "coreference_max_words": 50})}
    given_new = findings["discourse.coherence_entity_new_given_ratio_coref"]
    assert given_new["distribution"]["window_word_cap"] == 50
    assert given_new["distribution"]["window_words"] is not None
    assert "sentences_considered" in given_new["distribution"]


# --------------------------------------------------------- WordNet cohesion

def test_wordnet_lexical_chain_is_reported_alongside_the_identity_chain():
    wn, reason = coh.require_wordnet()
    if wn is None:
        pytest.skip(f"WordNet is not available in this environment: {reason}")
    sentences = ["The old car rattled down the lane.",
                "Overhead a hawk circled once.",
                "That automobile had not been serviced in years.",
                "The dog slept through the whole afternoon."]
    analysis = _analysis(_paragraph_of(sentences))
    findings = {item["metric_id"]: item for item in coherence_suite.measure(
        analysis, config={"features": {"lexical": True, "lexical_wordnet": True,
                                       "semantic": False, "entity": False,
                                       "connectives": False, "order_permutation": False}})}
    identity = findings["discourse.coherence_lexical_chain_coverage"]
    wordnet = findings["discourse.coherence_lexical_chain_coverage_wordnet"]
    assert identity["value"] is not None and wordnet["value"] is not None
    assert identity["distribution"]["backend"] == "identity"
    assert wordnet["distribution"]["backend"] == "wordnet"
    # "car" and "automobile" never repeat a word, so the WordNet-aware chain
    # covers strictly more of this fixture than the identity-only one.
    assert wordnet["value"] > identity["value"]


def test_hypernym_chain_catches_a_recurring_category_the_stricter_chains_miss():
    """The disagreement this channel exists to show: "car" and "truck" never
    repeat a word (defeats the identity chain) and are not each other's
    first-sense synonym either (defeats the WordNet synonym chain), but they
    are both close hyponyms of "motor vehicle" -- exactly the kind of
    category repetition only the hypernym-proximity chain can see."""

    wn, reason = coh.require_wordnet()
    if wn is None:
        pytest.skip(f"WordNet is not available in this environment: {reason}")
    sentences = ["The old car rattled down the lane.",
                "A battered truck rumbled past soon after.",
                "Overhead a hawk circled once in silence.",
                "A different truck honked twice near the corner."]
    analysis = _analysis(_paragraph_of(sentences))
    findings = {item["metric_id"]: item for item in coherence_suite.measure(
        analysis, config={"features": {"lexical": True, "lexical_wordnet": True,
                                       "lexical_wordnet_hypernym": True, "semantic": False,
                                       "entity": False, "connectives": False,
                                       "order_permutation": False}})}
    identity = findings["discourse.coherence_lexical_chain_coverage"]
    wordnet = findings["discourse.coherence_lexical_chain_coverage_wordnet"]
    hypernym = findings["discourse.coherence_lexical_chain_coverage_hypernym"]
    assert hypernym["value"] is not None
    assert hypernym["distribution"]["backend"] == "wordnet_hypernym"
    assert hypernym["distribution"]["mean_link_hypernym_distance"] is not None
    # "car" and "truck" (twice) link only under the loose, hypernym-proximity
    # criterion, so this channel covers strictly more of the fixture than
    # either stricter chain, which is the whole point of reporting it
    # separately rather than folding it into one of them.
    assert hypernym["value"] > identity["value"]
    assert hypernym["value"] > wordnet["value"]


def test_hypernym_chain_is_off_by_default_even_with_lexical_wordnet_on(sample_text):
    ids = _ids(coherence_suite.measure(
        _analysis(sample_text), config={"features": {"lexical_wordnet": True}}))
    assert "discourse.coherence_lexical_chain_coverage_hypernym" not in ids


def test_lexical_wordnet_is_off_by_default(sample_text):
    ids = _ids(coherence_suite.measure(_analysis(sample_text)))
    assert "discourse.coherence_lexical_chain_coverage_wordnet" not in ids


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


# ---------------------------------------------------- corpus profile round trip

_TRANSITION_ONLY_FEATURES = {"lexical": False, "lexical_wordnet": False,
                            "lexical_wordnet_hypernym": False, "semantic": False,
                            "entity": True, "coreference": False, "connectives": False,
                            "order_permutation": False}


def test_profile_vector_caches_the_surface_entity_grid_transition_table(sample_text):
    analysis = _analysis(sample_text)
    if analysis.nlp_unavailable:
        pytest.skip("spaCy is not available in this environment")
    vector = coherence_suite.profile_vector(analysis, {"entity_max_tracked": 150})
    assert vector
    assert set(vector) == set(coh.TRANSITION_KEYS)
    assert sum(vector.values()) == pytest.approx(1.0)


def test_corpus_profile_round_trip_carries_entity_grid_transition_tables(tmp_path, base_config,
                                                                         prose):
    """The round trip is the point: build a profile with the suite enabled and
    both the parse and model flags on (this suite needs both - see the module
    docstring), confirm ``feature_profiles`` actually gained a per-book
    transition-frequency row via ``profile_vector``, then confirm a grading
    run's corpus-delta finding uses those cached rows rather than reporting
    "no corpus profile available"."""

    from textgrader.corpus import build_profile, write_profile

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    book_count = 5
    for index in range(book_count):
        (corpus_dir / f"book{index}.txt").write_text(prose(400 + index, paragraphs=20),
                                                      encoding="utf-8")
    metrics_config = {"coherence_suite": {"enabled": True, "features": _TRANSITION_ONLY_FEATURES}}
    # metric_selection="auto" would fall back to "all" here because ``metrics``
    # is not None it already IS supplied, so this also exercises the ordinary
    # "enabled" selection path grade.py's own corpus builder uses.
    profile = build_profile([corpus_dir], metrics=metrics_config,
                            include_parse_metrics=True, include_model_metrics=True)
    if analysis_unavailable := not profile["feature_profiles"].get("coherence_suite"):
        pytest.skip(f"spaCy is not available in this environment "
                   f"(metric_errors={profile['metric_errors']})")

    rows = profile["feature_profiles"]["coherence_suite"]
    assert len(rows) == book_count
    for row in rows:
        assert row, "every book in this fixture has recurring entities"
        assert set(row) <= set(coh.TRANSITION_KEYS)
        assert sum(row.values()) == pytest.approx(1.0)

    write_profile(profile, tmp_path / "profile.json")
    manuscript_path = tmp_path / "manuscript.txt"
    manuscript_path.write_text(prose(999, paragraphs=20), encoding="utf-8")
    config = {**base_config, "corpus_profile": "profile.json",
              "metrics": {**base_config["metrics"], **metrics_config}}
    report = grade.analyze(manuscript_path, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]

    delta = next(item for item in report.results
                if item.metric_id == "discourse.coherence_entity_grid_transition_corpus_delta")
    assert delta.value is not None, delta.warning
    assert delta.distribution["corpus_size"] == book_count
    assert delta.distribution["backend"] == "surface_lemma"
    assert delta.distribution["role_schema"] == coh.ROLE_SCHEMA_VERSION


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
