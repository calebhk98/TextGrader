"""Contract tests for the experimental conversational-dynamics suite.

Mirrors the shape of ``test_coherence_suite.py``/``test_stylometry_suite.py``:
wiring/off-by-default/registry-parity checks, graceful degradation under a
missing optional package, degenerate-document survival, and the specific
separation scenarios the task's "Tests and validation" section names by
name: two-speaker alternation vs a one-speaker-dominant scene, distinct
speaker vocabularies vs deliberately identical voices, sparse attribution
suppressing every per-speaker claim, a question-answer fixture, and
dialogue-free text degrading cleanly.

``_dialogue()`` below builds one-quotation-per-paragraph fixtures with the
speech tag trailing the quote ('"Text," Name said.'). This is not an
arbitrary style choice: it is the one shaping that
``dialogue_attribution.classify_turns`` (which this suite reuses verbatim,
never re-parsing a quotation itself) resolves unambiguously. Two other
plausible shapes were tried while building this file and rejected because
the shared attribution code -- correctly, by its own documented "text
heuristic, not a parse" contract -- does not resolve them the way a human
reader would:

* Two tag-split quotes per paragraph ('"X," Alice said. "Y?" Bob asked.
  "Z," Alice said.') -- with nothing but a blank line between paragraphs,
  the merge rule in ``dialogue_attribution.turn_spans``/``DocumentAnalysis.
  turns`` (a gap counts as ending a turn only if it contains sentence-final
  punctuation) sees no such punctuation in a bare blank line and merges
  every paragraph into a single giant turn.
* A leading tag before every quote ('Alice said, "X." Bob said, "Y?"') --
  the gap between two quotes then carries the *next* quote's tag, and
  ``classify_turns`` checks a turn's ``before`` context ahead of its
  ``after`` context, so each quote after the first is mis-attributed to the
  speaker of the quote that follows it, not the one that precedes it.

Both are documented here, with the fixtures that demonstrated them kept in
``test__dialogue_helper_matches_the_documented_attribution_shape`` below, so
a future reader does not have to rediscover this by hand.
"""

from __future__ import annotations

import math
import statistics

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, conversation_suite as cs

PREFIX = "dialogue.conversation_"


def _dialogue(turns: list[tuple[str, str]]) -> str:
    """One spoken turn per paragraph, tag trailing the quote -- see module
    docstring for why this is the fixture shape that attributes cleanly."""

    paragraphs = []
    for speaker, text in turns:
        verb = "asked" if text.rstrip().endswith("?") else "said"
        paragraphs.append(f'"{text}" {speaker} {verb}.')
    return "\n\n".join(paragraphs)


def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book")


def _findings(text: str, config=None) -> dict[str, dict]:
    return {item["metric_id"]: item for item in cs.measure(_analysis(text), config=config)}


# ------------------------------------------------------------------- wiring

def test_off_by_default(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(PREFIX)]


def test_enabling_the_suite_turns_on_every_default_feature(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "conversation_suite": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(manuscript, config).results
          if item.metric_id.startswith(PREFIX)}
    for name in ("attribution_coverage", "speaker_dominance_gini", "speaker_alternation_rate",
                "question_response_rate", "turn_length_accommodation",
                "function_word_coordination", "contraction_convergence",
                "question_mark_convergence", "lexical_entrainment", "response_relevance",
                "politeness_strategy_rate", "sentiment_coupling", "speaker_sentiment_spread",
                "speaker_separability_accuracy", "graph_density", "graph_reciprocity"):
        assert PREFIX + name in ids, name
    # Off-by-default features must not leak an id even with the suite enabled.
    for name in ("pos_pattern_convergence", "dialogue_act_distribution",
                "dialogue_act_transition_entropy", "scene_style_drift"):
        assert PREFIX + name not in ids, name


def test_every_finding_uses_the_stable_prefix_and_family(sample_text):
    findings = cs.measure(_analysis(sample_text))
    assert findings
    for item in findings:
        assert item["metric_id"].startswith(PREFIX)
        assert item["family"] == "dialogue"


def test_every_finding_carries_sample_size_and_floor(sample_text):
    for item in cs.measure(_analysis(sample_text)):
        assert "sample_size" in item
        assert "min_sample" in item


def test_config_and_registry_declare_the_same_option_surface():
    import json
    from pathlib import Path

    spec = REGISTRY["conversation_suite"]
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    configured = dict(config["metrics"]["conversation_suite"])
    configured.pop("enabled")
    for key in [key for key in configured if key.startswith("_")]:
        assert configured.pop(key)  # every note must actually say something
    assert set(configured) == set(spec.defaults)
    assert configured["features"] == spec.defaults["features"]


# --------------------------------------------------------------- gating rule

def test_sentence_transformers_and_spacy_are_not_in_requires():
    """The critical gating rule: this suite's cost is "moderate" and its
    MetricSpec.requires names neither "spacy" nor "sentence_transformers", so
    corpus.py's needs_parse/needs_model guards never pull the whole suite --
    dominance, coordination, entrainment, none of which need a parse or a
    model -- out of a corpus profile just because two off-by-default
    features (pos_convergence, response_relevance_embedding) can use them."""

    spec = REGISTRY["conversation_suite"]
    assert "sentence_transformers" not in spec.requires
    assert "spacy" not in spec.requires
    assert spec.needs_model is False
    assert spec.needs_parse is False


def test_default_config_never_loads_the_dialogue_act_model(monkeypatch, sample_text):
    def _boom(model_name):
        raise AssertionError(f"dialogue-act model {model_name!r} must not load by default")

    monkeypatch.setattr(cs, "_load_dialogue_act_pipeline", _boom)
    findings = cs.measure(_analysis(sample_text))
    assert findings
    assert not any(item["metric_id"].startswith(PREFIX + "dialogue_act") for item in findings)


def test_default_config_never_calls_the_pos_convergence_feature(monkeypatch, sample_text):
    def _boom(*args, **kwargs):
        raise AssertionError("pos_convergence must not run under the default config")

    monkeypatch.setattr(cs, "_finding_pos_convergence", _boom)
    findings = cs.measure(_analysis(sample_text))
    assert findings


def test_default_config_never_calls_scene_drift(monkeypatch, sample_text):
    def _boom(*args, **kwargs):
        raise AssertionError("scene_drift must not run under the default config")

    monkeypatch.setattr(cs, "_finding_scene_drift", _boom)
    findings = cs.measure(_analysis(sample_text))
    assert findings


def test_response_relevance_embedding_upgrade_is_off_by_default(monkeypatch):
    """response_relevance itself is on by default (lexical backend, no
    download); only the embedding UPGRADE is gated, and disabled by default
    the embedding loader must never be touched."""

    def _boom(texts, model_name):
        raise AssertionError("embed_texts must not be called unless "
                             "response_relevance_embedding is explicitly enabled")

    monkeypatch.setattr(cs, "embed_texts", _boom)
    turns = []
    for i in range(12):
        turns.append(("Alice", f"This is a fairly ordinary sentence number {i} about the day."))
        turns.append(("Bob", f"That is a reasonably plain reply number {i} about the weather."))
    findings = _findings(_dialogue(turns))
    relevance = findings[PREFIX + "response_relevance"]
    assert relevance["value"] is not None
    assert relevance["distribution"]["backend"] == "lexical"


# ------------------------------------------------------------- degradation

def test_missing_optional_packages_degrade_only_the_metrics_that_need_them(monkeypatch,
                                                                          sample_text):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        findings = _findings(sample_text, config={"min_turns_per_speaker": 2,
                                                   "min_coordination_pairs": 1})
    finally:
        optional.reset_cache()

    # Dependency-free groups still produce real numbers.
    assert findings[PREFIX + "backchannel_rate"]["value"] is not None
    assert findings[PREFIX + "politeness_strategy_rate"]["value"] is not None
    assert findings[PREFIX + "function_word_coordination"]["value"] is not None

    # vaderSentiment/nrclex-dependent findings say why they could not run.
    sentiment = findings[PREFIX + "sentiment_coupling"]
    emotion = findings[PREFIX + "emotion_coupling"]
    assert sentiment["value"] is None and sentiment["warning"]
    assert emotion["value"] is None and emotion["warning"]

    # The graph still computes its manual density/reciprocity without
    # networkx; it just cannot cross-check them, which shows up as a note,
    # not a missing value.
    graph = findings[PREFIX + "graph_density"]
    if graph["value"] is not None:
        assert graph["distribution"].get("networkx_note")


def test_wordfreq_missing_degrades_rare_word_entrainment_to_a_length_proxy(monkeypatch):
    turns = []
    for i in range(15):
        turns.append(("Alice", f"We discussed the sesquipedalian phenomenon number {i} again."))
        turns.append(("Bob", f"The sesquipedalian phenomenon number {i} was discussed again."))
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "wordfreq")
    optional.reset_cache()
    try:
        findings = _findings(_dialogue(turns), config={"min_turns_per_speaker": 2,
                                                        "min_entrainment_pairs": 3})
    finally:
        optional.reset_cache()
    rare = findings[PREFIX + "rare_word_entrainment"]
    assert rare["value"] is not None
    assert "length" in rare["warning"]
    # Its sibling, which does not need wordfreq, is unaffected.
    assert findings[PREFIX + "lexical_entrainment"]["value"] is not None


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC",
                                  "He walked home. He walked home again. It rained."])
def test_survives_degenerate_and_dialogue_free_documents(text):
    findings = cs.measure(_analysis(text))
    assert findings
    for item in findings:
        assert item["metric_id"].startswith(PREFIX)
        if item["metric_id"] != PREFIX + "attribution_tagged_unnamed_gap":
            # every genuinely dialogue-dependent finding must be a clean
            # "no data" rather than a fabricated number, on text with no
            # quoted speech at all.
            assert item["value"] is None
            assert item["warning"]


def test_every_registered_metric_and_degenerate_documents_are_already_covered_elsewhere():
    """test_optional_metrics.py parametrizes REGISTRY for "runs without
    raising" and "survives a degenerate document"; this file adds only the
    suite-specific behaviour."""

    assert "conversation_suite" in REGISTRY


# ------------------------------------------------------- separation: dominance

def test_two_speaker_alternation_has_lower_dominance_and_higher_alternation_than_one_dominant():
    alternating = _dialogue([
        ("Alice", "Hi Bob, how are you doing."), ("Bob", "I am doing fine thanks Alice."),
        ("Alice", "That is good to hear today."), ("Bob", "What about you this week."),
        ("Alice", "I am doing well myself too."), ("Bob", "Glad to hear that from you."),
    ])
    dominant = _dialogue([
        ("Alice", "Line number one goes here now."), ("Alice", "Line number two goes here now."),
        ("Alice", "Line number three goes here now."), ("Alice", "Line number four goes here."),
        ("Alice", "Line number five goes here now."), ("Bob", "Just a single short line."),
    ])
    config = {"min_turns_per_speaker": 1}
    alt = _findings(alternating, config=config)
    dom = _findings(dominant, config=config)

    assert alt[PREFIX + "speaker_dominance_gini"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert dom[PREFIX + "speaker_dominance_gini"]["value"] == pytest.approx(1 / 3, abs=1e-9)
    assert dom[PREFIX + "speaker_dominance_gini"]["value"] > alt[PREFIX + "speaker_dominance_gini"]["value"]

    assert alt[PREFIX + "speaker_alternation_rate"]["value"] == pytest.approx(100.0)
    assert dom[PREFIX + "speaker_alternation_rate"]["value"] == pytest.approx(20.0)
    assert alt[PREFIX + "speaker_alternation_rate"]["value"] > dom[PREFIX + "speaker_alternation_rate"]["value"]

    assert dom[PREFIX + "top_speaker_turn_share"]["value"] == pytest.approx(5 / 6 * 100)
    assert alt[PREFIX + "top_speaker_turn_share"]["value"] == pytest.approx(50.0)


def test_gini_exact_values_on_hand_computable_vectors():
    assert cs._gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-9)
    assert cs._gini([1, 5]) == pytest.approx(1 / 3, abs=1e-9)
    assert cs._gini([1, 1, 1, 7]) == pytest.approx(0.45, abs=1e-9)
    assert cs._gini([]) is None
    assert cs._gini([5]) is None


# ---------------------------------------------------- separation: separability

def test_distinct_speaker_vocabularies_separate_better_than_identical_voices():
    """``_turn_vector`` deliberately does not look at raw content vocabulary
    (see the module docstring: length, contraction/question/exclamation
    markers, five function-word category rates -- "beyond" the existing
    function-word-DISTANCE metrics, not a content-word model). So the
    "distinct voices" fixture has to differ on those actual dimensions
    (contraction use, sentence length, terminal punctuation, auxiliary/
    article density) while the "identical voices" fixture is structurally
    identical and differs only in content words, which this vector cannot
    see and must not separate."""

    distinct_turns, identical_turns = [], []
    for i in range(10):
        distinct_turns.append(("Alice", f"I'm really glad you're here today {i}!"))
        distinct_turns.append(("Bob", f"The committee has decided that the matter will "
                                       f"proceed exactly as previously discussed {i}."))
        identical_turns.append(("Alice", f"The garden was full of golden light this "
                                          f"afternoon {i}."))
        identical_turns.append(("Bob", f"The house was full of quiet sound this "
                                        f"evening {i}."))
    config = {"min_turns_per_speaker": 2}
    distinct_result = _findings(_dialogue(distinct_turns),
                                config=config)[PREFIX + "speaker_separability_accuracy"]
    identical_result = _findings(_dialogue(identical_turns),
                                 config=config)[PREFIX + "speaker_separability_accuracy"]

    assert distinct_result["value"] is not None and identical_result["value"] is not None
    assert distinct_result["value"] > identical_result["value"]
    assert distinct_result["distribution"]["accuracy"] > 0.9
    assert identical_result["distribution"]["accuracy"] == pytest.approx(0.5, abs=0.15)


def test_separability_accuracy_helper_exact_on_a_trivial_fixture():
    """White-box: two speakers whose vectors are perfectly separated by one
    dimension must classify every turn correctly (accuracy 1.0)."""

    labeled = [("A", [0.0]) for _ in range(4)] + [("B", [10.0]) for _ in range(4)]
    result = cs._separability_accuracy(labeled)
    assert result is not None
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["chance"] == pytest.approx(0.5)
    assert result["excess"] == pytest.approx(0.5)


# --------------------------------------------------- separation: sparse attribution

def test_sparse_attribution_suppresses_every_per_speaker_claim_but_not_turn_adjacency():
    # Every turn is tagged only by pronoun ("she"/"he said"), which
    # dialogue_attribution's name heuristic cannot resolve to a speaker, so
    # almost nothing here is "named".
    turns = []
    for i in range(20):
        speaker_word = "She" if i % 2 == 0 else "He"
        turns.append(f'"This is turn number {i} in the conversation." {speaker_word} said.')
    text = "\n\n".join(turns)
    findings = _findings(text)

    coverage = findings[PREFIX + "attribution_coverage"]
    assert coverage["value"] == pytest.approx(0.0)

    for name in ("speaker_dominance_gini", "top_speaker_turn_share", "speaker_alternation_rate",
                "same_speaker_run_length", "function_word_coordination",
                "contraction_convergence", "lexical_entrainment", "sentiment_coupling",
                "speaker_sentiment_spread", "speaker_separability_accuracy", "graph_density"):
        item = findings[PREFIX + name]
        assert item["value"] is None, name
        assert item["warning"], name

    # Turn-adjacency findings do not need a name and must still work.
    assert findings[PREFIX + "backchannel_rate"]["value"] is not None
    assert findings[PREFIX + "politeness_strategy_rate"]["value"] is not None


def test_low_coverage_below_threshold_is_also_suppressed_even_with_two_named_speakers():
    """Two named speakers each clear min_turns, but they are a small island
    inside a much larger pool of pronoun-tagged turns: coverage stays below
    the default 25% gate, so the per-speaker claims must still be withheld."""

    turns = []
    for i in range(60):
        turns.append(f'"Unnamed line number {i} here." She said.')
    for i in range(9):
        turns.append(f'"Named follow-up number {i}." Alice said.')
        turns.append(f'"Named follow-up reply number {i}." Bob said.')
    text = "\n\n".join(turns)
    findings = _findings(text)
    assert findings[PREFIX + "attribution_coverage"]["value"] < 25.0
    dominance = findings[PREFIX + "speaker_dominance_gini"]
    assert dominance["value"] is None
    assert "below the" in dominance["warning"]


# ------------------------------------------------------------- separation: Q&A

def test_question_answer_fixture_gives_a_high_exact_response_rate():
    turns = [
        ("Alice", "Are you coming to the party?"), ("Bob", "Yes, I will be there."),
        ("Bob", "Did you finish the report?"), ("Alice", "No, not yet."),
        ("Alice", "Can you help me with this?"), ("Bob", "Sure, happy to help."),
    ]
    findings = _findings(_dialogue(turns))
    response = findings[PREFIX + "question_response_rate"]
    unanswered = findings[PREFIX + "unanswered_question_rate"]
    assert response["value"] == pytest.approx(100.0)
    assert unanswered["value"] == pytest.approx(0.0)
    assert response["sample_size"] == 3


def test_a_trailing_unanswered_question_is_detected_exactly():
    turns = [
        ("Alice", "Are you coming to the party?"), ("Bob", "Yes, I will be there."),
        ("Bob", "But do you actually want to go?"),
    ]
    findings = _findings(_dialogue(turns))
    response = findings[PREFIX + "question_response_rate"]
    unanswered = findings[PREFIX + "unanswered_question_rate"]
    # Two questions: the first is answered by Bob; the last has no reply at all.
    assert response["sample_size"] == 2
    assert response["value"] == pytest.approx(50.0)
    assert unanswered["value"] == pytest.approx(50.0)


def test_backchannel_rate_exact_on_a_short_reply_fixture():
    turns = [
        ("Alice", "I went to the market this morning."), ("Bob", "Okay."),
        ("Alice", "I bought some apples and some bread."), ("Bob", "Nice."),
        ("Alice", "Then I walked home along the river."), ("Bob", "I see, that sounds pleasant."),
    ]
    findings = _findings(_dialogue(turns))
    backchannel = findings[PREFIX + "backchannel_rate"]
    # 5 reply-like pairs (every adjacent pair here has a different speaker);
    # two of Bob's replies ("Okay.", "Nice.") are <= 2 words.
    assert backchannel["sample_size"] == 5
    assert backchannel["value"] == pytest.approx(100.0 * 2 / 5)


# --------------------------------------------------------- white-box primitives

def test_coordination_formula_exact_value_by_hand():
    """Two responders, one fully primed-and-matching, one never primed."""

    events = [
        {"speaker": "A", "text": "yes"}, {"speaker": "B", "text": "yes"},
        {"speaker": "A", "text": "no"}, {"speaker": "B", "text": "no"},
        {"speaker": "A", "text": "yes"}, {"speaker": "B", "text": "maybe"},
        {"speaker": "A", "text": "no"}, {"speaker": "B", "text": "maybe"},
    ]
    reply_pairs = cs._reply_pairs(events)
    marker = lambda text: text == "yes"
    rows, n_pairs = cs._coordination(reply_pairs, marker, min_primed=1)
    assert n_pairs == 7
    by_speaker = {row["speaker"]: row for row in rows}
    # B replies to A four times; A says "yes" twice (turns 0, 4). When A said
    # "yes", B's replies were "yes" then "maybe" -- conditional = 1/2 = 0.5.
    # B's overall reply-position baseline: "yes","no","maybe","maybe" -> 1/4.
    assert by_speaker["B"]["n_primed"] == 2
    assert by_speaker["B"]["conditional_rate"] == pytest.approx(50.0)
    assert by_speaker["B"]["baseline_rate"] == pytest.approx(25.0)
    assert by_speaker["B"]["coordination"] == pytest.approx(25.0)


def test_shuffle_contrast_real_rate_is_exact_and_deterministic():
    pairs = [("a b", "a c"), ("d e", "d f"), ("g h", "i j")]

    def overlap(prev, nxt):
        return 1.0 if set(prev.split()) & set(nxt.split()) else 0.0

    result = cs._shuffle_contrast(pairs, overlap, seed=0)
    # Real: pair0 shares "a" (1.0), pair1 shares "d" (1.0), pair2 shares
    # nothing (0.0) -> real_rate = 2/3 exactly, regardless of the shuffle.
    assert result["real_rate"] == pytest.approx(2 / 3)
    again = cs._shuffle_contrast(pairs, overlap, seed=0)
    assert again == result


def test_content_words_excludes_function_words_and_speech_verbs():
    words = cs._content_words('"She said the quiet river ran through the old valley."')
    assert "said" not in words
    assert "the" not in words
    assert "quiet" in words
    assert "valley" in words


def test_politeness_hits_recognizes_a_representative_sentence():
    hits = cs._politeness_hits("Could you please pass the salt, thanks?")
    assert {"indirect_request", "please", "gratitude"} <= hits


def test_graph_stats_reciprocity_and_density_hand_computed():
    events = [
        {"speaker": "A", "text": "x"}, {"speaker": "B", "text": "y"},
        {"speaker": "B", "text": "z"}, {"speaker": "A", "text": "w"},
        {"speaker": "A", "text": "v"}, {"speaker": "C", "text": "u"},
    ]
    reply_pairs = cs._reply_pairs(events)
    # Edges: A->B, B->A, A->C (3 distinct directed edges over 3 nodes).
    result = cs._graph_stats(reply_pairs)
    assert result["nodes"] == 3
    assert result["edges"] == 3
    assert result["density"] == pytest.approx(3 / (3 * 2))
    # Only (A,B)/(B,A) is reciprocated: 2 of 3 edges.
    assert result["reciprocity"] == pytest.approx(2 / 3)


# ------------------------------------------------------- helper self-documentation

def test__dialogue_helper_matches_the_documented_attribution_shape():
    """Regression pin for the module docstring's explanation of why
    ``_dialogue()`` looks the way it does."""

    text = _dialogue([("Alice", "Hello there."), ("Bob", "How are you?"),
                      ("Alice", "I am fine.")])
    events, method = cs._ordered_events(_analysis(text))
    assert method == "tag"
    assert [event["speaker"] for event in events] == ["Alice", "Bob", "Alice"]
    assert [event["text"] for event in events] == ["Hello there.", "How are you?", "I am fine."]


# -------------------------------------------------------- off-by-default features

def test_dialogue_act_feature_runs_and_reports_a_transition_entropy_when_enabled():
    module, reason = optional.require("transformers")
    if module is None:
        pytest.skip(f"transformers not usable in this environment: {reason}")
    turns = [
        ("Alice", "What is the timeline for this?"), ("Bob", "The deadline is next Friday."),
        ("Alice", "Will you send the report?"), ("Bob", "I will handle it personally."),
        ("Alice", "Thank you for the update."), ("Bob", "You are welcome, take care."),
    ]
    findings = _findings(_dialogue(turns), config={
        "features": {**cs.DEFAULT_FEATURES, "dialogue_act": True},
        "dialogue_act_max_turns": 50})
    distribution = findings[PREFIX + "dialogue_act_distribution"]
    entropy = findings[PREFIX + "dialogue_act_transition_entropy"]
    assert distribution["value"] is not None
    assert set(distribution["distribution"]["labels_percent"]) <= {
        "commissive", "directive", "inform", "question"}
    assert entropy["value"] is not None
    assert 0.0 <= entropy["value"] <= 1.0 + 1e-9


def test_pos_convergence_feature_runs_when_spacy_is_available():
    module, reason = optional.require("spacy")
    if module is None:
        pytest.skip(f"spacy not usable in this environment: {reason}")
    turns = []
    for i in range(8):
        turns.append(("Alice", f"The quick brown fox jumps over the lazy dog number {i}."))
        turns.append(("Bob", f"A sudden loud noise startled the sleeping cat number {i}."))
    findings = _findings(_dialogue(turns), config={
        "features": {**cs.DEFAULT_FEATURES, "pos_convergence": True},
        "min_turns_per_speaker": 2, "min_coordination_pairs": 1})
    item = findings[PREFIX + "pos_pattern_convergence"]
    assert item["value"] is not None or item["warning"]


def test_scene_drift_reports_insufficient_with_too_few_windows(sample_text):
    findings = _findings(sample_text, config={"features": {**cs.DEFAULT_FEATURES,
                                                           "scene_drift": True},
                                              "scene_window_words": 1_000_000})
    item = findings[PREFIX + "scene_style_drift"]
    assert item["value"] is None
    assert "window" in item["warning"]


def test_a_coupling_correlation_needs_enough_pairs_and_carries_its_interval():
    # r over five pairs has a 95% interval about +-0.9 wide, so the default
    # minimum is 30 pairs and the interval travels with the value.
    assert REGISTRY["conversation_suite"].defaults["min_coupling_pairs"] == 30
    low, high = cs._pearson_interval(0.5, 5)
    assert low < -0.4 and high > 0.9
    low, high = cs._pearson_interval(0.5, 103)
    assert low == pytest.approx(0.3392, abs=1e-3)
    assert high == pytest.approx(0.6323, abs=1e-3)
    assert cs._pearson_interval(1.0, 50) is None
