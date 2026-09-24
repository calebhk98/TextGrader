"""Contract tests for the experimental richer syntactic-complexity suite.

Mirrors the style ``tests/test_coherence_suite.py`` and ``tests/test_logic_suite.py``
already use for a ``cost="parse"`` suite with an off-by-default, model-backed
feature: registry/config consistency, off-by-default wiring, no unwanted model
load under the default config, graceful degradation without spaCy or benepar,
a truncated-sample-is-insufficient-data check, and a handful of synthetic
separation tests the task spec calls for directly (simple vs. subordinate,
dense noun phrases vs. clause-heavy prose, a sentence-length control, and a
check that per-sentence medians are not dominated by one huge sentence).

Only ONE test in this file (``test_real_benepar_separates_flat_from_embedded``)
loads the real benepar model; every other constituency test monkeypatches
``_load_benepar``/``_resolve_constituency`` so this file stays fast and does
not load a second copy of the ~1.2GB pipeline.
"""

from __future__ import annotations

import time
from collections import Counter

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, syntax_complexity_suite as scs
from textgrader.results import Action, StatusType

PREFIX = "syntax.complexity_"

SPACY_READY = optional.have("spacy")
requires_spacy = pytest.mark.skipif(not SPACY_READY, reason="spacy/en_core_web_sm not available")

BENEPAR_READY = optional.have("benepar")
requires_benepar = pytest.mark.skipif(not BENEPAR_READY, reason="benepar not installed")

NLTK_READY = optional.have("nltk")
requires_nltk = pytest.mark.skipif(not NLTK_READY, reason="nltk not available")


def _benepar_model_ready(model: str = "benepar_en3") -> bool:
    try:
        import nltk
        nltk.data.find(f"models/{model}")
        return True
    except Exception:
        return False


requires_benepar_model = pytest.mark.skipif(
    not (BENEPAR_READY and _benepar_model_ready()),
    reason="benepar_en3 model not downloaded")


def _analysis(text: str, **kwargs) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book", **kwargs)


def _findings(text: str, config=None, profile=None) -> dict[str, dict]:
    analysis = _analysis(text)
    return {item["metric_id"]: item for item in scs.measure(analysis, config=config, profile=profile)}


def _ids(findings) -> set[str]:
    return {item["metric_id"] for item in findings}


def _all_off_config(**overrides) -> dict:
    features = {"tunit_clause": False, "phrasal_elaboration": False, "dependency_topology": False,
               "syntactic_surprisal": False, "constituency": False}
    features.update(overrides)
    return {"enabled": True, "features": features}


# ---------------------------------------------------------------- wiring

def test_registered_and_off_by_default():
    assert "syntax_complexity_suite" in REGISTRY
    spec = REGISTRY["syntax_complexity_suite"]
    assert spec.family == "syntax"
    assert spec.cost == "parse"
    assert "benepar" in spec.requires
    for name in ("features", "long_dependency_threshold", "constituency_model",
                "constituency_sample_sentences", "constituency_max_sentences",
                "constituency_max_seconds", "constituency_seed"):
        assert name in spec.defaults, name
    for name in ("tunit_clause", "phrasal_elaboration", "dependency_topology",
                "syntactic_surprisal", "constituency"):
        assert name in spec.defaults["features"], name
    assert spec.defaults["features"]["constituency"] is False
    assert spec.defaults["features"]["tunit_clause"] is True


def test_config_json_matches_registry_defaults():
    import json
    from pathlib import Path
    config = json.loads(Path("config.json").read_text(encoding="utf-8"))
    entry = config["metrics"]["syntax_complexity_suite"]
    assert entry["enabled"] is False
    cleaned = {key: value for key, value in entry.items()
              if key != "enabled" and not key.startswith("_")}
    assert cleaned == REGISTRY["syntax_complexity_suite"].defaults


def test_off_by_default(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(PREFIX)]


def test_enabling_the_suite_turns_on_every_default_feature_but_not_constituency(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "syntax_complexity_suite": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(manuscript, config).results}
    matched = {mid for mid in ids if mid.startswith(PREFIX)}
    assert "syntax.complexity_l2sca_clauses_per_tunit" in matched
    assert "syntax.complexity_noun_phrase_length" in matched
    assert "syntax.complexity_branching_factor" in matched
    assert not any("constituency" in mid for mid in matched)


def test_every_metric_id_uses_the_stable_prefix(sample_text):
    findings = scs.measure(_analysis(sample_text))
    ids = _ids(findings)
    assert ids
    assert all(mid.startswith(PREFIX) for mid in ids)


def test_constituency_is_off_by_default_even_with_the_suite_enabled(sample_text):
    ids = _ids(scs.measure(_analysis(sample_text)))
    assert not any("constituency" in mid for mid in ids)


def test_no_benepar_is_loaded_under_the_default_config(monkeypatch, manuscript, base_config):
    """The same critical gating rule coherence_suite's rst/coreference tests
    check: this suite's cost stays "parse", so nothing but features.constituency's
    own off-by-default value protects a book from a benepar/transformers load."""

    def _boom(model_name):
        raise AssertionError(f"benepar model {model_name!r} must not load by default")

    monkeypatch.setattr(scs, "_load_benepar", _boom)
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "syntax_complexity_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)  # default features: constituency is off
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
    assert not any("constituency" in item.metric_id for item in report.results)


def test_missing_benepar_degrades_one_feature_only(monkeypatch, sample_text):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "benepar")
    optional.reset_cache()
    try:
        findings = _findings(sample_text, config=_all_off_config(constituency=True))
        assert findings
        for metric_id, item in findings.items():
            assert item["value"] is None
            assert "benepar" in (item["warning"] or "").lower() or "unavailable" in (item["warning"] or "").lower()
    finally:
        optional.reset_cache()


def test_missing_spacy_degrades_the_whole_suite_without_crashing(monkeypatch, sample_text):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        findings = _findings(sample_text)
        assert findings
        assert all(item["value"] is None for item in findings.values())
        assert all(item["warning"] for item in findings.values())
    finally:
        optional.reset_cache()


def test_degenerate_documents_do_not_crash():
    for text in ("", "Hi.", "A\n\nB\n\nC"):
        findings = scs.measure(_analysis(text))
        assert findings  # always names every metric id, value or not


# --------------------------------------------------------- T-unit approximation

@requires_spacy
def test_main_clause_head_separates_coordinated_main_clauses_from_subordinate_ones():
    """Direct unit test of the T-unit approximation's central rule."""

    import spacy
    nlp = spacy.load("en_core_web_sm")

    coordinated = nlp("She left and he stayed.")
    predicates = [t for t in coordinated if scs._is_predicate_head(t)]
    assert len(predicates) == 2
    assert all(scs._is_main_clause_head(t) for t in predicates), \
        "both coordinated main-clause verbs should form their own T-unit"

    subordinated = nlp("She left because he stayed.")
    predicates = [t for t in subordinated if scs._is_predicate_head(t)]
    assert len(predicates) == 2
    main = [t for t in predicates if scs._is_main_clause_head(t)]
    dependent = [t for t in predicates if not scs._is_main_clause_head(t)]
    assert len(main) == 1 and len(dependent) == 1, \
        "a subordinate clause's predicate must not be counted as its own T-unit"


@requires_spacy
def test_finite_nonfinite_ratio_is_undefined_not_infinite_with_no_nonfinite_verb():
    findings = _findings("She left. He stayed. They laughed. It rained. ",
                         config=_all_off_config(tunit_clause=True))
    item = findings["syntax.complexity_finite_nonfinite_clause_ratio"]
    # All-finite text: ratio is either a large finite number or explicitly
    # undefined, but must never silently become float('inf').
    assert item["value"] != float("inf")


@requires_spacy
def test_l2sca_headline_is_pooled_not_the_per_sentence_median():
    """Most sentences here have zero dependent clauses; a few (repeated
    heavily) have several. The median of per-sentence ratios collapses to
    0, exactly the resolution problem the coordinator flagged from the
    real-book table; the pooled (sum of numerators / sum of denominators)
    ratio must not collapse the same way."""

    text = " ".join(SIMPLE_SENTENCES * 6) + " " + " ".join(EMBEDDED_SENTENCES * 3)
    findings = _findings(text)
    item = findings["syntax.complexity_l2sca_dependent_clauses_per_clause"]
    pooled = item["value"]
    median = item["distribution"]["per_sentence_shape"]["median"]
    assert item["distribution"]["aggregation"] == "pooled"
    assert median == pytest.approx(0.0, abs=1e-9)
    assert pooled > 0.05, "the pooled ratio must keep resolution the median throws away"
    assert pooled != pytest.approx(median)


@requires_spacy
def test_l2sca_pooled_headline_equals_hand_computed_sum_over_sum():
    text = _repeat_sentences(SIMPLE_SENTENCES + EMBEDDED_SENTENCES, 3)
    for metric_id in ("syntax.complexity_l2sca_complex_nominals_per_clause",
                      "syntax.complexity_l2sca_clauses_per_tunit",
                      "syntax.complexity_l2sca_mean_sentence_length"):
        item = _findings(text)[metric_id]
        numerator = item["distribution"]["numerator_total"]
        denominator = item["distribution"]["denominator_total"]
        assert denominator > 0
        assert item["value"] == pytest.approx(numerator / denominator), metric_id
        assert item["sample_size"] == int(denominator), metric_id


@requires_spacy
def test_syntax_complexity_profile_needs_parse_metrics_flag():
    """The suite's cost is 'parse', so textgrader.corpus._metric_names drops
    it from profiling -- and profile_vector never runs -- unless the build
    passes --parse-metrics, even with the suite enabled in metrics config.
    This is what makes the config note's "--parse-metrics AND the suite
    enabled" wording true rather than aspirational."""

    import tempfile
    from pathlib import Path
    from textgrader.corpus import build_profile

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "book.txt"
        source.write_text(_repeat_sentences(SIMPLE_SENTENCES, 10), encoding="utf-8")

        without_flag = build_profile(
            [source], metrics={"syntax_complexity_suite": {"enabled": True}})
        assert "syntax_complexity_suite" not in without_flag["feature_profiles"]

        with_flag = build_profile(
            [source], include_parse_metrics=True,
            metrics={"syntax_complexity_suite": {"enabled": True}})
        assert "syntax_complexity_suite" in with_flag["feature_profiles"]
        rows = with_flag["feature_profiles"]["syntax_complexity_suite"]
        assert rows and rows[0]


@requires_spacy
def test_syntactic_surprisal_unavailable_warning_names_parse_metrics_flag():
    findings = _findings("The cat sat on the mat quietly today.",
                         config=_all_off_config(syntactic_surprisal=True))
    for metric_id in ("syntax.complexity_pos_bigram_cross_entropy",
                      "syntax.complexity_dependency_label_bigram_cross_entropy",
                      "syntax.complexity_sentence_syntactic_surprisal"):
        assert "--parse-metrics" in findings[metric_id]["warning"]


# ------------------------------------------------------ synthetic separation

def _repeat_sentences(sentences: list[str], times: int) -> str:
    return " ".join(sentences * times)


SIMPLE_SENTENCES = [
    "The cat sat.", "It was warm.", "The dog ran.", "She was happy.",
    "The sun rose.", "He was tired.", "The bird sang.", "They went home.",
]

EMBEDDED_SENTENCES = [
    "The hypothesis that the committee, which had convened after the report was "
    "published, would reject the proposal was disproved.",
    "Although the weather was cold, the children who had been waiting eagerly for "
    "the bus decided to walk instead, since the driver, who was often late, had "
    "not yet arrived.",
    "The report that the analyst, who had reviewed every prior submission, "
    "eventually filed contradicted the claim that the board had initially accepted.",
    "Because the witness who testified first contradicted the account that the "
    "detective, who had studied the case for years, had already reconstructed, "
    "the jury that had seemed convinced grew uncertain.",
]


@requires_spacy
def test_simple_coordinated_sentences_score_lower_than_deeply_subordinate_ones():
    """Rule 4's required separation test, applied to this suite's headline
    T-unit/clause and dependency-topology channels."""

    simple = _findings(_repeat_sentences(SIMPLE_SENTENCES, 5))
    embedded = _findings(_repeat_sentences(EMBEDDED_SENTENCES, 8))

    simple_dc = simple["syntax.complexity_l2sca_dependent_clauses_per_clause"]["value"]
    embedded_dc = embedded["syntax.complexity_l2sca_dependent_clauses_per_clause"]["value"]
    assert simple_dc < embedded_dc
    assert simple_dc == pytest.approx(0.0, abs=1e-9)
    assert embedded_dc > 0.3

    simple_cn = simple["syntax.complexity_l2sca_complex_nominals_per_clause"]["value"]
    embedded_cn = embedded["syntax.complexity_l2sca_complex_nominals_per_clause"]["value"]
    assert simple_cn < embedded_cn

    simple_depth = simple["syntax.complexity_branching_factor"]["distribution"]["max"]
    embedded_depth = embedded["syntax.complexity_branching_factor"]["distribution"]["max"]
    assert simple_depth <= embedded_depth


@requires_spacy
def test_dense_noun_phrases_score_higher_phrasal_elaboration_than_clause_heavy_prose():
    dense_np = _repeat_sentences([
        "The very old wooden house near the quiet river with the broken red roof stood alone.",
        "A tall dark stranger in a long grey coat with a worn leather bag walked slowly by.",
        "The small brown dog with the torn ear and the loud bark chased the ball again.",
    ], 10)
    clause_heavy = _repeat_sentences([
        "She left because he asked her to, although she did not want to go at all.",
        "He ran because the dog barked, and then he stopped because he was tired.",
        "They stayed because it rained, but they left once the storm had passed by.",
    ], 10)

    dense_findings = _findings(dense_np)
    clause_findings = _findings(clause_heavy)
    dense_np_len = dense_findings["syntax.complexity_noun_phrase_length"]["value"]
    clause_np_len = clause_findings["syntax.complexity_noun_phrase_length"]["value"]
    assert dense_np_len > clause_np_len

    dense_dc = dense_findings["syntax.complexity_l2sca_dependent_clauses_per_clause"]["value"]
    clause_dc = clause_findings["syntax.complexity_l2sca_dependent_clauses_per_clause"]["value"]
    assert clause_dc > dense_dc


@requires_spacy
def test_sentence_length_alone_does_not_explain_the_complexity_signal():
    """Two texts with comparable sentence length but very different
    subordination must not score the same on a clause-structure measure --
    residual variation beyond words/sentence must exist (task spec's
    "Sentence-length control test")."""

    flat = _repeat_sentences([
        "The cat ran quickly and the dog barked loudly and the bird flew away.",
        "She laughed warmly and he smiled kindly and they walked home together.",
        "The wind blew softly and the leaves fell gently and the day ended well.",
    ], 12)
    embedded = _repeat_sentences([
        "Because the cat ran quickly, the dog that had been barking loudly finally stopped.",
        "Since she laughed warmly, the man who had been smiling kindly walked away slowly.",
        "Although the wind blew softly, the leaves that had been falling gently piled up high.",
    ], 12)

    flat_findings = _findings(flat)
    embedded_findings = _findings(embedded)

    flat_len = flat_findings["syntax.complexity_l2sca_mean_sentence_length"]["value"]
    embedded_len = embedded_findings["syntax.complexity_l2sca_mean_sentence_length"]["value"]
    assert flat_len == pytest.approx(embedded_len, rel=0.35), \
        "the two texts should have comparable sentence length"

    flat_dc = flat_findings["syntax.complexity_l2sca_dependent_clauses_per_clause"]["value"]
    embedded_dc = embedded_findings["syntax.complexity_l2sca_dependent_clauses_per_clause"]["value"]
    assert embedded_dc > flat_dc + 0.2, \
        "subordination must differ sharply even though sentence length does not"


@requires_spacy
def test_pooled_headline_and_per_sentence_median_serve_different_purposes():
    """The MLS headline is now the pooled, L2SCA-style mean (total words /
    total sentences), which -- correctly -- IS pulled up by one huge
    sentence, the same way a real long run-on genuinely raises a whole
    text's average sentence length. That pooled sensitivity is the point of
    fixing the aggregation; robustness to the outlier still lives in
    ``distribution["per_sentence_shape"]``, which is not thrown away, so a
    reader who wants the median-based view the segmentation-bug context
    called for still has it available beside the pooled headline."""

    short_sentences = ["The cat sat quietly by the fire tonight."] * 25
    huge_sentence = ("The cat sat, and the dog watched, and the bird sang, and the mouse hid, "
                     "and the fire crackled, and the wind blew, and the rain fell, and the "
                     "night grew long, and the house creaked, and the clock ticked, and the "
                     "candle flickered, and the shadows danced, and the silence deepened, and "
                     "the hours passed, and the morning came at last after everyone had fallen "
                     "fast asleep in their own quiet corners of the old, creaking house.")
    text = " ".join(short_sentences) + " " + huge_sentence

    findings = _findings(text)
    sentence_length = findings["syntax.complexity_l2sca_mean_sentence_length"]
    pooled = sentence_length["value"]
    shape_summary = sentence_length["distribution"]["per_sentence_shape"]
    assert sentence_length["distribution"]["aggregation"] == "pooled"
    assert shape_summary["median"] < 10, \
        "the per-sentence median should still reflect the 25 short sentences"
    assert pooled > shape_summary["median"], \
        "the pooled mean is correctly pulled up by the one huge sentence -- that is the point"
    assert pooled == pytest.approx(shape_summary["mean"]), \
        "MLS's denominator is sentence count, so its pooled ratio equals the per-sentence mean"
    assert shape_summary["max"] > shape_summary["median"] * 5, \
        "the outlier sentence is still visible in the per-sentence shape's max"

    # The T-unit split is the other half of the story: this suite's own
    # T-unit approximation correctly reads the huge run-on as many short
    # coordinated T-units rather than one huge one, which is why the pooled
    # mean T-unit length stays low for the outlier while T-units per
    # sentence spikes for it instead.
    tunit_length = findings["syntax.complexity_l2sca_mean_tunit_length"]
    tunits_per_sentence = findings["syntax.complexity_l2sca_tunits_per_sentence"]
    assert tunit_length["value"] < 12
    assert (tunits_per_sentence["distribution"]["per_sentence_shape"]["max"]
           > tunits_per_sentence["value"] * 3)


# ------------------------------------------------------------ syntactic surprisal

@requires_spacy
def test_profile_vector_is_none_without_spacy(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        assert scs.profile_vector(_analysis("Some plain sentence here.")) is None
    finally:
        optional.reset_cache()


@requires_spacy
def test_syntactic_surprisal_needs_a_corpus_profile():
    findings = _findings("The cat sat on the mat quietly today.",
                         config=_all_off_config(syntactic_surprisal=True))
    for metric_id in ("syntax.complexity_pos_bigram_cross_entropy",
                      "syntax.complexity_dependency_label_bigram_cross_entropy",
                      "syntax.complexity_sentence_syntactic_surprisal"):
        item = findings[metric_id]
        assert item["value"] is None
        assert "corpus" in item["warning"]


@requires_spacy
def test_syntactic_surprisal_reads_the_pooled_profile_never_the_scored_document():
    """Structural proof of "never trains on the document being scored":
    profile_vector must not be called from inside measure()."""

    text_a = _repeat_sentences(SIMPLE_SENTENCES, 6)
    text_b = _repeat_sentences(EMBEDDED_SENTENCES, 6)
    vector_a = scs.profile_vector(_analysis(text_a))
    vector_b = scs.profile_vector(_analysis(text_b))
    assert vector_a and vector_b
    profile = {"feature_profiles": {"syntax_complexity_suite": [vector_a, vector_b]}}

    def _boom(*_a, **_k):
        raise AssertionError("measure() must never call profile_vector on the scored document")

    import unittest.mock as mock
    with mock.patch.object(scs, "profile_vector", side_effect=_boom):
        findings = _findings(text_b, config=_all_off_config(syntactic_surprisal=True), profile=profile)

    item = findings["syntax.complexity_pos_bigram_cross_entropy"]
    assert item["value"] is not None
    assert item["sample_size"] > 0
    surprisal = findings["syntax.complexity_sentence_syntactic_surprisal"]
    assert surprisal["value"] is not None
    assert len(surprisal["evidence"]) <= 10


# ------------------------------------------------------------------ constituency

def _fake_tree_for(depth_words: int) -> str:
    inner = "(NN word)"
    for _ in range(depth_words):
        inner = f"(NP (DT the) {inner})"
    return f"(S {inner} (VP (VBD happened)) (. .))"


def test_constituency_time_cap_excludes_parser_load_time(monkeypatch):
    """Mirrors coherence_suite's identical RST test: loading is a one-time
    process cost and must never itself trip the wall-clock cap."""

    class _FakeNlp:
        def __call__(self, text):
            class _Sent:
                _ = type("Ext", (), {"parse_string": _fake_tree_for(2)})()
            class _Doc:
                sents = [_Sent()]
            return _Doc()

    def _slow_load(model_name):
        time.sleep(1.5)
        return _FakeNlp(), None

    monkeypatch.setattr(scs, "_load_benepar", _slow_load)
    fake_analysis = type("FakeAnalysis", (), {"sentences": [f"Sentence {i}." for i in range(20)]})()
    trees, settings, _note = scs._resolve_constituency(
        fake_analysis, "fake-model", num_sentences=4, max_sentences_cap=10,
        max_seconds=1.0, seed=0)
    assert settings["sentences_parsed"] == 4
    assert settings["stopped_early_on_time_cap"] is False
    assert settings["elapsed_seconds"] < 1.0


def _constituency_only_config():
    return _all_off_config(constituency=True)


@requires_nltk
def test_constituency_truncated_sample_reports_insufficient_data(monkeypatch, manuscript, base_config):
    trees = [_fake_tree_for(1), _fake_tree_for(3)]
    settings = {"backend": "benepar", "model": "fake", "total_sentences_in_document": 500,
               "passages_requested": 8, "passage_sentences_target": 1, "max_sentences_cap": 60,
               "seed": 0, "passages_sampled": 8, "total_sentences_sampled": 8,
               "sentences_parsed": 2, "elapsed_seconds": 1.0, "max_seconds_cap": 180.0,
               "stopped_early_on_time_cap": True}
    note = "backend=benepar model=fake: parsed 2 of 8 sampled sentence(s) (stopped early: time cap reached)"
    monkeypatch.setattr(scs, "_resolve_constituency", lambda *a, **k: (trees, settings, note))

    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "syntax_complexity_suite": _constituency_only_config()}}
    report = grade.analyze(manuscript, config)
    items = {item.metric_id: item for item in report.results if "constituency" in item.metric_id}
    assert len(items) == 6
    for metric_id, item in items.items():
        assert item.sample_size == 2, (metric_id, item.sample_size)
        assert item.action is Action.INSUFFICIENT_DATA, (metric_id, item.action, item.warning)


@requires_nltk
def test_constituency_complete_sample_is_not_insufficient_data(monkeypatch, manuscript, base_config):
    trees = [_fake_tree_for(1), _fake_tree_for(3)]
    settings = {"backend": "benepar", "model": "fake", "total_sentences_in_document": 500,
               "passages_requested": 2, "passage_sentences_target": 1, "max_sentences_cap": 60,
               "seed": 0, "passages_sampled": 2, "total_sentences_sampled": 2,
               "sentences_parsed": 2, "elapsed_seconds": 1.0, "max_seconds_cap": 180.0,
               "stopped_early_on_time_cap": False}
    note = "backend=benepar model=fake: parsed 2 of 2 sampled sentence(s)"
    monkeypatch.setattr(scs, "_resolve_constituency", lambda *a, **k: (trees, settings, note))

    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "syntax_complexity_suite": _constituency_only_config()}}
    report = grade.analyze(manuscript, config)
    items = {item.metric_id: item for item in report.results if "constituency" in item.metric_id}
    assert len(items) == 6
    for metric_id, item in items.items():
        assert item.action is not Action.INSUFFICIENT_DATA, (metric_id, item.action, item.warning)


def test_constituency_degrades_cleanly_without_benepar(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "benepar")
    optional.reset_cache()
    try:
        findings = _findings("The cat sat on the mat. " * 40, config=_constituency_only_config())
        assert findings
        for item in findings.values():
            assert item["value"] is None
    finally:
        optional.reset_cache()


@requires_benepar_model
def test_real_benepar_separates_flat_from_embedded():
    """The one real, model-loading test in this file: a short, bounded
    sample on each side, so the real ~1.5s/sentence cost stays small."""

    config = {**_constituency_only_config(),
             "constituency_sample_sentences": 4, "constituency_max_sentences": 6,
             "constituency_max_seconds": 60.0}
    simple = _findings(_repeat_sentences(SIMPLE_SENTENCES, 1), config=config)
    embedded = _findings(_repeat_sentences(EMBEDDED_SENTENCES, 1), config=config)

    simple_depth = simple["syntax.complexity_constituency_tree_depth"]
    embedded_depth = embedded["syntax.complexity_constituency_tree_depth"]
    assert simple_depth["value"] is not None and embedded_depth["value"] is not None
    assert simple_depth["value"] < embedded_depth["value"]
    assert simple_depth["distribution"]["backend"] == "benepar"
    assert simple_depth["distribution"]["model"] == "benepar_en3"


# --------------------------------------------------------------- overlap disclosure

@requires_spacy
def test_overlapping_measures_reference_the_existing_metric_id():
    findings = _findings(_repeat_sentences(SIMPLE_SENTENCES + EMBEDDED_SENTENCES, 4))
    mean_len = findings["syntax.complexity_l2sca_mean_sentence_length"]
    assert "style.sentence_words_p50" in mean_len["distribution"]["overlaps_existing_metric_id"]
    dep_clause = findings["syntax.complexity_l2sca_dependent_clauses_per_clause"]
    assert "syntax.subordination_rate" in dep_clause["distribution"]["overlaps_existing_metric_id"]
    long_dep = findings["syntax.complexity_long_dependency_rate"]
    assert "syntax.dependency_distance_mean" in long_dep["distribution"]["overlaps_existing_metric_id"]
    finite_ratio = findings["syntax.complexity_finite_nonfinite_clause_ratio"]
    assert "syntax.finite_clauses_per_sentence" in finite_ratio["distribution"]["overlaps_existing_metric_id"]


@requires_spacy
def test_l2sca_findings_disclose_the_approximation():
    findings = _findings(_repeat_sentences(SIMPLE_SENTENCES, 5))
    for metric_id in ("syntax.complexity_l2sca_mean_sentence_length",
                      "syntax.complexity_l2sca_clauses_per_tunit"):
        assert "approximation" in (findings[metric_id]["warning"] or "").lower()
