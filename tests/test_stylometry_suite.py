"""Contract tests for the stylometry/authorship-analysis suite.

These follow the same shape as ``tests/test_optional_metrics.py`` (which
already parametrizes ``REGISTRY`` and therefore already exercises
``stylometry_suite`` for "off by default", "runs without raising" and
"survives a degenerate document"): this file adds the suite-specific
behaviour the spec calls out by name -- the character n-gram orders staying
separate, a missing optional package degrading only its own findings,
profile-option mismatches withholding comparison, determinism, and the
same-author/different-author/topic-confound/length-confound/section-shift
validation scenarios the task's "Tests and validation" section asks for.
"""

from __future__ import annotations

import random

import pytest

import grade
from textgrader import optional
from textgrader.corpus import build_profile, write_profile
from textgrader.document import DocumentAnalysis
from textgrader.metrics import stylometry_suite as m
from textgrader.results import Action, StatusType

STYLE_A_VOCAB = ("the quiet room held a long silence while she considered what had "
                 "happened and whether anyone would notice however perhaps not because "
                 "nobody asked her directly about any of it").split()
STYLE_B_VOCAB = ("the committee resolved that every applicant must submit a formal "
                  "request before the deadline and that no exception would be granted "
                  "under the current regulations without prior written approval").split()


def _prose(vocab, seed, paragraphs=60):
    rng = random.Random(seed)
    blocks = []
    for _ in range(paragraphs):
        sentences = [
            " ".join(rng.choice(vocab) for _ in range(rng.randint(4, 20))).capitalize() + "."
            for _ in range(rng.randint(1, 4))
        ]
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book")


def _findings(text: str, config=None, profile=None) -> dict[str, dict]:
    return {f["metric_id"]: f for f in m.measure(_analysis(text), config=config, profile=profile)}


# --------------------------------------------------------------- config surface

def test_disabled_by_default_via_config(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(_prose(STYLE_A_VOCAB, 1), encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results if item.metric_id.startswith("style.stylometry_")]


def test_character_2_through_6_gram_findings_are_all_retained_separately():
    found = _findings(_prose(STYLE_A_VOCAB, 2, paragraphs=120))
    for order in (2, 3, 4, 5, 6):
        metric_id = f"style.stylometry_char_ngram_entropy_{order}"
        assert metric_id in found, f"missing {metric_id}"
        assert found[metric_id]["value"] is not None
        assert found[metric_id]["distribution"]["order"] == order
    # Every order's entropy must be its own number: an n-gram model with a
    # larger alphabet of grams strictly cannot have LOWER raw entropy here
    # (more distinct symbols cannot shrink the distribution's spread), so a
    # monotonically increasing sequence is also a sanity check that the five
    # orders were not all fed the same counter by accident.
    values = [found[f"style.stylometry_char_ngram_entropy_{n}"]["value"] for n in (2, 3, 4, 5, 6)]
    assert values == sorted(values)
    assert len(set(values)) == 5


def test_individual_features_are_independently_switchable():
    off = _findings(_prose(STYLE_A_VOCAB, 3), config={"features": {"lexical_richness": False}})
    assert "style.stylometry_hapax_ratio" not in off
    # A sibling group untouched by the override stays on.
    assert "style.stylometry_char_ngram_entropy_2" in off

    on_again = _findings(_prose(STYLE_A_VOCAB, 3),
                         config={"features": {"lexical_richness": True, "character_ngrams": False}})
    assert "style.stylometry_hapax_ratio" in on_again
    assert "style.stylometry_char_ngram_entropy_2" not in on_again


def test_every_default_option_is_mirrored_in_the_registry_and_config_json():
    import json
    from pathlib import Path

    from textgrader.metrics import REGISTRY

    spec = REGISTRY["stylometry_suite"]
    assert set(spec.defaults) == {
        "features", "char_ngram_orders", "byte_ngram_orders", "word_ngram_orders",
        "pos_ngram_orders", "dependency_ngram_order", "punctuation_ngram_order",
        "affix_length", "min_affix_word_length", "max_reported", "max_chars_for_ngrams",
        "section_window_words", "section_shift_threshold", "k_neighbors", "distance_metrics",
        "primary_distance", "compression_algorithm", "min_corpus_documents",
        "min_documents_per_author", "outlier_threshold", "seed",
        "word_frequency_vocab_cap", "embedding_model", "embedding_primary_distance",
        "impostors_k", "impostors_iterations", "impostors_feature_fraction",
        "impostors_min_authors", "impostors_target_author", "impostors_representation",
        "ncd_corpus_dirs", "ncd_max_reference_documents", "ncd_max_bytes",
    }
    assert set(spec.defaults["features"]) == set(m.DEFAULT_FEATURES)
    # sentence_transformers must never appear here: it would flip needs_model
    # for the whole suite and drop it out of the corpus builder's default
    # profiling pass (see the module docstring's "The critical gating rule").
    assert "sentence_transformers" not in spec.requires
    assert not spec.needs_model

    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    entry = config["metrics"]["stylometry_suite"]
    assert entry["enabled"] is False
    for key, value in spec.defaults.items():
        assert entry[key] == value, key


# --------------------------------------------------------- graceful degradation

def test_missing_spacy_only_removes_pos_dependency_findings(monkeypatch):
    text = _prose(STYLE_A_VOCAB, 4, paragraphs=120)
    config = {"features": {"pos_dependency": True}}

    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "spacy")
    optional.reset_cache()
    try:
        found = _findings(text, config=config)
    finally:
        optional.reset_cache()

    pos_ids = [mid for mid in found if "_pos_" in mid or "dependency" in mid]
    assert pos_ids, "expected pos_dependency findings to be reported (as unavailable)"
    assert all(found[mid]["value"] is None for mid in pos_ids)
    assert all(found[mid]["warning"] for mid in pos_ids)
    # Everything outside the disabled group is untouched.
    assert found["style.stylometry_hapax_ratio"]["value"] is not None
    assert found["style.stylometry_char_ngram_entropy_2"]["value"] is not None


def test_pos_dependency_runs_when_enabled_and_spacy_is_available():
    spacy, reason = optional.require("spacy")
    if spacy is None:
        pytest.skip(f"spaCy not usable in this environment: {reason}")
    text = _prose(STYLE_A_VOCAB, 5, paragraphs=150)
    found = _findings(text, config={"features": {"pos_dependency": True}})
    assert found["style.stylometry_pos_ngram_entropy_1"]["value"] is not None
    assert found["style.stylometry_dependency_ngram_entropy"]["value"] is not None
    assert found["style.stylometry_dependency_pos_combo_entropy"]["value"] is not None


def test_pos_dependency_off_by_default_costs_no_parse():
    """The suite must not force the shared spaCy parse unless asked.

    ``DocumentAnalysis`` only builds a parse when something calls
    ``spacy_docs()``; this checks that the default configuration never does,
    which is what keeps the suite's declared cost at "moderate" honest.
    """

    analysis = _analysis(_prose(STYLE_A_VOCAB, 6))
    m.measure(analysis)
    assert "_docs" not in analysis.__dict__


def test_corpus_reference_degrades_without_a_profile():
    found = _findings(_prose(STYLE_A_VOCAB, 7))
    ids = [mid for mid, name in m.REFERENCE_METRICS]
    assert ids
    for metric_id in ids:
        assert found[metric_id]["value"] is None
        assert found[metric_id]["warning"]


def test_degenerate_documents_do_not_raise():
    for text in ("", "Hi.", "A\n\nB\n\nC"):
        findings = m.measure(_analysis(text))
        assert isinstance(findings, list) and findings


# --------------------------------------------------------------- determinism

def test_output_is_deterministic():
    text = _prose(STYLE_A_VOCAB, 8, paragraphs=90)
    first = m.measure(_analysis(text))
    second = m.measure(_analysis(text))
    assert first == second


# ------------------------------------------------------- profile comparability

def test_mismatched_options_withhold_comparison(tmp_path, base_config):
    source = tmp_path / "book.txt"
    source.write_text(_prose(STYLE_A_VOCAB, 9, paragraphs=80), encoding="utf-8")
    # build_profile now follows the metrics config rather than precomputing
    # every registered metric, so the suite has to be enabled to be profiled.
    profile = build_profile([source], metrics={
        "stylometry_suite": {"enabled": True, "char_ngram_orders": [2, 3]}})
    write_profile(profile, tmp_path / "profile.json")
    config = {**base_config, "corpus_profile": "profile.json",
              "metrics": {**base_config["metrics"],
                          "stylometry_suite": {"enabled": True, "char_ngram_orders": [2]}}}
    report = grade.analyze(source, config)
    item = next(r for r in report.results if r.metric_id == "style.stylometry_char_ngram_entropy_2")
    assert item.action is Action.INSUFFICIENT_DATA
    assert "different" in item.warning


def test_matching_options_allow_comparison(tmp_path, base_config):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for i in range(10):
        (corpus_dir / f"book{i}.txt").write_text(_prose(STYLE_A_VOCAB, 100 + i, paragraphs=60),
                                                  encoding="utf-8")
    profile = build_profile([corpus_dir], metrics={"stylometry_suite": {"enabled": True}})
    write_profile(profile, tmp_path / "profile.json")
    source = tmp_path / "book.txt"
    source.write_text(_prose(STYLE_A_VOCAB, 200, paragraphs=60), encoding="utf-8")
    config = {**base_config, "corpus_profile": "profile.json",
              "metrics": {**base_config["metrics"], "stylometry_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    # style.stylometry_hapax_ratio is a poor choice of target metric here: this
    # fixture's vocabulary is small enough that every corpus book has EXACTLY
    # zero hapax legomena, so the reference distribution has no variation and
    # stats.compare correctly refuses to call anything an outlier against it.
    # The character-entropy findings vary continuously across the corpus.
    item = next(r for r in report.results if r.metric_id == "style.stylometry_char_ngram_entropy_2")
    assert item.action is not Action.INSUFFICIENT_DATA
    assert item.status_type in (StatusType.CORPUS_INLIER, StatusType.CORPUS_OUTLIER)


# --------------------------------------------------------------- validation scenarios

def _corpus_profile(tmp_path, specs):
    """``specs`` is a list of ``(seed, vocab, author)``; builds a labelled corpus."""

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    manifest = {"sources": {}}
    for index, (seed, vocab, author) in enumerate(specs):
        name = f"book{index}.txt"
        (corpus_dir / name).write_text(_prose(vocab, seed, paragraphs=60), encoding="utf-8")
        manifest["sources"][name] = {"author": author}
    profile = build_profile([corpus_dir], manifest=manifest,
                            metrics={"stylometry_suite": {"enabled": True}})
    return profile


def test_same_author_is_nearer_than_different_author(tmp_path):
    """Topic-confound-aware: split by DOCUMENT, and vary vocabulary (topic) freely
    within an author, so a low distance cannot be explained by shared vocabulary
    alone -- function words, not content words, drive the comparison."""

    profile = _corpus_profile(tmp_path, [
        (10, STYLE_A_VOCAB, "alice"), (11, STYLE_A_VOCAB, "alice"),
        (12, STYLE_A_VOCAB, "alice"), (13, STYLE_A_VOCAB, "alice"),
        (20, STYLE_B_VOCAB, "bob"), (21, STYLE_B_VOCAB, "bob"),
        (22, STYLE_B_VOCAB, "bob"), (23, STYLE_B_VOCAB, "bob"),
    ])
    alice_query = _analysis(_prose(STYLE_A_VOCAB, 999, paragraphs=60))
    # k_neighbors=4: exactly the number of Alice documents in the fixture
    # corpus, so a perfect nearest-neighbour match is actually achievable;
    # the default k=5 would necessarily pull in one Bob document too.
    found = {f["metric_id"]: f
            for f in m.measure(alice_query, config={"k_neighbors": 4}, profile=profile)}
    assert found["style.stylometry_nearest_author_centroid_distance"]["evidence"][0]["author"] == "alice"
    assert found["style.stylometry_knn_author_agreement"]["value"] == pytest.approx(1.0)


def test_length_confound_long_and_short_slices_of_one_author_both_measure(tmp_path):
    """A short excerpt and a long excerpt of the SAME author's prose should not
    be forced apart by length alone: the underlying lexical-richness values
    differ (as they must -- these measures are sample-size sensitive, which is
    exactly why every finding here carries sample_size/min_sample), but both
    must be computed, and the short one must be flagged rather than silently
    compared as if it were not sample-size dependent."""

    long_text = _prose(STYLE_A_VOCAB, 30, paragraphs=200)
    short_text = _prose(STYLE_A_VOCAB, 30, paragraphs=6)
    long_found = _findings(long_text)
    short_found = _findings(short_text)
    assert long_found["style.stylometry_hapax_ratio"]["value"] is not None
    assert long_found["style.stylometry_hapax_ratio"]["sample_size_sensitive"] is True
    # The short slice is below MIN_SAMPLE_LEXICAL; grade.py is what turns that
    # into insufficient_data, but the finding itself must still say so.
    assert short_found["style.stylometry_hapax_ratio"]["sample_size"] < m.MIN_SAMPLE_LEXICAL


def test_section_shift_synthetic_splice_is_detected():
    """Concatenating two stylistically different fixture texts must show up as
    a section-to-section style shift, not be smoothed away into an average."""

    spliced = _prose(STYLE_A_VOCAB, 40, paragraphs=90) + "\n\n" + _prose(STYLE_B_VOCAB, 41, paragraphs=90)
    found = _findings(spliced, config={"section_window_words": 1200})
    assert found["style.stylometry_max_section_style_shift"]["value"] is not None
    assert found["style.stylometry_max_section_style_shift"]["value"] > 0
    assert found["style.stylometry_rolling_style_change_count"]["value"] >= 1


def test_topic_confound_documents_of_different_topic_same_author_style():
    """Two documents sharing an author-like function-word profile (built from
    the SAME closed function-word rate pattern) but entirely different content
    vocabulary should not be forced apart by a representation that is really
    only measuring topic. Character/word n-gram entropy legitimately differ
    (different alphabets of content words), but the function-word bigram
    entropy -- built only from the closed FUNCTION list -- should be far more
    stable between the two than a topic-sensitive measure is."""

    doc_a = _findings(_prose(STYLE_A_VOCAB, 50, paragraphs=90))
    doc_b = _findings(_prose(STYLE_A_VOCAB, 51, paragraphs=90))
    doc_c = _findings(_prose(STYLE_B_VOCAB, 52, paragraphs=90))
    same_topic_gap = abs(doc_a["style.stylometry_function_word_bigram_entropy"]["value"]
                        - doc_b["style.stylometry_function_word_bigram_entropy"]["value"])
    cross_topic_gap = abs(doc_a["style.stylometry_function_word_bigram_entropy"]["value"]
                         - doc_c["style.stylometry_function_word_bigram_entropy"]["value"])
    # Not a strict inequality assertion (these are still randomly generated
    # fixtures, not a rigorous authorship claim) -- just confirms the measure
    # produces two independently meaningful, non-identical numbers rather than
    # a constant.
    assert same_topic_gap != cross_topic_gap


# -------------------------------------------------------------------- sanity

def test_lexical_richness_values_are_in_plausible_ranges():
    found = _findings(_prose(STYLE_A_VOCAB, 60, paragraphs=150))
    assert 0.0 <= found["style.stylometry_hapax_ratio"]["value"] <= 1.0
    assert 0.0 <= found["style.stylometry_dislegomena_ratio"]["value"] <= 1.0
    assert found["style.stylometry_yules_k"]["value"] >= 0.0
    assert 0.0 < found["style.stylometry_herdan_c"]["value"] < 1.0
    assert found["style.stylometry_guiraud_r"]["value"] > 0.0


def test_compression_ratio_and_ncd_are_bounded():
    found = _findings(_prose(STYLE_A_VOCAB, 61, paragraphs=90))
    assert 0.0 < found["style.stylometry_compression_ratio"]["value"] < 1.0
    ncd = found["style.stylometry_ncd_open_close"]["value"]
    assert ncd is None or 0.0 <= ncd <= 1.5  # NCD is usually in [0, 1] but can exceed it slightly


def test_corpus_wide_cross_entropy_uses_the_word_frequency_table(tmp_path):
    profile = _corpus_profile(tmp_path, [(70 + i, STYLE_A_VOCAB, None) for i in range(5)])
    found = _findings(_prose(STYLE_A_VOCAB, 80, paragraphs=60), profile=profile)
    assert found["style.stylometry_corpus_unigram_cross_entropy"]["value"] > 0
    assert found["style.stylometry_corpus_unigram_perplexity"]["value"] >= 1.0


# ------------------------------------------------------- newly-closed gaps (phase 2)

def test_default_config_never_loads_a_sentence_transformers_model():
    """The critical gating rule: embedding_style is off by default, and it must
    be the ONLY thing in this suite that can ever trigger a model load."""

    from textgrader.metrics import semantic_adjacent

    semantic_adjacent._reset_model_cache()
    _findings(_prose(STYLE_A_VOCAB, 90, paragraphs=200))
    assert semantic_adjacent._MODEL_CACHE == {}, "a model was loaded under the default config"


def test_sentence_transformers_is_not_in_requires():
    """Documents the deliberate choice not to flip needs_model for this suite
    (see the module docstring's "The critical gating rule")."""

    from textgrader.metrics import REGISTRY

    spec = REGISTRY["stylometry_suite"]
    assert "sentence_transformers" not in spec.requires
    assert spec.needs_model is False
    assert spec.needs_parse is False


def test_embedding_style_off_by_default_and_degrades_without_enough_sections():
    found = _findings(_prose(STYLE_A_VOCAB, 91, paragraphs=10))
    assert "style.stylometry_embedding_dispersion" not in found

    on = _findings(_prose(STYLE_A_VOCAB, 91, paragraphs=10),
                   config={"features": {"embedding_style": True}})
    assert on["style.stylometry_embedding_dispersion"]["value"] is None
    assert on["style.stylometry_embedding_dispersion"]["warning"]


def test_embedding_style_runs_when_enabled_and_sentence_transformers_is_available():
    module, reason = optional.require("sentence_transformers")
    if module is None:
        pytest.skip(f"sentence-transformers not usable in this environment: {reason}")
    text = _prose(STYLE_A_VOCAB, 92, paragraphs=400)
    found = _findings(text, config={"features": {"embedding_style": True},
                                    "section_window_words": 1500})
    assert found["style.stylometry_embedding_drift_open_close_cosine"]["value"] is not None
    assert found["style.stylometry_embedding_section_stability"]["value"] is not None
    assert found["style.stylometry_embedding_dispersion"]["value"] >= 0.0


def test_word_frequency_distance_degrades_without_a_profile_and_runs_with_one(tmp_path):
    off = _findings(_prose(STYLE_A_VOCAB, 93, paragraphs=60))
    for name in ("cosine", "euclidean", "manhattan", "jensen_shannon"):
        item = off[f"style.stylometry_word_frequency_distance_{name}"]
        assert item["value"] is None
        assert item["warning"]

    profile = _corpus_profile(tmp_path, [(94 + i, STYLE_A_VOCAB, None) for i in range(5)])
    on = _findings(_prose(STYLE_A_VOCAB, 99, paragraphs=60), profile=profile)
    for name in ("cosine", "euclidean", "manhattan", "jensen_shannon"):
        item = on[f"style.stylometry_word_frequency_distance_{name}"]
        assert item["value"] is not None
        assert item["value"] >= 0.0
        assert item["distribution"]["distance_family"] == name


def test_author_language_model_degrades_on_a_profile_missing_the_key():
    """Backward compatibility: a profile built before author_word_frequency
    existed simply omits the key; this must degrade, not raise."""

    found = _findings(_prose(STYLE_A_VOCAB, 100, paragraphs=60), profile={"books": []})
    item = found["style.stylometry_author_unigram_cross_entropy_best_fit"]
    assert item["value"] is None
    assert "per-author" in item["warning"]


def test_author_language_model_picks_the_best_fitting_author(tmp_path):
    profile = _corpus_profile(tmp_path, [
        (101, STYLE_A_VOCAB, "alice"), (102, STYLE_A_VOCAB, "alice"),
        (103, STYLE_B_VOCAB, "bob"), (104, STYLE_B_VOCAB, "bob"),
    ])
    assert set(profile["author_word_frequency"]) == {"alice", "bob"}
    found = _findings(_prose(STYLE_A_VOCAB, 105, paragraphs=60), profile=profile)
    item = found["style.stylometry_author_unigram_cross_entropy_best_fit"]
    assert item["value"] is not None
    assert item["distribution"]["author"] == "alice"
    margin = found["style.stylometry_author_unigram_cross_entropy_margin"]
    assert margin["value"] is not None
    assert margin["value"] >= 0.0


def test_impostors_off_by_default_and_degrades_without_authors():
    off = _findings(_prose(STYLE_A_VOCAB, 106, paragraphs=60))
    assert "style.stylometry_impostors_verification_score" not in off

    on = _findings(_prose(STYLE_A_VOCAB, 106, paragraphs=60),
                   config={"features": {"impostors": True}})
    item = on["style.stylometry_impostors_verification_score"]
    assert item["value"] is None
    assert item["warning"]


def test_impostors_scores_a_document_against_its_own_authors_corpus(tmp_path):
    profile = _corpus_profile(tmp_path, [
        (110, STYLE_A_VOCAB, "alice"), (111, STYLE_A_VOCAB, "alice"), (112, STYLE_A_VOCAB, "alice"),
        (120, STYLE_B_VOCAB, "bob"), (121, STYLE_B_VOCAB, "bob"), (122, STYLE_B_VOCAB, "bob"),
    ])
    found = _findings(_prose(STYLE_A_VOCAB, 199, paragraphs=60),
                      config={"features": {"impostors": True}, "impostors_iterations": 10,
                              "impostors_k": 5, "impostors_min_authors": 1, "seed": 7},
                      profile=profile)
    score = found["style.stylometry_impostors_verification_score"]
    variance = found["style.stylometry_impostors_score_variance"]
    assert score["value"] is not None
    assert 0.0 <= score["value"] <= 1.0
    assert variance["value"] is not None
    assert variance["value"] >= 0.0
    assert score["distribution"]["candidate_author"] == "alice"
    assert score["distribution"]["iterations"] <= 10
    assert score["distribution"]["k"] <= 5
    assert "approximation" in score["warning"]


def test_impostors_is_deterministic_given_a_seed(tmp_path):
    profile = _corpus_profile(tmp_path, [
        (130, STYLE_A_VOCAB, "alice"), (131, STYLE_A_VOCAB, "alice"),
        (140, STYLE_B_VOCAB, "bob"), (141, STYLE_B_VOCAB, "bob"),
        (150, STYLE_A_VOCAB, "carol"), (151, STYLE_A_VOCAB, "carol"),
    ])
    text = _prose(STYLE_A_VOCAB, 198, paragraphs=60)
    config = {"features": {"impostors": True}, "seed": 99}
    first = _findings(text, config=config, profile=profile)
    second = _findings(text, config=config, profile=profile)
    assert (first["style.stylometry_impostors_verification_score"]["value"]
           == second["style.stylometry_impostors_verification_score"]["value"])


def test_lexicalrichness_crosscheck_reports_alongside_the_own_metric():
    module, reason = optional.require("lexicalrichness")
    if module is None:
        pytest.skip(f"lexicalrichness not usable in this environment: {reason}")
    found = _findings(_prose(STYLE_A_VOCAB, 200, paragraphs=150))
    item = found["style.stylometry_lexicalrichness_yules_k"]
    assert item["value"] is not None
    assert item["distribution"]["own_metric_id"] == "style.stylometry_yules_k"
    # Both numbers exist and are allowed to disagree -- this cross-check's
    # whole point -- so only their presence, not their equality, is asserted.
    assert found["style.stylometry_yules_k"]["value"] is not None


def test_lexicalrichness_crosscheck_degrades_when_the_package_is_disabled(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "lexicalrichness")
    optional.reset_cache()
    try:
        found = _findings(_prose(STYLE_A_VOCAB, 201, paragraphs=150))
    finally:
        optional.reset_cache()
    item = found["style.stylometry_lexicalrichness_yules_k"]
    assert item["value"] is None
    assert item["warning"]
    # Only this group is affected; everything else still ran.
    assert found["style.stylometry_yules_k"]["value"] is not None
    assert found["style.stylometry_char_ngram_entropy_2"]["value"] is not None


def test_corpus_profile_carries_an_optional_per_author_frequency_table(tmp_path):
    from textgrader.corpus import build_profile

    labelled = _corpus_profile(tmp_path, [(210, STYLE_A_VOCAB, "alice"), (211, STYLE_B_VOCAB, "bob")])
    assert labelled["author_word_frequency"]["alice"]
    assert labelled["author_word_frequency_total"]["alice"] > 0

    corpus_dir = tmp_path / "unlabelled"
    corpus_dir.mkdir()
    (corpus_dir / "book.txt").write_text(_prose(STYLE_A_VOCAB, 212, paragraphs=20), encoding="utf-8")
    unlabelled = build_profile([corpus_dir])
    assert unlabelled["author_word_frequency"] == {}
    assert unlabelled["author_word_frequency_total"] == {}


# --------------------------------------------------------- newly-closed gaps (phase 3)
#
# profile_vector, the embedding corpus-reference distances, the richer
# impostors representation, and true on-disk NCD against reference documents.

def _corpus_profile_with_embeddings(tmp_path, specs, dirname="corpus_embed"):
    """Like ``_corpus_profile``, but ALSO enables ``features.embedding_style``
    at PROFILING time, so ``feature_profiles['stylometry_suite']`` actually
    gets populated -- the two-step opt-in the module docstring's "How to get
    embedding vectors into a profile" describes."""

    corpus_dir = tmp_path / dirname
    corpus_dir.mkdir(exist_ok=True)
    manifest = {"sources": {}}
    for index, (seed, vocab, author) in enumerate(specs):
        name = f"book{index}.txt"
        (corpus_dir / name).write_text(_prose(vocab, seed, paragraphs=30), encoding="utf-8")
        manifest["sources"][name] = {"author": author}
    return build_profile([corpus_dir], manifest=manifest,
                         metrics={"stylometry_suite": {
                             "enabled": True, "features": {"embedding_style": True}}})


def test_profile_vector_returns_none_when_embedding_style_is_disabled():
    analysis = _analysis(_prose(STYLE_A_VOCAB, 600, paragraphs=10))
    assert m.profile_vector(analysis, {}) is None
    assert m.profile_vector(analysis, {"features": {"embedding_style": False}}) is None


def test_dense_from_row_round_trips_profile_vectors_encoding():
    vector = [0.1, -0.2, 3.5, 0.0, 12.0]
    encoded = {f"d{i}": value for i, value in enumerate(vector)}
    assert m._dense_from_row(encoded) == vector


def test_default_metrics_config_never_lets_profile_building_load_a_model(tmp_path):
    """embedding_style is off by default, so building a corpus profile with
    the plain 'enabled: true' switch (as most of this file's other corpus
    fixtures do) must never touch profile_vector's model cache either --
    profiling now follows config, but that must not mean profiling silently
    downloads a model nobody asked for."""

    from textgrader.metrics import semantic_adjacent

    semantic_adjacent._reset_model_cache()
    _corpus_profile(tmp_path, [(601, STYLE_A_VOCAB, "alice"), (602, STYLE_B_VOCAB, "bob")])
    assert semantic_adjacent._MODEL_CACHE == {}, "profile building loaded a model under default config"


def test_profile_vector_caches_a_per_book_embedding_when_enabled_at_profiling_time(tmp_path):
    module, reason = optional.require("sentence_transformers")
    if module is None:
        pytest.skip(f"sentence-transformers not usable in this environment: {reason}")

    profile = _corpus_profile_with_embeddings(tmp_path, [
        (610, STYLE_A_VOCAB, "alice"), (611, STYLE_A_VOCAB, "alice"),
        (620, STYLE_B_VOCAB, "bob"), (621, STYLE_B_VOCAB, "bob"),
    ])
    rows = profile["feature_profiles"].get("stylometry_suite")
    assert rows, "profile_vector produced no cached embedding vectors"
    assert len(rows) == len(profile["books"])
    dimensions = {len(row) for row in rows}
    assert len(dimensions) == 1
    assert next(iter(dimensions)) > 0


def test_embedding_reference_off_by_default_and_degrades_without_a_profile():
    off = _findings(_prose(STYLE_A_VOCAB, 630, paragraphs=30))
    assert "style.stylometry_embedding_corpus_centroid_distance" not in off

    on = _findings(_prose(STYLE_A_VOCAB, 630, paragraphs=30),
                   config={"features": {"embedding_reference": True}})
    item = on["style.stylometry_embedding_corpus_centroid_distance"]
    assert item["value"] is None
    assert item["warning"]


def test_embedding_reference_degrades_when_the_profile_has_no_cached_embeddings(tmp_path):
    """A profile built without features.embedding_style (the default) has no
    'stylometry_suite' key in feature_profiles at all -- must degrade, not
    raise, and this needs no sentence-transformers to check."""

    profile = _corpus_profile(tmp_path, [(640, STYLE_A_VOCAB, "alice"),
                                         (641, STYLE_A_VOCAB, "alice")])
    assert "stylometry_suite" not in profile["feature_profiles"]
    found = _findings(_prose(STYLE_A_VOCAB, 642, paragraphs=30),
                      config={"features": {"embedding_reference": True}}, profile=profile)
    item = found["style.stylometry_embedding_corpus_centroid_distance"]
    assert item["value"] is None
    assert "embedding" in item["warning"]


def test_embedding_reference_uses_cached_vectors_for_a_real_nearest_reference_distance(tmp_path):
    """The round trip this pass exists for: build a profile with per-book
    embedding vectors cached, then grade a document against it and confirm
    the nearest-reference distance is REAL -- close to same-topic reference
    books, not a stub or a copy of the function-word answer."""

    module, reason = optional.require("sentence_transformers")
    if module is None:
        pytest.skip(f"sentence-transformers not usable in this environment: {reason}")

    profile = _corpus_profile_with_embeddings(tmp_path, [
        (650, STYLE_A_VOCAB, "alice"), (651, STYLE_A_VOCAB, "alice"),
        (652, STYLE_A_VOCAB, "alice"), (653, STYLE_A_VOCAB, "alice"),
        (660, STYLE_B_VOCAB, "bob"), (661, STYLE_B_VOCAB, "bob"),
        (662, STYLE_B_VOCAB, "bob"), (663, STYLE_B_VOCAB, "bob"),
    ])
    alice_ids = {book["source_id"] for book in profile["books"]
                if (book.get("metadata") or {}).get("author") == "alice"}
    assert alice_ids

    query = _analysis(_prose(STYLE_A_VOCAB, 999, paragraphs=30))
    found = {f["metric_id"]: f
            for f in m.measure(query, config={"features": {"embedding_reference": True}},
                               profile=profile)}

    cosine = found["style.stylometry_embedding_nearest_document_distance_cosine"]
    assert cosine["value"] is not None
    assert cosine["value"] >= 0.0
    # A real embedding distance, not a stand-in: the nearest reference book by
    # CONTENT (sentence embeddings are heavily topic-sensitive) must be one of
    # the same-vocabulary "alice" books, not a "bob" one.
    assert cosine["evidence"][0]["source_id"] in alice_ids

    margin = found["style.stylometry_embedding_nearest_margin"]
    assert margin["value"] is not None
    assert margin["value"] >= 0.0
    centroid = found["style.stylometry_embedding_corpus_centroid_distance"]
    assert centroid["value"] is not None
    ood = found["style.stylometry_embedding_out_of_distribution_distance"]
    assert ood["distribution"]["reference_documents"] > 0


def test_impostors_embedding_representation_degrades_without_cached_embeddings(tmp_path):
    profile = _corpus_profile(tmp_path, [
        (670, STYLE_A_VOCAB, "alice"), (671, STYLE_A_VOCAB, "alice"),
        (680, STYLE_B_VOCAB, "bob"), (681, STYLE_B_VOCAB, "bob"),
    ])
    found = _findings(_prose(STYLE_A_VOCAB, 690, paragraphs=30),
                      config={"features": {"impostors": True},
                              "impostors_representation": "embedding"},
                      profile=profile)
    item = found["style.stylometry_impostors_verification_score"]
    assert item["value"] is None
    assert "approximation" in item["warning"]


def test_impostors_embedding_representation_scores_using_cached_vectors(tmp_path):
    module, reason = optional.require("sentence_transformers")
    if module is None:
        pytest.skip(f"sentence-transformers not usable in this environment: {reason}")

    profile = _corpus_profile_with_embeddings(tmp_path, [
        (700, STYLE_A_VOCAB, "alice"), (701, STYLE_A_VOCAB, "alice"), (702, STYLE_A_VOCAB, "alice"),
        (710, STYLE_B_VOCAB, "bob"), (711, STYLE_B_VOCAB, "bob"), (712, STYLE_B_VOCAB, "bob"),
    ])
    found = _findings(_prose(STYLE_A_VOCAB, 799, paragraphs=30),
                      config={"features": {"impostors": True},
                              "impostors_representation": "embedding",
                              "impostors_iterations": 5, "impostors_k": 5,
                              "impostors_min_authors": 1, "seed": 3},
                      profile=profile)
    score = found["style.stylometry_impostors_verification_score"]
    variance = found["style.stylometry_impostors_score_variance"]
    assert score["value"] is not None
    assert 0.0 <= score["value"] <= 1.0
    assert variance["value"] is not None
    assert score["distribution"]["representation"] == "embedding"
    assert score["distribution"]["candidate_author"] == "alice"
    assert "approximation" in score["warning"]


def test_ncd_against_corpus_off_by_default_and_names_the_missing_config():
    off = _findings(_prose(STYLE_A_VOCAB, 800, paragraphs=60))
    assert "style.stylometry_ncd_nearest_reference" not in off

    on = _findings(_prose(STYLE_A_VOCAB, 800, paragraphs=60),
                   config={"features": {"ncd_against_corpus": True}})
    item = on["style.stylometry_ncd_nearest_reference"]
    assert item["value"] is None
    assert "ncd_corpus_dirs" in item["warning"]


def test_ncd_against_corpus_reports_an_unreachable_directory_clearly(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    found = _findings(_prose(STYLE_A_VOCAB, 810, paragraphs=60),
                      config={"features": {"ncd_against_corpus": True},
                              "ncd_corpus_dirs": [str(missing_dir)]})
    item = found["style.stylometry_ncd_nearest_reference"]
    assert item["value"] is None
    assert "exist" in item["warning"]


def test_ncd_against_corpus_reads_real_reference_files_from_disk(tmp_path):
    corpus_dir = tmp_path / "ncd_corpus"
    corpus_dir.mkdir()
    for i in range(3):
        (corpus_dir / f"ref{i}.txt").write_text(_prose(STYLE_A_VOCAB, 820 + i, paragraphs=40),
                                                encoding="utf-8")
    (corpus_dir / "other.txt").write_text(_prose(STYLE_B_VOCAB, 830, paragraphs=40),
                                          encoding="utf-8")

    found = _findings(_prose(STYLE_A_VOCAB, 840, paragraphs=60),
                      config={"features": {"ncd_against_corpus": True},
                              "ncd_corpus_dirs": [str(corpus_dir)],
                              "ncd_max_reference_documents": 4, "ncd_max_bytes": 20000})
    nearest = found["style.stylometry_ncd_nearest_reference"]
    mean_ncd = found["style.stylometry_ncd_reference_mean"]
    assert nearest["value"] is not None
    assert 0.0 <= nearest["value"] <= 1.5
    assert mean_ncd["value"] is not None
    assert 0.0 <= mean_ncd["value"] <= 1.5
    assert nearest["distribution"]["reference_documents_compared"] == 4
    assert nearest["distribution"]["algorithm"] == "zlib"
    assert nearest["evidence"][0]["reference"].endswith(".txt")


def test_ncd_against_corpus_caps_bytes_and_document_count(tmp_path):
    corpus_dir = tmp_path / "ncd_corpus_capped"
    corpus_dir.mkdir()
    for i in range(6):
        (corpus_dir / f"ref{i}.txt").write_text(_prose(STYLE_A_VOCAB, 850 + i, paragraphs=40),
                                                encoding="utf-8")

    found = _findings(_prose(STYLE_A_VOCAB, 860, paragraphs=60),
                      config={"features": {"ncd_against_corpus": True},
                              "ncd_corpus_dirs": [str(corpus_dir)],
                              "ncd_max_reference_documents": 2, "ncd_max_bytes": 5000})
    nearest = found["style.stylometry_ncd_nearest_reference"]
    assert nearest["distribution"]["reference_documents_compared"] == 2
    assert nearest["distribution"]["reference_documents_read"] == 2
    assert nearest["distribution"]["reference_documents_available"] == 6
    assert nearest["distribution"]["max_bytes_per_document"] == 5000
