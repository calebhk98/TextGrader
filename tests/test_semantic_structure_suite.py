"""semantic_structure_suite: multiple non-generative semantic-structure and
topic-model channels, off by default, independently switchable, honest about
scale (BM25 unbounded vs. cosine), and never fitting a corpus-trained topic
model on the graded text itself.
"""

from __future__ import annotations

import random

import pytest

import grade
from textgrader import optional
from textgrader.corpus import build_profile
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, semantic_structure_suite as sss

PREFIX = "semantic.structure_"

requires_bm25 = pytest.mark.skipif(not optional.have("rank_bm25"),
                                   reason="rank_bm25 not available")
requires_sklearn = pytest.mark.skipif(not optional.have("sklearn"),
                                      reason="sklearn not available")
requires_topic_models = pytest.mark.skipif(
    not (optional.have("sklearn") and optional.have("tomotopy")),
    reason="sklearn and/or tomotopy not available")


def _analysis(text, **kwargs):
    return DocumentAnalysis.from_text(text, comparison_unit="book", **kwargs)


def _ids(findings):
    return {item["metric_id"] for item in findings}


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


# --------------------------------------------------------------------- wiring

def test_registered_with_expected_shape():
    spec = REGISTRY["semantic_structure_suite"]
    assert spec.family == "semantic"
    assert spec.module == "semantic_structure_suite"
    # The gating rule: sentence_transformers must never be in requires, or
    # corpus profiling would skip this whole suite (including its cheap,
    # default-on lexical/BM25/LSA/topic-model channels) as needs_model.
    assert "sentence_transformers" not in spec.requires
    assert spec.needs_model is False
    # Likewise spacy must not be in requires: the static_embedding_spacy
    # feature runs its own independent pipeline and must not force
    # needs_parse for the whole suite.
    assert spec.needs_parse is False
    assert spec.defaults["features"]["sentence_transformer"] is False
    assert spec.defaults["features"]["static_embedding_glove"] is False
    assert spec.defaults["features"]["static_embedding_spacy"] is False


def test_off_by_default(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(PREFIX)]


def test_enabling_the_suite_turns_on_every_default_feature(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "semantic_structure_suite": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(manuscript, config).results}
    matched = {mid for mid in ids if mid.startswith(PREFIX)}
    assert f"{PREFIX}lexical_tfidf_adjacent_sentence" in matched
    assert f"{PREFIX}bm25_adjacent_sentence" in matched
    assert f"{PREFIX}lsa_adjacent_sentence" in matched
    assert f"{PREFIX}disagreement_rank_correlation" in matched
    # Off-by-default channels stay off even with the suite enabled.
    assert not any("static_embedding" in mid for mid in matched)
    assert not any(mid.startswith(f"{PREFIX}sentence_transformer_") for mid in matched)


def test_every_metric_id_uses_the_stable_prefix(sample_text):
    findings = sss.measure(_analysis(sample_text))
    ids = _ids(findings)
    assert ids
    assert all(mid.startswith(PREFIX) for mid in ids)


def test_features_are_independently_switchable(sample_text):
    analysis = _analysis(sample_text)
    lexical_only = sss.measure(analysis, config={
        "features": {"lexical_tfidf": True, "bm25": False, "lsa": False,
                    "topic_models": False, "disagreement": False}})
    ids = _ids(lexical_only)
    assert f"{PREFIX}lexical_tfidf_adjacent_sentence" in ids
    assert not any(mid.startswith(f"{PREFIX}bm25_") for mid in ids)
    assert not any(mid.startswith(f"{PREFIX}lsa_") for mid in ids)
    assert not any(mid.startswith(f"{PREFIX}topic_") for mid in ids)
    assert not any(mid.startswith(f"{PREFIX}disagreement_") for mid in ids)

    bm25_only = sss.measure(analysis, config={
        "features": {"lexical_tfidf": False, "bm25": True, "lsa": False,
                    "topic_models": False, "disagreement": False}})
    bm25_ids = _ids(bm25_only)
    assert bm25_ids == {f"{PREFIX}bm25_adjacent_sentence", f"{PREFIX}bm25_adjacent_paragraph",
                        f"{PREFIX}bm25_centroid_relatedness", f"{PREFIX}bm25_window_drift",
                        f"{PREFIX}bm25_global_dispersion",
                        f"{PREFIX}bm25_intro_conclusion_similarity"}


@pytest.mark.parametrize("name", ["semantic_structure_suite"])
def test_runs_without_raising_on_the_shared_manuscript(name, manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"], name: {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    from textgrader.results import StatusType
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_survives_a_degenerate_document(text, tmp_path, base_config):
    source = tmp_path / "tiny.txt"
    source.write_text(text, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "semantic_structure_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    from textgrader.results import StatusType
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


# ------------------------------------------------------- nothing heavy by default

def test_default_config_loads_no_heavy_backend(sample_text, monkeypatch):
    """A test proving nothing heavy loads under default config (rule 6):
    monkeypatch every model-/download-backed loader to raise, and confirm the
    default (features unset) run never calls any of them."""

    def _boom(*args, **kwargs):
        raise AssertionError("a heavy backend was loaded under default config")

    monkeypatch.setattr(sss, "_load_glove", _boom)
    monkeypatch.setattr(sss, "_load_spacy_vectors", _boom)
    monkeypatch.setattr(sss.sem, "embed_texts", _boom)
    monkeypatch.setattr(sss.sem, "get_sentence_vectors", _boom)
    monkeypatch.setattr(sss.sem, "get_paragraph_vectors", _boom)

    findings = sss.measure(_analysis(sample_text))  # default config: no explicit features
    assert findings  # the cheap, default-on channels still ran


def test_disabled_features_never_touch_their_backend(sample_text, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("a disabled representation's backend was invoked")

    monkeypatch.setattr(sss, "_load_glove", _boom)
    monkeypatch.setattr(sss, "_load_spacy_vectors", _boom)
    monkeypatch.setattr(sss.sem, "embed_texts", _boom)
    monkeypatch.setattr(sss.sem, "get_sentence_vectors", _boom)
    monkeypatch.setattr(sss.sem, "get_paragraph_vectors", _boom)

    findings = sss.measure(_analysis(sample_text), config={
        "features": {"lexical_tfidf": True, "bm25": True, "lsa": True,
                    "static_embedding_glove": False, "static_embedding_spacy": False,
                    "sentence_transformer": False, "topic_models": False,
                    "disagreement": True}})
    assert findings


# ------------------------------------------------------------------ separation

def test_topic_consistent_vs_abrupt_shift_lexical_tfidf():
    """A topic-consistent paragraph should show a higher adjacent similarity
    than the sentence pair straddling an abrupt topic shift."""

    consistent = ("The garden had roses and the garden had trellises. The garden trellises "
                 "held roses every single spring morning. Roses and garden trellises filled "
                 "every garden corner with color.")
    shifted = ("The garden had roses and the garden had trellises. Meanwhile the stock market "
              "collapsed as bond yields spiked overnight. Traders shouted over falling futures "
              "prices on the exchange floor.")

    consistent_values = sss._adjacent_raw(
        DocumentAnalysis.from_text(consistent).sentences, True,
        sss._build_lexical_tfidf(DocumentAnalysis.from_text(consistent).sentences,
                                 DocumentAnalysis.from_text(consistent), "sentence", {})[1])
    shifted_values = sss._adjacent_raw(
        DocumentAnalysis.from_text(shifted).sentences, True,
        sss._build_lexical_tfidf(DocumentAnalysis.from_text(shifted).sentences,
                                 DocumentAnalysis.from_text(shifted), "sentence", {})[1])

    # The shifted text's minimum adjacent similarity (the shift itself) must
    # be lower than every adjacent similarity in the fully consistent text.
    assert min(shifted_values) < min(consistent_values)
    assert min(shifted_values) == pytest.approx(0.0, abs=1e-9)


def test_lexical_paraphrase_tfidf_differs_more_than_embeddings(monkeypatch):
    """A lexical paraphrase (same meaning, different words) should make
    TF-IDF report near-zero similarity while a stand-in 'embedding' backend
    (patched to reflect real semantic closeness) reports high similarity."""

    text = ("The dog was extremely happy to see her. The canine was overjoyed at her arrival.")
    analysis = DocumentAnalysis.from_text(text)
    sentences = analysis.sentences
    assert len(sentences) == 2

    _ok, tfidf_sim, _group, _scale, _note = sss._build_lexical_tfidf(sentences, analysis, "sentence", {})
    tfidf_value = tfidf_sim(0, 1)

    # Stand in for a real sentence-transformer embedding backend without a
    # network/model dependency: two paraphrases of the same idea score high.
    def fake_embedding_similarity(i, j):
        return 0.82

    assert tfidf_value < 0.3
    assert fake_embedding_similarity(0, 1) - tfidf_value > 0.4


@requires_bm25
def test_repeated_terms_with_semantic_discontinuity_bm25_vs_disagreement():
    """Repeating a term across an otherwise topic-discontinuous transition can
    keep a lexical/BM25 score misleadingly non-zero; this is exactly the case
    representation disagreement exists to surface, not something any single
    representation is expected to catch alone."""

    text = ("The bank of the river was thick with reeds and mud. The central bank raised "
           "interest rates again this morning. Investors reacted with alarm to the news.")
    analysis = DocumentAnalysis.from_text(text)
    sentences = analysis.sentences
    assert len(sentences) == 3
    _ok, bm25_sim, _group, _scale, _note = sss._build_bm25(sentences, analysis, "sentence", {})
    # "bank" recurs across the discontinuous transition (0->1), so BM25 must
    # not report a zero score there even though the topic genuinely shifted.
    assert bm25_sim(0, 1) > 0.0


def test_short_text_reports_insufficient_data():
    analysis = _analysis("Only one sentence here.")
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": True, "bm25": False, "lsa": False,
                    "topic_models": False, "disagreement": False}})
    by_id = _by_id(findings)
    adjacent = by_id[f"{PREFIX}lexical_tfidf_adjacent_sentence"]
    assert adjacent["value"] is None
    assert adjacent["sample_size"] == 0 or adjacent["sample_size"] < adjacent["min_sample"]
    assert "at least two" in adjacent["warning"]


def test_model_unavailable_fallbacks_remain_visible_and_independent(sample_text, monkeypatch):
    from textgrader import optional

    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "rank_bm25")
    optional.reset_cache()
    try:
        findings = sss.measure(_analysis(sample_text), config={
            "features": {"lexical_tfidf": True, "bm25": True, "lsa": True,
                        "topic_models": False, "disagreement": False}})
    finally:
        monkeypatch.delenv("TEXTGRADER_DISABLE_OPTIONAL", raising=False)
        optional.reset_cache()
    by_id = _by_id(findings)
    bm25_finding = by_id[f"{PREFIX}bm25_adjacent_sentence"]
    assert bm25_finding["value"] is None
    assert "unavailable" in bm25_finding["warning"]
    # lexical_tfidf, an independent channel, is unaffected.
    assert by_id[f"{PREFIX}lexical_tfidf_adjacent_sentence"]["value"] is not None


# ---------------------------------------------------------------------- BM25 scale

@requires_bm25
def test_bm25_is_never_reported_as_cosine():
    text = "The cat sat. The cat slept. The dog barked loudly outside."
    analysis = DocumentAnalysis.from_text(text)
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": False, "bm25": True, "lsa": False,
                    "topic_models": False, "disagreement": False}})
    by_id = _by_id(findings)
    adjacent = by_id[f"{PREFIX}bm25_adjacent_sentence"]
    assert adjacent["unit"] == "bm25_score"
    assert adjacent["distribution"]["scale"] == "unbounded"
    assert "UNBOUNDED" in adjacent["warning"]


# --------------------------------------------------------------------- overlap

def test_overlap_with_existing_metrics_is_named():
    text = ("The old house stood on the hill. The old house had stood there for a century. "
           "Nobody had lived in it for years.")
    analysis = DocumentAnalysis.from_text(text)
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": True, "bm25": False, "lsa": False,
                    "topic_models": False, "disagreement": False}})
    by_id = _by_id(findings)
    adjacent = by_id[f"{PREFIX}lexical_tfidf_adjacent_sentence"]
    assert "overlaps_existing_metric_id" in adjacent["distribution"]
    assert "semantic.adjacent_sentence_similarity" in adjacent["distribution"][
        "overlaps_existing_metric_id"]

    centroid = by_id[f"{PREFIX}lexical_tfidf_centroid_relatedness"]
    assert "coherence_global_context_overlap" in centroid["distribution"][
        "overlaps_existing_metric_id"]


# ---------------------------------------------------------------- disagreement

@requires_bm25
@requires_sklearn
def test_disagreement_findings_compare_active_representations(sample_text):
    findings = sss.measure(_analysis(sample_text), config={
        "features": {"lexical_tfidf": True, "bm25": True, "lsa": True,
                    "topic_models": False, "disagreement": True}})
    by_id = _by_id(findings)
    correlation = by_id[f"{PREFIX}disagreement_rank_correlation"]
    assert correlation["sample_size"] >= 1
    assert set(correlation["distribution"]["representations"]) >= {"lexical_tfidf", "bm25", "lsa"}
    for key in correlation["distribution"]["pairwise"]:
        assert "__" in key

    single = by_id[f"{PREFIX}disagreement_single_representation_only"]
    consensus = by_id[f"{PREFIX}disagreement_consensus_low_coherence"]
    assert single["unit"] == "%"
    assert consensus["unit"] == "%"


def test_disagreement_needs_two_active_representations():
    analysis = _analysis("One. Two. Three sentences here in total for this tiny test.")
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": True, "bm25": False, "lsa": False,
                    "topic_models": False, "disagreement": True}})
    by_id = _by_id(findings)
    correlation = by_id[f"{PREFIX}disagreement_rank_correlation"]
    assert correlation["value"] is None
    assert "at least two" in correlation["warning"]


def test_consensus_low_coherence_stays_near_configured_quantile_under_zero_inflation():
    """A zero-inflated representation's own 10th-PERCENTILE VALUE is often
    0.0 itself (real lexical_tfidf/bm25 adjacent-sentence scores on real
    prose: confirmed 98.3-99.7% "consensus" on real books under the old
    absolute-threshold design, since "flag everything <= 0.0" flags nearly
    every zero transition). Rank-based flagging must instead flag close to
    the configured quantile share regardless of how zero-inflated the
    representation is.
    """

    zero_inflated_a = [0.0] * 180 + [float(i) for i in range(1, 21)]
    zero_inflated_b = [0.0] * 170 + [float(i) * 0.5 for i in range(1, 31)]
    varied_c = [float(i % 50) / 50.0 for i in range(200)]
    rep_values = {"repA": zero_inflated_a, "repB": zero_inflated_b, "repC": varied_c}
    analysis = _analysis("Placeholder text for this synthetic disagreement test. " * 3)

    findings = sss._disagreement_findings(rep_values, analysis,
                                          {"disagreement_low_tail_quantile": 0.10})
    by_id = _by_id(findings)
    consensus = by_id[f"{PREFIX}disagreement_consensus_low_coherence"]
    single = by_id[f"{PREFIX}disagreement_single_representation_only"]

    assert consensus["value"] is not None
    assert consensus["value"] < 40.0, (
        f"consensus share {consensus['value']} is not near the configured 10% quantile -- "
        f"zero-inflation is still dominating the flag")
    assert single["value"] is not None

    diagnostics = consensus["distribution"]["per_representation"]
    assert set(diagnostics) == {"repA", "repB", "repC"}
    for rep, diag in diagnostics.items():
        assert diag["flagged_share_percent"] == pytest.approx(10.0, abs=2.0), (rep, diag)
    # repA and repB are heavily zero-inflated: far more than the flag quota
    # (20 of 200) sits tied at the boundary value (0.0), so the selection at
    # that boundary is reported as an arbitrary (though deterministic) tie.
    assert diagnostics["repA"]["degenerate_tie"] is True
    assert diagnostics["repA"]["tied_at_boundary_value"] == 180
    assert diagnostics["repC"]["degenerate_tie"] is False


# ---------------------------------------------------- intro/conclusion (not degenerate)

_LSA_DIFFERENT_ENDS_TEXT = (
    "The dog ran through the forest chasing a rabbit under tall green trees. "
    "Birds sang in the morning light as the dog kept running past the old oak. "
    "The forest grew quiet as afternoon came and the dog rested by a stream. "
    "Stock markets crashed overnight as investors panicked over bond yields. "
    "The central bank held an emergency meeting to discuss the falling economy. "
    "Traders on the exchange floor shouted prices as futures kept falling fast."
)


@requires_sklearn
@requires_bm25
def test_intro_conclusion_similarity_is_not_degenerate():
    """A fresh, separate fit on just the opening/closing snippet degenerates:
    a 2-document TruncatedSVD has at most one meaningful dimension, forcing
    cosine to exactly 1.0 regardless of content, and Okapi BM25's IDF goes
    negative for any term in more than half of a 2-document corpus.
    Confirmed on three real books under the old design: lsa_intro_conclusion_
    similarity was exactly 1.0 and bm25_intro_conclusion_similarity was
    negative on every one. Scoring within the whole-document fit instead
    must not reproduce either failure."""

    analysis = DocumentAnalysis.from_text(_LSA_DIFFERENT_ENDS_TEXT)
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": False, "bm25": True, "lsa": True, "topic_models": False,
                    "disagreement": False},
        "intro_conclusion_sentences": 2})
    by_id = _by_id(findings)
    lsa = by_id[f"{PREFIX}lsa_intro_conclusion_similarity"]
    bm25 = by_id[f"{PREFIX}bm25_intro_conclusion_similarity"]
    assert lsa["value"] is not None
    assert lsa["value"] < 0.95, lsa["value"]
    assert bm25["value"] is not None
    assert bm25["value"] >= 0.0, bm25["value"]


@requires_sklearn
def test_intro_conclusion_similarity_ranks_identical_ends_above_unrelated_ends():
    identical_ends_text = (
        "The dog ran through the forest chasing a rabbit under tall green trees. "
        "Birds sang in the morning light as the dog kept running past the old oak. "
        "Something happened in between that is unrelated filler text here now. "
        "More unrelated filler text about nothing in particular occurs here too. "
        "The dog ran through the forest chasing a rabbit under tall green trees. "
        "Birds sang in the morning light as the dog kept running past the old oak."
    )
    config = {"features": {"lexical_tfidf": False, "bm25": False, "lsa": True,
                          "topic_models": False, "disagreement": False},
             "intro_conclusion_sentences": 2}
    identical_value = _by_id(sss.measure(DocumentAnalysis.from_text(identical_ends_text),
                                         config=config))[
        f"{PREFIX}lsa_intro_conclusion_similarity"]["value"]
    unrelated_value = _by_id(sss.measure(DocumentAnalysis.from_text(_LSA_DIFFERENT_ENDS_TEXT),
                                         config=config))[
        f"{PREFIX}lsa_intro_conclusion_similarity"]["value"]
    assert identical_value is not None and unrelated_value is not None
    assert identical_value > unrelated_value


# ------------------------------------------------------------- corpus topic models

def _synthetic_profile(seed: int = 0, n_books: int = 10):
    vocab_a = ["dog", "cat", "forest", "tree", "river", "mountain", "castle", "king", "queen",
              "sword"]
    vocab_b = ["market", "stock", "price", "bank", "economy", "finance", "invest", "trade",
              "money", "debt"]
    rng = random.Random(seed)
    books, rows = [], []
    for i in range(n_books):
        topic_words = vocab_a if i % 2 == 0 else vocab_b
        counts = {word: float(rng.randint(5, 40)) for word in topic_words}
        counts.update({"__seed__": 0.0, "__sample_paragraphs__": 20.0, "__vocab_size__": 3000.0,
                      "__schema__": 1.0})
        rows.append(counts)
        books.append({"source_id": f"book{i}", "source_filename": f"book{i}.txt",
                     "source_path": f"book{i}.txt"})
    return {"books": books, "feature_profiles": {"semantic_structure_suite": rows}}


_ALTERNATING_TOPIC_PARAGRAPHS = [
    "The forest near the castle was full of ancient trees. A river ran past the old stone "
    "walls.",
    "The king once lived among the mountain paths. Legends spoke of a magic sword.",
    "A dog and a cat wandered near the queen's own knights, guarding the castle gate.",
    "Stock markets crashed hard this week. Investors watched the price of every trade fall.",
    "The central bank raised interest rates again. Economy watchers feared a debt crisis.",
    "Finance ministers met to discuss the failing market. Money flowed out of every bank.",
    "Back in the forest, the dog chased a cat past an old river bend near the mountain.",
    "A new trade deal boosted the economy. Stock prices rose as investors bought debt bonds.",
]


def test_topic_models_never_fit_on_the_graded_text_leave_one_out():
    profile = _synthetic_profile()
    analysis_in_corpus = _analysis("The dog ran. The cat slept.\n\nThe forest grew tall.",
                                   source="book2.txt")
    fit = sss._fit_topic_models(profile, analysis_in_corpus, {"topic_n_topics": 2})
    assert fit["meta"]["corpus_rows_excluded_leave_one_out"] == 1
    assert fit["meta"]["corpus_books_used"] == 9

    analysis_not_in_corpus = _analysis("The dog ran. The cat slept.\n\nThe forest grew tall.",
                                       source="not_in_corpus.txt")
    fit2 = sss._fit_topic_models(profile, analysis_not_in_corpus, {"topic_n_topics": 2})
    assert fit2["meta"]["corpus_rows_excluded_leave_one_out"] == 0
    assert fit2["meta"]["corpus_books_used"] == 10


@requires_topic_models
def test_topic_models_detect_alternating_topic_structure():
    profile = _synthetic_profile()
    text = "\n\n".join(_ALTERNATING_TOPIC_PARAGRAPHS) + "\n"
    analysis = _analysis(text, source="graded.txt")
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": False, "bm25": False, "lsa": False,
                    "topic_models": True, "disagreement": False},
        "topic_n_topics": 2, "hdp_iterations": 100}, profile=profile)
    by_id = _by_id(findings)
    for model in ("lda", "nmf", "hdp"):
        switch = by_id[f"{PREFIX}topic_{model}_switch_rate"]
        assert switch["value"] is not None
        # Seven adjacent pairs, six of which switch topic (the alternating
        # a/b/a/b/... structure above) -- a high switch rate.
        assert switch["value"] >= 40.0
        confidence = by_id[f"{PREFIX}topic_{model}_dominant_confidence"]
        assert confidence["value"] > 0.5
        assert "overlaps_existing_metric_id" in confidence["distribution"]


def test_topic_models_report_reason_below_minimum_corpus_size():
    profile = _synthetic_profile(n_books=2)
    analysis = _analysis("\n\n".join(_ALTERNATING_TOPIC_PARAGRAPHS), source="graded.txt")
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": False, "bm25": False, "lsa": False,
                    "topic_models": True, "disagreement": False}}, profile=profile)
    by_id = _by_id(findings)
    entropy = by_id[f"{PREFIX}topic_lda_distribution_entropy"]
    assert entropy["value"] is None
    assert "at least" in entropy["warning"]
    assert entropy["distribution"]["corpus_books_used"] == 2


def test_topic_models_report_no_profile_available():
    analysis = _analysis("\n\n".join(_ALTERNATING_TOPIC_PARAGRAPHS), source="graded.txt")
    findings = sss.measure(analysis, config={
        "features": {"lexical_tfidf": False, "bm25": False, "lsa": False,
                    "topic_models": True, "disagreement": False}}, profile=None)
    by_id = _by_id(findings)
    for model in ("lda", "nmf", "hdp"):
        entropy = by_id[f"{PREFIX}topic_{model}_distribution_entropy"]
        assert entropy["value"] is None
        assert entropy["warning"]


# ------------------------------------------------------- profile_vector hook

def test_profile_vector_is_bounded():
    text = "\n\n".join(_ALTERNATING_TOPIC_PARAGRAPHS * 5) + "\n"
    analysis = DocumentAnalysis.from_text(text)
    vector = sss.profile_vector(analysis, {"topic_profile_max_terms": 50})
    assert vector is not None
    term_entries = {k: v for k, v in vector.items() if not k.startswith("__")}
    assert len(term_entries) <= 50
    assert vector["__seed__"] == 0.0
    assert "__sample_paragraphs__" in vector


def test_profile_vector_off_when_feature_disabled():
    analysis = DocumentAnalysis.from_text("\n\n".join(_ALTERNATING_TOPIC_PARAGRAPHS))
    vector = sss.profile_vector(analysis, {"features": {"topic_models": False}})
    assert vector is None


def test_profile_vector_via_real_build_profile(corpus_dir):
    """End-to-end: textgrader.corpus.build_profile calls this suite's
    profile_vector hook once per book when the suite is enabled, exactly the
    path a real corpus build exercises."""

    profile = build_profile([corpus_dir],
                           metrics={"semantic_structure_suite": {"enabled": True}})
    rows = profile["feature_profiles"].get("semantic_structure_suite")
    assert rows is not None
    assert len(rows) == len(profile["books"])
    assert any(row for row in rows)


# ---------------------------------------------------------------- aggregation docs

def test_window_drift_and_dispersion_document_their_aggregation(sample_text):
    findings = sss.measure(_analysis(sample_text), config={
        "features": {"lexical_tfidf": True, "bm25": False, "lsa": False,
                    "topic_models": False, "disagreement": False}})
    by_id = _by_id(findings)
    dispersion = by_id[f"{PREFIX}lexical_tfidf_global_dispersion"]
    assert "standard deviation" in dispersion["distribution"]["aggregation"].lower()
    drift = by_id[f"{PREFIX}lexical_tfidf_window_drift"]
    assert drift["distribution"]["window_size"] == 5
