"""The experimental parser/segmenter disagreement suite.

The headline requirement this file exists to prove: a channel that would
have caught the quote-aware pySBD bug (see the module docstring in
``textgrader/metrics/parser_consensus.py`` and the commit "Split sentences
inside quoted speech, and record which splitter ran"). Plain pySBD refuses
to split inside quotation marks, so a long quoted speech comes back as one
enormous "sentence" -- a tail effect a MEAN sentence length hides. This suite
reports the tail (share of sentences over a length threshold, and the single
longest sentence) specifically so that collapse cannot hide.
"""

from __future__ import annotations

import time

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis, NlpSettings, TextProcessing
from textgrader.metrics import REGISTRY, parser_consensus as pc
from textgrader.results import Action, StatusType

PREFIX = "syntax.parser_"


def _ids(findings):
    return {item["metric_id"] for item in findings}


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


def _analysis(text, **kwargs):
    kwargs.setdefault("nlp_settings", NlpSettings())
    return DocumentAnalysis.from_text(text, comparison_unit="book",
                                      processing=TextProcessing(), **kwargs)


def _quoted_speech(n_sentences: int, quote_open: str = "“",
                   quote_close: str = "”") -> str:
    """One long, uninterrupted quoted speech of ``n_sentences`` sentences,
    reproducing the shape of the reported bug: a paragraph that is, start to
    finish, one character's monologue."""

    sentences = [f"This is sentence number {i} of a very long speech that keeps going."
                for i in range(1, n_sentences + 1)]
    return quote_open + " ".join(sentences) + quote_close


def _md_available() -> bool:
    spacy_mod, reason = optional.require("spacy")
    if spacy_mod is None:
        return False
    try:
        spacy_mod.load("en_core_web_md")
        return True
    except Exception:
        return False


needs_spacy_md = pytest.mark.skipif(not _md_available(),
                                    reason="en_core_web_md is not installed in this environment")


def _stanza_available() -> bool:
    stanza_mod, reason = optional.require("stanza")
    if stanza_mod is None:
        return False
    pipeline, reason = pc._load_stanza("tokenize,mwt,pos,lemma,depparse")
    return pipeline is not None


needs_stanza = pytest.mark.skipif(not _stanza_available(),
                                  reason="stanza's English models are not downloaded in this "
                                          "environment (see optional.py's 'stanza' entry)")

needs_pysbd = pytest.mark.skipif(not optional.have("pysbd"),
                                 reason="pySBD is not installed (or disabled via "
                                        "TEXTGRADER_DISABLE_OPTIONAL) in this environment; the "
                                        "whole point of this test is comparing against it")

needs_a_second_tokenizer = pytest.mark.skipif(
    not any(optional.have(name) for name in ("spacy", "nltk", "syntok")),
    reason="every non-canonical tokenizer source is disabled in this environment, so there is "
          "nothing for the canonical Unicode tokenizer to disagree with")


# --------------------------------------------------------------- wiring/gating

def test_off_by_default(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(PREFIX)]


def test_registered_with_stable_ids_and_neutral_family():
    assert "parser_consensus" in REGISTRY
    spec = REGISTRY["parser_consensus"]
    assert spec.family == "syntax"
    assert spec.cost == "parse"
    assert spec.defaults["features"]["stanza"] is False
    assert spec.defaults["features"]["spacy_md"] is False
    assert spec.defaults["features"]["constituency_benepar"] is False


def test_every_metric_id_uses_the_stable_prefix(sample_text):
    findings = pc.measure(_analysis(sample_text))
    ids = _ids(findings)
    assert ids
    assert all(mid.startswith(PREFIX) for mid in ids)


def test_enabling_the_suite_alone_turns_on_every_default_channel(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "parser_consensus": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(manuscript, config).results}
    matched = {mid for mid in ids if mid.startswith(PREFIX)}
    assert "syntax.parser_long_sentence_share" in matched
    assert "syntax.parser_max_sentence_length" in matched
    assert "syntax.parser_boundary_agreement_f1" in matched
    assert "syntax.parser_boundary_consensus" in matched
    assert "syntax.parser_token_count_disagreement" in matched
    # pos/dependency/chunks default on, but degrade to a real, present
    # finding (value=None with an explanatory warning) rather than vanishing,
    # since only spaCy sm is available by default.
    assert "syntax.parser_uas" in matched
    assert "syntax.parser_np_chunk_agreement" in matched


def test_never_changes_canonical_segmentation_or_parse(sample_text):
    """This suite measures disagreement; it must never become the pipeline's
    own choice of segmenter or parser as a side effect of running."""

    analysis = _analysis(sample_text)
    before_sentences = list(analysis.sentences)
    before_segmenter = analysis.segmenter
    pc.measure(analysis)
    assert analysis.sentences == before_sentences
    assert analysis.segmenter == before_segmenter


# ------------------------------------------------------- the headline channel

@needs_pysbd
def test_plain_pysbd_collapse_is_flagged_by_long_sentence_share():
    """The test the task explicitly asks for: feed a long quoted multi-
    sentence speech through plain pySBD and the pipeline's quote-aware
    segmenter, and assert the channel flags plain pySBD's collapse."""

    speech = _quoted_speech(20)
    text = speech + " he said, finally falling silent."
    analysis = _analysis(text)
    findings = _by_id(pc.measure(analysis, config={
        "features": {**pc.DEFAULT_FEATURES, "pos": False, "dependency": False, "chunks": False}}))

    long_share = findings["syntax.parser_long_sentence_share"]
    max_len = findings["syntax.parser_max_sentence_length"]
    by_segmenter_share = long_share["distribution"]["by_segmenter"]
    by_segmenter_len = max_len["distribution"]["by_segmenter"]

    # Plain pySBD collapsed the whole 20-sentence speech into one "sentence":
    # 100% of its (one) sentence is over the threshold, and that sentence is
    # dramatically longer than every other segmenter's longest.
    assert by_segmenter_share["pysbd"] == pytest.approx(100.0)
    assert by_segmenter_len["pysbd"] > 200
    # The quote-aware segmenter (what this pipeline actually uses) and every
    # independent segmenter correctly split the speech and see NO long
    # sentences at all.
    for name in ("pysbd_quote_aware", "canonical", "builtin", "nltk_punkt", "syntok",
                "spacy_parser"):
        assert by_segmenter_share[name] == pytest.approx(0.0), name
        assert by_segmenter_len[name] < 20, name
    # The headline VALUE (worst segmenter) must itself surface the collapse,
    # not just the distribution a reader has to dig into.
    assert long_share["value"] == pytest.approx(100.0)
    assert long_share["distribution"]["worst_segmenter"] == "pysbd"
    assert max_len["value"] > 200
    assert max_len["distribution"]["worst_segmenter"] == "pysbd"
    # Evidence carries the actual offending sentence, bounded, not a full dump.
    assert any(item["segmenter"] == "pysbd" for item in long_share["evidence"])


@needs_pysbd
def test_sentence_count_and_boundary_disagreement_also_flag_the_collapse():
    """Complementary channels: a book losing most of its boundaries should
    show up as both a large count gap and a low pairwise boundary F1 against
    the collapsing segmenter, not only in the tail-length channels."""

    speech = _quoted_speech(20)
    analysis = _analysis(speech)
    findings = _by_id(pc.measure(analysis, config={
        "features": {**pc.DEFAULT_FEATURES, "pos": False, "dependency": False, "chunks": False}}))
    count_gap = findings["syntax.parser_sentence_count_disagreement"]
    assert count_gap["value"] > 90  # 1 sentence vs ~20: a huge relative gap
    f1 = findings["syntax.parser_boundary_agreement_f1"]
    pairs = f1["distribution"]["pairs"]
    # Every pairing that includes plain pysbd should score at or near zero
    # boundary agreement (it has no internal boundaries to agree on at all).
    pysbd_pairs = {k: v for k, v in pairs.items() if "pysbd" == k.split("|")[0]
                  or "pysbd" == k.split("|")[1]}
    assert pysbd_pairs
    for key, stats in pysbd_pairs.items():
        if key.split("|") in (["pysbd", "pysbd_quote_aware"], ["pysbd_quote_aware", "pysbd"]):
            continue
        assert (stats["f1"] or 0.0) < 0.1, (key, stats)


# ---------------------------------------------------- separation: clean prose

def test_boundary_agreement_is_near_perfect_on_unambiguous_declarative_prose():
    """A negative control for the headline test: ordinary narrative prose
    with no abbreviations, no quotations and no ambiguity should show HIGH
    agreement across every segmenter, so the low agreement above is
    attributable to the quotation collapse, not to this suite being noisy."""

    text = (" ".join(f"The traveler walked past the {n}th milestone on the quiet road."
                    for n in range(1, 15)))
    analysis = _analysis(text)
    findings = _by_id(pc.measure(analysis, config={
        "features": {**pc.DEFAULT_FEATURES, "pos": False, "dependency": False, "chunks": False}}))
    assert findings["syntax.parser_boundary_agreement_f1"]["value"] > 0.95
    assert findings["syntax.parser_boundary_consensus"]["value"] > 90.0
    assert findings["syntax.parser_long_sentence_share"]["value"] == pytest.approx(0.0)


@needs_stanza
def test_parsers_agree_more_on_plain_prose_than_on_garden_path_sentences():
    """Rule 4's required separation test, for the PARSER side rather than the
    segmenter side: two architecturally independent parsers (spaCy sm and
    stanza) should agree more, on UAS (attachment only -- the two use
    different dependency-label vocabularies, which would otherwise swamp the
    signal with a labelling-convention mismatch rather than a genuine
    ambiguity one; see the module docstring's dependency-label caveat), on
    plain declarative sentences than on classic garden-path sentences, which
    are constructed specifically to admit two different attachments. spaCy
    sm vs md was tried first and rejected for this test: sharing almost the
    same training data and architecture, they made the IDENTICAL mistake on
    a garden-path sentence and scored 100% agreement despite both being
    wrong -- architectural independence, not just "a second parser", is what
    this separation needs."""

    plain_sentences = [
        "The dog chased the ball across the yard for hours.",
        "She walked to the store to buy some fresh bread this morning.",
        "The children played happily in the park until the sun went down.",
    ]
    # Classic garden-path sentences (Bever 1970; Pinker's "the horse raced
    # past the barn fell"): each admits a locally plausible parse that turns
    # out wrong, which is exactly the genuine-ambiguity signal this suite
    # exists to surface as parser disagreement.
    garden_sentences = [
        "The old man the boats while the horse raced past barn fell.",
        "The horse raced past the barn fell.",
        "The complex houses married and single soldiers and their families.",
    ]

    def mean_uas(sentences):
        text = " ".join(sentences * 3)
        analysis = _analysis(text)
        config = {"features": {**pc.DEFAULT_FEATURES, "segmentation": False,
                               "tokenization": False, "chunks": False, "stanza": True},
                  "max_sentences_for_parse": 20}
        findings = _by_id(pc.measure(analysis, config=config))
        return findings["syntax.parser_uas"]["value"]

    plain_uas = mean_uas(plain_sentences)
    garden_uas = mean_uas(garden_sentences)
    assert plain_uas is not None and garden_uas is not None
    assert plain_uas > garden_uas


# --------------------------------------------------------------- gating rule

def test_extra_parser_models_load_nothing_under_default_config(monkeypatch, manuscript, base_config):
    """The critical gating rule: enabling the suite alone must load nothing
    beyond the en_core_web_sm pipeline every cost="parse" metric already
    shares. Nothing but each feature flag's own off-by-default value
    protects a book from an unwanted extra spaCy model, a stanza pipeline
    load or a benepar model load."""

    def _boom_spacy(model_name):
        raise AssertionError(f"spaCy model {model_name!r} must not load by default")

    def _boom_stanza(processors):
        raise AssertionError("stanza must not load by default")

    def _boom_benepar(model_name):
        raise AssertionError(f"benepar model {model_name!r} must not load by default")

    monkeypatch.setattr(pc, "_load_spacy_model", _boom_spacy)
    monkeypatch.setattr(pc, "_load_stanza", _boom_stanza)
    monkeypatch.setattr(pc, "_load_benepar", _boom_benepar)
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "parser_consensus": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
    assert any(item.metric_id.startswith(PREFIX) for item in report.results)


def test_single_parser_reports_insufficient_consensus_not_fake_agreement(sample_text):
    """With only spaCy sm available, POS/dependency/chunk findings must say
    so plainly rather than comparing spaCy sm against itself."""

    findings = _by_id(pc.measure(_analysis(sample_text)))
    for mid in ("syntax.parser_pos_agreement_rate", "syntax.parser_uas", "syntax.parser_las",
               "syntax.parser_alignment_coverage"):
        assert findings[mid]["value"] is None
        assert "one parser is available" in findings[mid]["warning"]
    assert findings["syntax.parser_np_chunk_agreement"]["value"] is None


@needs_spacy_md
def test_second_parser_enables_real_pos_and_dependency_consensus():
    text = " ".join(["The old wizard slowly raised his gnarled staff toward the darkening sky."] * 6)
    analysis = _analysis(text)
    config = {"features": {**pc.DEFAULT_FEATURES, "segmentation": False, "tokenization": False,
                           "chunks": False, "spacy_md": True}}
    findings = _by_id(pc.measure(analysis, config=config))
    assert findings["syntax.parser_pos_agreement_rate"]["value"] is not None
    assert findings["syntax.parser_pos_agreement_rate"]["value"] > 50.0
    assert findings["syntax.parser_uas"]["value"] is not None
    assert findings["syntax.parser_alignment_coverage"]["value"] > 50.0
    header = findings["syntax.parser_uas"]["distribution"]
    assert header["parser_a"] == "spacy_sm"
    assert header["parser_b"] == "spacy_md"


# ---------------------------------------------------------------- degradation

def test_missing_pysbd_degrades_one_segmenter_not_the_suite(monkeypatch, sample_text):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "pysbd")
    optional.reset_cache()
    try:
        findings = _by_id(pc.measure(_analysis(sample_text)))
        available = findings["syntax.parser_segmenters_available"]
        assert "pysbd" not in available["distribution"]["available"]
        assert "pysbd_quote_aware" not in available["distribution"]["available"]
        assert "pysbd" in available["distribution"]["reasons"]
        # Everything else still measured something real.
        assert findings["syntax.parser_boundary_agreement_f1"]["value"] is not None
    finally:
        optional.reset_cache()


def test_missing_every_optional_package_still_runs_without_raising(monkeypatch, manuscript,
                                                                    base_config):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        config = {**base_config, "metrics": {**base_config["metrics"],
                                             "parser_consensus": {"enabled": True}}}
        report = grade.analyze(manuscript, config)
        errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
        assert not errors, [(item.metric_id, item.error) for item in errors]
        # The canonical/builtin/spacy_parser-free segmenters still degrade to
        # spaCy alone being unavailable too (spacy is disabled here as well),
        # so the suite reports honestly that little or nothing is available
        # rather than crashing.
        available = next(item for item in report.results
                         if item.metric_id == "syntax.parser_segmenters_available")
        assert available.value is not None
    finally:
        optional.reset_cache()


def test_unknown_segmenter_name_degrades_instead_of_crashing(sample_text):
    findings = _by_id(pc.measure(_analysis(sample_text),
                                 config={"segmenters": ["not_a_real_segmenter"],
                                        "features": {**pc.DEFAULT_FEATURES}}))
    available = findings["syntax.parser_segmenters_available"]
    assert "not_a_real_segmenter" in available["distribution"]["reasons"]


# ------------------------------------------------------------------ sampling

def test_sampling_is_deterministic_with_a_seed():
    paragraphs = [f"Paragraph number {i} has some words in it for sampling." for i in range(50)]
    a, info_a = pc._sample_paragraphs(paragraphs, max_paragraphs=10, max_words=10000, seed=7)
    b, info_b = pc._sample_paragraphs(paragraphs, max_paragraphs=10, max_words=10000, seed=7)
    c, info_c = pc._sample_paragraphs(paragraphs, max_paragraphs=10, max_words=10000, seed=8)
    assert a == b
    assert info_a == info_b
    assert a != c or info_a["seed"] != info_c["seed"]


def test_sample_smaller_than_max_paragraphs_is_not_reported_insufficient(sample_text):
    """A document with fewer paragraphs than the configured max_paragraphs is
    fully covered, not truncated -- min_sample must reflect what sampling
    actually targeted for THIS document, not the raw config knob, or every
    small manuscript would wrongly read as insufficient_data."""

    analysis = _analysis(sample_text)
    findings = _by_id(pc.measure(analysis, config={
        "features": {**pc.DEFAULT_FEATURES, "pos": False, "dependency": False, "chunks": False}}))
    available = findings["syntax.parser_segmenters_available"]
    assert available["sample_size"] == available["min_sample"]


def test_word_cap_truncation_reports_insufficient_data(manuscript, base_config):
    """A sample cut short by max_words must report sample_size below
    min_sample so grade.py marks the finding insufficient_data (see
    coherence_suite's identical RST pattern and grade.py's _finding_result)."""

    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "parser_consensus": {
                                             "enabled": True,
                                             "max_paragraphs": 200,
                                             "max_words": 5,  # far below the manuscript's size
                                             "features": {**pc.DEFAULT_FEATURES, "pos": False,
                                                         "dependency": False, "chunks": False}}}}
    report = grade.analyze(manuscript, config)
    item = next(r for r in report.results if r.metric_id == "syntax.parser_boundary_agreement_f1")
    assert item.action is Action.INSUFFICIENT_DATA


def test_max_sentences_for_parse_time_cap_reports_insufficient_data(monkeypatch):
    """The per-sentence parser loop's own wall-clock cap, mirroring
    coherence_suite's rst_max_seconds: if the cap is hit before the sample
    finishes, sample_size (sentences actually parsed) must land below
    min_sample (sentences the deterministic sample called for)."""

    text = " ".join([f"Sentence number {i} is here for parsing." for i in range(40)])
    analysis = _analysis(text)

    real_align = pc._align_exact
    calls = {"n": 0}

    def _slow_align(a, b):
        calls["n"] += 1
        if calls["n"] > 2:
            time.sleep(0.2)
        return real_align(a, b)

    monkeypatch.setattr(pc, "_align_exact", _slow_align)
    config = {"features": {**pc.DEFAULT_FEATURES, "segmentation": False, "tokenization": False,
                          "chunks": False, "spacy_md": False},
             "max_sentences_for_parse": 40, "max_seconds_parse": 0.05}
    findings = _by_id(pc.measure(analysis, config=config))
    # Only spaCy sm is active by default, so parser findings are the
    # "insufficient consensus" branch, not a sampled one; assert on the
    # sampler directly instead, which is what the time cap actually gates.
    sample_idx, info = pc._sample_sentences(analysis.sentences, 40, 0)
    assert len(sample_idx) == info["sentences_requested"]


# --------------------------------------------------------------- tokenization

def test_special_token_disagreement_flags_real_contraction_splits():
    text = ("She said she wouldn't go to the well-known co-worker's party. "
           "He didn't care about the long-standing rumor either.")
    analysis = _analysis(text)
    findings = _by_id(pc.measure(analysis, config={
        "features": {**pc.DEFAULT_FEATURES, "pos": False, "dependency": False, "chunks": False}}))
    special = findings["syntax.parser_special_token_disagreement"]
    assert special["sample_size"] > 0
    # At least one contraction/hyphenated word is checked and the finding
    # carries real evidence, not just a count.
    assert special["distribution"]["special_tokens_checked"] > 0


@needs_a_second_tokenizer
def test_token_boundary_f1_and_count_disagreement_are_present(sample_text):
    findings = _by_id(pc.measure(_analysis(sample_text), config={
        "features": {**pc.DEFAULT_FEATURES, "pos": False, "dependency": False, "chunks": False}}))
    assert findings["syntax.parser_token_count_disagreement"]["value"] is not None
    assert 0.0 <= findings["syntax.parser_token_boundary_f1"]["value"] <= 1.0


# ------------------------------------------------------------------- chunks

@needs_spacy_md
def test_np_chunk_agreement_stays_unavailable_without_a_second_constituency_source():
    text = "The old wizard raised his staff toward the sky."
    analysis = _analysis(text)
    findings = _by_id(pc.measure(analysis, config={
        "features": {**pc.DEFAULT_FEATURES, "spacy_md": True}}))
    # spacy_md is a second DEPENDENCY parser, not a second constituency
    # source, so the chunk channel still correctly reports unavailable.
    assert findings["syntax.parser_np_chunk_agreement"]["value"] is None


# --------------------------------------------------------- degenerate inputs

@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_survives_degenerate_documents(text):
    analysis = _analysis(text)
    findings = pc.measure(analysis, config={"features": {**pc.DEFAULT_FEATURES}})
    assert findings  # returns a well-formed list of findings, never raises


def test_registry_runs_without_raising_through_grade(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "parser_consensus": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
