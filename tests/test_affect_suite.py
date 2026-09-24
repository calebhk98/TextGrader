"""The experimental sentiment/emotion/affect/tone-trajectory suite (Task 15).

Every finding stays Polarity.NEUTRAL: this suite measures structure (level,
volatility, arc, dialogue-vs-narration, per-speaker spread, cross-engine
disagreement), never quality.  These tests check that structure directly,
with exact values on small controlled fixtures where the task's own
acceptance criteria ask for it (rule 7: assert exact values, not only
directions) and with direction/threshold checks on more realistic prose.
"""

from __future__ import annotations

import pytest

import grade
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY, affect_suite as affect

PREFIX = "discourse.affect_"


def _analysis(text: str, **kwargs) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book", **kwargs)


def _ids(findings):
    return {item["metric_id"] for item in findings}


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


def _require_or_skip(*packages: str) -> None:
    """Skip (not fail) a test that needs a real optional package for its
    numbers, exactly like the rest of this codebase's optional-metric tests
    do (see test_randomness_suite.py's ``optional.have(...)`` guards). This
    also covers ``TEXTGRADER_DISABLE_OPTIONAL=all`` runs, which simulate the
    package being absent without it actually being uninstalled.
    """

    missing = [name for name in packages if not optional.have(name)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}, not available in this environment")


@pytest.fixture(autouse=True)
def _restore_registry():
    """Undo any register_engine/cache mutation a test makes, even on failure."""

    before = set(affect._ENGINES)
    yield
    for name in set(affect._ENGINES) - before:
        affect._ENGINES.pop(name, None)
    affect._reset_caches()


# ------------------------------------------------------------------- wiring

def test_off_by_default(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(PREFIX)]


def test_registered_in_metrics_registry():
    assert "affect_suite" in REGISTRY
    spec = REGISTRY["affect_suite"]
    assert spec.module == "affect_suite"
    assert spec.family == "discourse"
    assert spec.cost == "moderate"
    # sentence_transformers is deliberately absent: including it would flip
    # needs_model and drop this suite out of default corpus profiling.
    assert "sentence_transformers" not in spec.requires
    assert spec.needs_model is False
    assert spec.needs_parse is False


def test_enabling_the_suite_turns_on_every_default_engine(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "affect_suite": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(manuscript, config).results}
    matched = {mid for mid in ids if mid.startswith(PREFIX)}
    assert matched
    for engine in ("vader", "nrc_valence", "afinn", "textblob_polarity", "textblob_subjectivity"):
        assert f"{PREFIX}{engine}_level" in matched
    # Off by default.
    assert f"{PREFIX}transformer_sentiment_level" not in matched
    assert f"{PREFIX}nrc_categories" in matched
    assert f"{PREFIX}empath_categories" in matched
    assert f"{PREFIX}disagreement" in matched


def test_every_finding_id_uses_the_stable_prefix_and_stays_neutral(sample_text):
    findings = affect.measure(_analysis(sample_text))
    assert findings
    assert all(item["metric_id"].startswith(PREFIX) for item in findings)
    # This module never sets a polarity; grade.py's MetricResult defaults to
    # Polarity.NEUTRAL when none is given, which is the contract this suite
    # relies on (see the module docstring: "no affect metric is quality").
    assert all("polarity" not in item for item in findings)


def test_features_are_independently_switchable(sample_text):
    analysis = _analysis(sample_text)
    without_vader = _ids(affect.measure(analysis, config={"features": {"vader": False}}))
    assert not any(mid.startswith(f"{PREFIX}vader_") for mid in without_vader)
    assert f"{PREFIX}afinn_level" in without_vader

    without_categories = _ids(affect.measure(analysis, config={
        "features": {"nrc_categories": False, "empath_categories": False,
                    "liwc_categories": False, "vad": False}}))
    assert f"{PREFIX}nrc_categories" not in without_categories
    assert f"{PREFIX}empath_categories" not in without_categories
    assert f"{PREFIX}liwc_categories" not in without_categories
    assert f"{PREFIX}vad_valence_level" not in without_categories
    assert f"{PREFIX}vader_level" in without_categories


# --------------------------------------------------------- separation tests
# Task 15's acceptance criteria: strongly positive vs negative, alternating
# emotional sentences (mean vs volatility), neutral technical prose, and a
# dialogue/narration tone contrast.

def test_strongly_positive_vs_negative_fixtures_separate():
    _require_or_skip("vaderSentiment", "afinn", "nrclex", "textblob")
    positive = _analysis("I love this. This is wonderful. What a brilliant, joyous day.")
    negative = _analysis("I hate this. This is terrible. What a dreadful, miserable day.")
    pos_findings = _by_id(affect.measure(positive))
    neg_findings = _by_id(affect.measure(negative))

    # The headline is the MEAN over sentences (see _level_finding's docstring
    # for why), not the median -- computed by hand from each engine's three
    # per-sentence scores rather than assumed.
    assert pos_findings[f"{PREFIX}vader_level"]["value"] == pytest.approx(0.6816, abs=1e-4)
    assert neg_findings[f"{PREFIX}vader_level"]["value"] == pytest.approx(-0.59183, abs=1e-4)
    assert pos_findings[f"{PREFIX}afinn_level"]["value"] == pytest.approx(1.244444, abs=1e-5)
    assert neg_findings[f"{PREFIX}afinn_level"]["value"] == pytest.approx(-1.066667, abs=1e-5)
    # The median is still available, in distribution, for comparison.
    assert pos_findings[f"{PREFIX}vader_level"]["distribution"]["median"] == pytest.approx(0.6369, abs=1e-4)
    assert pos_findings[f"{PREFIX}vader_level"]["distribution"]["aggregation"] == "mean over sentences"
    # Every signed engine agrees on sign here, which is the boring/expected case.
    for engine in ("vader", "afinn", "nrc_valence", "textblob_polarity"):
        assert pos_findings[f"{PREFIX}{engine}_level"]["value"] > 0
        assert neg_findings[f"{PREFIX}{engine}_level"]["value"] < 0


def test_mean_and_volatility_are_independent():
    """The key separation test: alternating affect keeps the mean near zero
    while volatility is at its structural maximum -- and the two metric ids
    must show exactly that, not one number standing in for both.

    A toy engine with an exact, content-addressed score keeps this fixture
    fully deterministic rather than depending on a third-party lexicon's
    exact numbers.
    """

    def make_scorer(settings):
        def score(text):
            if "POS" in text:
                return 1.0
            if "NEG" in text:
                return -1.0
            return 0.0
        return score, None

    affect.register_engine(affect.EngineSpec(
        name="toy_alternating", label="toy alternating", unit="score [-1,1]", signed=True,
        make_scorer=make_scorer))
    text = "POS one. NEG one. POS two. NEG two. POS three. NEG three."
    analysis = _analysis(text)
    findings = _by_id(affect.measure(analysis, config={"features": {
        name: False for name in affect.DEFAULT_FEATURES}}))

    level = findings[f"{PREFIX}toy_alternating_level"]
    volatility = findings[f"{PREFIX}toy_alternating_volatility"]
    arc = findings[f"{PREFIX}toy_alternating_arc"]
    reversal = findings[f"{PREFIX}toy_alternating_reversal_runs"]

    # The headline IS the mean (see _level_finding); here mean and median
    # coincide by symmetry, so this also doubles as a sanity check that the
    # mean-headline change did not disturb the symmetric case.
    assert level["value"] == pytest.approx(0.0)
    assert level["distribution"]["mean"] == pytest.approx(0.0)
    assert level["distribution"]["median"] == pytest.approx(0.0)
    assert level["distribution"]["aggregation"] == "mean over sentences"
    # Volatility is NOT zero even though the mean is: consecutive sentences
    # are always 2.0 apart on a [-1, 1] scale, the maximum possible.
    assert volatility["value"] == pytest.approx(2.0)
    assert arc["value"] == pytest.approx(0.0)
    assert arc["distribution"]["early_mean"] == pytest.approx(0.0)
    assert arc["distribution"]["late_mean"] == pytest.approx(0.0)
    # Every single adjacent pair reverses sign.
    assert reversal["value"] == pytest.approx(100.0)
    assert reversal["distribution"]["longest_positive_run_sentences"] == 1
    assert reversal["distribution"]["longest_negative_run_sentences"] == 1


def test_neutral_technical_prose_reads_as_neutral():
    _require_or_skip("vaderSentiment")
    text = ("The device operates at five volts of direct current. The manual lists three "
           "configuration options for the interface. Version two point one was released in "
           "March. The firmware update requires a restart of the controller. Each sensor "
           "reports a value once per second to the logging module.")
    findings = _by_id(affect.measure(_analysis(text)))
    level = findings[f"{PREFIX}vader_level"]
    share = findings[f"{PREFIX}vader_polarity_share"]
    # The mean is small but not exactly zero (one sentence carries a mild
    # signal); the median, in distribution, IS exactly zero here -- the
    # headline must read the former, not the latter.
    assert level["value"] == pytest.approx(0.068, abs=1e-4)
    assert level["distribution"]["median"] == pytest.approx(0.0)
    assert share["distribution"]["neutral_share"] >= 60.0
    assert share["distribution"]["neutral_share"] > share["distribution"]["positive_share"]


def test_level_headline_is_the_mean_on_a_zero_inflated_series():
    """The regression this suite must never repeat: a lexicon engine scores
    most ordinary sentences at exactly 0.0, and real books push that share
    past 50% (measured on Alice's Adventures in Wonderland, Tess of the
    d'Urbervilles and Flatland: 41-50% of sentences at exactly 0.0 for every
    signed engine). A median-based headline collapses to exactly 0.0 on
    every one of those books; the mean does not, and is what ..._level must
    report. A toy engine with an exact, content-addressed score keeps the
    fixture deterministic rather than depending on a lexicon's own numbers.
    """

    def make_scorer(settings):
        def score(text):
            return 1.0 if "POS" in text else 0.0
        return score, None

    affect.register_engine(affect.EngineSpec(
        name="toy_zero_inflated", label="toy zero-inflated", unit="score [0,1]", signed=True,
        make_scorer=make_scorer))
    sentences = [f"Neutral filler sentence number {i}." for i in range(8)]
    sentences += ["POS this is wonderful.", "POS this is amazing."]
    analysis = _analysis(" ".join(sentences))
    findings = _by_id(affect.measure(analysis, config={"features": {
        name: False for name in affect.DEFAULT_FEATURES}}))
    level = findings[f"{PREFIX}toy_zero_inflated_level"]

    assert level["sample_size"] == 10
    # Hand-computed: eight 0.0 sentences and two 1.0 sentences.
    assert level["value"] == pytest.approx(0.2)
    assert level["value"] != 0.0
    assert level["distribution"]["mean"] == pytest.approx(0.2)
    assert level["distribution"]["median"] == pytest.approx(0.0)  # what a median headline would report
    assert level["distribution"]["aggregation"] == "mean over sentences"
    assert level["distribution"]["zero_share"] == pytest.approx(80.0)
    assert level["distribution"]["nonzero_mean"] == pytest.approx(1.0)


def test_dialogue_and_narration_tones_separate():
    _require_or_skip("vaderSentiment")
    dialogue_lines = [f'"This is wonderful, I am delighted and thrilled, sentence {i}!" '
                     f'Alice said.' for i in range(25)]
    narration_lines = [f"The room felt cold, dreary and abandoned, item {i}." for i in range(25)]
    paragraphs = []
    for dialogue, narration in zip(dialogue_lines, narration_lines):
        paragraphs.append(dialogue)
        paragraphs.append(narration)
    analysis = _analysis("\n\n".join(paragraphs))
    findings = _by_id(affect.measure(analysis))
    gap = findings[f"{PREFIX}vader_dialogue_narration_gap"]
    assert gap["value"] == pytest.approx(1.21815, abs=1e-3)
    assert gap["distribution"]["dialogue_mean"] > 0
    assert gap["distribution"]["narration_mean"] < 0
    assert gap["distribution"]["dialogue_n"] >= affect.MIN_SAMPLE
    assert gap["distribution"]["narration_n"] >= affect.MIN_SAMPLE


# ------------------------------------------------------------ disagreement

def test_engines_disagree_and_disagreement_is_reported():
    """Negation is a known blind spot for plain lexicon sums: AFINN's
    ``.score`` has no negation handling, while VADER's does, so the two
    genuinely disagree on these sentences rather than being nudged apart
    artificially.
    """

    _require_or_skip("vaderSentiment", "afinn")
    base = ["This movie is not bad at all.", "I could not care less about this.",
           "The weather today is mild.", "He walked to the corner store quietly."]
    sentences = (base * 6)[:24]
    analysis = _analysis(" ".join(sentences))
    findings = _by_id(affect.measure(analysis))
    disagreement = findings[f"{PREFIX}disagreement"]
    assert disagreement["value"] is not None
    assert disagreement["value"] > 0  # engines are allowed to, and do, disagree
    pairs = {(item["engine_a"], item["engine_b"]): item for item in disagreement["evidence"]}
    afinn_vader = pairs[("afinn", "vader")]
    assert afinn_vader["sign_disagreement_rate"] == pytest.approx(100.0)
    assert afinn_vader["rank_correlation"] == pytest.approx(-1.0)
    assert afinn_vader["n"] == 24
    # Every pair is reported, not averaged away.
    assert len(disagreement["evidence"]) >= 3


def test_disagreement_reports_none_available_below_minimum_sample():
    analysis = _analysis("This is nice. That was bad.")
    findings = _by_id(affect.measure(analysis))
    disagreement = findings[f"{PREFIX}disagreement"]
    assert disagreement["value"] is None
    assert disagreement["warning"]


# --------------------------------------------------------------- registry

def test_register_engine_toy_and_score_text():
    affect.register_engine(affect.EngineSpec(
        name="toy_half", label="toy half", unit="score [0,1]", signed=False,
        make_scorer=lambda settings: (lambda text: 0.5, None)))
    assert "toy_half" in affect.list_engines()
    value, reason = affect.score_text("toy_half", "anything")
    assert value == pytest.approx(0.5)
    assert reason is None

    analysis = _analysis("One sentence. Another sentence here for good measure.")
    findings = _by_id(affect.measure(analysis, config={"features": {
        name: False for name in affect.DEFAULT_FEATURES}}))
    assert findings[f"{PREFIX}toy_half_level"]["value"] == pytest.approx(0.5)


def test_register_engine_rejects_duplicate_name_without_replace():
    affect.register_engine(affect.EngineSpec(
        name="toy_dup", label="toy", unit="u", signed=False,
        make_scorer=lambda settings: (lambda text: 0.0, None)))
    with pytest.raises(ValueError):
        affect.register_engine(affect.EngineSpec(
            name="toy_dup", label="toy again", unit="u", signed=False,
            make_scorer=lambda settings: (lambda text: 1.0, None)))
    # replace=True is allowed.
    affect.register_engine(affect.EngineSpec(
        name="toy_dup", label="toy again", unit="u", signed=False,
        make_scorer=lambda settings: (lambda text: 1.0, None)), replace=True)


def test_score_text_unknown_engine():
    value, reason = affect.score_text("no_such_engine", "text")
    assert value is None
    assert "unknown affect engine" in reason


def test_vad_placeholder_until_registered_then_wired_automatically():
    analysis = _analysis("One sentence here. Another one right here for good measure.")
    before = _by_id(affect.measure(analysis))
    for dimension in affect.VAD_DIMENSIONS:
        item = before[f"{PREFIX}vad_{dimension}_level"]
        assert item["value"] is None
        assert "no VAD engine registered" in item["warning"]

    affect.register_engine(affect.EngineSpec(
        name="vad_valence", label="toy VAD valence", unit="valence [0,1]", signed=True,
        kind="vad", make_scorer=lambda settings: (lambda text: 0.5, None)))
    after = _by_id(affect.measure(analysis))
    assert after[f"{PREFIX}vad_valence_level"]["value"] == pytest.approx(0.5)
    # The other two dimensions are still placeholders: registering one
    # dimension does not fabricate the other two.
    assert after[f"{PREFIX}vad_arousal_level"]["value"] is None


# ------------------------------------------------------------- per-speaker

def test_speaker_spread_needs_two_attributable_speakers():
    _require_or_skip("vaderSentiment")
    dense = "\n\n".join('"I cannot find it anywhere," Alice said.\n\n'
                        '"Then where did you look for it?" Bob said.' for _ in range(12))
    analysis = _analysis(dense)
    findings = _by_id(affect.measure(analysis))
    spread = findings[f"{PREFIX}vader_speaker_spread"]
    assert spread["value"] is not None
    assert spread["sample_size"] == 2
    assert set(spread["distribution"]["per_speaker_mean"]) == {"Alice", "Bob"}


def test_speaker_spread_unavailable_with_one_speaker():
    _require_or_skip("vaderSentiment")
    text = " ".join('"Only one voice here," Alice said.' for _ in range(20))
    analysis = _analysis(text)
    findings = _by_id(affect.measure(analysis))
    spread = findings[f"{PREFIX}vader_speaker_spread"]
    assert spread["value"] is None
    assert "speaker" in spread["warning"]


# -------------------------------------------------------- LIWC / Empath

def test_liwc_dictionary_parsing_is_verified_against_a_synthetic_fixture(tmp_path):
    _require_or_skip("liwc")
    dic_path = tmp_path / "toy.dic"
    dic_path.write_text(
        "%\n1\tposemo\n2\tnegemo\n%\nhappy\t1\njoy*\t1\nlove\t1\nsad\t2\nhate\t2\nterribl*\t2\n",
        encoding="utf-8")
    text = "He felt happy and joyful. She felt sad and terrible about it. " * 3
    analysis = _analysis(text)
    findings = _by_id(affect.measure(analysis, config={
        "liwc_dictionary_path": str(dic_path)}))
    item = findings[f"{PREFIX}liwc_categories"]
    assert item["value"] is not None
    assert item["distribution"]["category_counts"] == {"posemo": 6, "negemo": 6}
    assert item["distribution"]["possible_categories"] == 2


def test_liwc_without_a_configured_path_never_touches_the_package(monkeypatch):
    _real_require = optional.require

    def _guarded(name):
        if name == "liwc":
            raise AssertionError("liwc should never be imported without a configured path")
        return _real_require(name)

    monkeypatch.setattr(affect, "require", _guarded)
    analysis = _analysis("A perfectly ordinary sentence. And one more for good measure.")
    findings = _by_id(affect.measure(analysis))
    item = findings[f"{PREFIX}liwc_categories"]
    assert item["value"] is None
    assert "liwc_dictionary_path" in item["warning"]


def test_empath_analyze_never_calls_the_network(monkeypatch):
    """Empath() exposes a create_category() method that talks to a network
    backend; this suite must only ever call the local analyze() path.  A
    monkeypatched socket that raises on connect() proves nothing in this
    call path reaches the network.
    """

    module, reason = optional.require("empath")
    if module is None:
        pytest.skip(f"empath not installed: {reason}")

    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("affect_suite.empath_categories must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    text = "He was furious and terrified, screaming in the dark. " * 3
    analysis = _analysis(text)
    findings = _by_id(affect.measure(analysis))
    item = findings[f"{PREFIX}empath_categories"]
    assert item["value"] is not None
    assert item["distribution"]["active_categories"] > 0


# ---------------------------------------------------- transformer gating

def test_transformer_sentiment_is_off_by_default_and_never_loads(monkeypatch):
    """Rule 6's gating test: monkeypatch the loader itself to raise, so any
    attempt to import transformers/torch while the feature is off fails the
    test rather than merely going unnoticed.
    """

    _real_require = affect.require

    def _guarded(name):
        if name in ("transformers", "torch"):
            raise AssertionError(
                f"transformer_sentiment must not call require({name!r}) while disabled")
        return _real_require(name)

    monkeypatch.setattr(affect, "require", _guarded)
    analysis = _analysis("One sentence. Another sentence for good measure.")
    findings = _by_id(affect.measure(analysis))  # default config: feature is off
    assert f"{PREFIX}transformer_sentiment_level" not in findings


def test_transformer_sentiment_builds_when_explicitly_enabled(monkeypatch):
    """Exercises the real _make_transformer_scorer wrapper (sign mapping,
    truncation, model-name plumbing) against a fake transformers module
    rather than downloading the real checkpoint, which is exercised
    separately and manually (see the final report) precisely because it is
    the one heavy, off-by-default channel in this suite.
    """

    calls = []

    class _FakePipeline:
        def __call__(self, text, truncation=True):
            calls.append(text)
            return [{"label": "NEGATIVE", "score": 0.75}]

    class _FakeTransformers:
        @staticmethod
        def pipeline(task, model):
            assert task == "sentiment-analysis"
            return _FakePipeline()

    _real_require = affect.require

    def _fake_require(name):
        if name == "transformers":
            return _FakeTransformers(), None
        return _real_require(name)

    monkeypatch.setattr(affect, "require", _fake_require)
    analysis = _analysis("One sentence. Another sentence for good measure.")
    findings = _by_id(affect.measure(analysis, config={
        "features": {"transformer_sentiment": True}}))
    assert calls
    item = findings[f"{PREFIX}transformer_sentiment_level"]
    assert item["value"] == pytest.approx(-0.75)
    assert item["distribution"]["engine_version"]


# ------------------------------------------------------------ degradation

def test_a_missing_optional_package_degrades_one_engine_at_a_time(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        analysis = _analysis("One sentence. Another sentence here for good measure.")
        findings = affect.measure(analysis)
        assert findings
        for item in findings:
            if item["value"] is None:
                assert item["warning"]
    finally:
        optional.reset_cache()


def test_every_registered_metric_runs_on_degenerate_documents(base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "affect_suite": {"enabled": True}}}
    for text in ["", "Hi.", "A\n\nB\n\nC"]:
        for item in affect.measure(_analysis(text)):
            pass  # must not raise; grade.py-level degenerate coverage is in
                  # test_optional_metrics.py's parametrized sweep, which
                  # already includes affect_suite once registered.


def test_versions_and_resources_are_recorded():
    _require_or_skip("vaderSentiment", "nrclex")
    findings = _by_id(affect.measure(_analysis("One sentence. Another for measure.")))
    level = findings[f"{PREFIX}vader_level"]
    assert level["distribution"]["engine_version"]
    assert level["distribution"]["resource"]
    categories = findings[f"{PREFIX}nrc_categories"]
    assert categories["distribution"]["engine_version"]
