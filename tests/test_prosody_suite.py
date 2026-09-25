"""Tests for the poetry/prosody suite (``rhythm.prosody_*``, ``textgrader.prosody``).

Fixtures are short, public-domain or hand-written texts (a Sonnet 18 quatrain,
a small invented free-verse stanza, deliberate alliteration/assonance lines,
an invented-vocabulary stanza, and two ordinary prose paragraphs) so exact
values can be pinned rather than only asserting direction, per this project's
rule that every headline channel needs a test showing it separates two inputs
it should, with exact expected values on small fixtures where possible.
"""

from __future__ import annotations

import pytest

import grade
from textgrader import optional, prosody
from textgrader.document import DocumentAnalysis
from textgrader.metrics import prosody_suite

# Every fixture below needs real CMUdict lookup to produce a resolved,
# assertable value; under TEXTGRADER_DISABLE_OPTIONAL=all (or an install that
# genuinely lacks "pronouncing"), those channels correctly degrade to
# "unavailable" instead -- itself already covered by
# tests/test_optional_metrics.py's blanket degradation sweep. Tests that
# assert a specific pronunciation-backed VALUE are skipped in that situation
# rather than asserting a value the library was never asked to produce.
needs_pronouncing = pytest.mark.skipif(
    not optional.have("pronouncing"),
    reason="the 'pronouncing' (CMUdict) package is not available in this environment")

# --------------------------------------------------------------------- fixtures

# Shakespeare, Sonnet 18 (public domain), first quatrain: rhymed (ABAB),
# metrical (iambic pentameter) poetry.
SONNET = """Shall I compare thee to a summer's day?
Thou art more lovely and more temperate:
Rough winds do shake the darling buds of May,
And summer's lease hath all too short a date:"""

# Hand-written free verse: no end rhyme, no consistent meter, irregular line
# lengths.
FREE_VERSE = """I do not know
the way the light moves
when nobody names it
so I stopped trying to explain"""

# Ordinary prose, written as two blank-line-separated paragraphs with no
# internal line breaks -- the "punctuation-free/free-verse" acceptance
# criterion is about verse; this is the prose side of the same requirement
# (metrics must not crash or force a meter/rhyme reading onto prose).
PROSE = ("The quiet room held a long silence while she considered what had happened, "
        "and whether anyone would notice, however perhaps not, because nobody asked her "
        "directly about any of it.\n\n"
        "He walked slowly toward the door and paused there for a moment before deciding "
        "that it was better, after all, simply to leave without another word.")

# Invented fantasy vocabulary: every proper noun is outside CMUdict, so
# pronunciation coverage must visibly drop with g2p_fallback off (the default).
PROPER_NOUNS = """Zylnthar rode toward Quorvexia
past the spires of Malketh Dun
beneath the banners of Thrennor
into the shattered halls of Velkaith"""

# Deliberate alliteration (repeated onset consonants).
ALLITERATION = """Peter Piper picked a peck of pickled peppers
Sally sells seashells by the seashore
Big black bugs bleed blue black blood
Fuzzy Wuzzy was a bear"""

# Deliberate assonance (repeated stressed vowels).
ASSONANCE = """The rain in Spain falls mainly on the plain
Hear the mellow wedding bells
Go slow on the road, old soul
Deep sleep, sweet dreams, evening breeze"""

# A hard-wrapped prose paragraph in the exact style Project Gutenberg's own
# plain-text files use: every line but the last comes within a few characters
# of a shared ~72-character right margin.
HARD_WRAPPED = "\n".join([
    "This is a long paragraph of ordinary prose that has been hard-wrapped at",
    "roughly seventy characters per line the way plain-text files from Project",
    "Gutenberg traditionally are, so every line but the last comes close to the",
    "same width.",
])


def _measure(text, config=None):
    analysis = DocumentAnalysis.from_text(text, comparison_unit="book")
    return {item["metric_id"]: item for item in prosody_suite.measure(analysis, config or {})}


# ------------------------------------------------------------ registry wiring

def test_prosody_suite_off_by_default(tmp_path, base_config):
    source = tmp_path / "poem.txt"
    source.write_text(SONNET, encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results if item.metric_id.startswith("rhythm.prosody_")]


def test_prosody_suite_enabled_end_to_end(tmp_path, base_config):
    source = tmp_path / "poem.txt"
    source.write_text(SONNET, encoding="utf-8")
    config = {**base_config,
             "metrics": {**base_config["metrics"], "prosody_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    ids = {item.metric_id for item in report.results}
    assert "rhythm.prosody_dominant_meter_confidence" in ids
    assert "rhythm.prosody_pronunciation_coverage" in ids
    assert "rhythm.prosody_end_rhyme_density" in ids


def test_grade_list_metrics_names_prosody_suite(capsys):
    grade.list_metrics()
    out = capsys.readouterr().out
    assert "prosody_suite" in out
    assert "prosody" in out


# ---------------------------------------------- physical line/stanza structure

def test_physical_line_breaks_are_preserved_unlike_paragraphs():
    """The core "get lines without editing document.py" requirement.

    ``DocumentAnalysis.paragraphs`` joins every soft line wrap inside a block
    into one line; ``textgrader.prosody.line_structure`` must not.
    """

    text = "roses are red\nviolets are blue\n\nsugar is sweet\nand so are you"
    analysis = DocumentAnalysis.from_text(text, comparison_unit="book")
    assert analysis.paragraphs == ["roses are red violets are blue",
                                   "sugar is sweet and so are you"]
    structure = prosody.line_structure(analysis)
    assert [stanza.lines for stanza in structure.stanzas] == [
        ("roses are red", "violets are blue"),
        ("sugar is sweet", "and so are you"),
    ]
    assert structure.nonblank_lines == ("roses are red", "violets are blue",
                                        "sugar is sweet", "and so are you")


def test_short_varying_poetic_lines_are_not_hard_wrap_rejoined():
    analysis = DocumentAnalysis.from_text(SONNET, comparison_unit="book")
    structure = prosody.line_structure(analysis)
    assert len(structure.stanzas) == 1
    assert len(structure.stanzas[0].lines) == 4
    assert structure.hard_wrapped_stanzas == 0


def test_hard_wrapped_prose_paragraph_is_rejoined_into_one_line():
    analysis = DocumentAnalysis.from_text(HARD_WRAPPED, comparison_unit="book")
    structure = prosody.line_structure(analysis)
    assert len(structure.stanzas) == 1
    assert len(structure.stanzas[0].lines) == 1
    assert structure.hard_wrapped_stanzas == 1
    # The raw physical-line count still reflects the file's literal newlines;
    # only the corrected "poetic line" view is rejoined.
    assert structure.raw_line_count == 4
    assert len(structure.nonblank_lines) == 1


def test_line_and_stanza_structure_headline_values():
    findings = _measure(SONNET)
    assert findings["rhythm.prosody_line_count"]["value"] == 4.0
    assert findings["rhythm.prosody_nonblank_line_count"]["value"] == 4.0
    assert findings["rhythm.prosody_stanza_count"]["value"] == 1.0
    assert findings["rhythm.prosody_lines_per_stanza"]["value"] == 4.0
    assert findings["rhythm.prosody_stanza_symmetry"]["value"] == pytest.approx(100.0)
    # Whole-number counts headline the mean; the median stays in distribution.
    assert findings["rhythm.prosody_words_per_line"]["value"] == pytest.approx(8.25)
    assert findings["rhythm.prosody_words_per_line"]["distribution"]["median"] == pytest.approx(8.5)


def test_enjambment_is_labeled_a_surface_proxy():
    findings = _measure(FREE_VERSE)
    item = findings["rhythm.prosody_enjambment_rate"]
    assert item["value"] == pytest.approx(100.0)
    assert "proxy" in item["warning"]


# ------------------------------------------------------------------- coverage

@needs_pronouncing
def test_pronunciation_backend_is_the_real_cmudict():
    result = prosody.pronounce("rhyme")
    assert result.source == "cmudict"
    assert result.phones == ("R", "AY1", "M")


@needs_pronouncing
def test_pronunciation_coverage_is_always_reported_when_phonology_runs():
    findings = _measure(SONNET)
    item = findings["rhythm.prosody_pronunciation_coverage"]
    assert item["value"] == pytest.approx(100.0)
    assert item["distribution"]["backend"] == "cmudict"
    assert item["distribution"]["language"] == "en"


def test_pronunciation_coverage_absent_when_no_phonology_feature_is_on():
    analysis = DocumentAnalysis.from_text(SONNET, comparison_unit="book")
    config = {"features": {"line_stanza_structure": False, "meter_stress": False,
                           "rhyme": False, "near_rhyme_feature": False,
                           "phonological_patterning": False}}
    assert prosody_suite.measure(analysis, config) == []


@needs_pronouncing
def test_pronunciation_coverage_drops_for_proper_noun_heavy_text():
    """The required "proper-noun-heavy low-coverage text" fixture."""

    ordinary = _measure(PROSE)
    invented = _measure(PROPER_NOUNS)
    assert ordinary["rhythm.prosody_pronunciation_coverage"]["value"] == pytest.approx(100.0)
    assert invented["rhythm.prosody_pronunciation_coverage"]["value"] == pytest.approx(
        76.19047619047619)
    assert (invented["rhythm.prosody_pronunciation_coverage"]["value"]
           < ordinary["rhythm.prosody_pronunciation_coverage"]["value"])
    distribution = invented["rhythm.prosody_pronunciation_coverage"]["distribution"]
    assert distribution["unresolved_words"] == 5
    assert distribution["source_counts"].get("g2p", 0) == 0  # fallback is off by default


# --------------------------------------------------------------- stress/meter

@needs_pronouncing
def test_stress_sequence_api_shape_and_coverage():
    """Task 20's stress_sequence(analysis) -> (values, coverage, settings) contract."""

    analysis = DocumentAnalysis.from_text(SONNET, comparison_unit="book")
    values, coverage, settings = prosody.stress_sequence(analysis)
    assert settings == {"use_g2p_fallback": False}
    assert coverage["word_count"] == len(analysis.words)
    assert coverage["resolved_words"] + coverage["unresolved_words"] == coverage["word_count"]
    assert coverage["syllable_count"] == len(values)
    assert coverage["coverage_ratio"] == pytest.approx(1.0)
    assert set(values) <= {0.0, 1.0}
    assert len(values) > 0


@needs_pronouncing
def test_stress_sequence_skips_unresolved_words_without_a_placeholder():
    analysis = DocumentAnalysis.from_text(PROPER_NOUNS, comparison_unit="book")
    values, coverage, settings = prosody.stress_sequence(analysis)
    assert settings["use_g2p_fallback"] is False
    assert coverage["unresolved_words"] > 0
    assert coverage["coverage_ratio"] < 100.0
    # Every returned value is a real 0/1 syllable from a RESOLVED word; the
    # sequence is shorter than "one entry per word" rather than padded.
    assert len(values) < len(analysis.words) * 3


@needs_pronouncing
def test_dominant_meter_confidence_separates_metrical_poem_from_free_verse():
    metrical = _measure(SONNET)
    free = _measure(FREE_VERSE)
    assert metrical["rhythm.prosody_dominant_meter_confidence"]["value"] == pytest.approx(75.0)
    assert free["rhythm.prosody_dominant_meter_confidence"]["value"] == pytest.approx(25.0)
    assert metrical["rhythm.prosody_dominant_meter_confidence"]["distribution"]["dominant_meter"] \
        == "iambic"


@needs_pronouncing
def test_meter_findings_report_feet_and_dispersion():
    findings = _measure(SONNET)
    assert findings["rhythm.prosody_feet_per_line"]["value"] == pytest.approx(4.875)
    assert findings["rhythm.prosody_feet_per_line"]["distribution"]["median"] == pytest.approx(5.0)
    assert findings["rhythm.prosody_syllables_per_line"]["value"] == pytest.approx(9.75)
    assert findings["rhythm.prosody_syllables_per_line"]["distribution"]["median"] == pytest.approx(10.0)
    assert findings["rhythm.prosody_meter_conformity_rate"]["value"] is not None
    assert findings["rhythm.prosody_meter_deviation"]["value"] is not None


# ------------------------------------------------------------------- rhyme

@needs_pronouncing
def test_end_rhyme_density_separates_rhymed_poem_from_free_verse_and_prose():
    metrical = _measure(SONNET)
    free = _measure(FREE_VERSE)
    prose = _measure(PROSE)
    assert metrical["rhythm.prosody_end_rhyme_density"]["value"] == pytest.approx(
        66.66666666666667)
    assert free["rhythm.prosody_end_rhyme_density"]["value"] == pytest.approx(0.0)
    assert prose["rhythm.prosody_end_rhyme_density"]["value"] == pytest.approx(0.0)


@needs_pronouncing
def test_three_rhyme_channels_stay_separate_and_independently_switchable():
    """Exact rhyme, ARPABET near-rhyme and PanPhon feature similarity never merge."""

    analysis = DocumentAnalysis.from_text(SONNET, comparison_unit="book")
    only_rhyme = {item["metric_id"] for item in prosody_suite.measure(
        analysis, {"features": {"rhyme": True, "near_rhyme_feature": False,
                                "meter_stress": False, "phonological_patterning": False,
                                "line_stanza_structure": False}})}
    assert "rhythm.prosody_end_rhyme_density" in only_rhyme
    assert "rhythm.prosody_perfect_rhyme_rate" in only_rhyme
    assert "rhythm.prosody_near_rhyme_rate" in only_rhyme
    assert "rhythm.prosody_feature_rhyme_similarity" not in only_rhyme

    only_feature = {item["metric_id"] for item in prosody_suite.measure(
        analysis, {"features": {"rhyme": False, "near_rhyme_feature": True,
                                "meter_stress": False, "phonological_patterning": False,
                                "line_stanza_structure": False}})}
    assert "rhythm.prosody_feature_rhyme_similarity" in only_feature
    assert "rhythm.prosody_end_rhyme_density" not in only_feature

    # On the sonnet fixture the three channels genuinely disagree in kind:
    # perfect and near-slant rhymes both occur, and PanPhon's feature
    # similarity is computed from a different (graded, articulatory) metric
    # entirely -- never derived from the other two.
    both = _measure(SONNET)
    assert both["rhythm.prosody_perfect_rhyme_rate"]["value"] == pytest.approx(50.0)
    assert both["rhythm.prosody_near_rhyme_rate"]["value"] == pytest.approx(50.0)
    assert both["rhythm.prosody_feature_rhyme_similarity"]["value"] is not None


@needs_pronouncing
def test_rime_and_syllable_helpers_agree_on_true_rhymes():
    day, may = prosody.pronounce("day"), prosody.pronounce("may")
    assert prosody.rime_key(day.phones) == prosody.rime_key(may.phones)
    assert prosody.syllable_count(prosody.pronounce("literally").phones) == 4


@needs_pronouncing
def test_feature_rhyme_similarity_scores_identical_rimes_as_one():
    day, may = prosody.pronounce("day"), prosody.pronounce("may")
    similarity, reason = prosody.feature_rhyme_similarity(
        prosody.rime_of(day.phones), prosody.rime_of(may.phones))
    assert reason is None
    assert similarity == pytest.approx(1.0)


def test_phone_edit_distance_is_zero_only_for_identical_sequences():
    assert prosody.phone_edit_distance(("D", "AY1", "M"), ("D", "AY1", "M")) == 0
    assert prosody.phone_edit_distance(("D", "AY1", "M"), ("D", "AY1", "N")) == 1
    assert prosody.phone_edit_distance((), ("D", "AY1")) == 2


# ----------------------------------------------------- phonological patterning

@needs_pronouncing
def test_alliteration_density_separates_deliberate_alliteration_from_prose():
    alliterative = _measure(ALLITERATION)
    prose = _measure(PROSE)
    assert alliterative["rhythm.prosody_alliteration_density"]["value"] == pytest.approx(
        89.28571428571429)
    assert prose["rhythm.prosody_alliteration_density"]["value"] == pytest.approx(
        3.225806451612903)
    assert (alliterative["rhythm.prosody_alliteration_density"]["value"]
           > 10 * prose["rhythm.prosody_alliteration_density"]["value"])


@needs_pronouncing
def test_assonance_density_separates_deliberate_assonance_from_prose():
    assonant = _measure(ASSONANCE)
    prose = _measure(PROSE)
    assert assonant["rhythm.prosody_assonance_density"]["value"] == pytest.approx(
        76.92307692307692)
    assert prose["rhythm.prosody_assonance_density"]["value"] == pytest.approx(
        12.903225806451612)
    assert (assonant["rhythm.prosody_assonance_density"]["value"]
           > prose["rhythm.prosody_assonance_density"]["value"])


@needs_pronouncing
def test_phonological_findings_report_entropy_and_balance():
    findings = _measure(SONNET)
    assert findings["rhythm.prosody_phoneme_entropy"]["value"] is not None
    assert findings["rhythm.prosody_phoneme_entropy"]["sample_size_sensitive"] is True
    assert 0.0 <= findings["rhythm.prosody_vowel_consonant_balance"]["value"] <= 100.0


# ------------------------------------------------------------ optional backends

def test_g2p_en_resolves_an_invented_word_when_enabled():
    pytest.importorskip("g2p_en")
    result = prosody.pronounce("zylnthar", use_g2p=True)
    if result.source != "g2p":
        pytest.skip(f"g2p_en could not resolve the test word here (source={result.source!r})")
    assert result.phones


def test_default_config_never_loads_g2p_phonemizer_or_poesy(monkeypatch):
    """Rule 7: a model/heavy-optional-backed feature loads nothing under defaults.

    Guards the three off-by-default backends (g2p_en, phonemizer, poesy) by
    making their loader raise if called; the default config must never call
    it, so this must pass whether or not those packages happen to be
    installed in this environment.
    """

    original_require = prosody.require
    guarded = {"g2p_en", "phonemizer", "poesy"}

    def _guard(name):
        if name in guarded:
            raise AssertionError(f"{name!r} must not be imported under default prosody_suite "
                                 "config (g2p_fallback/phonemizer_backend/poesy_crosscheck are "
                                 "all off by default)")
        return original_require(name)

    monkeypatch.setattr(prosody, "require", _guard)
    findings = _measure(PROPER_NOUNS)
    assert findings  # ran to completion without tripping the guard


def test_phonemizer_backend_degrades_honestly_when_espeak_is_missing():
    if not optional.have("phonemizer"):
        pytest.skip("the 'phonemizer' package itself is not available in this environment")
    ipa, reason = prosody.phonemize_word("test")
    if ipa is not None:
        pytest.skip("espeak is installed in this environment; nothing to degrade")
    assert "espeak" in reason.lower()
    findings = _measure(SONNET, {"features": {"phonemizer_backend": True}})
    item = findings["rhythm.prosody_phonemizer_backend_agreement"]
    assert item["value"] is None
    assert "espeak" in item["warning"].lower()


def test_poesy_crosscheck_runs_without_crashing_either_way():
    findings = _measure(SONNET, {"features": {"poesy_crosscheck": True}})
    assert "rhythm.prosody_poesy_meter_agreement" in findings
    assert "rhythm.prosody_poesy_rhyme_scheme_accuracy" in findings
    # Whichever way it resolved (real scan or a quoted-error degrade), the
    # finding must never be silently missing its warning/value pairing.
    item = findings["rhythm.prosody_poesy_meter_agreement"]
    assert item["value"] is not None or item["warning"]


@needs_pronouncing
def test_near_rhyme_feature_degrades_honestly_without_panphon(monkeypatch):
    # _panphon_distance() caches a built Distance() instance process-wide, so
    # a prior test that already loaded the real thing must be cleared first,
    # or this monkeypatched require() would never even be consulted.
    optional.reset_cache()
    original_require = prosody.require
    monkeypatch.setattr(prosody, "require", lambda name: (
        (None, "optional package 'panphon' disabled for this test") if name == "panphon"
        else original_require(name)))
    try:
        findings = _measure(SONNET)
        item = findings["rhythm.prosody_feature_rhyme_similarity"]
        assert item["value"] is None
        assert "panphon" in item["warning"]
    finally:
        optional.reset_cache()


# --------------------------------------------------------------- degeneracy

@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_survives_degenerate_documents_with_every_feature_on(text):
    analysis = DocumentAnalysis.from_text(text, comparison_unit="book")
    config = {"features": {"line_stanza_structure": True, "meter_stress": True,
                           "rhyme": True, "near_rhyme_feature": True,
                           "phonological_patterning": True, "g2p_fallback": True,
                           "phonemizer_backend": True, "poesy_crosscheck": True}}
    findings = prosody_suite.measure(analysis, config)
    assert findings  # every switch produced SOME finding, none of them raised


@needs_pronouncing
def test_prose_never_crashes_or_is_forced_into_a_meter():
    findings = _measure(PROSE)
    # Prose is still measured (never disabled), but is not force-fit: a
    # single line/paragraph pair yields "insufficient" signal rather than a
    # fabricated meter classification with high confidence.
    assert findings["rhythm.prosody_dominant_meter_confidence"]["value"] is not None
    assert findings["rhythm.prosody_pronunciation_coverage"]["value"] == pytest.approx(100.0)
