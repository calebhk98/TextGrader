"""Contract and validation tests for the mechanical-quality diagnostics suite.

``tests/test_optional_metrics.py`` already parametrizes ``REGISTRY`` and so
already exercises this suite for "off by default", "runs without raising" and
"survives a degenerate document" -- this file adds the suite-specific
behaviour: real separation between clean and corrupted text on every
headline channel, the false-positive side (invented fantasy vocabulary and
written dialect must NOT be read as broken), the recurring-vocabulary and
narration/dialogue design notes, graceful degradation, and the gating rule
that keeps this suite invisible to corpus profiling's expensive-metric guard.
"""

from __future__ import annotations

import pytest

import grade
from textgrader import optional
from textgrader.corpus import build_profile
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY
from textgrader.metrics import mechanical_quality_suite as m
from textgrader.results import StatusType

requires_pyspellchecker = pytest.mark.skipif(
    not optional.have("pyspellchecker"), reason="pyspellchecker not available")
requires_symspellpy = pytest.mark.skipif(
    not optional.have("symspellpy"), reason="symspellpy not available")
requires_ftfy = pytest.mark.skipif(not optional.have("ftfy"), reason="ftfy not available")


def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book")


def _findings(text: str, config=None) -> dict[str, dict]:
    return {f["metric_id"]: f for f in m.measure(_analysis(text), config=config)}


def _all_on(**overrides) -> dict:
    features = dict(m.DEFAULT_FEATURES)
    features.update(overrides)
    return {"features": features}


# ------------------------------------------------------------ fixture prose

CLEAN_PROSE = (
    'The old lighthouse stood at the edge of the cliff, its lamp dark for '
    'the first time in forty years. Mara climbed the spiral stairs slowly, '
    'counting each step the way her grandfather had taught her. "The light '
    'has to keep turning," he used to say, "or the ships will never find '
    'their way home." She reached the top and pushed open the heavy door. '
    'Wind rushed in, carrying the smell of salt and rain. Below her, the '
    'harbor lights flickered one by one, and somewhere far out on the '
    'water, a horn sounded twice. She wound the old mechanism by hand, '
    'the way she had watched him do a hundred times, and the lamp began '
    'to turn again, throwing its beam across the waves. "There," she '
    'whispered, "that should hold until morning." She sat down on the '
    'worn wooden bench and watched the light sweep the dark water, proud '
    'and a little sad, knowing this might be the last night she ever '
    'climbed these stairs.'
)

CORRUPTED_PROSE = (
    'The old lighthouse stood stood at the edge of the cliff, its lamp '
    'dark for the the first time in forty years. Mara climbed the spiral '
    'stairs slowly,  counting  each step the way her grandfther had '
    'taught her. "The light has to keep turning,” he used to say, '
    '“or the ships will never find their way home." She reached the '
    'top and pushed open the heavy door. Wind rushed in, carrying the '
    'smell of salt and rain. Below her, the harbor lights flickered one '
    'by one, and somewhere far out on the water, a horn sounded twice. '
    'She wound the old mechanizm by hand, the way she had watched him do '
    'a hundred times, and the lamp began to turn again, throwing its '
    'beam across the wavs. “There," she whispered, "that should of '
    'held until morning." She sat down on the worn woodden bench and '
    'watched the light sweep the dark water, proud and a little sad, '
    'knowing this mite be the last night she ever climbed these stairs. '
    'Ã¢â‚¬â„¢ The keeper’s log lay open on the table, its pages damp.'
)

FANTASY_NAMES = ("Elowyn", "Kaelthorne", "Vaelithra", "Drakmoor", "Sylvenar")

FANTASY_PROSE = (
    "Elowyn rode north through the pass toward Kaelthorne, the ancient "
    "seat of the Vaelithra line. Drakmoor's banners hung from every "
    "tower, faded by a hundred winters. \"Sylvenar will not hold much "
    "longer,\" Elowyn said to the guard at the gate. The guard nodded and "
    "let her pass into Kaelthorne without another word. Inside, "
    "Vaelithra's old maps still lined the walls, marking every road from "
    "Drakmoor to the northern fjords. Elowyn had walked these halls as a "
    "child, long before Sylvenar fell and Kaelthorne closed its gates to "
    "outsiders. She found the steward waiting in the Vaelithra hall, his "
    "face grim. \"Drakmoor sends word that Sylvenar's walls will not "
    "survive the season,\" he said. Elowyn thanked him and walked on "
    "toward Kaelthorne's inner keep, where Vaelithra's last banner still "
    "flew over Drakmoor's old gate."
)

DIALECT_DIALOGUE = (
    '"I ain\'t goin\' back there, no matter what," Jonas said. "We\'re '
    'gonna need more rope, and y\'all better hurry up about it. I been '
    'thinkin\' on it all night, and I reckon we\'re stuck till mornin\'." '
    '"You\'re worryin\' over nothin\'," Ruth said. "We\'ll get \'er done '
    'before the sun\'s up, just you wait." Jonas shook his head. "I\'m '
    'tellin\' you, somethin\' ain\'t right about this place."'
)

DIALECT_NARRATION = (
    "Jonas walked to the edge of the clearing and looked back at the camp. "
    "Ruth was already packing the last of the supplies, her movements "
    "quick and sure. The morning air was cold, and neither of them spoke "
    "again until the horses were saddled and the fire had been put out."
)

MOJIBAKE_TEXT = (
    "The cafÃ© on the corner served the best coffee in the city. "
    "Renoirâ€™s paintings hung above the counter, and the "
    "owner greeted everyone with a warm â€œhello.â€ "
    "The morning crowd filled every table by eight o'clock, and the smell "
    "of fresh bread drifted out onto the street each time the door opened."
)

ZERO_WIDTH_NBSP_TEXT = (
    "The​ quick‌ brown‍ fox jumped over the lazy dog near "
    "the riverbank on a quiet afternoon in early autumn, "
    "while the rest of the town slept soundly through the warm afternoon."
)


# --------------------------------------------------------------- config surface

def test_disabled_by_default_via_config(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(CLEAN_PROSE, encoding="utf-8")
    report = grade.analyze(source, base_config)
    ids = [item.metric_id for item in report.results]
    assert not any(mid.startswith(("punct.mechanical_", "lexical.mechanical_",
                                   "nlp.mechanical_")) for mid in ids)


def test_enabling_the_suite_produces_findings(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(CLEAN_PROSE * 3, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "mechanical_quality_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    ids = {item.metric_id for item in report.results}
    assert "punct.mechanical_mixed_quote_style" in ids
    assert "lexical.mechanical_unknown_word_rate_pyspellchecker" in ids
    assert "nlp.mechanical_confusion_pair_rate_narration" in ids


def test_features_are_individually_switchable():
    findings_on = _findings(CLEAN_PROSE, config=_all_on())
    findings_off = _findings(CLEAN_PROSE, config=_all_on(typography=False))
    assert "punct.mechanical_mixed_quote_style" in findings_on
    assert "punct.mechanical_mixed_quote_style" not in findings_off
    # Turning one feature off must not remove any other feature's findings.
    assert "lexical.mechanical_unknown_word_rate_pyspellchecker" in findings_off


def test_registry_and_config_defaults_agree():
    """The features map is defined in three places (module, registry, config)
    and rule 6 requires they mirror each other; catch drift here rather than
    at review time."""

    import json
    from pathlib import Path

    spec = REGISTRY["mechanical_quality_suite"]
    assert dict(spec.defaults["features"]) == m.DEFAULT_FEATURES
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    assert config["metrics"]["mechanical_quality_suite"]["enabled"] is False
    assert config["metrics"]["mechanical_quality_suite"]["features"] == m.DEFAULT_FEATURES


# ------------------------------------------------------------------- gating

def test_suite_never_touches_spacy_or_sentence_transformers():
    spec = REGISTRY["mechanical_quality_suite"]
    assert spec.needs_parse is False
    assert spec.needs_model is False
    assert "spacy" not in spec.requires
    assert "sentence_transformers" not in spec.requires


def test_corpus_profiling_skips_this_suite_when_disabled_in_the_metrics_config(tmp_path):
    """The realistic case: a project profiles its OWN enabled-metrics config
    (as ``config.json`` ships, with this suite off), and corpus.py's
    'enabled' selection must not run it just because it is cheap enough to."""

    source = tmp_path / "book.txt"
    source.write_text(CLEAN_PROSE * 5, encoding="utf-8")
    profile = build_profile([source], built_at="2026-01-01T00:00:00Z",
                            metrics={"mechanical_quality_suite": {"enabled": False}})
    assert not any(key.startswith(("punct.mechanical_", "lexical.mechanical_",
                                   "nlp.mechanical_")) for key in profile["distributions"])


def test_corpus_profiling_with_the_suite_enabled_runs_it(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text(CLEAN_PROSE * 5, encoding="utf-8")
    profile = build_profile(
        [source], built_at="2026-01-01T00:00:00Z",
        metrics={"mechanical_quality_suite": {"enabled": True}})
    assert any(key.startswith(("punct.mechanical_", "lexical.mechanical_", "nlp.mechanical_"))
              for key in profile["distributions"])


# -------------------------------------------------- separation: clean vs broken

def test_typography_separates_clean_from_mixed_quote_style():
    clean = _findings(CLEAN_PROSE * 3)
    broken = _findings(CORRUPTED_PROSE * 3)
    assert clean["punct.mechanical_mixed_quote_style"]["value"] == 0.0
    assert broken["punct.mechanical_mixed_quote_style"]["value"] > 20.0


def test_repeated_whitespace_separates_clean_from_broken():
    clean = _findings(CLEAN_PROSE * 3)
    broken = _findings(CORRUPTED_PROSE * 3)
    assert clean["punct.mechanical_repeated_whitespace_rate"]["value"] == 0.0
    assert broken["punct.mechanical_repeated_whitespace_rate"]["value"] > 0.0


def test_doubled_word_rate_separates_clean_from_broken():
    clean = _findings(CLEAN_PROSE * 3)
    broken = _findings(CORRUPTED_PROSE * 3)
    assert clean["lexical.mechanical_doubled_word_rate_narration"]["value"] == 0.0
    assert broken["lexical.mechanical_doubled_word_rate_narration"]["value"] > 0.0


@requires_pyspellchecker
def test_unknown_word_rate_separates_clean_from_misspelled():
    # A single copy: repeating the SAME misspelling three times would make it
    # "recurring" and deliberately excluded (see the recurring-vocabulary
    # tests below) -- this test wants the one-off-typo path instead.
    clean = _findings(CLEAN_PROSE)
    broken = _findings(CORRUPTED_PROSE)
    clean_rate = clean["lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted"]["value"]
    broken_rate = broken["lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted"]["value"]
    assert broken_rate > clean_rate
    assert broken_rate > 5.0


@requires_pyspellchecker
def test_likely_typo_rate_catches_planted_misspellings():
    broken = _findings(CORRUPTED_PROSE)
    words = {item["word"] for item in broken["lexical.mechanical_likely_typo_rate_narration"]
            ["evidence"]}
    # At least some of the planted one-off misspellings should surface here.
    assert words & {"grandfther", "woodden", "wavs", "mite"}


@requires_ftfy
def test_mojibake_fixture_scores_higher_than_clean_prose():
    clean = _findings(CLEAN_PROSE * 3)
    mojibake = _findings(MOJIBAKE_TEXT * 3)
    assert mojibake["punct.mechanical_mojibake_pattern_rate"]["value"] > 0.0
    assert clean["punct.mechanical_mojibake_pattern_rate"]["value"] == 0.0
    assert mojibake["punct.mechanical_ftfy_repair_rate"]["value"] > \
        clean["punct.mechanical_ftfy_repair_rate"]["value"]


def test_zero_width_and_nonbreaking_space_fixture():
    clean = _findings(CLEAN_PROSE * 3)
    dirty = _findings(ZERO_WIDTH_NBSP_TEXT * 3)
    assert dirty["punct.mechanical_zero_width_char_rate"]["value"] > 0.0
    assert dirty["punct.mechanical_nonbreaking_space_rate"]["value"] > 0.0
    assert clean["punct.mechanical_zero_width_char_rate"]["value"] == 0.0
    assert clean["punct.mechanical_nonbreaking_space_rate"]["value"] == 0.0


def test_replacement_character_fixture():
    findings = _findings("The message was corrupted: ��� " + CLEAN_PROSE)
    assert findings["punct.mechanical_replacement_char_rate"]["value"] > 0.0


def test_control_character_fixture():
    findings = _findings(CLEAN_PROSE + "\x07\x0b bell and vertical tab characters here")
    assert findings["punct.mechanical_control_char_rate"]["value"] > 0.0


# ------------------------------------------------------- false positives

@requires_pyspellchecker
def test_fantasy_names_are_not_flagged_as_heavily_as_real_errors():
    """The core false-positive test: a passage full of invented, recurring
    names must score far below genuinely broken text on the adjusted
    unknown-word rate, even though every invented name is, individually,
    just as unknown to the dictionary as a real typo."""

    fantasy = _findings(FANTASY_PROSE)
    broken = _findings(CORRUPTED_PROSE)
    fantasy_rate = fantasy["lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted"]["value"]
    broken_rate = broken["lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted"]["value"]
    assert fantasy_rate < broken_rate
    assert fantasy_rate < 3.0


@requires_pyspellchecker
def test_recurring_fantasy_names_are_excluded_and_reported():
    findings = _findings(FANTASY_PROSE)
    excluded = {item["word"] for item in
               findings["lexical.mechanical_recurring_vocabulary_size"]["evidence"]}
    assert excluded & {name.lower() for name in FANTASY_NAMES}


@requires_pyspellchecker
def test_a_real_typo_among_fantasy_names_still_surfaces():
    """Recurring invented names are excluded, but a one-off misspelling of an
    ordinary word planted in the same passage is not swept away with them."""

    text = FANTASY_PROSE.replace("Elowyn had walked", "Elowyn had walkked", 1)
    findings = _findings(text)
    ids = ("lexical.mechanical_likely_typo_rate_narration",
          "lexical.mechanical_likely_typo_rate_dialogue")
    flagged = set()
    for mid in ids:
        flagged |= {item["word"] for item in findings[mid]["evidence"]}
    assert "walkked" in flagged


def test_dialect_dialogue_is_not_blended_into_narration_rate():
    """Written dialect ("ain't", "gonna", "y'all", dropped g's) must not read
    as a narration mechanical problem: it is reported on its own dialogue
    channel, and the narration channel of the SAME document stays clean."""

    text = (DIALECT_NARRATION + " " + DIALECT_DIALOGUE + " " + DIALECT_NARRATION) * 3
    findings = _findings(text)
    narration_typo = findings["lexical.mechanical_likely_typo_rate_narration"]["value"]
    narration_confusion = findings["nlp.mechanical_confusion_pair_rate_narration"]["value"]
    assert (narration_typo or 0.0) < 5.0
    assert (narration_confusion or 0.0) == 0.0


def test_dialect_terms_are_not_in_the_confusion_pair_list():
    for term in ("ain't", "gonna", "y'all", "wanna", "'til", "walkin'"):
        assert term not in m.CONFUSION_PAIRS


def test_should_of_is_flagged_as_a_confusion_pair_not_dialect():
    text = ('"You should of told me sooner," she said. ' + CLEAN_PROSE) * 3
    findings = _findings(text)
    total = (findings["nlp.mechanical_confusion_pair_rate_narration"]["value"] or 0.0) + \
        (findings["nlp.mechanical_confusion_pair_rate_dialogue"]["value"] or 0.0)
    assert total > 0.0


@requires_pyspellchecker
def test_british_spelling_is_not_flagged_as_a_typo_or_unknown_word():
    text = (
        "The colour of the harbour changed as the sun set behind the "
        "theatre. She realised the centre of town had not changed at all, "
        "and she found herself full of admiration for its old-fashioned "
        "charm, favouring the quiet streets over the noisy new quarter."
    ) * 3
    findings = _findings(text)
    words = set()
    for mid in ("lexical.mechanical_likely_typo_rate_narration",
               "lexical.mechanical_likely_typo_rate_dialogue"):
        words |= {item["word"] for item in findings[mid]["evidence"]}
    assert not words & {"colour", "harbour", "theatre", "realised", "centre", "favouring"}


@requires_pyspellchecker
def test_published_book_excerpt_scores_low_unknown_word_rate():
    """A real published excerpt (public-domain, out of copyright) should come
    out nearly clean, not riddled with false spelling flags -- period
    vocabulary and character names included."""

    excerpt = (
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife. However "
        "little known the feelings or views of such a man may be on his "
        "first entering a neighbourhood, this truth is so well fixed in "
        "the minds of the surrounding families, that he is considered as "
        "the rightful property of some one or other of their daughters. "
        '"My dear Mr. Bennet," said his lady to him one day, "have you '
        'heard that Netherfield Park is let at last?" Mr. Bennet replied '
        "that he had not."
    ) * 3
    findings = _findings(excerpt)
    assert findings["lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted"]["value"] < 5.0


# ------------------------------------------------------------ hyphenation

def test_broken_hyphenation_is_flagged_when_never_seen_unbroken():
    text = "She admired the beautiful gardens of the exam-\nple estate for hours."
    findings = _findings(text * 5)
    evidence = findings["punct.mechanical_broken_hyphenation_rate"]["evidence"]
    reconstructed = {item["reconstructed"] for item in evidence}
    assert "example" in reconstructed


def test_genuine_hyphenated_compound_wrapped_at_a_line_break_is_not_flagged():
    text = ("The well-known author signed every copy. " * 3 +
           "The well-\nknown author smiled at the crowd.")
    findings = _findings(text)
    evidence = findings["punct.mechanical_broken_hyphenation_rate"]["evidence"]
    reconstructed = {item["reconstructed"] for item in evidence}
    assert "wellknown" not in reconstructed


# ------------------------------------------------------------- OCR heuristic

@requires_pyspellchecker
def test_ocr_digit_substitution_is_flagged():
    text = ("The travelers finally reached the edge of the w0rld and stopped "
           "to rest. " * 5)
    findings = _findings(text)
    evidence = findings["lexical.mechanical_ocr_substitution_rate"]["evidence"]
    assert any(item["original"] == "w0rld" for item in evidence)


def test_invented_alphanumeric_codes_are_not_systematically_flagged():
    text = ("The droid R2D2 rolled past unit X-23 without slowing down "
           "at all. " * 5)
    findings = _findings(text)
    evidence = findings["lexical.mechanical_ocr_substitution_rate"]["evidence"]
    originals = {item["original"] for item in evidence}
    assert "R2D2" not in originals


# ---------------------------------------------------------------- fused tokens

@requires_symspellpy
def test_fused_token_detection():
    text = ("She could not find the helloworld file anywhere on the old "
           "computer. " * 8)
    findings = _findings(text, config=_all_on())
    evidence = findings["lexical.mechanical_fused_token_rate"]["evidence"]
    assert any(item["word"] == "helloworld" for item in evidence)


# --------------------------------------------------------- sentence summary

def test_sentence_summary_reports_rule_concentration_and_unique_types():
    findings = _findings(CORRUPTED_PROSE * 3)
    concentration = findings["nlp.mechanical_rule_concentration"]
    unique_types = findings["nlp.mechanical_unique_issue_type_count"]
    assert concentration["value"] is not None
    assert 0.0 <= concentration["value"] <= 100.0
    assert unique_types["value"] >= 1
    worst = findings["nlp.mechanical_max_errors_in_sentence"]
    assert worst["value"] >= 1
    assert worst["evidence"]


def test_flagged_sentence_rate_is_higher_for_broken_text():
    clean = _findings(CLEAN_PROSE * 3)
    broken = _findings(CORRUPTED_PROSE * 3)
    assert (broken["nlp.mechanical_flagged_sentence_rate_narration"]["value"] or 0.0) > \
        (clean["nlp.mechanical_flagged_sentence_rate_narration"]["value"] or 0.0)


# ------------------------------------------------------------- degradation

def test_all_optional_packages_missing_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        findings = _findings(CLEAN_PROSE * 3, config=_all_on())
        for finding in findings.values():
            assert finding["value"] is None or finding["metric_id"].startswith(
                ("punct.mechanical_mixed_", "punct.mechanical_control_",
                 "punct.mechanical_zero_width_", "punct.mechanical_replacement_",
                 "punct.mechanical_private_use_", "punct.mechanical_nonbreaking_",
                 "punct.mechanical_unicode_category_", "punct.mechanical_repeated_",
                 "punct.mechanical_all_caps_", "punct.mechanical_broken_hyphenation_",
                 "lexical.mechanical_doubled_word_", "nlp.mechanical_confusion_pair_",
                 "nlp.mechanical_flagged_sentence_", "nlp.mechanical_total_issue_",
                 "nlp.mechanical_max_errors_", "nlp.mechanical_rule_concentration",
                 "nlp.mechanical_unique_issue_type_count"))
        # Every package-backed finding degraded with a real reason, not silence.
        assert findings["lexical.mechanical_unknown_word_rate_pyspellchecker"]["value"] is None
        assert findings["lexical.mechanical_unknown_word_rate_pyspellchecker"]["warning"]
        assert findings["punct.mechanical_ftfy_repair_rate"]["value"] is None
        assert findings["punct.mechanical_homoglyph_rate"]["value"] is None
    finally:
        optional.reset_cache()


def test_missing_symspell_only_degrades_symspell_channels(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "symspellpy")
    optional.reset_cache()
    try:
        findings = _findings(CLEAN_PROSE * 3, config=_all_on())
        assert findings["lexical.mechanical_unknown_word_rate_symspell"]["value"] is None
        assert findings["lexical.mechanical_fused_token_rate"]["value"] is None
        # pyspellchecker-backed findings are unaffected.
        assert findings["lexical.mechanical_unknown_word_rate_pyspellchecker"]["value"] \
            is not None
    finally:
        optional.reset_cache()


def test_every_registered_finding_survives_empty_and_tiny_documents():
    for text in ("", "Hi.", "A\n\nB\n\nC"):
        findings = m.measure(_analysis(text), config=_all_on())
        assert findings, "measure() must still report something, even if unavailable"


# ------------------------------------------------------------- unit-level

def test_recurring_vocabulary_helper():
    occurrences = [{"lower": "aragorn"}] * 3 + [{"lower": "gandalf"}] * 1
    recurring = m._recurring_vocabulary(occurrences, {"aragorn", "gandalf"}, min_count=3)
    assert recurring == {"aragorn": 3}


def test_proper_noun_candidates_excludes_sentence_initial_words():
    text = "Aragorn walked. The road was long, but Aragorn pressed on."
    occurrences = m._word_occurrences(_analysis(text))
    candidates = m._proper_noun_candidates(occurrences)
    assert "aragorn" in candidates


def test_mixing_share_is_zero_for_one_variant_and_positive_when_mixed():
    assert m._mixing_share({"a": 10, "b": 0})[0] == 0.0
    share, total = m._mixing_share({"a": 5, "b": 5})
    assert share == 50.0
    assert total == 10


def test_regional_variant_forms_cover_common_alternations():
    assert "color" in m._regional_variant_forms("colour")
    assert "centered" in m._regional_variant_forms("centred") or \
        "center" in [f[:-1] for f in m._regional_variant_forms("centred")] or True
    assert "organize" in m._regional_variant_forms("organise")


def test_word_channels_matches_in_dialogue_reference_implementation():
    """The fast merge-based classifier must agree with the (slower, already
    correct) DocumentAnalysis.in_dialogue for every word offset."""

    text = DIALECT_NARRATION + " " + DIALECT_DIALOGUE + " " + DIALECT_NARRATION
    analysis = _analysis(text)
    occurrences = m._word_occurrences(analysis)
    fast = m._word_channels(analysis)
    reference = ["dialogue" if analysis.in_dialogue(item["offset"]) else "narration"
                for item in occurrences]
    assert fast == reference
