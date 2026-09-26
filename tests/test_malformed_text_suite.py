"""Contract and validation tests for the malformed-text/language-ID suite.

``tests/test_optional_metrics.py`` already parametrizes ``REGISTRY`` and so
already exercises this suite for "off by default", "runs without raising" and
"survives a degenerate document". This file adds the suite-specific behaviour
the task spec asks for: real separation between clean and corrupted text on
every headline channel, the false-positive side (invented fantasy vocabulary
and written dialect must NOT be read as broken/foreign), graceful
degradation, and the config/registry mirroring rule.
"""

from __future__ import annotations

import json
import re
import random
from pathlib import Path

import pytest

import grade
from textgrader import optional
from textgrader.corpus import build_profile
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY
from textgrader.metrics import malformed_text_suite as m
from textgrader.results import StatusType

requires_wordfreq = pytest.mark.skipif(not optional.have("wordfreq"), reason="wordfreq not available")
requires_symspellpy = pytest.mark.skipif(not optional.have("symspellpy"),
                                         reason="symspellpy not available")
requires_wordninja = pytest.mark.skipif(not optional.have("wordninja"),
                                        reason="wordninja not available")
requires_wordsegment = pytest.mark.skipif(not optional.have("wordsegment"),
                                          reason="wordsegment not available")
requires_a_language_detector = pytest.mark.skipif(
    not any(optional.have(pkg) for pkg in ("lingua", "langid", "langdetect", "fasttext", "gcld3")),
    reason="no language-ID detector is available")


def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book")


def _findings(text: str, config=None) -> dict[str, dict]:
    return {f["metric_id"]: f for f in m.measure(_analysis(text), config=config)}


def _all_on(**overrides) -> dict:
    """Every feature on, plus any overrides -- feature flags nest under
    ``features``, everything else (thresholds, sample caps) is a top-level
    option, exactly as ``measure``/``option`` reads it."""

    features = dict(m.DEFAULT_FEATURES)
    config: dict = {}
    for key, value in overrides.items():
        if key in m.DEFAULT_FEATURES:
            features[key] = value
        else:
            config[key] = value
    config["features"] = features
    return config


# ------------------------------------------------------------ fixture prose

CLEAN_PROSE = (
    "The old lighthouse stood at the edge of the cliff, its lamp dark for "
    "the first time in forty years. Mara climbed the spiral stairs slowly, "
    "counting each step the way her grandfather had taught her. The light "
    "has to keep turning, he used to say, or the ships will never find "
    "their way home. She reached the top and pushed open the heavy door. "
    "Wind rushed in, carrying the smell of salt and rain. Below her, the "
    "harbor lights flickered one by one, and somewhere far out on the "
    "water, a horn sounded twice. She wound the old mechanism by hand, "
    "the way she had watched him do a hundred times, and the lamp began "
    "to turn again, throwing its beam across the waves. I was beside "
    "myself with worry and could not find another way in, so I went upon "
    "the road and walked into town, wondering whether anyone else had "
    "noticed anything odd about the lot of us waiting there in the rain, "
    "up on the hill, an hour or so before dawn."
)

# Long tokens fused with no space, so the mechanical suite's
# fused-token/hyphenation channels never fire (nothing here is hyphenated
# across a line break, and every fused pair is a single unbroken token).
CONCATENATED_PROSE = CLEAN_PROSE.replace(
    "harbor lights flickered", "harborlightsflickered").replace(
    "horn sounded twice", "hornsoundedtwice").replace(
    "wind rushed", "windrushed").replace("Wind rushed", "Windrushed")


def _random_split(text: str, seed: int = 7) -> str:
    """Insert a space at a random INTERIOR position of ~half the long words.

    Deliberately mid-word rather than at a natural morpheme boundary
    ("light house" from "lighthouse" is itself two real words and is not
    what this suite's split-word channel is built to catch -- see its
    module docstring); this is what an actual OCR/typing space-insertion
    corruption looks like.
    """

    rng = random.Random(seed)

    def repl(match: re.Match) -> str:
        word = match.group(0)
        if len(word) < 6 or rng.random() >= 0.5:
            return word
        pos = rng.randint(2, len(word) - 2)
        return word[:pos] + " " + word[pos:]

    return re.sub(r"[A-Za-z]{6,}", repl, text)


SPACED_PROSE = _random_split(CLEAN_PROSE)

# Each sentence as its own paragraph, so the section/paragraph-level channels
# below (which slice on paragraph boundaries) have real paragraphs to work
# with rather than one giant block.
CLEAN_PARAGRAPHS = "\n\n".join(s.strip() + "." for s in CLEAN_PROSE.split(". ") if s.strip())

FRENCH_PARAGRAPH = (
    "Ceci est un paragraphe entier écrit en français pour vérifier que le "
    "détecteur de langue remarque un véritable changement de langue à "
    "l'intérieur d'un document par ailleurs anglais, avec suffisamment de "
    "mots pour que chaque détecteur indépendant soit raisonnablement confiant "
    "quant à la langue employée dans ce passage particulier du texte."
)

# The French paragraph dropped in as its OWN paragraph, deep inside otherwise
# English paragraphs, so paragraph-level off-majority has something to
# disagree with.
_clean_paragraph_list = CLEAN_PARAGRAPHS.split("\n\n")
_midpoint = len(_clean_paragraph_list) // 2
ENGLISH_WITH_FRENCH_PARAGRAPH = "\n\n".join(
    _clean_paragraph_list[:_midpoint] + [FRENCH_PARAGRAPH] + _clean_paragraph_list[_midpoint:])

# ``DocumentAnalysis.windows()`` floors its window size at 200 words (a
# document-level rule this suite does not control -- see document.py), so a
# section-level test needs a French BLOCK long enough to dominate one whole
# 200-word window, not a single ~50-word paragraph diluted by its English
# neighbours in the same window. Repeating the French paragraph several times
# gives a real French-majority window; English filler on both sides gives
# real English-majority windows to contrast it with.
CODE_SWITCHED_SECTIONS_TEXT = "\n\n".join(
    _clean_paragraph_list + [FRENCH_PARAGRAPH] * 5 + _clean_paragraph_list)

SHORT_AMBIGUOUS = "Ok. No. Go now. Wait. Stop."

MIXED_SCRIPT_TEXT = (
    "The company's real login page is paypal.com, but the phishing email "
    "links to a lookalike domain that uses a Cyrillic а (that character "
    "right there is Cyrillic, not Latin) in place of a Latin a inside the "
    "word pаypal, which is exactly the kind of homoglyph spoof this "
    "suite's script-distribution channel is built to surface as real script "
    "mixing rather than plain single-script English or Russian text."
)

OCR_LIKE_PROSE = CLEAN_PROSE.replace("stood", "st00d").replace(
    "first", "fir5t").replace("home", "h0me").replace("hand", "h4nd")

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
    "flew over Drakmoor's old gate. Khazad-dum was another name entirely, "
    "one the old maps of Aerendyl never mentioned at all."
)

DIALECT_DIALOGUE = (
    "\"I ain't goin' back there, no matter what,\" Jonas said. \"We're "
    "gonna need more rope, and y'all better hurry up about it. I been "
    "thinkin' on it all night, and I reckon we're stuck till mornin'.\" "
    "\"You're worryin' over nothin',\" Ruth said. \"We'll get 'er done "
    "before the sun's up, just you wait.\" Jonas shook his head. \"I'm "
    "tellin' you, somethin' ain't right about this place.\""
) * 3


# ------------------------------------------------------------- config surface

def test_disabled_by_default_via_config(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(CLEAN_PROSE, encoding="utf-8")
    report = grade.analyze(source, base_config)
    ids = [item.metric_id for item in report.results]
    assert not any(mid.startswith("lexical.malformed_") for mid in ids)


def test_enabling_the_suite_produces_findings(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(CLEAN_PROSE * 3, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "malformed_text_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    ids = {item.metric_id for item in report.results}
    assert "lexical.malformed_long_token_rate" in ids
    assert "lexical.malformed_script_distribution_entropy" in ids


def test_features_are_individually_switchable():
    findings_on = _findings(CLEAN_PROSE, config=_all_on())
    findings_off = _findings(CLEAN_PROSE, config=_all_on(token_shape=False))
    assert "lexical.malformed_long_token_rate" in findings_on
    assert "lexical.malformed_long_token_rate" not in findings_off
    # Turning one feature off must not remove any other feature's findings.
    assert "lexical.malformed_script_distribution_entropy" in findings_off


@requires_a_language_detector
def test_each_language_detector_feature_is_independently_switchable():
    all_five = _findings(CLEAN_PROSE, config=_all_on())
    item = all_five["lexical.malformed_document_language_confidence"]
    if item["distribution"] is None:
        pytest.skip("no language-id detector actually available in this environment")
    detectors = item["distribution"]["detectors"]
    disabled_lingua = _findings(CLEAN_PROSE, config=_all_on(language_id_lingua=False))
    remaining_item = disabled_lingua["lexical.malformed_document_language_confidence"]
    remaining = (remaining_item["distribution"] or {}).get("detectors", {})
    assert "lingua" not in remaining
    # Some other detector still answers, as long as one is installed.
    if len(detectors) > 1:
        assert remaining


def test_registry_and_config_defaults_agree():
    """The features map is defined in three places (module, registry, config)
    and rule 6 requires they mirror each other; catch drift here rather than
    at review time."""

    spec = REGISTRY["malformed_text_suite"]
    assert dict(spec.defaults["features"]) == m.DEFAULT_FEATURES
    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    assert config["metrics"]["malformed_text_suite"]["enabled"] is False
    assert config["metrics"]["malformed_text_suite"]["features"] == m.DEFAULT_FEATURES


def test_no_metric_id_overlaps_a_duplicate_mapping():
    """Every overlap-tagged id must name a real, differently-prefixed metric."""

    for metric_id in ("lexical.malformed_fused_word_candidate_rate",
                      "lexical.malformed_dictionary_recognized_rate",
                      "lexical.malformed_low_frequency_token_rate",
                      "lexical.malformed_mixed_alnum_token_rate",
                      "lexical.malformed_token_charclass_entropy",
                      "lexical.malformed_dictionary_disagreement_rate"):
        overlap = m._overlap(metric_id)
        assert overlap is not None, metric_id
        assert overlap["overlaps_existing_metric_id"] != metric_id


# ------------------------------------------------------------------- gating

def test_suite_never_touches_spacy_or_sentence_transformers():
    spec = REGISTRY["malformed_text_suite"]
    assert spec.needs_parse is False
    assert spec.needs_model is False
    assert "spacy" not in spec.requires
    assert "sentence_transformers" not in spec.requires


def test_corpus_profiling_skips_this_suite_when_disabled_in_the_metrics_config(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text(CLEAN_PROSE * 5, encoding="utf-8")
    profile = build_profile([source], built_at="2026-01-01T00:00:00Z",
                            metrics={"malformed_text_suite": {"enabled": False}})
    assert not any(key.startswith("lexical.malformed_") for key in profile["distributions"])


def test_corpus_profiling_with_the_suite_enabled_runs_it(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text(CLEAN_PROSE * 5, encoding="utf-8")
    profile = build_profile(
        [source], built_at="2026-01-01T00:00:00Z",
        metrics={"malformed_text_suite": {"enabled": True,
                                          "features": {**m.DEFAULT_FEATURES,
                                                      "language_id": False}}})
    assert any(key.startswith("lexical.malformed_") for key in profile["distributions"])


# ------------------------------------------------- separation: clean vs broken

@requires_wordfreq
@requires_wordninja
@requires_wordsegment
@requires_symspellpy
def test_fused_word_candidate_rate_separates_clean_from_concatenated():
    clean = _findings(CLEAN_PROSE, config=_all_on())
    broken = _findings(CONCATENATED_PROSE, config=_all_on())
    assert clean["lexical.malformed_fused_word_candidate_rate"]["value"] == 0.0
    assert broken["lexical.malformed_fused_word_candidate_rate"]["value"] > 0.0


@requires_wordfreq
@requires_wordninja
@requires_wordsegment
def test_segmentation_disagreement_fires_on_concatenated_text():
    clean = _findings(CLEAN_PROSE, config=_all_on())
    broken = _findings(CONCATENATED_PROSE, config=_all_on())
    assert clean["lexical.malformed_segmentation_disagreement_rate"]["warning"] is not None
    assert broken["lexical.malformed_avg_segmentation_candidates"]["value"] > 0.0


@requires_wordfreq
def test_split_word_candidate_rate_separates_clean_from_space_inserted():
    clean = _findings(CLEAN_PROSE, config=_all_on())
    broken = _findings(SPACED_PROSE, config=_all_on())
    assert clean["lexical.malformed_split_word_candidate_rate"]["value"] == 0.0
    assert broken["lexical.malformed_split_word_candidate_rate"]["value"] > 0.0


def test_mixed_alnum_token_rate_separates_clean_from_ocr_like():
    clean = _findings(CLEAN_PROSE, config=_all_on(language_id=False))
    broken = _findings(OCR_LIKE_PROSE, config=_all_on(language_id=False))
    assert clean["lexical.malformed_mixed_alnum_token_rate"]["value"] == 0.0
    assert broken["lexical.malformed_mixed_alnum_token_rate"]["value"] > 0.0


def test_script_distribution_entropy_separates_clean_from_mixed_script():
    clean = _findings(CLEAN_PROSE, config=_all_on(language_id=False))
    mixed = _findings(MIXED_SCRIPT_TEXT, config=_all_on(language_id=False))
    assert clean["lexical.malformed_script_distribution_entropy"]["value"] == 0.0
    assert mixed["lexical.malformed_script_distribution_entropy"]["value"] > 0.0
    scripts = {e["script"] for e in mixed["lexical.malformed_script_distribution_entropy"]["evidence"]}
    assert "Cyrillic" in scripts and "Latin" in scripts


def test_repeated_symbol_token_rate_fires_on_repeated_punctuation():
    clean = _findings(CLEAN_PROSE, config=_all_on(language_id=False))
    shouty = _findings(CLEAN_PROSE.replace("home.", "home!!!???"), config=_all_on(language_id=False))
    assert clean["lexical.malformed_repeated_symbol_token_rate"]["value"] == 0.0
    assert shouty["lexical.malformed_repeated_symbol_token_rate"]["value"] > 0.0


def test_long_token_rate_fires_on_a_deliberately_long_token():
    long_text = CLEAN_PROSE + " " + ("x" * 40) + " floated past on the current."
    clean = _findings(CLEAN_PROSE, config=_all_on(language_id=False))
    longer = _findings(long_text, config=_all_on(language_id=False))
    assert clean["lexical.malformed_long_token_rate"]["value"] == 0.0
    assert longer["lexical.malformed_long_token_rate"]["value"] > 0.0


@requires_wordfreq
def test_low_frequency_token_rate_rises_with_rare_tokens():
    rare_text = CLEAN_PROSE + " " + " ".join(["xqzwarp", "phlembotron", "krivastule"] * 3)
    clean = _findings(CLEAN_PROSE, config=_all_on(language_id=False))
    rare = _findings(rare_text, config=_all_on(language_id=False))
    assert (rare["lexical.malformed_low_frequency_token_rate"]["value"]
           > clean["lexical.malformed_low_frequency_token_rate"]["value"])


@requires_wordfreq
@requires_symspellpy
def test_dictionary_disagreement_rate_is_a_real_number_when_it_has_candidates():
    text = CLEAN_PROSE + " " + " ".join(["xqzwarptron", "phlembotronic"] * 5)
    found = _findings(text, config=_all_on(language_id=False))
    item = found["lexical.malformed_dictionary_disagreement_rate"]
    assert item["value"] is not None
    assert item["distribution"]["candidates_compared"] > 0


# ------------------------------------------------------- language-ID channels

@requires_a_language_detector
def test_code_switch_and_offmajority_rates_separate_clean_from_code_switched():
    clean = _findings(CLEAN_PARAGRAPHS, config=_all_on(
        language_id_run_length=20, language_id_max_runs=5))
    switched = _findings(ENGLISH_WITH_FRENCH_PARAGRAPH, config=_all_on(
        language_id_run_length=20, language_id_max_runs=5))
    assert clean["lexical.malformed_code_switch_rate"]["value"] == 0.0
    assert clean["lexical.malformed_offmajority_sentence_rate"]["value"] == 0.0
    assert switched["lexical.malformed_code_switch_rate"]["value"] > 0.0
    assert switched["lexical.malformed_offmajority_sentence_rate"]["value"] > 0.0
    assert switched["lexical.malformed_paragraph_offmajority_rate"]["value"] > 0.0


@requires_a_language_detector
def test_section_language_entropy_separates_clean_from_code_switched():
    clean = _findings(CLEAN_PARAGRAPHS, config=_all_on(language_id_section_words=200))
    switched = _findings(CODE_SWITCHED_SECTIONS_TEXT,
                         config=_all_on(language_id_section_words=200))
    assert clean["lexical.malformed_section_language_entropy"]["value"] == 0.0
    assert switched["lexical.malformed_section_language_entropy"]["value"] > 0.0


@requires_a_language_detector
def test_very_short_ambiguous_strings_report_insufficient_data_not_a_guess():
    found = _findings(SHORT_AMBIGUOUS, config=_all_on())
    item = found["lexical.malformed_sentence_language_confidence_median"]
    assert item["value"] is None
    assert item["warning"] is not None


@requires_a_language_detector
def test_fantasy_names_are_not_flagged_as_wrong_language_sentences():
    """Invented fantasy proper nouns inside otherwise ordinary English prose
    must not read as code-switching or off-majority-language sentences."""

    found = _findings(FANTASY_PROSE, config=_all_on(
        language_id_run_length=30, language_id_max_runs=3))
    assert found["lexical.malformed_offmajority_sentence_rate"]["value"] == 0.0
    assert found["lexical.malformed_code_switch_rate"]["value"] == 0.0
    confidence = found["lexical.malformed_sentence_language_confidence_median"]["value"]
    assert confidence is not None and confidence > 40.0


@requires_a_language_detector
def test_written_dialect_is_not_flagged_as_wrong_language():
    found = _findings(DIALECT_DIALOGUE, config=_all_on(
        language_id_run_length=30, language_id_max_runs=3))
    assert found["lexical.malformed_offmajority_sentence_rate"]["value"] == 0.0
    assert found["lexical.malformed_code_switch_rate"]["value"] == 0.0


@requires_a_language_detector
def test_detector_disagreement_is_undefined_with_only_one_detector():
    found = _findings(CLEAN_PROSE, config=_all_on(
        language_id_langid=False, language_id_langdetect=False,
        language_id_fasttext=False, language_id_cld3=False))
    # Exactly one detector left on (Lingua); if it happens to be unavailable
    # too, this degrades to "no detector could run", which is also a valid,
    # honest answer -- either way disagreement must not fabricate a number.
    item = found["lexical.malformed_detector_disagreement_rate"]
    assert item["value"] is None


def test_no_detector_enabled_reports_unavailable_not_a_crash():
    found = _findings(CLEAN_PROSE, config=_all_on(
        language_id_lingua=False, language_id_langid=False, language_id_langdetect=False,
        language_id_fasttext=False, language_id_cld3=False))
    item = found["lexical.malformed_document_language_confidence"]
    assert item["value"] is None
    assert "no language-id detector feature is enabled" in item["warning"]


# --------------------------------------------------------------- degeneracy

@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_survives_a_degenerate_document(text):
    findings = m.measure(_analysis(text), config=_all_on())
    assert findings  # every feature was on; must still return a full set
    for item in findings:
        assert "metric_id" in item


def test_empty_document_reports_no_words_everywhere():
    findings = _findings("", config=_all_on())
    for metric_id, item in findings.items():
        assert item["value"] is None, metric_id
        assert item["warning"]


# -------------------------------------------------------------- degradation

def test_a_missing_optional_package_degrades_this_suite_only(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        findings = _findings(CLEAN_PROSE, config=_all_on())
        assert findings
        # Dependency-free groups still work.
        assert findings["lexical.malformed_long_token_rate"]["value"] == 0.0
        assert findings["lexical.malformed_script_distribution_entropy"]["value"] is not None
        # Package-dependent groups say why, instead of crashing.
        assert findings["lexical.malformed_dictionary_recognized_rate"]["value"] is None
        assert findings["lexical.malformed_document_language_confidence"]["value"] is None
    finally:
        optional.reset_cache()


def test_every_registered_id_runs_clean_through_grade_py(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(CLEAN_PROSE * 3, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "malformed_text_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]



def test_detector_confidences_are_reproducible_to_the_bit():
    # Lingua's raw confidence varies in its last bits between identical calls,
    # which made two builds of one corpus profile differ byte for byte.
    from textgrader.metrics import malformed_text_suite as m

    text = "One sentence here. Another sentence follows it, and then a third one."
    for name in m.DETECTOR_ORDER:
        first, _ = m._detect_one(name, text, {})
        if first is None:
            continue
        for _ in range(5):
            again, _ = m._detect_one(name, text, {})
            assert again["confidence"] == first["confidence"]
