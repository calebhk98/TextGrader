"""Tests for the randomness/gibberish/compression/information-theory suite.

The corruption-ordering tests are the heart of this file: the whole point of
keeping every channel raw (see the module docstring) is that different kinds
of noise should show up on different channels, so these tests assert
*ordering* between real prose and several controlled corruptions rather than
pinning exact numbers, which would make the suite brittle for no benefit.
"""

from __future__ import annotations

import random
import string

import pytest

import grade
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY
from textgrader.metrics import randomness_suite as rs
from textgrader import optional


# ------------------------------------------------------------- synthetic text

_SUBJECTS = ["She", "He", "Alice", "Bob", "The old woman", "The young man", "The child",
             "The stranger"]
_VERBS = ["walked", "looked", "moved", "stepped", "wandered", "hurried", "crept"]
_ADVERBS = ["quietly", "slowly", "carefully", "silently", "nervously", "calmly"]
_PREPS = ["through", "around", "into", "toward", "past", "beside"]
_OBJECTS = ["the old house", "the quiet room", "the dark hallway", "the empty garden",
            "the narrow street", "the abandoned church", "the wooden door"]
_CONNECTORS = ["and then paused", "before turning back", "without a word",
               "while the wind blew", "as the light faded", "and said nothing"]


def _sentence(rng: random.Random) -> str:
    """A real (if formulaic) English sentence, not a bag-of-words salad.

    A word-order corruption test needs word order that carries information in
    the first place. Independently-sampled "word salad" (every word drawn
    uniformly from a flat vocabulary) has almost no bigram structure to
    destroy, so shuffling it barely changes an n-gram model's surprise -
    which was the first version of this fixture, and which made the
    corruption-baseline assertions below fail for the right reason: they were
    honestly measuring a text with no real syntax. Slot-filled templates give
    genuine, English-like local structure (articles precede nouns, adjectives
    precede the nouns they modify) while staying fully deterministic.
    """

    return (f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_ADVERBS)} "
            f"{rng.choice(_PREPS)} {rng.choice(_OBJECTS)}, {rng.choice(_CONNECTORS)}.")


def _real_prose(seed: int = 1, paragraphs: int = 90) -> str:
    rng = random.Random(seed)
    blocks = [" ".join(_sentence(rng) for _ in range(rng.randint(2, 5)))
              for _ in range(paragraphs)]
    return "\n\n".join(blocks)


def _shuffled_words(text: str, seed: int = 2) -> str:
    words = text.split()
    random.Random(seed).shuffle(words)
    return " ".join(words)


def _shuffled_chars(text: str, seed: int = 3) -> str:
    chars = list(text)
    random.Random(seed).shuffle(chars)
    return "".join(chars)


def _random_letters(length: int, seed: int = 4) -> str:
    rng = random.Random(seed)
    out, count = [], 0
    while count < length:
        word = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 9)))
        out.append(word)
        count += len(word) + 1
    # Punctuate every dozen words so the sentence segmenter has something to
    # split on; without it the whole thing is one "sentence".
    for index in range(11, len(out), 12):
        out[index] += "."
    return " ".join(out)


def _repeated_phrase(length: int) -> str:
    phrase = "the cat sat on the mat and the dog ran away. "
    return (phrase * (length // len(phrase) + 1))[:length]


REAL_PROSE = _real_prose()
CORRUPTIONS = {
    "real_prose": REAL_PROSE,
    "shuffled_words": _shuffled_words(REAL_PROSE),
    "shuffled_chars": _shuffled_chars(REAL_PROSE),
    "random_letters": _random_letters(len(REAL_PROSE)),
    "repeated_phrase": _repeated_phrase(len(REAL_PROSE)),
}


def _measure(text: str, **overrides) -> dict[str, dict]:
    config = {**rs.DEFAULTS, **overrides}
    analysis = DocumentAnalysis.from_text(text)
    return {item["metric_id"]: item for item in rs.measure(analysis, config=config)}


@pytest.fixture(scope="module")
def all_findings() -> dict[str, dict[str, dict]]:
    return {name: _measure(text) for name, text in CORRUPTIONS.items()}


# --------------------------------------------------------------- plain wiring

def test_registered_under_the_lexical_family_with_a_stable_module():
    spec = REGISTRY["randomness_suite"]
    assert spec.module == "randomness_suite"
    assert spec.family == "lexical"
    assert spec.cost == "moderate"
    assert "features" in spec.defaults


def test_off_by_default(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(REAL_PROSE, encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results
                if item.metric_id.startswith("style.randomness_")]


def test_enabled_emits_ids_under_the_required_prefix(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(REAL_PROSE, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "randomness_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    ids = [item.metric_id for item in report.results if item.metric_id.startswith("style.randomness_")]
    assert ids, "enabling the suite produced no findings at all"
    assert all(mid.startswith("style.randomness_") for mid in ids)
    assert len(ids) == len(set(ids)), "duplicate metric ids"


def test_config_json_documents_the_whole_option_surface():
    """Rule: every option in MetricSpec.defaults must also be in config.json."""

    import json
    from pathlib import Path

    config = json.loads(Path(__file__).resolve().parents[1].joinpath("config.json").read_text())
    configured = config["metrics"]["randomness_suite"]
    defaults = REGISTRY["randomness_suite"].defaults
    assert configured["enabled"] is False
    for key in defaults:
        assert key in configured, f"{key!r} is in MetricSpec.defaults but missing from config.json"
    for key in defaults["features"]:
        assert key in configured["features"], f"features.{key!r} missing from config.json"


# ------------------------------------------------------------- feature toggles

def test_features_merge_partially_without_losing_other_defaults():
    merged = rs._features({"features": {"compression": False}})
    assert merged["compression"] is False
    assert merged["char_entropy"] is True  # not mentioned, must keep its default
    assert merged["pos_dependency"] is False  # default itself


def test_disabling_a_feature_removes_exactly_its_metrics():
    with_compression = _measure(REAL_PROSE)
    without_compression = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "compression": False})
    compression_ids = {mid for mid in with_compression if "compression" in mid or mid.startswith(
        f"{rs.ID}ncd_")}
    assert compression_ids, "expected at least one compression/NCD id to exist"
    assert not (compression_ids & set(without_compression))
    # Something from an unrelated group must still be present.
    assert f"{rs.ID}char_entropy" in without_compression


def test_pos_dependency_is_off_by_default_and_opt_in():
    default_findings = _measure(REAL_PROSE)
    assert f"{rs.ID}pos_ngram_cross_entropy" not in default_findings

    opted_in = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "pos_dependency": True})
    if optional.have("spacy"):
        item = opted_in[f"{rs.ID}pos_ngram_cross_entropy"]
        assert item["value"] is not None
    else:
        item = opted_in[f"{rs.ID}pos_ngram_cross_entropy"]
        assert item["value"] is None and item["warning"]


# ----------------------------------------------------------- graceful failure

def test_missing_optional_compressors_degrade_without_crashing(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        findings = _measure(REAL_PROSE)
        # zlib/gzip/bz2/lzma are stdlib and must still produce a real number.
        assert findings[f"{rs.ID}compression_ratio_zlib"]["value"] is not None
        # The optional compressors must degrade to a visible reason, not a crash.
        for name in ("zstd", "brotli", "lz4", "snappy", "ppmd"):
            item = findings[f"{rs.ID}compression_ratio_{name}"]
            assert item["value"] is None
            assert item["warning"]
        # wordfreq-backed lexical channel degrades the same way.
        unknown = findings[f"{rs.ID}unknown_word_rate"]
        assert unknown["value"] is None and unknown["warning"]
    finally:
        optional.reset_cache()


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC", "word " * 5])
def test_degenerate_documents_never_crash(text):
    findings = _measure(text, features={**rs.DEFAULT_FEATURES, "pos_dependency": True})
    assert findings
    for item in findings.values():
        assert isinstance(item["metric_id"], str) and item["metric_id"].startswith(rs.ID)


# -------------------------------------------------------- sample-size honesty

def test_entropy_and_lm_findings_are_sample_size_sensitive(all_findings):
    findings = all_findings["real_prose"]
    for metric_id in (f"{rs.ID}char_entropy", f"{rs.ID}token_type_entropy",
                      f"{rs.ID}char_ngram_cross_entropy", f"{rs.ID}word_ngram_cross_entropy",
                      f"{rs.ID}compression_ratio_zlib", f"{rs.ID}lz_complexity"):
        item = findings[metric_id]
        assert item["sample_size_sensitive"] is True, metric_id
        assert item["sample_size"] is not None and item["sample_size"] > 0, metric_id


def test_settings_are_recorded_on_compression_findings(all_findings):
    item = all_findings["real_prose"][f"{rs.ID}compression_ratio_zlib"]
    dist = item["distribution"]
    assert dist["algorithm"] == "zlib"
    assert dist["level"] == rs.DEFAULTS["compression_level"]
    assert dist["block_chars"] > 0


# ------------------------------------------------------- corruption ordering

def test_character_level_channels_rank_repeated_lowest_and_random_highest(all_findings):
    """Repeated template text should be the easiest to predict/compress;
    shuffled characters and random letters should be the hardest."""

    def value(name, metric_id):
        return all_findings[name][metric_id]["value"]

    for metric_id in (f"{rs.ID}char_ngram_cross_entropy",):
        repeated = value("repeated_phrase", metric_id)
        real = value("real_prose", metric_id)
        shuffled_chars = value("shuffled_chars", metric_id)
        random_letters = value("random_letters", metric_id)
        assert repeated < real < shuffled_chars
        assert repeated < real < random_letters

    for metric_id in (f"{rs.ID}compression_ratio_zlib",):
        repeated = value("repeated_phrase", metric_id)
        real = value("real_prose", metric_id)
        shuffled_chars = value("shuffled_chars", metric_id)
        random_letters = value("random_letters", metric_id)
        # Higher ratio = more compressible = less random.
        assert repeated > real > shuffled_chars
        assert repeated > real > random_letters

    for metric_id in (f"{rs.ID}lz_complexity_normalized",):
        repeated = value("repeated_phrase", metric_id)
        shuffled_chars = value("shuffled_chars", metric_id)
        random_letters = value("random_letters", metric_id)
        assert repeated < shuffled_chars
        assert repeated < random_letters


def test_word_identity_channel_separates_real_words_from_random_letters(all_findings):
    """Random letters mint an almost entirely new vocabulary every occurrence;
    real prose and word-shuffled real prose share the same, small vocabulary."""

    def value(name, metric_id):
        return all_findings[name][metric_id]["value"]

    metric_id = f"{rs.ID}token_type_entropy"
    real = value("real_prose", metric_id)
    shuffled_words = value("shuffled_words", metric_id)
    random_letters = value("random_letters", metric_id)
    assert real == pytest.approx(shuffled_words, rel=0.05)
    assert random_letters > real * 1.5


def test_unknown_word_rate_flags_random_letters_not_real_prose(all_findings):
    if not optional.have("wordfreq"):
        pytest.skip("wordfreq not installed")
    real = all_findings["real_prose"][f"{rs.ID}unknown_word_rate"]["value"]
    random_letters = all_findings["random_letters"][f"{rs.ID}unknown_word_rate"]["value"]
    assert real < 5.0
    assert random_letters > 80.0


def test_consonant_cluster_rate_is_higher_for_random_letters(all_findings):
    real = all_findings["real_prose"][f"{rs.ID}consonant_cluster_rate"]["value"]
    random_letters = all_findings["random_letters"][f"{rs.ID}consonant_cluster_rate"]["value"]
    assert random_letters > real


def test_letter_frequency_diverges_more_for_random_letters_than_real_prose(all_findings):
    real = all_findings["real_prose"][f"{rs.ID}letter_frequency_divergence"]["value"]
    random_letters = all_findings["random_letters"][f"{rs.ID}letter_frequency_divergence"]["value"]
    assert random_letters > real


# ----------------------------------------------------- corruption baselines

def test_word_and_char_shuffle_baselines_increase_surprisal_more_than_sentence_shuffle():
    """The acceptance criterion in one test: word/char shuffling should visibly
    degrade this document's own n-gram model; sentence shuffling, scored with a
    local character model, should barely move it. Disagreement between
    channels is the point, not a bug."""

    findings = _measure(REAL_PROSE)
    word_ratio = findings[f"{rs.ID}word_shuffle_ratio"]["value"]
    char_ratio = findings[f"{rs.ID}char_shuffle_ratio"]["value"]
    sentence_ratio = findings[f"{rs.ID}sentence_shuffle_ratio"]["value"]
    assert word_ratio is not None and char_ratio is not None and sentence_ratio is not None
    assert word_ratio > 1.1
    assert char_ratio > 1.1
    assert abs(sentence_ratio - 1.0) < abs(word_ratio - 1.0)
    assert abs(sentence_ratio - 1.0) < abs(char_ratio - 1.0)


def test_corruption_ratios_are_deterministic_given_the_same_seed():
    first = _measure(REAL_PROSE)
    second = _measure(REAL_PROSE)
    for metric_id in (f"{rs.ID}word_shuffle_ratio", f"{rs.ID}char_shuffle_ratio",
                      f"{rs.ID}sentence_shuffle_ratio"):
        assert first[metric_id]["value"] == second[metric_id]["value"]


def test_ncd_channels_are_bounded_and_present():
    findings = _measure(REAL_PROSE)
    for metric_id in (f"{rs.ID}ncd_word_shuffle", f"{rs.ID}ncd_char_shuffle",
                      f"{rs.ID}ncd_sentence_shuffle"):
        value = findings[metric_id]["value"]
        assert value is not None
        assert 0.0 <= value <= 1.5  # NCD can exceed 1 slightly with small blocks


# --------------------------------------------------------- registry contract

def test_runs_without_raising_via_grade(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "randomness_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    from textgrader.results import StatusType
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
