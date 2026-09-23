"""Tests for the randomness/gibberish/compression/information-theory suite.

The corruption-ordering tests are the heart of this file: the whole point of
keeping every channel raw (see the module docstring) is that different kinds
of noise should show up on different channels, so these tests assert
*ordering* between real prose and several controlled corruptions rather than
pinning exact numbers, which would make the suite brittle for no benefit.
"""

from __future__ import annotations

import os
import random
import shutil
import string
import subprocess
from pathlib import Path

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
    findings = _measure(text, features={**rs.DEFAULT_FEATURES, "pos_dependency": True,
                                        "kenlm_language_model": True,
                                        "neural_language_model": True,
                                        "textdescriptives_cross_check": True})
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


# ------------------------------------------------ newly-available compressors
#
# zstandard, brotli, lz4 and pyppmd were installed but never exercised before
# this task: every one of these had only ever produced "unavailable". These
# tests run them for real and pin the bugs that surfaced the first time they
# did (see the module's ``_effective_setting``/``_library_version`` docstrings).

NEW_COMPRESSORS = ("zstd", "brotli", "lz4", "ppmd")


@pytest.mark.parametrize("name", NEW_COMPRESSORS)
def test_newly_available_compressors_produce_a_sane_ratio(name, all_findings):
    if not optional.have(rs._COMPRESSOR_MODULE_NAME[name]):
        pytest.skip(f"{name} not installed in this environment")
    item = all_findings["real_prose"][f"{rs.ID}compression_ratio_{name}"]
    zlib_item = all_findings["real_prose"][f"{rs.ID}compression_ratio_zlib"]
    assert item["value"] is not None and item["warning"] is None
    # Same ballpark as zlib on the same text, not an order of magnitude off in
    # either direction: a real bug here (e.g. compressing zero bytes, or
    # scoring compressed-and-then-recompressed data) shows up as a wildly
    # wrong ratio long before anyone reads the number closely.
    assert 0.2 * zlib_item["value"] < item["value"] < 6 * zlib_item["value"]


def test_lz4_library_version_is_reported():
    """Regression test: require("lz4") returns the lz4.frame submodule, which
    carries no __version__ of its own - only the top-level lz4 package does.
    _library_version silently returned None for every lz4 finding before this
    was special-cased."""

    if not optional.have("lz4"):
        pytest.skip("lz4 not installed in this environment")
    assert rs._library_version("lz4") is not None


@pytest.mark.parametrize("name,low,high", [("zstd", 1, 22), ("ppmd", 2, 16),
                                           ("bz2", 1, 9), ("lzma", 0, 9), ("brotli", 0, 11)])
def test_effective_setting_stays_inside_each_codecs_valid_range(name, low, high):
    for level in (-5, 0, 1, 6, 9, 16, 22, 30, 200):
        _, value = rs._effective_setting(name, level)
        assert low <= value <= high, (name, level, value)


def test_out_of_range_compression_level_still_compresses_instead_of_degrading():
    """Regression test: zstandard raises ValueError above level 22, and
    pyppmd's default variant raises for a negative max_order. Both used to be
    swallowed by _compress's blanket except-and-degrade, silently turning "the
    user configured an unusual level" into "zstd/ppmd is unavailable"."""

    data = b"the quick brown fox jumps over the lazy dog. " * 50
    if optional.have("zstandard"):
        compressed, error = rs._compress("zstd", data, 99)
        assert compressed is not None and error is None
    if optional.have("pyppmd"):
        compressed, error = rs._compress("ppmd", data, -3)
        assert compressed is not None and error is None


def test_compression_ratio_finding_records_the_effective_setting(all_findings):
    item = all_findings["real_prose"][f"{rs.ID}compression_ratio_ppmd"]
    if item["value"] is None:
        pytest.skip("pyppmd not installed in this environment")
    assert item["distribution"]["effective_setting"] == {"max_order": rs.DEFAULTS["compression_level"]}


# --------------------------------------------------------------------- snappy
#
# snappy was previously kept unavailable on purpose ("proving the degradation
# path works") even once libsnappy-dev made it importable; the user rejected
# that reasoning ("We don't need to prove degradation"), so it is wired up as
# a real channel like every other compressor. It is NOT held to the same
# "same ballpark as zlib" bound as the other newly-available compressors
# above: Snappy trades compression ratio for speed by design, so a
# meaningfully weaker ratio than zlib on the same text is its correct,
# honest behaviour, not a bug.

def test_snappy_is_a_real_channel_not_a_permanent_unavailable(all_findings):
    if not optional.have("snappy"):
        pytest.skip("python-snappy not installed in this environment")
    item = all_findings["real_prose"][f"{rs.ID}compression_ratio_snappy"]
    zlib_item = all_findings["real_prose"][f"{rs.ID}compression_ratio_zlib"]
    assert item["value"] is not None and item["warning"] is None
    assert item["value"] > 1.0  # a genuine compression, not a pass-through bug
    # Loose bound: catches a real bug (compressing zero bytes, double
    # compression) without asserting snappy matches zlib's strength, which it
    # is not designed to.
    assert 0.05 * zlib_item["value"] < item["value"] < 3 * zlib_item["value"]


def test_snappy_has_no_compression_level_and_says_so():
    """python-snappy's compress() takes no level/quality parameter at all;
    reporting the shared ``compression_level`` knob as though it had been
    used would misrepresent what actually happened."""

    setting_name, setting_value = rs._effective_setting("snappy", 6)
    assert setting_value is None
    if optional.have("snappy"):
        # And the finding itself must say the same thing, not just the helper.
        item = _measure(REAL_PROSE)[f"{rs.ID}compression_ratio_snappy"]
        assert item["distribution"]["effective_setting"] == {setting_name: None}


def test_snappy_ranks_repeated_lowest_entropy_and_random_highest(all_findings):
    if not optional.have("snappy"):
        pytest.skip("python-snappy not installed in this environment")

    def value(name):
        return all_findings[name][f"{rs.ID}compression_ratio_snappy"]["value"]

    repeated, real = value("repeated_phrase"), value("real_prose")
    shuffled_chars, random_letters = value("shuffled_chars"), value("random_letters")
    # Higher ratio = more compressible = less random, same reading as zlib's
    # channel above.
    assert repeated > real > shuffled_chars
    assert repeated > real > random_letters


# ------------------------------------------------------- corruption ordering

def test_character_level_channels_rank_repeated_lowest_and_random_highest(all_findings):
    """Repeated template text should be the easiest to predict/compress;
    shuffled characters and random letters should be the hardest."""

    def value(name, metric_id):
        return all_findings[name][metric_id]["value"]

    metric_ids = [f"{rs.ID}char_ngram_cross_entropy"]
    if optional.have("pyppmd"):
        metric_ids.append(f"{rs.ID}ppm_cross_entropy")
    for metric_id in metric_ids:
        repeated = value("repeated_phrase", metric_id)
        real = value("real_prose", metric_id)
        shuffled_chars = value("shuffled_chars", metric_id)
        random_letters = value("random_letters", metric_id)
        assert repeated < real < shuffled_chars, metric_id
        assert repeated < real < random_letters, metric_id

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


# ---------------------------------------------------- letter-bigram divergence

def test_bigram_reference_table_is_derived_from_the_shipped_corpus_profile():
    reference, total, reason = rs._bigram_reference_table()
    assert reason is None and reference is not None
    assert total > 0
    assert abs(sum(reference.values()) - 1.0) < 1e-9
    # The real top English bigrams, if this table means what it claims to.
    top = sorted(reference, key=reference.get, reverse=True)[:10]
    assert {"th", "he", "in", "er", "an"} & set(top)


def test_bigram_reference_table_is_cached():
    first, _, _ = rs._bigram_reference_table()
    second, _, _ = rs._bigram_reference_table()
    assert first is second


def test_bigram_reference_table_degrades_visibly_when_the_profile_is_unreadable(monkeypatch, tmp_path):
    missing = tmp_path / "does-not-exist.json"
    monkeypatch.setattr(rs, "PROSE_REFERENCE", missing)
    rs._reset_randomness_suite_caches()
    try:
        reference, total, reason = rs._bigram_reference_table()
        assert reference is None and total == 0
        assert reason and "does-not-exist.json" in reason
        findings = _measure(REAL_PROSE)
        item = findings[f"{rs.ID}letter_bigram_divergence"]
        assert item["value"] is None and item["warning"]
    finally:
        rs._reset_randomness_suite_caches()


def test_letter_bigram_divergence_is_unchanged_by_word_order(all_findings):
    """Shuffling word order cannot change which letters follow which inside a
    word, so this channel (unlike letter_frequency_divergence's cousin
    channels that look at sequence, not composition) should not move at
    all when only word order is corrupted."""

    real = all_findings["real_prose"][f"{rs.ID}letter_bigram_divergence"]["value"]
    shuffled_words = all_findings["shuffled_words"][f"{rs.ID}letter_bigram_divergence"]["value"]
    assert real == pytest.approx(shuffled_words, rel=1e-9)


def test_letter_bigram_divergence_is_higher_for_random_letters_than_real_prose(all_findings):
    real = all_findings["real_prose"][f"{rs.ID}letter_bigram_divergence"]["value"]
    random_letters = all_findings["random_letters"][f"{rs.ID}letter_bigram_divergence"]["value"]
    assert random_letters > real


def test_letter_bigram_divergence_records_its_source(all_findings):
    item = all_findings["real_prose"][f"{rs.ID}letter_bigram_divergence"]
    assert "prose_reference.json" in item["distribution"]["reference"]


def test_letter_bigram_divergence_is_off_by_a_dedicated_toggle():
    with_it = _measure(REAL_PROSE)
    without_it = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "letter_bigram_divergence": False})
    assert f"{rs.ID}letter_bigram_divergence" in with_it
    assert f"{rs.ID}letter_bigram_divergence" not in without_it
    # The single-letter channel is a different toggle and must still be there.
    assert f"{rs.ID}letter_frequency_divergence" in without_it


# ------------------------------------------------------- PPM language model

def test_ppm_cross_entropy_is_off_by_a_dedicated_toggle():
    with_it = _measure(REAL_PROSE)
    without_it = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "ppm_language_model": False})
    if optional.have("pyppmd"):
        assert with_it[f"{rs.ID}ppm_cross_entropy"]["value"] is not None
    assert f"{rs.ID}ppm_cross_entropy" not in without_it


def test_ppm_cross_entropy_degrades_visibly_without_pyppmd(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "pyppmd")
    optional.reset_cache()
    try:
        findings = _measure(REAL_PROSE)
        item = findings[f"{rs.ID}ppm_cross_entropy"]
        assert item["value"] is None and item["warning"]
    finally:
        optional.reset_cache()


def test_ppm_cross_entropy_records_its_method_and_order(all_findings):
    item = all_findings["real_prose"][f"{rs.ID}ppm_cross_entropy"]
    if item["value"] is None:
        pytest.skip("pyppmd not installed in this environment")
    assert item["distribution"]["algorithm"] == "ppmd"
    assert item["distribution"]["max_order"] == rs.DEFAULTS["ppm_max_order"]
    assert "compress(train" in item["distribution"]["method"]


def test_ppm_cross_entropy_is_distinct_from_the_compression_ratio_channel(all_findings):
    """A real predictive-model measurement, not the same number relabelled."""

    findings = all_findings["real_prose"]
    ppm = findings[f"{rs.ID}ppm_cross_entropy"]
    ratio = findings[f"{rs.ID}compression_ratio_ppmd"]
    if ppm["value"] is None or ratio["value"] is None:
        pytest.skip("pyppmd not installed in this environment")
    assert ppm["unit"] == "bits/char"
    assert ratio["unit"] == "ratio"


# ------------------------------------------------------- KenLM language model
#
# kenlm (the query-time Python bindings) is installed in this environment,
# and so is a compiled lmplz trainer - but never assumed to be on PATH, since
# pip cannot install it and a build toolchain's own temporary path is not
# durable or portable to another install. _discover_lmplz_for_tests looks on
# PATH first and only falls back to the one specific location this task's
# own environment is known to have built it in, purely so this suite's own
# test run exercises the real training path when it can; every test below
# still skips cleanly when neither is found.

def _discover_lmplz_for_tests() -> str | None:
    found = shutil.which("lmplz")
    if found:
        return found
    fallback = Path("/tmp/kenlm-src/build/bin/lmplz")
    return str(fallback) if fallback.is_file() else None


_KENLM_LMPLZ_PATH = _discover_lmplz_for_tests()
_HAVE_KENLM = optional.have("kenlm") and _KENLM_LMPLZ_PATH is not None


def test_kenlm_language_model_is_off_by_default():
    assert rs.DEFAULT_FEATURES["kenlm_language_model"] is False
    findings = _measure(REAL_PROSE)
    assert f"{rs.ID}kenlm_cross_entropy" not in findings


def test_kenlm_language_model_spawns_no_subprocess_under_default_config(monkeypatch):
    """The hard requirement: measuring with MetricSpec.defaults must never
    even ask optional.require for kenlm, let alone run lmplz - so corpus
    profiling (which uses exactly these defaults) cannot spawn a training
    subprocess no matter what is installed."""

    requested: list[str] = []
    real_require = optional.require

    def _tracking_require(name):
        requested.append(name)
        return real_require(name)

    ran: list[list[str]] = []
    real_run = subprocess.run

    def _tracking_run(command, *args, **kwargs):
        ran.append(list(command))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(rs, "require", _tracking_require)
    monkeypatch.setattr(rs.subprocess, "run", _tracking_run)
    _measure(REAL_PROSE)  # rs.DEFAULTS: kenlm_language_model is False
    assert "kenlm" not in requested
    assert not ran


def test_kenlm_language_model_degrades_visibly_without_the_kenlm_package(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "kenlm")
    optional.reset_cache()
    try:
        findings = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "kenlm_language_model": True})
        item = findings[f"{rs.ID}kenlm_cross_entropy"]
        assert item["value"] is None
        assert "pip install kenlm" in item["warning"] or "kenlm" in item["warning"]
    finally:
        optional.reset_cache()


def test_kenlm_language_model_reports_which_build_step_is_missing_when_lmplz_is_absent(monkeypatch):
    """The honesty constraint: 'lmplz not found' must name the actual missing
    piece (a compiled trainer pip cannot provide) and where it looked, not a
    bare 'kenlm unavailable' that would send someone to a pip install that
    cannot fix this."""

    if not optional.have("kenlm"):
        pytest.skip("kenlm python bindings not installed in this environment")
    monkeypatch.setattr(rs.shutil, "which", lambda name: None)
    findings = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "kenlm_language_model": True},
                        kenlm_lmplz_path="")
    item = findings[f"{rs.ID}kenlm_cross_entropy"]
    assert item["value"] is None
    assert "lmplz" in item["warning"]
    assert "build" in item["warning"].lower() or "compil" in item["warning"].lower()
    assert "pip install kenlm" in item["warning"]


def test_find_lmplz_binary_prefers_a_configured_path_over_path(tmp_path, monkeypatch):
    fake = tmp_path / "lmplz"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(rs.shutil, "which", lambda name: "/should/not/be/used")
    found, checked = rs._find_lmplz_binary(str(fake))
    assert found == str(fake)
    assert checked == [str(fake)]  # PATH is never even consulted once the configured path works


def test_find_lmplz_binary_falls_back_to_path_when_unconfigured(monkeypatch):
    monkeypatch.setattr(rs.shutil, "which", lambda name: "/usr/local/bin/lmplz")
    found, checked = rs._find_lmplz_binary(None)
    assert found == "/usr/local/bin/lmplz"
    assert checked == ["PATH"]


def test_find_lmplz_binary_reports_none_when_nowhere_is_found(monkeypatch):
    monkeypatch.setattr(rs.shutil, "which", lambda name: None)
    found, checked = rs._find_lmplz_binary("/does/not/exist")
    assert found is None
    assert checked == ["/does/not/exist", "PATH"]


@pytest.mark.skipif(not _HAVE_KENLM, reason="kenlm python bindings or a compiled lmplz not available")
def test_kenlm_language_model_trains_and_scores_real_text_when_enabled():
    findings = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "kenlm_language_model": True},
                        kenlm_lmplz_path=_KENLM_LMPLZ_PATH)
    item = findings[f"{rs.ID}kenlm_cross_entropy"]
    if item["value"] is None:
        pytest.skip(f"kenlm unavailable in this environment: {item['warning']}")
    assert item["value"] > 0
    assert item["unit"] == "bits/word"
    assert item["sample_size_sensitive"] is True
    assert item["distribution"]["algorithm"] == "kenlm"
    assert item["distribution"]["smoothing"] == "modified Kneser-Ney"
    assert item["distribution"]["order"] == rs.DEFAULTS["kenlm_order"]


@pytest.mark.skipif(not _HAVE_KENLM, reason="kenlm python bindings or a compiled lmplz not available")
def test_kenlm_cross_entropy_ranks_repeated_lowest_and_random_letters_highest():
    """Same reading as the from-scratch word/char n-gram channels: a real,
    Kneser-Ney-smoothed word model should find repeated text trivially
    predictable and an unfamiliar-vocabulary string the least predictable,
    with word-order shuffling and character shuffling landing in between."""

    def score(text):
        findings = _measure(text, features={**rs.DEFAULT_FEATURES, "kenlm_language_model": True},
                            kenlm_lmplz_path=_KENLM_LMPLZ_PATH)
        return findings[f"{rs.ID}kenlm_cross_entropy"]["value"]

    values = {name: score(text) for name, text in CORRUPTIONS.items()}
    if any(value is None for value in values.values()):
        pytest.skip("kenlm unavailable in this environment")
    # This is a word-level model, so word-order shuffling (which keeps every
    # whole-word token but destroys its context) is expected to hurt it more
    # than the two most extreme corruptions below - the actual, measured
    # behaviour, not an assumption: character shuffling and random letters
    # both mostly destroy whole words themselves, so KenLM's fixed,
    # training-derived vocabulary finds almost none of the "words" in either
    # of them recognisable at all, which costs even more than a familiar
    # vocabulary in a bizarre order.
    assert values["repeated_phrase"] < values["real_prose"] < values["shuffled_words"]
    assert values["shuffled_words"] < values["shuffled_chars"]
    assert values["shuffled_words"] < values["random_letters"]


@pytest.mark.skipif(not _HAVE_KENLM, reason="kenlm python bindings or a compiled lmplz not available")
def test_kenlm_language_model_uses_build_binary_when_available_for_quiet_loading():
    """Not a hard requirement (the ARPA file alone is a usable model), but
    when build_binary is findable this channel should use it: loading a
    KenLM binary model is silent, while loading the ARPA text file directly
    prints a progress notice straight to the process's own stderr on every
    single grading run."""

    findings = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "kenlm_language_model": True},
                        kenlm_lmplz_path=_KENLM_LMPLZ_PATH)
    item = findings[f"{rs.ID}kenlm_cross_entropy"]
    if item["value"] is None:
        pytest.skip(f"kenlm unavailable in this environment: {item['warning']}")
    build_binary = Path(_KENLM_LMPLZ_PATH).with_name("build_binary")
    if not (build_binary.is_file() and os.access(build_binary, os.X_OK)) and not shutil.which("build_binary"):
        pytest.skip("build_binary not available alongside lmplz or on PATH in this environment")
    assert item["distribution"]["model_format"] == "binary"


def test_kenlm_language_model_degrades_visibly_with_too_little_text():
    """No crash regardless of whether kenlm/lmplz happen to be available in
    this environment: too little text to build a held-out split degrades the
    same way "unavailable" always does elsewhere in this module."""

    findings = _measure("Hi.", features={**rs.DEFAULT_FEATURES, "kenlm_language_model": True})
    item = findings[f"{rs.ID}kenlm_cross_entropy"]
    assert item["value"] is None and item["warning"]


@pytest.mark.skipif(not _HAVE_KENLM, reason="kenlm python bindings or a compiled lmplz not available")
def test_kenlm_language_model_names_the_too_little_text_reason_specifically():
    """With kenlm and lmplz both genuinely available, the specific reason for
    a short document must be the held-out-split one, not a masked lmplz/kenlm
    availability problem."""

    findings = _measure("Hi.", features={**rs.DEFAULT_FEATURES, "kenlm_language_model": True},
                        kenlm_lmplz_path=_KENLM_LMPLZ_PATH)
    item = findings[f"{rs.ID}kenlm_cross_entropy"]
    assert item["value"] is None
    assert "sentences" in item["warning"]


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


# ------------------------------------------------------- neural language model
#
# The gating rule this whole group exists to satisfy: textgrader/corpus.py
# profiles this suite (cost="moderate", "sentence_transformers" not in
# requires) over every reference book using MetricSpec.defaults, so a model
# load hiding behind a default-True flag would silently cost every corpus
# build a multi-gigabyte download and a slow CPU forward pass per book.

def test_neural_language_model_is_off_by_default():
    assert rs.DEFAULT_FEATURES["neural_language_model"] is False
    findings = _measure(REAL_PROSE)
    assert f"{rs.ID}neural_lm_perplexity" not in findings


def test_neural_language_model_imports_nothing_under_default_config(monkeypatch):
    """The hard requirement: measuring with MetricSpec.defaults must never
    even ask optional.require for torch or transformers, let alone import
    them - so corpus profiling (which uses exactly these defaults) cannot
    trigger a model load no matter what is installed."""

    requested: list[str] = []
    real_require = optional.require

    def _tracking_require(name):
        requested.append(name)
        return real_require(name)

    monkeypatch.setattr(rs, "require", _tracking_require)
    _measure(REAL_PROSE)  # rs.DEFAULTS: neural_language_model is False
    assert "torch" not in requested
    assert "transformers" not in requested


def test_neural_language_model_degrades_visibly_without_torch_or_transformers(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "torch,transformers")
    optional.reset_cache()
    try:
        findings = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "neural_language_model": True})
        item = findings[f"{rs.ID}neural_lm_perplexity"]
        assert item["value"] is None and item["warning"]
    finally:
        optional.reset_cache()


_HAVE_NEURAL_LM = optional.have("torch") and optional.have("transformers")


@pytest.mark.skipif(not _HAVE_NEURAL_LM, reason="torch/transformers not installed")
def test_neural_language_model_scores_real_text_when_enabled():
    findings = _measure(REAL_PROSE, features={**rs.DEFAULT_FEATURES, "neural_language_model": True},
                        neural_lm_max_chars=1500)
    item = findings[f"{rs.ID}neural_lm_perplexity"]
    if item["value"] is None:
        pytest.skip(f"model unavailable in this environment: {item['warning']}")
    assert item["value"] > 0
    assert item["distribution"]["model"] == rs.DEFAULTS["neural_lm_model"]
    assert item["sample_size_sensitive"] is True
    assert "gibberish" in item["distribution"]["warning_not_a_gibberish_detector"]


@pytest.mark.skipif(not _HAVE_NEURAL_LM, reason="torch/transformers not installed")
def test_neural_language_model_reflects_the_documented_subword_gibberish_trap():
    """The caveat this channel's docstring makes at length, pinned down: on
    this exact model, purely random letters can score a LOWER (more
    "fluent") perplexity than the same real sentence with only its word
    order scrambled - the opposite of what the character n-gram and
    compression channels correctly show for the same corruption. This is
    documented behaviour, not a bug to chase away."""

    real = "The lighthouse keeper walked down to the shore at dawn."
    shuffled = "shore the at keeper down dawn walked lighthouse to The."
    random_letters = "qwtz bklm vxpr njgd hfsa ouei rlmt bzkq."

    def score(text):
        analysis = DocumentAnalysis.from_text(text)
        config = {**rs.DEFAULTS, "features": {**rs.DEFAULT_FEATURES, "neural_language_model": True},
                 "neural_lm_max_chars": 500}
        findings = {item["metric_id"]: item for item in rs.measure(analysis, config=config)}
        return findings[f"{rs.ID}neural_lm_perplexity"]["value"]

    real_ppl, shuffled_ppl, random_ppl = score(real), score(shuffled), score(random_letters)
    if None in (real_ppl, shuffled_ppl, random_ppl):
        pytest.skip("model unavailable in this environment")
    assert real_ppl < shuffled_ppl
    assert random_ppl < shuffled_ppl  # the trap: word-order noise scores worse than character noise.


@pytest.mark.skipif(not _HAVE_NEURAL_LM, reason="torch/transformers not installed")
def test_neural_language_model_still_ranks_real_prose_most_fluent(all_findings):
    """The one ordering this channel keeps even with the trap above: real,
    grammatical prose is never LESS fluent than a corruption of itself. Which
    corruption is worst is exactly the part this channel gets non-intuitively
    (see the docstring and the trap test above), so that part is not asserted
    here."""

    overrides = {"features": {**rs.DEFAULT_FEATURES, "neural_language_model": True},
                "neural_lm_max_chars": 1500}
    findings = {name: _measure(text, **overrides) for name, text in CORRUPTIONS.items()}
    real = findings["real_prose"][f"{rs.ID}neural_lm_perplexity"]["value"]
    if real is None:
        pytest.skip("model unavailable in this environment")
    for name in ("shuffled_words", "shuffled_chars", "random_letters"):
        other = findings[name][f"{rs.ID}neural_lm_perplexity"]["value"]
        assert real < other, name


# ------------------------------------------------------- textdescriptives cross-check

def test_textdescriptives_cross_check_is_off_by_default():
    assert rs.DEFAULT_FEATURES["textdescriptives_cross_check"] is False
    findings = _measure(REAL_PROSE)
    assert f"{rs.ID}textdescriptives_word_perplexity" not in findings


def test_textdescriptives_cross_check_degrades_visibly_without_the_package(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "textdescriptives")
    optional.reset_cache()
    try:
        findings = _measure(REAL_PROSE,
                            features={**rs.DEFAULT_FEATURES, "textdescriptives_cross_check": True})
        item = findings[f"{rs.ID}textdescriptives_word_perplexity"]
        assert item["value"] is None and item["warning"]
    finally:
        optional.reset_cache()


_HAVE_TEXTDESCRIPTIVES = optional.have("spacy") and optional.have("textdescriptives")


@pytest.mark.skipif(not _HAVE_TEXTDESCRIPTIVES, reason="spacy/textdescriptives not installed")
def test_textdescriptives_cross_check_reports_a_finite_value_next_to_unknown_word_rate():
    findings = _measure(REAL_PROSE,
                        features={**rs.DEFAULT_FEATURES, "textdescriptives_cross_check": True},
                        textdescriptives_max_chars=4000)
    item = findings[f"{rs.ID}textdescriptives_word_perplexity"]
    if item["value"] is None:
        pytest.skip(f"unavailable in this environment: {item['warning']}")
    assert item["value"] > 0
    assert item["distribution"]["cross_check_of"] == f"{rs.ID}unknown_word_rate"
    assert f"{rs.ID}unknown_word_rate" in findings  # the channel it is meant to sit beside


# --------------------------------------------------------- registry contract

def test_runs_without_raising_via_grade(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "randomness_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    from textgrader.results import StatusType
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]
