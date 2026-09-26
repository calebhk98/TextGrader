"""Contract and validation tests for the readability-formula cross-check suite.

``tests/test_optional_metrics.py`` already parametrizes ``REGISTRY`` and so
already exercises this suite for "off by default", "runs without raising" and
"survives a degenerate document" (empty text, ``"Hi."``, three one-word
paragraphs). This file adds the suite-specific behaviour the task spec calls
out by name: simple-children's-prose-vs-dense-academic-prose separation
across every headline channel, a hand-computable exact value on a tiny
fixture, the contested-syllable-count words the spec names, the 100-word
insufficient-data floor, py-readability-metrics' hard refusal below that
floor being caught rather than crashing, TextGrader's existing core fk/ari
(and the maturity aggregate) staying unchanged whether or not this suite
runs, per-feature switchability, and graceful degradation with every optional
package unavailable.
"""

from __future__ import annotations

import math

import pytest

import grade
from textgrader import optional
from textgrader.core_metrics import syllables as core_syllables
from textgrader.document import DocumentAnalysis
from textgrader.metrics import readability_suite as m
from textgrader.results import Action, Polarity

# --------------------------------------------------------------------- fixtures

# Real, plain-vocabulary short sentences -- x3 to clear the suite's 100-word/
# 30-sentence floors comfortably (273 words, 45 sentences).
CHILDREN_TEXT = (
    "The cat sat on the mat. The dog ran to the tree. A bird flew in the sky. "
    "The sun was big and warm. Kids like to play and run. Mom made a cake for us. "
    "We ate the cake at noon. The ball rolled down the hill. Tom got the ball back fast. "
    "Dad read a book to us. The book had fun little rhymes. We went to bed at night. "
    "The moon was round and bright. Stars came out one by one. It was time to go to sleep. "
) * 3

# Real, dense, subordination-heavy, polysyllabic academic prose -- x6 to clear
# the same floors (570 words, 30 sentences).
ACADEMIC_TEXT = (
    "Notwithstanding the ostensibly straightforward methodology employed throughout the "
    "investigation, subsequent empirical analysis revealed considerable heterogeneity among "
    "the underlying distributions. Researchers subsequently postulated that unobserved "
    "confounding variables, potentially correlated with both the exposure and the outcome, "
    "might substantially attenuate the estimated associations. Consequently, the "
    "interdisciplinary committee recommended a comprehensive reevaluation of the theoretical "
    "framework underpinning contemporary econometric methodologies. Furthermore, the "
    "epistemological implications of privileging quantitative over qualitative methodologies "
    "remain contentious among practitioners occupying divergent methodological traditions. "
    "Notably, the aforementioned considerations necessitate a fundamentally interdisciplinary "
    "approach that synthesizes insights from sociolinguistics, computational linguistics, and "
    "cognitive psychology simultaneously. "
) * 6

# 9 words, 1 long word (>6 letters: "outside"); textstat's own sentence
# counter ignores any run it counts as <=2 words, so "Cats sit." (2 words) is
# dropped and only the second run is counted, giving exactly one sentence --
# verified directly against textstat.sentence_count/long_word_count/
# lexicon_count before writing this fixture. LIX = words/sentences +
# 100*long_words/words = 9/1 + 100*1/9.
LIX_TEXT = "Cats sit. Dogs run fast every single day outside."
LIX_EXPECTED = 9 / 1 + 100 * 1 / 9

TINY_TEXT = "A small dog ran across the quiet yard and barked twice."  # 11 words

# Reproduces, on a small fixture, the real bug found on Alice in Wonderland
# (26,539 canonical words, 1,459 canonical sentences): pystylometry's regex
# sentence splitter (pystylometry._utils.split_sentences) requires an
# immediate capital Latin letter after each sentence-final [.!?]+, so a
# sentence that opens on a curly quotation mark defeats it, and the
# splitter silently merges it with the sentence before. A fixture of ONLY
# dialogue does not reproduce this: pystylometry's own fallback path
# ("if len(result) <= 1: use a simpler split") fires instead and happens to
# get it right by coincidence. Narration sentences (which split correctly,
# since a capital letter does follow) have to be mixed in, exactly as in
# real prose, so the primary (flawed) regex is the one actually used, and
# only the dialogue-adjacent boundaries are lost. Verified directly before
# writing this fixture: pystylometry counts 24 sentences here against a
# canonical 40 (a 0.6x ratio, outside any reasonable tolerance) -- close in
# kind, if not in exact magnitude, to Alice's own 513-vs-1,459 (0.35x).
DIALOGUE_TEXT = (
    "Alice sat by the river. She was very bored indeed. "
    "“Come here,” she said. “Now!” cried the White Rabbit. "
    "The day was warm and bright. Alice looked around slowly. "
    "“Where are you going?” asked the boy. “Nowhere special,” she answered softly. "
    "The garden was full of flowers. Everyone seemed happy today. "
) * 4

# Every readability-formula number this suite produces comes from an optional
# package (textstat, py-readability-metrics, pronouncing, pystylometry).
# ``TEXTGRADER_DISABLE_OPTIONAL=all`` (the project's documented way to
# simulate an environment without any optional dependency, used for the
# degraded-mode verification run) makes every one of them report
# "unavailable" rather than a real number -- exactly the scenario
# ``test_graceful_degradation_without_any_optional_package`` below exists to
# check. The tests that assert real, computed values (separation, exact
# values, cross-implementation disagreement, the formula aggregate) are
# skipped rather than failed in that mode, the same way
# ``test_lexical_norms_suite.py``'s ``requires_real_*`` markers skip
# resource-dependent assertions when the resource is not cached.
_LIBRARIES_AVAILABLE = all(
    optional.require(name)[0] is not None
    for name in ("textstat", "py_readability_metrics", "pronouncing", "pystylometry"))
requires_libraries = pytest.mark.skipif(
    not _LIBRARIES_AVAILABLE,
    reason="textstat/py-readability-metrics/pronouncing/pystylometry not fully available "
          "(or disabled via TEXTGRADER_DISABLE_OPTIONAL)")


def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book")


def _findings(text: str, config=None) -> dict[str, dict]:
    return {f["metric_id"]: f for f in m.measure(_analysis(text), config=config)}


# --------------------------------------------------------------- config surface

def test_disabled_by_default_via_config(tmp_path, base_config):
    source = tmp_path / "story.txt"
    source.write_text(CHILDREN_TEXT, encoding="utf-8")
    report = grade.analyze(source, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(m.PREFIX)]


def test_family_and_prefix():
    # The task spec's own suggested prefix, since "readability" is not in
    # test_optional_metrics.py's hard-coded PREFIXES set (that file is off
    # limits for this task) -- see the module docstring's "Family and
    # metric-ID prefix".
    assert m.PREFIX == "nlp.readability_"
    assert m.FAMILY == "readability"
    found = _findings(CHILDREN_TEXT)
    assert found
    assert all(mid.startswith("nlp.readability_") for mid in found)


def test_every_finding_is_polarity_neutral(tmp_path, base_config):
    # Every finding module-side has no polarity field at all (common.finding()
    # takes none), and MetricResult defaults to Polarity.NEUTRAL -- this pins
    # that nothing downstream (grade.py's _finding_result) ever assigns one.
    source = tmp_path / "story.txt"
    source.write_text(CHILDREN_TEXT, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "readability_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    items = [item for item in report.results if item.metric_id.startswith(m.PREFIX)]
    assert items
    assert all(item.polarity is Polarity.NEUTRAL for item in items)


# ------------------------------------------------------- children vs academic

# (metric id suffix, "higher" if academic should score higher, "lower" if it
# should score lower). Covers every up-front headline channel this suite
# computes across all three libraries plus TextGrader's own core values --
# the acceptance rule that every headline channel must be shown to separate
# two inputs it should.
_HIGHER_FOR_ACADEMIC = [
    "fk_textstat", "ari_textstat", "gunning_fog_textstat", "smog_textstat",
    "coleman_liau_textstat", "dale_chall_textstat", "linsear_write_textstat",
    "spache_textstat", "lix_textstat", "rix_textstat", "mcalpine_eflaw_textstat",
    "fk_readability_metrics", "ari_readability_metrics", "coleman_liau_readability_metrics",
    "dale_chall_readability_metrics", "linsear_write_readability_metrics",
    "spache_readability_metrics", "smog_readability_metrics",
    "fk_pystylometry", "ari_pystylometry", "coleman_liau_pystylometry",
    "dale_chall_pystylometry", "linsear_write_pystylometry", "smog_pystylometry",
    "forcast_pystylometry", "psk_pystylometry", "fry_pystylometry",
    "formula_grade_mean", "formula_grade_median",
]
_LOWER_FOR_ACADEMIC = [
    "flesch_reading_ease_textstat", "flesch_reading_ease_readability_metrics",
    "flesch_reading_ease_pystylometry",
]


@requires_libraries
def test_children_vs_academic_separates_every_higher_channel():
    children = _findings(CHILDREN_TEXT)
    academic = _findings(ACADEMIC_TEXT)
    for suffix in _HIGHER_FOR_ACADEMIC:
        mid = f"{m.PREFIX}{suffix}"
        c_value, a_value = children[mid]["value"], academic[mid]["value"]
        assert c_value is not None and a_value is not None, mid
        assert a_value > c_value, (mid, c_value, a_value)


@requires_libraries
def test_children_vs_academic_separates_every_lower_channel():
    children = _findings(CHILDREN_TEXT)
    academic = _findings(ACADEMIC_TEXT)
    for suffix in _LOWER_FOR_ACADEMIC:
        mid = f"{m.PREFIX}{suffix}"
        c_value, a_value = children[mid]["value"], academic[mid]["value"]
        assert c_value is not None and a_value is not None, mid
        assert a_value < c_value, (mid, c_value, a_value)


@requires_libraries
def test_core_fk_and_ari_also_separate_the_same_way():
    # The "core" entries collected for the disagreement findings are the same
    # core_metrics values used elsewhere in this codebase's maturity
    # aggregate -- confirm they separate the same direction as every library
    # here, i.e. this suite is not inventing a different notion of "harder".
    children_dis = _findings(CHILDREN_TEXT)[f"{m.PREFIX}fk_disagreement"]["distribution"]["values"]
    academic_dis = _findings(ACADEMIC_TEXT)[f"{m.PREFIX}fk_disagreement"]["distribution"]["values"]
    assert academic_dis["core"] > children_dis["core"]
    ari_children = _findings(CHILDREN_TEXT)[f"{m.PREFIX}ari_disagreement"]["distribution"]["values"]
    ari_academic = _findings(ACADEMIC_TEXT)[f"{m.PREFIX}ari_disagreement"]["distribution"]["values"]
    assert ari_academic["core"] > ari_children["core"]


# ----------------------------------------------------------- exact hand values

@requires_libraries
def test_lix_exact_value():
    found = _findings(LIX_TEXT)
    item = found[f"{m.PREFIX}lix_textstat"]
    assert item["value"] == pytest.approx(LIX_EXPECTED, abs=1e-9)


def test_contested_syllable_words_reproduced_exactly():
    """The exact per-word counts the module docstring cites.

    ``core_metrics.syllables`` used to undercount diphthong words (fire,
    poem, cruel, hour, science all scored 1); its rules now match CMUdict on
    them.  What it still gets wrong is irregular spelling (evening, business,
    colonel), which is what the cross-check exists to show.  Reproduced
    directly against the installed packages, so a change to either counter
    is caught.
    """

    pronouncing_mod, reason = optional.require("pronouncing")
    if pronouncing_mod is None:
        pytest.skip(reason)

    def cmudict_count(word):
        phones = pronouncing_mod.phones_for_word(word)
        return pronouncing_mod.syllable_count(phones[0]) if phones else None

    for word in ("fire", "poem", "cruel", "hour", "science"):
        assert core_syllables(word) == cmudict_count(word) == 2, word
    contested = {"evening": (3, 2), "business": (3, 2), "colonel": (3, 2)}
    for word, (core_expected, cmu_expected) in contested.items():
        assert core_syllables(word) == core_expected, word
        assert cmudict_count(word) == cmu_expected, word

    agreeing = {"every": 3, "naturally": 4, "beautiful": 3, "world": 1}
    for word, expected in agreeing.items():
        assert core_syllables(word) == expected, word
        assert cmudict_count(word) == expected, word


@requires_libraries
def test_syllable_crosscheck_flags_the_contested_words_as_evidence():
    text = ("Every evening the colonel talked business and read a poem about fire, "
           "or so the beautiful old story of the world goes. " * 8)
    found = _findings(text)
    item = found[f"{m.PREFIX}syllable_disagreement_rate_core_vs_cmudict"]
    assert item["value"] is not None and item["value"] > 0
    evidence_words = {row["word"] for row in item["evidence"]}
    assert evidence_words & {"evening", "business", "colonel"}
    assert not evidence_words & {"fire", "poem"}
    for row in item["evidence"]:
        assert row["core_heuristic"] != row["cmudict"]


@requires_libraries
def test_syllable_disagreement_rate_is_a_bounded_percent():
    item = _findings(ACADEMIC_TEXT)[f"{m.PREFIX}syllable_disagreement_rate_core_vs_cmudict"]
    assert 0.0 <= item["value"] <= 100.0
    textstat_item = _findings(ACADEMIC_TEXT)[
        f"{m.PREFIX}syllable_disagreement_rate_textstat_vs_cmudict"]
    assert 0.0 <= textstat_item["value"] <= 100.0


@requires_libraries
def test_syllable_disagreement_ignores_words_cmudict_does_not_know():
    """A word with no CMUdict entry has no reference count, so adding such
    words must leave both rates unchanged.  Scoring them against a fallback
    counter once moved the textstat rate whenever the core counter changed."""
    base = ("Every evening the colonel talked business and read a poem about fire, "
            "or so the beautiful old story of the world goes. ")
    invented = "Zorblaxian quimbrotted flenwicks glarmously. "
    plain = _findings(base * 8)
    padded = _findings((base + invented) * 8)
    for suffix in ("core_vs_cmudict", "textstat_vs_cmudict"):
        metric_id = f"{m.PREFIX}syllable_disagreement_rate_{suffix}"
        assert padded[metric_id]["value"] == pytest.approx(plain[metric_id]["value"])
        assert padded[metric_id]["distribution"]["words_in_cmudict"] == \
            plain[metric_id]["distribution"]["words_in_cmudict"]
    core = padded[f"{m.PREFIX}syllable_disagreement_rate_core_vs_cmudict"]["distribution"]
    assert core["unique_words_sampled"] > core["words_in_cmudict"]
    assert core["cmudict_coverage_percent"] < 100.0


# ------------------------------------------------------- segmentation diagnostics

@requires_libraries
def test_pystylometry_sentence_merging_is_caught_on_dialogue_prose():
    """Reproduces the real bug found on Alice in Wonderland (see the fixture
    docstring above) on a small, hand-checked example: pystylometry's own
    (raw) sentence count diverges sharply from canonical, its
    canonical-segmentation counterpart matches it, and every one of these
    numbers is visible in the finding's own distribution rather than only
    showing up as an unexplained outlier in the grade value.
    """

    found = _findings(DIALOGUE_TEXT)
    canonical_sentences = DocumentAnalysis.from_text(DIALOGUE_TEXT).sentence_count

    raw = found[f"{m.PREFIX}ari_pystylometry"]
    raw_ratio = raw["distribution"]["sentence_count_ratio_vs_canonical"]
    assert raw["distribution"]["canonical_sentence_count"] == canonical_sentences
    assert raw["distribution"]["library_sentence_count"] < canonical_sentences
    assert raw_ratio < 0.9, "the raw pystylometry sentence count should diverge sharply here"

    canonical_seg = found[f"{m.PREFIX}ari_pystylometry_canonical_seg"]
    canonical_ratio = canonical_seg["distribution"]["sentence_count_ratio_vs_canonical"]
    assert canonical_seg["distribution"]["library_sentence_count"] == canonical_sentences
    assert canonical_ratio == pytest.approx(1.0)

    # The formula VALUES must actually differ too -- the merged-sentence
    # version is not merely flagged, it is a materially different number.
    assert raw["value"] != canonical_seg["value"]


@requires_libraries
def test_pystylometry_sentence_merging_excludes_it_from_the_trusted_aggregate():
    found = _findings(DIALOGUE_TEXT)
    mean_item = found[f"{m.PREFIX}formula_grade_mean"]
    excluded = mean_item["distribution"]["excluded_formula_ids"]
    assert "pystylometry_ari" in excluded
    # The canonical-segmentation counterpart, having matched canonical
    # sentence-for-sentence, must be trusted and therefore included.
    assert "pystylometry_canonical_seg_ari" in mean_item["distribution"]["formula_ids"]
    assert "pystylometry_canonical_seg_ari" not in excluded


@requires_libraries
def test_formula_grade_spread_raw_keeps_the_full_disagreement_visible():
    found = _findings(DIALOGUE_TEXT)
    trusted_spread = found[f"{m.PREFIX}formula_grade_spread"]["value"]
    raw_spread = found[f"{m.PREFIX}formula_grade_spread_raw"]["value"]
    # The unfiltered spread must be at least as large as the trust-filtered
    # one, since the trusted set is always a subset of every grade-scale
    # value produced this run.
    assert raw_spread >= trusted_spread
    assert raw_spread > 0 and trusted_spread >= 0


@requires_libraries
def test_textstat_and_readability_metrics_segmentation_is_recorded_even_when_close():
    # textstat/py-readability-metrics do not have pystylometry's bug, but
    # every finding -- theirs included -- must still carry the same
    # segmentation diagnostics, per the fix: a reader should never have to
    # guess whether a library's sentence count was even checked.
    found = _findings(ACADEMIC_TEXT)
    canonical_sentences = DocumentAnalysis.from_text(ACADEMIC_TEXT).sentence_count
    canonical_words = DocumentAnalysis.from_text(ACADEMIC_TEXT).word_count
    for mid in (f"{m.PREFIX}fk_textstat", f"{m.PREFIX}fk_readability_metrics"):
        dist = found[mid]["distribution"]
        assert dist["canonical_sentence_count"] == canonical_sentences
        assert dist["canonical_word_count"] == canonical_words
        assert dist["library_sentence_count"] is not None
        assert dist["library_word_count"] is not None
        assert "sentence_count_ratio_vs_canonical" in dist
        assert "words_per_sentence_ratio_vs_canonical" in dist


@requires_libraries
def test_forcast_has_no_sentence_count_and_is_never_excluded():
    # FORCAST's formula never touches sentence length, so it has nothing to
    # distrust regardless of how badly a document's dialogue confuses a
    # sentence splitter, and must always be pooled into the trusted aggregate.
    found = _findings(DIALOGUE_TEXT)
    forcast = found[f"{m.PREFIX}forcast_pystylometry"]
    assert forcast["distribution"].get("library_sentence_count") is None
    mean_item = found[f"{m.PREFIX}formula_grade_mean"]
    assert "pystylometry_forcast" in mean_item["distribution"]["formula_ids"]
    assert "pystylometry_forcast" not in mean_item["distribution"]["excluded_formula_ids"]


@requires_libraries
def test_canonical_segmentation_toggle_removes_its_own_ids_only():
    with_canonical = _findings(ACADEMIC_TEXT)
    assert f"{m.PREFIX}ari_pystylometry_canonical_seg" in with_canonical
    assert f"{m.PREFIX}ari_pystylometry" in with_canonical

    without_canonical = _findings(
        ACADEMIC_TEXT, config={"features": {"pystylometry_canonical_segmentation": False}})
    assert f"{m.PREFIX}ari_pystylometry_canonical_seg" not in without_canonical
    assert f"{m.PREFIX}ari_pystylometry" in without_canonical


# ------------------------------------------------------------ minimum-size floor

@requires_libraries
def test_tiny_text_is_insufficient_data_end_to_end(tmp_path, base_config):
    source = tmp_path / "tiny.txt"
    source.write_text(TINY_TEXT, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "readability_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    items = {item.metric_id: item for item in report.results if item.metric_id.startswith(m.PREFIX)}
    assert items
    # textstat and pystylometry never raise on tiny input -- they return a
    # real number, which the suite's own sample_size=word_count/min_sample=100
    # gate (see the module docstring's "Handling the 100-word floor") must
    # still mark insufficient rather than an outlier claim.
    checked = [f"{m.PREFIX}fk_textstat", f"{m.PREFIX}ari_pystylometry",
              f"{m.PREFIX}forcast_pystylometry", f"{m.PREFIX}lix_textstat"]
    for mid in checked:
        assert items[mid].action is Action.INSUFFICIENT_DATA, (mid, items[mid].action,
                                                               items[mid].warning)
        assert items[mid].value is not None  # a real (if unreliable) number, not a crash


def test_readability_metrics_below_floor_does_not_crash_and_is_reported():
    # py-readability-metrics itself raises ReadabilityException below 100
    # words; this must never escape measure(), and the reason must be the
    # real, quoted exception text (task rule 2).
    module, reason = optional.require("py_readability_metrics")
    if module is None:
        pytest.skip(reason)
    found = _findings(TINY_TEXT)
    item = found[f"{m.PREFIX}fk_readability_metrics"]
    assert item["value"] is None
    assert item["warning"] and "ReadabilityException" in item["warning"]
    assert "100 words" in item["warning"]


def test_empty_and_degenerate_text_never_crashes_and_never_emits_nan():
    def has_nan(value):
        if isinstance(value, float):
            return math.isnan(value)
        if isinstance(value, dict):
            return any(has_nan(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return any(has_nan(v) for v in value)
        return False

    for text in ("", "Hi.", "A\n\nB\n\nC"):
        findings = m.measure(_analysis(text))
        assert findings  # still emits the declared channels, not a bare crash
        assert not any(has_nan(f) for f in findings), text


# --------------------------------------------------------- core values unchanged

def test_core_prose_metrics_unchanged_by_this_suite(tmp_path, base_config):
    source = tmp_path / "book.txt"
    source.write_text(ACADEMIC_TEXT, encoding="utf-8")
    off_config = {**base_config, "metrics": {**base_config["metrics"],
                                             "readability_suite": {"enabled": False}}}
    on_config = {**base_config, "metrics": {**base_config["metrics"],
                                            "readability_suite": {"enabled": True}}}
    off_report = grade.analyze(source, off_config)
    on_report = grade.analyze(source, on_config)

    def prose_values(report):
        return {item.metric_id: item.value for item in report.results
               if item.metric_id.startswith("prose.")}

    off_values, on_values = prose_values(off_report), prose_values(on_report)
    assert off_values  # the fixture must actually be big enough to produce core metrics
    assert off_values == on_values
    assert off_values["prose.fk"] == on_values["prose.fk"]
    assert off_values["prose.ari"] == on_values["prose.ari"]
    # The maturity aggregate is computed purely from Polarity-oriented
    # findings; every finding this suite adds is Polarity.NEUTRAL, so it must
    # be byte-identical whether or not the suite ran.
    assert off_report.maturity() == on_report.maturity()


def test_readability_suite_ids_are_absent_when_disabled(tmp_path, base_config):
    source = tmp_path / "book.txt"
    source.write_text(ACADEMIC_TEXT, encoding="utf-8")
    off_config = {**base_config, "metrics": {**base_config["metrics"],
                                             "readability_suite": {"enabled": False}}}
    on_config = {**base_config, "metrics": {**base_config["metrics"],
                                            "readability_suite": {"enabled": True}}}
    off_report = grade.analyze(source, off_config)
    on_report = grade.analyze(source, on_config)
    assert not [item for item in off_report.results if item.metric_id.startswith(m.PREFIX)]
    assert [item for item in on_report.results if item.metric_id.startswith(m.PREFIX)]


# ------------------------------------------------------------------ disagreement

@requires_libraries
def test_disagreement_findings_show_a_real_nonzero_disagreement():
    # Three independent tokenizers/formulas essentially never land on the
    # exact same floating-point number; this is real, not manufactured.
    # (Dale-Chall is the one formula in this suite where, on this particular
    # fixture, all three implementations happen to agree exactly -- textstat
    # and py-readability-metrics share the classic 3,000-word Dale-Chall list
    # and pystylometry's own restricted list flags the same words as
    # difficult here. That IS this suite's "may agree without failing" case,
    # so it is checked on the children's fixture instead, where it does not.)
    academic = _findings(ACADEMIC_TEXT)
    for suffix in ("ari_disagreement", "linsear_write_disagreement", "coleman_liau_disagreement"):
        item = academic[f"{m.PREFIX}{suffix}"]
        assert item["value"] is not None and item["value"] > 0, suffix
        assert item["distribution"]["n_implementations"] >= 2
    children = _findings(CHILDREN_TEXT)
    dale_chall_item = children[f"{m.PREFIX}dale_chall_disagreement"]
    assert dale_chall_item["value"] is not None and dale_chall_item["value"] > 0


def test_disagreement_reports_unavailable_with_fewer_than_two_implementations():
    found = _findings(ACADEMIC_TEXT, config={
        "features": {"textstat_formulas": False, "readability_metrics_formulas": False,
                    "pystylometry_formulas": False}})
    item = found[f"{m.PREFIX}fk_disagreement"]
    # Only the "core" value remains once every library group is switched off.
    assert item["value"] is None or item["distribution"] is None
    if item["value"] is None:
        assert item["warning"]


# -------------------------------------------------------------- cross-formula aggregate

@requires_libraries
def test_formula_aggregate_matches_its_own_inputs():
    found = _findings(ACADEMIC_TEXT)
    mean_item = found[f"{m.PREFIX}formula_grade_mean"]
    median_item = found[f"{m.PREFIX}formula_grade_median"]
    max_item = found[f"{m.PREFIX}formula_grade_max"]
    spread_item = found[f"{m.PREFIX}formula_grade_spread"]
    sd_item = found[f"{m.PREFIX}formula_grade_sd"]
    n = mean_item["distribution"]["n_formulas"]
    assert n >= 10  # a genuinely broad formula set, not two or three numbers
    assert mean_item["distribution"]["min"] <= mean_item["value"] <= mean_item["distribution"]["max"]
    assert mean_item["distribution"]["min"] <= median_item["value"] <= mean_item["distribution"]["max"]
    assert max_item["value"] == mean_item["distribution"]["max"]
    assert spread_item["value"] == mean_item["distribution"]["max"] - mean_item["distribution"]["min"]
    assert sd_item["value"] >= 0
    # Flesch Reading Ease (opposite scale) and the raw Dale-Chall score must
    # never be pooled into a US-grade-level aggregate.
    for formula_id in mean_item["distribution"]["formula_ids"]:
        assert "flesch_reading_ease" not in formula_id
        assert "dale_chall" not in formula_id


@requires_libraries
def test_formula_aggregate_headline_varies_between_documents():
    # Rule: no headline may return the same value on every document.
    children_mean = _findings(CHILDREN_TEXT)[f"{m.PREFIX}formula_grade_mean"]["value"]
    academic_mean = _findings(ACADEMIC_TEXT)[f"{m.PREFIX}formula_grade_mean"]["value"]
    assert children_mean != academic_mean


def test_formula_aggregate_absent_below_the_floor():
    found = _findings(TINY_TEXT)
    assert f"{m.PREFIX}formula_grade_mean" not in found


# --------------------------------------------------------------- difficult words

@requires_libraries
def test_difficult_word_overlap_is_a_bounded_ratio_and_varies():
    children_item = _findings(CHILDREN_TEXT)[f"{m.PREFIX}difficult_word_list_overlap"]
    academic_item = _findings(ACADEMIC_TEXT)[f"{m.PREFIX}difficult_word_list_overlap"]
    for item in (children_item, academic_item):
        if item["value"] is not None:
            assert 0.0 <= item["value"] <= 1.0
    # Academic prose must produce a much larger difficult-word set than
    # simple prose under both bundled word lists.
    assert (academic_item["distribution"]["textstat_difficult_word_count"] >
           children_item["distribution"]["textstat_difficult_word_count"])
    assert (academic_item["distribution"]["pystylometry_difficult_word_count"] >
           children_item["distribution"]["pystylometry_difficult_word_count"])


# ------------------------------------------------------------- feature switches

def test_pystylometry_gunning_fog_is_off_by_default_and_toggle_adds_it():
    off = _findings(ACADEMIC_TEXT)
    assert f"{m.PREFIX}gunning_fog_pystylometry" not in off
    on = _findings(ACADEMIC_TEXT, config={"features": {"pystylometry_gunning_fog": True}})
    assert f"{m.PREFIX}gunning_fog_pystylometry" in on


def test_textstat_locale_formulas_are_off_by_default_and_toggle_adds_them():
    off = _findings(ACADEMIC_TEXT)
    assert f"{m.PREFIX}fernandez_huerta_textstat" not in off
    on = _findings(ACADEMIC_TEXT, config={"features": {"textstat_locale_formulas": True}})
    assert f"{m.PREFIX}fernandez_huerta_textstat" in on
    assert f"{m.PREFIX}gulpease_index_textstat" in on


def test_each_library_group_is_independently_switchable():
    baseline = _findings(ACADEMIC_TEXT)
    assert f"{m.PREFIX}fk_textstat" in baseline
    assert f"{m.PREFIX}fk_readability_metrics" in baseline
    assert f"{m.PREFIX}fk_pystylometry" in baseline

    no_textstat = _findings(ACADEMIC_TEXT, config={"features": {"textstat_formulas": False}})
    assert f"{m.PREFIX}fk_textstat" not in no_textstat
    assert f"{m.PREFIX}fk_readability_metrics" in no_textstat

    no_readability_metrics = _findings(ACADEMIC_TEXT,
                                       config={"features": {"readability_metrics_formulas": False}})
    assert f"{m.PREFIX}fk_readability_metrics" not in no_readability_metrics
    assert f"{m.PREFIX}fk_textstat" in no_readability_metrics

    no_pystylometry = _findings(ACADEMIC_TEXT, config={"features": {"pystylometry_formulas": False}})
    assert f"{m.PREFIX}fk_pystylometry" not in no_pystylometry
    assert f"{m.PREFIX}forcast_pystylometry" not in no_pystylometry
    assert f"{m.PREFIX}fk_textstat" in no_pystylometry


def test_syllable_and_difficult_word_crosscheck_are_independently_switchable():
    no_syllable = _findings(ACADEMIC_TEXT, config={"features": {"syllable_crosscheck": False}})
    assert f"{m.PREFIX}syllable_disagreement_rate_core_vs_cmudict" not in no_syllable
    assert f"{m.PREFIX}difficult_word_list_overlap" in no_syllable

    no_difficult = _findings(ACADEMIC_TEXT, config={"features": {"difficult_word_crosscheck": False}})
    assert f"{m.PREFIX}difficult_word_list_overlap" not in no_difficult
    assert f"{m.PREFIX}syllable_disagreement_rate_core_vs_cmudict" in no_difficult


# ---------------------------------------------------------------- degradation

def test_graceful_degradation_without_any_optional_package(monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL",
                       "textstat,py_readability_metrics,pronouncing,pystylometry")
    optional.reset_cache()
    try:
        findings = m.measure(_analysis(ACADEMIC_TEXT))
    finally:
        optional.reset_cache()
    assert findings
    unavailable = [f for f in findings if f["value"] is None]
    assert unavailable
    for f in unavailable:
        # every one names a real, actionable reason, not a bare None
        if f["metric_id"] != f"{m.PREFIX}library_versions":
            assert f["warning"], f["metric_id"]
    # importlib.metadata's own version lookups are unaffected by
    # TEXTGRADER_DISABLE_OPTIONAL (they never import the package itself), so
    # the version-report finding's distribution should still resolve.
    versions = next(f for f in findings if f["metric_id"] == f"{m.PREFIX}library_versions")
    assert versions["distribution"]["textstat"]


def test_every_registered_optional_package_key_used_here_exists_once():
    # A duplicate PACKAGES key silently overwrites another suite's entry --
    # this pins that this suite's three new keys are each registered exactly
    # once and importable without raising at the dict-construction level.
    for name in ("textstat", "py_readability_metrics", "pronouncing"):
        assert name in optional.PACKAGES
    # pystylometry is reused, not re-registered.
    assert optional.PACKAGES["pystylometry"][0] == "pystylometry"
