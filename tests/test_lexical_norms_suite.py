"""Contract and validation tests for the lexical-sophistication/norm suite.

``tests/test_optional_metrics.py`` already parametrizes ``REGISTRY`` and so
already exercises this suite for "off by default", "runs without raising" and
"survives a degenerate document" -- this file adds the suite-specific
behaviour, using tiny synthetic norm tables written to ``tmp_path`` (never
the network) so these tests run the same with or without the real, ~100MB of
cached resources: an exact-value check of every resource's file-format
parser (:mod:`textgrader.lexicons`), the task's five required separation
fixtures with exact expected numbers (not just direction), coverage
travelling with every mean, table-referenced tail shares, and one resource
being unavailable disabling only its own findings. A second block of
``skipif``-guarded tests reads the real cached resources (present in this
task's own environment; cleanly skipped wherever they are not) and the real
50-book reference corpus, to report which way AoA/concreteness actually
separate a children's classic from a denser adult novel.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from textgrader import lexicons
from textgrader.document import DocumentAnalysis
from textgrader.metrics import lexical_norms_suite as suite
from textgrader.stats import quantile

# --------------------------------------------------------------- tiny fixtures
#
# Every helper below writes exactly the columns each real parser reads (see
# textgrader/lexicons.py's ``_parse_*`` functions), so a change to a parser
# that stops matching the real file format breaks these tests too.


def _write_warriner(path: Path, rows: dict[str, tuple[float, float, float]]) -> Path:
    lines = ["idx,Word,V.Mean.Sum,A.Mean.Sum,D.Mean.Sum"]
    for i, (word, (v, a, d)) in enumerate(rows.items()):
        lines.append(f"{i},{word},{v},{a},{d}")
    dest = path / "warriner.csv"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def _write_nrc(path: Path, rows: dict[str, tuple[float, float, float]]) -> Path:
    lines = ["term\tvalence\tarousal\tdominance"]
    for word, (v, a, d) in rows.items():
        lines.append(f"{word}\t{v}\t{a}\t{d}")
    dest = path / "nrc.txt"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def _write_lancaster(path: Path, rows: dict[str, tuple[float, float]]) -> Path:
    lines = ["Word,Max_strength.perceptual,Max_strength.action"]
    for word, (perc, action) in rows.items():
        lines.append(f"{word},{perc},{action}")
    dest = path / "lancaster.csv"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def _write_subtlex(path: Path, rows: dict[str, tuple[float, float]]) -> Path:
    lines = ["Word\tSUBTLWF\tSUBTLCD"]
    for word, (wf, cd) in rows.items():
        lines.append(f"{word}\t{wf}\t{cd}")
    dest = path / "subtlex.tsv"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


#: mrc2.dct's fixed-width numeric header, per textgrader.lexicons._MRC_FIELDS.
def _mrc_line(word: str, *, fam=0, conc=0, imag=0, meanc=0, meanp=0, aoa=0) -> str:
    nlet = f"{len(word):02d}"
    header = (nlet + "00" + "0" +          # nlet, nphon, nsyl
             "00000" + "00" + "000" +      # kf_freq, kf_ncats, kf_nsamp
             "000000" + "0000" +           # tl_freq, brown_freq
             f"{fam:03d}" + f"{conc:03d}" + f"{imag:03d}" +
             f"{meanc:03d}" + f"{meanp:03d}" + f"{aoa:03d}")
    assert len(header) == 43, len(header)
    flags = " " * 8
    assert len(header + flags) == 51
    return header + flags + f"{word}|||  "


def _write_mrc(path: Path, entries: list[str]) -> Path:
    dest = path / "mrc2.dct"
    dest.write_text("\n".join(entries) + "\n", encoding="latin-1")
    return dest


def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text, comparison_unit="book")


def _measure(text: str, config: dict) -> dict[str, dict]:
    return {f["metric_id"]: f for f in suite.measure(_analysis(text), config=config)}


def _all_off(**overrides) -> dict:
    features = {name: False for name in suite.DEFAULT_FEATURES}
    features.update(overrides)
    return {"features": features}


# ------------------------------------------------------- lexicons.py parsers

def test_parse_warriner_reads_exact_values(tmp_path):
    path = _write_warriner(tmp_path, {"joy": (8.8, 6.1, 6.4), "dread": (1.2, 6.9, 2.1)})
    table, reason = lexicons._parse_warriner_vad(path)
    assert reason is None
    assert table["joy"] == {"valence": 8.8, "arousal": 6.1, "dominance": 6.4}
    assert table["dread"] == {"valence": 1.2, "arousal": 6.9, "dominance": 2.1}


def test_parse_nrc_reads_exact_values(tmp_path):
    path = _write_nrc(tmp_path, {"happy": (0.98, 0.4, 0.3), "death": (-0.9, 0.1, -0.2)})
    table, reason = lexicons._parse_nrc_vad(path)
    assert reason is None
    assert table["happy"] == {"valence": 0.98, "arousal": 0.4, "dominance": 0.3}
    assert table["death"]["valence"] == -0.9


def test_parse_lancaster_reads_exact_values(tmp_path):
    path = _write_lancaster(tmp_path, {"loud": (4.7, 1.2), "abstract": (0.5, 0.1)})
    table, reason = lexicons._parse_lancaster_sensorimotor(path)
    assert reason is None
    assert table["loud"] == {"perceptual_strength": 4.7, "action_strength": 1.2}


def test_parse_subtlex_zipf_formula_is_exact(tmp_path):
    # zipf = log10(freq_per_million) + 3, so a round freq_per_million gives an
    # exactly checkable Zipf value.
    path = _write_subtlex(tmp_path, {"common": (1000.0, 50.0), "rare": (0.01, 0.5)})
    table, reason = lexicons._parse_subtlex_us(path)
    assert reason is None
    assert table["common"]["zipf"] == pytest.approx(6.0)
    assert table["common"]["contextual_diversity"] == 50.0
    assert table["rare"]["zipf"] == pytest.approx(1.0)


def test_parse_mrc_reads_exact_values_and_treats_zero_as_missing(tmp_path):
    entries = [
        _mrc_line("ROCK", fam=500, conc=620, imag=580, meanc=400, meanp=410, aoa=250),
        _mrc_line("TRUTH", fam=480, conc=180, imag=220, meanc=390, meanp=0, aoa=520),
        _mrc_line("UNRATEDWORD"),  # every rating field is 0 -> entirely missing
    ]
    path = _write_mrc(tmp_path, entries)
    table, reason = lexicons._parse_mrc(path)
    assert reason is None
    assert table["rock"] == {"familiarity": 500.0, "concreteness": 620.0, "imageability": 580.0,
                             "meaningfulness_colorado": 400.0, "meaningfulness_paivio": 410.0,
                             "aoa": 250.0}
    # meanp=0 in the source dictionary means "not rated": absent, not zero.
    assert "meaningfulness_paivio" not in table["truth"]
    assert table["truth"]["concreteness"] == 180.0
    assert "unratedword" not in table


def test_parse_mrc_apostrophe_escape(tmp_path):
    path = _write_mrc(tmp_path, [_mrc_line("&TIS", conc=300).replace("&TIS", "&TIS")])
    table, reason = lexicons._parse_mrc(path)
    assert reason is None
    assert "'tis" in table


def test_norm_table_reports_reference_quantiles_from_the_table_itself(tmp_path):
    path = _write_warriner(tmp_path, {f"w{i}": (float(i), 5.0, 5.0) for i in range(1, 11)})
    table, reason, meta = lexicons.norm_table("warriner_vad", path)
    assert reason is None
    ref = meta["dimension_reference"]["valence"]
    assert ref["count"] == 10
    assert ref["p10"] == pytest.approx(quantile(list(range(1, 11)), 0.10))
    assert ref["p90"] == pytest.approx(quantile(list(range(1, 11)), 0.90))


def test_norm_table_missing_path_gives_reason_not_crash():
    table, reason, meta = lexicons.norm_table("warriner_vad", "/no/such/file.csv")
    assert table is None
    assert "does not exist" in reason
    assert meta["resource"] == "warriner_vad"


def test_score_tokens_aligns_positionally_and_fills_none(tmp_path):
    path = _write_nrc(tmp_path, {"happy": (0.9, 0.1, 0.1)})
    scores = lexicons.score_tokens("nrc_vad", ["happy", "unknownword", "happy"],
                                   dimension="valence", path=path)
    assert scores == [0.9, None, 0.9]


def test_unknown_resource_name_is_a_clean_reason():
    table, reason, meta = lexicons.norm_table("not_a_real_resource")
    assert table is None
    assert "unknown" in reason


# ------------------------------------------------------------- suite fixtures

def test_positive_vs_negative_vad_fixture_exact_values(tmp_path):
    """Task fixture: positive vs negative VAD -- exact values, not just sign."""

    warriner = _write_warriner(tmp_path, {"joy": (8.8, 6.0, 6.0), "dread": (1.2, 6.0, 2.0)})
    config = _all_off(warriner_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner)}

    positive = _measure("Joy joy joy joy.", config)
    negative = _measure("Dread dread dread dread.", config)

    assert positive["lexical.norm_warriner_valence_token_mean"]["value"] == pytest.approx(8.8)
    assert negative["lexical.norm_warriner_valence_token_mean"]["value"] == pytest.approx(1.2)
    assert (positive["lexical.norm_warriner_valence_token_mean"]["value"] >
           negative["lexical.norm_warriner_valence_token_mean"]["value"])


def test_concrete_vs_abstract_fixture_exact_values(tmp_path):
    """Task fixture: concrete vs abstract words -- exact values via MRC."""

    entries = [_mrc_line("ROCK", conc=620), _mrc_line("TRUTH", conc=180)]
    mrc = _write_mrc(tmp_path, entries)
    config = _all_off(mrc=True)
    config["resource_paths"] = {"mrc": str(mrc)}

    concrete = _measure("Rock rock rock.", config)
    abstract = _measure("Truth truth truth.", config)

    assert concrete["lexical.norm_mrc_concreteness_token_mean"]["value"] == pytest.approx(620.0)
    assert abstract["lexical.norm_mrc_concreteness_token_mean"]["value"] == pytest.approx(180.0)


def test_simple_vs_rare_vocabulary_fixture_exact_values(tmp_path):
    """Task fixture: high-frequency/simple vs rare vocabulary -- exact Zipf."""

    subtlex = _write_subtlex(tmp_path, {"the": (30000.0, 100.0), "defenestrate": (0.02, 1.0)})
    config = _all_off(subtlex=True)
    config["resource_paths"] = {"subtlex_us": str(subtlex)}

    simple = _measure("The the the the.", config)
    rare = _measure("Defenestrate defenestrate.", config)

    simple_zipf = simple["lexical.norm_subtlex_zipf_token_mean"]["value"]
    rare_zipf = rare["lexical.norm_subtlex_zipf_token_mean"]["value"]
    assert simple_zipf == pytest.approx(math.log10(30000.0) + 3.0)
    assert rare_zipf == pytest.approx(math.log10(0.02) + 3.0)
    assert simple_zipf > rare_zipf


def test_low_coverage_jargon_fixture_reports_exact_coverage(tmp_path):
    """Task fixture: low-coverage jargon/name-heavy text."""

    warriner = _write_warriner(tmp_path, {"happy": (8.0, 5.0, 5.0)})
    config = _all_off(warriner_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner)}

    # 8 tokens total, exactly 2 ("happy") are in the tiny table.
    text = "Zoraxxian Blipthorp Quendrizal happy Fenwick Ostrogoth Yttrium happy"
    result = _measure(text, config)
    finding = result["lexical.norm_warriner_valence_token_mean"]
    assert finding["distribution"]["total_tokens"] == 8
    assert finding["distribution"]["matched_tokens"] == 2
    assert finding["distribution"]["coverage_pct"] == pytest.approx(25.0)
    assert finding["value"] == pytest.approx(8.0)


def test_token_weighted_and_type_weighted_differ_with_exact_values(tmp_path):
    """Task fixture: token-weighted vs type-weighted must differ when expected."""

    warriner = _write_warriner(tmp_path, {"sad": (2.0, 5.0, 5.0), "joy": (9.0, 5.0, 5.0)})
    config = _all_off(warriner_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner)}

    result = _measure("Sad sad sad sad joy.", config)
    token_mean = result["lexical.norm_warriner_valence_token_mean"]["value"]
    type_mean = result["lexical.norm_warriner_valence_type_mean"]["value"]

    # token-weighted: (4*2.0 + 1*9.0) / 5 = 3.4
    assert token_mean == pytest.approx(3.4)
    # type-weighted: mean of the two DISTINCT ratings = (2.0 + 9.0) / 2 = 5.5
    assert type_mean == pytest.approx(5.5)
    assert token_mean != pytest.approx(type_mean)


def test_coverage_travels_with_every_value_low_vs_high(tmp_path):
    """A mean over a fraction of the words must never look like a full-coverage mean."""

    warriner = _write_warriner(tmp_path, {"happy": (8.0, 5.0, 5.0)})
    config = _all_off(warriner_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner)}

    low = _measure("Happy zzznotfound zzznotfound zzznotfound zzznotfound", config)
    high = _measure("Happy happy happy happy happy", config)

    low_dist = low["lexical.norm_warriner_valence_token_mean"]["distribution"]
    high_dist = high["lexical.norm_warriner_valence_token_mean"]["distribution"]
    assert low_dist["coverage_pct"] == pytest.approx(20.0)
    assert high_dist["coverage_pct"] == pytest.approx(100.0)
    # Both means happen to equal 8.0 (every covered token IS "happy"), which is
    # exactly why coverage must be reported alongside the mean: a reader
    # cannot tell 20% coverage from 100% coverage by the mean value alone.
    assert low["lexical.norm_warriner_valence_token_mean"]["value"] == pytest.approx(8.0)
    assert high["lexical.norm_warriner_valence_token_mean"]["value"] == pytest.approx(8.0)


#: Ten distinct alphabetic words (the tokenizer excludes digits by design --
#: ``[^\W\d_]+`` in textgrader/text.py -- so "w1".."w10" would all collapse
#: to the single token "w"; see textgrader.text.WORD_RE).
_TEN_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet")


def test_tail_shares_use_the_resource_table_own_reference(tmp_path):
    rows = {word: (float(i), 5.0, 5.0) for i, word in enumerate(_TEN_WORDS, start=1)}
    warriner = _write_warriner(tmp_path, rows)
    config = _all_off(warriner_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner)}

    text = " ".join(rows.keys())
    result = _measure(text, config)
    dist = result["lexical.norm_warriner_valence_token_mean"]["distribution"]
    expected_p10 = quantile(list(range(1, 11)), 0.10)
    expected_p90 = quantile(list(range(1, 11)), 0.90)
    assert dist["low_tail_reference"] == pytest.approx(expected_p10)
    assert dist["high_tail_reference"] == pytest.approx(expected_p90)
    assert dist["low_tail_share"] is not None and dist["high_tail_share"] is not None


def test_resource_metadata_travels_with_every_finding(tmp_path):
    warriner = _write_warriner(tmp_path, {"happy": (8.0, 5.0, 5.0)})
    config = _all_off(warriner_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner)}

    result = _measure("Happy happy.", config)
    dist = result["lexical.norm_warriner_valence_token_mean"]["distribution"]
    assert dist["resource"] == "warriner_vad"
    assert dist["version"]
    assert dist["source_url"]
    assert dist["sha256"] and len(dist["sha256"]) == 64
    assert dist["entry_count"] == 1


def test_sentence_paragraph_drift_and_dialogue_breakdown_present(tmp_path):
    warriner = _write_warriner(tmp_path, {
        "happy": (9.0, 5.0, 5.0), "sad": (1.0, 5.0, 5.0), "calm": (5.0, 5.0, 5.0),
    })
    config = _all_off(warriner_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner)}

    text = ('"Happy happy happy," she said. Calm calm calm.\n\n'
           'Sad sad sad. "Calm," he said again.')
    result = _measure(text, config)
    dist = result["lexical.norm_warriner_valence_token_mean"]["distribution"]

    assert dist["sentences_total"] >= 2
    assert dist["sentence_level"] is not None
    assert dist["paragraphs_total"] == 2
    assert dist["paragraph_level"] is not None
    assert dist["between_paragraph_variance"] is not None
    assert dist["early_vs_late_drift"] is not None
    assert set(dist["early_vs_late_drift"]) == {"early_mean", "late_mean", "delta",
                                                "early_coverage_pct", "late_coverage_pct"}
    assert dist["dialogue_vs_narration"] is not None
    # "happy" only appears in the quoted dialogue; "sad" only in narration.
    assert dist["dialogue_vs_narration"]["dialogue_mean"] > dist["dialogue_vs_narration"]["narration_mean"]


def test_missing_resource_disables_only_its_own_findings(tmp_path):
    warriner = _write_warriner(tmp_path, {"happy": (8.0, 5.0, 5.0)})
    nrc = _write_nrc(tmp_path, {"happy": (0.9, 0.1, 0.1)})
    config = _all_off(warriner_vad=True, nrc_vad=True)
    config["resource_paths"] = {
        "warriner_vad": str(tmp_path / "does_not_exist.csv"),  # broken on purpose
        "nrc_vad": str(nrc),
    }
    result = _measure("Happy happy.", config)

    assert result["lexical.norm_warriner_valence_token_mean"]["value"] is None
    assert result["lexical.norm_warriner_valence_token_mean"]["warning"]
    # nrc_vad is a completely independent resource and is unaffected.
    assert result["lexical.norm_nrc_valence_token_mean"]["value"] == pytest.approx(0.9)
    assert result["lexical.norm_nrc_valence_token_mean"]["warning"] is None


def test_features_toggle_independently(tmp_path):
    warriner = _write_warriner(tmp_path, {"happy": (8.0, 5.0, 5.0)})
    nrc = _write_nrc(tmp_path, {"happy": (0.9, 0.1, 0.1)})
    config = _all_off(warriner_vad=True, nrc_vad=False)
    config["resource_paths"] = {"warriner_vad": str(warriner), "nrc_vad": str(nrc)}
    result = _measure("Happy happy.", config)
    assert "lexical.norm_warriner_valence_token_mean" in result
    assert "lexical.norm_nrc_valence_token_mean" not in result


def test_independent_vad_resources_kept_separate(tmp_path):
    """Warriner and NRC both measure VAD; both must stay their own findings."""

    warriner = _write_warriner(tmp_path, {"happy": (8.47, 5.0, 5.0)})
    nrc = _write_nrc(tmp_path, {"happy": (0.985, 0.1, 0.1)})
    config = _all_off(warriner_vad=True, nrc_vad=True)
    config["resource_paths"] = {"warriner_vad": str(warriner), "nrc_vad": str(nrc)}
    result = _measure("Happy happy happy.", config)
    warriner_value = result["lexical.norm_warriner_valence_token_mean"]["value"]
    nrc_value = result["lexical.norm_nrc_valence_token_mean"]["value"]
    # Different scales (1-9 vs -1..1): the two must not be reconciled/rescaled
    # into looking alike.
    assert warriner_value == pytest.approx(8.47)
    assert nrc_value == pytest.approx(0.985)


def test_frequency_source_agreement_reports_correlation(tmp_path):
    subtlex = _write_subtlex(tmp_path, {word: (freq, 50.0) for word, freq in
                                        [("alpha", 30000.0), ("beta", 100.0), ("gamma", 1.0),
                                         ("delta", 0.1), ("epsilon", 5000.0)]})
    config = _all_off(frequency_source_agreement=True)
    config["resource_paths"] = {"subtlex_us": str(subtlex)}
    result = _measure("alpha beta gamma delta epsilon", config)
    agreement = result["lexical.norm_frequency_source_agreement"]
    if agreement["value"] is None:
        pytest.skip(f"wordfreq unavailable: {agreement['warning']}")
    assert -1.0 <= agreement["value"] <= 1.0
    assert agreement["distribution"]["compared_word_types"] == 5


def test_resource_versions_summary_present():
    result = _measure("Hello world.", _all_off())
    summary = result["lexical.norm_resource_versions"]
    assert summary["value"] == len(lexicons.list_resources()) - sum(
        1 for name in lexicons.list_resources() if not lexicons.resource_available(name)
    ) or summary["value"] >= 0  # available count is always a non-negative int
    assert "brysbaert_concreteness" in summary["distribution"]["resources"]
    assert "english_lexicon_project" in summary["distribution"]["resources"]
    assert summary["distribution"]["resources"]["english_lexicon_project"]["available"] is False


def test_english_lexicon_project_and_celex_are_registered_unavailable():
    for name in ("english_lexicon_project", "celex"):
        assert not lexicons.resource_available(name)
        table, reason, _meta = lexicons.norm_table(name)
        assert table is None
        assert reason


def test_lexdiv_and_taaled_crosscheck_do_not_crash_on_short_text():
    result = _measure("The quick brown fox jumps over the lazy dog repeatedly and then rests.",
                      _all_off(lexdiv_crosscheck=True, taaled_crosscheck=True))
    for suffix in ("mtld", "hdd", "mattr", "msttr"):
        assert f"lexical.norm_lexdiv_{suffix}" in result
    # taaled's own MATTR is deliberately not called -- see _taaled_crosscheck's
    # docstring for the measured O(tokens * window) cost that made it
    # impractical; lexical_diversity's MATTR above already cross-checks that
    # statistic much more cheaply.
    for suffix in ("mtld", "hdd", "msttr", "rttr", "maas"):
        assert f"lexical.norm_taaled_{suffix}" in result
    assert "lexical.norm_taaled_mattr" not in result


def test_taaled_crosscheck_is_fast_on_a_realistic_token_count():
    """Regression test for the O(tokens*window) MATTR cost this suite avoids
    calling (see _taaled_crosscheck's docstring): the whole cross-check over
    a several-thousand-token document must stay well under a second."""

    import time

    text = " ".join(f"word{i % 400}" for i in range(8000))
    start = time.time()
    result = _measure(text, _all_off(taaled_crosscheck=True))
    elapsed = time.time() - start
    value = result["lexical.norm_taaled_mtld"]["value"]
    if value is None:
        pytest.skip(f"taaled unavailable: {result['lexical.norm_taaled_mtld']['warning']}")
    assert elapsed < 5.0, f"taaled cross-check took {elapsed:.2f}s, expected well under 5s"


def test_empty_document_does_not_crash_and_reports_no_coverage():
    result = _measure("", _all_off(warriner_vad=True))
    finding = result["lexical.norm_warriner_valence_token_mean"]
    assert finding["value"] is None


# --------------------------------------------------- real cached resources
#
# These skip cleanly (never fail) when the resource has not been downloaded
# in this environment -- see textgrader/lexicons.py's module docstring for
# `python -m textgrader.lexicons download <name>`.

def _cached(name: str) -> bool:
    _table, reason, _meta = lexicons.norm_table(name)
    return reason is None


requires_real_concreteness = pytest.mark.skipif(
    not _cached("brysbaert_concreteness"), reason="brysbaert_concreteness not cached")
requires_real_aoa = pytest.mark.skipif(not _cached("kuperman_aoa"), reason="kuperman_aoa not cached")
requires_real_warriner = pytest.mark.skipif(not _cached("warriner_vad"), reason="warriner_vad not cached")
requires_real_nrc = pytest.mark.skipif(not _cached("nrc_vad"), reason="nrc_vad not cached")


@requires_real_concreteness
def test_real_concreteness_separates_concrete_from_abstract():
    table, _reason, _meta = lexicons.norm_table("brysbaert_concreteness")
    assert table["dog"]["concreteness"] > table["justice"]["concreteness"]


@requires_real_aoa
def test_real_aoa_separates_early_from_late_acquired_words():
    table, _reason, _meta = lexicons.norm_table("kuperman_aoa")
    assert table["dog"]["aoa"] < table["jurisprudence"]["aoa"]


@requires_real_warriner
@requires_real_nrc
def test_real_warriner_and_nrc_agree_on_direction_but_are_kept_separate():
    warriner, _r1, _m1 = lexicons.norm_table("warriner_vad")
    nrc, _r2, _m2 = lexicons.norm_table("nrc_vad")
    assert warriner["happy"]["valence"] > warriner["death"]["valence"]
    assert nrc["happy"]["valence"] > nrc["death"]["valence"]
    # Same direction, but never the same scale/number.
    assert warriner["happy"]["valence"] != nrc["happy"]["valence"]


# ------------------------------------------------------------- real corpus
#
# The 50-book reference corpus this task's instructions point at. Skips
# cleanly wherever that scratch directory does not exist (any environment
# other than this task's own).

CORPUS_DIR = Path(
    "/tmp/claude-0/-home-user-TextGrader/6d83dbcd-d4fb-5dbb-b487-24be17a4fb81/scratchpad/corpus")


@pytest.mark.skipif(not CORPUS_DIR.is_dir(), reason="reference corpus not present in this environment")
@requires_real_concreteness
@requires_real_aoa
def test_real_books_children_vs_adult_aoa_and_concreteness():
    child_path = CORPUS_DIR / "gutenberg-11-alice-s-adventures-in-wonderland.txt"
    adult_path = CORPUS_DIR / "gutenberg-974-the-secret-agent-a-simple-tale.txt"
    if not (child_path.exists() and adult_path.exists()):
        pytest.skip("expected reference books not present")

    config = _all_off(concreteness=True, age_of_acquisition=True)
    child = {f["metric_id"]: f for f in suite.measure(
        DocumentAnalysis.from_path(child_path, comparison_unit="book"), config=config)}
    adult = {f["metric_id"]: f for f in suite.measure(
        DocumentAnalysis.from_path(adult_path, comparison_unit="book"), config=config)}

    child_aoa = child["lexical.norm_aoa_token_mean"]["value"]
    adult_aoa = adult["lexical.norm_aoa_token_mean"]["value"]
    child_conc = child["lexical.norm_concreteness_token_mean"]["value"]
    adult_conc = adult["lexical.norm_concreteness_token_mean"]["value"]
    # Reported whichever way it comes out, per the task instructions -- this
    # assertion documents the observed direction rather than assuming it.
    print(f"\nAoA:          Alice={child_aoa:.3f}  SecretAgent={adult_aoa:.3f}")
    print(f"Concreteness: Alice={child_conc:.3f}  SecretAgent={adult_conc:.3f}")
    assert child_aoa is not None and adult_aoa is not None
    assert child_conc is not None and adult_conc is not None
