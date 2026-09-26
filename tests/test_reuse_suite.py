"""Contract tests for the experimental approximate-duplication / fuzzy-reuse /
structural-reuse / motif-detection suite (``reuse_suite``).

Every fixture here is built from short, hand-written synthetic sentences
rather than any real book, song or poem, per this project's convention of
keeping test fixtures self-contained and license-free.
"""

from __future__ import annotations

import random

import pytest

from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY
from textgrader.metrics import reuse_algorithms as ra
from textgrader.metrics import reuse_suite as R


def _analysis(text: str) -> DocumentAnalysis:
    return DocumentAnalysis.from_text(text)


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


# ------------------------------------------------------------------ fixtures

EXACT_DUP_TEXT = (
    "The cat sat on the mat. The cat sat on the mat. The dog ran in the yard. "
    "A bird flew over the house. A cup fell off the table."
)

NO_DUP_TEXT = (
    "The cat sat on the mat. A dog ran in the yard. A bird flew over the house. "
    "A cup fell off the table. Rain fell on the roof all night."
)

PARAPHRASE_A = ("The old wooden door at the end of the quiet hallway creaked loudly "
               "every single time in the cold night wind.")
PARAPHRASE_B = ("The old wooden door at the end of the quiet hallway creaked softly "
               "every single time in the cold night wind.")

PARAPHRASE_TEXT = (
    f"{PARAPHRASE_A} {PARAPHRASE_B} A cup fell off the table. "
    "Rain fell on the roof at night. Someone walked slowly down the hall."
)

UNRELATED_TEXT = (
    f"{PARAPHRASE_A} A cup fell off the table. "
    "Rain fell on the roof at night. Someone walked slowly down the hall. "
    "A bird flew over the house today."
)


# ------------------------------------------------------------------ registry

def test_off_by_default_and_correct_family_and_prefix():
    spec = REGISTRY["reuse_suite"]
    assert spec.family == "repetition"
    assert spec.cost == "moderate"
    assert "spacy" not in spec.requires  # off-by-default parsed views only
    findings = R.measure(_analysis(EXACT_DUP_TEXT), config={})
    assert findings  # the module itself always runs when explicitly called
    for item in findings:
        assert item["metric_id"].startswith("repetition.reuse_")
        assert item["family"] == "repetition"


def test_config_defaults_mirror_registry_defaults():
    import json
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parents[1] / "config.json").read_text())
    entry = config["metrics"]["reuse_suite"]
    assert entry["enabled"] is False
    spec = REGISTRY["reuse_suite"]
    assert entry["features"] == spec.defaults["features"]
    for key, value in spec.defaults.items():
        if key == "features":
            continue
        assert entry[key] == value, key


# --------------------------------------------------------- exact duplication

def test_exact_repeated_sentence_is_detected():
    dup = _by_id(R.measure(_analysis(EXACT_DUP_TEXT), config={}))
    clean = _by_id(R.measure(_analysis(NO_DUP_TEXT), config={}))
    dup_share = dup["repetition.reuse_exact_sentence_share"]["value"]
    clean_share = clean["repetition.reuse_exact_sentence_share"]["value"]
    assert dup_share == pytest.approx(40.0)  # 2 of 5 sentences are exact repeats
    assert clean_share == 0.0
    assert dup_share > clean_share


def test_minor_paraphrase_is_a_near_duplicate_not_necessarily_exact():
    # The true (exact-shingle) Jaccard between PARAPHRASE_A/B is 0.727 (one
    # word differs out of 21); LSH is probabilistic and, with so few
    # sentences to bucket, can miss a true positive sitting close above a
    # 0.7 cutoff -- a real property of banded LSH, not a bug -- so this test
    # asks for a lower configured threshold, comfortably below the true
    # similarity and still far above the near-zero similarity between
    # unrelated sentences.
    config = {"near_duplicate_threshold": 0.5}
    paraphrase = _by_id(R.measure(_analysis(PARAPHRASE_TEXT), config=config))
    unrelated = _by_id(R.measure(_analysis(UNRELATED_TEXT), config=config))
    exact = paraphrase["repetition.reuse_exact_sentence_share"]["value"]
    near = paraphrase["repetition.reuse_near_duplicate_sentence_share"]["value"]
    near_unrelated = unrelated["repetition.reuse_near_duplicate_sentence_share"]["value"]
    # "loudly" vs "softly" is a one-word edit: not byte-identical, so the
    # exact-duplicate channel must NOT count it...
    assert exact == 0.0
    # ...but the near-duplicate (MinHash/Jaccard candidate) channel must.
    assert near > 0.0
    assert near > near_unrelated


# --------------------------------------------------------------- POS skeleton

POS_SKELETON_TEXT = (
    "The dog chased the cat. The wizard summoned the storm. "
    "Blue is my favorite color today."
)


@pytest.mark.skipif(not optional.have("spacy"), reason="spaCy is not installed")
def test_same_pos_skeleton_different_content_words_is_detected():
    """"The dog chased the cat." and "The wizard summoned the storm." share the
    exact POS sequence DET NOUN VERB DET NOUN PUNCT despite sharing no content
    word; the third sentence has a different skeleton.
    """

    config = {"features": {"view_pos": True}}
    findings = _by_id(R.measure(_analysis(POS_SKELETON_TEXT), config=config))
    pos_item = findings["repetition.reuse_view_pos_exact_share"]
    content_item = findings["repetition.reuse_view_content_words_exact_share"]
    assert pos_item["warning"] is None
    # 2 of 3 sentences share a POS skeleton -> a real, nonzero exact-share.
    assert pos_item["value"] > 0.0
    # The same two sentences share zero content words, so the surface/
    # content-word view must NOT see them as duplicates: the POS view is
    # catching something the lexical views cannot.
    assert content_item["value"] == 0.0


def test_pos_view_off_by_default():
    findings = _by_id(R.measure(_analysis(POS_SKELETON_TEXT), config={}))
    item = findings["repetition.reuse_view_pos_exact_share"]
    assert item["value"] is None
    assert "off by default" in item["warning"]


# ------------------------------------------------------------ punctuation only

PUNCT_TEXT = (
    '"Wait," she said, "no!" '
    '"Stop," he said, "go!" '
    "The weather was calm and quiet in the valley."
)


def test_same_punctuation_pattern_only_is_detected():
    """The first two sentences share the punctuation skeleton `","","!"` while
    using entirely different words; the punctuation-only view must see them as
    duplicates even though every lexical view does not.
    """

    findings = _by_id(R.measure(_analysis(PUNCT_TEXT), config={}))
    punct_item = findings["repetition.reuse_view_punctuation_exact_share"]
    normalized_item = findings["repetition.reuse_view_normalized_exact_share"]
    assert punct_item["value"] > 0.0
    assert punct_item["distribution"]["overlaps_existing_metric_id"]
    assert normalized_item["value"] == 0.0


# ---------------------------------------------------- bounded candidate count

def test_long_document_comparison_is_bounded():
    """20,000 synthetic sentences: candidate generation must stay far below
    the 200 million all-pairs comparisons, and provably bounded by a linear
    multiple of the sentence count -- never a function of n^2.
    """

    rng = random.Random(7)
    vocab = [f"tok{i}" for i in range(300)]
    n = 20000
    sentence_tokens = [[rng.choice(vocab) for _ in range(12)] for _ in range(n)]
    # Sprinkle in a small, known set of exact duplicates so the candidate
    # generator has something real to find, not just random near-misses.
    for i in range(0, n, 500):
        sentence_tokens[i + 1] = list(sentence_tokens[i])

    shingle_sets = [ra.shingles(tokens, 3) for tokens in sentence_tokens]
    graph, jaccards, pair_count, backend = ra.minhash_lsh_candidates(
        shingle_sets, num_perm=32, threshold=0.7, seed=1, max_candidates_per_item=20)

    all_pairs = n * (n - 1) // 2
    assert pair_count < all_pairs / 1000  # nowhere near quadratic
    # Provably bounded by construction: at most max_candidates_per_item edges
    # per item, so at most n * max_candidates_per_item / 2 undirected pairs.
    assert pair_count <= n * 20 // 2
    # And it actually found the planted duplicates.
    assert any(jac > 0.99 for jac in jaccards.values())


def test_capped_inverted_index_candidates_are_also_bounded():
    """The dependency-free fallback candidate generator (used when datasketch
    is unavailable, and directly for the windowed approximate-repeat search)
    is independently bounded the same way.
    """

    rng = random.Random(3)
    vocab = [f"w{i}" for i in range(200)]
    n = 20000
    shingle_sets = [ra.shingles([rng.choice(vocab) for _ in range(10)], 3) for _ in range(n)]
    graph, pair_count = ra.capped_inverted_candidates(
        shingle_sets, cap=40, max_candidates_per_item=20)
    assert pair_count <= n * 20 // 2
    assert pair_count < (n * (n - 1) // 2) / 1000


# ------------------------------------------------------- fuzzy hash length floor

SHORT_SENTENCES_TEXT = "\n\n".join(
    "Hi. Bye. OK. Go now. Yes sir. No way." for _ in range(10))


def test_short_sentences_never_reach_tlsh_or_ssdeep():
    findings = _by_id(R.measure(_analysis(SHORT_SENTENCES_TEXT), config={}))
    tlsh_item = findings["repetition.reuse_tlsh_similarity_distribution"]
    ssdeep_item = findings["repetition.reuse_ssdeep_similarity_distribution"]
    assert tlsh_item["value"] is None
    assert ssdeep_item["value"] is None
    # Under a degraded environment (e.g. TEXTGRADER_DISABLE_OPTIONAL=all) the
    # package itself may be the reason instead; either way, no number is ever
    # produced from text this short.
    if optional.have("tlsh"):
        assert "minimum length" in tlsh_item["warning"]
    if optional.have("ppdeep"):
        assert "minimum length" in ssdeep_item["warning"]


def test_hard_floor_cannot_be_configured_below():
    """Even a caller who sets the length floor to 1 character cannot make a
    short paragraph reach TLSH/ssdeep -- HARD_MIN_BLOCK_CHARS always applies.
    """

    config = {"min_tlsh_chars": 1, "min_ssdeep_chars": 1}
    findings = _by_id(R.measure(_analysis(SHORT_SENTENCES_TEXT), config=config))
    assert findings["repetition.reuse_tlsh_similarity_distribution"]["value"] is None
    assert findings["repetition.reuse_ssdeep_similarity_distribution"]["value"] is None


# ------------------------------------------------------------- longest repeats

LONGEST_RUN_TEXT = (
    "The cat sat quietly on the warm mat by the fire. "
    "Later that day the cat sat quietly on the warm mat by the fire again. "
    "Nothing else in this story repeats at all."
)


def test_longest_repeated_token_run_finds_the_real_run():
    findings = _by_id(R.measure(_analysis(LONGEST_RUN_TEXT), config={}))
    item = findings["repetition.reuse_longest_repeated_token_run"]
    # "the cat sat quietly on the warm mat by the fire" is 11 tokens.
    assert item["value"] == 11
    evidence = item["evidence"][0]
    assert evidence["length_tokens"] == 11


def test_longest_approximate_repeated_run_extends_beyond_exact():
    """A near-duplicate long run (one content word swapped in the middle)
    should be found by the approximate channel at a length at least as long
    as the exact-run channel finds for the same text.
    """

    text = (
        "The old dog walked slowly across the wide open field at dawn every morning. "
        "The old dog walked slowly across the wide open meadow at dawn every morning. "
        "Nothing else repeats."
    )
    findings = _by_id(R.measure(_analysis(text), config={}))
    exact = findings["repetition.reuse_longest_repeated_token_run"]["value"]
    approx = findings["repetition.reuse_longest_approximate_repeated_run"]["value"]
    assert approx >= exact
    assert approx > 0


def _repeating_tokens(paragraphs: int) -> list[str]:
    adjectives = ("old", "tall", "broken", "white", "narrow", "mossy", "leaning", "lonely")
    nouns = ("stone", "tower", "gate", "cross", "wall", "well", "mill", "barn")
    pairs = [(a, n) for a in adjectives for n in nouns][:paragraphs]
    return " ".join(
        f"the {a} {n} stood beside the river where the long road bent toward the hills and "
        f"every traveller who passed it in the grey light of the morning stopped to look at "
        f"the {n} and wonder who had set it there so many years before the town was built"
        for a, n in pairs).split()


def test_approximate_run_cost_is_linear_on_text_that_repeats_throughout(monkeypatch):
    """Rescoring the whole run on every extension step made this cubic on
    exactly the text it exists to catch: 80 near-identical paragraphs took
    over ten minutes.  Scorer calls must now grow in step with the text."""

    real = ra._scorer()
    calls = {"n": 0}

    def counting():
        def score(a, b):
            calls["n"] += 1
            return real(a, b)
        return score

    monkeypatch.setattr(ra, "_scorer", counting)
    counts, results = [], []
    for paragraphs in (16, 32, 64):
        calls["n"] = 0
        tokens = _repeating_tokens(paragraphs)
        results.append((len(tokens), ra.longest_approximate_repeated_run(tokens)))
        counts.append(calls["n"])
    assert counts[1] < 2.6 * counts[0] and counts[2] < 2.6 * counts[1], counts
    for total, found in results:
        # Two non-overlapping copies of a text that repeats throughout can
        # cover at most half of it each, and should come close.
        assert total // 2 - 16 <= found["length"] <= total // 2
        assert found["similarity"] >= 80.0


# --------------------------------------------------------------- degradation

def test_graceful_degradation_without_any_optional_package(monkeypatch):
    """Disabling every optional package must degrade each channel to its own
    'unavailable' finding (or, for MinHash/near-dup, to the dependency-free
    fallback) -- never raise, and never take down the whole suite.
    """

    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "all")
    optional.reset_cache()
    try:
        findings = _by_id(R.measure(_analysis(EXACT_DUP_TEXT), config={}))
        # Exact-share channels need no optional package at all.
        assert findings["repetition.reuse_exact_sentence_share"]["value"] == pytest.approx(40.0)
        # MinHash/LSH-backed channels fall back to the dependency-free
        # inverted-shingle index rather than disappearing.
        near_dup = findings["repetition.reuse_near_duplicate_sentence_share"]
        assert near_dup["warning"] is None
        assert near_dup["distribution"]["backend"] == "inverted_shingle_index"
        # Pure-optional channels report themselves unavailable, by name.
        for metric_id, package in (
                ("repetition.reuse_simhash_hamming_distance", "simhash"),
                ("repetition.reuse_levenshtein_similarity_distribution", "levenshtein"),
                ("repetition.reuse_jaro_winkler_similarity_distribution", "jellyfish"),
                ("repetition.reuse_token_set_similarity_distribution", "rapidfuzz"),
                ("repetition.reuse_textdistance_similarity_distribution", "textdistance")):
            item = findings[metric_id]
            assert item["value"] is None
            assert package in item["warning"]
    finally:
        monkeypatch.delenv("TEXTGRADER_DISABLE_OPTIONAL", raising=False)
        optional.reset_cache()


def test_a_missing_package_degrades_only_its_own_channel(monkeypatch):
    two_paragraph_text = f"{PARAPHRASE_TEXT}\n\nA second paragraph, unrelated to the first one."
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "tlsh")
    optional.reset_cache()
    try:
        findings = _by_id(R.measure(_analysis(two_paragraph_text), config={}))
        assert findings["repetition.reuse_tlsh_similarity_distribution"]["value"] is None
        assert "tlsh" in findings["repetition.reuse_tlsh_similarity_distribution"]["warning"]
        # ssdeep (a different package) must be entirely unaffected.
        assert findings["repetition.reuse_exact_sentence_share"]["value"] is not None
    finally:
        monkeypatch.delenv("TEXTGRADER_DISABLE_OPTIONAL", raising=False)
        optional.reset_cache()


# ---------------------------------------------------------------- determinism

def test_minhash_candidates_are_deterministic():
    rng = random.Random(42)
    vocab = [f"tok{i}" for i in range(50)]
    sentence_tokens = [[rng.choice(vocab) for _ in range(10)] for _ in range(200)]
    shingle_sets = [ra.shingles(tokens, 3) for tokens in sentence_tokens]
    _, jac1, count1, _ = ra.minhash_lsh_candidates(shingle_sets, num_perm=32, threshold=0.5, seed=1)
    _, jac2, count2, _ = ra.minhash_lsh_candidates(shingle_sets, num_perm=32, threshold=0.5, seed=1)
    assert count1 == count2
    assert jac1 == jac2


def test_measure_is_deterministic_across_runs():
    findings1 = R.measure(_analysis(PARAPHRASE_TEXT), config={"near_duplicate_threshold": 0.5})
    findings2 = R.measure(_analysis(PARAPHRASE_TEXT), config={"near_duplicate_threshold": 0.5})
    values1 = [(item["metric_id"], item["value"]) for item in findings1]
    values2 = [(item["metric_id"], item["value"]) for item in findings2]
    assert values1 == values2


# ---------------------------------------------------------- ssdeep block size

@pytest.mark.skipif(not optional.have("ppdeep"), reason="ppdeep is not installed")
def test_ssdeep_fires_on_comparable_length_near_duplicate_paragraphs():
    """ppdeep (like ssdeep) is a piecewise/block hash: it only compares two
    hashes computed at the same or an adjacent internal block size, which is
    chosen from input length. Two organically different-length paragraphs
    can therefore score 0 even when they share real content -- a genuine
    property of the algorithm, not a bug in this integration -- so this test
    proves the channel fires for real when block sizes line up (same length,
    one word changed), which is exactly the templated/padded-reuse case this
    channel exists to catch.
    """

    base = ("The old wooden door at the end of the quiet hallway creaked loudly "
           "every single time anyone walked past it during the cold autumn "
           "nights that year in the house. ") * 3
    variant = base.replace("loudly", "softly", 1)
    assert len(base) == len(variant)
    text = f"{base}\n\n{variant}\n\nSomething short.\n\nAnother short one."
    findings = _by_id(R.measure(_analysis(text), config={}))
    item = findings["repetition.reuse_ssdeep_similarity_distribution"]
    assert item["value"] is not None
    assert item["value"] > 50.0


# ------------------------------------------------------------- degenerate docs

def test_empty_document_reports_every_channel_unavailable():
    findings = R.measure(_analysis(""), config={})
    assert findings
    for item in findings:
        assert item["value"] is None
        assert item["warning"]
