"""Approximate duplication, fuzzy reuse, structural reuse, and motif detection.

TextGrader already measures *exact* repetition (:mod:`repeated_ngrams`,
:mod:`local_repetition`, :mod:`lexical_repetition_distance`,
:mod:`lexical_lemma_repetition`), whole-sentence rhetorical templates
(:mod:`discourse_constructions`), punctuation shapes (:mod:`punctuation_patterns`)
and embedding-based duplicate clusters (:mod:`semantic_clusters`). This suite
fills the gap between "identical" and "semantically similar": near-duplicate
sentences and paragraphs that differ by a word or two, sentences that share a
grammatical/punctuation/shape chassis but not vocabulary, and fuzzy-hash
similarity for longer blocks. Every finding stays ``Polarity.NEUTRAL`` (see
``grade.py``'s aggregation): a repeated refrain in a children's book and a
paraphrase-padded AI draft produce the same *numbers*, and only a human (or a
downstream, empirically validated aggregate) decides which is which.

Where a channel here overlaps an existing metric, that metric's id is named in
``distribution["overlaps_existing_metric_id"]`` rather than silently
duplicating it:

* the ``punctuation`` transformed view overlaps
  :mod:`punctuation_patterns`'s ``punct.repeated_sentence_skeleton_share`` /
  ``punct.adjacent_identical_skeleton_rate`` (same skeleton alphabet, different
  question: sentence-level *exact skeleton reuse rate*, computed once here for
  every transformed view uniformly, rather than the punctuation module's own
  richer adjacency/run/paragraph analysis, which this suite does not repeat).
* the ``content_words`` transformed view overlaps
  :mod:`local_repetition`'s ``style.local_lexical_repetition`` and
  :mod:`lexical_repetition_distance`'s ``repetition.content_word_reuse_distance``
  (those measure single-word reuse distance; this measures whole-sentence
  near-duplication after stripping function words).
* the ``lemma`` transformed view overlaps
  :mod:`lexical_lemma_repetition`'s ``repetition.lemma_reuse_distance`` (same
  lemmatized representation, sentence-level duplication instead of single-lemma
  reuse distance).
* ``repetition.reuse_template_motif_count`` /
  ``..._template_motif_coverage`` overlap :mod:`discourse_constructions`'s
  ``discourse.repeated_construction_share`` / ``discourse.top_construction_count``
  (that module abstracts a WHOLE sentence to a skeleton; this module finds
  repeated fixed-length WINDOWS of the abstraction, which can span sentence
  boundaries and catches a motif shorter than any one sentence).

Candidate generation, never all-pairs
--------------------------------------
Every comparison here is candidate-bound. Sentence- and paragraph-level near-
duplicate detection uses MinHash + LSH (:mod:`datasketch`) when installed, and
degrades to the same capped inverted-shingle index :mod:`semantic_clusters`
already uses for its blocked clustering when it is not -- never to an
all-pairs scan. The longest-approximate-repeat search shingles fixed-size,
strided token windows and runs the same capped inverted index over those.
SimHash uses its own library's ``SimhashIndex`` bucketing. See
:mod:`reuse_algorithms` for the implementations and
``tests/test_reuse_suite.py::test_long_document_comparison_is_bounded`` for a
measured candidate-pair count on a 20,000-sentence synthetic document.

Libraries, and how each was verified for real (not just imported)
-------------------------------------------------------------------
* **datasketch** -- ``MinHash(...).jaccard()`` against a shared three-of-four-
  word overlap estimated 0.6875 (true Jaccard 0.6), and
  ``MinHashLSH(threshold=0.5).query()`` returned the inserted near-duplicate
  key.
* **simhash** (the real ``iceb0y/simhash-py`` 64-bit implementation, not a
  same-named decoy) -- ``Simhash.__init__`` documents ``f`` (fingerprint
  bits) and a shingle regex/hash function; ``SimhashIndex`` is a real
  candidate-bucketing structure (deletion/permutation buckets), not a plain
  dict.
* **tlsh** (PyPI name ``py-tlsh``) -- ``tlsh.hash()`` on two 500+ character
  blocks differing by one word gave ``tlsh.diff() == 89``; the same two
  blocks against an unrelated topic gave ``257`` (bigger = more different).
  ``tlsh.hash()`` on a short string returns the literal sentinel ``"TNULL"``
  rather than a usable hash -- this suite treats that exactly like "below the
  minimum block length."
* **ppdeep** (the pure-Python, ssdeep-compatible alternative) -- its
  docstring says "Pure-Python library for computing fuzzy hashes (ssdeep)...
  Based on SpamSum by Dr. Andrew Tridgell." ``ppdeep.compare()`` of the same
  near-duplicate 500+ character blocks scored 99 (0-100, higher = more
  similar); the unrelated-topic pair scored 0. The real ``ssdeep`` package
  fails to build in this environment: ``pip install ssdeep`` raised
  ``cffi.VerificationError: CompileError: command
  '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1`` (it needs a
  system libfuzzy/C toolchain this container does not have). ssdeep/ppdeep is
  a *piecewise* hash: ``compare()`` only returns nonzero when the two hashes
  picked the same or an adjacent internal block size, itself chosen from
  input length -- two organically different-length paragraphs routinely
  score 0 even when they share content, which is why this channel's median
  sits at 0.0 on every real novel checked during development, and fires (99)
  only once the two blocks are made comparable in length. See
  ``config.json``'s ``_requires_ssdeep`` note and
  ``tests/test_reuse_suite.py::test_ssdeep_fires_on_comparable_length_near_duplicate_paragraphs``.
* **RapidFuzz** -- ``fuzz.ratio`` of a one-word edit scored ~95.7;
  ``token_sort_ratio``/``token_set_ratio`` of two reordered copies of the
  same words both scored 100.0.
* **python-Levenshtein** -- ``distance('kitten', 'sitting') == 3``,
  ``ratio(...) == 0.615...``, the textbook values.
* **Jellyfish** -- ``jaro_winkler_similarity('martha', 'marhta') ==
  0.9611...``, the textbook Winkler-paper value.
* **textdistance** -- docstring: "Compute distance between sequences. 30+
  algorithms"; ``sorensen.normalized_similarity('night', 'nacht') == 0.6``,
  the textbook Sorensen-Dice value.
* **pyahocorasick** -- an automaton built from
  ``{'he','she','his','hers'}`` correctly found all matches (including
  overlapping ones) in ``'ushers'``.

Fuzzy hashes on short strings
-------------------------------
TLSH and ppdeep both behave badly on short input: at real-world sentence
length (~100-150 characters) ppdeep's near-duplicate score for a genuine
one-word paraphrase dropped from 99 (at ~500 characters) to 38, and TLSH
refuses outright below its own complexity floor. Both channels therefore run
only over **paragraphs** (the "longer blocks" the task spec asks for), never
sentences, filtered by a configurable minimum character length
(``min_tlsh_chars`` / ``min_ssdeep_chars``) that itself can never be
configured below :data:`HARD_MIN_BLOCK_CHARS` --
``tests/test_reuse_suite.py::test_short_sentences_never_reach_fuzzy_hashes``
confirms a document made only of short sentences reports these two channels
as unavailable rather than computing on them.

Cost and the off-by-default parsed views
-------------------------------------------
``COST`` is ``moderate``: every default-on channel is dependency-free or
candidate-bound lexical work. The ``lemma``/``pos``/``dependency`` transformed
views need a real spaCy parse (part of speech disambiguates a lemma; POS/dep
tags do not exist without one) and are **off by default**, exactly like
:mod:`conversation_suite`'s ``pos_convergence`` feature -- turning one on
forces this document's shared spaCy parse (shared with every other ``parse``-
cost metric already enabled; itself the one-time cost the whole codebase
already amortizes), not a second one. ``spacy`` is deliberately left out of
this suite's ``REQUIRES``, so enabling one of these three views does not flip
the suite's registry cost or the corpus builder's "does this need
--parse-metrics" decision for the rest of the suite -- see this module's
``config.json`` entry's ``_requires_pos_dependency_lemma_views`` note.
"""

from __future__ import annotations

import importlib.metadata
import re
from collections import Counter
from typing import Any, Mapping, Sequence

from .. import text as textlib
from ..document import DocumentAnalysis
from ..optional import require
from ..stats import summarize
from .common import MODERATE, finding, option, rate
from .function_words import FUNCTION
from .lexical_repetition_distance import STOPWORDS as _LEXICAL_STOPWORDS
from .punctuation_patterns import skeleton as _punctuation_skeleton
from . import reuse_algorithms as ra

FAMILY = "repetition"
COST = MODERATE
# Deliberately excludes "spacy" and "sentence_transformers": every default-on
# channel below needs neither. See the module docstring's "Cost" section.
REQUIRES: tuple[str, ...] = ("datasketch", "rapidfuzz", "levenshtein", "jellyfish",
                            "textdistance", "simhash", "tlsh", "ppdeep", "ahocorasick")
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

FUNCTION_SET = frozenset(FUNCTION)
STOPWORDS = _LEXICAL_STOPWORDS
CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

_LIB_DISTRIBUTIONS = {
    "datasketch": "datasketch", "rapidfuzz": "rapidfuzz", "levenshtein": "Levenshtein",
    "jellyfish": "jellyfish", "textdistance": "textdistance", "simhash": "simhash",
    "tlsh": "py-tlsh", "ppdeep": "ppdeep", "ahocorasick": "pyahocorasick",
}

DEFAULT_FEATURES: dict[str, bool] = {
    "sentence_level": True,
    "paragraph_level": True,
    "simhash": True,
    "edit_distance_family": True,
    "textdistance_crosscheck": True,
    "tlsh": True,
    "ssdeep": True,
    "longest_repeated_run": True,
    "longest_approximate_run": True,
    "template_motifs": True,
    "view_normalized": True,
    "view_function_words": True,
    "view_stopwords": True,
    "view_punctuation": True,
    "view_word_shape": True,
    "view_content_words": True,
    # Off by default: force/share a spaCy parse. See the module docstring.
    "view_lemma": False,
    "view_pos": False,
    "view_dependency": False,
}

DEFAULT_SHINGLE_SIZE = 3
DEFAULT_MINHASH_NUM_PERM = 32
DEFAULT_MINHASH_SEED = 1
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.7
DEFAULT_NEAR_DUPLICATE_THRESHOLDS = (0.6, 0.7, 0.8, 0.9)
DEFAULT_MAX_CANDIDATES_PER_ITEM = 20
DEFAULT_MAX_SENTENCES = 20000
DEFAULT_MAX_PARAGRAPHS = 8000
DEFAULT_MIN_TLSH_CHARS = 300
DEFAULT_MIN_SSDEEP_CHARS = 200
#: Never overridden lower by config -- the hard floor behind "short sentences
#: must not abuse TLSH/ssdeep."
HARD_MIN_BLOCK_CHARS = 40
DEFAULT_APPROX_WINDOW_TOKENS = 8
DEFAULT_APPROX_STRIDE_TOKENS = 4
DEFAULT_APPROX_SIMILARITY_THRESHOLD = 80.0
DEFAULT_APPROX_MAX_WINDOWS = 20000
DEFAULT_APPROX_EXTEND_STEP = 4
DEFAULT_TEMPLATE_MOTIF_WINDOW = 5
DEFAULT_TEMPLATE_MOTIF_MIN_COUNT = 3
DEFAULT_SIMHASH_SHINGLE_SIZE = 4
DEFAULT_SIMHASH_INDEX_K = 3
DEFAULT_MAX_REPORTED = 15
#: Candidate generation runs at this LOOSER threshold, separate from
#: ``near_duplicate_threshold`` (which only FILTERS the resulting candidate
#: pairs for the near-duplicate share/gap metrics). Generating candidates at
#: the strict near-duplicate threshold directly would make every
#: "nearest-neighbor similarity among candidates" channel self-selecting: by
#: construction almost every surviving candidate would already be near the
#: near-duplicate ceiling, so the whole distribution reads as a flat 100%
#: regardless of the document -- measured directly on this suite's own
#: development corpus (Peter Pan, Alice, Anne of Green Gables, Dracula,
#: Little Women): every one of the four RapidFuzz/Levenshtein/Jaro-
#: Winkler/textdistance channels' median sat at exactly 100.0 on all five
#: books before this separation was introduced. A loose candidate net still
#: correctly finds every true near-duplicate (a lower LSH threshold can only
#: raise recall), and now also carries the weaker matches that give the
#: distribution real, book-dependent shape.
DEFAULT_CANDIDATE_THRESHOLD = 0.3


def _feature(config: Mapping[str, Any] | None, name: str) -> bool:
    """One ``features.<name>`` flag, defaulting individually per
    :data:`DEFAULT_FEATURES` -- see ``anomaly_suite._feature`` for why each
    flag needs its own default rather than requiring the whole map re-sent.
    """

    features = option(config, "features", {})
    default = DEFAULT_FEATURES.get(name, True)
    if not isinstance(features, Mapping):
        return default
    value = features.get(name, default)
    return default if value is None else bool(value)


def _library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name, distribution_name in _LIB_DISTRIBUTIONS.items():
        module, reason = require(name)
        if module is None:
            out[name] = None
            continue
        version = getattr(module, "__version__", None)
        if version is None:
            try:
                version = importlib.metadata.version(distribution_name)
            except Exception:  # pragma: no cover - metadata not installed
                version = "unknown"
        out[name] = version
    return out


# ------------------------------------------------------------- small helpers

def _shape_finding(metric_id: str, name: str, values: Sequence[float], unit: str, *,
                   sample_size: int | None = None, min_sample: int = MIN_SAMPLE,
                   distribution_extra: Mapping[str, Any] | None = None,
                   evidence: Sequence[Mapping[str, Any]] | None = None,
                   warning: str | None = None,
                   aggregation: str = "headline is the median of one best-match value per "
                                      "unit that had at least one candidate") -> dict[str, Any]:
    """A :func:`shape`-style finding (median + full distribution) that also
    accepts extra settings/version metadata to merge into ``distribution``.
    """

    clean = [float(v) for v in values if v is not None]
    if not clean:
        dist = dict(distribution_extra) if distribution_extra else {}
        dist["aggregation"] = aggregation
        return finding(metric_id, name, None, unit, family=FAMILY,
                       sample_size=sample_size if sample_size is not None else 0,
                       min_sample=min_sample, warning=warning or "no values to summarize",
                       distribution=dist)
    summary = summarize(clean)
    dist = dict(summary)
    if distribution_extra:
        dist.update(distribution_extra)
    dist["aggregation"] = aggregation
    return finding(metric_id, name, summary.get("median"), unit, family=FAMILY,
                   sample_size=sample_size if sample_size is not None else summary.get("count"),
                   min_sample=min_sample, distribution=dist, evidence=evidence, warning=warning)


def _exact_share(view_strings: Sequence[str]) -> tuple[float | None, int, int, Counter]:
    total = len(view_strings)
    counts = Counter(s for s in view_strings if s)
    if not total:
        return None, 0, 0, counts
    duplicated = sum(c for c in counts.values() if c > 1)
    return rate(duplicated, total), total, len(counts), counts


def _view_exact_finding(metric_id: str, name: str, view_strings: Sequence[str], *,
                        overlaps: Sequence[str] | None = None,
                        min_sample: int = MIN_SAMPLE,
                        no_data_warning: str = "no sentences to measure") -> dict[str, Any]:
    share, total, distinct, counts = _exact_share(view_strings)
    if not total:
        return finding(metric_id, name, None, "percent", family=FAMILY, sample_size=0,
                       min_sample=min_sample, warning=no_data_warning)
    dist: dict[str, Any] = {
        "distinct_forms": distinct,
        "aggregation": "share of units whose transformed-view string exactly matches at "
                      "least one other unit's",
    }
    if overlaps:
        dist["overlaps_existing_metric_id"] = list(overlaps)
    evidence = [{"form": form[:120], "count": count}
               for form, count in counts.most_common(DEFAULT_MAX_REPORTED) if count > 1]
    return finding(metric_id, name, share, "percent", family=FAMILY, sample_size=total,
                   min_sample=min_sample, distribution=dist, evidence=evidence or None)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _function_word_skeleton(tokens: Sequence[str]) -> list[str]:
    return [token if token in FUNCTION_SET else "_" for token in tokens]


# --------------------------------------------------------- transformed views

def _view_strings_lexical(name: str, raw_texts: Sequence[str],
                          word_tokens: Sequence[list[str]],
                          lower_tokens: Sequence[list[str]]) -> list[str]:
    if name == "normalized":
        return [" ".join(tokens) for tokens in lower_tokens]
    if name == "function_words":
        return [" ".join(t for t in tokens if t in FUNCTION_SET) for tokens in lower_tokens]
    if name == "stopwords":
        return [" ".join(t for t in tokens if t in STOPWORDS) for tokens in lower_tokens]
    if name == "punctuation":
        return [_punctuation_skeleton(text) for text in raw_texts]
    if name == "word_shape":
        return [" ".join(ra.word_shape(t) for t in tokens) for tokens in word_tokens]
    if name == "content_words":
        return [" ".join(t for t in tokens if t not in STOPWORDS and len(t) > 2)
               for tokens in lower_tokens]
    raise ValueError(name)  # pragma: no cover - internal misuse guard


def _view_strings_parsed(analysis: DocumentAnalysis, kind: str) -> list[str]:
    out: list[str] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            if kind == "lemma":
                toks = [t.lemma_.lower() for t in sent
                       if t.pos_ in CONTENT_POS and t.is_alpha]
            elif kind == "pos":
                toks = [t.pos_ for t in sent if not t.is_space]
            else:
                toks = [t.dep_ for t in sent if not t.is_space]
            out.append(" ".join(toks))
    return out


_LEXICAL_VIEWS = (
    ("normalized", "repetition.reuse_view_normalized_exact_share",
     "Exact-duplicate share, lowercase/normalized view", None),
    ("function_words", "repetition.reuse_view_function_words_exact_share",
     "Exact-duplicate share, function-words-only view", None),
    ("stopwords", "repetition.reuse_view_stopwords_exact_share",
     "Exact-duplicate share, stopwords-only view", None),
    ("punctuation", "repetition.reuse_view_punctuation_exact_share",
     "Exact-duplicate share, punctuation-only view",
     ("punct.repeated_sentence_skeleton_share", "punct.adjacent_identical_skeleton_rate")),
    ("word_shape", "repetition.reuse_view_word_shape_exact_share",
     "Exact-duplicate share, word-shape view", None),
    ("content_words", "repetition.reuse_view_content_words_exact_share",
     "Exact-duplicate share, content-words-only view",
     ("style.local_lexical_repetition", "repetition.content_word_reuse_distance")),
)

_PARSED_VIEWS = (
    ("lemma", "repetition.reuse_view_lemma_exact_share",
     "Exact-duplicate share, content-lemma view", ("repetition.lemma_reuse_distance",)),
    ("pos", "repetition.reuse_view_pos_exact_share",
     "Exact-duplicate share, POS-tag-sequence view", None),
    ("dependency", "repetition.reuse_view_dependency_exact_share",
     "Exact-duplicate share, dependency-label-sequence view", None),
)


def _transformed_view_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None,
                               raw_texts: Sequence[str], word_tokens: Sequence[list[str]],
                               lower_tokens: Sequence[list[str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, metric_id, label, overlaps in _LEXICAL_VIEWS:
        if not _feature(config, f"view_{name}"):
            out.append(finding(metric_id, label, None, "percent", family=FAMILY, sample_size=0,
                               min_sample=MIN_SAMPLE,
                               warning=f"disabled by features.view_{name}=false"))
            continue
        strings = _view_strings_lexical(name, raw_texts, word_tokens, lower_tokens)
        out.append(_view_exact_finding(metric_id, label, strings, overlaps=overlaps))

    for name, metric_id, label, overlaps in _PARSED_VIEWS:
        if not _feature(config, f"view_{name}"):
            out.append(finding(metric_id, label, None, "percent", family=FAMILY, sample_size=0,
                               min_sample=MIN_SAMPLE,
                               warning=f"off by default; enable features.view_{name} to turn "
                                       f"this on (it forces this document's shared spaCy parse)"))
            continue
        if analysis.nlp_unavailable:
            out.append(finding(metric_id, label, None, "percent", family=FAMILY, sample_size=0,
                               min_sample=MIN_SAMPLE, warning=analysis.nlp_unavailable))
            continue
        strings = _view_strings_parsed(analysis, name)
        out.append(_view_exact_finding(metric_id, label, strings, overlaps=overlaps,
                                       no_data_warning="no spaCy-segmented sentences to measure"))
    return out


# ---------------------------------------------------------- sentence pipeline

_SENTENCE_UNAVAILABLE_IDS = (
    ("repetition.reuse_exact_sentence_share", "Exact-duplicate sentence share"),
    ("repetition.reuse_near_duplicate_sentence_share", "Near-duplicate sentence share"),
    ("repetition.reuse_view_normalized_near_duplicate_share",
     "Near-duplicate sentence share, lowercase/normalized view"),
    ("repetition.reuse_minhash_jaccard_distribution", "MinHash-estimated Jaccard similarity"),
    ("repetition.reuse_simhash_hamming_distance", "SimHash nearest-neighbor Hamming distance"),
    ("repetition.reuse_levenshtein_similarity_distribution",
     "Normalized Levenshtein nearest-neighbor similarity"),
    ("repetition.reuse_jaro_winkler_similarity_distribution",
     "Jaro-Winkler nearest-neighbor similarity"),
    ("repetition.reuse_token_set_similarity_distribution", "Token-set fuzzy similarity"),
    ("repetition.reuse_token_sort_similarity_distribution", "Token-sort fuzzy similarity"),
    ("repetition.reuse_textdistance_similarity_distribution",
     "Sorensen-Dice character-bigram similarity (cross-check)"),
    ("repetition.reuse_near_duplicate_distance", "Distance between near-duplicate sentences"),
)


def _sentence_pipeline(sentences: Sequence[str],
                       config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    total = len(sentences)
    if total < 2 or not _feature(config, "sentence_level"):
        warning = ("needs at least two sentences" if total < 2
                  else "disabled by features.sentence_level=false")
        return [finding(mid, name, None, "percent" if "share" in mid else None, family=FAMILY,
                        sample_size=total, min_sample=MIN_SAMPLE, warning=warning)
               for mid, name in _SENTENCE_UNAVAILABLE_IDS]

    max_sentences = max(2, int(option(config, "max_sentences", DEFAULT_MAX_SENTENCES)))
    truncated = total > max_sentences
    used = list(sentences[:max_sentences])
    n = len(used)

    raw_texts = [_normalize_whitespace(s) for s in used]
    lower_tokens = [textlib.words(s) for s in used]
    lower_tokens = [[t.lower().replace("’", "'") for t in toks] for toks in lower_tokens]
    word_tokens = [textlib.words(s) for s in used]

    shingle_k = max(1, int(option(config, "shingle_size", DEFAULT_SHINGLE_SIZE)))
    num_perm = max(4, int(option(config, "minhash_num_perm", DEFAULT_MINHASH_NUM_PERM)))
    seed = int(option(config, "minhash_seed", DEFAULT_MINHASH_SEED))
    near_threshold = float(option(config, "near_duplicate_threshold",
                                  DEFAULT_NEAR_DUPLICATE_THRESHOLD))
    candidate_threshold = min(near_threshold, float(option(
        config, "candidate_threshold", DEFAULT_CANDIDATE_THRESHOLD)))
    thresholds = sorted({float(t) for t in option(
        config, "near_duplicate_thresholds", list(DEFAULT_NEAR_DUPLICATE_THRESHOLDS))})
    max_candidates = max(1, int(option(config, "max_candidates_per_item",
                                       DEFAULT_MAX_CANDIDATES_PER_ITEM)))

    surface_shingles = [ra.shingles(text.split(), shingle_k) for text in raw_texts]
    # Candidate generation runs at the LOOSER candidate_threshold; the near-
    # duplicate share/gap metrics below then filter these candidates' real
    # Jaccard by the stricter near_duplicate_threshold. See
    # DEFAULT_CANDIDATE_THRESHOLD's docstring for why the two are separate.
    graph, jaccards, pair_count, backend = ra.minhash_lsh_candidates(
        surface_shingles, num_perm=num_perm, threshold=candidate_threshold, seed=seed,
        max_candidates_per_item=max_candidates)

    exact_share, _, distinct_forms, exact_counts = _exact_share(raw_texts)
    exact_evidence = [{"text": form[:160], "count": count}
                      for form, count in exact_counts.most_common(DEFAULT_MAX_REPORTED)
                      if count > 1]

    near_flag = [False] * n
    gaps: list[int] = []
    near_pairs: list[tuple[float, int, int]] = []
    for key, jac in jaccards.items():
        i, j = tuple(key)
        if jac >= near_threshold:
            near_flag[i] = near_flag[j] = True
            gaps.append(abs(i - j))
            near_pairs.append((jac, i, j))
    near_share = rate(sum(near_flag), n)
    near_evidence = [{"jaccard": round(jac, 3), "text_a": raw_texts[i][:160],
                      "text_b": raw_texts[j][:160], "sentence_index_a": i, "sentence_index_b": j}
                     for jac, i, j in sorted(near_pairs, reverse=True)[:DEFAULT_MAX_REPORTED]]

    per_threshold = []
    for t in thresholds:
        flagged: set[int] = set()
        for key, jac in jaccards.items():
            if jac >= t:
                i, j = tuple(key)
                flagged.add(i)
                flagged.add(j)
        per_threshold.append({"threshold": t, "share_percent": rate(len(flagged), n)})

    base_dist = {"backend": backend, "candidate_pair_count": pair_count,
                "near_duplicate_threshold": near_threshold,
                "candidate_threshold": candidate_threshold, "shingle_size": shingle_k,
                "minhash_num_perm": num_perm, "minhash_seed": seed,
                "max_candidates_per_item": max_candidates}
    if truncated:
        base_dist["truncated_to_max_sentences"] = max_sentences
        base_dist["total_sentences"] = total

    out: list[dict[str, Any]] = [
        finding("repetition.reuse_exact_sentence_share", "Exact-duplicate sentence share",
               exact_share, "percent", family=FAMILY, sample_size=n, min_sample=MIN_SAMPLE,
               distribution={"distinct_forms": distinct_forms, "aggregation":
                            "share of sentences whose whitespace-normalized text exactly "
                            "matches at least one other sentence", **base_dist},
               evidence=exact_evidence or None),
        finding("repetition.reuse_near_duplicate_sentence_share",
               "Near-duplicate sentence share", near_share, "percent", family=FAMILY,
               sample_size=n, min_sample=MIN_SAMPLE, details=per_threshold,
               distribution=dict(base_dist, thresholds_evaluated=[t for t in thresholds],
                                aggregation="share of sentences with at least one candidate "
                                          "pair whose real Jaccard similarity clears "
                                          "near_duplicate_threshold; details gives the same "
                                          "share at each configured threshold"),
               evidence=near_evidence or None),
    ]

    if _feature(config, "view_normalized"):
        norm_tokens = [" ".join(toks) for toks in lower_tokens]
        norm_shingles = [ra.shingles(toks, shingle_k) for toks in lower_tokens]
        _, norm_jaccards, norm_pairs, norm_backend = ra.minhash_lsh_candidates(
            norm_shingles, num_perm=num_perm, threshold=candidate_threshold, seed=seed,
            max_candidates_per_item=max_candidates)
        norm_flag = [False] * n
        for key, jac in norm_jaccards.items():
            if jac >= near_threshold:
                i, j = tuple(key)
                norm_flag[i] = norm_flag[j] = True
        out.append(finding(
            "repetition.reuse_view_normalized_near_duplicate_share",
            "Near-duplicate sentence share, lowercase/normalized view",
            rate(sum(norm_flag), n), "percent", family=FAMILY, sample_size=n,
            min_sample=MIN_SAMPLE,
            distribution={"backend": norm_backend, "candidate_pair_count": norm_pairs,
                         "near_duplicate_threshold": near_threshold,
                         "aggregation": "share of sentences with at least one lowercase/"
                                       "normalized-view candidate pair whose real Jaccard "
                                       "clears near_duplicate_threshold"}))
    else:
        out.append(finding("repetition.reuse_view_normalized_near_duplicate_share",
                           "Near-duplicate sentence share, lowercase/normalized view", None,
                           "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                           warning="disabled by features.view_normalized=false"))

    jaccard_values = list(jaccards.values())
    out.append(_shape_finding(
        "repetition.reuse_minhash_jaccard_distribution",
        "MinHash-estimated Jaccard similarity among candidate sentence pairs",
        jaccard_values, "ratio", sample_size=len(jaccard_values),
        distribution_extra=dict(base_dist),
        aggregation="headline is the median MinHash-estimated Jaccard similarity, one value "
                   "per candidate PAIR (not per sentence)",
        warning=None if jaccard_values else "no candidate pairs found"))

    if _feature(config, "simhash"):
        shingle_size = max(1, int(option(config, "simhash_shingle_size",
                                         DEFAULT_SIMHASH_SHINGLE_SIZE)))
        index_k = max(1, int(option(config, "simhash_index_k", DEFAULT_SIMHASH_INDEX_K)))
        nearest, reason = ra.simhash_nearest_neighbors(raw_texts, shingle_k=shingle_size,
                                                        index_k=index_k, extra_candidates=graph)
        values = list(nearest.values())
        out.append(_shape_finding(
            "repetition.reuse_simhash_hamming_distance",
            "SimHash nearest-neighbor Hamming distance (bits, lower = more similar)",
            values, "bits", sample_size=len(values),
            distribution_extra={"fingerprint_bits": 64, "shingle_size": shingle_size,
                                "index_k": index_k, "library_version": _library_versions()["simhash"]},
            warning=None if values else reason))
    else:
        out.append(finding("repetition.reuse_simhash_hamming_distance",
                           "SimHash nearest-neighbor Hamming distance (bits, lower = more similar)",
                           None, "bits", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                           warning="disabled by features.simhash=false"))

    if _feature(config, "edit_distance_family"):
        fuzzy = ra.edit_and_fuzzy_nearest_neighbors(raw_texts, graph)
        library_versions = _library_versions()
        for key, metric_id, label, unit, lib_key in (
                ("levenshtein", "repetition.reuse_levenshtein_similarity_distribution",
                 "Normalized Levenshtein nearest-neighbor similarity (candidate pairs)",
                 "percent", "levenshtein"),
                ("jaro_winkler", "repetition.reuse_jaro_winkler_similarity_distribution",
                 "Jaro-Winkler nearest-neighbor similarity (candidate pairs)", "percent",
                 "jellyfish"),
                ("token_set", "repetition.reuse_token_set_similarity_distribution",
                 "RapidFuzz token-set fuzzy similarity (candidate pairs)", "percent",
                 "rapidfuzz"),
                ("token_sort", "repetition.reuse_token_sort_similarity_distribution",
                 "RapidFuzz token-sort fuzzy similarity (candidate pairs)", "percent",
                 "rapidfuzz")):
            values, reason = fuzzy[key]
            out.append(_shape_finding(
                metric_id, label, values, unit, sample_size=len(values),
                distribution_extra={"library_version": library_versions.get(lib_key)},
                warning=None if values else (reason or "no candidate pairs found")))
    else:
        for metric_id, label, unit in (
                ("repetition.reuse_levenshtein_similarity_distribution",
                 "Normalized Levenshtein nearest-neighbor similarity (candidate pairs)", "percent"),
                ("repetition.reuse_jaro_winkler_similarity_distribution",
                 "Jaro-Winkler nearest-neighbor similarity (candidate pairs)", "percent"),
                ("repetition.reuse_token_set_similarity_distribution",
                 "RapidFuzz token-set fuzzy similarity (candidate pairs)", "percent"),
                ("repetition.reuse_token_sort_similarity_distribution",
                 "RapidFuzz token-sort fuzzy similarity (candidate pairs)", "percent")):
            out.append(finding(metric_id, label, None, unit, family=FAMILY, sample_size=0,
                               min_sample=MIN_SAMPLE,
                               warning="disabled by features.edit_distance_family=false"))

    if _feature(config, "textdistance_crosscheck"):
        td_values, td_reason = ra.textdistance_crosscheck(raw_texts, graph)
        out.append(_shape_finding(
            "repetition.reuse_textdistance_similarity_distribution",
            "Sorensen-Dice character-bigram similarity, cross-check (candidate pairs)",
            td_values, "percent", sample_size=len(td_values),
            distribution_extra={"library_version": _library_versions()["textdistance"]},
            warning=None if td_values else (td_reason or "no candidate pairs found")))
    else:
        out.append(finding("repetition.reuse_textdistance_similarity_distribution",
                           "Sorensen-Dice character-bigram similarity, cross-check "
                           "(candidate pairs)", None, "percent", family=FAMILY, sample_size=0,
                           min_sample=MIN_SAMPLE,
                           warning="disabled by features.textdistance_crosscheck=false"))

    out.append(_shape_finding(
        "repetition.reuse_near_duplicate_distance",
        "Sentence-index distance between near-duplicate pairs (local refrain vs. book-wide "
        "template)", gaps, "sentences apart", sample_size=len(gaps),
        distribution_extra={"near_duplicate_threshold": near_threshold},
        aggregation="headline is the median absolute sentence-index gap, one value per "
                   "near-duplicate PAIR found at or above near_duplicate_threshold",
        warning=None if gaps else f"no sentence pair reached the near-duplicate threshold "
                                  f"of {near_threshold}"))

    return out


# --------------------------------------------------------- paragraph pipeline

_PARAGRAPH_UNAVAILABLE_IDS = (
    ("repetition.reuse_exact_paragraph_share", "Exact-duplicate paragraph share"),
    ("repetition.reuse_near_duplicate_paragraph_share", "Near-duplicate paragraph share"),
    ("repetition.reuse_tlsh_similarity_distribution",
     "TLSH fuzzy-hash similarity among paragraphs (diff score, lower = more similar)"),
    ("repetition.reuse_ssdeep_similarity_distribution",
     "ssdeep-compatible fuzzy-hash similarity among paragraphs (0-100, higher = more similar)"),
)


def _paragraph_pipeline(paragraphs: Sequence[str], config: Mapping[str, Any] | None
                        ) -> list[dict[str, Any]]:
    total = len(paragraphs)
    if total < 2 or not _feature(config, "paragraph_level"):
        warning = ("needs at least two paragraphs" if total < 2
                  else "disabled by features.paragraph_level=false")
        return [finding(mid, name, None, "percent" if "share" in mid else None, family=FAMILY,
                        sample_size=total, min_sample=min(MIN_SAMPLE, 10), warning=warning)
               for mid, name in _PARAGRAPH_UNAVAILABLE_IDS]

    max_paragraphs = max(2, int(option(config, "max_paragraphs", DEFAULT_MAX_PARAGRAPHS)))
    truncated = total > max_paragraphs
    used = list(paragraphs[:max_paragraphs])
    n = len(used)

    raw_texts = [_normalize_whitespace(p) for p in used]
    shingle_k = max(1, int(option(config, "shingle_size", DEFAULT_SHINGLE_SIZE)))
    num_perm = max(4, int(option(config, "minhash_num_perm", DEFAULT_MINHASH_NUM_PERM)))
    seed = int(option(config, "minhash_seed", DEFAULT_MINHASH_SEED))
    near_threshold = float(option(config, "near_duplicate_threshold",
                                  DEFAULT_NEAR_DUPLICATE_THRESHOLD))
    candidate_threshold = min(near_threshold, float(option(
        config, "candidate_threshold", DEFAULT_CANDIDATE_THRESHOLD)))
    thresholds = sorted({float(t) for t in option(
        config, "near_duplicate_thresholds", list(DEFAULT_NEAR_DUPLICATE_THRESHOLDS))})
    max_candidates = max(1, int(option(config, "max_candidates_per_item",
                                       DEFAULT_MAX_CANDIDATES_PER_ITEM)))

    shingle_sets = [ra.shingles(text.split(), shingle_k) for text in raw_texts]
    # See DEFAULT_CANDIDATE_THRESHOLD's docstring: candidate generation uses
    # the looser threshold so the TLSH/ssdeep distributions below are not
    # self-selected for near-ceiling similarity; near-duplicate share/gaps
    # still filter by the stricter near_duplicate_threshold.
    graph, jaccards, pair_count, backend = ra.minhash_lsh_candidates(
        shingle_sets, num_perm=num_perm, threshold=candidate_threshold, seed=seed,
        max_candidates_per_item=max_candidates)

    exact_share, _, distinct_forms, exact_counts = _exact_share(raw_texts)
    exact_evidence = [{"text": form[:200], "count": count}
                      for form, count in exact_counts.most_common(DEFAULT_MAX_REPORTED)
                      if count > 1]
    near_flag = [False] * n
    near_pairs: list[tuple[float, int, int]] = []
    for key, jac in jaccards.items():
        if jac >= near_threshold:
            i, j = tuple(key)
            near_flag[i] = near_flag[j] = True
            near_pairs.append((jac, i, j))
    near_share = rate(sum(near_flag), n)
    near_evidence = [{"jaccard": round(jac, 3), "text_a": raw_texts[i][:200],
                      "text_b": raw_texts[j][:200], "paragraph_index_a": i,
                      "paragraph_index_b": j}
                     for jac, i, j in sorted(near_pairs, reverse=True)[:DEFAULT_MAX_REPORTED]]
    per_threshold = []
    for t in thresholds:
        flagged: set[int] = set()
        for key, jac in jaccards.items():
            if jac >= t:
                i, j = tuple(key)
                flagged.add(i)
                flagged.add(j)
        per_threshold.append({"threshold": t, "share_percent": rate(len(flagged), n)})

    base_dist = {"backend": backend, "candidate_pair_count": pair_count,
                "near_duplicate_threshold": near_threshold}
    if truncated:
        base_dist["truncated_to_max_paragraphs"] = max_paragraphs
        base_dist["total_paragraphs"] = total

    out: list[dict[str, Any]] = [
        finding("repetition.reuse_exact_paragraph_share", "Exact-duplicate paragraph share",
               exact_share, "percent", family=FAMILY, sample_size=n,
               min_sample=min(MIN_SAMPLE, 10),
               distribution={"distinct_forms": distinct_forms, "aggregation":
                            "share of paragraphs whose whitespace-normalized text exactly "
                            "matches at least one other paragraph", **base_dist},
               evidence=exact_evidence or None),
        finding("repetition.reuse_near_duplicate_paragraph_share",
               "Near-duplicate paragraph share", near_share, "percent", family=FAMILY,
               sample_size=n, min_sample=min(MIN_SAMPLE, 10), details=per_threshold,
               distribution=dict(base_dist, aggregation="share of paragraphs with at least "
                                "one candidate pair whose real Jaccard clears "
                                "near_duplicate_threshold; details gives the same share at "
                                "each configured threshold"),
               evidence=near_evidence or None),
    ]

    min_tlsh_chars = max(HARD_MIN_BLOCK_CHARS,
                         int(option(config, "min_tlsh_chars", DEFAULT_MIN_TLSH_CHARS)))
    if _feature(config, "tlsh"):
        diffs, evidence, reason = ra.tlsh_pairwise(raw_texts, graph, min_chars=min_tlsh_chars)
        out.append(_shape_finding(
            "repetition.reuse_tlsh_similarity_distribution",
            "TLSH fuzzy-hash similarity among paragraphs (diff score, lower = more similar)",
            [float(d) for d in diffs], "tlsh_diff", sample_size=len(diffs),
            distribution_extra={"min_block_chars": min_tlsh_chars,
                                "library_version": _library_versions()["tlsh"]},
            aggregation="headline is the median TLSH diff score, one value per candidate "
                       "paragraph PAIR at or above min_block_chars (lower = more similar)",
            evidence=evidence[:DEFAULT_MAX_REPORTED] if evidence else None,
            warning=None if diffs else reason))
    else:
        out.append(finding("repetition.reuse_tlsh_similarity_distribution",
                           "TLSH fuzzy-hash similarity among paragraphs (diff score, lower = "
                           "more similar)", None, "tlsh_diff", family=FAMILY, sample_size=0,
                           min_sample=min(MIN_SAMPLE, 10),
                           warning="disabled by features.tlsh=false"))

    min_ssdeep_chars = max(HARD_MIN_BLOCK_CHARS,
                           int(option(config, "min_ssdeep_chars", DEFAULT_MIN_SSDEEP_CHARS)))
    if _feature(config, "ssdeep"):
        scores, evidence, reason = ra.ssdeep_pairwise(raw_texts, graph, min_chars=min_ssdeep_chars)
        out.append(_shape_finding(
            "repetition.reuse_ssdeep_similarity_distribution",
            "ssdeep-compatible (ppdeep) fuzzy-hash similarity among paragraphs (0-100, higher "
            "= more similar)", [float(s) for s in scores], "score_0_100",
            sample_size=len(scores),
            distribution_extra={"min_block_chars": min_ssdeep_chars,
                                "library_version": _library_versions()["ppdeep"],
                                "library": "ppdeep (pure-Python ssdeep-compatible; the real "
                                          "'ssdeep' package fails to build in this environment)"},
            aggregation="headline is the median ppdeep similarity score, one value per "
                       "candidate paragraph PAIR at or above min_block_chars (higher = more "
                       "similar; often 0 for organically different-length blocks -- see "
                       "config.json's _requires_ssdeep note)",
            evidence=evidence[:DEFAULT_MAX_REPORTED] if evidence else None,
            warning=None if scores else reason))
    else:
        out.append(finding("repetition.reuse_ssdeep_similarity_distribution",
                           "ssdeep-compatible (ppdeep) fuzzy-hash similarity among paragraphs "
                           "(0-100, higher = more similar)", None, "score_0_100", family=FAMILY,
                           sample_size=0, min_sample=min(MIN_SAMPLE, 10),
                           warning="disabled by features.ssdeep=false"))

    return out


# ------------------------------------------------------ longest repeat / motif

def _longest_run_findings(tokens: Sequence[str], config: Mapping[str, Any] | None
                          ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    total = len(tokens)

    if _feature(config, "longest_repeated_run"):
        found = ra.longest_repeated_token_run(tokens) if total >= 2 else None
        if found:
            length, first, second = found
            snippet = " ".join(tokens[first:first + length])
            joined = "\x1f".join(tokens)
            offsets = [0]
            for tok in tokens:
                offsets.append(offsets[-1] + len(tok) + 1)
            needle = "\x1f".join(tokens[first:first + length])
            char_hits = ra.occurrences_of(joined, needle)
            occurrence_count = 0
            for hit in char_hits:
                # A match only counts if it starts exactly on a token boundary.
                if hit in offsets[:-1]:
                    occurrence_count += 1
            out.append(finding(
                "repetition.reuse_longest_repeated_token_run",
                "Longest exact repeated token run", length, "tokens", family=FAMILY,
                sample_size=total, min_sample=200, unit_sensitive=True,
                distribution={"total_occurrences": max(occurrence_count, 2),
                             "aggregation": "single document-wide maximum, not an aggregate "
                                           "over multiple values"},
                evidence=[{"length_tokens": length, "first_position": first,
                          "second_position": second, "text": snippet[:200]}]))
        else:
            out.append(finding(
                "repetition.reuse_longest_repeated_token_run",
                "Longest exact repeated token run", 0 if total >= 2 else None, "tokens",
                family=FAMILY, sample_size=total, min_sample=200, unit_sensitive=True,
                warning=None if total >= 2 else "not enough tokens to search",
                distribution={"total_occurrences": 0} if total >= 2 else None))
    else:
        out.append(finding("repetition.reuse_longest_repeated_token_run",
                           "Longest exact repeated token run", None, "tokens", family=FAMILY,
                           sample_size=0, min_sample=200,
                           warning="disabled by features.longest_repeated_run=false"))

    if _feature(config, "longest_approximate_run"):
        window = max(2, int(option(config, "approx_window_tokens", DEFAULT_APPROX_WINDOW_TOKENS)))
        stride = max(1, int(option(config, "approx_stride_tokens", DEFAULT_APPROX_STRIDE_TOKENS)))
        threshold = float(option(config, "approx_similarity_threshold",
                                 DEFAULT_APPROX_SIMILARITY_THRESHOLD))
        max_windows = max(2, int(option(config, "approx_max_windows", DEFAULT_APPROX_MAX_WINDOWS)))
        extend_step = max(1, int(option(config, "approx_extend_step", DEFAULT_APPROX_EXTEND_STEP)))
        result = ra.longest_approximate_repeated_run(
            tokens, window=window, stride=stride, threshold=threshold,
            max_windows=max_windows, extend_step=extend_step) if total >= window * 2 else None
        if result:
            length = result["length"]
            first, second = result["first_position"], result["second_position"]
            out.append(finding(
                "repetition.reuse_longest_approximate_repeated_run",
                "Longest approximately-repeated token run", length, "tokens", family=FAMILY,
                sample_size=total, min_sample=200, unit_sensitive=True,
                distribution={"similarity_threshold": threshold,
                             "candidate_pairs_examined": result.get("candidate_pairs_examined"),
                             "window_tokens": window, "stride_tokens": stride,
                             "aggregation": "single document-wide maximum, not an aggregate "
                                           "over multiple values"},
                evidence=[{"length_tokens": length, "first_position": first,
                          "second_position": second, "similarity": result["similarity"],
                          "text_a": " ".join(tokens[first:first + length])[:200],
                          "text_b": " ".join(tokens[second:second + length])[:200]}]))
        else:
            out.append(finding(
                "repetition.reuse_longest_approximate_repeated_run",
                "Longest approximately-repeated token run", 0 if total >= window * 2 else None,
                "tokens", family=FAMILY, sample_size=total, min_sample=200, unit_sensitive=True,
                warning=None if total >= window * 2 else
                f"needs at least {window * 2} tokens (2x approx_window_tokens)"))
    else:
        out.append(finding("repetition.reuse_longest_approximate_repeated_run",
                           "Longest approximately-repeated token run", None, "tokens",
                           family=FAMILY, sample_size=0, min_sample=200,
                           warning="disabled by features.longest_approximate_run=false"))
    return out


def _template_motif_findings(tokens: Sequence[str], config: Mapping[str, Any] | None
                             ) -> list[dict[str, Any]]:
    if not _feature(config, "template_motifs"):
        return [
            finding("repetition.reuse_template_motif_count", "Repeated template motif count",
                   None, "motifs", family=FAMILY, sample_size=0, min_sample=30,
                   warning="disabled by features.template_motifs=false"),
            finding("repetition.reuse_template_motif_coverage",
                   "Token share covered by a repeated template motif", None, "percent",
                   family=FAMILY, sample_size=0, min_sample=30,
                   warning="disabled by features.template_motifs=false"),
        ]

    window = max(2, int(option(config, "template_motif_window", DEFAULT_TEMPLATE_MOTIF_WINDOW)))
    min_count = max(2, int(option(config, "template_motif_min_count",
                                  DEFAULT_TEMPLATE_MOTIF_MIN_COUNT)))
    total = len(tokens)
    overlaps = ["discourse.repeated_construction_share", "discourse.top_construction_count"]
    if total < window:
        warning = f"needs at least {window} tokens (template_motif_window)"
        return [
            finding("repetition.reuse_template_motif_count", "Repeated template motif count",
                   None, "motifs", family=FAMILY, sample_size=total, min_sample=30,
                   warning=warning),
            finding("repetition.reuse_template_motif_coverage",
                   "Token share covered by a repeated template motif", None, "percent",
                   family=FAMILY, sample_size=total, min_sample=30, warning=warning),
        ]

    skeleton = _function_word_skeleton(tokens)
    positions = ra.motif_positions(skeleton, window)
    repeated = {gram: pos for gram, pos in positions.items() if len(pos) >= min_count
               and any(part != "_" for part in gram)}
    covered = bytearray(total)
    for pos_list in repeated.values():
        for start in pos_list:
            for offset in range(start, min(start + window, total)):
                covered[offset] = 1
    coverage = rate(sum(covered), total)
    top = sorted(repeated.items(), key=lambda item: -len(item[1]))[:DEFAULT_MAX_REPORTED]
    evidence = [{"motif": " ".join(gram), "count": len(pos), "example_positions": pos[:5]}
               for gram, pos in top]

    return [
        finding("repetition.reuse_template_motif_count", "Repeated template motif count",
               len(repeated), "motifs", family=FAMILY, sample_size=total, min_sample=30,
               unit_sensitive=True,
               distribution={"motif_window": window, "min_count": min_count,
                            "overlaps_existing_metric_id": overlaps,
                            "aggregation": "count of distinct function-word/punctuation "
                                          f"skeleton windows of length {window} tokens "
                                          f"occurring at least {min_count} times"},
               evidence=evidence or None),
        finding("repetition.reuse_template_motif_coverage",
               "Token share covered by a repeated template motif", coverage, "percent",
               family=FAMILY, sample_size=total, min_sample=30,
               distribution={"motif_window": window, "min_count": min_count,
                            "overlaps_existing_metric_id": overlaps,
                            "aggregation": "share of tokens covered by at least one "
                                          "qualifying repeated motif window"},
               evidence=evidence[:5] or None),
    ]


# ------------------------------------------------------------------ measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    sentences = analysis.sentences
    paragraphs = analysis.paragraphs
    tokens = analysis.tokens

    out: list[dict[str, Any]] = []
    out.extend(_sentence_pipeline(sentences, config))
    out.extend(_paragraph_pipeline(paragraphs, config))
    out.extend(_longest_run_findings(tokens, config))
    out.extend(_template_motif_findings(tokens, config))

    used = list(sentences[:max(2, int(option(config, "max_sentences", DEFAULT_MAX_SENTENCES)))])
    word_tokens = [textlib.words(s) for s in used]
    lower_tokens = [[t.lower().replace("’", "'") for t in toks] for toks in word_tokens]
    raw_texts = [_normalize_whitespace(s) for s in used]
    out.extend(_transformed_view_findings(analysis, config, raw_texts, word_tokens, lower_tokens))

    return out
