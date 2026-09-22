"""A broad, deliberately un-opinionated stylometry and authorship-analysis suite.

Authorship and style are distributed across many weak, individually
unreliable signals: character n-grams that leak spelling and morphology,
function words used below conscious control, punctuation habits, sentence-
opening tics, vocabulary-growth curves, and how predictable the prose is to a
compressor. No single one of these identifies an author. A system that keeps
them all as separate numbers -- instead of averaging them into one
"authorship score" -- lets a reader see *which* signals agree and which
disagree, and disagreement between independent measurements is itself
evidence (two texts can share a topic and diverge in syntax, or share an
author and diverge in topic). This module follows the sibling family
``function_words`` in spirit but never merges into it: where that module
computes one Burrows's-Delta number against the corpus mean, this module adds
independent representations and independent distance families, on purpose,
and expects them to disagree with each other and with it sometimes.

Every measurement here is off by default (``stylometry_suite.enabled`` in
``config.json``) and every group of measurements has its own switch under
``features``, because turning the suite on is not the same decision as
wanting POS n-grams, which need a spaCy parse, or corpus-reference distances,
which need a profile with function-word feature vectors in it.

Two kinds of work happen in this module:

Within-document measurements
    lexical richness (hapax/dislegomena ratios, Yule's K/I, Honore's R,
    Sichel's S, Herdan's C, Guiraud's R, Maas's a^2, the Uber/Dugast index),
    vocabulary growth (Heaps' law) and rank-frequency (Zipf's law) fits,
    n-gram entropy over character, byte, word, function-word, punctuation,
    word-shape and affix representations, sentence-opening and
    capitalization/contraction habits, whole-document compressibility, and
    section-to-section stability -- all computable with no reference corpus
    at all, which is what lets this suite say something about a manuscript
    that has never been profiled against anything.

Corpus-reference measurements
    nearest/second-nearest/margin/centroid/dispersion/out-of-distribution
    distance, k-nearest-neighbour agreement, and a corpus-wide unigram
    cross-entropy, all read from a corpus profile built by
    :mod:`textgrader.corpus`. These need ``profile`` to be non-``None``, and
    the author-labelled variants (per-author centroid, within-author
    dispersion, standardized distance, kNN author agreement) need per-book
    ``metadata.author`` entries in that profile's manifest, which is optional
    and gracefully degrades to "unavailable, no author labels" when absent.
    Four more groups were added once the reference environment grew
    ``sentence-transformers``, ``torch`` and ``lexicalrichness``, and once
    :mod:`textgrader.corpus` gained a per-author frequency table (see below):
    a document/section-level sentence-embedding representation
    (``embedding_style``), a bounded impostors-style verification
    (``impostors``), per-author unigram cross-entropy
    (``author_language_model``), and a word-frequency distance family against
    the corpus's pooled vocabulary (``word_frequency_distance``). Every one of
    these is documented in its own section below.

Feature representations are cached on ``analysis._shared`` via
:meth:`DocumentAnalysis.memo`, because several distance and entropy
computations below reuse the same token/character/mark stream; nothing here
re-cleans, re-tokenizes, re-splits sentences, or re-parses dialogue --
everything is built from :class:`DocumentAnalysis`'s own properties, or from
existing sibling metric modules' *public* helpers (``function_words.vector``,
``function_words.FUNCTION``, ``punctuation_profile.MARK_RE``/``MARK_NAMES``,
``dialogue_contractions.CONTRACTION_SUFFIXES``/``WHOLE_WORD_CONTRACTIONS``,
``drift_chapter_zscores.get_sections``), which is reuse in the same sense the
project already practises for those constants, not a new coupling.

No third-party package is required for most of the within-document
measurements: every distance, entropy, regression and compression routine in
groups A-E is plain standard library (``math``, ``statistics``,
``collections.Counter``, ``zlib``, ``lzma``). This was a judgement call, not
an oversight -- scipy, numpy and scikit-learn are available in the reference
environment, but the underlying arithmetic (cosine/Manhattan/Euclidean/
Jensen-Shannon distance, ordinary least squares on ~20-1000 points) is simple
enough that adding a hard dependency would buy nothing except a new way for
``TEXTGRADER_DISABLE_OPTIONAL=all`` to have something to disable. spaCy (the
``pos_dependency`` feature group) and now sentence-transformers (the
``embedding_style`` group) and lexicalrichness (the
``lexicalrichness_crosscheck`` group) are the exceptions, and each defaults to
``False``/gracefully-degrading for a different reason -- see each group's own
section below. This suite is registered at ``cost="moderate"`` rather than
``"parse"`` or ``"model"``, and ``REQUIRES`` deliberately does **not** list
``sentence_transformers`` (see ``embedding_style``, below, for why): both
choices keep the corpus builder profiling this suite's cheap channels by
default, which misdeclaring either one would silently switch off.

Sentence-embedding style representation (``embedding_style``, off by default)
    :func:`_embedding_style_findings` embeds this document's own sections
    (the same ``get_sections`` split ``section_stability`` uses) with
    ``sentence-transformers`` and reports how far the opening section's
    embedding sits from the closing section's (three distance families:
    cosine, Euclidean, Manhattan -- Jensen-Shannon is not computed here
    because an embedding is not a probability distribution), the median and
    largest section-to-section embedding shift, and each section's distance
    from the document's own mean embedding. This is the "biggest new
    channel" this pass adds, and it is deliberately *within-document*: a
    corpus profile does not retain raw book text or per-book embedding
    vectors (see :mod:`textgrader.corpus`'s docstring on portability, and the
    note in "Still deferred" below), so there is nothing to embed on the
    corpus side to compare against directly. It still becomes
    corpus-comparable the same way every other scalar finding in this suite
    does: :mod:`textgrader.corpus` folds every scalar finding value into
    ``profile.distributions`` automatically for any suite it profiles, so a
    corpus rebuilt with ``metrics.stylometry_suite.features.embedding_style``
    turned on in *its own* config populates a real reference distribution for
    ``style.stylometry_embedding_dispersion`` (and the others), and
    ``grade.py`` then compares a manuscript's value against it exactly the
    way it already compares ``compression_ratio``. Loading the model is
    strictly gated behind this feature flag -- see "The critical gating rule"
    below.

Impostors-style verification (``impostors``, off by default)
    :func:`_impostors_findings` is a bounded, distance-based approximation of
    Koppel & Winter's impostors method, reconsidered now that the corpus
    profile has enough structure to make a *cheap* version honest. It picks
    a candidate author (the corpus author whose function-word centroid is
    nearest this document, or ``impostors_target_author`` if set), samples up
    to ``impostors_k`` other corpus authors as impostors, and repeats
    ``impostors_iterations`` times: sample a random ``impostors_feature_fraction``
    slice of the function-word list, and check whether this document sits
    closer to the candidate author's centroid than to *every* sampled
    impostor's centroid on that feature slice. The mean win rate is
    ``impostors_verification_score``; its variance across iterations is
    ``impostors_score_variance``. This is what the module docstring's
    previous pass judged infeasible under the label "impostors" -- and that
    judgement was right for the *literature's* impostors method, which fits
    a real classifier per iteration; what is implemented here is the
    distance-comparison idea the technique is built on, over data the corpus
    profile already stores (aligned per-book function-word vectors plus
    author labels), with no retraining and no raw text. Every finding says so
    in its warning: this is an approximation, not the citable technique, and
    should be read as one more weak signal among the suite's many, not a
    verdict.

Per-author language model (``author_language_model``, on by default)
    :func:`_author_language_model_findings` scores this document's
    cross-entropy under *every* author's unigram frequency table in
    ``profile.author_word_frequency`` (see "Corpus schema: the per-author
    frequency table" below) and reports the best (lowest-cross-entropy)
    author's cross-entropy and perplexity, plus the margin to the
    second-best author. Genuinely author-specific, unlike
    ``corpus_language_model``'s corpus-*wide* table.

Word-frequency distance family (``word_frequency_distance``, on by default)
    :func:`_word_frequency_distance_findings` treats the corpus's pooled
    ``word_frequency`` table as a second aligned representation (besides
    ``feature_profiles['function_words']``) and runs the same four distance
    families against it that ``corpus_reference`` already runs against the
    function-word vectors, restricted to the corpus's most frequent words
    unioned with this document's own most frequent words (bounded by
    ``word_frequency_vocab_cap``). This is what closes the "distance families
    beyond function words" gap: not a new distance family, but the same four
    applied to a second representation the profile already, genuinely holds.

lexicalrichness cross-check (``lexicalrichness_crosscheck``, on by default)
    :func:`_lexicalrichness_crosscheck_findings` reports ``lexicalrichness``'s
    own ``yulek``/``yulei``/``Herdan``/``Dugast``/``Maas``/``rttr`` alongside
    this module's own hand-rolled Yule's K/I, Herdan's C, Uber/Dugast and
    Guiraud's R (group B, above), in the project's established voice for a
    cross-check: report both, keep the disagreement, never silently prefer
    one. Each finding's ``distribution`` names which of this module's own
    metric ids it corresponds to.

The critical gating rule
    :class:`textgrader.metrics.MetricSpec.needs_model` is defined as
    ``"sentence_transformers" in self.requires`` -- nothing else -- and
    :mod:`textgrader.corpus` uses exactly that to decide whether to run a
    metric over every corpus book by default. A ``torch``-backed feature is
    invisible to that guard unless the *whole suite's* ``REQUIRES`` says so,
    and this suite's ``REQUIRES`` deliberately still does **not** include
    ``"sentence_transformers"``: doing so would flip ``needs_model`` for the
    *entire* suite (every within-document group in it, not just
    ``embedding_style``) and silently drop this suite out of the corpus
    builder's default profiling pass -- a real regression to the "cheap
    unless asked" property the previous pass built and documented (see the
    cost/``REQUIRES`` paragraph above). The correct, finer-grained guard is
    already in this module: ``embedding_style`` defaults to ``False`` in
    ``DEFAULT_FEATURES``, and :func:`_embedding_style_findings` is the *only*
    code path in this file that imports ``sentence_transformers`` (via
    :func:`textgrader.metrics.semantic_adjacent.embed_texts`, reused rather
    than duplicated), and only calls it when that flag is explicitly on.
    ``tests/test_stylometry_suite.py`` asserts this directly: measuring with
    the suite's default config never populates
    ``semantic_adjacent``'s model cache.

Corpus schema: the per-author frequency table
    :mod:`textgrader.corpus` now builds ``profile["author_word_frequency"]``
    (``{author: {word: count}}``) and ``profile["author_word_frequency_total"]``
    (``{author: total_tokens}``) alongside the existing corpus-wide
    ``word_frequency``, populated whenever a book's manifest entry gives an
    ``author``. Both keys are optional and additive: ``build_profile``'s
    existing keys, arguments and behaviour are unchanged, a profile with no
    author metadata gets empty dicts for both, and this module reads them
    with ``.get(..., {})`` so a profile built *before* this change (which has
    neither key at all) degrades to the same "no per-author table" warning
    rather than raising.

Still deferred (named here rather than faked):

* **pystylometry, PyDelta, MOWEN, JGAAP, stylo (R), stylometry-cli, fastText,
  Vowpal Wabbit, MALLET, KenLM, pyppmd/PPM.** Permanently out of scope: none
  are installed in the reference environment, several require an R or JVM
  runtime this project does not shell out to, and a from-scratch
  reimplementation of any of them well enough to trust its numbers is a
  larger project than any pass can responsibly deliver. Their standard-
  library-reachable *ideas* are kept (Delta-style distances, NCD, n-gram
  cross-entropy, growth-curve fits, and now an embedding representation and
  a distance-based impostors approximation); their specific implementations,
  and KenLM's actual language-model machinery, are not.
* **NCD against specific reference documents/authors.** Still needs the *raw
  text* of the reference side, and corpus profiles deliberately do not
  retain raw book text (see :mod:`textgrader.corpus`'s module docstring on
  portability). NCD remains an *internal* measurement, between this
  document's opening and closing text.
* **Per-book embedding vectors stored in the corpus profile
  (``feature_profiles['embedding']``), enabling a true nearest/centroid
  *embedding* distance against specific reference books.** This task's file
  list authorizes editing :mod:`textgrader.corpus` for the per-author
  frequency table only; adding a second, vector-valued ``feature_profiles``
  entry (parallel to ``feature_profiles['function_words']``) the way
  ``embedding_style`` would need is a further, structural corpus-schema
  change deliberately left for a future pass rather than smuggled in under a
  narrower authorization. ``embedding_style`` answers a real question
  without it (see above); it does not answer the specific "nearest reference
  book by embedding" question the way ``corpus_reference`` answers it for
  function words.
* **Full impostors/unmasking with a real classifier and a large,
  independently-sampled impostor pool.** ``impostors`` (above) is a bounded
  approximation over the corpus profile's own aligned data; it does not
  retrain a classifier and does not carry the original technique's
  statistical guarantees.
* **Byte n-grams beyond order 2, POS n-grams beyond order 4, more than one
  dependency n-gram order at once.** Available as ``byte_ngram_orders``,
  ``pos_ngram_orders`` and ``dependency_ngram_order`` options for a user who
  wants more; the defaults keep the finding count and the parse-gated cost
  reasonable.
"""

from __future__ import annotations

import lzma
import math
import random
import statistics
import warnings
import zlib
from collections import Counter
from typing import Any, Mapping, Sequence

from .. import stats as stats_module
from ..document import DocumentAnalysis
from ..optional import require
from .common import MODERATE, cosine_distance, finding, option, rate
from .common import tokens as tokenize
from .dialogue_contractions import CONTRACTION_SUFFIXES, WHOLE_WORD_CONTRACTIONS
from .drift_chapter_zscores import get_sections
from .function_words import FUNCTION, vector as function_word_vector
from .punctuation_profile import MARK_NAMES, MARK_RE, marks_for
from .semantic_adjacent import embed_texts as _embed_texts

FAMILY = "authorial"
COST = MODERATE
# spaCy is used opportunistically by the pos_dependency feature group (default
# off) and lexicalrichness by lexicalrichness_crosscheck (default on, degrades
# to unavailable()); neither changes needs_parse/needs_model. sentence_transformers
# is used by embedding_style (default off, via semantic_adjacent.embed_texts) but
# is DELIBERATELY NOT listed here: MetricSpec.needs_model is "sentence_transformers"
# in requires with no finer grain, so listing it would flip needs_model for this
# WHOLE suite and silently drop it from the corpus builder's default profiling
# pass (see the module docstring's "The critical gating rule"). The feature flag,
# not this tuple, is what keeps the model from loading uninvited.
REQUIRES: tuple[str, ...] = ("lexicalrichness",)
MIN_SAMPLE = 300
UNIT_SENSITIVE = False

PREFIX = "style.stylometry_"

FUNCTION_SET = frozenset(FUNCTION)

# ---------------------------------------------------------------- defaults

DEFAULT_FEATURES: dict[str, bool] = {
    "character_ngrams": True,
    "byte_ngrams": True,
    "word_ngrams": True,
    "function_word_ngrams": True,
    "punctuation_shape": True,
    "word_shape": True,
    "affixes": True,
    "sentence_openings": True,
    "contractions_capitalization": True,
    "lexical_richness": True,
    "vocabulary_growth": True,
    "section_stability": True,
    "compression": True,
    "corpus_language_model": True,
    "corpus_reference": True,
    # Off by default: forces the shared spaCy parse, tens of seconds on a novel.
    "pos_dependency": False,
    # Cheap: reads profile["author_word_frequency"], which is just a dict lookup
    # and a linear pass over this document's own tokens per candidate author.
    "author_language_model": True,
    # Cheap: reuses the corpus-wide word_frequency table already read by
    # corpus_language_model, just against more distance families.
    "word_frequency_distance": True,
    # Cheap: lexicalrichness's own routines are the same order of cost as this
    # module's hand-rolled ones; degrades to unavailable() if the package is
    # missing or disabled.
    "lexicalrichness_crosscheck": True,
    # Off by default: loads a sentence-transformers model (see the module
    # docstring's "The critical gating rule" -- this is the ONLY feature flag
    # in this suite that gates a model load, and it must stay off by default).
    "embedding_style": False,
    # Off by default: a heavier, more specialized analysis than the rest of
    # this suite's cheap channels, and only meaningful with author-labelled
    # corpus data (see the module docstring's "Impostors-style verification").
    "impostors": False,
}

DEFAULT_CHAR_NGRAM_ORDERS = (2, 3, 4, 5, 6)
DEFAULT_BYTE_NGRAM_ORDERS = (2,)
DEFAULT_WORD_NGRAM_ORDERS = (1, 2, 3)
DEFAULT_POS_NGRAM_ORDERS = (1, 2, 3, 4)
DEFAULT_DEPENDENCY_NGRAM_ORDER = 2
DEFAULT_PUNCTUATION_NGRAM_ORDER = 2
DEFAULT_AFFIX_LENGTH = 3
DEFAULT_MIN_AFFIX_WORD_LENGTH = 5
DEFAULT_MAX_REPORTED = 25
DEFAULT_MAX_CHARS_FOR_NGRAMS = 500_000
DEFAULT_SECTION_WINDOW_WORDS = 3000
DEFAULT_SECTION_SHIFT_THRESHOLD = 2.5
DEFAULT_K_NEIGHBORS = 5
ALL_DISTANCE_METRICS = ("cosine", "euclidean", "manhattan", "jensen_shannon")
DEFAULT_DISTANCE_METRICS = ALL_DISTANCE_METRICS
DEFAULT_PRIMARY_DISTANCE = "cosine"
DEFAULT_COMPRESSION_ALGORITHM = "zlib"
DEFAULT_MIN_CORPUS_DOCUMENTS = 4
DEFAULT_MIN_DOCUMENTS_PER_AUTHOR = 2
DEFAULT_OUTLIER_THRESHOLD = 3.5
DEFAULT_SEED = 42

# word_frequency_distance: how much of the corpus's/document's own vocabulary
# to align on. Bounded so a huge corpus or a huge manuscript cannot turn this
# into an unbounded-size dict comparison.
DEFAULT_WORD_FREQUENCY_VOCAB_CAP = 3000

# embedding_style
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
#: Distance families that apply to a dense embedding vector. Jensen-Shannon is
#: intentionally excluded: an embedding is a point in a learned vector space,
#: not a probability distribution, so treating it as one would not mean what
#: it means for the function-word or word-frequency representations.
EMBEDDING_DISTANCE_METRICS = ("cosine", "euclidean", "manhattan")
DEFAULT_EMBEDDING_PRIMARY_DISTANCE = "cosine"

# impostors: every one of these bounds the cost of the group. m iterations of
# (1 candidate + up to k impostors) distance comparisons over a
# feature_fraction-sized slice of the (57-word) function-word list is cheap
# regardless of corpus size -- it is bounded by these constants, not by the
# corpus -- which is what makes this feasible where a real impostors
# implementation (per-iteration classifier refitting) is not.
DEFAULT_IMPOSTORS_K = 10
DEFAULT_IMPOSTORS_ITERATIONS = 25
DEFAULT_IMPOSTORS_FEATURE_FRACTION = 0.5
DEFAULT_IMPOSTORS_MIN_AUTHORS = 2

# Lexical-richness measures are unstable on short samples; every finding in
# that group uses this floor rather than the module's general MIN_SAMPLE.
MIN_SAMPLE_LEXICAL = 1000
# A section-to-section comparison needs several sections to mean anything;
# matches the threshold the sibling book-drift modules use for the same reason.
MIN_SAMPLE_SECTIONS = 4


def _feature(config: Mapping[str, Any] | None, name: str, default: bool) -> bool:
    """One ``features.<name>`` flag, defaulting individually.

    ``grade.py`` merges ``MetricSpec.defaults`` with the user's config one
    level deep (``dict.update``), so a user who overrides ``features`` to turn
    on just one group would otherwise silently turn every *other* group off.
    Reading each flag with its own default, rather than requiring the whole
    map to be re-specified, is what makes "enable the suite and flip one
    switch" behave the way the config file's own comment says it does.
    """

    features = option(config, "features", {})
    if not isinstance(features, Mapping):
        return default
    value = features.get(name, default)
    return default if value is None else bool(value)


def _int_list(config: Mapping[str, Any] | None, name: str, default: Sequence[int]) -> list[int]:
    raw = option(config, name, list(default))
    try:
        return sorted({int(item) for item in raw if int(item) > 0})
    except (TypeError, ValueError):
        return sorted(set(default))


# ------------------------------------------------------------- small maths

def _shannon(counter: Counter) -> float | None:
    total = sum(counter.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counter.values() if n) + 0.0


def _entropy_finding(metric_id: str, name: str, counter: Counter, *, min_sample: int,
                     max_reported: int, unit_label: str,
                     distribution_extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    total = sum(counter.values())
    if not total:
        return finding(metric_id, name, None, "bits", family=FAMILY, sample_size=0,
                       min_sample=min_sample, sample_size_sensitive=True,
                       warning=f"no {unit_label} to measure")
    dist: dict[str, Any] = {"distinct": len(counter), "occurrences": total}
    if distribution_extra:
        dist.update(distribution_extra)
    evidence = [{"item": item, "count": count} for item, count in counter.most_common(max_reported)]
    return finding(metric_id, name, _shannon(counter), "bits", family=FAMILY, sample_size=total,
                   min_sample=min_sample, distribution=dist, evidence=evidence,
                   # Entropy over an open category set rises with how much text
                   # produced it; see common.py's module docstring.
                   sample_size_sensitive=True)


def _linreg(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float, float] | None:
    """Ordinary least squares, ``(slope, intercept, r_squared)``."""

    if len(xs) < 3:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 0:
        return None
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = covariance / var_x
    intercept = mean_y - slope * mean_x
    predicted = [slope * x + intercept for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, predicted))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return slope, intercept, (r2 if r2 is not None else 1.0)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    mean_a, mean_b = statistics.fmean(a), statistics.fmean(b)
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return None
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    return covariance / math.sqrt(var_a * var_b)


def _rank_correlation(items: Sequence[str], order_a: Sequence[str], order_b: Sequence[str]) -> float | None:
    """Spearman correlation between two rank orders over the same item set."""

    rank_a = {item: index for index, item in enumerate(order_a)}
    rank_b = {item: index for index, item in enumerate(order_b)}
    shared = [item for item in items if item in rank_a and item in rank_b]
    if len(shared) < 3:
        return None
    return _pearson([rank_a[item] for item in shared], [rank_b[item] for item in shared])


# ------------------------------------------------------------ distance family

def _euclidean(a: Mapping[str, float], b: Mapping[str, float], keys: Sequence[str]) -> float:
    return math.sqrt(sum((a.get(k, 0.0) - b.get(k, 0.0)) ** 2 for k in keys))


def _manhattan(a: Mapping[str, float], b: Mapping[str, float], keys: Sequence[str]) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def _jensen_shannon(a: Mapping[str, float], b: Mapping[str, float], keys: Sequence[str]) -> float | None:
    """Jensen-Shannon divergence, in bits, treating each vector as a distribution.

    Function-word rates are not themselves a probability distribution, so each
    vector is renormalized to sum to one over ``keys`` first. A vector that is
    all zero (no function word observed at all, only possible on a degenerate
    document) has no distribution to compare, so this returns ``None``.
    """

    sum_a = sum(max(a.get(k, 0.0), 0.0) for k in keys)
    sum_b = sum(max(b.get(k, 0.0), 0.0) for k in keys)
    if sum_a <= 0 or sum_b <= 0:
        return None
    pa = {k: max(a.get(k, 0.0), 0.0) / sum_a for k in keys}
    pb = {k: max(b.get(k, 0.0), 0.0) / sum_b for k in keys}
    mid = {k: 0.5 * (pa[k] + pb[k]) for k in keys}

    def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
        return sum(p[k] * math.log2(p[k] / q[k]) for k in keys if p[k] > 0 and q.get(k, 0) > 0)

    return 0.5 * kl(pa, mid) + 0.5 * kl(pb, mid)


def _distance(name: str, a: Mapping[str, float], b: Mapping[str, float],
             keys: Sequence[str]) -> float | None:
    if name == "cosine":
        return cosine_distance(a, b)
    if name == "euclidean":
        return _euclidean(a, b, keys)
    if name == "manhattan":
        return _manhattan(a, b, keys)
    if name == "jensen_shannon":
        return _jensen_shannon(a, b, keys)
    return None


def _dense_distance(name: str, a: Sequence[float], b: Sequence[float]) -> float | None:
    """The subset of :data:`EMBEDDING_DISTANCE_METRICS` that apply to a dense,
    fixed-order vector (a sentence-embedding), rather than the sparse,
    key-addressed representation :func:`_distance` compares.

    Cosine distance normalizes both vectors itself rather than assuming they
    already carry unit norm: ``sentence-transformers`` embeddings do (see
    ``embed_texts``'s ``normalize_embeddings=True``), but a centroid built by
    averaging several of them generally does not, and this function is used
    for exactly that comparison (``embedding_dispersion``).
    """

    if len(a) != len(b) or len(a) == 0:
        return None
    if name == "cosine":
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
        norm_b = math.sqrt(sum(float(x) * float(x) for x in b))
        if norm_a <= 0 or norm_b <= 0:
            return None
        return 1.0 - dot / (norm_a * norm_b)
    if name == "euclidean":
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    if name == "manhattan":
        return sum(abs(float(x) - float(y)) for x, y in zip(a, b))
    return None


# --------------------------------------------------------- shared, cached views

def _sampled_text(analysis: DocumentAnalysis, cap: int, chunks: int = 8) -> str:
    """The text used for character/byte n-gram profiling, capped and sampled.

    A 300,000-word novel is roughly 1.8M characters; building five separate
    n-gram Counters (orders 2-6) over the whole thing is still linear, but
    "linear" is not free five times over on the largest documents this tool
    targets. Above ``cap`` characters, evenly spaced excerpts across the whole
    document are used instead of the raw first ``cap`` characters, which
    matters for books whose opening chapter has a different register (a
    prologue, a different narrator) than the rest -- a purely front-loaded
    sample would describe that chapter, not the book. Sampling is
    deterministic (fixed, evenly spaced offsets), so no seed is needed here;
    ``seed`` is retained as a documented option because requirement (10) asks
    for one, and because a future randomized sampling strategy would need it,
    but no step in this module currently consumes it.
    """

    text = analysis.text
    if len(text) <= cap:
        return text
    per_chunk = max(1, cap // chunks)
    step = len(text) / chunks
    pieces = [text[int(i * step):int(i * step) + per_chunk] for i in range(chunks)]
    return "\n".join(pieces)


def _char_ngram_counter(text: str, n: int) -> Counter:
    if len(text) < n:
        return Counter()
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def _byte_ngram_counter(data: bytes, n: int) -> Counter:
    if len(data) < n:
        return Counter()
    return Counter(data[i:i + n].hex() for i in range(len(data) - n + 1))


def _word_ngram_counter(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _word_frequencies(analysis: DocumentAnalysis) -> Counter:
    return analysis.memo("stylometry.word_freq", lambda: Counter(analysis.tokens))


def _punctuation_sequence(analysis: DocumentAnalysis) -> list[str]:
    """Ordered mark labels, reusing ``punctuation_profile``'s own regex.

    ``punctuation_profile.marks_for`` returns a Counter (order lost); this
    module additionally needs the sequence order for n-grams and rank
    stability, so it re-runs the same shared ``MARK_RE`` (not a new
    classification of its own) and caches the ordered result separately.
    """

    return analysis.memo("stylometry.punct_sequence",
                         lambda: [m.lastgroup for m in MARK_RE.finditer(analysis.text)])


def _word_shape(word: str) -> str:
    """Capitalization/punctuation shape of one token.

    No digit category: :data:`textgrader.text.WORD_RE` excludes digits from
    what counts as a "word" at all, so a token here is always a letter run
    with, at most, internal apostrophes; a stylometric "digit shape" family
    would have to be built over the raw text instead, which this suite does
    not currently do (see the module docstring's Deferred section for the
    general policy on not inventing a second tokenizer).
    """

    if "'" in word:
        return "apostrophe"
    if word.isupper() and len(word) > 1:
        return "upper"
    if word[:1].isupper() and word[1:].islower():
        return "title"
    if word.islower():
        return "lower"
    return "mixed"


# ---------------------------------------------------------------- A. n-grams

def _character_ngram_findings(analysis: DocumentAnalysis, config, max_reported) -> list[dict[str, Any]]:
    orders = _int_list(config, "char_ngram_orders", DEFAULT_CHAR_NGRAM_ORDERS)
    cap = int(option(config, "max_chars_for_ngrams", DEFAULT_MAX_CHARS_FOR_NGRAMS))
    text = _sampled_text(analysis, cap).lower()
    out = []
    for n in orders:
        counter = _char_ngram_counter(text, n)
        out.append(_entropy_finding(
            f"{PREFIX}char_ngram_entropy_{n}", f"Character {n}-gram entropy", counter,
            min_sample=max(50, 20 * n), max_reported=max_reported, unit_label=f"{n}-grams",
            distribution_extra={"order": n, "sampled_characters": len(text),
                                "full_document_characters": len(analysis.text)}))
    return out


def _byte_ngram_findings(analysis: DocumentAnalysis, config, max_reported) -> list[dict[str, Any]]:
    orders = _int_list(config, "byte_ngram_orders", DEFAULT_BYTE_NGRAM_ORDERS)
    cap = int(option(config, "max_chars_for_ngrams", DEFAULT_MAX_CHARS_FOR_NGRAMS))
    data = _sampled_text(analysis, cap).encode("utf-8")
    out = []
    for n in orders:
        counter = _byte_ngram_counter(data, n)
        out.append(_entropy_finding(
            f"{PREFIX}byte_ngram_entropy_{n}", f"Byte {n}-gram entropy", counter,
            min_sample=max(50, 20 * n), max_reported=max_reported, unit_label=f"byte {n}-grams",
            distribution_extra={"order": n, "sampled_bytes": len(data),
                                "note": "n-gram values are hex-encoded bytes"}))
    return out


def _word_ngram_findings(analysis: DocumentAnalysis, config, max_reported) -> list[dict[str, Any]]:
    orders = _int_list(config, "word_ngram_orders", DEFAULT_WORD_NGRAM_ORDERS)
    tokens = analysis.tokens
    out = []
    for n in orders:
        counter = _word_ngram_counter(tokens, n)
        out.append(_entropy_finding(
            f"{PREFIX}word_ngram_entropy_{n}", f"Word {n}-gram entropy", counter,
            min_sample=max(30, 10 * n), max_reported=max_reported, unit_label=f"word {n}-grams",
            distribution_extra={"order": n}))
    return out


def _function_word_ngram_finding(analysis: DocumentAnalysis, max_reported) -> dict[str, Any]:
    """Bigrams over the subsequence of tokens that are function words.

    This is what the brief calls "stopword sequences": the two names describe
    the same idea (a closed, high-frequency word list used as a sequence
    rather than a bag), so only one representation is built rather than two
    near-duplicates under different names. Non-function words are skipped
    rather than treated as a break, so "in the old house" contributes the
    pair ("in", "the") even though "old" and "house" sit between them in the
    surface text; this is deliberate; the whole point of a function-word
    n-gram is to see how function words chain together independent of the
    content words filling the gaps.
    """

    sequence = [token for token in analysis.tokens if token in FUNCTION_SET]
    counter = _word_ngram_counter(sequence, 2)
    return _entropy_finding(
        f"{PREFIX}function_word_bigram_entropy", "Function-word bigram entropy", counter,
        min_sample=100, max_reported=max_reported, unit_label="function-word bigrams",
        distribution_extra={"function_word_tokens": len(sequence)})


def _punctuation_ngram_finding(analysis: DocumentAnalysis, config, max_reported) -> dict[str, Any]:
    order = max(2, int(option(config, "punctuation_ngram_order", DEFAULT_PUNCTUATION_NGRAM_ORDER)))
    sequence = _punctuation_sequence(analysis)
    counter = _word_ngram_counter(sequence, order)
    return _entropy_finding(
        f"{PREFIX}punctuation_ngram_entropy", f"Punctuation {order}-gram entropy", counter,
        min_sample=50, max_reported=max_reported, unit_label="punctuation n-grams",
        distribution_extra={"order": order, "possible_marks": len(MARK_NAMES)})


def _word_shape_finding(analysis: DocumentAnalysis, max_reported) -> dict[str, Any]:
    counter = Counter(_word_shape(word) for word in analysis.words)
    return _entropy_finding(
        f"{PREFIX}word_shape_entropy", "Word-shape entropy (case/apostrophe pattern)", counter,
        min_sample=100, max_reported=max_reported, unit_label="words")


def _affix_findings(analysis: DocumentAnalysis, config, max_reported) -> list[dict[str, Any]]:
    length = max(1, int(option(config, "affix_length", DEFAULT_AFFIX_LENGTH)))
    min_len = max(length + 1, int(option(config, "min_affix_word_length", DEFAULT_MIN_AFFIX_WORD_LENGTH)))
    words = [token for token in analysis.tokens if token.isalpha() and len(token) >= min_len]
    prefixes = Counter(word[:length] for word in words)
    suffixes = Counter(word[-length:] for word in words)
    return [
        _entropy_finding(f"{PREFIX}prefix_entropy", f"Word-prefix entropy ({length} letters)",
                         prefixes, min_sample=100, max_reported=max_reported, unit_label="words",
                         distribution_extra={"affix_length": length, "min_word_length": min_len}),
        _entropy_finding(f"{PREFIX}suffix_entropy", f"Word-suffix entropy ({length} letters)",
                         suffixes, min_sample=100, max_reported=max_reported, unit_label="words",
                         distribution_extra={"affix_length": length, "min_word_length": min_len}),
    ]


def _sentence_opening_finding(analysis: DocumentAnalysis, max_reported) -> dict[str, Any]:
    """Entropy of the sentence-opening WORD distribution.

    A different lens than ``sentence_openings`` (which reports the share of
    sentences that repeat another sentence's opening N-word phrase, and its
    own entropy over multi-word openings): this one looks at only the single
    opening word, lower-cased, over the whole document, which is closer to
    the classic stylometric "which word does this author reach for first"
    signal than a repetition-detection measure is.
    """

    openings = []
    for sentence in analysis.sentences:
        words = tokenize(sentence)
        if words:
            openings.append(words[0])
    counter = Counter(openings)
    return _entropy_finding(
        f"{PREFIX}sentence_opening_word_entropy", "Sentence-opening word entropy", counter,
        min_sample=MIN_SAMPLE_SECTIONS * 5, max_reported=max_reported, unit_label="sentences")


def _contraction_and_capitalization_findings(analysis: DocumentAnalysis,
                                             max_reported) -> list[dict[str, Any]]:
    tokens = analysis.tokens
    total = len(tokens)
    if not total:
        return [
            finding(f"{PREFIX}contraction_rate", "Contraction rate (whole document)", None,
                    "%", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning="no words in text"),
            finding(f"{PREFIX}emphatic_capitalization_rate", "ALL-CAPS word rate", None,
                    "%", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning="no words in text"),
        ]
    contractions = [token for token in tokens
                    if token in WHOLE_WORD_CONTRACTIONS or token.endswith(CONTRACTION_SUFFIXES)]
    allcaps = [word for word in analysis.words if word.isupper() and len(word) > 1 and word != "I"]
    return [
        finding(f"{PREFIX}contraction_rate",
                "Contraction rate, whole document (narration and dialogue together)",
                rate(len(contractions), total), "%", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE,
                evidence=[{"item": token, "count": count}
                          for token, count in Counter(contractions).most_common(max_reported)],
                warning=None if contractions else "no contractions found"),
        finding(f"{PREFIX}emphatic_capitalization_rate",
                "Rate of ALL-CAPS words (excluding the pronoun \"I\"), per 100 words",
                rate(len(allcaps), total), "%", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE,
                evidence=[{"item": word, "count": count}
                          for word, count in Counter(allcaps).most_common(max_reported)],
                warning=None if allcaps else "no ALL-CAPS words found"),
    ]


# ------------------------------------------------------- B. lexical richness

def _lexical_richness_findings(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    """Classic type/frequency-spectrum diversity measures.

    All of these are computed from one frequency spectrum (``N`` tokens,
    ``V`` types, ``V_r`` = number of types occurring exactly ``r`` times) and
    are, without exception, sensitive to sample size: a longer text has more
    room for repeats, which mechanically moves every one of them. That is why
    every finding below carries ``min_sample=1000`` and
    ``sample_size_sensitive=True`` rather than the module's general
    ``MIN_SAMPLE``, and why ``grade.py`` will refuse to compare two of these
    computed from differently sized documents.

    Formula choices, since the literature is not consistent about constants:

    * Yule's K uses the conventional ``1e4`` scaling.
    * Yule's I is implemented as ``N^2 / (M2 - N)`` (Sichel 1986's form),
      which is the reciprocal relationship to K without K's ``1e4`` constant,
      not ``1/K`` itself; the two disagree by a factor of ``1e4`` on purpose.
    * The Uber index and "Dugast's U" are the same formula in the stylometric
      literature under two names (Dugast introduced it; Uber is the more
      common name in English-language tooling); reporting one instead of two
      near-identical findings under different labels is a judgement call
      documented rather than hidden by silently picking one name.
    """

    tokens = analysis.tokens
    n = len(tokens)
    counter = _word_frequencies(analysis)
    vocab = len(counter)
    min_sample = MIN_SAMPLE_LEXICAL

    def _lex(metric_id: str, name: str, unit: str, value: float | None, *,
             warning: str | None = None) -> dict[str, Any]:
        return finding(metric_id, name, value, unit, family=FAMILY, sample_size=n,
                       min_sample=min_sample, sample_size_sensitive=True,
                       distribution={"tokens": n, "types": vocab}, warning=warning)

    if n == 0 or vocab == 0:
        w = "no words in text"
        return [
            _lex(f"{PREFIX}hapax_ratio", "Hapax legomena ratio", "ratio", None, warning=w),
            _lex(f"{PREFIX}dislegomena_ratio", "Dislegomena ratio", "ratio", None, warning=w),
            _lex(f"{PREFIX}yules_k", "Yule's K", "K", None, warning=w),
            _lex(f"{PREFIX}yules_i", "Yule's I", "I", None, warning=w),
            _lex(f"{PREFIX}honore_r", "Honore's R", "R", None, warning=w),
            _lex(f"{PREFIX}sichel_s", "Sichel's S", "S", None, warning=w),
            _lex(f"{PREFIX}herdan_c", "Herdan's C", "C", None, warning=w),
            _lex(f"{PREFIX}guiraud_r", "Guiraud's R", "R", None, warning=w),
            _lex(f"{PREFIX}maas_index", "Maas's a^2", "a^2", None, warning=w),
            _lex(f"{PREFIX}uber_index", "Uber/Dugast index", "U", None, warning=w),
        ]

    spectrum = Counter(counter.values())  # r -> V_r
    v1, v2 = spectrum.get(1, 0), spectrum.get(2, 0)
    m2 = sum(r * r * vr for r, vr in spectrum.items())

    hapax_ratio = v1 / vocab
    dislegomena_ratio = v2 / vocab
    yules_k = 1e4 * (m2 - n) / (n * n) if n else None
    yules_i = (n * n) / (m2 - n) if m2 != n else None
    honore_r = (100 * math.log(n) / (1 - v1 / vocab)) if v1 != vocab and n > 1 else None
    sichel_s = v2 / vocab
    herdan_c = math.log(vocab) / math.log(n) if n > 1 and vocab > 1 else None
    guiraud_r = vocab / math.sqrt(n)
    log_n = math.log(n) if n > 1 else None
    log_v = math.log(vocab)
    maas_index = ((log_n - log_v) / (log_n ** 2)) if log_n else None
    uber_index = ((log_n ** 2) / (log_n - log_v)) if log_n and log_n != log_v else None

    return [
        _lex(f"{PREFIX}hapax_ratio", "Hapax legomena ratio (types occurring once / vocabulary)",
             "ratio", hapax_ratio),
        _lex(f"{PREFIX}dislegomena_ratio",
             "Dislegomena ratio (types occurring exactly twice / vocabulary)", "ratio",
             dislegomena_ratio),
        _lex(f"{PREFIX}yules_k", "Yule's K (higher = less diverse)", "K", yules_k),
        _lex(f"{PREFIX}yules_i", "Yule's I (higher = more diverse)", "I", yules_i,
             warning=None if yules_i is not None else
             "M2 equals N (every type occurs exactly once); Yule's I is undefined"),
        _lex(f"{PREFIX}honore_r", "Honore's R", "R", honore_r,
             warning=None if honore_r is not None else
             "every token is a hapax legomenon; Honore's R is undefined"),
        _lex(f"{PREFIX}sichel_s", "Sichel's S (dislegomena / vocabulary)", "S", sichel_s),
        _lex(f"{PREFIX}herdan_c", "Herdan's C (log V / log N)", "C", herdan_c),
        _lex(f"{PREFIX}guiraud_r", "Guiraud's R (V / sqrt(N))", "R", guiraud_r),
        _lex(f"{PREFIX}maas_index", "Maas's a^2 (lower = more diverse)", "a^2", maas_index),
        _lex(f"{PREFIX}uber_index", "Uber/Dugast index (higher = more diverse)", "U", uber_index,
             warning=None if uber_index is not None else
             "log N equals log V (every token is a distinct type); undefined"),
    ]


# ---------------------------------------------------- C. vocabulary growth

def _heaps_and_zipf_findings(analysis: DocumentAnalysis, max_reported) -> list[dict[str, Any]]:
    tokens = analysis.tokens
    n = len(tokens)
    if n < 100:
        warning = f"only {n} words; need at least 100 to fit a growth curve"
        return [
            finding(f"{PREFIX}heaps_exponent", "Heaps' law exponent", None, "exponent",
                    family=FAMILY, sample_size=n, min_sample=1000, sample_size_sensitive=True,
                    warning=warning),
            finding(f"{PREFIX}heaps_fit_r2", "Heaps' law fit, R-squared", None, "r_squared",
                    family=FAMILY, sample_size=n, min_sample=1000, sample_size_sensitive=True,
                    warning=warning),
            finding(f"{PREFIX}zipf_slope", "Zipf rank-frequency slope", None, "slope",
                    family=FAMILY, sample_size=n, min_sample=1000, sample_size_sensitive=True,
                    warning=warning),
            finding(f"{PREFIX}zipf_r2", "Zipf rank-frequency fit, R-squared", None, "r_squared",
                    family=FAMILY, sample_size=n, min_sample=1000, sample_size_sensitive=True,
                    warning=warning),
        ]

    # Heaps' law: vocabulary size as a function of tokens seen so far, sampled
    # at ~20 log-spaced checkpoints in one linear pass (a running set, not one
    # rebuild per checkpoint).
    checkpoints = sorted({max(50, int(n * (i / 20) ** 2)) for i in range(1, 21)} | {n})
    seen: set[str] = set()
    growth: list[tuple[int, int]] = []
    next_checkpoint = 0
    for index, token in enumerate(tokens, start=1):
        seen.add(token)
        while next_checkpoint < len(checkpoints) and index >= checkpoints[next_checkpoint]:
            growth.append((checkpoints[next_checkpoint], len(seen)))
            next_checkpoint += 1
    heaps_fit = _linreg([math.log(count) for count, _ in growth],
                        [math.log(vocab) for _, vocab in growth])
    heaps_exponent, heaps_r2 = (heaps_fit[0], heaps_fit[2]) if heaps_fit else (None, None)

    # Zipf's law: log rank vs log frequency over the most frequent types.
    ranked = Counter(tokens).most_common(min(1000, len(set(tokens))))
    zipf_fit = _linreg([math.log(rank) for rank, _ in enumerate(ranked, start=1)],
                       [math.log(count) for _, count in ranked])
    zipf_slope, zipf_intercept, zipf_r2 = zipf_fit if zipf_fit else (None, None, None)

    return [
        finding(f"{PREFIX}heaps_exponent", "Heaps' law exponent (vocabulary ~ N^exponent)",
                heaps_exponent, "exponent", family=FAMILY, sample_size=n, min_sample=1000,
                sample_size_sensitive=True,
                distribution={"checkpoints": len(growth), "fit_r2": heaps_r2},
                evidence=[{"tokens_seen": c, "vocabulary": v} for c, v in growth[:max_reported]]),
        finding(f"{PREFIX}heaps_fit_r2", "Heaps' law fit, R-squared", heaps_r2, "r_squared",
                family=FAMILY, sample_size=n, min_sample=1000, sample_size_sensitive=True,
                distribution={"exponent": heaps_exponent}),
        finding(f"{PREFIX}zipf_slope", "Zipf rank-frequency slope (natural language: near -1)",
                zipf_slope, "slope", family=FAMILY, sample_size=n, min_sample=1000,
                sample_size_sensitive=True,
                distribution={"intercept": zipf_intercept, "ranks_fit": len(ranked),
                             "fit_r2": zipf_r2},
                evidence=[{"rank": r, "word": w, "count": c}
                          for r, (w, c) in enumerate(ranked[:max_reported], start=1)]),
        finding(f"{PREFIX}zipf_r2", "Zipf rank-frequency fit, R-squared", zipf_r2, "r_squared",
                family=FAMILY, sample_size=n, min_sample=1000, sample_size_sensitive=True,
                distribution={"slope": zipf_slope, "ranks_fit": len(ranked)}),
    ]


# --------------------------------------------------- D. section-to-section stability

# Fixed histogram edges for word/sentence length, shared across every section
# of one document so the buckets line up and a Jensen-Shannon divergence
# between two sections' histograms is comparing like buckets to like buckets.
WORD_LENGTH_EDGES = (2, 4, 6, 8, 10, 14, 20)
SENTENCE_LENGTH_EDGES = (5, 10, 15, 20, 30, 45, 70)


def _section_style_vectors(sections: Sequence[tuple[str, DocumentAnalysis]]):
    return [view.memo("stylometry.section_fw_vector", lambda v=view: function_word_vector(v.text))
            for _, view in sections]


def _insufficient_section(method: str | None, count: int,
                          ids: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    warning = (f"only {count} section(s) available (method={method}); need at least "
               f"{MIN_SAMPLE_SECTIONS} to compare a book with itself" if method else
               "text has no measurable sections")
    return [finding(mid, name, None, None, family=FAMILY, sample_size=count,
                    min_sample=MIN_SAMPLE_SECTIONS, warning=warning) for mid, name in ids]


def _section_stability_findings(analysis: DocumentAnalysis, config, max_reported) -> list[dict[str, Any]]:
    window_words = int(option(config, "section_window_words", DEFAULT_SECTION_WINDOW_WORDS))
    primary = str(option(config, "primary_distance", DEFAULT_PRIMARY_DISTANCE))
    threshold = float(option(config, "section_shift_threshold", DEFAULT_SECTION_SHIFT_THRESHOLD))
    sections, method = get_sections(analysis, window_words)
    ids = (
        (f"{PREFIX}function_word_rank_stability", "Function-word rank stability across sections"),
        (f"{PREFIX}punctuation_rank_stability", "Punctuation-mark rank stability across sections"),
        (f"{PREFIX}word_length_section_distance",
         "Word-length histogram distance between adjacent sections"),
        (f"{PREFIX}sentence_length_section_distance",
         "Sentence-length histogram distance between adjacent sections"),
        (f"{PREFIX}style_vector_drift_open_close",
         "Function-word style-vector distance, opening section to closing section"),
        (f"{PREFIX}section_style_stability", "Median section-to-section function-word distance"),
        (f"{PREFIX}max_section_style_shift",
         "Largest single section-to-section function-word distance"),
        (f"{PREFIX}rolling_style_change_count", "Section-to-section gaps exceeding the shift threshold"),
    )
    if len(sections) < MIN_SAMPLE_SECTIONS:
        return _insufficient_section(method, len(sections), ids)

    n = len(sections)
    fw_vectors = _section_style_vectors(sections)
    mark_counts = [marks_for(view) for _, view in sections]
    word_hists = [stats_module.histogram([len(w) for w in view.words], WORD_LENGTH_EDGES)
                 for _, view in sections]
    sentence_hists = [stats_module.histogram(view.sentence_lengths, SENTENCE_LENGTH_EDGES)
                      for _, view in sections]

    fw_orders = [sorted(FUNCTION, key=lambda w, vec=vec: -vec.get(w, 0.0)) for vec in fw_vectors]
    mark_orders = [sorted(MARK_NAMES, key=lambda m, counts=counts: -counts.get(m, 0))
                  for counts in mark_counts]

    fw_correlations = [_rank_correlation(FUNCTION, fw_orders[i], fw_orders[i + 1])
                       for i in range(n - 1)]
    mark_correlations = [_rank_correlation(MARK_NAMES, mark_orders[i], mark_orders[i + 1])
                         for i in range(n - 1)]
    word_hist_distances = [_jensen_shannon(word_hists[i], word_hists[i + 1],
                                           list(word_hists[i].keys())) for i in range(n - 1)]
    sentence_hist_distances = [_jensen_shannon(sentence_hists[i], sentence_hists[i + 1],
                                               list(sentence_hists[i].keys())) for i in range(n - 1)]
    style_distances = [_distance(primary, fw_vectors[i], fw_vectors[i + 1], FUNCTION)
                       for i in range(n - 1)]

    def _median(values: Sequence[float | None]) -> float | None:
        clean = [v for v in values if v is not None]
        return statistics.median(clean) if clean else None

    open_close = _distance(primary, fw_vectors[0], fw_vectors[-1], FUNCTION)
    clean_style = [v for v in style_distances if v is not None]
    change_count = 0
    for i, value in enumerate(clean_style):
        others = clean_style[:i] + clean_style[i + 1:]
        comparison = stats_module.compare(value, others)
        if comparison.robust_distance is not None and abs(comparison.robust_distance) > threshold:
            change_count += 1

    section_labels = [title or f"section {i + 1}" for i, (title, _) in enumerate(sections)]
    pair_evidence = [{"from": section_labels[i], "to": section_labels[i + 1],
                      f"{primary}_distance": style_distances[i]}
                     for i in range(n - 1)][:max_reported]

    return [
        finding(f"{PREFIX}function_word_rank_stability",
                f"Function-word rank stability across sections (method={method})",
                _median(fw_correlations), "spearman", family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS,
                distribution={"method": method, "pairwise": fw_correlations}),
        finding(f"{PREFIX}punctuation_rank_stability",
                f"Punctuation-mark rank stability across sections (method={method})",
                _median(mark_correlations), "spearman", family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS,
                distribution={"method": method, "pairwise": mark_correlations}),
        finding(f"{PREFIX}word_length_section_distance",
                f"Word-length histogram distance between adjacent sections (method={method})",
                _median(word_hist_distances), "bits (JS divergence)", family=FAMILY,
                sample_size=n, min_sample=MIN_SAMPLE_SECTIONS,
                distribution={"method": method, "edges": WORD_LENGTH_EDGES}),
        finding(f"{PREFIX}sentence_length_section_distance",
                f"Sentence-length histogram distance between adjacent sections (method={method})",
                _median(sentence_hist_distances), "bits (JS divergence)", family=FAMILY,
                sample_size=n, min_sample=MIN_SAMPLE_SECTIONS,
                distribution={"method": method, "edges": SENTENCE_LENGTH_EDGES}),
        finding(f"{PREFIX}style_vector_drift_open_close",
                f"Function-word style-vector distance, opening section to closing section "
                f"(method={method}, distance={primary})",
                open_close, primary, family=FAMILY, sample_size=n, min_sample=MIN_SAMPLE_SECTIONS,
                distribution={"method": method, "distance_family": primary,
                             "opening_section": section_labels[0], "closing_section": section_labels[-1]},
                warning=None if open_close is not None else
                f"the opening or closing section used no function words at all, so {primary} "
                f"distance between them is undefined"),
        finding(f"{PREFIX}section_style_stability",
                f"Median section-to-section function-word distance (method={method}, "
                f"distance={primary})",
                _median(style_distances), primary, family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS, evidence=pair_evidence,
                distribution={"method": method, "distance_family": primary}),
        finding(f"{PREFIX}max_section_style_shift",
                f"Largest single section-to-section function-word distance (method={method}, "
                f"distance={primary})",
                max(clean_style) if clean_style else None, primary, family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS, evidence=pair_evidence,
                distribution={"method": method, "distance_family": primary}),
        finding(f"{PREFIX}rolling_style_change_count",
                f"Section-to-section gaps whose robust z exceeds {threshold} (method={method})",
                change_count, "sections", family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS, unit_sensitive=True,
                distribution={"method": method, "threshold": threshold, "pairs": len(clean_style)}),
    ]


# --------------------------------------------------------------- E. compression

def _compress(data: bytes, algorithm: str) -> bytes:
    if algorithm == "lzma":
        return lzma.compress(data, preset=6)
    return zlib.compress(data, level=9)


def _compression_findings(analysis: DocumentAnalysis, config) -> list[dict[str, Any]]:
    algorithm = str(option(config, "compression_algorithm", DEFAULT_COMPRESSION_ALGORITHM))
    if algorithm not in ("zlib", "lzma"):
        algorithm = DEFAULT_COMPRESSION_ALGORITHM
    if analysis.word_count < 200:
        warning = f"only {analysis.word_count} words; need at least 200 to measure compressibility"
        return [
            finding(f"{PREFIX}compression_ratio", "Whole-document compression ratio", None,
                    "ratio", family=FAMILY, sample_size=analysis.word_count, min_sample=200,
                    warning=warning),
            finding(f"{PREFIX}ncd_open_close",
                    "Normalized Compression Distance, opening half vs closing half", None,
                    "NCD", family=FAMILY, sample_size=analysis.word_count, min_sample=200,
                    warning=warning),
        ]

    data = analysis.text.encode("utf-8")
    ratio = len(_compress(data, algorithm)) / len(data)

    paragraphs = analysis.paragraphs
    if len(paragraphs) >= 2:
        half = len(paragraphs) // 2
        first_text = "\n\n".join(paragraphs[:half])
        second_text = "\n\n".join(paragraphs[half:])
    else:
        midpoint = len(analysis.text) // 2
        first_text, second_text = analysis.text[:midpoint], analysis.text[midpoint:]
    first, second = first_text.encode("utf-8"), second_text.encode("utf-8")
    ncd, ncd_warning = None, None
    if not first or not second:
        ncd_warning = "one half of the document was empty; NCD needs text on both sides"
    else:
        c_first = len(_compress(first, algorithm))
        c_second = len(_compress(second, algorithm))
        c_joint = len(_compress(first + second, algorithm))
        denominator = max(c_first, c_second)
        ncd = (c_joint - min(c_first, c_second)) / denominator if denominator else None

    return [
        finding(f"{PREFIX}compression_ratio",
                f"Whole-document compression ratio ({algorithm}; lower = more redundant/predictable)",
                ratio, "ratio", family=FAMILY, sample_size=analysis.word_count, min_sample=200,
                distribution={"algorithm": algorithm, "raw_bytes": len(data)}),
        finding(f"{PREFIX}ncd_open_close",
                f"Normalized Compression Distance, opening half vs closing half ({algorithm}); "
                f"an internal proxy for cross-document NCD, which needs reference text this "
                f"suite's corpus profiles do not retain (see module docstring)",
                ncd, "NCD", family=FAMILY, sample_size=analysis.word_count, min_sample=200,
                distribution={"algorithm": algorithm, "first_half_words": len(tokenize(first_text)),
                             "second_half_words": len(tokenize(second_text))},
                warning=ncd_warning),
    ]


# --------------------------------------------------- F. corpus-wide language model

def _corpus_language_model_findings(analysis: DocumentAnalysis, profile) -> list[dict[str, Any]]:
    """Cross-entropy/perplexity of this document under a corpus-wide unigram model.

    Genuinely author-specific language models are deferred (see the module
    docstring): :mod:`textgrader.corpus` stores one pooled word-frequency
    table for the whole corpus, not one per author, so what is computable
    here is "how surprising is this document's vocabulary given the corpus at
    large", not "given this author". That is still a real, useful signal --
    a document whose word choices are unusual against the corpus's language
    model will score high regardless of whose corpus it is -- and it is
    labelled as corpus-wide throughout rather than implied to be more than it
    is.
    """

    tokens = analysis.tokens
    n = len(tokens)
    frequency = (profile or {}).get("word_frequency")
    total = (profile or {}).get("word_frequency_total")
    if not profile or not frequency or not total:
        warning = "no corpus profile with a word-frequency table configured"
        return [
            finding(f"{PREFIX}corpus_unigram_cross_entropy",
                    "Cross-entropy under the corpus-wide unigram model", None, "bits/token",
                    family=FAMILY, sample_size=n, min_sample=200, warning=warning),
            finding(f"{PREFIX}corpus_unigram_perplexity",
                    "Perplexity under the corpus-wide unigram model", None, "perplexity",
                    family=FAMILY, sample_size=n, min_sample=200, warning=warning),
        ]
    if n == 0:
        warning = "no words in text"
        return [
            finding(f"{PREFIX}corpus_unigram_cross_entropy",
                    "Cross-entropy under the corpus-wide unigram model", None, "bits/token",
                    family=FAMILY, sample_size=0, min_sample=200, warning=warning),
            finding(f"{PREFIX}corpus_unigram_perplexity",
                    "Perplexity under the corpus-wide unigram model", None, "perplexity",
                    family=FAMILY, sample_size=0, min_sample=200, warning=warning),
        ]

    vocab_size = len(frequency)
    denominator = total + vocab_size  # Laplace (add-one) smoothing over the corpus vocabulary
    unseen = 0
    surprisal_sum = 0.0
    for token in tokens:
        count = frequency.get(token, 0)
        if count == 0:
            unseen += 1
        surprisal_sum += -math.log2((count + 1) / denominator)
    cross_entropy = surprisal_sum / n
    perplexity = 2 ** cross_entropy

    return [
        finding(f"{PREFIX}corpus_unigram_cross_entropy",
                "Cross-entropy under the corpus-wide unigram model (add-one smoothed)",
                cross_entropy, "bits/token", family=FAMILY, sample_size=n, min_sample=200,
                distribution={"corpus_vocabulary": vocab_size, "corpus_tokens": total,
                             "unseen_token_share": unseen / n}),
        finding(f"{PREFIX}corpus_unigram_perplexity",
                "Perplexity under the corpus-wide unigram model (add-one smoothed)",
                perplexity, "perplexity", family=FAMILY, sample_size=n, min_sample=200,
                distribution={"corpus_vocabulary": vocab_size, "corpus_tokens": total,
                             "unseen_token_share": unseen / n}),
    ]


# ------------------------------------------------------ G. corpus-reference distances

#: ``(metric_id, name)`` for every finding this group can emit, in one place so
#: the "corpus not usable" fallback and the real computation report the exact
#: same id/name pairs.
REFERENCE_METRICS: tuple[tuple[str, str], ...] = tuple(
    (f"{PREFIX}nearest_document_distance_{name}", f"Nearest reference-document distance ({name})")
    for name in ALL_DISTANCE_METRICS
) + (
    (f"{PREFIX}second_nearest_document_distance", "Second-nearest reference-document distance"),
    (f"{PREFIX}nearest_margin", "Margin between nearest and second-nearest reference document"),
    (f"{PREFIX}corpus_centroid_distance", "Distance to the overall corpus centroid"),
    (f"{PREFIX}out_of_distribution_distance",
     "Out-of-distribution distance (this document's nearest-neighbour distance, as a robust "
     "z against how far apart the corpus's own documents typically sit from their nearest "
     "neighbour)"),
    (f"{PREFIX}nearest_author_centroid_distance", "Distance to the nearest reference author's centroid"),
    (f"{PREFIX}within_author_dispersion", "Within-author dispersion for the nearest reference author"),
    (f"{PREFIX}standardized_distance_nearest_author",
     "Standardized distance to the nearest author (centroid distance / within-author dispersion)"),
    (f"{PREFIX}knn_author_agreement", "k-nearest-neighbour author agreement"),
)
_REFERENCE_NAMES = dict(REFERENCE_METRICS)
#: Unit label per finding, reused so the unavailable-fallback path and the
#: real computation report the same unit rather than the fallback going unitless.
_REFERENCE_UNITS: dict[str, str] = {
    f"{PREFIX}nearest_document_distance_{name}": name for name in ALL_DISTANCE_METRICS
} | {
    f"{PREFIX}second_nearest_document_distance": "distance",
    f"{PREFIX}nearest_margin": "distance",
    f"{PREFIX}corpus_centroid_distance": "distance",
    f"{PREFIX}out_of_distribution_distance": "robust z",
    f"{PREFIX}nearest_author_centroid_distance": "distance",
    f"{PREFIX}within_author_dispersion": "distance",
    f"{PREFIX}standardized_distance_nearest_author": "standardized distance",
    f"{PREFIX}knn_author_agreement": "share",
}


def _reference_unavailable(warning: str, sample_size: int, min_docs: int) -> list[dict[str, Any]]:
    return [finding(metric_id, name, None, _REFERENCE_UNITS.get(metric_id), family=FAMILY,
                    sample_size=sample_size, min_sample=min_docs, warning=warning)
            for metric_id, name in REFERENCE_METRICS]


def _own_nearest_neighbor_distances(rows: Sequence[Mapping[str, float]], primary: str,
                                    cap: int = 300) -> list[float]:
    """Each reference document's distance to ITS OWN nearest neighbour.

    This is the reference distribution :func:`_corpus_reference_findings` uses
    for the out-of-distribution finding: not "is this document far from the
    centroid" but "is this document further from its nearest match than
    corpus documents usually are from theirs". Leave-one-out by construction
    -- a document is never its own neighbour. O(cap * M); ``cap`` bounds this
    for large corpora rather than letting a many-hundred-book corpus turn one
    finding into an O(M^2) pass with no ceiling.
    """

    m = len(rows)
    if m < 2:
        return []
    indices = list(range(m)) if m <= cap else [int(i * m / cap) for i in range(cap)]
    out = []
    for i in indices:
        best = None
        for j in range(m):
            if j == i:
                continue
            d = _distance(primary, rows[i], rows[j], FUNCTION)
            if d is not None and (best is None or d < best):
                best = d
        if best is not None:
            out.append(best)
    return out


def _corpus_reference_findings(analysis: DocumentAnalysis, config, profile,
                               max_reported) -> list[dict[str, Any]]:
    min_docs = max(2, int(option(config, "min_corpus_documents", DEFAULT_MIN_CORPUS_DOCUMENTS)))
    outlier_threshold = float(option(config, "outlier_threshold", DEFAULT_OUTLIER_THRESHOLD))
    rows = ((profile or {}).get("feature_profiles") or {}).get("function_words")
    books = (profile or {}).get("books")
    sample_size = analysis.word_count
    if not profile or not rows or not books or len(rows) != len(books):
        return _reference_unavailable(
            "no corpus profile with aligned function-word feature vectors configured "
            "(build one with textgrader.corpus and point corpus_profile at it)",
            sample_size, min_docs)
    if len(rows) < min_docs:
        return _reference_unavailable(
            f"corpus has only {len(rows)} document(s); need at least {min_docs} "
            f"(set corpus_reference.min_corpus_documents to lower the floor)",
            sample_size, min_docs)

    doc_vector = function_word_vector(analysis.text)
    if not any(doc_vector.values()):
        return _reference_unavailable("no function words found in this document", sample_size, min_docs)

    requested = [name for name in option(config, "distance_metrics", list(DEFAULT_DISTANCE_METRICS))
                if name in ALL_DISTANCE_METRICS]
    primary = str(option(config, "primary_distance", DEFAULT_PRIMARY_DISTANCE))
    if primary not in ALL_DISTANCE_METRICS:
        primary = DEFAULT_PRIMARY_DISTANCE
    m = len(rows)
    out: list[dict[str, Any]] = []

    # --- one nearest-document finding per requested distance family --------
    for name in ALL_DISTANCE_METRICS:
        metric_id = f"{PREFIX}nearest_document_distance_{name}"
        display_name = _REFERENCE_NAMES[metric_id]
        if name not in requested:
            out.append(finding(metric_id, display_name, None, name, family=FAMILY,
                               sample_size=m, min_sample=min_docs,
                               warning="excluded via the distance_metrics option"))
            continue
        distances = [(i, d) for i, row in enumerate(rows)
                    if (d := _distance(name, doc_vector, row, FUNCTION)) is not None]
        if not distances:
            out.append(finding(metric_id, display_name, None, name, family=FAMILY,
                               sample_size=m, min_sample=min_docs,
                               warning="no corpus row produced a comparable vector"))
            continue
        distances.sort(key=lambda item: item[1])
        nearest_i, nearest_d = distances[0]
        out.append(finding(metric_id, display_name, nearest_d, name, family=FAMILY,
                           sample_size=m, min_sample=min_docs,
                           distribution={"distance_family": name, "corpus_size": m},
                           evidence=[{"source_id": books[nearest_i].get("source_id"),
                                     "distance": nearest_d}]))

    # --- everything else uses one primary distance family -------------------
    primary_distances = [(i, d) for i, row in enumerate(rows)
                        if (d := _distance(primary, doc_vector, row, FUNCTION)) is not None]
    if not primary_distances:
        remaining = [pair for pair in REFERENCE_METRICS
                    if pair[0] not in {f"{PREFIX}nearest_document_distance_{n}" for n in ALL_DISTANCE_METRICS}]
        out.extend(finding(mid, name, None, _REFERENCE_UNITS.get(mid), family=FAMILY, sample_size=m,
                           min_sample=min_docs,
                           warning=f"no corpus row was comparable under the primary distance "
                                   f"({primary})")
                   for mid, name in remaining)
        return out
    primary_distances.sort(key=lambda item: item[1])
    nearest_i, nearest_d = primary_distances[0]
    second_pair = primary_distances[1] if len(primary_distances) > 1 else None

    out.append(finding(f"{PREFIX}second_nearest_document_distance", _REFERENCE_NAMES[
        f"{PREFIX}second_nearest_document_distance"], second_pair[1] if second_pair else None,
        primary, family=FAMILY, sample_size=m, min_sample=min_docs,
        distribution={"distance_family": primary},
        warning=None if second_pair else "corpus has only one comparable document",
        evidence=[{"source_id": books[second_pair[0]].get("source_id")}] if second_pair else None))
    out.append(finding(f"{PREFIX}nearest_margin", _REFERENCE_NAMES[f"{PREFIX}nearest_margin"],
        (second_pair[1] - nearest_d) if second_pair else None, primary, family=FAMILY,
        sample_size=m, min_sample=min_docs, distribution={"distance_family": primary},
        warning=None if second_pair else "corpus has only one comparable document"))

    centroid = {word: statistics.fmean(row.get(word, 0.0) for row in rows) for word in FUNCTION}
    centroid_distance = _distance(primary, doc_vector, centroid, FUNCTION)
    out.append(finding(f"{PREFIX}corpus_centroid_distance",
        _REFERENCE_NAMES[f"{PREFIX}corpus_centroid_distance"], centroid_distance, primary,
        family=FAMILY, sample_size=m, min_sample=min_docs,
        distribution={"distance_family": primary}))

    own_nearest = _own_nearest_neighbor_distances(rows, primary)
    ood = (stats_module.compare(nearest_d, own_nearest, threshold=outlier_threshold)
          if own_nearest else None)
    out.append(finding(f"{PREFIX}out_of_distribution_distance",
        _REFERENCE_NAMES[f"{PREFIX}out_of_distribution_distance"],
        ood.robust_distance if ood else None, "robust z", family=FAMILY, sample_size=m,
        min_sample=min_docs,
        distribution={"distance_family": primary, "method": ood.method if ood else None,
                     "reference_documents": len(own_nearest), "threshold": outlier_threshold,
                     "beyond_threshold": (abs(ood.robust_distance) > outlier_threshold
                                         if ood and ood.robust_distance is not None else None)},
        warning=None if own_nearest else "corpus is too small to build a nearest-neighbour baseline"))

    # --- author-labelled sub-family: needs corpus manifest metadata.author --
    authors = [((books[i].get("metadata") or {}).get("author")) for i in range(m)]
    min_per_author = max(1, int(option(config, "min_documents_per_author",
                                       DEFAULT_MIN_DOCUMENTS_PER_AUTHOR)))
    author_counts = Counter(a for a in authors if a)
    nearest_author = authors[nearest_i]
    label_ids = (f"{PREFIX}nearest_author_centroid_distance", f"{PREFIX}within_author_dispersion",
                f"{PREFIX}standardized_distance_nearest_author", f"{PREFIX}knn_author_agreement")
    if not nearest_author or author_counts.get(nearest_author, 0) < min_per_author:
        reason = ("no author metadata in this corpus profile's book manifest" if not author_counts
                 else f"the nearest reference document's author has fewer than {min_per_author} "
                      f"corpus documents to build a centroid/dispersion from")
        out.extend(finding(mid, _REFERENCE_NAMES[mid], None, _REFERENCE_UNITS.get(mid),
                           family=FAMILY, sample_size=m, min_sample=min_docs, warning=reason)
                   for mid in label_ids)
        return out

    author_rows = [rows[i] for i in range(m) if authors[i] == nearest_author]
    author_centroid = {word: statistics.fmean(row.get(word, 0.0) for row in author_rows)
                      for word in FUNCTION}
    author_centroid_distance = _distance(primary, doc_vector, author_centroid, FUNCTION)
    dispersion_values = [_distance(primary, row, author_centroid, FUNCTION) for row in author_rows]
    dispersion_values = [d for d in dispersion_values if d is not None]
    within_dispersion = statistics.fmean(dispersion_values) if dispersion_values else None
    standardized = (author_centroid_distance / within_dispersion
                   if author_centroid_distance is not None and within_dispersion else None)

    k = max(1, int(option(config, "k_neighbors", DEFAULT_K_NEIGHBORS)))
    neighbor_authors = [authors[i] for i, _ in primary_distances[:k] if authors[i]]
    knn_agreement, knn_note = None, None
    if neighbor_authors:
        modal_author, modal_count = Counter(neighbor_authors).most_common(1)[0]
        knn_agreement = modal_count / len(neighbor_authors)
    else:
        knn_note = "none of the k nearest reference documents carry an author label"

    out.append(finding(f"{PREFIX}nearest_author_centroid_distance",
        _REFERENCE_NAMES[f"{PREFIX}nearest_author_centroid_distance"], author_centroid_distance,
        primary, family=FAMILY, sample_size=m, min_sample=min_docs,
        distribution={"distance_family": primary, "author_documents": len(author_rows)},
        evidence=[{"author": nearest_author}]))
    out.append(finding(f"{PREFIX}within_author_dispersion",
        _REFERENCE_NAMES[f"{PREFIX}within_author_dispersion"], within_dispersion, primary,
        family=FAMILY, sample_size=len(author_rows), min_sample=min_per_author,
        distribution={"distance_family": primary}, evidence=[{"author": nearest_author}]))
    out.append(finding(f"{PREFIX}standardized_distance_nearest_author",
        _REFERENCE_NAMES[f"{PREFIX}standardized_distance_nearest_author"], standardized,
        "standardized distance", family=FAMILY, sample_size=len(author_rows),
        min_sample=min_per_author, distribution={"distance_family": primary},
        warning=None if standardized is not None else
        "within-author dispersion is zero or unavailable"))
    out.append(finding(f"{PREFIX}knn_author_agreement",
        _REFERENCE_NAMES[f"{PREFIX}knn_author_agreement"], knn_agreement, "share", family=FAMILY,
        sample_size=len(neighbor_authors), min_sample=1,
        distribution={"k": k, "distance_family": primary}, warning=knn_note))
    return out


# --------------------------------------------------------- H. POS / dependency

def _pos_sequences(analysis: DocumentAnalysis):
    """``(pos_tags, dep_labels, dep_pos_pairs, sentence_opening_pos)`` over one shared parse.

    Built in a single pass over ``analysis.spacy_docs()`` and memoized, so
    enabling every finding in this group still costs exactly one parse and
    one pass over its tokens, not four.
    """

    def build():
        pos_tags: list[str] = []
        dep_labels: list[str] = []
        dep_pos_pairs: list[str] = []
        opening_pos: list[str] = []
        for _, doc in analysis.spacy_docs():
            for sent in doc.sents:
                first = True
                for token in sent:
                    if token.is_space or token.is_punct:
                        continue
                    pos_tags.append(token.pos_)
                    dep_labels.append(token.dep_)
                    dep_pos_pairs.append(f"{token.dep_}/{token.pos_}")
                    if first:
                        opening_pos.append(token.pos_)
                        first = False
        return pos_tags, dep_labels, dep_pos_pairs, opening_pos

    return analysis.memo("stylometry.pos_sequences", build)


def _pos_dependency_findings(analysis: DocumentAnalysis, config, max_reported) -> list[dict[str, Any]]:
    orders = _int_list(config, "pos_ngram_orders", DEFAULT_POS_NGRAM_ORDERS)
    dep_order = max(2, int(option(config, "dependency_ngram_order", DEFAULT_DEPENDENCY_NGRAM_ORDER)))
    ids = [(f"{PREFIX}pos_ngram_entropy_{n}", f"POS {n}-gram entropy") for n in orders] + [
        (f"{PREFIX}dependency_ngram_entropy", f"Dependency-label {dep_order}-gram entropy"),
        (f"{PREFIX}dependency_pos_combo_entropy", "Dependency-relation + POS combination entropy"),
        (f"{PREFIX}sentence_opening_pos_entropy", "Sentence-opening POS entropy"),
    ]
    reason = analysis.nlp_unavailable
    if reason:
        return [finding(mid, name, None, "bits", family=FAMILY, sample_size=0, min_sample=100,
                        sample_size_sensitive=True, warning=f"spaCy parse unavailable: {reason}")
               for mid, name in ids]

    pos_tags, dep_labels, dep_pos_pairs, opening_pos = _pos_sequences(analysis)
    if not pos_tags:
        return [finding(mid, name, None, "bits", family=FAMILY, sample_size=0, min_sample=100,
                        sample_size_sensitive=True, warning="the parse produced no tokens")
               for mid, name in ids]

    out = [_entropy_finding(f"{PREFIX}pos_ngram_entropy_{n}", f"POS {n}-gram entropy",
                            _word_ngram_counter(pos_tags, n), min_sample=100,
                            max_reported=max_reported, unit_label=f"POS {n}-grams",
                            distribution_extra={"order": n})
          for n in orders]
    out.append(_entropy_finding(
        f"{PREFIX}dependency_ngram_entropy", f"Dependency-label {dep_order}-gram entropy",
        _word_ngram_counter(dep_labels, dep_order), min_sample=100, max_reported=max_reported,
        unit_label="dependency n-grams", distribution_extra={"order": dep_order}))
    out.append(_entropy_finding(
        f"{PREFIX}dependency_pos_combo_entropy", "Dependency-relation + POS combination entropy",
        Counter(dep_pos_pairs), min_sample=100, max_reported=max_reported, unit_label="tokens"))
    out.append(_entropy_finding(
        f"{PREFIX}sentence_opening_pos_entropy", "Sentence-opening POS entropy",
        Counter(opening_pos), min_sample=MIN_SAMPLE_SECTIONS * 5, max_reported=max_reported,
        unit_label="sentences"))
    return out


# --------------------------------------------- I. word-frequency distance family

def _bounded_vocabulary(doc_counts: Counter, corpus_frequency: Mapping[str, int],
                        cap: int) -> list[str]:
    """The corpus's ``cap`` commonest words unioned with this document's own
    ``cap`` commonest words, so the vocabulary a rare document introduces is
    not simply invisible to the comparison, while the total stays bounded
    regardless of how large the corpus or the document is."""

    top_corpus = {word for word, _ in Counter(corpus_frequency).most_common(cap)}
    top_doc = {word for word, _ in doc_counts.most_common(cap)}
    return sorted(top_corpus | top_doc)


def _word_frequency_distance_findings(analysis: DocumentAnalysis, config,
                                      profile) -> list[dict[str, Any]]:
    """Distance, in each of the four families, between this document's own
    word-frequency distribution and the corpus's pooled one.

    Closes the "distance families beyond function words" gap: ``profile``
    already carries one more aligned representation besides
    ``feature_profiles['function_words']`` -- the corpus-wide
    ``word_frequency`` table ``corpus_language_model`` already reads for
    cross-entropy -- so the same four distance families used for the
    function-word comparison apply here too, over a bounded shared vocabulary
    (see :func:`_bounded_vocabulary`).
    """

    frequency = (profile or {}).get("word_frequency")
    total = (profile or {}).get("word_frequency_total")
    sample_size = analysis.word_count

    def _unavailable(warning: str, size: int) -> list[dict[str, Any]]:
        return [finding(f"{PREFIX}word_frequency_distance_{name}",
                        f"Document-vs-corpus word-frequency distance ({name})", None, name,
                        family=FAMILY, sample_size=size, min_sample=200, warning=warning)
               for name in ALL_DISTANCE_METRICS]

    if not profile or not frequency or not total:
        return _unavailable("no corpus profile with a word-frequency table configured",
                            sample_size)
    doc_counts = _word_frequencies(analysis)
    n = sum(doc_counts.values())
    if n == 0:
        return _unavailable("no words in text", 0)
    cap = max(50, int(option(config, "word_frequency_vocab_cap", DEFAULT_WORD_FREQUENCY_VOCAB_CAP)))
    keys = _bounded_vocabulary(doc_counts, frequency, cap)
    if not keys:
        return _unavailable("no shared vocabulary between this document and the corpus", n)

    doc_vector = {word: doc_counts.get(word, 0) / n for word in keys}
    corpus_vector = {word: frequency.get(word, 0) / total for word in keys}
    return [
        finding(f"{PREFIX}word_frequency_distance_{name}",
                f"Document-vs-corpus word-frequency distance ({name})",
                _distance(name, doc_vector, corpus_vector, keys), name, family=FAMILY,
                sample_size=n, min_sample=200,
                distribution={"distance_family": name, "vocabulary_size": len(keys),
                             "vocabulary_cap": cap})
        for name in ALL_DISTANCE_METRICS
    ]


# ------------------------------------------------ J. per-author language model

def _score_authors_by_cross_entropy(
        tokens: Sequence[str], n: int, author_frequency: Mapping[str, Mapping[str, int]],
        author_totals: Mapping[str, int]) -> list[tuple[str, float, int, int]]:
    """``[(author, cross_entropy, author_vocabulary, author_tokens), ...]``, best fit first.

    Add-one (Laplace) smoothed over each author's OWN vocabulary -- the same
    smoothing :func:`_corpus_language_model_findings` uses for the corpus-wide
    table -- so an author-specific cross-entropy and the corpus-wide one read
    the same way and differ only in whose frequency table produced them.
    """

    scored = []
    for author, freq in (author_frequency or {}).items():
        total = (author_totals or {}).get(author)
        if not freq or not total:
            continue
        vocab_size = len(freq)
        denominator = total + vocab_size
        surprisal = sum(-math.log2((freq.get(token, 0) + 1) / denominator) for token in tokens)
        scored.append((author, surprisal / n, vocab_size, total))
    scored.sort(key=lambda row: row[1])
    return scored


def _author_language_model_findings(analysis: DocumentAnalysis, profile,
                                    max_reported: int) -> list[dict[str, Any]]:
    """Cross-entropy under the best-fitting REFERENCE AUTHOR's unigram model.

    Genuinely author-specific, unlike :func:`_corpus_language_model_findings`'s
    corpus-*wide* table: this reads ``profile['author_word_frequency']``,
    built by :mod:`textgrader.corpus` from books whose manifest entries carry
    an ``author`` (see the module docstring's "Corpus schema" section), and
    scores this document against every author's table rather than picking one
    in advance, so "best fit" is a real minimum over the corpus's authors
    rather than an assumption.
    """

    ids = (
        (f"{PREFIX}author_unigram_cross_entropy_best_fit",
         "Cross-entropy under the best-fitting reference author's unigram model"),
        (f"{PREFIX}author_unigram_perplexity_best_fit",
         "Perplexity under the best-fitting reference author's unigram model"),
        (f"{PREFIX}author_unigram_cross_entropy_margin",
         "Cross-entropy margin, best-fitting author vs second-best-fitting author"),
    )
    units = ("bits/token", "perplexity", "bits/token")
    tokens = analysis.tokens
    n = len(tokens)
    author_frequency = (profile or {}).get("author_word_frequency") or {}
    author_totals = (profile or {}).get("author_word_frequency_total") or {}

    if not profile or not author_frequency:
        warning = ("no per-author frequency table in this corpus profile (build one with "
                  "textgrader.corpus from books whose manifest gives each an 'author', or "
                  "rebuild an older profile that predates this table)")
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=n, min_sample=200,
                        warning=warning) for (mid, name), unit in zip(ids, units)]
    if n == 0:
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=0, min_sample=200,
                        warning="no words in text") for (mid, name), unit in zip(ids, units)]

    scored = _score_authors_by_cross_entropy(tokens, n, author_frequency, author_totals)
    if not scored:
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=n, min_sample=200,
                        warning="the per-author frequency table has no author with both a word "
                                "count and a recorded token total")
               for (mid, name), unit in zip(ids, units)]

    best_author, best_ce, best_vocab, best_total = scored[0]
    second = scored[1] if len(scored) > 1 else None
    margin = (second[1] - best_ce) if second else None
    evidence = [{"author": author, "cross_entropy": ce} for author, ce, *_ in scored[:max_reported]]
    common = {"author": best_author, "author_vocabulary": best_vocab, "author_tokens": best_total,
             "authors_scored": len(scored)}
    return [
        finding(ids[0][0], ids[0][1], best_ce, "bits/token", family=FAMILY, sample_size=n,
                min_sample=200, distribution=common, evidence=evidence),
        finding(ids[1][0], ids[1][1], 2 ** best_ce, "perplexity", family=FAMILY, sample_size=n,
                min_sample=200, distribution=common),
        finding(ids[2][0], ids[2][1], margin, "bits/token", family=FAMILY, sample_size=n,
                min_sample=200, distribution={**common, "second_author": second[0] if second else None},
                warning=None if margin is not None else
                "only one author has a usable frequency table; no second-best to compare"),
    ]


# -------------------------------------------------- K. impostors verification

_IMPOSTORS_CAVEAT = (
    "distance-based approximation of the impostors technique: repeated distance comparisons "
    "against author centroids already stored in the corpus profile, over random function-word "
    "subsets, rather than a refitted classifier per iteration; it does not carry the original "
    "technique's statistical guarantees (see the module docstring's 'Impostors-style "
    "verification')")


def _impostors_findings(analysis: DocumentAnalysis, config, profile) -> list[dict[str, Any]]:
    """A bounded, distance-based approximation of impostors verification.

    See the module docstring's "Impostors-style verification" section for the
    algorithm and for why this is an approximation rather than the literature's
    technique. Every finding this function returns carries
    :data:`_IMPOSTORS_CAVEAT` in its warning, on both the success and the
    degraded path, because that caveat is the one thing a reader must not miss
    about a number called "impostors verification score".
    """

    ids = (
        (f"{PREFIX}impostors_verification_score",
         "Impostors verification score (mean share of sampled impostor authors this document "
         "sits closer to the candidate author than to, across feature-subset iterations)"),
        (f"{PREFIX}impostors_score_variance",
         "Variance of the impostors verification score across feature-subset iterations"),
    )
    units = ("share", "share^2")

    def _unavailable(warning: str, size: int) -> list[dict[str, Any]]:
        combined = f"{warning}; {_IMPOSTORS_CAVEAT}"
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=size, min_sample=1,
                        warning=combined) for (mid, name), unit in zip(ids, units)]

    rows = ((profile or {}).get("feature_profiles") or {}).get("function_words")
    books = (profile or {}).get("books")
    if not profile or not rows or not books or len(rows) != len(books):
        return _unavailable("no corpus profile with aligned function-word feature vectors "
                            "and author metadata configured", 0)
    authors = [((books[i].get("metadata") or {}).get("author")) for i in range(len(books))]
    author_counts = Counter(author for author in authors if author)
    if not author_counts:
        return _unavailable("no author metadata in this corpus profile's book manifest", len(rows))

    doc_vector = function_word_vector(analysis.text)
    if not any(doc_vector.values()):
        return _unavailable("no function words found in this document", len(rows))

    primary = str(option(config, "primary_distance", DEFAULT_PRIMARY_DISTANCE))
    if primary not in ALL_DISTANCE_METRICS:
        primary = DEFAULT_PRIMARY_DISTANCE
    min_per_author = max(1, int(option(config, "min_documents_per_author",
                                       DEFAULT_MIN_DOCUMENTS_PER_AUTHOR)))

    target_author = option(config, "impostors_target_author", None)
    if target_author and author_counts.get(target_author, 0) < min_per_author:
        target_author = None
    if not target_author:
        distances = sorted(
            ((authors[i], d) for i in range(len(rows)) if authors[i]
             and (d := _distance(primary, doc_vector, rows[i], FUNCTION)) is not None),
            key=lambda item: item[1])
        if not distances:
            return _unavailable("no author-labelled corpus row was comparable under the "
                                "primary distance", len(rows))
        target_author = distances[0][0]
    if author_counts.get(target_author, 0) < min_per_author:
        return _unavailable(f"candidate author {target_author!r} has fewer than "
                            f"{min_per_author} corpus document(s) to build a centroid from",
                            author_counts.get(target_author, 0))

    other_authors = sorted(author for author in author_counts if author != target_author)
    min_impostors = max(1, int(option(config, "impostors_min_authors",
                                      DEFAULT_IMPOSTORS_MIN_AUTHORS)))
    if len(other_authors) < min_impostors:
        return _unavailable(f"need at least {min_impostors} other reference author(s) to sample "
                            f"impostors from; this corpus has {len(other_authors)}", len(rows))

    seed = int(option(config, "seed", DEFAULT_SEED))
    rng = random.Random(seed)
    k = max(1, int(option(config, "impostors_k", DEFAULT_IMPOSTORS_K)))
    m = max(1, int(option(config, "impostors_iterations", DEFAULT_IMPOSTORS_ITERATIONS)))
    fraction = min(1.0, max(0.05, float(option(config, "impostors_feature_fraction",
                                               DEFAULT_IMPOSTORS_FEATURE_FRACTION))))
    impostor_authors = (other_authors if len(other_authors) <= k
                        else rng.sample(other_authors, k))

    target_rows = [rows[i] for i in range(len(rows)) if authors[i] == target_author]
    impostor_rows = {author: [rows[i] for i in range(len(rows)) if authors[i] == author]
                     for author in impostor_authors}
    feature_pool = list(FUNCTION)
    subset_size = max(3, round(len(feature_pool) * fraction))

    def _centroid(rows_subset: Sequence[Mapping[str, float]],
                 keys: Sequence[str]) -> dict[str, float]:
        return {word: statistics.fmean(row.get(word, 0.0) for row in rows_subset) for word in keys}

    scores: list[float] = []
    for _ in range(m):
        subset = rng.sample(feature_pool, min(subset_size, len(feature_pool)))
        d_target = _distance(primary, doc_vector, _centroid(target_rows, subset), subset)
        if d_target is None:
            continue
        wins = counted = 0
        for impostor_book_rows in impostor_rows.values():
            if not impostor_book_rows:
                continue
            d_impostor = _distance(primary, doc_vector, _centroid(impostor_book_rows, subset), subset)
            if d_impostor is None:
                continue
            counted += 1
            if d_target < d_impostor:
                wins += 1
        if counted:
            scores.append(wins / counted)

    if not scores:
        return _unavailable("no iteration produced a comparable distance; this document and "
                            "the sampled corpus rows may share no function words", len(rows))

    mean_score = statistics.fmean(scores)
    variance = statistics.pvariance(scores) if len(scores) > 1 else 0.0
    common = {"candidate_author": target_author, "impostor_authors": sorted(impostor_rows),
             "k": len(impostor_rows), "iterations": len(scores),
             "feature_subset_size": subset_size, "distance_family": primary, "seed": seed}
    evidence = [{"iteration": index, "score": score} for index, score in enumerate(scores)][:25]
    return [
        finding(ids[0][0], ids[0][1], mean_score, "share", family=FAMILY,
                sample_size=len(target_rows), min_sample=min_per_author,
                distribution=common, evidence=evidence, warning=_IMPOSTORS_CAVEAT),
        finding(ids[1][0], ids[1][1], variance, "share^2", family=FAMILY,
                sample_size=len(scores), min_sample=2, distribution=common,
                warning=_IMPOSTORS_CAVEAT),
    ]


# --------------------------------------------- L. sentence-embedding representation

def _embedding_style_findings(analysis: DocumentAnalysis, config) -> list[dict[str, Any]]:
    """Sentence-embedding based section-to-section style drift within this document.

    See the module docstring's "Sentence-embedding style representation"
    section for what this does and does not compare against, and "The
    critical gating rule" for why loading the model is strictly conditioned
    on ``features.embedding_style`` (this function's only caller) being
    explicitly on: this is the only code path anywhere in this suite that
    can trigger a ``sentence_transformers`` import.
    """

    model_name = str(option(config, "embedding_model", DEFAULT_EMBEDDING_MODEL))
    primary = str(option(config, "embedding_primary_distance", DEFAULT_EMBEDDING_PRIMARY_DISTANCE))
    if primary not in EMBEDDING_DISTANCE_METRICS:
        primary = DEFAULT_EMBEDDING_PRIMARY_DISTANCE
    window_words = int(option(config, "section_window_words", DEFAULT_SECTION_WINDOW_WORDS))

    family_ids = [(f"{PREFIX}embedding_drift_open_close_{name}",
                   f"Sentence-embedding distance, opening section to closing section ({name})")
                 for name in EMBEDDING_DISTANCE_METRICS]
    other_ids = (
        (f"{PREFIX}embedding_section_stability",
         "Median section-to-section sentence-embedding distance"),
        (f"{PREFIX}embedding_max_section_shift",
         "Largest single section-to-section sentence-embedding distance"),
        (f"{PREFIX}embedding_dispersion",
         "Mean distance of each section's embedding from the document's own mean embedding"),
    )
    all_ids = family_ids + list(other_ids)

    sections, method = get_sections(analysis, window_words)
    if len(sections) < MIN_SAMPLE_SECTIONS:
        warning = (f"only {len(sections)} section(s) available (method={method}); need at "
                  f"least {MIN_SAMPLE_SECTIONS} to compare a book with itself" if method else
                  "text has no measurable sections")
        return [finding(mid, name, None, primary, family=FAMILY, sample_size=len(sections),
                        min_sample=MIN_SAMPLE_SECTIONS, warning=warning) for mid, name in all_ids]

    vectors, reason = _embed_texts([view.text for _, view in sections], model_name)
    if vectors is None:
        return [finding(mid, name, None, primary, family=FAMILY, sample_size=len(sections),
                        min_sample=MIN_SAMPLE_SECTIONS, warning=reason) for mid, name in all_ids]

    n = len(vectors)
    section_labels = [title or f"section {i + 1}" for i, (title, _) in enumerate(sections)]
    out: list[dict[str, Any]] = []
    for name in EMBEDDING_DISTANCE_METRICS:
        out.append(finding(
            f"{PREFIX}embedding_drift_open_close_{name}",
            f"Sentence-embedding distance, opening section to closing section ({name})",
            _dense_distance(name, vectors[0], vectors[-1]), name, family=FAMILY, sample_size=n,
            min_sample=MIN_SAMPLE_SECTIONS,
            distribution={"method": method, "model": model_name, "distance_family": name,
                         "opening_section": section_labels[0], "closing_section": section_labels[-1]}))

    pair_distances = [_dense_distance(primary, vectors[i], vectors[i + 1]) for i in range(n - 1)]
    clean = [d for d in pair_distances if d is not None]
    median_distance = statistics.median(clean) if clean else None
    max_distance = max(clean) if clean else None
    mean_vector = [statistics.fmean(dim) for dim in zip(*vectors)]
    dispersion_values = [d for d in (_dense_distance(primary, vec, mean_vector) for vec in vectors)
                         if d is not None]
    dispersion = statistics.fmean(dispersion_values) if dispersion_values else None

    pair_evidence = [{"from": section_labels[i], "to": section_labels[i + 1],
                      f"{primary}_distance": pair_distances[i]} for i in range(n - 1)][:25]
    common = {"method": method, "model": model_name, "distance_family": primary}

    out.append(finding(f"{PREFIX}embedding_section_stability",
                f"Median section-to-section sentence-embedding distance (method={method}, "
                f"distance={primary}, model={model_name})",
                median_distance, primary, family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS, evidence=pair_evidence, distribution=common))
    out.append(finding(f"{PREFIX}embedding_max_section_shift",
                f"Largest single section-to-section sentence-embedding distance (method={method}, "
                f"distance={primary}, model={model_name})",
                max_distance, primary, family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS, evidence=pair_evidence, distribution=common))
    out.append(finding(f"{PREFIX}embedding_dispersion",
                f"Mean distance of each section's embedding from the document's own mean "
                f"embedding (method={method}, distance={primary}, model={model_name})",
                dispersion, primary, family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE_SECTIONS, distribution=common))
    return out


# --------------------------------------------------- M. lexicalrichness cross-check

def _lexicalrichness_crosscheck_findings(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    """``lexicalrichness``'s own diversity measures, alongside this module's
    hand-rolled equivalents (group B, above) -- reported side by side rather
    than reconciled, per the project's established cross-check philosophy
    (see :mod:`lexical_mtld`'s module docstring for the precedent). Each
    finding's ``distribution.own_metric_id`` names the metric id this
    module's own computation of the "same" quantity lives under, so a reader
    can pull both numbers up together.
    """

    ids = (
        (f"{PREFIX}lexicalrichness_yules_k", "Yule's K (lexicalrichness)", "K",
         f"{PREFIX}yules_k", "yulek"),
        (f"{PREFIX}lexicalrichness_yules_i", "Yule's I (lexicalrichness)", "I",
         f"{PREFIX}yules_i", "yulei"),
        (f"{PREFIX}lexicalrichness_herdan_c", "Herdan's C (lexicalrichness)", "C",
         f"{PREFIX}herdan_c", "Herdan"),
        (f"{PREFIX}lexicalrichness_dugast_uber", "Dugast's Uber index (lexicalrichness)", "U",
         f"{PREFIX}uber_index", "Dugast"),
        (f"{PREFIX}lexicalrichness_maas_index", "Maas's a^2 (lexicalrichness)", "a^2",
         f"{PREFIX}maas_index", "Maas"),
        (f"{PREFIX}lexicalrichness_rttr", "Root TTR / Guiraud's R (lexicalrichness)", "R",
         f"{PREFIX}guiraud_r", "rttr"),
    )
    n = analysis.word_count

    def _degrade(warning: str, size: int) -> list[dict[str, Any]]:
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=size,
                        min_sample=MIN_SAMPLE_LEXICAL, sample_size_sensitive=True,
                        distribution={"own_metric_id": own_id}, warning=warning)
               for mid, name, unit, own_id, _ in ids]

    module, reason = require("lexicalrichness")
    if module is None:
        return _degrade(reason, n)
    if n == 0:
        return _degrade("no words in text", 0)

    try:
        lr = module.LexicalRichness(analysis.tokens, preprocessor=None, tokenizer=None)
        # A near-degenerate sample (every token distinct, or every token
        # identical) makes lexicalrichness divide by zero internally and warn
        # rather than raise; the non-finite result is caught below instead.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            values = {attribute: getattr(lr, attribute) for _, _, _, _, attribute in ids}
    except Exception as exc:  # pragma: no cover - defensive; lexicalrichness edge cases
        return _degrade(f"lexicalrichness raised {type(exc).__name__}: {exc}", n)

    out = []
    for mid, name, unit, own_id, attribute in ids:
        value = values.get(attribute)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            # lexicalrichness divides by (terms**2 - total) or similar on a
            # near-degenerate sample (e.g. every token distinct, or every
            # token identical) and returns +/-inf or nan rather than raising;
            # this module's own equivalent measure (group B) guards the same
            # edge case explicitly and reports None with a warning, so the
            # cross-check does too rather than writing a non-finite number
            # into a JSON report.
            value = None
        out.append(finding(mid, name, value, unit, family=FAMILY, sample_size=n,
                           min_sample=MIN_SAMPLE_LEXICAL, sample_size_sensitive=True,
                           distribution={"own_metric_id": own_id},
                           warning=None if value is not None else
                           f"lexicalrichness.{attribute} was undefined (non-finite) on this sample"))
    return out


# ---------------------------------------------------------------------- measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    max_reported = max(1, min(40, int(option(config, "max_reported", DEFAULT_MAX_REPORTED))))
    out: list[dict[str, Any]] = []

    if _feature(config, "character_ngrams", True):
        out.extend(_character_ngram_findings(analysis, config, max_reported))
    if _feature(config, "byte_ngrams", True):
        out.extend(_byte_ngram_findings(analysis, config, max_reported))
    if _feature(config, "word_ngrams", True):
        out.extend(_word_ngram_findings(analysis, config, max_reported))
    if _feature(config, "function_word_ngrams", True):
        out.append(_function_word_ngram_finding(analysis, max_reported))
    if _feature(config, "punctuation_shape", True):
        out.append(_punctuation_ngram_finding(analysis, config, max_reported))
    if _feature(config, "word_shape", True):
        out.append(_word_shape_finding(analysis, max_reported))
    if _feature(config, "affixes", True):
        out.extend(_affix_findings(analysis, config, max_reported))
    if _feature(config, "sentence_openings", True):
        out.append(_sentence_opening_finding(analysis, max_reported))
    if _feature(config, "contractions_capitalization", True):
        out.extend(_contraction_and_capitalization_findings(analysis, max_reported))
    if _feature(config, "lexical_richness", True):
        out.extend(_lexical_richness_findings(analysis))
    if _feature(config, "vocabulary_growth", True):
        out.extend(_heaps_and_zipf_findings(analysis, max_reported))
    if _feature(config, "section_stability", True):
        out.extend(_section_stability_findings(analysis, config, max_reported))
    if _feature(config, "compression", True):
        out.extend(_compression_findings(analysis, config))
    if _feature(config, "corpus_language_model", True):
        out.extend(_corpus_language_model_findings(analysis, profile))
    if _feature(config, "corpus_reference", True):
        out.extend(_corpus_reference_findings(analysis, config, profile, max_reported))
    if _feature(config, "pos_dependency", False):
        out.extend(_pos_dependency_findings(analysis, config, max_reported))
    if _feature(config, "word_frequency_distance", True):
        out.extend(_word_frequency_distance_findings(analysis, config, profile))
    if _feature(config, "author_language_model", True):
        out.extend(_author_language_model_findings(analysis, profile, max_reported))
    if _feature(config, "lexicalrichness_crosscheck", True):
        out.extend(_lexicalrichness_crosscheck_findings(analysis))
    if _feature(config, "impostors", False):
        out.extend(_impostors_findings(analysis, config, profile))
    if _feature(config, "embedding_style", False):
        out.extend(_embedding_style_findings(analysis, config))

    return out
