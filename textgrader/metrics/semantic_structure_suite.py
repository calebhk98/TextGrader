"""Multiple, independent non-generative semantic-structure and topic channels.

Every semantic-repetition metric already in this codebase
(:mod:`semantic_adjacent`, :mod:`semantic_window`, :mod:`semantic_paragraph`,
:mod:`semantic_clusters`, the ``semantic`` group of :mod:`coherence_suite`)
answers "how similar are these two units" with exactly one instrument:
sentence-transformer cosine similarity, falling back to a document-local
TF-IDF cosine when that model is not installed. That is one representation.
Different representations encode different assumptions -- BM25/TF-IDF are
lexical, LSA is a low-rank co-occurrence projection, LDA/NMF/HDP expose topic
mixtures, static word vectors (GloVe/fastText/spaCy) are word-level
distributional similarity, sentence-transformers are contextual. Text that
looks coherent to one representation can look discontinuous to another, and
this suite's whole point is to keep that disagreement rather than average it
away (see "Representation disagreement" below).

Every finding here is ``Polarity.NEUTRAL`` (the tool's default for optional
findings). A tightly focused short story and a sprawling multi-thread novel
both have legitimate semantic structure; a high or low value is a fact about
organization, not a verdict on quality.

**Representations** (``features`` in this metric's config; every one is
independently switchable, mirrored in ``config.json``'s ``_requires_*``
notes):

``lexical_tfidf`` (on by default, dependency-free)
    Document-local TF-IDF content-word cosine, built with
    :func:`semantic_adjacent.lexical_vectors` -- the exact function that
    family's own fallback already uses. Reused, not reimplemented.
``bm25`` (on by default, needs ``rank_bm25``)
    Okapi BM25 relevance of unit *i* as a query against unit *j* as a
    document, over this document's own sentence/paragraph corpus.  BM25
    scores are **unbounded**, not a cosine similarity in [-1, 1]; every
    finding under this representation says so and no code anywhere compares
    a BM25 score's magnitude against a cosine one.
``lsa`` (on by default, needs ``sklearn``)
    A within-document TruncatedSVD over this document's own TF-IDF matrix: a
    linear-algebra re-projection of THIS text, not a corpus-trained topic
    model.  The acceptance criteria list "LSA" and "topic-model" as separate
    required channels; this is the former.  See ``topic_models`` below for
    the latter, which genuinely is corpus-trained.
``static_embedding_glove`` (off by default, needs ``gensim`` + a downloaded
    vector set)
    Mean-pooled (aggregation=mean; TF-IDF-weighted mean and SIF are
    documented alternatives, not implemented -- see "Deferred" below) GloVe
    vectors loaded once per process via gensim's downloader API
    (``glove-wiki-gigaword-50`` by default: 66 MB, verified downloading and
    scoring correctly in this environment -- see the module's git history /
    task report for the measured load time).  Off by default because it is a
    real network download the first time it runs; cached under
    ``~/gensim-data`` afterward.
``static_embedding_spacy`` (off by default, needs a vector-bearing spaCy
    model, e.g. ``en_core_web_md``)
    A second, independent static-vector channel: spaCy's own ``doc.vector``
    (itself a mean pool over the sentence's/paragraph's tokens' static
    vectors), loaded through an independent, lightweight pipeline
    (``exclude=["parser","ner","tagger","lemmatizer","attribute_ruler",
    "tok2vec"]``) so this never touches the shared parse the ``"parse"``-cost
    metrics use, and never forces ``needs_parse`` for this whole suite.
``sentence_transformer`` (off by default even though it is the suite's most
    "real" contextual channel, needs ``sentence_transformers``)
    Reuses :func:`semantic_adjacent.get_sentence_vectors` /
    ``get_paragraph_vectors`` -- the SAME cached embedding array
    :mod:`semantic_adjacent`, :mod:`semantic_window`, :mod:`semantic_paragraph`,
    :mod:`semantic_clusters` and :mod:`coherence_suite` already computed for
    this document, at this model.  This suite never re-encodes with the same
    model. Off by default per the gating rule below: putting
    ``sentence_transformers`` in ``REQUIRES`` would flip ``needs_model`` for
    the WHOLE suite and silently drop even the cheap, default-on lexical/
    BM25/LSA/topic-model channels out of every corpus profiling pass unless
    ``--model-metrics`` were also passed (see
    :func:`textgrader.corpus._metric_names`). Keeping it out of ``REQUIRES``
    and behind its own ``features`` flag (default ``False``) means the
    default config loads nothing heavy, and a corpus profile can still be
    built for the corpus-trained topic channel without ever touching a
    transformer model.
``sentence_transformer`` intro/conclusion pair uses a direct, tiny
    :func:`semantic_adjacent.embed_texts` call (two new strings, not already
    cached) rather than re-running the whole-document encode.

For each active representation, six metrics are reported (unit = "cosine" for
every cosine-scale representation, "bm25_score" for BM25):

* ``semantic.structure_<rep>_adjacent_sentence`` / ``_adjacent_paragraph`` --
  the spec's "adjacent-sentence"/"adjacent-paragraph similarity/distance",
  reported as a full distribution with the ten lowest-similarity transitions
  as evidence ("lowest-similarity transitions" from the spec is folded into
  this finding's evidence rather than becoming a 37th metric id, the same
  pattern :mod:`semantic_adjacent` itself already uses). The headline is
  the mean.  Real prose makes lexical_tfidf/bm25 adjacent-sentence scores
  zero-inflated (most sentence pairs share no content word), so a median
  sits at exactly 0.0 and never moves; see :func:`_resolve_headline`.  The
  median, the full shape and ``zero_share_percent`` stay in
  ``distribution``.
* ``semantic.structure_<rep>_centroid_relatedness`` -- "similarity to
  document centroid".  Rather than a true vector centroid (which does not
  exist for BM25's asymmetric query/document scoring), every representation
  uses the SAME operational definition: each sentence's mean similarity to a
  capped (``centroid_reference_cap``, default 200), seeded random reference
  sample of the document's own sentences.  This is deliberately O(n *
  cap), not O(n^2) (see "Bound O(n^2) work" below), and is directly
  comparable across cosine-scale representations even though it is a proxy,
  not the literal mean-vector cosine, for any of them. Headline: the mean,
  as above.
* ``semantic.structure_<rep>_window_drift`` -- "local-window semantic
  drift": adjacent-similarity values are pooled into non-overlapping blocks
  of ``structure_window`` sentences (default 5) and the headline is the
  mean absolute change between consecutive
  block means.
* ``semantic.structure_<rep>_global_dispersion`` -- "global semantic
  dispersion": the standard deviation (never a mean/median fallback --
  dispersion is itself a spread statistic, and the finding's
  ``distribution`` says so explicitly) of a capped (``dispersion_pair_cap``,
  default 300), seeded random sample of NON-adjacent sentence-pair
  similarities.
* ``semantic.structure_<rep>_intro_conclusion_similarity`` -- "conclusion-to-
  introduction similarity" and "opening-to-closing semantic distance" are the
  same measurement asked two ways (distance = 1 - similarity for a
  cosine-scale representation); this suite reports the one finding, as the
  cosine (or, for bm25, the query-vs-synthetic-document score) between the
  AVERAGE of the first/last ``intro_conclusion_sentences`` sentences'
  (default 3) vectors/statistics -- taken from the SAME sentence-level fit
  every other metric above already uses (``group_sim``, see the
  representation-registry comment below), never a fresh, separate fit on
  just the two snippets. A fresh 2-document fit is exactly what made this
  metric degenerate in an earlier version of this suite: TruncatedSVD on two
  documents has at most one meaningful dimension (forcing cosine to exactly
  1.0 regardless of content) and Okapi BM25's IDF goes negative for a term
  in more than half of a 2-document corpus -- confirmed on three real books
  (Alice, The Secret Agent, Flatland), where ``lsa_intro_conclusion_
  similarity`` was exactly 1.0 and ``bm25_intro_conclusion_similarity`` was
  negative on every one of them under that design.

**Corpus-trained topic models -- the hard design problem.**  LDA/NMF/HDP must
be trained on the reference corpus, never on the graded text, and versioned
(the task spec, and this project's rule 15: leave-one-out corpus validation).
At grading time TextGrader has only the corpus *profile* JSON, never the raw
books, so the model has to be fit from whatever :func:`profile_vector` (this
suite's ``textgrader.corpus.build_profile`` hook) cached, once per book, when
the corpus was built.

The design used here:

1. A **fixed, versioned vocabulary** (``topic_vocab_size`` words, default
   3000; capped and ordered by ``wordfreq.top_n_list('en', ...)``, filtered to
   alphabetic content words of length >= 3 that are not on this suite's
   stopword list, falling back to a small built-in word list if ``wordfreq``
   is unavailable). It is FIXED across every book independently -- each
   book's row is built with no access to any other book's text -- which is
   what makes each book's cached vector comparable to every other's without
   ever building a corpus-wide vocabulary at profiling time (that would need
   two passes over the corpus, which the per-book ``profile_vector`` hook
   cannot do).
2. A **bounded, deterministic sample**: :func:`profile_vector` draws a
   seeded random sample of ``topic_profile_paragraph_sample`` paragraphs
   (default 30, capped to however many the book has), scans at most
   ``topic_profile_paragraph_word_cap`` words per sampled paragraph (default
   300), and counts occurrences of fixed-vocabulary words. Only the top
   ``topic_profile_max_terms`` terms by count (default 250) are kept, so one
   book's cached row is bounded regardless of vocabulary size or book length.
3. The row is stored under ``feature_profiles["semantic_structure_suite"]``
   (:mod:`textgrader.corpus`'s per-book vector mechanism -- see
   ``build_profile``'s own docstring on ``profile_vector``), one flat
   ``dict[str, float]`` per book, aligned index-for-index with
   ``profile["books"]``. Three reserved ``__``-prefixed keys
   (``__seed__``, ``__sample_paragraphs__``, ``__vocab_size__``) record the
   exact sampling that produced the row, so a reader can tell a row built
   under an older version of this module's constants from a current one;
   real vocabulary words never start with ``__`` so there is no collision.
4. At **grading time**, :func:`_fit_topic_models` reads every book's cached
   row, projects it back onto the CURRENT fixed vocabulary (a term whose
   position in an old vocabulary ordering has since changed is looked up by
   the word itself, never by position), excludes any row whose book's
   ``source_filename``/``source_path`` matches the graded document's own
   ``source`` (leave-one-out -- see the module's git history for the
   "excluded/used" counts this test verified against a real split-corpus
   profile), and fits scikit-learn's ``LatentDirichletAllocation`` and
   ``NMF`` plus tomotopy's ``HDPModel`` (MALLET is Java and out of scope
   entirely -- see the task instructions; tomotopy is the suggested
   Python-native HDP/LDA-family alternative) once per process, cached by
   ``(id(profile), vocab_size, n_topics, seed, source, features)``. The
   graded document's own paragraphs are then vectorized against the SAME
   fixed vocabulary and only ``.transform()``/``.infer()`` is called on them
   -- the corpus fit itself never sees the graded text.
5. Every ``semantic.structure_topic_*`` finding's ``distribution`` records
   ``vocab_size``, ``vocab_version``, the sampling seed, how many corpus rows
   were available/used/excluded (leave-one-out), ``n_topics`` actually used
   (bounded by both the config and the corpus size), and the fitting
   library's installed version (``sklearn.__version__`` / gensim's or
   tomotopy's), satisfying "record vocabulary size, sample size, seed and
   library version in each finding's distribution".

**Measured profile growth.** See this suite's entry in the task report for
the exact measured bytes added to a real, multi-book profile built with
``features.topic_models`` on; the design bounds it structurally regardless
of the corpus size, since it is
``books * min(topic_profile_max_terms, vocab_size) * (~1 average bytes per
JSON number/word entry)`` and does not grow with book length, sentence count,
or vocabulary size beyond the ``topic_profile_max_terms`` cap.

Building a profile that actually populates this cache needs
``semantic_structure_suite`` enabled (``features.topic_models`` -- on by
default -- left on) when ``textgrader.corpus`` builds it. This suite's
``cost`` is ``"moderate"``, not ``"parse"`` or ``"model"`` (none of its
default-on channels need the shared spaCy parse or a sentence-transformer
model), so **no** ``--parse-metrics``/``--model-metrics`` flag is required
for the topic-model profile rows to be built -- unlike
``coherence_suite``'s transition-frequency profile or
``syntax_complexity_suite``'s surprisal profile, both of which need
``--parse-metrics`` because their own suite's cost genuinely is ``"parse"``.

**Topic-transition metrics measured separately from semantic-transition
metrics** (an explicit acceptance criterion): the six per-representation
adjacent/centroid/drift/dispersion/intro-conclusion metrics above never read
the corpus profile and never involve a topic id; the seven
``semantic.structure_topic_<model>_*`` metrics below never involve a
cosine/BM25 similarity. They are two disjoint families sharing only this
module and its ``FAMILY``/prefix.

**A within-document topic model is a SEPARATE metric id family and already
exists.** :mod:`textgrader.sequences`' ``window_topic_id`` sequence (used by
:mod:`timeseries_suite`'s ``topic_transition_rate``/``topic_transition_entropy``/
``topic_dwell_time``) already fits a fresh NMF/LDA model over THIS document's
own fixed-size windows every time it runs -- exactly the "very long books"
within-document exception the task spec allows as "a separate, clearly named
metric ID". This suite does not duplicate it; every
``semantic.structure_topic_*`` finding's ``distribution`` carries an
``overlaps_existing_metric_id`` note pointing at it, because the two
disagreeing (a corpus-trained topic mixture vs. a document's own fresh fit)
is expected and is itself evidence, not a bug to reconcile.

**Representation disagreement is data, not noise** (rule 18 and this suite's
own acceptance criteria). Three findings compare the ACTIVE representations'
adjacent-sentence transition sequences directly:

* ``semantic.structure_disagreement_rank_correlation`` -- the median pairwise
  Spearman rank correlation between every pair of active representations'
  adjacent-transition scores (the full pairwise matrix is in
  ``distribution``). Rank correlation, not a raw-value comparison, is what
  lets a bounded cosine score and an unbounded BM25 score be compared at all.
* ``semantic.structure_disagreement_single_representation_only`` -- the share
  of transitions flagged "low-coherence" by exactly one active
  representation.
* ``semantic.structure_disagreement_consensus_low_coherence`` -- the share
  flagged by a majority of active representations, with the actual sentence
  pairs as evidence.

  Both flag the lowest ``disagreement_low_tail_quantile`` (default 10%) of
  each representation's OWN adjacent-transition scores BY RANK (see
  :func:`_low_tail_flags_by_rank`), never a shared absolute threshold (which
  BM25's unbounded scale would make meaningless) and never an absolute
  quantile THRESHOLD either: lexical_tfidf/bm25 scores on real prose are
  zero-inflated enough that their own 10th-percentile threshold is 0.0
  itself, so "flag every value <= 0.0" flagged 98.3-99.7% of transitions as
  "consensus" on three real books in an earlier version of this suite --
  nowhere near the configured 10%. Picking a fixed COUNT of the lowest
  values by rank (deterministic, per-representation-seeded tie-break --
  see :func:`_rep_tie_seed`'s docstring for why plain index order would
  itself manufacture spurious cross-representation agreement out of a
  shared tied block) flags close to the configured share regardless of
  zero-inflation. Each finding's ``distribution["per_representation"]``
  reports exactly how many transitions each active representation flagged
  and whether the flag boundary landed inside a tie too large to resolve
  (``degenerate_tie``), so a reader is never left assuming the boundary was
  informative when it was not.

**Overlap with existing metrics.** Where a channel here measures
substantially the same thing as an existing metric id, the overlap is named
directly in that finding's ``distribution["overlaps_existing_metric_id"]``
rather than left for a reader to notice on their own:

* ``lexical_tfidf``'s adjacent-sentence/paragraph findings overlap
  :mod:`semantic_adjacent`'s and :mod:`coherence_suite`'s lexical FALLBACK
  backend (same TF-IDF cosine over the same pairs).
* ``lexical_tfidf``'s centroid-relatedness finding overlaps
  ``discourse.coherence_global_context_overlap``.
* ``sentence_transformer``'s adjacent-sentence/paragraph findings overlap
  ``semantic.adjacent_sentence_similarity``/``semantic.paragraph_similarity``
  and ``discourse.coherence_local_semantic_cohesion``/
  ``discourse.coherence_paragraph_semantic_transition`` when the embedding
  backend is actually available (never when it fell back to lexical -- see
  the ``sentence_transformer`` backend builder, which reports itself
  unavailable rather than silently duplicate the ``lexical_tfidf``
  representation under a different name).
* Every ``semantic.structure_topic_*`` finding overlaps
  ``window_topic_id``/``topic_transition_rate``/``topic_transition_entropy``/
  ``topic_dwell_time`` (see above).

**Bounding O(n^2) work** (rule 12): adjacent-pair work is O(n). Centroid
relatedness is O(n * centroid_reference_cap). Global dispersion samples a
capped, seeded set of pairs rather than all C(n,2). The corpus topic-model
fit operates on at most ``corpus books`` rows of ``vocab_size`` (bounded)
features, never on raw text at grading time. BM25's ``get_scores`` call is
O(n) per query and is only ever called from the bounded contexts above (never
inside a full n^2 loop), and its per-query result is cached
(``scores_cache``) so a query index already scored is never re-scored.

**Deferred, with evidence:**

* **Paragraph-to-title relevance.** :class:`textgrader.document.DocumentAnalysis`
  carries no document-level title field -- only Markdown/plain-text SECTION
  headings via ``analysis.sections`` (see that property's docstring), which
  describe chapter/section boundaries, not a document title. Implementing
  this metric would mean inventing a definition of "title" this project's
  own document model does not have, so it is left out rather than guessed at,
  consistent with rule 4 (reuse ``DocumentAnalysis``; no metric independently
  decides what counts as something the shared pipeline does not already
  define).
* **TF-IDF-weighted mean and SIF aggregation for static embeddings.** Only
  plain mean pooling (content words, stopwords excluded) is implemented for
  ``static_embedding_glove``/``static_embedding_spacy``, named explicitly in
  each finding's ``warning``. The spec asks these to be "kept separate where
  implemented"; only one is implemented here, so there is nothing to keep
  separate from yet -- a second aggregation would be its own
  ``static_embedding_*_tfidf_weighted`` representation, not a silent change
  to this one's definition.
* **fastText.** Gensim's downloader API does not host a fastText WORD-vector
  set smaller than ``fasttext-wiki-news-subwords-300`` (about 1 GB); loading
  fastText's genuinely distinguishing subword-OOV behaviour needs
  ``gensim.models.fasttext.load_facebook_vectors`` against Facebook's own
  ``.bin`` release, a multi-gigabyte download this shared, 15 GB, four-CPU
  container should not be asked to hold alongside four sibling agents' own
  work. ``static_embedding_glove`` already exercises the identical
  "gensim-hosted static vector set, mean-pooled" code path end to end
  (verified: downloading and scoring ``glove-wiki-gigaword-50`` for real in
  this environment -- see the task report), so a fastText representation
  would add a large download without adding a new code path; it is left as a
  documented gap rather than implemented against a token-sized stand-in
  vector set that would not actually exercise fastText's subword behaviour.
* **MALLET.** Explicitly out of scope per the task instructions (Java, not
  Python). tomotopy is used for HDP instead, as suggested.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
import zlib
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from ..document import DocumentAnalysis
from ..optional import on_reset, require
from ..stats import quantile, summarize
from .common import MODERATE, finding, option, rate
from . import semantic_adjacent as sem

FAMILY = "semantic"
# None of this suite's default-on channels (lexical_tfidf, bm25, lsa,
# topic_models, disagreement) need the shared spaCy parse or a
# sentence-transformer model, so this stays "moderate": no --parse-metrics or
# --model-metrics flag is needed to populate the topic-model profile cache
# (see the module docstring's "hard design problem" section). The two
# feature-flagged static-vector channels and the sentence_transformer channel
# are heavier but are off by default and gated entirely through `features`,
# never through REQUIRES/COST (see the gating-rule note below).
COST = MODERATE
# Documentation only -- deliberately does NOT include "sentence_transformers"
# (see the module docstring's gating-rule note: that would flip needs_model
# for the WHOLE suite and drop even the cheap default-on channels out of
# every corpus profiling pass) and does NOT include "spacy" (that would flip
# needs_parse for the whole suite even though the spaCy-vector channel below
# runs its own independent, non-shared pipeline and never touches the shared
# parse).
REQUIRES: tuple[str, ...] = ("sklearn", "gensim", "rank_bm25", "tomotopy", "wordfreq")
MIN_SAMPLE = 6
UNIT_SENSITIVE = False

MIN_SAMPLE_PAIRS = 6
MIN_SAMPLE_PARAGRAPH_PAIRS = 3
MIN_SAMPLE_DISAGREEMENT = 6
MIN_TOPIC_PARAGRAPHS = 6
MIN_TOPIC_CORPUS_BOOKS = 4
DEFAULT_VOCAB_SIZE = 3000
DEFAULT_GLOVE_MODEL = "glove-wiki-gigaword-50"
DEFAULT_SPACY_VECTOR_MODEL = "en_core_web_md"
TOPIC_PROFILE_SCHEMA_VERSION = 1

DEFAULT_FEATURES: dict[str, bool] = {
    "lexical_tfidf": True,
    "bm25": True,
    "lsa": True,
    "static_embedding_glove": False,
    "static_embedding_spacy": False,
    "sentence_transformer": False,
    "topic_models": True,
    "disagreement": True,
}

REP_LABELS: dict[str, str] = {
    "lexical_tfidf": "TF-IDF lexical",
    "bm25": "BM25",
    "lsa": "within-document LSA/SVD",
    "static_embedding_glove": "GloVe static-embedding",
    "static_embedding_spacy": "spaCy static-vector",
    "sentence_transformer": "sentence-transformer",
}

_OVERLAP_NOTES: dict[tuple[str, str], str] = {
    ("lexical_tfidf", "adjacent_sentence"):
        "overlaps semantic.adjacent_sentence_similarity's lexical fallback and "
        "discourse.coherence_local_semantic_cohesion's lexical fallback (semantic_adjacent.py / "
        "coherence_suite.py): the same TF-IDF content-word cosine over the same adjacent "
        "sentence pairs.",
    ("lexical_tfidf", "adjacent_paragraph"):
        "overlaps semantic.paragraph_similarity's lexical fallback and "
        "discourse.coherence_paragraph_semantic_transition's lexical fallback: the same TF-IDF "
        "content-word cosine over the same adjacent paragraph pairs.",
    ("lexical_tfidf", "centroid_relatedness"):
        "overlaps discourse.coherence_global_context_overlap: both are a sentence's TF-IDF "
        "cosine similarity to a document-level lexical reference (this suite's reference is a "
        "capped random sample of sentences; coherence_suite's is the full-document mean vector).",
    ("sentence_transformer", "adjacent_sentence"):
        "overlaps semantic.adjacent_sentence_similarity and "
        "discourse.coherence_local_semantic_cohesion WHEN the sentence-transformers embedding "
        "backend is actually available: identical cosine similarity of the same cached sentence "
        "embeddings (semantic_adjacent.get_sentence_vectors).",
    ("sentence_transformer", "adjacent_paragraph"):
        "overlaps semantic.paragraph_similarity and "
        "discourse.coherence_paragraph_semantic_transition WHEN the sentence-transformers "
        "embedding backend is actually available: identical cosine similarity of the same "
        "cached paragraph embeddings (semantic_adjacent.get_paragraph_vectors).",
}

TOPIC_OVERLAP_NOTE = (
    "overlaps window_topic_id / topic_transition_rate|topic_transition_entropy|topic_dwell_time "
    "(textgrader.sequences / timeseries_suite): both measure NMF/LDA topic-switch and "
    "persistence statistics, but THIS suite's topic model is corpus-trained with leave-one-out "
    "exclusion of the graded book, while window_topic_id fits a fresh NMF/LDA model on THIS "
    "document's own fixed-size windows every time it runs. The two disagreeing is expected -- a "
    "corpus-referenced topic mixture and a document's own from-scratch fit answering a related "
    "but different question -- and is reported as data, not reconciled."
)

# ----------------------------------------------------------------- small helpers

_STOPWORDS = sem.STOPWORDS


def _tokens(text: str) -> list[str]:
    from ..text import words
    return [word.lower().replace("’", "'") for word in words(text)]


def _content_tokens(text: str) -> list[str]:
    return [word for word in _tokens(text) if len(word) > 2 and word not in _STOPWORDS]


def _sparse_dot(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(word, 0.0) for word, value in a.items())


def _sparse_group_cosine(vectors: Sequence[Mapping[str, float]], indices_a: Sequence[int],
                         indices_b: Sequence[int]) -> float | None:
    """Cosine similarity between the (re-normalized) SUM of the sparse
    vectors at ``indices_a`` and at ``indices_b`` -- the sparse-vector
    equivalent of averaging dense embeddings, used by group_sim so the
    intro/conclusion metric scores a group of sentences within the SAME
    whole-document TF-IDF space every other lexical_tfidf metric already
    uses, rather than refitting IDF on just the two groups (which is what
    made this metric degenerate -- see the module docstring)."""

    def merged(indices: Sequence[int]) -> dict[str, float]:
        total: dict[str, float] = {}
        for index in indices:
            for word, value in vectors[index].items():
                total[word] = total.get(word, 0.0) + value
        norm = math.sqrt(sum(value * value for value in total.values()))
        return {word: value / norm for word, value in total.items()} if norm else {}

    a, b = merged(indices_a), merged(indices_b)
    return _sparse_dot(a, b) if a and b else None


def _features(config: Mapping[str, Any] | None) -> dict[str, bool]:
    merged = dict(DEFAULT_FEATURES)
    merged.update(option(config, "features", {}) or {})
    return merged


def _reference_sample(n: int, cap: int, seed: int) -> list[int]:
    if n <= cap:
        return list(range(n))
    return sorted(random.Random(seed).sample(range(n), cap))


# The maximum number of (i, j) pairs enumerated outright before switching to
# rejection sampling in :func:`_sample_non_adjacent_pairs`. Below this, simply
# building and sampling the candidate list is cheap; above it (a 9,000-sentence
# novel has about 40 million i<j pairs), materializing every pair first would
# itself be an O(n^2) pass this function exists to avoid.
_PAIR_ENUMERATION_LIMIT = 200_000


def _sample_non_adjacent_pairs(n: int, cap: int, seed: int, min_gap: int = 2
                               ) -> list[tuple[int, int]]:
    """Up to ``cap`` distinct ``(i, j)`` pairs with ``j - i >= min_gap``,
    without ever materializing the full O(n^2) candidate set for a
    long document -- see "Bound O(n^2) work" in the module docstring."""

    max_pairs = n * (n - min_gap) // 2 if n > min_gap else 0
    if max_pairs <= 0:
        return []
    if max_pairs <= _PAIR_ENUMERATION_LIMIT:
        candidates = [(i, j) for i in range(n) for j in range(i + min_gap, n)]
        return candidates if len(candidates) <= cap else random.Random(seed).sample(candidates, cap)
    rng = random.Random(seed)
    seen: set[tuple[int, int]] = set()
    attempts, max_attempts = 0, cap * 20
    while len(seen) < cap and attempts < max_attempts:
        i = rng.randrange(0, n - min_gap)
        j = rng.randrange(i + min_gap, n)
        seen.add((i, j))
        attempts += 1
    return sorted(seen)


# ----------------------------------------------------------- representation registry
#
# Every builder has the signature ``build(units, analysis, kind, config) ->
# (ok, sim_fn, group_sim, scale, note)``.  ``kind`` is "sentence" or
# "paragraph"; only the sentence_transformer builder reads it, to reuse
# semantic_adjacent's cache for either one (see the module docstring).
# ``sim_fn(i, j)`` returns a float or ``None`` (a comparable pair could not be
# formed, e.g. both units were out-of-vocabulary for a static embedding).
# ``group_sim(indices_a, indices_b)`` scores the AVERAGE of the vectors/stats
# at ``indices_a`` against the average at ``indices_b``, all within the SAME
# fit this builder call already produced for ``units`` -- this is what the
# ``intro_conclusion_similarity`` metric uses (opening-k vs. closing-k
# sentence indices), rather than a fresh, separate fit on just those two
# snippets. A fresh 2-document fit is exactly the bug this design avoids: a
# TruncatedSVD or BM25 index fit on only two documents degenerates (at most
# one meaningful SVD dimension forces cosine to +-1 outright; BM25's IDF goes
# negative for any term in more than half of a 2-document corpus) --
# confirmed on three real books, where ``lsa_intro_conclusion_similarity``
# was exactly 1.0 and ``bm25_intro_conclusion_similarity`` was negative on
# every one of them under the old design. Scoring group averages within the
# SAME whole-document fit instead means every representation answers this
# metric with the real corpus/document statistics it already computed for
# every other metric, never a degenerate miniature refit. ``scale`` is
# "cosine" or "unbounded".

BuildResult = tuple[
    bool,
    Callable[[int, int], float | None] | None,
    Callable[[Sequence[int], Sequence[int]], float | None] | None,
    str, str,
]


def _build_lexical_tfidf(units: Sequence[str], analysis: DocumentAnalysis, kind: str,
                         config: Mapping[str, Any]) -> BuildResult:
    if len(units) < 1:
        return False, None, None, "cosine", "no units to compare"
    vectors = sem.lexical_vectors(list(units))

    def sim(i: int, j: int) -> float:
        return _sparse_dot(vectors[i], vectors[j])

    def group_sim(indices_a: Sequence[int], indices_b: Sequence[int]) -> float | None:
        return _sparse_group_cosine(vectors, indices_a, indices_b)

    return True, sim, group_sim, "cosine", (
        "backend=lexical_tfidf: document-local TF-IDF content-word cosine (dependency-free; IDF "
        "built from this unit list itself)")


def _build_bm25(units: Sequence[str], analysis: DocumentAnalysis, kind: str,
                config: Mapping[str, Any]) -> BuildResult:
    module, reason = require("rank_bm25")
    if module is None:
        return False, None, None, "unbounded", f"bm25 unavailable: {reason}"
    tokenized = [_content_tokens(unit) for unit in units]
    if not tokenized or not any(tokenized):
        return False, None, None, "unbounded", "no content tokens in any unit"
    try:
        model = module.BM25Okapi(tokenized)
    except Exception as exc:  # pragma: no cover - malformed corpus
        return False, None, None, "unbounded", f"rank_bm25 fit failed ({type(exc).__name__}: {exc})"

    # rank_bm25's own get_scores/get_batch_scores each rebuild a length-n numpy
    # array from self.doc_len on EVERY call, regardless of how many doc_ids are
    # requested -- so even "batch scores against one document" is O(n), which
    # would make every O(n) or O(n * cap) caller below (adjacent pairs,
    # centroid relatedness) an O(n^2) pass over a long document. A single-pair
    # score is instead computed directly here, in O(query length), from the
    # same public fields BM25Okapi.get_scores itself reads (idf, doc_freqs,
    # doc_len, avgdl, k1, b) -- see "Bound O(n^2) work" in the module
    # docstring. If a future rank_bm25 release renames or drops one of these,
    # the AttributeError/IndexError falls back to the (slower but correct)
    # public get_batch_scores rather than crashing the representation.
    group_sim: Callable[[Sequence[int], Sequence[int]], float | None] | None
    try:
        idf, doc_freqs, doc_len, avgdl = model.idf, model.doc_freqs, model.doc_len, model.avgdl
        k1, b = model.k1, model.b

        def sim(i: int, j: int) -> float:
            document = doc_freqs[j]
            length = doc_len[j]
            total = 0.0
            for term in tokenized[i]:
                term_idf = idf.get(term)
                if not term_idf:
                    continue
                frequency = document.get(term, 0)
                if not frequency:
                    continue
                total += term_idf * (frequency * (k1 + 1)
                                     / (frequency + k1 * (1 - b + b * length / avgdl)))
            return total

        def group_sim(indices_a: Sequence[int], indices_b: Sequence[int]) -> float | None:
            # "closing queried against the whole-document index": indices_a's
            # term frequencies and lengths are merged into one synthetic
            # document, indices_b's tokens are concatenated into one query,
            # and both are scored with the SAME idf/avgdl/k1/b the whole
            # document was fit with -- never a fresh 2-document BM25Okapi
            # fit, which is what made this degenerate (Okapi IDF goes
            # negative for a term in more than half of a 2-document corpus;
            # confirmed on three real books, where this metric was negative
            # for all of them under the old design -- see the module
            # docstring).
            document_terms: Counter = Counter()
            length = 0
            for index in indices_a:
                document_terms.update(doc_freqs[index])
                length += doc_len[index]
            if length == 0:
                return None
            query_tokens = [term for index in indices_b for term in tokenized[index]]
            if not query_tokens:
                return None
            total = 0.0
            for term in query_tokens:
                term_idf = idf.get(term)
                if not term_idf:
                    continue
                frequency = document_terms.get(term, 0)
                if not frequency:
                    continue
                total += term_idf * (frequency * (k1 + 1)
                                     / (frequency + k1 * (1 - b + b * length / avgdl)))
            return total
    except (AttributeError, IndexError):  # pragma: no cover - rank_bm25 internals changed
        def sim(i: int, j: int) -> float:
            return float(model.get_batch_scores(tokenized[i], [j])[0])

        group_sim = None

    note = ("backend=bm25: Okapi BM25 relevance of unit i (query) against unit j (document), "
            "over this document's own sentence/paragraph corpus (rank_bm25.BM25Okapi). "
            "UNBOUNDED, not a cosine similarity in [-1, 1] -- never compare its magnitude "
            "against a cosine-scale representation, only its rank/ordering within this document.")
    return True, sim, group_sim, "unbounded", note


def _build_lsa(units: Sequence[str], analysis: DocumentAnalysis, kind: str,
              config: Mapping[str, Any]) -> BuildResult:
    sk, reason = require("sklearn")
    if sk is None:
        return False, None, None, "cosine", f"sklearn unavailable: {reason}"
    if len(units) < 2:
        return False, None, None, "cosine", "needs at least two units"
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    # max_features bounds the TF-IDF matrix width; a full novel's distinct
    # vocabulary can otherwise run into the tens of thousands of columns,
    # which is what made TruncatedSVD's randomized solver measurably slow on
    # a 9,000-sentence book during this suite's own benchmarking (see the
    # task report). Capped by document frequency (TfidfVectorizer's own
    # ranking), not truncated arbitrarily.
    max_features = int(option(config, "lsa_max_features", 1500))
    try:
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1, max_features=max_features)
        matrix = vectorizer.fit_transform(list(units))
    except Exception as exc:
        return False, None, None, "cosine", f"TF-IDF vectorization failed ({type(exc).__name__}: {exc})"
    if matrix.shape[1] < 2:
        return False, None, None, "cosine", "too little distinct vocabulary to fit LSA"
    n_components = max(1, min(int(option(config, "lsa_components", 10)),
                              matrix.shape[1] - 1, len(units) - 1))
    seed = int(option(config, "seed", 0))
    # The SVD basis itself is fit on a capped, seeded sample of rows -- fitting
    # scales with row count, and a book-length document can have thousands of
    # sentences -- but every unit is then projected onto that SAME basis via
    # .transform() (not .fit_transform()), so every sentence still gets a
    # vector; only the (comparatively expensive) basis-fitting step is
    # bounded. See "Bound O(n^2) work" in the module docstring.
    fit_cap = int(option(config, "lsa_fit_sample_cap", 800))
    fit_rows = _reference_sample(matrix.shape[0], fit_cap, seed)
    try:
        svd = TruncatedSVD(n_components=n_components, random_state=seed)
        svd.fit(matrix[fit_rows])
        reduced = svd.transform(matrix)
    except Exception as exc:
        return False, None, None, "cosine", f"TruncatedSVD failed ({type(exc).__name__}: {exc})"
    np, _ = require("numpy")
    if np is None:  # pragma: no cover - numpy ships with sklearn
        return False, None, None, "cosine", "numpy unavailable (required by sklearn itself)"
    norms = np.linalg.norm(reduced, axis=1)
    norms[norms == 0] = 1.0
    reduced = reduced / norms[:, None]

    def sim(i: int, j: int) -> float:
        return float(reduced[i] @ reduced[j])

    def group_sim(indices_a: Sequence[int], indices_b: Sequence[int]) -> float | None:
        # Mean of the ALREADY-fitted (whole-document-basis) per-unit vectors
        # for each group, re-normalized, then cosine -- never a fresh SVD fit
        # on just the two groups. A 2-document TruncatedSVD has at most one
        # meaningful dimension, which forces every pair's cosine to exactly
        # +-1 regardless of content; confirmed on three real books, where
        # this metric was exactly 1.0 for all of them under that design (see
        # the module docstring).
        a = np.asarray(reduced[list(indices_a)]).mean(axis=0)
        b = np.asarray(reduced[list(indices_b)]).mean(axis=0)
        norm_a, norm_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float((a / norm_a) @ (b / norm_b)) if norm_a and norm_b else None

    explained = float(getattr(svd, "explained_variance_ratio_", [0.0]).sum())
    note = (f"backend=lsa: within-document TruncatedSVD over this document's own TF-IDF matrix "
            f"({n_components} components, explained_variance_ratio sum={explained:.3f}); a "
            f"linear-algebra re-projection of THIS document, not a corpus-trained topic model "
            f"-- see the 'topic_models' feature for the leave-one-out corpus-trained channel.")
    return True, sim, group_sim, "cosine", note


_GLOVE_CACHE: dict[str, tuple[Any, str | None]] = {}


def _load_glove(model_name: str) -> tuple[Any, str | None]:
    if model_name in _GLOVE_CACHE:
        return _GLOVE_CACHE[model_name]
    module, reason = require("gensim")
    if module is None:
        _GLOVE_CACHE[model_name] = (None, reason)
        return _GLOVE_CACHE[model_name]
    try:
        import gensim.downloader as api
        vectors = api.load(model_name)
        outcome: tuple[Any, str | None] = (vectors, None)
    except Exception as exc:  # pragma: no cover - network/model failure
        outcome = (None, f"gensim static-vector set {model_name!r} unavailable "
                         f"({type(exc).__name__}: {exc}); try loading it once with network "
                         f"access: python -c \"import gensim.downloader as api; "
                         f"api.load('{model_name}')\"")
    _GLOVE_CACHE[model_name] = outcome
    return outcome


def _reset_glove_cache() -> None:
    _GLOVE_CACHE.clear()


on_reset(_reset_glove_cache)


def _mean_pool(vectors_kv: Any, text: str, np: Any) -> Any:
    words = [word for word in _content_tokens(text) if word in vectors_kv]
    if not words:
        return None
    vector = sum(vectors_kv[word] for word in words) / len(words)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else None


def _dense_group_cosine(vectors: Sequence[Any], indices_a: Sequence[int],
                        indices_b: Sequence[int], np: Any) -> float | None:
    """Cosine between the mean of the (already-normalized, possibly-missing)
    dense vectors at ``indices_a`` and at ``indices_b``, skipping units with
    no vector (e.g. every content word was out-of-vocabulary). Used by
    every dense-vector representation's ``group_sim`` for the
    intro/conclusion metric, so it scores within the SAME per-unit vectors
    already computed for every other metric rather than a fresh, separate
    embedding of two new snippets."""

    def merged(indices: Sequence[int]) -> Any:
        present = [vectors[index] for index in indices if vectors[index] is not None]
        if not present:
            return None
        average = np.mean(np.stack(present), axis=0)
        norm = float(np.linalg.norm(average))
        return average / norm if norm else None

    a, b = merged(indices_a), merged(indices_b)
    return float(np.dot(a, b)) if a is not None and b is not None else None


def _build_glove(units: Sequence[str], analysis: DocumentAnalysis, kind: str,
                 config: Mapping[str, Any]) -> BuildResult:
    model_name = option(config, "glove_model", DEFAULT_GLOVE_MODEL)
    kv, reason = _load_glove(model_name)
    if kv is None:
        return False, None, None, "cosine", f"static_embedding_glove unavailable: {reason}"
    np, np_reason = require("numpy")
    if np is None:  # pragma: no cover - numpy ships with gensim
        return False, None, None, "cosine", f"numpy unavailable: {np_reason}"
    vectors = [_mean_pool(kv, unit, np) for unit in units]
    missing = sum(1 for vector in vectors if vector is None)
    if missing == len(vectors):
        return False, None, None, "cosine", (f"no unit had a recognized word in {model_name!r} "
                                             f"(aggregation=mean, content words only)")

    def sim(i: int, j: int) -> float | None:
        a, b = vectors[i], vectors[j]
        return None if a is None or b is None else float(np.dot(a, b))

    def group_sim(indices_a: Sequence[int], indices_b: Sequence[int]) -> float | None:
        return _dense_group_cosine(vectors, indices_a, indices_b, np)

    note = (f"backend=static_embedding_glove: {model_name!r} vectors via gensim's downloader "
            f"API, mean-pooled over content words (aggregation=mean; TF-IDF-weighted mean and "
            f"SIF are documented, not implemented), cosine similarity. {missing}/{len(vectors)} "
            f"units had no recognized vocabulary word and are excluded pairwise.")
    return True, sim, group_sim, "cosine", note


_SPACY_VECTOR_CACHE: dict[str, tuple[Any, str | None]] = {}


def _load_spacy_vectors(model_name: str) -> tuple[Any, str | None]:
    if model_name in _SPACY_VECTOR_CACHE:
        return _SPACY_VECTOR_CACHE[model_name]
    module, reason = require("spacy")
    if module is None:
        _SPACY_VECTOR_CACHE[model_name] = (None, reason)
        return _SPACY_VECTOR_CACHE[model_name]
    try:
        # An independent, lightweight pipeline -- vectors live on the vocab,
        # not on any of the excluded components -- so this never touches the
        # shared parse a cost="parse" metric would use, and never forces
        # needs_parse for this whole suite (see the module docstring).
        nlp = module.load(model_name, exclude=["parser", "ner", "tagger", "lemmatizer",
                                               "attribute_ruler", "tok2vec"])
        if not nlp.vocab.vectors.shape[0]:
            outcome: tuple[Any, str | None] = (
                None, f"spaCy model {model_name!r} has no static word vectors; try "
                     f"en_core_web_md or en_core_web_lg")
        else:
            outcome = (nlp, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"spaCy vector model {model_name!r} unavailable "
                         f"({type(exc).__name__}: {exc}); python -m spacy download {model_name}")
    _SPACY_VECTOR_CACHE[model_name] = outcome
    return outcome


def _reset_spacy_vector_cache() -> None:
    _SPACY_VECTOR_CACHE.clear()


on_reset(_reset_spacy_vector_cache)


def _build_spacy_vectors(units: Sequence[str], analysis: DocumentAnalysis, kind: str,
                         config: Mapping[str, Any]) -> BuildResult:
    model_name = option(config, "spacy_vector_model", DEFAULT_SPACY_VECTOR_MODEL)
    nlp, reason = _load_spacy_vectors(model_name)
    if nlp is None:
        return False, None, None, "cosine", f"static_embedding_spacy unavailable: {reason}"
    np, np_reason = require("numpy")
    if np is None:  # pragma: no cover - numpy ships with spacy
        return False, None, None, "cosine", f"numpy unavailable: {np_reason}"
    try:
        docs = list(nlp.pipe(list(units)))
    except Exception as exc:  # pragma: no cover - runtime failure
        return False, None, None, "cosine", (f"spaCy vector pipeline failed "
                                             f"({type(exc).__name__}: {exc})")
    vectors = []
    for doc in docs:
        if doc.has_vector and doc.vector_norm:
            vectors.append(doc.vector / doc.vector_norm)
        else:
            vectors.append(None)
    missing = sum(1 for vector in vectors if vector is None)
    if missing == len(vectors):
        return False, None, None, "cosine", f"no unit had a recognized {model_name!r} vector"

    def sim(i: int, j: int) -> float | None:
        a, b = vectors[i], vectors[j]
        return None if a is None or b is None else float(np.dot(a, b))

    def group_sim(indices_a: Sequence[int], indices_b: Sequence[int]) -> float | None:
        return _dense_group_cosine(vectors, indices_a, indices_b, np)

    note = (f"backend=static_embedding_spacy: {model_name!r} static word vectors, mean-pooled "
            f"by spaCy's own doc.vector (aggregation=mean), cosine similarity. "
            f"{missing}/{len(vectors)} units had no recognized vector and are excluded "
            f"pairwise.")
    return True, sim, group_sim, "cosine", note


def _build_sentence_transformer(units: Sequence[str], analysis: DocumentAnalysis, kind: str,
                                config: Mapping[str, Any]) -> BuildResult:
    model_name = option(config, "sentence_transformer_model", sem.DEFAULT_MODEL)
    if kind == "paragraph":
        backend, vectors, note = sem.get_paragraph_vectors(analysis, model_name)
    else:
        backend, vectors, note = sem.get_sentence_vectors(analysis, model_name)
    if backend != "embedding" or vectors is None or len(vectors) == 0:
        return False, None, None, "cosine", (f"sentence_transformer representation unavailable "
                                             f"or fell back to a lexical proxy: {note}")

    def sim(i: int, j: int) -> float:
        return float(vectors[i].dot(vectors[j]))

    def group_sim(indices_a: Sequence[int], indices_b: Sequence[int]) -> float | None:
        # Mean of the ALREADY-encoded (whole-document) sentence-transformer
        # vectors for each group, re-normalized, then cosine -- reuses
        # semantic_adjacent's cache exactly like every other metric here;
        # never a fresh embed_texts call on two new concatenated snippets.
        a = vectors[list(indices_a)].mean(axis=0)
        b = vectors[list(indices_b)].mean(axis=0)
        norm_a = float((a * a).sum() ** 0.5)
        norm_b = float((b * b).sum() ** 0.5)
        return float((a / norm_a).dot(b / norm_b)) if norm_a and norm_b else None

    return True, sim, group_sim, "cosine", f"backend=sentence_transformer: {note}"


REPRESENTATIONS: dict[str, Callable[..., BuildResult]] = {
    "lexical_tfidf": _build_lexical_tfidf,
    "bm25": _build_bm25,
    "lsa": _build_lsa,
    "static_embedding_glove": _build_glove,
    "static_embedding_spacy": _build_spacy_vectors,
    "sentence_transformer": _build_sentence_transformer,
}


# ---------------------------------------------------------- per-representation findings

def _unit_label(scale: str) -> str:
    return "cosine" if scale == "cosine" else "bm25_score"


def _resolve_headline(values: Sequence[float], summary: Mapping[str, Any]
                      ) -> tuple[float | None, str]:
    """The mean, always, with the median kept in ``distribution``.

    Real prose makes adjacent-sentence lexical/BM25 similarity zero-inflated:
    most sentence pairs share no content word at all, so the median sat at
    exactly 0.0 on Alice, The Secret Agent and Flatland alike while the mean
    was 2.08 / 0.75 / 1.94.  An earlier version switched to the mean only
    when the median was degenerate, but then one metric id meant a median for
    one book and a mean for another, and a corpus percentile compared the
    two.  A headline has to be the same statistic for every document, so it
    is the mean for all of them.
    """

    return summary.get("mean"), "mean (the median and full shape are in distribution)"


def _zero_share_percent(values: Sequence[float]) -> float:
    return 100.0 * sum(1 for value in values if value == 0) / len(values) if values else 0.0


def _adjacent_raw(units: Sequence[str], ok: bool,
                  sim_fn: Callable[[int, int], float | None] | None) -> list[float | None]:
    pairs = max(0, len(units) - 1)
    if not ok or pairs < 1 or sim_fn is None:
        return []
    return [sim_fn(i, i + 1) for i in range(pairs)]


def _adjacent_finding(rep: str, kind: str, units: Sequence[str], raw: list[float | None],
                      scale: str, note: str) -> dict[str, Any]:
    metric_id = f"semantic.structure_{rep}_adjacent_{kind}"
    name = f"Adjacent-{kind} {REP_LABELS[rep]} similarity"
    min_sample = MIN_SAMPLE_PAIRS if kind == "sentence" else MIN_SAMPLE_PARAGRAPH_PAIRS
    unit_label = _unit_label(scale)
    pairs = max(0, len(units) - 1)
    if pairs < 1:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=len(units),
                       min_sample=min_sample,
                       warning=f"needs at least two {kind}s; this text has {len(units)}")
    if not raw:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=min_sample, warning=note)
    idxs = [i for i, value in enumerate(raw) if value is not None]
    values = [raw[i] for i in idxs]
    if not values:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=min_sample,
                       warning=f"{note}; no comparable pair (every unit vector was missing)")
    summary = summarize(values)
    ranked = sorted(range(len(values)), key=lambda k: values[k])[:10]
    evidence = [{"index": idxs[k], "value": values[k],
                "unit_a": sem.truncate(units[idxs[k]]), "unit_b": sem.truncate(units[idxs[k] + 1])}
               for k in ranked]
    headline, method = _resolve_headline(values, summary)
    distribution = dict(summary)
    distribution["scale"] = scale
    distribution["low_tail_p10"] = quantile(values, 0.10)
    distribution["zero_share_percent"] = _zero_share_percent(values)
    distribution["aggregation"] = f"headline is the {method} of the per-pair values"
    overlap = _OVERLAP_NOTES.get((rep, f"adjacent_{kind}"))
    if overlap:
        distribution["overlaps_existing_metric_id"] = overlap
    return finding(metric_id, name, headline, unit_label, family=FAMILY,
                  sample_size=len(values), min_sample=min_sample, distribution=distribution,
                  evidence=evidence, warning=note)


def _centroid_finding(rep: str, sentences: Sequence[str], ok: bool,
                      sim_fn: Callable[[int, int], float | None] | None, scale: str, note: str,
                      config: Mapping[str, Any]) -> dict[str, Any]:
    metric_id = f"semantic.structure_{rep}_centroid_relatedness"
    name = f"{REP_LABELS[rep]} similarity of each sentence to a reference sample of the document"
    unit_label = _unit_label(scale)
    n = len(sentences)
    if n < 3:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=n,
                       min_sample=3, warning=f"needs at least three sentences; this text has {n}")
    if not ok or sim_fn is None:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=3, warning=note)
    seed = int(option(config, "seed", 0))
    cap = int(option(config, "centroid_reference_cap", 200))
    # Both the reference set AND the sentences queried against it are capped:
    # capping only the reference set still leaves an O(n * cap) pass with a
    # per-pair cost (get_batch_scores et al.) that is not free, so a
    # 300,000-word novel's ~15,000 sentences would still mean millions of
    # individual similarity calls. Sampling the QUERY side too (a distinct
    # seed so the two samples are not forced identical) bounds the whole
    # metric to O(query_cap * reference_cap), independent of document length
    # -- see "Bound O(n^2) work" in the module docstring.
    query_cap = int(option(config, "centroid_query_cap", 400))
    reference = _reference_sample(n, cap, seed)
    queries = _reference_sample(n, query_cap, seed + 1)
    means: list[float] = []
    for i in queries:
        values = [value for value in (sim_fn(i, j) for j in reference if j != i)
                 if value is not None]
        if values:
            means.append(statistics.fmean(values))
    if not means:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=3, warning=f"{note}; no comparable pair")
    summary = summarize(means)
    headline, method = _resolve_headline(means, summary)
    distribution = dict(summary)
    distribution.update({"scale": scale, "reference_sample_size": len(reference),
                         "query_sample_size": len(queries), "document_sentence_count": n,
                         "seed": seed, "zero_share_percent": _zero_share_percent(means),
                         "aggregation": f"headline is the {method} of each queried sentence's own "
                                       f"mean relatedness to the reference sample"})
    overlap = _OVERLAP_NOTES.get((rep, "centroid_relatedness"))
    if overlap:
        distribution["overlaps_existing_metric_id"] = overlap
    return finding(metric_id, name, headline, unit_label, family=FAMILY,
                  sample_size=len(means), min_sample=3, distribution=distribution, warning=note)


def _window_drift_finding(rep: str, sentences: Sequence[str], ok: bool,
                          sim_fn: Callable[[int, int], float | None] | None, scale: str,
                          note: str, config: Mapping[str, Any]) -> dict[str, Any]:
    metric_id = f"semantic.structure_{rep}_window_drift"
    name = (f"{REP_LABELS[rep]} local-window semantic drift (block-mean adjacent-similarity "
            f"change)")
    unit_label = _unit_label(scale)
    window = max(2, int(option(config, "structure_window", 5)))
    pairs = max(0, len(sentences) - 1)
    if not ok or sim_fn is None:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=pairs,
                       min_sample=window * 2, warning=note)
    if pairs < window * 2:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=pairs,
                       min_sample=window * 2,
                       warning=f"needs at least {window * 2} adjacent pairs "
                               f"(2x structure_window); found {pairs}")
    values = [value for value in (sim_fn(i, i + 1) for i in range(pairs)) if value is not None]
    if len(values) < window * 2:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=len(values),
                       min_sample=window * 2, warning=f"{note}; too few comparable pairs")
    blocks = [values[i:i + window] for i in range(0, len(values), window)]
    block_means = [statistics.fmean(block) for block in blocks if block]
    if len(block_means) < 2:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=len(values),
                       min_sample=window * 2, warning="fewer than two windows to compare")
    deltas = [abs(block_means[i] - block_means[i - 1]) for i in range(1, len(block_means))]
    summary = summarize(deltas)
    headline, method = _resolve_headline(deltas, summary)
    distribution = dict(summary)
    distribution.update({
        "scale": scale, "window_size": window, "block_count": len(block_means),
        "zero_share_percent": _zero_share_percent(deltas),
        "aggregation": f"headline is the {method} of the block-to-block absolute mean change",
    })
    return finding(metric_id, name, headline, unit_label, family=FAMILY,
                  sample_size=len(deltas), min_sample=1, distribution=distribution, warning=note)


def _dispersion_finding(rep: str, sentences: Sequence[str], ok: bool,
                        sim_fn: Callable[[int, int], float | None] | None, scale: str, note: str,
                        config: Mapping[str, Any]) -> dict[str, Any]:
    metric_id = f"semantic.structure_{rep}_global_dispersion"
    name = (f"{REP_LABELS[rep]} global semantic dispersion (spread of a capped random sample of "
            f"non-adjacent sentence-pair similarities)")
    unit_label = _unit_label(scale)
    n = len(sentences)
    if not ok or sim_fn is None:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=4, warning=note)
    if n < 4:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=n,
                       min_sample=4, warning=f"needs at least four sentences; found {n}")
    seed = int(option(config, "seed", 0))
    cap = int(option(config, "dispersion_pair_cap", 300))
    sample = _sample_non_adjacent_pairs(n, cap, seed, min_gap=2)
    if not sample:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=4, warning="no non-adjacent sentence pair available")
    values = [value for value in (sim_fn(i, j) for i, j in sample) if value is not None]
    if not values:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=4, warning=f"{note}; no comparable pair")
    summary = summarize(values)
    distribution = dict(summary)
    distribution.update({
        "scale": scale, "pairs_sampled": len(values), "pair_cap": cap, "seed": seed,
        "zero_share_percent": _zero_share_percent(values),
        "aggregation": "the headline value is the STANDARD DEVIATION of the sampled "
                       "similarities (dispersion is a spread statistic, never a median or mean "
                       "fallback -- see _resolve_headline's docstring for why that choice is "
                       "specific to a central-tendency headline); 'median'/'mean' above describe "
                       "the sampled similarities themselves, not the dispersion",
    })
    return finding(metric_id, name, summary.get("std"), unit_label, family=FAMILY,
                  sample_size=len(values), min_sample=4, distribution=distribution, warning=note)


def _intro_conclusion_finding(rep: str, sentences: Sequence[str], ok: bool,
                              group_sim: Callable[[Sequence[int], Sequence[int]], float | None]
                              | None, scale: str, note: str,
                              config: Mapping[str, Any]) -> dict[str, Any]:
    metric_id = f"semantic.structure_{rep}_intro_conclusion_similarity"
    name = (f"{REP_LABELS[rep]} opening-to-closing similarity (conclusion-to-introduction "
            f"similarity; opening-to-closing semantic DISTANCE is 1 - this value on a "
            f"cosine-scale representation)")
    unit_label = _unit_label(scale)
    k = max(1, int(option(config, "intro_conclusion_sentences", 3)))
    n = len(sentences)
    if n < 2 * k:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=n,
                       min_sample=2 * k,
                       warning=f"needs at least {2 * k} sentences (2x intro_conclusion_sentences); "
                               f"this text has {n}")
    if not ok or group_sim is None:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=2 * k, warning=note)
    opening_indices = list(range(k))
    closing_indices = list(range(n - k, n))
    value = group_sim(opening_indices, closing_indices)
    if value is None:
        return finding(metric_id, name, None, unit_label, family=FAMILY, sample_size=0,
                       min_sample=2 * k,
                       warning=f"{note}; no comparable pair (every unit vector was missing)")
    return finding(
        metric_id, name, value, unit_label, family=FAMILY, sample_size=2 * k, min_sample=2 * k,
        distribution={
            "opening_sentences": k, "closing_sentences": k, "scale": scale,
            "aggregation": "cosine (or, for bm25, the query-vs-synthetic-document score) between "
                          "the AVERAGE of the opening-k and the AVERAGE of the closing-k sentence "
                          "vectors/statistics, both drawn from the SAME whole-document fit this "
                          "representation already produced for every other metric -- never a "
                          "fresh, separate fit on just the two snippets, which degenerates (see "
                          "the module docstring)",
        }, warning=note)


def _representation_findings(rep: str, analysis: DocumentAnalysis, config: Mapping[str, Any]
                             ) -> tuple[list[dict[str, Any]], list[float | None]]:
    builder = REPRESENTATIONS[rep]
    sentences, paragraphs = analysis.sentences, analysis.paragraphs
    sok, ssim, sgroup, sscale, snote = builder(sentences, analysis, "sentence", config)
    pok, psim, _pgroup, pscale, pnote = builder(paragraphs, analysis, "paragraph", config)
    raw_sentence_values = _adjacent_raw(sentences, sok, ssim)
    findings = [
        _adjacent_finding(rep, "sentence", sentences, raw_sentence_values, sscale, snote),
        _adjacent_finding(rep, "paragraph", paragraphs, _adjacent_raw(paragraphs, pok, psim),
                          pscale, pnote),
        _centroid_finding(rep, sentences, sok, ssim, sscale, snote, config),
        _window_drift_finding(rep, sentences, sok, ssim, sscale, snote, config),
        _dispersion_finding(rep, sentences, sok, ssim, sscale, snote, config),
        _intro_conclusion_finding(rep, sentences, sok, sgroup, sscale, snote, config),
    ]
    return findings, raw_sentence_values


# --------------------------------------------------------- representation disagreement

def _spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    module, _reason = require("scipy.stats")
    if module is not None:
        try:
            rho, p_value = module.spearmanr(a, b)
            return float(rho) if rho == rho else None  # NaN check
        except Exception:  # pragma: no cover - scipy input guard
            pass

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average_rank
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    variance_a = sum((x - mean_a) ** 2 for x in ra)
    variance_b = sum((y - mean_b) ** 2 for y in rb)
    if variance_a <= 0 or variance_b <= 0:
        return None
    return covariance / math.sqrt(variance_a * variance_b)


def _rep_tie_seed(rep: str, base_seed: int) -> int:
    """A deterministic, per-representation tie-break seed.

    Not Python's built-in ``hash(str)``: that is salted per-process
    (``PYTHONHASHSEED``) unless explicitly disabled, which would make the
    tie-break -- and therefore which transitions get flagged -- silently
    different between two runs of an identical grading job. ``zlib.crc32``
    is a stable, dependency-free hash across processes and Python versions.
    """

    return (base_seed * 1_000_003) ^ zlib.crc32(rep.encode("utf-8"))


def _low_tail_flags_by_rank(clean: Sequence[tuple[int, float]], quantile_share: float,
                            tie_break_seed: int) -> tuple[set[int], dict[str, Any]]:
    """The lowest ``quantile_share`` of ``clean`` (``(index, value)`` pairs),
    chosen BY RANK rather than by an absolute quantile threshold.

    An absolute threshold (this suite's earlier approach: flag every value
    ``<= quantile(values, quantile_share)``) is what a zero-inflated
    representation breaks: lexical_tfidf and BM25 adjacent-sentence scores on
    real prose are mostly exactly 0.0 (most sentence pairs share no content
    word), so their own 10th-percentile THRESHOLD is 0.0 too, and "every
    value <= 0.0" flags nearly the whole document -- confirmed on three real
    books, where the resulting consensus share was 98.3-99.7% instead of
    anywhere near the configured 10%. Picking a fixed COUNT of the lowest
    values by rank flags close to ``quantile_share`` of the sample regardless
    of how many values are tied at the bottom.

    The tie-break is a deterministic pseudo-random order SEEDED PER
    REPRESENTATION (``tie_break_seed``, from :func:`_rep_tie_seed`), not
    "lowest index first": two lexical representations (lexical_tfidf and
    BM25 both measure the same content-word overlap) routinely share the
    exact same large block of zero-valued transitions on real prose, and
    breaking ties by plain index order would make both representations pick
    the identical arbitrary subset of that shared block every time --
    manufacturing spurious cross-representation "agreement" out of nothing
    but a shared tie-break rule rather than real agreement about which
    transition is least coherent. A per-representation seed avoids that
    while staying fully deterministic and reproducible.

    The tie is not hidden either way: when more values are tied at the exact
    value that sits at the flag boundary than the flag quota itself, WHICH
    of those tied transitions ends up flagged is an arbitrary (though
    deterministic) choice, and ``degenerate_tie`` says so, alongside how many
    were tied there, so a reader is never left assuming the boundary was
    informative when it was not.
    """

    n = len(clean)
    if n == 0:
        return set(), {"flagged": 0, "total": 0, "flagged_share_percent": None,
                       "tied_at_boundary_value": 0, "degenerate_tie": False}
    count = min(n, max(1, round(quantile_share * n)))
    rng = random.Random(tie_break_seed)
    tie_keys = {index: rng.random() for index, _ in clean}
    ordered = sorted(clean, key=lambda pair: (pair[1], tie_keys[pair[0]]))
    boundary_value = ordered[count - 1][1]
    tied_at_boundary = sum(1 for _, value in clean if value == boundary_value)
    flagged = {index for index, _ in ordered[:count]}
    return flagged, {
        "flagged": len(flagged), "total": n,
        "flagged_share_percent": 100.0 * len(flagged) / n,
        "boundary_value": boundary_value, "tied_at_boundary_value": tied_at_boundary,
        "degenerate_tie": tied_at_boundary > count,
    }


def _disagreement_findings(rep_values: Mapping[str, list[float | None]],
                           analysis: DocumentAnalysis,
                           config: Mapping[str, Any]) -> list[dict[str, Any]]:
    active = {rep: values for rep, values in rep_values.items()
             if values and sum(1 for v in values if v is not None) >= MIN_SAMPLE_DISAGREEMENT}
    reps = sorted(active)
    correlation_id = "semantic.structure_disagreement_rank_correlation"
    correlation_name = ("Median pairwise rank correlation of adjacent-sentence transition "
                        "scores across active representations")
    single_id = "semantic.structure_disagreement_single_representation_only"
    single_name = ("Adjacent-sentence transitions flagged low-coherence by exactly one active "
                  "representation")
    consensus_id = "semantic.structure_disagreement_consensus_low_coherence"
    consensus_name = ("Adjacent-sentence transitions flagged low-coherence by a majority of "
                      "active representations")
    if len(reps) < 2:
        warning = f"needs at least two active representations with enough transitions; found {len(reps)}"
        return [
            finding(correlation_id, correlation_name, None, "spearman_rho", family=FAMILY,
                   sample_size=0, min_sample=1, warning=warning),
            finding(single_id, single_name, None, "%", family=FAMILY, sample_size=0,
                   min_sample=1, warning=warning),
            finding(consensus_id, consensus_name, None, "%", family=FAMILY, sample_size=0,
                   min_sample=1, warning=warning),
        ]

    correlations: dict[str, float] = {}
    for a_index in range(len(reps)):
        for b_index in range(a_index + 1, len(reps)):
            rep_a, rep_b = reps[a_index], reps[b_index]
            values_a, values_b = active[rep_a], active[rep_b]
            n = min(len(values_a), len(values_b))
            paired = [(values_a[i], values_b[i]) for i in range(n)
                     if values_a[i] is not None and values_b[i] is not None]
            if len(paired) < MIN_SAMPLE_DISAGREEMENT:
                continue
            xs, ys = zip(*paired)
            rho = _spearman(list(xs), list(ys))
            if rho is not None:
                correlations[f"{rep_a}__{rep_b}"] = rho
    rho_values = list(correlations.values())
    finding_correlation = finding(
        correlation_id, correlation_name, statistics.median(rho_values) if rho_values else None,
        "spearman_rho", family=FAMILY, sample_size=len(rho_values), min_sample=1,
        distribution={
            "pairwise": correlations, "representations": reps,
            # Left as the median deliberately, unlike the adjacent/centroid/
            # window headlines above: rho_values is a small set of pairwise
            # Spearman correlations (continuous in [-1, 1], one per
            # representation PAIR, not per transition), so exact ties are not
            # the zero-inflation pattern _resolve_headline exists for --
            # checked directly on real books (Alice, The Secret Agent,
            # Flatland) and the median never sat at a degenerate tied value.
            "aggregation": "headline is the median pairwise Spearman rank correlation",
        },
        warning=None if rho_values else
        "no representation pair had enough paired transitions to correlate")

    low_tail_q = float(option(config, "disagreement_low_tail_quantile", 0.10))
    base_seed = int(option(config, "seed", 0))
    per_rep_low: dict[str, set[int]] = {}
    per_rep_diagnostics: dict[str, dict[str, Any]] = {}
    for rep, values in active.items():
        clean = [(i, v) for i, v in enumerate(values) if v is not None]
        if len(clean) < MIN_SAMPLE_DISAGREEMENT:
            continue
        flagged, diagnostics = _low_tail_flags_by_rank(clean, low_tail_q,
                                                       _rep_tie_seed(rep, base_seed))
        per_rep_low[rep] = flagged
        per_rep_diagnostics[rep] = diagnostics

    all_indices: set[int] = set()
    for flagged in per_rep_low.values():
        all_indices |= flagged
    flag_counts = Counter({index: sum(1 for flagged in per_rep_low.values() if index in flagged)
                           for index in all_indices})
    n_active = len(per_rep_low)
    single_flags = sorted(index for index, count in flag_counts.items() if count == 1)
    consensus_threshold = max(2, math.ceil(n_active / 2)) if n_active else None
    consensus_flags = sorted(index for index, count in flag_counts.items()
                             if consensus_threshold and count >= consensus_threshold)

    if not per_rep_low:
        warning = "no active representation had enough transitions to flag a low-coherence tail"
        finding_single = finding(single_id, single_name, None, "%", family=FAMILY, sample_size=0,
                                 min_sample=1, warning=warning)
        finding_consensus = finding(consensus_id, consensus_name, None, "%", family=FAMILY,
                                    sample_size=0, min_sample=1, warning=warning)
    else:
        sentences = analysis.sentences
        # Every representation flags close to low_tail_q of its OWN transitions
        # by rank (deterministic tie-break: lowest value, then lowest index),
        # never by an absolute quantile threshold -- a zero-inflated
        # representation's 10th-percentile THRESHOLD is often 0.0 itself, which
        # would otherwise flag nearly every zero transition (confirmed on real
        # books: lexical_tfidf/bm25 flagged 98-99.7% of transitions under the
        # old threshold rule). ``per_representation`` below reports exactly how
        # many each representation flagged and whether that flag boundary sat
        # inside a large tie (``degenerate_tie``), so a reader can see when the
        # selection at the edge was effectively arbitrary rather than hiding it.
        finding_single = finding(
            single_id, single_name,
            rate(len(single_flags), len(all_indices), 100.0) if all_indices else None, "%",
            family=FAMILY, sample_size=len(all_indices), min_sample=1,
            distribution={"representations_compared": sorted(per_rep_low),
                         "low_tail_quantile": low_tail_q,
                         "flagging_method": "lowest low_tail_quantile share BY RANK per "
                                           "representation (deterministic tie-break: value then "
                                           "index), not an absolute quantile threshold -- see "
                                           "per_representation for exactly how many each flagged",
                         "per_representation": per_rep_diagnostics,
                         "flagged_by_exactly_one": len(single_flags),
                         "total_flagged_by_any": len(all_indices)},
            evidence=[{"sentence_index": index,
                      "flagged_by": [rep for rep, flagged in per_rep_low.items()
                                    if index in flagged]}
                     for index in single_flags[:20]],
            warning=None if all_indices else "no transition was flagged by any representation")
        finding_consensus = finding(
            consensus_id, consensus_name,
            rate(len(consensus_flags), len(all_indices), 100.0) if all_indices else None, "%",
            family=FAMILY, sample_size=len(all_indices), min_sample=1,
            distribution={"representations_compared": sorted(per_rep_low),
                         "low_tail_quantile": low_tail_q,
                         "flagging_method": "lowest low_tail_quantile share BY RANK per "
                                           "representation (deterministic tie-break: value then "
                                           "index), not an absolute quantile threshold -- see "
                                           "per_representation for exactly how many each flagged",
                         "per_representation": per_rep_diagnostics,
                         "consensus_threshold": consensus_threshold,
                         "flagged_by_consensus": len(consensus_flags)},
            evidence=[{"sentence_index": index,
                      "flagged_by": [rep for rep, flagged in per_rep_low.items()
                                    if index in flagged],
                      "sentence_a": sem.truncate(sentences[index])
                                    if index < len(sentences) else None,
                      "sentence_b": sem.truncate(sentences[index + 1])
                                    if index + 1 < len(sentences) else None}
                     for index in consensus_flags[:20]],
            warning=None if all_indices else "no transition was flagged by any representation")
    return [finding_correlation, finding_single, finding_consensus]


# ------------------------------------------------------------- corpus topic models

_FALLBACK_VOCAB: tuple[str, ...] = tuple("""
time year people way day man thing woman life child world school state family student group
country problem hand part place case week company system program question work government
number night point home water room mother area money story fact month lot right study book
eye job word business issue side kind head house service friend father power hour game line
end member law car city community name president team minute idea body information back
parent face others level office door health person art war history party result change morning
reason research girl guy moment air teacher force education foot boy age policy process music
market sense nation plan college interest death experience effect use class control care field
development role effort rate heart drug show leader light voice wife whole police mind price
report decision son view relationship town road arm difference building tree love population
professional building rise cost land human owner rest walk order rock date table body model
letter conversation summer garden mountain river forest castle king queen village island animal
kitchen paper energy machine tool wall floor window street bridge ocean beach cloud rain snow
storm wind fire earth stone glass silver gold iron wood metal cotton silk paper ink pencil
brush color shape sound taste smell touch feeling emotion memory dream hope fear anger joy
sorrow peace war battle victory defeat journey travel voyage adventure quest mystery secret
treasure map compass sword shield armor magic spell curse blessing prophecy legend myth tale
""".split())


_VOCAB_CACHE: dict[int, tuple[list[str], str]] = {}


def _fixed_vocabulary(cap: int) -> tuple[list[str], str]:
    if cap in _VOCAB_CACHE:
        return _VOCAB_CACHE[cap]
    module, _reason = require("wordfreq")
    if module is not None:
        try:
            import importlib.metadata as _metadata
            try:
                version = _metadata.version("wordfreq")
            except Exception:  # pragma: no cover - metadata missing
                version = "unknown"
            candidates = module.top_n_list("en", cap * 4)
            words: list[str] = []
            seen: set[str] = set()
            for candidate in candidates:
                lowered = candidate.lower()
                if (lowered.isalpha() and len(lowered) >= 3 and lowered not in _STOPWORDS
                        and lowered not in seen):
                    words.append(lowered)
                    seen.add(lowered)
                if len(words) >= cap:
                    break
            result = (words, f"wordfreq:{version}:top{cap}:v1")
        except Exception:  # pragma: no cover - wordfreq runtime failure
            result = (list(_FALLBACK_VOCAB[:cap]),
                     f"builtin:v1:top{min(cap, len(_FALLBACK_VOCAB))}")
    else:
        result = (list(_FALLBACK_VOCAB[:cap]), f"builtin:v1:top{min(cap, len(_FALLBACK_VOCAB))}")
    _VOCAB_CACHE[cap] = result
    return result


def _reset_vocab_cache() -> None:
    _VOCAB_CACHE.clear()


on_reset(_reset_vocab_cache)


def profile_vector(analysis: DocumentAnalysis, config: Mapping[str, Any] | None
                   ) -> dict[str, float] | None:
    """The per-corpus-book vector :mod:`textgrader.corpus` caches under
    ``feature_profiles['semantic_structure_suite']``: a bounded, deterministic
    sample of this book's paragraph word counts, restricted to a fixed,
    versioned vocabulary. See the module docstring's "hard design problem"
    section for why this shape (rather than a corpus-wide vocabulary or the
    raw text) is what lets LDA/NMF/HDP be fit at grading time without ever
    reading the corpus's raw books again.

    Returns ``None`` (nothing cached for this book) when ``features.
    topic_models`` is off in the config the corpus is being profiled with,
    the book has fewer than two paragraphs, no fixed-vocabulary word was
    found in the sample, or the fixed vocabulary itself could not be built.
    """

    config = config or {}
    if not _features(config).get("topic_models", True):
        return None
    paragraphs = analysis.paragraphs
    if len(paragraphs) < 2:
        return None
    vocab_size = int(option(config, "topic_vocab_size", DEFAULT_VOCAB_SIZE))
    vocab, _version = _fixed_vocabulary(vocab_size)
    if not vocab:
        return None
    vocab_set = set(vocab)
    seed = int(option(config, "topic_profile_seed", 0))
    sample_size = min(int(option(config, "topic_profile_paragraph_sample", 30)), len(paragraphs))
    indices = sorted(random.Random(seed).sample(range(len(paragraphs)), sample_size))
    word_cap = int(option(config, "topic_profile_paragraph_word_cap", 300))
    counts: Counter[str] = Counter()
    for index in indices:
        for word in _tokens(paragraphs[index])[:word_cap]:
            if word in vocab_set:
                counts[word] += 1
    if not counts:
        return None
    max_terms = int(option(config, "topic_profile_max_terms", 250))
    row = {word: float(count) for word, count in counts.most_common(max_terms)}
    row["__seed__"] = float(seed)
    row["__sample_paragraphs__"] = float(len(indices))
    row["__vocab_size__"] = float(len(vocab))
    row["__schema__"] = float(TOPIC_PROFILE_SCHEMA_VERSION)
    return row


def _corpus_rows(profile: Mapping[str, Any] | None) -> list[dict[str, float]]:
    return list(((profile or {}).get("feature_profiles") or {}).get("semantic_structure_suite")
               or [])


def _book_matches_source(book: Mapping[str, Any], source: str | None) -> bool:
    if not source:
        return False
    return book.get("source_filename") == source or book.get("source_path") == source


def _corpus_term_matrix(profile: Mapping[str, Any] | None, vocab: Sequence[str],
                        exclude_source: str | None) -> tuple[list[list[float]], int, int]:
    rows = _corpus_rows(profile)
    books = (profile or {}).get("books") or []
    vocab_index = {word: i for i, word in enumerate(vocab)}
    matrix: list[list[float]] = []
    excluded = 0
    total_rows = 0
    for i, row in enumerate(rows):
        if not row:
            continue
        total_rows += 1
        book = books[i] if i < len(books) else {}
        if _book_matches_source(book, exclude_source):
            excluded += 1
            continue
        vector = [0.0] * len(vocab)
        hits = 0
        for term, count in row.items():
            if term.startswith("__"):
                continue
            index = vocab_index.get(term)
            if index is not None:
                vector[index] = count
                hits += 1
        if hits == 0:
            continue
        matrix.append(vector)
    return matrix, total_rows, excluded


_TOPIC_FIT_CACHE: dict[Any, Any] = {}
_TOPIC_MODEL_NAMES = ("lda", "nmf", "hdp")


def _reset_topic_fit_cache() -> None:
    _TOPIC_FIT_CACHE.clear()


on_reset(_reset_topic_fit_cache)


def _fit_topic_models_uncached(profile: Mapping[str, Any] | None, source: str | None,
                               config: Mapping[str, Any]) -> dict[str, Any]:
    vocab_size = int(option(config, "topic_vocab_size", DEFAULT_VOCAB_SIZE))
    vocab, vocab_version = _fixed_vocabulary(vocab_size)
    seed = int(option(config, "seed", 0))
    matrix, total_rows, excluded = _corpus_term_matrix(profile, vocab, source)
    n_books = len(matrix)
    meta: dict[str, Any] = {
        "vocab_size": len(vocab), "vocab_version": vocab_version, "seed": seed,
        "corpus_rows_total": total_rows, "corpus_rows_excluded_leave_one_out": excluded,
        "corpus_books_used": n_books,
    }
    if n_books < MIN_TOPIC_CORPUS_BOOKS:
        meta["reason"] = (
            f"needs a corpus profile with at least {MIN_TOPIC_CORPUS_BOOKS} usable "
            f"'semantic_structure_suite' feature_profiles rows after leave-one-out exclusion "
            f"(found {n_books}, {excluded} excluded as the graded book itself, {total_rows} "
            f"total rows); build one with semantic_structure_suite enabled and "
            f"features.topic_models on (the default) -- see config.json's "
            f"'_requires_topic_models' note")
        return {"models": {}, "meta": meta, "vocab": vocab}
    np, np_reason = require("numpy")
    if np is None:  # pragma: no cover - numpy ships with sklearn
        meta["reason"] = f"numpy unavailable: {np_reason}"
        return {"models": {}, "meta": meta, "vocab": vocab}
    array = np.array(matrix)
    requested_topics = int(option(config, "topic_n_topics", 8))
    n_topics = max(2, min(requested_topics, n_books - 1, len(vocab) - 1))
    meta["n_topics_requested"] = requested_topics
    meta["n_topics_used"] = n_topics
    models: dict[str, Any] = {}

    sk, sk_reason = require("sklearn")
    if sk is not None:
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=seed,
                                            max_iter=50)
            lda.fit(array)
            models["lda"] = {"model": lda, "kind": "sklearn", "library_version": sk.__version__}
        except Exception as exc:
            meta["lda_error"] = f"{type(exc).__name__}: {exc}"
        try:
            from sklearn.decomposition import NMF
            nmf = NMF(n_components=n_topics, random_state=seed, max_iter=400, init="nndsvda")
            nmf.fit(array)
            models["nmf"] = {"model": nmf, "kind": "sklearn", "library_version": sk.__version__}
        except Exception as exc:
            meta["nmf_error"] = f"{type(exc).__name__}: {exc}"
    else:
        meta["sklearn_reason"] = sk_reason

    tomotopy_module, tomo_reason = require("tomotopy")
    if tomotopy_module is not None:
        try:
            hdp = tomotopy_module.HDPModel(tw=tomotopy_module.TermWeight.ONE, seed=seed)
            for row in matrix:
                doc_tokens = [vocab[index] for index, count in enumerate(row)
                             for _ in range(int(count))]
                if doc_tokens:
                    hdp.add_doc(doc_tokens)
            iterations = int(option(config, "hdp_iterations", 200))
            hdp.train(iterations, workers=1)
            models["hdp"] = {"model": hdp, "kind": "tomotopy",
                             "library_version": getattr(tomotopy_module, "__version__", None),
                             "hdp_topic_slots": hdp.k, "hdp_live_topics": hdp.live_k}
        except Exception as exc:
            meta["hdp_error"] = f"{type(exc).__name__}: {exc}"
    else:
        meta["tomotopy_reason"] = tomo_reason

    return {"models": models, "meta": meta, "vocab": vocab}


def _profile_fingerprint(profile: Mapping[str, Any] | None) -> tuple[Any, ...]:
    """A cheap, structural stand-in for identity, used because a plain
    ``dict`` (what every real profile is -- loaded straight from JSON)
    cannot be weakly referenced the way :mod:`semantic_adjacent` guards its
    own per-analysis cache. ``id(profile)`` alone is not safe: a short-lived
    profile dict (exactly what this suite's own tests construct, one per
    test) can be garbage collected and a LATER, unrelated profile can be
    allocated at the same recycled address, which would otherwise hand back
    a completely different corpus's fitted topic models. Combining the
    object id with its book count, its ``feature_profiles`` sub-object's id
    and that sub-object's own row count makes a false cache hit require the
    new profile to coincidentally match on all four, rather than only one.
    """

    if profile is None:
        return (None,)
    books = profile.get("books")
    rows = _corpus_rows(profile)
    return (id(profile), len(books) if books is not None else -1, id(rows), len(rows))


def _fit_topic_models(profile: Mapping[str, Any] | None, analysis: DocumentAnalysis,
                      config: Mapping[str, Any]) -> dict[str, Any]:
    key = (int(option(config, "topic_vocab_size", DEFAULT_VOCAB_SIZE)),
          int(option(config, "topic_n_topics", 8)), int(option(config, "seed", 0)),
          analysis.source, int(option(config, "hdp_iterations", 200)))
    fingerprint = _profile_fingerprint(profile)
    cached = _TOPIC_FIT_CACHE.get(key)
    if cached is not None and profile is not None:
        cached_fingerprint, value = cached
        if cached_fingerprint == fingerprint:
            return value
    result = _fit_topic_models_uncached(profile, analysis.source, config)
    if profile is not None:
        _TOPIC_FIT_CACHE[key] = (fingerprint, result)
    return result


def _paragraph_term_vectors(paragraphs: Sequence[str], vocab: Sequence[str],
                            word_cap: int) -> list[list[float]]:
    vocab_index = {word: i for i, word in enumerate(vocab)}
    vectors = []
    for paragraph in paragraphs:
        vector = [0.0] * len(vocab)
        for word in _tokens(paragraph)[:word_cap]:
            index = vocab_index.get(word)
            if index is not None:
                vector[index] += 1.0
        vectors.append(vector)
    return vectors


def _topic_distributions(entry: Mapping[str, Any], paragraph_vectors: list[list[float]],
                         vocab: Sequence[str]) -> list[list[float] | None]:
    if entry["kind"] == "sklearn":
        np, _reason = require("numpy")
        if np is None:  # pragma: no cover - numpy ships with sklearn
            return [None] * len(paragraph_vectors)
        array = np.array(paragraph_vectors)
        distribution = entry["model"].transform(array)
        totals = distribution.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        normalized = distribution / totals
        return [row.tolist() if row.sum() > 0 else None for row in normalized]
    model = entry["model"]
    out: list[list[float] | None] = []
    for vector in paragraph_vectors:
        doc_tokens = [vocab[index] for index, count in enumerate(vector)
                     for _ in range(int(count))]
        if not doc_tokens:
            out.append(None)
            continue
        try:
            unseen = model.make_doc(doc_tokens)
            distribution, _log_likelihood = model.infer(unseen)
            out.append(list(float(p) for p in distribution))
        except Exception:  # pragma: no cover - tomotopy inference failure
            out.append(None)
    return out


_TOPIC_METRIC_STEMS: tuple[tuple[str, str, str], ...] = (
    ("distribution_entropy", "distribution entropy", "bits"),
    ("dominant_confidence", "dominant-topic probability", "probability"),
    ("switch_rate", "topic-switch rate between adjacent paragraphs", "%"),
    ("recurrence_interval", "recurrence interval (paragraphs between repeats of the same "
                           "dominant topic)", "paragraphs"),
    ("run_length", "persistence (dominant-topic run length)", "paragraphs"),
    ("active_topics", "active-topic count per paragraph", "count"),
    ("concentration", "balance/concentration across the document", "ratio"),
)


def _empty_topic_findings(model_name: str, reason: str,
                          base_distribution: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        finding(f"semantic.structure_topic_{model_name}_{stem}",
               f"{model_name.upper()} corpus-trained topic {label}", None, unit, family=FAMILY,
               sample_size=0, min_sample=MIN_TOPIC_PARAGRAPHS, distribution=dict(base_distribution),
               warning=reason)
        for stem, label, unit in _TOPIC_METRIC_STEMS
    ]


def _topic_findings(model_name: str, entry: Mapping[str, Any], meta: Mapping[str, Any],
                    dist_list: list[list[float] | None],
                    config: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = dict(meta)
    base.update({"library": entry["kind"], "library_version": entry.get("library_version")})
    if model_name == "hdp":
        base.update({"hdp_topic_slots": entry.get("hdp_topic_slots"),
                    "hdp_live_topics": entry.get("hdp_live_topics")})
    base["overlaps_existing_metric_id"] = TOPIC_OVERLAP_NOTE
    valid = [(i, d) for i, d in enumerate(dist_list) if d]
    n = len(valid)
    if n < MIN_TOPIC_PARAGRAPHS:
        return _empty_topic_findings(
            model_name,
            f"needs at least {MIN_TOPIC_PARAGRAPHS} paragraphs with a recognized "
            f"vocabulary word; found {n}", base)

    threshold = float(option(config, "topic_active_threshold", 0.1))
    entropies, confidences, actives, dominants = [], [], [], []
    for _, distribution in valid:
        entropies.append(-sum(p * math.log2(p) for p in distribution if p > 1e-12))
        confidences.append(max(distribution))
        actives.append(sum(1 for p in distribution if p >= threshold))
        dominants.append(int(max(range(len(distribution)), key=lambda k: distribution[k])))
    n_topics = len(valid[0][1])
    max_entropy = math.log2(n_topics) if n_topics > 1 else 0.0

    switches = [dominants[k] != dominants[k - 1] for k in range(1, len(dominants))]
    switch_rate = 100.0 * sum(switches) / len(switches) if switches else None

    last_seen: dict[int, int] = {}
    gaps = []
    for position, topic in enumerate(dominants):
        if topic in last_seen:
            gaps.append(position - last_seen[topic])
        last_seen[topic] = position

    runs = [len(list(group)) for _key, group in itertools.groupby(dominants)]

    usage = Counter(dominants)
    usage_total = sum(usage.values())
    usage_probs = [count / usage_total for count in usage.values()]
    usage_entropy = -sum(p * math.log2(p) for p in usage_probs if p > 0)
    usage_max = math.log2(len(usage)) if len(usage) > 1 else 0.0
    concentration = (1.0 - usage_entropy / usage_max) if usage_max else 1.0

    entropy_summary = summarize(entropies)
    entropy_headline, entropy_method = _resolve_headline(entropies, entropy_summary)
    confidence_summary = summarize(confidences)
    confidence_headline, confidence_method = _resolve_headline(confidences, confidence_summary)
    active_summary = summarize(actives)
    active_headline, active_method = _resolve_headline(actives, active_summary)
    gap_headline, gap_method = ((_resolve_headline(gaps, summarize(gaps))) if gaps else
                               (None, "mean"))
    run_summary = summarize(runs)
    run_headline, run_method = _resolve_headline(runs, run_summary)

    prefix = f"semantic.structure_topic_{model_name}"
    return [
        finding(f"{prefix}_distribution_entropy",
               f"{model_name.upper()} corpus-trained topic distribution entropy",
               entropy_headline, "bits", family=FAMILY, sample_size=n,
               min_sample=MIN_TOPIC_PARAGRAPHS,
               distribution={**base, "n_topics": n_topics, "max_possible_bits": max_entropy,
                            "mean": statistics.fmean(entropies),
                            "zero_share_percent": _zero_share_percent(entropies),
                            "aggregation": f"headline is the {entropy_method} of the per-paragraph "
                                          f"topic-distribution entropy"}),
        finding(f"{prefix}_dominant_confidence",
               f"{model_name.upper()} corpus-trained dominant-topic probability",
               confidence_headline, "probability", family=FAMILY, sample_size=n,
               min_sample=MIN_TOPIC_PARAGRAPHS,
               distribution={**base, "n_topics": n_topics,
                            "zero_share_percent": _zero_share_percent(confidences),
                            "aggregation": f"headline is the {confidence_method} of the "
                                          f"per-paragraph dominant-topic probability"}),
        finding(f"{prefix}_switch_rate",
               f"{model_name.upper()} corpus-trained topic-switch rate between adjacent "
               f"paragraphs", switch_rate, "%", family=FAMILY, sample_size=len(switches),
               min_sample=max(1, MIN_TOPIC_PARAGRAPHS - 1),
               distribution={**base, "n_topics": n_topics,
                            "aggregation": "share of adjacent paragraph pairs whose dominant "
                                          "topic differs (a rate, not a median/mean of samples)"},
               warning=None if switches else
               "fewer than two paragraphs with a valid topic distribution"),
        finding(f"{prefix}_recurrence_interval",
               f"{model_name.upper()} corpus-trained topic recurrence interval",
               gap_headline, "paragraphs", family=FAMILY,
               sample_size=len(gaps), min_sample=max(1, MIN_TOPIC_PARAGRAPHS - 1),
               distribution={**base, "n_topics": n_topics,
                            "zero_share_percent": _zero_share_percent(gaps) if gaps else 0.0,
                            "aggregation": f"headline is the {gap_method} of the paragraph gap "
                                          f"between successive occurrences of the same dominant "
                                          f"topic"},
               warning=None if gaps else "no dominant topic recurred"),
        finding(f"{prefix}_run_length",
               f"{model_name.upper()} corpus-trained dominant-topic persistence (run length)",
               run_headline, "paragraphs", family=FAMILY, sample_size=len(runs),
               min_sample=1,
               distribution={**base, "n_topics": n_topics, "longest_run": max(runs),
                            "zero_share_percent": _zero_share_percent(runs),
                            "aggregation": f"headline is the {run_method} of the dominant-topic "
                                          f"run lengths"}),
        finding(f"{prefix}_active_topics",
               f"{model_name.upper()} corpus-trained active-topic count per paragraph "
               f"(probability >= {threshold:g})", active_headline, "count",
               family=FAMILY, sample_size=n, min_sample=MIN_TOPIC_PARAGRAPHS,
               distribution={**base, "n_topics": n_topics, "active_threshold": threshold,
                            "zero_share_percent": _zero_share_percent(actives),
                            "aggregation": f"headline is the {active_method} of the per-paragraph "
                                          f"active-topic count"}),
        finding(f"{prefix}_concentration",
               f"{model_name.upper()} corpus-trained dominant-topic balance/concentration "
               f"(0=perfectly even topic use across the document, 1=one topic dominates every "
               f"paragraph)", concentration, "ratio", family=FAMILY, sample_size=n,
               min_sample=MIN_TOPIC_PARAGRAPHS,
               distribution={**base, "n_topics": n_topics,
                            "distinct_dominant_topics_used": len(usage),
                            "aggregation": "1 - normalized Shannon entropy of the document-wide "
                                          "dominant-topic usage distribution (a single "
                                          "whole-document statistic, not a median/mean of "
                                          "per-paragraph samples)"}),
    ]


def _topic_model_group(analysis: DocumentAnalysis, config: Mapping[str, Any],
                       profile: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    fit = _fit_topic_models(profile, analysis, config)
    meta, models, vocab = fit["meta"], fit["models"], fit["vocab"]
    if not models:
        reason = meta.get("reason") or "no corpus-trained topic model could be fit"
        out = []
        for model_name in _TOPIC_MODEL_NAMES:
            out.extend(_empty_topic_findings(model_name, reason, meta))
        return out
    word_cap = int(option(config, "topic_profile_paragraph_word_cap", 300))
    paragraph_vectors = _paragraph_term_vectors(analysis.paragraphs, vocab, word_cap)
    out = []
    for model_name, entry in models.items():
        dist_list = _topic_distributions(entry, paragraph_vectors, vocab)
        out.extend(_topic_findings(model_name, entry, meta, dist_list, config))
    for model_name in _TOPIC_MODEL_NAMES:
        if model_name in models:
            continue
        reason = (meta.get(f"{model_name}_error")
                 or meta.get("sklearn_reason" if model_name in ("lda", "nmf") else "tomotopy_reason")
                 or f"{model_name} model unavailable")
        out.extend(_empty_topic_findings(model_name, reason, meta))
    return out


# ------------------------------------------------------------------------- measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    features = _features(config)
    out: list[dict[str, Any]] = []
    rep_sentence_values: dict[str, list[float | None]] = {}
    for rep in REPRESENTATIONS:
        if not features.get(rep, DEFAULT_FEATURES.get(rep, False)):
            continue
        findings, raw_values = _representation_findings(rep, analysis, config)
        out.extend(findings)
        if any(value is not None for value in raw_values):
            rep_sentence_values[rep] = raw_values
    if features.get("topic_models", True):
        out.extend(_topic_model_group(analysis, config, profile))
    if features.get("disagreement", True):
        out.extend(_disagreement_findings(rep_sentence_values, analysis, config))
    return out
