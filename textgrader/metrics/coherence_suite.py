"""Whether sentences and paragraphs talk to one another, not just whether
each one is well-formed on its own.

Every other metric in this codebase asks a question about one unit of text
at a time: is this sentence passive, how long is this paragraph, does this
word repeat.  None of them can see a paragraph whose sentences are each
perfectly fine and collectively make no sense, because "makes no sense
together" is a relationship between units, not a property of one.  This
suite measures four largely independent kinds of relationship:

``lexical``
    Do adjacent sentences and paragraphs share vocabulary, and does a
    repeated keyword hold a thread across several of them (a lexical chain)?
    Dependency-free; this is the layer that always runs.
``semantic``
    Do adjacent sentences and paragraphs mean similar things, reusing the
    embedding cache the ``semantic_repetition`` family already built rather
    than encoding the document a second time.  Degrades to the same
    TF-IDF lexical fallback that family uses, and says so.
``entity``
    Does the same person/place/thing keep coming back, and in what
    grammatical role?  A minimal entity grid: subject, object, other mention,
    or absent, tracked by noun-chunk lemma (surface identity, not
    coreference: see the module docstring in :mod:`textgrader.coherence`).
    Also runs the same grid split by dialogue/narration channel, since that
    costs nothing beyond the one shared-parse walk the full-document grid
    already pays for.  Needs the shared spaCy parse.
``coreference``
    The same entity-grid questions, but backed by real neural coreference
    (``fastcoref``) instead of lemma matching, so "Alice" / "she" / "her"
    resolve to one chain.  **Off by default even when the suite is on**: it
    downloads and runs a transformer model and is the single most expensive
    thing this module can be asked to do, so it never fires unless a caller
    explicitly turns it on (see ``coreference_model``/``coreference_max_words``
    below).  Its findings are separate metric ids from ``entity``'s, not a
    replacement: the two disagreeing is itself evidence, not something to
    hide, so both are always reported once ``coreference`` is on.
``lexical_wordnet``
    A synonym/hypernym-aware lexical chain, alongside (not instead of) the
    identity-based one in ``lexical``.  Off by default: it needs the
    ``nltk`` ``wordnet`` corpus, which is a real download some environments
    do not have, and it makes a heuristic (first-sense) call that the
    identity chain does not have to make.
``connectives`` / ``order_permutation``
    Whether the text marks its own transitions (explicit connectives, by
    relation family, and whether they cluster at paragraph starts), and
    whether the sentences/paragraphs the writer chose actually outperform a
    random shuffle of themselves on a cohesion score.  Both are
    dependency-free.

Every one of those seven groups is switched on or off independently through
``features`` in this metric's config (see ``DEFAULT_FEATURES`` below); the
top-level ``coherence_suite.enabled`` switch on its own turns nothing on,
consistent with every other metric.  Once it is on, the original five groups
(``lexical``, ``semantic``, ``entity``, ``connectives``, ``order_permutation``)
default to on too, so a user opts *out* of the ones that were already
dependency-free or degrade gracefully rather than having to discover and opt
into every one of them.  ``coreference`` and ``lexical_wordnet`` are the
exception: both default to off even then, because both are new, both load an
optional dependency this module did not used to have, and neither is safe to
fire unasked on a 300,000-word novel (:mod:`textgrader.corpus`'s profiler
would never load a coreference model on its own either, since this module's
``cost`` stays ``"parse"`` - see ``COST`` below).

No polarity judgement is made anywhere in this module. High cohesion can be
repetitive prose; low cohesion can be a deliberate montage or a poem. Every
finding here is ``Polarity.NEUTRAL`` by the default the rest of the tool
already applies to optional findings, and nothing in this module aggregates
these numbers into a single "coherence score".

Closed since the first pass (real dependencies became available in this
environment):

* **Real coreference.** ``fastcoref`` is installed and is wired in as the
  ``coreference`` feature (see above): entity continuity can now follow "she"
  back to "Alice" rather than only matching repeated nouns.  It is reported
  as new, separate metric ids alongside the pre-existing surface/lemma ones,
  never in place of them - see :mod:`textgrader.coherence`'s docstring for
  why the disagreement between the two is itself data.  Loading the model
  hit a real version-skew bug between ``fastcoref==2.1.6`` and a newer
  ``transformers``; :func:`textgrader.coherence._shim_transformers_tied_weights`
  documents the one-attribute compatibility shim that works around it, and
  the ordinary ``optional.require``/try-except path still degrades to
  ``unavailable(...)`` if that shim ever stops being enough.
* **Real sentence embeddings.** ``sentence-transformers`` is installed; the
  ``semantic`` group's ``embedding`` backend runs end to end rather than
  falling back to the ``lexical`` TF-IDF proxy, and a finding still says
  honestly which backend actually produced it either way.
* **WordNet-based lexical cohesion.** The ``nltk`` ``wordnet`` corpus is
  present; ``lexical_wordnet`` adds a synonym/hypernym-aware chain next to
  the identity-based one, with the same "both reported, disagreement kept"
  rule as coreference.
* **Per-channel (dialogue vs. narration) entity grids.** Reconsidered now
  that real coreference exists: entity *identity* is resolved once, over the
  whole document (or, for the ``coreference`` backend, over its window - see
  below), so a chain that crosses a quotation mark is not severed by a
  channel split; only *which observations count toward a channel's rate* is
  filtered, the same principle the pre-existing ``lexical``
  narration/dialogue overlap channels already used.  ``entity``'s four
  role-sequence metrics (new/given, reintroduction distance, dangling rate,
  transition entropy) now have narration and dialogue variants.  The
  co-occurrence **graph** and the **order-permutation tests** deliberately do
  not: a co-occurrence window and a shuffle-based arrangement score are both
  questions about the document's overall structure, not about one channel's
  share of a role sequence, and channel-splitting either would answer a
  different, not obviously more useful, question rather than the same one
  more narrowly - inventing that split was judged not worth doing versus
  documenting the boundary here.

Still deferred:

* **RST parsing** (IsaNLP, Feng-Hirst, discopy, DisCoDisCo) and **PDTB
  implicit-relation labels**. No maintained, installable parser exists in
  this environment; faking a relation label or a tree depth from string
  matching would be worse than not reporting it. Only explicit,
  surface-matched connectives are measured. Permanently out of scope.
* **A fitted corpus reference distribution for entity-grid TRANSITION
  FREQUENCIES** (as opposed to their entropy, which is a single number and
  already flows through the existing mechanism below). ``textgrader.corpus``
  turns every scalar ``finding["value"]`` from a profiled metric into a
  per-book column and then a pooled distribution automatically - so
  ``discourse.coherence_entity_grid_transition_entropy``'s entropy *already*
  gets a real corpus reference distribution for free, the day someone builds
  a profile with both ``--parse-metrics`` and ``--model-metrics`` (this
  suite's ``cost="parse"`` and its ``sentence_transformers`` requirement mean
  both flags, not just one, are needed - see ``textgrader/corpus.py``'s
  ``_metric_names``).  What the corpus mechanism genuinely cannot carry,
  without a schema change to ``textgrader/corpus.py`` this task does not
  permit, is the full 16-way transition-frequency *table* itself (S→S, S→O,
  ...): ``build_profile`` only ever extracts one scalar ``value`` per metric
  id into a book's row, never a finding's ``distribution`` dict, and
  ``item_distributions`` only pools the fixed, pre-named list in
  ``ITEM_SOURCES`` (sentence/paragraph/word/turn counts), not an arbitrary
  categorical count table a metric produces at measurement time.  Emitting
  one metric id per transition type to route around that would add sixteen
  near-duplicate, low-value ids to this suite's surface for one narrow
  comparison; that trade was judged not worth it, so the transition counts
  and their entropy are reported as this module already did, and only the
  entropy number, not the table, is corpus-comparable today.
* **Pronoun-to-named-mention transition rate.** ``pov.entity_pronoun_ratio``
  already measures named-entity-to-pronoun balance in narration; adding a
  second, entity-grid-flavoured version of the same comparison here would be
  a restatement, not a new channel.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Mapping, Sequence

from .. import coherence as coh
from .. import text as textlib
from ..document import DocumentAnalysis
from ..optional import require
from .common import (PARSE, cosine_distance, finding, option, rate, shape,
                     summarize, unavailable)
from . import semantic_adjacent as sem

FAMILY = "discourse"
# The suite's cheapest groups (lexical, connectives, order_permutation) are
# dependency-free MODERATE work; entity needs the shared spaCy PARSE, and
# semantic can load a sentence-embedding MODEL. COST records the most
# expensive thing this module can be asked to do; a caller who only wants the
# dependency-free groups still gets the fast path, because grade.py's
# "> 1s" warning is about what actually ran, not what was declared.
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy", "sentence_transformers", "networkx", "fastcoref", "nltk")
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

# Sample-size floors for measurements whose unit isn't "sentences in the
# document" (pairs, paragraphs, words, graph nodes).
MIN_SAMPLE_PAIRS = 9
MIN_SAMPLE_PARAGRAPH_PAIRS = 4
MIN_SAMPLE_PARAGRAPHS = 5
MIN_SAMPLE_WORDS = 200
MIN_SAMPLE_GRAPH_NODES = 3

# features.coreference and features.lexical_wordnet are NOT in this default
# set of "on unless disabled" groups, even though every other group is: both
# load an optional dependency this suite did not use to have, coreference
# runs a transformer model, and the whole point of the gating rule in
# textgrader/corpus.py (needs_model checks REQUIRES for "sentence_transformers"
# only, so a fastcoref-backed feature is invisible to it) is that nothing may
# rely on that guard to keep a model-backed measurement from firing unasked.
# They are opt-in on top of an already-opt-in suite.
DEFAULT_FEATURES: dict[str, bool] = {
    "lexical": True,
    "lexical_wordnet": False,
    "semantic": True,
    "entity": True,
    "coreference": False,
    "connectives": True,
    "order_permutation": True,
}


def _features(config: Mapping[str, Any] | None) -> dict[str, bool]:
    merged = dict(DEFAULT_FEATURES)
    merged.update(option(config, "features", {}) or {})
    return merged


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    features = _features(config)
    out: list[dict[str, Any]] = []
    if features.get("lexical", True):
        out.extend(_lexical(analysis, config))
    if features.get("lexical_wordnet", False):
        out.append(_lexical_chain_coverage_wordnet(analysis, config))
    if features.get("semantic", True):
        out.extend(_semantic(analysis, config))
    if features.get("connectives", True):
        out.extend(_connectives(analysis, config))
    if features.get("entity", True):
        out.extend(_entity(analysis, config))
    if features.get("coreference", False):
        out.extend(_coreference_entity(analysis, config))
    if features.get("order_permutation", True):
        out.extend(_order(analysis, config))
    return out


# --------------------------------------------------------------- lexical group

def _overlap_evidence(units: list[str], values: list[float], limit: int = 15
                      ) -> list[dict[str, Any]]:
    ranked = sorted(range(len(values)), key=lambda i: -values[i])[:limit]
    return [{"index": i, "overlap": values[i], "unit_a": sem.truncate(units[i]),
             "unit_b": sem.truncate(units[i + 1])} for i in ranked]


def _overlap_finding(sentences: list[str], min_len: int, metric_id: str, name: str,
                     channel: str) -> dict[str, Any]:
    if len(sentences) < 2:
        return finding(metric_id, name, None, "jaccard", family=FAMILY, channel=channel,
                       sample_size=len(sentences), min_sample=MIN_SAMPLE_PAIRS,
                       warning=f"needs at least two sentences; this channel has {len(sentences)}")
    values = coh.adjacent_overlap(sentences, min_len)
    if not values:
        return finding(metric_id, name, None, "jaccard", family=FAMILY, channel=channel,
                       sample_size=0, min_sample=MIN_SAMPLE_PAIRS,
                       warning="no adjacent sentence pair had a content word on either side")
    return shape(metric_id, name, values, "jaccard", family=FAMILY, channel=channel,
                min_sample=MIN_SAMPLE_PAIRS, evidence=_overlap_evidence(sentences, values))[0]


def _adjacent_sentence_overlap(analysis: DocumentAnalysis, min_len: int) -> list[dict[str, Any]]:
    channels = (
        (analysis, "discourse.coherence_adjacent_sentence_overlap",
         "Adjacent-sentence content-word overlap", "full"),
        (analysis.narration, "discourse.coherence_adjacent_sentence_overlap_narration",
         "Adjacent-sentence content-word overlap (narration only)", "narration"),
        (analysis.dialogue, "discourse.coherence_adjacent_sentence_overlap_dialogue",
         "Adjacent-sentence content-word overlap (dialogue turns only, in speaking order; a "
         "turn may follow a different speaker's turn)", "dialogue"),
    )
    return [_overlap_finding(view.sentences, min_len, metric_id, name, channel)
            for view, metric_id, name, channel in channels]


def _adjacent_paragraph_overlap(analysis: DocumentAnalysis, min_len: int) -> dict[str, Any]:
    metric_id = "discourse.coherence_adjacent_paragraph_overlap"
    name = "Adjacent-paragraph content-word overlap"
    paragraphs = analysis.paragraphs
    if len(paragraphs) < 2:
        return finding(metric_id, name, None, "jaccard", family=FAMILY,
                       sample_size=len(paragraphs), min_sample=MIN_SAMPLE_PARAGRAPH_PAIRS,
                       warning=f"needs at least two paragraphs; this text has {len(paragraphs)}")
    values = coh.adjacent_overlap(paragraphs, min_len)
    if not values:
        return finding(metric_id, name, None, "jaccard", family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE_PARAGRAPH_PAIRS,
                       warning="no adjacent paragraph pair had a content word on either side")
    return shape(metric_id, name, values, "jaccard", family=FAMILY,
                min_sample=MIN_SAMPLE_PARAGRAPH_PAIRS,
                evidence=_overlap_evidence(paragraphs, values))[0]


def _lexical_chain_coverage(analysis: DocumentAnalysis, config: Mapping[str, Any],
                            min_len: int) -> dict[str, Any]:
    metric_id = "discourse.coherence_lexical_chain_coverage"
    name = "Sentence share covered by a repeated-keyword lexical chain"
    gap = int(option(config, "chain_gap", 3))
    min_chain_len = int(option(config, "chain_min_length", 2))
    sentences = analysis.sentences
    if not sentences:
        return finding(metric_id, name, None, "%", family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE, warning="no sentences to measure")
    chains = coh.lexical_chains(sentences, gap=gap, min_len=min_len)
    real_chains = [chain for chain in chains if len(chain) >= min_chain_len]
    covered: set[int] = set()
    for chain in real_chains:
        covered.update(chain)
    coverage = 100.0 * len(covered) / len(sentences)
    lengths = [len(chain) for chain in real_chains]
    ranked = sorted(real_chains, key=len, reverse=True)[:20]
    return finding(
        metric_id, name, coverage, "%", family=FAMILY, sample_size=len(sentences),
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"backend": "identity", "chain_count": len(real_chains),
                     "mean_chain_length": (sum(lengths) / len(lengths)) if lengths else None,
                     "longest_chain": max(lengths) if lengths else 0,
                     "gap": gap, "min_chain_length": min_chain_len},
        evidence=[{"length": len(chain), "first_sentence_index": chain[0],
                  "last_sentence_index": chain[-1]} for chain in ranked],
        warning=None if real_chains else
        "no repeated content word formed a chain of the minimum length")


def _global_context_overlap(analysis: DocumentAnalysis, min_len: int) -> dict[str, Any]:
    metric_id = "discourse.coherence_global_context_overlap"
    name = "Sentence similarity to the document's own lexical centroid"
    sentences = analysis.sentences
    if not sentences:
        return finding(metric_id, name, None, "cosine", family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE, warning="no sentences to measure")
    vectors = sem.lexical_vectors(sentences)
    centroid: dict[str, float] = {}
    for vector in vectors:
        for word, weight in vector.items():
            centroid[word] = centroid.get(word, 0.0) + weight
    if vectors:
        centroid = {word: value / len(vectors) for word, value in centroid.items()}
    values = []
    for vector in vectors:
        distance = cosine_distance(vector, centroid)
        if distance is not None:
            values.append(1.0 - distance)
    if not values:
        return finding(metric_id, name, None, "cosine", family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE,
                       warning="no sentence had a content word to compare against the "
                               "document's own centroid")
    result = shape(metric_id, name, values, "cosine", family=FAMILY, min_sample=MIN_SAMPLE)[0]
    result["warning"] = ("this is a TF-IDF lexical-overlap proxy for topical drift from the "
                         "document's own vocabulary centroid, not a semantic distance; see the "
                         "'semantic' feature group for the embedding-backed measurement")
    return result


def _lexical(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    min_len = int(option(config, "min_word_len", 3))
    out = list(_adjacent_sentence_overlap(analysis, min_len))
    out.append(_adjacent_paragraph_overlap(analysis, min_len))
    out.append(_lexical_chain_coverage(analysis, config, min_len))
    out.append(_global_context_overlap(analysis, min_len))
    return out


def _lexical_chain_coverage_wordnet(analysis: DocumentAnalysis,
                                    config: Mapping[str, Any]) -> dict[str, Any]:
    """The ``lexical_wordnet`` feature: :func:`_lexical_chain_coverage`'s
    identity-based chain, rerun with a WordNet synonym/hypernym key instead
    of the word itself, and reported as its OWN finding rather than replacing
    the identity one - see ``coh.wordnet_concept_key``'s docstring for why
    the two are expected to disagree, not just permitted to."""

    metric_id = "discourse.coherence_lexical_chain_coverage_wordnet"
    name = "Sentence share covered by a WordNet synonym/hypernym lexical chain"
    min_len = int(option(config, "min_word_len", 3))
    gap = int(option(config, "chain_gap", 3))
    min_chain_len = int(option(config, "chain_min_length", 2))
    sentences = analysis.sentences
    if not sentences:
        return finding(metric_id, name, None, "%", family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE, warning="no sentences to measure")
    wn, reason = analysis.memo("coherence_wordnet", coh.require_wordnet)
    if wn is None:
        return unavailable(metric_id, name, reason, family=FAMILY)
    chains = coh.lexical_chains(sentences, gap=gap, min_len=min_len,
                                key_fn=lambda word: coh.wordnet_concept_key(word, wn))
    real_chains = [chain for chain in chains if len(chain) >= min_chain_len]
    covered: set[int] = set()
    for chain in real_chains:
        covered.update(chain)
    coverage = 100.0 * len(covered) / len(sentences)
    lengths = [len(chain) for chain in real_chains]
    ranked = sorted(real_chains, key=len, reverse=True)[:20]
    return finding(
        metric_id, name, coverage, "%", family=FAMILY, sample_size=len(sentences),
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"backend": "wordnet", "chain_count": len(real_chains),
                     "mean_chain_length": (sum(lengths) / len(lengths)) if lengths else None,
                     "longest_chain": max(lengths) if lengths else 0,
                     "gap": gap, "min_chain_length": min_chain_len,
                     "word_sense_disambiguation": "none (first WordNet sense only)"},
        evidence=[{"length": len(chain), "first_sentence_index": chain[0],
                  "last_sentence_index": chain[-1]} for chain in ranked],
        warning=None if real_chains else
        "no shared or synonymous content word formed a chain of the minimum length")


# -------------------------------------------------------------- semantic group

def _local_semantic_cohesion(analysis: DocumentAnalysis, model_name: str,
                             low_tail: float) -> dict[str, Any]:
    metric_id = "discourse.coherence_local_semantic_cohesion"
    name = "Adjacent-sentence semantic similarity: shape and low-tail rate"
    sentences = analysis.sentences
    pairs = max(0, len(sentences) - 1)
    if pairs < 1:
        return finding(metric_id, name, None, "cosine", family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE_PAIRS,
                       warning=f"needs at least two sentences; this text has {len(sentences)}")
    backend, vectors, note = sem.get_sentence_vectors(analysis, model_name)
    values = [sem.similarity_at(backend, vectors, i, i + 1) for i in range(pairs)]
    summary = summarize(values)
    low_share = 100.0 * sum(1 for value in values if value <= low_tail) / len(values)
    summary.update({"variance": (summary.get("std") or 0.0) ** 2,
                    "low_tail_rate_percent": low_share, "low_tail_threshold": low_tail,
                    "backend": backend, "model": model_name})
    return finding(metric_id, name, summary.get("median"), "cosine", family=FAMILY,
                  sample_size=pairs, min_sample=MIN_SAMPLE_PAIRS, distribution=summary,
                  warning=note)


def _paragraph_semantic_transition(analysis: DocumentAnalysis, model_name: str,
                                   low_tail: float) -> dict[str, Any]:
    metric_id = "discourse.coherence_paragraph_semantic_transition"
    name = "Paragraph-to-paragraph semantic similarity: shape and low-tail rate"
    paragraphs = analysis.paragraphs
    pairs = max(0, len(paragraphs) - 1)
    if pairs < 1:
        return finding(metric_id, name, None, "cosine", family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE_PARAGRAPH_PAIRS,
                       warning=f"needs at least two paragraphs; this text has {len(paragraphs)}")
    backend, vectors, note = sem.get_paragraph_vectors(analysis, model_name)
    values = [sem.similarity_at(backend, vectors, i, i + 1) for i in range(pairs)]
    summary = summarize(values)
    low_share = 100.0 * sum(1 for value in values if value <= low_tail) / len(values)
    summary.update({"variance": (summary.get("std") or 0.0) ** 2,
                    "low_tail_rate_percent": low_share, "low_tail_threshold": low_tail,
                    "backend": backend, "model": model_name})
    return finding(metric_id, name, summary.get("median"), "cosine", family=FAMILY,
                  sample_size=pairs, min_sample=MIN_SAMPLE_PARAGRAPH_PAIRS, distribution=summary,
                  warning=note)


def _semantic(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    model_name = option(config, "semantic_model", sem.DEFAULT_MODEL)
    low_tail = float(option(config, "semantic_low_tail_threshold", 0.15))
    return [_local_semantic_cohesion(analysis, model_name, low_tail),
            _paragraph_semantic_transition(analysis, model_name, low_tail)]


# ---------------------------------------------------------- connectives group

# Anywhere-in-sentence, closed-class markers by PDTB-style relation family.
# Deliberately small per family (surface matching, not sense disambiguation:
# see discourse_causal's docstring for why "since"/"so"/"still" have readings
# this cannot tell apart from their connective one). The point of this table
# is the family aggregation and the paragraph-boundary comparison below, not
# an exhaustive connective inventory: discourse_connectives.py and
# discourse_causal.py already cover sentence-initial and causal-only ground.
FAMILY_MARKERS: dict[str, tuple[tuple[str, ...], ...]] = {
    "causal": (("because",), ("as", "a", "result"), ("due", "to"), ("owing", "to"),
              ("so", "that")),
    "contrastive": (("but",), ("however",), ("yet",), ("on", "the", "other", "hand"),
                    ("whereas",), ("conversely",)),
    "temporal": (("meanwhile",), ("afterward",), ("afterwards",), ("beforehand",),
                ("eventually",), ("subsequently",)),
    "additive": (("also",), ("moreover",), ("furthermore",), ("in", "addition"),
                ("besides",), ("additionally",)),
    "conditional": (("unless",), ("provided", "that"), ("as", "long", "as"),
                    ("even", "if",)),
    "exemplification": (("for", "example"), ("for", "instance"), ("such", "as"),
                        ("namely",)),
    "conclusion": (("therefore",), ("thus",), ("in", "conclusion"), ("consequently",),
                  ("hence",)),
    "elaboration": (("in", "other", "words"), ("that", "is",), ("specifically",),
                    ("in", "particular")),
    "concession": (("although",), ("even", "though"), ("despite",), ("nonetheless",),
                  ("regardless",)),
}


def _build_family_index() -> dict[str, list[tuple[str, tuple[str, ...]]]]:
    index: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for family, phrases in FAMILY_MARKERS.items():
        for phrase in phrases:
            index.setdefault(phrase[0], []).append((family, phrase))
    return index


_FAMILY_INDEX = _build_family_index()


def _count_family_matches(tokens: list[str]) -> Counter:
    """One counted match per family per starting position (not per family per
    sentence), so a sentence with two causal markers counts twice. Indexed by
    first word so this is linear in sentence length rather than
    families x phrases x length."""

    counts: Counter = Counter()
    for index, word in enumerate(tokens):
        candidates = _FAMILY_INDEX.get(word)
        if not candidates:
            continue
        for family, phrase in candidates:
            if tokens[index:index + len(phrase)] == list(phrase):
                counts[family] += 1
    return counts


def _paragraph_initial_flags(analysis: DocumentAnalysis) -> list[bool]:
    flags: list[bool] = []
    for group in analysis.sentences_by_paragraph:
        flags.extend(index == 0 for index in range(len(group)))
    return flags


def _connectives(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    max_reported = int(option(config, "connective_max_reported", 25))
    family_id = "discourse.coherence_connective_family_rate"
    family_name = ("Explicit discourse-connective rate by relation family (matched anywhere in "
                  "the sentence)")
    boundary_id = "discourse.coherence_paragraph_boundary_connective_ratio"
    boundary_name = ("Ratio of connective rate at paragraph-initial sentences to the rest "
                     "(whether transitions are marked at structural boundaries)")

    sentences = analysis.sentences
    words_total = analysis.word_count
    if not sentences or not words_total:
        warning = "no sentences to measure" if not sentences else "no words in text"
        return [
            finding(family_id, family_name, None, "per 1,000 words", family=FAMILY,
                   sample_size=0, min_sample=MIN_SAMPLE_WORDS, warning=warning),
            finding(boundary_id, boundary_name, None, "ratio", family=FAMILY, sample_size=0,
                   min_sample=MIN_SAMPLE_PARAGRAPHS, warning=warning),
        ]

    flags = _paragraph_initial_flags(analysis)
    family_counts: Counter = Counter()
    initial_hits = initial_sentences = other_hits = other_sentences = 0
    for sentence, is_initial in zip(sentences, flags):
        tokens = [word.lower() for word in textlib.words(sentence)]
        hits = _count_family_matches(tokens)
        family_counts.update(hits)
        total_hits = sum(hits.values())
        if is_initial:
            initial_sentences += 1
            initial_hits += total_hits
        else:
            other_sentences += 1
            other_hits += total_hits

    total_matches = sum(family_counts.values())
    family_finding = finding(
        family_id, family_name, rate(total_matches, words_total, 1000.0), "per 1,000 words",
        family=FAMILY, sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        distribution={"family_counts": dict(family_counts),
                     "family_rate_per_1000_words": {family: rate(count, words_total, 1000.0)
                                                    for family, count in family_counts.items()}},
        evidence=[{"family": family, "count": count}
                 for family, count in family_counts.most_common(max_reported)],
        warning=None if total_matches else "no family connective matched")

    initial_rate = rate(initial_hits, initial_sentences, 100.0) if initial_sentences else None
    other_rate = rate(other_hits, other_sentences, 100.0) if other_sentences else None
    ratio = (initial_rate / other_rate
            if initial_rate is not None and other_rate else None)
    boundary_finding = finding(
        boundary_id, boundary_name, ratio, "ratio", family=FAMILY,
        sample_size=initial_sentences, min_sample=MIN_SAMPLE_PARAGRAPHS,
        distribution={"paragraph_initial_rate_per_100_sentences": initial_rate,
                     "other_sentence_rate_per_100_sentences": other_rate,
                     "paragraph_initial_sentences": initial_sentences,
                     "other_sentences": other_sentences},
        warning=None if ratio is not None else
        "a connective family matched in only one of the two groups (or neither); the ratio is "
        "undefined")
    return [family_finding, boundary_finding]


# ---------------------------------------------------------------- entity group

# One (metric_id, name) pair per role-sequence measurement, per channel. The
# base ("") set is the pre-existing full-document surface metrics whose ids
# must not move (see the module contract); "_narration"/"_dialogue" are the
# new per-channel surface variants (see the docstring's "Closed since the
# first pass" section for why a channel split is now judged safe); "_coref"
# is the new real-coreference backend, windowed rather than channel-split
# (see coh.resolve_coreference's docstring for why).
_GRID_METRIC_STEMS = (
    ("entity_new_given_ratio", "New-vs-given entity mention ratio"),
    ("entity_reintroduction_distance",
     "Distance in sentences between repeated mentions of the same entity"),
    ("entity_dangling_rate", "Entities introduced once and never mentioned again"),
    ("entity_grid_transition_entropy",
     "Entropy of adjacent subject/object/other/absent entity-grid transitions"),
)
_GRAPH_METRIC_STEM = ("entity_graph_density", "Density of the entity co-occurrence graph")

_CHANNEL_LABELS = {"": "", "narration": " (narration only)",
                   "dialogue": " (dialogue turns only, in speaking order)",
                   "coref": " (real coreference, fastcoref-backed, windowed)"}


def _grid_ids(suffix: str) -> tuple[tuple[str, str], ...]:
    tag = f"_{suffix}" if suffix else ""
    label = _CHANNEL_LABELS[suffix]
    return tuple((f"discourse.coherence_{stem}{tag}", f"{name}{label}")
                 for stem, name in _GRID_METRIC_STEMS)


def _graph_ids(suffix: str) -> tuple[str, str]:
    tag = f"_{suffix}" if suffix else ""
    stem, name = _GRAPH_METRIC_STEM
    return f"discourse.coherence_{stem}{tag}", f"{name}{_CHANNEL_LABELS[suffix]}"


def _entity_given_new(metric_id: str, name: str, per_sentence: Sequence[Mapping[str, str]],
                      total_sentences: int, *, channel: str, backend: Mapping[str, Any],
                      no_mentions_reason: str) -> dict[str, Any]:
    seen: set[str] = set()
    new_count = given_count = carried_over = empty_sentences = 0
    prev_keys: set[str] = set()
    for roles in per_sentence:
        if not roles:
            empty_sentences += 1
        current_keys = set(roles)
        carried_over += len(current_keys & prev_keys)
        for key in current_keys:
            if key in seen:
                given_count += 1
            else:
                new_count += 1
                seen.add(key)
        prev_keys = current_keys
    if new_count == 0:
        warning = no_mentions_reason
    elif given_count == 0:
        warning = "every mention was a first mention of its entity; the ratio is undefined"
    else:
        warning = None
    ratio = (new_count / given_count) if given_count else None
    return finding(metric_id, name, ratio, "ratio", family=FAMILY, sample_size=total_sentences,
                  min_sample=MIN_SAMPLE, channel=(channel or "full"),
                  distribution={**backend, "new_mentions": new_count, "given_mentions": given_count,
                               "carried_over_from_previous_sentence": carried_over,
                               "sentences_with_no_tracked_entity": empty_sentences,
                               "distinct_entities": len(seen)},
                  warning=warning)


def _entity_reintroduction(metric_id: str, name: str, per_sentence: Sequence[Mapping[str, str]],
                           *, channel: str, backend: Mapping[str, Any]) -> dict[str, Any]:
    last_seen: dict[str, int] = {}
    gaps: list[int] = []
    for index, roles in enumerate(per_sentence):
        for key in roles:
            if key in last_seen:
                gaps.append(index - last_seen[key])
            last_seen[key] = index
    if not gaps:
        return finding(metric_id, name, None, "sentences", family=FAMILY, sample_size=0,
                      min_sample=MIN_SAMPLE, channel=(channel or "full"),
                      distribution=dict(backend) if backend else None,
                      warning="no entity was mentioned more than once")
    result = shape(metric_id, name, gaps, "sentences", family=FAMILY, min_sample=MIN_SAMPLE,
                   channel=(channel or "full"),
                   evidence=[{"gap_sentences": gap} for gap in sorted(gaps, reverse=True)[:20]])[0]
    if backend:
        result["distribution"] = {**backend, **(result["distribution"] or {})}
    return result


def _entity_dangling(metric_id: str, name: str, per_sentence: Sequence[Mapping[str, str]],
                     total_sentences: int, lookback: int, *, channel: str,
                     backend: Mapping[str, Any]) -> dict[str, Any]:
    first_seen: dict[str, int] = {}
    counts: dict[str, int] = {}
    for index, roles in enumerate(per_sentence):
        for key in roles:
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, index)
    # An entity first introduced near the very end of the document has not
    # had a fair chance to come back, so it is excluded from the denominator
    # rather than scored as dangling for running out of text.
    eligible = [key for key, index in first_seen.items() if index < total_sentences - lookback]
    if not eligible:
        return finding(metric_id, name, None, "%", family=FAMILY, sample_size=0,
                      min_sample=MIN_SAMPLE, channel=(channel or "full"),
                      distribution=dict(backend) if backend else None,
                      warning=f"every tracked entity was introduced in the last {lookback} "
                              f"sentences, too close to the end to judge whether it dangles")
    dangling = [key for key in eligible if counts[key] == 1]
    return finding(
        metric_id, name, 100.0 * len(dangling) / len(eligible), "%", family=FAMILY,
        sample_size=len(eligible), min_sample=MIN_SAMPLE, channel=(channel or "full"),
        distribution={**backend, "dangling_entities": len(dangling),
                     "eligible_entities": len(eligible), "lookback_sentences": lookback},
        evidence=[{"entity": key, "first_sentence_index": first_seen[key]}
                 for key in sorted(dangling, key=lambda item: first_seen[item])[:25]])


def _entity_transition_entropy(metric_id: str, name: str, rows: Mapping[str, Sequence[str]],
                               tracked: Sequence[str], *, channel: str,
                               backend: Mapping[str, Any]) -> dict[str, Any]:
    if not tracked:
        return finding(metric_id, name, None, "bits", family=FAMILY, sample_size=0,
                      min_sample=MIN_SAMPLE, channel=(channel or "full"),
                      distribution=dict(backend) if backend else None,
                      warning="no entity was mentioned in two or more sentences")
    counts = coh.transition_counts(rows)
    total_transitions = sum(counts.values())
    entropy = coh.entropy_of_counts(counts)
    return finding(
        metric_id, name, entropy, "bits", family=FAMILY, sample_size=total_transitions,
        min_sample=MIN_SAMPLE, sample_size_sensitive=True, channel=(channel or "full"),
        distribution={**backend, "role_schema": coh.ROLE_SCHEMA_VERSION,
                     "tracked_entities": len(tracked), "possible_transition_types": 16,
                     "max_possible_bits": math.log2(min(16, len(counts))) if counts else 0.0},
        evidence=[{"from": a, "to": b, "count": count}
                 for (a, b), count in sorted(counts.items(), key=lambda item: -item[1])[:25]],
        warning=None if total_transitions else "no adjacent-sentence transition to measure")


def _entity_graph(metric_id: str, name: str, rows: Mapping[str, Sequence[str]],
                  tracked: Sequence[str], window: int, min_mentions: int,
                  freq: Mapping[str, int], backend: Mapping[str, Any]) -> dict[str, Any]:
    module, reason = require("networkx")
    if module is None:
        return unavailable(metric_id, name, reason, family=FAMILY)
    graph_tracked = [key for key in tracked if freq.get(key, 0) >= min_mentions]
    if len(graph_tracked) < MIN_SAMPLE_GRAPH_NODES:
        return finding(metric_id, name, None, "ratio", family=FAMILY,
                      sample_size=len(graph_tracked), min_sample=MIN_SAMPLE_GRAPH_NODES,
                      distribution=dict(backend) if backend else None,
                      warning=f"needs at least {MIN_SAMPLE_GRAPH_NODES} entities mentioned "
                              f"{min_mentions}+ times each; found {len(graph_tracked)}")
    sub_rows = {key: rows[key] for key in graph_tracked}
    graph = coh.build_entity_graph(module, sub_rows, window)
    stats = coh.graph_stats(module, graph)
    return finding(
        metric_id, name, stats["density"], "ratio", family=FAMILY, sample_size=stats["nodes"],
        min_sample=MIN_SAMPLE_GRAPH_NODES,
        distribution={**backend, **stats, "co_occurrence_window_sentences": window,
                     "min_mentions_to_track": min_mentions,
                     "networkx_version": getattr(module, "__version__", None)},
        warning=None if stats["edges"] else "no two entities ever co-occurred within the window")


def _grid_findings(suffix: str, per_sentence: Sequence[Mapping[str, str]], total_sentences: int,
                   lookback: int, max_tracked: int, *, channel: str,
                   backend: Mapping[str, Any], no_mentions_reason: str) -> list[dict[str, Any]]:
    """The four role-sequence metrics (given/new, reintroduction, dangling,
    transition entropy) for one (channel, backend) combination; the shared
    body behind every surface, per-channel and coreference entity finding."""

    ids = _grid_ids(suffix)
    freq = coh.entity_frequency(per_sentence)
    tracked = [key for key, _ in sorted(freq.items(), key=lambda item: -item[1])[:max_tracked]]
    rows = coh.grid_rows(per_sentence, tracked)
    return [
        _entity_given_new(*ids[0], per_sentence, total_sentences, channel=channel,
                          backend=backend, no_mentions_reason=no_mentions_reason),
        _entity_reintroduction(*ids[1], per_sentence, channel=channel, backend=backend),
        _entity_dangling(*ids[2], per_sentence, total_sentences, lookback, channel=channel,
                         backend=backend),
        _entity_transition_entropy(*ids[3], rows, tracked, channel=channel, backend=backend),
    ]


_SURFACE_BACKEND = {"backend": "surface_lemma",
                    "entity_identity": "noun-chunk root lemma, case-folded "
                                       "(surface-based, not coreference)"}
_SURFACE_NO_MENTIONS = "no entity mentions found (no noun chunk was headed by a noun or proper noun)"


def _entity(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    all_ids = [pair for suffix in ("", "narration", "dialogue") for pair in _grid_ids(suffix)]
    all_ids.append(_graph_ids(""))
    reason = analysis.nlp_unavailable
    if reason:
        return [unavailable(metric_id, name, reason, family=FAMILY) for metric_id, name in all_ids]

    lookback = int(option(config, "entity_lookback_sentences", 10))
    max_tracked = int(option(config, "entity_max_tracked", 150))
    graph_window = int(option(config, "entity_graph_window_sentences", 3))
    min_mentions_for_graph = int(option(config, "entity_min_mentions_for_graph", 2))

    per_sentence, channels = analysis.memo(
        "coherence_entity_mentions",
        lambda: coh.entity_mentions_by_sentence(analysis.spacy_sents_by_channel()))
    total_sentences = len(per_sentence)
    if total_sentences == 0:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE, warning="the shared parse produced no sentences")
               for metric_id, name in all_ids]

    out = list(_grid_findings("", per_sentence, total_sentences, lookback, max_tracked,
                              channel="full", backend=_SURFACE_BACKEND,
                              no_mentions_reason=_SURFACE_NO_MENTIONS))

    for channel_name in ("narration", "dialogue"):
        indices = coh.channel_indices(channels, channel_name)
        channel_rows = coh.project_rows(per_sentence, indices)
        out.extend(_grid_findings(channel_name, channel_rows, len(channel_rows), lookback,
                                  max_tracked, channel=channel_name, backend=_SURFACE_BACKEND,
                                  no_mentions_reason=_SURFACE_NO_MENTIONS))

    freq = coh.entity_frequency(per_sentence)
    tracked = [key for key, _ in sorted(freq.items(), key=lambda item: -item[1])[:max_tracked]]
    rows = coh.grid_rows(per_sentence, tracked)
    graph_id, graph_name = _graph_ids("")
    out.append(_entity_graph(graph_id, graph_name, rows, tracked, graph_window,
                             min_mentions_for_graph, freq, _SURFACE_BACKEND))
    return out


def _coreference_entity(analysis: DocumentAnalysis, config: Mapping[str, Any]
                        ) -> list[dict[str, Any]]:
    """The ``coreference`` feature: the same four role-sequence metrics as
    ``_entity``, real-coreference-backed via ``fastcoref``, plus its own
    co-occurrence graph - all under distinct, ``_coref``-suffixed metric ids,
    never replacing the surface ones.  Off by default; see the module
    docstring and ``coh.resolve_coreference`` for why and how it is bounded.
    """

    ids = list(_grid_ids("coref"))
    graph_id = _graph_ids("coref")
    all_ids = ids + [graph_id]
    reason = analysis.nlp_unavailable
    if reason:
        return [unavailable(metric_id, name, reason, family=FAMILY) for metric_id, name in all_ids]

    model_name = option(config, "coreference_model", coh.DEFAULT_COREF_MODEL)
    max_words = int(option(config, "coreference_max_words", 4000))
    lookback = int(option(config, "entity_lookback_sentences", 10))
    max_tracked = int(option(config, "entity_max_tracked", 150))
    graph_window = int(option(config, "entity_graph_window_sentences", 3))
    min_mentions_for_graph = int(option(config, "entity_min_mentions_for_graph", 2))

    per_sentence, _channels, settings, note = analysis.memo(
        "coherence_coref_chains",
        lambda: coh.resolve_coreference(analysis, model_name, max_words))
    if not per_sentence:
        return [unavailable(metric_id, name, note or "coreference produced no result",
                           family=FAMILY) for metric_id, name in all_ids]

    total_sentences = len(per_sentence)
    out = list(_grid_findings("coref", per_sentence, total_sentences, lookback, max_tracked,
                              channel="full", backend=settings,
                              no_mentions_reason="fastcoref found no chain of two or more "
                                                 "mentions in the window"))
    for item in out:
        # The window note belongs on every coreference finding, exactly like
        # semantic_adjacent's backend_note is on every semantic finding, so a
        # reader never has to cross-reference which metric explains the cap.
        item["warning"] = note if not item["warning"] else f"{note}; {item['warning']}"

    freq = coh.entity_frequency(per_sentence)
    tracked = [key for key, _ in sorted(freq.items(), key=lambda item: -item[1])[:max_tracked]]
    rows = coh.grid_rows(per_sentence, tracked)
    graph_finding = _entity_graph(graph_id[0], graph_id[1], rows, tracked, graph_window,
                                  min_mentions_for_graph, freq, settings)
    if not graph_finding["warning"]:
        graph_finding["warning"] = note
    out.append(graph_finding)
    return out


# ----------------------------------------------------------- permutation group

def _sentence_order(analysis: DocumentAnalysis, config: Mapping[str, Any], rng: random.Random,
                    permutations: int, min_len: int) -> dict[str, Any]:
    metric_id = "discourse.coherence_sentence_order_percentile"
    name = "Percentile of the real sentence order among random within-paragraph reshuffles"
    min_sentences = int(option(config, "order_min_sentences_per_paragraph", 4))
    max_sentences = int(option(config, "order_max_sentences_per_paragraph", 40))
    max_paragraphs = int(option(config, "order_max_paragraphs_sampled", 30))

    candidates = [group for group in analysis.sentences_by_paragraph
                 if min_sentences <= len(group) <= max_sentences]
    sampled_note = None
    if len(candidates) > max_paragraphs:
        candidates = rng.sample(candidates, max_paragraphs)
        sampled_note = f"sampled {max_paragraphs} of the eligible paragraphs"
    if not candidates:
        return finding(metric_id, name, None, "percentile", family=FAMILY, sample_size=0,
                      min_sample=MIN_SAMPLE_PARAGRAPHS,
                      warning=f"no paragraph has between {min_sentences} and {max_sentences} "
                              f"sentences")

    percentiles = []
    for group in candidates:
        percentile, _ = coh.permutation_percentile(
            group, lambda items: coh.order_score_from_overlap(items, min_len),
            permutations, rng)
        if percentile is not None:
            percentiles.append(percentile)
    if not percentiles:
        return finding(metric_id, name, None, "percentile", family=FAMILY, sample_size=0,
                      min_sample=MIN_SAMPLE_PARAGRAPHS,
                      warning="no eligible paragraph produced a permutation score")

    thresholds = (50, 75, 90, 95)
    result = shape(metric_id, name, percentiles, "percentile", family=FAMILY,
                  min_sample=MIN_SAMPLE_PARAGRAPHS)[0]
    result["distribution"] = {
        **(result["distribution"] or {}),
        "paragraphs_tested": len(candidates), "permutations_per_paragraph": permutations,
        "scoring_function": "sum of adjacent content-word Jaccard overlap",
        "share_of_paragraphs_beating_threshold_percent": {
            threshold: 100.0 * sum(1 for value in percentiles if value >= threshold)
                      / len(percentiles) for threshold in thresholds},
    }
    if sampled_note:
        result["warning"] = sampled_note
    return result


def _paragraph_order(analysis: DocumentAnalysis, config: Mapping[str, Any], rng: random.Random,
                     permutations: int, min_len: int, seed: int) -> dict[str, Any]:
    metric_id = "discourse.coherence_paragraph_order_percentile"
    name = "Percentile of the real paragraph order among random whole-document reshuffles"
    cap = int(option(config, "order_max_paragraphs_for_doc", 60))
    paragraphs = analysis.paragraphs
    if len(paragraphs) < 2:
        return finding(metric_id, name, None, "percentile", family=FAMILY,
                      sample_size=len(paragraphs), min_sample=MIN_SAMPLE_PARAGRAPHS,
                      warning=f"needs at least two paragraphs to permute; this text has "
                              f"{len(paragraphs)}")

    warning = None
    window = paragraphs
    if len(paragraphs) > cap:
        start = rng.randrange(0, len(paragraphs) - cap + 1)
        window = paragraphs[start:start + cap]
        warning = (f"document has {len(paragraphs)} paragraphs; sampled a contiguous "
                  f"{cap}-paragraph window starting at paragraph {start} to bound permutation "
                  f"cost")

    percentile, real_score = coh.permutation_percentile(
        window, lambda items: coh.order_score_from_overlap(items, min_len), permutations, rng)
    return finding(
        metric_id, name, percentile, "percentile", family=FAMILY, sample_size=len(window),
        min_sample=MIN_SAMPLE_PARAGRAPHS,
        distribution={"paragraphs_tested": len(window), "permutations": permutations,
                     "seed": seed, "real_score": real_score,
                     "scoring_function": "sum of adjacent content-word Jaccard overlap"},
        warning=warning)


def _order(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    permutations = int(option(config, "permutations", 50))
    seed = int(option(config, "seed", 0))
    min_len = int(option(config, "min_word_len", 3))
    return [
        _sentence_order(analysis, config, random.Random(seed), permutations, min_len),
        _paragraph_order(analysis, config, random.Random(seed + 1), permutations, min_len, seed),
    ]
