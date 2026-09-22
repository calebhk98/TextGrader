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
    grammatical role?  A minimal, honestly-labelled entity grid: subject,
    object, other mention, or absent, tracked by noun-chunk lemma because no
    coreference model is available in this environment (see the module
    docstring in :mod:`textgrader.coherence` for why that is not
    coreference).  Needs the shared spaCy parse.
``connectives`` / ``order_permutation``
    Whether the text marks its own transitions (explicit connectives, by
    relation family, and whether they cluster at paragraph starts), and
    whether the sentences/paragraphs the writer chose actually outperform a
    random shuffle of themselves on a cohesion score.  Both are
    dependency-free.

Every one of those five groups is switched on or off independently through
``features`` in this metric's config (see ``DEFAULT_FEATURES`` below); the
top-level ``coherence_suite.enabled`` switch on its own turns nothing on,
consistent with every other metric, but once it is on all five groups default
to on too, so a user opts *out* of the expensive ones rather than having to
discover and opt into every one of them.

No polarity judgement is made anywhere in this module. High cohesion can be
repetitive prose; low cohesion can be a deliberate montage or a poem. Every
finding here is ``Polarity.NEUTRAL`` by the default the rest of the tool
already applies to optional findings, and nothing in this module aggregates
these numbers into a single "coherence score".

Deferred (see the task spec's own library table for the fuller list this was
drawn from):

* **Real coreference** (fastcoref, Coreferee, spaCy's coref pipe, BookNLP).
  None is installed, and none can be exercised without a model download this
  environment cannot fetch and test. Entity continuity below is explicitly
  surface/lemma based instead, and every finding that depends on it says so.
* **RST parsing** (IsaNLP, Feng-Hirst, discopy, DisCoDisCo) and **PDTB
  implicit-relation labels**. No maintained, installable parser exists in
  this environment; faking a relation label or a tree depth from string
  matching would be worse than not reporting it. Only explicit,
  surface-matched connectives are measured.
* **TAACO / ReaderBench / a corpus reference distribution for entity-grid
  transition likelihood.** These need either an external tool this
  environment cannot run or a fitted reference corpus this task does not
  ship; the transition frequencies and their entropy are reported instead of
  a likelihood-under-a-corpus number, so nothing here is invented against a
  reference sample that does not exist.
* **Per-channel (dialogue vs. narration) entity grids and lexical chains.**
  The adjacent-sentence overlap metric is split into ``full``/``narration``/
  ``dialogue`` because it is cheap to; the entity grid, the graph and the
  permutation tests are run on the full document only, because a coreference
  chain that crosses a quotation mark (a narrator naming a character a
  speaker then refers to as "I") is exactly the kind of continuity a
  channel split would sever.
* **Pronoun-to-named-mention transition rate.** ``pov.entity_pronoun_ratio``
  already measures named-entity-to-pronoun balance in narration; adding a
  second, entity-grid-flavoured version of the same comparison here would be
  a restatement, not a new channel.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Any, Mapping

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
REQUIRES: tuple[str, ...] = ("spacy", "sentence_transformers", "networkx")
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

# Sample-size floors for measurements whose unit isn't "sentences in the
# document" (pairs, paragraphs, words, graph nodes).
MIN_SAMPLE_PAIRS = 9
MIN_SAMPLE_PARAGRAPH_PAIRS = 4
MIN_SAMPLE_PARAGRAPHS = 5
MIN_SAMPLE_WORDS = 200
MIN_SAMPLE_GRAPH_NODES = 3

DEFAULT_FEATURES: dict[str, bool] = {
    "lexical": True,
    "semantic": True,
    "entity": True,
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
    if features.get("semantic", True):
        out.extend(_semantic(analysis, config))
    if features.get("connectives", True):
        out.extend(_connectives(analysis, config))
    if features.get("entity", True):
        out.extend(_entity(analysis, config))
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
        distribution={"chain_count": len(real_chains),
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

_ENTITY_METRIC_NAMES = (
    ("discourse.coherence_entity_new_given_ratio", "New-vs-given entity mention ratio"),
    ("discourse.coherence_entity_reintroduction_distance",
     "Distance in sentences between repeated mentions of the same entity"),
    ("discourse.coherence_entity_dangling_rate",
     "Entities introduced once and never mentioned again"),
    ("discourse.coherence_entity_grid_transition_entropy",
     "Entropy of adjacent subject/object/other/absent entity-grid transitions"),
    ("discourse.coherence_entity_graph_density", "Density of the entity co-occurrence graph"),
)


def _entity_given_new(per_sentence: list[dict[str, str]], total_sentences: int) -> dict[str, Any]:
    metric_id, name = _ENTITY_METRIC_NAMES[0]
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
        warning = "no entity mentions found (no noun chunk was headed by a noun or proper noun)"
    elif given_count == 0:
        warning = "every mention was a first mention of its entity; the ratio is undefined"
    else:
        warning = None
    ratio = (new_count / given_count) if given_count else None
    return finding(metric_id, name, ratio, "ratio", family=FAMILY, sample_size=total_sentences,
                  min_sample=MIN_SAMPLE,
                  distribution={"new_mentions": new_count, "given_mentions": given_count,
                               "carried_over_from_previous_sentence": carried_over,
                               "sentences_with_no_tracked_entity": empty_sentences,
                               "distinct_entities": len(seen),
                               "entity_identity": "noun-chunk root lemma, case-folded "
                                                  "(surface-based, not coreference)"},
                  warning=warning)


def _entity_reintroduction(per_sentence: list[dict[str, str]]) -> dict[str, Any]:
    metric_id, name = _ENTITY_METRIC_NAMES[1]
    last_seen: dict[str, int] = {}
    gaps: list[int] = []
    for index, roles in enumerate(per_sentence):
        for key in roles:
            if key in last_seen:
                gaps.append(index - last_seen[key])
            last_seen[key] = index
    if not gaps:
        return finding(metric_id, name, None, "sentences", family=FAMILY, sample_size=0,
                      min_sample=MIN_SAMPLE,
                      warning="no entity was mentioned more than once")
    return shape(metric_id, name, gaps, "sentences", family=FAMILY, min_sample=MIN_SAMPLE,
                evidence=[{"gap_sentences": gap} for gap in sorted(gaps, reverse=True)[:20]])[0]


def _entity_dangling(per_sentence: list[dict[str, str]], total_sentences: int,
                     lookback: int) -> dict[str, Any]:
    metric_id, name = _ENTITY_METRIC_NAMES[2]
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
                      min_sample=MIN_SAMPLE,
                      warning=f"every tracked entity was introduced in the last {lookback} "
                              f"sentences, too close to the end to judge whether it dangles")
    dangling = [key for key in eligible if counts[key] == 1]
    return finding(
        metric_id, name, 100.0 * len(dangling) / len(eligible), "%", family=FAMILY,
        sample_size=len(eligible), min_sample=MIN_SAMPLE,
        distribution={"dangling_entities": len(dangling), "eligible_entities": len(eligible),
                     "lookback_sentences": lookback},
        evidence=[{"entity": key, "first_sentence_index": first_seen[key]}
                 for key in sorted(dangling, key=lambda item: first_seen[item])[:25]])


def _entity_transition_entropy(rows: dict[str, list[str]], tracked: list[str]) -> dict[str, Any]:
    metric_id, name = _ENTITY_METRIC_NAMES[3]
    if not tracked:
        return finding(metric_id, name, None, "bits", family=FAMILY, sample_size=0,
                      min_sample=MIN_SAMPLE,
                      warning="no entity was mentioned in two or more sentences")
    counts = coh.transition_counts(rows)
    total_transitions = sum(counts.values())
    entropy = coh.entropy_of_counts(counts)
    return finding(
        metric_id, name, entropy, "bits", family=FAMILY, sample_size=total_transitions,
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"role_schema": coh.ROLE_SCHEMA_VERSION, "tracked_entities": len(tracked),
                     "possible_transition_types": 16,
                     "max_possible_bits": math.log2(min(16, len(counts))) if counts else 0.0},
        evidence=[{"from": a, "to": b, "count": count}
                 for (a, b), count in sorted(counts.items(), key=lambda item: -item[1])[:25]],
        warning=None if total_transitions else "no adjacent-sentence transition to measure")


def _entity_graph(rows: dict[str, list[str]], tracked: list[str], window: int,
                  min_mentions: int, freq: dict[str, int]) -> dict[str, Any]:
    metric_id, name = _ENTITY_METRIC_NAMES[4]
    module, reason = require("networkx")
    if module is None:
        return unavailable(metric_id, name, reason, family=FAMILY)
    graph_tracked = [key for key in tracked if freq.get(key, 0) >= min_mentions]
    if len(graph_tracked) < MIN_SAMPLE_GRAPH_NODES:
        return finding(metric_id, name, None, "ratio", family=FAMILY,
                      sample_size=len(graph_tracked), min_sample=MIN_SAMPLE_GRAPH_NODES,
                      warning=f"needs at least {MIN_SAMPLE_GRAPH_NODES} entities mentioned "
                              f"{min_mentions}+ times each; found {len(graph_tracked)}")
    sub_rows = {key: rows[key] for key in graph_tracked}
    graph = coh.build_entity_graph(module, sub_rows, window)
    stats = coh.graph_stats(module, graph)
    return finding(
        metric_id, name, stats["density"], "ratio", family=FAMILY, sample_size=stats["nodes"],
        min_sample=MIN_SAMPLE_GRAPH_NODES,
        distribution={**stats, "co_occurrence_window_sentences": window,
                     "min_mentions_to_track": min_mentions,
                     "networkx_version": getattr(module, "__version__", None)},
        warning=None if stats["edges"] else "no two entities ever co-occurred within the window")


def _entity(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    reason = analysis.nlp_unavailable
    if reason:
        return [unavailable(metric_id, name, reason, family=FAMILY)
               for metric_id, name in _ENTITY_METRIC_NAMES]

    lookback = int(option(config, "entity_lookback_sentences", 10))
    max_tracked = int(option(config, "entity_max_tracked", 150))
    graph_window = int(option(config, "entity_graph_window_sentences", 3))
    min_mentions_for_graph = int(option(config, "entity_min_mentions_for_graph", 2))

    per_sentence, _ = analysis.memo(
        "coherence_entity_mentions",
        lambda: coh.entity_mentions_by_sentence(analysis.spacy_sents()))
    total_sentences = len(per_sentence)
    if total_sentences == 0:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=0,
                       min_sample=MIN_SAMPLE, warning="the shared parse produced no sentences")
               for metric_id, name in _ENTITY_METRIC_NAMES]

    freq = coh.entity_frequency(per_sentence)
    tracked = [key for key, _ in sorted(freq.items(), key=lambda item: -item[1])[:max_tracked]]
    rows = coh.grid_rows(per_sentence, tracked)

    return [
        _entity_given_new(per_sentence, total_sentences),
        _entity_reintroduction(per_sentence),
        _entity_dangling(per_sentence, total_sentences, lookback),
        _entity_transition_entropy(rows, tracked),
        _entity_graph(rows, tracked, graph_window, min_mentions_for_graph, freq),
    ]


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
