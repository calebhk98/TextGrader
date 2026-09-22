"""Shared entity-grid, lexical-chain and permutation utilities for the
``coherence_suite`` metric family.

Nothing here is a metric.  This module holds the pieces that are genuinely
shared between several coherence measurements - a Jaccard overlap, a lexical
chain builder, an entity-grid role classifier, a deterministic permutation
scorer - so that :mod:`textgrader.metrics.coherence_suite` reads as a set of
small ``measure``-shaped functions rather than repeating the same windowed
loop five times with slightly different bookkeeping.

Judgement calls made once, here, rather than re-litigated per metric:

* **Entity identity is surface identity.**  Two mentions are "the same
  entity" if the lemma of a noun chunk's syntactic head matches, case-folded.
  This is not coreference: it will not link "the old woman" to "she" three
  sentences later, and it will wrongly merge two different rooms both called
  "the room".  No coreference model is installed in this environment
  (``fastcoref``/``coreferee``/spaCy's own coref pipe are all unavailable),
  so this is the honest fallback the spec asks for, and every finding that
  uses it says so.
* **Roles are a closed four-way schema**: subject (``S``), object (``O``),
  other mention (``X``), or absent (``-``).  This is deliberately the
  minimal Barzilay/Lapata-style schema, not the fuller PropBank-style role
  set some entity-grid implementations use, because deriving reliable
  finer-grained roles from a dependency parse alone (no semantic role
  labeler is installed) invites more precision than the input supports.
* **Permutations reorder, they never resample.**  Every permutation test in
  this module shuffles the exact list of sentences or paragraphs the document
  already has; it never invents, drops or rewords a unit.  That is what makes
  the result a measurement of order sensitivity rather than of vocabulary.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics.semantic_adjacent import STOPWORDS
from .text import words as split_words

ROLE_SCHEMA_VERSION = "sxo-v1"

SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass", "expl", "agent"})
OBJECT_DEPS = frozenset({"dobj", "obj", "iobj", "pobj", "dative", "attr", "oprd"})


# --------------------------------------------------------------- lexical

def content_words(text: str, min_len: int = 3) -> list[str]:
    """Lower-cased, stop-word-filtered tokens; the unit every overlap measure
    in this module compares.  Reuses the closed-class stop list the semantic
    similarity family already curated, instead of a second copy of it."""

    return [word.lower() for word in split_words(text)
            if len(word) >= min_len and word.lower() not in STOPWORDS]


def jaccard(a: Iterable[str], b: Iterable[str], default: float | None = None) -> float | None:
    """Jaccard overlap of two content-word sets, or ``default`` if both are empty.

    ``default=None`` (the reporting default) makes an empty-vs-empty pair
    honestly undefined rather than silently perfect or silently zero.
    ``default=0.0`` is used by the permutation scorers, which need a total
    order over shuffles and cannot skip a pair.
    """

    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return default
    return len(set_a & set_b) / len(union)


def adjacent_overlap(units: Sequence[str], min_len: int = 3) -> list[float]:
    """Jaccard content-word overlap between each unit and the one after it.

    Pairs where both units contribute no content words are left out of the
    result entirely (rather than scored 0 or 1), since "no overlap" and
    "nothing to overlap" are different findings.
    """

    token_sets = [set(content_words(unit, min_len)) for unit in units]
    out: list[float] = []
    for a, b in zip(token_sets, token_sets[1:]):
        value = jaccard(a, b)
        if value is not None:
            out.append(value)
    return out


def lexical_chains(units: Sequence[str], *, gap: int = 3, min_len: int = 3
                   ) -> list[list[int]]:
    """Chains of unit indices connected by a shared content word.

    A chain is a maximal run of units in which each member shares at least
    one content word with a member no more than ``gap`` units earlier.  This
    is the classic lexical-chain idea (Morris & Hirst) reduced to its
    cheapest honest form: exact word repetition, not a thesaurus walk, so it
    never claims a semantic link it cannot support.  Cost is linear in the
    number of content-word occurrences: one dict lookup per token, no
    pairwise unit comparison.
    """

    open_chains: dict[str, list[int]] = {}
    finished: list[list[int]] = []
    for index, unit in enumerate(units):
        seen_this_unit: set[str] = set()
        for word in content_words(unit, min_len):
            if word in seen_this_unit:
                continue
            seen_this_unit.add(word)
            chain = open_chains.get(word)
            if chain is not None and index - chain[-1] <= gap:
                chain.append(index)
            else:
                if chain is not None:
                    finished.append(chain)
                open_chains[word] = [index]
    finished.extend(open_chains.values())
    return finished


# ----------------------------------------------------------------- entities

def chunk_role(dep: str) -> str:
    if dep in SUBJECT_DEPS:
        return "S"
    if dep in OBJECT_DEPS:
        return "O"
    return "X"

_ROLE_RANK = {"S": 3, "O": 2, "X": 1}


def entity_mentions_by_sentence(spacy_sents: Iterable[Any]
                                ) -> tuple[list[dict[str, str]], list[str]]:
    """``(mentions_per_sentence, sentence_order)`` from noun chunks.

    ``mentions_per_sentence[i]`` maps an entity key (a lemma) to the single
    best role ``spaCy`` evidenced for it in sentence ``i`` (subject beats
    object beats other, so a name used as both subject and object of the same
    sentence is not double counted). Only chunks headed by a common or proper
    noun become entities; a chunk headed by a pronoun is real evidence of
    *something* being talked about but, without coreference, this module has
    no way to say what, so pronoun-headed chunks are not tracked as entities.
    """

    per_sentence: list[dict[str, str]] = []
    for sent in spacy_sents:
        roles: dict[str, str] = {}
        try:
            chunks = list(sent.noun_chunks)
        except Exception:  # pragma: no cover - defensive against parser edge cases
            chunks = []
        for chunk in chunks:
            root = chunk.root
            if root.pos_ not in ("NOUN", "PROPN"):
                continue
            key = root.lemma_.lower().strip()
            if not key:
                continue
            role = chunk_role(root.dep_)
            if _ROLE_RANK.get(role, 0) > _ROLE_RANK.get(roles.get(key, ""), 0):
                roles[key] = role
        per_sentence.append(roles)
    return per_sentence, []


def entity_frequency(per_sentence: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for roles in per_sentence:
        for key in roles:
            counts[key] = counts.get(key, 0) + 1
    return counts


def grid_rows(per_sentence: Sequence[Mapping[str, str]], entities: Sequence[str]
             ) -> dict[str, list[str]]:
    """Dense role sequence (including ``"-"`` for absent) per tracked entity."""

    return {key: [roles.get(key, "-") for roles in per_sentence] for key in entities}


def transition_counts(rows: Mapping[str, Sequence[str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for sequence in rows.values():
        for a, b in zip(sequence, sequence[1:]):
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1
    return counts


def entropy_of_counts(counts: Mapping[Any, int]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def build_entity_graph(module: Any, rows: Mapping[str, Sequence[str]], window: int):
    """A ``networkx.Graph`` linking entities that co-occur within ``window``
    sentences of each other at least once.  ``module`` is the already
    ``optional.require``-d ``networkx`` module; this function never imports it
    itself, so a caller without the package never pays for the attempt."""

    graph = module.Graph()
    graph.add_nodes_from(rows)
    keys = list(rows)
    # Sentence -> the entities present in it, built once instead of an
    # all-pairs scan of (entity, entity, sentence).
    length = len(next(iter(rows.values()))) if rows else 0
    present_at: list[list[str]] = [[] for _ in range(length)]
    for key, sequence in rows.items():
        for index, role in enumerate(sequence):
            if role != "-":
                present_at[index].append(key)
    for start in range(length):
        window_entities: set[str] = set()
        for offset in range(window + 1):
            index = start + offset
            if index >= length:
                break
            window_entities.update(present_at[index])
        window_entities_list = sorted(window_entities)
        for i, a in enumerate(window_entities_list):
            for b in window_entities_list[i + 1:]:
                graph.add_edge(a, b)
    return graph


def graph_stats(module: Any, graph: Any) -> dict[str, Any]:
    nodes = graph.number_of_nodes()
    if nodes == 0:
        return {"nodes": 0, "edges": 0, "density": None, "average_degree": None,
                "average_clustering": None, "connected_components": 0,
                "largest_component_share": None}
    components = list(module.connected_components(graph))
    largest = max((len(component) for component in components), default=0)
    degrees = [degree for _, degree in graph.degree()]
    return {
        "nodes": nodes, "edges": graph.number_of_edges(),
        "density": module.density(graph),
        "average_degree": sum(degrees) / nodes if nodes else None,
        "average_clustering": module.average_clustering(graph) if nodes else None,
        "connected_components": len(components),
        "largest_component_share": largest / nodes if nodes else None,
    }


# -------------------------------------------------------------- permutation

def permutation_percentile(real_items: Sequence[Any], score_fn: Callable[[Sequence[Any]], float],
                           permutations: int, rng: random.Random) -> tuple[float | None, float]:
    """Where the real order's score ranks among ``permutations`` reshuffles.

    Returns ``(percentile, real_score)``.  The percentile is the share of the
    ``permutations + 1`` samples (the real order plus every shuffle) that
    score at or below the real order, so ``100.0`` means the real order beat
    every shuffle tried and ``50.0`` means it looked like a typical shuffle.
    Reshuffling never changes the multiset of items, only their order, so a
    high percentile is evidence about arrangement, not about vocabulary.
    """

    if len(real_items) < 2:
        return None, 0.0
    real_score = score_fn(real_items)
    samples = [real_score]
    pool = list(real_items)
    for _ in range(max(0, permutations)):
        rng.shuffle(pool)
        samples.append(score_fn(pool))
    beaten_or_tied = sum(1 for value in samples if value <= real_score)
    return 100.0 * beaten_or_tied / len(samples), real_score


def order_score_from_overlap(units: Sequence[str], min_len: int = 3) -> float:
    """Sum of adjacent content-word overlap; the default, dependency-free
    scoring function for both permutation tests.  Empty-vs-empty pairs score
    0.0 here (not left out, unlike :func:`adjacent_overlap`'s reporting mode)
    because a permutation scorer needs one total order over every shuffle."""

    token_sets = [set(content_words(unit, min_len)) for unit in units]
    return sum(jaccard(a, b, default=0.0) for a, b in zip(token_sets, token_sets[1:]))


__all__ = [
    "ROLE_SCHEMA_VERSION", "content_words", "jaccard", "adjacent_overlap",
    "lexical_chains", "chunk_role", "entity_mentions_by_sentence", "entity_frequency",
    "grid_rows", "transition_counts", "entropy_of_counts", "build_entity_graph",
    "graph_stats", "permutation_percentile", "order_score_from_overlap",
]
