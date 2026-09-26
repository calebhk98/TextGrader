"""Text as a graph: entity, character, topic and lexical network statistics.

Every other discourse-level metric in this codebase (``coherence_suite``'s
entity grid, its lexical chains, its semantic-similarity pairs) measures a
RELATIONSHIP between adjacent or nearby units.  A graph makes the same kind of
evidence into a TOPOLOGY: not just "does entity X reappear near entity Y" but
"is this novel one dominant protagonist radiating out to a large supporting
cast (a star), a relay of two-character scenes (a chain), or two largely
separate storylines that only meet once (two communities)".  Those three
shapes have the same node count, and can even have the same edge count, and
are nonetheless obviously different books; degree centralization and
community structure are what tell them apart, which raw counts cannot.

This suite measures shape.  It never judges it: a single-protagonist story and
a flat ensemble cast are both legitimate topologies for AI-generated prose to
match against human examples of either kind, so every finding here is
``Polarity.NEUTRAL`` (the default for an optional metric) and nothing in this
module aggregates these numbers into a single "network quality" score.

Graph constructions, each with its edge meaning stated explicitly (see
:mod:`textgrader.graphs` for the shared machinery behind each one):

``lexical_chain``
    Nodes are sentence indices.  An edge links two sentences that are
    consecutive members of the same repeated-content-word lexical chain (see
    :func:`textgrader.coherence.lexical_chains`).  Dependency-free.
``surface_name``
    Nodes are surface, non-sentence-initial capitalized name candidates (a
    regex proxy, not NER -- see :data:`textgrader.graphs._NAME_RE`).  An edge
    links two names that co-occur within a sentence window.  Dependency-free:
    this is the suite's answer to "can a character/entity-like graph be built
    at all without a spaCy parse" (see the module's cost note below).
``entity_cooccurrence``
    Nodes are surface entities: the SAME noun-chunk-lemma entities
    ``coherence_suite``'s ``entity`` feature already tracks (reused, not
    re-extracted, via the identical ``analysis.memo`` cache key -- enabling
    both suites on the same document costs the noun-chunk walk once).  An
    edge links two entities that co-occur within a sentence window.  Needs the
    shared spaCy parse.
``character_cooccurrence``
    The same construction, restricted to PROPER-NOUN-headed mentions only
    (see :func:`textgrader.graphs.character_mentions_by_sentence`) -- a
    narrower, more defensible "this is probably a character or named place"
    signal than any noun chunk.  Needs the shared spaCy parse.
``coreference_entity``
    The same construction again, this time over real ``fastcoref``-resolved
    coreference chains (the SAME cached chains ``coherence_suite``'s
    ``coreference`` feature uses, via
    :func:`textgrader.coherence.resolve_coreference`), so "Alice"/"she"/"her"
    collapse to one node instead of three.  **Off by default even when the
    suite is on**: it loads a transformer model over a bounded window of the
    document, exactly like ``coherence_suite``'s identically-named feature.
``quote_speaker``
    Nodes are identified speakers (:mod:`dialogue_attribution`'s speech-tag
    speaker recovery).  An edge links two DIFFERENT speakers whose identified
    turns are adjacent in speaking order (an untagged turn in between does not
    break the adjacency -- see :func:`_quote_speaker_graph`).  Dependency-free.
``paragraph_entity_overlap``
    Nodes are PARAGRAPHS (not entities).  An edge links two paragraphs within
    a bounded paragraph window that mention at least one of the same tracked
    entities, weighted by how many they share.  Needs the shared spaCy parse
    (it reuses the same entity extraction as ``entity_cooccurrence``).
``topic_transition``
    Paragraphs are clustered into a small number of topics by TF-IDF k-means
    (deterministic, seeded).  Nodes are topic ids; an edge links two different
    topics whose paragraphs are adjacent in the document, weighted by how
    often that transition happens.  Needs ``scikit-learn`` (already a base
    dependency of this project; this is clustering, not a neural model).
``sentence_semantic`` / ``paragraph_semantic``
    Nodes are sentences (or paragraphs).  An edge links a unit to its nearest
    neighbours by embedding cosine similarity (a bounded k-NN, never a
    complete graph -- see ``semantic_graph_k``/``semantic_graph_max_units``
    below).  **Off by default**: reuses
    :mod:`textgrader.metrics.semantic_adjacent`'s embedding cache and degrades
    to its TF-IDF lexical fallback, with the same honest backend labelling,
    when ``sentence-transformers`` is not installed.
``dependency_relation``
    Nodes are POS tags.  An edge links two POS tags that stood in a head/child
    dependency relation anywhere in the document, weighted by how often.  A
    small, cheap, closed-vocabulary graph (there are only a couple of dozen
    POS tags), not a per-sentence syntax tree.  **Off by default**: this is
    the spec's explicitly OPTIONAL dependency-relation aggregate graph, added
    for completeness rather than because it answers a load-bearing question
    the rest of this suite does not.
``temporal_drift``
    Not a graph of its own: it builds the ``entity_cooccurrence`` construction
    twice, once over each half of the document (by paragraph), and reports how
    much the edge set changed (an edge-set Jaccard dissimilarity, labelled
    explicitly as a proxy for graph edit distance -- true graph edit distance
    is NP-hard and this codebase does not attempt it) and how much the
    community structure persisted for entities common to both halves (a
    pairwise co-community agreement rate, Rand-Index-style, not corrected for
    chance).  Needs the same shared parse as ``entity_cooccurrence``.
``booknlp``
    Real, neural character coreference and quote-speaker attribution (David
    Bamman's BookNLP), in place of the surface/coreference-lemma heuristics
    above, over a bounded PREFIX of the document (see
    :func:`textgrader.graphs.resolve_booknlp`).  **Off by default and very
    heavy**: on this project's small English model, measured at roughly 40
    words/second of CPU processing plus a one-time ~20s, ~880MB-resident model
    load (see :mod:`textgrader.graphs`'s docstring for the exact numbers from
    a real run against a Project Gutenberg novel excerpt) -- a whole
    300,000-word novel would take on the order of two hours, so this feature
    only ever sends a bounded prefix (``booknlp_max_words``), never the whole
    book, and is never eligible for corpus profiling (this suite's cost stays
    ``"parse"``, and BookNLP needs neither ``sentence_transformers`` nor
    another flag ``MetricSpec.needs_model`` would catch, so this feature flag
    is the ONLY thing standing between it and an unasked-for two-hour run --
    see the "Gating" note below for why that matters more here than anywhere
    else in this suite).  BookNLP's own state-dict-loading bug against current
    ``transformers``/``torch`` (a legacy ``bert.embeddings.position_ids``
    buffer key old checkpoints still carry) is worked around by
    :func:`textgrader.optional.shim_booknlp_transformers`, the same pattern
    ``shim_fastcoref_transformers``/``shim_benepar_transformers`` already use
    for their own version-skew bugs -- see that function's docstring for the
    exact error this reproduced and why the fix cannot mask a real mismatch.

Every graph reports the SAME statistics battery
(:func:`textgrader.graphs.graph_statistics`): raw node/edge counts (marked
unit-sensitive, with density as their normalized sibling), degree
mean/variance/entropy, largest-component share and connected-component count,
isolate share, clustering/transitivity, degree assortativity, average
shortest path/diameter/radius on the largest component (bounded; skipped
above a node-count cap, never silently truncated), Louvain community
count/modularity/community-size entropy (seeded, algorithm named), PageRank
concentration (Gini), betweenness and degree centralization, and edge-weight
entropy -- plus, where ``igraph`` is installed, an independent recomputation
of density/transitivity/components/modularity kept SIDE BY SIDE with
``networkx``'s own numbers, never averaged (see :mod:`textgrader.graphs`'s
module docstring for why).  A node/edge count that reflects a graph with a
node-reappearance sequence (every construction except ``sentence_semantic``/
``paragraph_semantic``/``dependency_relation``, whose nodes are not something
that "reappears" in the same sense) also reports reappearance-distance shape:
how many units typically pass before the same node comes back.

**Gating.**  ``MetricSpec.needs_model`` -- the switch that keeps corpus
profiling from silently downloading and running a model -- checks
``REQUIRES`` for exactly the string ``"sentence_transformers"``.  This
suite's ``REQUIRES`` deliberately does NOT include it, even though the
``sentence_semantic``/``paragraph_semantic`` features can load that package:
including it would flip ``needs_model`` for the WHOLE suite, which would also
silently exclude every dependency-free and parse-only graph (lexical chain,
surface name, entity/character co-occurrence, quote-speaker, paragraph
overlap) from ever being profiled by a corpus build that does not separately
pass ``--model-metrics`` -- exactly the failure mode this project's rules
warn against for this suite specifically.  Instead, every model-backed
feature (``sentence_semantic``, ``paragraph_semantic``, ``coreference_entity``,
``booknlp``) is its own ``features`` entry, off by default, and loads nothing
under the default config: see ``tests/test_graph_suite.py``'s
``test_default_config_loads_no_model_or_heavy_backend`` for a test that
monkeypatches every one of those loaders to raise and confirms the default
config never calls them.  This suite's ``cost`` is ``"parse"`` (several
features need the shared spaCy parse), so ``config.json``'s entry notes,
next to every parse-dependent feature, that a corpus profile needs
``--parse-metrics`` before that feature is ever precomputed -- the same
convention ``coherence_suite`` and ``syntax_complexity_suite`` already use.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from .. import coherence as coh
from .. import graphs as gr
from ..document import DocumentAnalysis
from ..optional import require
from .common import PARSE, finding, option, shape, unavailable
from . import dialogue_attribution as dlg
from . import semantic_adjacent as sem

FAMILY = "discourse"
# entity/character/paragraph-overlap/dependency/temporal-drift all need the
# shared spaCy parse; the dependency-free features (lexical_chain,
# surface_name, quote_speaker) still run without one, exactly the way
# coherence_suite's dependency-free groups do despite that module's cost also
# being "parse" -- see this module's "Gating" docstring section.
COST = PARSE
# Deliberately NOT "sentence_transformers" -- see the module docstring's
# "Gating" section for why. "booknlp" is listed because it affects neither
# needs_parse nor needs_model (MetricSpec only checks for "spacy" and
# "sentence_transformers" respectively), so listing it here only documents a
# real optional dependency without changing any gating behaviour.
REQUIRES: tuple[str, ...] = ("spacy", "networkx", "fastcoref", "booknlp")
MIN_SAMPLE = 3
UNIT_SENSITIVE = False

#: Below this many nodes, a graph's statistics are still computed and
#: reported (see :func:`_graph_findings`), but with ``sample_size`` below
#: ``min_sample`` so ``grade.py``'s own comparator marks the result
#: ``Action.INSUFFICIENT_DATA`` and withholds it from corpus comparison,
#: rather than this module silently discarding a real, computable number.
MIN_GRAPH_NODES = 3

DEFAULT_FEATURES: dict[str, bool] = {
    "lexical_chain": True,
    "surface_name": True,
    "entity_cooccurrence": True,
    "character_cooccurrence": True,
    "paragraph_entity_overlap": True,
    "quote_speaker": True,
    "topic_transition": True,
    "temporal_drift": True,
    "sentence_semantic": False,
    "paragraph_semantic": False,
    "coreference_entity": False,
    "dependency_relation": False,
    "booknlp": False,
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
    if features.get("lexical_chain", True):
        out.extend(_lexical_chain_graph(analysis, config))
    if features.get("surface_name", True):
        out.extend(_surface_name_graph(analysis, config))
    if features.get("entity_cooccurrence", True):
        out.extend(_entity_cooccurrence_graph(analysis, config))
    if features.get("character_cooccurrence", True):
        out.extend(_character_cooccurrence_graph(analysis, config))
    if features.get("paragraph_entity_overlap", True):
        out.extend(_paragraph_entity_overlap_graph(analysis, config))
    if features.get("quote_speaker", True):
        out.extend(_quote_speaker_graph(analysis, config))
    if features.get("topic_transition", True):
        out.extend(_topic_transition_graph(analysis, config))
    if features.get("temporal_drift", True):
        out.extend(_temporal_drift(analysis, config))
    if features.get("sentence_semantic", False):
        out.extend(_semantic_graph(analysis, config, "sentence"))
    if features.get("paragraph_semantic", False):
        out.extend(_semantic_graph(analysis, config, "paragraph"))
    if features.get("coreference_entity", False):
        out.extend(_coreference_entity_graph(analysis, config))
    if features.get("dependency_relation", False):
        out.extend(_dependency_relation_graph(analysis, config))
    if features.get("booknlp", False):
        out.extend(_booknlp_graphs(analysis, config))
    return out


# ------------------------------------------------------------- shared machinery

def _ids(prefix: str, label: str, with_recurrence: bool) -> dict[str, tuple[str, str]]:
    ids = {
        "nodes": (f"discourse.graph_{prefix}_node_count", f"{label}: node count"),
        "edges": (f"discourse.graph_{prefix}_edge_count", f"{label}: edge count"),
        "density": (f"discourse.graph_{prefix}_density", f"{label}: edge density"),
        "community": (f"discourse.graph_{prefix}_community_structure",
                      f"{label}: community structure (modularity)"),
        "centralization": (f"discourse.graph_{prefix}_centralization",
                           f"{label}: degree centralization"),
        "path": (f"discourse.graph_{prefix}_path_length",
                 f"{label}: average shortest-path length (largest component)"),
    }
    if with_recurrence:
        ids["recurrence"] = (f"discourse.graph_{prefix}_recurrence_distance",
                             f"{label}: node reappearance distance")
    return ids


def _all_ids(prefix: str, label: str, with_recurrence: bool) -> list[tuple[str, str]]:
    return list(_ids(prefix, label, with_recurrence).values())


def _unavailable_all(prefix: str, label: str, reason: str, with_recurrence: bool = True
                     ) -> list[dict[str, Any]]:
    return [unavailable(metric_id, name, reason, family=FAMILY)
            for metric_id, name in _all_ids(prefix, label, with_recurrence)]


def _graph_findings(prefix: str, label: str, module: Any, graph: Any,
                    construction: Mapping[str, Any], *, present_at: Sequence[Sequence[Any]] | None,
                    weighted: bool = True, seed: int = 0, min_nodes: int = MIN_GRAPH_NODES
                    ) -> list[dict[str, Any]]:
    """The shared statistics battery for one graph, as this suite's findings.

    Every graph type in this module funnels through here so the same six (or
    seven, with ``present_at``) metric ids and the same statistics are always
    reported the same way -- see the module docstring for what each id means.
    """

    ids = _ids(prefix, label, present_at is not None)
    n = graph.number_of_nodes()
    if n == 0:
        warning = "this construction produced no graph nodes at all"
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=0,
                        min_sample=min_nodes, distribution=dict(construction), warning=warning)
                for metric_id, name in ids.values()]

    # Below min_nodes, the statistics are still computed and reported (a
    # 2-node graph's density is a perfectly well-defined 1.0, not "unknown"):
    # grade.py's own comparator turns a real value whose sample_size is below
    # its min_sample into Action.INSUFFICIENT_DATA and withholds it from
    # corpus comparison, which is the honest way to say "too small to trust",
    # rather than this module silently discarding a real, computable number.
    small_graph_note = (
        f"this graph has only {n} node(s), below the {min_nodes} this suite's network "
        f"statistics need to be read as reliable rather than an artefact of a handful of "
        f"co-occurrences" if n < min_nodes else None)

    stats = gr.graph_statistics(module, graph, seed=seed, weighted=weighted)
    cross = gr.igraph_cross_check(graph, weighted=weighted)
    out: list[dict[str, Any]] = []

    out.append(finding(*ids["nodes"], stats["nodes"], "nodes", family=FAMILY, sample_size=n,
                       min_sample=min_nodes, unit_sensitive=True, distribution=dict(construction)))
    out.append(finding(*ids["edges"], stats["edges"], "edges", family=FAMILY, sample_size=n,
                       min_sample=min_nodes, unit_sensitive=True,
                       distribution={**construction, "possible_edges": n * (n - 1) // 2}))

    density_dist = {
        **construction, "average_degree": stats["average_degree"],
        "degree_variance": stats["degree_variance"], "degree_entropy": stats["degree_entropy"],
        "largest_component_nodes": stats["largest_component_nodes"],
        "largest_component_share": stats["largest_component_share"],
        "connected_components": stats["connected_components"],
        "isolate_share": stats["isolate_share"],
        "average_clustering": stats["average_clustering"], "transitivity": stats["transitivity"],
        "edge_weight_entropy": stats["edge_weight_entropy"],
        "networkx_version": getattr(module, "__version__", None),
    }
    if cross.get("available"):
        density_dist["igraph_cross_check"] = {
            "networkx_version": getattr(module, "__version__", None), "igraph": cross,
            "density_disagreement": gr.disagreement(stats["density"], cross["density"]),
            "transitivity_disagreement": gr.disagreement(stats["transitivity"], cross["transitivity"]),
            "connected_components_disagreement": gr.disagreement(
                stats["connected_components"], cross["connected_components"]),
        }
    else:
        density_dist["igraph_cross_check"] = cross
    out.append(finding(*ids["density"], stats["density"], "ratio", family=FAMILY, sample_size=n,
                       min_sample=min_nodes, distribution=density_dist,
                       warning=None if stats["edges"] else "the graph has no edges"))

    community_dist = {
        **construction, "community_count": stats["community_count"],
        "community_size_entropy": stats["community_size_entropy"],
        "algorithm": stats["community_algorithm"], "seed": stats["community_seed"],
    }
    if cross.get("available") and cross.get("modularity") is not None:
        community_dist["igraph_cross_check"] = {
            "modularity": cross["modularity"], "community_count": cross["community_count"],
            "algorithm": cross["community_algorithm"],
            "modularity_disagreement": gr.disagreement(stats["modularity"], cross["modularity"]),
            "community_count_disagreement": gr.disagreement(
                stats["community_count"], cross["community_count"]),
        }
    else:
        community_dist["igraph_cross_check"] = cross if not cross.get("available") else None
    out.append(finding(*ids["community"], stats["modularity"], "modularity", family=FAMILY,
                       sample_size=n, min_sample=min_nodes, distribution=community_dist,
                       warning=None if stats["edges"] else "the graph has no edges to partition"))

    centralization_dist = {
        **construction, "assortativity": stats["assortativity"],
        "pagerank_gini": stats["pagerank_gini"],
        "betweenness_centralization": stats["betweenness_centralization"],
        "betweenness_settings": stats["betweenness_settings"],
    }
    out.append(finding(*ids["centralization"], stats["degree_centralization"], "ratio",
                       family=FAMILY, sample_size=n, min_sample=min_nodes,
                       distribution=centralization_dist))

    path_dist = {
        **construction, "diameter": stats["diameter"], "radius": stats["radius"],
        "largest_component_nodes": stats["largest_component_nodes"],
        "largest_component_share": stats["largest_component_share"],
    }
    out.append(finding(*ids["path"], stats["average_shortest_path_length"], "edges",
                       family=FAMILY, sample_size=stats["largest_component_nodes"],
                       min_sample=min_nodes, distribution=path_dist,
                       warning=stats["path_skipped_reason"]))

    if present_at is not None:
        gaps = gr.reappearance_gaps(present_at)
        if gaps:
            rec = shape(*ids["recurrence"], gaps, "units", family=FAMILY, min_sample=min_nodes)[0]
            rec["distribution"] = {**construction, **(rec["distribution"] or {})}
        else:
            rec = finding(*ids["recurrence"], None, "units", family=FAMILY, sample_size=0,
                          min_sample=min_nodes, distribution=dict(construction),
                          warning="no node appeared in more than one unit")
        out.append(rec)
    if small_graph_note:
        for item in out:
            item["warning"] = (small_graph_note if not item["warning"]
                               else f"{small_graph_note}; {item['warning']}")
    return out


def _node_frequency(present_at: Sequence[Sequence[Any]]) -> dict[Any, int]:
    freq: dict[Any, int] = {}
    for nodes in present_at:
        for node in nodes:
            freq[node] = freq.get(node, 0) + 1
    return freq


def _filter_present_at(present_at: Sequence[Sequence[Any]], min_mentions: int
                       ) -> list[list[Any]]:
    freq = _node_frequency(present_at)
    keep = {node for node, count in freq.items() if count >= min_mentions}
    return [[node for node in nodes if node in keep] for nodes in present_at]


# ------------------------------------------------------------- lexical_chain

def _lexical_chain_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                         ) -> list[dict[str, Any]]:
    prefix, label = "lexical_chain", "Lexical-chain graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason, with_recurrence=False)

    min_len = int(option(config, "min_word_len", 3))
    gap = int(option(config, "lexical_chain_gap", 3))
    min_chain_len = int(option(config, "lexical_chain_min_length", 2))
    sentences = analysis.sentences
    if not sentences:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=0,
                        min_sample=MIN_GRAPH_NODES, warning="no sentences to build a graph from")
                for metric_id, name in _all_ids(prefix, label, False)]

    chains = coh.lexical_chains(sentences, gap=gap, min_len=min_len)
    real_chains = [chain for chain in chains if len(chain) >= min_chain_len]
    graph = gr.chain_path_graph(module, real_chains, len(sentences))
    construction = {"graph_type": "lexical_chain", "unit": "sentence", "chain_gap": gap,
                    "chain_min_length": min_chain_len, "min_word_len": min_len,
                    "chains_used": len(real_chains), "edge_definition":
                    "two sentences are linked if they are consecutive members of the same "
                    "repeated-content-word lexical chain; weight = number of chains linking them"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=None)


# --------------------------------------------------------------- surface_name

def _surface_name_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                        ) -> list[dict[str, Any]]:
    prefix, label = "surface_name", "Surface capitalized-name co-occurrence graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason)

    window = int(option(config, "surface_name_window_sentences", 3))
    min_mentions = int(option(config, "min_mentions_for_graph", 2))
    sentences = analysis.sentences
    if not sentences:
        return _unavailable_all(prefix, label, "no sentences to scan for names")

    raw_present_at = gr.surface_name_sequence(sentences)
    present_at = _filter_present_at(raw_present_at, min_mentions)
    graph = gr.cooccurrence_graph(module, present_at, window)
    construction = {"graph_type": "surface_name", "unit": "sentence", "backend": "surface_regex",
                    "window_sentences": window, "min_mentions_for_graph": min_mentions,
                    "names_found_before_filtering": len(_node_frequency(raw_present_at)),
                    "edge_definition":
                    "two capitalized name candidates are linked with weight = number of "
                    "sentence-window start positions in which both were present (see "
                    "textgrader.graphs's module docstring for the exact rule)",
                    "note": "dependency-free heuristic (regex, not NER); works even when the "
                            "shared spaCy parse is unavailable"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=present_at)


# --------------------------------------------------------- entity_cooccurrence

def _entity_rows(analysis: DocumentAnalysis, config: Mapping[str, Any]
                 ) -> tuple[list[str], dict[str, Any]] | None:
    """Tracked entities + dense role rows, from the SAME cache key
    ``coherence_suite`` uses, so enabling both suites costs one noun-chunk
    walk, not two."""

    max_tracked = int(option(config, "entity_max_tracked", 150))
    per_sentence, _channels = analysis.memo(
        "coherence_entity_mentions",
        lambda: coh.entity_mentions_by_sentence(analysis.spacy_sents_by_channel()))
    if not per_sentence:
        return None
    freq = coh.entity_frequency(per_sentence)
    tracked = [key for key, _ in sorted(freq.items(), key=lambda item: -item[1])[:max_tracked]]
    if not tracked:
        return None
    return tracked, coh.grid_rows(per_sentence, tracked)


def _entity_cooccurrence_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                               ) -> list[dict[str, Any]]:
    prefix, label = "entity", "Named-entity co-occurrence graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason)
    parse_reason = analysis.nlp_unavailable
    if parse_reason:
        return _unavailable_all(prefix, label, parse_reason)

    window = int(option(config, "entity_graph_window_sentences", 3))
    min_mentions = int(option(config, "min_mentions_for_graph", 2))
    computed = _entity_rows(analysis, config)
    if computed is None:
        return _unavailable_all(prefix, label,
                                "no entity mentions found (no noun chunk was headed by a noun "
                                "or proper noun)")
    tracked, rows = computed
    freq = {key: sum(1 for role in seq if role != "-") for key, seq in rows.items()}
    graph_tracked = [key for key in tracked if freq.get(key, 0) >= min_mentions]
    present_at = gr.present_at_from_rows({key: rows[key] for key in graph_tracked})
    graph = gr.cooccurrence_graph(module, present_at, window)
    construction = {"graph_type": "entity_cooccurrence", "unit": "sentence",
                    "backend": "surface_lemma",
                    "entity_identity": "noun-chunk root lemma, case-folded (surface-based, not "
                                       "coreference; see the coreference_entity feature)",
                    "window_sentences": window, "min_mentions_for_graph": min_mentions,
                    "entities_tracked": len(graph_tracked),
                    "edge_definition":
                    "two entities are linked with weight = number of sentence-window start "
                    "positions in which both were present"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=present_at)


# ------------------------------------------------------- character_cooccurrence

def _character_cooccurrence_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                                  ) -> list[dict[str, Any]]:
    prefix, label = "character", "Character (proper-noun) co-occurrence graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason)
    parse_reason = analysis.nlp_unavailable
    if parse_reason:
        return _unavailable_all(prefix, label, parse_reason)

    window = int(option(config, "character_graph_window_sentences", 3))
    min_mentions = int(option(config, "min_mentions_for_graph", 2))
    # Capped exactly like entity_cooccurrence's entity_max_tracked: without a
    # cap, a long novel's proper-noun surface extraction can track hundreds of
    # distinct strings (real character names, place names, and spaCy PROPN
    # mis-tags alike), and this suite's exact graph algorithms (path length,
    # community detection) are then run on a graph sized by however large the
    # book's proper-noun vocabulary happens to be rather than by a bounded
    # setting - exactly the unbounded-by-document-length cost this project's
    # rules require every construction to avoid.
    max_tracked = int(option(config, "character_max_tracked", 150))
    raw_present_at = analysis.memo(
        "graph_suite_character_mentions",
        lambda: gr.character_mentions_by_sentence(analysis.spacy_sents_by_channel()))
    if not any(raw_present_at):
        return _unavailable_all(prefix, label,
                                "no proper-noun-headed noun chunk was found (no probable "
                                "character/named-place mention)")
    freq = _node_frequency(raw_present_at)
    top_tracked = {name for name, _count in
                  sorted(freq.items(), key=lambda item: -item[1])[:max_tracked]}
    capped_present_at = [[name for name in names if name in top_tracked]
                        for names in raw_present_at]
    present_at = _filter_present_at(capped_present_at, min_mentions)
    graph = gr.cooccurrence_graph(module, present_at, window)
    construction = {"graph_type": "character_cooccurrence", "unit": "sentence",
                    "backend": "surface_propn",
                    "entity_identity": "proper-noun-headed noun-chunk surface text, case-preserved "
                                       "(narrower than entity_cooccurrence's noun-chunk-lemma "
                                       "identity; see the coreference_entity feature for "
                                       "coreference-resolved character identity, and the "
                                       "booknlp feature for real character coreference)",
                    "window_sentences": window, "min_mentions_for_graph": min_mentions,
                    "character_max_tracked": max_tracked,
                    "characters_tracked": len(_node_frequency(present_at)),
                    "distinct_proper_nouns_before_capping": len(freq),
                    "edge_definition":
                    "two proper-noun mentions are linked with weight = number of "
                    "sentence-window start positions in which both were present"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=present_at)


# --------------------------------------------------- paragraph_entity_overlap

def _paragraph_entity_overlap_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                                    ) -> list[dict[str, Any]]:
    prefix, label = "paragraph_overlap", "Paragraph entity-overlap graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason, with_recurrence=False)
    parse_reason = analysis.nlp_unavailable
    if parse_reason:
        return _unavailable_all(prefix, label, parse_reason, with_recurrence=False)

    window = int(option(config, "paragraph_overlap_window", 5))
    min_mentions = int(option(config, "min_mentions_for_graph", 2))
    paragraphs = analysis.paragraphs
    if len(paragraphs) < MIN_GRAPH_NODES:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=len(paragraphs),
                        min_sample=MIN_GRAPH_NODES,
                        warning=f"needs at least {MIN_GRAPH_NODES} paragraphs; this text has "
                                f"{len(paragraphs)}")
                for metric_id, name in _all_ids(prefix, label, False)]

    computed = _entity_rows(analysis, config)
    if computed is None:
        return _unavailable_all(prefix, label,
                                "no entity mentions found (no noun chunk was headed by a noun "
                                "or proper noun)", with_recurrence=False)
    tracked, rows = computed
    freq = {key: sum(1 for role in seq if role != "-") for key, seq in rows.items()}
    graph_tracked = [key for key in tracked if freq.get(key, 0) >= min_mentions]
    sentence_present_at = gr.present_at_from_rows({key: rows[key] for key in graph_tracked})
    paragraph_present_at = gr.aggregate_to_paragraphs(sentence_present_at,
                                                      analysis.paragraph_sentence_counts)
    graph = gr.paragraph_overlap_graph(module, paragraph_present_at, window)
    construction = {"graph_type": "paragraph_entity_overlap", "unit": "paragraph",
                    "backend": "surface_lemma", "paragraph_window": window,
                    "min_mentions_for_graph": min_mentions, "entities_tracked": len(graph_tracked),
                    "edge_definition":
                    "two paragraphs within the paragraph window are linked, weight = number of "
                    "distinct tracked entities they both mention"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=None)


# ----------------------------------------------------------------- quote_speaker

def _quote_speaker_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                         ) -> list[dict[str, Any]]:
    prefix, label = "quote_speaker", "Quote-speaker interaction graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason, with_recurrence=False)

    turns = dlg.classify_turns(analysis)
    speaker_sequence = [item["speaker"] for item in turns if item["speaker"]]
    if len(set(speaker_sequence)) < MIN_GRAPH_NODES:
        return [finding(metric_id, name, None, None, family=FAMILY,
                        sample_size=len(set(speaker_sequence)), min_sample=MIN_GRAPH_NODES,
                        warning=f"needs at least {MIN_GRAPH_NODES} identified speakers; this "
                                f"text has {len(set(speaker_sequence))}")
                for metric_id, name in _all_ids(prefix, label, False)]

    graph, monologue_runs = gr.turn_adjacency_graph(module, speaker_sequence)
    construction = {"graph_type": "quote_speaker", "unit": "identified spoken turn",
                    "backend": "speech_tag_speaker_recovery",
                    "turns_with_a_recovered_speaker": len(speaker_sequence),
                    "total_turns": len(turns), "monologue_runs": monologue_runs,
                    "edge_definition":
                    "two different identified speakers are linked, weight = number of times "
                    "one's turn is immediately followed by the other's in speaking order "
                    "(an untagged turn in between does not break the adjacency)"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=None)


# -------------------------------------------------------------- topic_transition

def _topic_transition_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                            ) -> list[dict[str, Any]]:
    prefix, label = "topic_transition", "Topic-transition graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason, with_recurrence=False)
    sklearn_module, sk_reason = require("sklearn")
    if sklearn_module is None:
        return _unavailable_all(prefix, label, sk_reason, with_recurrence=False)

    n_topics = int(option(config, "topic_n_topics", 6))
    seed = int(option(config, "seed", 0))
    max_paragraphs = int(option(config, "topic_max_paragraphs", 2000))
    paragraphs = analysis.paragraphs[:max_paragraphs]
    if len(paragraphs) < n_topics * 2:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=len(paragraphs),
                        min_sample=n_topics * 2,
                        warning=f"needs at least {n_topics * 2} paragraphs for {n_topics} "
                                f"topics; this text has {len(paragraphs)}")
                for metric_id, name in _all_ids(prefix, label, False)]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    try:
        matrix = TfidfVectorizer(max_features=2000, stop_words="english").fit_transform(paragraphs)
        labels = KMeans(n_clusters=n_topics, random_state=seed, n_init=10).fit_predict(matrix)
    except Exception as exc:  # pragma: no cover - degenerate-vocabulary guard
        return _unavailable_all(prefix, label,
                                f"topic clustering failed ({type(exc).__name__}: {exc})",
                                with_recurrence=False)
    topic_sequence = [f"topic_{label}" for label in labels]
    graph, dwell_transitions = gr.turn_adjacency_graph(module, topic_sequence)
    construction = {"graph_type": "topic_transition", "unit": "paragraph",
                    "backend": "tfidf_kmeans", "n_topics": n_topics, "seed": seed,
                    "paragraphs_clustered": len(paragraphs),
                    "same_topic_adjacent_paragraphs": dwell_transitions,
                    "edge_definition":
                    "two different topic ids are linked, weight = number of times consecutive "
                    "paragraphs transition between them; a paragraph staying on the same topic "
                    "as its predecessor does not create an edge and is counted separately as "
                    "same_topic_adjacent_paragraphs"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=None)


# ----------------------------------------------------------------- temporal_drift

def _partition_communities(module: Any, present_at: Sequence[Sequence[str]], window: int,
                           seed: int) -> tuple[Any, dict[str, int]]:
    graph = gr.cooccurrence_graph(module, present_at, window)
    community = gr.community_detection(module, graph, seed=seed)
    membership: dict[str, int] = {}
    for index, group in enumerate(community["communities"]):
        for node in group:
            membership[node] = index
    return graph, membership


def _temporal_drift(analysis: DocumentAnalysis, config: Mapping[str, Any]
                    ) -> list[dict[str, Any]]:
    edge_id = ("discourse.graph_temporal_edge_distance",
              "Entity co-occurrence graph edge-set change, early half vs. late half "
              "(edge-set Jaccard dissimilarity; a labelled proxy for graph edit distance, "
              "which is NP-hard and not attempted here)")
    community_id = ("discourse.graph_temporal_community_persistence",
                    "Entity community-membership agreement, early half vs. late half, for "
                    "entities present in both (pairwise co-community agreement rate, "
                    "Rand-Index-style, not corrected for chance)")
    all_ids = [edge_id, community_id]

    module, reason = require("networkx")
    if module is None:
        return [unavailable(metric_id, name, reason, family=FAMILY) for metric_id, name in all_ids]
    parse_reason = analysis.nlp_unavailable
    if parse_reason:
        return [unavailable(metric_id, name, parse_reason, family=FAMILY)
                for metric_id, name in all_ids]

    window = int(option(config, "entity_graph_window_sentences", 3))
    min_mentions = int(option(config, "min_mentions_for_graph", 2))
    seed = int(option(config, "seed", 0))
    paragraphs = analysis.paragraphs
    if len(paragraphs) < 2 * MIN_GRAPH_NODES:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=len(paragraphs),
                        min_sample=2 * MIN_GRAPH_NODES,
                        warning=f"needs at least {2 * MIN_GRAPH_NODES} paragraphs to split into "
                                f"early/late halves; this text has {len(paragraphs)}")
                for metric_id, name in all_ids]

    computed = _entity_rows(analysis, config)
    if computed is None:
        return [unavailable(metric_id, name,
                            "no entity mentions found (no noun chunk was headed by a noun or "
                            "proper noun)", family=FAMILY) for metric_id, name in all_ids]
    tracked, rows = computed
    freq = {key: sum(1 for role in seq if role != "-") for key, seq in rows.items()}
    graph_tracked = [key for key in tracked if freq.get(key, 0) >= min_mentions]
    present_at = gr.present_at_from_rows({key: rows[key] for key in graph_tracked})

    counts = analysis.paragraph_sentence_counts
    midpoint_paragraph = len(counts) // 2
    midpoint_sentence = sum(counts[:midpoint_paragraph])
    early, late = present_at[:midpoint_sentence], present_at[midpoint_sentence:]
    if len(early) < 2 or len(late) < 2:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=0,
                        min_sample=1, warning="one half of the document had too few sentences "
                                              "to build a graph")
                for metric_id, name in all_ids]

    early_graph, early_membership = _partition_communities(module, early, window, seed)
    late_graph, late_membership = _partition_communities(module, late, window, seed)

    def _edge_set(graph: Any) -> set[frozenset]:
        return {frozenset(edge) for edge in graph.edges()}

    early_edges, late_edges = _edge_set(early_graph), _edge_set(late_graph)
    union = early_edges | late_edges
    jaccard_distance = 1.0 - (len(early_edges & late_edges) / len(union)) if union else None
    edge_finding = finding(
        *edge_id, jaccard_distance, "ratio", family=FAMILY,
        sample_size=len(union), min_sample=MIN_GRAPH_NODES,
        distribution={"graph_type": "temporal_drift_edges", "window_sentences": window,
                     "min_mentions_for_graph": min_mentions,
                     "early_half_nodes": early_graph.number_of_nodes(),
                     "early_half_edges": early_graph.number_of_edges(),
                     "late_half_nodes": late_graph.number_of_nodes(),
                     "late_half_edges": late_graph.number_of_edges(),
                     "shared_edges": len(early_edges & late_edges), "union_edges": len(union)},
        warning=None if union else "neither half produced any co-occurrence edge")

    common_nodes = sorted(set(early_membership) & set(late_membership))
    if len(common_nodes) < 2:
        community_finding = finding(
            *community_id, None, "ratio", family=FAMILY, sample_size=len(common_nodes),
            min_sample=2, warning=f"needs at least 2 entities present in both halves; found "
                                  f"{len(common_nodes)}")
    else:
        agree = total = 0
        for i, a in enumerate(common_nodes):
            for b in common_nodes[i + 1:]:
                total += 1
                same_early = early_membership[a] == early_membership[b]
                same_late = late_membership[a] == late_membership[b]
                agree += int(same_early == same_late)
        community_finding = finding(
            *community_id, agree / total if total else None, "ratio", family=FAMILY,
            sample_size=total, min_sample=1,
            distribution={"graph_type": "temporal_drift_communities", "window_sentences": window,
                         "entities_common_to_both_halves": len(common_nodes),
                         "node_pairs_compared": total, "seed": seed,
                         "algorithm": "networkx.algorithms.community.louvain_communities"})
    return [edge_finding, community_finding]


# --------------------------------------------------------------- sentence/paragraph semantic

def _semantic_graph(analysis: DocumentAnalysis, config: Mapping[str, Any], unit: str
                    ) -> list[dict[str, Any]]:
    prefix, label = f"{unit}_semantic", f"{unit.capitalize()} semantic-similarity graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason, with_recurrence=False)

    model_name = option(config, "semantic_model", sem.DEFAULT_MODEL)
    k = int(option(config, "semantic_graph_k", 5))
    min_similarity = float(option(config, "semantic_graph_min_similarity", 0.5))
    max_units = int(option(config, "semantic_graph_max_units", 1500))
    seed = int(option(config, "seed", 0))

    units = analysis.sentences if unit == "sentence" else analysis.paragraphs
    if len(units) < MIN_GRAPH_NODES:
        return [finding(metric_id, name, None, None, family=FAMILY, sample_size=len(units),
                        min_sample=MIN_GRAPH_NODES,
                        warning=f"needs at least {MIN_GRAPH_NODES} {unit}s; this text has "
                                f"{len(units)}")
                for metric_id, name in _all_ids(prefix, label, False)]

    sampled = len(units) > max_units
    if sampled:
        indices = sorted(random.Random(seed).sample(range(len(units)), max_units))
        selected = [units[i] for i in indices]
    else:
        indices = list(range(len(units)))
        selected = units

    getter = sem.get_sentence_vectors if unit == "sentence" else sem.get_paragraph_vectors
    backend, vectors, note = getter(analysis, model_name)
    if sampled:
        # Vectors were computed (and cached) over the WHOLE document by
        # get_sentence_vectors/get_paragraph_vectors; select the sampled
        # subset's rows rather than re-encoding, so sampling only bounds the
        # O(n^2) pairwise-similarity step below, not the embedding step.
        vectors = vectors[indices] if backend == "embedding" else [vectors[i] for i in indices]

    graph = module.Graph()
    graph.add_nodes_from(range(len(selected)))
    edges_added = 0
    for i in range(len(selected)):
        scored = []
        for j in range(len(selected)):
            if i == j:
                continue
            similarity = sem.similarity_at(backend, vectors, i, j)
            if similarity >= min_similarity:
                scored.append((similarity, j))
        scored.sort(reverse=True)
        for _similarity, j in scored[:k]:
            if not graph.has_edge(i, j):
                graph.add_edge(i, j, weight=1)
                edges_added += 1

    construction = {"graph_type": f"{unit}_semantic", "unit": unit, "backend": backend,
                    "model": model_name, "k_nearest_neighbors": k,
                    "min_similarity_threshold": min_similarity,
                    "units_in_document": len(units), "units_in_graph": len(selected),
                    "sampled": sampled, "sample_seed": seed if sampled else None,
                    "edge_definition":
                    "unit i is linked to unit j if j is among i's k nearest neighbours by "
                    "embedding cosine similarity, at or above the similarity threshold "
                    "(undirected: a link needs only one side to nominate the other)"}
    result = _graph_findings(prefix, label, module, graph, construction, present_at=None)
    for item in result:
        item["warning"] = note if not item["warning"] else f"{note}; {item['warning']}"
    return result


# ---------------------------------------------------------- coreference_entity

def _coreference_entity_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                              ) -> list[dict[str, Any]]:
    prefix, label = "coref_entity", "Coreference-resolved entity co-occurrence graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason)
    parse_reason = analysis.nlp_unavailable
    if parse_reason:
        return _unavailable_all(prefix, label, parse_reason)

    model_name = option(config, "coreference_model", coh.DEFAULT_COREF_MODEL)
    max_words = int(option(config, "coreference_max_words", 4000))
    window = int(option(config, "entity_graph_window_sentences", 3))
    min_mentions = int(option(config, "min_mentions_for_graph", 2))

    # Same cache key coherence_suite's "coreference" feature uses: enabling
    # both suites' coreference features on one document runs fastcoref once.
    per_sentence, _channels, settings, note = analysis.memo(
        "coherence_coref_chains",
        lambda: coh.resolve_coreference(analysis, model_name, max_words))
    if not per_sentence:
        return _unavailable_all(prefix, label, note or "coreference produced no result")

    freq = coh.entity_frequency(per_sentence)
    tracked = [key for key, count in freq.items() if count >= min_mentions]
    rows = coh.grid_rows(per_sentence, tracked)
    present_at = gr.present_at_from_rows(rows)
    graph = gr.cooccurrence_graph(module, present_at, window)
    construction = {"graph_type": "coreference_entity", "unit": "sentence", **settings,
                    "window_sentences": window, "min_mentions_for_graph": min_mentions,
                    "entities_tracked": len(tracked),
                    "edge_definition":
                    "two coreference-resolved entity clusters are linked with weight = number "
                    "of sentence-window start positions in which both were present"}
    result = _graph_findings(prefix, label, module, graph, construction, present_at=present_at)
    for item in result:
        item["warning"] = note if not item["warning"] else f"{note}; {item['warning']}"
    return result


# --------------------------------------------------------------- dependency_relation

def _dependency_relation_graph(analysis: DocumentAnalysis, config: Mapping[str, Any]
                               ) -> list[dict[str, Any]]:
    prefix, label = "dependency", "Dependency-relation (POS-pair) aggregate graph"
    module, reason = require("networkx")
    if module is None:
        return _unavailable_all(prefix, label, reason, with_recurrence=False)
    parse_reason = analysis.nlp_unavailable
    if parse_reason:
        return _unavailable_all(prefix, label, parse_reason, with_recurrence=False)

    graph = module.Graph()
    relation_total = 0
    for token in analysis.spacy_tokens():
        if token.dep_ == "ROOT" or token.head is token:
            continue
        a, b = token.head.pos_, token.pos_
        if not a or not b:
            continue
        graph.add_node(a)
        graph.add_node(b)
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += 1
        else:
            graph.add_edge(a, b, weight=1)
        relation_total += 1

    construction = {"graph_type": "dependency_relation", "unit": "POS tag",
                    "relations_observed": relation_total,
                    "edge_definition":
                    "two POS tags are linked with weight = number of head/child dependency "
                    "relations observed anywhere in the document between tokens of those tags "
                    "(direction is not tracked; the graph is undirected)"}
    return _graph_findings(prefix, label, module, graph, construction, present_at=None)


# ------------------------------------------------------------------------ booknlp

def _booknlp_graphs(analysis: DocumentAnalysis, config: Mapping[str, Any]
                    ) -> list[dict[str, Any]]:
    char_prefix, char_label = "booknlp_character", "BookNLP character co-occurrence graph"
    speak_prefix, speak_label = "booknlp_speaker", "BookNLP quote-speaker interaction graph"
    all_ids = _all_ids(char_prefix, char_label, True) + _all_ids(speak_prefix, speak_label, False)

    module, reason = require("networkx")
    if module is None:
        return [unavailable(metric_id, name, reason, family=FAMILY) for metric_id, name in all_ids]

    model_size = option(config, "booknlp_model", gr.DEFAULT_BOOKNLP_MODEL_SIZE)
    pipeline = option(config, "booknlp_pipeline", gr.DEFAULT_BOOKNLP_PIPELINE)
    max_words = int(option(config, "booknlp_max_words", 3000))
    window = int(option(config, "character_graph_window_sentences", 3))

    present_at, speaker_sequence, settings, note = analysis.memo(
        "graph_suite_booknlp",
        lambda: gr.resolve_booknlp(analysis.text, max_words, model_size, pipeline))
    if not present_at and not speaker_sequence:
        return [unavailable(metric_id, name, note or "BookNLP produced no result", family=FAMILY)
                for metric_id, name in all_ids]

    char_graph = gr.cooccurrence_graph(module, present_at, window) if present_at else module.Graph()
    char_construction = {"graph_type": "booknlp_character", "unit": "BookNLP sentence", **settings,
                         "window_sentences": window,
                         "edge_definition":
                         "two BookNLP character ids are linked with weight = number of "
                         "sentence-window start positions in which both were present"}
    char_result = _graph_findings(char_prefix, char_label, module, char_graph, char_construction,
                                  present_at=present_at)

    if len(set(speaker_sequence)) >= MIN_GRAPH_NODES:
        speak_graph, monologue_runs = gr.turn_adjacency_graph(module, speaker_sequence)
        speak_construction = {"graph_type": "booknlp_speaker", "unit": "BookNLP quote", **settings,
                              "monologue_runs": monologue_runs,
                              "edge_definition":
                              "two different BookNLP-attributed speakers are linked, weight = "
                              "number of times one's quote is immediately followed by the "
                              "other's"}
        speak_result = _graph_findings(speak_prefix, speak_label, module, speak_graph,
                                       speak_construction, present_at=None)
    else:
        speak_result = [finding(metric_id, name, None, None, family=FAMILY,
                                sample_size=len(set(speaker_sequence)), min_sample=MIN_GRAPH_NODES,
                                distribution=dict(settings),
                                warning=f"needs at least {MIN_GRAPH_NODES} BookNLP-identified "
                                        f"speakers; found {len(set(speaker_sequence))}")
                        for metric_id, name in _all_ids(speak_prefix, speak_label, False)]

    result = char_result + speak_result
    for item in result:
        item["warning"] = note if not item["warning"] else f"{note}; {item['warning']}"
    return result
