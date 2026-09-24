"""Graph construction and network-statistics helpers for ``graph_suite``.

Nothing here is a metric; this module holds the pieces genuinely shared
between the suite's dozen graph constructions, the way :mod:`textgrader.coherence`
holds the entity-grid/lexical-chain machinery ``coherence_suite`` builds on.
:mod:`textgrader.metrics.graph_suite` decides which graphs to build and what
to report; this module builds them and computes their statistics.

Judgement calls made once, here, rather than re-litigated per graph:

* **Every co-occurrence graph is a sliding-sentence-window graph, and its edge
  weight has one fixed meaning.**  For a window of ``w`` sentences, walk every
  start position ``s`` in the document; let ``W(s)`` be the set of nodes
  present in sentences ``[s, s+w]``.  Every pair of nodes in ``W(s)`` gets its
  edge weight incremented by one.  This is *not* "number of sentences the pair
  shares a mention in" (that undercounts a pair that is close together but
  never in the exact same sentence) and it is *not* a literature-standard
  "scene" window (this codebase has no scene boundaries to build one from,
  only sentences and paragraphs) -- it is its own explicit, reproducible
  quantity, documented here once so every graph that calls
  :func:`cooccurrence_graph` means the same thing by "edge weight".
* **Community detection is Louvain, seeded, every time.**  ``networkx``'s
  ``louvain_communities`` is stochastic (it breaks modularity ties by a random
  tie-break), so an unseeded run is a different partition on every call and a
  document could not even be compared against itself between two grading
  runs.  The seed and algorithm name are recorded in every finding that uses
  this, per the project rule that determinism is not automatic and must be
  shown, not assumed.
* **Path statistics (average shortest path, diameter, radius) run on the
  largest connected component only, and only up to a node-count cap.**  These
  are undefined on a disconnected graph and O(n^3) (or O(n * (n+e)) for BFS
  from every node) on a connected one; a 2,000-character co-occurrence graph
  from a long series is not hypothetical here, so both bounds are load-bearing,
  not decorative.  Above the cap, this module reports that it was skipped
  rather than silently returning a value computed on a truncated graph.
* **Betweenness centrality is sampled once the graph is large enough that
  exact betweenness would be slow**, using ``networkx``'s own ``k=`` sampling
  parameter with a fixed seed, exactly the way ``k`` bounds it in the library
  itself.
* **A second, independent graph library is used as a disagreement channel,
  never to "double-check and average".**  Where ``igraph`` is installed, this
  module recomputes density, transitivity, connected-component count and a
  Louvain-style modularity/community count independently (igraph's own
  ``community_multilevel``, not the same code path as ``networkx``'s
  ``louvain_communities``) and reports both values side by side plus their
  absolute difference.  Two implementations of "the same statistic" landing on
  different numbers is real information about how sensitive that statistic is
  to implementation choices (tie-breaking, floating-point order of
  operations), not noise to reconcile -- see the project rule this suite
  follows throughout.  ``rustworkx`` is also installed and available for a
  future third channel; it is not wired in here because one independent
  cross-check already answers the "do two implementations agree" question,
  and a third library recomputing exactly what igraph already recomputes
  would not add new information, only more code to keep in sync.
"""

from __future__ import annotations

import math
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .optional import on_reset, require, shim_booknlp_transformers
from .spans import sentence_noun_chunks

GRAPH_SCHEMA_VERSION = "graph-suite-v1"

DEFAULT_COMMUNITY_SEED = 0
#: Above this many nodes, average-shortest-path/diameter/radius are skipped
#: rather than computed on a truncated sample: a book-length character network
#: can have hundreds of nodes and BFS-from-every-node is O(n*(n+e)); this is a
#: hard safety cap, not a target.
DEFAULT_PATH_NODE_CAP = 1500
#: Above this many nodes, betweenness centrality is estimated from a sample of
#: ``k`` source nodes (networkx's own supported parameter) rather than exactly.
DEFAULT_BETWEENNESS_EXACT_CAP = 200
DEFAULT_BETWEENNESS_SAMPLE_K = 100


# ------------------------------------------------------------ construction

def cooccurrence_graph(module: Any, present_at: Sequence[Sequence[Any]], window: int) -> Any:
    """A weighted ``networkx.Graph`` from a per-unit (sentence) presence list.

    ``present_at[i]`` is the list/set of node identities present in unit
    ``i`` (a sentence, most often).  See the module docstring for the exact
    edge-weight definition.  ``module`` is the already ``optional.require``-d
    ``networkx`` module; this function never imports it itself.
    """

    graph = module.Graph()
    all_nodes = {node for entities in present_at for node in entities}
    graph.add_nodes_from(all_nodes)
    length = len(present_at)
    for start in range(length):
        window_nodes: set = set()
        for offset in range(window + 1):
            index = start + offset
            if index >= length:
                break
            window_nodes.update(present_at[index])
        ordered = sorted(window_nodes, key=str)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, weight=1)
    return graph


def present_at_from_rows(rows: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """``[[node, ...], ...]`` per unit, from a dense role-sequence mapping
    (the shape :func:`textgrader.coherence.grid_rows` already returns): a
    node is "present" in a unit when its role there is not ``"-"``."""

    length = len(next(iter(rows.values()))) if rows else 0
    present_at: list[list[str]] = [[] for _ in range(length)]
    for key, sequence in rows.items():
        for index, role in enumerate(sequence):
            if role != "-":
                present_at[index].append(key)
    return present_at


def aggregate_to_paragraphs(present_at: Sequence[Sequence[str]],
                            paragraph_sentence_counts: Sequence[int]) -> list[list[str]]:
    """Collapse a per-sentence presence list to a per-paragraph one.

    A node is present in a paragraph if it was present in any sentence of
    that paragraph.  ``paragraph_sentence_counts[i]`` is how many of
    ``present_at``'s consecutive entries belong to paragraph ``i`` (exactly
    :attr:`DocumentAnalysis.paragraph_sentence_counts`), so this never
    re-derives paragraph boundaries a second time.
    """

    out: list[list[str]] = []
    cursor = 0
    for count in paragraph_sentence_counts:
        seen: set[str] = set()
        for offset in range(count):
            if cursor + offset < len(present_at):
                seen.update(present_at[cursor + offset])
        out.append(sorted(seen))
        cursor += count
    return out


def character_mentions_by_sentence(sents_channels) -> list[list[str]]:
    """Per-sentence lists of PROPER-NOUN-headed noun-chunk mentions.

    This is the "character" channel's own extraction, deliberately narrower
    than :func:`textgrader.coherence.entity_mentions_by_sentence` (which
    tracks any noun- or proper-noun-headed chunk, so it also catches "the
    room" and "her mother's letter"): a proper name is the only signal this
    codebase has for "this is probably a character or named place" without
    guessing, so the character co-occurrence graph is restricted to it.  Keyed
    by the chunk root's surface text (not its lemma, and not case-folded): a
    proper noun's spelling is its identity here, unlike a common noun where
    lemma-folding "rooms"/"room" together is the right call.  A multi-word
    name's chunk root is usually its last token ("Long John Silver" -> chunk
    root "Silver"), so this under-merges compound names into their final
    word; no alias merging is attempted here since this suite only merges
    aliases on real coreference evidence (see the ``coreference`` feature),
    never by guessing from string shape.
    """

    per_sentence: list[list[str]] = []
    for sent, _channel, _offset in sents_channels:
        names: list[str] = []
        try:
            chunks = sentence_noun_chunks(sent)
        except Exception:  # pragma: no cover - defensive against parser edge cases
            chunks = []
        seen: set[str] = set()
        for chunk in chunks:
            root = chunk.root
            if root.pos_ != "PROPN":
                continue
            key = root.text.strip()
            if key and key not in seen:
                seen.add(key)
                names.append(key)
        per_sentence.append(names)
    return per_sentence


def paragraph_overlap_graph(module: Any, node_lists: Sequence[Sequence[str]], window: int) -> Any:
    """A weighted ``networkx.Graph`` whose NODES ARE PARAGRAPH INDICES.

    Unlike every other construction in this module (where nodes are entities
    or speakers), the paragraph entity-overlap graph's nodes are the
    paragraphs themselves: an edge links two paragraphs within ``window``
    paragraphs of each other, weighted by how many distinct entities they
    both mention.  Bounded to nearby paragraphs, not all pairs, for the same
    reason :func:`cooccurrence_graph` is windowed: a name that recurs across
    most of a long novel's paragraphs would otherwise make this an almost
    complete graph on the paragraph count, which is both slow (O(paragraphs^2))
    and not a useful topology (everything overlaps with everything).
    """

    graph = module.Graph()
    graph.add_nodes_from(range(len(node_lists)))
    sets = [set(nodes) for nodes in node_lists]
    for i in range(len(sets)):
        limit = min(i + window + 1, len(sets))
        for j in range(i + 1, limit):
            shared = len(sets[i] & sets[j])
            if shared:
                graph.add_edge(i, j, weight=shared)
    return graph


def turn_adjacency_graph(module: Any, speaker_sequence: Sequence[str]) -> tuple[Any, int]:
    """A weighted ``networkx.Graph`` of who-speaks-after-whom.

    An edge links two DIFFERENT speakers whose turns are adjacent in
    ``speaker_sequence`` (only turns with a recovered speaker are in the
    sequence at all; an untagged turn breaks adjacency rather than being
    treated as a speaker of its own -- see the caller for how the sequence is
    built).  Consecutive turns by the SAME speaker are not an edge (a graph
    has no undirected self-loops here); they are counted separately and
    returned as the second item, ``monologue_runs`` (how many times a speaker
    held the floor for two or more turns in a row), since that is a real
    fact about the exchange that an edge-only view would silently drop.
    """

    graph = module.Graph()
    graph.add_nodes_from(set(speaker_sequence))
    monologue_runs = 0
    for a, b in zip(speaker_sequence, speaker_sequence[1:]):
        if a == b:
            monologue_runs += 1
            continue
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += 1
        else:
            graph.add_edge(a, b, weight=1)
    return graph, monologue_runs


def chain_path_graph(module: Any, chains: Sequence[Sequence[int]], node_count: int) -> Any:
    """A ``networkx.Graph`` over unit (sentence) indices from lexical chains.

    Each chain (a list of unit indices sharing a repeated content word,
    :func:`textgrader.coherence.lexical_chains`'s output) contributes an edge
    between each CONSECUTIVE pair of its members, weighted by how many chains
    link that pair -- not every pair within the chain, which would turn an
    n-member chain into a clique of `n choose 2` edges and make a single long
    chain dominate the whole graph's edge count.  Isolated units (mentioned in
    no chain of the configured minimum length) are still added as nodes, so
    the graph's node count is the same "how much text" quantity every other
    finding in this suite uses (all sentences), not just the covered ones.
    """

    graph = module.Graph()
    graph.add_nodes_from(range(node_count))
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += 1
            else:
                graph.add_edge(a, b, weight=1)
    return graph


_NAME_STOPWORDS = frozenset({
    "The", "A", "An", "He", "She", "They", "It", "I", "We", "You", "And",
    "But", "So", "Then", "When", "As", "With", "At", "In", "On", "If",
    "That", "This", "There", "Here", "Who", "What", "Yes", "No", "Oh",
    "Mr", "Mrs", "Dr", "Miss", "Chapter", "Book", "Part",
})
#: One to three capitalized words in a row, not sentence-initial (the token
#: must be preceded by a lower-case word, a comma/semicolon/colon, or a
#: closing quotation mark, none of which end a sentence), so "The dog
#: barked." does not nominate "The" as a name.  This is a heuristic surface
#: proxy for a proper name, not NER: it will catch "Long John Silver" and
#: "Aunt Polly" and it will also catch a capitalized word placed for emphasis
#: or a title-cased heading fragment that survived cleanup.  Allowing a
#: closing quotation mark right before the name is deliberate and a real
#: source of false positives -- dialogue attribution is exactly where a name
#: most often follows a comma ("Come here," Alice said.), but a sentence that
#: begins immediately after a closing quotation ("Stop." Alice walked in.)
#: will also match, incorrectly, since this regex cannot tell a
#: sentence-ending quote from a mid-utterance one.  It exists specifically to
#: answer whether a "who talks to whom" graph can be built at all WITHOUT a
#: spaCy parse (see the module docstring's "cheap graphs" note); the
#: parse-backed entity/character graphs in graph_suite.py are the more
#: trustworthy channel and this is reported alongside them, under its own
#: metric ids, not in place of them.
_NAME_RE = re.compile(r"(?<=[a-z,;:\"'’”]\s)([A-Z][a-z']+(?:\s[A-Z][a-z']+){0,2})")


def surface_name_sequence(sentences: Sequence[str]) -> list[list[str]]:
    """Per-sentence lists of surface, non-sentence-initial capitalized names.

    Dependency-free: works even where ``analysis.nlp_unavailable`` is true.
    See :data:`_NAME_RE` for exactly what counts as a name candidate and why.
    """

    out: list[list[str]] = []
    for sentence in sentences:
        found = []
        for match in _NAME_RE.finditer(sentence):
            name = match.group(1)
            first_word = name.split()[0]
            if first_word in _NAME_STOPWORDS:
                continue
            found.append(name)
        out.append(found)
    return out


# ------------------------------------------------------------------ recurrence

def reappearance_gaps(present_at: Sequence[Sequence[Any]]) -> list[int]:
    """Gaps (in units) between successive appearances of the same node.

    Generic over any co-occurrence graph's ``present_at`` list: works
    identically for the entity, character, coreference and surface-name
    graphs.  This is the graph-level twin of
    :func:`textgrader.coherence.transition_counts`'s reintroduction-distance
    idea, generalized past the closed S/O/X/absent role schema to plain node
    presence, since a co-occurrence graph does not track grammatical role.
    """

    last_seen: dict[Any, int] = {}
    gaps: list[int] = []
    for index, nodes in enumerate(present_at):
        for node in nodes:
            if node in last_seen:
                gaps.append(index - last_seen[node])
            last_seen[node] = index
    return gaps


# -------------------------------------------------------------------- stats

def gini(values: Sequence[float]) -> float | None:
    """Gini coefficient of a non-negative sample; ``0`` is perfectly equal,
    approaching ``1`` is maximally concentrated in one item."""

    numbers = sorted(value for value in values if value is not None and value >= 0)
    n = len(numbers)
    if n == 0:
        return None
    total = sum(numbers)
    if total <= 0:
        return 0.0
    index_sum = sum((i + 1) * value for i, value in enumerate(numbers))
    return (2 * index_sum) / (n * total) - (n + 1) / n


def degree_centralization(graph: Any) -> float | None:
    """Freeman's degree centralization: 1.0 for a perfect star, 0.0 for a
    regular graph (every node the same degree, e.g. a cycle or a complete
    graph).  ``None`` below 3 nodes, where the normalizing denominator is 0."""

    n = graph.number_of_nodes()
    if n < 3:
        return None
    degrees = [d for _, d in graph.degree()]
    max_degree = max(degrees)
    numerator = sum(max_degree - d for d in degrees)
    denominator = (n - 1) * (n - 2)
    return numerator / denominator if denominator else None


def betweenness_centralization(module: Any, graph: Any, *, seed: int,
                               exact_cap: int = DEFAULT_BETWEENNESS_EXACT_CAP,
                               sample_k: int = DEFAULT_BETWEENNESS_SAMPLE_K
                               ) -> tuple[float | None, dict[str, Any]]:
    """Freeman's betweenness centralization, on normalized betweenness.

    Exact for graphs up to ``exact_cap`` nodes; above that, ``networkx``'s own
    ``k=`` sampling (a fixed, seeded random sample of source nodes) is used,
    and the settings say so, since a sampled betweenness centralization is not
    exactly the same quantity as the exact one and must never be compared as
    though it were.
    """

    n = graph.number_of_nodes()
    if n < 3:
        return None, {"method": "undefined", "nodes": n}
    sampled = n > exact_cap
    k = min(sample_k, n) if sampled else None
    values = module.betweenness_centrality(graph, k=k, seed=seed if sampled else None,
                                           normalized=True, weight=None)
    scores = list(values.values())
    max_score = max(scores)
    denominator = n - 1  # max possible sum of (max - score) for a star, normalized scores
    centralization = sum(max_score - score for score in scores) / denominator if denominator else None
    settings = {"method": "sampled" if sampled else "exact", "sample_k": k, "seed": seed if sampled else None,
               "nodes": n}
    return centralization, settings


def community_detection(module: Any, graph: Any, *, seed: int = DEFAULT_COMMUNITY_SEED,
                        weighted: bool = True) -> dict[str, Any]:
    """Louvain communities, modularity, and community-size entropy, seeded.

    Returns ``{"communities": [...], "modularity": float|None,
    "community_count": int, "community_size_entropy": float|None,
    "algorithm": str, "seed": int}``.  A graph with fewer than 2 nodes or no
    edges has no meaningful partition; ``communities`` is then one singleton
    set per node (or empty), ``modularity`` is ``None``, and the algorithm
    name still records what would have run.
    """

    algorithm = "networkx.algorithms.community.louvain_communities"
    n = graph.number_of_nodes()
    if n == 0 or graph.number_of_edges() == 0:
        communities = [{node} for node in graph.nodes()]
        return {"communities": communities, "modularity": None,
               "community_count": len(communities), "community_size_entropy": None,
               "algorithm": algorithm, "seed": seed}
    weight_key = "weight" if weighted else None
    communities = module.algorithms.community.louvain_communities(
        graph, weight=weight_key, seed=seed)
    modularity = module.algorithms.community.modularity(graph, communities, weight=weight_key)
    sizes = [len(c) for c in communities]
    total = sum(sizes)
    entropy = (-sum((s / total) * math.log2(s / total) for s in sizes if s) if total else None)
    return {"communities": communities, "modularity": modularity,
           "community_count": len(communities), "community_size_entropy": entropy,
           "algorithm": algorithm, "seed": seed}


def bounded_path_stats(module: Any, graph: Any, *,
                       node_cap: int = DEFAULT_PATH_NODE_CAP) -> dict[str, Any]:
    """Average shortest path, diameter and radius on the LARGEST connected
    component only (undefined on a disconnected graph), skipped above
    ``node_cap`` nodes rather than computed on a truncated sample.  Returns
    the component's own size and coverage share regardless, since those are
    cheap and always meaningful."""

    n = graph.number_of_nodes()
    if n == 0:
        return {"largest_component_nodes": 0, "largest_component_share": None,
               "average_shortest_path_length": None, "diameter": None, "radius": None,
               "skipped_reason": "no nodes"}
    components = list(module.connected_components(graph))
    largest = max(components, key=len)
    coverage = len(largest) / n
    if len(largest) < 2:
        return {"largest_component_nodes": len(largest), "largest_component_share": coverage,
               "average_shortest_path_length": None, "diameter": None, "radius": None,
               "skipped_reason": "largest component has fewer than two nodes"}
    if len(largest) > node_cap:
        return {"largest_component_nodes": len(largest), "largest_component_share": coverage,
               "average_shortest_path_length": None, "diameter": None, "radius": None,
               "skipped_reason": f"largest component has {len(largest)} nodes, above the "
                                 f"{node_cap}-node cap for exact path statistics"}
    sub = graph.subgraph(largest)
    return {"largest_component_nodes": len(largest), "largest_component_share": coverage,
           "average_shortest_path_length": module.average_shortest_path_length(sub),
           "diameter": module.diameter(sub), "radius": module.radius(sub),
           "skipped_reason": None}


def graph_statistics(module: Any, graph: Any, *, seed: int = DEFAULT_COMMUNITY_SEED,
                     weighted: bool = True) -> dict[str, Any]:
    """The full network-statistics battery for one graph, in one dict.

    Every graph type in this suite reports this same shape, so a reader never
    has to remember which stats a particular graph gives -- see the module
    docstring for what each judgement call inside here means.
    """

    n = graph.number_of_nodes()
    e = graph.number_of_edges()
    if n == 0:
        return {"nodes": 0, "edges": 0, "density": None, "average_degree": None,
               "degree_variance": None, "degree_entropy": None,
               "largest_component_nodes": 0, "largest_component_share": None,
               "connected_components": 0, "isolate_share": None,
               "average_clustering": None, "transitivity": None, "assortativity": None,
               "average_shortest_path_length": None, "diameter": None, "radius": None,
               "degree_centralization": None, "betweenness_centralization": None,
               "pagerank_gini": None, "edge_weight_entropy": None,
               "community_count": 0, "modularity": None, "community_size_entropy": None,
               "community_algorithm": None, "community_seed": seed}

    degrees = [d for _, d in graph.degree()]
    degree_mean = sum(degrees) / n
    degree_variance = statistics.pvariance(degrees) if n > 1 else 0.0
    degree_counts: dict[int, int] = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    degree_entropy = -sum((c / n) * math.log2(c / n) for c in degree_counts.values() if c)
    isolate_count = sum(1 for d in degrees if d == 0)

    path = bounded_path_stats(module, graph)
    community = community_detection(module, graph, seed=seed, weighted=weighted)

    # A regular graph (every node the same degree) makes the Pearson
    # correlation's denominator zero; networkx computes it via scipy and that
    # emits a ConstantInputWarning rather than raising, so the degenerate case
    # is checked for directly instead of relying on an exception that never
    # comes.
    if degree_variance == 0:
        assortativity = None
    else:
        try:
            assortativity = module.degree_pearson_correlation_coefficient(graph)
            if assortativity is not None and not math.isfinite(assortativity):
                assortativity = None
        except Exception:  # pragma: no cover - defensive
            assortativity = None

    degree_central = degree_centralization(graph)
    betweenness_central, betweenness_settings = betweenness_centralization(module, graph, seed=seed)

    try:
        pagerank = module.pagerank(graph, weight="weight" if weighted else None)
        pagerank_gini = gini(list(pagerank.values()))
    except Exception:  # pragma: no cover - non-convergence on a pathological graph
        pagerank_gini = None

    weights = [data.get("weight", 1) for _, _, data in graph.edges(data=True)] if weighted else []
    weight_counts: dict[Any, int] = {}
    for w in weights:
        weight_counts[w] = weight_counts.get(w, 0) + 1
    edge_weight_entropy = (-sum((c / len(weights)) * math.log2(c / len(weights))
                                for c in weight_counts.values() if c)
                          if weights else None)

    return {
        "nodes": n, "edges": e,
        "density": module.density(graph),
        "average_degree": degree_mean, "degree_variance": degree_variance,
        "degree_entropy": degree_entropy,
        "largest_component_nodes": path["largest_component_nodes"],
        "largest_component_share": path["largest_component_share"],
        "connected_components": module.number_connected_components(graph),
        "isolate_share": isolate_count / n,
        "average_clustering": module.average_clustering(graph, weight="weight" if weighted else None),
        "transitivity": module.transitivity(graph),
        "assortativity": assortativity,
        "average_shortest_path_length": path["average_shortest_path_length"],
        "diameter": path["diameter"], "radius": path["radius"],
        "path_skipped_reason": path["skipped_reason"],
        "degree_centralization": degree_central,
        "betweenness_centralization": betweenness_central,
        "betweenness_settings": betweenness_settings,
        "pagerank_gini": pagerank_gini,
        "edge_weight_entropy": edge_weight_entropy,
        "community_count": community["community_count"], "modularity": community["modularity"],
        "community_size_entropy": community["community_size_entropy"],
        "community_algorithm": community["algorithm"], "community_seed": community["seed"],
    }


# ---------------------------------------------------------- library disagreement

_IGRAPH_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_igraph_cache() -> None:
    _IGRAPH_CACHE.clear()


on_reset(_reset_igraph_cache)


def igraph_cross_check(graph: Any, *, weighted: bool = True) -> dict[str, Any] | None:
    """Recompute density/transitivity/components/modularity with ``igraph``,
    independently of the ``networkx`` code path above, and return both sides
    plus their absolute difference.  ``{"available": False, ...}`` when
    ``igraph`` is not installed -- the caller folds that into its own
    finding's warning rather than failing, exactly like every other optional
    channel in this project.  igraph's ``community_multilevel`` (Louvain) has
    no seed parameter to fix, unlike ``networkx``'s ``louvain_communities``:
    its tie-breaking is deterministic given the same edge order, which this
    function always builds in the same way from ``graph.edges()``, so this
    channel is still reproducible even without an explicit seed.
    """

    module, reason = require("igraph")
    if module is None:
        return {"available": False, "reason": reason}
    nodes = list(graph.nodes())
    if not nodes:
        return {"available": True, "nodes": 0, "edges": 0}
    index = {node: i for i, node in enumerate(nodes)}
    edges = [(index[a], index[b]) for a, b in graph.edges()]
    weights = [graph[a][b].get("weight", 1) for a, b in graph.edges()] if weighted else None
    ig_graph = module.Graph(n=len(nodes), edges=edges)
    if weights:
        ig_graph.es["weight"] = weights
    density = ig_graph.density()
    transitivity = ig_graph.transitivity_undirected(mode="zero")
    components = len(ig_graph.connected_components())
    modularity = community_count = None
    if edges:
        try:
            clustering = ig_graph.community_multilevel(weights="weight" if weights else None)
            modularity, community_count = clustering.modularity, len(clustering)
        except Exception:  # pragma: no cover - disconnected/degenerate graph guard
            modularity = community_count = None
    return {
        "available": True, "library": "igraph", "version": getattr(module, "__version__", None),
        "nodes": len(nodes), "edges": len(edges),
        "density": density, "transitivity": transitivity, "connected_components": components,
        "modularity": modularity, "community_count": community_count,
        "community_algorithm": "igraph.Graph.community_multilevel (Louvain)",
    }


def disagreement(networkx_value: float | None, igraph_value: float | None) -> float | None:
    if networkx_value is None or igraph_value is None:
        return None
    return abs(networkx_value - igraph_value)


# -------------------------------------------------------------------- BookNLP

#: BookNLP's small English model: fast enough to bound with a word cap (see
#: graph_suite.py's ``booknlp_max_words``); the "big" model exists but is not
#: offered as a default here since the small model already costs roughly a
#: second per 40 words on CPU (measured; see graph_suite.py's docstring).
DEFAULT_BOOKNLP_MODEL_SIZE = "small"
DEFAULT_BOOKNLP_PIPELINE = "entity,quote,supersense,event,coref"

_BOOKNLP_CACHE: dict[tuple[str, str], tuple[Any, str | None]] = {}


def _reset_booknlp_cache() -> None:
    _BOOKNLP_CACHE.clear()


on_reset(_reset_booknlp_cache)


def _load_booknlp(model_size: str, pipeline: str) -> tuple[Any, str | None]:
    """Cached across every call in this process, exactly like
    :func:`textgrader.coherence._load_rst_parser`: constructing ``BookNLP``
    loads several transformer checkpoints (entity tagger, coref, speaker
    attribution), which took ~16-22s and held ~880MB resident when measured
    against this project's small English model -- see graph_suite.py's
    docstring for the exact numbers -- so it must happen at most once per
    process no matter how many documents ask for it.
    """

    key = (model_size, pipeline)
    if key in _BOOKNLP_CACHE:
        return _BOOKNLP_CACHE[key]
    module, reason = require("booknlp")
    if module is None:
        _BOOKNLP_CACHE[key] = (None, reason)
        return _BOOKNLP_CACHE[key]
    try:
        with shim_booknlp_transformers():
            model = module.BookNLP("en", {"pipeline": pipeline, "model": model_size})
        outcome: tuple[Any, str | None] = (model, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"BookNLP ({model_size!r} model) unavailable "
                         f"({type(exc).__name__}: {exc}); pip install --no-deps booknlp (its "
                         f"declared dependencies pull in an unused ~570MB tensorflow install and "
                         f"upgrade numpy; --no-deps avoids both -- see requirements-embeddings.txt)")
    _BOOKNLP_CACHE[key] = outcome
    return outcome


def _read_tsv(path: Path) -> list[dict[str, str]]:
    import csv
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def resolve_booknlp(text: str, max_words: int, model_size: str = DEFAULT_BOOKNLP_MODEL_SIZE,
                    pipeline: str = DEFAULT_BOOKNLP_PIPELINE
                    ) -> tuple[list[list[str]], list[str], dict[str, Any], str | None]:
    """Run real BookNLP over a bounded word-count prefix of ``text``.

    Returns ``(character_present_at, speaker_sequence, settings, note)``:
    ``character_present_at`` is a per-sentence list of BookNLP character ids
    (as strings; BookNLP's own coreference-resolved ``PER``-category entity
    clusters, from its ``.entities`` output, one row per mention), in
    BookNLP's own sentence segmentation (never the shared spaCy or pySBD one
    -- BookNLP re-tokenizes and re-sentences internally, so its sentence
    indices are its own and are not comparable position-for-position to
    ``analysis.sentences``).  ``speaker_sequence`` is the ordered list of
    character ids BookNLP's own quote-attribution assigned to each detected
    quotation (from ``.quotes``), for the turn-adjacency graph.

    Always bounded to a PREFIX of the document, the same reason
    :func:`textgrader.coherence.resolve_coreference` bounds fastcoref: at
    roughly 40 words/second on CPU for the small model (measured: 66s of
    processing, after a one-time ~20s model load, for a 2,500-word excerpt of
    a Project Gutenberg novel -- see graph_suite.py's docstring), running this
    on a 300,000-word novel would take on the order of two hours. ``settings``
    records exactly how much text was actually sent to BookNLP and how long it
    took; ``note`` is always set, either why BookNLP could not run at all or a
    description of the prefix and timing, so a finding produced this way can
    never be mistaken for a whole-book analysis.
    """

    import time

    words = text.split()
    prefix_words = words[:max(1, max_words)]
    prefix_text = " ".join(prefix_words)
    settings = {"backend": "booknlp", "model": model_size, "pipeline": pipeline,
               "words_considered": len(prefix_words), "word_cap": max_words,
               "words_in_document": len(words)}
    model, reason = _load_booknlp(model_size, pipeline)
    if model is None:
        return [], [], settings, reason
    if not prefix_words:
        return [], [], settings, "no words in the document to send to BookNLP"

    with tempfile.TemporaryDirectory(prefix="textgrader_booknlp_") as tmp:
        input_path = Path(tmp) / "input.txt"
        input_path.write_text(prefix_text, encoding="utf-8")
        book_id = "book"
        started = time.monotonic()
        try:
            model.process(str(input_path), tmp, book_id)
        except Exception as exc:  # pragma: no cover - runtime/OOM failure
            return [], [], settings, f"BookNLP processing failed ({type(exc).__name__}: {exc})"
        elapsed = time.monotonic() - started
        settings["elapsed_seconds"] = round(elapsed, 1)

        entities = _read_tsv(Path(tmp) / f"{book_id}.entities")
        tokens = _read_tsv(Path(tmp) / f"{book_id}.tokens")
        quotes = _read_tsv(Path(tmp) / f"{book_id}.quotes")

    token_to_sentence: dict[int, int] = {}
    max_sentence = -1
    for row in tokens:
        try:
            token_id = int(row["token_ID_within_document"])
            sentence_id = int(row["sentence_ID"])
        except (KeyError, ValueError):
            continue
        token_to_sentence[token_id] = sentence_id
        max_sentence = max(max_sentence, sentence_id)

    present_at: list[list[str]] = [[] for _ in range(max_sentence + 1)]
    for row in entities:
        if row.get("cat") != "PER":
            continue
        try:
            start_token = int(row["start_token"])
            char_id = row["COREF"]
        except (KeyError, ValueError):
            continue
        sentence_id = token_to_sentence.get(start_token)
        if sentence_id is not None:
            present_at[sentence_id].append(char_id)

    speaker_sequence: list[str] = []
    for row in sorted(quotes, key=lambda r: int(r.get("quote_start", 0) or 0)):
        char_id = row.get("char_id")
        if char_id and char_id != "-1":
            speaker_sequence.append(char_id)

    settings["sentences_seen"] = max_sentence + 1
    settings["character_mentions"] = sum(len(nodes) for nodes in present_at)
    settings["quotes_with_a_speaker"] = len(speaker_sequence)
    note = (f"backend=booknlp: BookNLP ({model_size!r} model) processed the first "
           f"{len(prefix_words):,} of this document's {len(words):,} words (cap "
           f"{max_words:,}) in {settings['elapsed_seconds']:.1f}s; character and speaker ids are "
           f"BookNLP's own coreference clusters, sentenced by BookNLP's own segmenter, not the "
           f"shared spaCy/pySBD one this suite's other channels use, and this is a SAMPLE, not a "
           f"whole-book analysis")
    return present_at, speaker_sequence, settings, note


__all__ = [
    "GRAPH_SCHEMA_VERSION", "cooccurrence_graph", "present_at_from_rows",
    "aggregate_to_paragraphs", "character_mentions_by_sentence", "paragraph_overlap_graph",
    "turn_adjacency_graph", "chain_path_graph",
    "surface_name_sequence", "reappearance_gaps", "gini", "degree_centralization",
    "betweenness_centralization", "community_detection", "bounded_path_stats",
    "graph_statistics", "igraph_cross_check", "disagreement",
    "DEFAULT_BOOKNLP_MODEL_SIZE", "DEFAULT_BOOKNLP_PIPELINE", "resolve_booknlp",
]
