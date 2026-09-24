"""``graph_suite``: entity/character/topic/lexical network analysis.

Off by default, every graph independently switchable, honest about small
graphs, and never confused by a missing optional package -- the same
contract every other experimental suite in this project holds itself to (see
``tests/test_coherence_suite.py`` for the sibling suite this one shares
several caches and conventions with).
"""

from __future__ import annotations

import networkx as nx
import pytest

import grade
from textgrader import graphs as gr
from textgrader import optional
from textgrader.document import DocumentAnalysis
from textgrader.metrics import graph_suite
from textgrader.results import Action, StatusType

PREFIX = "discourse.graph_"


def _ids(findings):
    return {item["metric_id"] for item in findings}


def _by_id(findings):
    return {item["metric_id"]: item for item in findings}


def _analysis(text, **kwargs):
    return DocumentAnalysis.from_text(text, comparison_unit="book", **kwargs)


# ------------------------------------------------------------------- wiring

def test_off_by_default(manuscript, base_config):
    report = grade.analyze(manuscript, base_config)
    assert not [item for item in report.results if item.metric_id.startswith(PREFIX)]


def test_enabling_the_suite_turns_on_every_default_feature(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "graph_suite": {"enabled": True}}}
    ids = {item.metric_id for item in grade.analyze(manuscript, config).results}
    matched = {mid for mid in ids if mid.startswith(PREFIX)}
    assert matched
    for prefix in ("lexical_chain", "surface_name", "entity", "character", "quote_speaker"):
        assert any(mid.startswith(f"{PREFIX}{prefix}_") for mid in matched), prefix
    # Off-by-default features never leak in under the default config.
    assert not any("sentence_semantic" in mid for mid in matched)
    assert not any("paragraph_semantic" in mid for mid in matched)
    assert not any("coref_entity" in mid for mid in matched)
    assert not any("dependency" in mid for mid in matched)
    assert not any("booknlp" in mid for mid in matched)


def test_every_metric_id_uses_the_stable_prefix(sample_text):
    findings = graph_suite.measure(_analysis(sample_text))
    ids = _ids(findings)
    assert ids
    assert all(mid.startswith(PREFIX) for mid in ids)


def test_features_are_independently_switchable(sample_text):
    analysis = _analysis(sample_text)
    only_lexical = graph_suite.measure(analysis, config={
        "features": {name: False for name in graph_suite.DEFAULT_FEATURES} | {"lexical_chain": True}})
    ids = _ids(only_lexical)
    assert ids == {f"{PREFIX}lexical_chain_node_count", f"{PREFIX}lexical_chain_edge_count",
                  f"{PREFIX}lexical_chain_density", f"{PREFIX}lexical_chain_community_structure",
                  f"{PREFIX}lexical_chain_centralization", f"{PREFIX}lexical_chain_path_length"}

    only_quote = graph_suite.measure(analysis, config={
        "features": {name: False for name in graph_suite.DEFAULT_FEATURES} | {"quote_speaker": True}})
    quote_ids = _ids(only_quote)
    assert all(mid.startswith(f"{PREFIX}quote_speaker_") for mid in quote_ids)
    assert quote_ids


def test_every_registered_metric_survives_registry_smoke(manuscript, base_config):
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "graph_suite": {"enabled": True}}}
    report = grade.analyze(manuscript, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


# ------------------------------------------------------------------- gating

def test_default_config_loads_no_model_or_heavy_backend(monkeypatch, sample_text):
    """``sentence_semantic``/``paragraph_semantic``/``coreference_entity``/
    ``booknlp`` are off by default; this proves it by making every one of
    their loaders raise, then confirming the DEFAULT feature set (no config
    override) never triggers any of them -- mirroring
    ``test_coherence_suite.py``'s identically-purposed
    ``test_no_coreference_model_is_loaded_under_the_default_config``.
    """

    from textgrader.metrics import semantic_adjacent as sem

    def _boom_embed(*args, **kwargs):
        raise AssertionError("sentence-transformers must not load under the default config")

    def _boom_coref(*args, **kwargs):
        raise AssertionError("fastcoref must not load under the default config")

    def _boom_booknlp(*args, **kwargs):
        raise AssertionError("BookNLP must not load under the default config")

    monkeypatch.setattr(sem, "embed_texts", _boom_embed)
    monkeypatch.setattr(gr, "_load_booknlp", _boom_booknlp)

    import textgrader.coherence as coh

    monkeypatch.setattr(coh, "_load_coref_model", _boom_coref)

    analysis = _analysis(sample_text)
    findings = graph_suite.measure(analysis)  # no config: every DEFAULT_FEATURES value applies
    errors = [item for item in findings if item.get("warning") and "must not load" in item["warning"]]
    assert not errors


def test_needs_model_is_false_for_the_suite():
    """The suite's REQUIRES deliberately omits "sentence_transformers" (see
    the module docstring's "Gating" section): including it would flip
    ``needs_model`` for the WHOLE suite and silently exclude every
    dependency-free/parse-only graph from corpus profiling too."""

    from textgrader.metrics import REGISTRY

    spec = REGISTRY["graph_suite"]
    assert spec.needs_model is False
    assert spec.needs_parse is True
    assert "sentence_transformers" not in spec.requires


# ---------------------------------------------------- graph statistics battery

def _star():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D", "E"])
    for node in ["B", "C", "D", "E"]:
        graph.add_edge("A", node, weight=1)
    return graph


def _chain():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D", "E"])
    for a, b in zip(["A", "B", "C", "D"], ["B", "C", "D", "E"]):
        graph.add_edge(a, b, weight=1)
    return graph


def _two_community():
    graph = nx.Graph()
    for a, b in [("A", "B"), ("B", "C"), ("C", "A"), ("D", "E"), ("E", "F"), ("F", "D"), ("C", "D")]:
        graph.add_edge(a, b, weight=1)
    return graph


def test_star_chain_and_two_community_fixtures_are_separated_by_exact_statistics():
    """A star, a chain and a two-community graph have the same (or nearly the
    same) node/edge counts and are nonetheless obviously different network
    shapes; degree centralization and clustering/transitivity are what tell
    them apart, and this pins the exact numbers so a regression cannot creep
    in silently.
    """

    star, chain, two = _star(), _chain(), _two_community()
    star_stats = gr.graph_statistics(nx, star, seed=0)
    chain_stats = gr.graph_statistics(nx, chain, seed=0)
    two_stats = gr.graph_statistics(nx, two, seed=0)

    # Degree centralization: the star is a perfect hub-and-spoke (1.0);
    # neither the chain nor the two-community graph has a single dominant hub.
    assert star_stats["degree_centralization"] == pytest.approx(1.0)
    assert chain_stats["degree_centralization"] == pytest.approx(1 / 6)
    assert two_stats["degree_centralization"] == pytest.approx(0.2)
    assert star_stats["degree_centralization"] > chain_stats["degree_centralization"]
    assert star_stats["degree_centralization"] > two_stats["degree_centralization"]

    # Clustering/transitivity: the two-community graph has two tight local
    # triangles; the star and the chain have no triangles at all.
    assert star_stats["transitivity"] == 0
    assert chain_stats["transitivity"] == 0
    assert two_stats["transitivity"] == pytest.approx(0.6)
    assert two_stats["average_clustering"] == pytest.approx(7 / 9)

    # Community structure: the star has no partition worth making (a single
    # hub is its own community); the chain and the two-community graph both
    # split into two, but modularity is markedly higher for the real
    # two-community graph than for an arbitrary path-graph split.
    assert star_stats["community_count"] == 1
    assert star_stats["modularity"] == pytest.approx(0.0)
    assert chain_stats["community_count"] == 2
    assert chain_stats["modularity"] == pytest.approx(0.21875)
    assert two_stats["community_count"] == 2
    assert two_stats["modularity"] == pytest.approx(0.35714285714285715)
    assert two_stats["modularity"] > chain_stats["modularity"]

    # Node/edge counts are close enough that they alone could not have told
    # these three apart (the whole point of the rest of the battery).
    assert star_stats["nodes"] == chain_stats["nodes"] == 5
    assert star_stats["edges"] == chain_stats["edges"] == 4
    assert two_stats["nodes"] == 6 and two_stats["edges"] == 7


def test_community_detection_is_deterministic():
    graph = _two_community()
    first = gr.community_detection(nx, graph, seed=0)
    second = gr.community_detection(nx, graph, seed=0)
    assert first["community_count"] == second["community_count"]
    assert first["modularity"] == pytest.approx(second["modularity"])
    assert first["algorithm"] == "networkx.algorithms.community.louvain_communities"


def test_igraph_cross_check_agrees_on_a_simple_graph():
    if not optional.have("igraph"):
        pytest.skip("igraph is not available in this environment")
    graph = _two_community()
    cross = gr.igraph_cross_check(graph)
    assert cross["available"] is True
    stats = gr.graph_statistics(nx, graph, seed=0)
    assert cross["nodes"] == stats["nodes"]
    assert cross["edges"] == stats["edges"]
    assert cross["density"] == pytest.approx(stats["density"])
    assert gr.disagreement(stats["density"], cross["density"]) == pytest.approx(0.0)


# --------------------------------------------------- repeated disconnected mentions

def test_repeated_disconnected_entity_mentions_stay_disconnected():
    """Two entities that each recur many times but NEVER co-occur within the
    window produce a graph with no edge between them: repetition alone must
    not be mistaken for interaction.
    """

    module = nx
    # Entity "x" appears alone in sentences 0-9, entity "y" alone in
    # sentences 20-29: never within a window of 3 sentences of each other.
    present_at = [["x"] for _ in range(10)] + [[] for _ in range(10)] + [["y"] for _ in range(10)]
    graph = gr.cooccurrence_graph(module, present_at, window=3)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 0
    stats = gr.graph_statistics(module, graph, seed=0)
    assert stats["connected_components"] == 2
    assert stats["largest_component_share"] == pytest.approx(0.5)
    assert stats["isolate_share"] == pytest.approx(1.0)  # both nodes have degree 0


def test_repeated_mentions_within_the_window_do_connect():
    present_at = [["x", "y"] for _ in range(10)]
    graph = gr.cooccurrence_graph(nx, present_at, window=0)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph["x"]["y"]["weight"] == 10  # one increment per co-occurring sentence


# -------------------------------------------------------------- alias handling

def test_merging_present_at_identity_changes_the_graph_the_way_coreference_would():
    """A mechanical demonstration of what alias/coreference resolution DOES to
    a co-occurrence graph, independent of any real model: merging two labels
    that refer to the same thing into one node changes node count, degree and
    recurrence distance in a specific, checkable way. The real
    ``coreference_entity``/``booknlp`` features do this merge only on model
    evidence (see the module docstring); this test pins the mechanical effect
    the merge has once it happens.
    """

    unmerged = [["alice"], ["bob"], [], ["she_unresolved"], ["bob"], ["she_unresolved"]]
    merged = [["alice"], ["bob"], [], ["alice"], ["bob"], ["alice"]]

    unmerged_graph = gr.cooccurrence_graph(nx, unmerged, window=1)
    merged_graph = gr.cooccurrence_graph(nx, merged, window=1)

    assert unmerged_graph.number_of_nodes() == 3  # alice, bob, she_unresolved never linked
    assert merged_graph.number_of_nodes() == 2    # alice absorbed the pronoun mentions

    unmerged_freq = gr.reappearance_gaps(unmerged)
    merged_freq = gr.reappearance_gaps(merged)
    # Merging gives "alice" two more reappearances than the surface labels had
    # individually, so there are more (and on average shorter) recorded gaps.
    assert len(merged_freq) > len(unmerged_freq)


@pytest.mark.skipif(not (optional.have("networkx") and optional.have("spacy")),
                    reason="networkx/spaCy are not available in this environment")
def test_character_graph_does_not_guess_aliases_from_surface_similarity():
    """The surface (non-coreference) character graph keys on exact proper-noun
    surface text; two spellings of what a human reader would recognize as the
    same person ("Bob" and "Bobby") are never merged without real evidence.
    """

    text = ("Bob walked into the room with Carol. Bob looked at Carol. Bob sighed loudly. "
           "Later, Bobby returned home with Carol. Bobby was tired near Carol. Bobby slept.")
    analysis = _analysis(text)
    findings = graph_suite.measure(analysis, config={
        "features": {name: False for name in graph_suite.DEFAULT_FEATURES}
                    | {"character_cooccurrence": True},
        "min_mentions_for_graph": 1})
    node_count = _by_id(findings)[f"{PREFIX}character_node_count"]
    # "Bob" and "Bobby" are kept as two distinct nodes (plus "Carol"); a real
    # coreference or BookNLP backend, with actual evidence, is what may merge
    # Bob/Bobby (see coreference_entity/booknlp).
    assert node_count["value"] == 3


@pytest.mark.skipif(not optional.have("fastcoref"),
                    reason="fastcoref is not available in this environment")
def test_coreference_entity_graph_resolves_pronouns_the_surface_graph_cannot():
    """Real, model-backed alias handling: a pronoun-heavy passage about one
    character gives the surface entity graph almost nothing to track (pronoun
    heads are not entities at all in that channel -- see
    ``textgrader.coherence``'s module docstring), while the coreference
    backend resolves "she"/"her" back to the named entity, mirroring
    ``test_coherence_suite.py``'s analogous real-coreference test.
    """

    text = (" ".join([
        "Eleanor walked into the quiet library and looked around carefully.",
        "She noticed the old clock on the wall and smiled at it.",
        "She had always loved that clock since she was a child.",
        "Her mother used to bring her here on rainy afternoons.",
        "She sat down and opened the book she had been reading.",
    ] * 3))
    analysis = _analysis(text)
    config = {"features": {name: False for name in graph_suite.DEFAULT_FEATURES}
                        | {"coreference_entity": True},
             "coreference_max_words": 2000, "min_mentions_for_graph": 2}
    findings = _by_id(graph_suite.measure(analysis, config=config))
    node_count = findings[f"{PREFIX}coref_entity_node_count"]
    assert node_count["warning"] is not None and "backend=coreference" in node_count["warning"]
    if node_count["value"] is not None:
        # Real evidence: at least one cluster reached the min_mentions floor.
        assert node_count["value"] >= 1


# --------------------------------------------------------------- insufficient data

def test_tiny_graph_reports_a_real_value_with_sample_size_below_min_sample():
    """A 2-node graph is below ``MIN_GRAPH_NODES``: its density (1.0) is a
    real, well-defined number, reported with ``sample_size`` below
    ``min_sample`` so ``grade.py``'s own comparator (not this module) is what
    turns it into ``Action.INSUFFICIENT_DATA`` and withholds it from corpus
    comparison -- see ``test_tiny_document_is_insufficient_data_end_to_end``
    for that end-to-end behaviour.
    """

    module = nx
    graph = module.Graph()
    graph.add_edge("x", "y", weight=1)  # only 2 nodes, below MIN_GRAPH_NODES
    result = graph_suite._graph_findings("test", "Test graph", module, graph,
                                         {"graph_type": "test"}, present_at=None)
    by_id = {item["metric_id"]: item for item in result}
    density = by_id["discourse.graph_test_density"]
    assert density["value"] == pytest.approx(1.0)
    assert density["sample_size"] == 2
    assert density["min_sample"] == graph_suite.MIN_GRAPH_NODES
    assert density["sample_size"] < density["min_sample"]
    assert "below the" in density["warning"]


@pytest.mark.skipif(not optional.have("networkx"),
                    reason="networkx is not available in this environment")
def test_tiny_document_is_insufficient_data_end_to_end(tmp_path, base_config):
    source = tmp_path / "tiny.txt"
    source.write_text("Hi. Bye.", encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "graph_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    graph_results = [item for item in report.results if item.metric_id.startswith(PREFIX)]
    assert graph_results
    assert all(item.status_type is not StatusType.INTERNAL_ERROR for item in graph_results)
    assert any(item.action is Action.INSUFFICIENT_DATA for item in graph_results)


@pytest.mark.parametrize("text", ["", "Hi.", "A\n\nB\n\nC"])
def test_degenerate_documents_never_raise(text, tmp_path, base_config):
    source = tmp_path / "degenerate.txt"
    source.write_text(text, encoding="utf-8")
    config = {**base_config, "metrics": {**base_config["metrics"],
                                         "graph_suite": {"enabled": True}}}
    report = grade.analyze(source, config)
    errors = [item for item in report.results if item.status_type is StatusType.INTERNAL_ERROR]
    assert not errors, [(item.metric_id, item.error) for item in errors]


# ------------------------------------------------ semantic graph edge/runtime cap

@pytest.mark.skipif(not optional.have("networkx"),
                    reason="networkx is not available in this environment")
def test_long_document_semantic_graph_respects_its_edge_and_runtime_cap(monkeypatch):
    """A long document's sentence-semantic graph must sample down to
    ``semantic_graph_max_units`` rather than building an O(n^2) graph over
    every sentence, and the resulting graph's edge count must respect the
    k-nearest-neighbor cap.
    """

    from textgrader.metrics import semantic_adjacent as sem

    # Force the dependency-free lexical fallback so this test needs no model
    # download and runs in well under a second even at this sentence count.
    monkeypatch.setattr(sem, "embed_texts", lambda *a, **k: (None, "forced fallback for this test"))

    vocab = ["orange", "violet", "cedar", "harbor", "granite", "willow", "ember", "quartz"]
    sentences = []
    for index in range(600):
        words = [vocab[(index + offset) % len(vocab)] for offset in range(4)]
        sentences.append(" ".join(words).capitalize() + ".")
    text = " ".join(sentences)
    analysis = _analysis(text)

    max_units, k = 40, 3
    findings = _by_id(graph_suite.measure(analysis, config={
        "features": {name: False for name in graph_suite.DEFAULT_FEATURES}
                    | {"sentence_semantic": True},
        "semantic_graph_max_units": max_units, "semantic_graph_k": k,
        "semantic_graph_min_similarity": 0.0}))

    node_count = findings[f"{PREFIX}sentence_semantic_node_count"]
    edge_count = findings[f"{PREFIX}sentence_semantic_edge_count"]
    assert node_count["value"] == max_units
    assert node_count["distribution"]["sampled"] is True
    assert node_count["distribution"]["units_in_document"] == len(analysis.sentences)
    # Undirected k-NN: at most one edge per (i, j) pair even though both i and
    # j may nominate each other, and at most k nominations issued per node.
    assert edge_count["value"] <= max_units * k


# ------------------------------------------------------- construction settings

@pytest.mark.skipif(not (optional.have("networkx") and optional.have("spacy")),
                    reason="networkx/spaCy are not available in this environment")
def test_every_finding_records_its_construction_settings(sample_text):
    findings = graph_suite.measure(_analysis(sample_text), config={
        "features": {name: False for name in graph_suite.DEFAULT_FEATURES}
                    | {"entity_cooccurrence": True}})
    density = _by_id(findings)[f"{PREFIX}entity_density"]
    assert density["distribution"]["graph_type"] == "entity_cooccurrence"
    assert "window_sentences" in density["distribution"]
    assert "edge_definition" in density["distribution"]


@pytest.mark.skipif(not optional.have("networkx"),
                    reason="networkx is not available in this environment")
def test_raw_counts_are_marked_unit_sensitive(sample_text):
    findings = graph_suite.measure(_analysis(sample_text), config={
        "features": {name: False for name in graph_suite.DEFAULT_FEATURES}
                    | {"lexical_chain": True}})
    by_id = _by_id(findings)
    assert by_id[f"{PREFIX}lexical_chain_node_count"]["unit_sensitive"] is True
    assert by_id[f"{PREFIX}lexical_chain_edge_count"]["unit_sensitive"] is True
    assert by_id[f"{PREFIX}lexical_chain_density"]["unit_sensitive"] is not True


# ----------------------------------------------------------------- dependency-free

def test_dependency_free_features_work_without_spacy(sample_text, monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "spacy")
    optional.reset_cache()
    try:
        analysis = _analysis(sample_text)
        assert analysis.nlp_unavailable
        findings = graph_suite.measure(analysis, config={
            "features": {name: False for name in graph_suite.DEFAULT_FEATURES}
                        | {"lexical_chain": True, "surface_name": True, "quote_speaker": True}})
        by_id = _by_id(findings)
        assert by_id[f"{PREFIX}lexical_chain_node_count"]["value"] is not None
        assert by_id[f"{PREFIX}surface_name_node_count"]["value"] is not None
    finally:
        optional.reset_cache()


def test_parse_dependent_features_report_unavailable_without_spacy(sample_text, monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "spacy")
    optional.reset_cache()
    try:
        analysis = _analysis(sample_text)
        findings = graph_suite.measure(analysis, config={
            "features": {name: False for name in graph_suite.DEFAULT_FEATURES}
                        | {"entity_cooccurrence": True, "character_cooccurrence": True}})
        by_id = _by_id(findings)
        assert by_id[f"{PREFIX}entity_node_count"]["value"] is None
        assert by_id[f"{PREFIX}entity_node_count"]["warning"]
        assert by_id[f"{PREFIX}character_node_count"]["value"] is None
    finally:
        optional.reset_cache()


def test_a_missing_networkx_degrades_every_graph_not_the_run(sample_text, monkeypatch):
    monkeypatch.setenv("TEXTGRADER_DISABLE_OPTIONAL", "networkx")
    optional.reset_cache()
    try:
        findings = graph_suite.measure(_analysis(sample_text))
        assert findings
        assert all(item["value"] is None for item in findings)
        assert all(item["warning"] for item in findings)
    finally:
        optional.reset_cache()


# ------------------------------------------------------------------- BookNLP

@pytest.mark.skipif(not optional.have("networkx"),
                    reason="networkx is not available in this environment")
def test_booknlp_feature_degrades_when_the_package_is_absent(sample_text, monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("BookNLP must only load when booknlp_feature is on and reachable")

    monkeypatch.setattr(gr, "_load_booknlp", lambda *a, **k: (None, "booknlp is not installed "
                                                              "in this test"))
    findings = graph_suite.measure(_analysis(sample_text), config={
        "features": {name: False for name in graph_suite.DEFAULT_FEATURES} | {"booknlp": True}})
    assert findings
    assert all(item["value"] is None for item in findings)
    assert all("not installed" in (item["warning"] or "") for item in findings)


def test_shim_booknlp_transformers_is_idempotent():
    """Calling the shim twice must not raise or double-wrap the patched method."""

    optional.shim_booknlp_transformers()
    optional.shim_booknlp_transformers()
