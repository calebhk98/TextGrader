# 13. Add entity, character, topic, and lexical graph analysis

## Goal

Represent text as graphs and extract network-science features. Support character/entity co-occurrence graphs for fiction, entity-reference graphs for essays/news, paragraph/sentence similarity graphs, and optional lexical/topic recurrence graphs.

## Suggested TextGrader integration

- Config switch: `graph_suite`
- Primary module: `textgrader/metrics/graph_suite.py`
- Helper: `textgrader/graphs.py` for graph construction/caching.
- Family: `discourse`
- Metric IDs: `discourse.graph_...`
- Cost: `moderate`/`parse`/`model` depending on node/edge source.

## Graph types to build

- Named-entity co-occurrence graph.
- Coreference-resolved entity graph.
- Character interaction/co-occurrence graph.
- Quote-speaker interaction graph when speaker attribution exists.
- Paragraph entity-overlap graph.
- Sentence semantic-similarity graph.
- Paragraph semantic-similarity graph.
- Topic-transition graph.
- Lexical-chain graph.
- Optional dependency-relation aggregate graph.

## Metrics to add per graph where meaningful

- Node count and edge count, with normalized density sibling.
- Degree mean/variance/entropy.
- Largest-component share.
- Number of connected components.
- Isolate share.
- Average clustering coefficient.
- Transitivity.
- Assortativity.
- Average shortest-path length on the largest connected component.
- Diameter/radius where computationally safe.
- Modularity/community count.
- Community-size entropy.
- PageRank concentration/Gini.
- Betweenness-centrality concentration.
- Degree-centralization measures.
- Recurrence/reappearance distance for entities/characters.
- Community persistence across chapters/windows.
- Graph edit/distance between early and late sections.
- Character/entity dominance.
- Edge-weight entropy.
- Optional graph-anomaly score relative to corpus graphs.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **NetworkX** | Primary easy-to-audit graph construction and metrics. |
| **igraph** | Faster graph algorithms and independent results on large graphs. |
| **graph-tool** | Optional high-performance/community algorithms where installation permits. |
| **rustworkx** | Additional fast graph implementation. |
| **PyTorch Geometric** | Optional graph embeddings/anomaly models; do not require for core graph statistics. |
| **PyOD graph/anomaly capabilities** | Experimental graph anomaly channels where applicable. |
| **BookNLP** | Character/entity/coreference/quote-speaker extraction for books. |
| **fastcoref/Coreferee/spaCy coref** | Entity-chain construction when BookNLP is not used. |

## Why

Narratives and expository texts have topology. A novel with one dominant protagonist and several communities differs structurally from one with a flat ensemble. An essay/news article that introduces many disconnected entities differs from one that develops a connected subject. Graphs make these relationships measurable in ways token statistics cannot.

## Implementation details

1. Define each graph construction explicitly. Edge meaning must be stable and documented: same sentence, same paragraph, coreference transition, quote exchange, semantic threshold, etc.
2. Emit graph-size-normalized metrics alongside raw counts. Raw node/edge counts are unit-sensitive.
3. Cache extracted entities/coreference chains and graph objects.
4. For community detection, use deterministic seeds and report algorithm name/settings.
5. For semantic graphs, cap k-nearest-neighbor edges or threshold edges so a long document does not become a complete O(n²) graph.
6. For character graphs, merge aliases only when a source such as BookNLP/coreference provides evidence. Do not aggressively guess aliases.
7. Provide channel/graph-type identifiers in metric IDs.
8. Graph algorithms that require connected graphs must operate on the largest component and report its coverage.

## Corpus/profile requirements

Store scalar graph metric distributions and graph-construction settings. Full graphs need not be embedded in corpus JSON; optional artifacts can be stored separately.

## Tests and validation

- Star graph narrative fixture vs chain-like fixture vs two-community fixture.
- Repeated disconnected entity mentions.
- Alias/coreference handling.
- Very small graph returns insufficient data.
- Long document semantic graph respects edge/runtime cap.

## Acceptance criteria

- Multiple graph constructions exist, not only character networks.
- Standard network measures and longitudinal graph-change measures are emitted.
- Graph definitions/settings are recorded and corpus comparable.
- Fiction benefits from BookNLP when installed but non-fiction still gets entity/semantic graphs.

---

## Shared context (from the combined task document)

This document is intended to be split into 24 Git issues. Each numbered section is written to be implementable on its own by a developer who has the TextGrader repository but does not have the surrounding discussion.

The project philosophy for these tasks is intentionally broad: **a metric is a sensor, not an opinion**. Do not remove a measurement merely because it is correlated with another measurement or because theory suggests it should be weak. Preserve raw outputs, preserve disagreements between implementations, and let later validation determine usefulness. These tasks are about adding measurement channels, not deciding in advance which channels deserve weight in a future quality score.

Unless a task explicitly says otherwise, new metrics should be **off by default**, should fail independently, should not change the existing maturity aggregate, and should use `Polarity.NEUTRAL` until the direction has been empirically validated.

# Cross-task implementation rules

These rules are repeated here as a final checklist, but each issue above is intended to remain usable independently.

1. **Do not delete correlated metrics.** Correlation is diagnostic metadata, not a deletion criterion.
2. **Do not change the existing maturity aggregate as part of these tasks.** New optional findings should remain neutral until separately validated for direction and aggregation.
3. **Do not silently create a “quality” aggregate inside one task.** Preserve raw measurements first. A future quality score can learn or explicitly weight them later.
4. **Reuse `DocumentAnalysis`.** No metric should independently decide what counts as a word, sentence, paragraph, quotation, dialogue or cleaned manuscript.
5. **Cache expensive shared representations in `analysis._shared`.** This includes parses, embeddings, coreference chains, feature vectors, graph structures and ordered sequences.
6. **Everything optional must degrade locally.** Add Python libraries to `textgrader.optional.PACKAGES`; external binaries/R/Java tools need equivalent availability/error handling.
7. **Keep new metrics off by default.** They are experimental until benchmarked.
8. **Stable metric IDs matter.** Corpus profiles depend on them. Do not expose library-generated random/positional names without a stable mapping.
9. **Store settings/version metadata.** A metric computed with different model, corpus, window, n-gram order or library version may not be comparable.
10. **Use normalized siblings for raw counts.** Raw counts remain useful but should be marked unit-sensitive and not compared across different text lengths.
11. **Respect sample-size limits.** Report `insufficient_data` instead of unstable numbers.
12. **Bound O(n²) work.** Long books are a primary use case. Use windows, candidate generation, sampling, kNN graphs, pair caps and offline corpus preparation.
13. **Preserve evidence, but bound it.** Include representative sentences/paragraphs/pairs with offsets or section IDs; never dump the full intermediate model output into the report.
14. **Run synthetic corruption tests.** For every new family, test clean text and controlled transformations: word shuffle, sentence shuffle, duplication, deletion, random substitution, OCR-like corruption, style splice, and other relevant perturbations.
15. **Use leave-one-out corpus validation.** Never fit a reference relationship/model using the same text being evaluated when the goal is out-of-sample comparison.
16. **Benchmark runtime.** Update `benchmark.py`/README cost class when a suite materially changes cost.
17. **Update registry/documentation/tests.** Every normal optional metric suite gets a `MetricSpec`, config example, README/list-metrics summary, graceful-degradation test coverage and degenerate-document coverage.
18. **Treat library disagreement as data.** If two parsers, readability libraries, sentiment engines, spell checkers or stylometry implementations disagree, keep that disagreement instead of trying to make them identical.
