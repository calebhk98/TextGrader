# 2. Add discourse coherence and cohesion analysis

## Goal

Measure whether sentences and paragraphs are connected to one another, not merely whether individual sentences are grammatical. Add lexical cohesion, entity continuity, coreference continuity, discourse-relation structure, entity-grid/entity-graph measures, semantic-ordering measures, and permutation-based coherence tests.

## Suggested TextGrader integration

- Config switch: `coherence_suite`
- Primary module: `textgrader/metrics/coherence_suite.py`
- Optional helper module: `textgrader/coherence.py` for shared entity-grid/permutation utilities.
- Family: `discourse`
- Metric ID prefix: `discourse.coherence_`
- Cost: `moderate`, `parse`, or `model` depending on subtest; one suite may report partial results when only some dependencies exist.
- Reuse canonical sentences/paragraphs and the existing shared spaCy parse/embedding cache where possible.

## Metrics to add

**Lexical/local cohesion**

- Adjacent-sentence content-word overlap.
- Adjacent-sentence lemma overlap.
- Adjacent-sentence noun overlap.
- Adjacent-sentence named-entity overlap.
- Adjacent-paragraph overlap equivalents.
- Global lexical overlap against the document centroid/context.
- Repeated-keyword chain length.
- Lexical-chain count, mean length, longest chain, and coverage.
- Local semantic similarity mean/median/variance/low-tail rate.
- Paragraph-to-paragraph semantic transition statistics.

**Entity continuity / entity-grid metrics**

- Entities introduced per sentence.
- Entities carried over from previous sentence.
- New-vs-given entity ratio.
- Dangling entity rate: entities introduced once and never referenced again.
- Mean/median/max reintroduction distance.
- Coreference-chain count and length distribution.
- Pronoun-to-named-mention transition rates.
- Subject/object/other entity-grid transition frequencies.
- Entity-grid transition entropy.
- Entity-grid transition likelihood under the reference corpus.
- Entity-graph density, connected components, average degree, clustering, and largest-component share.

**Discourse-relation metrics**

- Explicit connective rate by relation family: causal, contrastive, temporal, additive, conditional, exemplification, conclusion, elaboration, concession.
- Implicit discourse-relation distribution when a parser provides it.
- RST relation distribution.
- RST tree depth, branching, nuclearity balance, and relation entropy.
- Discourse-segment length distribution.
- Connective/relation mismatch or low-confidence rate where a parser exposes confidence.

**Order sensitivity / permutation tests**

- Sentence-order advantage: percentile of real paragraph order relative to randomized sentence permutations.
- Paragraph-order advantage for multi-paragraph documents.
- Entity-grid order discrimination score.
- Semantic-order discrimination score.
- RST/discourse-order discrimination where available.
- Fraction of paragraphs whose original order beats at least 50%, 75%, 90%, and 95% of permutations.
- Median rank of original ordering.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **TAACO** | Broad lexical/semantic cohesion indices and independent cross-checks. |
| **ReaderBench** | Cohesion, discourse, complexity, reading-oriented measurements. |
| **CoherenceFramework / entity-grid implementations** | Entity-grid and entity-graph coherence. Use maintained equivalents if the named project is stale. |
| **discopy** | PDTB-style discourse relation work where compatible. |
| **DisCoDisCo** | Discourse segmentation/connective/relation predictions. |
| **IsaNLP RST parser** | Rhetorical Structure Theory parsing. |
| **Feng-Hirst RST parser / maintained RST alternatives** | Independent RST parse/measurement. |
| **PDTB resources/tools** | Explicit/implicit discourse relation labels and connective inventories. |
| **fastcoref** | Coreference chains. |
| **Coreferee** | Independent coreference implementation. |
| **spaCy coreference components** | Additional coreference channel when installed. |
| **BookNLP** | Long-document entity/coreference continuity, especially fiction. |
| **NetworkX / igraph** | Entity graphs and graph statistics. |
| **sentence-transformers** | Existing semantic coherence channel; reuse current embedding cache rather than re-encoding. |

## Why

Word-level and sentence-level correctness cannot detect a paragraph whose sentences are individually plausible but badly ordered or unrelated. Coherence is inherently relational. Entity grids, lexical chains, semantic transitions, discourse parsing, and sentence-order discrimination attack the problem from different angles. Their disagreement is valuable: a paragraph may be lexically cohesive but logically disordered, or entity-coherent but semantically repetitive.

## Implementation details

1. Start with dependency-free lexical overlap and permutation infrastructure so the suite always emits something.
2. Use canonical sentence boundaries from `DocumentAnalysis`. The permutation test must only reorder those existing sentences; it must not re-segment text.
3. Use a deterministic RNG seed in config, e.g. `permutations=50`, `seed=0`. For very large documents sample paragraphs/windows rather than exploding runtime.
4. Cache entity/coreference representations in `analysis._shared` so graph, continuity, and later tasks can reuse them.
5. Represent entity-grid roles at minimum as subject/object/other/absent. Compute transition counts for adjacent sentences and optionally lag-2 transitions.
6. If no reliable coreference model exists, still provide exact lemma/entity-string continuity and clearly label it as surface/entity-string based.
7. Treat dialogue and narration separately where possible. Emit `channel="dialogue"`, `channel="narration"`, and `channel="full"` findings rather than hiding channel effects.
8. For RST/PDTB tools, persist relation counts and tree summaries, not the full parse tree in every metric result. Detailed trees may be stored as optional diagnostics/artifacts.
9. Permutation baselines must preserve the same sentences and vocabulary. Their purpose is order sensitivity, not general language-likeness.
10. Avoid assigning quality polarity initially. High cohesion can be pathological repetition; low cohesion can be deliberate poetry or montage.

## Corpus/profile requirements

Store distributions for scalar outputs. For entity-grid likelihood, optionally store corpus transition probabilities with smoothing and the exact role schema/version. For discourse parsers, store model/version identifiers because relation inventories can change across implementations.

## Tests and validation

- Ordered vs shuffled paragraph fixture: original should usually outperform its permutations on at least some order-sensitive metrics.
- Repeated-topic but shuffled text: lexical overlap may remain high while ordering scores fall; retain both results.
- Entity-continuity fixture with pronouns and repeated named entities.
- Poetry/fragment fixture must not crash; metrics may report insufficient data.
- Missing coreference/RST/PDTB dependencies must not prevent lexical metrics from running.
- Deterministic permutation score under fixed seed.
- Large-document runtime cap test.

## Acceptance criteria

- The suite emits lexical, entity, discourse, and ordering measurements separately.
- It includes sentence-permutation and paragraph-permutation baselines.
- It can run partially without heavy models.
- Entity-grid/coreference/discourse outputs can be compared to corpus distributions.
- The original and shuffled versions of the same text can be meaningfully distinguished by at least some tests without using vocabulary changes.

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
