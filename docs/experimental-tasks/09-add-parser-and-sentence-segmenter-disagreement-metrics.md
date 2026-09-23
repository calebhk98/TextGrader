# 9. Add parser and sentence-segmenter disagreement metrics

## Goal

Run multiple independent sentence segmenters and syntactic parsers, then measure how much they disagree. Treat parser disagreement as a signal in its own right, especially for poetry, malformed text, OCR damage, fragments, and syntactically unusual prose.

## Suggested TextGrader integration

- Config switch: `parser_consensus`
- Primary module: `textgrader/metrics/parser_consensus.py`
- Family: `syntax`
- Metric ID prefix: `syntax.parser_`
- Cost: `parse` and potentially expensive. Support configurable parser/segmenter lists and maximum words.
- Do **not** replace the canonical TextGrader segmentation/parser. This task measures disagreement against it and among alternatives.

## Metrics to add

**Sentence segmentation consensus**

- Sentence-count disagreement across segmenters.
- Boundary precision/recall/F1 pairwise against canonical boundaries.
- Boundary consensus rate.
- Boundaries supported by all/majority/only-one segmenter.
- Mean/maximum character offset disagreement.
- Paragraphs with highest boundary disagreement.

**Tokenization consensus**

- Token-count disagreement.
- Token-boundary F1.
- Contraction/hyphen/apostrophe disagreement counts.
- Special-token disagreement rate.

**POS/morphology consensus**

- POS agreement rate on aligned tokens.
- Morphological-feature agreement.
- Per-tag confusion matrix summaries.

**Dependency consensus**

- Unlabeled attachment agreement between parsers after token alignment.
- Labeled attachment agreement.
- Root agreement.
- Dependency-label agreement.
- Parse-depth disagreement.
- Dependency-distance disagreement.
- Sentence-level parser disagreement distribution.

**Constituency/chunk/NER consensus where available**

- Span-boundary overlap/F1.
- Noun-phrase span agreement.
- Named-entity span/type agreement.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **spaCy** | Existing shared parse and one reference implementation. |
| **Stanza** | Independent tokenization/POS/dependency/NER/constituency where models exist. |
| **Stanford CoreNLP** | **Not used: Java, and TextGrader is Python only.** Independent Java pipeline for tokenization, POS, dependencies, constituency, NER. |
| **SuPar** | Additional dependency/constituency parsers. |
| **benepar** | Constituency parsing. |
| **UDPipe** | Independent tokenization/tagging/dependency parser. |
| **Trankit** | Additional multilingual NLP pipeline. |
| **pySBD** | Existing sentence segmenter. |
| **syntok** | Independent sentence/token segmentation. |
| **segtok** | Independent segmenter. |
| **BlingFire** | Fast sentence/token segmentation. |
| **NLTK Punkt** | Statistical sentence segmentation baseline. |

## Why

Any one parser can fail confidently. Agreement across independent parsers is an empirical proxy for how conventional/easy-to-parse a passage is. More importantly, different disagreement patterns can identify specific input problems. TextGrader should measure this without making one parser the unquestioned ground truth.

## Implementation details

1. Add a canonical alignment layer operating on character offsets. Never compare parser token index 17 directly unless spans align.
2. Sentence-boundary comparisons should use source character offsets in canonical analyzed text.
3. For token mismatches, calculate both exact-boundary agreement and a relaxed alignment for downstream POS/dependency comparison.
4. Dependency comparison should only include aligned token spans; report alignment coverage so a low LAS/UAS caused by tokenization mismatch is distinguishable.
5. Cache each external parser result in `analysis._shared`.
6. Support `parsers=[...]`, `segmenters=[...]`, `max_words`, and timeout settings.
7. Every parser is optional. The metric should still compare whichever two or more are available. With only one implementation, emit insufficient/unavailable consensus findings rather than pretending consensus exists.
8. Keep pairwise disagreement and all-system consensus separately.
9. Include bounded evidence: top sentences/paragraphs with disagreement and parser outputs summarized, not full parse dumps.

## Corpus/profile requirements

Store scalar disagreement distributions and parser/model versions. Corpus comparison is only valid for the same parser set/model versions and alignment policy.

## Tests and validation

- Abbreviations, initials, decimals and quotations for sentence segmentation.
- Hyphenated/contraction tokenization.
- Normal prose with expected high agreement.
- Fragmented text/poetry with expected higher disagreement.
- Missing-parser combinations.
- Parser timeout/failure isolation.
- Character-offset alignment correctness.

## Acceptance criteria

- Multiple segmenters and parsers can be configured independently.
- Pairwise and consensus disagreement are measurable.
- Tokenization coverage is reported before POS/dependency agreement.
- The suite never changes TextGrader's canonical analysis as a side effect.
- External parser failure is contained to its own channel.

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
6. **Everything optional must degrade locally, and must be Python.** Add Python libraries to `textgrader.optional.PACKAGES`. TextGrader is Python only: do not add a tool that needs R, a JVM, or another language runtime, even where a table below names one. A compiled helper a Python package calls (a KenLM binary, say) still needs the same availability/error handling.
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
