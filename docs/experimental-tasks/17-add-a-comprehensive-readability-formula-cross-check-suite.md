# 17. Add a comprehensive readability-formula cross-check suite

## Goal

Compute a broad collection of classical readability formulas using both TextGrader and independent library implementations where possible. Keep every formula and implementation disagreement instead of reducing them to a single grade level.

## Suggested TextGrader integration

- Config switch: `readability_suite`
- Primary module: `textgrader/metrics/readability_suite.py`
- Family: `readability`
- Metric ID prefix: `nlp.readability_` or extend accepted prefixes with `readability`.
- Cost: `fast`/`moderate`.

## Metrics to add

- Flesch Reading Ease.
- Flesch-Kincaid Grade.
- Gunning Fog.
- SMOG.
- Coleman-Liau.
- Automated Readability Index.
- Dale-Chall.
- Linsear Write.
- FORCAST.
- Powers-Sumner-Kearl.
- Fry score/grade where implementable.
- LIX.
- RIX.
- Spache.
- Additional formulas exposed by installed readability libraries, retained under stable IDs.
- Mean/median/max formula-implied grade.
- Spread/range/SD across formulas.
- Pairwise implementation disagreement for formulas TextGrader already computes.
- Syllable-counter disagreement.
- Difficult-word-list disagreement where multiple implementations exist.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **textstat** | Broad formula coverage. |
| **py-readability-metrics** | Independent implementations and extra formulas. |
| **LFTK** | Readability-related handcrafted features. |
| **pystylometry** | Additional readability formula implementations. |
| Existing TextGrader core metrics | Keep as primary existing calculations; compare rather than replace. |
| **pronouncing/cmudict** | Optional syllable-count cross-check. |

## Why

Readability formulas are highly correlated but differ in tokenization, syllable counting, difficult-word lists, sentence-length assumptions, and mathematical weighting. Those differences are exactly the kind of redundant-but-not-identical signals TextGrader wants.

## Implementation details

1. Do not change existing core readability values in this task.
2. Add library outputs under distinct metric IDs, e.g. `nlp.readability_fk_textstat` vs existing core `fk`.
3. Where several libraries expose the same named formula, report each and their absolute/relative disagreement.
4. Reuse canonical words/sentences for TextGrader implementations, but accept that third-party libraries may tokenize differently; record that as an implementation difference rather than forcing identical preprocessing if the library does not expose it.
5. Add a syllable-count comparison on a bounded sample or all unique words if inexpensive.
6. Mark formulas requiring minimum sentence/word counts insufficient on tiny inputs.
7. Keep formulas neutral with respect to quality; readability level is not quality.

## Corpus/profile requirements

Store all formula outputs and disagreement distributions. Record library version. Existing maturity behavior must remain unchanged unless a separate future task explicitly adds new oriented measures.

## Tests and validation

- Simple children's prose vs dense academic prose.
- Tiny text minimum-size behavior.
- Words with contested syllable counts.
- Ensure existing TextGrader core outputs are unchanged.
- Confirm two implementations may disagree without test failure; tests should validate calculation plumbing and bounded ranges, not force equality.

## Acceptance criteria

- Broad formula set is available.
- Independent implementations remain separately visible.
- Formula and syllable-counter disagreement are explicit metrics.
- Existing core scoring behavior is untouched.

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
