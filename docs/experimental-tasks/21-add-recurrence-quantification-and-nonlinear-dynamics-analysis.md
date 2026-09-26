# 21. Add recurrence-quantification and nonlinear-dynamics analysis

## Goal

Apply recurrence plots and nonlinear-dynamics/complexity measures to text-derived sequences and semantic trajectories. This is explicitly experimental. Keep every result separate and validate on synthetic sequences before interpreting real text.

## Suggested TextGrader integration

- Config switch: `nonlinear_dynamics_suite`
- Primary module: `textgrader/metrics/nonlinear_dynamics_suite.py`
- Reuse Task 5 sequence registry and Task 12 semantic vectors if available.
- Family: `book_drift`
- Metric ID prefix: `drift.nonlinear_`
- Cost: `moderate`; recurrence matrices must be capped for long documents.

## Metrics to add

**Recurrence Quantification Analysis**

- Recurrence rate.
- Determinism.
- Average diagonal-line length.
- Longest diagonal line.
- Diagonal-line entropy.
- Laminarity.
- Trapping time.
- Longest vertical line.
- Recurrence-time statistics.
- Trend/nonstationarity measures from recurrence structure.

**Nonlinear/scaling metrics**

- Hurst exponent.
- Detrended fluctuation analysis exponent.
- Correlation/fractal dimension where sample size supports it.
- Sample entropy.
- Approximate entropy.
- Permutation entropy.
- Lempel-Ziv complexity.
- Ordinal-pattern distribution/entropy.
- Optional Lyapunov-exponent estimates only when sample-size and method assumptions are met; otherwise skip.

Apply to selected channels such as sentence length, paragraph length, semantic distance, sentiment, lexical rarity, parse depth, punctuation density, syllable/stress sequences, and low-dimensional semantic trajectories.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **PyRQA** | Recurrence plots and RQA measures. |
| **nolds** | Hurst, DFA, Lyapunov/fractal/nonlinear measures. |
| **AntroPy** | Entropy, LZ and fractal/scaling cross-checks. |
| **EntropyHub** | Additional entropy estimators. |
| **ordpy** | Ordinal patterns/permutation entropy/complexity. |
| **NumPy/SciPy** | Embedding, distance matrices, numeric support. |

## Why

Writing has long-range structure and recurrence. Some texts repeat motifs, rhythms, or semantic states in patterned ways; others drift. RQA and nonlinear statistics offer a radically different measurement basis from ordinary NLP. Even weak predictive value is worth measuring under the project's experimental philosophy.

## Implementation details

1. Start with scalar sequences. Add semantic-vector recurrence only after a bounded/downsampled design exists.
2. Make embedding dimension, delay and recurrence threshold explicit configuration.
3. Provide automated/default threshold selection but report the chosen threshold.
4. Cap recurrence matrix size. For long sequences, downsample, use windows, or sample fixed-length subsequences and report strategy.
5. Return insufficient data rather than unstable estimates on short sequences.
6. Keep AntroPy/nolds/EntropyHub versions of similar measures independently if they use different estimators.
7. Do not attach quality polarity.

## Corpus/profile requirements

Store settings and distributions for every scalar. Comparison requires matching embedding/threshold/downsampling strategy.

## Tests and validation

- Constant sequence.
- Periodic sequence.
- Chaotic/noisy synthetic sequence.
- Shuffled version of same values.
- Too-short sequence.
- Runtime/memory cap on long sequence.

## Acceptance criteria

- Core RQA measures and multiple nonlinear/entropy measures are available.
- Parameters/downsampling are explicit and reproducible.
- Short/large inputs fail gracefully.
- Similar estimators from different libraries may coexist as independent metrics.

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
