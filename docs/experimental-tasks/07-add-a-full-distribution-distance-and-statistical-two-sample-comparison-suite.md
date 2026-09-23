# 7. Add a full distribution-distance and statistical two-sample comparison suite

## Goal

Expand TextGrader's whole-distribution comparison beyond Wasserstein distance. For any metric that produces sentence-, paragraph-, word-, or turn-level samples, calculate several independent distribution distances/tests and retain them separately.

## Suggested TextGrader integration

- Config switch: `distribution_distance_suite`
- Primary module: `textgrader/metrics/distribution_distance_suite.py`
- Extend/reuse `distribution_shape.py` helpers rather than inventing a second pooled-corpus representation.
- Family: `distribution_shape`
- Metric IDs: `style.distribution_<source>_<distance>` or extend tests to recognize a `distribution.` prefix.
- Cost: `moderate`.

## Metrics to add

For each eligible manuscript sample vs pooled/reference sample:

- 1D Wasserstein / Earth Mover distance.
- Energy distance.
- Kolmogorov-Smirnov statistic and p-value where assumptions are acceptable.
- Cramér-von Mises statistic and p-value.
- Anderson-Darling k-sample statistic where supported.
- Jensen-Shannon distance/divergence for discretized distributions.
- KL divergence in both directions when smoothing makes it finite.
- Symmetrized KL.
- Hellinger distance.
- Bhattacharyya coefficient/distance.
- Total variation distance.
- Maximum Mean Discrepancy (MMD).
- Sliced Wasserstein where multivariate samples exist.
- Optimal transport cost for appropriate feature distributions.
- Quantile-vector distance.
- Tail-specific distance: lower/upper decile mismatch.
- Distance correlation where the comparison is relational rather than a two-sample metric.

Apply these to existing/future distributions such as sentence length, paragraph length, word length, turn length, dependency distance, parse depth, lexical rarity, sentiment, semantic transitions, etc.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **SciPy** | Wasserstein, energy distance, KS, Cramér-von Mises, Anderson-Darling, entropy utilities. |
| **POT (Python Optimal Transport)** | Optimal transport and sliced-Wasserstein families. |
| **hyppo** | MMD, energy and independence/two-sample methods. |
| **dcor** | Distance correlation and related energy statistics. |
| **NumPy** | Histograms, quantiles, smoothing, vectorized calculations. |

## Why

A single distance cannot describe every way distributions differ. Wasserstein is sensitive to transport/shift; KS is sensitive to maximum CDF separation; tail metrics notice rare extremes; JS/Hellinger compare probability mass; MMD can detect broader distribution differences. TextGrader explicitly wants correlated but non-identical sensors.

## Implementation details

1. Create one shared API accepting manuscript values, reference values, and measurement metadata.
2. Never compare distributions with incompatible units or preprocessing.
3. For histogram-based divergences, use common bin edges determined from the union/reference and configurable smoothing. Store binning method in settings.
4. Report effect/distance and statistical-test p-value separately. A tiny p-value on a huge sample is not an effect size.
5. Add minimum sample sizes per method and reject pathological calculations rather than returning NaN/inf.
6. Where reference corpus values are stored per-document rather than pooled, explicitly choose whether the test compares to pooled observations or to a distribution of document summaries. Keep those as different metric IDs.
7. Multivariate OT/MMD should use bounded, standardized vectors and explicit dimension limits.

## Corpus/profile requirements

Reuse pooled distribution storage where possible. Add enough metadata to identify source measurement, units, sample unit, histogram binning, smoothing, and whether reference values were pooled or per-document summaries.

## Tests and validation

- Same-distribution synthetic samples.
- Mean-shift samples.
- Same mean but changed variance.
- Same mean/variance but bimodal vs unimodal.
- Tail-only contamination.
- Verify different distances rank these cases differently.
- Tiny sample handling and no NaN/inf leakage.

## Acceptance criteria

- Wasserstein is retained and at least the listed independent alternatives are added where mathematically valid.
- Effect sizes/distances are not replaced by p-values.
- Each distance has a stable metric ID and corpus distribution.
- Distribution source and preprocessing are auditable from result settings/details.

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
