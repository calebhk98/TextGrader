# 22. Add a meta-analysis layer for dependence, correlation, disagreement, and residual features between TextGrader metrics

## Goal

Analyze TextGrader's own metric outputs. Preserve ordinary correlation statistics but, more importantly, create residual/disagreement features that detect when normally related measurements diverge. This directly supports the project's requirement to keep highly correlated tests because the difference between them can be informative.

## Suggested TextGrader integration

- This is partly a corpus/profile feature rather than an ordinary single-text metric.
- Config switch: `metric_relationships`
- Primary module: `textgrader/metrics/metric_relationships.py`
- Shared helper: `textgrader/relationships.py`
- Family: `distribution_shape`
- Metric IDs: `style.relationship_...`
- Cost: `moderate`.

## Corpus-level analyses to compute

For eligible numeric metrics across the reference corpus:

- Pearson correlation.
- Spearman correlation.
- Kendall tau.
- Distance correlation.
- Mutual information.
- Partial correlation where configured.
- Robust correlation alternatives where available.
- Regression models predicting one metric from one or more related metrics.
- Residual distributions for those relationships.
- Principal components / factor-like summaries as diagnostics only; do not use them to delete original metrics.
- Correlation stability under leave-one-out or bootstrap samples.

## Document-level metrics to emit

For configured/automatically discovered strongly related metric pairs/groups:

- Pairwise standardized disagreement.
- Residual of metric A after predicting it from metric B.
- Symmetric residual/disagreement version.
- Multivariate residual after predicting a metric from a group.
- Number of relationship residuals above reference 90/95/99th percentile.
- Maximum residual severity.
- Mean top-k residual severity.
- Cross-family mismatch examples such as:
  - Syntax complexity unusually low given vocabulary sophistication.
  - Readability unusually easy/hard given lexical rarity.
  - Semantic coherence unusually low given lexical overlap.
  - Sentence-length variation unusually low/high given paragraph variation.
  - Words/paragraph unusual given words/sentence and sentences/paragraph.
- Detector disagreement among correlated implementations, e.g. readability formulas, parsers, spell checkers, sentiment engines.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **SciPy** | Pearson/Spearman/Kendall and statistical utilities. |
| **statsmodels** | Regression, robust models, partial relationships, diagnostics. |
| **pingouin** | Convenient partial/robust correlation methods where useful. |
| **dcor** | Distance correlation. |
| **hyppo** | Independence/dependence tests and MMD/energy-style relationships. |
| **scikit-learn** | Mutual information, robust/scaled regressions, nonlinear regressors where configured. |
| **NumPy/pandas** | Corpus metric matrix and residual calculations. |

## Why

If words/sentence, words/paragraph and sentences/paragraph are 90% correlated, the remaining 10% is precisely where unusual structure can appear. Deleting one because of correlation loses that signal. Modeling the expected relationship and scoring residuals turns redundancy into a feature.

## Implementation details

1. Build the relationship model from the reference corpus only.
2. Do not automatically include every possible pair in final output; the profile may contain a full matrix, while the grade run emits configured/top residuals and selected known relationships.
3. Add an option such as `discover_min_abs_spearman=0.7` to discover strongly related pairs, but retain user-configured pairs even if correlation is lower.
4. Robustly scale metrics and ignore pairs with inadequate joint coverage.
5. Store regression coefficients/model artifacts and residual distribution.
6. For nonlinear relationships, optionally fit simple splines/random forests, but keep linear residuals too; do not replace the interpretable baseline.
7. The relationship metric must run after ordinary metrics have been measured. If the existing runner cannot support post-processing, add a clearly isolated post-metric phase rather than making metrics call one another recursively.
8. No residual feature should cause the underlying metrics to be deleted or hidden.

## Corpus/profile requirements

Store metric matrix coverage, correlation matrices, configured/discovered relationships, model coefficients/artifacts, residual distributions and feature versions. Rebuild when the source metric set changes.

## Tests and validation

- Synthetic perfectly correlated pair: near-zero residual.
- Synthetic mostly correlated pair with one anomaly: high residual only for anomaly.
- Existing words/sentence, words/paragraph, sentences/paragraph relationship.
- Missing source metrics should make only dependent residual findings unavailable.
- Leave-one-out profile validation to ensure the test document is not used to fit its own expected relationship.

## Acceptance criteria

- Multiple dependence statistics are available at corpus level.
- Document-level residual/disagreement metrics exist.
- Correlated metrics remain intact.
- Reference fitting is leakage-free.
- The relationship layer can identify an unusual combination even when both raw metrics are individually corpus-inliers.

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
