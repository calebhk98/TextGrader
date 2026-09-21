# 6. Add multivariate anomaly/outlier detection over TextGrader feature vectors

## Goal

Treat each corpus text as a point in a multivariate feature space and run many independent anomaly detectors. Emit each detector's score separately. The purpose is to identify texts that are unusual in combinations of metrics even when no single metric is extreme.

## Suggested TextGrader integration

- Config switch: `anomaly_suite`
- Primary module: `textgrader/metrics/anomaly_suite.py`
- Helper: `textgrader/feature_matrix.py` to build stable document vectors from corpus/profile features.
- Family: `distribution_shape`
- Metric IDs: `style.anomaly_<detector>` so current optional-metric prefix tests continue to see them.
- Cost: `moderate`; model fitting should happen during corpus build/profile validation, not every grade when avoidable.

## Metrics to add

At minimum emit independent standardized anomaly scores/ranks for:

- Isolation Forest.
- Local Outlier Factor / novelty mode.
- One-Class SVM.
- Elliptic Envelope / robust covariance.
- Mahalanobis distance.
- kNN distance.
- HBOS.
- ECOD.
- COPOD.
- ABOD.
- KDE-based anomaly score.
- PCA reconstruction/outlier score.
- Gaussian-mixture low-likelihood score.
- SOS where available.
- HDBSCAN outlier score where a stable reference fit exists.
- Consensus count: number of detectors above their own configured reference percentile. Keep this as a count, not a replacement for individual outputs.
- Detector disagreement/dispersion across normalized ranks.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **scikit-learn** | IsolationForest, LOF, OneClassSVM, EllipticEnvelope, PCA, covariance, mixtures where appropriate. |
| **PyOD** | ECOD, COPOD, ABOD, KNN, HBOS, KDE, GMM, SOS, IForest, LOF, PCA and additional detectors. |
| **hdbscan** | Density/outlier scores. |
| **SciPy / NumPy** | Mahalanobis, robust numerical helpers, rank normalization. |
| **pandas** | Stable corpus feature matrix construction and missing-value diagnostics. |

## Why

Text can be ordinary on every one-dimensional metric but unusual in combination: for example, very simple syntax combined with unusually rare vocabulary, or high sentence variation combined with extremely uniform punctuation. Different anomaly algorithms encode different geometries. Their disagreement is useful and must not be collapsed prematurely.

## Implementation details

1. Define an explicit feature-selection policy. By default include comparable scalar metrics with adequate corpus coverage; exclude raw counts, labels, unavailable values, and obviously unit-sensitive metrics.
2. Record the exact feature list/order in the profile/model artifact.
3. Standardize robustly where appropriate; store the scaler fitted on the reference corpus.
4. Fit detectors on the reference corpus during profile build or an explicit preparation command. Grading should call stored detector state or recompute only inexpensive closed-form distances.
5. Missing document features require a policy: either impute from corpus median with a missingness indicator or use only detectors supporting missingness. Do not silently zero-fill.
6. Expose raw detector score, corpus percentile/rank, and threshold decision separately when possible.
7. Do not use a detector's binary `predict` result as the only metric.
8. Add a minimum corpus-size rule; multivariate models are unstable with tiny reference sets.
9. Keep separate models for different comparison units/genres if the corpus supports them rather than pooling incompatible units.

## Corpus/profile requirements

Store feature schema, transformation/scaler metadata, model parameters/artifacts, training corpus size, contamination/threshold settings, random seed, and library/model version. Provide a way to rebuild models when metric schema changes.

## Tests and validation

- Obvious synthetic outlier in 2D/3D fixture feature space.
- Combination outlier where each individual feature lies inside its marginal reference range.
- Missing-feature behavior.
- Small-corpus refusal.
- Determinism under seed.
- Verify each detector remains independently visible.
- Verify detector model/settings mismatch prevents invalid corpus comparison.

## Acceptance criteria

- At least the listed detector families are supported when dependencies exist.
- Individual detector scores and disagreements are retained.
- Models are trained from the reference corpus rather than the graded document.
- Feature schema/versioning prevents accidental incompatible comparisons.
- One broken detector does not suppress the rest.

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
