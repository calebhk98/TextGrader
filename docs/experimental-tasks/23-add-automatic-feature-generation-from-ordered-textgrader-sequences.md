# 23. Add automatic feature generation from ordered TextGrader sequences

## Goal

Automatically expand existing per-sentence/per-paragraph/per-window measurements into hundreds or thousands of candidate document-level features. This task is intentionally exploratory and should support a “comprehensive” mode for research plus a bounded default mode for ordinary runs.

## Suggested TextGrader integration

- Config switch: `automatic_feature_generation`
- Primary module: `textgrader/metrics/automatic_features.py`
- Shared helper/schema: reuse Task 5 `textgrader/sequences.py`.
- Family: `distribution_shape` or source-family derived from each sequence.
- Metric IDs: stable sanitized `style.autofeature_<sequence>_<extractor>_<feature>` IDs.
- Cost: `moderate` to potentially expensive; must have caps.

## Source channels

Use every suitable ordered numeric sequence exposed by TextGrader, including future ones. Examples:

- Sentence length.
- Paragraph length.
- Word length.
- Lexical rarity.
- Parse depth.
- Dependency distance.
- Clause count.
- Punctuation density.
- POS entropy.
- Semantic adjacent distance.
- Topic probability/transition values.
- Sentiment/VAD/emotion trajectories.
- Entity count/continuity.
- Dialogue fraction.
- Prosody/stress/syllable sequences.
- Mechanical-error rate by window.

## Feature families to generate

- Quantiles and robust summaries.
- Range/interquartile range/MAD.
- Skew/kurtosis.
- Autocorrelation and partial autocorrelation.
- Trend coefficients.
- Change statistics.
- Run statistics.
- Peak counts.
- FFT coefficients/spectral summaries.
- Energy measures.
- Entropy/complexity measures.
- catch22/catch24 features.
- tsfresh feature families.
- Optional cross-sequence features for aligned channels.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **tsfresh** | Main automatic large-scale time-series feature extraction. |
| **pycatch22** | Compact diverse feature family. |
| **scikit-learn** | Optional feature selection/ranking during validation, never silent deletion from raw measurement store. |
| **featuretools** (experimental) | Consider only for compositional aggregation over structured section/sentence tables; keep separate from tsfresh if used. |
| **SciPy / NumPy / statsmodels** | Stable local feature implementations and cross-checks. |

## Why

Hand-designing every transformation for every sequence is slow and biased by intuition. Automated feature libraries can expose useful patterns nobody would think to implement manually. The project explicitly wants to test even unlikely signals. This task turns each base sequence into a broad experimental sensor bank.

## Implementation details

1. Define three modes:
   - `minimal`: compact handpicked stable features.
   - `standard`: catch22 plus selected tsfresh features.
   - `comprehensive`: broad tsfresh extraction subject to runtime/output limits.
2. Stable metric IDs are critical. Map extractor feature names to sanitized IDs and include extractor/library version in profile metadata.
3. Apply minimum-length and finite-value filters before extraction.
4. Do not write enormous arrays to the report. Each extracted scalar is one finding; optionally cap reported automatic features to those configured or those with a corpus profile. The raw research output can be a separate artifact if needed.
5. During corpus validation, calculate feature stability, missingness, runtime, correlation and effect under synthetic corruptions. **Do not automatically delete highly correlated features.** Flag them for analysis instead.
6. Provide allowlist/denylist regexes for feature names so problematic or extremely slow extractors can be disabled.
7. Cache source sequences; do not recompute base NLP features.
8. Add an upper bound on number of emitted findings and a deterministic selection rule when limits are reached.

## Corpus/profile requirements

Profiles may become large. Support a dedicated `automatic_features` section/artifact containing feature schema, library versions, settings, distributions and coverage. Do not bloat the base profile uncontrollably without a documented storage strategy.

## Tests and validation

- Constant, trend, alternating, periodic and shuffled sequences.
- Comprehensive mode must be deterministic for same versions/settings.
- Runtime/output cap test.
- Missing/NaN-producing extractor must not crash the suite.
- Profile round-trip and option mismatch behavior.

## Acceptance criteria

- At least minimal, standard and comprehensive extraction modes exist.
- tsfresh and catch22 can expand multiple TextGrader sequences.
- Feature IDs/settings are stable enough for corpus profiles.
- Correlation is reported/analyzed but does not silently remove metrics.
- A single bad automatic feature cannot break the run.

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
