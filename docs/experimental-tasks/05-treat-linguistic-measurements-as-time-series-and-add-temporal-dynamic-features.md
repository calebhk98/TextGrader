# 5. Treat linguistic measurements as time series and add temporal/dynamic features

## Goal

Represent a document as ordered sequences of sentence-, paragraph-, or window-level measurements and analyze their dynamics with general time-series methods. This is intentionally broader than existing sentence-length autocorrelation and drift metrics.

## Suggested TextGrader integration

- Config switch: `timeseries_suite`
- Primary module: `textgrader/metrics/timeseries_suite.py`
- Shared helper: `textgrader/sequences.py` for canonical named numeric sequences.
- Family: `book_drift` or `sentence_rhythm` depending on finding; IDs under `drift.timeseries_` and `rhythm.timeseries_`.
- Cost: `moderate`.

## Source sequences

At minimum generate ordered numeric channels for:

- Sentence word count.
- Sentence character count.
- Paragraph word count.
- Paragraph sentence count.
- Clause count per sentence.
- Parse depth per sentence when available.
- Dependency-distance mean per sentence when available.
- Punctuation count per sentence.
- Content-word rarity per sentence.
- POS entropy per sentence.
- Character entropy per sentence.
- Named-entity count per sentence.
- Pronoun rate per sentence/window.
- Dialogue fraction by paragraph/window.
- Sentiment/emotion score when available.
- Semantic similarity to previous sentence.
- Semantic distance from document centroid.
- Topic probability / topic ID transitions when available.
- Any future metric that exposes a safe ordered numeric series through a shared sequence registry.

## Metrics to add per eligible sequence

- Mean/median/variance/robust dispersion, even if similar summaries exist elsewhere, because these are part of the time-series feature context.
- Autocorrelation at multiple lags.
- Partial autocorrelation.
- Linear trend slope/intercept/R².
- Piecewise trend summaries.
- Stationarity-test statistics where sample size permits.
- Number/strength of peaks and troughs.
- Longest above/below-median run.
- Change-point count and location summary.
- Spectral entropy.
- Dominant frequency and spectral concentration.
- Wavelet energy by scale.
- Hurst exponent.
- DFA/scaling exponent.
- Permutation entropy.
- Recurrence-related features where configured.
- catch22/catch24 feature vector.
- Selected tsfresh features.
- Drift detector event count and earliest/latest event.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **tsfresh** | Large automated family of time-series features. |
| **pycatch22** | 22 deliberately diverse time-series characteristics; optionally catch24 with mean/variance. |
| **AntroPy** | Entropy and fractal/scaling features. |
| **PyRQA** | Recurrence-quantification features. |
| **statsmodels** | ACF/PACF, AR models, stationarity tests and trend/statistical diagnostics. |
| **scipy.signal** | Peaks, periodograms, FFT/Welch-derived features, filtering helpers. |
| **PyWavelets** | Multiscale/wavelet energy and coefficients. |
| **nolds** | Nonlinear dynamics, Hurst/DFA/Lyapunov/fractal measures where defensible. |
| **ordpy** | Ordinal-pattern/permutation statistics. |
| **sktime** | Time-series transformations/detection tools where useful. |
| **skchange** | Change-point and segment-anomaly detection. |
| **ruptures** | Existing change-point dependency; reuse rather than duplicate. |
| **River** | Streaming drift detectors such as ADWIN/Page-Hinkley. |

## Why

Two documents can have identical global means and distributions while arranging those measurements very differently over time. A novel can gradually lengthen sentences, alternate systematically between terse and long passages, or exhibit periodic template behavior. Ordered dynamics are information that ordinary aggregate statistics discard.

## Implementation details

1. Add a shared sequence registry/cache in `DocumentAnalysis._shared`. A sequence should have a name, values, unit, sample unit (`sentence`, `paragraph`, `window`), and provenance.
2. The suite should discover available sequences and apply feature families independently. Missing parse/model sequences are skipped with explicit reasons.
3. Do not emit thousands of tsfresh features by default. Provide `feature_set="minimal"|"comprehensive"` or explicit feature allowlists; the comprehensive mode is the experimental target.
4. Prefix derived metric IDs with both source sequence and transform, e.g. `rhythm.timeseries_sentence_words_catch22_CO_f1ecac` or a sanitized stable equivalent.
5. Maintain stable metric IDs across library upgrades by mapping library feature names explicitly.
6. Record minimum-length requirements per algorithm. Many nonlinear features are nonsense on 8 samples.
7. Keep raw and transformed sequence summaries available in `details` only if bounded; do not dump a 50,000-point series into JSON by default.
8. Use deterministic settings/seeds.

## Corpus/profile requirements

Every selected derived scalar can be profiled normally. Store feature-set version and source-sequence definition in metric settings. Do not compare a feature created from 500-word windows with one created from 2,500-word windows.

## Tests and validation

- Constant sequence.
- Alternating short/long sequence.
- Monotonic trend.
- Periodic synthetic sequence.
- Random permutation of a real sequence: same distribution, changed temporal features.
- Tiny text must return insufficient data without numerical warnings becoming internal errors.
- Benchmark runtime on a novel-sized fixture and enforce configurable feature caps.

## Acceptance criteria

- Multiple linguistic source sequences are exposed.
- catch22/catch24 and a configurable tsfresh feature set work.
- Classical ACF/trend/spectral/wavelet/nonlinear features are available.
- Time ordering can change results even when the underlying value distribution is unchanged.
- Outputs are stable, bounded, corpus-comparable, and failure-isolated.

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
