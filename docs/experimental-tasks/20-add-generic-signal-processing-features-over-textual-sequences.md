# 20. Add generic signal-processing features over textual sequences

## Goal

Apply classical signal-processing tools to ordered text-derived sequences such as sentence length, word length, punctuation density, syllables, lexical rarity, POS codes, stress patterns, semantic deltas, and other channels. This intentionally tests methods that may be weak; the goal is empirical coverage.

## Suggested TextGrader integration

- Config switch: `signal_processing_suite`
- Primary module: `textgrader/metrics/signal_processing_suite.py`
- Reuse the sequence registry created by Task 5 if present; otherwise implement a minimal compatible sequence provider.
- Family: `sentence_rhythm` / `book_drift`
- Metric IDs: `rhythm.signal_<sequence>_<feature>` or `drift.signal_...`
- Cost: `moderate`.

## Source sequences

At minimum support:

- Sentence word lengths.
- Paragraph word lengths.
- Word lengths as a token sequence.
- Syllable counts per word/sentence where available.
- Punctuation count/density per sentence.
- Lexical rarity/frequency per token or sentence.
- POS-tag numeric sequence with a stable encoding.
- Dependency-depth/distance sequences.
- Semantic adjacent-distance sequence.
- Sentiment trajectory.
- Stress sequence from prosody tools.

## Metrics to add

- FFT power spectrum summaries.
- Dominant frequency.
- Spectral centroid.
- Spectral bandwidth/spread.
- Spectral flatness.
- Spectral entropy.
- Low/mid/high-frequency energy shares.
- Welch periodogram versions of major spectral summaries.
- Autocorrelation/cross-correlation beyond existing sentence-length ACF.
- Peak count/prominence/distance distribution.
- Zero-crossing rate after centering where meaningful.
- Cepstral coefficients or simple cepstral peak summaries where appropriate.
- Wavelet coefficient energy by scale.
- Wavelet entropy.
- Multiscale variance.
- Cross-spectrum/coherence between paired sequences where aligned, e.g. sentence length vs sentiment or rarity.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **SciPy `signal` / FFT** | Periodograms, Welch spectra, peaks, correlations, spectral features. |
| **NumPy FFT** | Basic FFT and deterministic numeric operations. |
| **PyWavelets** | Wavelet transforms and multiscale energy. |
| **statsmodels** | Supporting autocorrelation/cross-correlation diagnostics. |
| **librosa** (optional, experimental) | Reusable spectral/feature utilities if they cleanly accept numeric sequences; do not require audio semantics. |

## Why

Repeated prose templates, rhythmic alternation, periodic paragraphing, stress/meter, and generated regularity can produce frequency-domain patterns invisible to means and distributions. Many such features may prove useless; that is acceptable. They should be empirically evaluated rather than dismissed theoretically.

## Implementation details

1. Numeric sequence definitions and normalization must be explicit. For categorical POS tags, use a stable mapping and treat resulting spectral interpretation as experimental.
2. Detrend/center sequences for spectral calculations where appropriate and emit both raw/detrended variants only if useful.
3. Require minimum sequence lengths and return insufficient data for tiny texts.
4. Use normalized frequency units such as cycles per sentence/paragraph rather than implying time in seconds.
5. Avoid enormous FFT outputs; emit derived scalar summaries and bounded top peaks.
6. Cross-spectral metrics only run on sequences aligned to the same unit.
7. Record windowing method and parameters for Welch/wavelets.

## Corpus/profile requirements

Profile scalar signal features and record sequence definition, normalization, windowing and transform settings.

## Tests and validation

- Constant sequence.
- Alternating sequence with obvious period 2.
- Sinusoidal synthetic sequence.
- Randomized version with same distribution.
- Trend-only sequence.
- Verify dominant-frequency/spectral features respond as expected.

## Acceptance criteria

- FFT/Welch, peak, wavelet and cross-sequence signal features are available.
- Time/frequency units are documented honestly.
- Features are corpus-comparable under identical settings.
- Experimental categorical-sequence transforms are labeled as such rather than given linguistic certainty.

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
