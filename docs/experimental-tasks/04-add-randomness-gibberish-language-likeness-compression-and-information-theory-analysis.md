# 4. Add randomness, gibberish, language-likeness, compression, and information-theory analysis

## Goal

Detect multiple kinds of “noise”: character garbage, plausible-word salad, syntactically strange text, repeated/template text, and structurally random text. Use several n-gram levels, several compressors, entropy families, and complexity measures because each is sensitive to different pathologies.

## Suggested TextGrader integration

- Config switch: `randomness_suite`
- Primary module: `textgrader/metrics/randomness_suite.py`
- Family: `lexical` or a new registry family such as `randomness`; keep metric IDs under an existing test prefix such as `style.randomness_` unless test infrastructure is extended.
- Metric ID prefix: `style.randomness_`
- Cost: `moderate`; optional pre-trained KenLM models may raise cost but should not become generative/model metrics.

## Metrics to add

**Language-model likelihood**

- Character 2/3/4/5/6-gram cross-entropy/perplexity.
- Byte n-gram cross-entropy/perplexity.
- Word unigram/bigram/trigram/4-gram perplexity.
- POS n-gram perplexity.
- Dependency-label n-gram perplexity.
- Punctuation-sequence n-gram perplexity.
- Difference between word-level and character-level surprisal.
- Fraction of sentences/paragraphs below reference likelihood thresholds.
- Lowest-likelihood sentence/paragraph with bounded evidence.

**Compression**

For each compressor, emit compressed bytes, compression ratio, bits/character, bits/token, and where feasible Normalized Compression Distance to reference texts or corpus centroids/representatives.

- zlib/DEFLATE.
- gzip.
- bz2/bzip2.
- lzma/xz.
- Zstandard.
- Brotli.
- LZ4.
- Snappy.
- PPMd.
- Any additional easily available compressor should be pluggable rather than replacing the listed set.

**Entropy / complexity**

- Character Shannon entropy.
- Byte entropy.
- Token entropy.
- Word-length entropy.
- POS entropy.
- Punctuation entropy cross-check.
- Conditional entropy for adjacent tokens/POS tags.
- Mutual information at configurable lags.
- Entropy rate estimates.
- Permutation entropy.
- Spectral entropy.
- SVD entropy.
- Approximate entropy.
- Sample entropy.
- Lempel-Ziv complexity.
- Rényi entropy for selected orders.
- Tsallis entropy for selected q values.
- Excess-entropy proxies.
- Compression-vs-Shannon residuals.

**Corruption sensitivity**

- Word-shuffle degradation in word n-gram likelihood.
- Sentence-shuffle degradation in discourse/language likelihood.
- Character-shuffle degradation.
- POS-shuffle degradation.
- Ratio of original score to shuffled baselines.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **KenLM** | Fast character/word/POS/dependency n-gram language models. |
| **NLTK language-model module** | Slower/reference implementation and smoothing cross-checks. |
| **MALLET** | Additional classical/probabilistic modeling where useful. |
| **gibberish-detector** | Character-level gibberish detector as an independent weak sensor. |
| **pyppmd** | PPMd compression and compression-based likelihood/NCD. |
| **zstandard** | Zstd compressor. |
| **brotli** | Brotli compressor. |
| **lz4** | LZ4 compressor. |
| **python-snappy** | Snappy compressor. |
| Python stdlib `zlib`, `gzip`, `bz2`, `lzma` | Dependency-free compression channels. |
| **AntroPy** | Permutation, spectral, SVD, approximate/sample entropy, LZ complexity, fractal/scaling measures. |
| **EntropyHub** | Large family of entropy algorithms. |
| **dit** | Information-theoretic quantities beyond basic entropy. |
| **SciPy / NumPy** | Shannon entropy, KL/JS components, mutual-information helpers and numeric summaries. |

## Why

Random text can fail at different levels. Character garbage fails character models. Word salad may pass spelling and unigram statistics but fail word/POS n-grams. Sentence shuffling can preserve words and syntax while destroying discourse order. Repetitive machine/template text may have abnormally strong compression. No single “gibberish score” sees all of these.

## Implementation details

1. Keep all raw subtests. Do not define one combined randomness number in this task.
2. Train/store n-gram models during corpus-building or ship explicit reference models. Do not fit the reference model to the document being graded.
3. Use smoothing suitable for unseen n-grams. Record order and smoothing configuration in profile settings.
4. Compression must operate on a documented byte representation, ideally UTF-8 of canonical analyzed text. Store the encoding/normalization setting.
5. Implement a shared compressor adapter returning compressed length and errors consistently.
6. NCD against a whole corpus can be expensive; support a configurable representative subset or nearest corpus candidates.
7. Entropy measures that require minimum sequence length must return insufficient-data findings rather than unstable numbers.
8. Shuffle baselines must be seeded and bounded. Use a small default permutation count and allow larger validation runs offline.
9. Preserve disagreement among compressors and entropy estimators.
10. Do not mark high entropy automatically “good” or “bad”; leave neutral until calibrated by text type.

## Corpus/profile requirements

Store scalar distributions. For KenLM, store model path/hash/order/training corpus identity. For NCD, retain reference document identifiers or representative compressed artifacts if required. For shuffle-derived scores, store permutation count and seed/method.

## Tests and validation

Use at least these synthetic corruption classes:

- Random ASCII/Unicode characters.
- Shuffled characters.
- Valid English words in random order.
- Grammatically plausible but semantically random sentences.
- Shuffled sentences from a coherent paragraph.
- Heavy phrase repetition/template text.
- Normal prose.
- Poetry.

Verify different channels respond differently. The desired result is not perfect agreement.

## Acceptance criteria

- Character, word, POS, dependency and punctuation language-likeness are independently measurable.
- At least the stdlib compressors plus optional PPMd/Zstd/Brotli/LZ4/Snappy are supported.
- Multiple entropy/complexity families are emitted separately.
- Corruption/shuffle baselines exist.
- Missing optional compressors/models never break the run.

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
