# 1. Add a comprehensive stylometry and authorship-analysis suite

## Goal

Add a broad stylometric/authorship suite that can describe authorial style, compare a text with reference authors or corpus texts, detect internal style shifts, and expose multiple independent distance families. The implementation must not collapse highly correlated measures into one score. It should retain the individual distances and feature families because disagreement among them is itself useful.

## Suggested TextGrader integration

- Config switch: `stylometry_suite`
- Primary module: `textgrader/metrics/stylometry_suite.py`
- Family: `authorial`
- Metric ID prefix: `style.stylometry_`
- Cost: `moderate` for dependency-free features; parse/model-backed subfeatures may be skipped or marked unavailable unless their shared analysis is already available.
- Reuse `DocumentAnalysis`; do not independently clean, tokenize, split sentences, or parse dialogue.
- Add Python dependencies to `textgrader.optional.PACKAGES` and degrade individual subfeatures when unavailable.
- External R/Java/CLI tools must be optional. If integrated directly, wrap them so failure produces an unavailable/error finding rather than failing the grading run.

## Metrics to add

Emit as many individual findings as practical rather than one opaque authorship score.

**Reference-author / reference-document distance metrics**

- Nearest-author distance.
- Second-nearest-author distance.
- Nearest-vs-second-nearest margin.
- Distance to reference-author centroid.
- Distance to overall corpus centroid.
- Within-author dispersion for the nearest reference author.
- Standardized distance relative to that author's within-author dispersion.
- Out-of-distribution distance.
- k-nearest-neighbor author agreement.
- Section-to-section style stability within the analyzed text.
- Rolling nearest-author identity and identity-switch count where reference authors exist.
- Cross-entropy/perplexity under author-specific language models.
- Normalized Compression Distance to reference authors/documents.
- Impostor/unmasking-style verification scores where the library supports them.

**Feature representations — keep each separately**

- Character 2-, 3-, 4-, 5-, and 6-grams.
- Byte n-grams.
- Word 1-, 2-, and 3-grams.
- Function-word frequencies and function-word n-grams.
- Stopword sequences.
- POS 1-, 2-, 3-, and 4-grams.
- Dependency-label n-grams.
- Dependency-relation + POS combinations.
- Punctuation unigrams and punctuation n-grams.
- Word-shape n-grams, such as capitalization/digit/punctuation patterns.
- Prefix and suffix n-grams.
- Sentence-opening lexical patterns.
- Sentence-opening POS/dependency patterns.
- Contraction preferences.
- Capitalization habits.
- Affix frequencies.
- Word-length histogram.
- Sentence-length histogram.
- Paragraph-length histogram.
- Function-word rank profile.
- Punctuation rank profile.

**Lexical/stylometric scalar measures**

- Hapax legomena ratio.
- Dislegomena ratio.
- Yule's K.
- Yule's I.
- Honoré's R.
- Sichel's S.
- Herdan's C.
- Guiraud's R.
- Maas index.
- Dugast index/variants supported by the selected implementation.
- Uber index.
- Vocabulary growth curve.
- Heaps-law exponent and fit residual.
- Zipf slope, intercept, goodness-of-fit, and residual statistics.
- Stopword entropy.
- Function-word rank stability across sections.
- Punctuation rank stability across sections.
- Word-length histogram distance between sections.
- Sentence-length histogram distance between sections.
- Style-vector drift from opening to close.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **pystylometry** | Burrows' Delta, Cosine Delta, Zeta, Kilgarriff chi-square, MinMax, John's Delta, NCD, lexical-richness measures, n-gram entropy, independent cross-checks. |
| **stylo** (R) | Delta variants, PCA, clustering, bootstrap consensus, rolling stylometry, mature independent implementation. |
| **PyDelta** | Independent Burrows/Eder/Delta-family implementation. Keep disagreement with TextGrader's existing `function_words` metric. |
| **MOWEN** | Authorship attribution/verification, style change, impostors, unmasking, feature/event drivers, distance combinations. |
| **JGAAP** | Additional older authorship event cullers and distance functions; useful specifically because it is methodologically independent. |
| **stylometry-cli** | Character-trigram/frequent-word Delta-style cross-checks where usable. |
| **scikit-learn** | TF-IDF/count vectors, LinearSVC, logistic regression, nearest centroid, kNN, PCA/SVD, calibration, feature normalization. |
| **fastText** | Lightweight supervised style/authorship classifier and word/character-subword representation. |
| **Vowpal Wabbit** | Very large sparse feature spaces and optional online authorship/style classification. |
| **MALLET** | MaxEnt/classical statistical classification as an independent implementation. |
| **KenLM** | Author-specific character/word/POS n-gram language models and cross-entropy/perplexity. |
| **pyppmd / PPM compressors** | PPM-based compression likelihood and normalized compression distance. |
| **textdistance** | Multiple profile/vector/string distances; expose the useful ones independently rather than choosing one. |
| **SciPy / NumPy** | Rank correlations, standardized distances, histogram distances, robust summaries. |

## Why

Authorship and style are distributed across many weak signals. Character n-grams capture spelling, morphology, punctuation, contractions, and fragments of syntax simultaneously. Function words are comparatively topic-resistant. POS/dependency patterns expose syntactic habits. Compression and n-gram cross-entropy provide completely different views of predictability. A system that preserves all of these is more useful than one classifier that returns an author label.

The suite should also work when no author labels exist. In that case it still emits internal style stability, vocabulary/style statistics, section-to-section distances, corpus-centroid distance, and corpus-neighbor distances.

## Implementation details

1. Build reusable stylometric feature extractors that accept `DocumentAnalysis` or text slices from it. Cache each feature representation in `analysis._shared` because multiple distance calculations will reuse it.
2. Keep representations separated. Do not concatenate everything and only report one vector distance.
3. For corpus comparison, extend corpus profiles with the feature profiles required for each enabled stylometry representation. Store enough metadata to verify feature settings match: n-gram order, vocabulary size/culling threshold, lowercasing, stopword/function-word list version, normalization, and whether dialogue/narration were included.
4. When author labels are available in corpus metadata, create per-author centroids and within-author distributions. When they are not available, treat corpus texts as reference documents and emit nearest-document/centroid/OOD metrics only.
5. Use cosine, Manhattan, Euclidean, standardized Euclidean, Delta-family distances, Jensen-Shannon where appropriate, and library-specific distances as separate findings.
6. Add rolling/section-level analysis using `analysis.sections` or canonical windows. Do not re-read the source file. Report median section distance, maximum section distance, early-vs-late distance, and change count above configurable thresholds.
7. For KenLM/PPM approaches, train reference models offline as part of corpus/profile construction when enabled. The grading path should load artifacts, not retrain a model for every document.
8. For supervised classifiers, return calibrated probability/margin only when training data is sufficient. Never invent a probability from raw SVM distance.
9. Retain topic-resistant and topic-sensitive variants separately. This allows later validation to determine whether a feature is measuring author, genre, or topic.
10. All high-dimensional models must be deterministic under a configured seed.

## Corpus/profile requirements

The corpus builder must optionally retain:

- Per-document stylometric feature vectors or a compact serialized representation.
- Optional author labels and other metadata if supplied.
- Feature vocabulary/culling settings.
- Per-author centroid/dispersion statistics.
- Corpus centroid/covariance where safe.
- Optional KenLM/PPM model artifact paths or identifiers.
- Metric distributions for every scalar output so ordinary TextGrader percentile/severity handling can be used.

Do not force author metadata into the normal corpus format when it is absent. The suite must still work against unlabeled corpora.

## Tests and validation

- Same-author vs different-author tests using a small fixture corpus.
- Topic-confound test: ensure the test split is by document, not random chunks from one document.
- Length-confound test: compare long and short slices from the same author.
- Section-shift synthetic test by concatenating two stylistically different fixture texts.
- Confirm character 2/3/4/5/6-gram findings are all retained separately.
- Confirm missing external tools only remove their own findings.
- Confirm profile option mismatch withholds comparison rather than comparing incompatible vectors.
- Confirm deterministic output with a fixed seed.
- Add a validation report showing correlations among stylometry findings, but do **not** use correlation as a reason to delete them.

## Acceptance criteria

- The suite emits multiple independent stylometric families rather than a single authorship result.
- It works with unlabeled corpora and becomes richer when author labels are present.
- It exposes nearest, second-nearest, margin, centroid, dispersion, OOD, and section-stability measurements.
- It includes character/byte/word/function-word/POS/dependency/punctuation/shape/affix feature families.
- It includes the listed lexical stylometric statistics.
- KenLM/compression-based measurements are supported as optional independent channels.
- No external dependency can crash the main TextGrader run.
- All comparable scalar findings can be profiled and compared to the example/reference corpus.

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
