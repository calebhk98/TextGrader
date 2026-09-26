# 18. Add approximate duplication, fuzzy reuse, structural reuse, and motif detection

## Goal

Detect exact and approximate textual reuse at several representation levels: surface wording, normalized wording, lemmas, function words, POS patterns, punctuation patterns, and fuzzy hashes. This extends current exact repeated n-grams/local repetition into paraphrase-like and template-like reuse.

## Suggested TextGrader integration

- Config switch: `reuse_suite`
- Primary module: `textgrader/metrics/reuse_suite.py`
- Family: `repetition`
- Metric ID prefix: `repetition.reuse_`
- Cost: `moderate`; book-scale pairwise comparisons must use indexing/LSH/candidate generation.

## Metrics to add

- Exact duplicate sentence fraction.
- Near-duplicate sentence fraction at multiple similarity thresholds.
- Exact/near-duplicate paragraph fraction.
- Longest repeated substring/token sequence.
- Longest approximate repeated sequence.
- Repeated template count.
- MinHash/Jaccard similarity distribution among sentence/paragraph candidates.
- SimHash Hamming-distance nearest-neighbor distribution.
- TLSH/ssdeep similarity where appropriate for longer blocks.
- Normalized Levenshtein distance nearest-neighbor statistics.
- Jaro/Jaro-Winkler nearest-neighbor statistics.
- Token-set/token-sort fuzzy similarity statistics.
- Reuse distance: how far apart repeated/near-repeated units occur.

Run equivalent reuse analysis after transformations:

- Lowercase/normalized text.
- Lemmas.
- Function words only.
- Stopwords only.
- POS tag sequences.
- Dependency-label sequences where available.
- Punctuation only.
- Word-shape patterns.
- Content words only.

The purpose of transformed views is to detect repeated grammatical or stylistic chassis even when surface vocabulary changes.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **datasketch** | MinHash/LSH approximate Jaccard candidate generation. |
| **simhash** implementations | SimHash fingerprints and Hamming distance. |
| **TLSH** | Locality-sensitive fuzzy hashing for sufficiently long blocks. |
| **ssdeep** | Context-triggered piecewise fuzzy hashing where available. |
| **RapidFuzz** | Fast token/string fuzzy similarity. |
| **python-Levenshtein** | Fast edit distance. |
| **textdistance** | Multiple string/token distance families. |
| **Jellyfish** | Jaro/Jaro-Winkler/phonetic string comparisons. |
| **pyahocorasick** | Fast exact multi-pattern matching where useful. |
| Suffix array/tree libraries or custom efficient implementation | Longest repeated substrings/sequences. |

## Why

Exact n-grams miss paraphrased templates; semantic embeddings can miss structural reuse. Function-word/POS/punctuation-only comparisons can reveal repeated scaffolding even when content words change. Fuzzy-hash and edit-distance methods supply independent notions of “same-ish.”

## Implementation details

1. Do not perform all-pairs comparison on all sentences/paragraphs. Use MinHash/LSH, fingerprint buckets, n-gram inverted indexes, or windowed candidate generation.
2. Keep each representation and similarity method separate.
3. Raw duplicate counts are unit-sensitive; include normalized shares/rates.
4. Preserve distance between occurrences so local rhetorical repetition differs from book-wide template recurrence.
5. Add configurable minimum unit length; fuzzy hashes behave poorly on very short strings.
6. Exclude trivial boilerplate detected by canonical preprocessing where appropriate, but do not silently remove repeated prose.
7. Keep strongest bounded evidence pairs with offsets/section identifiers.

## Corpus/profile requirements

Profile normalized reuse rates and similarity summaries. Store transform settings, thresholds and hash/library versions.

## Tests and validation

- Exact repeated sentence.
- Minor edit/paraphrase.
- Same POS skeleton with different content words.
- Same punctuation pattern only.
- Long book-like fixture to ensure subquadratic candidate generation.
- Very short sentences must not abuse TLSH/ssdeep.

## Acceptance criteria

- Exact, fuzzy, hash-based and transformed-structure reuse channels exist.
- Approximate search is bounded for long documents.
- Evidence identifies representative repeated pairs.
- Existing exact n-gram metrics remain untouched and complementary.

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
