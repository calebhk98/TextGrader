# 12. Add multiple non-generative semantic-structure and topic-model channels

## Goal

Describe semantic organization using multiple independent representations rather than relying only on sentence-transformer embeddings. Add static embeddings, LSA/SVD, LDA, NMF, HDP, BM25/lexical similarity, and topic-transition statistics.

## Suggested TextGrader integration

- Config switch: `semantic_structure_suite`
- Primary module: `textgrader/metrics/semantic_structure_suite.py`
- Family: `semantic`
- Metric ID prefix: `semantic.structure_`
- Cost: `moderate`/`model` depending on representation.

## Metrics to add per representation

- Adjacent-sentence similarity/distance.
- Adjacent-paragraph similarity/distance.
- Similarity to document centroid.
- Local-window semantic drift.
- Global semantic dispersion.
- Lowest-similarity transitions.
- Topic distribution entropy.
- Dominant-topic probability/confidence.
- Topic-switch count/rate.
- Topic recurrence interval.
- Topic persistence/run lengths.
- Number of active topics above threshold.
- Topic balance/concentration.
- Paragraph-to-title relevance when a title is available.
- Conclusion-to-introduction similarity.
- Opening-to-closing semantic distance.
- Agreement/disagreement across semantic representations.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **Gensim** | Word2Vec, FastText wrappers, Doc2Vec, LSI/LSA, LDA, HDP. |
| **GloVe vectors** | Independent static embedding representation; load pre-trained vectors rather than requiring a specific Python package. |
| **fastText** | Subword-aware static embeddings and optional document vectors. |
| **spaCy static vectors** | Another independent static vector channel when a vector model is installed. |
| **scikit-learn** | TF-IDF + TruncatedSVD/LSA, NMF, LDA. |
| **tomotopy** | Efficient topic modeling, including LDA/HDP-family models. |
| **MALLET** | Independent LDA/topic model implementation. |
| **rank_bm25 / BM25 implementation** | Lexical retrieval-style similarity between sentences/paragraphs. |
| **sentence-transformers** | Existing embedding channel; reuse current encoding cache and retain as one representation among many. |

## Why

Different semantic methods encode different assumptions. BM25/TF-IDF are lexical. LSA captures low-rank co-occurrence. LDA/NMF expose topic mixtures. Static embeddings encode word-level distributional similarity. Sentence-transformers encode contextual semantics. Text that looks coherent to one representation may look discontinuous to another; preserve that disagreement.

## Implementation details

1. Add a representation registry so each representation can expose sentence and/or paragraph vectors/topic distributions with metadata.
2. Reuse existing semantic metrics/caches. Do not re-encode sentences with the same sentence-transformer model.
3. Topic models should normally be trained on the reference corpus, not on one short test document. For very long books, an optional within-document topic model may be a separate metric ID.
4. Keep model-specific topic IDs local; compare statistical properties such as entropy/persistence rather than pretending topic 3 from LDA equals topic 3 from NMF.
5. For Word2Vec/GloVe/FastText document vectors, define aggregation explicitly: mean, TF-IDF-weighted mean, SIF if added. Keep alternative aggregations separate where implemented.
6. BM25 similarity should use a clearly defined document/sentence corpus and avoid treating BM25 scores as cosine-like bounded similarities.
7. Add representation-disagreement metrics: rank correlation among adjacent-transition scores, transitions flagged only by one model, and consensus low-coherence transitions.
8. Record model artifact IDs, training corpus, dimensions and preprocessing.

## Corpus/profile requirements

Store trained topic/model artifacts or references plus scalar metric distributions. Model settings/version must match before corpus comparison. For static pre-trained vectors, record exact vector set/version.

## Tests and validation

- Topic-consistent paragraph vs abrupt topic shift.
- Lexical paraphrase where TF-IDF differs more than embeddings.
- Repeated lexical terms with semantic discontinuity.
- Short text insufficient-data handling.
- Model-unavailable fallbacks remain visible and independent.

## Acceptance criteria

- At least lexical/BM25, LSA, topic-model, static-embedding, and existing contextual-embedding channels are represented.
- Topic transitions and semantic transitions are measured separately.
- Representation disagreement is exposed.
- Corpus-trained models are versioned and not silently retrained on the graded text.

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
