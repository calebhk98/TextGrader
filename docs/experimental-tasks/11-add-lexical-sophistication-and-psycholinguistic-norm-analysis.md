# 11. Add lexical sophistication and psycholinguistic norm analysis

## Goal

Measure word choice along many external normative dimensions: frequency, range, age of acquisition, concreteness, imageability, familiarity, valence/arousal/dominance, sensorimotor grounding, contextual distinctiveness, association strength, and lexical sophistication/diversity. Preserve means, tails, variance, and within-document drift separately.

## Suggested TextGrader integration

- Config switch: `lexical_norms_suite`
- Primary module: `textgrader/metrics/lexical_norms_suite.py`
- Helper: `textgrader/lexicons.py` for cached/versioned norm lookup tables.
- Family: `lexical`
- Metric ID prefix: `lexical.norm_`
- Cost: `moderate`; external norm files should be loaded once and cached.

## Metrics to add

For each available norm/resource, emit:

- Coverage percentage.
- Mean.
- Median.
- Standard deviation/robust dispersion.
- 10th/25th/75th/90th quantiles.
- Low-tail and high-tail share using norm-specific thresholds/reference quantiles.
- Sentence-level mean distribution.
- Paragraph-level mean distribution.
- Early-vs-late drift.
- Between-paragraph variance.
- Dialogue-vs-narration difference where applicable.

Norm dimensions to include:

- General frequency and Zipf frequency.
- Corpus range/dispersion.
- SUBTLEX frequency/contextual diversity.
- Age of acquisition.
- Concreteness.
- Imageability.
- Familiarity.
- Valence.
- Arousal.
- Dominance.
- Sensorimotor strength by modality/action dimension.
- Contextual distinctiveness.
- Lexical association strength.
- Academic/general vocabulary bands where data is available.
- Morphological family size where available.
- Reaction-time / lexical-decision norms.

Also expose/cross-check lexical sophistication/diversity metrics from TAALES/TAALED/LFTK instead of replacing existing MATTR/MTLD/HD-D.

## Libraries / tools / resources

| Library/tool/resource | Add/use it for |
| --- | --- |
| **TAALES** | Lexical sophistication, frequency/range, n-grams, association and contextual-distinctiveness measures. |
| **TAALED** | Large set of lexical-diversity variants. |
| **LFTK** | Broad handcrafted lexical/surface/syntax/readability features and independent implementations. |
| **wordfreq** | Existing general-language frequency; keep as one channel. |
| **WordNet** | Synset/polysemy/semantic relation counts and lexical structure. |
| **SUBTLEX-US / compatible SUBTLEX resource** | Subtitle frequency and contextual diversity. |
| **English Lexicon Project** | Lexical-decision/reaction-time and lexical norm information where licensing permits. |
| **MRC Psycholinguistic Database** | Familiarity, imageability, concreteness and related norms. |
| **CELEX** | Lexical/morphological/frequency information where licensed/available. |
| **Kuperman age-of-acquisition norms** | AoA. |
| **Brysbaert concreteness norms** | Concreteness. |
| **Warriner VAD** | Valence/arousal/dominance. |
| **NRC VAD** | Independent/broader VAD norms. |
| **Lancaster Sensorimotor Norms** | Perceptual/action grounding. |
| **WordNorms or equivalent aggregation package** | Unified loading where licensing and versions are explicit. |

## Why

Two texts with identical average word length or frequency can differ strongly in concreteness, AoA, emotional norms, sensorimotor grounding, dispersion, or tail behavior. These are measurable dimensions of lexical choice and can be useful even when highly correlated.

## Implementation details

1. Do not silently redistribute copyrighted/licensed datasets. Add download/setup instructions or user-supplied resource paths where required.
2. Normalize tokens using canonical TextGrader tokens/lemmas. Keep surface-form and lemma-based lookups as separate findings if both are useful.
3. Record coverage for every norm. A mean over 12% of words must not look equivalent to a mean over 93%.
4. Weighting policy must be explicit: token-weighted vs type-weighted results should be separate if both are emitted.
5. Cache resource tables globally or in `analysis._shared`; do not reload per metric.
6. Preserve independent resource versions even when two resources measure the same concept, e.g. Warriner vs NRC VAD.
7. For ambiguous words with multiple WordNet senses, do not claim sense-specific norms without disambiguation. Emit surface lexical properties separately.
8. Add optional moving-window trends for each major norm, but cap output volume.

## Corpus/profile requirements

Store scalar distributions plus resource/version/coverage policy. A corpus built with one norm dataset version should not be directly compared against a different one without warning.

## Tests and validation

- High-frequency/simple vocabulary vs rare vocabulary.
- Concrete vs abstract word fixture.
- Positive vs negative VAD fixture.
- Low-coverage text with names/technical jargon.
- Verify token-weighted and type-weighted calculations differ when expected.
- Missing licensed resource only disables its metrics.

## Acceptance criteria

- The suite covers frequency/range, AoA, concreteness/imageability/familiarity, VAD, sensorimotor and lexical-decision dimensions when resources are available.
- Coverage is always reported.
- Means do not replace variance/quantiles/tails/drift.
- Independent resources measuring similar concepts remain separate.

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
