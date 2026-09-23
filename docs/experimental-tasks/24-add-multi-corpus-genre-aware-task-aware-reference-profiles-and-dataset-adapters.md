# 24. Add multi-corpus, genre-aware, task-aware reference profiles and dataset adapters

## Goal

Expand TextGrader from one generic reference corpus to a library of optional reference corpora and metadata-aware profiles. A text should be able to receive several independent reference-fit vectors — e.g. general prose, fiction, newspaper, essays, poetry, academic prose — without requiring the grader to force a single genre classification first.

## Suggested TextGrader integration

- This task touches `corpus_builder/`, `textgrader/corpus.py`, configuration, and report/profile selection more than one metric module.
- Config block suggestion: `reference_profiles` or `corpora` rather than a normal `metrics` switch.
- Optional metric module: `textgrader/metrics/reference_fit.py`
- Family: `distribution_shape`
- Metric IDs: `style.reference_fit_<profile>_...`
- Preserve current single-profile behavior for backward compatibility.

## Corpora/datasets to support or provide adapters/documented recipes for

**General/literary corpora**

- Project Gutenberg — existing acquisition path; retain.
- Brown Corpus.
- British National Corpus where licensing permits.
- Open American National Corpus.
- WikiText and other clean encyclopedic text sources where licensing permits.
- Google Books Ngrams for aggregate historical frequency signals where appropriate, not as a direct document corpus.

**News**

- Reuters Corpus / Reuters-21578 where licensing permits.
- Other legally usable news datasets with clear source/license metadata.

**Syntax/annotation**

- Penn Treebank.
- Universal Dependencies corpora.
- GUM.
- OntoNotes where licensed/available.

**Discourse/coherence**

- RST Discourse Treebank.
- Penn Discourse Treebank.

**Authorship/stylometry**

- PAN authorship attribution/verification/style-change corpora.
- Public-domain author-grouped literary corpora built from Gutenberg metadata.

**Literature/characters**

- LitBank.
- Other book-scale entity/coreference datasets where usable.

**Logic/NLI**

- SNLI.
- MNLI.
- Other NLI/contradiction sets only as calibration/validation data, not as a reference for prose quality.

**Essay/scoring**

- Public or licensed essay-scoring datasets where legal terms permit use.
- Human-scored writing datasets with explicit rubric metadata where available.

**Lexical frequency/norm resources**

- SUBTLEX.
- COCA if licensing permits.
- BNC frequency lists where permitted.
- Psycholinguistic norm datasets from Task 11.

## Metrics / outputs to add

For each configured reference profile:

- Existing per-metric percentile/severity under that profile.
- Overall profile-fit distance using a documented multivariate feature set.
- Number/share of measured metrics inside the profile's central bands.
- Median absolute standardized distance from profile center.
- Nearest reference profile by distance — report as descriptive only, not forced genre truth.
- Second-nearest profile and margin.
- Reference-profile disagreement: which profile considers the text most/least typical on which metrics.
- Genre/reference-fit vector containing one independent score per profile.
- Coverage: number of metrics that could be validly compared for each profile.
- Time-period fit where profiles are historical/dated.

## Libraries / tools

| Tool/resource | Add/use it for |
| --- | --- |
| Existing `corpus_builder/` | Extend metadata/adapters instead of creating an unrelated download pipeline. |
| **NLTK corpora** | Brown/Reuters and other packaged corpora where appropriate. |
| **datasets** (Hugging Face datasets library) | Optional standardized access to openly licensed datasets; pin dataset/config/revision. |
| **Universal Dependencies tooling** | Download/normalize UD treebanks where syntax-specific reference profiles are needed. |
| **PAN datasets/tools** | Authorship/style-change validation/reference data. |
| **pandas/pyarrow** | Larger metadata/profile tables and optional artifact storage. |
| Existing TextGrader profile/validation code | Leave-one-out calibration and comparison-unit safeguards must remain central. |

## Why

A poem, newspaper article, Victorian novel and student essay should not be expected to occupy the same region of every metric. Hard-coding genre-specific scoring rules into individual metrics would be brittle. Multiple independent reference profiles let the same raw measurement be interpreted against several populations while preserving the raw values.

## Implementation details

1. Preserve `corpus_profile` backward compatibility. Add an optional list/map of additional profiles.
2. Every profile needs metadata: corpus name, source, license/terms note, language, date range, genre/domain labels, comparison unit, text-processing fingerprint, metric settings, build date and dataset revision/hash.
3. Do not automatically download corpora with restrictive licenses. Provide adapters/configuration and clear setup errors.
4. Never compare incompatible text-processing settings or metric settings; reuse current safeguards.
5. A document can be compared to multiple profiles in one run. Do not choose one and discard the others.
6. Profile-fit aggregate must report feature count/coverage and should use robust standardized distances. Keep several candidate distances if useful rather than one authoritative genre score.
7. Leave-one-out validation must be available per profile. Profiles with poor calibration should expose that fact.
8. Allow profiles grouped by comparison unit: book-to-books, chapter-to-chapters, essay-to-essays, poem-to-poems, etc.
9. Add profile aliases/configuration so large external artifact paths are not repeated throughout config.
10. Dataset adapters must record exact revision/version so results are reproducible.

## Tests and validation

- Two tiny synthetic reference profiles with deliberately different distributions; same test text should receive different percentiles.
- Incompatible processing fingerprint must withhold comparison.
- Multiple-profile results coexist in one report.
- Backward compatibility with existing single `corpus_profile` configuration.
- Missing/licensed-unavailable dataset adapter fails gracefully.
- Leave-one-out validation on a small profile.
- Comparison-unit mismatch test.

## Acceptance criteria

- TextGrader supports multiple simultaneous reference profiles without losing current behavior.
- Profiles carry source/license/version/genre/time/comparison-unit metadata.
- A text receives a reference-fit vector rather than a forced genre label.
- Existing per-metric comparisons can be repeated against each valid profile.
- The listed major corpora/dataset classes have adapters or documented reproducible ingestion recipes where licensing allows.

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
