# 15. Add sentiment, emotion, affect, and tone-trajectory analysis

## Goal

Measure emotional/tone structure using multiple independent lexicons/models. Preserve overall level, variance, local volatility, arcs, dialogue/narration differences, and disagreement among sentiment systems.

## Suggested TextGrader integration

- Config switch: `affect_suite`
- Primary module: `textgrader/metrics/affect_suite.py`
- Family: `discourse` or `lexical` depending on metric.
- Metric IDs: `discourse.affect_...`
- Cost: `moderate`.

## Metrics to add

For each sentiment/emotion implementation where possible:

- Document mean/median sentiment.
- Positive/negative/neutral share.
- Sentiment variance and interquartile range.
- Sentence/paragraph sentiment volatility.
- Adjacent sentiment delta distribution.
- Emotional reversal count.
- Longest positive/negative run.
- Early/middle/late arc values.
- Linear/nonlinear trend features.
- Dialogue-vs-narration difference.
- Per-speaker affect profiles where attribution exists.
- Emotion-category distribution: anger, fear, joy, sadness, disgust, surprise, trust, anticipation, etc. according to resource ontology.
- Emotion entropy/concentration.
- Valence/arousal/dominance trajectory.
- System disagreement: sign disagreement, rank correlation, high-confidence disagreement rate.
- Correlation between affect and sentence length/rhythm.
- Correlation between affect and lexical rarity/syntactic complexity where sample size permits.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **VADER** | Rule/lexicon sentiment, especially punctuation/intensification-sensitive prose/dialogue. |
| **AFINN** | Independent lexicon sentiment. |
| **TextBlob / Pattern sentiment** | Another independent baseline. |
| **NRC Emotion Lexicon** | Emotion-category counts. |
| **NRC VAD** | Valence/arousal/dominance. |
| **LIWC** | Broad psychological/style categories if the user supplies a licensed dictionary. Do not redistribute proprietary data. |
| **Empath** | Broad lexical topic/psychological categories. |
| **SEANCE** | Sentiment/cognition feature suite. |
| **sentimentr** (R) | Sentence-aware sentiment with valence shifters as an independent implementation. |
| **scipy/statsmodels** | Arc/trend/correlation calculations. |

## Why

Average sentiment alone is nearly useless for many texts. Emotional dynamics, variance, reversal, speaker differences, and cross-system disagreement are more informative. These measures are descriptive sensors, not judgments that positive writing is better than negative writing.

## Implementation details

1. Apply tools at sentence and paragraph level, then aggregate. Avoid running a black-box “whole book” score only.
2. Reuse canonical sentence/paragraph segmentation.
3. Keep each sentiment engine separate and add explicit disagreement metrics.
4. Map emotion lexicon categories without forcing ontologies to match. NRC categories remain NRC categories; LIWC categories remain LIWC categories.
5. Licensed dictionaries must be user-supplied and configured by path; missing license/resource produces unavailable findings only for those channels.
6. Provide trajectory windows for long documents and retain section-level distributions without dumping every point in output.
7. All correlations with other sequences should include sample size and only run with sufficient aligned samples.

## Corpus/profile requirements

Store per-engine scalar distributions and engine/resource version. Corpus comparison must be withheld across incompatible resource versions.

## Tests and validation

- Strongly positive vs negative fixtures.
- Alternating emotional sentences to distinguish mean from volatility.
- Neutral technical prose.
- Dialogue/narration tone contrast.
- Ensure engines are allowed to disagree and that disagreement is reported.

## Acceptance criteria

- Multiple independent affect systems are supported.
- Mean, variance, volatility, arc, runs, dialogue/narration and disagreement metrics exist.
- Licensed resources are optional and not bundled illegally.
- No affect metric is automatically interpreted as quality.

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
