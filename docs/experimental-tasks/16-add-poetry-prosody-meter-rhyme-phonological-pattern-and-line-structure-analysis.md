# 16. Add poetry/prosody, meter, rhyme, phonological-pattern, and line-structure analysis

## Goal

Make TextGrader meaningfully analyze poetry rather than treating line breaks and fragments only as prose anomalies. Add meter/stress/rhyme/phoneme/alliteration/assonance/consonance/line/stanza measurements, while also allowing these signals to run experimentally on prose.

## Suggested TextGrader integration

- Config switch: `prosody_suite`
- Primary module: `textgrader/metrics/prosody_suite.py`
- Helper: `textgrader/prosody.py` for line/stanza/phoneme representations.
- Family: `sentence_rhythm` or introduce `prosody`; metric IDs under `rhythm.prosody_` to fit existing prefix tests.
- Cost: `moderate`.

## Metrics to add

**Line/stanza structure**

- Line count and nonblank poetic-line count.
- Words/line and syllables/line distributions.
- Line-length CV/entropy.
- Stanza count, lines/stanza distribution and stanza symmetry.
- Repeated line-length patterns.
- Enjambment proxies: lines lacking terminal punctuation/syntactic closure.

**Stress/meter**

- Stress-pattern distribution.
- Dominant meter candidate and confidence if provided by Prosodic.
- Meter conformity rate.
- Meter variation/deviation rate.
- Feet/line distribution.
- Stress entropy and periodicity.

**Rhyme**

- End-rhyme density.
- Perfect rhyme rate.
- Slant/near-rhyme rate where supported.
- Rhyme-scheme regularity.
- Rhyme-class entropy.
- Internal-rhyme proxy/rate.
- Rhyme recurrence distance.

**Phonological patterning**

- Phoneme entropy.
- Vowel/consonant balance.
- Alliteration density.
- Assonance density.
- Consonance density.
- Repeated onset/coda patterns.
- Phoneme n-gram repetition.
- Sound-pattern recurrence by line/paragraph.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **Prosodic** | Meter, stress, syllable/phoneme and rhyme-oriented analysis. |
| **Poesy** | Older poetry-processing layer; use as an optional independent channel if still runnable. |
| **CMU Pronouncing Dictionary / cmudict** | Pronunciation/stress lookup. |
| **pronouncing** | Convenient CMUdict access/rhyme operations. |
| **g2p-en** | Grapheme-to-phoneme fallback for unknown words. |
| **phonemizer** | Additional pronunciation backend. |
| **PanPhon** | Phonological feature vectors/distances for near-rhyme and sound similarity. |
| **regex / spaCy** | Line punctuation/syntactic-closure proxies where appropriate. |

## Why

Poetry intentionally violates many prose conventions. A general text-analysis system that claims broad genre support should measure the structure poetry actually uses. Phonological regularity can also be an experimental signal for prose cadence, slogans, advertising, speeches, and repetitive/generated text.

## Implementation details

1. Preserve physical line breaks from the canonical analyzed text. Do not use sentence segmentation as a substitute for poetic lines.
2. Detect likely poetry only as metadata/confidence; do not disable prosody metrics on prose. The user explicitly wants weird tests to remain available.
3. Use pronunciation lookup with fallback. Always report pronunciation coverage.
4. Keep exact rhyme, phonetic-near-rhyme and embedding/feature-based phonetic similarity separate.
5. Enjambment must be labeled a proxy unless a syntactic analysis establishes stronger evidence.
6. Handle punctuation-free/free-verse text without treating it as an internal error.
7. Store bounded example lines for strongest repeated/rhyme patterns.

## Corpus/profile requirements

Profile poetry against poetry when available, but do not prevent comparison against general corpora. Record pronunciation backend/version and language. Add genre-specific reference profiles later rather than hard-coding poetry expectations into the metric.

## Tests and validation

- Rhymed metrical poem fixture.
- Free verse fixture.
- Ordinary prose fixture.
- Unknown/proper-noun-heavy words for pronunciation coverage.
- Deliberate alliteration/assonance fixture.

## Acceptance criteria

- Line, stanza, stress/meter, rhyme and phonological-pattern metrics exist.
- Pronunciation coverage is explicit.
- Free verse and prose do not crash or get force-fit into a meter.
- Metrics remain neutral and corpus-relative.

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
