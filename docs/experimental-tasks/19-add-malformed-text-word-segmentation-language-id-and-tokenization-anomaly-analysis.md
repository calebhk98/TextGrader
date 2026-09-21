# 19. Add malformed-text, word-segmentation, language-ID, and tokenization-anomaly analysis

## Goal

Detect text that is malformed at the token/language level: fused words, missing spaces, accidental splits, code-switching, wrong-language fragments, bizarre token shapes, OCR damage, and disagreement among language/segmentation systems.

## Suggested TextGrader integration

- Config switch: `malformed_text_suite`
- Primary module: `textgrader/metrics/malformed_text_suite.py`
- Family: `lexical`
- Metric ID prefix: `lexical.malformed_`
- Cost: `fast`/`moderate`.

## Metrics to add

- Fraction of tokens confidently recognized by general frequency/dictionaries.
- Very-low-frequency token share.
- Fused-word candidate rate.
- Split-word candidate rate.
- Word-segmentation correction cost/ambiguity.
- Average number of plausible segmentations for suspicious tokens where available.
- Long token rate and long alphabetic token tail.
- Mixed alphanumeric token rate.
- Repeated-symbol token rate.
- Token character-class entropy.
- Language ID for document, paragraph, and sentence.
- Language-ID confidence distribution.
- Code-switch rate.
- Language-ID entropy over document sections.
- Disagreement among language detectors.
- Fraction of sentences classified as a different language than document majority.
- Segmentation-tool disagreement on suspicious no-space strings.
- Dictionary/frequency/language-model disagreement for suspicious tokens.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **wordninja** | Statistical splitting of concatenated words. |
| **wordsegment** | Independent word segmentation. |
| **SymSpellPy** | Segmentation/spelling correction with edit distance. |
| **fastText language identification** | Fast language-ID channel. |
| **Lingua** | High-accuracy language detection, useful for short text. |
| **langid.py** | Independent statistical language ID. |
| **langdetect** | Independent baseline despite known quirks; disagreement is useful. |
| **CLD3 / pycld3** | Neural/compact language identification where installable. |
| **wordfreq** | Existing frequency signal. |
| **regex/unicodedata** | Token-shape and script detection. |

## Why

A text can look statistically bizarre because it is in the wrong language, contains fused words, or is damaged by OCR/encoding. Measuring those conditions separately prevents downstream metrics from being misinterpreted. Disagreement among language detectors is particularly useful on short, mixed, or corrupted passages.

## Implementation details

1. Use canonical tokens but retain character offsets for evidence.
2. Only run expensive segmentation candidates on suspicious tokens, not every ordinary word.
3. Separate word-segmentation confidence from “correction.” Do not alter the canonical document.
4. Language-ID should operate at document + paragraph + sentence levels with configurable minimum characters.
5. Normalize detector labels to BCP-47/ISO codes where possible, but retain original detector label in details.
6. Set deterministic seeds for detectors that are nondeterministic or initialize them once.
7. Script detection should distinguish Latin/Cyrillic/Greek/etc. so homoglyph/mixed-script anomalies can be identified.

## Corpus/profile requirements

Profile normalized rates and disagreement statistics. Record detector versions and language configuration.

## Tests and validation

- Normal English prose.
- Concatenated English words.
- Random spaces inserted inside words.
- English text with Spanish/French paragraph.
- Very short ambiguous strings.
- Mixed-script homoglyph fixture.
- OCR-like tokens.

## Acceptance criteria

- Multiple language detectors and multiple segmentation systems are supported.
- Disagreement is explicitly reported.
- The task never silently “fixes” input text.
- Suspicious token evidence is bounded and offset-aware.

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
