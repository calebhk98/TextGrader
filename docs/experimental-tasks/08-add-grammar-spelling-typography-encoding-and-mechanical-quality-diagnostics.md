# 8. Add grammar, spelling, typography, encoding, and mechanical-quality diagnostics

## Goal

Add independent mechanical-correctness sensors for grammar, agreement, spelling, punctuation rules, typography consistency, encoding damage, OCR-like corruption, Unicode anomalies, and formatting artifacts. Preserve rule categories rather than emitting only one error count.

## Suggested TextGrader integration

- Config switch: `mechanical_quality_suite`
- Primary module: `textgrader/metrics/mechanical_quality_suite.py`
- Family: `punctuation`, `lexical`, or `syntax` per finding.
- Metric ID prefixes: `nlp.mechanical_`, `punct.mechanical_`, `lexical.mechanical_`.
- Cost: `moderate`; LanguageTool can be expensive and should be bounded/cached.

## Metrics to add

**Grammar/style-rule diagnostics**

- Total LanguageTool matches per 1,000 words.
- Separate rates by LanguageTool category/rule family: grammar, agreement, punctuation, capitalization, typography, confusion pairs, repeated words, style, miscellaneous.
- Unique rule count.
- Repeated same-rule concentration.
- Sentences with any rule match.
- Maximum errors in one sentence/paragraph.

**Spelling / lexical validity**

- Unknown-word rate under Hunspell/Enchant.
- Unknown-word rate under SymSpell dictionary.
- Unknown-word rate under pyspellchecker.
- Contextual spelling issue rate under JamSpell if available.
- Spell-checker disagreement rate.
- Likely OCR substitution rate.
- Repeated-token / fused-token / split-token anomalies.

**Unicode/encoding/typography**

- ftfy repair candidate count and characters changed.
- Mojibake pattern count.
- Unicode-category distribution/entropy.
- Control-character rate.
- Zero-width character count/rate.
- Private-use character rate.
- Replacement-character `�` rate.
- Confusable/homoglyph candidate rate where feasible.
- Mixed straight/curly quote rate.
- Mixed apostrophe style rate.
- Mixed dash/hyphen style rate.
- Non-breaking-space rate.
- Repeated whitespace rate.
- Suspicious line-break/hyphenation rate.
- All-caps anomaly rate and unusual capitalization patterns.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **LanguageTool / language_tool_python** | **Not used: LanguageTool is a Java server, and TextGrader is Python only.** Grammar, spelling, punctuation, capitalization, style and rule-category diagnostics. |
| **Hunspell / pyenchant** | Independent dictionary-based spelling. |
| **SymSpellPy** | Edit-distance spelling/segmentation anomalies. |
| **pyspellchecker** | Another independent dictionary/frequency spell checker. |
| **JamSpell** | Contextual spell-checking channel when available. |
| **wordfreq** | Existing frequency information for ranking unknown/suspicious tokens. |
| **ftfy** | Mojibake/Unicode repair detection. |
| **charset-normalizer** | Encoding-confidence/anomaly information when source bytes are available. |
| **regex** | Unicode-aware suspicious-pattern detection with timeout discipline. |
| Python `unicodedata` | Unicode classes, normalization, combining marks and control categories. |
| **confusable_homoglyphs** or equivalent | Optional confusable/homoglyph detection. |

## Why

Mechanical corruption can masquerade as every other kind of text anomaly. OCR damage can break parsers, inflate vocabulary rarity, lower coherence, and cause sentence-segmentation disagreement. Measuring it directly allows later systems to distinguish “bad prose” from “bad input.” Multiple spell/grammar implementations are useful because dictionaries and rule systems disagree.

## Implementation details

1. LanguageTool results must be grouped by stable category and rule ID. Do not only report total errors.
2. Use per-1,000-word or per-sentence rates alongside raw counts. Raw counts should be marked unit-sensitive.
3. Spell-checkers should exclude obvious proper nouns, URLs, identifiers, and tokens already classified as non-word by the canonical pipeline where reasonable. Keep a raw variant if useful.
4. Preserve checker disagreement: number of tokens flagged by all, majority, only one, etc.
5. `ftfy` should be used diagnostically. Do not silently replace analyzed text inside this metric; report how much it *would* repair.
6. Unicode normalization comparisons should retain counts by category and representative bounded evidence.
7. Where source bytes are unavailable because `DocumentAnalysis` only has decoded text, encoding-confidence features must report unavailable rather than fabricate a result.
8. Grammar tools must support a max-text or chunking policy for book-length input.

## Corpus/profile requirements

Profile normalized rates and disagreement metrics. Record LanguageTool version/rule language and dictionary versions where possible. Rules change over time; model/version mismatch should withhold direct percentile comparison.

## Tests and validation

- Clean prose fixture.
- Common grammar errors.
- Misspellings.
- OCR-like substitutions (`rn`/`m`, broken hyphenation, digit-letter confusions).
- Mojibake fixture.
- Zero-width and non-breaking-space fixture.
- Mixed typography fixture.
- Proper-noun-heavy text to ensure spelling false positives are visible but bounded.

## Acceptance criteria

- Mechanical errors are separated by category and implementation.
- At least one grammar engine, multiple spelling channels, and Unicode/encoding diagnostics are available.
- Checker disagreement is retained.
- No repair tool silently mutates the canonical analyzed text.
- Book-length inputs are bounded and failure-isolated.

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
