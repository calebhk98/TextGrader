# 14. Expand dialogue analysis with conversational dynamics and speaker interaction

## Goal

Build on TextGrader's existing dialogue metrics to analyze interaction dynamics: speaker accommodation, lexical entrainment, response matching, turn-taking, question/answer behavior, dialogue acts, politeness, emotional coupling, and per-character stylistic separability.

## Suggested TextGrader integration

- Config switch: `conversation_suite`
- Primary module: `textgrader/metrics/conversation_suite.py`
- Family: `dialogue`
- Metric ID prefix: `dialogue.conversation_`
- Cost: `moderate`/`model` depending on dialogue-act/politeness models.
- Reuse existing dialogue extraction and speaker attribution. Never re-parse raw quotes differently inside this suite.

## Metrics to add

- Identified-speaker coverage and confidence cross-check.
- Speaker turn-count distribution and dominance/Gini.
- Turn-length matching between adjacent speakers.
- Sentence-length accommodation.
- Function-word accommodation/coordination.
- Lexical entrainment: shared word/lemma adoption after another speaker uses it.
- Rare-word entrainment.
- Contraction-rate convergence.
- Punctuation/question/exclamation convergence.
- POS-pattern convergence.
- Speaker style distance and separability beyond existing function-word measures.
- Response semantic relevance.
- Question-response rate.
- Unanswered-question rate.
- Backchannel/short-response rate.
- Dialogue-act distribution and transition matrix/entropy.
- Politeness-strategy distribution.
- Sentiment/emotion response coupling.
- Speaker sentiment differentiation.
- Reciprocity of turn exchange.
- Conversation graph statistics.
- Alternation/run lengths by speaker.
- Scene-level changes in speaker style/interaction.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **ConvoKit** | Linguistic coordination, conversational feature extraction, politeness/interaction tools and conversation structures. |
| **BookNLP** | Character resolution and quote-speaker attribution for fiction. |
| **spaCy** | Existing POS/dependency/token features for speaker turns. |
| **sentence-transformers** | Response relevance if already available; reuse shared models/caches. |
| **NetworkX / igraph** | Speaker interaction graphs. |
| **scikit-learn** | Speaker-style classifiers/separability and simple dialogue-act models if training data is available. |
| **NRC/VADER/other sentiment resources** | Optional response-emotion coupling; share implementation with Task 15. |

## Why

Dialogue quality/structure is not captured by quote percentage or contraction rate alone. Characters can have distinct voices, converge on one another, exchange questions, dominate interactions, or produce mechanically alternating template dialogue. Conversational-analysis methods provide additional sensors while reusing the project's existing dialogue channel.

## Implementation details

1. Use existing canonical turns/speaker attribution. If attribution coverage is too low, speaker-specific findings must report insufficient data, mirroring current safeguards.
2. Represent each speaker turn as an ordered event with speaker ID, text, sentence count, word count, and optional feature vector.
3. Accommodation must be directional where possible: does speaker B increase use of feature X after speaker A uses it?
4. Keep raw style differences and convergence measures separate.
5. Conversation graph nodes are speakers; weighted directed edges represent adjacent turns or replies.
6. Dialogue-act models should be optional and versioned. Emit label distribution and transition entropy, not merely the most common act.
7. ConvoKit outputs should be mapped to stable TextGrader metric IDs rather than dumping opaque feature names without documentation.
8. Analyze full-document dialogue and, where sample sizes allow, scene/chapter windows.

## Corpus/profile requirements

Store scalar distributions plus attribution settings/model versions. Corpus comparisons of speaker-level statistics require comparable minimum speaker/turn coverage.

## Tests and validation

- Two-speaker alternating dialogue.
- One-speaker-dominant scene.
- Distinct speaker vocabularies vs deliberately identical voices.
- Sparse attribution should suppress speaker-specific claims.
- Question-answer fixture.
- Dialogue-free text should return insufficient/unavailable findings cleanly.

## Acceptance criteria

- Coordination/entrainment, turn dynamics, question-response, speaker separability and interaction-graph metrics are present.
- Existing dialogue extraction/attribution is reused.
- Low attribution coverage cannot generate confident speaker metrics.
- Heavy dialogue models remain optional.

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
