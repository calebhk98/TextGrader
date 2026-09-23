# 3. Add logic, consistency, entailment, and argument-structure analysis

## Goal

Add measurable proxies for whether claims relate logically to surrounding claims: entailment, contradiction, premise/conclusion structure, proposition consistency, connective semantics, semantic-role consistency, and argument-mining structure. This task does **not** claim to solve formal logic. It should expose many narrow signals instead of one false “logicality” oracle.

## Suggested TextGrader integration

- Config switch: `logic_suite`
- Primary module: `textgrader/metrics/logic_suite.py`
- Helper: `textgrader/propositions.py` for proposition/OpenIE normalization and caching.
- Family: `discourse`
- Metric ID prefix: `discourse.logic_`
- Cost: `model` for NLI; `parse`/external for OpenIE/argument tools.

## Metrics to add

- Adjacent-sentence entailment probability distribution.
- Adjacent-sentence contradiction probability distribution.
- Non-adjacent contradiction scan within configurable windows.
- Paragraph-level self-contradiction rate.
- Highest contradiction pair and evidence excerpts.
- “Therefore/thus/hence” relation: premise-to-conclusion entailment/support score.
- “However/but/nevertheless” relation: contrast/contradiction compatibility score.
- “Because/since” causal-link semantic score.
- Conditional-marker consistency around “if/unless/provided that”.
- Proposition duplication/paraphrase rate.
- Proposition introduction rate and unsupported-new-proposition proxy.
- OpenIE subject-relation-object consistency across mentions.
- Entity-property contradiction candidates: same entity/relation with incompatible objects.
- Temporal-order contradiction candidates where explicit dates/times exist.
- Negation-flip candidates.
- Claim/premise/conclusion density from argument mining.
- Premise-to-claim distance and argument-chain length.
- Support vs attack edge counts where the argument model provides them.
- Semantic-role pattern consistency and argument omission rates.
- Modal/hedge distribution around claims.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **SentenceTransformers CrossEncoder NLI models** | Pairwise entailment/neutral/contradiction probabilities. Prefer compact NLI classifiers; this is analysis, not generative text production. |
| **transformers** | Direct access to NLI/sequence-classification models when needed. |
| **Stanford OpenIE / CoreNLP** | Subject-relation-object proposition extraction, dependencies, coreference, constituency. |
| **AllenNLP legacy SRL/OpenIE models** | Independent semantic-role/proposition extraction where models remain usable. |
| **PropBank** | Predicate/argument frames and labels. |
| **VerbNet** | Verb-class semantic constraints. |
| **FrameNet** | Frame-semantic structure. |
| **WordNet** | Antonymy, hypernymy and lexical relations for contradiction/support heuristics. |
| **ConceptNet** | Optional commonsense relation checks; never treat absence as falsehood. |
| **glazing or equivalent lexical-resource wrapper** | Unified access to multiple lexical-semantic resources if maintained. |
| **Argument-mining models/toolkits** | Claim/premise/support/attack extraction. Keep model identity/version in results. |
| **dateparser / dateutil** | Optional normalization of explicit temporal expressions for contradiction checks. |

## Why

Coherence is not logical validity. Text can be smooth and topically consistent while making contradictory or unsupported claims. Pairwise NLI, proposition extraction, semantic roles, connective-specific tests, and argument mining expose different failure modes. None should be promoted to “truth checking”; they are structural/semantic consistency sensors.

## Implementation details

1. Create a bounded candidate-pair generator. Do not run all-pairs NLI on a 100,000-sentence corpus. Pair adjacent sentences, same-paragraph sentences, sentences sharing entities, and a sampled long-range window.
2. Cache NLI outputs by normalized sentence-pair hash in `analysis._shared`.
3. Keep entailment, neutral, and contradiction probabilities rather than only the argmax label.
4. For explicit connective tests, detect connectives using existing discourse modules and evaluate the sentence/clause pair surrounding the connective.
5. Extract OpenIE propositions and normalize subjects/objects through coreference where available. Store only bounded evidence examples in the report.
6. For contradiction candidates, require both a semantic signal and a shared entity/topic signal to reduce nonsense pairings.
7. Temporal contradiction checks should only fire when times/dates can be normalized with reasonable confidence.
8. ConceptNet/WordNet relations are supporting features, not ground truth. Missing graph edges must never count as contradiction.
9. Argument-mining labels must remain model-specific findings. Do not pretend different argument ontologies are identical; expose independent channels.
10. Add configurable caps: `max_pairs`, `window_sentences`, `max_evidence`, and model batch size.

## Corpus/profile requirements

Store distributions for contradiction rate, entailment rate, connective-specific relation scores, proposition repetition, argument density, argument-chain statistics, and role-pattern summaries. Record model identifier/version in profile settings so incompatible model outputs are not compared.

## Tests and validation

- Synthetic contradiction pair: “X is open” / “X is not open.”
- Synthetic entailment pair.
- Neutral unrelated pair.
- Connective tests for “therefore”, “however”, and “because”.
- Long document must obey pair cap.
- Missing NLI model should leave OpenIE/lexical metrics intact.
- NLI model/version mismatch must withhold corpus comparison.
- Explicitly test that the metric does not label unsupported commonsense lookup as factual contradiction.

## Acceptance criteria

- Separate entailment, contradiction, connective, proposition, and argument-structure outputs exist.
- Pairwise computation is bounded and deterministic.
- The suite is usable without pretending to perform formal proof or fact checking.
- Heavy-model failure is isolated.
- All scalar outputs can participate in corpus comparison when model/settings match.

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
