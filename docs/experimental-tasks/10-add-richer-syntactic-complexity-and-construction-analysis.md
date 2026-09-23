# 10. Add richer syntactic-complexity and construction analysis

## Goal

Expand syntax well beyond mean dependency depth. Add T-unit/clause measures, phrasal elaboration, constituency-tree features, dependency topology, construction diversity, production-rule statistics, and syntactic surprisal.

## Suggested TextGrader integration

- Config switch: `syntax_complexity_suite`
- Primary module: `textgrader/metrics/syntax_complexity_suite.py`
- Family: `syntax`
- Metric ID prefix: `syntax.complexity_`
- Cost: `parse`.

## Metrics to add

**T-unit / clause complexity**

- T-units per sentence/paragraph.
- Mean T-unit length.
- Clauses per T-unit.
- Dependent clauses per clause.
- Dependent clauses per T-unit.
- Coordinate phrases per clause/T-unit.
- Complex nominals per clause/T-unit.
- Verb phrases per T-unit.
- Mean clause length.
- Relative/adverbial/complement clause counts and ratios beyond existing basic metrics.
- Finite/nonfinite clause ratios.

**Phrasal elaboration**

- Mean noun-phrase length.
- Mean noun-phrase tree depth.
- Pre-/post-modifier counts.
- Prepositional-phrase attachment density.
- Appositive rate.
- Participial modifier rate.
- Nominal postmodifier diversity.

**Dependency topology**

- Mean/max tree depth.
- Branching-factor distribution.
- Tree imbalance.
- Dependency directionality ratio.
- Dependency-length mean, SD, quantiles and tail rate.
- Crossing/nonprojective dependency rate.
- Root POS/lemma distributions.
- Subtree-size distribution.
- Dependency-label entropy and transition entropy.

**Constituency / grammar**

- Constituency tree depth.
- Phrase-type distribution.
- Production-rule frequencies.
- Production-rule entropy.
- Repeated parse-template rate.
- Distinct subtree-pattern count/rate.
- Sentence syntactic-template diversity.

**Syntactic surprisal**

- POS n-gram cross-entropy.
- Dependency-label sequence cross-entropy.
- Constituency production-rule cross-entropy.
- Sentence-level syntactic surprisal distribution.

## Libraries / tools

| Library/tool | Add/use it for |
| --- | --- |
| **TAASSC** | Syntactic sophistication/complexity, phrasal elaboration, subordination, construction and lexicogrammatical measures. |
| **spaCy** | Existing dependency/morphology parse; reuse shared docs. |
| **benepar** | Constituency trees. |
| **SuPar** | Independent dependency/constituency parses. |
| **Stanford CoreNLP** | **Not used: Java, and TextGrader is Python only.** Constituency/dependency parsing, Tregex/Semgrex-compatible structures. |
| **Stanza** | Independent syntax/morphology/constituency. |
| **NLTK PCFG/tree utilities** | Production rules, tree traversals, optional grammar models. |
| **Tregex / Semgrex** | Construction pattern matching on constituency/dependency structures. |
| **Universal Dependencies resources** | Stable relation inventory and cross-parser interpretation. |
| **KenLM / NLTK LM** | Syntactic sequence language models. |

## Why

Syntactic complexity is multidimensional. Clause subordination, noun-phrase elaboration, dependency distance, branching, and construction diversity can move independently. A single parse-depth number is useful but necessarily incomplete. TextGrader's philosophy favors keeping those dimensions separate.

## Implementation details

1. Reuse the shared spaCy parse for all features it can support.
2. Only invoke constituency parsers if those submetrics are enabled/available. Cache parse trees.
3. Define T-unit and clause extraction rules explicitly and document limitations. Prefer established TAASSC-style definitions where possible.
4. Emit independent TextGrader calculations and TAASSC outputs separately when both exist; implementation disagreement is useful.
5. Production-rule/template metrics may be sample-size-sensitive; mark them accordingly.
6. Train syntactic n-gram models from the corpus, never from the test document alone.
7. Keep per-sentence distributions available through `shape(...)` where helpful.
8. Include top/bottom evidence sentences for extreme complexity measures.

## Corpus/profile requirements

Store distributions and any corpus-trained POS/dependency/production language models. Record parser/model/version and construction definition version.

## Tests and validation

- Simple coordinated sentences vs deeply subordinate sentences.
- Dense noun phrases vs clause-heavy prose.
- Same lexical content with altered syntax if possible.
- Parser missing/failure behavior.
- Sentence-length control test: verify syntax metrics are not merely aliases for words/sentence; correlation is allowed but residual variation should exist.

## Acceptance criteria

- T-unit, clause, phrasal, dependency-topology, constituency, and syntactic-surprisal groups are represented.
- Existing parse metrics are reused/cross-checked rather than overwritten.
- Independent library outputs remain visible.
- Corpus comparisons are parser/settings aware.

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
