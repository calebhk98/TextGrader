# TextGrader

Measures English prose against a reference corpus and reports what it finds as
evidence, not as instructions to rewrite. Built for authoring agents that draft,
measure and revise in a loop, and usable by hand.

There is no quality score and a run never fails. Readability and surface-style
measures are correlated, unusual prose is not bad prose, and a value far from a
reference corpus is often a deliberate choice. What you get is a structured
list of measurements, each saying how far from the reference it sits, how sure
that is, and how much text it was measured from.

## Install

```console
git clone https://github.com/calebhk98/TextGrader && cd TextGrader
pip install -r requirements.txt            # optional: nothing here is required
python -m spacy download en_core_web_sm     # for the syntax metrics
pip install -r requirements-embeddings.txt  # for the semantic metrics (pulls torch)
```

Python 3.9+. The core and 24 of the 57 metrics run on the standard library
alone. Anything missing degrades to a visible result naming the package and the
install command, never a crash.

## Use

```console
python3 grade.py draft.md                       # human-readable
python3 grade.py draft.md --json                # machine-readable
python3 grade.py chapter_07.md --comparison-unit chapter
python3 grade.py draft.md --enable mtld --enable-family sentence_rhythm
python3 grade.py --list-metrics                 # everything available, and its cost
```

Nine metrics are on by default: the fast, dependency-free, generic ones.
Everything else is opt-in.

## Reading a result

`summary.top_findings` is the severity-ordered short list to read first. Every
result looks like this:

```json
{
  "metric_id": "rhythm.in_run_share_short",
  "family": "sentence_rhythm",
  "value": 55.0, "unit": "%",
  "direction": "high", "severity": 4.1, "confidence": "high",
  "sample_size": 221, "comparison_unit": "chapter", "channel": "narration",
  "action": "review",
  "corpus": {"corpus_median": 17.0, "corpus_count": 40, "percentile": 98.8},
  "distribution": {"median": 3, "p10": 1, "p90": 9, "entropy": 1.9,
                   "lag1_autocorrelation": 0.31, "bimodality": 0.61},
  "evidence": [{"sentence_index": 412, "run": 7, "text": "He stopped. ..."}]
}
```

Branch on `action`:

| `action` | meaning |
| --- | --- |
| `review` | far enough from the reference to be worth a look |
| `informational` | measured, and unremarkable or not comparable |
| `insufficient_data` | not enough text, or not enough corpus, to say anything |
| `rule_violation` | a project rule you configured was broken |
| `unavailable` | the measurement did not happen; `warning` says why |
| `error` | the metric crashed; the run continued and the count is visible |

`severity` is a robust distance from the corpus centre, so findings in different
units rank against each other. `distribution` carries the shape of the
underlying sample, because a mean is rarely the interesting fact: `14 14 14 14`
and `4 7 13 29` share one and are not the same prose.

A comparison is withheld, with a reason, when the corpus is too small, the
document is too small, or a count-based metric would be compared across
different units (a chapter against a shelf of novels).

`summary.maturity.percentile` is the aggregate: the median of the per-metric
corpus percentiles, after orienting each by its `polarity` so higher always
means more developed. 18 of the core metrics are graded; `front`, `and2`,
`andrate`, `negative` and the size counts are `neutral` and excluded, because
more or fewer fronted clauses is a style choice rather than a competence.
`metric_count` says how many went in.

`polarity` is not `direction`. `direction` is observational - is this document
above or below the corpus centre. `polarity` is semantic - does above mean
more. Averaging percentiles without orienting them first produces a number
that means nothing.

It exists because a findings list cannot show aggregate drift: every metric
can stay comfortably inside its band while the whole moves, and a revision
that does that produces no finding to see. It is also the scalar a
draft-measure-revise loop needs in order to tell whether the last iteration
helped. What it claims is that the document sits at this position among the
corpus texts on the measures the corpus defines; it does not claim a higher
number is a better book.

Set `analysis.benchmark` to the name of one text in the corpus profile and
each run also reports where it falls behind that text, ranked by gap in
corpus standard deviations so that findings in different units compare.

`summary.scorecard` is the census: how many measures were taken, how many sit
inside their reference, and how many were `not_taken`. That last one is
reported separately and folded into neither side, because it is the number
that moves when instrumentation breaks: a metric that errored is not a pass
and an unavailable one is not a failure, so a tool that counts only failures
can read 100% while three of its measures are silently dead. `failing` is
grouped by family, since eleven findings in one family is a habit and eleven
across eleven families is noise. It is a count, not a quality score, and a run
still always exits zero.

## Configure

`config.json`, resolved relative to itself. `--config other.json` reaches the
project reports too.

```json
{
  "corpus_profile": "my-corpus.json",
  "analysis": {"comparison_unit": "chapter"},
  "text_processing": {"strip_gutenberg": true, "segmenter": "auto"},
  "nlp": {"model": "en_core_web_sm", "disable": ["ner"]},
  "metrics": {"mtld": {"enabled": true}},
  "project_rules": {
    "em_dash": "forbid",
    "banned_phrases": [{"id": "filter", "pattern": "\\bI noticed\\b"}]
  }
}
```

`text_processing` is recorded in every corpus profile and compared against the
manuscript's, so grading against a differently-prepared corpus warns you.
Project rules are the only thing reported as a violation, and are off until
configured; patterns are compiled and bounded at load, so a bad one is a
visible result rather than an exception or a hang.

Keys are checked at load. An unrecognised one is reported as a
`config.<key>` result with `action: "unavailable"`, naming the key and
suggesting the intended spelling when it is close, because a key nothing reads
is otherwise indistinguishable from a measure that ran and found nothing. A run
still exits zero. Keys beginning with `_` are treated as comments, since JSON
has none. `project_rules.hard_line_breaks` and `project_rules.chapter_length`
are recognised but unimplemented, and say so if you set them.

### Chapter bands

A single whole-book reading-level target cannot express a book whose level is
supposed to move, and it hides the movement: a manuscript reading
Flesch-Kincaid 6.8 against a target of 7.0 looks fine while its last fourteen
chapters average 8.85 against a floor of 7.2. Set `chapter_bands` and
`chapter_report.py` judges each band:

```json
"chapter_bands": {
  "metric": "fk",
  "bands": [
    {"chapters": "1-10",  "floor": 5.5},
    {"chapters": "11-22", "floor": 6.5},
    {"chapters": "23-36", "floor": 7.2, "ceiling": 9.5, "exempt": {"28": "why"}}
  ]
}
```

The **band average** is judged, not each chapter. Chapters outside the band's
range are listed as a pointer to where to look and do not themselves fail the
band - judging both made bands that no revision could satisfy. A band may set
a ceiling as well as a floor, so a chapter running four grades hot is caught
too. An exempt chapter leaves the average and is reported, so an exemption is
a decision on the record.

## Corpus

Grading needs a profile, not the books. Acquire, profile, then check the
profile holds up:

```console
python3 build_corpus.py --config config.json            # download public-domain text
python3 -m textgrader.corpus corpus/books -o corpus.json --comparison-unit book
python3 validate_corpus.py corpus/books                 # leave-one-out calibration
```

`validate_corpus.py` drops each text, rebuilds the reference from the rest,
grades the dropped text against it, and reports how often each metric flags a
book that belongs to its own population. That is the check that catches a
badly-calibrated metric, and it found one: rates floored at zero were being
judged by a symmetric rule, so the whole upper tail read as outlying.

`--parse-metrics` and `--model-metrics` also profile the spaCy and embedding
metrics. They are opt-in because of what they cost per book.

The bundled `data/prose_reference.json` is 50 public-domain novels from
Gutenberg and Standard Ebooks, built with the `text_processing` settings in
this repository's `config.json` and recorded in the profile so a mismatch with
your manuscript is reported rather than assumed. `data/absolutes_reference.json`
is the same 50 books. Rebuild both from a shelf that matches what you write:
percentiles are only as relevant as the corpus they come from, and this one is
general English-language fiction weighted to the 19th and early 20th century.

Ten of the twelve children's classics the Lexile coefficients in
`core_metrics.lexile` were fitted against are in it, so the docstring's
calibration claim can be checked against the shipped corpus rather than taken
on trust. The two missing are *The Railway Children* and *A Little Princess*.

"Grading needs a profile, not the books" holds for every metric and for all but
three of the project reports. `tics`, `number_report` and `quotable` count
user-configured regex patterns, which a profile built before those patterns
existed cannot anticipate, so they read `*.txt` from `corpus_dirs` and say so
when it is empty rather than comparing against nothing.

A corpus records the `analysis.lexile_frequency_source` it was built with, and
carries a Lexile distribution only when one was set. `none`, `bundled` and
`wordfreq` are three different scales rather than three readings of one, so a
run whose source differs from its corpus keeps the Lexile value and has the
percentile withheld, with a result saying why.

`data/word_frequency.json` is deliberately not rebuilt alongside the corpus.
The Lexile coefficients in `core_metrics.lexile` were least-squares fitted
against that exact table, so replacing it silently invalidates the calibration.
Refit the coefficients first if you replace it.

## Cost

Run `python3 benchmark.py` for figures from your machine and your text. The
shape of it on a full-length novel:

- the dependency-free pipeline and metrics: **seconds**, dominated by sentence
  segmentation. `text_processing.segmenter: "builtin"` trades accuracy for a
  large speedup.
- the 13 `parse` metrics: **tens of seconds**, sharing one spaCy parse, so
  enabling all of them costs about the same as enabling one.
- the 4 `model` metrics: **tens of seconds**, sharing one embedding pass.

`grade.py` warns on any metric that took over a second on the document in front
of it, so a slow run always says what was slow.

## The metrics

### sentence rhythm

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `length_quantiles` | fast | - | yes | Sentence and paragraph length quantiles. |
| `sentence_length_autocorrelation` | fast | - | no | Lag-1..n autocorrelation of sentence length: catches metronomic prose. |
| `sentence_length_deltas` | fast | - | no | Distribution of the change in length between adjacent sentences. |
| `sentence_length_entropy` | fast | - | no | Entropy of the sentence-length distribution, normalized for range. |
| `sentence_run_lengths` | fast | - | no | Run-length distribution of short/medium/long sentence bands. |
| `sentence_segmentation` | fast | pysbd | no | Which segmenter was used, and how much it disagrees with the built-in one. |
| `timeseries_suite` | moderate | statsmodels, ruptures, pycatch22, PyWavelets, tsfresh | no | Trend, autocorrelation, spectral, catch24 and wavelet features over named linguistic sequences; pick the sequences and the feature groups separately in config. |

### paragraph rhythm

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `paragraph_rhythm` | fast | - | no | Paragraph length dispersion, quantiles and autocorrelation. |
| `single_sentence_paragraph_runs` | fast | - | no | Runs of consecutive one-sentence paragraphs. |

### syntax

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `clause_structure` | parse | spacy | no | Mean token depth in the dependency tree (not per-sentence tree depth). |
| `clause_types` | parse | spacy | no | Relative, adverbial and complement clause rates. |
| `coordination_ratio` | parse | spacy | no | Coordination against subordination. |
| `dependency_distance` | parse | spacy | no | Mean and SD of dependency distance, a real syntactic-complexity measure. |
| `finite_clauses` | parse | spacy | no | Finite clauses per sentence. |
| `opening_patterns` | parse | spacy | no | POS/dependency sentence-opening shapes, with no hard-coded vocabulary. |
| `parse_depth` | parse | spacy | no | Per-sentence maximum dependency-tree depth. |
| `parser_consensus` | parse | spacy, pysbd, nltk, syntok, stanza, benepar | no | Where independent sentence splitters, tokenizers and parsers disagree on the same sampled text: long-sentence share and maximum length per splitter (the tail, which is where a quote-collapse shows), boundary F1, and, once a second parser is enabled, POS, dependency and noun-phrase agreement. |
| `passive_voice` | parse | spacy | no | Share of clauses in the passive voice. |
| `syntax_complexity_suite` | parse | spacy, benepar | no | L2SCA-style T-unit and clause ratios pooled over the text (a labelled spaCy approximation, not L2SCA itself), phrasal elaboration, dependency-tree topology, corpus-trained syntactic surprisal (needs a profile built with `--parse-metrics`), and opt-in benepar constituency measures over a bounded sample. |
| `pos_distribution` | parse | spacy | no | Share of each open-class part of speech. |
| `tense_consistency` | parse | spacy | no | Rate of sentence-to-sentence tense changes in narration. |

### lexical

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `hdd` | moderate | lexicalrichness | no | HD-D, a hypergeometric length-resistant diversity measure. |
| `mattr` | moderate | - | yes | Moving-average type-token ratio, length-resistant lexical diversity. |
| `mtld` | moderate | lexicalrichness | no | Measure of Textual Lexical Diversity. |
| `nominalizations` | parse | spacy | no | Suffix-matched nominalization density; a proxy, not a parse of derivation. |
| `mechanical_quality_suite` | moderate | pyspellchecker, symspellpy, ftfy, confusable-homoglyphs | no | Typography, encoding, homoglyph, hyphenation and spelling checks with two independent spell checkers, invented names excluded as recurring vocabulary, and every dialect-sensitive rate split between narration and dialogue. |
| `randomness_suite` | moderate | wordfreq, zstandard, brotli, lz4, snappy, pyppmd, kenlm | no | Language-likeness, multi-codec compression and entropy channels, including a KenLM model trained on the text itself. |
| `word_rarity` | moderate | wordfreq | no | Zipf word-rarity distribution from general-language frequencies. |

### repetition

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `lemma_repetition` | parse | spacy | no | Repetition measured over lemmas, so walk/walked/walking cannot hide. |
| `local_repetition` | moderate | - | yes | Content-word reuse inside sliding windows. |
| `repeated_ngrams` | moderate | - | yes | Repeated word sequences, scored by excess occurrences rather than by type count. |
| `repetition_distance` | moderate | - | no | How soon a content word is reused, in tokens. |
| `sentence_openings` | fast | - | yes | How often a sentence starts with the same few words as another. |

### semantic repetition

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `adjacent_sentence_similarity` | model | sentence_transformers | no | Embedding similarity between neighbouring sentences. |
| `duplicate_sentence_clusters` | model | sentence_transformers | no | Clusters of sentences that restate one another. |
| `local_similarity_window` | model | sentence_transformers | no | Similarity to the previous three and five sentences. |
| `paragraph_similarity` | model | sentence_transformers | no | Paragraph-to-paragraph semantic similarity. |

### dialogue

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `character_voice` | fast | - | no | Pairwise distance between transcript speakers' function-word profiles. |
| `dialogue_attribution` | fast | - | no | Speech tag against action beat against untagged turn. |
| `dialogue_channels` | moderate | - | no | Every core shape measured separately for dialogue and for narration. |
| `dialogue_contractions` | fast | - | yes | Contraction rate inside spoken text, excluding possessives. |
| `dialogue_runs` | fast | - | no | Consecutive spoken turns with no narration between them. |
| `dialogue_tags` | fast | - | yes | Speech-tag density and how elaborate the tags are. |
| `dialogue_turn_lengths` | moderate | - | no | Distribution of spoken turn lengths in words and sentences. |
| `speaker_function_words` | fast | - | no | Per-speaker function-word profile and pairwise distance. |
| `speaker_style` | fast | - | no | Questions, exclamations and contractions per identified speaker. |

### discourse

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `causal_connectives` | fast | - | no | Causal and explanatory connective rates. |
| `coherence_suite` | parse | spacy, networkx, nltk, sentence-transformers, fastcoref, isanlp-rst | no | Lexical, WordNet and semantic adjacency, surface *and* coreference-resolved entity grids kept side by side, a corpus-referenced entity-grid transition table, and a sampled real RST parse (tree depth, segment length, nuclearity balance, relation-family entropy). |
| `hedges_boosters` | fast | - | no | Hedge, booster and modal rates. |
| `logic_suite` | parse | spacy, transformers, fastcoref, nltk, python-dateutil | no | Surface contradiction candidates and proposition triples, cross-checked against an NLI model and coreference when installed, plus opt-in semantic role labelling, REBEL relation extraction, an argument-relation classifier and WordNet/PropBank/VerbNet/FrameNet lexical checks. The REBEL and argument-mining models are CC BY-NC-SA 4.0 (non-commercial). |
| `rhetorical_constructions` | moderate | - | no | Repeated rhetorical templates, discovered rather than listed. |
| `sentence_initial_connectives` | fast | - | no | Rate of sentences opening on However, Indeed, Moreover and the like. |

### pov

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `entity_pronoun_ratio` | parse | spacy | no | Named entities against pronouns: over-naming or pronoun saturation. |
| `narration_pov` | fast | - | no | Person-marking rates measured in narration only, free of dialogue. |
| `pov_block_confidence` | fast | - | no | Per-block POV call with an explicit evidence count; no evidence means no call. |
| `pov_pronouns` | fast | - | yes | Person-marking pronoun rates and block-by-block POV evidence. |

### punctuation

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `punctuation` | fast | - | yes | Rate of each punctuation mark per 1,000 words. |
| `punctuation_entropy` | moderate | - | no | Entropy of the punctuation mix, a regularity signal. |
| `punctuation_patterns` | moderate | - | no | Repeated punctuation shapes across consecutive sentences and paragraphs. |
| `punctuation_profile` | moderate | - | no | Full per-mark distribution, per sentence and per 1,000 words. |

### book drift

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `change_points` | moderate | ruptures | no | Where the style changes abruptly. |
| `chapter_zscores` | moderate | - | no | Which section looks unlike the rest of this book, and on which measures. |
| `rolling_drift` | moderate | - | no | Gradual style drift from the opening to the close. |

### authorial

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `function_words` | fast | - | no | Burrows's Delta against the corpus function-word profiles. |
| `stylometry_suite` | moderate | lexicalrichness, sentence-transformers | no | Authorship channels kept separate on purpose: n-gram profiles, lexical richness, section stability, impostors, and nearest-reference distances over cached per-book embeddings. |

### distribution shape

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `anomaly_suite` | moderate | scikit-learn, pyod, hdbscan | no | Fifteen multivariate anomaly detectors over the document's core-metric vector against the corpus, each reported on its own, plus a consensus count and a cross-detector disagreement score. |
| `distribution_distance_suite` | moderate | scipy | no | A two-sample distance battery (Wasserstein, energy, KS, Cramer-von Mises, Anderson-Darling, Jensen-Shannon, KL, Hellinger, MMD, tail mismatch and more) against the corpus's pooled sentence, paragraph, word and turn distributions, each distance kept apart from any p-value. |
| `distribution_shape` | fast | - | yes | The text's sentence, paragraph, word and turn distributions held against the corpus's pooled ones. |

Every switch in the `*_suite` rows above is off by default, and each one
takes a `features` map (or, for `timeseries_suite`, separate `sequences` and
`feature_groups` lists) so individual measurement groups can be turned on and
off without the others.  `config.json` carries a `_..._requires` note beside
each group saying what it needs installed and what it does without it: some
degrade to a labelled weaker backend, some report `unavailable`.

Their `needs` column lists what a suite *can* use, not what it declares in the
registry.  The neural channels (an NLI model, coreference, a causal language
model, sentence embeddings) are off by default and deliberately kept out of
each suite's `requires`, because `needs_model` is what decides whether the
corpus builder profiles a metric at all, and widening it to gate one expensive
channel would drop dozens of cheap ones out of corpus profiling.  Turning a
neural channel on is always an explicit act in `config.json`.

Two channels need something pip cannot install: `randomness_suite`'s KenLM
channel needs the `lmplz` binary built from source (the wheel only queries a
model), and any WordNet channel needs the nltk corpus downloaded, not just the
package.  Both say so when they cannot run.

## What these measurements do not claim

The honesty is part of the output, not a footnote. Each of these is stated in
the metric's own `name`, docstring and `warning` as well:

- `clause_structure` is **mean token depth**, not tree depth. `parse_depth` is
  the real per-sentence maximum.
- `nominalizations` is **suffix matching** on parsed nouns, not derivational
  morphology.
- the core `subord` / `relcl` / `simple` metrics are **lexical proxies** that
  match cue words with a regex. `clause_types` is the parse-based replacement.
- causal `since` is not separated from temporal `since`, and the hedge
  `seemed` is not separated from a literal one. These are surface counts.
- bare `'s` is excluded from contraction counts, because `"she's happy"` and
  `"the dog's bowl"` are indistinguishable without a parse. The excluded share
  is reported separately.
- prose speaker attribution is a **biased sample**: only tagged turns can be
  attributed. Compare speakers with each other; never read a per-speaker zero
  as a fact about a character.
- without `sentence-transformers`, the `semantic_*` metrics fall back to
  lexical overlap, which scores a real paraphrase near zero where embeddings
  score it near 0.8. A low value from that backend is not evidence of no
  repetition, and every finding says which backend produced it.
- single-quote dialogue is not parsed: an apostrophe and a closing single quote
  are the same character. The parser reports the limitation rather than
  guessing.

## Project reports

`textgrader/reports/` holds thirteen reports that check a manuscript against an
author's own conventions: cast lists, tic patterns, register thresholds,
formatting policy, citation conventions. They are **off by default** and do
nothing without `project_measures` in your configuration.
`examples/project_measures.example.json` is a worked example, not a default.

## Develop

```console
python3 -m pytest
```

Layout: root holds the scripts you invoke and the configuration; `textgrader/`
is the importable analysis package, named to match the project as Python
packages conventionally are; `corpus_builder/` acquires text and shares no code
with the analysis.

A metric is one module in `textgrader/metrics/` plus one row in the registry in
`textgrader/metrics/__init__.py`. `textgrader/metrics/common.py` documents the
contract and `rhythm_autocorrelation.py` is the worked example. Read
`textgrader/document.py` first: it builds the shared pipeline every metric
measures, so no metric decides for itself what a word, a sentence, a quotation
or a clean manuscript is.

Design rationale lives in module docstrings rather than here, so it stays next
to the code it explains: `document.py` for the pipeline, `stats.py` for the
estimator ladder and the distribution summaries, `results.py` for the result
schema, `corpus.py` for what a profile records and why.

## License

See `LICENSE`.
