# TextGrader

TextGrader measures English prose and reports **evidence**, not instructions to
rewrite. It is built for authoring agents that draft, measure and revise in a
loop, so every result carries what such a reader needs to decide whether to act:
which family of style it belongs to, which way the value is unusual, how
unusual, how much text it was measured from, and whether the corpus can support
the claim at all.

Nothing here produces a quality score. Readability and surface-style measures
are correlated, unusual prose is not bad prose, and a measurement that lands far
from a reference corpus is often a deliberate choice. A run never "fails".

Results come in three kinds, which are kept apart on purpose:

* **corpus results** locate prose inside a reference distribution;
* **project rules** report the author's own configured choices being broken;
* **diagnostics** point at material that deserves a human look.

## One document, measured once

Every number in a report describes the same text. The pipeline runs once, in
`textgrader/document.py`, and every metric reads the result:

```
raw text
  -> canonical cleanup       Gutenberg boilerplate, Markdown headings,
  |                          transcript lines, quote normalization
  -> tokens / sentences / paragraphs      one configured segmenter, not a
  |                                       regular expression per metric
  -> quotations / dialogue / narration    dialogue and narration are themselves
  |                                       DocumentAnalysis objects
  -> spaCy Doc (optional, chunked, lazy, shared)
  -> every metric
```

This is a correctness feature rather than a tidiness one. Core analysis used to
strip Gutenberg boilerplate and Markdown headings while the optional metrics
received the raw file, so two numbers in one report could describe two different
documents; sentence segmentation was a regular expression for most metrics and
pySBD for one; dialogue was re-parsed, differently, in four places. A metric now
never decides for itself what a word, a sentence, a quotation or a clean
manuscript is.

`analysis.dialogue` and `analysis.narration` are ordinary documents, so any
metric can be pointed at one channel. `analysis.sections` splits on Markdown
headings and `analysis.windows(n)` on word count, which is what the book-level
drift metrics compare.

## Where things live

```
grade.py                 the entry point: one manuscript in, structured report out
build_corpus.py          acquire public-domain text from the configured providers
build_manuscript.py      assemble numbered chapter files into one manuscript
benchmark.py             time every step on your own machine and text
validate_corpus.py       leave-one-out check: does the corpus comparison hold up

config.json              the only file that describes YOUR project
data/                    reference data shipped with TextGrader
examples/                worked example configurations, never defaults

textgrader/              the library; everything importable lives here
  document.py            DocumentAnalysis: the shared pipeline
  core_metrics.py        the core measurements every run makes
  stats.py               distribution shape and robust corpus comparison
  text.py                words, paragraphs, quotations, transcripts
  results.py             the result and report schema
  rules.py               compiling and bounding user-supplied patterns
  project.py             loading project configuration
  paths.py               where the package keeps its own files
  corpus.py              building and loading corpus profiles
  optional.py            every third-party import, in one place
  metrics/               one module per registered metric, plus the registry
  reports/               the project reports, runnable as python -m
  corpus_builder/        acquisition providers, used only by build_corpus.py
```

There used to be a second top-level `measures/` directory beside `textgrader/`,
and the split was in the wrong place. `measures/prose_grade.py` held the core
calculator, which is library code, and two callers loaded it by file path with
`importlib.util.spec_from_file_location`, each getting its own copy of the
module and its own caches. The other twelve files were project reports, a
genuinely different kind of thing, but each opened with a `sys.path.insert` so
it could import the library sitting next to it.

So the core calculator moved into the library as `textgrader/core_metrics.py`,
and the reports became `textgrader/reports/`, a real package that imports
normally and runs as `python -m textgrader.reports.register`. The acquisition
code moved the same way, from a second top-level `corpus_builder/` to
`textgrader/corpus_builder/`. There is one package now, with a subpackage for
each kind of thing in it.

The rule for what stays in root is: things you invoke but never import, plus
tool configuration. `textgrader.corpus` is `python -m` rather than a root script
because it has a library API as well as a command line.

The two kinds are still distinct, and the distinction is the reason `reports/`
exists at all:

| | `textgrader/metrics/` | `textgrader/reports/` |
| --- | --- | --- |
| answers | what is this prose like | does this manuscript follow my rules |
| returns | structured findings | a table for a person, and an exit code |
| runs | in process, on the shared pipeline | as a subprocess |
| needs config | no | yes: it does nothing without `project_measures` |
| on by default | the fast generic ones | none |


## Requirements

Python 3.9 or newer. **No third-party package is required.** Core analysis, the
corpus builder and 24 of the 57 metrics run on the standard library alone.

```console
pip install -r requirements.txt
python -m spacy download en_core_web_sm       # for the parse-based metrics
pip install -r requirements-embeddings.txt    # semantic similarity (pulls torch)
```

Every optional library is reached through `textgrader/optional.py` and degrades
to a visible result with the reason and the install command. A metric that
loses its library does not take the run with it. To prove that on a machine
that has everything installed:

```console
TEXTGRADER_DISABLE_OPTIONAL=all python3 grade.py draft.md
TEXTGRADER_DISABLE_OPTIONAL=spacy,wordfreq python3 grade.py draft.md
```

### What each dependency actually buys

Degrading gracefully is not the same as degrading harmlessly, so the three
cases are kept apart:

**The library is the measurement.** Without spaCy there is no parse, so the 13
`parse` metrics report `unavailable` and say which package and model to
install. Nothing is guessed.

**The library is faster or better tested, and the fallback is exact.** SciPy
and NumPy are in this group. `tests/test_dependency_agreement.py` pins the
dependency-free skew, excess kurtosis, quantiles, Shannon entropy, Spearman
correlation with ties, and the Wasserstein distance against SciPy and NumPy to
twelve significant figures, so "no dependency" never quietly means
"approximate". The same tests are what stop a future edit drifting.

**The library changes what the metric can detect.** There is exactly one of
these, and it is worth stating plainly because the fallback looks like it
works. Without `sentence-transformers`, the four `semantic_*` metrics fall back
to TF-IDF lexical overlap. On this pair:

> The dog was extremely happy to see her.
> The canine was overjoyed at her arrival.

the embedding backend scores **0.79** and the lexical fallback scores **0.00**,
because the two sentences share no content words. Catching a restatement that
changes its vocabulary is the entire job of that metric family, so a low number
from the lexical backend is not evidence of no repetition. Every such finding
says so, in the `warning` and in each evidence row's `backend` field, and
`tests/test_dependency_agreement.py` asserts both halves of that gap.

If you only install one optional package, install spaCy. If you care about
AI-ish restatement specifically, install `sentence-transformers` too and accept
the cost in the table below.

## Analyze a manuscript

```console
python3 grade.py draft.md
python3 grade.py draft.md --json
python3 grade.py draft.md --json-out report.json
python3 grade.py chapter_07.md --comparison-unit chapter
python3 grade.py draft.md --enable mtld --enable-family sentence_rhythm
python3 grade.py --list-metrics
```

## Reading the output as an agent

`summary.top_findings` is the short, severity-ordered list to read first.
Every result looks like this:

```json
{
  "metric_id": "rhythm.in_run_share_short",
  "family": "sentence_rhythm",
  "name": "Sentences inside a run of 3+ short ones",
  "value": 55.0,
  "unit": "%",
  "direction": "high",
  "severity": 4.1,
  "confidence": "high",
  "sample_size": 221,
  "comparison_unit": "chapter",
  "channel": "narration",
  "action": "review",
  "corpus": {"corpus_median": 17.0, "corpus_count": 40, "percentile": 98.8,
             "method": "median/MAD", "outlier": true},
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
| `rule_violation` | an explicitly configured project rule was broken |
| `unavailable` | the measurement did not happen, and `warning` says why |
| `error` | the metric crashed; the run continues and the count is visible |

`severity` is the robust distance from the corpus centre, so findings across
different units are ranked against each other. `channel` says whether the
number describes the whole text, dialogue only, or narration only.

## Distributions, not averages

A mean is rarely the interesting fact about a text. These two share a mean
sentence length of fourteen words and are not the same prose:

```
A: 14, 14, 14, 14, 14, 14
B:  4,  7, 13, 29,  9, 22
```

So wherever a metric has a sample, it publishes the sample's shape in
`distribution`: count, mean, median, std, CV, min, max, p10/p25/p75/p90, IQR,
MAD, Shannon entropy, lag-1 autocorrelation, skew, a bimodality coefficient,
and a two-group split with its separation. The headline `value` is normally the
median, because the mean of a bimodal sample describes neither mode.

Named buckets (`stats.histogram`), band run lengths (`stats.run_lengths`) and
the first Wasserstein distance between two whole distributions
(`stats.wasserstein`, SciPy when present and an exact pure-Python fallback
otherwise) are available to any metric.

**On bimodality specifically**: the bimodality coefficient is a moment-based
screening statistic, not a test. It says the sample is shaped less like one hump
than a normal distribution would be. `two_group_split` then does a
one-dimensional two-means split and reports `separation`, the between-group
share of total variance, so "this text has two modes at 5 and 27 words" is a
claim with a number behind it. Neither can name *why* there are two groups.
Hartigan's dip test would be a stronger screen and is not implemented, because
it needs a dependency for one number.

## When a difference is a finding

`textgrader/stats.py` chooses an estimator by what the corpus can support, and
says which it used:

| method | when | why |
| --- | --- | --- |
| `median/MAD` | the corpus varies around its median | the default; threshold is a modified-z of 3.5 |
| `median/IQR` | MAD is zero but the tails differ | a discrete metric over a small corpus (`3, 3, 3, 3, 5, 9`) has MAD 0, which would otherwise make every non-3 infinitely distant |
| `empirical percentile` | MAD and IQR are both zero | the observed range is all that is left |
| `insufficient variation` | every observation is identical | **no outlier claim is made** |

Three separate gates can withhold a comparison, and each is reported instead of
being absorbed:

* **corpus too small.** Below 8 observations no outlier is called at all; below
  20 the claim is marked `confidence: "low"` because the tails are not
  determined. Five books do not get the authority of fifty.
* **manuscript too small.** Each metric declares the sample size it needs
  (`MIN_SAMPLE`). A passive rate from 3 sentences and one from 3,000 do not
  deserve equal standing, and the small one becomes `insufficient_data`.
  Separately, a document under 40 sentences or 500 words gets no corpus
  comparison at all, so a two-word file can no longer collect a pile of
  confident outliers.
* **unit mismatch.** See below.

## Books, chapters and scenes

A corpus profile records `comparison_unit`: `book`, `chapter`, `scene` or
`passage`. Scale-dependent metrics (word count, sentence count, paragraph count)
are refused across a mismatch, because a chapter "failing" document length
against a shelf of novels is a fact about how books are divided, not about the
chapter. Rates and ratios are never refused: that is what expressing a
measurement as a rate is for.

An unspecified unit resolves to `book` on both sides, so the common case of one
whole document against a corpus of whole documents still compares. Declaring
`analysis.comparison_unit: "chapter"` is what turns the protection on.

## Performance, for agents that run this in a loop

`python3 benchmark.py` times every step on your machine and your text. Figures
below are one core, 309,000 words, 17,654 sentences, pySBD segmentation.

Anything over a second is called out, both in the cost column of
`--list-metrics` and at run time.

| shared pipeline step | seconds | note |
| --- | ---: | --- |
| cleanup, words, paragraphs, quotations | 0.4 | linear |
| **sentence segmentation (pySBD)** | **5.3** | `text_processing.segmenter: "builtin"` does the same job in **0.25s** with cruder abbreviation handling |
| dialogue view | 1.3 | segmenting the spoken channel |
| narration view | 3.7 | segmenting the narrated channel |
| **spaCy parse** | **28.0** | `en_core_web_sm`, `ner` disabled; shared by all 13 parse metrics |
| **sentence embedding** | **34** sentences, **25** paragraphs | `all-MiniLM-L6-v2` on CPU; each is shared by the metrics wanting that unit |

Individual metrics, once the pipeline is warm: 43 of 57 are under half a second.
The ones that are not:

| metric | seconds |
| --- | ---: |
| `repeated_ngrams` | 3.8 |
| `chapter_zscores` | 2.3 (the other two `book_drift` metrics then cost about 0) |
| `duplicate_sentence_clusters` | 1.3 (lexical fallback; embeddings are slower) |
| `dialogue_turn_lengths` | 1.3 |
| `punctuation_profile`, `word_rarity`, `tense_consistency`, `local_repetition` | 0.8 to 0.9 |

Totals on the same 309,000-word text:

| run | seconds |
| --- | ---: |
| shipped defaults (9 metrics) | ~16 |
| everything except the `parse` and `model` metrics (40) | ~29 |
| plus the 13 `parse` metrics | ~61 |
| plus the 4 `model` metrics, with embeddings installed | ~130 |

Without `sentence-transformers` installed, the four semantic metrics use the
lexical fallback and cost about 3 seconds instead of 70. That is the trade: an
order of magnitude of runtime for the one measurement the fallback cannot make.

Roughly two thirds of a no-parse run is sentence segmentation, shared by every
metric rather than paid per metric. Practical guidance:

* Turning on any parse metric adds the 28-second parse once, not once per
  metric; all 13 together cost about 32 seconds including their own work.
* The same is true of the embedding: `adjacent_sentence_similarity` and
  `local_similarity_window` share one encoding of the sentences, and
  `paragraph_similarity` and `duplicate_sentence_clusters` share one of the
  paragraphs, so the four together cost roughly what the first two do.
* For a fast revision loop, grade one chapter at a time with
  `--comparison-unit chapter`, or set `text_processing.segmenter: "builtin"`,
  which takes a no-parse run of everything from ~29s to ~23s.
* `grade.py` attaches a warning to any metric that took over a second on the
  document in front of it, so a slow run always says which metric was slow.
* `nlp.max_words` refuses a parse above a size you choose, rather than stalling.


## Configuration

Relative paths resolve from the configuration file, not the working directory.
The distributed `config.json` contains no personal style rules and names no
book. `--config other.json` reaches the bundled reports too, through the
`TEXTGRADER_CONFIG` environment variable; it used to reach only the new metrics
while every legacy report silently kept using the repository's own file.

```json
{
  "corpus_profile": "my-corpus.json",
  "corpus_dirs": [],
  "analysis": {
    "comparison_unit": "chapter",
    "min_sentences_for_corpus": 40,
    "min_words_for_corpus": 500,
    "lexile_frequency_source": "none"
  },
  "text_processing": {
    "strip_gutenberg": true,
    "strip_markdown_headings": true,
    "strip_transcript": true,
    "drop_marker_paragraphs": true,
    "normalize_quotes": false,
    "segmenter": "auto",
    "language": "en",
    "transcript": null
  },
  "nlp": {"model": "en_core_web_sm", "disable": ["ner"], "max_words": null},
  "regex": {"engine": "auto", "timeout_seconds": 2.0},
  "metrics": {"mtld": {"enabled": true}},
  "project_rules": {
    "em_dash": "forbid",
    "quote_style": "straight",
    "banned_phrases": [
      {"id": "filter_word", "name": "Filter-word check", "pattern": "\\bI noticed\\b"}
    ]
  }
}
```

`corpus_dirs` is separate from `corpus_profile` and is only read by the two
project reports that scan raw corpus text rather than a profile. Grading itself
never needs the books.

`text_processing` is no longer decorative: it is applied by `grade.py`, recorded
in every corpus profile, and compared between the two. Grading a manuscript
against a corpus prepared differently produces a visible warning naming the
fields that differ.

An unknown key under `text_processing` or `nlp` is an error rather than a silent
no-op, so a typo cannot quietly disable a cleanup step.

### Project rules and user-supplied patterns

Rules are the only thing reported as a violation, and they are off until
configured. Every pattern is compiled once at load: a pattern that does not
compile becomes an `unavailable` result naming the rule instead of an exception
mid-run, and a pattern that compiles but backtracks catastrophically is cut off
by the `regex` package's matching timeout (`regex.timeout_seconds`). Without
that package installed there is no timeout available, so scanning is bounded by
input size instead and the result says so.

### External metric commands

`metric_commands` runs your own programs and reads one JSON result object, or a
list of them, from stdout. `{manuscript}` is substituted. Non-zero exits and
invalid JSON become visible internal errors. Commands never run unless
`allow_external_metric_commands` is explicitly `true`.

## The metrics

`python3 grade.py --list-metrics` prints this table with the costs measured on
your machine. Cost is `fast` (under half a second on a 300,000-word novel),
`moderate` (up to a few seconds) or `parse` (needs the shared spaCy parse).

The nine metrics on by default are the ones that are fast, dependency-free,
generic, and meaningful without any project-specific configuration. Everything
else is opt-in, including all thirteen of the bundled project reports in
`measures/`, which enforce an author's own house policy and mean nothing without
it.

### sentence rhythm

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `length_quantiles` | fast | - | yes | Sentence and paragraph length quantiles. |
| `sentence_length_autocorrelation` | fast | - | no | Lag-1..n autocorrelation of sentence length: catches metronomic prose. |
| `sentence_length_deltas` | fast | - | no | Distribution of the change in length between adjacent sentences. |
| `sentence_length_entropy` | fast | - | no | Entropy of the sentence-length distribution, normalized for range. |
| `sentence_run_lengths` | fast | - | no | Run-length distribution of short/medium/long sentence bands. |
| `sentence_segmentation` | fast | pysbd | no | Which segmenter was used, and how much it disagrees with the built-in one. |

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
| `passive_voice` | parse | spacy | no | Share of clauses in the passive voice. |
| `pos_distribution` | parse | spacy | no | Share of each open-class part of speech. |
| `tense_consistency` | parse | spacy | no | Rate of sentence-to-sentence tense changes in narration. |

### lexical

| switch | cost | needs | on by default | what it measures |
| --- | --- | --- | --- | --- |
| `hdd` | moderate | lexicalrichness | no | HD-D, a hypergeometric length-resistant diversity measure. |
| `mattr` | moderate | - | yes | Moving-average type-token ratio, length-resistant lexical diversity. |
| `mtld` | moderate | lexicalrichness | no | Measure of Textual Lexical Diversity. |
| `nominalizations` | parse | spacy | no | Suffix-matched nominalization density; a proxy, not a parse of derivation. |
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
| `hedges_boosters` | fast | - | no | Hedge, booster and modal rates. |
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

### What these metrics do not claim

Honesty about the limits is part of the output, not a footnote:

* `clause_structure` averages ancestor count over every token. That is **mean
  token depth**, not tree depth. It is named that way now, and
  `syntax.max_parse_depth` in the `parse_depth` module is the real per-sentence
  maximum depth.
* `nominalizations` is **suffix matching** on parsed nouns, not derivational
  morphology. The metric name, the docstring and every finding say so.
* The `subord` / `relcl` / `simple` core metrics are **lexical proxies** that
  match cue words with a regular expression. The `clause_types` module is the
  parse-based replacement.
* `discourse_causal` cannot separate causal `since` from temporal `since`, and
  `discourse_hedges` cannot separate the hedge `seemed` from a literal one.
  These are surface counts and are labelled as such.
* `dialogue_contractions` excludes bare `'s` entirely, because `"she's happy"`
  and `"the dog's bowl"` are indistinguishable without a parse. The excluded
  mass is reported separately as `style.dialogue_ambiguous_s_rate` rather than
  guessed at.
* Speaker attribution in prose is a **biased sample**: only tagged turns can be
  attributed, and a two-hander drops the tag once established. Compare speakers
  with each other, never read a per-speaker zero as a fact about a character.
* Without `sentence-transformers`, the four `semantic_*` metrics fall back to a
  TF-IDF lexical overlap, which is a different and weaker quantity. See
  "What each dependency actually buys" above: the fallback scores a real
  paraphrase at 0.00 where embeddings score it at 0.79.
* Single-quote dialogue is not parsed. An apostrophe and a closing single quote
  are the same character. The parser reports the limitation rather than
  guessing.

## Build a corpus profile

Corpus acquisition is a separate, dependency-free step with a small provider
interface: Gutenberg, Standard Ebooks, Internet Archive, Wikisource, Google
Books and the Library of Congress. A failed source does not stop the rest.

```console
python3 build_corpus.py --list-providers
python3 build_corpus.py --config config.json
python3 build_corpus.py --config config.json --healthcheck
```

Then profile the downloaded text:

```console
python3 -m textgrader.corpus books/ --name "Public-domain fiction" -o corpus.json
python3 -m textgrader.corpus chapters/ --comparison-unit chapter -o chapters.json
python3 -m textgrader.corpus books/ --config config.json -o corpus.json
python3 -m textgrader.corpus books/ --config config.json --parse-metrics -o full.json
```

Pass the same `--config` used for grading. The builder reproduces its
`text_processing` and its metric options, records both in the profile, and
`grade.py` refuses a comparison whose options do not match: a MATTR over a
100-word window and one over a 50-word window are different measurements.

`--parse-metrics` also profiles the thirteen spaCy metrics. The builder used to
skip everything needing a parse outright, so passive voice, syntax, tense, POS
and nominalizations could be measured on a manuscript but never compared with
anything. They are opt-in because they are slow, not because they are unwanted.
`--model-metrics` does the same for the four semantic metrics, which download
and run a sentence-embedding model over every book in the corpus.

A profile contains its schema, parser and metric-definition versions, corpus
name, comparison unit, build timestamp, text-processing fingerprint, per-source
IDs, filenames and SHA-256 hashes, per-source metric values, robust
distributions with their full shape, function-word feature profiles and a word
frequency table. Runtime grading needs only the profile, never the raw books.

For byte-reproducible output set `SOURCE_DATE_EPOCH`, or call `build_profile`
with a fixed `built_at`.

## Does the corpus comparison actually work?

A reference corpus is a claim about a population, and a tool that compares
against it should be checked against that claim rather than trusted. Every book
in a reference corpus belongs to the population it describes, so grading one
against the others ought to flag very little. Whatever it flags anyway is a
false positive, and counting those per metric says which measurements are
trustworthy.

```console
python3 validate_corpus.py corpus/books
python3 validate_corpus.py corpus/books --enable-family sentence_rhythm
python3 validate_corpus.py corpus/books --json-out validation.json
```

For each text it drops that text, rebuilds the reference from the rest, grades
the dropped text against it, and records every corpus outlier. Then it
aggregates into a per-metric rate.

Dropping a text's value from the pooled distribution is exactly what rebuilding
from the remaining texts gives, because a profile stores per-text values rather
than a fitted summary. That identity is what makes the check affordable -
otherwise it would re-measure every book once per hold-out - and
`tests/test_validate_corpus.py` asserts it against a genuine rebuild rather
than assuming it.

Read the output with one caveat in mind: a varied corpus **should** contain
outliers, and a metric that finds them is doing its job. What this separates is
"this metric flags a few unusual books" from "this metric flags half the
corpus", and only the second is a calibration problem. The companion column is
each metric's coefficient of variation across the corpus, because a metric that
never flags anything may be well behaved or may simply be constant, and those
are not the same thing.


## Project reports

These live in `textgrader/reports/`, predate the structured pipeline, and are
all **off by default**. Each is
one author's house policy made runnable, so each is now driven entirely by
`project_measures` in your configuration: cast lists, tic patterns and their
target rates, register thresholds and chapter exemptions, style target bands,
formatting policy, citation filename conventions and dialogue peer books. With
no configuration they print their descriptive table, apply no threshold, and
exit successfully, because "no configured policy" is not a violation.

`examples/project_measures.example.json` holds the complete set of values that
used to be hard-coded, preserved as a worked example of what the configuration
can express. It describes one specific manuscript and is not a default.

## Approximate Lexile and the bundled word frequency table

`word_frequency.json` was built from Gutenberg text that still contained the
licence boilerplate, and the Lexile coefficients were fitted against that
contaminated table rather than a published corpus. It is kept, and it is off.
`analysis.lexile_frequency_source` chooses:

* `none` (default) - approximate Lexile is unavailable rather than wrong;
* `bundled` - reproduces every number this repository ever printed, boilerplate
  and all;
* `wordfreq` - clean general-language frequencies, on a different scale from the
  one the coefficients were fitted to.

All three facts are reported with the value. The `word_rarity` metric is the
honest replacement: a full Zipf distribution rather than one fitted number.

## Tests

```console
python3 -m pytest
```

pytest, not `unittest discover`. The old documented command silently skipped
every module-level test function in `tests/test_build_corpus.py`, so the suite
could look green while not running. The suite parametrises over the whole metric
registry, so every registered metric is checked against the empty string, a
one-word document, a document with no dialogue and a realistic manuscript, and
against `TEXTGRADER_DISABLE_OPTIONAL=all`.

## Extending

A metric is one module under `textgrader/metrics/` plus one row in the registry
in `textgrader/metrics/__init__.py`:

```python
FAMILY = "sentence_rhythm"
COST = FAST
REQUIRES = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

def measure(analysis, config=None, profile=None):
    return [finding("rhythm.my_metric", "My metric", value, "%",
                    family=FAMILY, sample_size=n, min_sample=MIN_SAMPLE,
                    distribution=summarize(values), evidence=rows)]
```

`textgrader/metrics/common.py` documents the contract in full and
`rhythm_autocorrelation.py` is the worked example. The rules that matter: read
everything from `analysis`, never raise on degenerate input, publish the shape
of a sample rather than its mean, keep evidence short, and reach every optional
package through `textgrader/optional.py`.
