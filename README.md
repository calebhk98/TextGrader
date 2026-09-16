# TextGrader

TextGrader is a lightweight, reproducible English-fiction prose analysis tool.
It reports measurements as **evidence**, not automatic instructions to rewrite
prose. It deliberately separates three kinds of result:

* **descriptive and corpus results** locate prose within a reference distribution;
* **project rules** enforce an author's explicitly configured house choices; and
* **diagnostics** identify material that deserves human inspection.

There is no “maturity”, “human”, or overall prose-quality score. Readability and
surface-style measures are correlated, and unusual prose is not necessarily bad.
Corpus comparisons are two-sided and use median/MAD robust distance.

## Requirements

Python 3.9 or newer. The core has no third-party dependencies.

## Analyze a manuscript

```console
python3 grade.py draft.md
python3 grade.py draft.md --json
python3 grade.py draft.md --json-out report.json
```

Every result has a stable `metric_id`, a `status_type`, sample information, and
optional corpus statistics, details, warnings, or errors. JSON accounting is
built from these objects rather than terminal words. A failed optional metric
process becomes an `internal_error`, remains in the result count, and makes the
command exit 2; it can never silently disappear.

Tiny and empty inputs, missing profiles, and unavailable measures are reported
as `unavailable` rather than treated as prose findings. Approximate Lexile is
currently disabled: the old frequency file contained Gutenberg boilerplate and
its calibration was not reproducible.

## Configuration

Relative paths are resolved from the configuration file, not the current
working directory. The distributed configuration has no personal style rules.
Rules only take effect when explicitly added:

```json
{
  "manuscript": "draft.md",
  "corpus_profile": "my-corpus.json",
  "project_rules": {
    "em_dash": "forbid",
    "banned_phrases": [
      {"id": "filter_word", "name": "Filter-word check", "pattern": "\\bI noticed\\b"}
    ]
  }
}
```

An empty `project_rules` object means no violations are possible. Rules are not
corpus observations and do not turn diagnostics into failures.

Advanced users can configure external structured metrics with
`metric_commands`. Each command must exit zero and print one result object or a
list of objects as JSON. `{manuscript}` in an argument is replaced with the
input path. Non-zero exits and invalid JSON become visible internal errors.

## Build a self-contained corpus profile

TextGrader does **not** download corpus texts. Supply locally acquired UTF-8
`.txt` files (or directories) to the profile builder:

```console
python3 corpus_profile.py books/ another/book.txt \
  --name "Public-domain fiction sample" -o corpus_profile.json
python3 corpus_profile.py books/ --manifest corpus-manifest.json -o profile.json
```

The profile contains its schema/parser/metric versions, corpus name, build
timestamp, preprocessing settings, source IDs, filenames, SHA-256 hashes,
per-book counts, optional manifest metadata, robust distributions, and a word
frequency table. Runtime analysis needs only the profile, never the raw books.
Duplicate filename stems remain distinct through hash-derived source IDs.

For byte-reproducible output, set `SOURCE_DATE_EPOCH` or build programmatically
with a fixed `built_at`. Corpus acquisition is intentionally outside this tool.

The optional manifest is JSON whose `sources` member is keyed by relative
filename (or is a list with `filename`/`path` fields). Values can include an
`id` and arbitrary metadata such as author, publication year, or license.

## Shared text processing

All new analysis and profile code uses the shared `textgrader.text` layer for
Unicode-aware words, sentence and paragraph segmentation, Markdown-heading and
Gutenberg removal, quote/dialogue parsing, and configurable transcript lines.
Straight and curly double quotes are normalized consistently. Single-quote
dialogue is not guessed: the parser reports that limitation. Transcript speaker
names are not limited to a built-in cast or a fixed username length.

The regex-labelled clause measurements are intentionally called **lexical
proxies**. They do not claim to parse English grammar. A future NLP extra can
add syntactic analyses without making a heavy dependency mandatory.

## Chapters and legacy commands

Chapter filenames may begin with an arbitrary-length integer such as
`100_epilogue.md`; shared parsing does not truncate it to two characters.
Standalone scripts in `measures/` remain available during migration, but
`grade.py` is the supported structured entry point. Project-specific character
sheets, citation filename conventions, cast aliases, chapter bands, targets,
and prose bans are not defaults.

## Tests

```console
python3 -m unittest discover -s tests -v
```

### Opt-in style and "AI-ish" diagnostics

All additional style diagnostics are **off by default**. They do not claim to
identify authorship or produce a human-likeness score: they expose inspectable
signals that an authoring AI (or a person) can use to review repetitive or
uncharacteristic prose. Enable only the measurements useful to your project:

```json
{
  "metrics": {
    "repeated_ngrams": {"enabled": true, "sizes": [3, 4, 5, 6]},
    "mattr": {"enabled": true, "window": 100},
    "length_quantiles": {"enabled": true},
    "passive_voice": {"enabled": true}
  },
  "nlp": {"model": "en_core_web_sm"}
}
```

Available switches are `repeated_ngrams`, `sentence_openings`,
`local_repetition`, `function_words`, `mattr`, `punctuation`,
`length_quantiles`, `pov_pronouns`, `dialogue_contractions`, `dialogue_tags`,
`passive_voice`, `clause_structure`, `pos_distribution`, `tense_consistency`,
`nominalizations`, and `character_voice`. Each implementation lives in its own
module under `textgrader/metrics/`. Corpus profiles include distributions for
dependency-free scalar metrics and function-word profiles, so new profiles can
compare an enabled manuscript metric to the corpus rather than to a hard-coded
threshold. Rebuild older profiles to add these distributions.

The five grammatical metrics require the optional spaCy package and an English
model. Install them, for example, with `pip install spacy` followed by
`python -m spacy download en_core_web_sm`. If they are enabled without a usable
model, their results remain visible as unavailable warnings; core operation has
no third-party dependency. Character voice distance currently recognizes
transcript-style `Speaker: dialogue` lines and reports when there are not enough
speakers to compare.
