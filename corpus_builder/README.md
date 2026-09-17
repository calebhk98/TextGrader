# corpus_builder

Acquires public-domain text from digital libraries. Separate from `textgrader/`
on purpose: it builds a corpus, it does not grade one, and it shares no code
with the analysis package.

Providers: Project Gutenberg, Standard Ebooks, Internet Archive, Wikisource,
Google Books, Library of Congress. Metadata enrichment from Open Library and
Google Books. Plain text, HTML and EPUB extraction use the standard library
only; there is no third-party dependency here.

Driven by `build_corpus.py` at the repository root:

```console
python3 build_corpus.py --list-providers
python3 build_corpus.py --config config.json --healthcheck
python3 build_corpus.py --config config.json
```

Settings live under `corpus_builder` in the project configuration, or in a
standalone JSON file. `examples/corpus_builder.science_fiction.json` is a
worked example.

A provider that fails does not stop the others. A search result is not treated
as permission: records without a usable full-text format are skipped rather
than guessed at, and the manifest records where every file came from.
