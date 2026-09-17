# Sample corpus

Put plain-text reference books here, then build a profile from them:

```console
python3 -m textgrader.corpus sample_corpus/ --name "my corpus" \
    --comparison-unit book --config config.json -o my-corpus.json
```

and point `corpus_profile` in `config.json` at the result. Grading reads only
the profile, never the books, so this directory is not needed at run time.

`python3 build_corpus.py --config config.json` can fill it for you from the
public-domain providers configured under `corpus_builder`.

`--comparison-unit` matters: it records whether one file here is a whole book,
a chapter or a scene. TextGrader refuses to compare a count-based metric across
a mismatch, because a chapter "failing" document length against a shelf of
novels is a fact about how books are divided, not about the chapter.
