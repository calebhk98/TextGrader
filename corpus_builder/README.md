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

## Task 24: dataset adapters for genre/task-aware reference profiles

`dataset_adapters.py` extends this package with adapters for several whole,
pre-assembled corpora, each producing the same `books/*.txt` + `manifest.json`
shape `providers/` already does, so `textgrader.corpus.build_profile` reads
either path identically:

```console
python3 -m textgrader.corpus <output_dir>/books -o profile.json \
    --config config.json --comparison-unit <book|chapter|scene|passage>
```

**Never commit a built corpus or a built profile into this repository** (see
`.gitignore`); build them in a scratch directory outside the repo, or keep
them in your own private storage.

### Implemented, real adapters

| Adapter | Source | License / terms | Setup |
| --- | --- | --- | --- |
| `build_brown_corpus` | NLTK Brown Corpus (Francis & Kucera, 1964/1979), 500 documents across 15 genre categories (news, fiction, learned, humor, romance, science_fiction, ...) | "Distributed with the permission of the copyright holder, redistribution permitted" (bundled README); customary scholarly/research use | `NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c "import nltk; nltk.download('brown')"` once |
| `build_reuters_corpus` | NLTK Reuters-21578 (ApteMod), ~10,700 newswire stories, 90 topic categories | "Free... for research purposes only" (standard Reuters-21578 notice) | `NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c "import nltk; nltk.download('reuters')"` once |
| `build_ud_treebank` | A Universal Dependencies English treebank release (`UD_English-EWT`, `UD_English-GUM`, ...), read via `raw.githubusercontent.com` (a plain static file host; a sandboxed egress policy that blocks `github.com`/`codeload.github.com` archive downloads may still allow this) | EWT: CC BY-SA 4.0. GUM: CC BY-NC-SA 4.0 (non-commercial, matching this project's own terms) | none; pin `ref` to a released tag, e.g. `"r2.18"` |
| `build_litbank_corpus` | The 100 public-domain Gutenberg novels LitBank's annotations are built over (fetched through the existing `GutenbergProvider`, not LitBank's own, much larger annotation download) | CC BY 4.0 (LitBank itself; the underlying texts are public-domain Gutenberg works) | none |
| `build_wikitext_corpus` | WikiText (`Salesforce/wikitext` on Hugging Face) via the `datasets` library, pinned to one dataset revision | CC BY-SA (Wikipedia text) | optional: `pip install --dry-run datasets` first (already satisfied if `datasets` is installed for another reason), then `pip install datasets` |

Every adapter's `manifest.json` records `source`, `license_note`, `language`,
`dataset_revision` (an NLTK version, a UD release tag, a Hugging Face
revision, or LitBank's own pinned commit) and a per-document `sha256`, so a
profile built from it is reproducible: the same pinned revision always
produces the same bytes.

`NLTK_ALLOW_PROXIED_URLOPEN=1` is required in a sandboxed/proxied container
(such as this project's own development containers) for `nltk.download(...)`
to reach NLTK's data servers through the outbound proxy; it is a one-time
setup step, not something a grading run needs.

A worked example (Brown News vs. Brown Fiction, two genuinely different
reference profiles built from the SAME adapter):

```console
python3 - <<'PY'
from corpus_builder.dataset_adapters import build_brown_corpus
build_brown_corpus("/tmp/brown_news", categories=["news"])
build_brown_corpus("/tmp/brown_fiction", categories=["fiction"])
PY
python3 -m textgrader.corpus /tmp/brown_news/books -o /tmp/brown_news.json \
    --config config.json --comparison-unit passage --name "Brown: news"
python3 -m textgrader.corpus /tmp/brown_fiction/books -o /tmp/brown_fiction.json \
    --config config.json --comparison-unit passage --name "Brown: fiction"
```

Then add both under `reference_profiles` in `config.json` (see that key's
`_notes`) and grade any manuscript to get one reference-fit reading against
each, alongside the shipped `corpus_profile`, via the `reference_fit` metric
(`metrics.reference_fit.enabled: true`).

### Documented recipes only (never auto-downloaded)

These datasets require registration, a signed data-use agreement, or a paid
licence, so an anonymous script has no legitimate way to fetch them.
`dataset_adapters.LICENSED_RECIPES` documents, for each one, what to obtain
and from where; `dataset_adapters.require_local_licensed_corpus(name,
source_dir)` validates a copy you have already obtained yourself (raising
`AdapterError` with the recipe text if the directory is missing or empty)
rather than downloading anything:

- **British National Corpus (BNC)** — register at the Oxford Text Archive /
  BNC Consortium; free for non-commercial research after registration.
- **Penn Treebank (PTB)**, **OntoNotes 5.0**, **RST Discourse Treebank**,
  **Penn Discourse TreeBank** — each requires an LDC licence/membership
  (`catalog.ldc.upenn.edu`); RST-DT and PDTB are both built over the
  (also licensed) Penn Treebank WSJ text.
- **COCA** — licensed/paid full-text access via `corpusdata.org`; the free web
  interface does not permit bulk document download.
- **Licensed essay-scoring sets** (e.g. the ETS Corpus of Non-Native Written
  English) — obtained under that dataset's own terms, together with its
  scoring rubric, recorded alongside the profile's metadata so a fit-distance
  reading can be interpreted against the rubric it was scored under. The
  publicly downloadable Hewlett/ASAP essays are usable for non-commercial
  research but need a Kaggle account/API token, so they are documented rather
  than auto-fetched.
- **PAN** (authorship attribution/verification/style-change) — distributed
  per shared-task edition, almost always gated behind Zenodo/TIRA
  registration or shared-task submission terms; see `dataset_adapters.PAN_NOTE`
  for the full reasoning. Several editions are CC-BY/CC-BY-NC once public;
  check the specific edition's own licence before using it.
- **Open American National Corpus (OANC) / MASC** are already public domain
  and need **no** recipe — download directly from `anc.org/data/oanc/` and
  feed the `.txt` files straight to `build_profile`. Only the separate,
  rights-bundling "full" ANC (second release) needs an LDC licence; see
  `LICENSED_RECIPES["anc_restricted"]`.

### Documented as calibration data, not a reference profile

**SNLI, MNLI** and other NLI/entailment/contradiction datasets are
sentence-pair judgment data for validating `textgrader.metrics.logic_suite`'s
`nli_entailment` feature. They are not prose written for its own sake, so
this project never wraps them as a `reference_profiles` corpus: comparing a
manuscript's style against a corpus of isolated sentence pairs would not
describe any genre of writing. See `dataset_adapters.NLI_CALIBRATION_NOTE`.

### Google Books Ngrams

Aggregate historical n-gram frequency counts, not a document corpus (there is
no per-document text to profile), so there is no adapter here; it belongs
beside `wordfreq`/lexicon-style frequency resources (Task 11's psycholinguistic
norm loader), not `reference_profiles`.
