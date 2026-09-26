"""Bulk-corpus adapters: turn a whole pre-existing dataset into the plain-text
folder + ``manifest.json`` shape :mod:`textgrader.corpus` already reads.

``corpus_builder/providers`` acquires INDIVIDUAL books one candidate at a time
(search a catalog, fetch one text, repeat).  Every dataset in this module is
the opposite shape: one fixed, already-assembled archive (a linguistics
corpus, a treebank release, a Hugging Face dataset) that is downloaded or read
once and then split into per-document ``.txt`` files.  Reusing the
``CorpusProvider``/``Candidate`` search-and-fetch protocol for that would mean
inventing a fake one-candidate-per-dataset "search" for something that was
never a catalog, so this module writes ``books/*.txt`` and ``manifest.json``
directly, in exactly the shape :func:`corpus_builder.builder._save` and
:func:`textgrader.corpus._manifest_entries`/``_source_files`` already expect
(a ``"sources"`` mapping keyed by filename, each entry an object; a flat
directory of ``.txt`` files).  ``textgrader.corpus.build_profile`` does not
care which of the two paths produced its input.

Every function here is a task-24 dataset adapter for a Task 24 reference
profile: it downloads (or reads from a local, already-licensed copy), writes
one ``.txt`` file per document, and returns the output directory.  Nothing
here builds a TextGrader corpus *profile*; run
``python -m textgrader.corpus <output_dir>/books -o profile.json --config
config.json --comparison-unit <unit>`` afterwards, exactly as for a
Gutenberg/Standard-Ebooks corpus.

Determinism
    Every adapter records the exact package version, dataset revision, commit
    or release tag it read, in ``manifest.json``, so a profile built from it
    can be rebuilt from the same inputs later.  None of these functions
    silently follow "latest" without recording what "latest" resolved to.

Licensing
    Only openly, anonymously downloadable corpora are fetched automatically,
    and each one's exact terms are recorded in the manifest's
    ``license_note``.  Corpora that need registration, a signed use
    agreement, or a paid licence (the Penn Treebank, OntoNotes, the BNC, the
    RST/PDTB discourse treebanks, COCA, most licensed essay-scoring sets) are
    never fetched here -- see ``LICENSED_RECIPES`` and
    :func:`require_local_licensed_corpus` -- and NLI/contradiction sets
    (SNLI, MNLI) are documented as calibration data, never wrapped as a prose
    reference profile at all -- see ``NLI_CALIBRATION_NOTE``.

Package boundary
    ``corpus_builder`` otherwise depends on nothing beyond the standard
    library (see its README).  The adapters below are the deliberate,
    documented exception: each imports its one third-party package (``nltk``,
    ``datasets``) lazily, inside the function that needs it, and raises
    :class:`AdapterError` with an actionable message -- never a bare
    ``ImportError`` -- when it is missing.  This module intentionally does
    NOT route those imports through ``textgrader.optional.require``:
    ``corpus_builder`` has no import dependency on the ``textgrader`` analysis
    package at all (see this package's README), and this module keeps that
    boundary rather than reaching across it for one shared helper.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .net import NetworkError, fetch, make_opener

USER_AGENT = "TextGrader-CorpusBuilder-DatasetAdapters/1.0"


class AdapterError(RuntimeError):
    """A dataset could not be built: a missing package, missing local files,
    an undownloaded NLTK corpus, or a licence this project will not fetch
    automatically.  Always raised with the exact remedy in the message --
    never presented as "not installed" without saying what to do about it.
    """


# ------------------------------------------------------------- shared helpers

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-") or "document"


def _write_manifest(output_dir: Path, sources: Mapping[str, Mapping[str, Any]],
                    extra: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "builder": {"name": "TextGrader Dataset Adapter", "version": "1.0.0"},
        "sources": dict(sources),
        **extra,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def _write_document(books_dir: Path, filename: str, text: str) -> Path:
    books_dir.mkdir(parents=True, exist_ok=True)
    path = books_dir / filename
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")
    return path


# ------------------------------------------------------------------- NLTK/Brown

def _require_nltk():
    try:
        import nltk
    except ImportError as exc:
        raise AdapterError(
            "the 'nltk' package is required for this adapter: pip install nltk") from exc
    return nltk


#: Every category the Brown Corpus ships, in its own printed order -- kept
#: here (rather than re-derived from ``brown.categories()``) so a caller can
#: see the whole menu, and what each label means, without importing NLTK.
BROWN_CATEGORIES = ("adventure", "belles_lettres", "editorial", "fiction",
                    "government", "hobbies", "humor", "learned", "lore",
                    "mystery", "news", "religion", "reviews", "romance",
                    "science_fiction")

BROWN_LICENSE_NOTE = (
    "NLTK Brown Corpus (W. N. Francis and H. Kucera, Brown University, 1964; "
    "revised 1971/1979). Bundled README: 'Distributed with the permission of "
    "the copyright holder, redistribution permitted.' Customary use is "
    "scholarly/research, matching this project's non-commercial terms.")


def build_brown_corpus(output_dir: str | Path, *, categories: Sequence[str] | None = None,
                       exclude_fileids: Sequence[str] = ()) -> Path:
    """One ``.txt`` per Brown Corpus file, grouped by its Brown category.

    Each category (news, fiction, learned, humor, ...) is a natural,
    ready-made genre split -- exactly the "multi-profile test" the task
    documentation asks for: build one TextGrader profile per category (see
    ``python -m textgrader.corpus``) and a document's reference-fit vector
    against ``news`` and against ``fiction`` will differ.

    ``exclude_fileids`` holds out specific files (by their Brown id, e.g.
    ``"ca01"``) from the written corpus, so they can be graded afterwards as
    genuinely out-of-sample demonstration texts -- never included in both the
    profile and the thing measured against it.

    Requires the NLTK ``brown`` corpus already downloaded: run once,
    ``NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c "import nltk; nltk.download('brown')"``
    (the environment variable is needed in a sandboxed/proxied container; see
    ``corpus_builder/README.md``).
    """

    nltk = _require_nltk()
    try:
        from nltk.corpus import brown
        from nltk.tokenize.treebank import TreebankWordDetokenizer
    except ImportError as exc:  # pragma: no cover - nltk always ships these
        raise AdapterError(f"NLTK is installed but incomplete: {exc}") from exc
    try:
        available = set(brown.categories())
    except LookupError as exc:
        raise AdapterError(
            "the NLTK 'brown' corpus is not downloaded. Run once: "
            "NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c \"import nltk; "
            "nltk.download('brown')\"") from exc
    wanted = list(categories) if categories else sorted(available)
    unknown = sorted(set(wanted) - available)
    if unknown:
        raise AdapterError(f"unknown Brown categories: {', '.join(unknown)}; "
                           f"available: {', '.join(sorted(available))}")

    output_dir = Path(output_dir)
    detok = TreebankWordDetokenizer()
    sources: dict[str, dict[str, Any]] = {}
    excluded = set(exclude_fileids)
    for category in wanted:
        for fileid in brown.fileids(categories=category):
            if fileid in excluded:
                continue
            paragraphs = ["  ".join(detok.detokenize(sentence) for sentence in paragraph)
                         for paragraph in brown.paras(fileid)]
            text = "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())
            filename = f"brown-{fileid}.txt"
            path = _write_document(output_dir / "books", filename, text)
            sources[filename] = {
                "id": f"brown:{fileid}", "author": None,
                "title": f"Brown Corpus {fileid} ({category})",
                "genre": category, "sha256": f"sha256:{_sha256(path)}",
            }
    if not sources:
        raise AdapterError("no Brown files matched the requested categories/exclusions")
    _write_manifest(output_dir, sources, {
        "corpus": "brown", "source": "NLTK Brown Corpus",
        "license_note": BROWN_LICENSE_NOTE, "language": "en", "date_range": "1961",
        "dataset_revision": f"nltk=={nltk.__version__}",
        "categories": wanted, "excluded_fileids": sorted(excluded),
        "document_count": len(sources),
    })
    return output_dir


REUTERS_LICENSE_NOTE = (
    "NLTK Reuters-21578 (ApteMod version). Standard notice: 'The copyright "
    "for the text of newswire articles and Reuters annotations in the "
    "Reuters-21578 collection resides with Reuters Ltd. Reuters Ltd. and "
    "Carnegie Group, Inc. have agreed to allow the free distribution of this "
    "data for research purposes only.' Non-commercial research use only, "
    "matching this project's terms.")


def build_reuters_corpus(output_dir: str | Path, *, categories: Sequence[str] | None = None,
                         min_words: int = 150, max_documents: int | None = None) -> Path:
    """One ``.txt`` per Reuters-21578 (ApteMod) newswire story.

    ``categories`` filters to stories carrying at least one of the given
    topic labels (``reuters.categories()`` lists all 90 -- ``"acq"``,
    ``"earn"``, ``"grain"``, ...); ``None`` keeps every story. Reuters stories
    are short newswire text, so ``min_words`` drops the ones too short to be
    a useful reference-profile observation on their own (most TextGrader
    prose metrics need a genuine sentence sample; a two-line wire brief is
    not one) -- this is the same floor ``grade.py`` itself applies to a
    document before comparing it to any corpus.

    Requires the NLTK ``reuters`` corpus already downloaded: run once,
    ``NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c "import nltk; nltk.download('reuters')"``.
    """

    nltk = _require_nltk()
    try:
        from nltk.corpus import reuters
    except ImportError as exc:  # pragma: no cover
        raise AdapterError(f"NLTK is installed but incomplete: {exc}") from exc
    try:
        all_categories = set(reuters.categories())
    except LookupError as exc:
        raise AdapterError(
            "the NLTK 'reuters' corpus is not downloaded. Run once: "
            "NLTK_ALLOW_PROXIED_URLOPEN=1 python3 -c \"import nltk; "
            "nltk.download('reuters')\"") from exc
    wanted = set(categories) if categories else None
    if wanted:
        unknown = sorted(wanted - all_categories)
        if unknown:
            raise AdapterError(f"unknown Reuters categories: {', '.join(unknown)}")

    output_dir = Path(output_dir)
    sources: dict[str, dict[str, Any]] = {}
    for fileid in reuters.fileids():
        file_categories = reuters.categories(fileid)
        if wanted and not (wanted & set(file_categories)):
            continue
        text = reuters.raw(fileid)
        if len(text.split()) < min_words:
            continue
        filename = f"reuters-{_slug(fileid)}.txt"
        path = _write_document(output_dir / "books", filename, text)
        sources[filename] = {
            "id": f"reuters:{fileid}", "author": None,
            "title": f"Reuters-21578 {fileid}", "genre": "news",
            "topics": list(file_categories), "sha256": f"sha256:{_sha256(path)}",
        }
        if max_documents and len(sources) >= max_documents:
            break
    if not sources:
        raise AdapterError("no Reuters stories matched the requested categories/min_words")
    _write_manifest(output_dir, sources, {
        "corpus": "reuters-21578", "source": "NLTK Reuters-21578 (ApteMod)",
        "license_note": REUTERS_LICENSE_NOTE, "language": "en", "date_range": "1987",
        "dataset_revision": f"nltk=={nltk.__version__}",
        "categories": sorted(wanted) if wanted else "all", "min_words": min_words,
        "document_count": len(sources),
    })
    return output_dir


# ------------------------------------------------------- Universal Dependencies

#: sent_id/newdoc parsing needs only these two comment prefixes and the token
#: table's own blank-line-terminated blocks; a full CoNLL-U parser is not
#: needed to recover the original sentence text UD already stores verbatim.
_UD_TEXT_RE = re.compile(r"^# text\s*=\s*(.*)$")
_UD_NEWDOC_RE = re.compile(r"^# newdoc(?:\s+id\s*=\s*(.*))?$")
_UD_NEWPAR_RE = re.compile(r"^# newpar(?:\s+id\s*=\s*(.*))?$")


def _conllu_documents(text: str) -> list[tuple[str, str]]:
    """(doc_id, plain_text) pairs recovered from one ``.conllu`` file's
    ``# newdoc id``/``# newpar id``/``# text`` comment lines -- the exact
    original, detokenized sentence UD stores beside its own tokenization, so
    no re-detokenization heuristic is needed."""

    documents: list[tuple[str, list[list[str]]]] = []
    doc_index = -1
    for line in text.splitlines():
        newdoc = _UD_NEWDOC_RE.match(line)
        if newdoc:
            doc_index += 1
            documents.append((newdoc.group(1) or f"doc{doc_index}", [[]]))
            continue
        if doc_index < 0:  # a file with no explicit newdoc marker is one document
            doc_index = 0
            documents.append(("doc0", [[]]))
        if _UD_NEWPAR_RE.match(line):
            documents[doc_index][1].append([])
            continue
        sentence = _UD_TEXT_RE.match(line)
        if sentence:
            documents[doc_index][1][-1].append(sentence.group(1))
    return [(doc_id, "\n\n".join(" ".join(sentence for sentence in paragraph if sentence)
                                 for paragraph in paragraphs if paragraph))
           for doc_id, paragraphs in documents]


def build_ud_treebank(output_dir: str | Path, *, repo: str, ref: str,
                      splits: Sequence[str] = ("train", "dev", "test"),
                      file_prefix: str | None = None, timeout: float = 30.0,
                      min_words: int = 150) -> Path:
    """One ``.txt`` per document recovered from a Universal Dependencies
    treebank release, read from ``raw.githubusercontent.com`` (a plain static
    file host that this project's sandboxed egress policy allows, unlike
    ``github.com``/``codeload.github.com`` archive downloads, which a
    container's outbound policy may block -- see ``corpus_builder/README.md``).

    ``repo`` is the GitHub repository under the ``UniversalDependencies``
    organization (e.g. ``"UD_English-EWT"``, ``"UD_English-GUM"``); ``ref`` is
    a released tag (e.g. ``"r2.18"``), pinned explicitly so the corpus is
    reproducible -- this function never resolves "latest". Each treebank's
    ``.conllu`` files already carry the untokenized original sentence on a
    ``# text = ...`` comment and paragraph/document boundaries on
    ``# newpar``/``# newdoc`` comments, so the exact source prose is
    recovered without any re-detokenization heuristic.

    UD's own text is short per document (a handful of paragraphs, e.g. one
    blog post or one newswire story); ``min_words`` drops documents too short
    to be a useful reference-profile observation, matching
    :func:`build_reuters_corpus`'s reasoning.
    """

    prefix = file_prefix or repo.lower().replace("ud_english-", "en_")
    opener = make_opener(USER_AGENT)
    output_dir = Path(output_dir)
    sources: dict[str, dict[str, Any]] = {}
    fetched_files: list[str] = []
    for split in splits:
        filename = f"{prefix}-ud-{split}.conllu"
        url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{ref}/{filename}"
        try:
            raw = fetch(opener, url, timeout).decode("utf-8")
        except NetworkError as exc:
            raise AdapterError(f"could not fetch {url}: {exc}") from exc
        fetched_files.append(filename)
        for doc_id, doc_text in _conllu_documents(raw):
            if len(doc_text.split()) < min_words:
                continue
            out_filename = f"{_slug(repo)}-{split}-{_slug(doc_id)}.txt"
            path = _write_document(output_dir / "books", out_filename, doc_text)
            sources[out_filename] = {
                "id": f"ud:{repo}:{ref}:{split}:{doc_id}", "author": None,
                "title": doc_id, "genre": None, "split": split,
                "sha256": f"sha256:{_sha256(path)}",
            }
    if not sources:
        raise AdapterError(f"no documents recovered from {repo}@{ref} (splits={splits})")
    license_note = ("CC BY-SA 4.0 (Universal Dependencies English-EWT)" if "EWT" in repo.upper()
                    else "CC BY-NC-SA 4.0 (Universal Dependencies English-GUM; "
                         "non-commercial, matching this project's terms)" if "GUM" in repo.upper()
                    else "see the treebank's own LICENSE.txt in its GitHub repository")
    _write_manifest(output_dir, sources, {
        "corpus": repo, "source": f"https://github.com/UniversalDependencies/{repo}",
        "license_note": license_note, "language": "en",
        "dataset_revision": ref, "fetched_files": fetched_files,
        "document_count": len(sources), "min_words": min_words,
    })
    return output_dir


# ------------------------------------------------------------------ LitBank

#: Pinned to a specific commit of dbamman/litbank's README, never "the
#: current main branch", so the same set of 100 Gutenberg ids is fetched
#: every time this adapter runs. LitBank ships its 100-title book list as a
#: Markdown table (``|Gutenberg ID|Date|Author|Title|``) inside README.md
#: itself, not a separate data file -- confirmed by reading the repository at
#: this commit; see :func:`_litbank_book_rows`.
LITBANK_COMMIT = "3e50db0ffc033d7ccbb94f4d88f6b99210328ed8"
LITBANK_README_URL = f"https://raw.githubusercontent.com/dbamman/litbank/{LITBANK_COMMIT}/README.md"
_LITBANK_ROW_RE = re.compile(r"^\|(\d+)\|(\d{3,4})\|([^|]*)\|([^|]*)\|\s*$")
LITBANK_LICENSE_NOTE = (
    "LitBank (Bamman, Sims, Field, Popat, Underwood et al.) is licensed under "
    "a Creative Commons Attribution 4.0 International License (see "
    "https://github.com/dbamman/litbank). LitBank's own annotations are not "
    "fetched here -- only the same 100-title Project Gutenberg book list its "
    "annotations are built over, retrieved through this project's existing "
    "Gutenberg provider -- since only the underlying prose text, not "
    "LitBank's coreference/entity/event annotations, is used as a reference "
    "corpus for prose-style comparison.")


def _litbank_book_rows(readme_text: str) -> list[tuple[str, str, str, str]]:
    """(gutenberg_id, date, author, title) for each of the 100 rows LitBank's
    README.md lists under its ``|Gutenberg ID|Date|Author|Title|`` table."""

    rows = []
    for line in readme_text.splitlines():
        match = _LITBANK_ROW_RE.match(line.strip())
        if match:
            rows.append((match.group(1), match.group(2), match.group(3).strip(),
                        match.group(4).strip()))
    return rows


def build_litbank_corpus(output_dir: str | Path, *, max_books: int | None = None,
                         timeout: float = 30.0, cache_dir: str | Path | None = None) -> Path:
    """LitBank's 100 public-domain, Gutenberg-sourced novels/novel excerpts,
    fetched through the existing :class:`GutenbergProvider` -- reusing this
    project's own acquisition path rather than a second one, per this
    module's docstring.  LitBank's annotation layer (entities, coreference,
    events, quotations) is a separate, much larger download this function
    does not fetch; see the module docstring for why only the underlying text
    is used here.
    """

    from .models import Candidate
    from .providers.gutenberg import GutenbergProvider

    opener = make_opener(USER_AGENT)
    try:
        readme = fetch(opener, LITBANK_README_URL, timeout).decode("utf-8")
    except NetworkError as exc:
        raise AdapterError(f"could not fetch LitBank's README (book list): {exc}") from exc
    rows = _litbank_book_rows(readme)
    if max_books:
        rows = rows[:max_books]
    if not rows:
        raise AdapterError(
            "no book rows recognised in LitBank's README at the pinned commit; its "
            "|Gutenberg ID|Date|Author|Title| table format may have changed -- update "
            "dataset_adapters.LITBANK_COMMIT/_LITBANK_ROW_RE")

    output_dir = Path(output_dir)
    cache = Path(cache_dir) if cache_dir else output_dir / ".cache"
    provider = GutenbergProvider(cache_dir=cache, timeout=timeout)
    sources: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for gid, date, author, title in rows:
        candidate = Candidate("gutenberg", gid, title, [author] if author else [])
        try:
            document = provider.fetch(candidate)
        except NetworkError as exc:
            errors.append(f"{gid} ({title}): {exc}")
            continue
        filename = f"litbank-{gid}-{_slug(title)}.txt"
        path = _write_document(output_dir / "books", filename, document.text)
        sources[filename] = {
            "id": f"litbank:gutenberg:{gid}", "author": author, "title": title,
            "genre": "fiction", "first_publish_year": date, "source_url": document.source_url,
            "sha256": f"sha256:{_sha256(path)}",
        }
    if not sources:
        raise AdapterError("no LitBank title could be downloaded: " + "; ".join(errors[:5]))
    _write_manifest(output_dir, sources, {
        "corpus": "litbank", "source": "LitBank book list (public-domain Gutenberg novels)",
        "license_note": LITBANK_LICENSE_NOTE, "language": "en",
        "dataset_revision": f"litbank-booklist@{LITBANK_COMMIT}",
        "document_count": len(sources),
        "download_errors": errors,
    })
    return output_dir


# ------------------------------------------------------- Hugging Face datasets

def build_wikitext_corpus(output_dir: str | Path, *, config_name: str = "wikitext-2-raw-v1",
                          revision: str = "b08601e04326c79dfdd32d625aee71d232d685c3",
                          split: str = "train", max_documents: int = 300,
                          min_words: int = 150) -> Path:
    """WikiText articles via the Hugging Face ``datasets`` library, pinned to
    one dataset revision so the exact bytes downloaded are reproducible.

    WikiText articles are wrapped one per Wikipedia page, separated in the
    raw split by a ``" = Title = \\n"`` heading line; this groups the raw
    lines back into one document per heading rather than treating WikiText's
    already-tokenized line stream as one giant document. ``max_documents``
    bounds a first, small profile (this project ships nothing built from this
    adapter; see ``corpus_builder/README.md``).

    Optional: ``pip install datasets`` (run ``pip install --dry-run
    datasets`` first, per this project's dependency rules) if it is not
    already installed.
    """

    try:
        import datasets
    except ImportError as exc:
        raise AdapterError(
            "the 'datasets' package is required for the WikiText adapter: "
            "run 'pip install --dry-run datasets' first, then 'pip install "
            "datasets' if nothing else would be downgraded") from exc

    try:
        dataset = datasets.load_dataset("Salesforce/wikitext", config_name,
                                        split=split, revision=revision)
    except Exception as exc:  # network failure, revision moved, etc.
        raise AdapterError(f"could not load Salesforce/wikitext@{revision} "
                           f"({config_name}, split={split}): {exc}") from exc

    output_dir = Path(output_dir)
    sources: dict[str, dict[str, Any]] = {}
    heading_re = re.compile(r"^\s*=\s*([^=].*?)\s*=\s*$")
    current_title, current_lines = None, []

    def _flush():
        if current_title is None or not current_lines:
            return
        text = "\n".join(line for line in current_lines if line.strip())
        if len(text.split()) < min_words:
            return
        filename = f"wikitext-{_slug(current_title)}.txt"
        path = _write_document(output_dir / "books", filename, text)
        sources[filename] = {
            "id": f"wikitext:{current_title}", "author": None, "title": current_title,
            "genre": "encyclopedic", "sha256": f"sha256:{_sha256(path)}",
        }

    for row in dataset:
        line = row.get("text", "")
        match = heading_re.match(line)
        # A single "= Title =" (not "== Section ==") starts a new article.
        if match and not line.strip().startswith("=="):
            _flush()
            current_title, current_lines = match.group(1), []
            if max_documents and len(sources) >= max_documents:
                break
            continue
        if current_title is not None:
            current_lines.append(line)
    _flush()

    if not sources:
        raise AdapterError("no WikiText article met min_words; lower it or use a larger split")
    _write_manifest(output_dir, sources, {
        "corpus": "wikitext", "source": f"huggingface:Salesforce/wikitext:{config_name}",
        "license_note": "Creative Commons Attribution-ShareAlike License "
                        "(Wikipedia text; see https://huggingface.co/datasets/Salesforce/wikitext)",
        "language": "en", "dataset_revision": revision,
        "dataset_config": config_name, "dataset_split": split,
        "document_count": len(sources), "min_words": min_words,
    })
    return output_dir


# ------------------------------------------------------ licence-restricted sets

#: Corpora this project will never download automatically: each needs a
#: registration, a signed data-use agreement, institutional/paid access, or
#: (PAN) a shared-task submission account, so an anonymous script has no
#: legitimate way to fetch them.  Every entry is a *recipe* -- what to obtain,
#: from where, under what terms -- for someone who already holds a licence to
#: point ``require_local_licensed_corpus`` at.
LICENSED_RECIPES: dict[str, str] = {
    "bnc": (
        "British National Corpus (BNC), 100M words. Licensed by Oxford Text "
        "Archive/BNC Consortium; the XML edition is available to registered "
        "users (free for non-commercial research after registration, or "
        "purchasable). Register and download from http://www.natcorp.ox.ac.uk/ "
        "or https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554 , "
        "extract the World Edition XML, convert each <w>/<s> text to plain "
        "prose (stdlib xml.etree is enough; no BNC-specific tool needed), and "
        "point this project's build_profile at the resulting .txt files."),
    "penn_treebank": (
        "Penn Treebank (PTB), LDC99T42. Requires an LDC membership or "
        "per-corpus licence purchase: https://catalog.ldc.upenn.edu/LDC99T42 . "
        "Once obtained, its WSJ .mrg files' leaf tokens (via nltk.corpus.BracketParseCorpusReader "
        "or nltk.tree.Tree.fromstring) give plain prose; treat one .mrg "
        "section as one document."),
    "ontonotes": (
        "OntoNotes 5.0, LDC2013T19. Requires an LDC licence: "
        "https://catalog.ldc.upenn.edu/LDC2013T19 . The English newswire/"
        "broadcast/web/conversational-telephone-speech sections each make a "
        "natural separate reference profile (they differ enormously in "
        "register); extract plain text from the .onf/.parse files' leaves."),
    "rst_dt": (
        "RST Discourse Treebank, LDC2002T07. Requires an LDC licence: "
        "https://catalog.ldc.upenn.edu/LDC2002T07 . Built over a subset of "
        "the (also licensed) Penn Treebank's WSJ text; the .out files contain "
        "the raw article text alongside the RST tree annotation."),
    "pdtb": (
        "Penn Discourse TreeBank 3.0, LDC2019T05. Requires an LDC licence: "
        "https://catalog.ldc.upenn.edu/LDC2019T05 . Also built over licensed "
        "WSJ text (see penn_treebank/rst_dt above); the raw .txt files under "
        "its raw/ directory are plain prose, ready for build_profile."),
    "coca": (
        "Corpus of Contemporary American English (COCA). Full-text access is "
        "licensed/paid via https://www.corpusdata.org/ ; a free web interface "
        "exists but does not permit bulk download of full documents. Obtain "
        "a licensed text release before pointing build_profile at it."),
    "anc_restricted": (
        "Open American National Corpus (OANC) itself is public domain and "
        "needs no recipe here (download directly from "
        "https://anc.org/data/oanc/ and feed its .txt files to build_profile). "
        "The Manually Annotated Sub-Corpus (MASC) is likewise open. Only the "
        "SEPARATE 'full' American National Corpus (second release), which "
        "bundles some rights-restricted text, needs a licence; obtain it "
        "from the Linguistic Data Consortium if that specific release is "
        "wanted instead of the open OANC/MASC."),
    "essay_scoring_licensed": (
        "Human-scored essay datasets with restrictive terms (e.g. the ETS "
        "Corpus of Non-Native Written English, LDC2014T06, or a licensed "
        "state assessment's writing sample set). Obtain under that dataset's "
        "own licence (often an LDC membership or a direct agreement with the "
        "issuing testing organization), including its scoring rubric; record "
        "the rubric/scale alongside the profile's metadata so a fit-distance "
        "reading can be interpreted against the same rubric it was scored "
        "under. The publicly downloadable Hewlett/ASAP Automated Student "
        "Assessment Prize essays (Kaggle 'asap-aes') are usable for "
        "non-commercial research under Kaggle's competition rules, but "
        "require a Kaggle account/API token to fetch, so they are documented "
        "here rather than auto-downloaded."),
}

NLI_CALIBRATION_NOTE = (
    "SNLI and MNLI (and other NLI/entailment/contradiction datasets) are "
    "sentence-pair judgment data for validating a contradiction/entailment "
    "detector's calibration -- see textgrader.metrics.logic_suite's "
    "'nli_entailment' feature, which they can calibrate against. They are "
    "NOT prose written for its own sake, so this project never wraps them as "
    "a reference_profiles corpus profile: comparing a manuscript's prose "
    "style against a corpus of isolated, decontextualized sentence pairs "
    "would not describe any genre of writing a document could actually be "
    "measured against.")

PAN_NOTE = (
    "PAN shared-task corpora (authorship attribution/verification, "
    "style-change detection) are distributed per-edition, almost always "
    "through Zenodo or the PAN/TIRA infrastructure, and most editions require "
    "either agreeing to task-specific terms during a shared-task submission "
    "or registering with Zenodo/TIRA before the archive is released to you -- "
    "there is no single, stable, anonymously-fetchable URL across editions "
    "the way NLTK's or a pinned GitHub release's is. This project therefore "
    "documents PAN as a recipe rather than an adapter: once a specific "
    "edition's archive has been obtained under its own terms (checking that "
    "edition's LICENSE/README; several PAN authorship-verification sets are "
    "released CC-BY or CC-BY-NC once public), point "
    "require_local_licensed_corpus at the extracted per-author text files.")


def require_local_licensed_corpus(name: str, source_dir: str | Path) -> Path:
    """Validate a licence-restricted corpus the caller has already obtained.

    Never downloads anything.  Raises :class:`AdapterError` naming the
    documented recipe (see ``LICENSED_RECIPES``) whenever ``source_dir`` does
    not exist or holds no ``.txt``/``.md`` files, so a misconfigured path
    fails with the exact remedy rather than a bare ``FileNotFoundError``.
    """

    recipe = LICENSED_RECIPES.get(name)
    if recipe is None:
        raise AdapterError(
            f"unknown licensed corpus {name!r}; known recipes: "
            f"{', '.join(sorted(LICENSED_RECIPES))}")
    path = Path(source_dir)
    if not path.is_dir():
        raise AdapterError(f"{name}: {source_dir} does not exist. {recipe}")
    texts = list(path.rglob("*.txt")) + list(path.rglob("*.md"))
    if not texts:
        raise AdapterError(f"{name}: {source_dir} has no .txt/.md files yet. {recipe}")
    return path


__all__ = [
    "AdapterError", "BROWN_CATEGORIES", "build_brown_corpus", "build_reuters_corpus",
    "build_ud_treebank", "build_litbank_corpus", "build_wikitext_corpus",
    "LICENSED_RECIPES", "NLI_CALIBRATION_NOTE", "PAN_NOTE",
    "require_local_licensed_corpus",
]
