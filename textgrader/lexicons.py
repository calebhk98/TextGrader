"""Cached, versioned loading of psycholinguistic and affective norm tables.

This is the **single place** any norm/affect lexicon (age of acquisition,
concreteness, imageability, familiarity, valence/arousal/dominance,
sensorimotor strength, subtitle frequency/contextual diversity, lexical-
decision norms, ...) is downloaded, parsed, cached and looked up.
:mod:`textgrader.metrics.lexical_norms_suite` is the first caller; a later
task (affect/sentiment) is expected to call :func:`score_tokens` directly for
its own per-sentence VAD scoring rather than re-implementing a loader, which
is why the public surface below is deliberately small and format-agnostic.

Public API
----------

``norm_table(name, path=None) -> (table, reason, metadata)``
    ``table`` is ``None`` (with ``reason`` explaining why) or a mapping
    ``{lowercased_word: {dimension_key: float}}``. ``metadata`` is always a
    dict (see below), even when the table could not be loaded.

``score_tokens(name, tokens, dimension=None) -> list[float | None]``
    One score per token, aligned positionally with ``tokens`` (already
    lower-cased the way :attr:`textgrader.document.DocumentAnalysis.tokens`
    lower-cases them -- this module does not re-normalize case or curly
    quotes). ``None`` where the resource has no entry for that token, or
    where the resource itself is unavailable. ``dimension`` may be omitted
    when the resource has exactly one.

``reference_quantiles(name, dimension) -> dict | None``
    The value distribution *of the norm table itself* (not of any document):
    ``count``, ``mean``, ``std``, ``p10``/``p25``/``p50``/``p75``/``p90``,
    ``min``, ``max``. Metrics use this as the "norm-specific reference
    quantile" a document's low/high tail share is measured against, so the
    threshold is drawn from the lexicon's own spread rather than invented per
    metric. Cached alongside the table.

``resource_info(name) -> ResourceSpec`` / ``list_resources() -> list[str]``
    Static metadata (title, homepage, citation, license, dimensions, source
    URL, cache filename) without loading or downloading anything.

``download(name, force=False) -> DownloadResult``
    Fetches and caches one resource (or, with ``name="all"``, every
    resource that has a working, licence-clear download path). Never raises;
    failures come back as ``DownloadResult(ok=False, message=...)`` with the
    real HTTP/network error or the registration/licensing wall quoted.
    Callable as ``python -m textgrader.lexicons download <name|all>``.

Every ``metadata`` dict carries ``resource``, ``version``, ``source_url``,
``sha256`` and ``entry_count``, because a corpus profile built against one
version of a norm file must never be silently compared against another (see
the cross-task rule this task inherits). A metric that reports a norm value
without this metadata beside it is a bug in that metric, not in this module.

Caching and overrides
----------------------

Downloads are cached under ``~/.cache/textgrader/lexicons/<name>/`` (override
the root with the ``TEXTGRADER_LEXICON_CACHE_DIR`` environment variable).
**Nothing here is ever committed to the repository.** A caller -- typically
``lexical_norms_suite``'s ``resource_paths`` config option -- may instead
point a resource at a user-supplied file with ``path=``; the override is
parsed the same way and is never re-downloaded or written back to the cache.
Parsed tables are cached in-process (keyed by resolved file path and its
mtime) so a book-length grading run or a multi-document corpus build parses
each resource file once, per :mod:`textgrader.metrics.common`'s caching rule.

Resources and how each was obtained
------------------------------------

Every resource below was fetched from its own authoritative publication site
or a peer institution's OSF mirror of it (**not** re-hosted or bundled by
TextGrader), and it is worth recording which sources turned out to be
unreachable: ``crr.ugent.be`` (Marc Brysbaert's own site, historically the
canonical host for the Brysbaert/Warriner/Kuperman/SUBTLEX-US files) returned
its own "under reconstruction" notice for every path tried during this task's
research, and the Internet Archive's Wayback Machine could not be reached
through this environment's egress proxy at all (``ws_closed_mid_exchange``
on every attempt). The OSF mirrors the original authors themselves deposited
their data on were reachable instead and are what this module actually
downloads from; see each :class:`ResourceSpec` for the exact URL used.

Two resources named in the task spec could not be obtained at all, and are
registered with ``download_url=None`` and a ``notes`` string quoting exactly
what was found, so they fail loudly and specifically rather than silently:

``english_lexicon_project``
    ``elexicon.wustl.edu`` -- the project's own home page -- now serves an
    unrelated open-source room-booking application (LibreBooking; its
    ``README.md`` begins "Welcome to LibreBooking... This is a community
    effort to keep the OpenSource GPLv3 LibreBooking alive"), so the
    project's original bulk-download page is simply gone, not merely
    registration-walled. No alternative bulk (non-web-query) mirror of the
    lexical-decision RT/accuracy data was found. ``path=`` support is kept so
    a user with their own licensed export can still use it.
``celex``
    CELEX is distributed by the LDC under a paid licence
    (LDC96L14/LDC96L05); there is no free redistribution path, and this
    project does not hold a licence to fetch or bundle it. ``path=`` support
    is kept for a user who has their own licensed copy.

TAALES itself (as opposed to TAALED, its lexical-diversity sibling, which is
a real, installable PyPI package -- see ``lexical_norms_suite``'s module
docstring) is a Java desktop application distributed as a downloadable tool,
not a library or a data file; nothing here fetches it, and its acceptance
criteria are covered instead by the PyPI packages
:mod:`textgrader.metrics.lexical_norms_suite` cross-checks against.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .optional import require
from .stats import summarize

_USER_AGENT = ("TextGrader/lexicons (+https://github.com/; research use; "
               "contact via repository issues)")
_TIMEOUT = 60.0


def _cache_root() -> Path:
    override = os.environ.get("TEXTGRADER_LEXICON_CACHE_DIR")
    return Path(override).expanduser() if override else Path.home() / ".cache" / "textgrader" / "lexicons"


def cache_dir(name: str) -> Path:
    path = _cache_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------- resource specs

@dataclass(frozen=True)
class Dimension:
    key: str
    unit: str
    scale: tuple[float, float] | None
    description: str


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    title: str
    version: str
    homepage: str
    citation: str
    license: str
    dimensions: tuple[Dimension, ...]
    #: URL actually fetched by ``download()``; ``None`` means "known to be
    #: unobtainable without a licence/registration this project does not
    #: have" -- see ``notes`` for exactly what was tried.
    download_url: str | None
    filename: str
    #: "xlsx" | "csv" | "tsv" | "mrc_fixed_width" | "zip:<member-path>"
    format: str
    parser: str  # name of a module-level ``_parse_<x>`` function
    notes: str = ""


def _dims(*items: tuple[str, str, tuple[float, float] | None, str]) -> tuple[Dimension, ...]:
    return tuple(Dimension(key, unit, scale, desc) for key, unit, scale, desc in items)


RESOURCES: dict[str, ResourceSpec] = {
    "brysbaert_concreteness": ResourceSpec(
        name="brysbaert_concreteness", title="Brysbaert/Warriner/Kuperman concreteness ratings",
        version="brysbaert-2014",
        homepage="https://doi.org/10.3758/s13428-013-0403-5",
        citation="Brysbaert, M., Warriner, A.B., & Kuperman, V. (2014). Concreteness ratings "
                 "for 40 thousand generally known English word lemmas. Behavior Research "
                 "Methods, 46, 904-911.",
        license="Free for research/educational use; cite the paper.",
        dimensions=_dims(("concreteness", "1-5 rating", (1.0, 5.0),
                          "how concrete/perceptible (5) vs abstract (1) a word's referent is")),
        # crr.ugent.be (the paper's own host) is down ("under reconstruction");
        # this is the OSF copy the co-authors themselves deposited.
        download_url="https://osf.io/download/8ne54/",
        filename="Concreteness_ratings_Brysbaert_et_al_BRM.xlsx", format="xlsx",
        parser="_parse_brysbaert_concreteness"),
    "kuperman_aoa": ResourceSpec(
        name="kuperman_aoa", title="Kuperman age-of-acquisition ratings", version="kuperman-2012",
        homepage="https://doi.org/10.3758/s13428-012-0210-4",
        citation="Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012). Age-of-"
                 "acquisition ratings for 30,000 English words. Behavior Research Methods, "
                 "44, 978-990.",
        license="Free for research/educational use; cite the paper.",
        dimensions=_dims(
            ("aoa", "years", (0.0, 25.0), "estimated age (years) a word is typically learned, "
             "surface/base form"),
            ("aoa_lemma", "years", (0.0, 25.0), "the same rating aggregated to the word's "
             "highest-frequency lemma form")),
        download_url="https://osf.io/download/bx7vm/", filename="AoA_51715_words.xlsx",
        format="xlsx", parser="_parse_kuperman_aoa"),
    "warriner_vad": ResourceSpec(
        name="warriner_vad", title="Warriner/Kuperman/Brysbaert valence-arousal-dominance norms",
        version="warriner-2013",
        homepage="https://doi.org/10.3758/s13428-012-0314-x",
        citation="Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, "
                 "arousal, and dominance for 13,915 English lemmas. Behavior Research Methods, "
                 "45, 1191-1207.",
        license="Free for research/educational use; cite the paper.",
        dimensions=_dims(
            ("valence", "1-9 rating", (1.0, 9.0), "unhappy/happy"),
            ("arousal", "1-9 rating", (1.0, 9.0), "calm/excited"),
            ("dominance", "1-9 rating", (1.0, 9.0), "controlled/in-control")),
        download_url="https://osf.io/download/pyzv9/",
        filename="Ratings_Warriner_et_al.csv", format="csv", parser="_parse_warriner_vad"),
    "nrc_vad": ResourceSpec(
        name="nrc_vad", title="NRC Valence-Arousal-Dominance Lexicon", version="nrc-vad-2.1",
        homepage="https://saifmohammad.com/WebPages/nrc-vad.html",
        citation="Mohammad, S. (2018). Obtaining reliable human ratings of valence, arousal, "
                 "and dominance for 20,000 English words. Proceedings of ACL 2018.",
        license="Free for non-commercial research/educational use (NRC licence); see the "
                "resource's own README.txt for the full terms.",
        dimensions=_dims(
            ("valence", "-1 to 1", (-1.0, 1.0), "unpleasant/pleasant (unigram subset)"),
            ("arousal", "-1 to 1", (-1.0, 1.0), "calm/excited (unigram subset)"),
            ("dominance", "-1 to 1", (-1.0, 1.0), "weak/powerful (unigram subset)")),
        download_url="https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip",
        filename="NRC-VAD-Lexicon-v2.1.zip",
        format="zip:NRC-VAD-Lexicon-v2.1/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt",
        parser="_parse_nrc_vad"),
    "lancaster_sensorimotor": ResourceSpec(
        name="lancaster_sensorimotor", title="Lancaster Sensorimotor Norms", version="lynott-2020",
        homepage="https://doi.org/10.3758/s13428-019-01316-z",
        citation="Lynott, D., Connell, L., Brysbaert, M., Brand, J., & Carney, J. (2020). The "
                 "Lancaster Sensorimotor Norms: multidimensional measures of perceptual and "
                 "action strength for 40,000 English words. Behavior Research Methods, 52, "
                 "1271-1291.",
        license="Free for research use (CC BY-SA 4.0, per the OSF deposit); cite the paper.",
        dimensions=_dims(
            ("perceptual_strength", "0-5 rating", (0.0, 5.0),
             "strength of the word's dominant perceptual modality (vision/audition/touch/"
             "smell/taste/interoception)"),
            ("action_strength", "0-5 rating", (0.0, 5.0),
             "strength of the word's dominant effector/action modality (foot-leg/hand-arm/"
             "head/mouth/torso)")),
        download_url="https://osf.io/download/48wsc/",
        filename="Lancaster_sensorimotor_norms_for_39707_words.csv", format="csv",
        parser="_parse_lancaster_sensorimotor"),
    "subtlex_us": ResourceSpec(
        name="subtlex_us", title="SUBTLEX-US subtitle word frequency / contextual diversity",
        version="brysbaert-new-2009",
        homepage="https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus",
        citation="Brysbaert, M., & New, B. (2009). Moving beyond Kucera and Francis: A "
                 "critical evaluation of current word frequency norms. Behavior Research "
                 "Methods, 41, 977-990.",
        license="Free for research use (CC BY-SA, per the openlexicon mirror); cite the paper.",
        dimensions=_dims(
            ("zipf", "zipf (log10 per billion)", (0.0, 8.0),
             "Zipf-scale frequency computed from SUBTLEXWF, Van Heuven et al. (2014)'s formula "
             "-- an independent frequency source from wordfreq, kept as its own finding"),
            ("contextual_diversity", "%", (0.0, 100.0),
             "percentage of the 8,388 films/episodes the word occurs in (SUBTLEXCD)")),
        # crr.ugent.be is down; lexique.org is Christophe Pallier's openlexicon mirror,
        # which hosts the original authors' own distributed file unmodified.
        download_url="http://www.lexique.org/databases/SUBTLEX-US/SUBTLEXus74286wordstextversion.tsv",
        filename="SUBTLEXus74286wordstextversion.tsv", format="tsv", parser="_parse_subtlex_us"),
    "glasgow_norms": ResourceSpec(
        name="glasgow_norms", title="The Glasgow Norms", version="scott-2019",
        homepage="https://doi.org/10.3758/s13428-018-1099-3",
        citation="Scott, G.G., Keitel, A., Becirspahic, M., Yao, B., & Sereno, S.C. (2019). "
                 "The Glasgow Norms: ratings of 5,500 words on nine scales. Behavior Research "
                 "Methods, 51, 1258-1270.",
        license="Free for research/educational use; cite the paper.",
        dimensions=_dims(
            ("arousal", "1-9 rating", (1.0, 9.0), "calm/excited"),
            ("valence", "1-9 rating", (1.0, 9.0), "unhappy/happy"),
            ("dominance", "1-9 rating", (1.0, 9.0), "controlled/in-control"),
            ("concreteness", "1-7 rating", (1.0, 7.0), "abstract/concrete"),
            ("imageability", "1-7 rating", (1.0, 7.0), "hard/easy to form a mental image of"),
            ("familiarity", "1-7 rating", (1.0, 7.0), "unfamiliar/familiar"),
            ("age_of_acquisition", "1-7 banded rating", (1.0, 7.0),
             "banded self-report age of acquisition (independent of Kuperman's AoA)"),
            ("semantic_size", "1-7 rating", (1.0, 7.0), "how small/large the referent is"),
            ("gender_association", "1-7 rating", (1.0, 7.0), "feminine/masculine association")),
        download_url="https://osf.io/download/ud367/", filename="GlasgowNorms.xlsx",
        format="xlsx", parser="_parse_glasgow_norms"),
    "mrc": ResourceSpec(
        name="mrc", title="MRC Psycholinguistic Database (machine-usable dictionary)",
        version="mrc2.00",
        homepage="https://doi.org/10.3758/BF03202594",
        citation="Coltheart, M. (1981). The MRC Psycholinguistic Database. Quarterly Journal "
                 "of Experimental Psychology, 33A, 497-505. Wilson, M.D. (1988). MRC "
                 "Psycholinguistic Database: machine readable dictionary, version 2. "
                 "Behavior Research Methods, Instruments & Computers, 20, 6-11.",
        license="Distributed for research use (Oxford Text Archive item 1054 / CC "
                "BY-NC-SA per the community mirror used here); cite the paper.",
        dimensions=_dims(
            ("familiarity", "100-700 rating", (100.0, 700.0), "Colheart/Pavio familiarity, "
             "0 in the source dictionary means 'not rated' and is treated as missing"),
            ("concreteness", "100-700 rating", (100.0, 700.0), "0 means 'not rated'"),
            ("imageability", "100-700 rating", (100.0, 700.0), "0 means 'not rated'"),
            ("meaningfulness_colorado", "100-700 rating", (100.0, 700.0),
             "Colorado-norms meaningfulness; 0 means 'not rated'"),
            ("meaningfulness_paivio", "100-700 rating", (100.0, 700.0),
             "Paivio-norms meaningfulness; 0 means 'not rated'"),
            ("aoa", "100-700 rating", (100.0, 700.0),
             "age-of-acquisition rating on the MRC's own scale -- independent of, and on a "
             "different scale from, Kuperman's AoA; 0 means 'not rated'")),
        download_url="https://raw.githubusercontent.com/samzhang111/mrc-psycholinguistics/master/mrc2.dct",
        filename="mrc2.dct", format="mrc_fixed_width", parser="_parse_mrc"),
    "english_lexicon_project": ResourceSpec(
        name="english_lexicon_project", title="English Lexicon Project (lexical decision norms)",
        version="balota-2007", homepage="http://elexicon.wustl.edu",
        citation="Balota, D.A., et al. (2007). The English Lexicon Project. Behavior Research "
                 "Methods, 39, 445-459.",
        license="Unknown/unreachable -- see notes.",
        dimensions=_dims(
            ("lexical_decision_rt", "ms", (300.0, 1200.0), "mean lexical-decision reaction time"),
            ("lexical_decision_accuracy", "proportion", (0.0, 1.0), "lexical-decision accuracy")),
        download_url=None, filename="", format="", parser="",
        notes="elexicon.wustl.edu now serves an unrelated open-source room-booking "
              "application (LibreBooking; its own README.md opens 'Welcome to LibreBooking "
              "... This is a community effort to keep the OpenSource GPLv3 LibreBooking "
              "alive'), not the ELP data or even a query form -- the project's original "
              "bulk-download page is gone, not merely registration-walled. No alternative "
              "non-web-query mirror of the RT/accuracy data was found. Set "
              "metrics.lexical_norms_suite.resource_paths.english_lexicon_project to a "
              "locally supplied export to use this resource anyway."),
    "celex": ResourceSpec(
        name="celex", title="CELEX lexical database", version="celex-2",
        homepage="https://catalog.ldc.upenn.edu/LDC96L14",
        citation="Baayen, R.H., Piepenbrock, R., & Gulikers, L. (1995). The CELEX Lexical "
                 "Database (release 2). LDC96L14/LDC96L05.",
        license="LDC paid licence; no free redistribution path.",
        dimensions=_dims(
            ("morphological_family_size", "count", (0.0, 200.0),
             "number of derived/compound forms sharing the word's root")),
        download_url=None, filename="", format="", parser="",
        notes="CELEX is distributed by the Linguistic Data Consortium under a paid licence "
              "(LDC96L14 for English); this project holds no such licence and none of "
              "CELEX's data is redistributed by any free mirror this task found. Set "
              "metrics.lexical_norms_suite.resource_paths.celex to a locally licensed copy "
              "to use this resource anyway."),
}


def list_resources() -> list[str]:
    return sorted(RESOURCES)


def resource_info(name: str) -> ResourceSpec | None:
    return RESOURCES.get(name)


def resource_available(name: str) -> bool:
    spec = RESOURCES.get(name)
    return spec is not None and spec.download_url is not None


# ----------------------------------------------------------------------- fetch

@dataclass
class DownloadResult:
    ok: bool
    message: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_get(url: str, retries: int = 4) -> bytes:
    """GET with a few backed-off retries on 429/503.

    OSF (several resources here are mirrored through it) rate-limits bursts
    of anonymous downloads with a plain 429; a fixed small number of retries
    with growing delays clears it in practice without turning a transient
    rate limit into a permanent "unavailable".
    """

    import time

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    last_error: urllib.error.HTTPError | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 503) or attempt == retries:
                raise
            time.sleep(2.0 * (attempt + 1))
    raise last_error  # pragma: no cover - unreachable, loop always returns/raises


def download(name: str, force: bool = False) -> DownloadResult:
    """Fetch and cache one resource, or every fetchable one with ``name="all"``."""

    if name == "all":
        results = {item: download(item, force=force) for item in list_resources()
                   if resource_available(item)}
        ok = all(item.ok for item in results.values())
        lines = [f"{key}: {'ok' if item.ok else 'FAILED - ' + item.message}"
                for key, item in results.items()]
        return DownloadResult(ok=ok, message="\n".join(lines),
                              metadata={key: item.metadata for key, item in results.items()})

    spec = RESOURCES.get(name)
    if spec is None:
        return DownloadResult(False, f"unknown resource {name!r}; known: {', '.join(list_resources())}")
    if spec.download_url is None:
        return DownloadResult(False, f"{name} has no automatic download path: {spec.notes}")

    dest = cache_dir(name) / spec.filename
    if dest.exists() and not force:
        return DownloadResult(True, f"already cached at {dest}", dest, _read_meta(name) or {})

    try:
        payload = _http_get(spec.download_url)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(500).decode("utf-8", "replace")
        except Exception:  # pragma: no cover - best-effort error detail
            pass
        return DownloadResult(False, f"HTTP {exc.code} fetching {spec.download_url}: {exc.reason}"
                                     f"{' -- ' + body if body else ''}")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return DownloadResult(False, f"could not fetch {spec.download_url}: {type(exc).__name__}: {exc}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    sha256 = _sha256_of(dest)

    extracted_path = dest
    if spec.format.startswith("zip:"):
        member = spec.format.split(":", 1)[1]
        try:
            with zipfile.ZipFile(dest) as archive:
                data = archive.read(member)
        except (KeyError, zipfile.BadZipFile) as exc:
            return DownloadResult(False, f"downloaded {dest} but could not extract {member!r}: {exc}")
        extracted_path = cache_dir(name) / Path(member).name
        extracted_path.write_bytes(data)

    table, reason = _PARSERS[spec.parser](extracted_path if spec.format.startswith("zip:") else dest)
    entry_count = len(table) if table is not None else 0
    meta = {
        "resource": name, "version": spec.version, "source_url": spec.download_url,
        "sha256": sha256, "cached_file": str(dest), "extracted_file": str(extracted_path),
        "entry_count": entry_count, "dimensions": [d.key for d in spec.dimensions],
    }
    _write_meta(name, meta)
    if table is None:
        return DownloadResult(False, f"downloaded {dest} but could not parse it: {reason}", dest, meta)
    return DownloadResult(True, f"cached {entry_count} entries at {dest}", dest, meta)


def _meta_path(name: str) -> Path:
    return cache_dir(name) / "metadata.json"


def _write_meta(name: str, meta: Mapping[str, Any]) -> None:
    try:
        _meta_path(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:  # pragma: no cover - best-effort provenance record
        pass


def _read_meta(name: str) -> dict[str, Any] | None:
    try:
        return json.loads(_meta_path(name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------- parsers
#
# Every parser returns ``(table, reason)`` where ``table`` maps a lowercased,
# single-token word to ``{dimension_key: float}``. Multi-word entries (e.g.
# concreteness's "roller coaster") are dropped: this module looks words up
# one canonical token at a time, the same unit every other TextGrader metric
# counts in, and a bigram can never match one.

def _clean_word(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    word = raw.strip().lower()
    if not word or " " in word or "\t" in word:
        return None
    return word


def _parse_xlsx_table(path: Path, word_col: str, dim_cols: Mapping[str, str],
                      sheet_index: int = 0) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    module, reason = require("openpyxl")
    if module is None:
        return None, reason
    try:
        workbook = module.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.worksheets[sheet_index]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows)
    except Exception as exc:  # pragma: no cover - malformed/truncated download
        return None, f"could not read {path}: {type(exc).__name__}: {exc}"
    index = {str(name).strip(): i for i, name in enumerate(header) if name is not None}
    if word_col not in index:
        return None, f"{path} has no {word_col!r} column (found {sorted(index)})"
    word_i = index[word_col]
    dim_i = {dim: index[col] for dim, col in dim_cols.items() if col in index}
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        word = _clean_word(row[word_i]) if word_i < len(row) else None
        if word is None:
            continue
        entry: dict[str, float] = {}
        for dim, i in dim_i.items():
            value = row[i] if i < len(row) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                entry[dim] = float(value)
        if entry:
            table[word] = entry
    return table, None


def _parse_brysbaert_concreteness(path: Path):
    return _parse_xlsx_table(path, "Word", {"concreteness": "Conc.M"})


def _parse_kuperman_aoa(path: Path):
    table, reason = _parse_xlsx_table(
        path, "Word", {"aoa": "AoA_Kup"})
    if table is None:
        return None, reason
    lemma_table, lemma_reason = _parse_xlsx_table(
        path, "Lemma_highest_PoS", {"aoa_lemma": "AoA_Kup_lem"})
    if lemma_table:
        for word, entry in lemma_table.items():
            table.setdefault(word, {}).update(entry)
    return table, None


def _parse_glasgow_norms(path: Path):
    module, reason = require("openpyxl")
    if module is None:
        return None, reason
    dims = ("arousal", "valence", "dominance", "concreteness", "imageability",
           "familiarity", "age_of_acquisition", "semantic_size", "gender_association")
    try:
        workbook = module.load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        next(rows)  # dimension-group header row (AROU, VAL, ... merged cells)
        next(rows)  # "word, length, M, SD, N, M, SD, N, ..." header row
    except Exception as exc:  # pragma: no cover - malformed/truncated download
        return None, f"could not read {path}: {type(exc).__name__}: {exc}"
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        word = _clean_word(row[0]) if row else None
        if word is None:
            continue
        entry: dict[str, float] = {}
        for index, dim in enumerate(dims):
            mean_col = 2 + 3 * index
            if mean_col < len(row) and isinstance(row[mean_col], (int, float)):
                entry[dim] = float(row[mean_col])
        if entry:
            table[word] = entry
    return table, None


def _parse_warriner_vad(path: Path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            table: dict[str, dict[str, float]] = {}
            for row in reader:
                word = _clean_word(row.get("Word"))
                if word is None:
                    continue
                entry: dict[str, float] = {}
                for dim, col in (("valence", "V.Mean.Sum"), ("arousal", "A.Mean.Sum"),
                                 ("dominance", "D.Mean.Sum")):
                    raw = row.get(col)
                    if raw not in (None, ""):
                        try:
                            entry[dim] = float(raw)
                        except ValueError:
                            pass
                if entry:
                    table[word] = entry
    except OSError as exc:
        return None, f"could not read {path}: {exc}"
    return table, None


def _parse_nrc_vad(path: Path):
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            table: dict[str, dict[str, float]] = {}
            for row in reader:
                word = _clean_word(row.get("term"))
                if word is None:
                    continue
                entry = {}
                for dim in ("valence", "arousal", "dominance"):
                    raw = row.get(dim)
                    if raw not in (None, ""):
                        try:
                            entry[dim] = float(raw)
                        except ValueError:
                            pass
                if entry:
                    table[word] = entry
    except OSError as exc:
        return None, f"could not read {path}: {exc}"
    return table, None


def _parse_lancaster_sensorimotor(path: Path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            table: dict[str, dict[str, float]] = {}
            for row in reader:
                word = _clean_word(row.get("Word"))
                if word is None:
                    continue
                entry: dict[str, float] = {}
                for dim, col in (("perceptual_strength", "Max_strength.perceptual"),
                                 ("action_strength", "Max_strength.action")):
                    raw = row.get(col)
                    if raw not in (None, ""):
                        try:
                            entry[dim] = float(raw)
                        except ValueError:
                            pass
                if entry:
                    table[word] = entry
    except OSError as exc:
        return None, f"could not read {path}: {exc}"
    return table, None


#: The eleven raw modality columns, kept out of ``_parse_lancaster_sensorimotor``'s
#: two headline dimensions (see that function) to avoid a 22-metric-id
#: explosion; ``lexical_norms_suite`` reads this separately for one bounded
#: "modality profile" finding (rule 8: cap output volume).
LANCASTER_MODALITY_COLUMNS = {
    "auditory": "Auditory.mean", "gustatory": "Gustatory.mean", "haptic": "Haptic.mean",
    "interoceptive": "Interoceptive.mean", "olfactory": "Olfactory.mean",
    "visual": "Visual.mean", "foot_leg": "Foot_leg.mean", "hand_arm": "Hand_arm.mean",
    "head": "Head.mean", "mouth": "Mouth.mean", "torso": "Torso.mean",
}


def lancaster_modality_table(path: Path) -> tuple[dict[str, dict[str, float]] | None, str | None]:
    """The eleven raw Lancaster modality means, word -> {modality: value}."""

    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            table: dict[str, dict[str, float]] = {}
            for row in reader:
                word = _clean_word(row.get("Word"))
                if word is None:
                    continue
                entry = {}
                for modality, col in LANCASTER_MODALITY_COLUMNS.items():
                    raw = row.get(col)
                    if raw not in (None, ""):
                        try:
                            entry[modality] = float(raw)
                        except ValueError:
                            pass
                if entry:
                    table[word] = entry
    except OSError as exc:
        return None, f"could not read {path}: {exc}"
    return table, None


def _zipf_from_per_million(freq_per_million: float) -> float | None:
    """Van Heuven et al. (2014)'s Zipf transform: log10(per-million rate) + 3.

    E.g. SUBTLEX-US's SUBTLWF for "the" is ~29,449 per million, giving Zipf
    ``log10(29449) + 3 ~= 7.47`` -- in the same ballpark as wordfreq's own
    Zipf for "the" (~7.7), which is the point of keeping this as an
    independent cross-check on the same scale rather than a duplicate.
    """

    if freq_per_million <= 0:
        return None
    import math
    return math.log10(freq_per_million) + 3.0


def _parse_subtlex_us(path: Path):
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            table: dict[str, dict[str, float]] = {}
            for row in reader:
                word = _clean_word(row.get("Word"))
                if word is None:
                    continue
                entry: dict[str, float] = {}
                wf = row.get("SUBTLWF")
                if wf not in (None, ""):
                    try:
                        zipf = _zipf_from_per_million(float(wf))
                        if zipf is not None:
                            entry["zipf"] = zipf
                    except ValueError:
                        pass
                cd = row.get("SUBTLCD")
                if cd not in (None, ""):
                    try:
                        entry["contextual_diversity"] = float(cd)
                    except ValueError:
                        pass
                if entry:
                    table[word] = entry
    except OSError as exc:
        return None, f"could not read {path}: {exc}"
    return table, None


#: Byte offsets of the numeric header fields in mrc2.dct's fixed-width
#: format, verified against a real MRC2 line -- e.g. "HAPPY" (JJ, common
#: adjective sense) parses to fam=621, conc=355, imag=511, meanc=568,
#: matching the database's documented 100-700 rating range -- and against
#: samzhang111/mrc-psycholinguistics's independent SQLAlchemy port of the
#: same format (github.com/samzhang111/mrc-psycholinguistics/blob/master/
#: extract.py), which uses the identical slice boundaries.
_MRC_FIELDS = {
    "nlet": slice(0, 2), "nphon": slice(2, 4), "nsyl": slice(4, 5),
    "kf_freq": slice(5, 10), "kf_ncats": slice(10, 12), "kf_nsamp": slice(12, 15),
    "tl_freq": slice(15, 21), "brown_freq": slice(21, 25),
    "familiarity": slice(25, 28), "concreteness": slice(28, 31), "imageability": slice(31, 34),
    "meaningfulness_colorado": slice(34, 37), "meaningfulness_paivio": slice(37, 40),
    "aoa": slice(40, 43),
}
_MRC_RATING_DIMENSIONS = ("familiarity", "concreteness", "imageability",
                          "meaningfulness_colorado", "meaningfulness_paivio", "aoa")
_MRC_HEADER_LEN = 51


def _parse_mrc(path: Path):
    table: dict[str, dict[str, float]] = {}
    try:
        with open(path, encoding="latin-1") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if len(line) < _MRC_HEADER_LEN:
                    continue
                tail = line[_MRC_HEADER_LEN:]
                parts = tail.split("|")
                if not parts or not parts[0]:
                    continue
                # '&' stands in for an apostrophe at the start of a headword
                # ('ARRY, 'EM, ...); nothing else in the format needs
                # unescaping. Only the primary written word form is used.
                word = _clean_word(parts[0].replace("&", "'"))
                if word is None:
                    continue
                entry: dict[str, float] = {}
                for dim in _MRC_RATING_DIMENSIONS:
                    raw = line[_MRC_FIELDS[dim]]
                    try:
                        value = int(raw)
                    except ValueError:
                        continue
                    if value > 0:  # 0 means "not rated" in this dictionary
                        entry[dim] = float(value)
                if entry:
                    # A handful of headwords repeat across parts of speech
                    # (e.g. "HAPPY" as both V and JJ); keep the entry with
                    # more ratings filled in rather than the alphabetically
                    # first one.
                    existing = table.get(word)
                    if existing is None or len(entry) > len(existing):
                        table[word] = entry
    except OSError as exc:
        return None, f"could not read {path}: {exc}"
    return table, None


_PARSERS: dict[str, Callable[[Path], tuple[dict[str, dict[str, float]] | None, str | None]]] = {
    "_parse_brysbaert_concreteness": _parse_brysbaert_concreteness,
    "_parse_kuperman_aoa": _parse_kuperman_aoa,
    "_parse_glasgow_norms": _parse_glasgow_norms,
    "_parse_warriner_vad": _parse_warriner_vad,
    "_parse_nrc_vad": _parse_nrc_vad,
    "_parse_lancaster_sensorimotor": _parse_lancaster_sensorimotor,
    "_parse_subtlex_us": _parse_subtlex_us,
    "_parse_mrc": _parse_mrc,
}


# ------------------------------------------------------------------- resolution

def _resolve_path(name: str, override: str | Path | None) -> Path | None:
    if override:
        path = Path(override).expanduser()
        return path if path.exists() else None
    spec = RESOURCES.get(name)
    if spec is None or spec.download_url is None:
        return None
    if spec.format.startswith("zip:"):
        member = spec.format.split(":", 1)[1]
        path = cache_dir(name) / Path(member).name
    else:
        path = cache_dir(name) / spec.filename
    return path if path.exists() else None


#: In-process cache: (name, resolved-path-string, mtime) -> (table, metadata).
#: Never per-metric, never per-sentence -- see the module docstring's caching
#: rule and textgrader.metrics.common's identical rule for shared parses.
_TABLE_CACHE: dict[tuple[str, str, float], tuple[dict[str, dict[str, float]], dict[str, Any]]] = {}


def _base_metadata(name: str, path: Path | None) -> dict[str, Any]:
    spec = RESOURCES.get(name)
    meta: dict[str, Any] = {
        "resource": name, "version": spec.version if spec else None,
        "source_url": spec.download_url if spec else None,
        "homepage": spec.homepage if spec else None,
        "license": spec.license if spec else None,
        "sha256": None, "entry_count": 0, "cached_file": str(path) if path else None,
    }
    if path is not None and path.exists():
        cached = _read_meta(name)
        if cached and cached.get("cached_file") == str(path):
            meta["sha256"] = cached.get("sha256")
        else:
            meta["sha256"] = _sha256_of(path)
    return meta


def norm_table(name: str, path: str | Path | None = None
              ) -> tuple[dict[str, dict[str, float]] | None, str | None, dict[str, Any]]:
    """Load (and cache) one resource's per-word dimension table.

    Returns ``(table, reason, metadata)``. ``table`` is ``None`` when the
    resource is unavailable; ``reason`` then explains exactly why (a quoted
    download/parse error, a missing optional package, or -- for
    ``english_lexicon_project``/``celex`` -- the licensing note recorded on
    the :class:`ResourceSpec`). ``metadata`` is always populated.
    """

    spec = RESOURCES.get(name)
    if spec is None:
        return None, f"unknown lexicon resource {name!r}", _base_metadata(name, None)

    resolved = _resolve_path(name, path)
    if resolved is None:
        if path:
            reason = f"configured path {path!r} for resource {name!r} does not exist"
        elif spec.download_url is None:
            reason = f"{name} is not automatically downloadable: {spec.notes}"
        else:
            reason = (f"{name} is not cached; run "
                     f"`python -m textgrader.lexicons download {name}` "
                     f"(source: {spec.download_url})")
        return None, reason, _base_metadata(name, None)

    mtime = resolved.stat().st_mtime
    cache_key = (name, str(resolved), mtime)
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached[0], None, dict(cached[1])

    parser = _PARSERS.get(spec.parser)
    if parser is None:  # pragma: no cover - a resource registered without a parser
        return None, f"{name} has no registered parser ({spec.parser!r})", _base_metadata(name, resolved)
    table, reason = parser(resolved)
    metadata = _base_metadata(name, resolved)
    if table is None:
        return None, reason, metadata
    metadata["entry_count"] = len(table)
    metadata["dimension_reference"] = {
        dim.key: _reference_quantiles([entry[dim.key] for entry in table.values() if dim.key in entry])
        for dim in spec.dimensions
    }
    _TABLE_CACHE[cache_key] = (table, metadata)
    return table, None, dict(metadata)


def _reference_quantiles(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    summary = summarize(values, shape=False)
    return {key: summary.get(key) for key in
           ("count", "mean", "std", "min", "max", "p10", "p25", "median", "p75", "p90")}


def reference_quantiles(name: str, dimension: str, path: str | Path | None = None
                        ) -> dict[str, Any] | None:
    """The norm table's own value distribution for one dimension.

    Used as the "norm-specific reference quantile" a document's own low/high
    tail share is measured against (task spec, "Metrics to add"): the
    threshold comes from how the *lexicon* is spread, not an invented
    per-metric cutoff.
    """

    _, _, metadata = norm_table(name, path)
    return (metadata.get("dimension_reference") or {}).get(dimension)


def score_tokens(name: str, tokens: Sequence[str], dimension: str | None = None,
                 path: str | Path | None = None) -> list[float | None]:
    """One score per token (aligned positionally), or all ``None`` if unavailable.

    ``tokens`` are looked up exactly as given (already lower-cased the way
    :attr:`DocumentAnalysis.tokens` lower-cases and apostrophe-folds them);
    this function does no further normalization. Intended for a sibling
    module (e.g. sentence-level VAD scoring) that wants one resource's
    numbers without touching parsing or caching itself.
    """

    table, reason, metadata = norm_table(name, path)
    if table is None:
        return [None] * len(tokens)
    if dimension is None:
        dims = metadata.get("dimensions") or (
            [d.key for d in RESOURCES[name].dimensions] if name in RESOURCES else [])
        if len(dims) != 1:
            raise ValueError(
                f"resource {name!r} has {len(dims)} dimensions ({dims}); pass dimension=...")
        dimension = dims[0]
    return [table.get(token, {}).get(dimension) for token in tokens]


# ------------------------------------------------------------------------- CLI

def _cli(argv: Sequence[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("download", "list", "status"):
        print(__doc__.splitlines()[0])
        print("usage: python -m textgrader.lexicons download <name|all>")
        print("       python -m textgrader.lexicons list")
        print("       python -m textgrader.lexicons status")
        return 1
    if argv[1] == "list":
        for name in list_resources():
            spec = RESOURCES[name]
            flag = "downloadable" if spec.download_url else "NOT AVAILABLE (see notes)"
            print(f"{name:28s} {spec.version:20s} {flag}")
        return 0
    if argv[1] == "status":
        for name in list_resources():
            table, reason, meta = norm_table(name)
            if table is not None:
                print(f"{name:28s} cached, {meta['entry_count']} entries, sha256={meta['sha256']}")
            else:
                print(f"{name:28s} unavailable: {reason}")
        return 0
    if len(argv) < 3:
        print("usage: python -m textgrader.lexicons download <name|all>")
        return 1
    result = download(argv[2], force="--force" in argv[3:])
    print(result.message)
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    raise SystemExit(_cli(sys.argv))
