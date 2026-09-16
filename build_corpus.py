#!/usr/bin/env python3
"""Build a local fiction reference corpus from Project Gutenberg.

TextGrader corpus builder
=========================

This script intentionally separates *acquisition* from *analysis*. It downloads
plain-text fiction into a gitignored ``corpus/books/`` directory and writes an
auditable manifest containing the metadata and lightweight text classifications
used to select each book.

Project Gutenberg's weekly CSV catalog is used for discovery. Optional original
publication-year enrichment uses Open Library's low-volume Search API and is
cached locally. POV and tense are inferred from the downloaded narration using
simple, documented heuristics; low-confidence classifications can be rejected.

No third-party Python packages are required.

Examples:
    python build_corpus.py --genre science_fiction --pov third --max-year 1989 --count 40
    python build_corpus.py --config corpus_scifi_third.json
    python build_corpus.py --list-genres

The tool exits non-zero when it cannot satisfy ``minimum_count`` (default 30),
even if it managed to download some books. That is deliberate: a tiny reference
set should not silently masquerade as a useful corpus.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import difflib
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

VERSION = "1.0.0"

PG_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"

# Deliberately broad aliases. Project Gutenberg metadata is inconsistent and
# largely based on Library of Congress subject headings/bookshelves, so matching
# several historically common terms is more useful than pretending there is a
# clean genre taxonomy.
GENRE_TERMS: dict[str, tuple[str, ...]] = {
    "science_fiction": (
        "science fiction",
        "science-fiction",
        "interplanetary voyages",
        "space warfare",
        "life on other planets",
        "robots -- fiction",
        "time travel -- fiction",
        "future life -- fiction",
    ),
    "fantasy": (
        "fantasy fiction",
        "fantasy",
        "fairy tales",
        "magic -- fiction",
        "dragons -- fiction",
        "imaginary places -- fiction",
        "mythological fiction",
    ),
    "mystery": (
        "detective and mystery stories",
        "detective fiction",
        "mystery fiction",
        "crime -- fiction",
        "murder -- investigation -- fiction",
    ),
    "horror": (
        "horror tales",
        "horror fiction",
        "ghost stories",
        "supernatural -- fiction",
        "occult fiction",
    ),
    "romance": (
        "love stories",
        "romance fiction",
        "courtship -- fiction",
        "man-woman relationships -- fiction",
    ),
    "adventure": (
        "adventure stories",
        "adventure fiction",
        "sea stories",
        "survival -- fiction",
        "voyages and travels -- fiction",
    ),
    "historical": (
        "historical fiction",
        "history -- fiction",
    ),
    "western": (
        "western stories",
        "western fiction",
        "west (u.s.) -- fiction",
        "frontier and pioneer life -- fiction",
    ),
    "children": (
        "juvenile fiction",
        "children's stories",
        "children -- fiction",
        "young adult fiction",
    ),
    "gothic": (
        "gothic fiction",
        "gothic romance",
        "castles -- fiction",
    ),
    "nautical": (
        "sea stories",
        "seafaring life -- fiction",
        "sailors -- fiction",
        "ships -- fiction",
    ),
    "war": (
        "war stories",
        "war -- fiction",
        "soldiers -- fiction",
    ),
    "literary": (
        "psychological fiction",
        "domestic fiction",
        "bildungsromans",
        "social life and customs -- fiction",
    ),
}

# These are not genre exclusions; they eliminate obvious non-novel material or
# collections that distort book-level prose statistics. Users can disable or
# replace them in config.
DEFAULT_EXCLUDE_TERMS = (
    "poetry",
    "poems",
    "drama",
    "plays",
    "essays",
    "speeches",
    "correspondence",
    "letters",
    "bibliography",
    "periodicals",
    "short stories",
    "short story",
    "collection",
    "anthologies",
    "anthology",
)

FICTION_HINTS = (
    "fiction",
    "novel",
    "stories",
    "tale",
    "romance",
    "adventure",
    "mystery",
    "fantasy",
    "horror",
    "detective",
    "science fiction",
    "western",
)

FIRST_PERSON = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"
}
THIRD_PERSON = {
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
}

PAST_MARKERS = {
    "was", "were", "had", "did", "said", "went", "came", "saw", "thought",
    "knew", "looked", "seemed", "began", "found", "felt", "took", "made",
    "told", "asked", "answered", "stood", "sat", "gave", "left", "heard",
}
PRESENT_MARKERS = {
    "is", "are", "has", "does", "says", "goes", "comes", "sees", "thinks",
    "knows", "looks", "seems", "begins", "finds", "feels", "takes", "makes",
    "tells", "asks", "answers", "stands", "sits", "gives", "leaves", "hears",
}

WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)


@dataclasses.dataclass
class Settings:
    output_dir: str = "corpus"
    count: int = 40
    minimum_count: int = 30
    language: str = "en"
    genres: list[str] = dataclasses.field(default_factory=list)
    include_terms: list[str] = dataclasses.field(default_factory=list)
    exclude_terms: list[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_EXCLUDE_TERMS))
    pov: str = "any"  # any, first, third
    min_pov_confidence: float = 0.35
    tense: str = "any"  # any, past, present
    min_tense_confidence: float = 0.25
    min_year: int | None = None
    max_year: int | None = 1989
    year_policy: str = "best_effort"  # ignore, best_effort, strict
    min_words: int = 30000
    max_words: int = 250000
    max_books_per_author: int = 2
    max_candidates: int = 800
    seed: int = 20260916
    request_delay: float = 2.0
    timeout: float = 30.0
    catalog_url: str = PG_CATALOG_URL
    refresh_catalog: bool = False
    openlibrary_email: str | None = None
    year_lookup_limit: int = 100
    keep_failed_downloads: bool = False


@dataclasses.dataclass
class Candidate:
    gutenberg_id: int
    title: str
    authors: str
    language: str
    subjects: str
    bookshelves: str
    locc: str
    issued: str
    genre_terms: list[str]
    genre_score: int


class CorpusError(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def canonical_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.casefold().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def safe_slug(value: str, limit: int = 72) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", canonical_text(value)).strip("-")
    return (slug[:limit].rstrip("-") or "book")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        raise CorpusError(f"Invalid JSON in {path}: {exc}") from exc


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    os.replace(temp, path)


def make_opener(email: str | None = None) -> urllib.request.OpenerDirector:
    contact = email.strip() if email else None
    ua = f"TextGrader-CorpusBuilder/{VERSION}"
    if contact:
        ua += f" ({contact})"
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("User-Agent", ua),
        ("Accept-Encoding", "identity"),
    ]
    return opener


def fetch_bytes(
    opener: urllib.request.OpenerDirector,
    url: str,
    *,
    timeout: float,
    attempts: int = 3,
) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            with opener.open(url, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 4))
    raise CorpusError(f"Failed to download {url}: {last_exc}")


def ensure_catalog(settings: Settings, cache_dir: Path, opener: urllib.request.OpenerDirector) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "pg_catalog.csv.gz"
    if target.exists() and not settings.refresh_catalog:
        return target

    print(f"Downloading Project Gutenberg catalog: {settings.catalog_url}")
    data = fetch_bytes(opener, settings.catalog_url, timeout=settings.timeout)
    # Validate before replacing a good cache.
    try:
        decompressed = gzip.decompress(data)
    except OSError as exc:
        raise CorpusError("Downloaded Gutenberg catalog was not valid gzip data") from exc
    if b"Text#" not in decompressed[:500]:
        raise CorpusError("Downloaded Gutenberg catalog did not look like pg_catalog.csv")
    temp = target.with_suffix(".tmp")
    temp.write_bytes(data)
    os.replace(temp, target)
    return target


def row_value(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return str(row[name]).strip()
    return ""


def catalog_rows(path: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "Text#" not in reader.fieldnames:
            raise CorpusError(f"Unsupported Gutenberg CSV columns in {path}: {reader.fieldnames}")
        yield from reader


def combined_metadata(row: Mapping[str, str]) -> str:
    return " | ".join([
        row_value(row, "Title"),
        row_value(row, "Subjects"),
        row_value(row, "Bookshelves"),
    ]).casefold()


def match_terms(haystack: str, terms: Iterable[str]) -> list[str]:
    found = []
    for term in terms:
        if term.casefold() in haystack:
            found.append(term)
    return found


def looks_like_fiction(meta: str) -> bool:
    return any(term in meta for term in FICTION_HINTS)


def primary_author(authors: str) -> str:
    if not authors:
        return "unknown"
    # PG normally separates multiple authors with semicolons. Keep surname + first
    # names together; only use this for balancing, not bibliographic identity.
    return canonical_text(authors.split(";")[0]) or "unknown"


def candidate_from_row(row: Mapping[str, str], settings: Settings) -> Candidate | None:
    try:
        gid = int(row_value(row, "Text#"))
    except ValueError:
        return None

    type_ = row_value(row, "Type").casefold()
    if type_ and type_ != "text":
        return None

    language = row_value(row, "Language")
    langs = {part.strip() for part in re.split(r"[;, ]+", language) if part.strip()}
    if settings.language and settings.language not in langs:
        return None

    meta = combined_metadata(row)
    if not looks_like_fiction(meta):
        return None

    if any(term.casefold() in meta for term in settings.exclude_terms):
        return None
    if settings.include_terms and not all(term.casefold() in meta for term in settings.include_terms):
        return None

    genre_hits: list[str] = []
    for genre in settings.genres:
        terms = GENRE_TERMS.get(genre)
        if terms is None:
            raise CorpusError(f"Unknown genre {genre!r}. Use --list-genres.")
        hits = match_terms(meta, terms)
        if not hits:
            return None  # genres are ANDed when multiple are requested
        genre_hits.extend(hits)

    title = row_value(row, "Title")
    authors = row_value(row, "Authors")
    if not title or not authors:
        return None

    # Higher score = more explicit genre metadata match.
    score = len(set(genre_hits)) * 10
    if "fiction" in row_value(row, "Subjects").casefold():
        score += 2
    if any(genre.replace("_", " ") in row_value(row, "Bookshelves").casefold() for genre in settings.genres):
        score += 4

    return Candidate(
        gutenberg_id=gid,
        title=title,
        authors=authors,
        language=language,
        subjects=row_value(row, "Subjects"),
        bookshelves=row_value(row, "Bookshelves"),
        locc=row_value(row, "LoCC"),
        issued=row_value(row, "Issued"),
        genre_terms=sorted(set(genre_hits)),
        genre_score=score,
    )


def discover_candidates(path: Path, settings: Settings) -> list[Candidate]:
    candidates = []
    for row in catalog_rows(path):
        cand = candidate_from_row(row, settings)
        if cand:
            candidates.append(cand)

    rng = random.Random(settings.seed)
    # Shuffle before stable sort so equally ranked books aren't always the oldest
    # Gutenberg IDs or alphabetically clustered authors.
    rng.shuffle(candidates)
    candidates.sort(key=lambda c: c.genre_score, reverse=True)
    return candidates[: settings.max_candidates]


def normalize_title_for_match(title: str) -> str:
    title = re.split(r"\s*[:;]\s*", title, maxsplit=1)[0]
    title = re.sub(r"\b(volume|vol\.?|part)\s+[ivxlcdm0-9]+\b.*$", "", title, flags=re.I)
    return canonical_text(title)


def author_tokens(authors: str) -> set[str]:
    # PG format is often "Surname, Given, YYYY-YYYY". Years are useless for
    # matching; surname tokens carry the most signal.
    cleaned = re.sub(r"\b\d{3,4}(?:-\d{0,4})?\b", " ", authors)
    toks = [t for t in canonical_text(cleaned).split() if len(t) > 1]
    return set(toks)


def openlibrary_year(
    cand: Candidate,
    *,
    opener: urllib.request.OpenerDirector,
    cache: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    key = str(cand.gutenberg_id)
    if key in cache:
        return cache[key]

    query = {
        "title": cand.title,
        "author": re.sub(r"\b\d{3,4}(?:-\d{0,4})?\b", "", cand.authors.split(";")[0]).strip(" ,"),
        "fields": "key,title,author_name,first_publish_year",
        "limit": "5",
    }
    url = OPENLIBRARY_SEARCH_URL + "?" + urllib.parse.urlencode(query)
    try:
        payload = json.loads(fetch_bytes(opener, url, timeout=settings.timeout).decode("utf-8"))
    except (CorpusError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        result = {"year": None, "confidence": 0.0, "key": None, "error": str(exc)}
        cache[key] = result
        return result

    wanted_title = normalize_title_for_match(cand.title)
    wanted_authors = author_tokens(cand.authors)
    scored: list[tuple[float, dict[str, Any]]] = []
    for doc in payload.get("docs", []):
        year = doc.get("first_publish_year")
        if not isinstance(year, int):
            continue
        title_score = difflib.SequenceMatcher(
            None, wanted_title, normalize_title_for_match(str(doc.get("title", "")))
        ).ratio()
        doc_auth = canonical_text(" ".join(doc.get("author_name") or [])).split()
        author_overlap = 0.0
        if wanted_authors and doc_auth:
            author_overlap = len(wanted_authors.intersection(doc_auth)) / max(1, min(len(wanted_authors), len(set(doc_auth))))
        score = 0.72 * title_score + 0.28 * author_overlap
        scored.append((score, doc))

    if not scored:
        result = {"year": None, "confidence": 0.0, "key": None}
    else:
        score, doc = max(scored, key=lambda pair: pair[0])
        if score < 0.62:
            result = {"year": None, "confidence": round(score, 3), "key": doc.get("key")}
        else:
            result = {
                "year": int(doc["first_publish_year"]),
                "confidence": round(score, 3),
                "key": doc.get("key"),
            }
    cache[key] = result
    return result


def year_allowed(year: int | None, settings: Settings) -> bool:
    if settings.year_policy == "ignore" or (settings.min_year is None and settings.max_year is None):
        return True
    if year is None:
        return settings.year_policy == "best_effort"
    if settings.min_year is not None and year < settings.min_year:
        return False
    if settings.max_year is not None and year > settings.max_year:
        return False
    return True


def strip_gutenberg(text: str) -> str:
    start = START_RE.search(text)
    if start:
        text = text[start.end():]
    end = END_RE.search(text)
    if end:
        text = text[:end.start()]
    # A few common pre-marker remnants. Keep this intentionally conservative;
    # analysis should not delete legitimate front matter based on guesses.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"


def decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def gutenberg_text_urls(gid: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
    ]


def download_gutenberg_text(
    cand: Candidate,
    *,
    opener: urllib.request.OpenerDirector,
    settings: Settings,
) -> tuple[str, str]:
    errors = []
    for url in gutenberg_text_urls(cand.gutenberg_id):
        try:
            data = fetch_bytes(opener, url, timeout=settings.timeout, attempts=2)
            text = strip_gutenberg(decode_text(data))
            # HTML error pages and tiny metadata stubs should not pass as books.
            if len(text) < 5000 or text.lstrip().lower().startswith("<!doctype html"):
                errors.append(f"{url}: response too small/not text")
                continue
            return text, url
        except CorpusError as exc:
            errors.append(str(exc))
    raise CorpusError("; ".join(errors[-2:]) or "no usable text URL")


def narration_without_quotes(text: str) -> str:
    """Remove text inside straight/curly double quotes with a state machine.

    This is intentionally not a full literary-dialogue parser. It is used only
    to prevent dialogue pronouns/verb tense from dominating a coarse POV/tense
    classification. Paragraph-spanning quotation conventions can still confuse
    it, which is why every classification carries a confidence score.
    """
    out: list[str] = []
    in_quote = False
    quote_char: str | None = None
    for ch in text:
        if ch == "“":
            in_quote = True
            quote_char = "”"
            out.append(" ")
        elif ch == "”" and in_quote and quote_char == "”":
            in_quote = False
            quote_char = None
            out.append(" ")
        elif ch == '"':
            in_quote = not in_quote
            quote_char = '"' if in_quote else None
            out.append(" ")
        elif not in_quote:
            out.append(ch)
        elif ch == "\n":
            out.append("\n")
        else:
            out.append(" ")
    return "".join(out)


def classify_pov(text: str) -> dict[str, Any]:
    narration = narration_without_quotes(text)
    words = [w.casefold().replace("’", "'") for w in WORD_RE.findall(narration)]
    counts = Counter(words)
    first = sum(counts[w] for w in FIRST_PERSON)
    third = sum(counts[w] for w in THIRD_PERSON)
    total = first + third
    if total < 40:
        return {"label": "unknown", "confidence": 0.0, "first": first, "third": third, "sample": total}
    first_share = first / total
    # Confidence is separation from 50/50, damped for small samples.
    separation = abs(first_share - 0.5) * 2.0
    sample_factor = min(1.0, total / 400.0)
    confidence = separation * sample_factor
    if first_share >= 0.62:
        label = "first"
    elif first_share <= 0.38:
        label = "third"
    else:
        label = "mixed"
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "first_share": round(first_share, 4),
        "first": first,
        "third": third,
        "sample": total,
    }


def classify_tense(text: str) -> dict[str, Any]:
    narration = narration_without_quotes(text)
    words = [w.casefold().replace("’", "'") for w in WORD_RE.findall(narration)]
    counts = Counter(words)
    past = sum(counts[w] for w in PAST_MARKERS)
    present = sum(counts[w] for w in PRESENT_MARKERS)
    # Regular past-tense verbs add signal, but cap their contribution so prose
    # with lots of -ed adjectives cannot dominate the classifier.
    ed_count = sum(1 for w in words if len(w) > 4 and w.endswith("ed"))
    past += min(ed_count, max(50, past * 2))
    total = past + present
    if total < 60:
        return {"label": "unknown", "confidence": 0.0, "past": past, "present": present, "sample": total}
    past_share = past / total
    separation = abs(past_share - 0.5) * 2.0
    sample_factor = min(1.0, total / 600.0)
    confidence = separation * sample_factor
    if past_share >= 0.62:
        label = "past"
    elif past_share <= 0.38:
        label = "present"
    else:
        label = "mixed"
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "past_share": round(past_share, 4),
        "past": past,
        "present": present,
        "sample": total,
    }


def classification_allowed(result: Mapping[str, Any], wanted: str, minimum_confidence: float) -> bool:
    if wanted == "any":
        return True
    return result.get("label") == wanted and float(result.get("confidence", 0.0)) >= minimum_confidence


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def settings_from_dict(data: Mapping[str, Any]) -> Settings:
    valid = {field.name for field in dataclasses.fields(Settings)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise CorpusError(f"Unknown config keys: {', '.join(unknown)}")
    kwargs = dict(data)
    for list_key in ("genres", "include_terms", "exclude_terms"):
        if list_key in kwargs and isinstance(kwargs[list_key], str):
            kwargs[list_key] = [kwargs[list_key]]
    settings = Settings(**kwargs)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if settings.count < 1:
        raise CorpusError("count must be >= 1")
    if settings.minimum_count < 1 or settings.minimum_count > settings.count:
        raise CorpusError("minimum_count must be >= 1 and <= count")
    if settings.count < 30:
        print("WARNING: target count is below 30; this is usually too small for stable corpus distributions.", file=sys.stderr)
    if settings.pov not in {"any", "first", "third"}:
        raise CorpusError("pov must be any, first, or third")
    if settings.tense not in {"any", "past", "present"}:
        raise CorpusError("tense must be any, past, or present")
    if settings.year_policy not in {"ignore", "best_effort", "strict"}:
        raise CorpusError("year_policy must be ignore, best_effort, or strict")
    for genre in settings.genres:
        if genre not in GENRE_TERMS:
            raise CorpusError(f"Unknown genre {genre!r}. Use --list-genres.")
    if settings.min_words < 1000 or settings.max_words <= settings.min_words:
        raise CorpusError("word-count range is invalid")
    if settings.max_books_per_author < 1:
        raise CorpusError("max_books_per_author must be >= 1")
    if settings.year_lookup_limit < 0:
        raise CorpusError("year_lookup_limit must be >= 0")


def merge_cli(settings: Settings, args: argparse.Namespace) -> Settings:
    data = dataclasses.asdict(settings)
    overrides = {
        "output_dir": args.output,
        "count": args.count,
        "minimum_count": args.minimum_count,
        "genres": args.genre,
        "pov": args.pov,
        "tense": args.tense,
        "min_year": args.min_year,
        "max_year": args.max_year,
        "year_policy": args.year_policy,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_books_per_author": args.max_books_per_author,
        "seed": args.seed,
        "openlibrary_email": args.openlibrary_email,
        "refresh_catalog": True if args.refresh_catalog else None,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value
    merged = settings_from_dict(data)
    return merged


def build_manifest_header(settings: Settings, catalog_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "builder": {"name": "TextGrader Corpus Builder", "version": VERSION},
        "created_at": utc_now(),
        "selection": dataclasses.asdict(settings),
        "catalog": {
            "source": settings.catalog_url,
            "local_sha256": sha256_file(catalog_path),
        },
        "books": [],
        "rejections": {},
    }


def inc(counter: Counter[str], reason: str) -> None:
    counter[reason] += 1


def build_corpus(settings: Settings) -> tuple[dict[str, Any], int]:
    root = Path(settings.output_dir)
    books_dir = root / "books"
    cache_dir = root / ".cache"
    manifest_path = root / "manifest.json"
    year_cache_path = cache_dir / "openlibrary_years.json"
    books_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    opener = make_opener(settings.openlibrary_email)
    catalog_path = ensure_catalog(settings, cache_dir, opener)
    candidates = discover_candidates(catalog_path, settings)
    if not candidates:
        raise CorpusError("No Gutenberg catalog candidates matched the requested metadata filters.")

    print(f"Catalog candidates after metadata filters: {len(candidates)}")
    manifest = build_manifest_header(settings, catalog_path)
    rejections: Counter[str] = Counter()
    year_cache: dict[str, Any] = load_json(year_cache_path, {})
    author_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    year_lookups = 0

    # Reuse existing accepted files from a prior run when possible, but always
    # reclassify them so changed classifier code doesn't leave stale metadata.
    for idx, cand in enumerate(candidates, 1):
        if len(selected) >= settings.count:
            break
        author_key = primary_author(cand.authors)
        if author_counts[author_key] >= settings.max_books_per_author:
            inc(rejections, "author_cap")
            continue

        year_info = {"year": None, "confidence": 0.0, "key": None}
        needs_year = settings.year_policy != "ignore" and (settings.min_year is not None or settings.max_year is not None)
        if needs_year:
            cached = str(cand.gutenberg_id) in year_cache
            if not cached and year_lookups >= settings.year_lookup_limit:
                if settings.year_policy == "strict":
                    inc(rejections, "year_lookup_limit")
                    continue
            else:
                if not cached:
                    year_lookups += 1
                year_info = openlibrary_year(cand, opener=opener, cache=year_cache, settings=settings)
                save_json_atomic(year_cache_path, year_cache)
                if not cached:
                    # Open Library asks unidentified clients to stay at <=1 rps.
                    ol_delay = 0.4 if settings.openlibrary_email else 1.05
                    time.sleep(max(settings.request_delay / 4.0, ol_delay))

            if not year_allowed(year_info.get("year"), settings):
                inc(rejections, "publication_year")
                continue

        filename = f"pg{cand.gutenberg_id}-{safe_slug(cand.title)}.txt"
        path = books_dir / filename
        source_url: str | None = None
        try:
            if path.exists() and path.stat().st_size > 5000:
                text = path.read_text(encoding="utf-8", errors="replace")
                source_url = f"cached:pg{cand.gutenberg_id}"
            else:
                text, source_url = download_gutenberg_text(cand, opener=opener, settings=settings)
                path.write_text(text, encoding="utf-8", newline="\n")
                time.sleep(settings.request_delay)
        except (CorpusError, OSError) as exc:
            inc(rejections, "download_error")
            print(f"[{idx}/{len(candidates)}] reject PG{cand.gutenberg_id}: download failed: {exc}", file=sys.stderr)
            continue

        words = word_count(text)
        if words < settings.min_words:
            inc(rejections, "too_short")
            if not settings.keep_failed_downloads:
                path.unlink(missing_ok=True)
            continue
        if words > settings.max_words:
            inc(rejections, "too_long")
            if not settings.keep_failed_downloads:
                path.unlink(missing_ok=True)
            continue

        pov = classify_pov(text)
        if not classification_allowed(pov, settings.pov, settings.min_pov_confidence):
            inc(rejections, "pov")
            if not settings.keep_failed_downloads:
                path.unlink(missing_ok=True)
            continue

        tense = classify_tense(text)
        if not classification_allowed(tense, settings.tense, settings.min_tense_confidence):
            inc(rejections, "tense")
            if not settings.keep_failed_downloads:
                path.unlink(missing_ok=True)
            continue

        book = {
            "gutenberg_id": cand.gutenberg_id,
            "title": cand.title,
            "authors": cand.authors,
            "language": cand.language,
            "gutenberg_issued": cand.issued,
            "subjects": cand.subjects,
            "bookshelves": cand.bookshelves,
            "locc": cand.locc,
            "matched_genre_terms": cand.genre_terms,
            "genre_score": cand.genre_score,
            "first_publish_year": year_info.get("year"),
            "publication_year_source": "openlibrary" if year_info.get("year") is not None else None,
            "publication_year_match_confidence": year_info.get("confidence"),
            "openlibrary_key": year_info.get("key"),
            "inferred_pov": pov,
            "inferred_tense": tense,
            "word_count": words,
            "file": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "source_url": source_url,
            "retrieved_at": utc_now(),
        }
        selected.append(book)
        author_counts[author_key] += 1
        print(
            f"[{len(selected):>2}/{settings.count}] PG{cand.gutenberg_id} | "
            f"{words:,} words | {pov['label']} ({pov['confidence']:.2f}) | "
            f"{tense['label']} ({tense['confidence']:.2f}) | "
            f"{year_info.get('year') or '?'} | {cand.title}"
        )

        manifest["books"] = selected
        manifest["rejections"] = dict(rejections)
        manifest["updated_at"] = utc_now()
        save_json_atomic(manifest_path, manifest)

    manifest["books"] = selected
    manifest["rejections"] = dict(rejections)
    manifest["updated_at"] = utc_now()
    manifest["summary"] = {
        "requested": settings.count,
        "minimum_required": settings.minimum_count,
        "selected": len(selected),
        "unique_primary_authors": len(author_counts),
        "total_words": sum(int(b["word_count"]) for b in selected),
        "year_lookups_performed": year_lookups,
        "passed_minimum": len(selected) >= settings.minimum_count,
    }
    save_json_atomic(manifest_path, manifest)

    print("\nCorpus summary")
    print("--------------")
    print(f"Selected: {len(selected)} / {settings.count} target")
    print(f"Unique primary authors: {len(author_counts)}")
    print(f"Total words: {manifest['summary']['total_words']:,}")
    if rejections:
        print("Rejections: " + ", ".join(f"{k}={v}" for k, v in rejections.most_common()))
    print(f"Manifest: {manifest_path}")

    return manifest, 0 if len(selected) >= settings.minimum_count else 2


def write_example_config(path: Path) -> None:
    config = dataclasses.asdict(Settings())
    config.update({
        "genres": ["science_fiction"],
        "pov": "third",
        "tense": "past",
        "count": 40,
        "minimum_count": 30,
        "max_year": 1989,
        "year_policy": "best_effort",
        "max_books_per_author": 2,
    })
    save_json_atomic(path, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a 30+ book local fiction corpus for TextGrader from Project Gutenberg.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--write-example-config", type=Path, metavar="PATH", help="write an example JSON config and exit")
    parser.add_argument("--list-genres", action="store_true", help="list built-in genre names and exit")
    parser.add_argument("--output", help="corpus output directory")
    parser.add_argument("--count", type=int, help="target number of accepted books")
    parser.add_argument("--minimum-count", type=int, help="minimum acceptable corpus size; exits non-zero below this")
    parser.add_argument("--genre", action="append", choices=sorted(GENRE_TERMS), help="genre filter; repeat to require multiple genre matches")
    parser.add_argument("--pov", choices=["any", "first", "third"], help="inferred narration POV")
    parser.add_argument("--tense", choices=["any", "past", "present"], help="inferred narration tense")
    parser.add_argument("--min-year", type=int, help="minimum first-publication year")
    parser.add_argument("--max-year", type=int, help="maximum first-publication year")
    parser.add_argument("--year-policy", choices=["ignore", "best_effort", "strict"], help="handling of unknown publication years")
    parser.add_argument("--min-words", type=int, help="minimum stripped text word count")
    parser.add_argument("--max-words", type=int, help="maximum stripped text word count")
    parser.add_argument("--max-books-per-author", type=int, help="author balance cap")
    parser.add_argument("--seed", type=int, help="deterministic candidate sampling seed")
    parser.add_argument("--openlibrary-email", help="contact email for Open Library User-Agent; gets higher documented API limit")
    parser.add_argument("--refresh-catalog", action="store_true", help="redownload Gutenberg catalog")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_genres:
        for genre in sorted(GENRE_TERMS):
            print(genre)
        return 0
    if args.write_example_config:
        write_example_config(args.write_example_config)
        print(f"Wrote {args.write_example_config}")
        return 0

    settings = Settings()
    if args.config:
        config_data = load_json(args.config, None)
        if not isinstance(config_data, dict):
            raise CorpusError(f"Config {args.config} must contain a JSON object")
        settings = settings_from_dict(config_data)
    settings = merge_cli(settings, args)

    try:
        _manifest, status = build_corpus(settings)
        return status
    except KeyboardInterrupt:
        print("Interrupted. Partial downloads and cached metadata are preserved.", file=sys.stderr)
        return 130
    except CorpusError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
