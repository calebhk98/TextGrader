"""Project Gutenberg catalog and text provider."""
from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path
from typing import Iterable

from .base import HTTPProvider
from ..extractors import plain_text
from ..models import Candidate, Document, SearchQuery
from ..net import NetworkError

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
FICTION_HINTS = ("fiction", "novel", "stories", "tale", "romance", "adventure", "mystery", "fantasy", "horror", "detective", "western")
FORM_TERMS = {
    "fiction": FICTION_HINTS,
    "short_story": ("short stories", "short story"),
    "essay": ("essays", "essay"),
    "speech": ("speeches", "speech", "orations"),
    "poetry": ("poetry", "poems", "verse"),
    "drama": ("drama", "plays"),
}
GENRE_TERMS = {
    "science_fiction": ("science fiction", "science-fiction", "interplanetary voyages", "space warfare", "life on other planets", "robots -- fiction", "time travel -- fiction", "future life -- fiction"),
    "fantasy": ("fantasy fiction", "fantasy", "fairy tales", "magic -- fiction", "dragons -- fiction", "imaginary places -- fiction", "mythological fiction"),
    "mystery": ("detective and mystery stories", "detective fiction", "mystery fiction", "crime -- fiction", "murder -- investigation -- fiction"),
    "horror": ("horror tales", "horror fiction", "ghost stories", "supernatural -- fiction", "occult fiction"),
    "romance": ("love stories", "romance fiction", "courtship -- fiction", "man-woman relationships -- fiction"),
    "adventure": ("adventure stories", "adventure fiction", "sea stories", "survival -- fiction", "voyages and travels -- fiction"),
    "historical": ("historical fiction", "history -- fiction"),
    "western": ("western stories", "western fiction", "west (u.s.) -- fiction", "frontier and pioneer life -- fiction"),
    "children": ("juvenile fiction", "children's stories", "children -- fiction", "young adult fiction"),
    "gothic": ("gothic fiction", "gothic romance", "castles -- fiction"),
    "nautical": ("sea stories", "seafaring life -- fiction", "sailors -- fiction", "ships -- fiction"),
    "war": ("war stories", "war -- fiction", "soldiers -- fiction"),
    "literary": ("psychological fiction", "domestic fiction", "bildungsromans", "social life and customs -- fiction"),
}
START_RE = re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)


def _value(row, name: str) -> str:
    return str(row.get(name) or "").strip()


def infer_work_type(metadata: str) -> str | None:
    for form in ("short_story", "poetry", "essay", "speech", "drama", "fiction"):
        if any(term in metadata for term in FORM_TERMS[form]):
            return form
    return None


def strip_boilerplate(text: str) -> str:
    start = START_RE.search(text)
    if start:
        text = text[start.end():]
    end = END_RE.search(text)
    if end:
        text = text[:end.start()]
    return text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


class GutenbergProvider(HTTPProvider):
    name = "gutenberg"
    health_url = "https://www.gutenberg.org/robots.txt"

    def __init__(self, *, cache_dir: Path, catalog_url: str = CATALOG_URL, refresh: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cache_dir = cache_dir
        self.catalog_url = catalog_url
        self.refresh = refresh

    def _catalog(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target = self.cache_dir / "pg_catalog.csv.gz"
        if not target.exists() or self.refresh:
            data = self._get(self.catalog_url)
            try:
                gzip.decompress(data)
            except OSError as exc:
                raise NetworkError("Gutenberg catalog was not valid gzip data") from exc
            target.write_bytes(data)
        return target

    def search(self, query: SearchQuery) -> Iterable[Candidate]:
        with gzip.open(self._catalog(), "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                language = _value(row, "Language")
                if query.language and query.language not in re.split(r"[;, ]+", language):
                    continue
                meta = " | ".join((_value(row, "Title"), _value(row, "Subjects"), _value(row, "Bookshelves"))).casefold()
                form = infer_work_type(meta)
                if query.work_types and "any" not in query.work_types and form not in query.work_types:
                    continue
                if query.exclude_terms and any(term.casefold() in meta for term in query.exclude_terms):
                    continue
                if query.include_terms and not all(term.casefold() in meta for term in query.include_terms):
                    continue
                hits: list[str] = []
                if any(not any(term in meta for term in GENRE_TERMS.get(genre, ())) for genre in query.genres):
                    continue
                for genre in query.genres:
                    hits.extend(term for term in GENRE_TERMS.get(genre, ()) if term in meta)
                authors = [part.strip() for part in _value(row, "Authors").split(";") if part.strip()]
                if not _value(row, "Title") or not authors or (query.include_authors and not any(any(w.casefold() in a.casefold() for w in query.include_authors) for a in authors)):
                    continue
                gid = _value(row, "Text#")
                if not gid.isdigit():
                    continue
                yield Candidate(self.name, gid, _value(row, "Title"), authors, form, None, language, [s.strip() for s in _value(row, "Subjects").split(";") if s.strip()], {"gutenberg": gid}, {"issued": _value(row, "Issued"), "bookshelves": _value(row, "Bookshelves"), "locc": _value(row, "LoCC")}, matched_terms=sorted(set(hits)), score=len(set(hits)) * 10)

    def fetch(self, candidate: Candidate) -> Document:
        gid = candidate.provider_id
        urls = [f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt", f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt", f"https://www.gutenberg.org/files/{gid}/{gid}.txt", f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8"]
        errors: list[str] = []
        for url in urls:
            try:
                text = strip_boilerplate(plain_text(self._get(url)))
                if len(text) >= 5000 and not text.lstrip().lower().startswith("<!doctype html"):
                    return Document(text, url)
                errors.append(f"{url}: response too small/not text")
            except NetworkError as exc:
                errors.append(str(exc))
        raise NetworkError("; ".join(errors[-2:]))
