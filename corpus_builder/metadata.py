"""Replaceable metadata enrichment chain."""
from __future__ import annotations

import difflib
import re
import urllib.parse
from typing import Any, Protocol

from .models import Candidate
from .net import NetworkError, fetch_json, make_opener


class MetadataEnricher(Protocol):
    name: str
    def enrich(self, candidate: Candidate) -> dict[str, Any] | None: ...


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


class OpenLibraryMetadata:
    name = "openlibrary"

    def __init__(self, *, timeout: float, email: str | None = None) -> None:
        agent = "TextGrader-CorpusBuilder/2.0" + (f" ({email})" if email else "")
        self.opener, self.timeout = make_opener(agent), timeout

    def enrich(self, candidate: Candidate) -> dict[str, Any] | None:
        params = urllib.parse.urlencode({"title": candidate.title, "author": candidate.authors[0] if candidate.authors else "", "fields": "key,title,author_name,first_publish_year", "limit": 5})
        payload = fetch_json(self.opener, "https://openlibrary.org/search.json?" + params, self.timeout)
        scored = []
        for doc in payload.get("docs", []):
            year = doc.get("first_publish_year")
            if isinstance(year, int):
                title_score = difflib.SequenceMatcher(None, _canonical(candidate.title), _canonical(str(doc.get("title", "")))).ratio()
                authors = _canonical(" ".join(doc.get("author_name") or []))
                author_score = max((_canonical(a).split(",")[0] in authors for a in candidate.authors), default=False)
                scored.append((0.8 * title_score + 0.2 * author_score, doc))
        if not scored:
            return None
        score, doc = max(scored, key=lambda pair: pair[0])
        return {"publication_year": int(doc["first_publish_year"]), "confidence": round(score, 3), "key": doc.get("key"), "source": self.name} if score >= 0.62 else None


class GoogleBooksMetadata:
    name = "google_books"

    def __init__(self, *, timeout: float, api_key: str | None = None) -> None:
        self.opener, self.timeout, self.api_key = make_opener("TextGrader-CorpusBuilder/2.0"), timeout, api_key

    def enrich(self, candidate: Candidate) -> dict[str, Any] | None:
        query = f'intitle:"{candidate.title}"' + (f' inauthor:"{candidate.authors[0]}"' if candidate.authors else "")
        params = {"q": query, "maxResults": 5}
        if self.api_key:
            params["key"] = self.api_key
        payload = fetch_json(self.opener, "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode(params), self.timeout)
        for item in payload.get("items", []):
            info = item.get("volumeInfo", {})
            match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(info.get("publishedDate", "")))
            score = difflib.SequenceMatcher(None, _canonical(candidate.title), _canonical(str(info.get("title", "")))).ratio()
            if match and score >= 0.7:
                return {"publication_year": int(match.group()), "confidence": round(score, 3), "key": item.get("id"), "source": self.name}
        return None


def enrich_first(candidate: Candidate, enrichers: list[MetadataEnricher]) -> dict[str, Any]:
    errors: list[str] = []
    for enricher in enrichers:
        try:
            result = enricher.enrich(candidate)
            if result:
                return result
        except NetworkError as exc:
            errors.append(f"{enricher.name}: {exc}")
    return {"publication_year": None, "confidence": 0.0, "key": None, "source": None, "errors": errors}
