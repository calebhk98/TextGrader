"""Provider-neutral candidate selection and manifest generation."""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from .classifiers import classification_allowed, classify_pov, classify_tense, word_count
from .config import Settings
from .metadata import GoogleBooksMetadata, OpenLibraryMetadata, enrich_first
from .models import Candidate, SearchQuery
from .net import NetworkError
from .providers import PROVIDER_TYPES


class BuildError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _canonical(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in value if not unicodedata.combining(c)).casefold()).strip()


def _slug(value: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", _canonical(value)).strip("-")[:72].rstrip("-") or "document")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _dedupe_key(candidate: Candidate) -> str:
    for identifier in ("isbn_13", "isbn_10", "lccn", "oclc"):
        if candidate.identifiers.get(identifier):
            return f"{identifier}:{candidate.identifiers[identifier]}"
    author = candidate.authors[0] if candidate.authors else ""
    return f"title:{_canonical(candidate.title)}|author:{_canonical(author).split(',')[0]}"


def _candidate_allowed(candidate: Candidate, settings: Settings) -> bool:
    metadata = " ".join([candidate.title, *candidate.subjects]).casefold()
    form_terms = {
        "short_story": ("short stor",),
        "poetry": ("poetry", "poems", "verse"),
        "essay": ("essay",),
        "speech": ("speech", "oration"),
        "drama": ("drama", "plays"),
        "fiction": ("fiction", "novel", "romance"),
    }
    if candidate.work_type is None:
        candidate.work_type = next((form for form, terms in form_terms.items() if any(term in metadata for term in terms)), None)
    if settings.work_types and "any" not in settings.work_types and candidate.work_type not in settings.work_types:
        return False
    if settings.include_terms and not all(term.casefold() in metadata for term in settings.include_terms):
        return False
    if settings.exclude_terms and any(term.casefold() in metadata for term in settings.exclude_terms):
        return False
    return not settings.include_authors or any(term.casefold() in " ".join(candidate.authors).casefold() for term in settings.include_authors)


def _make_providers(settings: Settings, cache: Path):
    common = {"timeout": settings.timeout}
    providers = []
    for name in settings.providers:
        kind = PROVIDER_TYPES[name]
        if name == "gutenberg":
            providers.append(kind(cache_dir=cache, catalog_url=settings.catalog_url, refresh=settings.refresh_catalog, **common))
        elif name == "google_books":
            providers.append(kind(api_key=settings.google_books_api_key, **common))
        else:
            providers.append(kind(**common))
    return providers


def _make_enrichers(settings: Settings):
    values = []
    for name in settings.metadata_providers:
        if name == "openlibrary":
            values.append(OpenLibraryMetadata(timeout=settings.timeout, email=settings.openlibrary_email))
        elif name == "google_books":
            values.append(GoogleBooksMetadata(timeout=settings.timeout, api_key=settings.google_books_api_key))
    return values


def _year_allowed(year: int | None, settings: Settings) -> bool:
    if settings.year_policy == "ignore" or (settings.min_year is None and settings.max_year is None):
        return True
    if year is None:
        return settings.year_policy == "best_effort"
    return (settings.min_year is None or year >= settings.min_year) and (settings.max_year is None or year <= settings.max_year)


def discover(settings: Settings, providers) -> tuple[list[tuple[Any, Candidate]], dict[str, str]]:
    # Give every configured source a share instead of allowing the first large
    # catalog to crowd all later providers out of the candidate pool.
    per_provider_limit = max(settings.count, (settings.max_candidates + len(providers) - 1) // len(providers))
    query = SearchQuery(settings.language, settings.genres, settings.work_types, settings.include_terms, settings.exclude_terms, settings.include_authors, per_provider_limit)
    found: list[tuple[Any, Candidate]] = []
    failures: dict[str, str] = {}
    seen: set[str] = set()
    for provider in providers:
        try:
            for candidate in provider.search(query):
                if not _candidate_allowed(candidate, settings):
                    continue
                key = _dedupe_key(candidate)
                if key not in seen:
                    seen.add(key)
                    found.append((provider, candidate))
                if sum(1 for source, _candidate in found if source is provider) >= per_provider_limit:
                    break
        except (NetworkError, OSError, ValueError) as exc:
            failures[provider.name] = str(exc)
    random.Random(settings.seed).shuffle(found)
    found.sort(key=lambda pair: pair[1].score, reverse=True)
    return found[: settings.max_candidates], failures


def build(settings: Settings) -> tuple[dict[str, Any], int]:
    root, cache, books = Path(settings.output_dir), Path(settings.output_dir) / ".cache", Path(settings.output_dir) / "books"
    cache.mkdir(parents=True, exist_ok=True)
    books.mkdir(parents=True, exist_ok=True)
    providers = _make_providers(settings, cache)
    candidates, failures = discover(settings, providers)
    if not candidates:
        detail = "; ".join(f"{name}: {error}" for name, error in failures.items())
        raise BuildError("No provider returned matching candidates" + (f" ({detail})" if detail else ""))
    enrichers = _make_enrichers(settings)
    year_cache_path = cache / "publication_years.json"
    try:
        year_cache = json.loads(year_cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        year_cache = {}
    selected: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    author_counts: Counter[str] = Counter()
    represented_authors: set[str] = set()
    lookups = 0
    manifest = {"schema_version": 2, "builder": {"name": "TextGrader Corpus Builder", "version": "2.0.0"}, "created_at": _now(), "selection": dataclasses.asdict(settings), "providers": {name: {"error": error} for name, error in failures.items()}, "books": [], "sources": {}, "rejections": {}}

    for provider, candidate in candidates:
        if len(selected) >= settings.count:
            break
        author = _canonical(candidate.authors[0]) if candidate.authors else "unknown"
        if settings.max_books_per_author is not None and author_counts[author] >= settings.max_books_per_author:
            rejected["author_cap"] += 1
            continue
        if settings.max_authors is not None and author not in represented_authors and len(represented_authors) >= settings.max_authors:
            rejected["author_count_cap"] += 1
            continue
        year_info = {"publication_year": candidate.publication_year, "confidence": 1.0 if candidate.publication_year else 0.0, "source": provider.name if candidate.publication_year else None, "key": None}
        needs_year = settings.year_policy != "ignore" and (settings.min_year is not None or settings.max_year is not None)
        if needs_year and candidate.publication_year is None:
            if candidate.key in year_cache:
                year_info = year_cache[candidate.key]
            elif lookups < settings.year_lookup_limit:
                year_info = enrich_first(candidate, enrichers)
                # Do not make a temporary provider outage permanent. Successful
                # matches and definitive no-match results are safe to cache;
                # network failures are retried on a later run.
                if not year_info.get("errors"):
                    year_cache[candidate.key] = year_info
                    _save(year_cache_path, year_cache)
                lookups += 1
            elif settings.year_policy == "strict":
                rejected["year_lookup_limit"] += 1
                continue
        if not _year_allowed(year_info.get("publication_year"), settings):
            rejected["publication_year"] += 1
            continue
        filename = f"{candidate.provider}-{_slug(candidate.provider_id)}-{_slug(candidate.title)}.txt"
        path = books / filename
        try:
            if path.exists() and path.stat().st_size > 100:
                text, source_url = path.read_text(encoding="utf-8", errors="replace"), f"cached:{candidate.key}"
            else:
                document = provider.fetch(candidate)
                text, source_url = document.text, document.source_url
                path.write_text(text, encoding="utf-8", newline="\n")
                if settings.request_delay:
                    time.sleep(settings.request_delay)
        except (NetworkError, OSError, ValueError) as exc:
            rejected["download_error"] += 1
            continue
        words = word_count(text)
        pov, tense = classify_pov(text), classify_tense(text)
        reason = "too_short" if words < settings.min_words else "too_long" if words > settings.max_words else "pov" if not classification_allowed(pov, settings.pov, settings.min_pov_confidence) else "tense" if not classification_allowed(tense, settings.tense, settings.min_tense_confidence) else None
        if reason:
            rejected[reason] += 1
            if not settings.keep_failed_downloads:
                path.unlink(missing_ok=True)
            continue
        book = {"provider": candidate.provider, "provider_id": candidate.provider_id, "title": candidate.title, "authors": candidate.authors, "work_type": candidate.work_type, "language": candidate.language, "subjects": candidate.subjects, "identifiers": candidate.identifiers, "provider_metadata": candidate.metadata, "matched_genre_terms": candidate.matched_terms, "first_publish_year": year_info.get("publication_year"), "publication_year_source": year_info.get("source"), "publication_year_match_confidence": year_info.get("confidence"), "metadata_key": year_info.get("key"), "inferred_pov": pov, "inferred_tense": tense, "word_count": words, "file": str(path.relative_to(root)), "sha256": _sha(path), "source_url": source_url, "retrieved_at": _now()}
        selected.append(book)
        author_counts[author] += 1
        represented_authors.add(author)
        manifest["sources"][path.name] = {"id": candidate.key, "provider": candidate.provider, "provider_id": candidate.provider_id, "title": candidate.title, "authors": candidate.authors, "publication_year": year_info.get("publication_year"), "work_type": candidate.work_type, "sha256": book["sha256"]}
        manifest.update({"books": selected, "rejections": dict(rejected), "updated_at": _now()})
        _save(root / "manifest.json", manifest)

    manifest.update({"books": selected, "rejections": dict(rejected), "updated_at": _now(), "summary": {"requested": settings.count, "minimum_required": settings.minimum_count, "selected": len(selected), "unique_primary_authors": len(represented_authors), "total_words": sum(item["word_count"] for item in selected), "year_lookups_performed": lookups, "passed_minimum": len(selected) >= settings.minimum_count}})
    _save(root / "manifest.json", manifest)
    return manifest, 0 if len(selected) >= settings.minimum_count else 2


def provider_health(settings: Settings) -> dict[str, dict[str, Any]]:
    cache = Path(settings.output_dir) / ".cache"
    return {provider.name: dataclasses.asdict(provider.healthcheck()) for provider in _make_providers(settings, cache)}
