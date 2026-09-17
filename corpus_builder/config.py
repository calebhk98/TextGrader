"""Settings loading and validation."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .providers import GENRE_TERMS, PROVIDER_TYPES


class ConfigError(RuntimeError):
    pass


@dataclass
class Settings:
    output_dir: str = "corpus"
    count: int = 40
    minimum_count: int = 30
    language: str = "en"
    providers: list[str] = field(default_factory=lambda: ["gutenberg"])
    metadata_providers: list[str] = field(default_factory=lambda: ["openlibrary", "google_books"])
    work_types: list[str] = field(default_factory=lambda: ["fiction"])
    genres: list[str] = field(default_factory=list)
    include_terms: list[str] = field(default_factory=list)
    exclude_terms: list[str] = field(default_factory=list)
    include_authors: list[str] = field(default_factory=list)
    max_authors: int | None = None
    pov: str = "any"
    min_pov_confidence: float = 0.35
    tense: str = "any"
    min_tense_confidence: float = 0.25
    min_year: int | None = None
    max_year: int | None = 1989
    year_policy: str = "best_effort"
    min_words: int = 30000
    max_words: int = 250000
    max_books_per_author: int | None = 2
    max_candidates: int = 800
    seed: int = 20260916
    request_delay: float = 2.0
    timeout: float = 30.0
    catalog_url: str = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv.gz"
    refresh_catalog: bool = False
    openlibrary_email: str | None = None
    google_books_api_key: str | None = None
    year_lookup_limit: int = 100
    keep_failed_downloads: bool = False


def validate(settings: Settings) -> None:
    if settings.count < 1 or settings.minimum_count < 1 or settings.minimum_count > settings.count:
        raise ConfigError("count and minimum_count must be positive, with minimum_count <= count")
    unknown = sorted(set(settings.providers) - set(PROVIDER_TYPES))
    if not settings.providers:
        raise ConfigError("providers must contain at least one acquisition source")
    if unknown:
        raise ConfigError(f"Unknown providers: {', '.join(unknown)}")
    unknown_metadata = sorted(set(settings.metadata_providers) - {"openlibrary", "google_books"})
    if unknown_metadata:
        raise ConfigError(f"Unknown metadata providers: {', '.join(unknown_metadata)}")
    unknown_genres = sorted(set(settings.genres) - set(GENRE_TERMS))
    if unknown_genres:
        raise ConfigError(f"Unknown genres: {', '.join(unknown_genres)}")
    valid_forms = {"any", "fiction", "short_story", "essay", "speech", "poetry", "drama"}
    if not settings.work_types or set(settings.work_types) - valid_forms:
        raise ConfigError("work_types must contain any, fiction, short_story, essay, speech, poetry, or drama")
    if settings.pov not in {"any", "first", "third"} or settings.tense not in {"any", "past", "present"}:
        raise ConfigError("pov and tense must use their documented values")
    if settings.year_policy not in {"ignore", "best_effort", "strict"}:
        raise ConfigError("year_policy must be ignore, best_effort, or strict")
    if settings.min_words < 1 or settings.max_words <= settings.min_words:
        raise ConfigError("word-count range is invalid")
    if settings.max_candidates < settings.count:
        raise ConfigError("max_candidates must be >= count")
    if settings.timeout <= 0 or settings.request_delay < 0 or settings.year_lookup_limit < 0:
        raise ConfigError("timeout must be positive; delays and lookup limits cannot be negative")
    if settings.max_books_per_author is not None and settings.max_books_per_author < 1:
        raise ConfigError("max_books_per_author must be null or >= 1")
    if settings.max_authors is not None and settings.max_authors < 1:
        raise ConfigError("max_authors must be null or >= 1")


def from_mapping(data: Mapping[str, Any]) -> Settings:
    # A project's config.json may hold corpus settings alongside grader settings.
    if "corpus_builder" in data:
        nested = data["corpus_builder"]
        if not isinstance(nested, Mapping):
            raise ConfigError("corpus_builder must be a JSON object")
        data = nested
    fields = {item.name for item in dataclasses.fields(Settings)}
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ConfigError(f"Unknown config keys: {', '.join(unknown)}")
    values = dict(data)
    for key in ("providers", "metadata_providers", "work_types", "genres", "include_terms", "exclude_terms", "include_authors"):
        if isinstance(values.get(key), str):
            values[key] = [values[key]]
    try:
        settings = Settings(**values)
        validate(settings)
    except TypeError as exc:
        raise ConfigError(f"Invalid configuration value: {exc}") from exc
    return settings


def load(path: Path) -> Settings:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Config {path} must contain a JSON object")
    settings = from_mapping(data)
    output = Path(settings.output_dir)
    if not output.is_absolute():
        settings.output_dir = str((path.resolve().parent / output).resolve())
    return settings
