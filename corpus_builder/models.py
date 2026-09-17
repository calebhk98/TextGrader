"""Provider-neutral corpus builder data models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol


@dataclass
class Candidate:
    provider: str
    provider_id: str
    title: str
    authors: list[str]
    work_type: str | None = None
    publication_year: int | None = None
    language: str | None = None
    subjects: list[str] = field(default_factory=list)
    identifiers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    download_options: list[dict[str, str]] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    score: int = 0

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.provider_id}"


@dataclass
class Document:
    text: str
    source_url: str
    media_type: str = "text/plain"


@dataclass
class ProviderStatus:
    available: bool
    detail: str


@dataclass
class SearchQuery:
    language: str
    genres: list[str]
    work_types: list[str]
    include_terms: list[str]
    exclude_terms: list[str]
    include_authors: list[str]
    limit: int

    @property
    def text(self) -> str:
        return " ".join(self.include_terms + self.genres).replace("_", " ").strip()


class CorpusProvider(Protocol):
    name: str

    def search(self, query: SearchQuery) -> Iterable[Candidate]: ...
    def fetch(self, candidate: Candidate) -> Document: ...
    def healthcheck(self) -> ProviderStatus: ...
