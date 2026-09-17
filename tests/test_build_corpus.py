import csv
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_corpus
from corpus_builder import Candidate, Document, ProviderStatus, SearchQuery
from corpus_builder import builder
from corpus_builder.config import Settings, from_mapping
from corpus_builder.extractors import epub_text, html_text
from corpus_builder.metadata import enrich_first
from corpus_builder.net import NetworkError, fetch
from corpus_builder.providers import PROVIDER_TYPES
from corpus_builder.providers.gutenberg import GutenbergProvider
from textgrader.corpus import build_profile


def _write_catalog(path: Path, rows: list[dict[str, str]]) -> None:
    columns = ["Text#", "Type", "Issued", "Title", "Language", "Authors", "Subjects", "LoCC", "Bookshelves"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as catalog:
        catalog.write(buffer.getvalue())


def _row(subjects: str = "Science fiction; Space warfare -- Fiction") -> dict[str, str]:
    return {"Text#": "123", "Type": "Text", "Issued": "2006-01-01", "Title": "A Test Voyage", "Language": "en", "Authors": "Writer, Alice, 1900-1980", "Subjects": subjects, "LoCC": "PS", "Bookshelves": "Science Fiction"}


def _query(work_types: list[str]) -> SearchQuery:
    return SearchQuery("en", [], work_types, [], [], [], 20)


def test_all_documented_providers_implement_the_small_interface(tmp_path: Path) -> None:
    expected = {"gutenberg", "standard_ebooks", "internet_archive", "wikisource", "google_books", "library_of_congress"}
    assert set(PROVIDER_TYPES) == expected
    for name, provider_type in PROVIDER_TYPES.items():
        kwargs = {"cache_dir": tmp_path} if name == "gutenberg" else {}
        provider = provider_type(**kwargs)
        assert provider.name == name
        assert callable(provider.search)
        assert callable(provider.fetch)
        assert callable(provider.healthcheck)


def test_gutenberg_form_filter_allows_short_stories_poetry_essays_and_speeches(tmp_path: Path) -> None:
    cases = {"short_story": "Short stories", "poetry": "Poetry", "essay": "Essays", "speech": "Speeches"}
    for index, (form, subject) in enumerate(cases.items()):
        cache = tmp_path / str(index)
        cache.mkdir()
        _write_catalog(cache / "pg_catalog.csv.gz", [{**_row(subject), "Text#": str(index + 1)}])
        candidates = list(GutenbergProvider(cache_dir=cache).search(_query([form])))
        assert len(candidates) == 1
        assert candidates[0].work_type == form


def test_gutenberg_filters_author_and_exclusion_terms(tmp_path: Path) -> None:
    _write_catalog(tmp_path / "pg_catalog.csv.gz", [_row()])
    provider = GutenbergProvider(cache_dir=tmp_path)
    query = SearchQuery("en", ["science_fiction"], ["fiction"], [], [], ["Alice"], 20)
    assert next(iter(provider.search(query))).provider_id == "123"
    query.exclude_terms = ["space warfare"]
    assert list(provider.search(query)) == []


def test_html_and_epub_extractors_are_dependency_free() -> None:
    assert html_text(b"<html><style>bad</style><p>Hello &amp; welcome.</p></html>").strip() == "Hello & welcome."
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as book:
        book.writestr("chapter.xhtml", "<html><body><h1>One</h1><p>Story text.</p></body></html>")
    extracted = epub_text(archive.getvalue())
    assert "One" in extracted and "Story text." in extracted


def test_epub_rejects_excessively_compressed_entries() -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as book:
        book.writestr("chapter.xhtml", "x" * 100_000)
    try:
        epub_text(archive.getvalue())
    except ValueError as exc:
        assert "compression ratio" in str(exc)
    else:
        raise AssertionError("compression bomb was accepted")


def test_fetch_enforces_streaming_response_limit() -> None:
    class Response(io.BytesIO):
        headers = {}
        def __enter__(self): return self
        def __exit__(self, *_): return None
    class Opener:
        def open(self, *_args, **_kwargs): return Response(b"12345")
    try:
        fetch(Opener(), "https://example.test", 1, attempts=1, max_bytes=4)
    except NetworkError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("oversized response was accepted")


def test_nested_project_config_and_author_limits_are_supported() -> None:
    settings = from_mapping({"corpus_builder": {"count": 2, "minimum_count": 1, "work_types": ["poetry"], "max_authors": 1, "max_books_per_author": None}})
    assert settings.work_types == ["poetry"]
    assert settings.max_authors == 1
    assert settings.max_books_per_author is None


def test_metadata_enrichment_falls_back_after_an_outage() -> None:
    class Broken:
        name = "broken"

        def enrich(self, candidate):
            raise NetworkError("offline")

    class Working:
        name = "working"

        def enrich(self, candidate):
            return {"publication_year": 1901, "source": self.name}

    result = enrich_first(Candidate("test", "1", "Title", ["Author"]), [Broken(), Working()])
    assert result == {"publication_year": 1901, "source": "working"}


class FakeProvider:
    name = "fake"

    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    def search(self, query):
        return iter(self.candidates)

    def fetch(self, candidate):
        text = " ".join(["She walked home and he said they had looked around."] * 250)
        return Document(text, f"https://example.test/{candidate.provider_id}.txt")

    def healthcheck(self):
        return ProviderStatus(True, "ok")


def test_builder_deduplicates_sources_and_enforces_distinct_author_cap(tmp_path: Path) -> None:
    candidates = [
        Candidate("fake", "1", "Same Book", ["Alice"], "fiction", identifiers={"isbn_13": "X"}),
        Candidate("fake", "2", "Same Book Elsewhere", ["Alice"], "fiction", identifiers={"isbn_13": "X"}),
        Candidate("fake", "3", "Second", ["Bob"], "fiction"),
    ]
    provider = FakeProvider(candidates)
    settings = Settings(output_dir=str(tmp_path), count=2, minimum_count=1, providers=["gutenberg"], metadata_providers=[], max_authors=1, max_books_per_author=None, year_policy="ignore", min_words=1000, max_words=10000, request_delay=0)
    with patch.object(builder, "_make_providers", return_value=[provider]), patch.object(builder, "_make_enrichers", return_value=[]):
        manifest, status = builder.build(settings)
    assert status == 0
    assert len(manifest["books"]) == 1
    assert manifest["summary"]["unique_primary_authors"] == 1
    assert manifest["rejections"]["author_count_cap"] == 1


def test_builder_writes_provider_neutral_audit_manifest(tmp_path: Path) -> None:
    provider = FakeProvider([Candidate("fake", "42", "A Voyage", ["Alice"], "fiction", 1901, "en", ["space"], {"isbn_13": "123"})])
    settings = Settings(output_dir=str(tmp_path), count=1, minimum_count=1, providers=["gutenberg"], metadata_providers=[], year_policy="strict", min_words=1000, max_words=10000, request_delay=0)
    with patch.object(builder, "_make_providers", return_value=[provider]), patch.object(builder, "_make_enrichers", return_value=[]):
        manifest, status = builder.build(settings)
    book = manifest["books"][0]
    assert status == 0
    assert (book["provider"], book["provider_id"]) == ("fake", "42")
    assert book["identifiers"] == {"isbn_13": "123"}
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest
    profile = build_profile([tmp_path / "books"], manifest=manifest)
    assert profile["books"][0]["source_id"] == "fake:42"
    assert profile["books"][0]["metadata"]["provider"] == "fake"


def test_main_reports_invalid_nested_config_without_traceback(tmp_path: Path, capsys) -> None:
    config = tmp_path / "invalid.json"
    config.write_text('{"corpus_builder": {"not_a_setting": true}}', encoding="utf-8")
    assert build_corpus.main(["--config", str(config)]) == 1
    assert "ERROR: Unknown config keys: not_a_setting" in capsys.readouterr().err


def test_checked_in_example_config_enables_all_sources() -> None:
    config = Path(__file__).resolve().parents[1] / "examples/corpus_builder.science_fiction.json"
    settings = from_mapping(json.loads(config.read_text(encoding="utf-8")))
    assert set(settings.providers) == set(PROVIDER_TYPES)
    assert settings.genres == ["science_fiction"]
    assert settings.work_types == ["fiction"]
