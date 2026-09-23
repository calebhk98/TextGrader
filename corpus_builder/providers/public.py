"""Providers for public-domain repositories with web APIs or feeds."""
from __future__ import annotations

import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Any, Iterable

from .base import HTTPProvider
from ..extractors import extract, html_text
from ..models import Candidate, Document, SearchQuery
from ..net import NetworkError, fetch_json


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)] if value else []


def _year(value: Any) -> int | None:
    match = re.search(r"\b(1[0-9]{3}|20[0-9]{2})\b", str(value or ""))
    return int(match.group()) if match else None


class StandardEbooksProvider(HTTPProvider):
    name = "standard_ebooks"
    # The public Atom feed is usable without OPDS client authentication.
    health_url = "https://standardebooks.org/feeds/atom/new-releases"
    feed_url = health_url

    def search(self, query: SearchQuery) -> Iterable[Candidate]:
        root = ET.fromstring(self._get(self.feed_url))
        ns = {"a": "http://www.w3.org/2005/Atom", "dc": "http://purl.org/dc/terms/", "opds": "http://opds-spec.org/2010/catalog"}
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", default="", namespaces=ns).strip()
            authors = [node.findtext("a:name", default="", namespaces=ns).strip() for node in entry.findall("a:author", ns)]
            subjects = [node.get("term", "") for node in entry.findall("a:category", ns)]
            meta = " ".join([title, *authors, *subjects]).casefold()
            if query.text and not all(term in meta for term in query.text.casefold().split()):
                continue
            if query.include_authors and not any(term.casefold() in " ".join(authors).casefold() for term in query.include_authors):
                continue
            links = [{"url": node.get("href", ""), "media_type": node.get("type", "")} for node in entry.findall("a:link", ns) if "epub" in node.get("type", "") and node.get("rel", "") == "enclosure"]
            if not links:
                continue
            entry_id = entry.findtext("a:id", default=title, namespaces=ns).rstrip("/").rsplit("/", 1)[-1]
            work_type = "fiction" if "Fiction" in subjects else None
            if query.work_types and "any" not in query.work_types and work_type not in query.work_types:
                continue
            yield Candidate(self.name, entry_id, title, [a for a in authors if a], work_type, None, query.language, [s for s in subjects if s], {"standard_ebooks": entry_id}, download_options=links)

    def fetch(self, candidate: Candidate) -> Document:
        option = candidate.download_options[0]
        url = direct_download_url(option["url"])
        return Document(extract(self._get(url), option["media_type"], url), url, option["media_type"])


def direct_download_url(url: str) -> str:
    """The URL that returns the file itself rather than a download page.

    Standard Ebooks answers its plain ``.epub`` link with an HTML "Your
    download has started" page whose meta refresh points at the same path
    plus ``?source=download``; only that second URL returns the EPUB.
    Extracting the page instead fails as "not a zip file", so without this
    every Standard Ebooks book in a corpus fails to download.
    """

    if "standardebooks.org" in url and url.endswith(".epub"):
        return f"{url}?source=download"
    return url


class InternetArchiveProvider(HTTPProvider):
    name = "internet_archive"
    health_url = "https://archive.org/metadata/gutenberg"

    def search(self, query: SearchQuery) -> Iterable[Candidate]:
        terms = ["mediatype:texts"]
        if query.language:
            terms.append(f"language:{query.language}")
        if query.text:
            terms.append(f"({query.text})")
        params = urllib.parse.urlencode({"q": " AND ".join(terms), "fl[]": ["identifier", "title", "creator", "subject", "date"], "rows": query.limit, "output": "json"}, doseq=True)
        payload = fetch_json(self.opener, "https://archive.org/advancedsearch.php?" + params, self.timeout)
        for doc in payload.get("response", {}).get("docs", []):
            identifier = str(doc.get("identifier", ""))
            if identifier:
                yield Candidate(self.name, identifier, str(doc.get("title") or identifier), _list(doc.get("creator")), None, _year(doc.get("date")), query.language, _list(doc.get("subject")), {"internet_archive": identifier})

    def fetch(self, candidate: Candidate) -> Document:
        metadata = fetch_json(self.opener, f"https://archive.org/metadata/{urllib.parse.quote(candidate.provider_id)}", self.timeout)
        files = metadata.get("files", [])
        names = [str(item.get("name", "")) for item in files]
        name = next((n for n in names if n.endswith("_djvu.txt")), None) or next((n for n in names if n.lower().endswith(".txt") and "metadata" not in n.lower()), None)
        if not name:
            raise NetworkError(f"No full-text derivative for archive item {candidate.provider_id}")
        url = f"https://archive.org/download/{urllib.parse.quote(candidate.provider_id)}/{urllib.parse.quote(name)}"
        return Document(extract(self._get(url), "text/plain", url), url)


class WikisourceProvider(HTTPProvider):
    name = "wikisource"
    health_url = "https://en.wikisource.org/w/api.php?action=query&meta=siteinfo&format=json"

    def search(self, query: SearchQuery) -> Iterable[Candidate]:
        language = (query.language or "en").split("-")[0]
        base = f"https://{language}.wikisource.org/w/api.php"
        params = urllib.parse.urlencode({"action": "query", "generator": "search", "gsrsearch": query.text or "incategory:Novels", "gsrnamespace": "0", "gsrlimit": min(query.limit, 50), "prop": "info|categories", "inprop": "url", "format": "json", "formatversion": "2"})
        payload = fetch_json(self.opener, base + "?" + params, self.timeout)
        for page in payload.get("query", {}).get("pages", []):
            page_id = str(page.get("pageid", ""))
            if page_id:
                yield Candidate(self.name, page_id, str(page.get("title", page_id)), [], None, None, language, [str(c.get("title", "")) for c in page.get("categories", [])], {"wikisource": page_id}, metadata={"api": base})

    def fetch(self, candidate: Candidate) -> Document:
        params = urllib.parse.urlencode({"action": "parse", "pageid": candidate.provider_id, "prop": "text", "format": "json", "formatversion": "2"})
        url = candidate.metadata["api"] + "?" + params
        payload = fetch_json(self.opener, url, self.timeout)
        markup = str(payload.get("parse", {}).get("text", "")).encode()
        return Document(html_text(markup), url, "text/html")


class GoogleBooksProvider(HTTPProvider):
    name = "google_books"
    health_url = "https://www.googleapis.com/books/v1/volumes?q=public_domain&maxResults=1"

    def __init__(self, *, api_key: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key

    def search(self, query: SearchQuery) -> Iterable[Candidate]:
        params: dict[str, Any] = {"q": query.text or "subject:fiction", "filter": "full", "printType": "books", "maxResults": min(query.limit, 40)}
        if query.language:
            params["langRestrict"] = query.language
        if self.api_key:
            params["key"] = self.api_key
        payload = fetch_json(self.opener, "https://www.googleapis.com/books/v1/volumes?" + urllib.parse.urlencode(params), self.timeout)
        for item in payload.get("items", []):
            info, access = item.get("volumeInfo", {}), item.get("accessInfo", {})
            links = []
            for kind in ("epub",):
                link = access.get(kind, {}).get("downloadLink")
                if link:
                    links.append({"url": link, "media_type": f"application/{kind}+zip"})
            if links:
                identifiers = {str(x.get("type", "")).casefold(): str(x.get("identifier", "")) for x in info.get("industryIdentifiers", [])}
                yield Candidate(self.name, str(item.get("id")), str(info.get("title", item.get("id"))), _list(info.get("authors")), None, _year(info.get("publishedDate")), str(info.get("language") or query.language), _list(info.get("categories")), identifiers, download_options=links)

    def fetch(self, candidate: Candidate) -> Document:
        option = candidate.download_options[0]
        return Document(extract(self._get(option["url"]), option["media_type"], option["url"]), option["url"], option["media_type"])


class LibraryOfCongressProvider(HTTPProvider):
    name = "library_of_congress"
    health_url = "https://www.loc.gov/books/?fo=json&c=1"

    def search(self, query: SearchQuery) -> Iterable[Candidate]:
        params = urllib.parse.urlencode({"fo": "json", "q": query.text, "c": min(query.limit, 100)})
        payload = fetch_json(self.opener, "https://www.loc.gov/books/?" + params, self.timeout)
        for item in payload.get("results", []):
            options = []
            for resource in item.get("resources", []):
                for file in resource.get("files", []) if isinstance(resource.get("files"), list) else []:
                    for value in file if isinstance(file, list) else [file]:
                        url = value.get("url") if isinstance(value, dict) else None
                        mime = value.get("mimetype", "") if isinstance(value, dict) else ""
                        if url and ("text" in mime or str(url).lower().endswith((".txt", ".html"))):
                            options.append({"url": str(url), "media_type": str(mime or "text/plain")})
            if options:
                identifier = str(item.get("id", "")).rstrip("/").rsplit("/", 1)[-1]
                yield Candidate(self.name, identifier, str(item.get("title", identifier)), _list(item.get("contributor")), None, _year(item.get("date")), query.language, _list(item.get("subject")), {"loc": str(item.get("id", ""))}, download_options=options)

    def fetch(self, candidate: Candidate) -> Document:
        option = candidate.download_options[0]
        return Document(extract(self._get(option["url"]), option["media_type"], option["url"]), option["url"], option["media_type"])
