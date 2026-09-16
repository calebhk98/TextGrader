"""Dependency-free text extraction for provider responses."""
from __future__ import annotations

import html
import io
import re
import zipfile
from html.parser import HTMLParser


class _HTMLText(HTMLParser):
    BLOCKS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "blockquote", "section", "article"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "nav"}:
            self.ignored += 1
        elif not self.ignored and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self.ignored:
            self.ignored -= 1
        elif not self.ignored and tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored:
            self.parts.append(data)


def plain_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def html_text(data: bytes) -> str:
    parser = _HTMLText()
    parser.feed(plain_text(data))
    text = html.unescape("".join(parser.parts))
    return re.sub(r"\n(?:[ \t]*\n){2,}", "\n\n", text).strip() + "\n"


def epub_text(data: bytes) -> str:
    """Extract reading-order XHTML when possible, falling back to archive order."""
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        names = book.namelist()
        ordered: list[str] = []
        container = next((n for n in names if n.endswith("META-INF/container.xml")), None)
        if container:
            rootfile = re.search(rb'full-path=["\']([^"\']+)', book.read(container))
            if rootfile:
                opf_name = rootfile.group(1).decode("utf-8")
                opf = plain_text(book.read(opf_name))
                base = opf_name.rpartition("/")[0]
                hrefs = dict(re.findall(r'<item\b[^>]*\bid=["\']([^"\']+)["\'][^>]*\bhref=["\']([^"\']+)', opf))
                spine = re.findall(r'<itemref\b[^>]*\bidref=["\']([^"\']+)', opf)
                ordered = [f"{base}/{hrefs[item]}".lstrip("/") for item in spine if item in hrefs]
        if not ordered:
            ordered = [n for n in names if n.lower().endswith((".xhtml", ".html", ".htm"))]
        return "\n\n".join(html_text(book.read(name)).strip() for name in ordered if name in names).strip() + "\n"


def extract(data: bytes, media_type: str, url: str = "") -> str:
    kind = media_type.casefold()
    lower_url = url.casefold()
    if "epub" in kind or lower_url.endswith(".epub"):
        return epub_text(data)
    if "html" in kind or lower_url.endswith((".html", ".htm")):
        return html_text(data)
    return plain_text(data)
