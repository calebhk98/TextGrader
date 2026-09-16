"""Canonical, dependency-free text processing used by all metrics.

The routines here deliberately make modest claims.  In particular, sentence
segmentation is punctuation based rather than a grammatical parser.  Keeping
these choices in one module is preferable to subtly different regular
expressions in every metric, and makes the preprocessing recorded in a corpus
profile reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Pattern


# ``[^\W\d_]`` is a Unicode letter without depending on a third-party regex
# library. Apostrophes are retained only when surrounded by letters.
WORD_RE = re.compile(r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*", re.UNICODE)
HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}(?:\s+|$).*?(?:\n|$)")
SETEXT_HEADING_RE = re.compile(r"(?m)^.*\S.*\n\s{0,3}(?:=+|-+)\s*(?:\n|$)")
GUTENBERG_START_RE = re.compile(
    r"(?im)^\s*\*{0,3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG\b.*$"
)
GUTENBERG_END_RE = re.compile(
    r"(?im)^\s*\*{0,3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG\b.*$"
)

SINGLE_QUOTE_WARNING = (
    "Single-quote dialogue was detected but is not parsed; dialogue statistics "
    "include only straight or curly double-quoted speech."
)


def words(text: str) -> list[str]:
    """Return Unicode-aware word tokens, preserving internal apostrophes."""

    return WORD_RE.findall(text)


def paragraphs(text: str) -> list[str]:
    """Split prose on blank lines and join soft line wraps within a paragraph."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n+", normalized)
    return [re.sub(r"[ \t]*\n[ \t]*", " ", block).strip()
            for block in blocks if block.strip()]


def sentences(text: str) -> list[str]:
    """Split text into punctuation-delimited sentences.

    Closing quotation marks and brackets remain attached to their sentence.
    An unterminated final fragment is returned as a sentence when it has words.
    This is intentionally a transparent heuristic, not a claim to NLP parsing.
    """

    result: list[str] = []
    for paragraph in paragraphs(text):
        start = 0
        for match in re.finditer(r"[.!?\u2026]+(?:[\"\u201d\u2019)\]]+)?(?=\s|$)", paragraph):
            item = paragraph[start:match.end()].strip()
            if words(item):
                result.append(item)
            start = match.end()
        tail = paragraph[start:].strip()
        if words(tail):
            result.append(tail)
    return result


def remove_markdown_headings(text: str) -> str:
    """Remove ATX and Setext Markdown headings, retaining surrounding prose."""

    return HEADING_RE.sub("", SETEXT_HEADING_RE.sub("", text))


def normalize_quotes(text: str, *, single: bool = True) -> str:
    """Normalize typographic quote characters to their ASCII equivalents.

    Callers doing dialogue analysis should normally use :func:`split_dialogue`
    first, because normalizing single quotes makes apostrophes and British
    single-quoted speech indistinguishable.
    """

    table = {ord("\u201c"): '"', ord("\u201d"): '"'}
    if single:
        table.update({ord("\u2018"): "'", ord("\u2019"): "'"})
    return text.translate(table)


def strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer delimited by standard markers.

    Text without markers is returned unchanged.  Either marker may occur on
    its own, which is useful for imperfect local source files.
    """

    start = GUTENBERG_START_RE.search(text)
    if start:
        newline = text.find("\n", start.end())
        text = text[newline + 1 if newline >= 0 else start.end():]
    end = GUTENBERG_END_RE.search(text)
    if end:
        text = text[:end.start()]
    return text.strip()


@dataclass(frozen=True)
class TranscriptConfig:
    """Configuration for transcript/chat line detection.

    ``pattern`` may be supplied to describe the complete line and takes
    precedence over ``username_pattern`` and ``separator``.  The default has
    no arbitrary username-length limit and supports Unicode word characters.
    """

    pattern: str | Pattern[str] | None = None
    username_pattern: str = r"[^\W\d][\w.-]*"
    separator: str = r":"
    ignore_case: bool = False

    def compiled(self) -> Pattern[str]:
        flags = re.UNICODE | (re.IGNORECASE if self.ignore_case else 0)
        if isinstance(self.pattern, re.Pattern):
            return self.pattern
        expression = self.pattern or (
            rf"^\s*(?P<username>{self.username_pattern})\s*{self.separator}"
            rf"\s*(?P<message>.*)\s*$"
        )
        return re.compile(expression, flags)


def transcript_lines(
    text: str, config: TranscriptConfig | None = None
) -> list[re.Match[str]]:
    """Return matches for lines recognized as configured transcript entries."""

    matcher = (config or TranscriptConfig()).compiled()
    return [match for line in text.splitlines() if (match := matcher.match(line))]


def strip_transcript(
    text: str, config: TranscriptConfig | None = None
) -> tuple[str, float]:
    """Remove transcript lines and return their percentage of all word tokens."""

    matcher = (config or TranscriptConfig()).compiled()
    kept: list[str] = []
    removed_words = 0
    for line in text.splitlines():
        if matcher.match(line):
            removed_words += len(words(line))
        else:
            kept.append(line)
    total = len(words(text))
    share = 100.0 * removed_words / total if total else 0.0
    return "\n".join(kept), share


def _looks_like_single_quote_dialogue(text: str) -> bool:
    # Curly single quotes are unambiguous enough to flag. Straight single
    # quotes are flagged only as paired delimiters containing whitespace, so
    # ordinary contractions and possessives do not trigger the limitation.
    if re.search(r"\u2018[^\u2019\n]+\u2019", text):
        return True
    return bool(re.search(r"(?:^|[\s\u2014])'[^'\n]*\s+[^'\n]*'(?:\s|[.,!?]|$)", text))


def parse_quotations(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start, end, content)`` for double-quoted spans.

    Straight and curly double quotes are accepted, including multiline and
    arbitrarily long spans. Unclosed quotations run to end-of-text so their
    words are not silently assigned to narration.
    """

    spans: list[tuple[int, int, str]] = []
    opening: tuple[int, str] | None = None
    for index, char in enumerate(text):
        if opening is None:
            if char == "\u201c":
                opening = (index, "\u201d")
            elif char == '"':
                opening = (index, '"')
        elif char == opening[1]:
            begin, _ = opening
            spans.append((begin, index + 1, text[begin + 1:index]))
            opening = None
    if opening is not None:
        begin, _ = opening
        spans.append((begin, len(text), text[begin + 1:]))
    return spans


@dataclass(frozen=True)
class DialogueSplit:
    spoken: str
    narrated: str
    quotations: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def split_dialogue(text: str) -> DialogueSplit:
    """Separate double-quoted speech from narration without length cutoffs."""

    spans = parse_quotations(text)
    spoken = "\n\n".join(content for _, _, content in spans)
    narration: list[str] = []
    last = 0
    for start, end, _ in spans:
        narration.append(text[last:start])
        last = end
    narration.append(text[last:])
    warnings = (SINGLE_QUOTE_WARNING,) if _looks_like_single_quote_dialogue(text) else ()
    return DialogueSplit(spoken, " ".join(narration),
                         tuple(content for _, _, content in spans), warnings)
