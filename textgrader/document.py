"""The single place a word, sentence, paragraph, quotation or clean text is decided.

Before this module every metric made its own choices.  Core analysis stripped
Gutenberg boilerplate and Markdown headings while the optional metrics received
the raw file, so two numbers in one report could describe two different
documents.  Sentence segmentation was a regular expression in ``text.py`` for
most metrics and pySBD for one.  Dialogue was re-parsed, differently, in four
places.

:class:`DocumentAnalysis` fixes that by computing the pipeline once::

    raw text
      -> canonical cleanup      (Gutenberg, headings, transcripts, quotes)
      -> tokens / sentences / paragraphs / quotations / dialogue / narration
      -> spaCy Doc (optional, chunked, lazy)
      -> every metric

Everything is cached, so a metric that wants narration-only sentence lengths
costs a list comprehension rather than another parse.  ``analysis.dialogue``
and ``analysis.narration`` are themselves :class:`DocumentAnalysis` objects, so
any metric written against the general interface can be pointed at one channel.

Performance, for the AI agents that run this repeatedly: the dependency-free
part of the pipeline is linear and handles a 300,000-word novel in roughly a
second.  spaCy is not: see :meth:`DocumentAnalysis.spacy_docs`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import text as textlib
from .optional import require

# Paragraphs that are only a scene-break marker are not prose.
MARKER_RE = re.compile(r"^[\s*_=—-]+$")

#: Markdown headings, and the plain-text chapter headings a .txt novel uses.
#: Without the second form every Project Gutenberg book is one section, so a
#: chapter-level comparison has no chapters to build from and the book-drift
#: metrics fall back to fixed windows on every corpus text.
SECTION_RE = re.compile(
    r"(?m)^(?:"
    r"[ \t]{0,3}\#{1,6}[ \t]+(?P<atx>[^\n]*)"
    r"|(?P<setext>[^\n]*\S[^\n]*)\n[ \t]{0,3}(?:=+|-+)[ \t]*"
    r"|[ \t]{0,3}(?P<plain>(?:CHAPTER|Chapter|BOOK|Book|PART|Part)"
    r"[ \t]+[IVXLCDM0-9][^\n]{0,80})"
    r")[ \t]*$")

COMPARISON_UNITS = ("book", "chapter", "scene", "passage", "unknown")

#: The core size metrics: these ARE how much text there is.
SCALE_DEPENDENT_IDS = frozenset({
    "_words", "_sentences", "_paragraphs",
    "word_count", "sentence_count", "paragraph_count",
    "prose.words", "prose.sentences", "prose.paragraphs",
})


@dataclass(frozen=True)
class TextProcessing:
    """What "the text" means for this run.  Recorded in corpus profiles.

    Grading and profile building must agree on every field here or the numbers
    describe different documents.  :meth:`fingerprint` is what gets compared.
    """

    strip_gutenberg: bool = True
    strip_markdown_headings: bool = True
    strip_transcript: bool = True
    drop_marker_paragraphs: bool = True
    normalize_quotes: bool = False
    #: ``auto`` prefers pySBD, then spaCy, then the built-in splitter.
    segmenter: str = "auto"
    language: str = "en"
    #: pySBD's own whitespace/quote cleanup.  Off: it rewrites offsets.
    segmenter_clean: bool = False
    transcript: Mapping[str, Any] | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "TextProcessing":
        settings = dict(config or {})
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(settings) - known)
        if unknown:
            raise ValueError(
                f"unknown text_processing options: {', '.join(unknown)}; "
                f"valid options are {', '.join(sorted(known))}")
        return cls(**settings)

    def transcript_config(self) -> textlib.TranscriptConfig:
        return textlib.TranscriptConfig(**dict(self.transcript or {}))

    def fingerprint(self) -> dict[str, Any]:
        """The settings a corpus profile must match, as plain JSON."""

        return {
            "strip_gutenberg": self.strip_gutenberg,
            "strip_markdown_headings": self.strip_markdown_headings,
            "strip_transcript": self.strip_transcript,
            "drop_marker_paragraphs": self.drop_marker_paragraphs,
            "normalize_quotes": self.normalize_quotes,
            "segmenter": self.segmenter,
            "language": self.language,
            "segmenter_clean": self.segmenter_clean,
            "transcript": dict(self.transcript) if self.transcript else None,
        }


@dataclass(frozen=True)
class NlpSettings:
    model: str = "en_core_web_sm"
    #: Pipeline components to switch off.  ``ner`` roughly halves the cost and
    #: only the entity metrics need it, so it is off unless asked for.
    disable: tuple[str, ...] = ("ner",)
    #: spaCy's own default limit is 1,000,000 characters; we chunk below it.
    max_chars_per_chunk: int = 400_000
    #: Refuse to parse beyond this many words rather than stalling an agent.
    max_words: int | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "NlpSettings":
        settings = dict(config or {})
        disable = settings.pop("disable", None)
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(settings) - known)
        if unknown:
            raise ValueError(f"unknown nlp options: {', '.join(unknown)}")
        item = cls(**settings)
        if disable is not None:
            item = replace(item, disable=tuple(disable))
        return item


class Segmenter:
    """Resolve the sentence splitter once, then reuse it.

    Building a ``pysbd.Segmenter`` per paragraph made segmentation the single
    most expensive step in the pipeline; it is built once per run here.
    """

    __slots__ = ("wanted", "language", "clean", "used", "warning", "_impl")

    def __init__(self, processing: "TextProcessing") -> None:
        self.wanted = processing.segmenter
        self.language = processing.language
        self.clean = processing.segmenter_clean
        self.used = "builtin"
        self.warning: str | None = None
        self._impl = None
        if self.wanted in ("auto", "pysbd"):
            pysbd, reason = require("pysbd")
            if pysbd is not None:
                try:
                    self._impl = pysbd.Segmenter(language=self.language, clean=self.clean)
                    self.used = "pysbd"
                except Exception as exc:  # pragma: no cover - pySBD language guard
                    reason = f"pysbd failed ({type(exc).__name__}: {exc})"
            if self._impl is None and self.wanted == "pysbd":
                self.warning = reason
        elif self.wanted == "spacy":
            self.warning = ("segmenter='spacy' applies to the parsed document; the "
                            "built-in splitter was used for the dependency-free views")
        elif self.wanted != "builtin":
            self.warning = f"unknown segmenter {self.wanted!r}; used the built-in splitter"

    def split(self, body: str) -> list[str]:
        if self._impl is None:
            return textlib.sentences(body)
        try:
            found = [item.strip() for item in self._impl.segment(body)]
        except Exception:  # pragma: no cover - pySBD input guard
            return textlib.sentences(body)
        return [item for item in found if textlib.words(item)]


@dataclass
class DocumentAnalysis:
    """One text, prepared once, shared by every metric.

    Construct with :meth:`from_text` or :meth:`from_path`; the constructor
    takes the already-canonical body so that derived views (dialogue,
    narration, a chapter slice) do not re-run cleanup on text that has had it.
    """

    text: str
    raw: str = ""
    source: str | None = None
    processing: TextProcessing = field(default_factory=TextProcessing)
    nlp_settings: NlpSettings = field(default_factory=NlpSettings)
    comparison_unit: str = "unknown"
    #: Word share of the original that was removed as transcript/chat lines.
    transcript_share: float = 0.0
    raw_word_count: int = 0
    warnings: tuple[str, ...] = ()
    segmenter: str = "builtin"
    _nlp: Any = None
    _channel: str = "full"
    _segmenter: Any = None
    #: Memo for :meth:`windows`, so three book-drift metrics asking for the
    #: same slicing pay one segmentation between them rather than three.
    _windows: dict = field(default_factory=dict)
    #: Free-form per-document memo shared between cooperating metrics.
    _shared: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ build

    @classmethod
    def from_text(cls, raw: str, *, processing: TextProcessing | None = None,
                  nlp_settings: NlpSettings | None = None, source: str | None = None,
                  comparison_unit: str = "unknown", nlp: Any = None) -> "DocumentAnalysis":
        processing = processing or TextProcessing()
        body = raw
        if processing.strip_gutenberg:
            body = textlib.strip_gutenberg(body)
        if processing.strip_markdown_headings:
            body = textlib.remove_markdown_headings(body)
        share = 0.0
        if processing.strip_transcript:
            body, share = textlib.strip_transcript(body, processing.transcript_config())
        if processing.normalize_quotes:
            body = textlib.normalize_quotes(body, single=False)
        segmenter = Segmenter(processing)
        return cls(text=body, raw=raw, source=source, processing=processing,
                   nlp_settings=nlp_settings or NlpSettings(),
                   comparison_unit=comparison_unit if comparison_unit in COMPARISON_UNITS else "unknown",
                   transcript_share=share, raw_word_count=len(textlib.words(raw)),
                   warnings=(segmenter.warning,) if segmenter.warning else (),
                   segmenter=segmenter.used, _nlp=nlp, _segmenter=segmenter)

    @classmethod
    def from_path(cls, path: str | Path, **kwargs: Any) -> "DocumentAnalysis":
        path = Path(path)
        kwargs.setdefault("source", str(path))
        return cls.from_text(path.read_text(encoding="utf-8"), **kwargs)

    def derive(self, body: str, channel: str) -> "DocumentAnalysis":
        """A view over part of this document, sharing settings and the parser.

        Cleanup is deliberately not repeated: ``body`` is already canonical.
        """

        return DocumentAnalysis(
            text=body, raw=body, source=self.source, processing=self.processing,
            nlp_settings=self.nlp_settings, comparison_unit=self.comparison_unit,
            transcript_share=0.0, raw_word_count=len(textlib.words(body)),
            warnings=self.warnings, segmenter=self.segmenter, _nlp=self._nlp,
            _channel=channel, _segmenter=self._sentence_segmenter)

    @property
    def channel(self) -> str:
        """``full``, ``dialogue``, ``narration`` or a caller-chosen slice name."""

        return self._channel

    @property
    def _sentence_segmenter(self) -> Segmenter:
        if self._segmenter is None:
            self._segmenter = Segmenter(self.processing)
        return self._segmenter

    # ------------------------------------------------------- canonical tokens

    @cached_property
    def words(self) -> list[str]:
        """Unicode word tokens of the canonical text, original case."""

        return textlib.words(self.text)

    @cached_property
    def tokens(self) -> list[str]:
        """Lower-cased words with curly apostrophes folded, for counting."""

        return [word.lower().replace("’", "'") for word in self.words]

    @cached_property
    def paragraphs(self) -> list[str]:
        found = textlib.paragraphs(self.text)
        if self.processing.drop_marker_paragraphs:
            found = [item for item in found if not MARKER_RE.match(item)]
        return [item for item in found if textlib.words(item)]

    @cached_property
    def sentences_by_paragraph(self) -> list[list[str]]:
        """Sentences grouped by the paragraph they came from.

        This is the primitive.  Segmentation runs inside each paragraph so that
        a missing full stop at the end of one paragraph cannot glue it to the
        next, and both the flat sentence list and the per-paragraph sentence
        counts are views of this one pass: asking for the counts used to
        re-segment the whole document a second time.
        """

        splitter = self._sentence_segmenter
        return [splitter.split(paragraph) for paragraph in self.paragraphs]

    @cached_property
    def sentences(self) -> list[str]:
        """Sentences from the configured segmenter, never a per-metric regex."""

        return [item for group in self.sentences_by_paragraph for item in group]

    @cached_property
    def sentence_lengths(self) -> list[int]:
        return [len(textlib.words(item)) for item in self.sentences]

    @cached_property
    def paragraph_lengths(self) -> list[int]:
        return [len(textlib.words(item)) for item in self.paragraphs]

    @cached_property
    def paragraph_sentence_counts(self) -> list[int]:
        return [len(group) for group in self.sentences_by_paragraph]

    @cached_property
    def word_count(self) -> int:
        return len(self.words)

    @cached_property
    def sentence_count(self) -> int:
        return len(self.sentences)

    @cached_property
    def paragraph_count(self) -> int:
        return len(self.paragraphs)

    # -------------------------------------------------------------- dialogue

    @cached_property
    def dialogue_split(self) -> textlib.DialogueSplit:
        return textlib.split_dialogue(self.text)

    @cached_property
    def quotation_spans(self) -> list[tuple[int, int, str]]:
        """``(start, end, content)`` for every double-quoted span, in order."""

        return textlib.parse_quotations(self.text)

    @cached_property
    def quotations(self) -> list[str]:
        return [content for _, _, content in self.quotation_spans]

    @cached_property
    def turns(self) -> list[str]:
        """Spoken turns, rejoining a quotation split by its own attribution.

        ``"A," she says, "B."`` is one turn; ``"A," he says. "B," she says.``
        is two.  The difference is whether the narration between the spans
        closes a sentence, which is the only signal available without knowing
        who is speaking.
        """

        out: list[str] = []
        current: list[str] = []
        last = 0
        for start, end, content in self.quotation_spans:
            gap = self.text[last:start]
            if current and re.search(r"[.!?…]", gap):
                out.append(" ".join(current))
                current = []
            current.append(content)
            last = end
        if current:
            out.append(" ".join(current))
        return [item for item in out if textlib.words(item)]

    @cached_property
    def dialogue(self) -> "DocumentAnalysis":
        """A document made only of spoken text.  Turns are paragraph-separated
        so the sentence splitter cannot run two speakers together."""

        return self.derive("\n\n".join(self.turns), "dialogue")

    @cached_property
    def narration(self) -> "DocumentAnalysis":
        """A document made only of the text outside quotation marks."""

        pieces: list[str] = []
        last = 0
        for start, end, _ in self.quotation_spans:
            pieces.append(self.text[last:start])
            last = end
        pieces.append(self.text[last:])
        return self.derive("".join(pieces), "narration")

    @cached_property
    def dialogue_word_share(self) -> float | None:
        spoken = len(textlib.words(self.dialogue.text))
        total = self.word_count
        return 100.0 * spoken / total if total else None

    def in_dialogue(self, offset: int) -> bool:
        """Whether a character offset in :attr:`text` falls inside a quotation."""

        for start, end, _ in self.quotation_spans:
            if start <= offset < end:
                return True
            if offset < start:
                break
        return False

    # ------------------------------------------------------------- segments

    @cached_property
    def sections(self) -> list[tuple[str, "DocumentAnalysis"]]:
        """``(title, view)`` per Markdown-headed section of the original text.

        Headings are removed by the canonical cleanup, so the split is taken
        from :attr:`raw` before cleanup and each part is then cleaned on its
        own.  A document with no headings yields one untitled section, which is
        what lets a book-level metric fall back to :meth:`windows`.
        """

        marks = [(match.start(),
                  (match.group("atx") or match.group("setext")
                   or match.group("plain") or "").strip(),
                  match.end()) for match in SECTION_RE.finditer(self.raw)]
        if not marks:
            return [("", self)]
        out: list[tuple[str, DocumentAnalysis]] = []
        preamble = self.raw[:marks[0][0]]
        if textlib.words(preamble):
            out.append(("", self._section("", preamble)))
        for index, (_, title, end) in enumerate(marks):
            stop = marks[index + 1][0] if index + 1 < len(marks) else len(self.raw)
            body = self.raw[end:stop]
            if textlib.words(body):
                out.append((title, self._section(title, body)))
        return out or [("", self)]

    def _section(self, title: str, body: str) -> "DocumentAnalysis":
        item = DocumentAnalysis.from_text(
            body, processing=self.processing, nlp_settings=self.nlp_settings,
            source=f"{self.source or ''}#{title}" if title else self.source,
            comparison_unit=self.comparison_unit, nlp=self._nlp)
        item._segmenter = self._sentence_segmenter
        item._channel = f"section:{title}" if title else "section"
        return item

    def windows(self, words_per_window: int = 2000) -> list["DocumentAnalysis"]:
        """Consecutive fixed-size views, for drift measurement without headings.

        Windows break on paragraph boundaries so a rolling comparison is never
        measuring half a paragraph against half of another.
        """

        size = max(200, int(words_per_window))
        if size in self._windows:
            return self._windows[size]
        out: list[DocumentAnalysis] = []
        current: list[int] = []
        count = 0
        for index, length in enumerate(self.paragraph_lengths):
            current.append(index)
            count += length
            if count >= size:
                out.append(self._paragraph_slice(current, f"window:{len(out)}"))
                current, count = [], 0
        if current and (count >= size // 4 or not out):
            out.append(self._paragraph_slice(current, f"window:{len(out)}"))
        self._windows[size] = out
        return out

    def _paragraph_slice(self, indices: Sequence[int], channel: str) -> "DocumentAnalysis":
        """A view over whole paragraphs of this document, already segmented.

        A window is a contiguous run of paragraphs this document has already
        split into sentences, so the slice inherits that work.  Re-segmenting
        it was the single largest cost in the book-drift metrics, which slice
        the whole document into a hundred and fifty windows.
        """

        paragraphs = [self.paragraphs[index] for index in indices]
        view = self.derive("\n\n".join(paragraphs), channel)
        view.__dict__["paragraphs"] = paragraphs
        view.__dict__["sentences_by_paragraph"] = [self.sentences_by_paragraph[index]
                                                   for index in indices]
        return view

    def memo(self, key: str, build):
        """Per-document cache shared between metrics that need the same work.

        Only for values derived purely from this document, so that two metrics
        asking the same question of the same text cannot get two answers.
        """

        if key not in self._shared:
            self._shared[key] = build()
        return self._shared[key]

    # ------------------------------------------------------------------ spaCy

    @cached_property
    def _pipeline(self) -> tuple[Any, str | None]:
        if self._nlp is not None:
            return self._nlp, None
        spacy, reason = require("spacy")
        if spacy is None:
            return None, reason
        try:
            return spacy.load(self.nlp_settings.model,
                              disable=list(self.nlp_settings.disable)), None
        except Exception as exc:
            return None, (f"spaCy model {self.nlp_settings.model!r} unavailable "
                          f"({type(exc).__name__}: {exc}); run "
                          f"python -m spacy download {self.nlp_settings.model}")

    @property
    def nlp(self) -> Any:
        return self._pipeline[0]

    @property
    def nlp_unavailable(self) -> str | None:
        """Why there is no parse, or ``None`` when there is one."""

        pipeline, reason = self._pipeline
        if pipeline is None:
            return reason or "no NLP pipeline"
        limit = self.nlp_settings.max_words
        if limit and self.word_count > limit:
            return (f"document is {self.word_count:,} words, above the configured "
                    f"nlp.max_words of {limit:,}")
        return None

    @cached_property
    def _chunks(self) -> list[tuple[int, str]]:
        """Paragraph-aligned slices under the per-chunk character limit."""

        limit = max(1000, self.nlp_settings.max_chars_per_chunk)
        if len(self.text) <= limit:
            return [(0, self.text)]
        out: list[tuple[int, str]] = []
        start = 0
        while start < len(self.text):
            end = min(start + limit, len(self.text))
            if end < len(self.text):
                split = self.text.rfind("\n\n", start, end)
                if split <= start:
                    split = self.text.rfind("\n", start, end)
                if split <= start:
                    split = self.text.rfind(" ", start, end)
                if split > start:
                    end = split
            out.append((start, self.text[start:end]))
            start = end
        return out

    @cached_property
    def _docs(self) -> list[tuple[int, Any]]:
        pipeline = self.nlp
        if pipeline is None or self.nlp_unavailable:
            return []
        offsets = [offset for offset, _ in self._chunks]
        try:
            parsed = list(pipeline.pipe([body for _, body in self._chunks]))
        except Exception:  # pragma: no cover - model/runtime failure
            return []
        return list(zip(offsets, parsed))

    def spacy_docs(self) -> list[tuple[int, Any]]:
        """``(character_offset, Doc)`` for each chunk, or ``[]`` without spaCy.

        The document is chunked at paragraph boundaries because spaCy refuses
        inputs over a million characters and its memory use is superlinear.
        Each ``Doc`` therefore carries the offset of its chunk within
        :attr:`text`; add it to ``token.idx`` for a document-wide offset.

        Cost, measured on ``en_core_web_sm`` with ``ner`` disabled: about
        7,000-12,000 words per second on one core.  A 300,000-word novel is
        therefore roughly 30-45 seconds, and well over a minute with ``ner``
        enabled.  Nothing else in TextGrader is remotely that expensive; every
        metric that calls this is marked in the README's cost table.
        """

        return self._docs

    def spacy_sents(self) -> Iterator[Any]:
        for _, doc in self.spacy_docs():
            yield from doc.sents

    def spacy_tokens(self) -> Iterator[Any]:
        for _, doc in self.spacy_docs():
            yield from doc

    def token_offset(self, doc_offset: int, token: Any) -> int:
        return doc_offset + token.idx

    def dialogue_char_share(self, start: int, end: int) -> float:
        """Share of the characters in ``[start, end)`` that sit inside a quotation."""

        if end <= start:
            return 0.0
        inside = 0
        for open_at, close_at, _ in self.quotation_spans:
            if close_at <= start:
                continue
            if open_at >= end:
                break
            inside += min(end, close_at) - max(start, open_at)
        return inside / (end - start)

    def spacy_sents_by_channel(self, threshold: float = 0.5):
        """``(sentence_span, channel, offset)`` over the ONE shared parse.

        A narration-only parse metric used to ask for ``analysis.narration``
        and parse it again, which doubled the most expensive step in the tool
        for one extra number.  Classifying the sentences of the shared parse by
        how much of each one lies inside quotation marks costs nothing and
        answers the same question.  A sentence that is part speech and part
        attribution is reported as ``mixed`` rather than forced into a channel.
        """

        for offset, doc in self.spacy_docs():
            for sent in doc.sents:
                start, end = offset + sent.start_char, offset + sent.end_char
                share = self.dialogue_char_share(start, end)
                channel = ("dialogue" if share >= threshold
                           else "narration" if share <= 0.15 else "mixed")
                yield sent, channel, offset

    # ------------------------------------------------------------- provenance

    def describe(self) -> dict[str, Any]:
        """What was analyzed, for the report header and for corpus matching."""

        return {
            "source": self.source,
            "channel": self._channel,
            "comparison_unit": self.comparison_unit,
            "segmenter": self.segmenter,
            "text_processing": self.processing.fingerprint(),
            "raw_words": self.raw_word_count,
            "analyzed_words": self.word_count,
            "sentences": self.sentence_count,
            "paragraphs": self.paragraph_count,
            "transcript_word_share": self.transcript_share,
            "dialogue_word_share": self.dialogue_word_share,
            "warnings": list(self.warnings) + list(self.dialogue_split.warnings),
        }


#: What ``unknown`` means.  Both a corpus profile of complete texts and an
#: unqualified input document are one whole thing, so they compare; a caller
#: who is grading parts says so with ``analysis.comparison_unit``.
DEFAULT_COMPARISON_UNIT = "book"


def resolve_unit(unit: str) -> str:
    return unit if unit in COMPARISON_UNITS and unit != "unknown" else DEFAULT_COMPARISON_UNIT


def is_scale_dependent(metric_id: str) -> bool:
    """Whether the raw value grows with how much text you supply.

    Counts do; rates, ratios and shares do not, which is the entire reason to
    express a measurement as a rate.
    """

    # Matched by id, not by suffix.  A loose "ends with _words" test caught
    # every ``*_per_1000_words`` rate, which are the most carefully normalized
    # measurements in the tool, and refused to compare them across units.
    return (metric_id in SCALE_DEPENDENT_IDS
            or metric_id.endswith(("_count", "_total")))


def units_comparable(metric_id: str, document_unit: str, corpus_unit: str) -> bool:
    """Whether a scale-dependent metric may be compared across these units.

    Only the size metrics are refused: a chapter's word count against a corpus
    of novels is a fact about how books are divided, not about the chapter.

    Everything else compares freely, because everything else is a rate, and a
    rate does not care how much text produced it.  More text makes a rate a
    better estimate, not a different number, which is why a corpus of short
    works is a perfectly good corpus.  What a rate DOES depend on is how much
    text each observation had: the centre is the same at any granularity, but
    the spread is not, so an outlier threshold has to come from a corpus built
    at the granularity you are grading.  ``--split-sections`` builds one.

    An unspecified unit on either side resolves to ``book``.
    """

    if not is_scale_dependent(metric_id):
        return True
    return resolve_unit(document_unit) == resolve_unit(corpus_unit)
