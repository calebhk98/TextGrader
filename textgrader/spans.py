"""Per-sentence noun chunks and entities without re-walking the whole document.

spaCy's ``Span.noun_chunks`` and ``Span.ents`` are computed by walking the
*whole* ``Doc``'s chunks or entities and keeping the ones inside the span.
Asked once per sentence, that is quadratic in the length of the book:
measured on *Treasure Island* (69,000 words, one ``Doc``), the coherence
entity grid spent 300 of its 310 seconds in ``Span.noun_chunks``, calling
spaCy's noun-chunk iterator 4.3 million times for 3,709 sentences.

These helpers index a ``Doc``'s chunks and entities by sentence once, keep
the index in ``doc.user_data``, and return exactly what spaCy would: the
items that lie entirely inside the sentence.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Any

_CHUNKS = "textgrader.noun_chunks_by_sentence"
_ENTS = "textgrader.ents_by_sentence"


def _index(doc: Any, key: str, items_of) -> dict[int, list[Any]]:
    cached = doc.user_data.get(key)
    if cached is not None:
        return cached
    starts = [sent.start for sent in doc.sents]
    ends = {sent.start: sent.end for sent in doc.sents}
    index: dict[int, list[Any]] = {}
    for item in items_of(doc):
        position = bisect_right(starts, item.start) - 1
        if position < 0:
            continue
        sent_start = starts[position]
        # spaCy keeps an item only when it lies wholly inside the span.
        if item.end <= ends[sent_start]:
            index.setdefault(sent_start, []).append(item)
    doc.user_data[key] = index
    return index


def sentence_noun_chunks(sent: Any) -> list[Any]:
    """``list(sent.noun_chunks)``, in linear rather than quadratic total time."""

    return list(_index(sent.doc, _CHUNKS, lambda doc: doc.noun_chunks).get(sent.start, ()))


def sentence_ents(sent: Any) -> list[Any]:
    """``list(sent.ents)``, in linear rather than quadratic total time."""

    return list(_index(sent.doc, _ENTS, lambda doc: doc.ents).get(sent.start, ()))
