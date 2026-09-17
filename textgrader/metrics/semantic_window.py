"""Maximum semantic similarity to any of the previous few sentences.

``semantic_adjacent`` only compares a sentence to the one right before it, so
a paraphrase loop that skips a beat - restate, one filler sentence, restate
again - reads as low adjacent similarity even though a reader would notice
the repetition immediately. This module instead looks back over a small
window (``N`` previous sentences, configurable) and keeps the single highest
similarity found in it, for each ``N`` in ``windows``. A high maximum means
*some* recent sentence said almost the same thing; the distribution of that
maximum across the whole text is the "paraphrase loop" signature.

Backends and their honesty rules are shared with, and documented in,
``semantic_adjacent`` (see that module's docstring). Every finding here
carries the same backend note in its ``warning``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import finding, option
from .semantic_adjacent import (
    COST, FAMILY, MIN_SAMPLE, REQUIRES, UNIT_SENSITIVE, DEFAULT_MODEL,
    EVIDENCE_SNIPPET_CHARS, get_sentence_vectors, similarity_at, truncate,
)

__all__ = ["FAMILY", "COST", "REQUIRES", "MIN_SAMPLE", "UNIT_SENSITIVE", "measure"]

DEFAULT_WINDOWS = (3, 5)
HIGH_THRESHOLD = 0.8
MAX_EVIDENCE = 15


def _window_values(backend: str, vectors: Any, sentence_count: int,
                   window: int) -> list[tuple[int, int, float]]:
    """``(sentence_index, best_prior_index, max_similarity)`` for each sentence
    that has at least one predecessor inside ``window``."""

    out: list[tuple[int, int, float]] = []
    for i in range(1, sentence_count):
        lo = max(0, i - window)
        best_j, best_value = -1, -1.0
        for j in range(lo, i):
            value = similarity_at(backend, vectors, i, j)
            if value > best_value:
                best_j, best_value = j, value
        if best_j >= 0:
            out.append((i, best_j, best_value))
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    model_name = option(config, "model", DEFAULT_MODEL)
    windows = [int(n) for n in option(config, "windows", list(DEFAULT_WINDOWS)) if int(n) >= 1]
    sentences = analysis.sentences
    out: list[dict[str, Any]] = []

    if len(sentences) < 2 or not windows:
        warning = (f"needs at least two sentences and one window size; this text has "
                   f"{len(sentences)} sentences and windows={windows}")
        for window in windows or DEFAULT_WINDOWS:
            out.append(finding(f"semantic.max_similarity_prev{window}",
                               f"Maximum similarity to any of the previous {window} sentences",
                               None, "cosine", family=FAMILY, sample_size=0,
                               min_sample=MIN_SAMPLE, warning=warning))
            out.append(finding(f"semantic.paraphrase_loop_share_prev{window}",
                               f"Share of sentences restating one of the previous {window}",
                               None, "%", family=FAMILY, sample_size=0,
                               min_sample=MIN_SAMPLE, warning=warning))
        return out

    backend, vectors, note = get_sentence_vectors(analysis, model_name)

    for window in windows:
        rows = _window_values(backend, vectors, len(sentences), window)
        values = [value for _, _, value in rows]
        distribution = summarize(values) if values else {"count": 0}
        ranked = sorted(rows, key=lambda row: -row[2])[:MAX_EVIDENCE]
        evidence = [{
            "backend": backend,
            "sentence_index": i,
            "matched_sentence_index": j,
            "similarity": value,
            "sentence": truncate(sentences[i], EVIDENCE_SNIPPET_CHARS),
            "matched_sentence": truncate(sentences[j], EVIDENCE_SNIPPET_CHARS),
        } for i, j, value in ranked]
        share = 100.0 * sum(1 for value in values if value >= HIGH_THRESHOLD) / len(values) \
            if values else None

        out.append(finding(
            f"semantic.max_similarity_prev{window}",
            f"Maximum similarity to any of the previous {window} sentences",
            distribution.get("median"), "cosine", family=FAMILY, sample_size=len(values),
            distribution=distribution, evidence=evidence, min_sample=MIN_SAMPLE,
            warning=note))
        out.append(finding(
            f"semantic.paraphrase_loop_share_prev{window}",
            f"Share of sentences restating one of the previous {window}",
            share, "%", family=FAMILY, sample_size=len(values), min_sample=MIN_SAMPLE,
            warning=note, distribution={"backend": backend, "threshold": HIGH_THRESHOLD}))
    return out
