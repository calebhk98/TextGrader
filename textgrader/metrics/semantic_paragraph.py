"""Paragraph-to-paragraph semantic similarity: repetition at discourse level.

``semantic_adjacent`` and ``semantic_window`` work at sentence granularity and
therefore see only local restatement. A scene that gets re-summarized a page
later, or a chapter that keeps re-establishing the same beat in a fresh
paragraph, shows up here instead: two paragraphs can share very few sentence
pairs above threshold while still being substantially the same content.

Two findings are published:

``semantic.adjacent_paragraph_similarity``
    similarity between each paragraph and the one right after it - the
    paragraph-level analogue of ``semantic_adjacent``.
``semantic.paragraph_max_prior_similarity``
    each paragraph's maximum similarity to any *earlier* paragraph, which
    catches repetition that is not adjacent (the callback three chapters
    later that says the same thing as chapter one).

That second one is deliberately **not** a global all-pairs comparison. Even
at O(n^2) with n only in the low thousands, a book with 3,000 paragraphs is
4.5 million cosine comparisons, and on the fallback backend (a Python-level
sparse dot product) that is too slow to be a "moderate"-cost metric. Instead
each paragraph is compared only to the ``lookback`` paragraphs immediately
before it (default 50), for O(n * lookback) comparisons total. That makes the
metric **local-to-medium range, not global**: a duplicate scene more than
``lookback`` paragraphs earlier will not be found by this metric.  It is
also the metric family's only defense against a genuinely global near-copy
appearing far from where it first occurred, so a corpus profile that cares
about that should raise ``lookback`` rather than assume this catches it by
default.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import finding, option
from .semantic_adjacent import (
    COST, FAMILY, MIN_SAMPLE, REQUIRES, UNIT_SENSITIVE, DEFAULT_MODEL,
    EVIDENCE_SNIPPET_CHARS, get_paragraph_vectors, similarity_at, truncate,
)

__all__ = ["FAMILY", "COST", "REQUIRES", "MIN_SAMPLE", "UNIT_SENSITIVE", "measure"]

DEFAULT_LOOKBACK = 50
MAX_EVIDENCE = 15

_EMPTY_IDS = (
    ("semantic.adjacent_paragraph_similarity", "Adjacent-paragraph semantic similarity", "cosine"),
    ("semantic.paragraph_max_prior_similarity",
     "Paragraph's maximum similarity to an earlier paragraph", "cosine"),
)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    model_name = option(config, "model", DEFAULT_MODEL)
    lookback = max(1, int(option(config, "lookback", DEFAULT_LOOKBACK)))
    paragraphs = analysis.paragraphs
    pair_count = max(0, len(paragraphs) - 1)

    if pair_count == 0:
        warning = f"needs at least two paragraphs; this text has {len(paragraphs)}"
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE, warning=warning)
                for mid, name, unit in _EMPTY_IDS]

    backend, vectors, note = get_paragraph_vectors(analysis, model_name)

    adjacent = [similarity_at(backend, vectors, i, i + 1) for i in range(pair_count)]
    adjacent_summary = summarize(adjacent)
    adjacent_ranked = sorted(range(pair_count), key=lambda i: -adjacent[i])[:MAX_EVIDENCE]
    adjacent_evidence = [{
        "backend": backend,
        "paragraph_index": i,
        "similarity": adjacent[i],
        "paragraph_a": truncate(paragraphs[i], EVIDENCE_SNIPPET_CHARS),
        "paragraph_b": truncate(paragraphs[i + 1], EVIDENCE_SNIPPET_CHARS),
    } for i in adjacent_ranked]

    # Skip paragraph 0: it has no earlier paragraph to compare against.
    max_prior: list[float] = []
    max_prior_match: list[int] = []
    for i in range(1, len(paragraphs)):
        lo = max(0, i - lookback)
        best_j, best_value = -1, -1.0
        for j in range(lo, i):
            value = similarity_at(backend, vectors, i, j)
            if value > best_value:
                best_j, best_value = j, value
        max_prior.append(best_value)
        max_prior_match.append(best_j)

    prior_summary = summarize(max_prior) if max_prior else {"count": 0}
    prior_ranked = sorted(range(len(max_prior)), key=lambda k: -max_prior[k])[:MAX_EVIDENCE]
    prior_evidence = [{
        "backend": backend,
        "paragraph_index": k + 1,
        "matched_paragraph_index": max_prior_match[k],
        "gap_paragraphs": (k + 1) - max_prior_match[k],
        "similarity": max_prior[k],
        "paragraph": truncate(paragraphs[k + 1], EVIDENCE_SNIPPET_CHARS),
        "matched_paragraph": truncate(paragraphs[max_prior_match[k]], EVIDENCE_SNIPPET_CHARS),
    } for k in prior_ranked]

    return [
        finding("semantic.adjacent_paragraph_similarity", "Adjacent-paragraph semantic similarity",
                adjacent_summary.get("median"), "cosine", family=FAMILY, sample_size=pair_count,
                distribution=adjacent_summary, evidence=adjacent_evidence, min_sample=MIN_SAMPLE,
                warning=note),
        finding("semantic.paragraph_max_prior_similarity",
                "Paragraph's maximum similarity to an earlier paragraph",
                prior_summary.get("median"), "cosine", family=FAMILY, sample_size=len(max_prior),
                distribution=prior_summary, evidence=prior_evidence, min_sample=MIN_SAMPLE,
                warning=f"{note}; compared only against the previous {lookback} paragraphs "
                        f"(local-to-medium range, not a global duplicate search)"),
    ]
