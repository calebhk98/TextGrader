"""Runs of consecutive one-sentence paragraphs.

A single-sentence paragraph is not a problem by itself; used for emphasis it
is one of the oldest tricks in prose.  The problem is a run of them: page
after page of one-line paragraphs is a strong, easy-to-produce tell of
generated text, where every beat gets its own paragraph break regardless of
whether the content earns one.  This module measures the fact directly rather
than through a paragraph-length average, which a handful of long paragraphs
elsewhere in the text can dilute past recognition.

Reuses :func:`textgrader.stats.run_lengths` on a two-value label ("single" or
"multi") so the run-length distribution is computed the same way sentence-band
runs are in :mod:`rhythm_runs`.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import run_lengths, summarize
from .common import FAST, finding, rate

FAMILY = "paragraph_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

MAX_EVIDENCE = 8
SNIPPET_CHARS = 90


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")


def _runs_with_start(labels: Sequence[str]) -> list[tuple[str, int, int]]:
    """``(label, start_index, length)`` for every maximal run, in order."""

    out: list[tuple[str, int, int]] = []
    if not labels:
        return out
    current, start = labels[0], 0
    for index in range(1, len(labels)):
        if labels[index] != current:
            out.append((current, start, index - start))
            current, start = labels[index], index
    out.append((current, start, len(labels) - start))
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    counts = analysis.paragraph_sentence_counts
    total = len(counts)
    if total == 0:
        warning = "no paragraphs to measure"
        return [
            finding("rhythm.single_sentence_paragraph_share", "Share of one-sentence paragraphs",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.single_sentence_paragraph_run_length",
                    "Run length of consecutive one-sentence paragraphs",
                    None, "paragraphs", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.single_sentence_paragraph_max_run",
                    "Longest run of consecutive one-sentence paragraphs",
                    None, "paragraphs", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.single_sentence_paragraph_in_run_share",
                    "Share of paragraphs inside a run of 3+ one-sentence paragraphs",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    labels = ["single" if count == 1 else "multi" for count in counts]
    single_count = labels.count("single")
    distribution = run_lengths(labels)
    single_runs = distribution.get("single", [])
    run_summary = summarize(single_runs) if single_runs else {"count": 0}

    runs = [(start, length) for label, start, length in _runs_with_start(labels)
            if label == "single"]
    longest = sorted(runs, key=lambda item: -item[1])[:MAX_EVIDENCE]
    evidence = [{"run_length": length, "start_paragraph_index": start,
                 "text": _snippet(analysis.paragraphs[start])}
                for start, length in longest]

    in_run = sum(length for length in single_runs if length >= 3)

    return [
        finding("rhythm.single_sentence_paragraph_share", "Share of one-sentence paragraphs",
                rate(single_count, total), "percent", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE, distribution={"single_count": single_count}),
        finding("rhythm.single_sentence_paragraph_run_length",
                "Run length of consecutive one-sentence paragraphs",
                run_summary.get("mean"), "paragraphs", family=FAMILY,
                sample_size=len(single_runs),
                distribution={**run_summary, "headline": "mean: these are small whole-number counts, whose median sits on one value for nearly every book"}, min_sample=MIN_SAMPLE,
                evidence=evidence,
                warning=None if single_runs else "no one-sentence paragraphs in this text"),
        finding("rhythm.single_sentence_paragraph_max_run",
                "Longest run of consecutive one-sentence paragraphs",
                max(single_runs) if single_runs else None, "paragraphs", family=FAMILY,
                sample_size=len(single_runs), min_sample=MIN_SAMPLE,
                warning=None if single_runs else "no one-sentence paragraphs in this text"),
        finding("rhythm.single_sentence_paragraph_in_run_share",
                "Share of paragraphs inside a run of 3+ one-sentence paragraphs",
                rate(in_run, total), "percent", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE, distribution={"paragraphs_in_runs_of_3plus": in_run}),
    ]
