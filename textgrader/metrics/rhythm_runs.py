"""Runs of same-band sentence length: the better version of a "short runs" count.

Banding every sentence short/medium/long and then asking how long each band's
runs are is a stronger test for monotony than a single "average sentence
length" or even the lag-1 autocorrelation in :mod:`rhythm_autocorrelation`.
Autocorrelation can be pulled toward zero by a handful of genuinely varied
sentences even while most of the text sits in a five-sentence stretch of
short, clipped lines; the run-length distribution per band shows that stretch
directly, and the "share of sentences inside a run of 3+" turns it into one
number a reader can act on without inspecting the whole distribution.

Bands are cut with two configurable thresholds (``short_max``, ``long_min``)
rather than a fixed vocabulary, because "short" means something different in
a thriller written in clipped sentences than in a discursive literary novel;
the thresholds are still explicit here, in the finding's evidence, so nothing
is hidden in the code.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..stats import band_labels, run_lengths, summarize
from .common import FAST, finding, option, rate

FAMILY = "sentence_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

BANDS = ("short", "medium", "long")
# Longest runs kept as evidence per band; short lists an agent can act on.
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
    short_max = float(option(config, "short_max", 8))
    long_min = float(option(config, "long_min", 25))
    lengths = analysis.sentence_lengths
    total = len(lengths)
    if total == 0:
        warning = "no sentences to band"
        out: list[dict[str, Any]] = []
        for band in BANDS:
            out.append(finding(f"rhythm.band_share_{band}", f"Share of sentences in the {band} band",
                                None, "percent", family=FAMILY, sample_size=0,
                                min_sample=MIN_SAMPLE, warning=warning))
            out.append(finding(f"rhythm.run_length_{band}", f"Run length of consecutive {band} sentences",
                                None, "sentences", family=FAMILY, sample_size=0,
                                min_sample=MIN_SAMPLE, warning=warning))
            out.append(finding(f"rhythm.in_run_share_{band}",
                                f"Share of sentences inside a run of 3+ {band} sentences",
                                None, "percent", family=FAMILY, sample_size=0,
                                min_sample=MIN_SAMPLE, warning=warning))
            out.append(finding(f"rhythm.max_run_{band}", f"Longest run of {band} sentences",
                                None, "sentences", family=FAMILY, sample_size=0,
                                min_sample=MIN_SAMPLE, warning=warning))
        return out

    # cuts are upper bounds: <= short_max is "short", <= long_min - 1 is
    # "medium", anything left over falls into the last name, "long".
    labels = band_labels(lengths, [short_max, long_min - 1], list(BANDS))
    distribution = run_lengths(labels)
    runs = _runs_with_start(labels)
    band_counts = {band: labels.count(band) for band in BANDS}
    runs_by_band: dict[str, list[tuple[int, int]]] = {band: [] for band in BANDS}
    for label, start, length in runs:
        runs_by_band[label].append((start, length))

    out = []
    for band in BANDS:
        share = rate(band_counts[band], total)
        out.append(finding(
            f"rhythm.band_share_{band}", f"Share of sentences in the {band} band",
            share, "percent", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
            distribution={"short_max": short_max, "long_min": long_min, "count": band_counts[band]}))

        band_runs = distribution.get(band, [])
        run_summary = summarize(band_runs) if band_runs else {"count": 0}
        longest = sorted(runs_by_band[band], key=lambda item: -item[1])[:MAX_EVIDENCE]
        evidence = [{"band": band, "run_length": length, "start_sentence_index": start,
                     "text": _snippet(analysis.sentences[start])}
                    for start, length in longest]
        out.append(finding(
            f"rhythm.run_length_{band}", f"Run length of consecutive {band} sentences",
            run_summary.get("mean"), "sentences", family=FAMILY,
            sample_size=len(band_runs),
            distribution={**run_summary, "headline": "mean: these are small whole-number counts, whose median sits on one value for nearly every book"}, min_sample=MIN_SAMPLE,
            evidence=evidence,
            warning=None if band_runs else f"no {band}-band sentences in this text"))

        in_run = sum(length for length in band_runs if length >= 3)
        out.append(finding(
            f"rhythm.in_run_share_{band}", f"Share of sentences inside a run of 3+ {band} sentences",
            rate(in_run, total), "percent", family=FAMILY, sample_size=total,
            min_sample=MIN_SAMPLE, distribution={"sentences_in_runs_of_3plus": in_run}))

        out.append(finding(
            f"rhythm.max_run_{band}", f"Longest run of {band} sentences",
            max(band_runs) if band_runs else None, "sentences", family=FAMILY,
            sample_size=len(band_runs), min_sample=MIN_SAMPLE,
            warning=None if band_runs else f"no {band}-band sentences in this text"))
    return out
