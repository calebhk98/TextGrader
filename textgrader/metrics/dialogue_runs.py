"""Consecutive spoken turns with no narration between them: scene pacing.

A run of turns with no intervening narration sentence is a rapid exchange:
"Yes." / "No." / "Really?" with nothing but paragraph breaks between them.
Long runs read fast; a manuscript with none at all, or one dominated by a
single very long run, is worth flagging either way, which is why the run
LENGTH DISTRIBUTION is published (via ``stats.run_lengths``) rather than a
mean run length that would blur "mostly two-line exchanges with one twelve-
turn argument" into an unremarkable-looking average.

"No narration between them" is judged the strict way: the text between two
consecutive turns must contain zero words (``textlib.words``), not merely
lack a recognized speech verb.  A wordless gap is unambiguous; anything with
even one word of narration, however brief, breaks the run.  Turn offsets
come from ``dialogue_attribution.turn_spans``, which keeps the character
positions ``analysis.turns`` itself discards.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from .. import text as textlib
from ..stats import run_lengths, summarize
from .common import FAST, finding, rate
from .dialogue_attribution import turn_spans

FAMILY = "dialogue"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

# A run of at least this many turns is where an exchange starts reading as a
# sustained volley rather than an isolated back-and-forth.
RUN_SHARE_THRESHOLD = 3
EVIDENCE_LIMIT = 10


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    text = analysis.text
    spans = turn_spans(analysis)
    total = len(spans)
    if total == 0:
        warning = "no spoken turns found"
        return [
            finding("dialogue.unnarrated_run_length", "Unnarrated dialogue run length", None,
                    "turns", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.max_unnarrated_run", "Longest unnarrated dialogue run", None,
                    "turns", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.in_unnarrated_run_share",
                    f"Share of turns inside a run of {RUN_SHARE_THRESHOLD}+ unnarrated turns",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    # A new run id starts wherever the gap to the previous turn has any words
    # in it; stats.run_lengths then just counts how long each id's block is.
    labels = [0]
    run_id = 0
    for index in range(1, total):
        gap = text[spans[index - 1]["end"]:spans[index]["start"]]
        if textlib.words(gap):
            run_id += 1
        labels.append(run_id)

    lengths_by_id = run_lengths(labels)
    run_length_list = [length for lengths in lengths_by_id.values() for length in lengths]

    runs: list[tuple[int, int]] = []
    run_start = 0
    for index in range(1, total + 1):
        if index == total or labels[index] != labels[run_start]:
            runs.append((run_start, index))
            run_start = index

    longest = max(run_length_list) if run_length_list else None
    in_run_turns = sum(length for length in run_length_list if length >= RUN_SHARE_THRESHOLD)
    in_run_share = rate(in_run_turns, total)

    top_runs = sorted(runs, key=lambda pair: pair[1] - pair[0], reverse=True)[:EVIDENCE_LIMIT]
    evidence = [{"offset": spans[start]["start"], "length": end - start,
                 "first_turn": spans[start]["text"][:80]} for start, end in top_runs]

    summary = summarize(run_length_list)
    return [
        finding("dialogue.unnarrated_run_length", "Unnarrated dialogue run length",
                summary.get("median"), "turns", family=FAMILY, sample_size=len(run_length_list),
                distribution=summary, min_sample=MIN_SAMPLE, evidence=evidence),
        finding("dialogue.max_unnarrated_run", "Longest unnarrated dialogue run", longest,
                "turns", family=FAMILY, sample_size=len(run_length_list), min_sample=MIN_SAMPLE,
                evidence=evidence[:1]),
        finding("dialogue.in_unnarrated_run_share",
                f"Share of turns inside a run of {RUN_SHARE_THRESHOLD}+ unnarrated turns",
                in_run_share, "percent", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE),
    ]
