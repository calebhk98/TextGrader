"""Person-marking pronoun rates and where the point of view seems to drift.

Three bugs made the legacy version unreliable:

1. ``max(GROUPS, key=...)`` always returns *some* key, even when every count
   in the block is zero.  Python dict order made "first" the default winner,
   so a block with no POV pronouns at all (a scene of pure description, a
   block of dialogue tags) was silently reported as first-person evidence
   with no evidence behind it.  A block now gets a dominant person only when
   it actually contains at least one POV pronoun; otherwise it is ``None`` /
   ``"unknown"`` and is never counted as a drift.
2. Counting dialogue and narration together makes "I" and "you" inside
   quoted speech look like narrator drift.  A third-person novel is full of
   first- and second-person pronouns the moment a character opens their
   mouth; that is not the narrator's point of view changing.  Drift is now
   measured on narration only.  The whole-text rates are still reported,
   clearly labelled, because a manuscript's overall pronoun mix is a useful
   fact in its own right, just not evidence of drift.
3. ``style.pov_drift`` was a raw count of distinct dominant labels minus one,
   so a 200,000-word novel with the same handful of narration lapses as a
   20,000-word one "drifts more" purely for being longer.  It is kept, for
   compatibility, but ``style.pov_drift_rate`` (drifts per 100 narration
   blocks) is the number worth comparing across documents of different
   length.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from .common import FAST, finding, option, rate

FAMILY = "pov"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

GROUPS: dict[str, list[str]] = {
    "first": ["i", "me", "my", "mine", "we", "us", "our", "ours"],
    "second": ["you", "your", "yours"],
    "third": ["he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs"],
}

DEFAULT_BLOCK_WORDS = 500


def _counts(tokens: Sequence[str]) -> Counter[str]:
    return Counter(tokens)


def _group_totals(counts: Counter[str]) -> dict[str, int]:
    return {group: sum(counts[word] for word in words) for group, words in GROUPS.items()}


def _dominant(totals: Mapping[str, int]) -> str | None:
    """The POV group with the most evidence in a block, or ``None`` when the
    block carries no POV pronoun at all.  Ties are broken toward whichever
    group's word list was declared first, which only matters when there is
    already at least one pronoun to break a tie over."""

    if not any(totals.values()):
        return None
    return max(totals, key=lambda group: totals[group])


def _blocks(tokens: Sequence[str], block_words: int) -> list[dict[str, Any]]:
    out = []
    for start in range(0, len(tokens), block_words):
        chunk = tokens[start:start + block_words]
        totals = _group_totals(_counts(chunk))
        out.append({"start": start, **totals, "dominant": _dominant(totals)})
    return out


def _drift(blocks: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """``(drift_count, blocks_with_evidence)``: transitions between two
    consecutive blocks that both had a dominant person, and how many blocks
    that comparison was even possible for."""

    labelled = [block["dominant"] for block in blocks if block["dominant"] is not None]
    drift = sum(1 for a, b in zip(labelled, labelled[1:]) if a != b)
    return drift, len(labelled)


def _rate_findings(tokens: Sequence[str], suffix: str, label: str) -> list[dict[str, Any]]:
    total = len(tokens)
    counts = _counts(tokens)
    totals = _group_totals(counts)
    out = []
    for group in GROUPS:
        out.append(finding(
            f"style.pov_{group}{suffix}", f"{group.title()}-person pronoun rate{label}",
            rate(totals[group], total, 1000.0), "per 1,000 words", family=FAMILY,
            sample_size=total, min_sample=MIN_SAMPLE,
            warning=None if total else "no words to measure pronoun rates over"))
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    block_words = max(20, int(option(config, "block_words", DEFAULT_BLOCK_WORDS)))

    out = _rate_findings(analysis.tokens, "", " (whole text)")

    narration = analysis.narration
    out.extend(_rate_findings(narration.tokens, "_narration", " (narration only)"))

    narration_tokens = narration.tokens
    if not narration_tokens:
        out.append(finding("style.pov_drift", "POV block drift (narration)", None, "changes",
                           family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                           warning="no narration text to measure POV drift over"))
        out.append(finding("style.pov_drift_rate", "POV block drift rate (narration)", None,
                           "changes per 100 blocks", family=FAMILY, sample_size=0,
                           min_sample=MIN_SAMPLE,
                           warning="no narration text to measure POV drift over"))
        return out

    blocks = _blocks(narration_tokens, block_words)
    drift_count, evidenced_blocks = _drift(blocks)
    evidence = [{"start": b["start"], "dominant": b["dominant"], **{g: b[g] for g in GROUPS}}
                for b in blocks[:25]]

    out.append(finding(
        "style.pov_drift", "POV block drift (narration)", drift_count, "changes",
        family=FAMILY, sample_size=len(blocks), min_sample=MIN_SAMPLE,
        unit_sensitive=True, evidence=evidence,
        warning=None if evidenced_blocks >= 2 else
        "fewer than two narration blocks carried any POV pronoun; drift cannot be assessed"))
    out.append(finding(
        "style.pov_drift_rate", "POV block drift rate (narration)",
        rate(drift_count, evidenced_blocks) if evidenced_blocks else None,
        "changes per 100 blocks with POV evidence", family=FAMILY,
        sample_size=evidenced_blocks, min_sample=MIN_SAMPLE,
        distribution={"total_blocks": len(blocks), "blocks_with_evidence": evidenced_blocks},
        warning=None if evidenced_blocks >= 2 else
        "fewer than two narration blocks carried any POV pronoun; drift cannot be assessed"))
    return out
