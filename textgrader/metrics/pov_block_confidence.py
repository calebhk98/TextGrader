"""Per-block POV calls that refuse to guess when there is no evidence.

This module exists because of one bug. The older whole-text POV metric splits
the text into blocks and, for each one, does the equivalent of
``max(GROUPS, key=lambda g: counts[g])``. When a block happens to contain zero
person-marking pronouns, every group's count is zero, the tie is broken by
dict ordering, and the block is silently reported as "first person" - evidence
manufactured out of nothing, because Python's ``max`` cannot express "no
answer".

Here, a block gets a label only when it has at least ``MIN_EVIDENCE_PRONOUNS``
person-marking pronouns; otherwise it is labelled ``"unknown"`` and carries its
(low) evidence count so a reader can see why. A drift is counted only between
two *consecutive* blocks that both cleared that bar, so a stretch of
description-only prose in between two dialogue-heavy scenes cannot manufacture
a false drift out of its own silence.

Blocks are measured over narration only (see :mod:`pov_narration` for why
dialogue is excluded), split at paragraph boundaries by
:meth:`DocumentAnalysis.windows` so a block never cuts a paragraph in half.
Drift is reported as a rate per 100 blocks, not a raw count, because a whole
novel processed at the same block size will otherwise always look "worse"
than a single chapter simply for having more blocks to drift between.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import FAST, finding, option, rate

FAMILY = "pov"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 5
UNIT_SENSITIVE = False

GROUPS: dict[str, set[str]] = {
    "first": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"},
    "second": {"you", "your", "yours", "yourself", "yourselves"},
    "third": {"he", "him", "his", "himself", "she", "her", "hers", "herself",
              "they", "them", "their", "theirs", "themselves"},
}

# Below this many person-marking pronouns, a block's majority group is not
# distinguishable from noise and the block is labelled "unknown" instead.
MIN_EVIDENCE_PRONOUNS = 3


def _counts(tokens: list[str]) -> dict[str, int]:
    counter = Counter(tokens)
    return {group: sum(counter[word] for word in words) for group, words in GROUPS.items()}


def _blocks(narration: DocumentAnalysis, block_words: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    for view in narration.windows(block_words):
        counts = _counts(view.tokens)
        evidence_count = sum(counts.values())
        label = (max(counts.items(), key=lambda item: item[1])[0]
                 if evidence_count >= MIN_EVIDENCE_PRONOUNS else "unknown")
        out.append({"word_offset": offset, "words": view.word_count,
                    "counts": counts, "evidence_count": evidence_count, "label": label})
        offset += view.word_count
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    block_words = int(option(config, "block_words", 1000))
    blocks = _blocks(analysis.narration, block_words)
    total_blocks = len(blocks)
    confident = [block for block in blocks if block["label"] != "unknown"]
    no_evidence = [block for block in blocks if block["label"] == "unknown"]

    drift_count = sum(
        1 for a, b in zip(blocks, blocks[1:])
        if a["label"] != "unknown" and b["label"] != "unknown" and a["label"] != b["label"])

    block_evidence = [{"word_offset": block["word_offset"], "words": block["words"],
                       "counts": block["counts"], "evidence_count": block["evidence_count"],
                       "label": block["label"]} for block in blocks[:25]]
    warning = None if total_blocks else "no narration blocks to measure"

    out = [finding(
        "pov.block_drift_rate", "POV drift transitions between confidently-labelled blocks",
        rate(drift_count, total_blocks, 100.0), "per 100 blocks", family=FAMILY,
        sample_size=total_blocks, channel="narration", min_sample=MIN_SAMPLE,
        distribution={"drift_count": drift_count, "block_count": total_blocks,
                      "block_words": block_words,
                      "min_evidence_pronouns": MIN_EVIDENCE_PRONOUNS},
        evidence=block_evidence, warning=warning)]

    out.append(finding(
        "pov.confident_block_share", "Share of blocks with enough pronoun evidence for a POV call",
        rate(len(confident), total_blocks, 100.0), "percent", family=FAMILY,
        sample_size=total_blocks, channel="narration", min_sample=MIN_SAMPLE,
        distribution={"confident_blocks": len(confident), "block_count": total_blocks},
        warning=warning))

    out.append(finding(
        "pov.no_evidence_block_count",
        "Blocks below the pronoun-evidence minimum, given no POV label",
        len(no_evidence), "blocks", family=FAMILY, sample_size=total_blocks,
        channel="narration", min_sample=MIN_SAMPLE,
        distribution={"min_evidence_pronouns": MIN_EVIDENCE_PRONOUNS,
                      "block_count": total_blocks},
        warning=warning))

    if confident:
        dominant, dominant_count = Counter(block["label"] for block in confident).most_common(1)[0]
        dominant_warning = None
    else:
        dominant, dominant_count = None, 0
        dominant_warning = "no block had enough pronoun evidence for a POV call"
    out.append(finding(
        "pov.dominant_pov", "Dominant point of view across confidently-labelled blocks",
        dominant, "person", family=FAMILY, sample_size=len(confident), channel="narration",
        min_sample=MIN_SAMPLE,
        distribution={"dominant_block_count": dominant_count,
                      "label_counts": dict(Counter(block["label"] for block in confident))},
        warning=dominant_warning))
    return out
