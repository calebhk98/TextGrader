"""Person-marking pronoun rates measured in narration only.

The existing whole-text POV metric folds dialogue and narration together, so
a third-person novel whose characters happen to say "I" and "you" a lot in
their own speech can look like it drifts into first or second person when it
never does. Dialogue is not narration: a character saying "I told you" is
inside quotation marks and says nothing about who is telling the story.

This module measures pronoun rates in :attr:`DocumentAnalysis.narration`
only, and reports the same first-person rate for dialogue alongside it,
clearly labelled, purely as a point of contrast: a large gap between the two
is the expected, healthy shape for a third-person novel with lively dialogue,
and a small gap is itself informative (a narrator who talks like their
characters, or a first-person narrator whose "I" rate barely rises above the
dialogue baseline).

The dominant-person margin is the dominant pronoun group's share of every
person-marking pronoun found in narration; a margin near one is an
unambiguous point of view, and a margin near a third with three groups in
play is a narrator whose person is not being consistently marked at all.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import FAST, finding, rate

FAMILY = "pov"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 300
UNIT_SENSITIVE = False

GROUPS: dict[str, set[str]] = {
    "first": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"},
    "second": {"you", "your", "yours", "yourself", "yourselves"},
    "third": {"he", "him", "his", "himself", "she", "her", "hers", "herself",
              "they", "them", "their", "theirs", "themselves"},
}


def _counts(tokens: list[str]) -> dict[str, int]:
    counter = Counter(tokens)
    return {group: sum(counter[word] for word in words) for group, words in GROUPS.items()}


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    narration = analysis.narration
    n_words = narration.word_count
    n_counts = _counts(narration.tokens)
    total_person = sum(n_counts.values())

    out: list[dict[str, Any]] = []
    for group in ("first", "second", "third"):
        out.append(finding(
            f"pov.narration_{group}_rate", f"{group.title()}-person pronoun rate (narration only)",
            rate(n_counts[group], n_words, 1000.0), "per 1,000 words", family=FAMILY,
            sample_size=n_words, channel="narration", min_sample=MIN_SAMPLE,
            evidence=[{"group": group, "count": n_counts[group]}],
            warning=None if n_words else "no narration words to measure"))

    dialogue = analysis.dialogue
    d_words = dialogue.word_count
    d_counts = _counts(dialogue.tokens)
    out.append(finding(
        "pov.dialogue_first_rate",
        "First-person pronoun rate inside dialogue (contrast only, not a POV claim)",
        rate(d_counts["first"], d_words, 1000.0), "per 1,000 words", family=FAMILY,
        sample_size=d_words, channel="dialogue", min_sample=MIN_SAMPLE,
        distribution={"dialogue_second_rate": rate(d_counts["second"], d_words, 1000.0),
                      "dialogue_third_rate": rate(d_counts["third"], d_words, 1000.0),
                      "dialogue_counts": d_counts, "dialogue_words": d_words},
        evidence=[{"group": group, "count": d_counts[group]} for group in ("first", "second", "third")],
        warning=None if d_words else "no dialogue words to measure"))

    if total_person:
        dominant_group, dominant_count = max(n_counts.items(), key=lambda item: item[1])
        margin = dominant_count / total_person
        margin_warning = None
    else:
        dominant_group, dominant_count, margin = None, 0, None
        margin_warning = "no person-marking pronouns found in narration"
    out.append(finding(
        "pov.narration_dominant_margin",
        "Dominant narration point of view, as a share of person-marking pronouns",
        margin, "share", family=FAMILY, sample_size=total_person, channel="narration",
        min_sample=MIN_SAMPLE,
        distribution={"dominant_person": dominant_group, "dominant_count": dominant_count,
                      "counts_by_person": n_counts},
        warning=margin_warning))
    return out
