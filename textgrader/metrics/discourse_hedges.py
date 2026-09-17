"""Hedges, boosters and modals: surface markers of stance, not a stance parse.

Three closed-class lists, curated once from general English usage:

``HEDGES``
    softens a claim (perhaps, seemed, sort of, I think).
``BOOSTERS``
    sharpens one (clearly, always, of course).
``MODALS``
    marks obligation, possibility or ability (must, could, might).

These are lexical counts, nothing more. "Seemed" is a hedge in "she seemed
tired" and a plain verb of appearance in "the ghost seemed at the window";
"every" is counted as a booster (a totalizing claim) but is also the ordinary
word for iterating a set; modal "can" is as often about ability as about
possibility. None of that sense distinction is made here. What the rate can
say is only: how often does this vocabulary appear, and does hedging or
boosting dominate the mix. A pragmatic reading of stance is a different,
harder task that this module does not attempt.

Measured in narration separately for the hedge rate (the one most often read
as a narrator's tic - "perhaps", "seemed", "rather" - rather than a character's
speech habit), so a chatty, hedging cast does not get mistaken for a hedging
narrator.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import FAST, finding, rate

FAMILY = "discourse"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

HEDGES: dict[str, tuple[str, ...]] = {
    "perhaps": ("perhaps",), "maybe": ("maybe",), "seemed": ("seemed",),
    "appeared": ("appeared",), "somewhat": ("somewhat",), "rather": ("rather",),
    "fairly": ("fairly",), "apparently": ("apparently",), "presumably": ("presumably",),
    "arguably": ("arguably",), "possibly": ("possibly",),
    "sort of": ("sort", "of"), "kind of": ("kind", "of"),
    "i think": ("i", "think"), "i guess": ("i", "guess"),
}
BOOSTERS: dict[str, tuple[str, ...]] = {
    "clearly": ("clearly",), "obviously": ("obviously",), "certainly": ("certainly",),
    "definitely": ("definitely",), "undoubtedly": ("undoubtedly",), "absolutely": ("absolutely",),
    "of course": ("of", "course"), "indeed": ("indeed",), "truly": ("truly",),
    "always": ("always",), "never": ("never",), "every": ("every",), "completely": ("completely",),
}
MODALS: dict[str, tuple[str, ...]] = {
    word: (word,) for word in
    ("must", "should", "could", "would", "might", "may", "can", "shall", "ought")
}


def _by_length(table: Mapping[str, tuple[str, ...]]) -> dict[int, dict[tuple[str, ...], str]]:
    out: dict[int, dict[tuple[str, ...], str]] = {}
    for name, phrase in table.items():
        out.setdefault(len(phrase), {})[phrase] = name
    return out


_HEDGE_BY_LENGTH = _by_length(HEDGES)
_BOOSTER_BY_LENGTH = _by_length(BOOSTERS)
_MODAL_BY_LENGTH = _by_length(MODALS)


def _count(tokens: list[str], by_length: Mapping[int, dict[tuple[str, ...], str]]) -> Counter:
    counts: Counter[str] = Counter()
    total = len(tokens)
    for index in range(total):
        for length, table in by_length.items():
            if index + length > total:
                continue
            name = table.get(tuple(tokens[index:index + length]))
            if name:
                counts[name] += 1
    return counts


def _evidence(counts: Counter, words: int, limit: int = 25) -> list[dict[str, Any]]:
    return [{"word": name, "count": count, "rate_per_1000_words": rate(count, words, 1000.0)}
            for name, count in counts.most_common(limit)]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    words = analysis.word_count
    tokens = analysis.tokens
    hedge_counts = _count(tokens, _HEDGE_BY_LENGTH)
    booster_counts = _count(tokens, _BOOSTER_BY_LENGTH)
    modal_counts = _count(tokens, _MODAL_BY_LENGTH)

    hedge_total = sum(hedge_counts.values())
    booster_total = sum(booster_counts.values())
    modal_total = sum(modal_counts.values())

    hedge_rate = rate(hedge_total, words, 1000.0)
    booster_rate = rate(booster_total, words, 1000.0)
    modal_rate = rate(modal_total, words, 1000.0)

    out = [
        finding("discourse.hedge_rate", "Hedge rate", hedge_rate, "per 1,000 words",
                family=FAMILY, sample_size=words, min_sample=MIN_SAMPLE,
                evidence=_evidence(hedge_counts, words),
                warning=None if words else "no words to measure"),
        finding("discourse.booster_rate", "Booster rate", booster_rate, "per 1,000 words",
                family=FAMILY, sample_size=words, min_sample=MIN_SAMPLE,
                evidence=_evidence(booster_counts, words),
                warning=None if words else "no words to measure"),
        finding("discourse.modal_rate", "Modal-verb rate", modal_rate, "per 1,000 words",
                family=FAMILY, sample_size=words, min_sample=MIN_SAMPLE,
                evidence=_evidence(modal_counts, words),
                warning=None if words else "no words to measure"),
    ]

    if booster_total:
        hedge_booster_ratio = hedge_total / booster_total
        ratio_warning = None
    else:
        hedge_booster_ratio = None
        ratio_warning = ("no boosters found; the hedge:booster ratio is undefined"
                          if hedge_total else "no hedges or boosters found")
    out.append(finding(
        "discourse.hedge_booster_ratio", "Hedge-to-booster ratio", hedge_booster_ratio, "ratio",
        family=FAMILY, sample_size=hedge_total + booster_total, min_sample=MIN_SAMPLE,
        distribution={"hedge_count": hedge_total, "booster_count": booster_total},
        warning=ratio_warning))

    narration = analysis.narration
    n_words = narration.word_count
    n_hedge_counts = _count(narration.tokens, _HEDGE_BY_LENGTH)
    n_hedge_total = sum(n_hedge_counts.values())
    out.append(finding(
        "discourse.hedge_rate_narration", "Hedge rate (narration only)",
        rate(n_hedge_total, n_words, 1000.0), "per 1,000 words", family=FAMILY,
        sample_size=n_words, channel="narration", min_sample=MIN_SAMPLE,
        evidence=_evidence(n_hedge_counts, n_words),
        warning=None if n_words else "no narration words to measure"))
    return out
