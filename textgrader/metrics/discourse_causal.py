"""Rate of causal and explanatory connectives: because, therefore, as a result...

These markers are where a text stops to explain itself instead of trusting a
reader to infer the connection.  A high rate is not automatically a flaw (some
genres, and some narrators, explain constantly on purpose) but it is a fact
worth surfacing, especially split by whether the explaining happens inside a
character's own speech or in the narrator's voice: a character who reasons
aloud a great deal is a characterisation choice, a narrator who does it a
great deal is closer to a habit of the prose itself.

Word-level matching cannot disambiguate sense. "Since" is causal in "since he
lied, she left" and purely temporal in "since Tuesday"; "thus" and "hence" have
rare non-causal spatial uses ("thus far"); none of that is resolved here. The
rate below counts surface occurrences of the marker, not confirmed causal
readings, and the docstring says so instead of pretending otherwise.
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

# Canonical marker -> lower-cased word sequence.  Lengths vary from one word
# to four, so matching is done by grouping markers by length rather than by
# building one combined pattern.
CAUSAL: dict[str, tuple[str, ...]] = {
    "because": ("because",),
    "therefore": ("therefore",),
    "hence": ("hence",),
    "thus": ("thus",),
    "consequently": ("consequently",),
    "since": ("since",),
    "so that": ("so", "that"),
    "which meant": ("which", "meant"),
    "in other words": ("in", "other", "words"),
    "that is why": ("that", "is", "why"),
    "as a result": ("as", "a", "result"),
    "due to": ("due", "to"),
    "owing to": ("owing", "to"),
}

_BY_LENGTH: dict[int, dict[tuple[str, ...], str]] = {}
for _name, _phrase in CAUSAL.items():
    _BY_LENGTH.setdefault(len(_phrase), {})[_phrase] = _name


def _count_markers(tokens: list[str]) -> Counter:
    """Count every marker occurrence in one linear pass over the token list.

    At most four phrase lengths are checked per position, so this stays O(n)
    over the token count regardless of how many markers are in the dictionary.
    """

    counts: Counter[str] = Counter()
    total = len(tokens)
    for index in range(total):
        for length, table in _BY_LENGTH.items():
            if index + length > total:
                continue
            name = table.get(tuple(tokens[index:index + length]))
            if name:
                counts[name] += 1
    return counts


def _evidence(counts: Counter, words: int, limit: int = 25) -> list[dict[str, Any]]:
    return [{"marker": name, "count": count,
             "rate_per_1000_words": rate(count, words, 1000.0)}
            for name, count in counts.most_common(limit)]


def _channel_finding(metric_id: str, label: str, view: DocumentAnalysis) -> dict[str, Any]:
    counts = _count_markers(view.tokens)
    words = view.word_count
    matched = sum(counts.values())
    return finding(
        metric_id, f"Causal/explanatory marker rate ({label})",
        rate(matched, words, 1000.0), "per 1,000 words", family=FAMILY,
        sample_size=words, channel=view.channel, min_sample=MIN_SAMPLE,
        evidence=_evidence(counts, words),
        warning=None if words else f"no {label} words to measure")


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    total_counts = _count_markers(analysis.tokens)
    total_words = analysis.word_count
    total_matched = sum(total_counts.values())
    out = [finding(
        "discourse.causal_marker_rate", "Causal/explanatory marker rate",
        rate(total_matched, total_words, 1000.0), "per 1,000 words", family=FAMILY,
        sample_size=total_words, min_sample=MIN_SAMPLE,
        evidence=_evidence(total_counts, total_words),
        warning=None if total_words else "no words to measure")]

    out.append(_channel_finding("discourse.causal_marker_rate_dialogue", "dialogue only",
                                analysis.dialogue))
    out.append(_channel_finding("discourse.causal_marker_rate_narration", "narration only",
                                analysis.narration))
    return out
