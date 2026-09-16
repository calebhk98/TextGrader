"""How each spoken turn is attributed to its speaker, and how reliably.

Fiction marks who is talking three ways: a speech tag adjacent to the quote
("Ruth said"), an action beat that stands in for a tag ("Ruth crossed the
room." right before or after the line, with no verb of speaking at all), or
nothing, leaving the reader to infer the speaker from context.  The existing
``dialogue_tags.py`` looks for a speech verb in a fixed +/-100 character
window around each quotation, which is wide enough to attach a stray "said"
to the wrong quote in a fast exchange: ``"Yes." "No," she said. "Really?"``
would credit "she said" to whichever of the three quotes happens to fall
within 100 characters of it, not necessarily the one it actually tags.  This
module instead looks only at the narration that actually sits between two
quotations, or between a quotation and the nearest paragraph break, which is
the only span that could possibly belong to that turn.  ``analysis.
quotation_spans`` gives the bounds; :func:`turn_spans` redoes the same
merge ``DocumentAnalysis.turns`` performs (a quotation split by its own
attribution is one turn), but keeps the character offsets that ``.turns``
itself discards, because every metric in this family needs them.

Classification is a text heuristic, not a parse: "action beat" means adjacent
narration that is non-empty and does not contain a recognized speech verb,
not narration that has been confirmed to contain a verb at all.  A parser
could tell the two apart; this fast, dependency-free pass cannot, and says so
in its evidence rather than pretending otherwise.

Speaker names, where one is found, come from a capitalized token sitting
immediately next to the matched speech verb ("Ruth said" / "said Ruth"), not
from any list of character names, so the same code works on any novel.  A
turn with a tag but no adjacent capitalized token (most commonly a pronoun,
"she said") is still classified as ``speech_tag`` but carries no speaker; the
speaker-grouping metrics elsewhere in this family are therefore drawn from a
smaller, biased sample of the turns classified here.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .. import text as textlib
from .common import MODERATE, finding, rate

FAMILY = "dialogue"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# Verbs of speaking ("verba dicendi"), not character names: this is what lets
# attribution work on any novel instead of being tied to one hard-coded cast.
SPEECH_VERBS = frozenset({
    "said", "asked", "replied", "whispered", "shouted", "murmured", "cried",
    "called", "answered", "exclaimed", "muttered", "added", "continued",
    "interrupted", "snapped", "yelled", "laughed", "sighed", "gasped",
    "stated", "remarked", "noted", "observed", "questioned", "demanded",
    "announced", "admitted", "insisted", "protested", "suggested", "warned",
    "urged", "begged", "pleaded", "wondered", "mused", "retorted",
    "countered", "conceded", "chuckled", "grinned", "offered", "ventured",
    "breathed", "hissed", "growled", "barked", "drawled", "stammered",
    "stuttered", "repeated", "echoed", "agreed", "argued", "explained",
    "declared",
})
# The two tags plain enough to be the baseline (matching dialogue_tags.py's
# BASIC set); any other speech verb counts as an "elaborate" tag.
BASIC_TAG_VERBS = frozenset({"said", "asked"})

SPEECH_VERB_RE = re.compile(r"\b(" + "|".join(sorted(SPEECH_VERBS)) + r")\b", re.IGNORECASE)
_NAME_BEFORE_RE = re.compile(r"([A-Z][A-Za-z'’-]*)\s*,?\s*$")
_NAME_AFTER_RE = re.compile(r"^\s*,?\s*([A-Z][A-Za-z'’-]*)")
# Capitalized words that are grammatical, not a name, and would otherwise be
# picked up for sitting next to a speech verb ("She said", "The man said").
NAME_STOPWORDS = frozenset({
    "He", "She", "They", "It", "I", "We", "You", "The", "A", "An", "And",
    "But", "So", "Then", "When", "As", "With", "At", "In", "On", "If",
    "That", "This", "There", "Here", "Who", "What", "Someone", "Somebody",
})


def turn_spans(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    """Spoken turns with offsets, replicating ``DocumentAnalysis.turns``.

    ``analysis.turns`` already rejoins a quotation split by its own
    attribution ("A," she says, "B." is one turn), but returns bare strings.
    Every metric in this family needs the character offsets that merge
    discards, so the identical rule is applied here directly to
    ``analysis.quotation_spans``.  ``internal`` carries the narration between
    spans that got merged into one turn, which is exactly where an inline tag
    such as ", she said," lives.
    """

    text = analysis.text
    out: list[dict[str, Any]] = []
    parts: list[str] = []
    internal: list[str] = []
    start: int | None = None
    last = 0
    for span_start, span_end, content in analysis.quotation_spans:
        gap = text[last:span_start]
        if parts:
            if re.search(r"[.!?…]", gap):
                out.append({"start": start, "end": last, "text": " ".join(parts),
                            "internal": internal})
                parts, internal, start = [], [], None
            else:
                internal.append(gap)
        if start is None:
            start = span_start
        parts.append(content)
        last = span_end
    if parts:
        out.append({"start": start, "end": last, "text": " ".join(parts), "internal": internal})
    return [item for item in out if textlib.words(item["text"])]


def _tail_after_blank_line(segment: str) -> str:
    index = segment.rfind("\n\n")
    return segment[index + 2:] if index != -1 else segment


def _head_before_blank_line(segment: str) -> str:
    index = segment.find("\n\n")
    return segment[:index] if index != -1 else segment


def _speaker_in(candidate: str, verb_match: re.Match) -> str | None:
    before = _NAME_BEFORE_RE.search(candidate[:verb_match.start()])
    if before and before.group(1) not in NAME_STOPWORDS:
        return before.group(1)
    after = _NAME_AFTER_RE.search(candidate[verb_match.end():])
    if after and after.group(1) not in NAME_STOPWORDS:
        return after.group(1)
    return None


def classify_turns(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    """Label every turn ``speech_tag``, ``action_beat`` or ``untagged``.

    A turn is a ``speech_tag`` when a speech verb sits in the narration
    between it and its neighbour: inside the merged turn for a split
    attribution, or immediately before/after it, cut off at the nearest
    paragraph break so a tag from a different scene or speaker cannot attach.
    Failing that, any non-empty adjacent narration is an ``action_beat``.
    """

    text = analysis.text
    spans = turn_spans(analysis)
    out: list[dict[str, Any]] = []
    for index, record in enumerate(spans):
        prev_end = spans[index - 1]["end"] if index > 0 else 0
        next_start = spans[index + 1]["start"] if index + 1 < len(spans) else len(text)
        before = _tail_after_blank_line(text[prev_end:record["start"]])
        after = _head_before_blank_line(text[record["end"]:next_start])
        candidates = list(record["internal"]) + [before, after]
        label, verb, speaker = "untagged", None, None
        for candidate in candidates:
            match = SPEECH_VERB_RE.search(candidate)
            if match:
                label = "speech_tag"
                verb = match.group(1).lower()
                speaker = _speaker_in(candidate, match)
                break
        if label == "untagged" and any(textlib.words(candidate) for candidate in candidates):
            label = "action_beat"
        out.append({"start": record["start"], "end": record["end"], "text": record["text"],
                    "label": label, "verb": verb,
                    "elaborate": bool(verb) and verb not in BASIC_TAG_VERBS,
                    "speaker": speaker})
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    turns = classify_turns(analysis)
    total = len(turns)
    if total == 0:
        warning = "no spoken turns found"
        return [
            finding("dialogue.speech_tag_rate", "Speech-tag rate", None, "percent",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.action_beat_rate", "Action-beat rate", None, "percent",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.untagged_turn_rate", "Untagged-turn rate", None, "percent",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.elaborate_tag_share", "Elaborate speech-tag share", None,
                    "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    counts = Counter(item["label"] for item in turns)
    tagged = [item for item in turns if item["label"] == "speech_tag"]
    elaborate = sum(1 for item in tagged if item["elaborate"])
    evidence = [{"offset": item["start"], "label": item["label"], "verb": item["verb"],
                 "speaker": item["speaker"], "text": item["text"][:120]}
                for item in turns[:25]]
    return [
        finding("dialogue.speech_tag_rate", "Speech-tag rate",
                rate(counts.get("speech_tag", 0), total), "percent", family=FAMILY,
                sample_size=total, min_sample=MIN_SAMPLE, evidence=evidence),
        finding("dialogue.action_beat_rate", "Action-beat rate",
                rate(counts.get("action_beat", 0), total), "percent", family=FAMILY,
                sample_size=total, min_sample=MIN_SAMPLE),
        finding("dialogue.untagged_turn_rate", "Untagged-turn rate",
                rate(counts.get("untagged", 0), total), "percent", family=FAMILY,
                sample_size=total, min_sample=MIN_SAMPLE,
                warning="untagged turns have no recoverable speaker; every metric elsewhere "
                        "in this family that groups turns by speaker only sees the rest"),
        finding("dialogue.elaborate_tag_share",
                "Share of speech tags using a verb other than said/asked",
                rate(elaborate, len(tagged)), "percent", family=FAMILY,
                sample_size=len(tagged), min_sample=MIN_SAMPLE,
                warning=None if tagged else "no speech-tagged turns to assess"),
    ]
