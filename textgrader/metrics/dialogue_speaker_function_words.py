"""Do characters sound alike underneath their subject matter?

Content words track what a character is talking about; function words (the,
of, but, she, would...) track HOW they talk, largely independent of topic,
which is exactly why Burrows's Delta and other authorship-attribution methods
are built on them rather than on vocabulary.  The same idea applied within
one book, across speakers, answers a different question than
``dialogue_speaker_style``'s question/exclamation/contraction rates: not "do
they punctuate differently" but "does their underlying grammatical texture
differ at all".  If every character's function-word profile is close to
every other's, "all my characters sound alike" is one number away, not a
close reading away.

The function-word list is imported from ``function_words.py`` rather than
copied, so the two metrics can never quietly drift apart. Speaker grouping
reuses ``dialogue_speaker_style.identify_speakers``, with the same caveat:
the speech-tag path is a biased sample of the more explicitly tagged turns
(see that module's docstring), so a thin profile here can mean "this
character is rarely tagged by name", not "this character has no voice".

The headline is the DISTRIBUTION of pairwise cosine distances between every
qualifying pair of speakers, not any one pair's number, because "are my
characters distinct" is a claim about the whole cast.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .. import text as textlib
from ..stats import summarize
from .common import MODERATE, cosine_distance, finding, option
from .dialogue_speaker_style import identify_speakers
from .function_words import FUNCTION

FAMILY = "dialogue"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
# A handful of pairwise distances (a few speakers) before "spread" means much.
MIN_SAMPLE = 3
UNIT_SENSITIVE = False

EVIDENCE_LIMIT = 25


def _profile(turns: list[str]) -> dict[str, float]:
    tokens = [word.lower().replace("’", "'") for turn in turns for word in textlib.words(turn)]
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {word: 1000.0 * counts[word] / total for word in FUNCTION}


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    min_turns = int(option(config, "min_turns", 8))
    groups, method = identify_speakers(analysis)
    qualifying = {name: turns for name, turns in groups.items() if len(turns) >= min_turns}
    names = sorted(qualifying)

    if len(names) < 2:
        warning = (f"found {len(groups)} named speaker(s) ({method} attribution), "
                   f"{len(names)} with at least {min_turns} turns; need at least two "
                   f"function-word profiles to compare")
        return [
            finding("dialogue.speaker_function_word_distance",
                    "Spread of pairwise speaker function-word distance", None, "cosine distance",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.closest_speaker_pair_distance",
                    "Closest speaker pair by function-word profile", None, "cosine distance",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.speaker_profile_count",
                    "Speakers with a usable function-word profile", len(names), "speakers",
                    family=FAMILY, sample_size=len(names), min_sample=MIN_SAMPLE, warning=warning),
        ]

    profiles = {name: _profile(qualifying[name]) for name in names}
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            distance = cosine_distance(profiles[a], profiles[b])
            if distance is not None:
                pairs.append({"speaker_a": a, "speaker_b": b, "distance": distance})

    if not pairs:
        warning = "no speaker had enough words to build a function-word profile"
        return [
            finding("dialogue.speaker_function_word_distance",
                    "Spread of pairwise speaker function-word distance", None, "cosine distance",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.closest_speaker_pair_distance",
                    "Closest speaker pair by function-word profile", None, "cosine distance",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("dialogue.speaker_profile_count",
                    "Speakers with a usable function-word profile", len(names), "speakers",
                    family=FAMILY, sample_size=len(names), min_sample=MIN_SAMPLE, warning=warning),
        ]

    ordered = sorted(pairs, key=lambda pair: pair["distance"])
    closest, farthest = ordered[0], ordered[-1]
    distances = [pair["distance"] for pair in pairs]
    summary = summarize(distances)

    return [
        finding("dialogue.speaker_function_word_distance",
                "Spread of pairwise speaker function-word distance", summary.get("median"),
                "cosine distance", family=FAMILY, sample_size=len(pairs), distribution=summary,
                min_sample=MIN_SAMPLE, evidence=ordered[:EVIDENCE_LIMIT]),
        finding("dialogue.closest_speaker_pair_distance",
                "Closest speaker pair by function-word profile", closest["distance"],
                "cosine distance", family=FAMILY, sample_size=len(pairs), min_sample=MIN_SAMPLE,
                evidence=[{"pair": "closest", **closest}, {"pair": "most_distant", **farthest}]),
        finding("dialogue.speaker_profile_count", "Speakers with a usable function-word profile",
                len(names), "speakers", family=FAMILY, sample_size=len(names),
                min_sample=MIN_SAMPLE,
                warning=None if method == "transcript" else
                "speaker profiles come from speech-tagged turns with a nearby capitalized "
                "name; a thin profile can mean 'rarely tagged by name', not 'no voice'"),
    ]
