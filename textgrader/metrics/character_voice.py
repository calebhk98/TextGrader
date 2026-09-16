"""Pairwise function-word distance between transcript-style speakers.

This is one of the few metrics that legitimately needs text *before* the
canonical cleanup strips it: transcript-formatted lines ("Username: message")
are exactly what :attr:`DocumentAnalysis.processing`'s ``strip_transcript``
step removes from ``analysis.text``, because those lines are ordinarily chat
noise around a manuscript, not prose to grade.  For this metric they are the
whole point, so it reads :attr:`DocumentAnalysis.raw` (the original,
uncleaned text) with the same :class:`~textgrader.text.TranscriptConfig` the
pipeline itself uses, rather than re-inventing transcript detection.

Each speaker's lines become a small stylistic fingerprint (the same
function-word rates :mod:`function_words` uses, plus contraction, question
and exclamation rates) and every pair of speakers is compared by cosine
distance.  A single mean pairwise distance hides whether the cast is
uniformly distinct or has one outlier voice pulling the average up while
most pairs actually sound alike, so the full distribution of pairwise
distances is published and the median, not the mean, is the headline value.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from .. import text as textlib
from ..document import DocumentAnalysis
from ..stats import summarize
from .common import FAST, cosine_distance, finding
from .function_words import FUNCTION

FAMILY = "dialogue"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 3
UNIT_SENSITIVE = False

MAX_EVIDENCE = 25


def _vector(text: str) -> dict[str, float]:
    tokens = [word.lower() for word in textlib.words(text)]
    total = len(tokens) or 1
    counts = Counter(tokens)
    vector = {f"f:{word}": counts[word] / total for word in FUNCTION}
    vector.update({
        "contractions": sum("'" in token or "’" in token for token in tokens) / total,
        "questions": text.count("?") / total,
        "exclamations": text.count("!") / total,
    })
    return vector


def _speaker_lines(analysis: DocumentAnalysis) -> dict[str, list[str]]:
    matches = textlib.transcript_lines(analysis.raw, analysis.processing.transcript_config())
    groups: dict[str, list[str]] = {}
    for match in matches:
        info = match.groupdict()
        groups.setdefault(info.get("username") or "unknown", []).append(info.get("message") or "")
    return groups


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    groups = _speaker_lines(analysis)
    speakers = sorted(groups)

    if len(speakers) < 2:
        return [finding(
            "style.character_voice_distance", "Mean pairwise character voice distance", None,
            "cosine distance", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
            warning="need transcript-style speaker lines for at least two characters")]

    pairs = []
    for index, speaker_a in enumerate(speakers):
        for speaker_b in speakers[index + 1:]:
            distance = cosine_distance(_vector(" ".join(groups[speaker_a])),
                                       _vector(" ".join(groups[speaker_b])))
            pairs.append({"a": speaker_a, "b": speaker_b, "distance": distance,
                         "lines_a": len(groups[speaker_a]), "lines_b": len(groups[speaker_b])})

    distances = [pair["distance"] for pair in pairs if pair["distance"] is not None]
    if not distances:
        return [finding(
            "style.character_voice_distance", "Mean pairwise character voice distance", None,
            "cosine distance", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
            evidence=pairs[:MAX_EVIDENCE],
            warning="no speaker pair produced a comparable function-word vector "
                    "(a speaker with no words has an all-zero vector)")]

    summary = summarize(distances)
    evidence = sorted(pairs, key=lambda pair: -(pair["distance"] or -1))[:MAX_EVIDENCE]

    return [finding(
        "style.character_voice_distance", "Mean pairwise character voice distance",
        summary.get("median"), "cosine distance", family=FAMILY, sample_size=len(distances),
        distribution=summary, min_sample=MIN_SAMPLE, evidence=evidence)]
