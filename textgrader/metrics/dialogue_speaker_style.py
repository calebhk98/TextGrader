"""Whether individual characters sound different from one another.

``dialogue_channels`` asks whether dialogue as a whole differs from
narration.  This module asks the harder question one level down: do
individual speakers differ from each other?  A cast that is all one register
under different names is a common failure of generated or first-draft
dialogue, and it is invisible to any metric that treats "dialogue" as a
single bucket.

Speaker identification deliberately uses no cast list, because this
repository is being de-hardcoded away from any one book. A speaker is
recognized one of two ways, in :func:`identify_speakers`:

* a transcript-style ``Name: message`` line (``textgrader.text.
  transcript_lines``, read from ``analysis.raw`` since canonical cleanup
  strips these lines out of ``analysis.text`` when it recognizes the
  format); every line here comes with an explicit name, so this source is
  not a biased sample.
* failing that, a capitalized token sitting next to a speech verb adjacent
  to a quotation (``dialogue_attribution.classify_turns``).  A turn with a
  tag but no adjacent capitalized token ("she said") contributes to nobody's
  profile, so this source IS a biased sample: characters who are usually
  tagged by name will be over-represented relative to characters mostly
  tagged by pronoun, and untagged turns are invisible to it entirely.
  ``dialogue.identified_speaker_count`` exists precisely to flag how much of
  the cast this method could actually name.

The contraction rate deliberately does not treat any apostrophe as a
contraction: "John's coat" is a possessive, and a naive apostrophe count
would call it one. A curated suffix set (n't, 'll, 're, 've, 'd, 'm) is
matched instead. Bare "'s" is excluded entirely rather than guessed at,
because "she's happy" (is) and "the dog's bowl" (possessive) are not
distinguishable without a parse; excluding it undercounts is/has
contractions but never miscounts a possessive as one.

Per-speaker rows are the evidence; the headline values are the SPREAD of
each rate across speakers (via ``stats.summarize``), because the interesting
claim is "do these characters differ", not "what does any one of them do".
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .. import text as textlib
from ..stats import summarize
from .common import MODERATE, finding, option, rate
from .dialogue_attribution import classify_turns

FAMILY = "dialogue"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 3
UNIT_SENSITIVE = False

CONTRACTION_RE = re.compile(r"(?:n't|'ll|'re|'ve|'d|'m)$")
# Below this many matched lines, "Name:" is more likely a stray label than an
# actual chat transcript, so the speech-tag heuristic is used instead.
TRANSCRIPT_MIN_LINES = 10
EVIDENCE_LIMIT = 25


def identify_speakers(analysis: DocumentAnalysis) -> tuple[dict[str, list[str]], str]:
    """Group spoken turns by speaker.  Returns ``(turns_by_speaker, method)``.

    ``method`` is ``"transcript"`` when every line came with an explicit
    name, or ``"tag"`` when only speech-tagged turns with a nearby
    capitalized token could be attributed (see the module docstring for why
    that source is a biased sample).

    Transcript lines are only used when the configuration KEEPS them.  With the
    default ``text_processing.strip_transcript`` they are not part of the
    document any more, and reading them from :attr:`DocumentAnalysis.raw`
    measured text that every other metric had excluded.  On a manuscript with
    chat chapters that silently replaced its prose dialogue with unpunctuated
    chat lines, where contractions are written without apostrophes and
    questions without question marks, so both rates collapsed to zero and were
    then compared against a corpus of prose dialogue.
    """

    matches = ([] if analysis.processing.strip_transcript
               else textlib.transcript_lines(analysis.raw))
    if len(matches) >= TRANSCRIPT_MIN_LINES:
        groups: dict[str, list[str]] = {}
        for match in matches:
            info = match.groupdict()
            name = (info.get("username") or "").strip()
            message = (info.get("message") or "").strip()
            if name and message:
                groups.setdefault(name, []).append(message)
        if len(groups) >= 2:
            return groups, "transcript"

    groups = {}
    for item in classify_turns(analysis):
        if item["label"] == "speech_tag" and item["speaker"]:
            groups.setdefault(item["speaker"], []).append(item["text"])
    return groups, "tag"


def _contraction_rate(turns: list[str]) -> tuple[float | None, int]:
    tokens = [word.lower().replace("’", "'") for turn in turns for word in textlib.words(turn)]
    hits = sum(1 for token in tokens if CONTRACTION_RE.search(token))
    return rate(hits, len(tokens)), len(tokens)


def _unavailable(qualifying_count: int, total_count: int, method: str,
                 min_turns: int, warning: str | None = None) -> list[dict[str, Any]]:
    warning = warning or (
        f"found {total_count} named speaker(s) ({method} attribution), "
        f"{qualifying_count} with at least {min_turns} turns; need at least two to "
        f"say whether characters sound different from each other")
    return [
        finding("dialogue.speaker_question_rate", "Spread of per-speaker question rate", None,
                "per 100 turns", family=FAMILY, sample_size=qualifying_count,
                min_sample=MIN_SAMPLE, warning=warning),
        finding("dialogue.speaker_exclamation_rate", "Spread of per-speaker exclamation rate",
                None, "per 100 turns", family=FAMILY, sample_size=qualifying_count,
                min_sample=MIN_SAMPLE, warning=warning),
        finding("dialogue.speaker_contraction_rate", "Spread of per-speaker contraction rate",
                None, "percent", family=FAMILY, sample_size=qualifying_count,
                min_sample=MIN_SAMPLE, warning=warning),
        finding("dialogue.identified_speaker_count", "Identified speakers with enough turns",
                qualifying_count, "speakers", family=FAMILY, sample_size=total_count,
                min_sample=MIN_SAMPLE, warning=warning),
    ]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    min_turns = int(option(config, "min_turns", 8))
    min_coverage = float(option(config, "min_attributed_share", 25.0))
    groups, method = identify_speakers(analysis)
    qualifying = {name: turns for name, turns in groups.items() if len(turns) >= min_turns}

    if len(qualifying) < 2:
        return _unavailable(len(qualifying), len(groups), method, min_turns)

    # How much of the dialogue this attribution actually accounts for. A
    # per-speaker rate read off a tenth of the turns is not comparable with one
    # read off nearly all of them, and Victorian prose tags almost every turn
    # while modern prose drops the tag once a two-hander is established. Left
    # ungated, these metrics rank by how explicitly a book attributes speech.
    attributed = sum(len(turns) for turns in groups.values())
    total_turns = len(analysis.turns) if method == "tag" else attributed
    coverage = 100.0 * attributed / total_turns if total_turns else 0.0
    if method == "tag" and coverage < min_coverage:
        reason = (f"only {coverage:.0f}% of {total_turns} spoken turns could be attributed to a "
                  f"speaker (speech tag with a nearby capitalized name), below the "
                  f"{min_coverage:.0f}% these rates need to be comparable; a book that drops "
                  f"the tag once a two-hander is established cannot be held against one that "
                  f"tags every turn")
        return _unavailable(len(qualifying), len(groups), method, min_turns, reason)

    rows = []
    for name, turns in qualifying.items():
        questions = sum(1 for turn in turns if turn.rstrip().endswith("?"))
        exclamations = sum(1 for turn in turns if turn.rstrip().endswith("!"))
        contraction_rate, token_count = _contraction_rate(turns)
        rows.append({
            "speaker": name, "turns": len(turns), "words": token_count,
            "question_rate": rate(questions, len(turns)),
            "exclamation_rate": rate(exclamations, len(turns)),
            "contraction_rate": contraction_rate,
        })
    rows.sort(key=lambda row: -row["turns"])
    evidence = [{"attributed_turn_share": round(coverage, 1), "method": method}] + rows[:EVIDENCE_LIMIT]

    def spread(key: str) -> dict[str, Any]:
        return summarize([row[key] for row in rows if row[key] is not None])

    question_summary = spread("question_rate")
    exclamation_summary = spread("exclamation_rate")
    contraction_summary = spread("contraction_rate")

    return [
        finding("dialogue.speaker_question_rate",
                "Spread of per-speaker question rate", question_summary.get("median"),
                "per 100 turns", family=FAMILY, sample_size=len(rows),
                distribution=question_summary, min_sample=MIN_SAMPLE, evidence=evidence),
        finding("dialogue.speaker_exclamation_rate",
                "Spread of per-speaker exclamation rate", exclamation_summary.get("median"),
                "per 100 turns", family=FAMILY, sample_size=len(rows),
                distribution=exclamation_summary, min_sample=MIN_SAMPLE, evidence=evidence),
        finding("dialogue.speaker_contraction_rate",
                "Spread of per-speaker contraction rate", contraction_summary.get("median"),
                "percent", family=FAMILY, sample_size=len(rows),
                distribution=contraction_summary, min_sample=MIN_SAMPLE, evidence=evidence,
                warning="excludes bare 's; it is ambiguous between contraction and possessive "
                        "without a parse"),
        finding("dialogue.identified_speaker_count", "Identified speakers with enough turns",
                len(qualifying), "speakers", family=FAMILY, sample_size=len(groups),
                min_sample=MIN_SAMPLE,
                evidence=[{"speaker": name, "turns": len(turns)} for name, turns in
                         sorted(groups.items(), key=lambda kv: -len(kv[1]))[:EVIDENCE_LIMIT]],
                warning=None if method == "transcript" else
                "derived from speech tags with a nearby capitalized name; untagged and "
                "pronoun-tagged turns cannot be attributed and are excluded"),
    ]
