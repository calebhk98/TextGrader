"""A suffix proxy for nominalization density, not a parse of derivation.

"Nominalization" properly means a noun derived from a verb or adjective
("decide" -> "decision"), which is a fact about a word's morphological
history that spaCy's dependency parser does not represent.  What this module
actually measures is cheaper and narrower: the share of common nouns whose
lemma ends in one of a handful of suffixes that nominalizations commonly take
(-tion, -sion, -ment, -ness, -ity, -ance, -ence).  That is a useful proxy for
"abstract, Latinate, bureaucratic-sounding noun", which correlates with
nominalization but is not the same thing: it will miss nominalizations with
other endings (e.g. "upkeep", "growth", "arrival"... "arrival" does end in
one of the suffixes, but "breakthrough" does not) and it will count ordinary,
non-derived nouns that merely happen to share an ending ("ambulance",
"province", "kindness" is genuinely derived so that one is fine, but
"business" is not derived from "busy" in the relevant sense despite the
ending). The metric name and every finding say "suffix proxy" for exactly
this reason: it should never be read as a measurement of derivational
morphology.

Proper nouns are excluded (spaCy tags them ``PROPN``, not ``NOUN``, so this
falls out of the part-of-speech filter already). The lemma's suffix is
checked rather than the surface form's, so a plural like "decisions" is
still recognized even though its surface ending is "-sions", not "-sion".
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, rate, unavailable

FAMILY = "lexical"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

SUFFIXES = ("tion", "sion", "ment", "ness", "ity", "ance", "ence")

NAME = "Nominalization density (suffix proxy)"
PROXY_WARNING = (
    "Suffix proxy, not a parse of derivational morphology: counts common nouns whose lemma "
    "ends in -tion/-sion/-ment/-ness/-ity/-ance/-ence. It both misses nominalizations with "
    "other endings and can count ordinary nouns that merely share an ending. Proper nouns "
    "are excluded."
)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable("nlp.nominalization_density", NAME, analysis.nlp_unavailable, family=FAMILY)]

    noun_count = 0
    hit_count = 0
    evidence: list[dict[str, Any]] = []
    for offset, doc in analysis.spacy_docs():
        for token in doc:
            if token.pos_ != "NOUN":
                continue
            noun_count += 1
            lemma = (token.lemma_ or token.text).lower()
            if not lemma.endswith(SUFFIXES):
                continue
            hit_count += 1
            if len(evidence) < 25:
                evidence.append({"text": token.text, "lemma": lemma,
                                 "offset": analysis.token_offset(offset, token)})

    warning = PROXY_WARNING if noun_count else f"{PROXY_WARNING} No common nouns were parsed."
    return [finding(
        "nlp.nominalization_density", NAME, rate(hit_count, noun_count) if noun_count else None,
        "%", family=FAMILY, sample_size=noun_count, min_sample=MIN_SAMPLE,
        evidence=evidence, warning=warning)]
