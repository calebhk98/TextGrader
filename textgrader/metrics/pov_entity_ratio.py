"""PERSON entities against personal pronouns, measured in narration only.

Two failure modes hide from a pronoun count alone. Over-naming repeats a
character's name instead of using a pronoun ("Sarah walked. Sarah opened the
door. Sarah sighed.") and reads as stilted or as narration written to be
skimmed by search rather than read; pronoun saturation is the opposite
failure, where the antecedent is used once and pronouns carry every reference
after it until the reader loses track of who "she" is. Named-entity
recognition is the only reliable way to count "how often was a person named"
without hand-listing every character's name, which is why this metric needs
spaCy's ``ner`` component and the plain pronoun-rate metrics in
:mod:`pov_narration` do not.

The shared pipeline (:class:`textgrader.document.NlpSettings`) disables
``ner`` by default because it roughly doubles parse time and only this metric
needs it. That default must be respected, not worked around: this module
never loads its own spaCy pipeline and never mutates ``analysis``. It checks
whether the pipeline it was handed actually has ``ner`` available and, if not,
says so plainly (the fix is a configuration change - ``nlp.disable`` must not
include ``"ner"`` - not a code change here).

Both counts are restricted to narration by filtering the shared parse with
:meth:`DocumentAnalysis.in_dialogue`, so the parse runs once over the whole
document (as every ``PARSE``-cost metric should) rather than re-parsing a
narration-only view.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, option, rate, unavailable

FAMILY = "pov"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 300
UNIT_SENSITIVE = False

METRIC_IDS = ("pov.person_entity_rate", "pov.narration_pronoun_rate", "pov.entity_pronoun_ratio")
NAMES = {
    "pov.person_entity_rate": "PERSON-entity mention rate (narration only)",
    "pov.narration_pronoun_rate": "Personal-pronoun rate (narration only)",
    "pov.entity_pronoun_ratio": "PERSON entities per personal pronoun (narration only)",
}

PERSONAL_PRONOUNS = {
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "they", "them", "their", "theirs", "themselves",
}


def _unavailable(reason: str) -> list[dict[str, Any]]:
    return [unavailable(metric_id, NAMES[metric_id], reason, family=FAMILY, channel="narration")
            for metric_id in METRIC_IDS]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    reason = analysis.nlp_unavailable
    if reason:
        return _unavailable(reason)
    if not option(config, "enable_ner", True):
        return _unavailable("disabled by this metric's own enable_ner=False configuration")
    if "ner" in analysis.nlp_settings.disable:
        return _unavailable(
            "the shared spaCy pipeline has 'ner' in nlp.disable; this metric needs named-entity "
            "recognition, so nlp.disable must not include 'ner' for it to run")
    pipe_names = list(getattr(analysis.nlp, "pipe_names", []) or [])
    if "ner" not in pipe_names:
        return _unavailable(
            "the loaded spaCy pipeline has no 'ner' component despite nlp.disable allowing it; "
            "nothing to do here but report the mismatch")

    person_entities = 0
    person_evidence: list[dict[str, Any]] = []
    pronoun_count = 0
    for offset, doc in analysis.spacy_docs():
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            start = offset + ent.start_char
            if analysis.in_dialogue(start):
                continue
            person_entities += 1
            if len(person_evidence) < 25:
                person_evidence.append({"offset": start, "text": ent.text[:120]})
        for token in doc:
            if token.lower_ not in PERSONAL_PRONOUNS:
                continue
            start = offset + token.idx
            if analysis.in_dialogue(start):
                continue
            pronoun_count += 1

    narration_words = analysis.narration.word_count
    entity_rate = rate(person_entities, narration_words, 1000.0)
    pronoun_rate = rate(pronoun_count, narration_words, 1000.0)
    ratio = (person_entities / pronoun_count) if pronoun_count else None

    warning = None if narration_words else "no narration words to measure"
    return [
        finding("pov.person_entity_rate", NAMES["pov.person_entity_rate"], entity_rate,
                "per 1,000 words", family=FAMILY, sample_size=narration_words,
                channel="narration", min_sample=MIN_SAMPLE, evidence=person_evidence,
                warning=warning),
        finding("pov.narration_pronoun_rate", NAMES["pov.narration_pronoun_rate"], pronoun_rate,
                "per 1,000 words", family=FAMILY, sample_size=narration_words,
                channel="narration", min_sample=MIN_SAMPLE,
                distribution={"pronoun_count": pronoun_count}, warning=warning),
        finding("pov.entity_pronoun_ratio", NAMES["pov.entity_pronoun_ratio"], ratio, "ratio",
                family=FAMILY, sample_size=pronoun_count, channel="narration",
                min_sample=MIN_SAMPLE,
                distribution={"person_entities": person_entities, "pronoun_count": pronoun_count},
                warning=(warning or ("no personal pronouns found in narration; the ratio is "
                                    "undefined" if pronoun_count == 0 else None))),
    ]
