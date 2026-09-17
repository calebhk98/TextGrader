"""Share of clausal subjects marked passive by the dependency parse.

A clause's subject is either the underlying agent role (``nsubj``, active) or
a demoted one (``nsubjpass``, passive; or an active-looking ``nsubj`` whose
verb still carries a passive auxiliary, "was being watched") and counting
across every clausal subject in the document gives the share written in the
passive voice.

Reported for narration separately from the whole text, because passive
constructions cluster differently in dialogue ("It got broken, okay?") than
in narration, and a report that only sees the blended number cannot tell an
author whose narration is heavily passive from one whose characters just
talk that way.

The narration figure comes out of the SAME parse as the blended one, by
classifying each parsed sentence by how much of it lies inside quotation
marks.  Parsing ``analysis.narration`` separately would give an almost
identical answer for twice the cost, and the parse is by a wide margin the
most expensive thing TextGrader does.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, rate, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

SNIPPET_CHARS = 120


def _counts(analysis: DocumentAnalysis) -> dict[str, dict[str, Any]]:
    """One pass over the shared parse, split by channel."""

    tally = {name: {"subjects": 0, "passive": 0, "evidence": []}
             for name in ("full", "narration")}
    for sent, channel, offset in analysis.spacy_sents_by_channel():
        buckets = [tally["full"]] + ([tally["narration"]] if channel == "narration" else [])
        for token in sent:
            if token.dep_ not in ("nsubj", "nsubjpass"):
                continue
            is_passive = (token.dep_ == "nsubjpass"
                          or any(child.dep_ == "auxpass" for child in token.head.children))
            for bucket in buckets:
                bucket["subjects"] += 1
                if is_passive:
                    bucket["passive"] += 1
                    if len(bucket["evidence"]) < 25:
                        bucket["evidence"].append(
                            {"text": sent.text.strip()[:SNIPPET_CHARS],
                             "offset": analysis.token_offset(offset, token)})
    return tally


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    reason = analysis.nlp_unavailable
    if reason:
        return [unavailable("nlp.passive_voice", "Passive voice", reason, family=FAMILY),
                unavailable("nlp.passive_voice_narration", "Passive voice (narration only)",
                            reason, family=FAMILY, channel="narration")]
    tally = _counts(analysis)
    out = []
    for channel, metric_id, name in (
            ("full", "nlp.passive_voice", "Passive voice"),
            ("narration", "nlp.passive_voice_narration", "Passive voice (narration only)")):
        bucket = tally[channel]
        subjects = bucket["subjects"]
        out.append(finding(
            metric_id, name, rate(bucket["passive"], subjects) if subjects else None, "%",
            family=FAMILY, channel=channel, sample_size=subjects, min_sample=MIN_SAMPLE,
            evidence=bucket["evidence"],
            warning=None if subjects else f"no clausal subjects were parsed in {channel} text"))
    return out
