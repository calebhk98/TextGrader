"""Finite clauses per sentence: a syntactic-complexity count, not a proxy.

A finite clause has its own tensed or agreement-bearing verb; counting them is
the standard way corpus linguistics measures subordination load without
relying on surface cues like comma count. A token counts as heading a finite
clause when it is a ``VERB`` or ``AUX`` whose morphology carries ``Tense`` or
``VerbForm=Fin`` *and* whose dependency relation makes it a clause head rather
than a helper: ``aux``, ``auxpass`` and ``cop`` are excluded so that "has run"
or "is tired" is counted once (on the head verb/predicate), not once per
auxiliary. Non-finite dependents (infinitival ``xcomp``, participial ``acl``)
fall out on their own because they lack finite morphology, with no need to
hard-code which dependency labels are "non-finite".

A sentence with one finite clause is a simple sentence; three or more is
noticeably compound or embedded prose. Both shares are reported alongside the
full per-sentence distribution because "sentences average 1.4 clauses" hides
whether that is every sentence sitting near 1.4, or a mix of simple and
heavily layered ones.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, rate, shape, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# Dependents of a finite verb that are not themselves a separate clause head.
_HELPER_DEPS = {"aux", "auxpass", "cop"}

_METRIC_NAMES = {
    "syntax.finite_clauses_per_sentence": "Finite clauses per sentence",
    "syntax.single_clause_sentence_share": "Sentences with exactly one finite clause",
    "syntax.multi_clause_sentence_share": "Sentences with three or more finite clauses",
}


def _is_finite_clause_head(token: Any) -> bool:
    if token.pos_ not in ("VERB", "AUX"):
        return False
    if token.dep_ in _HELPER_DEPS:
        return False
    if token.morph.get("Tense"):
        return True
    return "Fin" in token.morph.get("VerbForm")


def _sentence_counts(analysis: DocumentAnalysis) -> list[float]:
    counts: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            counts.append(float(sum(1 for token in sent if _is_finite_clause_head(token))))
    return counts


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, name, analysis.nlp_unavailable, family=FAMILY)
                for metric_id, name in _METRIC_NAMES.items()]

    counts = _sentence_counts(analysis)
    total = len(counts)
    no_data = "no sentences to count finite clauses over"
    shape_id = "syntax.finite_clauses_per_sentence"
    single_id = "syntax.single_clause_sentence_share"
    multi_id = "syntax.multi_clause_sentence_share"

    out = shape(shape_id, _METRIC_NAMES[shape_id], counts, "clauses", family=FAMILY,
                min_sample=MIN_SAMPLE,
                evidence=[{"first_sentence_counts": [int(v) for v in counts[:20]]}]
                if counts else None)

    single = sum(1 for value in counts if value == 1)
    multi = sum(1 for value in counts if value >= 3)
    out.append(finding(
        single_id, _METRIC_NAMES[single_id], rate(single, total), "%", family=FAMILY,
        sample_size=total, min_sample=MIN_SAMPLE, warning=None if total else no_data))
    out.append(finding(
        multi_id, _METRIC_NAMES[multi_id], rate(multi, total), "%", family=FAMILY,
        sample_size=total, min_sample=MIN_SAMPLE, warning=None if total else no_data))
    return out
