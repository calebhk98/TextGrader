"""Coordination against subordination: "and ... and ..." versus embedding.

Two sentences can share a length and a clause count and still read completely
differently: one strings clauses together with coordinating conjunctions
("she opened the door, and the room was dark, and she froze"), the other nests
them ("when she opened the door onto the dark room, she froze"). Neither is a
lexical proxy; both sides are read off the dependency labels the parser
already assigns:

* coordination: ``conj`` (a coordinated clause or phrase) plus ``cc`` (the
  conjunction itself, "and"/"but"/"or").
* subordination: ``advcl``, ``relcl``, ``ccomp``, ``xcomp``, ``csubj``,
  ``acl`` and ``mark`` (the subordinator itself, "because"/"although"/"that").

Both are reported as rates per 100 sentences, plus their ratio. A ratio well
above 1 is coordination-heavy, additive prose; well below 1 is heavily
embedded prose. The ratio is undefined, not zero or infinite, when the
denominator is zero, and that case is reported as such rather than faked.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, rate, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

_COORDINATION_DEPS = ("conj", "cc")
_SUBORDINATION_DEPS = ("advcl", "relcl", "ccomp", "xcomp", "csubj", "acl", "mark")

_METRIC_NAMES = {
    "syntax.coordination_rate": "Coordination rate",
    "syntax.subordination_rate": "Subordination rate",
    "syntax.coordination_subordination_ratio": "Coordination-to-subordination ratio",
}


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, name, analysis.nlp_unavailable, family=FAMILY)
                for metric_id, name in _METRIC_NAMES.items()]

    dep_counts: Counter[str] = Counter()
    sentence_count = 0
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            sentence_count += 1
            for token in sent:
                dep_counts[token.dep_] += 1

    no_data = "no sentences to compute coordination/subordination rates over"
    coordination_count = sum(dep_counts.get(dep, 0) for dep in _COORDINATION_DEPS)
    subordination_count = sum(dep_counts.get(dep, 0) for dep in _SUBORDINATION_DEPS)
    coordination_rate = rate(coordination_count, sentence_count)
    subordination_rate = rate(subordination_count, sentence_count)

    if not sentence_count:
        ratio, ratio_warning = None, no_data
    elif subordination_count == 0:
        ratio, ratio_warning = None, (
            "no subordinating relation found; coordination-to-subordination ratio is "
            "undefined rather than infinite")
    else:
        ratio, ratio_warning = coordination_count / subordination_count, None

    coord_id, subord_id = "syntax.coordination_rate", "syntax.subordination_rate"
    ratio_id = "syntax.coordination_subordination_ratio"
    no_warning = None if sentence_count else no_data
    return [
        finding(coord_id, _METRIC_NAMES[coord_id], coordination_rate, "per 100 sentences",
                family=FAMILY, sample_size=sentence_count, min_sample=MIN_SAMPLE,
                warning=no_warning,
                evidence=[{"dependency_label": dep, "count": dep_counts.get(dep, 0)}
                          for dep in _COORDINATION_DEPS]),
        finding(subord_id, _METRIC_NAMES[subord_id], subordination_rate, "per 100 sentences",
                family=FAMILY, sample_size=sentence_count, min_sample=MIN_SAMPLE,
                warning=no_warning,
                evidence=[{"dependency_label": dep, "count": dep_counts.get(dep, 0)}
                          for dep in _SUBORDINATION_DEPS]),
        finding(ratio_id, _METRIC_NAMES[ratio_id], ratio, "ratio", family=FAMILY,
                sample_size=sentence_count, min_sample=MIN_SAMPLE, warning=ratio_warning),
    ]
