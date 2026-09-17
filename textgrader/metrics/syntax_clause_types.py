"""Clause-type rates read straight from the dependency parse.

``measures/prose_grade.py`` estimates "sentences with a subordinate clause" and
"sentences with a relative clause" by matching regexes such as a leading
``which``/``that``/``who`` or a comma followed by a subordinator. That is a
lexical proxy: it misses a relative clause with an elided relativizer ("the
book I read"), an adverbial clause introduced by a word the regex does not
list, and it can be fooled by "that" used as a demonstrative. A dependency
parse resolves the actual grammatical relation, so this metric replaces the
guess with a count of the labels that mean the same thing structurally:

* ``relcl``     - relative clause modifying a noun.
* ``advcl``     - adverbial clause (temporal, causal, conditional, ...).
* ``ccomp``/``xcomp`` - clausal complement, closed or open (a verb's object
  clause, e.g. "she said [that he left]" / "she wants [to leave]").
* ``csubj``/``csubjpass`` - a clause acting as the subject itself
  ("[what he said] surprised her").

Each is reported as a rate per 100 sentences rather than a raw count, so
documents of different length compare directly.
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

_METRIC_DEPS = {
    "syntax.relative_clause_rate": ("relcl",),
    "syntax.adverbial_clause_rate": ("advcl",),
    "syntax.complement_clause_rate": ("ccomp", "xcomp"),
    "syntax.clausal_subject_rate": ("csubj", "csubjpass"),
}
_METRIC_NAMES = {
    "syntax.relative_clause_rate": "Relative clause rate",
    "syntax.adverbial_clause_rate": "Adverbial clause rate",
    "syntax.complement_clause_rate": "Complement clause rate",
    "syntax.clausal_subject_rate": "Clausal subject rate",
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

    no_data = "no sentences to compute clause-type rates over"
    out = []
    for metric_id, deps in _METRIC_DEPS.items():
        count = sum(dep_counts.get(dep, 0) for dep in deps)
        evidence = [{"dependency_label": dep, "count": dep_counts.get(dep, 0)} for dep in deps]
        out.append(finding(
            metric_id, _METRIC_NAMES[metric_id], rate(count, sentence_count),
            "per 100 sentences", family=FAMILY, sample_size=sentence_count,
            min_sample=MIN_SAMPLE, evidence=evidence,
            warning=None if sentence_count else no_data))
    return out
