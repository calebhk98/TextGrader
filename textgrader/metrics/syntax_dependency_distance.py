"""Mean dependency distance: how far apart a sentence holds its dependents.

Dependency distance is the linear distance, in tokens, between a dependent and
its syntactic head (``|token.i - token.head.i|``).  A sentence built from short
local attachments (adjective next to its noun, subject next to its verb) has a
low mean distance; a sentence that defers its verb behind a long relative
clause, or stacks several PPs before resolving the noun they modify, has a
high one.  This is the measure TextDescriptives and the psycholinguistic
literature call dependency distance, and unlike counting parse-tree depth it
reflects the working-memory cost of holding a dependency open, which is a
better proxy for how hard a sentence is to parse while reading than depth
alone.

Each sentence contributes one number: the mean distance over its non-root
tokens (the root has no head and is excluded).  The per-sentence values are
then a sample in their own right, published as a shape, and the run's overall
mean and SD (of those per-sentence means, following the same convention as
TextDescriptives) summarize it in two numbers for a report that only wants
two numbers.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import PARSE, finding, shape, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

_METRIC_NAMES = {
    "syntax.sentence_dependency_distance": "Per-sentence mean dependency distance",
    "syntax.dependency_distance_mean": "Mean dependency distance",
    "syntax.dependency_distance_sd": "Dependency distance variability",
}


def _sentence_means(analysis: DocumentAnalysis) -> list[float]:
    means: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            distances = [abs(token.i - token.head.i) for token in sent if token.dep_ != "ROOT"]
            if distances:
                means.append(statistics.fmean(distances))
    return means


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, name, analysis.nlp_unavailable, family=FAMILY)
                for metric_id, name in _METRIC_NAMES.items()]

    means = _sentence_means(analysis)
    summary = summarize(means)
    no_data = "no sentence had a non-root token to measure dependency distance over"
    shape_id = "syntax.sentence_dependency_distance"
    mean_id = "syntax.dependency_distance_mean"
    sd_id = "syntax.dependency_distance_sd"

    out = shape(shape_id, _METRIC_NAMES[shape_id], means, "tokens", family=FAMILY,
                min_sample=MIN_SAMPLE,
                evidence=[{"first_sentence_means": [round(v, 2) for v in means[:20]]}]
                if means else None)
    out.append(finding(
        mean_id, _METRIC_NAMES[mean_id], summary.get("mean"), "tokens", family=FAMILY,
        sample_size=summary.get("count"), min_sample=MIN_SAMPLE,
        warning=None if means else no_data))
    out.append(finding(
        sd_id, _METRIC_NAMES[sd_id], summary.get("std"), "tokens", family=FAMILY,
        sample_size=summary.get("count"), min_sample=MIN_SAMPLE,
        warning=None if means else no_data))
    return out
