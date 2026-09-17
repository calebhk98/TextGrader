"""Real per-sentence parse-tree depth: the longest root-to-leaf path.

The legacy ``clause_structure`` metric (``textgrader/metrics/clause_structure.py``)
sums ``len(list(token.ancestors))`` for every token in the document and divides
by the token count.  That is the *mean* number of ancestors a token has, i.e.
mean token depth: a sentence with one deeply embedded clause and nine flat
tokens can score the same as a sentence with a shallow but uniform structure,
because the average blends depth with how many tokens sit at each depth.  It
also mixes tokens from different sentences into one running average, so a
report cannot tell a uniformly-moderate manuscript from one that alternates
simple and heavily embedded sentences.

This metric instead asks, per sentence, "how many edges separate the root
from its deepest descendant" (``max(len(list(token.ancestors)) for token in
sent)``, since the deepest node in a tree is necessarily a leaf, so scanning
every token and keeping the max finds the true longest root-to-leaf path
without a separate leaf search). That is real tree depth, not mean token
depth, and it is reported per sentence so its distribution, not just its
average, is visible.
"""

from __future__ import annotations

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
    "syntax.max_parse_depth": "Per-sentence maximum parse depth",
    "syntax.mean_max_parse_depth": "Mean maximum parse depth",
}


def _sentence_max_depths(analysis: DocumentAnalysis) -> list[float]:
    depths: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            tokens = list(sent)
            if not tokens:
                continue
            depths.append(float(max(sum(1 for _ in token.ancestors) for token in tokens)))
    return depths


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, name, analysis.nlp_unavailable, family=FAMILY)
                for metric_id, name in _METRIC_NAMES.items()]

    depths = _sentence_max_depths(analysis)
    summary = summarize(depths)
    no_data = "no sentences to measure parse depth over"

    out = shape("syntax.max_parse_depth", _METRIC_NAMES["syntax.max_parse_depth"],
                depths, "levels", family=FAMILY, min_sample=MIN_SAMPLE,
                evidence=[{"first_sentence_depths": depths[:20]}] if depths else None)
    out.append(finding(
        "syntax.mean_max_parse_depth", _METRIC_NAMES["syntax.mean_max_parse_depth"],
        summary.get("mean"), "levels", family=FAMILY, sample_size=summary.get("count"),
        min_sample=MIN_SAMPLE, warning=None if depths else no_data))
    return out
