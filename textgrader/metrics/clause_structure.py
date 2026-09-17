"""Mean dependency token depth: how far the average token sits from a root.

The name this metric used to go by, "mean dependency parse depth", claimed
more than the computation delivers.  It sums ``len(list(token.ancestors))``
for *every* token in the document and divides by the token count, which is
the mean number of ancestors a token has: mean *token* depth, not tree depth.
A sentence built from one deeply embedded clause and nine flat, shallow
tokens can score the same as a sentence with a uniformly moderate structure,
because the average blends how deep the tree goes with how many tokens sit
at each level.  It also pools tokens from every sentence into one running
mean, so the number cannot tell a uniformly-moderate manuscript from one that
alternates simple and heavily embedded sentences.

The metric is kept, correctly named, as ``nlp.mean_token_depth``.  The old id
``nlp.parse_depth`` is still emitted, computing the identical number, so a
corpus profile keyed on it keeps working, but every finding under that id
carries an explicit deprecation warning.  For actual per-sentence tree depth
(the longest root-to-leaf path in each sentence, which is what "parse depth"
should mean), use ``syntax.max_parse_depth`` in
``textgrader/metrics/syntax_parse_depth.py``.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

_NAME = "Mean dependency token depth"
_DEPRECATION_WARNING = (
    "Deprecated alias for nlp.mean_token_depth. Despite its old name this is the mean "
    "number of ancestors per token averaged over the whole document, not a per-sentence "
    "tree depth. For real per-sentence maximum parse-tree depth, use syntax.max_parse_depth."
)


def _token_depths(analysis: DocumentAnalysis) -> list[int]:
    depths: list[int] = []
    for _, doc in analysis.spacy_docs():
        for token in doc:
            depths.append(sum(1 for _ in token.ancestors))
    return depths


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [
            unavailable("nlp.mean_token_depth", _NAME, analysis.nlp_unavailable, family=FAMILY),
            unavailable("nlp.parse_depth", _NAME, analysis.nlp_unavailable, family=FAMILY),
        ]

    depths = _token_depths(analysis)
    mean_depth = statistics.fmean(depths) if depths else None
    no_data = "no tokens were parsed to measure dependency depth over"

    return [
        finding("nlp.mean_token_depth", _NAME, mean_depth, "levels", family=FAMILY,
                sample_size=len(depths), min_sample=MIN_SAMPLE,
                warning=None if depths else no_data),
        finding("nlp.parse_depth", _NAME, mean_depth, "levels", family=FAMILY,
                sample_size=len(depths), min_sample=MIN_SAMPLE,
                warning=_DEPRECATION_WARNING if depths else f"{_DEPRECATION_WARNING} {no_data}"),
    ]
