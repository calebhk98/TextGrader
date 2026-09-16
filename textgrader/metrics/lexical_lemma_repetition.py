"""Content-word repetition measured over lemmas, so inflection cannot hide it.

:mod:`lexical_repetition_distance` catches "shimmer ... shimmer" but not
"shimmer ... shimmering ... shimmered": three different surface tokens that
are the same word doing the same repetitive work. Lemmatizing NOUN, VERB, ADJ,
ADV and PROPN tokens before measuring reuse distance closes that gap; it needs
a real parse (part of speech disambiguates "saw" the noun from "saw" the verb,
which a lemmatizer cannot do from spelling alone), which is why this is the
PARSE-cost sibling of the surface-form metric rather than a config flag on it.

The payoff is the evidence: rather than three separate low counts for "walk",
"walked" and "walking", the report can say the manuscript used that one lemma
thirty-one times, with the breakdown of which surface forms it appeared as.
That is the actionable half of this metric; the reuse-distance shape and the
lemma type-token ratio are the two-number summary of the same signal.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import PARSE, finding, unavailable

FAMILY = "repetition"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
WITHIN_CLOSE = 50

_IDS_NAMES = (
    ("repetition.lemma_reuse_distance", "Lemma reuse distance"),
    ("repetition.lemma_type_token_ratio", "Lemma type-token ratio"),
    ("repetition.top_lemma_count", "Most-repeated lemma"),
)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable(mid, name, analysis.nlp_unavailable, family=FAMILY)
                for mid, name in _IDS_NAMES]

    last_seen: dict[str, int] = {}
    gaps: list[int] = []
    close_hits: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    lemma_counts: Counter = Counter()
    surface_by_lemma: dict[str, Counter] = defaultdict(Counter)

    # `position` walks every spaCy token, not just the content-POS ones, so a
    # gap means an actual distance in the document rather than a distance in a
    # filtered subsequence that would compress every skipped function word away.
    position = 0
    for token in analysis.spacy_tokens():
        if token.pos_ in CONTENT_POS and token.is_alpha:
            lemma = token.lemma_.lower()
            lemma_counts[lemma] += 1
            surface_by_lemma[lemma][token.text.lower()] += 1
            previous = last_seen.get(lemma)
            if previous is not None:
                gap = position - previous
                gaps.append(gap)
                if gap <= WITHIN_CLOSE:
                    close_hits[lemma].append((gap, previous, position))
            last_seen[lemma] = position
        position += 1

    total_content = sum(lemma_counts.values())
    if not total_content:
        return [unavailable(mid, name, "no content-word (noun/verb/adj/adv/propn) tokens parsed",
                            family=FAMILY) for mid, name in _IDS_NAMES]

    ttr = len(lemma_counts) / total_content

    distance_finding: dict[str, Any]
    if gaps:
        distribution = summarize(gaps)
        reuse_counts = Counter({lemma: len(hits) for lemma, hits in close_hits.items()})
        distance_evidence = []
        for lemma, count in reuse_counts.most_common(15):
            hits = close_hits[lemma]
            distance_evidence.append({
                "lemma": lemma, "reuses_within_50_tokens": count,
                "example_offsets": [{"previous_index": p, "index": i, "gap": g}
                                    for g, p, i in hits[:3]],
            })
        distance_finding = finding(
            _IDS_NAMES[0][0], _IDS_NAMES[0][1], distribution.get("median"), "tokens",
            family=FAMILY, sample_size=len(gaps), distribution=distribution,
            evidence=distance_evidence, min_sample=MIN_SAMPLE)
    else:
        distance_finding = finding(
            _IDS_NAMES[0][0], _IDS_NAMES[0][1], None, "tokens", family=FAMILY,
            sample_size=0, min_sample=MIN_SAMPLE,
            warning="no content-word lemma recurred in this text")

    ttr_finding = finding(
        _IDS_NAMES[1][0], _IDS_NAMES[1][1], 100.0 * ttr, "%", family=FAMILY,
        sample_size=total_content, min_sample=MIN_SAMPLE)

    top_lemma, top_count = lemma_counts.most_common(1)[0]
    top_evidence = []
    for lemma, count in lemma_counts.most_common(20):
        surfaces = surface_by_lemma[lemma].most_common(6)
        top_evidence.append({
            "lemma": lemma, "count": count,
            "surface_forms": {surface: n for surface, n in surfaces},
        })
    top_finding = finding(
        _IDS_NAMES[2][0], _IDS_NAMES[2][1], top_count, "occurrences", family=FAMILY,
        sample_size=total_content, evidence=top_evidence, min_sample=MIN_SAMPLE,
        warning=f"most-repeated lemma is {top_lemma!r}")

    return [distance_finding, ttr_finding, top_finding]
