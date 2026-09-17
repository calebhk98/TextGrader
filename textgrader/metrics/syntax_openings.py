"""Sentence-opening shapes, discovered from POS/dependency labels, not vocabulary.

``tics.py`` flags "sentence opens She/He + verb" as a named, hard-coded tic for
one manuscript: it works only because someone already read the book and wrote
down the exact words. It cannot say whether a different manuscript opens on
"There was", or "It seemed", or a fronted adverbial, and it cannot be reused
without editing the word list.

This metric builds the same kind of signal without any vocabulary at all. For
each sentence, the leading punctuation and quotation marks are skipped (a
dash or an opening quote is not part of the syntax), and the first ``depth``
remaining tokens contribute a ``POS+DEP`` tag each, joined into one string:
"PRON+nsubj VERB+ROOT" is what "He ran." and "She left." both produce, so the
proxy the legacy tic hard-codes falls out of the parse as one pattern among
however many the manuscript actually uses, whatever the pronoun or verb.

The pattern distribution is reported by its top share (how dominant the single
most common opening is), its entropy (how many effectively-distinct openings
there are), and the top patterns themselves as evidence, each with one example
sentence. The same is reported for narration alone, because dialogue openings
("Wait." / question inversions / vocatives) follow different conventions and
would otherwise dilute a narration-voice signal.

The narration-only reading needs its own spaCy parse (:attr:`DocumentAnalysis.
narration` is a fresh view, not a slice of the cached document parse), so a
book with substantial dialogue costs roughly double this metric's share of the
run's parse time. That is unavoidable without caching a second parse tree
elsewhere, and is why it is skipped below :data:`MIN_SAMPLE` narration
sentences.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, option, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

_METRIC_NAMES = {
    "syntax.opening_pattern_top_share": "Top sentence-opening pattern share",
    "syntax.opening_pattern_entropy": "Sentence-opening pattern entropy",
    "syntax.opening_pattern_top_share_narration":
        "Top sentence-opening pattern share (narration only)",
}


def _entropy(counts: Counter) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def _opening_counts(analysis: DocumentAnalysis, depth: int) -> dict[str, tuple]:
    """Opening signatures for the whole text and for narration, in one pass.

    Narration used to be taken from ``analysis.narration``, which is a separate
    document and therefore a SECOND full spaCy parse: on a 300,000-word novel
    that doubled the most expensive operation in the tool for one extra
    percentage.  Classifying the sentences of the shared parse by how much of
    each one falls inside quotation marks answers the same question for free.
    """

    buckets = {name: [Counter(), {}, 0] for name in ("full", "narration")}
    for sent, channel, _ in analysis.spacy_sents_by_channel():
        content = [token for token in sent if not token.is_punct]
        if not content:
            continue
        signature = " ".join(f"{token.pos_}+{token.dep_}" for token in content[:depth])
        for name in ("full", "narration") if channel == "narration" else ("full",):
            counts, examples, total = buckets[name]
            counts[signature] += 1
            examples.setdefault(signature, sent.text.strip())
            buckets[name][2] = total + 1
    return {name: (counts, examples, total) for name, (counts, examples, total)
            in buckets.items()}


def _top_evidence(counts: Counter, examples: Mapping[str, str],
                  limit: int) -> list[dict[str, Any]]:
    return [{"pattern": pattern, "count": count, "example": examples.get(pattern, "")[:120]}
            for pattern, count in counts.most_common(limit)]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, name, analysis.nlp_unavailable, family=FAMILY)
                for metric_id, name in _METRIC_NAMES.items()]

    depth = int(option(config, "depth", 2))
    max_reported = int(option(config, "max_reported", 25))
    no_data = "no sentence had a non-punctuation token to build an opening pattern from"

    buckets = _opening_counts(analysis, depth)
    counts, examples, total = buckets["full"]
    evidence = _top_evidence(counts, examples, max_reported)
    top_count = counts.most_common(1)[0][1] if counts else 0

    top_share_id = "syntax.opening_pattern_top_share"
    entropy_id = "syntax.opening_pattern_entropy"
    out = [
        finding(top_share_id, _METRIC_NAMES[top_share_id],
                100 * top_count / total if total else None, "%", family=FAMILY,
                sample_size=total, min_sample=MIN_SAMPLE, evidence=evidence,
                warning=None if total else no_data),
        finding(entropy_id, _METRIC_NAMES[entropy_id], _entropy(counts), "bits",
                family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                warning=None if total else no_data),
    ]

    n_counts, n_examples, n_total = buckets["narration"]
    if n_total >= MIN_SAMPLE:
        n_top_count = n_counts.most_common(1)[0][1] if n_counts else 0
        out.append(finding(
            "syntax.opening_pattern_top_share_narration",
            _METRIC_NAMES["syntax.opening_pattern_top_share_narration"],
            100 * n_top_count / n_total if n_total else None, "%", family=FAMILY,
            sample_size=n_total, min_sample=MIN_SAMPLE, channel="narration",
            evidence=_top_evidence(n_counts, n_examples, max_reported),
            warning=None if n_total else no_data))
    else:
        out.append(unavailable(
            "syntax.opening_pattern_top_share_narration",
            _METRIC_NAMES["syntax.opening_pattern_top_share_narration"],
            f"narration channel has {n_total} sentences, below the "
            f"{MIN_SAMPLE} needed for a separate narration-only reading", family=FAMILY,
            channel="narration"))
    return out
