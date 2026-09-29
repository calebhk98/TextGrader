"""Characteristic vocabulary: how often this text uses the corpus's everyday words.

Every other measurement in the tool describes a shape (how long sentences
run, how varied the words are); none says which words to use.  Style is also
vocabulary: an author's rate of "said", "like", "just" or "had" is as stable a
signature as sentence length, and it is the one a writer can change most
directly.

The profile records the corpus's everyday vocabulary -- its most frequent
words that occur in at least half of its observations, so names and
one-book topics drop out by dispersion -- with every observation's rate per
1,000 words (see ``corpus._word_rate_tables``).  This metric reports this
text's rate of each of those words as ``lexical.word_rate.<word>``, which the
grader compares with the corpus like any other measurement.

It also reports ``lexical.keyness_overused``: words this text uses far more
than the corpus does, ranked by Dunning's log-likelihood (G2) against the
corpus's total word counts, with how many corpus observations use each one.
That list is evidence for the writer to reason about, since an overused word
may be a name the story needs.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import FAST, finding, option, unavailable

FAMILY = "vocabulary"
COST = FAST
REQUIRES: tuple[str, ...] = ()
PREFIX = "lexical.word_rate."
MIN_SAMPLE = 500
#: G2 above 10.83 is p < 0.001 for one degree of freedom.
G2_THRESHOLD = 10.83


def _g2(a: int, b: int, c: int, d: int) -> float:
    """Dunning log-likelihood for a word seen ``a`` times in ``c`` tokens vs ``b`` in ``d``."""
    expected_a = c * (a + b) / (c + d)
    expected_b = d * (a + b) / (c + d)
    total = 0.0
    if a:
        total += a * math.log(a / expected_a)
    if b:
        total += b * math.log(b / expected_b)
    return 2 * total


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    rates = (profile or {}).get("word_rates") or {}
    vocabulary = rates.get("vocabulary") or []
    if not vocabulary:
        return [unavailable(PREFIX.rstrip("."), "Everyday-vocabulary word rates",
                            "the corpus profile has no word_rates table; rebuild it with this "
                            "version of python -m textgrader.corpus", family=FAMILY)]
    tokens = analysis.tokens
    total = len(tokens)
    counts = Counter(tokens)
    out = [finding(f"{PREFIX}{word}", f"Rate of '{word}'",
                   1000.0 * counts[word] / total if total else None, "per 1,000 words",
                   family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE)
           for word in vocabulary]

    corpus_counts = profile.get("word_frequency") or {}
    corpus_total = profile.get("word_frequency_total") or sum(corpus_counts.values())
    document_frequency = profile.get("word_document_frequency") or {}
    observations = profile.get("book_count") or len(profile.get("books") or [])
    limit = int(option(config or {}, "overuse_limit", 25))
    overused = []
    if total and corpus_total:
        for word, count in counts.items():
            corpus_count = corpus_counts.get(word, 0)
            if count / total <= corpus_count / corpus_total:
                continue
            g2 = _g2(count, corpus_count, total, corpus_total)
            if g2 >= G2_THRESHOLD:
                overused.append((g2, word, count, corpus_count))
    overused.sort(key=lambda item: (-item[0], item[1]))
    evidence = [{"word": word, "g2": round(g2, 1),
                 "this_text_per_1000": round(1000.0 * count / total, 2),
                 "corpus_per_1000": round(1000.0 * corpus_count / corpus_total, 3),
                 "corpus_observations_using_it": f"{document_frequency.get(word, 0)} of {observations}"}
                for g2, word, count, corpus_count in overused[:limit]]
    out.append(finding("lexical.keyness_overused",
                       "Words this text uses far more than the corpus (log-likelihood, p < 0.001)",
                       len(overused), "words", family=FAMILY, sample_size=total,
                       min_sample=MIN_SAMPLE, evidence=evidence,
                       distribution={"note": "a count, not compared with the corpus; the "
                                             "evidence lists the words, most over-used first"}))
    return out
