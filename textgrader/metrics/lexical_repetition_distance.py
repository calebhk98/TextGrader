"""How soon a content word comes back, in tokens.

Type-token measures (MTLD, HD-D, MATTR) describe how varied the vocabulary is
overall; they cannot tell a writer that they used "shimmer" twice in one
paragraph. This module answers that narrower, more actionable question
directly: for every content word, how far away (in tokens) was its previous
occurrence? A short gap is a word doing double duty within a reader's short
memory of the page; a long gap is just the vocabulary of the piece.

This is a single-pass, O(n) measurement (one dict of "last position seen per
word"), unlike :mod:`repeated_ngrams` and :mod:`local_repetition`, which scan
windows of the token stream and so cost more for the same signal. It also
tracks distance rather than density, which a fixed window cannot: a repeat at
gap 45 and a repeat at gap 900 both count the same in a 100-token window
scheme but are not the same experience for a reader.

A word is "content" here if it is at least ``min_length`` characters and not a
closed-class function word; see ``STOPWORDS`` below for why the small set in
``local_repetition.py`` needed extending rather than reuse as-is.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import MODERATE, finding, option, rate
from .local_repetition import COMMON as _BASE_STOPWORDS

# local_repetition.py's stopword set has only 20 entries and, being used there
# without a length filter, never needed the four-or-more-letter function words
# (this, that, have, were, which, ...) that would otherwise dominate this
# metric's default min_length=4 view of "content". It is extended rather than
# duplicated outright so the two repetition metrics keep the same closed-class
# core if that base set ever changes.
_EXTRA_STOPWORDS = {
    "this", "that", "these", "those", "have", "has", "had", "having",
    "were", "been", "being", "are", "am",
    "not", "nor", "neither", "either",
    "from", "into", "onto", "upon", "toward", "towards", "against", "without",
    "within", "along", "among", "across", "behind", "beyond", "except",
    "unless", "until", "since", "though", "although", "whether",
    "than", "then", "when", "where", "which", "what", "whose", "whom", "who",
    "why", "how",
    "will", "would", "should", "could", "might", "must", "shall", "can", "cannot",
    "just", "because", "while", "after", "before", "above", "below", "between",
    "during", "through", "about", "again", "further", "once", "here", "there",
    "their", "your", "yours", "mine", "ours", "theirs", "its", "hers", "him",
    "her", "them",
    "also", "only", "very", "such", "some", "more", "most", "other", "each",
    "every", "both", "does", "did", "done", "doing", "dont", "cant", "wont",
}
STOPWORDS = _BASE_STOPWORDS | _EXTRA_STOPWORDS

FAMILY = "repetition"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

WITHIN_CLOSE = 50
WITHIN_FAR = 200


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    min_length = int(option(config, "min_length", 4))
    tokens = analysis.tokens

    last_seen: dict[str, int] = {}
    gaps: list[int] = []
    close_hits: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for index, token in enumerate(tokens):
        if len(token) < min_length or token in STOPWORDS:
            continue
        previous = last_seen.get(token)
        if previous is not None:
            gap = index - previous
            gaps.append(gap)
            if gap <= WITHIN_CLOSE:
                close_hits[token].append((gap, previous, index))
        last_seen[token] = index

    ids = ("repetition.content_word_reuse_distance", "repetition.reuse_within_50_share",
           "repetition.reuse_within_200_share")
    names = ("Content-word reuse distance", "Reuses within 50 tokens",
             "Reuses within 200 tokens")
    if not gaps:
        warning = "no content word recurred in this text" if tokens else "no words to measure"
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE, warning=warning)
                for mid, name, unit in zip(ids, names, ("tokens", "%", "%"))]

    reuse_counts = Counter({word: len(hits) for word, hits in close_hits.items()})
    evidence = []
    for word, count in reuse_counts.most_common(20):
        hits = close_hits[word]
        evidence.append({
            "word": word, "reuses_within_50_tokens": count,
            "example_offsets": [{"previous_index": p, "index": i, "gap": g}
                                for g, p, i in hits[:3]],
        })

    distribution = summarize(gaps)
    within_close = rate(sum(1 for gap in gaps if gap <= WITHIN_CLOSE), len(gaps))
    within_far = rate(sum(1 for gap in gaps if gap <= WITHIN_FAR), len(gaps))

    return [
        finding(ids[0], names[0], distribution.get("median"), "tokens", family=FAMILY,
                sample_size=len(gaps), distribution=distribution, evidence=evidence,
                min_sample=MIN_SAMPLE),
        finding(ids[1], names[1], within_close, "%", family=FAMILY, sample_size=len(gaps),
                min_sample=MIN_SAMPLE),
        finding(ids[2], names[2], within_far, "%", family=FAMILY, sample_size=len(gaps),
                min_sample=MIN_SAMPLE),
    ]
