"""Shannon entropy of sentence length: how much rhythmic vocabulary is in use.

A text can have a healthy mean and standard deviation while still drawing its
sentences from a narrow set of lengths repeated over and over - eight words,
nine words, eight words, twenty, eight words, nine words.  Entropy answers a
different question than dispersion does: not "how far apart are the values"
but "how many distinct lengths carry real weight, and how evenly". A low
entropy relative to the range of lengths actually used is a narrow rhythmic
vocabulary, which reads as mechanical even when no two sentences are
identical.

Raw entropy in bits is not comparable across texts with different length
ranges: a text limited to lengths 5-10 has a lower ceiling than one spanning
5-40 even if both use their available lengths with equal evenness.  The
normalized form divides by the maximum possible entropy for the number of
distinct lengths actually observed (log2 of that count), so two texts can be
compared on evenness alone.  The modal-length share is the plainest version of
the same idea: what fraction of all sentences sit at the single most common
length.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import shannon_entropy
from .common import FAST, finding, rate, top

FAMILY = "sentence_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    lengths = analysis.sentence_lengths
    total = len(lengths)
    if total == 0:
        warning = "no sentences to measure"
        return [
            finding("rhythm.sentence_length_entropy", "Sentence-length entropy",
                    None, "bits", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.sentence_length_entropy_normalized",
                    "Sentence-length entropy, normalized for range",
                    None, "ratio", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.modal_sentence_length_share", "Share of sentences at the modal length",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    counts = Counter(lengths)
    distinct = len(counts)
    entropy = shannon_entropy(lengths) or 0.0
    max_entropy = math.log2(distinct) if distinct > 1 else 0.0
    normalized = (entropy / max_entropy) if max_entropy > 0 else 0.0
    modal_length, modal_count = counts.most_common(1)[0]
    modal_share = rate(modal_count, total)

    return [
        finding("rhythm.sentence_length_entropy", "Sentence-length entropy",
                entropy, "bits", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                distribution={"distinct_lengths": distinct, "max_possible_bits": max_entropy},
                evidence=[{"length": length, "count": count}
                          for length, count in counts.most_common(10)]),
        finding("rhythm.sentence_length_entropy_normalized",
                "Sentence-length entropy, normalized for range",
                normalized, "ratio", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                distribution={"distinct_lengths": distinct, "raw_entropy_bits": entropy},
                warning=None if distinct > 1 else
                "every sentence has the same length; normalization is undefined so 0.0 was used"),
        finding("rhythm.modal_sentence_length_share", "Share of sentences at the modal length",
                modal_share, "percent", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                distribution={"modal_length_words": modal_length, "modal_count": modal_count},
                evidence=top(counts, 10, key="length_words")),
    ]
