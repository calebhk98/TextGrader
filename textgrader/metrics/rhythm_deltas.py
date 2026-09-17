"""Sentence-to-sentence change in length, not just its order.

:mod:`rhythm_autocorrelation` asks whether successive sentence lengths are
correlated.  This module asks a plainer question about the same fact: how big
is the jump from one sentence to the next, and how often is there barely any
jump at all?  A text that moves "12, 13, 11, 14, 12, 13" sentence by sentence
has a near-zero autocorrelation signature that looks unremarkable, but every
delta is tiny; that is a metronome an autocorrelation number alone can miss,
and it is exactly the kind of flatness that makes generated prose feel inert.

Two shapes are published: the unsigned jump size (how much length changes,
regardless of direction) and the signed jump (whether prose tends to expand,
contract, or has no directional bias at all - a long tail of one sign would
mean sentences drift steadily longer or shorter through the passage).  A third
finding turns "how flat is this" into a single number: the share of adjacent
pairs that barely change (<= 2 words) and the entropy of the delta mix.  Real
prose has some flat pairs by chance; a text made almost entirely of them is
not choosing rhythm, it is failing to vary it.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import shannon_entropy, summarize
from .common import FAST, finding, rate

FAMILY = "sentence_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# A change of two words or fewer reads as "about the same length" to a reader;
# it is small enough to be noise in ordinary composition but, if it dominates
# the whole distribution of pairs, it is the text's actual rhythm.
FLAT_THRESHOLD = 2


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    lengths = analysis.sentence_lengths
    pair_count = max(0, len(lengths) - 1)
    if pair_count == 0:
        warning = (f"needs at least two sentences; this text has {len(lengths)}")
        return [
            finding("rhythm.sentence_length_delta_abs", "Absolute sentence-length delta",
                    None, "words", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.sentence_length_delta_signed", "Signed sentence-length delta",
                    None, "words", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
            finding("rhythm.sentence_length_flat_pair_rate", "Flat adjacent-sentence-pair rate",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    warning=warning),
        ]

    signed = [lengths[i + 1] - lengths[i] for i in range(pair_count)]
    absolute = [abs(delta) for delta in signed]
    flat_count = sum(1 for delta in absolute if delta <= FLAT_THRESHOLD)

    abs_summary = summarize(absolute)
    signed_summary = summarize(signed)
    flat_share = rate(flat_count, pair_count)
    delta_entropy = shannon_entropy(signed)

    return [
        finding("rhythm.sentence_length_delta_abs", "Absolute sentence-length delta",
                abs_summary.get("median"), "words", family=FAMILY, sample_size=pair_count,
                distribution=abs_summary, min_sample=MIN_SAMPLE,
                evidence=[{"first_deltas": absolute[:20]}]),
        finding("rhythm.sentence_length_delta_signed", "Signed sentence-length delta",
                signed_summary.get("median"), "words", family=FAMILY, sample_size=pair_count,
                distribution=signed_summary, min_sample=MIN_SAMPLE,
                evidence=[{"first_deltas": signed[:20]}]),
        finding("rhythm.sentence_length_flat_pair_rate",
                f"Share of adjacent sentence pairs within {FLAT_THRESHOLD} words of each other",
                flat_share, "percent", family=FAMILY, sample_size=pair_count,
                distribution={"flat_threshold_words": FLAT_THRESHOLD, "flat_pairs": flat_count,
                              "total_pairs": pair_count, "delta_entropy_bits": delta_entropy},
                min_sample=MIN_SAMPLE),
    ]
