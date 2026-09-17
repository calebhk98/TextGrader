"""How often sentences start on the same few words.

A repeated opening ("She looked..." "She turned..." "She felt...") is one of
the most reader-visible forms of repetition, and one an author is least
likely to notice themselves, because each sentence is read in isolation while
writing it.  The repeated-share number answers "how much of this text opens
predictably"; it says nothing about how *concentrated* the repetition is,
though, and that is where an average misleads: a text where every sentence
opens on one of three rotating phrases and a text where ten different
openings repeat exactly twice each can have the same repeated-share while
reading completely differently.  The entropy of the opening distribution
(bits, over the sentences that share an opening with at least one other
sentence) tells them apart: low entropy means a small closed set of stock
openings, high entropy means many different openings each repeating a little.

Narration is measured separately from the whole text because dialogue turns
routinely open the same way for structural reasons ("I ", "You ", "No,") that
have nothing to do with the narrator's habits.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from .common import FAST, finding, option, rate, tokens as tokenize

FAMILY = "repetition"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

DEFAULT_WORDS = 3


def _opening_entropy(counter: Counter, total: int) -> float | None:
    """Shannon entropy, in bits, of the opening-phrase distribution.

    ``stats.shannon_entropy`` bins a numeric sample; an opening phrase is a
    label, not a number, so its probabilities are computed directly from the
    counts here instead.
    """

    if not total:
        return None
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count) + 0.0


def _openings(sentences: Sequence[str], words: int) -> list[str]:
    out = []
    for sentence in sentences:
        prefix = tokenize(sentence)[:words]
        if prefix:
            out.append(" ".join(prefix))
    return out


def _channel_findings(sentences: Sequence[str], words: int, suffix: str,
                      label: str) -> list[dict[str, Any]]:
    openings = _openings(sentences, words)
    total = len(openings)
    counter = Counter(openings)
    repeated = sum(count for count in counter.values() if count > 1)
    if total == 0:
        warning = "no sentences to measure opening repetition over"
        return [
            finding(f"style.sentence_opening_repetition{suffix}",
                    f"Repeated sentence openings{label}", None, "%", family=FAMILY,
                    sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding(f"style.sentence_opening_entropy{suffix}",
                    f"Entropy of sentence-opening distribution{label}", None, "bits",
                    family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                    sample_size_sensitive=True, warning=warning),
        ]
    evidence = [{"opening": opening, "count": count}
                for opening, count in counter.most_common(25) if count > 1]
    return [
        finding(f"style.sentence_opening_repetition{suffix}",
                f"Repeated sentence openings{label}", rate(repeated, total), "%",
                family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE, evidence=evidence),
        finding(f"style.sentence_opening_entropy{suffix}",
                f"Entropy of sentence-opening distribution{label}",
                _opening_entropy(counter, total), "bits", family=FAMILY, sample_size=total,
                min_sample=MIN_SAMPLE,
                # More text means more distinct openings, so this rises with
                # sample size whatever the prose is doing. Graded against a
                # corpus of whole novels it flagged all 53 chapters of a book.
                sample_size_sensitive=True,
                distribution={"distinct_openings": len(counter), "max_words": words}),
    ]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    words = max(1, int(option(config, "words", DEFAULT_WORDS)))
    out = _channel_findings(analysis.sentences, words, "", "")
    narration = analysis.narration
    if narration.sentence_count >= MIN_SAMPLE and narration.sentence_count < analysis.sentence_count:
        out.extend(_channel_findings(narration.sentences, words, "_narration", " (narration only)"))
    return out
