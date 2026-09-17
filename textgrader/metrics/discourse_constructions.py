"""Repeated rhetorical templates, discovered rather than listed.

This is the generalised, de-hardcoded replacement for the old ``measures/tics.py``
script, which hand-coded one author's habits as regular expressions found by a
human reading their own manuscript. That approach cannot see a different
writer's habit, and it stops being honest the moment it is pointed at anyone
else's prose.

The idea it keeps is sound: a sentence can be abstracted into a *skeleton* by
keeping its function words and punctuation and replacing everything else with
a placeholder. "It's not the cold, it's the wind." and "It's not the plan,
it's the timing." reduce to the same skeleton, ``it 's not _ , it 's _ .``,
whether or not anyone ever wrote down that this book likes that construction.
Counting skeleton frequencies finds repeated SHAPE automatically, for any
manuscript, in any voice.

A repeated shape is not automatically a defect. Some of the most recognisable
prose styles in print are built from a small set of deliberately reused
sentence shapes; this module reports the share of text that uses one, and how
concentrated the shape distribution is, and leaves the judgment of "voice" or
"tic" to the reader of the report, the same way rhythm and repetition metrics
elsewhere in this package do.

Function words come from :mod:`function_words`'s closed-class ``FUNCTION``
list, the same list :mod:`function_words` uses for Burrows's Delta. Clitics
("'s", "n't", "'re", ...) are always kept literally rather than tested against
that list, because they are grammatical morphemes in their own right and
splitting "it's" into "it" + "'s" (rather than treating "it's" as one content
token) is what lets the skeleton see past the contraction to the shape
underneath.

Cost: one linear pass over the sentences building a skeleton string each
(itself a linear pass over that sentence's characters), plus one counting
pass over the skeletons. Both are O(n) in the size of the text; there is no
sentence-against-sentence comparison.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import MODERATE, finding, option, rate
from .function_words import FUNCTION

FAMILY = "discourse"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

FUNCTION_SET = set(FUNCTION)

# A skeleton token is either a run of letters, a clitic (apostrophe plus
# letters, so "it's" tokenizes as "it" then "'s"), or any single character
# that is neither a letter, a digit nor whitespace. This is a purpose-built
# tokenizer for shape abstraction, distinct from the canonical word/sentence
# splitting in ``textgrader.text``: it must keep punctuation and clitics that
# canonical word tokens deliberately discard.
_TOKEN_RE = re.compile(r"[A-Za-z]+|'[A-Za-z]+|[^\sA-Za-z']")

EXAMPLE_TRUNCATE = 110


def _skeleton(sentence: str) -> str:
    parts: list[str] = []
    for token in _TOKEN_RE.findall(sentence):
        if token.startswith("'"):
            parts.append(token.lower())
        elif token[0].isalpha():
            lowered = token.lower()
            parts.append(lowered if lowered in FUNCTION_SET else "_")
        else:
            parts.append(token)
    return " ".join(parts)


def _entropy(counts: Mapping[str, int]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    max_reported = int(option(config, "max_reported", 30))
    min_count = int(option(config, "min_count", 3))

    sentences = analysis.sentences
    total = len(sentences)
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for sentence in sentences:
        key = _skeleton(sentence)
        if not key:
            continue
        counts[key] += 1
        if key not in examples:
            trimmed = sentence.strip()
            examples[key] = (trimmed[:EXAMPLE_TRUNCATE] + "..."
                             if len(trimmed) > EXAMPLE_TRUNCATE else trimmed)

    repeated = {key: count for key, count in counts.items() if count >= min_count}
    repeated_sentence_count = sum(repeated.values())
    share = rate(repeated_sentence_count, total, 100.0)

    top_skeletons = sorted(repeated.items(), key=lambda item: item[1], reverse=True)[:max_reported]
    evidence = [{"skeleton": key, "count": count, "example": examples[key]}
                for key, count in top_skeletons]

    warning = None if total else "no sentences to measure"
    out = [finding(
        "discourse.repeated_construction_share",
        f"Share of sentences matching a skeleton seen {min_count}+ times",
        share, "percent", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
        distribution={"distinct_skeletons": len(counts), "repeated_skeletons": len(repeated),
                      "min_count": min_count},
        evidence=evidence, warning=warning)]

    entropy = _entropy(counts)
    out.append(finding(
        "discourse.construction_entropy",
        "Entropy of the sentence-skeleton distribution (low = a few shapes dominate)",
        entropy, "bits", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
        distribution={"distinct_skeletons": len(counts),
                      "max_possible_bits": math.log2(len(counts)) if len(counts) > 1 else 0.0},
        evidence=evidence, warning=warning))

    top_count = top_skeletons[0][1] if top_skeletons else 0
    out.append(finding(
        "discourse.top_construction_count",
        "Occurrences of the single most-repeated sentence skeleton",
        top_count, "sentences", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
        distribution={"top_skeleton": top_skeletons[0][0] if top_skeletons else None},
        evidence=evidence,
        warning=warning or (None if top_skeletons else
                            f"no skeleton reached the minimum count of {min_count}")))
    return out
