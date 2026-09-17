"""Part-of-speech mix, open-class and closed-class, plus its entropy.

The legacy version only reported the open word classes (noun, proper noun,
verb, adjective, adverb): the words a lexical-choice reading usually cares
about.  Left out entirely were the closed classes that carry a text's
grammatical scaffolding: pronouns, adpositions, determiners and the two
conjunction classes, plus auxiliaries.  Those are just as diagnostic of style
(a pronoun-heavy, determiner-light mix reads very differently from a
determiner-heavy one) and are cheap to add since the parse already produces
them; they are now reported alongside the open classes under the same
``nlp.pos_<tag>`` id shape.

A per-tag share is still only half the picture: two documents can have
identical noun and verb shares while one spreads its remaining budget evenly
across the other eight tags and the other concentrates it in two.  The
entropy of the *full* observed tag distribution (not only the tags reported
individually) captures that in one number: low entropy means the text leans
on a small number of tags disproportionately, high entropy means a flatter,
more evenly used tag set.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import PARSE, finding, rate, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

OPEN_CLASS = ("NOUN", "PROPN", "VERB", "ADJ", "ADV")
CLOSED_CLASS = ("PRON", "ADP", "DET", "CCONJ", "SCONJ", "AUX")
KEEP = tuple(sorted(OPEN_CLASS + CLOSED_CLASS))

_LABELS = {
    "NOUN": "Noun", "PROPN": "Proper noun", "VERB": "Verb", "ADJ": "Adjective",
    "ADV": "Adverb", "PRON": "Pronoun", "ADP": "Adposition", "DET": "Determiner",
    "CCONJ": "Coordinating conjunction", "SCONJ": "Subordinating conjunction", "AUX": "Auxiliary",
}


def _label_entropy(counter: Counter[str], total: int) -> float | None:
    """Entropy, in bits, of a categorical distribution given as raw counts."""

    if not total:
        return None
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count) + 0.0


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    if analysis.nlp_unavailable:
        out = [unavailable(f"nlp.pos_{tag.lower()}", f"{_LABELS[tag]} share",
                           analysis.nlp_unavailable, family=FAMILY) for tag in KEEP]
        out.append(unavailable("nlp.pos_entropy", "Entropy of the POS-tag distribution",
                               analysis.nlp_unavailable, family=FAMILY))
        return out

    counter: Counter[str] = Counter()
    total = 0
    for _, doc in analysis.spacy_docs():
        for token in doc:
            if not token.is_alpha:
                continue
            counter[token.pos_] += 1
            total += 1

    no_data = "no alphabetic tokens were parsed"
    out = [finding(f"nlp.pos_{tag.lower()}", f"{_LABELS[tag]} share",
                   rate(counter[tag], total) if total else None, "%", family=FAMILY,
                   sample_size=total, min_sample=MIN_SAMPLE, warning=None if total else no_data)
           for tag in KEEP]
    out.append(finding(
        "nlp.pos_entropy", "Entropy of the POS-tag distribution",
        _label_entropy(counter, total), "bits", family=FAMILY, sample_size=total,
        min_sample=MIN_SAMPLE, distribution={"distinct_tags": len(counter), "tag_counts": dict(counter)},
        warning=None if total else no_data))
    return out
