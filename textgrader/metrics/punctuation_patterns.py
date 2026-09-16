"""Repeated punctuation shapes: the mechanical tell that a mean rate misses.

A "comma rate" can look perfectly normal while every third sentence is built
from the exact same punctuation skeleton -- ``,",".`` over and over -- because
a rate only sees totals, never sequence. This module reduces each sentence to
its skeleton: the ordered sequence of punctuation marks it contains (using the
same mark classification as ``punctuation_profile``, so ``"He said," she
said.`` becomes ``",".`` and a bare declarative becomes ``.``). Each paragraph
is then reduced to the sequence of its sentences' skeletons, joined with
``|``, which catches paragraph-level formula ("action beat. Dialogue,
tag." repeated verbatim) that sentence-level repetition alone would not.

Four questions are asked of the result:

* what share of sentences have a skeleton that recurs somewhere else in the
  text (mechanical, but not necessarily consecutive);
* what share of *adjacent* sentence pairs share a skeleton (mechanical *and*
  back to back, which reads worse);
* the longest unbroken run of sentences sharing one skeleton;
* the same repetition question one level up, at the paragraph level, because
  "paragraph after paragraph with identical structure" is a distinct and
  worse failure than any one sentence-level statistic captures.

Cost is MODERATE: building a skeleton is a regex pass per sentence rather than
one pass over the whole text, and the reported evidence needs a first example
per repeated skeleton.

Performance note: grouping sentence skeletons into paragraphs uses
``analysis.paragraph_sentence_counts``, which (like ``analysis.sentences``)
comes from ``DocumentAnalysis`` and, under the ``pysbd`` segmenter, resplits
every paragraph independently rather than reusing the whole-document sentence
list. Measured on a 400,000-word book: about 6.8 seconds total, the large
majority of it inside that resegmentation rather than in this module's own
skeleton-building or counting. This is a cost of the shared pipeline, not of
this metric, and re-deriving paragraph sentence counts locally would mean
re-splitting sentences ourselves, which the module contract forbids.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import run_lengths
from .common import MODERATE, finding, option, rate
from .punctuation_profile import MARK_RE

FAMILY = "punctuation"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False


def skeleton(sentence: str) -> str:
    """The ordered punctuation-only sequence of a sentence, e.g. ``",".``."""

    return "".join(match.group() for match in MARK_RE.finditer(sentence))


def _repeated_evidence(skeletons: list[str], texts: list[str], limit: int) -> list[dict[str, Any]]:
    counts = Counter(skeletons)
    first_example: dict[str, str] = {}
    for pattern, text in zip(skeletons, texts):
        first_example.setdefault(pattern, text)
    out = []
    for pattern, count in counts.most_common():
        if count < 2 or not pattern:
            continue
        out.append({"skeleton": pattern, "count": count,
                    "example": first_example[pattern][:120]})
        if len(out) >= limit:
            break
    return out


def _share_repeated(skeletons: list[str]) -> tuple[float | None, int]:
    counts = Counter(skeletons)
    repeated = sum(count for count in counts.values() if count > 1)
    return rate(repeated, len(skeletons), 100.0), repeated


def _max_run(skeletons: list[str]) -> int:
    runs = run_lengths(skeletons)
    return max((max(lengths) for lengths in runs.values()), default=0)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    max_reported = int(option(config, "max_reported", 25))
    sentences = analysis.sentences
    total_sentences = len(sentences)
    if total_sentences == 0:
        warning = "no sentences to measure"
        return [
            finding("punct.repeated_sentence_skeleton_share", "Share of sentences with a repeated punctuation shape",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("punct.adjacent_identical_skeleton_rate", "Adjacent sentence pairs with an identical punctuation shape",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("punct.max_skeleton_run", "Longest run of consecutive sentences with an identical punctuation shape",
                    None, "sentences", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
            finding("punct.repeated_paragraph_skeleton_share", "Share of paragraphs with a repeated punctuation shape",
                    None, "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
        ]

    sentence_skeletons = [skeleton(sentence) for sentence in sentences]
    share_repeated, repeated_count = _share_repeated(sentence_skeletons)
    evidence = _repeated_evidence(sentence_skeletons, sentences, max_reported)

    pairs = list(zip(sentence_skeletons, sentence_skeletons[1:]))
    identical_pairs = sum(1 for a, b in pairs if a == b)
    adjacent_rate = rate(identical_pairs, len(pairs), 100.0) if pairs else None

    max_run = _max_run(sentence_skeletons)

    out = [
        finding("punct.repeated_sentence_skeleton_share",
                "Share of sentences with a punctuation shape used elsewhere in the text",
                share_repeated, "percent", family=FAMILY, sample_size=total_sentences,
                min_sample=MIN_SAMPLE, evidence=evidence,
                distribution={"distinct_skeletons": len(set(sentence_skeletons)),
                              "repeated_sentence_count": repeated_count}),
        finding("punct.adjacent_identical_skeleton_rate",
                "Share of adjacent sentence pairs sharing a punctuation shape",
                adjacent_rate, "percent", family=FAMILY, sample_size=len(pairs),
                min_sample=MIN_SAMPLE,
                warning=None if pairs else "needs at least two sentences"),
        finding("punct.max_skeleton_run",
                "Longest run of consecutive sentences with an identical punctuation shape",
                max_run, "sentences", family=FAMILY, sample_size=total_sentences,
                min_sample=MIN_SAMPLE),
    ]

    paragraph_skeletons: list[str] = []
    index = 0
    for count in analysis.paragraph_sentence_counts:
        paragraph_skeletons.append("|".join(sentence_skeletons[index:index + count]))
        index += count
    total_paragraphs = len(paragraph_skeletons)
    if total_paragraphs:
        paragraph_share, paragraph_repeated = _share_repeated(paragraph_skeletons)
        paragraph_evidence = _repeated_evidence(
            paragraph_skeletons, analysis.paragraphs, max_reported)
        out.append(finding(
            "punct.repeated_paragraph_skeleton_share",
            "Share of paragraphs whose sentence-by-sentence punctuation shape recurs "
            "elsewhere in the text",
            paragraph_share, "percent", family=FAMILY, sample_size=total_paragraphs,
            min_sample=min(MIN_SAMPLE, 10), evidence=paragraph_evidence,
            distribution={"distinct_paragraph_skeletons": len(set(paragraph_skeletons)),
                          "repeated_paragraph_count": paragraph_repeated,
                          "longest_paragraph_run": _max_run(paragraph_skeletons)}))
    else:
        out.append(finding(
            "punct.repeated_paragraph_skeleton_share",
            "Share of paragraphs whose sentence-by-sentence punctuation shape recurs "
            "elsewhere in the text",
            None, "percent", family=FAMILY, sample_size=0, min_sample=min(MIN_SAMPLE, 10),
            warning="no paragraphs to measure"))
    return out
