"""A full punctuation profile: every mark, not just the seven the original
``punctuation`` module happened to count.

``punctuation.py`` skips commas, periods and quotation marks entirely, which
throws away the marks that carry most of a prose style's texture. This module
covers comma, period, question mark, exclamation mark, semicolon, colon, em
dash, en dash, hyphen, ellipsis, double quote, single quote/apostrophe,
parenthesis and bracket.

Two independent quirks of written English are folded together deliberately,
and said so in each finding's name:

* straight and curly variants of a quote mark (``"`` and ``"``/``"``, ``'`` and
  ``'``/``'``) are counted as the same mark. A writer, or their typesetting
  pipeline, does not choose one on purpose.
* "single quote/apostrophe" is one bucket. There is no way to tell, from
  punctuation alone, an apostrophe in "don't" from a single-quoted aside, so
  this count is dominated by contractions and possessives; it is not a
  dialogue-style signal the way the double-quote count is.

Every mark is reported both per 1,000 words (comparable across texts of
different length) and per sentence (comparable across texts with different
sentence length, which matters because a mark like the comma is fundamentally
a sentence-internal event). Comma placement is the one whose *shape* matters
most, so its per-sentence counts are also published as a full distribution
rather than a single rate: a text that never uses a comma and one that uses
exactly two in every sentence can share a mean and be nothing alike.

Dialogue and narration are compared directly for the marks that plausibly
differ in kind rather than just in rate: comma (dialogue tends to run short,
comma-spliced clauses), em dash (a narration device as often as a speech
interruption) and ellipsis (trailing off is a speech mannerism). This is
attached as ``details`` on the whole-document finding for em dash and
ellipsis, and as its own metric for comma, which gets first-class treatment
per the brief.

Dependency-free: a single regex pass over the text, no spaCy, no re-splitting
of sentences (segmentation comes from ``analysis.sentences``).

Performance note: this module is linear and cheap by itself (well under a
second on a 400,000-word book), but reading the dialogue and narration
channels' *sentence counts* is not free when ``pysbd`` is the configured
segmenter, because ``analysis.dialogue``/``analysis.narration`` are freshly
built ``DocumentAnalysis`` views whose ``.sentences`` must be resegmented from
scratch rather than sliced out of the whole-document sentence list. Measured
on a 400,000-word synthetic book with heavy dialogue: about 8.8 seconds total,
almost all of it inside pysbd re-segmenting the narration and dialogue text.
This is a cost of the shared pipeline's segmenter choice, not of counting
punctuation, and it is paid once per report if another metric (this family's
``comma_rate_dialogue``/``comma_rate_narration`` findings need it regardless
of what else runs) also touches those channels, since they are cached on the
``DocumentAnalysis`` instance. Re-splitting sentences locally to avoid this
would violate the "never re-split yourself" rule, so it is accepted and
reported here rather than worked around.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import MODERATE, finding, rate

FAMILY = "punctuation"
# Counting marks is linear and fast on its own; this is MODERATE (matching the
# registry) because the comma/em-dash/ellipsis dialogue-vs-narration findings
# need those channels' sentence counts, which can force a costly resegmentation
# under the pysbd segmenter. See the performance note above.
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# Order matches the brief. Longer/more specific alternatives are tried first
# so "--" or "..." are not double-counted as runs of hyphens or periods.
MARK_RE = re.compile(
    r"(?P<ellipsis>\.{3,}|…)"
    r"|(?P<em_dash>-{2,}|—)"
    r"|(?P<en_dash>–)"
    r"|(?P<double_quote>[\"“”])"
    r"|(?P<single_quote>['‘’])"
    r"|(?P<comma>,)"
    r"|(?P<period>\.)"
    r"|(?P<question>\?)"
    r"|(?P<exclamation>!)"
    r"|(?P<semicolon>;)"
    r"|(?P<colon>:)"
    r"|(?P<hyphen>-)"
    r"|(?P<parenthesis>[()])"
    r"|(?P<bracket>[\[\]{}])"
)

MARK_NAMES = ("comma", "period", "question", "exclamation", "semicolon", "colon",
              "em_dash", "en_dash", "hyphen", "ellipsis", "double_quote",
              "single_quote", "parenthesis", "bracket")

MARK_LABELS = {
    "comma": "Comma", "period": "Period", "question": "Question mark",
    "exclamation": "Exclamation mark", "semicolon": "Semicolon", "colon": "Colon",
    "em_dash": "Em dash", "en_dash": "En dash", "hyphen": "Hyphen",
    "ellipsis": "Ellipsis (curly or three-dot, counted together)",
    "double_quote": "Double quote (straight and curly counted together)",
    "single_quote": "Single quote/apostrophe (straight and curly counted together)",
    "parenthesis": "Parenthesis", "bracket": "Bracket",
}


def count_marks(text: str) -> Counter:
    """Every punctuation mark in ``text``, classified by :data:`MARK_RE`."""

    return Counter(match.lastgroup for match in MARK_RE.finditer(text))


def marks_for(view: DocumentAnalysis) -> Counter:
    """Counted once per view. Re-scanning the text per mark made this metric
    twenty times slower than every other dependency-free measurement."""

    return view.memo("punct.marks", lambda: count_marks(view.text))


def _channel_rate(view: DocumentAnalysis, mark: str) -> tuple[float | None, int]:
    count = marks_for(view).get(mark, 0)
    sentences = view.sentence_count
    return (count / sentences if sentences else None), sentences


def _empty(words_total: int, sentences_total: int) -> list[dict[str, Any]]:
    warning = "no words in text" if not words_total else "no sentences in text"
    out: list[dict[str, Any]] = []
    for mark in MARK_NAMES:
        label = MARK_LABELS[mark]
        out.append(finding(f"punct.{mark}_per_1000_words", f"{label} per 1,000 words",
                            None, "per_1000_words", family=FAMILY, sample_size=words_total,
                            min_sample=MIN_SAMPLE, warning=warning))
        out.append(finding(f"punct.{mark}_per_sentence", f"{label} per sentence",
                            None, "per_sentence", family=FAMILY, sample_size=sentences_total,
                            min_sample=MIN_SAMPLE, warning=warning))
    out.append(finding("punct.commas_per_sentence_shape", "Comma count per sentence, shape",
                        None, "commas", family=FAMILY, sample_size=sentences_total,
                        min_sample=MIN_SAMPLE, warning=warning))
    out.append(finding("punct.comma_rate_dialogue", "Commas per sentence, dialogue only",
                        None, "per_sentence", family=FAMILY, channel="dialogue",
                        min_sample=MIN_SAMPLE, warning=warning))
    out.append(finding("punct.comma_rate_narration", "Commas per sentence, narration only",
                        None, "per_sentence", family=FAMILY, channel="narration",
                        min_sample=MIN_SAMPLE, warning=warning))
    return out


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    sentences_total = analysis.sentence_count
    if not words_total or not sentences_total:
        return _empty(words_total, sentences_total)

    from ..stats import summarize  # local import: keeps module import cheap

    counts = marks_for(analysis)
    out: list[dict[str, Any]] = []
    for mark in MARK_NAMES:
        count = counts.get(mark, 0)
        label = MARK_LABELS[mark]
        per_1000 = rate(count, words_total, 1000.0)
        per_sentence = rate(count, sentences_total, 1.0)
        details = None
        if mark in ("em_dash", "ellipsis"):
            dialogue_rate, dialogue_n = _channel_rate(analysis.dialogue, mark)
            narration_rate, narration_n = _channel_rate(analysis.narration, mark)
            details = [
                {"channel": "dialogue", "per_sentence": dialogue_rate, "sample_size": dialogue_n},
                {"channel": "narration", "per_sentence": narration_rate, "sample_size": narration_n},
            ]
        out.append(finding(f"punct.{mark}_per_1000_words", f"{label} per 1,000 words",
                            per_1000, "per_1000_words", family=FAMILY, sample_size=words_total,
                            min_sample=MIN_SAMPLE, evidence=[{"count": count}]))
        out.append(finding(f"punct.{mark}_per_sentence", f"{label} per sentence",
                            per_sentence, "per_sentence", family=FAMILY,
                            sample_size=sentences_total, min_sample=MIN_SAMPLE,
                            details=details))

    comma_counts = [sentence.count(",") for sentence in analysis.sentences]
    comma_summary = summarize(comma_counts)
    out.append(finding(
        "punct.commas_per_sentence_shape", "Comma count per sentence, shape",
        comma_summary.get("median"), "commas", family=FAMILY, sample_size=sentences_total,
        distribution=comma_summary, min_sample=MIN_SAMPLE,
        evidence=[{"first_sentence_counts": comma_counts[:20]}]))

    dialogue = analysis.dialogue
    narration = analysis.narration
    dialogue_rate, dialogue_n = _channel_rate(dialogue, "comma")
    narration_rate, narration_n = _channel_rate(narration, "comma")
    out.append(finding("punct.comma_rate_dialogue", "Commas per sentence, dialogue only",
                        dialogue_rate, "per_sentence", family=FAMILY, channel="dialogue",
                        sample_size=dialogue_n, min_sample=MIN_SAMPLE,
                        warning=None if dialogue_n else "no dialogue found in this text"))
    out.append(finding("punct.comma_rate_narration", "Commas per sentence, narration only",
                        narration_rate, "per_sentence", family=FAMILY, channel="narration",
                        sample_size=narration_n, min_sample=MIN_SAMPLE,
                        warning=None if narration_n else "no narration found in this text"))
    return out
