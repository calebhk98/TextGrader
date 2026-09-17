"""The same core shapes, measured separately for dialogue and for narration.

Every other metric in this repository treats "the text" as one voice.  This
one asks the more basic question a dialogue-heavy manuscript actually needs
answered: does the way characters talk differ from the way the narrator
describes things?  If words-per-sentence, lexical diversity, word length and
punctuation use are all about the same in both channels, the characters do
not sound different from the narrator, which is a finding worth stating in
its own right rather than leaving buried inside one blended average.

Five shapes are measured in both ``analysis.dialogue`` and
``analysis.narration`` (themselves ``DocumentAnalysis`` views, already
computed by the shared pipeline): words per sentence, type-token ratio,
mean word length, punctuation marks per 1,000 words, and dialogue's share of
all words.  The first four are reported per channel *and* as a
dialogue-minus-narration gap, which is the actionable number: a gap near
zero means "this author writes dialogue and narration in the same register",
a large gap means the two channels are doing different work.

Type-token ratio here is not the sliding-window MATTR that ``mattr.py``
computes over the whole text (an O(n * window) pass that would be run twice
more expensively here).  Instead the channel's tokens are cut into
non-overlapping fixed-size windows and each window's TTR is reported as a
full distribution, which is O(n), cheaper, and, because dialogue and
narration usually differ hugely in length, more comparable across the two
channels than a single whole-channel ratio would be.

This module's own work is linear and cheap, but the first metric to touch
both ``analysis.dialogue`` and ``analysis.narration`` pays for sentence
segmentation of two full-length derived views.  Measured on a 400,000-word
book with pySBD installed (the default segmenter), that is about 6 of this
module's roughly 8 second total; with the built-in segmenter, or once
another metric has already forced those two views, it is well under a
second. That cost is a property of the shared pipeline's segmenter choice,
not of anything this module does, and the same cost is paid by any other
metric (``rhythm_autocorrelation`` among them) that measures narration
sentences separately from the whole text.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..stats import summarize
from .common import FAST, finding, option

FAMILY = "dialogue"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# Enough windows to make the TTR spread meaningful; below this a single value
# is still reported, just without much confidence in its shape.
TTR_MIN_SAMPLE = 3
DEFAULT_TTR_WINDOW = 200

PUNCTUATION_MARKS = {
    ";": "semicolon", ":": "colon", "(": "parenthesis", "…": "ellipsis",
    "!": "exclamation", "?": "question", "—": "em_dash", "–": "en_dash",
}


def _window_ttr(tokens: list[str], window: int) -> list[float]:
    """Type-token ratio of consecutive, non-overlapping ``window``-token chunks.

    A final chunk under half the window size is dropped rather than reported
    on its own, so a channel whose length is not a clean multiple of the
    window does not contribute one artificially noisy short-window value.
    """

    if not tokens:
        return []
    size = min(window, len(tokens))
    values = []
    for start in range(0, len(tokens), size):
        chunk = tokens[start:start + size]
        if len(chunk) < max(2, size // 2) and values:
            continue
        if len(chunk) >= 2:
            values.append(len(set(chunk)) / len(chunk))
    return values


def _punctuation_rate(view: DocumentAnalysis) -> tuple[float | None, dict[str, float]]:
    words = view.word_count
    if not words:
        return None, {}
    counts = Counter(view.text)
    per_mark = {name: 1000.0 * counts.get(char, 0) / words for char, name in PUNCTUATION_MARKS.items()}
    return sum(per_mark.values()), per_mark


def _channel_findings(view: DocumentAnalysis, window: int, suffix: str,
                      channel: str) -> tuple[list[dict[str, Any]], dict[str, Any],
                                             dict[str, Any], dict[str, Any], float | None]:
    lengths = view.sentence_lengths
    sentence_summary = summarize(lengths)
    ttr_values = _window_ttr(view.tokens, window)
    ttr_summary = summarize(ttr_values)
    word_lengths = [len(word) for word in view.words]
    word_length_summary = summarize(word_lengths)
    punct_rate, per_mark = _punctuation_rate(view)

    out = [
        finding(f"dialogue.words_per_sentence_{suffix}", f"Words per sentence ({channel})",
                sentence_summary.get("median"), "words", family=FAMILY, sample_size=len(lengths),
                distribution=sentence_summary, channel=channel, min_sample=MIN_SAMPLE,
                evidence=[{"first_sentence_lengths": lengths[:20]}] if lengths else None,
                warning=None if lengths else f"no sentences in {channel}"),
        finding(f"dialogue.ttr_{suffix}", f"Type-token ratio, {window}-word windows ({channel})",
                ttr_summary.get("median"), "ratio", family=FAMILY, sample_size=len(ttr_values),
                distribution=ttr_summary, channel=channel, min_sample=TTR_MIN_SAMPLE,
                warning=None if ttr_values else f"fewer than two words in {channel}"),
        finding(f"dialogue.word_length_{suffix}", f"Mean word length ({channel})",
                word_length_summary.get("median"), "characters", family=FAMILY,
                sample_size=len(word_lengths), distribution=word_length_summary, channel=channel,
                min_sample=MIN_SAMPLE, warning=None if word_lengths else f"no words in {channel}"),
        finding(f"dialogue.punctuation_{suffix}", f"Punctuation marks per 1,000 words ({channel})",
                punct_rate, "per 1,000 words", family=FAMILY, sample_size=view.word_count,
                channel=channel, min_sample=MIN_SAMPLE,
                evidence=[{"per_mark": per_mark}] if per_mark else None,
                warning=None if per_mark else f"no words in {channel}"),
    ]
    return out, sentence_summary, ttr_summary, word_length_summary, punct_rate


def _gap(metric_id: str, name: str, unit: str, dialogue_summary: Mapping[str, Any],
         narration_summary: Mapping[str, Any], min_sample: int) -> dict[str, Any]:
    d = dialogue_summary.get("median")
    n = narration_summary.get("median")
    value = (d - n) if d is not None and n is not None else None
    return finding(metric_id, name, value, unit, family=FAMILY, channel="full",
                   min_sample=min_sample, distribution={"dialogue_median": d, "narration_median": n},
                   warning=None if value is not None else
                   "need a usable measurement in both dialogue and narration to compare")


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    window = max(2, int(option(config, "ttr_window", DEFAULT_TTR_WINDOW)))
    dialogue_view = analysis.dialogue
    narration_view = analysis.narration

    d_findings, d_sent, d_ttr, d_wl, d_punct = _channel_findings(
        dialogue_view, window, "dialogue", "dialogue")
    n_findings, n_sent, n_ttr, n_wl, n_punct = _channel_findings(
        narration_view, window, "narration", "narration")

    out = d_findings + n_findings
    out.append(_gap("dialogue.words_per_sentence_gap",
                    "Words-per-sentence gap, dialogue minus narration", "words",
                    d_sent, n_sent, MIN_SAMPLE))
    out.append(_gap("dialogue.ttr_gap", "Type-token ratio gap, dialogue minus narration",
                    "ratio", d_ttr, n_ttr, TTR_MIN_SAMPLE))
    out.append(_gap("dialogue.word_length_gap", "Mean word length gap, dialogue minus narration",
                    "characters", d_wl, n_wl, MIN_SAMPLE))

    punct_gap = (d_punct - n_punct) if d_punct is not None and n_punct is not None else None
    out.append(finding("dialogue.punctuation_gap",
                       "Punctuation-rate gap, dialogue minus narration", punct_gap,
                       "per 1,000 words", family=FAMILY, channel="full", min_sample=MIN_SAMPLE,
                       distribution={"dialogue": d_punct, "narration": n_punct},
                       warning=None if punct_gap is not None else
                       "need a usable measurement in both dialogue and narration to compare"))

    share = analysis.dialogue_word_share
    out.append(finding("dialogue.word_share", "Share of words spoken as dialogue", share,
                       "percent", family=FAMILY, channel="full", sample_size=analysis.word_count,
                       min_sample=MIN_SAMPLE,
                       warning=None if analysis.word_count else "empty document"))
    return out
