"""Which section of a book reads unlike the rest of it, and on what.

A style report for a whole novel is one number per metric; it cannot say
"chapter 14 is the one that changes voice". This module and its two siblings
(``drift_rolling``, ``drift_change_points``) all work over *sections* of one
document rather than over a corpus of documents, because the question is
"does this book agree with itself" rather than "does this book match other
books".

Sections come from ``analysis.sections`` (a Markdown-heading split) when that
yields at least three parts; a document with no headings, or only one or two,
falls back to ``analysis.windows(window_words)``, fixed-size paragraph-aligned
slices. Every finding's evidence records which of the two was used, because
"chapter 3" means something different from "the third 2,500-word window".

This module owns the feature extraction shared by all three drift modules
(:func:`section_features`, :data:`FEATURE_NAMES`, :func:`get_sections`):
``drift_rolling`` and ``drift_change_points`` both import it from here rather
than repeating it, so the definition of "a section's style fingerprint" is
made once. The features are all dependency-free and each answers a distinct
question a mean cannot: sentence-length center and spread, paragraph size,
average word length (a lexical-density proxy), comma density, how much of the
section is dialogue, lexical diversity over a fixed window (so length does not
confound it), and how choppy the prose is (share of very short sentences).

The per-section deviation itself is a robust z-score, via ``stats.compare``,
of each section's feature value against the *other* sections of the same
book. ``stats.compare`` already implements the median/MAD -> median/IQR ->
empirical-percentile fallback ladder, so a feature that happens to be
constant across most sections (MAD 0) does not turn one section's very
existence into an infinite-looking outlier; that ladder is reused rather than
reimplemented here. A robust z-score needs at least a handful of sections to
mean anything, so fewer than four (headed or windowed) is reported as
insufficient data rather than as a spuriously confident deviation.

Performance note: computing ``section_features`` needs each section's own
sentence lengths, and ``analysis.sections``/``analysis.windows`` build fresh
``DocumentAnalysis`` views that resegment their own text rather than slicing
the whole document's cached sentence list. Under the ``pysbd`` segmenter this
dominates the module's cost: measured on a 400,000-word book with no
headings (156 windows), about 7.1 seconds total, nearly all of it inside
pysbd resegmenting each window rather than in the feature arithmetic or
``stats.compare`` itself, and it did not shrink when tried with far fewer,
larger windows -- the cost tracks total words resegmented, not call count.
This is a cost of the shared pipeline's segmenter choice; a book split by
real chapter headings pays it once per chapter rather than once per
2,500-word window, but the total is comparable either way for a book this
size. Avoiding it would mean re-splitting sentences locally, which the module
contract forbids.
"""

from __future__ import annotations

import statistics
from typing import Any, Mapping, Sequence

from .. import stats as stats_module
from ..document import DocumentAnalysis
from .common import MODERATE, finding, option, rate

FAMILY = "book_drift"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 4
UNIT_SENSITIVE = False

# A section shorter than this has too little text for its own type-token
# ratio to mean anything; whatever tokens it has are used instead.
TTR_WINDOW = 500

# Sentences at or under this length are "choppy" for the short_sentence_share
# feature; matches the plain-language convention used elsewhere in the repo.
SHORT_SENTENCE_MAX = 10

# A robust z-score above this is called "deviating". Lower than the 3.5 used
# for corpus outliers (see stats.OUTLIER_DISTANCE) because the reference group
# here is only the book's other sections -- often single digits of them, not
# a corpus of dozens of books -- so a stricter bar would almost never fire.
DEVIATION_THRESHOLD = 2.5

FEATURE_NAMES = (
    "mean_sentence_length", "sentence_length_cv", "mean_paragraph_words",
    "mean_word_length", "comma_rate_per_sentence", "dialogue_word_share",
    "type_token_ratio", "short_sentence_share",
)


def get_sections(analysis: DocumentAnalysis,
                  window_words: int) -> tuple[list[tuple[str, DocumentAnalysis]], str]:
    """``(sections, method)`` where ``method`` is ``"heading"`` or ``"window"``.

    Every drift finding's evidence should record ``method`` so a reader knows
    whether "section 3" is an actual chapter or an arbitrary word-count slice.
    """

    return analysis.memo(f"drift.sections:{window_words}",
                         lambda: _build_sections(analysis, window_words))


def _build_sections(analysis: DocumentAnalysis,
                    window_words: int) -> tuple[list[tuple[str, DocumentAnalysis]], str]:
    headed = analysis.sections
    if len(headed) >= 3:
        return [(title or f"section {index + 1}", view)
                for index, (title, view) in enumerate(headed)], "heading"
    windows = analysis.windows(window_words)
    return [(f"window {index + 1}", view) for index, view in enumerate(windows)], "window"


def section_features(view: DocumentAnalysis) -> dict[str, float | None]:
    """A small, cheap style fingerprint for one section.

    Memoized on the section itself: the three book-drift metrics want the same
    fingerprint, and recomputing it meant segmenting every section three times.
    """

    return view.memo("drift.features", lambda: _section_features(view))


def _section_features(view: DocumentAnalysis) -> dict[str, float | None]:
    sentence_lengths = view.sentence_lengths
    paragraph_lengths = view.paragraph_lengths
    words = view.words
    sentence_total = len(sentence_lengths)

    mean_sentence_length = statistics.fmean(sentence_lengths) if sentence_lengths else None
    sentence_length_cv = None
    if len(sentence_lengths) > 1 and mean_sentence_length:
        sd = statistics.stdev(sentence_lengths)
        sentence_length_cv = 100 * sd / mean_sentence_length
    mean_paragraph_words = statistics.fmean(paragraph_lengths) if paragraph_lengths else None
    mean_word_length = statistics.fmean(len(word) for word in words) if words else None
    comma_rate_per_sentence = (view.text.count(",") / sentence_total) if sentence_total else None
    dialogue_word_share = view.dialogue_word_share

    tokens = view.tokens
    window = tokens[:TTR_WINDOW] if tokens else []
    type_token_ratio = (len(set(window)) / len(window)) if window else None

    short_sentence_share = rate(
        sum(1 for length in sentence_lengths if length <= SHORT_SENTENCE_MAX),
        sentence_total, 100.0) if sentence_total else None

    return {
        "mean_sentence_length": mean_sentence_length,
        "sentence_length_cv": sentence_length_cv,
        "mean_paragraph_words": mean_paragraph_words,
        "mean_word_length": mean_word_length,
        "comma_rate_per_sentence": comma_rate_per_sentence,
        "dialogue_word_share": dialogue_word_share,
        "type_token_ratio": type_token_ratio,
        "short_sentence_share": short_sentence_share,
    }


def _insufficient(method: str | None, count: int) -> list[dict[str, Any]]:
    warning = (f"only {count} section(s) available (method={method}); need at least "
               f"{MIN_SAMPLE} to compare a book against itself" if method else
               "text has no measurable sections")
    return [
        finding("drift.max_section_deviation", "Largest per-section style deviation",
                None, "robust z", family=FAMILY, sample_size=count, min_sample=MIN_SAMPLE,
                warning=warning),
        finding("drift.deviating_section_count", "Sections that deviate from the rest of the book",
                None, "sections", family=FAMILY, sample_size=count, min_sample=MIN_SAMPLE,
                warning=warning),
        finding("drift.section_deviation", "Distribution of per-section maximum deviation",
                None, "robust z", family=FAMILY, sample_size=count, min_sample=MIN_SAMPLE,
                warning=warning),
    ]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    window_words = int(option(config, "window_words", 2500))
    sections, method = get_sections(analysis, window_words)
    if len(sections) < MIN_SAMPLE:
        return _insufficient(method, len(sections))

    features = [section_features(view) for _, view in sections]
    n = len(sections)

    per_section_z: list[dict[str, float]] = []
    for i in range(n):
        row: dict[str, float] = {}
        for feature in FEATURE_NAMES:
            value = features[i][feature]
            if value is None:
                continue
            reference = [features[j][feature] for j in range(n)
                        if j != i and features[j][feature] is not None]
            comparison = stats_module.compare(value, reference)
            if comparison.robust_distance is not None:
                row[feature] = comparison.robust_distance
        per_section_z.append(row)

    section_max = [max((abs(z) for z in row.values()), default=0.0) for row in per_section_z]
    worst_index = max(range(n), key=lambda i: section_max[i]) if n else 0
    deviating = sum(1 for value in section_max if value > DEVIATION_THRESHOLD)

    def top_features(row: dict[str, float], limit: int = 2) -> list[dict[str, Any]]:
        ranked = sorted(row.items(), key=lambda item: abs(item[1]), reverse=True)
        return [{"feature": name, "z": z} for name, z in ranked[:limit]]

    evidence = [
        {"section": title, "words": sections[i][1].word_count,
         "max_abs_z": section_max[i], "top_features": top_features(per_section_z[i])}
        for i, (title, _) in enumerate(sections)
    ]
    evidence.sort(key=lambda row: row["max_abs_z"], reverse=True)

    worst_title = sections[worst_index][0]
    summary = stats_module.summarize(section_max)

    return [
        finding("drift.max_section_deviation",
                f"Largest per-section style deviation (method={method})",
                section_max[worst_index], "robust z", family=FAMILY, sample_size=n,
                min_sample=MIN_SAMPLE, evidence=evidence[:25],
                distribution={"most_deviating_section": worst_title, "method": method,
                              "top_features": top_features(per_section_z[worst_index])}),
        finding("drift.deviating_section_count",
                f"Sections with a maximum robust z above {DEVIATION_THRESHOLD} (method={method})",
                deviating, "sections", family=FAMILY, sample_size=n, min_sample=MIN_SAMPLE,
                evidence=[row for row in evidence if row["max_abs_z"] > DEVIATION_THRESHOLD][:25]),
        finding("drift.section_deviation",
                f"Distribution of each section's maximum feature deviation (method={method})",
                summary.get("median"), "robust z", family=FAMILY, sample_size=n,
                distribution=summary, min_sample=MIN_SAMPLE),
    ]
