"""Which sentence segmenter actually ran, and how much it would have mattered.

Sentence segmentation used to be decided independently by every metric that
needed it; :class:`~textgrader.document.DocumentAnalysis` now resolves one
segmenter for the whole run (pySBD by default, falling back to spaCy's
sentencizer or the built-in punctuation splitter), so a metric asking "how
many sentences" and one asking "what is sentence 40" always agree.  That
makes the old version of this module, which built its own separate
``pysbd.Segmenter`` and reported its sentence count and mean length,
duplicate a measurement the pipeline already makes for free.

What is still genuinely useful, and not a duplicate of anything else, is
knowing which segmenter a given run actually used (``analysis.segmenter``,
since ``auto`` silently falls back when pySBD is not installed) and how much
that choice matters: the built-in punctuation splitter is a plain heuristic,
and if it agrees with the configured segmenter almost everywhere, a report
can trust sentence-level metrics even in an environment without pySBD.  This
module reports that count difference and shows a handful of the boundaries
where the two disagree.

``style.sentences_pysbd`` and ``style.wps_pysbd`` are kept for corpus-profile
compatibility and still report pySBD's own count and mean sentence length,
independent of which segmenter this particular run ended up using.  When the
run already used pySBD, that is simply ``analysis.sentences`` and costs
nothing extra.  Otherwise a one-off pySBD segmenter is built, but it is run
paragraph by paragraph, the same way :class:`~textgrader.document.Segmenter`
runs it, rather than on the whole document in one call: pySBD's own
algorithm does not scale linearly with input length, and feeding it an entire
400,000-word novel as a single string (as the legacy module did) measured at
well over ten seconds, against well under one paragraph-chunked.
"""

from __future__ import annotations

import difflib
import statistics
from typing import Any, Mapping, Sequence

from .. import text as textlib
from ..document import DocumentAnalysis
from ..optional import require
from .common import FAST, finding, option

FAMILY = "sentence_rhythm"
COST = FAST
REQUIRES: tuple[str, ...] = ("pysbd",)
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

MAX_EXAMPLES = 5
SNIPPET_CHARS = 120


def _pysbd_counts(analysis: DocumentAnalysis, language: str,
                  clean: bool) -> tuple[int | None, float | None, str | None]:
    if analysis.segmenter == "pysbd" and language == analysis.processing.language \
            and clean == analysis.processing.segmenter_clean:
        # This run's own segmentation already is pySBD's; no second pass needed.
        lengths = analysis.sentence_lengths
        return len(lengths), (statistics.fmean(lengths) if lengths else None), None

    pysbd, reason = require("pysbd")
    if pysbd is None:
        return None, None, reason
    try:
        segmenter = pysbd.Segmenter(language=language, clean=clean)
        found: list[str] = []
        for paragraph in analysis.paragraphs:
            found.extend(sentence for sentence in segmenter.segment(paragraph)
                        if textlib.words(sentence))
    except Exception as exc:  # pragma: no cover - pySBD language/runtime guard
        return None, None, f"pysbd failed ({type(exc).__name__}: {exc})"
    if not found:
        return 0, None, None
    lengths = [len(textlib.words(sentence)) for sentence in found]
    return len(found), statistics.fmean(lengths), None


def _diff_examples(configured: Sequence[str], builtin: Sequence[str]) -> list[dict[str, Any]]:
    """A few boundaries where the configured and built-in splits disagree,
    found by diffing the two sentence lists rather than the raw text."""

    if list(configured) == list(builtin):
        return []
    matcher = difflib.SequenceMatcher(a=list(configured), b=list(builtin), autojunk=False)
    examples: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        examples.append({
            "kind": tag,
            "configured": [s[:SNIPPET_CHARS] for s in configured[i1:i2]][:3],
            "builtin": [s[:SNIPPET_CHARS] for s in builtin[j1:j2]][:3],
        })
        if len(examples) >= MAX_EXAMPLES:
            break
    return examples


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    configured = analysis.sentences
    builtin = textlib.sentences(analysis.text)
    count_diff = len(configured) - len(builtin)
    examples = _diff_examples(configured, builtin)

    language = option(config, "language", "en")
    clean = bool(option(config, "clean", False))
    pysbd_count, pysbd_wps, pysbd_reason = _pysbd_counts(analysis, language, clean)

    same_segmenter = analysis.segmenter == "builtin"
    disagreement_warning = ("the configured segmenter is already the built-in splitter; "
                            "there is nothing to disagree with") if same_segmenter else None

    return [
        finding("style.segmenter_used", "Sentence segmenter used", analysis.segmenter, None,
                family=FAMILY, sample_size=len(configured), min_sample=MIN_SAMPLE,
                distribution={"sentence_count": len(configured), "warnings": list(analysis.warnings)}),
        finding("style.sentences_pysbd", "pySBD sentence count", pysbd_count, "sentences",
                family=FAMILY, sample_size=pysbd_count or 0, min_sample=MIN_SAMPLE,
                warning=pysbd_reason),
        finding("style.wps_pysbd", "Words per pySBD sentence", pysbd_wps, "words/sentence",
                family=FAMILY, sample_size=pysbd_count or 0, min_sample=MIN_SAMPLE,
                warning=pysbd_reason),
        finding("style.segmenter_disagreement",
                "Sentence-count difference between the configured and built-in segmenter",
                count_diff, "sentence count difference", family=FAMILY,
                sample_size=len(configured), min_sample=MIN_SAMPLE,
                distribution={"configured_segmenter": analysis.segmenter,
                             "configured_sentence_count": len(configured),
                             "builtin_sentence_count": len(builtin),
                             "differing_blocks": len(examples)},
                evidence=examples, warning=disagreement_warning),
    ]
