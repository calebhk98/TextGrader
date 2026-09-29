"""One number for "does this read like the corpus", calibrated on the corpus itself.

Each compared measurement already has a percentile against the corpus.  Its
*atypicality* is how far that percentile sits from the middle, ``|p - 50| / 50``:
0 at the corpus median, 1 at or past the corpus's lowest or highest value.
The text's atypicality is the mean of those, taken first within each family
and then across families, so a family is one vote however many measurements
it has.  Without that, 300 word rates or 60 rhythm variants would outvote a
family of three, and switching a large family on or off would move the score
by its size rather than by what it found.

A mean atypicality of 0.4 means nothing on its own, because every real
chapter is atypical somewhere.  So the corpus's own observations are scored
the same way, each one against the others (leave-one-out, so a chapter is
never compared with itself), over exactly the measurements this text was
compared on.  The score is the share of the corpus's own observations that
are at least as atypical as this text:

* around 50: as typical as a middling chapter of the corpus;
* near 100: more typical than almost every chapter the corpus has;
* near 0: more atypical than every chapter the corpus has.

A genuine held-out chapter by the corpus author lands anywhere from 0 to 100
with equal chance; that is what calibrated means.  Because the calibration
reuses whatever set was compared, turning measurements on or off changes
which questions are asked, never the scale of the answer.

Only measurements whose per-observation values the profile keeps in order
can be calibrated: the book-row columns and the word-rate table.  A compared
measurement with only a pooled distribution (the item-level shapes) is
counted as ``uncalibrated`` and left out, rather than scored on a scale the
corpus cannot be scored on.
"""

from __future__ import annotations

import bisect
import statistics
from typing import Any, Iterable, Mapping

#: Fewer corpus observations than this and the leave-one-out percentiles are
#: too coarse (steps of 10 points or more) to rank anything.
MIN_OBSERVATIONS = 10


def atypicality(percentile: float) -> float:
    return min(1.0, abs(float(percentile) - 50.0) / 50.0)


def _leave_one_out(values: list[float | None]) -> list[float | None]:
    """Each observation's percentile among the others, same midrank rule as ``stats.compare``."""
    present = sorted(value for value in values if value is not None)
    others = len(present) - 1
    out: list[float | None] = []
    for value in values:
        if value is None or others < 1:
            out.append(None)
            continue
        less = bisect.bisect_left(present, value)
        equal = bisect.bisect_right(present, value) - less - 1
        out.append(100.0 * (less + 0.5 * equal) / others)
    return out


def _column(profile: Mapping[str, Any], key: str, prefix: str) -> list[float | None] | None:
    """The per-observation values of ``key``, in book order, or ``None`` if not kept in order."""
    books = profile.get("books")
    if not isinstance(books, list) or not books:
        return None
    rates = profile.get("word_rates") or {}
    if key.startswith(prefix) and rates.get("per_book"):
        vocabulary = rates.get("vocabulary") or []
        word = key[len(prefix):]
        if word in vocabulary and len(rates["per_book"]) == len(books):
            index = vocabulary.index(word)
            return [row[index] for row in rates["per_book"]]
        return None
    column = []
    for book in books:
        value = book.get(key) if isinstance(book, dict) else None
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        column.append(float(value) if ok else None)
    if all(value is None for value in column):
        return None
    reference = (profile.get("distributions") or {}).get(key)
    if isinstance(reference, dict) and reference.get("values") is not None:
        # The column must be the distribution the comparison used; an alias
        # distribution stored under another name is not calibratable here.
        if sorted(value for value in column if value is not None) != sorted(reference["values"]):
            return None
    return column


def _family_means(items: Iterable[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for family, value in items:
        grouped.setdefault(family, []).append(value)
    return {family: statistics.fmean(values) for family, values in grouped.items()}


def _share_at_least(corpus: list[float], value: float) -> float:
    """Percent of ``corpus`` at least ``value``, ties counted half, like the percentiles."""
    above = sum(1 for item in corpus if item > value + 1e-12)
    ties = sum(1 for item in corpus if abs(item - value) <= 1e-12)
    return 100.0 * (above + 0.5 * ties) / len(corpus)


def score(compared: Iterable[tuple[str, str, str, float]], profile: Mapping[str, Any] | None,
          *, word_rate_prefix: str = "lexical.word_rate.") -> dict[str, Any] | None:
    """The authorship score over ``compared`` (metric_id, profile key, family, percentile) rows."""
    if not profile:
        return None
    books = profile.get("books")
    if not isinstance(books, list) or len(books) < MIN_OBSERVATIONS:
        return {"score": None, "note": f"the corpus has fewer than {MIN_OBSERVATIONS} "
                                       f"observations, too few to calibrate against"}
    text_items: list[tuple[str, float]] = []
    corpus_items: list[list[tuple[str, float]]] = [[] for _ in books]
    uncalibrated = []
    seen = set()
    for metric_id, key, family, percentile in compared:
        if metric_id in seen:
            continue
        seen.add(metric_id)
        column = _column(profile, key, word_rate_prefix)
        if column is None:
            uncalibrated.append(metric_id)
            continue
        text_items.append((family, atypicality(percentile)))
        for index, value in enumerate(_leave_one_out(column)):
            if value is not None:
                corpus_items[index].append((family, atypicality(value)))
    if not text_items:
        return {"score": None, "uncalibrated": len(uncalibrated),
                "note": "no compared measurement has per-observation corpus values"}
    text_families = _family_means(text_items)
    corpus_families = [_family_means(items) for items in corpus_items]
    text_overall = statistics.fmean(text_families.values())
    corpus_overall = [statistics.fmean(fams[f] for f in text_families if f in fams)
                      for fams in corpus_families if any(f in fams for f in text_families)]
    families = {}
    for family, value in sorted(text_families.items()):
        reference = [fams[family] for fams in corpus_families if family in fams]
        families[family] = {
            "measurements": sum(1 for fam, _ in text_items if fam == family),
            "atypicality": round(value, 4),
            "corpus_median_atypicality": round(statistics.median(reference), 4),
            "score": round(_share_at_least(reference, value), 1),
        }
    ordered = sorted(corpus_overall)
    return {
        "score": round(_share_at_least(corpus_overall, text_overall), 1),
        "atypicality": round(text_overall, 4),
        "corpus_atypicality": {"min": round(ordered[0], 4),
                               "median": round(statistics.median(ordered), 4),
                               "max": round(ordered[-1], 4)},
        "observations": len(corpus_overall),
        "measurements": len(text_items),
        "families": families,
        "uncalibrated": len(uncalibrated),
        "method": "mean |percentile - 50| / 50 per family, families weighted equally; score is "
                  "the percent of corpus observations (each scored leave-one-out over the same "
                  "measurements) at least as atypical as this text",
    }
