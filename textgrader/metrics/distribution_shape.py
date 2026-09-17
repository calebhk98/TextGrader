"""Compare a text's distributions with the corpus's, item for item.

Every other corpus comparison in TextGrader reduces a text to one number and
holds it against thirty other numbers, one per corpus book.  That throws away
almost everything the corpus knows, and it makes the comparison depend on how
the corpus happens to be divided into files: a 3,000-word chapter's average has
a much wider spread than a 110,000-word book's, so a threshold calibrated on
books is wrong for chapters.

This compares populations instead.  The corpus profile stores the quantile
curve of every sentence length, paragraph length, word length, comma count and
spoken-turn length in the whole corpus, pooled across every book.  A document
has its own curve for the same quantities.  Both are distributions of the same
kind of item, so they are directly comparable whatever the documents were:
half a million published sentences against one chapter's four hundred is a
perfectly good comparison, because the unit on both sides is the sentence.

Two readings come out of it:

``distance``
    the first Wasserstein distance between the two curves, in the metric's own
    units.  "This text's sentence lengths would have to move an average of 4.2
    words each to look like the corpus."  A mean cannot say that, and two texts
    with identical means can differ wildly on it.

``bands``
    the share of the text's items falling into each corpus quintile.  A text
    matching the corpus puts 20% in each; a metronomic one puts most of its
    mass in one band.  The bands are cut from the corpus's own quantiles, so
    they mean something without knowing the units.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..corpus import ITEM_SOURCES, _item_values
from ..document import DocumentAnalysis
from ..stats import band_shares, curve_distance, curve_quantile, quantile_curve
from .common import FAST, finding, option, unavailable

FAMILY = "distribution_shape"
COST = FAST
REQUIRES: tuple[str, ...] = ()
#: Items, not documents. Below this the text's own curve is too rough to
#: compare, and the finding says so rather than reporting a noisy distance.
MIN_SAMPLE = 40
UNIT_SENSITIVE = False

#: Quintiles by default: enough resolution to see a concentration, few enough
#: that each band still holds a readable share.
DEFAULT_BANDS = 5

LABELS = {
    "sentence_words": "Sentence length",
    "paragraph_words": "Paragraph length in words",
    "paragraph_sentences": "Paragraph length in sentences",
    "word_characters": "Word length",
    "sentence_commas": "Commas per sentence",
    "turn_words": "Spoken turn length",
}
UNITS = {
    "sentence_words": "words", "paragraph_words": "words",
    "paragraph_sentences": "sentences", "word_characters": "characters",
    "sentence_commas": "commas", "turn_words": "words",
}


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    stored = (profile or {}).get("item_distributions") or {}
    bands = max(2, int(option(config, "bands", DEFAULT_BANDS)))
    if not stored:
        return [unavailable(f"shape.{name}_distance", f"{LABELS[name]} shape",
                            "the corpus profile has no pooled item distributions; rebuild it "
                            "to compare distributions rather than per-book averages",
                            family=FAMILY, unit=UNITS[name])
                for name in ITEM_SOURCES]

    mine = _item_values(analysis)
    out: list[dict[str, Any]] = []
    for name in ITEM_SOURCES:
        entry = stored.get(name) or {}
        corpus_curve = entry.get("quantiles") or []
        values = mine.get(name) or []
        label, unit = LABELS[name], UNITS[name]
        if not corpus_curve:
            out.append(unavailable(f"shape.{name}_distance", f"{label} shape",
                                   f"the corpus profile has no pooled {name} distribution",
                                   family=FAMILY, unit=unit))
            continue
        if len(values) < MIN_SAMPLE:
            out.append(unavailable(
                f"shape.{name}_distance", f"{label} shape",
                f"this text has {len(values)} {name.replace('_', ' ')} item(s), below the "
                f"{MIN_SAMPLE} needed for its own distribution to be worth comparing",
                family=FAMILY, unit=unit))
            continue

        curve = quantile_curve(values)
        distance = curve_distance(curve, corpus_curve)
        edges = [curve_quantile(corpus_curve, (index + 1) / bands)
                 for index in range(bands - 1)]
        shares = band_shares(values, edges)
        expected = 100.0 / bands
        # How lopsided the text is across bands that are equal by construction.
        concentration = max(shares) - expected if shares else None

        quantiles = {f"p{int(q * 100)}": curve_quantile(curve, q)
                     for q in (0.0, .10, .25, .50, .75, .90, 1.0)}
        corpus_quantiles = {f"p{int(q * 100)}": curve_quantile(corpus_curve, q)
                            for q in (0.0, .10, .25, .50, .75, .90, 1.0)}
        out.append(finding(
            f"shape.{name}_distance", f"{label}: distance from the corpus distribution",
            distance, unit, family=FAMILY, sample_size=len(values), min_sample=MIN_SAMPLE,
            distribution={"text": quantiles, "corpus": corpus_quantiles,
                          "corpus_items": entry.get("count"),
                          "corpus_sources": entry.get("sources")},
            evidence=[{"band": index + 1,
                       "upper_bound": (edges[index] if index < len(edges) else
                                       corpus_quantiles["p100"]),
                       "text_share": round(share, 1), "corpus_share": round(expected, 1)}
                      for index, share in enumerate(shares)]))
        out.append(finding(
            f"shape.{name}_concentration", f"{label}: excess share in one corpus band",
            concentration, "percentage points", family=FAMILY, sample_size=len(values),
            min_sample=MIN_SAMPLE,
            warning=f"bands are the corpus's own {bands}-quantiles, so a text matching the "
                    f"corpus scores near 0 and one crowded into a single band approaches "
                    f"{100 - expected:.0f}"))
    return out
