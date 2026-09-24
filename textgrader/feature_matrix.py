"""Build a stable, aligned document-feature matrix from a corpus profile.

Several planned measurements -- multivariate anomaly detection chief among
them -- need to treat each corpus book as one point in a shared numeric space
and this document as one more point in that same space.  A corpus profile
already stores that shape for free: :func:`textgrader.corpus.build_profile`
writes one flat dict per book (``profile["books"][i]``), keyed by metric id,
with every scalar finding any profiled metric produced for that book.  Rows
are guaranteed aligned by construction (``build_profile`` appends one row per
book, in file order, never skipping an index), so ``books[i]`` and any
``feature_profiles[name][i]`` this project builds elsewhere describe the same
text.  This module turns that raw material into something safe to compute a
distance or a covariance over.

Three things a naive "just take every numeric key" approach gets wrong, and
that this module exists to get right:

Feature selection
    Not every numeric key in a book row is a comparable feature.  Some are raw
    counts (``word_count``, and every metric id containing "count", such as
    ``dialogue.identified_speaker_count``) that grow with the length of the
    book rather than describing its style -- exactly the ``unit_sensitive``
    distinction :mod:`textgrader.metrics.common` already draws for
    within-document findings, applied here to the profile's own columns.
    Those are excluded outright.  What is left must also have "adequate
    corpus coverage": a metric that only a few corpus books happened to
    produce a number for (a short book with no dialogue, say) is not a safe
    axis to build a shared distance over, so a column below ``min_coverage``
    is dropped from the schema entirely rather than imputed for every book.

    The default candidate set is deliberately narrow: every "core" prose
    metric (:mod:`textgrader.core_metrics`) that is a rate, a percentage, a
    mean or a coefficient of variation rather than a raw count, plus the
    optional Lexile score.  These are the one feature source guaranteed
    defined the same way in *every* TextGrader corpus profile --
    ``include_core_metrics=True`` is ``build_profile``'s own default, and
    computing them again for one document costs a single
    ``core_metrics.measure`` call, no parse, no model, no re-running whichever
    optional suites happened to be enabled when a particular corpus was
    built.  A caller that wants a richer feature space can pass its own
    ``columns`` explicitly (see :func:`build_schema`); this module does not
    itself go fetch columns from other metric modules, on purpose -- see
    ``textgrader/metrics/anomaly_suite.py``'s module docstring for why that
    would make its own declared cost class dishonest.

Missing values
    A book missing one feature (its own metric could not measure it, or
    disagreed and returned ``None``) must not be silently zero-filled: a
    document that used no relative clauses at all and a document whose
    relative-clause rate was simply never computed are different situations,
    and treating both as literally ``0`` invents a data point.  Every missing
    value is instead imputed at that column's own corpus median, and the fact
    that it was imputed travels with the vector (:attr:`FeatureSchema.missing`
    for corpus rows, the second element of :func:`document_vector`'s return
    for a graded document) so a caller can discount or flag it rather than
    treat an invented number as observed.

Scale
    Every column is centered on its own corpus median and scaled by its own
    robust dispersion (1.4826 * median absolute deviation, the standard
    consistency-corrected estimator, falling back to the population standard
    deviation when the MAD is exactly zero -- a column with fewer than half
    its values distinct -- and to a scale of 1.0 if that is *also* zero, a
    constant column).  A median/MAD estimate is used rather than mean/std
    because the whole point of this module is to feed anomaly detectors,
    which is exactly the setting where a couple of genuinely extreme
    reference books should not be allowed to widen the ruler that judges
    every other book, including themselves.

Nothing here fits a model.  This module answers one question --
"what are the columns, and what is this book/document's standardized position
on them" -- and leaves what to do with that matrix (fit a detector, compute a
centroid, run a regression) to its caller.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .corpus import CORE_METRIC_KEYS

#: Corpus columns that are raw counts, not rates -- see the module docstring's
#: "Feature selection".  Kept as a literal set rather than re-derived from
#: :mod:`textgrader.corpus`'s ``SCALE_DEPENDENT`` so this module has no import
#: dependency on that constant's exact name continuing to exist; the two are
#: intentionally the same three keys.
_RAW_COUNT_KEYS = frozenset({"_words", "_sentences", "_paragraphs"})

#: The core prose metrics that are rates/percentages/means rather than raw
#: counts, plus the optional Lexile score -- see the module docstring.
DEFAULT_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    key for key in CORE_METRIC_KEYS if key not in _RAW_COUNT_KEYS) + ("lexile",)

#: A column needs at least this fraction of corpus books carrying a real
#: (non-imputed) numeric value before it is trusted as a feature at all.
DEFAULT_MIN_COVERAGE = 0.7

#: Median-absolute-deviation-to-standard-deviation consistency constant for
#: a normal distribution: the usual choice so a robust scale and a classical
#: one agree on approximately-normal data.
_MAD_SCALE = 1.4826


def _numeric(value: Any) -> float | None:
    """A finite float, or ``None`` for anything that is not a real number.

    Excludes ``bool`` explicitly: ``isinstance(True, int)`` is true in Python,
    and a stray boolean finding (there are none among the current core/rate
    metrics, but a future one is plausible) must never be treated as 0.0/1.0
    on a numeric axis.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if value == value and value not in (float("inf"), float("-inf")) else None


@dataclass(frozen=True)
class FeatureSchema:
    """The feature space fit from one corpus profile: columns, their
    imputation/scaling parameters, and the corpus's own standardized matrix.
    """

    #: Selected metric ids, in a fixed, deterministic order.
    columns: tuple[str, ...]
    #: Column -> corpus median (used both to impute and to center).
    medians: dict[str, float]
    #: Column -> robust scale (never zero; see the module docstring).
    scales: dict[str, float]
    #: Column -> fraction of corpus books with a real (non-imputed) value.
    coverage: dict[str, float]
    #: Column -> why a candidate column was NOT kept (denylisted, too little
    #: coverage, or dropped by the ``max_features`` cap). Purely diagnostic.
    dropped: dict[str, str]
    #: One id per corpus row, aligned with ``matrix``/``missing``.
    book_ids: tuple[str, ...]
    #: Standardized, imputed corpus rows: ``matrix[i][j]`` is book ``i``'s
    #: z-score on ``columns[j]``.
    matrix: tuple[tuple[float, ...], ...]
    #: ``missing[i][j]`` is True when book ``i`` had no real value for
    #: ``columns[j]`` and its median was substituted.
    missing: tuple[tuple[bool, ...], ...]

    @property
    def n_books(self) -> int:
        return len(self.book_ids)


def select_columns(books: Sequence[Mapping[str, Any]], *, candidates: Sequence[str],
                    deny: Sequence[str] = (), min_coverage: float = DEFAULT_MIN_COVERAGE,
                    max_features: int | None = None) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Which of ``candidates`` are usable columns, plus their coverage.

    A column is dropped (with a human-readable reason) rather than silently
    omitted, because "why isn't metric X in the feature list" is exactly the
    question a reader of this suite's evidence will ask.
    """

    deny_set = {key.lower() for key in deny}
    n = len(books)
    coverage: dict[str, float] = {}
    dropped: dict[str, str] = {}
    kept: list[str] = []
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key.lower() in deny_set or "count" in key.lower() or key in _RAW_COUNT_KEYS:
            dropped[key] = "denylisted as a raw count / unit-sensitive metric"
            continue
        present = sum(1 for book in books if _numeric(book.get(key)) is not None)
        rate = present / n if n else 0.0
        coverage[key] = rate
        if present < 2:
            dropped[key] = "fewer than two corpus books carry a numeric value for this metric"
        elif rate < min_coverage:
            dropped[key] = (f"present in only {rate:.0%} of corpus books "
                            f"(needs at least {min_coverage:.0%})")
        else:
            kept.append(key)
    # Deterministic ordering: highest coverage first, name as the tiebreak, so
    # the same corpus always yields the same schema regardless of dict order.
    kept.sort(key=lambda key: (-coverage[key], key))
    if max_features is not None and len(kept) > max_features:
        for key in kept[max_features:]:
            dropped[key] = (f"corpus schema capped at max_features={max_features}; "
                            f"this column had lower coverage than the ones kept")
        kept = kept[:max_features]
    return kept, coverage, dropped


def build_schema(books: Sequence[Mapping[str, Any]], *, columns: Sequence[str] | None = None,
                 deny: Sequence[str] = (), min_coverage: float = DEFAULT_MIN_COVERAGE,
                 max_features: int | None = None) -> FeatureSchema:
    """Fit a :class:`FeatureSchema` from a corpus profile's ``books`` list.

    ``columns`` defaults to :data:`DEFAULT_FEATURE_COLUMNS`; pass an explicit
    list to consider a different (or narrower) candidate set.  ``books`` is
    used exactly as :mod:`textgrader.corpus` writes it -- a list of flat
    ``{metric_id: value, ...}`` dicts, one per reference text.
    """

    candidates = list(columns) if columns is not None else list(DEFAULT_FEATURE_COLUMNS)
    kept, coverage, dropped = select_columns(
        books, candidates=candidates, deny=deny, min_coverage=min_coverage,
        max_features=max_features)

    medians: dict[str, float] = {}
    scales: dict[str, float] = {}
    for key in kept:
        values = [value for value in (_numeric(book.get(key)) for book in books)
                 if value is not None]
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values) * _MAD_SCALE
        if mad <= 1e-9:
            mad = statistics.pstdev(values) if len(values) > 1 else 0.0
        medians[key] = median
        scales[key] = mad if mad > 1e-9 else 1.0

    book_ids: list[str] = []
    matrix: list[tuple[float, ...]] = []
    missing: list[tuple[bool, ...]] = []
    for index, book in enumerate(books):
        row: list[float] = []
        miss_row: list[bool] = []
        for key in kept:
            value = _numeric(book.get(key))
            is_missing = value is None
            filled = medians[key] if is_missing else value
            row.append((filled - medians[key]) / scales[key])
            miss_row.append(is_missing)
        book_ids.append(str(book.get("source_id") or index))
        matrix.append(tuple(row))
        missing.append(tuple(miss_row))

    return FeatureSchema(columns=tuple(kept), medians=medians, scales=scales,
                         coverage=coverage, dropped=dropped, book_ids=tuple(book_ids),
                         matrix=tuple(matrix), missing=tuple(missing))


def document_vector(schema: FeatureSchema,
                    values: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Standardize one document's own ``{metric_id: value}`` mapping onto
    ``schema``'s columns, imputing any missing feature at the corpus median
    exactly as :func:`build_schema` does for a corpus row, so a document and
    the corpus it is compared against are never scaled two different ways.
    """

    row: list[float] = []
    missing: list[bool] = []
    for key in schema.columns:
        value = _numeric(values.get(key))
        is_missing = value is None
        filled = schema.medians[key] if is_missing else value
        row.append((filled - schema.medians[key]) / schema.scales[key])
        missing.append(is_missing)
    return tuple(row), tuple(missing)


__all__ = ["FeatureSchema", "DEFAULT_FEATURE_COLUMNS", "DEFAULT_MIN_COVERAGE",
           "select_columns", "build_schema", "document_vector"]
