"""Meta-analysis layer: dependence, disagreement and residual features between
TextGrader's OWN metrics.

Every other metric in this codebase measures the text.  This one measures the
*measurements*: it fits, from the reference corpus, how strongly related pairs
and groups of TextGrader's own metric values normally move together, and then
reports where a graded document's own combination diverges from that pattern.
This is the project's philosophy made literal: correlated metrics are kept
because the residual between them -- the part a simple linear relationship
does NOT explain -- can be the most informative single number in the report
(see the module's task doc; words/sentence, words/paragraph and
sentences/paragraph are 90% correlated, and the remaining 10% is precisely
where unusual structure shows up).

Where the fit comes from
    Entirely from ``profile["books"]`` -- the corpus profile's already-computed
    per-book metric values (see :mod:`textgrader.corpus`).  No change to
    ``corpus.py`` is needed or made: every column this module reads was
    already being written by whichever suite computed it, either as a bare
    core-metric key (``"wps"``, ``"wpp"``, ...; see
    ``corpus.CORE_METRIC_KEYS``) or under that suite's own metric id
    (``"semantic.structure_bm25_centroid_relatedness"``, ...).  The fit is
    rebuilt at GRADING time (never at profile-build time: ``measure()`` below
    is a deliberate no-op, see "Two entry points"), cached once per profile per
    process -- see :func:`_fit`.

Leave-one-out
    If the graded document's own source matches one of the profile's books
    (``source_filename``/``source_path``/``source_id`` -- the same three
    fields ``grade.py``'s ``_named_book`` matches on), that book is excluded
    from the fit before anything is computed, exactly the way
    ``semantic_structure_suite``'s topic-model fit excludes the graded book's
    own row (see that module's ``_book_matches_source``).  Without this, a
    document that happens to BE a corpus member would help fit the very
    relationship it is then compared against, understating how unusual its
    own combination looks -- the in-sample optimism leave-one-out validation
    exists to remove.  Tested directly in ``tests/test_metric_relationships.py``.

Two entry points, one isolated post-metric phase
    ``measure(analysis, config, profile)`` -- the ordinary per-suite entry
    point every metric module exposes -- returns ``[]`` unconditionally.  This
    suite's whole point is comparing metrics that only exist once every other
    metric (core AND optional) has already run on THIS document, which is not
    true yet when the ordinary metric loop calls it (``grade.py``'s
    ``_optional_results``): a residual of "words per paragraph given words per
    sentence" needs this document's OWN words-per-sentence value, not just the
    corpus's.  The real computation lives in :func:`relationship_findings`,
    called from ``grade.py``'s own, separately isolated post-metric phase
    (``_relationship_results``), added after every ordinary metric (core and
    optional) has produced its findings -- see that function's docstring for
    exactly where.  This keeps metrics from ever calling one another
    recursively: this module reads other metrics' ALREADY-COMPUTED report
    values, passed in as a plain ``{corpus_column_key: value}`` mapping
    (``document_values``); it never imports or calls another metric module.

Pair/group discovery
    Numeric metric columns present in the (leave-one-out) training books, at
    or above ``min_joint_coverage`` (fraction of books carrying a value for
    both sides of a pair), form the candidate pool -- always including the
    core prose metrics and every metric named in a configured pair or group,
    and otherwise capped at ``max_candidate_metrics`` (alphabetical, so a
    corpus with many optional suites enabled does not silently reorder which
    columns get considered from one run to the next).  DISCOVERY (never
    configured pairs) is further restricted to metric ids THIS document
    itself has a measured value for: the corpus profile carries every suite
    that was ever profiled, including ones off by default on an ordinary
    grading run, and discovering a pair whose predictor or target this run
    never measured would spend one of ``max_pairs``'s slots on a finding that
    can only ever come out "unavailable".  Spearman rank correlation (robust
    to the nonlinear-but-monotonic relationships common among rates) is
    computed for every candidate pair; a pair whose
    ``abs(spearman) >= discover_min_abs_spearman`` (default 0.7) is
    auto-discovered, UNLESS it is at or above ``identity_abs_spearman``
    (default 0.99), in which case it is almost certainly two measurements of
    the same underlying thing (a rate and its complement, the same count
    under two names) rather than a genuine relationship -- its residual would
    be about zero forever, so it is reported separately as a "near identity"
    (see ``style.relationship_pca_diagnostic``'s
    ``distribution["near_identities"]``) rather than spending a discovery
    slot.  Every pair named in ``pairs`` (config) is kept regardless of its
    correlation strength or the identity threshold -- the task spec's own
    words/paragraph-given-words/sentence-and-sentences/paragraph relationship
    is one of the ``DEFAULT_PAIRS``/``DEFAULT_GROUPS`` below, alongside the
    spec's other named cross-family examples (syntax-given-vocabulary,
    readability-given-lexical-rarity, semantic-coherence-given-lexical-
    overlap, sentence-length-variation-given-paragraph-variation).
    Auto-discovered pairs are capped at ``max_pairs`` (deterministic: by
    descending ``abs(spearman)``, ties broken alphabetically) so a corpus
    with hundreds of correlated columns cannot make one document's report
    unboundedly long; configured pairs are never dropped by this cap.  A pair
    below ``min_joint_coverage`` -- configured or discovered -- is reported
    unavailable rather than fit from too little data (this project's rule
    against reporting an unstable number as if it were a real one).

What is reported, per pair (``A`` = the modeled/predicted side, ``B`` = the
predictor; for a configured pair this is the declared ``[A, B]`` order, for a
discovered one it is alphabetical so the same pair always gets the same id):
    ``style.relationship_disagreement_<A>__<B>``
        signed difference between A's and B's own robust corpus z-scores
        (:func:`textgrader.relationships.robust_z`, the same median/MAD
        machinery :mod:`textgrader.stats` uses elsewhere in this codebase) --
        "these two normally-moving-together metrics moved in different
        directions relative to their own marginal distributions".
    ``style.relationship_residual_<A>_given_<B>``
        the interpretable linear baseline: A's value minus what a corpus-fit
        line predicts from B, standardized by that fit's own residual scale.
        Kept even where a nonlinear channel might do better (task spec rule:
        "keep linear residuals as the interpretable baseline").
    ``style.relationship_symmetric_<A>__<B>``
        an orthogonal (total-least-squares) residual: perpendicular distance
        from the point (A, B) to the corpus-fit line that treats neither side
        as the cause, so swapping A and B does not change the finding --
        distinct from the two possible directional residuals.
    ``style.relationship_residual_<target>_given_<predictors...>``
        (configured groups only) a multivariate linear residual, e.g. the
        spec's own words-per-paragraph given words-per-sentence AND
        sentences-per-paragraph.

Document-level aggregates, over every evaluated residual/symmetric/
multivariate finding (never the plain pairwise disagreement, which is not
itself a modeled residual):
    ``style.relationship_residuals_above_p90/p95/p99``
        how many of THIS document's relationship findings exceed that
        relationship's OWN in-sample |standardized residual| reference
        percentile (90th/95th/99th) -- each relationship supplies its own
        threshold, since a syntax/vocabulary residual and a paragraph-rhythm
        residual are not on the same scale before standardization.
    ``style.relationship_max_residual_severity`` / ``_topk_residual_severity``
        the single largest, and the mean of the ``top_k`` largest, absolute
        standardized residuals, with evidence naming which relationship(s).

Corpus-level diagnostics
    Pearson, Spearman, Kendall tau-b, distance correlation, a histogram
    mutual-information estimate, bootstrap and leave-one-out correlation
    stability, and (for a configured group) partial correlations controlling
    for the other predictors, all live in each pair/group finding's
    ``distribution`` -- reported alongside the document-level residual they
    describe, per the task spec's "reported in distribution or as clearly
    corpus-level findings". ``style.relationship_pca_diagnostic`` is a
    separate, corpus-only finding (the same numeric value for every document
    graded against a given profile): the explained-variance share of the
    leading components of the candidate pool, reported strictly as a
    diagnostic -- see the task spec's own words: never used to drop or
    replace an original metric.

Every implementation in this module (correlation, distance correlation,
mutual information, OLS/orthogonal regression, bootstrap/leave-one-out
stability, PCA via power iteration) is pure Python; see
:mod:`textgrader.relationships`'s module docstring for why that was chosen
over ``dcor``/``pingouin``/``hyppo``/``statsmodels``.  This suite therefore has
no optional-package dependency at all (``REQUIRES = ()``): it degrades to
"insufficient data"/"no corpus profile", never to a missing import.

Every finding is ``Polarity.NEUTRAL``; none feed the maturity aggregate.
Off by default; experimental.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .. import relationships as rel
from ..document import DocumentAnalysis
from .common import MODERATE, finding, option

FAMILY = "distribution_shape"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
UNIT_SENSITIVE = False

PREFIX = "style.relationship_"
FEATURE_VERSION = 1

DEFAULT_DISCOVER_MIN_ABS_SPEARMAN = 0.7
#: A discovered pair at or above this is treated as two measurements of the
#: same underlying thing (a rate and its complement, the same count under two
#: names) rather than a genuine relationship -- see ``_discover_pairs``.
DEFAULT_IDENTITY_ABS_SPEARMAN = 0.99
DEFAULT_MAX_PAIRS = 12
DEFAULT_MIN_JOINT_COVERAGE = 0.7
DEFAULT_BOOTSTRAP_SAMPLES = 200
DEFAULT_SEED = 0
DEFAULT_TOP_K = 3
DEFAULT_MAX_CANDIDATE_METRICS = 120
DEFAULT_MIN_CORPUS_DOCUMENTS = 12
DEFAULT_DISTANCE_SAMPLE_CAP = 200
PERCENTILE_THRESHOLDS = (90.0, 95.0, 99.0)

#: Always eligible for discovery/config, regardless of what other suites the
#: corpus profile happened to have enabled -- these are ``corpus.CORE_METRIC_KEYS``
#: minus the raw size counts (excluded below as unit-sensitive).
CORE_CANDIDATES = (
    "fk", "ari", "wps", "slcv", "wpp", "spp", "wlen", "long7", "sttr", "top100",
    "commas", "subord", "relcl", "simple", "u10", "b2035", "shortruns", "front",
    "and2", "andrate", "negative",
)

#: Raw counts/size fields: comparing THEIR relationships would mostly measure
#: how long the document is, not how it is written (see corpus.SCALE_DEPENDENT).
#: Never auto-discovered; still usable if a caller explicitly configures them.
EXCLUDED_FROM_DISCOVERY = {
    "word_count", "sentence_count", "paragraph_count",
    "_words", "_sentences", "_paragraphs",
    "mean_sentence_words", "mean_paragraph_words", "mean_word_characters",
}

#: The task spec's own named cross-family mismatch examples, kept on by
#: default (list of [target, predictor]) so they are always evaluated when
#: the corpus profile has the columns -- see the module docstring.
DEFAULT_PAIRS: list[list[str]] = [
    ["subord", "sttr"],                                          # syntax given vocabulary
    ["fk", "lexical.word_zipf"],                                 # readability given lexical rarity
    ["semantic.structure_bm25_centroid_relatedness", "top100"],  # coherence given lexical overlap
    ["slcv", "rhythm.paragraph_words_cv"],                       # sentence- given paragraph-variation
]

#: The task spec's required multivariate example: words/paragraph given
#: words/sentence AND sentences/paragraph.
DEFAULT_GROUPS: dict[str, list[str]] = {"wpp": ["wps", "spp"]}

DEFAULT_FEATURES: dict[str, bool] = {
    "disagreement": True, "residual": True, "symmetric": True, "multivariate": True,
    "percentile_counts": True, "max_severity": True, "topk_severity": True,
    "pca_diagnostic": True, "bootstrap_stability": True, "loo_stability": True,
    "mutual_information": True, "partial_correlation": True, "distance_correlation": True,
}

DEFAULTS: dict[str, Any] = {
    "features": DEFAULT_FEATURES,
    "discover_min_abs_spearman": DEFAULT_DISCOVER_MIN_ABS_SPEARMAN,
    "identity_abs_spearman": DEFAULT_IDENTITY_ABS_SPEARMAN,
    "pairs": DEFAULT_PAIRS,
    "groups": DEFAULT_GROUPS,
    "max_pairs": DEFAULT_MAX_PAIRS,
    "min_joint_coverage": DEFAULT_MIN_JOINT_COVERAGE,
    "bootstrap_samples": DEFAULT_BOOTSTRAP_SAMPLES,
    "seed": DEFAULT_SEED,
    "top_k": DEFAULT_TOP_K,
    "max_candidate_metrics": DEFAULT_MAX_CANDIDATE_METRICS,
    "min_corpus_documents": DEFAULT_MIN_CORPUS_DOCUMENTS,
    "distance_sample_cap": DEFAULT_DISTANCE_SAMPLE_CAP,
}


def _feature(config: Mapping[str, Any] | None, name: str, default: bool = True) -> bool:
    features = option(config, "features", {})
    if not isinstance(features, Mapping):
        return default
    value = features.get(name, default)
    return default if value is None else bool(value)


def _sanitize(metric_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(metric_id)).strip("_") or "metric"


# --------------------------------------------------------------- book access

def _book_matches_source(book: Mapping[str, Any], source: str | None) -> bool:
    """Whether ``book`` is the same text as the document being graded.

    Matches the same three fields ``grade.py``'s ``_named_book`` and
    ``semantic_structure_suite``'s ``_book_matches_source`` use, so a document
    re-graded from a different relative path but the same corpus source is
    still recognized (``source_id`` survives that; ``source_path`` does not).
    """

    if not source:
        return False
    return (book.get("source_filename") == source or book.get("source_path") == source
           or book.get("source_id") == source)


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        import math
        return float(value) if math.isfinite(float(value)) else None
    return None


def _column(books: Sequence[Mapping[str, Any]], key: str) -> list[float | None]:
    return [_numeric(book.get(key)) for book in books]


def _coverage(books: Sequence[Mapping[str, Any]], a: str, b: str) -> tuple[int, float]:
    ca, cb = _column(books, a), _column(books, b)
    joint = sum(1 for x, y in zip(ca, cb) if x is not None and y is not None)
    return joint, (joint / len(books) if books else 0.0)


def _series(books: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    return [v for v in _column(books, key) if v is not None]


# ------------------------------------------------------------- candidate pool

def _candidate_pool(books: Sequence[Mapping[str, Any]], configured_members: set[str],
                    max_candidates: int) -> list[str]:
    all_numeric: set[str] = set()
    for book in books:
        for key, value in book.items():
            if key in EXCLUDED_FROM_DISCOVERY:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                all_numeric.add(key)
    # A core metric only occupies budget when this corpus actually has it; an
    # absent core column (a profile with core_metrics turned off, or a test
    # fixture that never wrote it) must not crowd out real candidate columns.
    guaranteed = (set(CORE_CANDIDATES) & all_numeric) | configured_members
    extra = sorted(all_numeric - guaranteed)[:max(0, max_candidates - len(guaranteed))]
    return sorted(guaranteed | set(extra))


def _discover_pairs(books: Sequence[Mapping[str, Any]], pool: Sequence[str],
                    min_joint_coverage: float, min_abs_spearman: float,
                    identity_abs_spearman: float,
                    already: set[frozenset]) -> tuple[list[tuple[str, str, float]],
                                                      list[tuple[str, str, float]]]:
    """Every candidate pair meeting the discovery threshold, most-correlated first.

    Returns ``(found, near_identities)``.  A pair at or above
    ``identity_abs_spearman`` (default 0.99) is almost certainly two
    measurements of the same underlying thing (a rate and its complement, a
    count reported two ways) rather than a genuine cross-metric relationship:
    its residual is about zero forever and would otherwise spend one of the
    ``max_pairs`` discovery slots on a finding with nothing to say. Such a
    pair is reported separately, as ``near_identities`` (surfaced in the PCA/
    diagnostic finding's distribution -- see the module docstring), never
    silently dropped.
    """

    found: list[tuple[str, str, float]] = []
    near_identities: list[tuple[str, str, float]] = []
    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            key = frozenset((a, b))
            if key in already:
                continue
            joint, coverage = _coverage(books, a, b)
            if coverage < min_joint_coverage:
                continue
            pairs = rel.paired(_column(books, a), _column(books, b))
            value = rel.spearman(pairs)
            if value is None:
                continue
            if abs(value) >= identity_abs_spearman:
                near_identities.append((a, b, value))
            elif abs(value) >= min_abs_spearman:
                found.append((a, b, value))
    found.sort(key=lambda item: (-abs(item[2]), item[0], item[1]))
    near_identities.sort(key=lambda item: (-abs(item[2]), item[0], item[1]))
    return found, near_identities


# ------------------------------------------------------------------ pair fit

def _bounded(pairs: Sequence[tuple[float, float]], cap: int) -> list[tuple[float, float]]:
    return list(pairs[:cap]) if len(pairs) > cap else list(pairs)


def _fit_pair(books: Sequence[Mapping[str, Any]], target: str, predictor: str,
             config: Mapping[str, Any]) -> dict[str, Any] | None:
    joint, coverage = _coverage(books, target, predictor)
    min_coverage = float(option(config, "min_joint_coverage", DEFAULT_MIN_JOINT_COVERAGE))
    if joint < 6 or coverage < min_coverage:
        return None
    pairs = rel.paired(_column(books, target), _column(books, predictor))
    linear = rel.fit_residual_model(books, target, [predictor])
    orthogonal = rel.fit_orthogonal_model(pairs)
    if linear is None or orthogonal is None:
        return None
    cap = int(option(config, "distance_sample_cap", DEFAULT_DISTANCE_SAMPLE_CAP))
    bounded_pairs = _bounded(pairs, cap)
    seed = int(option(config, "seed", DEFAULT_SEED))
    samples = int(option(config, "bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES))
    stats: dict[str, Any] = {
        "n": len(pairs), "joint_coverage": coverage,
        "pearson": rel.pearson(pairs), "spearman": rel.spearman(pairs),
        "kendall_tau": rel.kendall_tau(bounded_pairs),
    }
    if _feature(config, "distance_correlation", True):
        stats["distance_correlation"] = rel.distance_correlation(bounded_pairs)
    if _feature(config, "mutual_information", True):
        stats["mutual_information_bits"] = rel.mutual_information(pairs)
    if _feature(config, "bootstrap_stability", True):
        stats["bootstrap_spearman_std"] = rel.bootstrap_correlation_stability(pairs, samples, seed)
    if _feature(config, "loo_stability", True):
        stats["loo_spearman_std"] = rel.leave_one_out_stability(pairs)
    return {"target": target, "predictor": predictor, "linear": linear,
           "orthogonal": orthogonal, "series_target": _series(books, target),
           "series_predictor": _series(books, predictor), "stats": stats}


def _fit_group(books: Sequence[Mapping[str, Any]], target: str, predictors: Sequence[str],
              config: Mapping[str, Any]) -> dict[str, Any] | None:
    linear = rel.fit_residual_model(books, target, predictors)
    if linear is None:
        return None
    n = linear["n"]
    partials: dict[str, float | None] = {}
    if _feature(config, "partial_correlation", True) and len(predictors) > 1:
        rows = [row for row in books
               if all(row.get(k) is not None for k in (target, *predictors))]
        target_series = [float(row[target]) for row in rows]
        pred_series = {p: [float(row[p]) for row in rows] for p in predictors}
        for p in predictors:
            others = [pred_series[q] for q in predictors if q != p]
            partials[p] = rel.partial_correlation(target_series, pred_series[p], others)
    return {"target": target, "predictors": list(predictors), "linear": linear,
           "n": n, "partial_correlations": partials}


def _pca_diagnostic(books: Sequence[Mapping[str, Any]], pool: Sequence[str]) -> dict[str, Any] | None:
    full = [key for key in pool
           if all(_numeric(book.get(key)) is not None for book in books)]
    full = full[:40]
    if len(full) < 3:
        return None
    rows = [[_numeric(book[key]) for key in full] for book in books]
    result = rel.principal_components(rows, max_components=3)
    if result is None:
        return None
    result["columns"] = full
    return result


# --------------------------------------------------------------------- cache

_FIT_CACHE: dict[Any, Any] = {}


def _reset_fit_cache() -> None:
    _FIT_CACHE.clear()


try:  # pragma: no cover - trivial wiring, exercised via tests calling reset
    from ..optional import on_reset
    on_reset(_reset_fit_cache)
except Exception:  # pragma: no cover - optional module should always import
    pass


def _profile_fingerprint(profile: Mapping[str, Any] | None) -> tuple[Any, ...]:
    """See ``semantic_structure_suite._profile_fingerprint`` for why a plain
    ``dict``'s identity needs this much (a short-lived test profile can be
    garbage-collected and a later, unrelated one allocated at the same id)."""

    if profile is None:
        return (None,)
    books = profile.get("books")
    return (id(profile), len(books) if books is not None else -1,
           id(books) if books is not None else None)


def _fit(profile: Mapping[str, Any] | None, source: str | None,
         config: Mapping[str, Any],
         document_keys: frozenset[str] | None = None) -> dict[str, Any]:
    """Fit (or fetch the cached fit for) this profile, this excluded source,
    these tunables, and this document's OWN set of measured metric keys (which
    bounds pair DISCOVERY -- see ``_fit_uncached``).  Cached once per profile
    per process -- see the module docstring's "Where the fit comes from"."""

    pairs_config = tuple(tuple(item) for item in (option(config, "pairs", DEFAULT_PAIRS) or []))
    groups_config = tuple(sorted((k, tuple(v)) for k, v in
                                 (option(config, "groups", DEFAULT_GROUPS) or {}).items()))
    key = (source, pairs_config, groups_config,
          round(float(option(config, "discover_min_abs_spearman", DEFAULT_DISCOVER_MIN_ABS_SPEARMAN)), 6),
          round(float(option(config, "identity_abs_spearman", DEFAULT_IDENTITY_ABS_SPEARMAN)), 6),
          int(option(config, "max_pairs", DEFAULT_MAX_PAIRS)),
          round(float(option(config, "min_joint_coverage", DEFAULT_MIN_JOINT_COVERAGE)), 6),
          int(option(config, "bootstrap_samples", DEFAULT_BOOTSTRAP_SAMPLES)),
          int(option(config, "seed", DEFAULT_SEED)),
          int(option(config, "max_candidate_metrics", DEFAULT_MAX_CANDIDATE_METRICS)),
          int(option(config, "min_corpus_documents", DEFAULT_MIN_CORPUS_DOCUMENTS)),
          int(option(config, "distance_sample_cap", DEFAULT_DISTANCE_SAMPLE_CAP)),
          tuple(sorted((k, bool(v)) for k, v in
                       (option(config, "features", DEFAULT_FEATURES) or {}).items())),
          tuple(sorted(document_keys)) if document_keys is not None else None)
    fingerprint = _profile_fingerprint(profile)
    cached = _FIT_CACHE.get(key)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    result = _fit_uncached(profile, source, config, document_keys)
    _FIT_CACHE[key] = (fingerprint, result)
    return result


def _fit_uncached(profile: Mapping[str, Any] | None, source: str | None,
                  config: Mapping[str, Any],
                  document_keys: frozenset[str] | None) -> dict[str, Any]:
    min_corpus = int(option(config, "min_corpus_documents", DEFAULT_MIN_CORPUS_DOCUMENTS))
    all_books = list((profile or {}).get("books") or [])
    excluded_indices = {i for i, book in enumerate(all_books) if _book_matches_source(book, source)}
    books = [book for i, book in enumerate(all_books) if i not in excluded_indices]
    meta = {
        "feature_version": FEATURE_VERSION, "corpus_books_total": len(all_books),
        "corpus_books_excluded_leave_one_out": len(excluded_indices), "corpus_books_used": len(books),
    }
    if len(books) < min_corpus:
        meta["reason"] = (f"only {len(books)} usable reference book(s) after leave-one-out "
                          f"exclusion (needs at least {min_corpus}); set "
                          f"metric_relationships.min_corpus_documents to lower the floor")
        return {"books": [], "pairs": {}, "groups": {}, "pca": None, "near_identities": [],
               "meta": meta}

    configured_pairs_raw = [tuple(item) for item in (option(config, "pairs", DEFAULT_PAIRS) or [])
                            if isinstance(item, (list, tuple)) and len(item) == 2]
    groups = dict(option(config, "groups", DEFAULT_GROUPS) or {})
    configured_members = {name for pair in configured_pairs_raw for name in pair}
    for target, predictors in groups.items():
        configured_members.add(target)
        configured_members.update(predictors)

    max_candidates = int(option(config, "max_candidate_metrics", DEFAULT_MAX_CANDIDATE_METRICS))
    pool = _candidate_pool(books, configured_members, max_candidates)
    # Pair DISCOVERY (never configured pairs, which are always attempted
    # regardless -- see the module docstring) is restricted to metric ids this
    # document itself has a value for. The corpus profile carries every
    # profiled suite, including ones off by default on an ordinary grading
    # run (e.g. a volatility channel from an off-by-default affect suite);
    # discovering a pair whose predictor or target this run never measured
    # would only ever emit an "unavailable" finding and spend one of
    # ``max_pairs``'s slots on it.
    discovery_pool = ([key for key in pool if key in document_keys]
                      if document_keys is not None else pool)
    min_joint_coverage = float(option(config, "min_joint_coverage", DEFAULT_MIN_JOINT_COVERAGE))
    min_abs_spearman = float(option(config, "discover_min_abs_spearman",
                                    DEFAULT_DISCOVER_MIN_ABS_SPEARMAN))
    identity_abs_spearman = float(option(config, "identity_abs_spearman",
                                         DEFAULT_IDENTITY_ABS_SPEARMAN))
    max_pairs = int(option(config, "max_pairs", DEFAULT_MAX_PAIRS))

    already = {frozenset(pair) for pair in configured_pairs_raw}
    discovered, near_identities = _discover_pairs(books, discovery_pool, min_joint_coverage,
                                                  min_abs_spearman, identity_abs_spearman, already)
    discovered = discovered[:max_pairs]

    selected: list[tuple[str, str, str]] = []  # (target, predictor, origin)
    seen: set[frozenset] = set()
    for target, predictor in configured_pairs_raw:
        pair_key = frozenset((target, predictor))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        selected.append((target, predictor, "configured"))
    for a, b, _score in discovered:
        pair_key = frozenset((a, b))
        if pair_key in seen:
            continue
        seen.add(pair_key)
        # Deterministic direction for a pair with no declared meaning: the
        # alphabetically later id is treated as the modeled/predicted side.
        selected.append((b, a, "discovered"))

    pair_fits: dict[tuple[str, str], dict[str, Any]] = {}
    for target, predictor, origin in selected:
        fit = _fit_pair(books, target, predictor, config)
        if fit is not None:
            fit["origin"] = origin
            fit["coverage"] = _coverage(books, target, predictor)[1]
            pair_fits[(target, predictor)] = fit

    group_fits: dict[str, dict[str, Any]] = {}
    for target, predictors in groups.items():
        joint = sum(1 for book in books
                   if all(_numeric(book.get(k)) is not None for k in (target, *predictors)))
        if joint / len(books) < min_joint_coverage:
            continue
        fit = _fit_group(books, target, predictors, config)
        if fit is not None:
            group_fits[target] = fit

    pca = None
    if _feature(config, "pca_diagnostic", True):
        pca = _pca_diagnostic(books, pool)

    meta.update({
        "candidate_pool_size": len(pool), "candidate_pool": pool,
        "discovery_pool_size": len(discovery_pool),
        "configured_pairs": len(configured_pairs_raw), "discovered_pairs": len(discovered),
        "near_identities_found": len(near_identities),
        "pairs_fit": len(pair_fits), "groups_fit": len(group_fits),
        "discover_min_abs_spearman": min_abs_spearman, "identity_abs_spearman": identity_abs_spearman,
        "min_joint_coverage": min_joint_coverage, "max_pairs": max_pairs,
    })
    return {"books": books, "pairs": pair_fits, "groups": group_fits, "pca": pca,
           "near_identities": near_identities, "meta": meta}


# ------------------------------------------------------------------ findings

def _unavailable(reason: str, extra_ids: Sequence[str] = ()) -> list[dict[str, Any]]:
    ids = [f"{PREFIX}unavailable"] + list(extra_ids)
    return [finding(metric_id, "Metric-relationship layer", None, None, family=FAMILY,
                    warning=reason)
           for metric_id in ids]


def _pair_findings(key: tuple[str, str], fit: Mapping[str, Any],
                   document_values: Mapping[str, Any],
                   features: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (findings, severity_entries).  ``severity_entries`` feed the
    document-level aggregate (max/top-k/percentile-count findings)."""

    target, predictor = key
    doc_target = document_values.get(target)
    doc_predictor = document_values.get(predictor)
    a, b = sorted((target, predictor))
    stats = dict(fit["stats"])
    stats["target"] = target
    stats["predictor"] = predictor
    stats["origin"] = fit["origin"]
    out: list[dict[str, Any]] = []
    severities: list[dict[str, Any]] = []
    missing = []
    if doc_target is None:
        missing.append(target)
    if doc_predictor is None:
        missing.append(predictor)
    missing_reason = (f"this document has no measured value for {', '.join(missing)}; the "
                      f"corpus fit is available but nothing to compare it against" if missing
                      else None)

    if features.get("disagreement", True):
        metric_id = f"{PREFIX}disagreement_{_sanitize(a)}__{_sanitize(b)}"
        value = None
        if not missing:
            z_target = rel.robust_z(doc_target, fit["series_target"])
            z_predictor = rel.robust_z(doc_predictor, fit["series_predictor"])
            if z_target is not None and z_predictor is not None:
                # Signed so the id's alphabetical order always determines the
                # sign, regardless of which side is the modeled "target".
                value = (z_target - z_predictor) if target == a else (z_predictor - z_target)
        out.append(finding(metric_id,
                           f"Standardized disagreement between {a} and {b}",
                           value, "robust sigma", family=FAMILY, distribution=dict(stats),
                           warning=missing_reason))

    if features.get("residual", True):
        metric_id = f"{PREFIX}residual_{_sanitize(target)}_given_{_sanitize(predictor)}"
        value = None
        if not missing:
            value = rel.standardized_residual(fit["linear"], doc_target, [doc_predictor])
        dist = dict(stats)
        dist["coefficients"] = fit["linear"]["coefficients"]
        dist["residual_reference_p90"] = rel.residual_reference_percentile(fit["linear"], 90.0)
        dist["residual_reference_p95"] = rel.residual_reference_percentile(fit["linear"], 95.0)
        dist["residual_reference_p99"] = rel.residual_reference_percentile(fit["linear"], 99.0)
        out.append(finding(metric_id,
                           f"Residual of {target} predicted from {predictor} (linear baseline)",
                           value, "robust sigma", family=FAMILY, distribution=dist,
                           warning=missing_reason))
        if value is not None:
            severities.append({"metric_id": metric_id, "kind": "residual",
                               "abs_value": abs(value),
                               "reference_p90": dist["residual_reference_p90"],
                               "reference_p95": dist["residual_reference_p95"],
                               "reference_p99": dist["residual_reference_p99"]})

    if features.get("symmetric", True):
        metric_id = f"{PREFIX}symmetric_{_sanitize(a)}__{_sanitize(b)}"
        value = None
        if not missing:
            point = (document_values.get(target), document_values.get(predictor))
            value = rel.orthogonal_residual(fit["orthogonal"], point)
        dist = dict(stats)
        offsets = fit["orthogonal"]["offsets"]
        median = fit["orthogonal"]["offset_median"]
        scale = fit["orthogonal"]["offset_scale"]
        magnitudes = sorted(abs((o - median) / scale) for o in offsets)
        dist["reference_p90"] = rel.quantile(magnitudes, 0.90) if magnitudes else None
        dist["reference_p95"] = rel.quantile(magnitudes, 0.95) if magnitudes else None
        dist["reference_p99"] = rel.quantile(magnitudes, 0.99) if magnitudes else None
        out.append(finding(metric_id,
                           f"Order-independent (orthogonal) residual between {a} and {b}",
                           value, "robust sigma", family=FAMILY, distribution=dist,
                           warning=missing_reason))
        if value is not None:
            severities.append({"metric_id": metric_id, "kind": "symmetric",
                               "abs_value": abs(value),
                               "reference_p90": dist["reference_p90"],
                               "reference_p95": dist["reference_p95"],
                               "reference_p99": dist["reference_p99"]})
    return out, severities


def _group_findings(target: str, fit: Mapping[str, Any],
                    document_values: Mapping[str, Any],
                    features: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not features.get("multivariate", True):
        return [], []
    predictors = fit["predictors"]
    doc_target = document_values.get(target)
    doc_predictors = [document_values.get(p) for p in predictors]
    missing = [name for name, value in zip([target, *predictors], [doc_target, *doc_predictors])
              if value is None]
    metric_id = f"{PREFIX}residual_{_sanitize(target)}_given_{'_'.join(sorted(_sanitize(p) for p in predictors))}"
    value = None
    if not missing:
        value = rel.standardized_residual(fit["linear"], doc_target, doc_predictors)
    dist = {
        "target": target, "predictors": predictors, "n": fit["n"],
        "coefficients": fit["linear"]["coefficients"],
        "partial_correlations": fit.get("partial_correlations", {}),
        "residual_reference_p90": rel.residual_reference_percentile(fit["linear"], 90.0),
        "residual_reference_p95": rel.residual_reference_percentile(fit["linear"], 95.0),
        "residual_reference_p99": rel.residual_reference_percentile(fit["linear"], 99.0),
    }
    warning = (f"this document has no measured value for {', '.join(missing)}; the corpus fit "
              f"is available but nothing to compare it against" if missing else None)
    result = finding(metric_id,
                     f"Multivariate residual of {target} predicted from "
                     f"{', '.join(predictors)}",
                     value, "robust sigma", family=FAMILY, distribution=dist, warning=warning)
    severities = []
    if value is not None:
        severities.append({"metric_id": metric_id, "kind": "multivariate", "abs_value": abs(value),
                           "reference_p90": dist["residual_reference_p90"],
                           "reference_p95": dist["residual_reference_p95"],
                           "reference_p99": dist["residual_reference_p99"]})
    return [result], severities


def _aggregate_findings(severities: Sequence[Mapping[str, Any]],
                        config: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    reason = "no relationship residual could be computed for this document" if not severities else None
    if _feature(config, "percentile_counts", True) and not severities:
        for threshold in PERCENTILE_THRESHOLDS:
            out.append(finding(f"{PREFIX}residuals_above_p{int(threshold)}",
                               f"Relationship residuals above their own reference "
                               f"{int(threshold)}th percentile",
                               None, "residuals", family=FAMILY, warning=reason))
    if _feature(config, "max_severity", True) and not severities:
        out.append(finding(f"{PREFIX}max_residual_severity", "Largest relationship-residual severity",
                           None, "robust sigma", family=FAMILY, warning=reason))
    if _feature(config, "topk_severity", True) and not severities:
        out.append(finding(f"{PREFIX}topk_residual_severity",
                           "Mean severity of the top-k relationship residuals",
                           None, "robust sigma", family=FAMILY, warning=reason))
    if not severities:
        return out
    if _feature(config, "percentile_counts", True):
        for threshold in PERCENTILE_THRESHOLDS:
            key = f"reference_p{int(threshold)}"
            flagged = [item for item in severities
                      if item.get(key) is not None and item["abs_value"] > item[key]]
            out.append(finding(
                f"{PREFIX}residuals_above_p{int(threshold)}",
                f"Relationship residuals above their own reference {int(threshold)}th percentile",
                len(flagged), "residuals", family=FAMILY, sample_size=len(severities),
                distribution={"flagged": [item["metric_id"] for item in flagged],
                             "evaluated": len(severities)}))
    if _feature(config, "max_severity", True):
        worst = max(severities, key=lambda item: item["abs_value"])
        out.append(finding(f"{PREFIX}max_residual_severity", "Largest relationship-residual severity",
                           worst["abs_value"], "robust sigma", family=FAMILY,
                           sample_size=len(severities),
                           evidence=[{"metric_id": worst["metric_id"], "kind": worst["kind"]}]))
    if _feature(config, "topk_severity", True):
        top_k = int(option(config, "top_k", DEFAULT_TOP_K))
        ordered = sorted(severities, key=lambda item: -item["abs_value"])[:top_k]
        mean_top = sum(item["abs_value"] for item in ordered) / len(ordered)
        out.append(finding(f"{PREFIX}topk_residual_severity",
                           f"Mean severity of the top {len(ordered)} relationship residuals",
                           mean_top, "robust sigma", family=FAMILY, sample_size=len(severities),
                           evidence=[{"metric_id": item["metric_id"], "kind": item["kind"],
                                     "abs_value": item["abs_value"]} for item in ordered]))
    return out


def _pca_finding(pca: Mapping[str, Any] | None,
                 near_identities: Sequence[tuple[str, str, float]]) -> dict[str, Any]:
    """The corpus-only PCA diagnostic, plus the near-identity pairs discovery
    excluded from becoming their own findings (see ``_discover_pairs``) --
    surfaced here, never silently dropped."""

    metric_id = f"{PREFIX}pca_diagnostic"
    name = "Explained-variance share of the leading components of TextGrader's own metrics"
    identities_payload = [{"a": a, "b": b, "spearman": value} for a, b, value in near_identities]
    if not pca:
        return finding(metric_id, name, None, "share", family=FAMILY,
                       distribution={"near_identities": identities_payload} if identities_payload
                       else None,
                       warning="not enough fully-covered numeric columns in the candidate pool "
                               "to compute a diagnostic PCA")
    dist = dict(pca)
    dist["near_identities"] = identities_payload
    return finding(metric_id, name, pca["explained_variance_share"][0]
                   if pca["explained_variance_share"] else None, "share", family=FAMILY,
                   sample_size=pca["n_rows"], distribution=dist,
                   warning="diagnostic only; never used to drop or replace an original metric")


# --------------------------------------------------------------------- entry

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The ordinary per-suite hook.  Deliberately a no-op -- see the module
    docstring's "Two entry points": every real finding needs this document's
    OWN already-measured metric values, which do not exist yet at the point
    ``grade.py``'s ordinary metric loop calls this. Real output comes from
    :func:`relationship_findings`, called from ``grade.py``'s post-metric
    phase."""

    return []


def relationship_findings(analysis: DocumentAnalysis, document_values: Mapping[str, Any],
                          config: Mapping[str, Any] | None = None,
                          profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """The real computation, called once per graded document from ``grade.py``'s
    isolated post-metric phase (``_relationship_results``) with every other
    metric's already-computed value in ``document_values``."""

    config = config or {}
    books = (profile or {}).get("books") if profile else None
    if not profile or not books:
        return _unavailable("no corpus profile configured; the relationship layer needs a "
                            "reference corpus of other books' metric values to fit against")

    corpus_unit = profile.get("comparison_unit")
    doc_unit = analysis.comparison_unit
    if corpus_unit and doc_unit and doc_unit != "unknown" and doc_unit != corpus_unit:
        return _unavailable(
            f"this corpus profile's comparison unit ('{corpus_unit}') does not match this "
            f"document's ('{doc_unit}'); comparing metric relationships across units would "
            f"measure how the text was divided, not how its metrics relate")

    document_keys = frozenset(document_values.keys())
    fit = _fit(profile, analysis.source, config, document_keys)
    if not fit["pairs"] and not fit["groups"] and fit["meta"].get("reason"):
        return _unavailable(fit["meta"]["reason"])

    features = option(config, "features", DEFAULT_FEATURES) or {}
    findings: list[dict[str, Any]] = []
    severities: list[dict[str, Any]] = []
    for key, pair_fit in sorted(fit["pairs"].items()):
        pair_findings, pair_severities = _pair_findings(key, pair_fit, document_values, features)
        findings.extend(pair_findings)
        severities.extend(pair_severities)
    for target, group_fit in sorted(fit["groups"].items()):
        group_findings, group_severities = _group_findings(target, group_fit, document_values,
                                                            features)
        findings.extend(group_findings)
        severities.extend(group_severities)
    findings.extend(_aggregate_findings(severities, config))
    if _feature(config, "pca_diagnostic", True):
        findings.append(_pca_finding(fit["pca"], fit.get("near_identities", [])))
    return findings
