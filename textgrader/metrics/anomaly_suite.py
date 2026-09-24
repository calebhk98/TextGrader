"""Multivariate anomaly/outlier detection over TextGrader's own feature space.

Every other corpus-comparison metric in this project asks one axis at a time:
is this sentence-length variance unusual, is this comma rate unusual, is this
subordinate-clause share unusual. A text can sit squarely inside every one of
those marginal ranges and still be a combination no reference book resembles
-- simple syntax paired with an unusually narrow vocabulary, say, or dense
subordination paired with almost no punctuation variety. Answering that needs
a *multivariate* question: is this document's whole feature vector far from
the cloud of reference books, in the space all those features share at once.
No single geometry answers that well on every corpus shape, so this module
runs many independent anomaly detectors over the same feature vectors and
reports every one of them separately -- never a single blended "anomaly
score" -- because which detectors agree and which do not is itself the
useful signal (see the project philosophy: a metric is a sensor, not an
opinion, and disagreement between independent measurements is data).

Where the feature vectors come from
    :mod:`textgrader.feature_matrix` builds one row per reference book from
    the corpus profile's already-computed ``books`` list, using (by default)
    every "core" prose metric that is a rate/percentage/mean rather than a
    raw count -- see that module's docstring for exactly why. This suite
    deliberately does NOT reach into other optional suites' finding values to
    build a richer feature space: doing so would mean re-running whichever
    modules happen to own those metric ids, at grading time, for every
    corpus-derived column -- a hidden cost that would depend on which other
    suites a given corpus profile happened to have enabled, and would make
    this suite's own declared ``COST = MODERATE`` untrue. The core-metric
    feature set is small (about twenty columns), always present in a profile
    built with ``include_core_metrics=True`` (``build_profile``'s own
    default), and costs exactly one extra ``core_metrics.measure`` call on
    this document -- no parse, no model, no re-running a sibling suite.  A
    caller who wants a different candidate set can pass ``columns`` in the
    config; this suite does not attempt to discover richer columns itself.

When detectors are fit
    "Fit at profile-build time" would need a place to store a serialized
    model and a way to detect when the corpus or metric schema changed
    underneath it -- a real model-registry feature this task's allowed files
    (no changes to ``textgrader/corpus.py``) do not include room for. Instead,
    every detector here is refit at GRADING time, directly from
    ``profile["books"]`` (about fifty rows, twenty columns). Measured end to
    end against the shipped reference corpus (via ``grade.py``, one real held-
    out novel, all fifteen detectors on): about 14 seconds. Most of that is
    NOT per-book work -- it is the one-time cost, paid once per Python
    process, of PyOD's first call into numba (its histogram/ABOD code paths
    are JIT-compiled on first use, independent of corpus size: about 4-5
    seconds on its own, confirmed by timing a second call immediately after
    the first) plus scikit-learn/PyOD's own first import. What scales with the
    corpus is small by comparison: fitting a detector on ~49 rows and scoring
    one query point is single-digit milliseconds once its backing library is
    warm, and this suite bounds the leave-one-out reference pass
    (``loo_reference_size``, default 15) specifically so that per-book cost
    never grows past a controlled ceiling regardless of how large the corpus
    profile is (see :func:`_loo_reference_scores`). This is what earns the
    suite ``COST = MODERATE`` rather than ``COST = FAST`` -- a real, measured
    number, not "fast because the data is small" -- and it also means every
    detector automatically incorporates the current corpus with no separate
    "rebuild the model" step: whatever ``build_profile`` last wrote to
    ``profile["books"]`` is what gets fit, every time.

    During corpus PROFILING itself (``build_profile`` calling
    ``measure(analysis, config=..., profile=None)`` once per reference book,
    to decide what to store in each book's own row) this module is a no-op:
    every finding here needs a profile of *other* books to compare against,
    which does not exist yet while a corpus is being built one book at a
    time, and fitting an anomaly model against a corpus that is still being
    assembled would be measuring a moving target. ``measure`` recognizes
    ``profile is None`` immediately and returns ``unavailable()`` for every
    finding without importing scikit-learn/PyOD/hdbscan at all, so enabling
    this suite never adds cost to ``build_corpus.py`` -- consistent with this
    project's gating rule that nothing expensive may run during corpus
    profiling unless a metric explicitly asks for it (this one never does).

Leave-one-out corpus validation
    Never fits a detector on the reference set the fitted document is
    compared against without holding out that exact reference book: this
    document's own score comes from a detector fit on ALL reference books
    (fine -- the document is not one of them), but the "is this score
    unusual FOR A REFERENCE BOOK" yardstick each detector needs (the
    ``corpus_percentile``/``threshold`` reported alongside the raw score)
    is built by refitting the SAME detector on every-book-but-one and scoring
    the held-out book, for a bounded number of reference books
    (``loo_reference_size``, evenly sampled through the corpus so the
    resulting reference distribution is deterministic and does not depend on
    corpus file order). Scoring a reference book with a model that was
    trained on that same book folded in would systematically understate how
    unusual real reference books look to that detector -- the in-sample
    optimism every leave-one-out validation exists to remove.

Missing document features
    A graded document missing one of the schema's feature columns (a metric
    that returned no value for this particular text) is imputed at that
    column's corpus median and flagged, never zero-filled -- see
    :func:`textgrader.feature_matrix.document_vector`. Every finding's
    ``distribution.missing_document_features`` names exactly which columns,
    if any, were estimated rather than measured for this document.

Detectors
    Every finding is its own detector, independently switchable under
    ``features`` and independently degrading when its package is missing --
    see ``config.json``'s ``_features_requires`` note. ``scikit-learn`` backs
    Isolation Forest, Local Outlier Factor (novelty mode), One-Class SVM,
    Elliptic Envelope (robust covariance), a hand-rolled robust Mahalanobis
    distance (via ``MinCovDet``, falling back to plain covariance if the
    robust fit fails), a mean-distance-to-k-nearest-neighbours detector, PCA
    reconstruction error, and Gaussian-mixture negative log-likelihood.
    ``PyOD`` backs HBOS, ECOD, COPOD, ABOD, KDE-based and SOS scores -- the
    families PyOD covers that scikit-learn does not. ``hdbscan`` backs a
    GLOSH-style density outlier score; on a reference corpus whose feature
    space has no real density-cluster structure at the configured
    ``hdbscan_min_cluster_size`` (every book is "noise" -- checked and
    reported honestly, not silently), that ONE finding degrades to
    unavailable rather than reporting a meaningless number, while every other
    detector is unaffected -- one broken/inapplicable detector never
    suppresses the rest. ``style.anomaly_consensus_count`` counts how many of
    the AVAILABLE detectors placed this document above its own
    ``consensus_percentile`` (default: the 90th) of that detector's
    leave-one-out reference scores -- a count, never a replacement for the
    individual findings. ``style.anomaly_disagreement`` is the standard
    deviation of this document's corpus-percentile rank across every
    detector that produced one: near zero means the detectors agree on how
    unusual this document is, large means they do not, and either is worth
    knowing.

Evidence
    Every detector finding carries the same bounded evidence: this
    document's five most-standardized-deviant feature columns (by absolute
    z-score against the corpus median), each with its raw standardized
    deviation, the corpus median it was measured against, and whether it was
    imputed. A bare anomaly score tells a writer nothing to act on; the
    features that pushed the vector away from the reference cloud do.

Determinism
    Every detector that takes a ``random_state`` is seeded with the
    configured ``seed`` (default 42); PCA reconstruction, robust covariance
    and the closed-form distances have no randomness to seed. Two runs
    against the same profile and the same document produce the same numbers.

Comparison-unit safety
    A corpus profile's ``comparison_unit`` (book/chapter/scene) must match
    this document's own, when both are known, or the comparison is refused
    outright: a single chapter's feature vector compared against whole-book
    reference vectors would measure how the text happens to be divided, not
    how anomalous it is (see the project's ``units_comparable`` rule
    elsewhere in this codebase, applied here at the whole-vector level
    because these detectors have no per-metric unit check to fall back on).

Small-corpus refusal
    Multivariate models are unstable with a handful of reference points --
    with ~20 feature columns, fewer than a few dozen reference books leaves
    the fitted covariance/density surface dominated by noise. Every finding
    reports ``sample_size`` as the number of reference books actually used
    and ``min_sample`` as ``min_corpus_documents`` (default 20); a corpus
    below that floor gets an honest "unavailable" naming the shortfall on
    every finding, never a confident-looking number computed from too little
    data.

Still deferred
    A separate offline "prepare" command that persists a versioned, trained
    model artifact (rather than refitting at grading time). The design note
    accompanying this task judged refitting-at-grading-time acceptable given
    a reference corpus of this size, and measuring it here confirms that;
    building a model-registry format, invalidation rule and CLI for a
    persisted artifact is a larger, separable piece of work this pass does
    not attempt. Per-genre/per-comparison-unit SEPARATE models (rather than
    one pooled fit, or an outright refusal to mix units): this module refuses
    to mix comparison units, which is the safe subset of that requirement,
    but does not yet split a single-comparison-unit corpus by any further
    metadata (genre, era) into multiple parallel models -- the shipped
    reference profile carries no such field to split on.
"""

from __future__ import annotations

import contextlib
import math
import statistics
from typing import Any, Mapping, Sequence

from ..core_metrics import measure as core_measure
from ..document import DocumentAnalysis
from ..feature_matrix import build_schema, document_vector
from ..optional import require
from .common import MODERATE, finding, option

try:  # threadpoolctl ships as scikit-learn's own runtime dependency.
    import threadpoolctl as _threadpoolctl
except Exception:  # pragma: no cover - scikit-learn absent entirely
    _threadpoolctl = None


def _single_threaded():
    """Cap BLAS/OpenMP worker threads to one for the life of the ``with`` block.

    Measured on this project's own reference corpus (50 books, ~22 features):
    a single ``NearestNeighbors`` fit+query over data this small took 225ms
    under this container's default thread count, and 0.9ms with threads
    capped at one -- a container shared between several concurrent agents
    hands scikit-learn's pairwise-distance dispatcher far more threads than
    fitting forty-nine 22-dimensional points can ever use, and the thread-pool
    setup/teardown itself dominates the actual arithmetic. Sixteen detectors
    each refitting up to ``loo_reference_size`` times is exactly the regime
    where that per-call overhead, not the math, decides whether this suite is
    "moderate" or "unusably slow" -- capping it is what keeps the measurement
    honest to its declared cost class. ``threadpoolctl`` (not the
    ``OMP_NUM_THREADS``-family environment variables) is used because it
    takes effect at call time regardless of whether NumPy/BLAS already
    initialized its thread pool before this module ran, which an environment
    variable set this late in the process would not reliably do.
    """

    if _threadpoolctl is None:
        return contextlib.nullcontext()
    return _threadpoolctl.threadpool_limits(1)

FAMILY = "distribution_shape"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("sklearn",)
DEFAULT_MIN_CORPUS_DOCUMENTS = 20
MIN_SAMPLE = DEFAULT_MIN_CORPUS_DOCUMENTS
UNIT_SENSITIVE = False

PREFIX = "style.anomaly_"

DEFAULT_MAX_FEATURES = 20
DEFAULT_MIN_COVERAGE = 0.7
DEFAULT_SEED = 42
DEFAULT_LOO_REFERENCE_SIZE = 15
DEFAULT_CONSENSUS_PERCENTILE = 90.0
DEFAULT_CONTAMINATION = 0.1
DEFAULT_ISOLATION_FOREST_ESTIMATORS = 50
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 5
DEFAULT_EVIDENCE_FEATURES = 5

#: scikit-learn-backed detectors.
_SKLEARN_DETECTORS = ("isolation_forest", "lof", "one_class_svm", "elliptic_envelope",
                     "mahalanobis", "knn", "pca", "gmm")
#: PyOD-backed detectors.
_PYOD_DETECTORS = ("hbos", "ecod", "copod", "abod", "kde", "sos")
#: hdbscan-backed detector.
_HDBSCAN_DETECTORS = ("hdbscan",)

ALL_DETECTORS = _SKLEARN_DETECTORS + _PYOD_DETECTORS + _HDBSCAN_DETECTORS

_DETECTOR_PACKAGE = {name: "sklearn" for name in _SKLEARN_DETECTORS}
_DETECTOR_PACKAGE.update({name: "pyod" for name in _PYOD_DETECTORS})
_DETECTOR_PACKAGE.update({name: "hdbscan" for name in _HDBSCAN_DETECTORS})

_DETECTOR_LABELS = {
    "isolation_forest": "Isolation Forest anomaly score",
    "lof": "Local Outlier Factor anomaly score (novelty mode)",
    "one_class_svm": "One-Class SVM anomaly score",
    "elliptic_envelope": "Elliptic Envelope (robust covariance) anomaly score",
    "mahalanobis": "Robust Mahalanobis distance from the corpus centroid",
    "knn": "Mean distance to the k nearest reference books",
    "pca": "PCA reconstruction-error anomaly score",
    "gmm": "Gaussian-mixture negative log-likelihood",
    "hbos": "HBOS (Histogram-Based Outlier Score)",
    "ecod": "ECOD (Empirical Cumulative-distribution-based Outlier Detection)",
    "copod": "COPOD (Copula-Based Outlier Detection)",
    "abod": "ABOD (Angle-Based Outlier Detection)",
    "kde": "KDE-based anomaly score",
    "sos": "SOS (Stochastic Outlier Selection)",
    "hdbscan": "HDBSCAN outlier score (GLOSH)",
}

DEFAULT_FEATURES: dict[str, bool] = {name: True for name in ALL_DETECTORS}
DEFAULT_FEATURES.update({"consensus_count": True, "disagreement": True})


def _feature(config: Mapping[str, Any] | None, name: str, default: bool = True) -> bool:
    """One ``features.<name>`` flag, defaulting individually -- see
    ``stylometry_suite._feature`` for why each flag needs its own default
    rather than requiring the whole map to be re-specified.
    """

    features = option(config, "features", {})
    if not isinstance(features, Mapping):
        return default
    value = features.get(name, default)
    return default if value is None else bool(value)


# ------------------------------------------------------------- small helpers

def _percentile_rank(value: float, reference: Sequence[float]) -> float | None:
    if not reference:
        return None
    below_or_equal = sum(1 for item in reference if item <= value)
    return 100.0 * below_or_equal / len(reference)


def _threshold(reference: Sequence[float], percentile: float) -> float | None:
    if not reference:
        return None
    ordered = sorted(reference)
    index = min(len(ordered) - 1, max(0, round(percentile / 100.0 * (len(ordered) - 1))))
    return ordered[index]


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("sklearn", "pyod", "hdbscan"):
        module, reason = require(name)
        versions[name] = getattr(module, "__version__", "unknown") if module is not None else reason
    return versions


def _feature_evidence(columns: Sequence[str], vector: Sequence[float], missing: Sequence[bool],
                      medians: Mapping[str, float], limit: int) -> list[dict[str, Any]]:
    rows = sorted(zip(columns, vector, missing), key=lambda item: -abs(item[1]))
    return [{"feature": name, "standardized_deviation": round(float(value), 3),
             "corpus_median": medians[name], "imputed": bool(is_missing)}
           for name, value, is_missing in rows[:limit]]


# -------------------------------------------------------------- pyod dispatch

def _pyod_model(name: str, train, seed: int):
    """Fit one PyOD detector. ``train`` must already be a numpy array: ECOD
    and COPOD read ``.shape`` directly rather than going through scikit-learn's
    input validation, so a plain list of lists raises ``AttributeError``
    inside PyOD rather than degrading -- converting once here, rather than
    trusting every PyOD detector to accept the same input shapes sklearn
    does, is what keeps that failure from surfacing as an internal error.
    """

    from pyod.models.abod import ABOD
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.hbos import HBOS
    from pyod.models.kde import KDE
    from pyod.models.sos import SOS

    n = len(train)
    if name == "hbos":
        model = HBOS(contamination=DEFAULT_CONTAMINATION)
    elif name == "ecod":
        model = ECOD(contamination=DEFAULT_CONTAMINATION)
    elif name == "copod":
        model = COPOD(contamination=DEFAULT_CONTAMINATION)
    elif name == "abod":
        model = ABOD(contamination=DEFAULT_CONTAMINATION, method="fast",
                     n_neighbors=max(2, min(10, n - 1)))
    elif name == "kde":
        model = KDE(contamination=DEFAULT_CONTAMINATION)
    elif name == "sos":
        perplexity = max(1.0, min(4.5, (n - 1) / 3.0))
        model = SOS(contamination=DEFAULT_CONTAMINATION, perplexity=perplexity)
    else:  # pragma: no cover - guarded by ALL_DETECTORS membership
        return None
    model.fit(train)
    return model


# ------------------------------------------------------------ detector dispatch

def _score_one(name: str, train: Sequence[Sequence[float]], query: Sequence[float],
               seed: int, config: Mapping[str, Any] | None) -> tuple[float | None, str | None]:
    """Fit ``name`` on ``train`` and score ``query``. Higher = more anomalous.

    Returns ``(score, None)`` on success or ``(None, reason)`` on any
    failure -- a missing package, a degenerate fit, or the detector's own
    numerical failure -- so one bad detector never raises out of
    :func:`measure`, and the reason is the actual exception text, never a
    guess (see this project's rule against silently swallowing errors).

    A fresh :func:`_single_threaded` context is entered on every call, not
    once around the whole detector loop: ``threadpoolctl`` limits whichever
    BLAS/OpenMP libraries are ALREADY loaded at the moment its context is
    entered, and every detector here imports its backing library lazily, on
    first use. A single outer context entered before scikit-learn/PyOD/
    hdbscan's first import misses every thread pool those libraries only
    register once they are actually imported -- measured on this project's
    own reference corpus, that mistake alone was the difference between a
    kNN fit+query taking 0.9ms and 225ms. Re-entering per call costs a cheap
    library rescan, not a thread spawn, and is what actually keeps every
    detector limited regardless of import order.
    """

    package = _DETECTOR_PACKAGE.get(name)
    if package is not None:
        module, reason = require(package)
        if module is None:
            return None, reason
    with _single_threaded():
        return _score_one_inner(name, train, query, seed, config)


def _score_one_inner(name: str, train: Sequence[Sequence[float]], query: Sequence[float],
                     seed: int, config: Mapping[str, Any] | None) -> tuple[float | None, str | None]:
    try:
        train_rows = [list(row) for row in train]
        query_row = list(query)
        if name == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            n_estimators = int(option(config, "isolation_forest_n_estimators",
                                      DEFAULT_ISOLATION_FOREST_ESTIMATORS))
            model = IsolationForest(n_estimators=n_estimators, random_state=seed,
                                    contamination=DEFAULT_CONTAMINATION).fit(train_rows)
            return float(-model.score_samples([query_row])[0]), None
        if name == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            k = max(2, min(20, len(train_rows) - 1))
            model = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(train_rows)
            return float(-model.score_samples([query_row])[0]), None
        if name == "one_class_svm":
            from sklearn.svm import OneClassSVM
            nu = float(option(config, "one_class_svm_nu", DEFAULT_CONTAMINATION))
            model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu).fit(train_rows)
            return float(-model.decision_function([query_row])[0]), None
        if name == "elliptic_envelope":
            from sklearn.covariance import EllipticEnvelope
            model = EllipticEnvelope(random_state=seed, contamination=DEFAULT_CONTAMINATION,
                                     support_fraction=0.9).fit(train_rows)
            return float(-model.decision_function([query_row])[0]), None
        if name == "mahalanobis":
            from sklearn.covariance import EmpiricalCovariance, MinCovDet
            try:
                model = MinCovDet(random_state=seed, support_fraction=0.9).fit(train_rows)
            except Exception:
                model = EmpiricalCovariance().fit(train_rows)
            distance_sq = float(model.mahalanobis([query_row])[0])
            return math.sqrt(max(distance_sq, 0.0)), None
        if name == "knn":
            from sklearn.neighbors import NearestNeighbors
            k = max(1, min(10, len(train_rows)))
            model = NearestNeighbors(n_neighbors=k).fit(train_rows)
            distances, _ = model.kneighbors([query_row])
            return float(statistics.fmean(distances[0])), None
        if name == "pca":
            from sklearn.decomposition import PCA
            p = len(query_row)
            cap = max(1, min(len(train_rows) - 1, p - 1 if p > 1 else 1))
            n_components = max(1, min(cap, int(option(config, "pca_components", max(2, p // 2)))))
            model = PCA(n_components=n_components, random_state=seed).fit(train_rows)
            reconstructed = model.inverse_transform(model.transform([query_row]))[0]
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(query_row, reconstructed))), None
        if name == "gmm":
            from sklearn.mixture import GaussianMixture
            components = max(1, min(int(option(config, "gmm_components", 2)),
                                    max(1, len(train_rows) // 10)))
            model = GaussianMixture(n_components=components, random_state=seed,
                                    covariance_type="diag", reg_covar=1e-3).fit(train_rows)
            return float(-model.score_samples([query_row])[0]), None
        if name in _PYOD_DETECTORS:
            numpy_module, numpy_reason = require("numpy")
            if numpy_module is None:
                return None, numpy_reason
            train_array = numpy_module.asarray(train_rows, dtype=float)
            query_array = numpy_module.asarray([query_row], dtype=float)
            model = _pyod_model(name, train_array, seed)
            if model is None:  # pragma: no cover - guarded by ALL_DETECTORS membership
                return None, f"pyod detector {name!r} could not be constructed"
            score = float(model.decision_function(query_array)[0])
            if score != score:  # NaN guard: a degenerate affinity/kernel for this feature space
                return None, (f"pyod {name!r} returned a non-finite score; its internal "
                              f"affinity/kernel computation failed to stay finite for this "
                              f"query, which can happen for a genuinely extreme point (SOS's "
                              f"perplexity-based neighbourhood calibration in particular can "
                              f"fail exactly for the far-outlier case it exists to score) or "
                              f"for too few reference rows relative to feature count")
            return score, None
        if name == "hdbscan":
            import hdbscan as hdbscan_module
            numpy_module, numpy_reason = require("numpy")
            if numpy_module is None:
                return None, numpy_reason
            train_array = numpy_module.asarray(train_rows, dtype=float)
            query_array = numpy_module.asarray([query_row], dtype=float)
            min_cluster_size = int(option(config, "hdbscan_min_cluster_size",
                                          DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE))
            clusterer = hdbscan_module.HDBSCAN(
                min_cluster_size=min_cluster_size, prediction_data=True).fit(train_array)
            if len(set(clusterer.labels_) - {-1}) == 0:
                return None, (f"this reference corpus's feature space has no density-based "
                              f"cluster structure at hdbscan_min_cluster_size={min_cluster_size} "
                              f"(every reference book was labelled noise); HDBSCAN's outlier "
                              f"score is undefined without at least one real cluster")
            _, strengths = hdbscan_module.approximate_predict(clusterer, query_array)
            return float(1.0 - strengths[0]), None
    except Exception as exc:  # a single detector's own numerical failure
        return None, f"{type(exc).__name__}: {exc}"
    return None, f"unknown detector {name!r}"  # pragma: no cover - guarded by ALL_DETECTORS


def _loo_reference_scores(name: str, matrix: Sequence[Sequence[float]], seed: int,
                          config: Mapping[str, Any] | None,
                          sample_size: int) -> tuple[list[float], str | None]:
    """Leave-one-out scores for up to ``sample_size`` reference books.

    Bounded and evenly sampled through the corpus (not random) so the same
    corpus always yields the same reference distribution: a detector that
    refits per held-out book costs roughly ``sample_size`` fits, and this
    suite's declared cost class is "moderate", not "however many reference
    books happen to be in the corpus" -- see the module docstring.
    """

    n = len(matrix)
    if sample_size >= n:
        indices = list(range(n))
    else:
        step = n / sample_size
        indices = sorted({int(i * step) for i in range(sample_size)})
    scores: list[float] = []
    last_warning: str | None = None
    for i in indices:
        train = matrix[:i] + matrix[i + 1:]
        score, warning = _score_one(name, train, matrix[i], seed, config)
        if score is not None:
            scores.append(score)
        elif warning:
            last_warning = warning
    return scores, (last_warning if not scores else None)


# ------------------------------------------------------------------- findings

def _unavailable_all(detector_names: Sequence[str], n_books: int, min_corpus: int,
                     reason: str) -> list[dict[str, Any]]:
    out = [finding(f"{PREFIX}{name}", _DETECTOR_LABELS.get(name, name), None, "score",
                  family=FAMILY, sample_size=n_books, min_sample=min_corpus, warning=reason)
          for name in detector_names]
    out.append(finding(f"{PREFIX}consensus_count",
                       "Number of detectors that flagged this document as an outlier",
                       None, "detectors", family=FAMILY, sample_size=n_books,
                       min_sample=min_corpus, warning=reason))
    out.append(finding(f"{PREFIX}disagreement",
                       "Disagreement across detectors' normalized corpus-percentile ranks",
                       None, "percentile points", family=FAMILY, sample_size=n_books,
                       min_sample=min_corpus, warning=reason))
    return out


def _detector_finding(name: str, doc_score: float | None, doc_warning: str | None,
                      reference: Sequence[float], ref_warning: str | None,
                      n_books: int, min_corpus: int, consensus_percentile: float,
                      settings: Mapping[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    metric_id = f"{PREFIX}{name}"
    label = _DETECTOR_LABELS.get(name, name)
    if doc_score is None:
        return finding(metric_id, label, None, "score", family=FAMILY, sample_size=n_books,
                       min_sample=min_corpus, warning=doc_warning or "this detector could not be fit",
                       distribution=dict(settings))
    percentile = _percentile_rank(doc_score, reference)
    threshold = _threshold(reference, consensus_percentile)
    dist = dict(settings)
    dist.update({
        "corpus_percentile": percentile,
        "reference_scores_n": len(reference),
        "threshold_percentile": consensus_percentile,
        "threshold_value": threshold,
        "above_threshold": bool(threshold is not None and doc_score > threshold),
    })
    warning = None if reference else (
        f"leave-one-out reference scores could not be computed ({ref_warning}); the raw score "
        f"is reported with no corpus percentile or threshold")
    return finding(metric_id, label, doc_score,
                   "raw detector score (each detector's own scale; not comparable across "
                   "detector families -- see distribution.corpus_percentile for a comparable "
                   "reading)", family=FAMILY, sample_size=n_books, min_sample=min_corpus,
                   distribution=dist, warning=warning, evidence=evidence)


def _consensus_finding(scored: Mapping[str, dict[str, Any]], consensus_percentile: float,
                       n_books: int, min_corpus: int, settings: Mapping[str, Any]) -> dict[str, Any]:
    available = [name for name, result in scored.items() if result["doc_score"] is not None]
    flagged = [name for name in available
              if scored[name].get("above_threshold")]
    metric_id = f"{PREFIX}consensus_count"
    label = "Number of detectors that flagged this document as an outlier"
    if not available:
        return finding(metric_id, label, None, "detectors", family=FAMILY, sample_size=n_books,
                       min_sample=min_corpus,
                       warning="no detector could be fit; see each style.anomaly_<detector> "
                               "finding for why")
    dist = dict(settings)
    dist.update({"detectors_available": available, "detectors_flagged": flagged,
                "consensus_percentile": consensus_percentile})
    return finding(metric_id,
                   f"{label} (of {len(available)} available, above their own "
                   f"{consensus_percentile:.0f}th leave-one-out corpus percentile)",
                   len(flagged), "detectors", family=FAMILY, sample_size=n_books,
                   min_sample=min_corpus, distribution=dist,
                   evidence=[{"detector": name} for name in flagged])


def _disagreement_finding(scored: Mapping[str, dict[str, Any]], n_books: int, min_corpus: int,
                          settings: Mapping[str, Any]) -> dict[str, Any]:
    metric_id = f"{PREFIX}disagreement"
    label = ("Disagreement across detectors' normalized corpus-percentile ranks (low = detectors "
             "agree on how unusual this document is; high = they do not)")
    per_detector = {name: result["percentile"] for name, result in scored.items()
                    if result.get("percentile") is not None}
    if len(per_detector) < 2:
        return finding(metric_id, label, None, "percentile points", family=FAMILY,
                       sample_size=n_books, min_sample=min_corpus,
                       warning="fewer than two detectors produced a comparable corpus-percentile "
                               "rank for this document")
    dispersion = statistics.pstdev(per_detector.values())
    dist = dict(settings)
    dist["per_detector_percentile"] = per_detector
    return finding(metric_id, label, dispersion, "percentile points (std dev)", family=FAMILY,
                   sample_size=n_books, min_sample=min_corpus, distribution=dist)


# ------------------------------------------------------------------------ main

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    detector_names = tuple(name for name in ALL_DETECTORS if _feature(config, name, True))
    min_corpus = int(option(config, "min_corpus_documents", DEFAULT_MIN_CORPUS_DOCUMENTS))

    books = (profile or {}).get("books") if profile else None
    if not profile or not books:
        return _unavailable_all(
            detector_names, 0, min_corpus,
            "no corpus profile configured; multivariate anomaly detection needs a reference "
            "corpus of feature vectors to compare this document against")

    corpus_unit = profile.get("comparison_unit")
    doc_unit = analysis.comparison_unit
    if corpus_unit and doc_unit and doc_unit != "unknown" and doc_unit != corpus_unit:
        return _unavailable_all(
            detector_names, len(books), min_corpus,
            f"this corpus profile's comparison unit ('{corpus_unit}') does not match this "
            f"document's ('{doc_unit}'); comparing a document against reference vectors built "
            f"from a different kind of unit would measure how the text was divided, not how "
            f"anomalous it is, so the comparison is refused")

    columns = option(config, "columns", None)
    if columns is not None and not isinstance(columns, (list, tuple)):
        columns = None
    min_coverage = float(option(config, "min_coverage", DEFAULT_MIN_COVERAGE))
    max_features = int(option(config, "max_features", DEFAULT_MAX_FEATURES))
    schema = build_schema(books, columns=columns, min_coverage=min_coverage,
                          max_features=max_features)
    n_books = schema.n_books

    if n_books < min_corpus:
        return _unavailable_all(
            detector_names, n_books, min_corpus,
            f"only {n_books} reference book(s) in this corpus profile; multivariate anomaly "
            f"detection needs at least {min_corpus} to fit a stable model (set "
            f"anomaly_suite.min_corpus_documents to lower the floor, at the cost of a less "
            f"stable fit)")
    if not schema.columns:
        return _unavailable_all(
            detector_names, n_books, min_corpus,
            "no usable numeric feature column had adequate corpus coverage; see "
            "anomaly_suite.columns / anomaly_suite.min_coverage")

    lexile_source = profile.get("lexile_frequency_source", "none")
    try:
        core = core_measure(analysis, floor=1, lexile_source=lexile_source) or {}
    except Exception as exc:
        return _unavailable_all(
            detector_names, n_books, min_corpus,
            f"could not compute this document's own core metrics: {type(exc).__name__}: {exc}")

    doc_values = {key: core.get(key) for key in schema.columns}
    doc_vector, doc_missing = document_vector(schema, doc_values)
    missing_features = [key for key, is_missing in zip(schema.columns, doc_missing) if is_missing]
    evidence = _feature_evidence(schema.columns, doc_vector, doc_missing, schema.medians,
                                 int(option(config, "evidence_features", DEFAULT_EVIDENCE_FEATURES)))

    seed = int(option(config, "seed", DEFAULT_SEED))
    loo_size = min(n_books, int(option(config, "loo_reference_size", DEFAULT_LOO_REFERENCE_SIZE)))
    consensus_percentile = float(option(config, "consensus_percentile", DEFAULT_CONSENSUS_PERCENTILE))

    settings = {
        "feature_columns": list(schema.columns), "feature_count": len(schema.columns),
        "corpus_book_count": n_books, "loo_reference_size": loo_size, "seed": seed,
        "missing_document_features": missing_features, "feature_dropped": dict(schema.dropped),
        "library_versions": _library_versions(),
    }

    scored: dict[str, dict[str, Any]] = {}
    with _single_threaded():
        for name in detector_names:
            doc_score, doc_warning = _score_one(name, schema.matrix, doc_vector, seed, config)
            reference, ref_warning = ([], None) if doc_score is None else _loo_reference_scores(
                name, schema.matrix, seed, config, loo_size)
            threshold = _threshold(reference, consensus_percentile)
            scored[name] = {
                "doc_score": doc_score, "doc_warning": doc_warning,
                "reference": reference, "ref_warning": ref_warning,
                "percentile": _percentile_rank(doc_score, reference) if doc_score is not None else None,
                "above_threshold": bool(doc_score is not None and threshold is not None
                                        and doc_score > threshold),
            }

    findings = [
        _detector_finding(name, scored[name]["doc_score"], scored[name]["doc_warning"],
                          scored[name]["reference"], scored[name]["ref_warning"], n_books,
                          min_corpus, consensus_percentile, settings, evidence)
        for name in detector_names
    ]
    if _feature(config, "consensus_count", True):
        findings.append(_consensus_finding(scored, consensus_percentile, n_books, min_corpus,
                                           settings))
    if _feature(config, "disagreement", True):
        findings.append(_disagreement_finding(scored, n_books, min_corpus, settings))
    return findings
