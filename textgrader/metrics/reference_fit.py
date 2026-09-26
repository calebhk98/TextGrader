"""How well this document fits each of several reference corpora at once.

Every other corpus-comparison metric in this project asks how ONE axis
(sentence length, comma rate, subordination) sits against ONE reference
population. This module asks a different, complementary question over the
SAME feature space :mod:`textgrader.feature_matrix` already builds for
:mod:`textgrader.metrics.anomaly_suite`: taken as a whole vector, how far does
this document sit from each of several named reference populations at once --
never collapsed into one "this is genre X" verdict, because the whole point of
Task 24 is a reference-FIT VECTOR, one independent score per profile, that a
reader (or another metric) can act on without TextGrader deciding a genre for
them first.

Where the profiles come from
    The ``profile`` every metric module already receives (``corpus_profile``)
    is included under the fixed alias ``"primary"``. Additional profiles
    configured under ``reference_profiles`` in ``config.json`` reach this
    module through ``config["_reference_profiles"]`` -- a mapping of
    alias -> already-loaded corpus profile dict, injected ONLY for this one
    metric by ``grade.py``'s ``_optional_results`` (see that function; every
    other metric's ``config`` is unaffected). Aliases are sanitized to
    ``[a-z0-9_]+`` so they are safe metric-id fragments. This module does not
    itself read ``config.json``'s ``reference_profiles`` block or load any
    file from disk -- that loading, and the shared safeguards (preprocessing
    fingerprint, metric-definition version, Lexile source, comparison unit),
    live once in ``grade.py``'s ``load_reference_profiles``, never duplicated
    here.

    ``"primary"`` gets one further, LOCAL safety check
    (:func:`_profile_compatible`) that ``grade.py``'s existing
    ``corpus_profile`` handling does not itself enforce (that legacy path only
    WARNS on a fingerprint/unit mismatch; see ``check_preprocessing`` in
    ``grade.py``). This module holds every alias, including ``"primary"``, to
    the same withhold-on-mismatch rule the Task 24 test suite requires for
    the additional profiles, so a manuscript's own reference-fit vector is
    never computed against an incompatible primary profile just because the
    legacy path would have let it through with only a warning elsewhere in
    the report.

Feature space and distances
    Reuses :func:`textgrader.feature_matrix.build_schema`/``document_vector``
    exactly as ``anomaly_suite`` does: each profile's reference books become a
    median/MAD-standardized matrix, and this document is standardized onto
    the SAME columns/scale. ``lexile`` is dropped from the candidate columns
    here specifically (unlike ``anomaly_suite``, which keeps it): a
    manuscript's Lexile and a profile's Lexile can be computed from different
    frequency tables (see ``lexile_frequency_source`` in ``textgrader.corpus``
    and ``grade.py``'s ``check_lexile_source``), and threading that
    per-profile mismatch flag into this module would need information
    ``grade.py`` does not pass down per alias. Every other core-prose column
    is scale-free and immune to that particular hazard.

    From the standardized vector, three robust candidate distances are kept
    side by side -- deliberately not blended into one score (see the project
    philosophy: a metric is a sensor, not an opinion):

    ``distance_mean_abs_z``
        mean absolute standardized deviation across features: a general "how
        far off, on average" reading.
    ``distance_rms_z``
        root-mean-square standardized deviation: the same idea, penalizing a
        few very deviant features more than ``mean_abs_z`` does.
    ``median_abs_z``
        the median absolute standardized deviation, explicitly requested by
        the task documentation: robust to one or two wildly deviant columns
        dominating the reading the way a mean-based distance can.

    ``central_band_share`` is the percentage of this document's features
    landing within ``central_band_z`` (default 1.0) standardized units of the
    profile's centre -- how much of the vector looks unremarkable, as
    opposed to how far the remarkable part is.

Coverage
    ``coverage`` reports how many of the candidate core-prose columns were
    actually usable for this profile (enough corpus books carried a value,
    see ``feature_matrix.build_schema``'s ``min_coverage``) and, of those, how
    many this specific document itself measured rather than had imputed at
    the profile's median -- both numbers the task documentation asks for
    under "how many metrics could be validly compared."

Leave-one-out calibration
    For each profile, up to ``loo_sample_size`` of its own reference books
    (evenly sampled, so the result is deterministic and does not depend on
    corpus file order) are held out one at a time; a schema refit on the
    REMAINING books scores the held-out book's own stored feature values
    exactly as ``document_vector`` scores a graded manuscript. The MEDIAN of
    those held-out books' own median-|z| readings is this profile's
    self-consistency: how far a genuine member of the corpus typically sits
    from the rest of it. A profile whose own held-out members already read as
    distant from each other (above ``loo_calibration_threshold``, default
    1.5) is not a stable yardstick, and ``style.reference_fit_<alias>_loo_calibration``
    says so in its warning rather than silently reporting the number -- the
    explicit "profiles with poor calibration must say so" requirement.

Cross-profile findings
    ``nearest_profile``/``second_nearest_profile``/``nearest_margin`` (by
    ``distance_mean_abs_z``) and ``disagreement`` (which profile finds which
    SHARED feature column most/least typical for this document, ranked by how
    much the profiles disagree) only appear once at least two profiles
    produced a usable distance; both are explicitly reported as descriptive,
    never a forced genre classification, per the module's opening paragraph.

Determinism
    Every distance and the leave-one-out sampling are closed-form/deterministic
    (evenly-spaced indices, exactly as ``anomaly_suite._loo_reference_scores``);
    two runs against the same profile and document produce the same numbers.

Off by default; experimental. See ``config.json``'s ``metrics.reference_fit``
entry.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Mapping, Sequence

from ..core_metrics import measure as core_measure
from ..document import DocumentAnalysis
from ..feature_matrix import DEFAULT_FEATURE_COLUMNS, build_schema, document_vector
from .common import MODERATE, finding, option

FAMILY = "distribution_shape"
COST = MODERATE
REQUIRES: tuple[str, ...] = ()
UNIT_SENSITIVE = False

PREFIX = "style.reference_fit_"
PRIMARY_ALIAS = "primary"

#: Every core-prose rate/percentage/mean the corpus builder always computes,
#: minus "lexile" -- see the module docstring's "Feature space and distances".
CANDIDATE_COLUMNS: tuple[str, ...] = tuple(
    key for key in DEFAULT_FEATURE_COLUMNS if key != "lexile")

DEFAULT_MIN_CORPUS_DOCUMENTS = 8
DEFAULT_MIN_COVERAGE = 0.6
DEFAULT_MAX_FEATURES = 20
DEFAULT_CENTRAL_BAND_Z = 1.0
DEFAULT_LOO_SAMPLE_SIZE = 12
DEFAULT_LOO_CALIBRATION_THRESHOLD = 1.5

_ALIAS_RE = re.compile(r"[^a-z0-9]+")


def _sanitize_alias(alias: str) -> str:
    return _ALIAS_RE.sub("_", str(alias).casefold()).strip("_") or "profile"


def _profile_compatible(profile: Mapping[str, Any],
                        analysis: DocumentAnalysis) -> tuple[bool, str | None]:
    """Whether ``profile`` may be held against this document at all.

    Applied to EVERY alias, including ``"primary"`` -- see the module
    docstring's "Where the profiles come from".
    """

    recorded = profile.get("text_processing")
    if recorded is None:
        return False, ("this profile predates recorded text_processing settings; rebuild it "
                       "before using it as a reference_fit profile")
    if dict(recorded) != analysis.processing.fingerprint():
        return False, ("corpus and manuscript were prepared differently; comparisons withheld "
                       "(see corpus.text_processing / corpus.reference_profiles.*.text_processing)")
    unit = profile.get("comparison_unit")
    doc_unit = getattr(analysis, "comparison_unit", "unknown")
    if unit and unit != "unknown" and doc_unit not in (None, "unknown") and unit != doc_unit:
        return False, (f"this profile describes '{unit}' units and the document is "
                       f"'{doc_unit}'; comparisons withheld")
    return True, None


def _collect_profiles(profile: Mapping[str, Any] | None,
                      config: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    profiles: dict[str, Mapping[str, Any]] = {}
    if profile:
        profiles[PRIMARY_ALIAS] = profile
    extra = option(config, "_reference_profiles", {}) or {}
    for alias, extra_profile in extra.items():
        if not isinstance(extra_profile, Mapping):
            continue
        key = _sanitize_alias(alias)
        if key == PRIMARY_ALIAS and PRIMARY_ALIAS in profiles:
            # The primary corpus_profile always wins the reserved alias; an
            # extra profile that happens to sanitize to the same name is
            # renamed rather than silently dropped or silently overwriting it.
            key = f"{key}_2"
        profiles[key] = extra_profile
    return profiles


class _AliasResult:
    __slots__ = ("alias", "label", "mean_abs_z", "median_abs_z", "central_band_share",
                "per_metric_z", "findings")

    def __init__(self, alias, label, mean_abs_z=None, median_abs_z=None,
                central_band_share=None, per_metric_z=None, findings=()):
        self.alias, self.label = alias, label
        self.mean_abs_z, self.median_abs_z = mean_abs_z, median_abs_z
        self.central_band_share = central_band_share
        self.per_metric_z = per_metric_z or {}
        self.findings = list(findings)


def _unavailable_alias(alias: str, label: str, reason: str, n_books: int | None,
                       min_corpus: int) -> _AliasResult:
    names = ("distance_mean_abs_z", "distance_rms_z", "median_abs_z",
             "central_band_share", "coverage", "loo_calibration")
    findings = [finding(f"{PREFIX}{alias}_{name}",
                        f"{label}: reference fit ({name.replace('_', ' ')})",
                        None, family=FAMILY, sample_size=n_books, min_sample=min_corpus,
                        warning=reason)
               for name in names]
    return _AliasResult(alias, label, findings=findings)


def _loo_calibration(alias: str, label: str, books: Sequence[Mapping[str, Any]],
                     columns: Sequence[str], config: Mapping[str, Any] | None) -> dict[str, Any]:
    sample_size = min(len(books), int(option(config, "loo_sample_size", DEFAULT_LOO_SAMPLE_SIZE)))
    threshold = float(option(config, "loo_calibration_threshold",
                             DEFAULT_LOO_CALIBRATION_THRESHOLD))
    metric_id = f"{PREFIX}{alias}_loo_calibration"
    name = f"{label}: leave-one-out self-fit (typical median |z| of a held-out reference text)"
    n = len(books)
    if n < 3 or sample_size < 3:
        return finding(metric_id, name, None, family=FAMILY, sample_size=n,
                       warning=f"only {n} reference book(s); leave-one-out calibration needs at "
                               f"least 3")
    step = n / sample_size
    indices = sorted({int(index * step) for index in range(sample_size)})
    distances: list[float] = []
    for index in indices:
        rest = list(books[:index]) + list(books[index + 1:])
        schema = build_schema(rest, columns=columns)
        if not schema.columns:
            continue
        held_out = {key: books[index].get(key) for key in schema.columns}
        vector, _ = document_vector(schema, held_out)
        distances.append(statistics.median(abs(value) for value in vector))
    if len(distances) < 3:
        return finding(metric_id, name, None, family=FAMILY, sample_size=n,
                       warning="too few held-out reference books produced a usable feature "
                               "schema to calibrate against")
    typical = statistics.median(distances)
    quality = "adequate" if typical <= threshold else "poor"
    warning = None
    if quality == "poor":
        warning = (f"this profile's own held-out reference texts have a typical median "
                  f"standardized distance of {typical:.2f} from the rest of the same profile "
                  f"(threshold {threshold:.2f}); the profile is not tightly self-consistent on "
                  f"these features, so a document's distance against it should be read "
                  f"cautiously")
    return finding(metric_id, name, typical, "standardized units (median |z|)", family=FAMILY,
                  sample_size=len(distances), min_sample=3,
                  distribution={"held_out_books": len(distances), "quality": quality,
                               "calibration_threshold": threshold,
                               "loo_median_abs_z_values": [round(value, 4) for value in distances]},
                  warning=warning)


def _profile_fit(alias: str, profile: Mapping[str, Any], analysis: DocumentAnalysis,
                 config: Mapping[str, Any] | None) -> _AliasResult:
    label = profile.get("corpus_name") or alias
    min_corpus = int(option(config, "min_corpus_documents", DEFAULT_MIN_CORPUS_DOCUMENTS))
    books = profile.get("books")
    if not isinstance(books, list) or not books:
        return _unavailable_alias(alias, label, "this profile has no per-book rows to compare "
                                                "a feature vector against", 0, min_corpus)
    ok, reason = _profile_compatible(profile, analysis)
    if not ok:
        return _unavailable_alias(alias, label, reason, len(books), min_corpus)
    if len(books) < min_corpus:
        return _unavailable_alias(
            alias, label,
            f"only {len(books)} reference book(s) in this profile; reference_fit needs at "
            f"least {min_corpus} (set metrics.reference_fit.min_corpus_documents to lower the "
            f"floor, at the cost of a less stable fit)", len(books), min_corpus)

    min_coverage = float(option(config, "min_coverage", DEFAULT_MIN_COVERAGE))
    max_features = int(option(config, "max_features", DEFAULT_MAX_FEATURES))
    schema = build_schema(books, columns=CANDIDATE_COLUMNS, min_coverage=min_coverage,
                          max_features=max_features)
    if not schema.columns:
        return _unavailable_alias(alias, label, "no usable numeric feature column had adequate "
                                                "corpus coverage in this profile", len(books),
                                  min_corpus)

    try:
        core = core_measure(analysis, floor=1, lexile_source="none") or {}
    except Exception as exc:
        return _unavailable_alias(
            alias, label, f"could not compute this document's own core metrics: "
                         f"{type(exc).__name__}: {exc}", len(books), min_corpus)

    doc_values = {key: core.get(key) for key in schema.columns}
    doc_vector, doc_missing = document_vector(schema, doc_values)
    abs_z = [abs(value) for value in doc_vector]
    mean_abs = statistics.fmean(abs_z)
    median_abs = statistics.median(abs_z)
    rms = math.sqrt(statistics.fmean(value * value for value in doc_vector))
    band_z = float(option(config, "central_band_z", DEFAULT_CENTRAL_BAND_Z))
    inside = sum(1 for value in abs_z if value <= band_z)
    central_share = 100.0 * inside / len(abs_z)
    missing = [key for key, is_missing in zip(schema.columns, doc_missing) if is_missing]
    measured = len(schema.columns) - len(missing)

    common = {"sample_size": len(books), "min_sample": min_corpus}
    findings = [
        finding(f"{PREFIX}{alias}_distance_mean_abs_z",
               f"{label}: overall fit distance (mean absolute standardized deviation)",
               mean_abs, "standardized units", family=FAMILY,
               distribution={"feature_columns": list(schema.columns)}, **common),
        finding(f"{PREFIX}{alias}_distance_rms_z",
               f"{label}: overall fit distance (RMS standardized deviation)",
               rms, "standardized units", family=FAMILY, **common),
        finding(f"{PREFIX}{alias}_median_abs_z",
               f"{label}: median absolute standardized distance from profile centre",
               median_abs, "standardized units", family=FAMILY, **common),
        finding(f"{PREFIX}{alias}_central_band_share",
               f"{label}: share of metrics within {band_z:g} standardized units of centre",
               central_share, "%", family=FAMILY, **common),
        finding(f"{PREFIX}{alias}_coverage",
               f"{label}: metrics validly compared", measured, "metrics", family=FAMILY,
               distribution={"candidate_metrics": len(CANDIDATE_COLUMNS),
                            "profile_metrics": len(schema.columns),
                            "document_imputed_metrics": missing,
                            "feature_dropped": dict(schema.dropped)},
               **common),
        _loo_calibration(alias, label, books, schema.columns, config),
    ]
    per_metric_z = dict(zip(schema.columns, doc_vector))
    return _AliasResult(alias, label, mean_abs, median_abs, central_share, per_metric_z, findings)


def _cross_profile_findings(results: Mapping[str, _AliasResult]) -> list[dict[str, Any]]:
    usable = {alias: result for alias, result in results.items()
             if result.mean_abs_z is not None}
    if len(usable) < 2:
        return []
    ranked = sorted(usable.items(), key=lambda item: item[1].mean_abs_z)
    table = [{"alias": alias, "label": result.label,
             "distance_mean_abs_z": round(result.mean_abs_z, 4)} for alias, result in ranked]
    nearest_alias, nearest = ranked[0]
    second_alias, second = ranked[1]
    margin = second.mean_abs_z - nearest.mean_abs_z
    findings = [
        finding(f"{PREFIX}nearest_profile",
               "Nearest reference profile by overall fit distance (descriptive only, not a "
               "forced genre label)", nearest_alias, family=FAMILY, details=table),
        finding(f"{PREFIX}second_nearest_profile",
               "Second-nearest reference profile (descriptive only)", second_alias,
               family=FAMILY, details=table),
        finding(f"{PREFIX}nearest_margin",
               "Gap between the nearest and second-nearest profile's fit distance", margin,
               "standardized units", family=FAMILY),
    ]
    shared = set.intersection(*(set(result.per_metric_z) for result in usable.values()))
    rows = []
    for metric in sorted(shared):
        per_alias = {alias: abs(result.per_metric_z[metric]) for alias, result in usable.items()}
        most_typical = min(per_alias, key=per_alias.get)
        least_typical = max(per_alias, key=per_alias.get)
        rows.append({"metric": metric, "most_typical_profile": most_typical,
                    "least_typical_profile": least_typical,
                    "spread": round(per_alias[least_typical] - per_alias[most_typical], 4),
                    "per_profile_abs_z": {alias: round(value, 4) for alias, value in per_alias.items()}})
    rows.sort(key=lambda row: -row["spread"])
    findings.append(finding(
        f"{PREFIX}disagreement",
        f"Reference-profile disagreement: which of {len(usable)} profiles finds each shared "
        f"metric most/least typical for this document", len(rows), "metrics", family=FAMILY,
        evidence=rows[:15], distribution={"profiles": sorted(usable)}))
    findings.append(finding(
        f"{PREFIX}vector", "Reference-fit vector: one independent fit reading per profile",
        None, family=FAMILY,
        details=[{"alias": alias, "label": result.label,
                 "distance_mean_abs_z": round(result.mean_abs_z, 4),
                 "median_abs_z": round(result.median_abs_z, 4),
                 "central_band_share": round(result.central_band_share, 2)}
                for alias, result in ranked]))
    return findings


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    profiles = _collect_profiles(profile, config)
    if not profiles:
        return [finding(f"{PREFIX}vector",
                        "Reference-fit vector: one independent fit reading per profile", None,
                        family=FAMILY,
                        warning="no reference profile is configured (neither corpus_profile nor "
                                "reference_profiles); nothing to compare this document against")]
    results = {alias: _profile_fit(alias, alias_profile, analysis, config)
              for alias, alias_profile in profiles.items()}
    findings: list[dict[str, Any]] = []
    for alias in sorted(results):
        findings.extend(results[alias].findings)
    findings.extend(_cross_profile_findings(results))
    return findings
