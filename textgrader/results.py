"""Stable, serializable measurement results.

Presentation is deliberately absent from this module.  A finding is data; a
terminal renderer and a JSON consumer must reach the same accounting result.

The schema is shaped for the tool's main reader, which is an authoring agent
revising its own draft.  Such a reader needs four things a bare number cannot
give it: which family of style the measurement belongs to, which way the value
is unusual, how unusual, and whether there was enough text to say so.  Those
are ``family``, ``direction``, ``severity`` and ``sample_size``/``confidence``.

``action`` is the field to branch on.  It is never a demand:

``review``
    the measurement is far enough from the reference to be worth a look.
``informational``
    measured, and unremarkable or uncomparable.
``insufficient_data``
    there was not enough text, or not enough corpus, to say anything.
``rule_violation``
    an explicitly configured project rule was broken.  Rules are the author's
    own choices; everything else is evidence.
``unavailable`` / ``error``
    the measurement did not happen, and why.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StatusType(str, Enum):
    INFORMATIONAL = "informational"
    CORPUS_INLIER = "corpus_inlier"
    CORPUS_OUTLIER = "corpus_outlier"
    PROJECT_RULE = "project_rule_violation"
    DIAGNOSTIC = "diagnostic_review"
    UNAVAILABLE = "unavailable"
    INTERNAL_ERROR = "internal_error"


class Action(str, Enum):
    REVIEW = "review"
    INFORMATIONAL = "informational"
    INSUFFICIENT_DATA = "insufficient_data"
    RULE_VIOLATION = "rule_violation"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


#: How each action counts in :meth:`Report.scorecard`.  Every member of
#: ``Action`` appears in exactly one of these, so a new action cannot be added
#: without deciding what it means for the count.
_PASSING = frozenset({Action.INFORMATIONAL})
_FAILING = frozenset({Action.REVIEW, Action.RULE_VIOLATION})
_NOT_TAKEN = frozenset({Action.INSUFFICIENT_DATA, Action.UNAVAILABLE, Action.ERROR})

_ACTION_BY_STATUS = {
    StatusType.CORPUS_OUTLIER: Action.REVIEW,
    StatusType.DIAGNOSTIC: Action.REVIEW,
    StatusType.PROJECT_RULE: Action.RULE_VIOLATION,
    StatusType.UNAVAILABLE: Action.UNAVAILABLE,
    StatusType.INTERNAL_ERROR: Action.ERROR,
}


@dataclass
class MetricResult:
    metric_id: str
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None
    status: str = "available"
    status_type: StatusType = StatusType.INFORMATIONAL
    corpus: Optional[Dict[str, Any]] = None
    sample_size: Optional[int] = None
    details: List[Dict[str, Any]] = field(default_factory=list)
    warning: Optional[str] = None
    error: Optional[str] = None
    # --- agent-facing additions -------------------------------------------
    #: Style family, e.g. ``sentence_rhythm``, ``dialogue``, ``discourse``.
    family: Optional[str] = None
    #: ``high``, ``low``, ``typical`` or ``unknown`` relative to the corpus.
    direction: str = "unknown"
    #: Robust distance from the corpus centre; ``None`` without a comparison.
    severity: Optional[float] = None
    #: ``book``, ``chapter``, ``scene``, ``passage`` or ``unknown``.
    comparison_unit: str = "unknown"
    #: ``high``, ``low`` or ``none``: how much the corpus supports the claim.
    confidence: str = "none"
    #: The shape of the underlying sample, not just its centre.
    distribution: Optional[Dict[str, Any]] = None
    #: A short, quotable list of what produced the number.
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    #: Left unset, it is derived from ``status_type``.
    action: Optional[Action] = None
    #: What the measurement channel was: ``full``, ``dialogue``, ``narration``.
    channel: str = "full"

    def __post_init__(self):
        if isinstance(self.status_type, str):
            self.status_type = StatusType(self.status_type)
        if self.action is None:
            self.action = _ACTION_BY_STATUS.get(self.status_type, Action.INFORMATIONAL)
        elif not isinstance(self.action, Action):
            self.action = Action(self.action)

    def to_dict(self):
        data = asdict(self)
        data["status_type"] = self.status_type.value
        data["action"] = self.action.value
        return data


@dataclass
class Report:
    schema_version: int = 2
    source: Optional[str] = None
    corpus_profile: Optional[Dict[str, Any]] = None
    #: What was actually analyzed: cleanup settings, counts, segmenter.
    document: Optional[Dict[str, Any]] = None
    results: List[MetricResult] = field(default_factory=list)

    def to_dict(self):
        return {"schema_version": self.schema_version, "source": self.source,
                "corpus_profile": self.corpus_profile,
                "document": self.document,
                "results": [result.to_dict() for result in self.results],
                "summary": self.summary()}

    def scorecard(self):
        """How many measurements were taken, and how many sit inside their reference.

        A severity-ranked findings list has no denominator, and a denominator
        is the point. "3 to review" says nothing without knowing whether 91
        things were measured or 65: the tool this one replaces printed only
        failures for a while, and its author's note on why that changed reads

            Seeing only you failed 5 metrics, when there are 300, hides that
            you passed 295.

        ``not_taken`` is reported as its own number and folded into neither
        side, because that is the one that moves silently. A metric that
        errored is not a pass; a metric that was unavailable is not a failure;
        both are measurements that did not happen. The same tool once ran for
        weeks with three measures crashing and printing empty sections while
        its scorecard read "23 of 23 measures passing (100%)" - a measure
        producing no output contributed neither a pass nor a failure, so
        losing one was invisible. The real count was 59.

        Configuration results are counted apart. A misspelt key is worth
        seeing, but it is not a measurement that failed to happen, and putting
        it in that number would blunt the number's one job.

        This is not a quality score and does not make a run fail. It counts
        what was measured and how much of it sits outside its reference, both
        of which this tool already computes and already prints one at a time.
        """

        measures = [item for item in self.results if item.family != "configuration"]
        passing = [item for item in measures if item.action in _PASSING]
        failing = [item for item in measures if item.action in _FAILING]
        not_taken = [item for item in measures if item.action in _NOT_TAKEN]
        by_family: dict[str, dict[str, int]] = {}
        for bucket, items in (("passing", passing), ("failing", failing),
                              ("not_taken", not_taken)):
            for item in items:
                family = by_family.setdefault(item.family or "other",
                                              {"passing": 0, "failing": 0, "not_taken": 0})
                family[bucket] += 1
        measured = len(passing) + len(failing)
        return {
            "passing": len(passing),
            "failing": len(failing),
            "measured": measured,
            "not_taken": len(not_taken),
            "passing_share": round(100.0 * len(passing) / measured, 1) if measured else None,
            "configuration_issues": len(self.results) - len(measures),
            # Grouped, because eleven findings in one family is a habit and
            # eleven across eleven families is noise.
            "by_family": dict(sorted(by_family.items())),
            "failing_detail": [
                {"metric_id": item.metric_id, "family": item.family,
                 "severity": item.severity, "action": item.action.value}
                for item in sorted(failing, key=lambda item: -(item.severity or 0.0))],
            "not_taken_detail": [
                {"metric_id": item.metric_id, "family": item.family,
                 "action": item.action.value, "warning": item.warning or item.error}
                for item in not_taken],
        }

    def summary(self):
        """Accounting plus the short list an agent should act on first."""

        counts = {kind.value: 0 for kind in StatusType}
        actions = {kind.value: 0 for kind in Action}
        for result in self.results:
            counts[result.status_type.value] += 1
            actions[result.action.value] += 1
        review = [item for item in self.results if item.action is Action.REVIEW]
        review.sort(key=lambda item: -(item.severity or 0.0))
        families: dict[str, int] = {}
        for item in review:
            families[item.family or "other"] = families.get(item.family or "other", 0) + 1
        return {
            "total": len(self.results),
            "by_status_type": counts,
            "by_action": actions,
            "scorecard": self.scorecard(),
            "has_internal_errors": bool(counts[StatusType.INTERNAL_ERROR.value]),
            "review_families": families,
            "top_findings": [
                {"metric_id": item.metric_id, "family": item.family, "value": item.value,
                 "unit": item.unit, "direction": item.direction, "severity": item.severity,
                 "corpus_median": (item.corpus or {}).get("corpus_median",
                                                          (item.corpus or {}).get("median")),
                 "sample_size": item.sample_size, "confidence": item.confidence,
                 "comparison_unit": item.comparison_unit, "channel": item.channel}
                for item in review[:15]
            ],
        }
