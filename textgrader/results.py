"""Stable, serializable measurement results.

Presentation is deliberately absent from this module.  A finding is data; a
terminal renderer and a JSON consumer must reach the same accounting result.
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

    def __post_init__(self):
        if isinstance(self.status_type, str):
            self.status_type = StatusType(self.status_type)

    def to_dict(self):
        data = asdict(self)
        data["status_type"] = self.status_type.value
        return data


@dataclass
class Report:
    schema_version: int = 1
    source: Optional[str] = None
    corpus_profile: Optional[Dict[str, Any]] = None
    results: List[MetricResult] = field(default_factory=list)

    def to_dict(self):
        return {"schema_version": self.schema_version, "source": self.source,
                "corpus_profile": self.corpus_profile,
                "results": [result.to_dict() for result in self.results],
                "summary": self.summary()}

    def summary(self):
        counts = {kind.value: 0 for kind in StatusType}
        for result in self.results:
            counts[result.status_type.value] += 1
        return {"total": len(self.results), "by_status_type": counts,
                "has_internal_errors": bool(counts[StatusType.INTERNAL_ERROR.value])}
