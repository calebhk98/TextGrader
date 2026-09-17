"""Common HTTP-provider behavior."""
from __future__ import annotations

from ..models import ProviderStatus
from ..net import NetworkError, fetch, make_opener


class HTTPProvider:
    name = "unknown"
    health_url = ""

    def __init__(self, *, timeout: float = 30.0, user_agent: str = "TextGrader-CorpusBuilder/2.0") -> None:
        self.timeout = timeout
        self.opener = make_opener(user_agent)

    def _get(self, url: str) -> bytes:
        return fetch(self.opener, url, self.timeout)

    def healthcheck(self) -> ProviderStatus:
        try:
            self._get(self.health_url)
            return ProviderStatus(True, "reachable")
        except NetworkError as exc:
            return ProviderStatus(False, str(exc))
