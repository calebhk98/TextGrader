"""Small HTTP and JSON helpers shared by providers and enrichers."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class NetworkError(RuntimeError):
    pass


MAX_RESPONSE_BYTES = 50 * 1024 * 1024


def make_opener(user_agent: str) -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", user_agent), ("Accept-Encoding", "identity")]
    return opener


def fetch(opener, url: str, timeout: float, attempts: int = 3,
          max_bytes: int = MAX_RESPONSE_BYTES) -> bytes:
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with opener.open(url, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > max_bytes:
                    raise NetworkError(f"Response from {url} exceeds {max_bytes} bytes")
                chunks: list[bytes] = []
                total = 0
                while chunk := response.read(min(64 * 1024, max_bytes - total + 1)):
                    total += len(chunk)
                    if total > max_bytes:
                        raise NetworkError(f"Response from {url} exceeds {max_bytes} bytes")
                    chunks.append(chunk)
                return b"".join(chunks)
        except NetworkError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 4))
    raise NetworkError(f"Failed to download {url}: {last}")


def fetch_json(opener, url: str, timeout: float) -> dict[str, Any]:
    try:
        value = json.loads(fetch(opener, url, timeout).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise NetworkError(f"Invalid JSON from {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise NetworkError(f"Expected a JSON object from {url}")
    return value
