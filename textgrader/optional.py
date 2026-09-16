"""Central, cached access to optional third-party packages.

Every optional library reaches TextGrader through this module so that a
missing, broken, or too-slow dependency degrades one metric instead of the
run.  ``require`` never raises: callers receive ``None`` and a human-readable
reason they can put in a result warning.

Set ``TEXTGRADER_DISABLE_OPTIONAL`` to a comma-separated list of names (or
``all``) to simulate an environment without them, which is how the test suite
checks graceful degradation.
"""

from __future__ import annotations

import importlib
import os
import threading
from typing import Any

# name -> (module to import, pip install hint)
PACKAGES: dict[str, tuple[str, str]] = {
    "spacy": ("spacy", "pip install spacy && python -m spacy download en_core_web_sm"),
    "pysbd": ("pysbd", "pip install pysbd"),
    "numpy": ("numpy", "pip install numpy"),
    "scipy": ("scipy", "pip install scipy"),
    "scipy.stats": ("scipy.stats", "pip install scipy"),
    "pandas": ("pandas", "pip install pandas"),
    "regex": ("regex", "pip install regex"),
    "wordfreq": ("wordfreq", "pip install wordfreq"),
    "lexicalrichness": ("lexicalrichness", "pip install lexicalrichness"),
    "sentence_transformers": ("sentence_transformers", "pip install sentence-transformers"),
    "sklearn": ("sklearn", "pip install scikit-learn"),
    "ruptures": ("ruptures", "pip install ruptures"),
    "textdescriptives": ("textdescriptives", "pip install textdescriptives"),
}

_lock = threading.Lock()
_cache: dict[str, tuple[Any, str | None]] = {}
#: Modules that keep their own cache of something built from an optional
#: package register a callback here, so clearing this cache really does restore
#: a first-run state.  Without it a test that simulates a missing package
#: leaves a module holding the "unavailable" answer for the rest of the process.
_reset_hooks: list = []


def _disabled() -> set[str]:
    raw = os.environ.get("TEXTGRADER_DISABLE_OPTIONAL", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def require(name: str) -> tuple[Any, str | None]:
    """Return ``(module, None)`` or ``(None, reason)``.  Never raises."""

    disabled = _disabled()
    if "all" in disabled or name in disabled:
        return None, f"optional package '{name}' disabled by TEXTGRADER_DISABLE_OPTIONAL"
    with _lock:
        if name in _cache:
            return _cache[name]
    module_name, hint = PACKAGES.get(name, (name, f"pip install {name}"))
    try:
        module = importlib.import_module(module_name)
        outcome: tuple[Any, str | None] = (module, None)
    except Exception as exc:  # ImportError, but a broken build can raise anything
        outcome = (None, f"optional package '{name}' unavailable ({type(exc).__name__}: {exc}); {hint}")
    with _lock:
        _cache.setdefault(name, outcome)
        return _cache[name]


def have(name: str) -> bool:
    return require(name)[0] is not None


def on_reset(callback) -> None:
    """Register a cache to clear whenever :func:`reset_cache` is called."""

    with _lock:
        if callback not in _reset_hooks:
            _reset_hooks.append(callback)


def reset_cache() -> None:
    """Forget cached import outcomes; used by tests that toggle availability."""

    with _lock:
        _cache.clear()
        hooks = list(_reset_hooks)
    for hook in hooks:
        hook()


def installed() -> dict[str, str | None]:
    """Report every optional package and why it is unusable, if it is."""

    return {name: require(name)[1] for name in sorted(PACKAGES)}
