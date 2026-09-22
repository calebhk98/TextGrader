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
    # spaCy component metrics (entropy/perplexity, readability); used directly
    # by randomness_suite's textdescriptives cross-check (off by default,
    # needs its own spaCy pipeline with the component attached).
    "textdescriptives": ("textdescriptives", "pip install textdescriptives"),
    "networkx": ("networkx", "pip install networkx"),
    # Compression channels for textgrader.metrics.randomness_suite. Every one
    # of these is optional: the stdlib zlib/gzip/bz2/lzma channels cover the
    # suite's acceptance criteria on their own, and each of these degrades to
    # one "unavailable" finding rather than to a missing suite. All four -
    # zstandard, brotli, lz4, pyppmd - are installed and exercised for real;
    # pyppmd additionally backs a genuine PPM predictive-model channel
    # (cross-entropy, not only a compression ratio). snappy needs the system
    # libsnappy-dev package, which this environment does not have, so it
    # stays unavailable, proving the degradation path still works.
    "zstandard": ("zstandard", "pip install zstandard"),
    "brotli": ("brotli", "pip install brotli"),
    "lz4": ("lz4.frame", "pip install lz4"),
    "snappy": ("snappy", "pip install python-snappy"),
    "pyppmd": ("pyppmd", "pip install pyppmd"),
    # A pretrained causal language model for randomness_suite's neural-LM
    # perplexity channel (features.neural_language_model, off by default).
    # Heavy (torch is a multi-gigabyte install) and only imported when that
    # flag is explicitly on - never during corpus profiling, which runs this
    # suite's other, cheap channels with default settings. See that channel's
    # docstring for why its perplexity is not a gibberish detector.
    "torch": ("torch", "pip install torch"),
    "transformers": ("transformers", "pip install transformers"),
    # ADF stationarity test for timeseries_suite's "stationarity" feature.
    # Everything else that suite computes (ACF, trend, spectral, Hurst, DFA,
    # permutation entropy, change points, Page-Hinkley) is dependency-free or
    # already covered by numpy/scipy/ruptures above; this is the one classical
    # time-series statistic worth a real implementation rather than a proxy.
    "statsmodels": ("statsmodels", "pip install statsmodels"),
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
