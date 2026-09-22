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
    # Neural coreference resolution for coherence_suite's optional
    # "coreference" feature (off by default: see that module's docstring).
    # Pulls in torch and a transformer encoder, so it is never imported
    # unless that feature flag is explicitly on.
    "fastcoref": ("fastcoref", "pip install fastcoref"),
    # WordNet synonymy/hypernymy for coherence_suite's optional
    # "lexical_wordnet" feature. The package alone is not enough - the
    # 'wordnet' corpus itself must also be downloaded
    # (python -m nltk.downloader wordnet); coherence.py checks for that
    # separately and reports it as its own unavailable reason.
    "nltk": ("nltk", "pip install nltk && python -m nltk.downloader wordnet"),
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
    # NLI cross-encoder backend for logic_suite's "nli_entailment" feature (off
    # by default). Heavy (pulls in a transformer checkpoint) and never imported
    # unless that flag is explicitly on -- see the module docstring's gating
    # note for why that matters here more than anywhere else in this codebase.
    "transformers": ("transformers", "pip install transformers"),
    # Only transformers' own import is touched directly; this entry exists so
    # `installed()`/`requirements.txt` account for the CPU wheel transformers
    # needs, and so a broken torch build reports through the same channel as
    # every other optional dependency instead of raising on import.
    "torch": ("torch", "pip install torch"),
    # Coreference resolution for logic_suite's "coreference_resolution"
    # feature (off by default). See textgrader/propositions.py for why a
    # pronoun subject is otherwise dropped from every cross-sentence check.
    "fastcoref": ("fastcoref", "pip install fastcoref"),
    # WordNet antonym/hypernym relations for logic_suite's "lexical_opposition"
    # feature. The `nltk` package alone is not enough -- its corpus data is a
    # separate download; textgrader.propositions checks for that data itself
    # and reports a LookupError as an actionable "unavailable", not a crash.
    "nltk": ("nltk", "pip install nltk"),
    # Date parsing for logic_suite's "temporal_ordering" feature: which of two
    # differing dates on the same subject+predicate is earlier, not just that
    # they differ.
    "dateutil": ("dateutil.parser", "pip install python-dateutil"),
    # catch22: 22 canonical, published time-series features (Lubba et al.
    # 2019) for timeseries_suite's optional "catch22_*" feature group. Kept
    # as its own package rather than reimplemented because the whole point
    # of catch22 is that its 22 features are a specific, citable, externally
    # validated selection, not a set this codebase should be re-deriving.
    "pycatch22": ("pycatch22", "pip install pycatch22"),
    # Discrete wavelet transform for timeseries_suite's optional
    # "wavelet_energy"/"wavelet_entropy" feature group (energy per scale and
    # Shannon entropy of that per-scale distribution).
    "pywt": ("pywt", "pip install PyWavelets"),
    # A configurable, off-by-default tsfresh feature set for timeseries_suite's
    # optional "tsfresh" feature group. Heavy (a large dependency tree; the
    # "comprehensive" preset alone can extract close to 800 numbers from one
    # sequence) so it never runs unless "tsfresh" is explicitly selected in
    # feature_groups -- see that group's own note in timeseries_suite.py for
    # why it is still exactly one finding per sequence regardless of preset.
    "tsfresh": ("tsfresh", "pip install tsfresh"),
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


def shim_fastcoref_transformers() -> None:
    """Supply the tied-weight default newer ``transformers`` expects, for
    ``fastcoref``'s sake.

    ``fastcoref==2.1.6``'s model class never runs the tied-weight bookkeeping
    that ``transformers`` 5.x expects every ``PreTrainedModel`` subclass to
    have completed, so ``from_pretrained`` can raise ``AttributeError: ... has
    no attribute 'all_tied_weights_keys'`` before a single weight is read.
    fastcoref's coref head is a span classifier with no input/output embedding
    to tie, so an empty mapping is the correct value rather than a guess: this
    only supplies the default the class would itself have set on the newer init
    path, and only when the attribute is missing, so an already-compatible
    ``transformers`` is never touched.  A model that initializes properly sets
    the attribute per instance, which shadows this class-level default.

    Both callers that load fastcoref -- the coherence suite and the logic
    suite's proposition helper -- go through here, so the two cannot disagree
    about whether coreference is available, which is exactly what happened
    when each carried its own loader.
    """

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:  # pragma: no cover - transformers itself unavailable
        return
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


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
