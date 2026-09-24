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
    # one "unavailable" finding rather than to a missing suite, on an
    # install that genuinely lacks it. All five - zstandard, brotli, lz4,
    # snappy, pyppmd - are installed and exercised for real here; pyppmd
    # additionally backs a genuine PPM predictive-model channel
    # (cross-entropy, not only a compression ratio). snappy needs the system
    # libsnappy-dev package (not a pip install), which this environment now
    # has.
    "zstandard": ("zstandard", "pip install zstandard"),
    "brotli": ("brotli", "pip install brotli"),
    "lz4": ("lz4.frame", "pip install lz4"),
    "snappy": ("snappy", "pip install python-snappy (also needs the system libsnappy-dev "
                         "package)"),
    "pyppmd": ("pyppmd", "pip install pyppmd"),
    # Query-time bindings for randomness_suite's optional "kenlm_language_model"
    # feature (off by default). The kenlm wheel gives kenlm.Model but NOT the
    # trainer, lmplz, which is a separate C++ program built from KenLM's own
    # source (a C++ compiler and Boost); that channel finds lmplz itself, via
    # config or PATH, and reports its own actionable "unavailable" when it
    # cannot -- this hint only covers the Python side.
    "kenlm": ("kenlm", "pip install kenlm (query-time bindings only; the lmplz trainer "
                       "must be built from source, see "
                       "https://github.com/kpu/kenlm#compiling)"),
    # Real constituency-tree parsing for syntax_complexity_suite's optional
    # "constituency" feature (off by default even when the suite is on: see
    # that module's docstring). Pulls in torch (already required by
    # fastcoref/isanlp_rst above) plus its own pretrained parser checkpoint,
    # downloaded once via `python -c "import benepar; benepar.download('benepar_en3')"`.
    "benepar": ("benepar", "pip install benepar && python -c \"import benepar; "
                          "benepar.download('benepar_en3')\""),
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
    # The same "transformers" install also backs three more off-by-default
    # logic_suite features added in a later pass -- "semantic_role_labeling"
    # (a seq2seq PropBank SRL model), "relation_extraction" (a seq2seq
    # closed-schema relation extractor, REBEL) and "argument_mining" (a
    # RoBERTa argument-relation classifier) -- each its own Hugging Face hub
    # checkpoint, no new package for any of them.
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
    # The same `nltk` install also backs three more off-by-default logic_suite
    # features -- "propbank_argument_structure", "verbnet_class_consistency"
    # and "framenet_frame_consistency" -- each needing its own separately
    # downloaded corpus (propbank, verbnet, framenet_v17 respectively; none
    # satisfies another), checked and reported the identical way.
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
    # Real, independent Burrows/Cosine/Argamon-quadratic/weighted Delta,
    # Zeta and Kilgarriff's chi-squared for stylometry_suite's optional
    # "pystylometry_reference" feature (off by default; reads real reference
    # text from disk at grading time). Lightweight (only "rich" as its own
    # dependency). This is the package genuinely on PyPI for this family --
    # see stylometry_suite's module docstring for why the literally-named
    # "pydelta" package is NOT it (a biological-taxonomy package) and why the
    # real PyDelta project (github.com/cophi-wue/pydelta) could not be
    # installed here either.
    "pystylometry": ("pystylometry", "pip install pystylometry"),
    # A tiny, independent character-bigram gibberish classifier for
    # randomness_suite's optional "gibberish_detector_package" feature (off by
    # default; see the module docstring's gating rule). Trained fresh on each
    # document's own held-out split, exactly like this suite's from-scratch
    # n-gram channels -- never on the package's own bundled reference file --
    # so it degrades to "unavailable" on an install that lacks it rather than
    # to an import error.
    "gibberish_detector": ("gibberish_detector", "pip install gibberish-detector"),
    # Streaming ADWIN change-detector for timeseries_suite's optional "adwin"
    # feature group, run beside (not instead of) the suite's own hand-written
    # Page-Hinkley detector -- two detectors disagreeing about where a book
    # changes is data, not redundancy. Pure Python/C wheel, no model download.
    "river": ("river", "pip install river"),
    # VADER lexicon-and-rule sentiment scoring for
    # textgrader.sequences' "sentence_sentiment_compound" sequence. The
    # lexicon ships inside the wheel (vaderSentiment/vader_lexicon.txt); unlike
    # nltk's VADER data this needs no separate download or network call, which
    # is exactly what let this sequence go in where an earlier pass deferred
    # it as blocked.
    "vaderSentiment": ("vaderSentiment.vaderSentiment", "pip install vaderSentiment"),
    # NRC emotion-lexicon scoring for textgrader.sequences'
    # "sentence_emotion_valence" sequence. Its lexicon also ships inside the
    # wheel (nrclex/data/nrc_en.json) and is read via importlib.resources, so
    # -- like vaderSentiment above -- no download or network call is needed at
    # runtime, even though the package's own declared dependencies (nltk,
    # textblob) are pulled in at install time for its optional
    # load_raw_text() path, which this codebase never calls (it tokenizes its
    # own sentences and calls load_token_list() instead).
    "nrclex": ("nrclex", "pip install NRCLex"),
    # Real RST (Rhetorical Structure Theory) discourse-tree parsing for
    # coherence_suite's optional "rst" feature (off by default). Imports the
    # 'isanlp_rst.parser' submodule directly so a caller gets the module that
    # actually exposes Parser. Needs the non-PyPI 'isanlp' package too (pip
    # install git+https://github.com/iinemo/isanlp.git); constructing the
    # parser itself (not just importing the package) downloads ~5 GB of
    # model checkpoints on first use and holds ~5-6 GB resident once loaded,
    # which is why textgrader.coherence caches one parser per process (see
    # _load_rst_parser) and why the feature samples a bounded, spread-out set
    # of passages instead of parsing a whole book -- see coherence_suite's
    # module docstring for the full cost story.
    "isanlp_rst": ("isanlp_rst.parser", "pip install isanlp-rst (also: pip install "
                  "git+https://github.com/iinemo/isanlp.git); pulls in torch/transformers and "
                  "downloads ~5 GB of model checkpoints on first use"),
    # A real, independent anomaly/outlier-detection library (61 detectors
    # spanning tabular/time-series/graph/text/image/audio data as of writing)
    # for textgrader.metrics.anomaly_suite's HBOS/ECOD/COPOD/ABOD/KDE/SOS
    # channels, which scikit-learn does not implement. Confirmed via
    # `pip show pyod` before use, per this project's rule that a package's
    # name is not evidence of what it does.
    "pyod": ("pyod", "pip install pyod"),
    # Density-based clustering (McInnes & Healy) for anomaly_suite's GLOSH
    # outlier-score channel. Confirmed via `pip show hdbscan` before use, same
    # reason as pyod above.
    "hdbscan": ("hdbscan", "pip install hdbscan"),
    # Independent, frequency-based dictionary spell checking for
    # mechanical_quality_suite's spelling channels. Verified for real: its
    # SpellChecker().correction('wrold') returns 'world' (see that module's
    # docstring). Bundles its own dictionary; no network call at runtime.
    "pyspellchecker": ("spellchecker", "pip install pyspellchecker"),
    # A second, independent edit-distance dictionary (SymSpell) for the same
    # suite's spelling channels, plus real word segmentation used for
    # fused-token detection. Bundles its own frequency dictionary (English
    # only); verified for real: word_segmentation('helloworld') returns
    # 'hello world' (see mechanical_quality_suite's docstring).
    "symspellpy": ("symspellpy", "pip install symspellpy"),
    # Mixed-script homoglyph detection for mechanical_quality_suite's
    # "confusables" feature. Verified for real: is_dangerous('paypal') is
    # False (plain ASCII) and is_dangerous('pаypal') (Cyrillic а) is
    # True -- see that module's docstring. Imported as its own submodule
    # because confusable_homoglyphs/__init__.py does not import it itself.
    "confusable_homoglyphs": ("confusable_homoglyphs.confusables",
                             "pip install confusable-homoglyphs"),
    # Mojibake/Unicode repair detection for mechanical_quality_suite's
    # "encoding_ftfy" feature, used diagnostically only -- it never rewrites
    # the analyzed text (see that module's docstring). Verified for real:
    # fix_text('MÃ¼nchen') returns 'München'.
    "ftfy": ("ftfy", "pip install ftfy"),
    # An independent, pure-Python rule-based sentence/token segmenter for
    # parser_consensus's default "segmentation"/"tokenization" features (the
    # 'syntok' segmenter/tokenizer sources). Verified for real: correctly
    # keeps "Mr." and "Dr." sentence-final and gives exact character-span
    # boundaries via token.offset (see parser_consensus.py's docstring).
    # Lightweight (only 'regex' as its own dependency, already installed).
    "syntok": ("syntok", "pip install syntok"),
    # Independent tokenization/POS/dependency parsing (and, with the right
    # processors, constituency parsing) for parser_consensus's off-by-default
    # "stanza" feature (POS/dependency/chunk consensus's second parser) and
    # the "chunks" feature's constituency source. The package alone is not
    # enough: its English models are a separate, non-PyPI download this
    # module never triggers itself (every stanza.Pipeline call passes
    # download_method=None) -- run e.g.
    # `python -c "import stanza; stanza.download('en', processors='tokenize,mwt,pos,lemma,depparse')"`
    # first (~320MB on disk, verified in this environment). Depends on torch,
    # so it lives in requirements-embeddings.txt rather than requirements.txt.
    "stanza": ("stanza", "pip install stanza (in requirements-embeddings.txt; also needs "
                        "stanza.download('en', processors=...) models, ~320MB, never "
                        "triggered by this codebase itself)"),
    # A second, independent graph library for graph_suite's disagreement
    # channel (density/transitivity/components/modularity recomputed
    # separately from networkx, never averaged with it -- see
    # textgrader/graphs.py's module docstring). Confirmed for real: pip
    # installs cleanly alongside the current numpy/scipy/scikit-learn with no
    # version changes to any of them.
    "igraph": ("igraph", "pip install igraph"),
    # A third, fast graph implementation, installed and available for
    # graph_suite but not currently wired into the disagreement channel
    # (igraph already answers "do two implementations agree"; see
    # textgrader/graphs.py's module docstring for why a third recomputation
    # of the same statistics was judged not worth the extra code to keep in
    # sync). Torch-free, no version conflicts with anything else installed.
    "rustworkx": ("rustworkx", "pip install rustworkx"),
    # Real character/entity/coreference/quote-speaker extraction for
    # graph_suite's optional "booknlp" feature (off by default; very heavy --
    # see that module's docstring for measured time/memory). Installed with
    # --no-deps deliberately: BookNLP's own declared dependencies include
    # tensorflow (an unused ~570MB download; nothing in the installed
    # package tree imports it -- confirmed with
    # `grep -rn tensorflow site-packages/booknlp` returning nothing) and an
    # unconstrained numpy bound that would upgrade this project's pinned
    # numpy 1.x to 2.x, which this project's pip-install rules refuse to do
    # for a dependency this project does not otherwise need upgraded.
    # `--no-deps` avoids both; spacy/torch/transformers, which BookNLP does
    # actually use, are already satisfied by this project's own requirements.
    "booknlp": ("booknlp.booknlp", "pip install --no-deps booknlp (installing it normally also "
                                   "pulls in an unused ~570MB tensorflow and upgrades numpy to "
                                   "2.x; see textgrader/optional.py's comment for why --no-deps "
                                   "is used instead)"),
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


def shim_benepar_transformers() -> None:
    """Restore ``T5Tokenizer(Fast).build_inputs_with_special_tokens``.

    benepar's retokenizer (``benepar/retokenization.py``) calls this method
    to locate a T5 tokenizer's special-token positions. ``transformers``
    5.17.0's rewritten tokenizer classes (``TokenizersBackend``) no longer
    define it at all, so loading ``benepar_en3`` (a T5-based checkpoint)
    raises ``AttributeError: T5Tokenizer has no attribute
    build_inputs_with_special_tokens`` -- reproduced directly in this
    environment before this shim existed. The restored method is exactly
    T5's own pre-5.x rule: a single sequence gets one trailing EOS id, a
    pair gets EOS after each sequence. Installed only when the attribute is
    actually missing (``hasattr`` guard), so a future transformers release
    that restores it is left alone, the same pattern as
    :func:`shim_fastcoref_transformers`.

    Both suites that load benepar (``syntax_complexity_suite`` and
    ``parser_consensus``) go through here.  Before they shared it, one loaded
    benepar and the other reported it unavailable in the same environment.
    """

    try:
        from transformers import T5Tokenizer, T5TokenizerFast
    except Exception:  # pragma: no cover - transformers itself unavailable
        return

    def _build(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    for cls in (T5Tokenizer, T5TokenizerFast):
        if not hasattr(cls, "build_inputs_with_special_tokens"):
            cls.build_inputs_with_special_tokens = _build


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


_booknlp_shim_applied = False


def shim_booknlp_transformers() -> None:
    """Let ``torch.nn.Module.load_state_dict`` ignore a legacy BERT buffer key
    BookNLP's shipped checkpoints still carry.

    Reproduced directly in this environment: constructing
    ``booknlp.booknlp.BookNLP`` raises ``RuntimeError: Error(s) in loading
    state_dict for Tagger: Unexpected key(s) in state_dict:
    "bert.embeddings.position_ids"`` before a single real weight mismatch is
    even considered. BookNLP's small-model checkpoints were saved from an
    older ``transformers`` BERT implementation that registered
    ``position_ids`` as a persistent buffer; newer ``transformers`` versions
    stopped registering it (it is derived at forward time instead), so the
    current model object has no such key to receive it and ``strict=True``
    loading (PyTorch's default, which BookNLP's own loading code never
    overrides) refuses the whole checkpoint over one now-unused buffer.

    The fix is narrow and additive: patch ``load_state_dict`` to drop only
    keys that (a) end in ``"position_ids"`` and (b) are not present in the
    receiving module's own ``state_dict()`` -- i.e. exactly the legacy keys a
    newer transformers build no longer creates -- and pass everything else
    through to the original implementation unchanged.  A model whose current
    architecture DOES register a matching key is unaffected (that key stays
    in ``own_keys`` and is kept), so this can never mask a real shape or name
    mismatch elsewhere in a checkpoint.  Applied at most once per process
    (the ``_booknlp_shim_applied`` guard), the same idempotency style
    :func:`shim_fastcoref_transformers`/:func:`shim_benepar_transformers` use,
    so calling this repeatedly (once per document that enables the
    ``booknlp`` feature) never double-wraps the method.
    """

    global _booknlp_shim_applied
    if _booknlp_shim_applied:
        return
    try:
        import torch
    except Exception:  # pragma: no cover - torch itself unavailable
        return
    original = torch.nn.Module.load_state_dict

    def _patched(self, state_dict, *args, **kwargs):
        own_keys = set(self.state_dict().keys())
        filtered = {key: value for key, value in state_dict.items()
                   if key in own_keys or not key.endswith("position_ids")}
        return original(self, filtered, *args, **kwargs)

    torch.nn.Module.load_state_dict = _patched
    _booknlp_shim_applied = True


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
