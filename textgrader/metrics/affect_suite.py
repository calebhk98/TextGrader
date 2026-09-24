"""Sentiment, emotion, affect and tone-trajectory analysis (experimental).

This suite measures; it never judges.  A tragedy should read dark and a
comedy light, a villain's chapters should sound angrier than a lullaby, and
none of that is a defect.  Every finding here stays ``Polarity.NEUTRAL`` (the
default this codebase's ``finding()``/``MetricResult`` already fall back to
when a metric passes no polarity), and nothing in this module compares "more
positive" to "better."  What it reports is *structure*: how much a document's
affect moves, where it lands early versus late, whether dialogue reads
differently from narration, whether named speakers sound different from one
another, and how much independent sentiment/emotion systems agree or
disagree about the same sentence.

Library survey (see the task document, ``docs/experimental-tasks/
15-add-sentiment-emotion-affect-and-tone-trajectory-analysis.md``):

* **sentimentr** is explicitly out of scope: it is an R package, and
  TextGrader is Python-only end to end (see ``textgrader/optional.py``'s
  module docstring and rule 6 of the shared cross-task rules).  No R runtime
  is invoked anywhere in this module.
* **SEANCE** (Sentiment Analysis and Cognition Engine) is, as far as this
  environment could determine, distributed only as a compiled desktop
  application (a Java-based GUI, from the original SEANCE/TAALES/TAACO
  family of tools by Kristopher Kyle et al.), not as an installable Python
  package.  ``pip index`` has no package literally named ``seance`` that is
  this tool (searched via the same "a package's name is not evidence of
  what it does" rule this project applies everywhere else); nothing on PyPI
  reimplements its specific dictionary blend. SEANCE is therefore deferred,
  not silently dropped: its niche (a broad sentiment/cognition-process
  lexicon blend) is covered here by the combination of VADER, AFINN,
  NRC EmoLex and Empath, each kept as its own independent, disagreeing
  channel rather than merged into one number.
* **LIWC** is commercially licensed and never bundled.  ``liwc`` (PyPI) is a
  pure, local parser for the ``.dic`` *file format* LIWC dictionaries use --
  not the dictionary itself -- verified for real against a small synthetic
  fixture (a two-category, wildcard-bearing ``.dic``) before being trusted:
  ``liwc.load_token_parser(...)`` correctly matched ``"happy"``/``"joyful"``
  (via the ``joy*`` wildcard) to a ``posemo`` category and left a stopword
  unmatched.  This module's ``liwc_categories`` channel is a pure no-op
  (nothing is imported or read) unless a user points ``liwc_dictionary_path``
  at their own licensed ``.dic`` file.

Everything else in the task's library table is real and installed:
VADER and the NRC Word-Emotion Association Lexicon are already loaded
elsewhere in this codebase (see "Reuse, not re-implementation" below); AFINN,
TextBlob/Pattern and Empath are new, verified installs; NRC VAD is Task 11's
responsibility (see "The engine registry" below); scipy/statsmodels are used
indirectly through :mod:`textgrader.stats`, which already implements the
arc/trend/correlation machinery this module calls.

Reuse, not re-implementation
-----------------------------
:mod:`textgrader.sequences` already loads and calls ``vaderSentiment`` (for
``sentence_sentiment_compound``) and ``nrclex`` (for
``sentence_emotion_valence``).  This module's built-in ``"vader"`` and
``"nrc_valence"`` engines call ``vaderSentiment``/``nrclex`` through their own
tiny scorer factories below (:func:`_make_vader_scorer`,
:func:`_make_nrc_valence_scorer`) rather than a second copy of
``sequences.py``'s sentence-loop, because this suite's engines need a bare
``str -> float`` function it can also call on a single dialogue turn or a
future conversation-suite utterance (see "Per-sentence scoring is
importable" below), not only on a whole ``DocumentAnalysis``.  The actual
library calls are identical to ``sequences.py``'s
(``SentimentIntensityAnalyzer().polarity_scores(text)["compound"]``;
``NRCLex(); load_token_list(...); affect_frequencies``): nothing about how
either package is invoked is reinvented, only the wrapper shape.  The NRC
emotion-*category* distribution (anger/fear/joy/... -- a different channel
from the valence-balance trajectory) reads ``NRCLex.affect_frequencies``'
other keys directly for the same reason: ``sequences.py`` only ever computes
the positive-minus-negative balance, and the category ontology this task
asks for needs the rest of that same dictionary.

The engine registry
--------------------
Every trajectory/volatility/arc/dialogue-gap/speaker-spread/correlation
finding in this module is driven by a small, open registry of named
per-sentence scorers, :data:`_ENGINES` (populated by :func:`register_engine`,
read by :func:`list_engines`).  An :class:`EngineSpec` is a name, a label, a
unit, whether its sign is meaningful (``signed``), and a
``make_scorer(settings) -> (str -> float | None, reason)`` factory.  Nothing
past the registry cares whether an engine is VADER, AFINN or something
registered from outside this module.

**Task 11 integration (two lines per dimension).**  Task 11 owns loading
every psycholinguistic norm table, including NRC VAD and Warriner VAD, in
``textgrader/lexicons.py``, which this module deliberately never imports or
downloads on its own.  Once that module exists, its orchestrator (wherever
suites get wired together after both branches merge) registers a VAD engine
per dimension with two lines each::

    from textgrader.metrics import affect_suite
    from textgrader import lexicons

    affect_suite.register_engine(affect_suite.EngineSpec(
        name="vad_valence", label="NRC-VAD valence", unit="valence [0,1]",
        signed=True, kind="vad", resource="NRC-VAD lexicon",
        version=lambda: lexicons.VAD_VERSION,
        make_scorer=lambda settings: lexicons.vad_scorer("valence", settings)))
    # ... same shape again for name="vad_arousal" and name="vad_dominance".

``make_scorer`` must return ``(scorer, None)`` on success or ``(None,
reason)`` when the resource cannot be loaded (missing table, bad path, ...),
matching every built-in engine below and :func:`textgrader.optional.require`'s
own ``(value, reason)`` convention.  Once registered, ``vad_valence`` gets
the exact same ``discourse.affect_vad_valence_level`` /
``..._volatility`` / ``..._arc`` / ``..._dialogue_narration_gap`` /
``..._speaker_spread`` / ``..._sequence_correlation`` findings every other
engine gets, gated by the same ``features`` map (default: enabled, since
``EngineSpec.default_enabled`` defaults to ``True``) -- no other change to
this file is needed.  Until that registration happens, :func:`measure`
reports three explicit placeholder ``unavailable`` findings under the
forward-stable ids ``discourse.affect_vad_valence_level``,
``..._vad_arousal_level`` and ``..._vad_dominance_level`` (see
:func:`_vad_placeholder_findings`), so a reader sees the channel is planned
rather than silently missing, and so a corpus profile built before and after
Task 11 lands compares the same metric ids.

Per-sentence scoring is importable
-----------------------------------
Task 14's conversation suite may want one speaker's per-utterance affect
without building a full ``DocumentAnalysis``.  :func:`score_text` is the
single-string entry point for that: ``score_text("vader", utterance)``
returns ``(score, None)`` or ``(None, reason)`` through the identical
scorer-factory path every aggregate finding in this module uses, so a
caller never has to special-case "is this engine available" logic of its
own.

Design choices worth stating up front
--------------------------------------
* **Sentence and paragraph units come from ``DocumentAnalysis``.**  No
  metric here re-splits sentences, re-finds dialogue or re-derives
  paragraphs; ``analysis.sentences``, ``analysis.dialogue``,
  ``analysis.narration`` and ``analysis.paragraph_sentence_counts`` are used
  exactly as the shared pipeline built them.
* **Mean and volatility are always separate findings.**  A document whose
  sentences alternate strongly positive/strongly negative averages near
  zero and is *not* neutral; ``..._level`` reports the former,
  ``..._volatility`` the mean absolute adjacent change, so the two can never
  be read as one number.  The corruption test in this module's test suite
  (``test_mean_and_volatility_are_independent``) asserts this on a small,
  exact fixture, not only "in general."
* **Category-set channels (NRC categories, Empath, LIWC) are whole-document,
  not per-sentence.**  Measured directly: ``nrclex.NRCLex()`` and
  ``empath.Empath().analyze()`` both pay a large *per-call* setup cost
  (rebuilding an inverted category index from their category table) that is
  independent of how much text is scored -- calling ``Empath().analyze()``
  once per sentence measured about 8.4s over 3,000 short sentences in this
  environment; calling it once over the concatenation of the same 3,000
  sentences measured about 0.03s for the identical token count.  A
  per-sentence *category distribution* trajectory would therefore cost
  hundreds of times more than the single whole-document call this task's
  category-distribution requirement actually needs, for no reader benefit:
  "how much of this document leans on the 'aggression' category" does not
  need a sentence-by-sentence reading the way sentiment polarity does.  The
  polarity/valence *trajectory* engines (VADER, NRC valence-balance, AFINN,
  TextBlob, a future VAD) remain genuinely per-sentence, which is where this
  task's volatility/arc/reversal/run measurements need them to be.
* **The neutral band is one tunable, applied to each engine's own raw
  value.**  ``neutral_band`` (default 0.05) marks a per-sentence score as
  "neutral" rather than weakly positive/negative.  Different engines have
  different natural scales (VADER and TextBlob polarity are bounded to
  [-1, 1]; AFINN here is normalized to a per-word rate; a future VAD engine
  might be centered differently), so the same absolute epsilon is not
  equally strict for every engine.  Rather than inventing a per-engine
  epsilon this module cannot validate, every polarity-share/reversal/run/
  disagreement finding's ``distribution`` records ``neutral_band`` (or
  ``neutral_band`` inside the ``evidence`` for the disagreement pairs) next
  to the concrete counts it produced, so the choice is checkable rather than
  hidden inside an unlabeled ratio.
* **Trajectory windows are capped regardless of book length.**  The arc
  finding is always a fixed three-way split (early/middle/late thirds), never
  a per-chapter or per-N-sentence rolling series; paragraph-level volatility
  is reported as one summary number (mean absolute paragraph-to-paragraph
  change), not a list of every paragraph's value.  Evidence lists (the arc
  finding's highest/lowest-scoring sentences, the NRC/Empath/LIWC top
  categories) are capped at a small fixed size with explicit sentence
  offsets, never a dump of every scored unit.
* **Disagreement is reported, never resolved.**  :func:`_disagreement_finding`
  computes sign-disagreement rate, Spearman rank correlation and a
  high-confidence disagreement rate (both engines in their own top quartile
  of |score|, opposite sign) for every pair of enabled *signed* engines, and
  keeps every pair's numbers in ``evidence`` rather than averaging the
  engines into one score.  Two engines disagreeing about one sentence's
  sentiment is exactly the kind of fact this project's cross-task rule 18
  ("treat library disagreement as data") asks to be kept, not smoothed away.

Measured cost
-------------
This suite is ``cost="moderate"``: it needs neither a spaCy parse nor
sentence-embeddings, so it is linear in sentence count, but it runs five
independent engines over every sentence rather than one, and each engine's
own library does real tokenization/tagging work per call -- this is
genuinely more expensive than a single-pass count-based metric, not an
inefficiency specific to this module (:func:`_textblob_sentiment`'s
docstring quantifies and removes the one clearly avoidable duplication,
sharing TextBlob's polarity/subjectivity computation instead of paying for
it twice). Measured directly (warm process -- these packages' own one-time
import cost, dominated by regex/bytecode compilation on first use, is
excluded, since a real grading run pays it once per process, not once per
document) on two Project Gutenberg novels, default config:

* *Alice's Adventures in Wonderland* (26,539 words, 1,459 sentences): about
  3.0s for every default channel except ``dialogue_narration_gap`` and
  ``speaker_profiles``; about 4.6s with those two added.
* *Tess of the d'Urbervilles* (152,575 words, 8,032 sentences): about 16s
  without those two channels; about 26s with them.

``dialogue_narration_gap`` and ``speaker_profiles`` are the two channels
worth turning off first for faster corpus-wide profiling of many long
books (see config.json's ``_requires_dialogue_narration_gap`` /
``_requires_speaker_profiles`` notes): both re-score already-scored text
(the dialogue/narration views, or every qualifying speaker's turns) with
every enabled engine, which is what the extra time above buys. Peak
resident memory measured well under 200 MB for both books with
``transformer_sentiment`` off (the default); enabling it adds a
transformer checkpoint's own memory footprint on top, the same as every
other transformer-backed channel elsewhere in this codebase.
"""

from __future__ import annotations

import functools
import importlib.metadata
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .. import text as textlib
from ..document import DocumentAnalysis
from ..optional import on_reset, require
from ..stats import autocorrelation, quantile, run_lengths, summarize
from .common import MODERATE, finding, option, unavailable
from .dialogue_speaker_style import identify_speakers

FAMILY = "discourse"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("vaderSentiment", "nrclex", "afinn", "textblob", "empath", "liwc",
                             "transformers", "torch")
MIN_SAMPLE = 20
CORRELATION_MIN_SAMPLE = 30
UNIT_SENSITIVE = False

PREFIX = "discourse.affect_"

DEFAULT_NEUTRAL_BAND = 0.05
DEFAULT_SPEAKER_MIN_TURNS = 8
DEFAULT_HIGH_CONFIDENCE_QUANTILE = 0.75
DEFAULT_EMPATH_TOP_CATEGORIES = 15
DEFAULT_TRANSFORMER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_TRANSFORMER_MAX_CHARS = 512

#: Every engine/measurement-family switch, mirrored in
#: ``textgrader.metrics.REGISTRY["affect_suite"].defaults["features"]`` and in
#: ``config.json``.  An engine registered later (see "The engine registry" in
#: the module docstring) that is NOT a key here still runs: :func:`_enabled`
#: falls back to that :class:`EngineSpec`'s own ``default_enabled``, so a
#: brand-new engine is never silently on or off by an accident of this dict.
DEFAULT_FEATURES: dict[str, bool] = {
    "vader": True,
    "nrc_valence": True,
    "afinn": True,
    "textblob_polarity": True,
    "textblob_subjectivity": True,
    # Off by default: loads a Hugging Face transformer checkpoint. This
    # suite's cost is "moderate", not "parse" or "model" (it needs neither a
    # spaCy parse nor sentence_transformers), so MetricSpec.needs_model does
    # NOT exclude affect_suite from corpus profiling -- see
    # MetricSpec.needs_model's own docstring in textgrader/metrics/__init__.py
    # ("only checks for sentence_transformers"). This flag defaulting to
    # False is therefore the only thing standing between a transformer model
    # and a book nobody asked to run one against, exactly the situation
    # logic_suite's module docstring describes for its own model flags.
    "transformer_sentiment": False,
    "nrc_categories": True,
    "empath_categories": True,
    # On by default but inert without liwc_dictionary_path configured (see
    # _liwc_category_finding): LIWC is commercially licensed and never
    # bundled, so leaving this on costs nothing until a user supplies their
    # own dictionary.
    "liwc_categories": True,
    "vad": True,
    "dialogue_narration_gap": True,
    "speaker_profiles": True,
    "sequence_correlations": True,
    "disagreement": True,
}


# ------------------------------------------------------------ the engine registry

@dataclass(frozen=True)
class EngineOutcome:
    """One engine's per-sentence scores for one document, with provenance."""

    values: tuple[float | None, ...]
    unit: str
    version: str
    resource: str
    #: Set when the engine could not be built at all (missing package,
    #: missing resource). ``values`` is then always ``()``.
    unavailable: str | None = None


@dataclass(frozen=True)
class EngineSpec:
    """A named per-sentence affect scorer, plus enough metadata to compare it honestly.

    ``signed`` marks an engine whose sign is meaningful (a polarity score or
    a VAD valence dimension): only signed engines get positive/negative/
    neutral-share, reversal/run and cross-engine disagreement findings. An
    unsigned engine (TextBlob subjectivity, a future VAD arousal/dominance)
    still gets level/volatility/arc/dialogue-gap/speaker-spread/correlation
    findings, because "how much the intensity moves around" stays meaningful
    even when "which way" does not.

    ``make_scorer(settings)`` returns ``(scorer, None)`` or ``(None,
    reason)``, matching :func:`textgrader.optional.require`'s convention.
    The returned ``scorer`` is a plain ``str -> float | None`` function,
    which is what lets the same engine score a whole document's sentences,
    one dialogue turn, or a single string handed in by another suite (see
    :func:`score_text`) through one code path.
    """

    name: str
    label: str
    unit: str
    signed: bool
    requires: tuple[str, ...] = ()
    default_enabled: bool = True
    #: Informational grouping only ("polarity", "subjectivity", "vad", "model").
    kind: str = "polarity"
    resource: str = ""
    version: Callable[[], str] = lambda: "unknown"
    make_scorer: Callable[[Mapping[str, Any]],
                          tuple[Callable[[str], float | None] | None, str | None]] | None = None


_ENGINES: dict[str, EngineSpec] = {}
_SCORER_CACHE: dict[tuple[str, str], tuple[Any, str | None]] = {}


def register_engine(spec: EngineSpec, *, replace: bool = False) -> None:
    """Add one per-sentence affect engine to the registry.

    See the module docstring's "The engine registry" section for the exact
    two-line-per-dimension call Task 11's orchestrator makes here for NRC VAD
    and Warriner VAD once ``textgrader/lexicons.py`` exists.  Raises
    ``ValueError`` on a name collision unless ``replace=True``, so two suites
    registering the same engine name by accident fail loudly rather than one
    silently shadowing the other.
    """

    if not replace and spec.name in _ENGINES:
        raise ValueError(f"affect engine {spec.name!r} is already registered; "
                         f"pass replace=True to override it")
    _ENGINES[spec.name] = spec


def list_engines() -> list[str]:
    """Every registered engine name, built-in or added via :func:`register_engine`."""

    return sorted(_ENGINES)


def _scorer_for(spec: EngineSpec, settings: Mapping[str, Any]) -> tuple[Any, str | None]:
    key = (spec.name, repr(sorted(settings.items())))
    if key in _SCORER_CACHE:
        return _SCORER_CACHE[key]
    try:
        outcome = spec.make_scorer(settings) if spec.make_scorer else (None, "no scorer factory")
    except Exception as exc:  # pragma: no cover - defensive: a third-party build failure
        outcome = (None, f"{type(exc).__name__}: {exc}")
    _SCORER_CACHE[key] = outcome
    return outcome


def score_text(engine_name: str, text: str,
               settings: Mapping[str, Any] | None = None) -> tuple[float | None, str | None]:
    """Score one arbitrary string with a registered engine.

    Independent of any :class:`DocumentAnalysis`: this is the per-sentence
    primitive every aggregate finding in this module is built from, exposed
    directly so another suite (Task 14's per-speaker conversation analysis)
    can score one utterance without importing this module's aggregation
    machinery or building a full document.  Returns ``(score, None)`` or
    ``(None, reason)``; never raises.
    """

    spec = _ENGINES.get(engine_name)
    if spec is None:
        return None, f"unknown affect engine {engine_name!r}; known engines are {list_engines()}"
    scorer, reason = _scorer_for(spec, dict(settings or {}))
    if scorer is None:
        return None, reason
    try:
        value = scorer(text)
    except Exception as exc:  # pragma: no cover - defensive: a third-party scoring failure
        return None, f"{type(exc).__name__}: {exc}"
    return (float(value) if value is not None and math.isfinite(value) else None), None


def _engine_values_for(analysis: DocumentAnalysis, spec: EngineSpec,
                       settings: Mapping[str, Any]) -> EngineOutcome:
    """Per-sentence scores of ``spec`` over ``analysis``, memoized on the document."""

    key = f"affect_suite:engine:{spec.name}:{sorted(settings.items())}"

    def build() -> EngineOutcome:
        scorer, reason = _scorer_for(spec, settings)
        if scorer is None:
            return EngineOutcome((), spec.unit, spec.version(), spec.resource, reason)
        values: list[float | None] = []
        for sentence in analysis.sentences:
            try:
                value = scorer(sentence)
            except Exception:  # pragma: no cover - defensive: one bad sentence, not the run
                value = None
            values.append(float(value) if value is not None and math.isfinite(value) else None)
        return EngineOutcome(tuple(values), spec.unit, spec.version(), spec.resource, None)

    return analysis.memo(key, build)


def _pkg_version(pkg: str) -> str:
    try:
        return importlib.metadata.version(pkg)
    except Exception:  # pragma: no cover - package not installed
        return "unknown"


# --------------------------------------------------------------- built-in engines

def _make_vader_scorer(settings: Mapping[str, Any]):
    module, reason = require("vaderSentiment")
    if module is None:
        return None, reason
    analyzer = module.SentimentIntensityAnalyzer()
    return (lambda text: float(analyzer.polarity_scores(text)["compound"])), None


def _make_nrc_valence_scorer(settings: Mapping[str, Any]):
    module, reason = require("nrclex")
    if module is None:
        return None, reason

    def score(text: str) -> float:
        tokens = [word.lower() for word in textlib.words(text)]
        lexicon = module.NRCLex()
        lexicon.load_token_list(tokens)
        frequencies = lexicon.affect_frequencies
        return float(frequencies.get("positive", 0.0) - frequencies.get("negative", 0.0))

    return score, None


def _make_afinn_scorer(settings: Mapping[str, Any]):
    module, reason = require("afinn")
    if module is None:
        return None, reason
    analyzer = module.Afinn(language=str(settings.get("afinn_language", "en")),
                            emoticons=bool(settings.get("afinn_emoticons", False)))

    def score(text: str) -> float | None:
        words = textlib.words(text)
        if not words:
            return None
        return float(analyzer.score(text)) / len(words)

    return score, None


@functools.lru_cache(maxsize=None)
def _textblob_sentiment(text: str) -> tuple[float, float] | None:
    """``(polarity, subjectivity)`` for one string, computed once and shared.

    ``textblob_polarity`` and ``textblob_subjectivity`` are two separate
    engines in the registry, each scoring every sentence in a document. Both
    read the same ``TextBlob(text).sentiment`` pair, so without this cache a
    document is tokenized and POS-tagged by TextBlob's PatternAnalyzer twice
    for every single sentence -- measured on Gutenberg's *Alice's Adventures
    in Wonderland* (about 1,500 sentences), TextBlob's own
    ``assessments``/``find_tokens`` calls alone accounted for roughly 1.2s of
    this suite's whole-document runtime before this cache, almost exactly
    double the cost one full pass needs. Because ``measure()`` scores one
    engine at a time over every sentence before moving to the next, the
    second engine's whole pass becomes a pure cache lookup. Unbounded on
    purpose (cleared per :func:`_reset_caches`, and never shared across
    processes): a book's sentences are rarely exact duplicates of each
    other, so the ceiling on useful entries is the document's own sentence
    count, already paid for once either way.
    """

    module, reason = require("textblob")
    if module is None:
        return None
    sentiment = module.TextBlob(text).sentiment
    return float(sentiment.polarity), float(sentiment.subjectivity)


def _make_textblob_scorer(attribute: str) -> Callable[[Mapping[str, Any]], Any]:
    index = 0 if attribute == "polarity" else 1

    def factory(settings: Mapping[str, Any]):
        module, reason = require("textblob")
        if module is None:
            return None, reason

        def score(text: str) -> float | None:
            result = _textblob_sentiment(text)
            return result[index] if result is not None else None

        return score, None
    return factory


_TRANSFORMER_PIPELINE_CACHE: dict[str, Any] = {}


def _transformer_pipeline(module: Any, model_name: str) -> Any:
    if model_name not in _TRANSFORMER_PIPELINE_CACHE:
        _TRANSFORMER_PIPELINE_CACHE[model_name] = module.pipeline("sentiment-analysis",
                                                                  model=model_name)
    return _TRANSFORMER_PIPELINE_CACHE[model_name]


def _make_transformer_scorer(settings: Mapping[str, Any]):
    module, reason = require("transformers")
    if module is None:
        return None, reason
    model_name = str(settings.get("transformer_model", DEFAULT_TRANSFORMER_MODEL))
    max_chars = max(16, int(settings.get("transformer_max_chars", DEFAULT_TRANSFORMER_MAX_CHARS)))
    try:
        pipeline = _transformer_pipeline(module, model_name)
    except Exception as exc:
        return None, (f"failed to load transformer sentiment model {model_name!r} "
                      f"({type(exc).__name__}: {exc})")

    def score(text: str) -> float | None:
        clipped = text[:max_chars].strip()
        if not clipped:
            return None
        result = pipeline(clipped, truncation=True)[0]
        sign = 1.0 if str(result.get("label", "")).upper().startswith("POS") else -1.0
        return sign * float(result.get("score", 0.0))

    return score, None


register_engine(EngineSpec(
    name="vader", label="VADER compound polarity", unit="compound score [-1,1]",
    signed=True, requires=("vaderSentiment",), kind="polarity",
    resource="vaderSentiment bundled lexicon (vader_lexicon.txt)",
    version=lambda: _pkg_version("vaderSentiment"), make_scorer=_make_vader_scorer))

register_engine(EngineSpec(
    name="nrc_valence", label="NRC EmoLex positive-minus-negative balance",
    unit="affect balance [-1,1]", signed=True, requires=("nrclex",), kind="polarity",
    resource="NRC Word-Emotion Association Lexicon (bundled nrc_en.json)",
    version=lambda: _pkg_version("NRCLex"), make_scorer=_make_nrc_valence_scorer))

register_engine(EngineSpec(
    name="afinn", label="AFINN wordlist sentiment", unit="AFINN score per word",
    signed=True, requires=("afinn",), kind="polarity", resource="AFINN-en-165 wordlist",
    version=lambda: _pkg_version("afinn"), make_scorer=_make_afinn_scorer))

register_engine(EngineSpec(
    name="textblob_polarity", label="TextBlob (Pattern) polarity", unit="polarity [-1,1]",
    signed=True, requires=("textblob",), kind="polarity",
    resource="TextBlob bundled PatternAnalyzer",
    version=lambda: _pkg_version("textblob"), make_scorer=_make_textblob_scorer("polarity")))

register_engine(EngineSpec(
    name="textblob_subjectivity", label="TextBlob (Pattern) subjectivity",
    unit="subjectivity [0,1]", signed=False, requires=("textblob",), kind="subjectivity",
    resource="TextBlob bundled PatternAnalyzer",
    version=lambda: _pkg_version("textblob"),
    make_scorer=_make_textblob_scorer("subjectivity")))

register_engine(EngineSpec(
    name="transformer_sentiment", label="transformer sentiment classifier",
    unit="signed confidence [-1,1]", signed=True, requires=("transformers", "torch"),
    default_enabled=False, kind="model",
    resource="Hugging Face sentiment-analysis pipeline (see settings.transformer_model)",
    version=lambda: _pkg_version("transformers"), make_scorer=_make_transformer_scorer))


def _reset_caches() -> None:
    _SCORER_CACHE.clear()
    _TRANSFORMER_PIPELINE_CACHE.clear()
    _LIWC_PARSER_CACHE.clear()
    _textblob_sentiment.cache_clear()
    global _EMPATH_INSTANCE
    _EMPATH_INSTANCE = None


on_reset(_reset_caches)


# ------------------------------------------------------------------- small helpers

def _features(config: Mapping[str, Any] | None) -> dict[str, Any]:
    configured = option(config, "features", {}) or {}
    return {**DEFAULT_FEATURES, **configured}


def _enabled(features: Mapping[str, Any], spec: EngineSpec) -> bool:
    return bool(features.get(spec.name, spec.default_enabled))


def _clean_pairs(values: Sequence[float | None]) -> list[tuple[int, float]]:
    return [(index, value) for index, value in enumerate(values) if value is not None]


def _adjacent_deltas(pairs: list[tuple[int, float]]) -> list[float]:
    return [later[1] - earlier[1] for earlier, later in zip(pairs, pairs[1:])]


def _sign_label(value: float, band: float) -> str:
    if value > band:
        return "positive"
    if value < -band:
        return "negative"
    return "neutral"


def _paragraph_means(values: Sequence[float | None],
                     paragraph_sentence_counts: Sequence[int]) -> list[float]:
    means: list[float] = []
    index = 0
    for count in paragraph_sentence_counts:
        chunk = [value for value in values[index:index + count] if value is not None]
        if chunk:
            means.append(statistics.fmean(chunk))
        index += count
    return means


def _thirds(pairs: list[tuple[int, float]]) -> tuple[list[float], list[float], list[float]]:
    n = len(pairs)
    a, b = n // 3, 2 * n // 3
    return ([v for _, v in pairs[:a]], [v for _, v in pairs[a:b]], [v for _, v in pairs[b:]])


def _linear_trend(pairs: list[tuple[int, float]]) -> tuple[float | None, float | None]:
    if len(pairs) < 3:
        return None, None
    xs = [float(index) for index, _ in pairs]
    ys = [value for _, value in pairs]
    try:
        regression = statistics.linear_regression(xs, ys)
        slope = regression.slope
    except statistics.StatisticsError:
        return None, None
    try:
        r = statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        r = None
    return slope, r


def _snippet(text: str, limit: int = 120) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit - 1] + "…"


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    try:
        return statistics.correlation(_rank(x), _rank(y))
    except statistics.StatisticsError:
        return None


#: A small closed stoplist for content-word rarity, independent of (and
#: smaller than) semantic_adjacent's or textgrader.sequences' -- only needs
#: to exclude the handful of words frequent enough to dominate a per-sentence
#: median rarity otherwise. A local copy rather than an import of
#: textgrader.sequences' private ``_RARITY_STOPWORDS``: that sequence's
#: values are NOT index-aligned with ``analysis.sentences`` (it silently
#: skips sentences with no qualifying content word), which this module's
#: per-sentence correlation needs to be, so the rarity series is recomputed
#: here with ``None`` kept in place of a skip instead.
_RARITY_STOPWORDS = frozenset("""
a an the and or but if of at by for with about against between into through
during before after above below to from up down in out on off over under is
am are was were be been being have has had having do does did doing would
could might must shall this that these those it he she they them his her
their as which who what
""".split())


def _sentence_rarity_aligned(analysis: DocumentAnalysis) -> tuple[list[float | None], str | None]:
    """Median content-word Zipf rarity per sentence, aligned 1:1 with ``analysis.sentences``.

    ``None`` for a sentence with no qualifying content word, rather than
    dropping it, which is what a correlation against another per-sentence
    series needs.
    """

    module, reason = require("wordfreq")
    if module is None:
        return [None] * len(analysis.sentences), reason
    values: list[float | None] = []
    for sentence in analysis.sentences:
        words = [word.lower() for word in textlib.words(sentence)
                if len(word) > 2 and word.lower() not in _RARITY_STOPWORDS]
        if not words:
            values.append(None)
            continue
        scores = [module.zipf_frequency(word, "en") for word in words]
        values.append(statistics.median(scores))
    return values, None


def _speaker_groups(analysis: DocumentAnalysis):
    return analysis.memo("affect_suite:speakers", lambda: identify_speakers(analysis))


# --------------------------------------------------------------- per-engine findings

def _unavailable_engine(outcome: EngineOutcome, metric_id: str, name: str) -> dict[str, Any] | None:
    if outcome.unavailable and not outcome.values:
        return unavailable(metric_id, name, outcome.unavailable, family=FAMILY)
    return None


def _level_finding(spec: EngineSpec, outcome: EngineOutcome) -> dict[str, Any]:
    metric_id, name = f"{PREFIX}{spec.name}_level", f"Sentiment/affect level, {spec.label}"
    guard = _unavailable_engine(outcome, metric_id, name)
    if guard:
        return guard
    values = [value for value in outcome.values if value is not None]
    summary = summarize(values)
    return finding(metric_id, name, summary.get("median"), spec.unit, family=FAMILY,
                   sample_size=summary.get("count"), min_sample=MIN_SAMPLE,
                   distribution={**summary, "engine_version": outcome.version,
                                "resource": outcome.resource},
                   warning=None if values else "no sentence produced a usable score")


def _polarity_share_finding(spec: EngineSpec, outcome: EngineOutcome, band: float) -> dict[str, Any]:
    metric_id = f"{PREFIX}{spec.name}_polarity_share"
    name = f"Positive/negative/neutral sentence share, {spec.label}"
    guard = _unavailable_engine(outcome, metric_id, name)
    if guard:
        return guard
    pairs = _clean_pairs(outcome.values)
    if not pairs:
        return unavailable(metric_id, name, "no sentence produced a usable score", family=FAMILY)
    labels = [_sign_label(value, band) for _, value in pairs]
    total = len(labels)
    positive = 100.0 * labels.count("positive") / total
    negative = 100.0 * labels.count("negative") / total
    neutral = 100.0 * labels.count("neutral") / total
    return finding(metric_id, name, positive, "percent", family=FAMILY, sample_size=total,
                   min_sample=MIN_SAMPLE,
                   distribution={"positive_share": positive, "negative_share": negative,
                                "neutral_share": neutral, "neutral_band": band})


def _volatility_finding(analysis: DocumentAnalysis, spec: EngineSpec,
                        outcome: EngineOutcome) -> dict[str, Any]:
    metric_id = f"{PREFIX}{spec.name}_volatility"
    name = f"Sentence-to-sentence volatility, {spec.label}"
    guard = _unavailable_engine(outcome, metric_id, name)
    if guard:
        return guard
    pairs = _clean_pairs(outcome.values)
    if len(pairs) < 2:
        return unavailable(metric_id, name, "fewer than two sentences produced a usable score",
                           family=FAMILY)
    deltas = _adjacent_deltas(pairs)
    mean_abs_delta = statistics.fmean(abs(delta) for delta in deltas)
    paragraph_means = _paragraph_means(outcome.values, analysis.paragraph_sentence_counts)
    paragraph_volatility = (
        statistics.fmean(abs(a - b) for a, b in zip(paragraph_means, paragraph_means[1:]))
        if len(paragraph_means) >= 2 else None)
    return finding(metric_id, name, mean_abs_delta, spec.unit, family=FAMILY,
                   sample_size=len(deltas), min_sample=MIN_SAMPLE,
                   distribution={"adjacent_delta": summarize(deltas),
                                "paragraph_volatility": paragraph_volatility,
                                "paragraph_count": len(paragraph_means),
                                "lag1_autocorrelation": autocorrelation([v for _, v in pairs], 1)})


def _reversal_runs_finding(spec: EngineSpec, outcome: EngineOutcome, band: float) -> dict[str, Any]:
    metric_id = f"{PREFIX}{spec.name}_reversal_runs"
    name = f"Emotional reversal rate and longest same-sign runs, {spec.label}"
    guard = _unavailable_engine(outcome, metric_id, name)
    if guard:
        return guard
    pairs = _clean_pairs(outcome.values)
    if len(pairs) < 2:
        return unavailable(metric_id, name, "fewer than two sentences produced a usable score",
                           family=FAMILY)
    labels = [_sign_label(value, band) for _, value in pairs]
    signed_labels = [label for label in labels if label != "neutral"]
    reversals = sum(1 for a, b in zip(signed_labels, signed_labels[1:]) if a != b)
    transitions = max(0, len(signed_labels) - 1)
    reversal_rate = 100.0 * reversals / transitions if transitions else None
    runs = run_lengths(labels)
    longest_positive = max(runs.get("positive", []) or [0])
    longest_negative = max(runs.get("negative", []) or [0])
    return finding(metric_id, name, reversal_rate, "reversals per 100 sign transitions",
                   family=FAMILY, sample_size=transitions, min_sample=MIN_SAMPLE,
                   distribution={"reversal_count": reversals, "sign_transitions": transitions,
                                "longest_positive_run_sentences": longest_positive,
                                "longest_negative_run_sentences": longest_negative,
                                "neutral_band": band},
                   warning=None if transitions else
                   "fewer than two non-neutral sentences; no sign transition to measure")


def _arc_finding(analysis: DocumentAnalysis, spec: EngineSpec, outcome: EngineOutcome) -> dict[str, Any]:
    metric_id = f"{PREFIX}{spec.name}_arc"
    name = f"Early/middle/late arc, {spec.label}"
    guard = _unavailable_engine(outcome, metric_id, name)
    if guard:
        return guard
    pairs = _clean_pairs(outcome.values)
    if len(pairs) < 3:
        return unavailable(metric_id, name, "fewer than three sentences produced a usable score",
                           family=FAMILY)
    early, middle, late = _thirds(pairs)
    early_mean = statistics.fmean(early) if early else None
    middle_mean = statistics.fmean(middle) if middle else None
    late_mean = statistics.fmean(late) if late else None
    value = (late_mean - early_mean) if early_mean is not None and late_mean is not None else None
    slope, r = _linear_trend(pairs)
    curvature = (middle_mean - (early_mean + late_mean) / 2.0
                if None not in (early_mean, middle_mean, late_mean) else None)
    ordered = sorted(pairs, key=lambda item: item[1])
    bounded = ordered[:3] + ordered[-3:] if len(ordered) > 6 else ordered
    evidence = [{"sentence_index": index, "value": val,
                "text": _snippet(analysis.sentences[index])}
               for index, val in bounded]
    return finding(metric_id, name, value, spec.unit, family=FAMILY, sample_size=len(pairs),
                   min_sample=MIN_SAMPLE,
                   distribution={"early_mean": early_mean, "middle_mean": middle_mean,
                                "late_mean": late_mean, "early_n": len(early),
                                "middle_n": len(middle), "late_n": len(late),
                                "linear_slope_per_sentence": slope, "linear_r": r,
                                "quadratic_curvature_proxy": curvature},
                   evidence=evidence)


def _dialogue_gap_finding(analysis: DocumentAnalysis, spec: EngineSpec,
                          settings: Mapping[str, Any]) -> dict[str, Any]:
    metric_id = f"{PREFIX}{spec.name}_dialogue_narration_gap"
    name = f"Dialogue-minus-narration gap, {spec.label}"
    dialogue_outcome = _engine_values_for(analysis.dialogue, spec, settings)
    narration_outcome = _engine_values_for(analysis.narration, spec, settings)
    dialogue_values = [value for value in dialogue_outcome.values if value is not None]
    narration_values = [value for value in narration_outcome.values if value is not None]
    if len(dialogue_values) < MIN_SAMPLE or len(narration_values) < MIN_SAMPLE:
        reason = dialogue_outcome.unavailable or narration_outcome.unavailable or (
            f"need at least {MIN_SAMPLE} scored sentences in both dialogue and narration; "
            f"found {len(dialogue_values)} and {len(narration_values)}")
        return unavailable(metric_id, name, reason, family=FAMILY)
    dialogue_mean = statistics.fmean(dialogue_values)
    narration_mean = statistics.fmean(narration_values)
    return finding(metric_id, name, dialogue_mean - narration_mean, spec.unit, family=FAMILY,
                   sample_size=min(len(dialogue_values), len(narration_values)),
                   min_sample=MIN_SAMPLE,
                   distribution={"dialogue_mean": dialogue_mean, "narration_mean": narration_mean,
                                "dialogue_n": len(dialogue_values),
                                "narration_n": len(narration_values)})


def _speaker_spread_finding(analysis: DocumentAnalysis, spec: EngineSpec, outcome: EngineOutcome,
                            settings: Mapping[str, Any], min_turns: int) -> dict[str, Any]:
    metric_id = f"{PREFIX}{spec.name}_speaker_spread"
    name = f"Spread of per-speaker mean affect, {spec.label}"
    guard = _unavailable_engine(outcome, metric_id, name)
    if guard:
        return guard
    groups, method = _speaker_groups(analysis)
    qualifying = {speaker: turns for speaker, turns in groups.items() if len(turns) >= min_turns}
    if len(qualifying) < 2:
        return unavailable(
            metric_id, name,
            f"found {len(groups)} named speaker(s) ({method} attribution), "
            f"{len(qualifying)} with at least {min_turns} turns; need at least two to say "
            f"whether characters' affect differs from each other", family=FAMILY)
    means: dict[str, float] = {}
    for speaker, turns in qualifying.items():
        scores = [score for turn in turns
                 if (score := score_text(spec.name, turn, settings)[0]) is not None]
        if scores:
            means[speaker] = statistics.fmean(scores)
    if len(means) < 2:
        return unavailable(metric_id, name,
                           "fewer than two speakers produced a usable score", family=FAMILY)
    spread = statistics.pstdev(list(means.values()))
    ranked = sorted(means.items(), key=lambda item: item[1])
    evidence = [{"speaker": speaker, "mean": value, "turns": len(qualifying[speaker])}
               for speaker, value in ranked[:10]]
    return finding(metric_id, name, spread, spec.unit, family=FAMILY, sample_size=len(means),
                   min_sample=2, distribution={"per_speaker_mean": means,
                                               "attribution_method": method},
                   evidence=evidence)


def _correlation_finding(analysis: DocumentAnalysis, spec: EngineSpec,
                         outcome: EngineOutcome) -> dict[str, Any]:
    metric_id = f"{PREFIX}{spec.name}_sequence_correlation"
    name = f"Correlation with sentence length and lexical rarity, {spec.label}"
    guard = _unavailable_engine(outcome, metric_id, name)
    if guard:
        return guard
    values = outcome.values
    lengths = [float(n) for n in analysis.sentence_lengths]
    rarity_values, rarity_reason = _sentence_rarity_aligned(analysis)

    length_pairs = [(v, l) for v, l in zip(values, lengths) if v is not None]
    rarity_pairs = [(v, r) for v, r in zip(values, rarity_values) if v is not None and r is not None]

    def _pearson(pairs: list[tuple[float, float]]) -> float | None:
        if len(pairs) < CORRELATION_MIN_SAMPLE:
            return None
        try:
            return statistics.correlation(*zip(*pairs))
        except statistics.StatisticsError:
            return None

    length_r, rarity_r = _pearson(length_pairs), _pearson(rarity_pairs)
    warning = None
    if length_r is None and rarity_r is None:
        warning = (f"fewer than {CORRELATION_MIN_SAMPLE} aligned sentences for a length "
                  f"correlation ({len(length_pairs)}) or a rarity correlation "
                  f"({len(rarity_pairs)}" + (f"; {rarity_reason}" if rarity_reason else "") + ")")
    return finding(metric_id, name, length_r, "pearson r", family=FAMILY,
                   sample_size=len(length_pairs), min_sample=CORRELATION_MIN_SAMPLE,
                   distribution={"length_r": length_r, "length_n": len(length_pairs),
                                "rarity_r": rarity_r, "rarity_n": len(rarity_pairs),
                                "rarity_unavailable_reason": rarity_reason},
                   warning=warning)


def _engine_findings(analysis: DocumentAnalysis, spec: EngineSpec, outcome: EngineOutcome,
                     band: float, features: Mapping[str, Any], settings: Mapping[str, Any],
                     min_turns: int) -> list[dict[str, Any]]:
    out = [_level_finding(spec, outcome), _volatility_finding(analysis, spec, outcome)]
    if spec.signed:
        out.append(_polarity_share_finding(spec, outcome, band))
        out.append(_reversal_runs_finding(spec, outcome, band))
    out.append(_arc_finding(analysis, spec, outcome))
    if features.get("dialogue_narration_gap", True):
        out.append(_dialogue_gap_finding(analysis, spec, settings))
    if features.get("speaker_profiles", True):
        out.append(_speaker_spread_finding(analysis, spec, outcome, settings, min_turns))
    if features.get("sequence_correlations", True):
        out.append(_correlation_finding(analysis, spec, outcome))
    return out


# ------------------------------------------------------------------ disagreement

def _disagreement(engine_outcomes: Mapping[str, EngineOutcome], band: float,
                  high_confidence_quantile: float) -> list[dict[str, Any]]:
    signed = [(name, outcome) for name, outcome in engine_outcomes.items()
             if _ENGINES[name].signed and outcome.values and not outcome.unavailable]
    report: list[dict[str, Any]] = []
    for i in range(len(signed)):
        for j in range(i + 1, len(signed)):
            a_name, a_outcome = signed[i]
            b_name, b_outcome = signed[j]
            n = min(len(a_outcome.values), len(b_outcome.values))
            aligned = [(a_outcome.values[k], b_outcome.values[k]) for k in range(n)
                      if a_outcome.values[k] is not None and b_outcome.values[k] is not None]
            if len(aligned) < MIN_SAMPLE:
                continue
            a_values = [x for x, _ in aligned]
            b_values = [y for _, y in aligned]
            sign_pairs = [(x, y) for x, y in aligned
                         if _sign_label(x, band) != "neutral" and _sign_label(y, band) != "neutral"]
            sign_disagreements = sum(1 for x, y in sign_pairs
                                     if _sign_label(x, band) != _sign_label(y, band))
            sign_rate = 100.0 * sign_disagreements / len(sign_pairs) if sign_pairs else None
            rank_correlation = _spearman(a_values, b_values)
            a_threshold = quantile(sorted(abs(v) for v in a_values), high_confidence_quantile)
            b_threshold = quantile(sorted(abs(v) for v in b_values), high_confidence_quantile)
            high_pairs = [(x, y) for x, y in aligned
                         if a_threshold is not None and b_threshold is not None
                         and abs(x) >= a_threshold and abs(y) >= b_threshold]
            high_disagreements = sum(1 for x, y in high_pairs if _sign_label(x, band) != _sign_label(y, band))
            high_rate = 100.0 * high_disagreements / len(high_pairs) if high_pairs else None
            report.append({
                "engine_a": a_name, "engine_b": b_name, "n": len(aligned),
                "sign_disagreement_rate": sign_rate, "sign_compared": len(sign_pairs),
                "rank_correlation": rank_correlation,
                "high_confidence_disagreement_rate": high_rate,
                "high_confidence_n": len(high_pairs), "neutral_band": band,
                "high_confidence_quantile": high_confidence_quantile,
            })
    return report


def _disagreement_finding(engine_outcomes: Mapping[str, EngineOutcome], band: float,
                          high_confidence_quantile: float) -> dict[str, Any]:
    metric_id, name = f"{PREFIX}disagreement", "Cross-engine sentiment disagreement"
    pairs_report = _disagreement(engine_outcomes, band, high_confidence_quantile)
    if not pairs_report:
        return unavailable(
            metric_id, name,
            f"fewer than two enabled signed engines with at least {MIN_SAMPLE} aligned scored "
            f"sentences to compare", family=FAMILY)
    rates = [pair["sign_disagreement_rate"] for pair in pairs_report
            if pair["sign_disagreement_rate"] is not None]
    mean_sign_rate = statistics.fmean(rates) if rates else None
    return finding(metric_id, name, mean_sign_rate, "percent", family=FAMILY,
                   sample_size=sum(pair["n"] for pair in pairs_report),
                   distribution={"pairs_compared": len(pairs_report)}, evidence=pairs_report)


# ------------------------------------------------------------- category channels

#: NRC's own eight basic (Plutchik) emotion categories; kept as NRC names it,
#: never remapped onto another ontology (see the module docstring's
#: "Map emotion lexicon categories without forcing ontologies to match" rule).
NRC_BASIC_CATEGORIES = ("anger", "anticipation", "disgust", "fear", "joy", "sadness",
                        "surprise", "trust")


def _category_entropy(shares: Mapping[str, float]) -> tuple[float | None, float | None]:
    total = sum(shares.values())
    if total <= 0:
        return None, None
    normalized = {key: value / total for key, value in shares.items() if value > 0}
    if not normalized:
        return None, None
    entropy = -sum(p * math.log2(p) for p in normalized.values())
    return entropy, 100.0 * max(normalized.values())


def _nrc_category_finding(analysis: DocumentAnalysis) -> dict[str, Any]:
    metric_id, name = f"{PREFIX}nrc_categories", "NRC EmoLex emotion-category distribution"
    module, reason = require("nrclex")
    if module is None:
        return unavailable(metric_id, name, reason, family=FAMILY)
    tokens = [word.lower() for word in analysis.words]
    if not tokens:
        return unavailable(metric_id, name, "no words in text", family=FAMILY)
    lexicon = module.NRCLex()
    lexicon.load_token_list(tokens)
    frequencies = lexicon.affect_frequencies
    shares = {category: frequencies.get(category, 0.0) for category in NRC_BASIC_CATEGORIES}
    entropy, top_share = _category_entropy(shares)
    evidence = [{"category": category, "share_of_words": share}
               for category, share in sorted(shares.items(), key=lambda item: -item[1])[:10]]
    return finding(metric_id, name, entropy, "bits", family=FAMILY, sample_size=len(tokens),
                   min_sample=MIN_SAMPLE,
                   distribution={"category_shares": shares,
                                "positive_share": frequencies.get("positive"),
                                "negative_share": frequencies.get("negative"),
                                "top_category_share_percent": top_share,
                                "possible_categories": len(NRC_BASIC_CATEGORIES),
                                "engine_version": _pkg_version("NRCLex"),
                                "resource": "NRC Word-Emotion Association Lexicon"},
                   evidence=evidence)


_EMPATH_INSTANCE: Any = None


def _empath_instance(module: Any) -> Any:
    global _EMPATH_INSTANCE
    if _EMPATH_INSTANCE is None:
        _EMPATH_INSTANCE = module.Empath()
    return _EMPATH_INSTANCE


def _empath_category_finding(analysis: DocumentAnalysis, top_k: int) -> dict[str, Any]:
    metric_id = f"{PREFIX}empath_categories"
    name = "Empath broad lexical/topical category distribution"
    module, reason = require("empath")
    if module is None:
        return unavailable(metric_id, name, reason, family=FAMILY)
    if not analysis.words:
        return unavailable(metric_id, name, "no words in text", family=FAMILY)
    lexicon = _empath_instance(module)
    # One call over the whole document, not per sentence: Empath's own
    # per-call cost (rebuilding an inverted category index) dominates over
    # token count -- see the module docstring's cost note.
    result = lexicon.analyze(analysis.text, normalize=True) or {}
    active = {category: share for category, share in result.items() if share > 0}
    entropy, top_share = _category_entropy(active)
    top_categories = sorted(active.items(), key=lambda item: -item[1])[:max(1, top_k)]
    return finding(metric_id, name, entropy, "bits", family=FAMILY, sample_size=len(analysis.words),
                   min_sample=MIN_SAMPLE,
                   distribution={"active_categories": len(active),
                                "possible_categories": len(lexicon.cats),
                                "top_category_share_percent": top_share,
                                "engine_version": _pkg_version("empath"),
                                "resource": "Empath bundled category lexicon (~200 categories)"},
                   evidence=[{"category": category, "share": share}
                            for category, share in top_categories])


_LIWC_PARSER_CACHE: dict[str, tuple[Any, list[str]]] = {}


def _liwc_parser(path: str) -> tuple[Any, list[str] | None, str | None]:
    if path in _LIWC_PARSER_CACHE:
        parse, categories = _LIWC_PARSER_CACHE[path]
        return parse, categories, None
    module, reason = require("liwc")
    if module is None:
        return None, None, reason
    try:
        parse, categories = module.load_token_parser(path)
    except Exception as exc:
        return None, None, f"failed to parse LIWC dictionary at {path!r} ({type(exc).__name__}: {exc})"
    _LIWC_PARSER_CACHE[path] = (parse, categories)
    return parse, categories, None


def _liwc_category_finding(analysis: DocumentAnalysis, path: str | None) -> dict[str, Any]:
    metric_id, name = f"{PREFIX}liwc_categories", "User-supplied LIWC category distribution"
    if not path:
        return unavailable(
            metric_id, name,
            "no liwc_dictionary_path configured; LIWC is commercially licensed and is never "
            "bundled -- point this option at your own licensed .dic file to enable this channel",
            family=FAMILY)
    parse, categories, reason = _liwc_parser(path)
    if parse is None:
        return unavailable(metric_id, name, reason, family=FAMILY)
    tokens = [word.lower() for word in analysis.words]
    if not tokens:
        return unavailable(metric_id, name, "no words in text", family=FAMILY)
    counts = Counter(category for token in tokens for category in parse(token))
    total = sum(counts.values())
    entropy, top_share = _category_entropy(dict(counts)) if total else (None, None)
    return finding(metric_id, name, entropy, "bits", family=FAMILY, sample_size=len(tokens),
                   min_sample=MIN_SAMPLE,
                   distribution={"category_counts": dict(counts.most_common(25)),
                                "possible_categories": len(categories or []),
                                "total_hits": total, "top_category_share_percent": top_share,
                                "dictionary_path": path,
                                "resource": "user-supplied LIWC .dic (not bundled)"},
                   evidence=[{"category": category, "count": count}
                            for category, count in counts.most_common(10)],
                   warning=None if total else "no token in the text matched the supplied dictionary")


# ------------------------------------------------------------------- VAD placeholders

#: The three dimensions NRC-VAD/Warriner-VAD norms provide; Task 11 registers
#: an engine named ``vad_<dimension>`` for each (see the module docstring).
VAD_DIMENSIONS = ("valence", "arousal", "dominance")


def _vad_placeholder_findings() -> list[dict[str, Any]]:
    out = []
    for dimension in VAD_DIMENSIONS:
        engine_name = f"vad_{dimension}"
        if engine_name in _ENGINES:
            continue  # a real engine is registered; the generic loop already covers it
        metric_id = f"{PREFIX}{engine_name}_level"
        out.append(unavailable(
            metric_id, f"NRC/Warriner VAD {dimension} trajectory",
            "no VAD engine registered; Task 11 owns loading NRC VAD / Warriner VAD norms in "
            "textgrader.lexicons -- once merged, its orchestrator calls "
            f"textgrader.metrics.affect_suite.register_engine(EngineSpec(name={engine_name!r}, "
            "...)) and this metric id starts reporting real values with no other change to this "
            "module; see register_engine's docstring", family=FAMILY))
    return out


# ------------------------------------------------------------------------- measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    features = _features(config)
    band = float(option(config, "neutral_band", DEFAULT_NEUTRAL_BAND))
    min_turns = max(1, int(option(config, "speaker_min_turns", DEFAULT_SPEAKER_MIN_TURNS)))
    high_confidence_quantile = float(option(config, "high_confidence_quantile",
                                            DEFAULT_HIGH_CONFIDENCE_QUANTILE))
    settings = {
        "afinn_language": option(config, "afinn_language", "en"),
        "afinn_emoticons": option(config, "afinn_emoticons", False),
        "transformer_model": option(config, "transformer_model", DEFAULT_TRANSFORMER_MODEL),
        "transformer_max_chars": option(config, "transformer_max_chars",
                                       DEFAULT_TRANSFORMER_MAX_CHARS),
    }

    out: list[dict[str, Any]] = []
    engine_outcomes: dict[str, EngineOutcome] = {}
    for engine_name in list_engines():
        spec = _ENGINES[engine_name]
        if not _enabled(features, spec):
            continue
        outcome = _engine_values_for(analysis, spec, settings)
        engine_outcomes[engine_name] = outcome
        out.extend(_engine_findings(analysis, spec, outcome, band, features, settings, min_turns))

    if features.get("disagreement", True):
        out.append(_disagreement_finding(engine_outcomes, band, high_confidence_quantile))
    if features.get("nrc_categories", True):
        out.append(_nrc_category_finding(analysis))
    if features.get("empath_categories", True):
        out.append(_empath_category_finding(
            analysis, int(option(config, "empath_top_categories", DEFAULT_EMPATH_TOP_CATEGORIES))))
    if features.get("liwc_categories", True):
        out.append(_liwc_category_finding(analysis, option(config, "liwc_dictionary_path", None)))
    if features.get("vad", True):
        out.extend(_vad_placeholder_findings())

    return out
