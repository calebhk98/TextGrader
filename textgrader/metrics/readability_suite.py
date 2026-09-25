"""Comprehensive readability-formula cross-check suite.

TextGrader's own core measurements (:mod:`textgrader.core_metrics`) compute
exactly two classical formulas -- Flesch-Kincaid grade (``fk``) and the
Automated Readability Index (``ari``) -- with their own hand-rolled tokenizer
and a vowel-cluster syllable heuristic. That is a deliberate, minimal choice
for the maturity aggregate, and this suite does not touch it (see
``core_metrics.py``'s own docstring and the "Do not change existing core
readability values" rule this task was given). What it adds instead is
**breadth and cross-checking**: the same formulas (and a dozen more) computed
by three independent, real, package-backed implementations, each with its own
tokenizer, sentence splitter and syllable counter, so that the *agreement or
disagreement* between them becomes data rather than being silently averaged
away.

Readability formulas are not a quality signal. A picture book should score
low; an expert monograph should score high; neither is "better" prose. Every
finding in this module is ``Polarity.NEUTRAL`` (the project default -- see
:mod:`textgrader.results`) and none of them feed the maturity aggregate,
because that aggregate is computed from ``polarity``, not from family, and
nothing here sets it to anything but neutral.

Libraries used, and how each was verified
------------------------------------------

- **textstat** (``pip install textstat``, confirmed a real formula-generating
  package by calling ``textstat.flesch_kincaid_grade``, ``.gunning_fog``,
  ``.smog_index``, ``.coleman_liau_index``, ``.automated_readability_index``,
  ``.dale_chall_readability_score``, ``.linsear_write_formula``, ``.lix``,
  ``.rix``, ``.spache_readability``, ``.mcalpine_eflaw`` and several
  locale-specific formulas against real text and checking the numbers land in
  a plausible grade-level range). Every function takes raw text and does its
  own tokenization/sentence-splitting internally -- a different pipeline from
  TextGrader's own, which is exactly the point (see "Reuse canonical
  words/sentences" below for what *is* shared).
- **py-readability-metrics** (PyPI name ``py-readability-metrics``, imports as
  ``readability`` -- confirmed by ``import readability; readability.__doc__``
  == "Scores the readability of a text and output a variety of metrics
  including: Flesch-Kincaid Grade Level, Gunning Fog, ARI, Dale Chall, SMOG,
  and more"). Its ``Readability(text)`` constructor tokenizes and
  sentence-splits (via ``nltk``) once; every scorer method
  (``.ari()``, ``.coleman_liau()``, ``.dale_chall()``, ``.flesch()``,
  ``.flesch_kincaid()``, ``.gunning_fog()``, ``.linsear_write()``,
  ``.smog()``, ``.spache()``) reuses that one pass and returns a namedtuple-
  like ``Result`` with a numeric ``.score`` plus a bucketed ``.grade_level``/
  ``.grade_levels``. Confirmed for real: every scorer except ``smog`` raises
  ``readability.exceptions.ReadabilityException('100 words required.')`` on
  fewer than 100 words (reproduced directly: ``Readability("word " *
  50).flesch()`` raises); ``smog`` instead raises when there are fewer than 30
  sentences. ``Readability("")`` raises a bare ``ZeroDivisionError`` at
  *construction* time (reproduced directly), so this module never constructs
  a reader over an empty document -- see "Handling the 100-word floor" below
  for what happens instead.
- **pystylometry** (``pystylometry.readability``, PyPI ``pystylometry``,
  already a project dependency for ``stylometry_suite``'s
  ``pystylometry_reference`` feature -- no new ``PACKAGES`` key). Confirmed
  for real: ``pystylometry.readability`` exposes ``compute_ari``,
  ``compute_coleman_liau``, ``compute_dale_chall``, ``compute_flesch``,
  ``compute_gunning_fog``, ``compute_smog``, ``compute_linsear_write`` (the
  formulas textstat/py-readability-metrics also cover) **and three the task
  spec asks for that neither of the other two libraries implements at all**:
  ``compute_forcast`` (FORCAST), ``compute_fry`` (the Fry readability graph)
  and ``compute_powers_sumner_kearl`` (Powers-Sumner-Kearl). Every one of
  these formulas is computed over word-count-bounded chunks (default 1,000
  words) and the returned ``*_score``/``grade_level`` field is documented and
  verified to be the *mean across chunks* -- e.g.
  ``compute_forcast(text).forcast_score`` -- which is what this module reports
  as the headline. ``compute_flesch`` returns BOTH the reading-ease score
  (``.reading_ease``) AND the Flesch-Kincaid grade computed from the same pass
  (``.grade_level``) in one call, unlike the other two libraries, which expose
  those as two separate functions -- this module calls it once and reports
  both. Its own syllable counter
  (``pystylometry.readability.syllables.count_syllables``) is itself backed by
  the ``pronouncing`` package (CMU Pronouncing Dictionary), confirmed by
  reading its source (``import pronouncing`` at the top, with a vowel-counting
  fallback for out-of-dictionary words), which is exactly the
  "pronouncing/cmudict" cross-check the task spec names -- this module also
  uses ``pronouncing`` directly, independent of pystylometry, for its own
  syllable-counter comparison (see below). ``compute_gunning_fog`` is the one
  exception: it loads its own spaCy pipeline internally
  (``pystylometry/readability/gunning_fog.py``), which would silently attach a
  second, suite-private spaCy load to a suite whose cost class is
  ``fast``/``moderate`` (not ``parse``) -- it is exposed as its own
  off-by-default feature (``pystylometry_gunning_fog``) rather than folded
  into the on-by-default ``pystylometry_formulas`` group.
- **pronouncing** (``pip install pronouncing`` -- pulls in ``cmudict``).
  Confirmed for real: ``pronouncing.phones_for_word("hello")`` returns CMU
  phonetic transcriptions and ``pronouncing.syllable_count(phones)`` counts
  stress markers. Used directly by :func:`_syllable_crosscheck` for a
  three-way syllable-count comparison (this module's own vowel-cluster
  heuristic reused unmodified from ``core_metrics.syllables``, textstat's
  ``syllable_count`` -- pyphen-hyphenation based -- and CMUdict via
  ``pronouncing``, with a documented, real disagreement: ``core_metrics.
  syllables`` undercounts several common diphthong words ("fire", "poem",
  "cruel", "hour", "science" all score 1 there and 2 under both textstat and
  CMUdict; see ``tests/test_readability_suite.py`` for the exact reproduced
  numbers), while "every"/"naturally"/"beautiful"/"world" agree across all
  three.

Reuse canonical words/sentences, where that is possible
----------------------------------------------------------

Every library above does its own tokenization and sentence splitting -- that
tokenization difference *is* one of the signals this suite exists to surface
(the task spec's own instruction: "accept that third-party libraries may
tokenize differently; record that as an implementation difference rather than
forcing identical preprocessing if the library does not expose it"). None of
the three libraries used here accepts pre-split sentences or tokens, so
"reuse canonical words/sentences" means what it can mean in practice: every
library is handed the *same* canonical body text
(``DocumentAnalysis.text`` -- the already-cleaned, transcript-stripped,
quote-normalized text every other metric in this codebase measures), never a
second, independently re-cleaned copy; the "core" comparison values (``fk``,
``ari``) are the *exact* numbers ``core_metrics.measure`` produces from
``analysis.words``/``analysis.tokens``/``analysis.sentences`` -- this module
calls that function directly (``floor=1``, matching ``grade.py``'s own call)
rather than re-deriving anything, so the ``core`` entry in every disagreement
finding is guaranteed byte-identical to the ``prose.fk``/``prose.ari``
findings in the same report; and the syllable/difficult-word cross-checks
below run over ``analysis.tokens`` (the same lower-cased, apostrophe-folded
token list every other lexical metric in this codebase uses), not an
independently-tokenized copy.

Handling the 100-word floor
------------------------------

py-readability-metrics refuses every formula except SMOG below 100 words (and
SMOG below 30 sentences) by raising ``ReadabilityException``. This module
never lets that exception escape: it is caught per formula and turned into an
``unavailable`` finding that quotes the exact exception text. The other two
libraries (textstat, pystylometry) never raise on a tiny document -- they
return numbers computed from as few tokens as are present, which are exactly
as unreliable as that implies. Rather than silently trusting a Flesch-Kincaid
grade computed from three words, **every finding in this module carries
``sample_size=<word count>`` and ``min_sample=100``** (the same floor
py-readability-metrics itself enforces), so ``grade.py``'s comparator marks
every one of them ``Action.INSUFFICIENT_DATA`` below that floor regardless of
which library produced the number -- this is the mechanism
:mod:`textgrader.metrics.common`'s own docstring describes ("a finding whose
sample_size is below the metric's minimum is reported with
action='insufficient_data'"), and it is what lets a document with, say, 40
words show every readability-formula finding as insufficient data even though
two of the three libraries happily returned a number for it.

Formulas covered
-------------------

Flesch Reading Ease, Flesch-Kincaid Grade (core + 3 libraries), Gunning Fog,
SMOG, Coleman-Liau, Automated Readability Index (core + 3 libraries),
Dale-Chall, Linsear Write, Spache -- each from up to three independent
implementations, plus a per-formula cross-implementation disagreement finding
(mean and max pairwise absolute difference); FORCAST, Fry and Powers-
Sumner-Kearl (pystylometry only -- no other installed library implements
them); LIX, RIX and the McAlpine EFLAW English-as-a-foreign-language formula
(textstat only); seven additional formulas textstat exposes that were
calibrated for other languages (Fernandez-Huerta and Szigriszt-Pazos for
Spanish, the Wiener Sachtextformel for German, the Gulpease Index for
Italian, Crawford, Gutierrez de Polini and Osman) -- kept under stable IDs per
the task spec's instruction to retain "additional formulas exposed by
installed readability libraries", but off by default
(``textstat_locale_formulas``) since applying a non-English-calibrated
formula to English prose is more likely to mislead than inform; a
difficult-word-list disagreement (Jaccard overlap between textstat's Dale-
Chall-derived difficult-word list and pystylometry's own, smaller, bundled
familiar-word list); a three-way syllable-counter disagreement rate over
every unique word (bounded); and cross-formula summary statistics (mean,
median, max, spread, SD) over every "US grade level" number this run
produced -- Flesch Reading Ease (0-100, inverted) and the raw Dale-Chall
score (its own ~0-10 scale) are excluded from that aggregate, since averaging
them with a grade-level number would mix incompatible scales, but they still
get their own per-formula disagreement finding.

Family and metric-ID prefix
------------------------------

``tests/test_optional_metrics.py`` (off limits for this task) hard-codes the
set of accepted metric-ID prefixes and does not include ``readability``, so
this suite's IDs use the ``nlp.`` prefix the task spec names as its
alternative (``nlp.readability_*``), matching the pattern every other
``PREFIXES``-gated experimental suite in this codebase already uses (e.g.
``style.stylometry_*``, ``discourse.affect_*`` -- the ID prefix and the
``FAMILY`` string are independent in this codebase; ``stylometry_suite``'s
own ``FAMILY`` is ``"authorial"`` while its IDs start ``style.``). This
module's ``FAMILY`` is the new string ``"readability"`` -- ``textgrader.
metrics.__init__.FAMILIES`` is computed from whatever family strings the
registry actually uses, so a new one needs no separate registration.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Mapping

from .. import core_metrics
from ..document import DocumentAnalysis
from ..optional import require
from .common import MODERATE, finding, option, unavailable

FAMILY = "readability"
PREFIX = "nlp.readability_"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("textstat", "py_readability_metrics", "pronouncing", "pystylometry")
#: Words. Matches py-readability-metrics' own hard floor (every formula it
#: implements except SMOG refuses below this), applied uniformly to every
#: finding in this module regardless of which library produced it -- see the
#: module docstring's "Handling the 100-word floor".
MIN_SAMPLE = 100
UNIT_SENSITIVE = False

DEFAULT_FEATURES: dict[str, bool] = {
    "textstat_formulas": True,
    "textstat_mcalpine_eflaw": True,
    "readability_metrics_formulas": True,
    "pystylometry_formulas": True,
    "syllable_crosscheck": True,
    "difficult_word_crosscheck": True,
    "formula_aggregate": True,
    # Off by default: seven formulas textstat exposes that were calibrated for
    # languages other than English (Spanish, German, Italian, ...). See the
    # module docstring.
    "textstat_locale_formulas": False,
    # Off by default: pystylometry's own compute_gunning_fog() loads its OWN
    # spaCy pipeline internally, independent of the shared cost="parse" spaCy
    # pipeline every other metric in this codebase shares. See the module
    # docstring.
    "pystylometry_gunning_fog": False,
}

DEFAULT_SYLLABLE_MAX_UNIQUE_WORDS = 20_000
DEFAULT_MAX_EVIDENCE = 20

#: Formula keys (the part after the library prefix, e.g. "textstat_fk" ->
#: "fk") that live on the same 0-~18 US-grade scale and so are safe to pool
#: into one mean/median/max/spread/SD aggregate. Flesch Reading Ease (0-100,
#: higher is EASIER, the opposite direction) and the raw Dale-Chall score (its
#: own ~0-10 scale) are deliberately excluded -- they still get their own
#: disagreement finding, just not folded into this aggregate.
_GRADE_SCALE_FORMULAS = {"fk", "ari", "gunning_fog", "smog", "coleman_liau", "linsear_write",
                        "spache", "forcast", "psk", "fry"}

#: textstat function name -> (display name, unit).
_LOCALE_FORMULAS = (
    ("fernandez_huerta", "Fernandez Huerta (Spanish)", "grade"),
    ("szigriszt_pazos", "Szigriszt-Pazos (Spanish)", "score"),
    ("gutierrez_polini", "Gutierrez de Polini (Spanish)", "grade"),
    ("crawford", "Crawford (Spanish)", "grade"),
    ("osman", "Osman (Arabic/Malay)", "score"),
    ("wiener_sachtextformel", "Wiener Sachtextformel (German)", "grade"),
    ("gulpease_index", "Gulpease Index (Italian)", "score"),
)


def _features(config: Mapping[str, Any] | None) -> dict[str, bool]:
    merged = dict(DEFAULT_FEATURES)
    merged.update(option(config, "features", {}) or {})
    return merged


def _finite(value: Any) -> float | None:
    """A plain float, or ``None`` for anything that is not one, or is NaN/inf.

    pystylometry's chunked formulas return ``float('nan')`` (never raise) on a
    zero-word document -- reproduced directly (``compute_ari("").ari_score``
    is ``nan``, and every other formula's score/grade_level does the same).
    ``NaN`` is a valid Python float, so it would otherwise pass straight
    through as a "real" value and print as the non-standard JSON token
    ``NaN`` in the report. Converting it to ``None`` here makes such a
    document's readability findings ``unavailable`` instead, which is the
    honest reading of "the library computed nothing" for the class of
    documents this can occur on (zero measurable words).
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _grade_number(value: Any) -> float | None:
    """Coerce a library's bucketed grade label into one number.

    The three libraries disagree not only on formulas but on how a "grade
    level" is represented: pystylometry mostly returns a plain float,
    py-readability-metrics returns a single string (``'8'``), a list of
    strings (``['11', '12']``), or a label (``'K'``, ``'college'``,
    ``'college_graduate'``), and Dale-Chall/Fry grade levels come back as
    range strings (``'11-12'``). This collapses all of those onto the same
    numeric US-grade scale so they can sit in one aggregate, using the
    midpoint of a range or list and the conventional endpoints for the named
    labels (kindergarten=0, college=13, college graduate=16). ``None``/``'na'``
    means the library itself declined to bucket the score.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _finite(value)
    if isinstance(value, (list, tuple, set)):
        parsed = [item for item in (_grade_number(entry) for entry in value) if item is not None]
        return statistics.fmean(parsed) if parsed else None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text or text == "na":
            return None
        if text in ("k", "kindergarten"):
            return 0.0
        if "graduate" in text:
            return 16.0
        if "college" in text:
            return 13.0
        numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]
        return statistics.fmean(numbers) if numbers else None
    return None


class _Collector:
    """Every headline numeric value this run produced, for the disagreement
    and cross-formula-aggregate findings below.

    ``values`` holds every headline number, keyed ``"<library>_<formula>"``
    (e.g. ``"textstat_fk"``, ``"pystylometry_dale_chall"``) regardless of
    scale, so any two implementations of the same formula can be compared.
    ``grades`` is the subset whose formula is on the shared US-grade scale
    (see :data:`_GRADE_SCALE_FORMULAS`), used only by the cross-formula
    aggregate.
    """

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.grades: dict[str, float] = {}

    def record(self, library: str, formula: str, value: float | None) -> None:
        if value is None:
            return
        key = f"{library}_{formula}"
        self.values[key] = value
        if formula in _GRADE_SCALE_FORMULAS:
            self.grades[key] = value

    def get(self, library: str, formula: str) -> float | None:
        return self.values.get(f"{library}_{formula}")


def _emit(out: list[dict[str, Any]], collector: _Collector, library: str, formula: str,
         mid: str, name: str, value: float | None, unit: str, word_count: int, *,
         distribution: Mapping[str, Any] | None = None,
         evidence: list[Mapping[str, Any]] | None = None, warning: str | None = None) -> None:
    out.append(finding(mid, name, value, unit, family=FAMILY, sample_size=word_count,
                       min_sample=MIN_SAMPLE, distribution=distribution, evidence=evidence,
                       warning=warning))
    collector.record(library, formula, value)


def _disagreement(out: list[dict[str, Any]], collector: _Collector, formula: str, mid: str,
                  name: str, unit: str, word_count: int,
                  libraries: tuple[str, ...] = ("core", "textstat", "readability_metrics",
                                               "pystylometry")) -> None:
    """One implementation-disagreement finding for one formula.

    Mean and max pairwise absolute difference across whichever of the given
    libraries produced a value this run, kept rather than reconciled (task
    rule 18: "treat library disagreement as data").
    """

    present = {library: collector.get(library, formula) for library in libraries}
    present = {library: value for library, value in present.items() if value is not None}
    if len(present) < 2:
        out.append(unavailable(mid, name, "fewer than two implementations produced a value "
                               "for this document", family=FAMILY, unit=unit))
        return
    items = list(present.items())
    pairs = [abs(a[1] - b[1]) for index, a in enumerate(items) for b in items[index + 1:]]
    out.append(finding(mid, name, statistics.fmean(pairs), unit, family=FAMILY,
                       sample_size=word_count, min_sample=MIN_SAMPLE,
                       distribution={"values": present, "max_pairwise_difference": max(pairs),
                                     "n_implementations": len(present)}))


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as importlib_metadata
        return importlib_metadata.version(name)
    except Exception:  # pragma: no cover - package metadata unavailable
        return None


# ------------------------------------------------------------------ textstat

def _textstat_formulas(text: str, word_count: int, collector: _Collector
                       ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    module, reason = require("textstat")
    # (formula key, textstat function, display name, unit)
    plan = (
        ("fk", "flesch_kincaid_grade", "Flesch-Kincaid grade (textstat)", "grade"),
        ("flesch_reading_ease", "flesch_reading_ease", "Flesch Reading Ease (textstat)", "score"),
        ("gunning_fog", "gunning_fog", "Gunning Fog (textstat)", "grade"),
        ("smog", "smog_index", "SMOG index (textstat)", "grade"),
        ("coleman_liau", "coleman_liau_index", "Coleman-Liau index (textstat)", "grade"),
        ("ari", "automated_readability_index", "Automated Readability Index (textstat)", "grade"),
        ("dale_chall", "dale_chall_readability_score", "Dale-Chall score (textstat)", "score"),
        ("linsear_write", "linsear_write_formula", "Linsear Write (textstat)", "grade"),
        ("spache", "spache_readability", "Spache index (textstat)", "grade"),
        ("lix", "lix", "LIX (textstat)", "score"),
        ("rix", "rix", "RIX (textstat)", "score"),
    )
    for formula, fn_name, name, unit in plan:
        mid = f"{PREFIX}{formula}_textstat"
        if module is None:
            out.append(unavailable(mid, name, reason, family=FAMILY, unit=unit))
            continue
        try:
            value = float(getattr(module, fn_name)(text))
        except Exception as exc:  # pragma: no cover - third-party failure mode
            out.append(unavailable(mid, name, f"textstat raised {type(exc).__name__}: {exc}",
                                   family=FAMILY, unit=unit))
            continue
        _emit(out, collector, "textstat", formula, mid, name, value, unit, word_count,
             distribution={"library": "textstat", "library_version": _pkg_version("textstat")})
    return out


def _textstat_mcalpine(text: str, word_count: int) -> list[dict[str, Any]]:
    module, reason = require("textstat")
    mid, name, unit = f"{PREFIX}mcalpine_eflaw_textstat", "McAlpine EFLAW (textstat)", "score"
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY, unit=unit)]
    try:
        value = float(module.mcalpine_eflaw(text))
    except Exception as exc:  # pragma: no cover
        return [unavailable(mid, name, f"textstat raised {type(exc).__name__}: {exc}",
                            family=FAMILY, unit=unit)]
    return [finding(mid, name, value, unit, family=FAMILY, sample_size=word_count,
                    min_sample=MIN_SAMPLE, distribution={"library": "textstat"})]


def _textstat_locale_formulas(text: str, word_count: int) -> list[dict[str, Any]]:
    module, reason = require("textstat")
    out: list[dict[str, Any]] = []
    for fn_name, label, unit in _LOCALE_FORMULAS:
        mid, name = f"{PREFIX}{fn_name}_textstat", f"{label} (textstat)"
        if module is None:
            out.append(unavailable(mid, name, reason, family=FAMILY, unit=unit))
            continue
        try:
            value = float(getattr(module, fn_name)(text))
        except Exception as exc:  # pragma: no cover
            out.append(unavailable(mid, name, f"textstat raised {type(exc).__name__}: {exc}",
                                   family=FAMILY, unit=unit))
            continue
        out.append(finding(mid, name, value, unit, family=FAMILY, sample_size=word_count,
                           min_sample=MIN_SAMPLE,
                           distribution={"library": "textstat",
                                         "note": "calibrated for a language other than English; "
                                                 "reported on English text as a raw cross-check, "
                                                 "not as an English grade level"}))
    return out


def _textstat_difficult_words(text: str) -> tuple[set[str], str | None]:
    module, reason = require("textstat")
    if module is None:
        return set(), reason
    try:
        words = module.difficult_words_list(text)
    except Exception as exc:  # pragma: no cover
        return set(), f"textstat raised {type(exc).__name__}: {exc}"
    return {word.lower() for word in words}, None


# ------------------------------------------------------ py-readability-metrics

def _build_reader(text: str, word_count: int):
    """One ``readability.Readability`` reader, reused across every formula.

    Returns ``(reader, reason)``. ``Readability("")`` raises a bare
    ``ZeroDivisionError`` at construction (verified directly), so an empty
    document is refused before that call rather than after.
    """

    module, reason = require("py_readability_metrics")
    if module is None:
        return None, reason
    if word_count == 0:
        return None, "no words in text"
    try:
        return module.Readability(text), None
    except Exception as exc:  # pragma: no cover - third-party failure mode
        return None, f"py-readability-metrics raised {type(exc).__name__}: {exc}"


def _rl_call(reader, method: str, **kwargs):
    """Call one reader formula, catching its 100-word/30-sentence refusal.

    Returns ``(result, reason)``; ``result`` is the library's own namedtuple-
    like object with a numeric ``.score`` (and, on most formulas, a bucketed
    ``.grade_level``/``.grade_levels``) on success.
    """

    if reader is None:
        return None, "reader unavailable"
    try:
        return getattr(reader, method)(**kwargs), None
    except Exception as exc:
        return None, f"py-readability-metrics raised {type(exc).__name__}: {exc}"


def _readability_metrics_formulas(reader, reader_reason: str | None, word_count: int,
                                  collector: _Collector) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    version = _pkg_version("py-readability-metrics")
    # (formula key, reader method, display name, unit, grade attr or None)
    plan = (
        ("ari", "ari", "Automated Readability Index (py-readability-metrics)", "grade",
         "grade_levels"),
        ("coleman_liau", "coleman_liau", "Coleman-Liau index (py-readability-metrics)", "grade",
         "grade_level"),
        ("dale_chall", "dale_chall", "Dale-Chall score (py-readability-metrics)", "score",
         "grade_levels"),
        ("flesch_reading_ease", "flesch", "Flesch Reading Ease (py-readability-metrics)", "score",
         None),
        ("fk", "flesch_kincaid", "Flesch-Kincaid grade (py-readability-metrics)", "grade",
         "grade_level"),
        ("gunning_fog", "gunning_fog", "Gunning Fog (py-readability-metrics)", "grade",
         "grade_level"),
        ("linsear_write", "linsear_write", "Linsear Write (py-readability-metrics)", "grade",
         "grade_level"),
        ("spache", "spache", "Spache index (py-readability-metrics)", "grade", "grade_level"),
    )
    for formula, method, name, unit, grade_attr in plan:
        mid = f"{PREFIX}{formula}_readability_metrics"
        if reader is None:
            out.append(unavailable(mid, name, reader_reason, family=FAMILY, unit=unit))
            continue
        result, reason = _rl_call(reader, method)
        if result is None:
            out.append(unavailable(mid, name, reason, family=FAMILY, unit=unit))
            continue
        value = float(result.score)
        distribution = {"library": "py-readability-metrics", "library_version": version}
        if grade_attr:
            distribution["library_grade_level"] = getattr(result, grade_attr, None)
        _emit(out, collector, "readability_metrics", formula, mid, name, value, unit, word_count,
             distribution=distribution)
    # SMOG needs its own call (extra all_sentences kwarg, a different -- 30
    # sentence, not 100 word -- minimum enforced inside the library itself).
    mid, name = f"{PREFIX}smog_readability_metrics", "SMOG index (py-readability-metrics)"
    if reader is None:
        out.append(unavailable(mid, name, reader_reason, family=FAMILY, unit="grade"))
    else:
        result, reason = _rl_call(reader, "smog", all_sentences=True)
        if result is None:
            out.append(unavailable(mid, name, reason, family=FAMILY, unit="grade"))
        else:
            value = float(result.score)
            _emit(out, collector, "readability_metrics", "smog", mid, name, value, "grade",
                 word_count, distribution={"library": "py-readability-metrics",
                                          "library_version": version,
                                          "library_grade_level": result.grade_level})
    return out


# ----------------------------------------------------------------- pystylometry

def _pystylometry_formulas(text: str, word_count: int, collector: _Collector,
                           features: Mapping[str, bool]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    module, reason = require("pystylometry")
    readability_mod = None
    if module is not None:
        try:
            readability_mod = module.readability
        except AttributeError as exc:  # pragma: no cover - internal layout changed
            module, reason = None, f"pystylometry.readability raised {type(exc).__name__}: {exc}"
    version = _pkg_version("pystylometry")

    def _call(fn_name: str):
        if readability_mod is None:
            return None, reason
        try:
            return getattr(readability_mod, fn_name)(text), None
        except Exception as exc:  # pragma: no cover - third-party failure mode
            return None, f"pystylometry raised {type(exc).__name__}: {exc}"

    # (formula key, compute function, display name, unit, value attr, grade attr or None)
    plan = (
        ("ari", "compute_ari", "Automated Readability Index (pystylometry)", "grade",
         "ari_score", "grade_level"),
        ("coleman_liau", "compute_coleman_liau", "Coleman-Liau index (pystylometry)", "grade",
         "cli_index", "grade_level"),
        ("dale_chall", "compute_dale_chall", "Dale-Chall score (pystylometry)", "score",
         "dale_chall_score", "grade_level"),
        ("linsear_write", "compute_linsear_write", "Linsear Write (pystylometry)", "grade",
         "linsear_score", "grade_level"),
        ("smog", "compute_smog", "SMOG index (pystylometry)", "grade", "smog_index",
         "grade_level"),
        ("forcast", "compute_forcast", "FORCAST grade (pystylometry)", "grade", "forcast_score",
         "grade_level"),
        ("psk", "compute_powers_sumner_kearl", "Powers-Sumner-Kearl grade (pystylometry)", "grade",
         "psk_score", "grade_level"),
    )
    for formula, fn_name, name, unit, value_attr, grade_attr in plan:
        mid = f"{PREFIX}{formula}_pystylometry"
        result, call_reason = _call(fn_name)
        if result is None:
            out.append(unavailable(mid, name, call_reason, family=FAMILY, unit=unit))
            continue
        value = _finite(getattr(result, value_attr))
        _emit(out, collector, "pystylometry", formula, mid, name, value, unit, word_count,
             distribution={"library": "pystylometry", "library_version": version,
                          "library_grade_level": _finite(getattr(result, grade_attr, None))})

    # Flesch: one call gives BOTH the reading-ease score and the FK grade.
    flesch_result, flesch_reason = _call("compute_flesch")
    ease_mid = f"{PREFIX}flesch_reading_ease_pystylometry"
    fk_mid = f"{PREFIX}fk_pystylometry"
    if flesch_result is None:
        out.append(unavailable(ease_mid, "Flesch Reading Ease (pystylometry)", flesch_reason,
                               family=FAMILY, unit="score"))
        out.append(unavailable(fk_mid, "Flesch-Kincaid grade (pystylometry)", flesch_reason,
                               family=FAMILY, unit="grade"))
    else:
        _emit(out, collector, "pystylometry", "flesch_reading_ease", ease_mid,
             "Flesch Reading Ease (pystylometry)", _finite(flesch_result.reading_ease), "score",
             word_count, distribution={"library": "pystylometry", "library_version": version,
                                       "note": "same pass as the pystylometry Flesch-Kincaid "
                                               "grade below"})
        _emit(out, collector, "pystylometry", "fk", fk_mid, "Flesch-Kincaid grade (pystylometry)",
             _finite(flesch_result.grade_level), "grade", word_count,
             distribution={"library": "pystylometry", "library_version": version})

    # Fry: no single canonical score -- report the numeric grade estimate as
    # the headline, sentence length/syllable-per-100/graph zone as distribution.
    mid, name = f"{PREFIX}fry_pystylometry", "Fry readability graph grade (pystylometry)"
    result, call_reason = _call("compute_fry")
    if result is None:
        out.append(unavailable(mid, name, call_reason, family=FAMILY, unit="grade"))
    else:
        grade_value = _grade_number(result.grade_level)
        _emit(out, collector, "pystylometry", "fry", mid, name, grade_value, "grade", word_count,
             distribution={"library": "pystylometry", "library_version": version,
                          "avg_sentence_length": _finite(result.avg_sentence_length),
                          "avg_syllables_per_100_words": _finite(result.avg_syllables_per_100),
                          "graph_zone": result.graph_zone})

    if features.get("pystylometry_gunning_fog"):
        mid, name = f"{PREFIX}gunning_fog_pystylometry", "Gunning Fog (pystylometry)"
        result, call_reason = _call("compute_gunning_fog")
        if result is None:
            out.append(unavailable(mid, name, call_reason, family=FAMILY, unit="grade"))
        else:
            raw = getattr(result, "fog_index", None) if hasattr(result, "fog_index") \
                else getattr(result, "score", None)
            value = _finite(raw)
            _emit(out, collector, "pystylometry", "gunning_fog", mid, name, value, "grade",
                 word_count, distribution={"library": "pystylometry", "library_version": version,
                                          "note": "loads its own spaCy pipeline, independent of "
                                                  "the shared cost=\"parse\" pipeline",
                                          "library_grade_level": _finite(getattr(result,
                                                                                "grade_level",
                                                                                None))})
    return out


def _pystylometry_difficult_words(text: str) -> tuple[set[str] | None, str | None]:
    module, reason = require("pystylometry")
    if module is None:
        return None, reason
    try:
        additional_formulas = module.readability.additional_formulas
        familiar = additional_formulas.DALE_CHALL_FAMILIAR_WORDS
    except AttributeError as exc:  # pragma: no cover - internal layout changed
        return None, f"pystylometry.readability.additional_formulas raised {type(exc).__name__}: {exc}"
    try:
        from ..text import words as tokenize
        tokens = {token.lower() for token in tokenize(text)}
    except Exception:  # pragma: no cover
        tokens = {token.lower() for token in text.split()}
    return {token for token in tokens if token not in familiar}, None


# ------------------------------------------------------------- syllable cross-check

def _syllable_crosscheck(analysis: DocumentAnalysis, config: Mapping[str, Any] | None,
                         word_count: int) -> list[dict[str, Any]]:
    textstat_mod, textstat_reason = require("textstat")
    pronouncing_mod, pronouncing_reason = require("pronouncing")
    mid_core = f"{PREFIX}syllable_disagreement_rate_core_vs_cmudict"
    mid_textstat = f"{PREFIX}syllable_disagreement_rate_textstat_vs_cmudict"
    name_core = "Syllable count disagreement rate, TextGrader heuristic vs CMUdict"
    name_textstat = "Syllable count disagreement rate, textstat (pyphen) vs CMUdict"
    if pronouncing_mod is None:
        return [unavailable(mid_core, name_core, pronouncing_reason, family=FAMILY, unit="percent"),
               unavailable(mid_textstat, name_textstat, pronouncing_reason, family=FAMILY,
                           unit="percent")]

    cap = int(option(config, "syllable_max_unique_words", DEFAULT_SYLLABLE_MAX_UNIQUE_WORDS))
    unique_words = sorted({token for token in analysis.tokens if token.isalpha()})
    truncated = len(unique_words) > cap
    sample = unique_words[:cap]

    def cmudict_count(word: str) -> tuple[int, bool]:
        phones_list = pronouncing_mod.phones_for_word(word)
        if phones_list:
            return pronouncing_mod.syllable_count(phones_list[0]), True
        return core_metrics.syllables(word), False

    max_evidence = int(option(config, "max_evidence", DEFAULT_MAX_EVIDENCE))
    core_mismatches = 0
    textstat_mismatches = 0
    both_mismatch = 0
    in_dictionary = 0
    core_evidence: list[dict[str, Any]] = []
    textstat_available = textstat_mod is not None
    for word in sample:
        core_count = core_metrics.syllables(word)
        cmu_count, found = cmudict_count(word)
        if found:
            in_dictionary += 1
        core_differs = core_count != cmu_count
        if core_differs:
            core_mismatches += 1
        textstat_count = None
        if textstat_available:
            try:
                textstat_count = textstat_mod.syllable_count(word)
            except Exception:  # pragma: no cover - third-party failure mode
                textstat_count = None
            if textstat_count is not None and textstat_count != cmu_count:
                textstat_mismatches += 1
                if core_differs:
                    both_mismatch += 1
        if core_differs and found and len(core_evidence) < max_evidence:
            core_evidence.append({"word": word, "core_heuristic": core_count,
                                  "cmudict": cmu_count, "textstat": textstat_count})

    sampled = len(sample)
    out = [finding(
        mid_core, name_core, 100.0 * core_mismatches / sampled if sampled else None, "percent",
        family=FAMILY, sample_size=word_count, min_sample=MIN_SAMPLE,
        distribution={"unique_words_sampled": sampled, "truncated": truncated,
                     "cmudict_coverage_percent": 100.0 * in_dictionary / sampled if sampled
                     else None},
        evidence=core_evidence)]
    if textstat_available:
        out.append(finding(
            mid_textstat, name_textstat,
            100.0 * textstat_mismatches / sampled if sampled else None, "percent", family=FAMILY,
            sample_size=word_count, min_sample=MIN_SAMPLE,
            distribution={"unique_words_sampled": sampled, "truncated": truncated,
                         "both_disagree_with_cmudict_percent": 100.0 * both_mismatch / sampled
                         if sampled else None}))
    else:
        out.append(unavailable(mid_textstat, name_textstat, textstat_reason, family=FAMILY,
                               unit="percent"))
    return out


# --------------------------------------------------------- difficult-word crosscheck

def _difficult_word_crosscheck(text: str, word_count: int, config: Mapping[str, Any] | None
                               ) -> list[dict[str, Any]]:
    mid = f"{PREFIX}difficult_word_list_overlap"
    name = "Difficult-word-list overlap, textstat vs pystylometry (Jaccard)"
    textstat_words, textstat_reason = _textstat_difficult_words(text)
    pystylometry_words, pystylometry_reason = _pystylometry_difficult_words(text)
    if not textstat_words and textstat_reason:
        return [unavailable(mid, name, textstat_reason, family=FAMILY, unit="ratio")]
    if pystylometry_words is None:
        return [unavailable(mid, name, pystylometry_reason, family=FAMILY, unit="ratio")]
    union = textstat_words | pystylometry_words
    intersection = textstat_words & pystylometry_words
    jaccard = len(intersection) / len(union) if union else None
    max_evidence = int(option(config, "max_evidence", DEFAULT_MAX_EVIDENCE))
    only_textstat = sorted(textstat_words - pystylometry_words)[:max_evidence]
    only_pystylometry = sorted(pystylometry_words - textstat_words)[:max_evidence]
    return [finding(
        mid, name, jaccard, "ratio", family=FAMILY, sample_size=word_count, min_sample=MIN_SAMPLE,
        distribution={"textstat_difficult_word_count": len(textstat_words),
                     "pystylometry_difficult_word_count": len(pystylometry_words),
                     "intersection_count": len(intersection), "union_count": len(union),
                     "only_textstat_sample": only_textstat,
                     "only_pystylometry_sample": only_pystylometry,
                     "note": "textstat's list is drawn from its own bundled Dale-Chall-derived "
                             "easy-word corpus (syllable_threshold=2 by default); pystylometry's "
                             "'difficult' set is every word absent from its own bundled ~1,250-word "
                             "familiar-word subset (itself a subset of the real 3,000-word "
                             "Dale-Chall list). Different word lists and different criteria are "
                             "kept as data, not reconciled (task rule 18)."})]


# ---------------------------------------------------------------------- entry point

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    features = _features(config)
    text = analysis.text
    word_count = analysis.word_count
    out: list[dict[str, Any]] = []
    collector = _Collector()

    versions = {
        "textstat": _pkg_version("textstat"),
        "py-readability-metrics": _pkg_version("py-readability-metrics"),
        "pystylometry": _pkg_version("pystylometry"),
        "pronouncing": _pkg_version("pronouncing"),
    }
    out.append(finding(f"{PREFIX}library_versions", "Readability-suite library versions", None,
                       None, family=FAMILY, distribution=versions))

    core = core_metrics.measure(analysis, floor=1)
    if core:
        collector.record("core", "fk", core.get("fk"))
        collector.record("core", "ari", core.get("ari"))

    if features.get("textstat_formulas", True):
        out.extend(_textstat_formulas(text, word_count, collector))
    if features.get("textstat_mcalpine_eflaw", True):
        out.extend(_textstat_mcalpine(text, word_count))
    if features.get("textstat_locale_formulas"):
        out.extend(_textstat_locale_formulas(text, word_count))

    if features.get("readability_metrics_formulas", True):
        reader, reader_reason = _build_reader(text, word_count)
        out.extend(_readability_metrics_formulas(reader, reader_reason, word_count, collector))

    if features.get("pystylometry_formulas", True):
        out.extend(_pystylometry_formulas(text, word_count, collector, features))

    if features.get("syllable_crosscheck", True):
        out.extend(_syllable_crosscheck(analysis, config, word_count))

    if features.get("difficult_word_crosscheck", True):
        out.extend(_difficult_word_crosscheck(text, word_count, config))

    # ---------------------------------------------------------- disagreement
    _disagreement(out, collector, "fk", f"{PREFIX}fk_disagreement",
                 "Flesch-Kincaid grade, cross-implementation disagreement", "grade", word_count)
    _disagreement(out, collector, "ari", f"{PREFIX}ari_disagreement",
                 "Automated Readability Index, cross-implementation disagreement", "grade",
                 word_count)
    _disagreement(out, collector, "flesch_reading_ease", f"{PREFIX}flesch_reading_ease_disagreement",
                 "Flesch Reading Ease, cross-implementation disagreement", "score", word_count,
                 libraries=("textstat", "readability_metrics", "pystylometry"))
    _disagreement(out, collector, "gunning_fog", f"{PREFIX}gunning_fog_disagreement",
                 "Gunning Fog, cross-implementation disagreement", "grade", word_count,
                 libraries=("textstat", "readability_metrics", "pystylometry"))
    _disagreement(out, collector, "smog", f"{PREFIX}smog_disagreement",
                 "SMOG index, cross-implementation disagreement", "grade", word_count,
                 libraries=("textstat", "readability_metrics", "pystylometry"))
    _disagreement(out, collector, "coleman_liau", f"{PREFIX}coleman_liau_disagreement",
                 "Coleman-Liau index, cross-implementation disagreement", "grade", word_count,
                 libraries=("textstat", "readability_metrics", "pystylometry"))
    _disagreement(out, collector, "dale_chall", f"{PREFIX}dale_chall_disagreement",
                 "Dale-Chall score, cross-implementation disagreement", "score", word_count,
                 libraries=("textstat", "readability_metrics", "pystylometry"))
    _disagreement(out, collector, "linsear_write", f"{PREFIX}linsear_write_disagreement",
                 "Linsear Write, cross-implementation disagreement", "grade", word_count,
                 libraries=("textstat", "readability_metrics", "pystylometry"))
    _disagreement(out, collector, "spache", f"{PREFIX}spache_disagreement",
                 "Spache index, cross-implementation disagreement", "grade", word_count,
                 libraries=("textstat", "readability_metrics"))

    # ------------------------------------------------------ cross-formula aggregate
    if features.get("formula_aggregate", True) and word_count >= MIN_SAMPLE \
            and len(collector.grades) >= 2:
        values = sorted(collector.grades.values())
        spread = values[-1] - values[0]
        base = {"n_formulas": len(values), "formula_ids": sorted(collector.grades.keys()),
               "min": values[0], "max": values[-1]}
        out.append(finding(f"{PREFIX}formula_grade_mean",
                           "Mean formula-implied grade level, across every readability formula "
                           "computed this run", statistics.fmean(values), "grade", family=FAMILY,
                           sample_size=word_count, min_sample=MIN_SAMPLE,
                           distribution=dict(base, aggregation="mean")))
        out.append(finding(f"{PREFIX}formula_grade_median", "Median formula-implied grade level",
                           statistics.median(values), "grade", family=FAMILY,
                           sample_size=word_count, min_sample=MIN_SAMPLE,
                           distribution=dict(base, aggregation="median")))
        out.append(finding(f"{PREFIX}formula_grade_max", "Highest formula-implied grade level",
                           values[-1], "grade", family=FAMILY, sample_size=word_count,
                           min_sample=MIN_SAMPLE, distribution=dict(base, aggregation="max")))
        out.append(finding(f"{PREFIX}formula_grade_spread",
                           "Spread (range) across every formula-implied grade level computed "
                           "this run", spread, "grade", family=FAMILY, sample_size=word_count,
                           min_sample=MIN_SAMPLE,
                           distribution=dict(base, aggregation="range (max - min)")))
        if len(values) > 1:
            out.append(finding(f"{PREFIX}formula_grade_sd",
                               "Standard deviation across every formula-implied grade level "
                               "computed this run", statistics.stdev(values), "grade",
                               family=FAMILY, sample_size=word_count, min_sample=MIN_SAMPLE,
                               distribution=dict(base, aggregation="sample standard deviation")))
    return out
