"""Grammar-adjacent, spelling, typography, encoding and mechanical-quality
diagnostics -- Python-only, and deliberately conservative about invented
vocabulary and dialect.

TextGrader grades prose that is often fantasy or science fiction (invented
names, places and coinages), historical fiction (period spelling), or
dialogue written in dialect ("ain't", "gonna", "y'all"). A spell checker that
flags every invented name and a grammar checker that flags every line of
dialect would be worse than useless here, so every channel in this suite is
built to separate "this looks mechanically broken" from "this looks
deliberate": recurring unknown words are treated as vocabulary, not errors
(see :func:`_recurring_vocabulary` below); narration and dialogue are scored
and reported separately wherever non-standard grammar in dialogue would
otherwise read as an error (see the ``_narration``/``_dialogue`` metric id
pairs); and every heuristic that could fire on real prose is deliberately
narrowed until it does not, per the module's own test suite
(``tests/test_mechanical_quality_suite.py``), including a false-positive test
against a fantasy passage full of invented names and a page of written
dialect.

LanguageTool is explicitly not used (see the task spec: it is a Java server,
and TextGrader is Python only). What replaces its role:

* **Grammar/style-rule diagnostics** (the brief's LanguageTool section) are
  replaced by a small, hand-written rule set of unambiguous, near-universal
  written-English mechanical errors -- doubled consecutive words, and a fixed
  list of "should of"/"would of"/"could of"/"must of"/"might of" (never a
  deliberate dialect spelling; the spoken contraction is "should've", not
  "should of") -- plus the combined bookkeeping the brief asks of a rule
  engine: total rate, a "which single issue type dominates" concentration
  number, the sentence with the most flags, and how many distinct issue types
  fired at all. See group F, below.
* **Spelling / lexical validity** uses two independent, real dictionaries
  (``pyspellchecker`` and ``symspellpy``, both installed and verified below --
  see "Package verification"), kept as separate channels on purpose so their
  disagreement is itself a reported number, plus edit-distance-based
  likely-typo scoring and SymSpell's real word-segmentation for fused-token
  detection. Hunspell/pyenchant and JamSpell are not integrated: pyenchant
  needs the system ``libenchant`` library (a non-Python native dependency this
  project has no other use for), and JamSpell has no maintained, installable
  PyPI wheel for this Python version at the time of writing (``pip install
  jamspell`` has no matching distribution); both gaps are named here rather
  than silently skipped, per the "not installed is never a reason" rule --
  they are excluded for the quoted reasons above, not because installation
  was not attempted.
* **Unicode/encoding/typography** is the suite's most confident group: mixed
  quote/apostrophe/dash/ellipsis style, control characters, zero-width
  characters, the replacement character, private-use characters, a
  diagnostic (never text-mutating) ``ftfy`` repair estimate, an independent
  mojibake pattern scan, and homoglyph detection via ``confusable_homoglyphs``
  (verified below to flag genuine mixed-script confusables, e.g. a Cyrillic
  "а" inside a Latin word, while leaving plain ASCII alone -- naively calling
  its ``is_confusable`` on any Latin word returns a nonempty list, because
  every Latin letter has some cross-script lookalike; ``is_dangerous`` is the
  function that actually answers "is this string a plausible spoof").

Package verification (rule: a package's name is not evidence of what it
does)::

    >>> import spellchecker; spellchecker.SpellChecker().correction('wrold')
    'world'
    >>> import symspellpy; sym_spell.word_segmentation('helloworld')
    Composition(segmented_string='hello world', ...)
    >>> from confusable_homoglyphs import confusables
    >>> confusables.is_dangerous('paypal')            # plain ASCII
    False
    >>> confusables.is_dangerous('p\\u0430ypal')       # Cyrillic а for a
    True
    >>> import ftfy; ftfy.fix_text('MÃ¼nchen')
    'München'

Every one of these was actually imported and exercised in this environment;
none is assumed to work from its name or its PyPI description alone.

Design notes worth reading before changing a threshold:

* **Recurring unknown vocabulary is excluded, and reported.**
  :func:`_recurring_vocabulary` builds a per-document set of unknown words
  that occur at least ``recurring_min_count`` times (default 3); these are
  treated as the manuscript's own vocabulary (character names, invented
  places, coinages) rather than errors, and every "adjusted" rate excludes
  them. The excluded words themselves are always reported as evidence on
  ``lexical.mechanical_recurring_vocabulary_size``, so a writer can see
  exactly what was set aside and why, and correct the module's judgement by
  eye if it ever guesses wrong.
* **A capitalized word that is not the first word of its sentence-like span
  is a proper-noun candidate** and is excluded from the "adjusted" spelling
  rate the same way (see :func:`_word_occurrences`): sentence-initial
  position is inferred cheaply from the character immediately before the
  word in the canonical text (a terminal mark, an opening quote, or the start
  of the text) rather than by re-deriving sentence boundaries, so this stays
  a single linear pass over ``analysis.text``.
* **Dialogue and narration are scored separately, not blended.** Every
  finding that risks reading ordinary characterisation (dialect, sentence
  fragments, deliberately doubled words like "no, no") as a mechanical error
  is computed once per channel and published as two findings
  (``..._narration`` / ``..._dialogue``) rather than one mixed number. The
  purely mechanical/typographic groups (encoding, typography, raw spelling)
  are computed over the whole document, because a mojibake byte sequence or a
  stray non-breaking space is not a dialect choice in either channel.
* **ftfy is diagnostic only.** :func:`_ftfy_findings` never replaces
  ``analysis.text``; it reports what ``ftfy.fix_text`` *would* change, as a
  rate and as bounded before/after evidence, per the task's implementation
  rule 5.
* **Nothing here needs the shared spaCy parse or a sentence-embedding
  model.** ``COST`` is ``moderate`` (bounded dictionary lookups and a handful
  of linear regex passes) and ``REQUIRES`` lists only pip packages, so this
  suite is invisible to neither ``needs_parse`` nor ``needs_model`` --
  ``tests/test_mechanical_quality_suite.py`` asserts directly that enabling
  it never triggers a spaCy parse or downloads a sentence-transformers model.
* **charset-normalizer / real encoding-confidence detection is not
  implemented.** :class:`~textgrader.document.DocumentAnalysis` only ever
  holds decoded ``str`` text (see ``DocumentAnalysis.from_text``'s
  signature); there are no raw bytes left by the time any metric runs, and a
  metric that computed a "confidence this was decoded correctly" number from
  already-decoded text would be fabricating a source-bytes measurement it
  cannot actually make. Every encoding-adjacent finding here is instead
  built from the decoded text itself (mojibake substring patterns, replacement
  characters, ftfy's own diagnostic repair pass), which is exactly what rule
  7 of the task's implementation details asks for in this situation.

Every measurement group is switched independently under ``features`` (see
``DEFAULT_FEATURES``), mirroring ``stylometry_suite``'s convention, and the
whole suite is off by default in ``config.json``.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..optional import on_reset, require
from .common import MODERATE, finding, option, rate, unavailable

FAMILY_PUNCTUATION = "punctuation"
FAMILY_LEXICAL = "lexical"
FAMILY_SYNTAX = "syntax"

FAMILY = FAMILY_LEXICAL  # fallback family for grade.py's spec.family lookup
# Measured with every feature on: ~6s on a 26,000-word novel (Alice in
# Wonderland), ~21s on a 162,000-word novel (Dracula) -- most of it the two
# dictionary loads (cached per process after the first call) and the shared
# narration/dialogue resegmentation pysbd cost that punctuation_profile's
# module docstring also documents. No step is superlinear: two earlier
# passes were found and fixed during development, an accidental O(n^2)
# string-prefix scan in the sentence-initial heuristic (32s on its own on
# Dracula) and O(words * quotations) dialogue-channel classification calling
# DocumentAnalysis.in_dialogue per word instead of one linear merge pass
# (see _word_occurrences and _word_channels below) -- both are why this is
# "moderate", not "fast", despite being dependency-cheap per call.
COST = MODERATE
# pyspellchecker and symspellpy each ship their own bundled dictionary (no
# network call, no system library); confusable_homoglyphs ships its own
# Unicode confusables table; ftfy needs nothing beyond itself. None of these
# touches spacy or sentence_transformers, so MetricSpec.needs_parse/needs_model
# stay False for this whole suite -- see the module docstring.
REQUIRES: tuple[str, ...] = ("pyspellchecker", "symspellpy", "ftfy", "confusable_homoglyphs")
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

MIN_SAMPLE_WORDS = 200
MIN_SAMPLE_SENTENCES = 15

DEFAULT_FEATURES: dict[str, bool] = {
    # Typography: mixed quote/apostrophe/dash/ellipsis rendering, repeated
    # whitespace, ALL-CAPS share. No third-party package; unambiguous.
    "typography": True,
    # Encoding, stdlib-only half: control/zero-width/replacement/private-use
    # character rates and a Unicode-category entropy. No package needed.
    "encoding": True,
    # Encoding, ftfy half: diagnostic mojibake-repair estimate plus an
    # independent, package-free mojibake pattern scan kept as its own
    # channel (disagreement between the two is reported, not reconciled).
    "encoding_ftfy": True,
    # Mixed-script homoglyph detection (confusable_homoglyphs). Bounded
    # automatically: it only ever examines the (normally near-zero) share of
    # word tokens containing a non-ASCII character.
    "confusables": True,
    # Broken line-wrap hyphenation ("exam-\nple"), confirmed against the
    # document's own dictionary and its own genuinely-hyphenated compounds
    # so a real "well-known" wrapped at the hyphen is not flagged.
    "hyphenation": True,
    # Independent dictionary-based unknown-word rate, pyspellchecker.
    "spelling_pyspellchecker": True,
    # Independent dictionary-based unknown-word rate, SymSpell.
    "spelling_symspell": True,
    # Checker agreement/disagreement breakdown; needs both checkers above.
    "spelling_disagreement": True,
    # Likely-typo scoring (edit-distance-1/2, low-frequency, recurring
    # vocabulary excluded) plus the excluded-vocabulary report itself.
    "likely_typos": True,
    # Doubled consecutive words, scored separately for narration/dialogue.
    "doubled_words": True,
    # Fused-token detection via SymSpell's real word segmentation.
    "fused_tokens": True,
    # OCR-style single-character substitution (digit-for-letter, "rn"/"m").
    "ocr_substitution": True,
    # A small, dialect-safe fixed list of "should of"/"would of"/... errors.
    "confusion_pairs": True,
    # Combined rule-engine bookkeeping: total rate, rule concentration,
    # unique issue-type count, flagged-sentence rate, worst single sentence.
    "sentence_summary": True,
}

# Words unknown to the dictionary but seen at least this many times are
# treated as the manuscript's own vocabulary (character names, invented
# places, coinages), not spelling errors.
DEFAULT_RECURRING_MIN_COUNT = 3
# A word seen more than this many times is never scored as a "likely typo"
# even below the recurring threshold: a real typo is not usually retyped
# identically five separate times.
DEFAULT_LIKELY_TYPO_MAX_COUNT = 2
# Bounds the cost of pyspellchecker's correction() calls (group C): a word
# with no edit-distance-1 dictionary match makes correction() generate every
# edit-distance-2 string, which is expensive per call and routine on archaic/
# foreign/invented words -- see the bound's own comment where it is applied.
DEFAULT_LIKELY_TYPO_MAX_CANDIDATES = 500
DEFAULT_MAX_REPORTED = 25
# Bounds the cost of SymSpell segmentation (group D): only unknown tokens at
# least this long are worth checking for a missing space, and at most this
# many distinct candidates are checked per document.
DEFAULT_FUSED_MIN_LENGTH = 8
DEFAULT_FUSED_MAX_CANDIDATES = 400
DEFAULT_LANGUAGE = "en"

CONFUSION_PAIRS = ("should of", "would of", "could of", "must of", "might of")

# British/American suffix alternations. pyspellchecker's bundled dictionary
# is American-English only, and TextGrader grades a lot of historical and
# British-set fiction, so without this a whole book written in British
# spelling would score every "colour"/"endeavour"/"centre"-type word as
# either an unknown word or a likely typo of its American spelling -- a
# systematic false positive on real, correctly-spelled published prose (see
# the module docstring's fantasy-names/dialect concern; this is the same
# problem in a different costume). Applied both ways so an American-spelled
# document checked in a hypothetically British-only dictionary would be
# equally protected. Each pair is a real, common alternation, not a fuzzy
# edit-distance guess, kept deliberately small and literal to avoid hiding a
# genuine typo behind a coincidental suffix match.
REGIONAL_SUFFIX_PAIRS = (
    ("our", "or"), ("ise", "ize"), ("ised", "ized"), ("ising", "izing"),
    ("isation", "ization"), ("isations", "izations"), ("yse", "yze"),
    ("ysed", "yzed"), ("ysing", "yzing"), ("ogue", "og"), ("re", "er"),
)


def _regional_variant_forms(word: str) -> list[str]:
    """Candidate American (or British) spellings of ``word``.

    Substitution happens anywhere the pattern occurs, not only at the end of
    the word, because the alternation is a morpheme ("colour" -> "color"),
    and an inflected form ("favouring", "neighbourhood") still carries it
    mid-word. This is safe against false matches because every candidate
    produced here is thrown away unless it is ALSO a real dictionary word
    (see :func:`_known_allowing_regional_variants`); a coincidental
    substring match that does not land on a real word costs one dictionary
    lookup and nothing else.
    """

    forms = []
    for a, b in REGIONAL_SUFFIX_PAIRS:
        if a in word and len(word) > len(a):
            forms.append(word.replace(a, b))
        if b in word and len(word) > len(b):
            forms.append(word.replace(b, a))
    return forms


def _known_allowing_regional_variants(word: str, known_word) -> bool:
    if known_word(word):
        return True
    return any(known_word(form) for form in _regional_variant_forms(word))

SENTENCE_BOUNDARY_CHARS = set(".!?…\"“”‘’':;\n")

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_ALNUM_RUN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?![A-Za-z0-9])")
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]{2,})-\n[ \t]*([a-z]{2,})")
_SAME_LINE_HYPHEN_RE = re.compile(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b")
_ALL_CAPS_RE = re.compile(r"\b[A-Z]{2,}\b")
_WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")
_DOUBLE_QUOTE_STRAIGHT_RE = re.compile('"')
_DOUBLE_QUOTE_CURLY_RE = re.compile("[“”]")
_SINGLE_QUOTE_STRAIGHT_RE = re.compile(r"(?<![A-Za-z])'|'(?![A-Za-z])")
_SINGLE_QUOTE_CURLY_RE = re.compile("[‘’]")
_EM_DASH_RE = re.compile("—")
_DOUBLE_HYPHEN_RE = re.compile(r"-{2,}")
_SPACED_EN_DASH_RE = re.compile(r"(?<=\s)–(?=\s)")
_ELLIPSIS_CHAR_RE = re.compile("…")
_ELLIPSIS_DOTS_RE = re.compile(r"\.{3,}")
_ZERO_WIDTH_RE = re.compile("[​‌‍﻿]")
_NBSP_RE = re.compile(" ")
_REPLACEMENT_RE = re.compile("�")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A short, deliberately narrow list of literal mojibake substrings: UTF-8
# text that was decoded as (or re-encoded through) Latin-1/CP1252. Kept
# separate from ftfy on purpose -- see the module docstring's "disagreement
# between the two is reported, not reconciled".
_MOJIBAKE_PATTERNS_RE = re.compile(
    "Ã[ -¿]"          # Ã followed by a Latin-1 continuation byte
    "|â\u0080[\u0090-\u009f]"   # â€<x>, the classic curly-quote/dash mangle
    "|Â[ -¿]"         # Â followed by a Latin-1 punctuation byte
)

_DIGIT_TO_LETTER = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g"}


# --------------------------------------------------------------- small maths

def _entropy(counter: Mapping[Any, int]) -> float | None:
    total = sum(counter.values())
    if not total:
        return None
    import math
    return -sum((n / total) * math.log2(n / total) for n in counter.values() if n)


def _mixing_share(counts: Mapping[str, int]) -> tuple[float | None, int]:
    """% of occurrences NOT in the single most common variant; 0 = consistent."""

    total = sum(counts.values())
    if total < 2:
        return (None if total == 0 else 0.0), total
    top = max(counts.values())
    return 100.0 * (total - top) / total, total


def _feature(config: Mapping[str, Any] | None, name: str) -> bool:
    features = option(config, "features", {})
    if not isinstance(features, Mapping):
        return DEFAULT_FEATURES.get(name, False)
    value = features.get(name, DEFAULT_FEATURES.get(name, False))
    return DEFAULT_FEATURES.get(name, False) if value is None else bool(value)


# ------------------------------------------------------- cached spell checkers

_PYSPELLCHECKER_CACHE: dict[str, tuple[Any, str | None]] = {}
_SYMSPELL_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_caches() -> None:
    _PYSPELLCHECKER_CACHE.clear()
    _SYMSPELL_CACHE.clear()


on_reset(_reset_caches)


def _pyspellchecker(language: str) -> tuple[Any, str | None]:
    if language in _PYSPELLCHECKER_CACHE:
        return _PYSPELLCHECKER_CACHE[language]
    module, reason = require("pyspellchecker")
    if module is None:
        outcome = (None, reason)
    else:
        try:
            outcome = (module.SpellChecker(language=language), None)
        except Exception as exc:  # pragma: no cover - unsupported language code
            outcome = (None, f"pyspellchecker could not load language {language!r} "
                             f"({type(exc).__name__}: {exc})")
    _PYSPELLCHECKER_CACHE[language] = outcome
    return outcome


def _symspell(language: str) -> tuple[Any, str | None]:
    """A SymSpell instance over its own bundled English frequency dictionary.

    SymSpellPy ships one bundled dictionary (English); a non-English
    ``language`` degrades this one channel to "unavailable" rather than
    silently scoring the wrong language against it.
    """

    if language in _SYMSPELL_CACHE:
        return _SYMSPELL_CACHE[language]
    if language != "en":
        outcome = (None, f"symspellpy's bundled dictionary only covers English; "
                         f"language={language!r} was requested")
        _SYMSPELL_CACHE[language] = outcome
        return outcome
    module, reason = require("symspellpy")
    if module is None:
        outcome = (None, reason)
    else:
        try:
            from importlib import resources
            path = str(resources.files("symspellpy")
                       .joinpath("frequency_dictionary_en_82_765.txt"))
            sym_spell = module.SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            sym_spell.load_dictionary(path, term_index=0, count_index=1)
            outcome = (sym_spell, None)
        except Exception as exc:  # pragma: no cover - broken install
            outcome = (None, f"symspellpy dictionary failed to load "
                             f"({type(exc).__name__}: {exc})")
    _SYMSPELL_CACHE[language] = outcome
    return outcome


# ------------------------------------------------------------- shared, cached

def _word_occurrences(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    """Every word token with its offset and a cheap sentence-initial flag.

    Sentence-initial position is read from the character immediately before
    the word in ``analysis.text`` (a terminal mark, an opening quote/bracket,
    or the very start of the text) rather than by re-deriving sentence
    boundaries, so this stays one linear pass. It is a proxy, not a parse:
    good enough to separate "capitalized because it opens a clause" from
    "capitalized because it is a name", which is all the spelling-adjustment
    groups below need it for.
    """

    def build() -> list[dict[str, Any]]:
        text = analysis.text
        out = []
        for match in _WORD_RE.finditer(text):
            word = match.group(0)
            start = match.start()
            # Walk back over spaces/tabs without slicing the whole prefix --
            # ``text[:start].rstrip(...)`` copies up to the entire document on
            # every single word, which is quadratic overall (measured: 32s on
            # a 160,000-word novel, versus a fraction of a second this way).
            pos = start
            while pos > 0 and text[pos - 1] in " \t":
                pos -= 1
            sentence_initial = pos == 0 or text[pos - 1] in SENTENCE_BOUNDARY_CHARS
            out.append({"word": word, "lower": word.lower().replace("’", "'"),
                       "offset": start, "sentence_initial": sentence_initial})
        return out

    return analysis.memo("mechanical.word_occurrences", build)


def _word_channels(analysis: DocumentAnalysis) -> list[str]:
    """``"narration"``/``"dialogue"`` for each entry of :func:`_word_occurrences`,
    in the same order, computed with one merge pass over both already-sorted
    sequences (occurrences by text order, ``quotation_spans`` by start).

    ``DocumentAnalysis.in_dialogue`` answers the same question for one offset
    at a time by scanning ``quotation_spans`` from its start on every call;
    called once per word over a whole novel that is O(words * quotations)
    instead of O(words + quotations), and was measured costing tens of
    seconds on a single dialogue-heavy book. This does the same classification
    as one linear sweep instead, cached alongside the occurrences it labels.
    """

    def build() -> list[str]:
        spans = analysis.quotation_spans
        occurrences = _word_occurrences(analysis)
        out = []
        index = 0
        total_spans = len(spans)
        for item in occurrences:
            offset = item["offset"]
            while index < total_spans and spans[index][1] <= offset:
                index += 1
            if index < total_spans and spans[index][0] <= offset < spans[index][1]:
                out.append("dialogue")
            else:
                out.append("narration")
        return out

    return analysis.memo("mechanical.word_channels", build)


def _recurring_vocabulary(occurrences: Sequence[Mapping[str, Any]], unknown: set[str],
                          min_count: int) -> dict[str, int]:
    """Unknown words seen often enough to be this manuscript's own vocabulary."""

    counts = Counter(item["lower"] for item in occurrences if item["lower"] in unknown)
    return {word: count for word, count in counts.items() if count >= min_count}


def _proper_noun_candidates(occurrences: Sequence[Mapping[str, Any]]) -> set[str]:
    """Lowercased forms seen capitalized, mid-span, at least once.

    A word excluded here only because ONE occurrence looked like a name is a
    deliberately generous exclusion: the cost of missing a real typo that
    happens to share a spelling with a capitalized word is much lower than
    the cost of flagging an invented name throughout a book.
    """

    return {item["lower"] for item in occurrences
            if not item["sentence_initial"] and item["word"][:1].isupper()
            and len(item["word"]) > 1}


# -------------------------------------------------------- A. typography group

def _typography_findings(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    text = analysis.text
    words_total = analysis.word_count
    out = []

    def mixing_finding(metric_id: str, name: str, counts: Mapping[str, int]) -> dict[str, Any]:
        share, total = _mixing_share(counts)
        return finding(metric_id, name, share, "percent", family=FAMILY_PUNCTUATION,
                       sample_size=total, min_sample=10,
                       evidence=[{"variant": k, "count": v} for k, v in counts.items() if v],
                       warning=None if total else "no marks of this kind were found")

    out.append(mixing_finding(
        "punct.mechanical_mixed_quote_style", "Mixed straight/curly double-quote style",
        {"straight": len(_DOUBLE_QUOTE_STRAIGHT_RE.findall(text)),
         "curly": len(_DOUBLE_QUOTE_CURLY_RE.findall(text))}))
    out.append(mixing_finding(
        "punct.mechanical_mixed_apostrophe_style",
        "Mixed straight/curly apostrophe style (dominated by contractions, "
        "same caveat as punctuation_profile's single_quote bucket)",
        {"straight": len(_SINGLE_QUOTE_STRAIGHT_RE.findall(text)),
         "curly": len(_SINGLE_QUOTE_CURLY_RE.findall(text))}))
    out.append(mixing_finding(
        "punct.mechanical_mixed_dash_style",
        "Mixed interruption-dash style (em dash / double hyphen / spaced en dash)",
        {"em_dash": len(_EM_DASH_RE.findall(text)),
         "double_hyphen": len(_DOUBLE_HYPHEN_RE.findall(text)),
         "spaced_en_dash": len(_SPACED_EN_DASH_RE.findall(text))}))
    out.append(mixing_finding(
        "punct.mechanical_mixed_ellipsis_style", "Mixed ellipsis style (… vs ...)",
        {"char": len(_ELLIPSIS_CHAR_RE.findall(text)),
         "dots": len(_ELLIPSIS_DOTS_RE.findall(text))}))

    if not words_total:
        out.append(finding("punct.mechanical_repeated_whitespace_rate",
                           "Repeated-whitespace rate (excludes the classic two-space-"
                           "after-period convention)", None, "per_1000_words",
                           family=FAMILY_PUNCTUATION, sample_size=0, min_sample=MIN_SAMPLE_WORDS,
                           warning="no words in text"))
        out.append(finding("punct.mechanical_all_caps_word_rate", "ALL-CAPS word rate",
                           None, "per_1000_words", family=FAMILY_PUNCTUATION, sample_size=0,
                           min_sample=MIN_SAMPLE_WORDS, warning="no words in text"))
        return out

    suspicious = 0
    examples = []
    for match in _WHITESPACE_RUN_RE.finditer(text):
        preceding = text[:match.start()].rstrip()
        two_space_convention = len(match.group(0)) == 2 and preceding and preceding[-1] in ".!?\"')]"
        if not two_space_convention:
            suspicious += 1
            if len(examples) < DEFAULT_MAX_REPORTED:
                examples.append({"offset": match.start(),
                                 "context": text[max(0, match.start() - 20):match.end() + 20]})
    out.append(finding("punct.mechanical_repeated_whitespace_rate",
                       "Repeated-whitespace rate (excludes the classic two-space-after-"
                       "period convention)", rate(suspicious, words_total, 1000.0),
                       "per_1000_words", family=FAMILY_PUNCTUATION, sample_size=words_total,
                       min_sample=MIN_SAMPLE_WORDS, evidence=examples[:10]))

    all_caps = len(_ALL_CAPS_RE.findall(text))
    out.append(finding("punct.mechanical_all_caps_word_rate",
                       "ALL-CAPS word rate (emphasis/shouting, or an encoding artefact "
                       "if unexpectedly high)", rate(all_caps, words_total, 1000.0),
                       "per_1000_words", family=FAMILY_PUNCTUATION, sample_size=words_total,
                       min_sample=MIN_SAMPLE_WORDS))
    return out


# ---------------------------------------------------------- B. encoding group

def _encoding_findings(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    text = analysis.text
    words_total = analysis.word_count
    out = []

    def char_rate_finding(metric_id: str, name: str, pattern: re.Pattern,
                          evidence_label: str) -> dict[str, Any]:
        matches = list(pattern.finditer(text))
        count = len(matches)
        value = rate(count, words_total, 1000.0) if words_total else None
        evidence = [{"offset": m.start(), evidence_label: repr(m.group(0)),
                    "context": text[max(0, m.start() - 15):m.start() + 15]}
                   for m in matches[:10]]
        return finding(metric_id, name, value, "per_1000_words", family=FAMILY_PUNCTUATION,
                       sample_size=words_total, min_sample=MIN_SAMPLE_WORDS, evidence=evidence,
                       warning=None if words_total else "no words in text")

    out.append(char_rate_finding("punct.mechanical_control_char_rate",
                                 "Control-character rate (excluding tab/newline/CR)",
                                 _CONTROL_RE, "char"))
    out.append(char_rate_finding("punct.mechanical_zero_width_char_rate",
                                 "Zero-width character rate", _ZERO_WIDTH_RE, "char"))
    out.append(char_rate_finding("punct.mechanical_replacement_char_rate",
                                 "Replacement-character (�) rate",
                                 _REPLACEMENT_RE, "char"))
    out.append(char_rate_finding("punct.mechanical_nonbreaking_space_rate",
                                 "Non-breaking-space rate", _NBSP_RE, "char"))

    private_use = sum(1 for ch in text if 0xE000 <= ord(ch) <= 0xF8FF
                      or 0xF0000 <= ord(ch) <= 0xFFFFD or 0x100000 <= ord(ch) <= 0x10FFFD)
    out.append(finding("punct.mechanical_private_use_char_rate",
                       "Private-use-area character rate",
                       rate(private_use, words_total, 1000.0) if words_total else None,
                       "per_1000_words", family=FAMILY_PUNCTUATION, sample_size=words_total,
                       min_sample=MIN_SAMPLE_WORDS,
                       warning=None if words_total else "no words in text"))

    if not text:
        out.append(finding("punct.mechanical_unicode_category_entropy",
                           "Unicode general-category entropy", None, "bits",
                           family=FAMILY_PUNCTUATION, sample_size=0, min_sample=MIN_SAMPLE_WORDS,
                           warning="no text"))
    else:
        categories = Counter(unicodedata.category(ch) for ch in text)
        out.append(finding("punct.mechanical_unicode_category_entropy",
                           "Unicode general-category entropy (garbled/mixed-encoding "
                           "text spreads across far more categories than clean prose)",
                           _entropy(categories), "bits", family=FAMILY_PUNCTUATION,
                           sample_size=len(text), min_sample=MIN_SAMPLE_WORDS,
                           distribution={"distinct_categories": len(categories)},
                           evidence=[{"category": c, "count": n}
                                    for c, n in categories.most_common(10)],
                           sample_size_sensitive=True))
    return out


def _ftfy_findings(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    """Diagnostic only: reports what ``ftfy.fix_text`` would change, never
    replaces ``analysis.text`` (task implementation rule 5)."""

    text = analysis.text
    words_total = analysis.word_count
    module, reason = require("ftfy")
    if module is None:
        return [
            unavailable("punct.mechanical_ftfy_repair_rate", "ftfy repair-candidate rate",
                       reason, family=FAMILY_PUNCTUATION),
            unavailable("punct.mechanical_mojibake_pattern_rate", "Mojibake pattern rate",
                       reason, family=FAMILY_PUNCTUATION),
        ]
    out = []
    if not text:
        out.append(finding("punct.mechanical_ftfy_repair_rate", "ftfy repair-candidate rate",
                           None, "percent", family=FAMILY_PUNCTUATION, sample_size=0,
                           min_sample=MIN_SAMPLE_WORDS, warning="no text"))
    else:
        try:
            fixed, explanation = module.fix_and_explain(text)
        except Exception:  # pragma: no cover - ftfy internal failure
            fixed, explanation = text, None
        changed = sum(1 for a, b in zip(text, fixed) if a != b) + abs(len(text) - len(fixed))
        repair_pct = 100.0 * changed / len(text)
        ops = [{"operation": op, "encoding": enc}
              for op, enc in (explanation or []) if op != "apply"][:10]
        out.append(finding(
            "punct.mechanical_ftfy_repair_rate",
            "ftfy repair-candidate rate (diagnostic only; the analyzed text is "
            "never rewritten by this metric)", repair_pct, "percent",
            family=FAMILY_PUNCTUATION, sample_size=len(text), min_sample=MIN_SAMPLE_WORDS,
            distribution={"characters_examined": len(text), "characters_changed": changed},
            evidence=ops, warning=None if changed else None))

    mojibake_matches = list(_MOJIBAKE_PATTERNS_RE.finditer(text))
    out.append(finding(
        "punct.mechanical_mojibake_pattern_rate",
        "Mojibake pattern rate (independent regex scan, kept alongside the ftfy "
        "estimate above rather than reconciled with it -- see the module "
        "docstring)", rate(len(mojibake_matches), words_total, 1000.0) if words_total else None,
        "per_1000_words", family=FAMILY_PUNCTUATION, sample_size=words_total,
        min_sample=MIN_SAMPLE_WORDS,
        evidence=[{"offset": m.start(), "text": text[max(0, m.start() - 10):m.end() + 10]}
                 for m in mojibake_matches[:10]],
        warning=None if words_total else "no words in text"))
    return out


def _confusable_findings(analysis: DocumentAnalysis) -> list[dict[str, Any]]:
    module, reason = require("confusable_homoglyphs")
    words_total = analysis.word_count
    if module is None:
        return [unavailable("punct.mechanical_homoglyph_rate",
                            "Mixed-script homoglyph rate", reason, family=FAMILY_PUNCTUATION)]
    if not words_total:
        return [finding("punct.mechanical_homoglyph_rate", "Mixed-script homoglyph rate",
                        None, "per_1000_words", family=FAMILY_PUNCTUATION, sample_size=0,
                        min_sample=MIN_SAMPLE_WORDS, warning="no words in text")]
    non_ascii_words = [w for w in analysis.words if any(ord(ch) > 127 for ch in w)]
    flagged = []
    for word in non_ascii_words:
        try:
            if module.is_dangerous(word):
                flagged.append(word)
        except Exception:  # pragma: no cover - malformed input to the checker
            continue
    return [finding(
        "punct.mechanical_homoglyph_rate",
        "Mixed-script homoglyph rate (flags real script-mixing, e.g. a Cyrillic "
        "а inside a Latin word; plain ASCII and single-script non-English "
        "text both score zero)", rate(len(flagged), words_total, 1000.0), "per_1000_words",
        family=FAMILY_PUNCTUATION, sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        distribution={"non_ascii_word_tokens": len(non_ascii_words)},
        evidence=[{"word": w} for w in flagged[:DEFAULT_MAX_REPORTED]])]


def _hyphenation_findings(analysis: DocumentAnalysis, known_word) -> list[dict[str, Any]]:
    text = analysis.text
    words_total = analysis.word_count
    same_line_pairs = {(a.lower(), b.lower()) for a, b in _SAME_LINE_HYPHEN_RE.findall(text)}
    flagged = []
    for match in _HYPHEN_BREAK_RE.finditer(text):
        first, second = match.group(1), match.group(2)
        pair = (first.lower(), second.lower())
        if pair in same_line_pairs:
            continue  # a genuine hyphenated compound that also appears unbroken
        joined = (first + second).lower()
        if known_word is None or known_word(joined):
            flagged.append({"offset": match.start(), "reconstructed": first + second,
                           "original": match.group(0)})
    warning = None if words_total else "no words in text"
    if words_total and known_word is None:
        warning = ("no spell checker was available to confirm the reconstructed word, "
                   "so this count is unconfirmed line-wrap hyphenation only")
    return [finding(
        "punct.mechanical_broken_hyphenation_rate",
        "Broken line-wrap hyphenation rate (a word split across a line break whose "
        "hyphen never appears when the same two halves are written on one line)",
        rate(len(flagged), words_total, 1000.0) if words_total else None, "per_1000_words",
        family=FAMILY_PUNCTUATION, sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        evidence=flagged[:DEFAULT_MAX_REPORTED], warning=warning)]


# --------------------------------------------------------- C. spelling groups

def _known_word_functions(config: Mapping[str, Any] | None):
    """``(pyspell_known, pyspell_reason, symspell_known, symspell_reason)``."""

    language = option(config, "language", DEFAULT_LANGUAGE)
    checker, py_reason = _pyspellchecker(language)
    sym_spell, sym_reason = _symspell(language)

    def pyspell_known(word: str) -> bool:
        return word in checker.word_frequency.dictionary

    def symspell_known(word: str) -> bool:
        return word in sym_spell.words

    return (pyspell_known if checker else None, py_reason,
           symspell_known if sym_spell else None, sym_reason)


def _spelling_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None,
                       pyspell_known, py_reason, symspell_known, sym_reason,
                       do_pyspell: bool, do_symspell: bool, do_disagreement: bool
                       ) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    """Returns ``(findings, pyspell_unknown_lowered, symspell_unknown_lowered)``."""

    occurrences = _word_occurrences(analysis)
    words_total = analysis.word_count
    out: list[dict[str, Any]] = []
    py_unknown: set[str] = set()
    sym_unknown: set[str] = set()

    if not words_total:
        no_words_ids = []
        if do_pyspell:
            no_words_ids += [("lexical.mechanical_unknown_word_rate_pyspellchecker",
                              "Unknown-word rate (pyspellchecker)"),
                             ("lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted",
                              "Unknown-word rate (pyspellchecker), proper nouns and "
                              "recurring vocabulary excluded")]
        if do_symspell:
            no_words_ids += [("lexical.mechanical_unknown_word_rate_symspell",
                              "Unknown-word rate (SymSpell)"),
                             ("lexical.mechanical_unknown_word_rate_symspell_adjusted",
                              "Unknown-word rate (SymSpell), proper nouns and recurring "
                              "vocabulary excluded")]
        return ([finding(mid, name, None, "per_1000_words", family=FAMILY_LEXICAL,
                         sample_size=0, min_sample=MIN_SAMPLE_WORDS, warning="no words in text")
                for mid, name in no_words_ids], py_unknown, sym_unknown)

    recurring_min_count = int(option(config, "recurring_min_count", DEFAULT_RECURRING_MIN_COUNT))
    proper_nouns = _proper_noun_candidates(occurrences)

    channels = _word_channels(analysis)

    def channel(items, checker) -> dict[str, float | None]:
        """Adjusted-rate breakdown by dialogue/narration for the evidence block."""

        by_channel = {"narration": [0, 0], "dialogue": [0, 0]}
        for item, ch in zip(items, channels):
            by_channel[ch][0] += 1
            if checker(item["lower"]) is False:
                by_channel[ch][1] += 1
        return {ch: rate(unk, tot, 1000.0) for ch, (tot, unk) in by_channel.items()}

    def build_pair(checker, prefix: str, label: str) -> tuple[list[dict], set[str]]:
        unknown_lower = {item["lower"] for item in occurrences
                         if not checker(item["lower"])}
        raw_count = sum(1 for item in occurrences if item["lower"] in unknown_lower)
        recurring = _recurring_vocabulary(occurrences, unknown_lower, recurring_min_count)
        regional = {word for word in unknown_lower
                   if _known_allowing_regional_variants(word, checker)}
        excluded = proper_nouns | set(recurring) | regional
        adjusted_count = sum(1 for item in occurrences
                             if item["lower"] in unknown_lower and item["lower"] not in excluded)
        pair = [
            finding(f"lexical.mechanical_unknown_word_rate_{prefix}",
                   f"Unknown-word rate ({label}), every unrecognized token",
                   rate(raw_count, words_total, 1000.0), "per_1000_words",
                   family=FAMILY_LEXICAL, sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
                   evidence=[{"count": raw_count}]),
            finding(f"lexical.mechanical_unknown_word_rate_{prefix}_adjusted",
                   f"Unknown-word rate ({label}), proper-noun candidates, recurring "
                   f"vocabulary and British/American spelling variants excluded (see "
                   f"lexical.mechanical_recurring_vocabulary_size)",
                   rate(adjusted_count, words_total, 1000.0), "per_1000_words",
                   family=FAMILY_LEXICAL, sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
                   distribution={"proper_noun_candidates_excluded": len(proper_nouns),
                                "recurring_vocabulary_excluded": len(recurring),
                                "regional_spelling_variants_excluded": len(regional),
                                "raw_unknown_count": raw_count,
                                "adjusted_unknown_count": adjusted_count},
                   details=[{"channel": ch, "per_1000_words": value}
                           for ch, value in channel(occurrences, checker).items()]),
        ]
        return pair, unknown_lower

    if do_pyspell:
        if pyspell_known is None:
            out.append(unavailable("lexical.mechanical_unknown_word_rate_pyspellchecker",
                                   "Unknown-word rate (pyspellchecker)", py_reason,
                                   family=FAMILY_LEXICAL))
            out.append(unavailable("lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted",
                                   "Unknown-word rate (pyspellchecker), adjusted", py_reason,
                                   family=FAMILY_LEXICAL))
        else:
            pair, py_unknown = build_pair(pyspell_known, "pyspellchecker", "pyspellchecker")
            out.extend(pair)

    if do_symspell:
        if symspell_known is None:
            out.append(unavailable("lexical.mechanical_unknown_word_rate_symspell",
                                   "Unknown-word rate (SymSpell)", sym_reason,
                                   family=FAMILY_LEXICAL))
            out.append(unavailable("lexical.mechanical_unknown_word_rate_symspell_adjusted",
                                   "Unknown-word rate (SymSpell), adjusted", sym_reason,
                                   family=FAMILY_LEXICAL))
        else:
            pair, sym_unknown = build_pair(symspell_known, "symspell", "SymSpell")
            out.extend(pair)

    if do_disagreement:
        if pyspell_known is None or symspell_known is None:
            missing = py_reason if pyspell_known is None else sym_reason
            out.append(unavailable("lexical.mechanical_checker_disagreement_rate",
                                   "Spell-checker disagreement rate", missing,
                                   family=FAMILY_LEXICAL))
        else:
            counts = {"both_unknown": 0, "only_pyspellchecker": 0, "only_symspell": 0,
                     "both_known": 0}
            for item in occurrences:
                py_unk = not pyspell_known(item["lower"])
                sym_unk = not symspell_known(item["lower"])
                if py_unk and sym_unk:
                    counts["both_unknown"] += 1
                elif py_unk:
                    counts["only_pyspellchecker"] += 1
                elif sym_unk:
                    counts["only_symspell"] += 1
                else:
                    counts["both_known"] += 1
            disagreement = counts["only_pyspellchecker"] + counts["only_symspell"]
            out.append(finding(
                "lexical.mechanical_checker_disagreement_rate",
                "Spell-checker disagreement rate (share of words exactly one "
                "dictionary calls unknown)", rate(disagreement, words_total, 1000.0),
                "per_1000_words", family=FAMILY_LEXICAL, sample_size=words_total,
                min_sample=MIN_SAMPLE_WORDS, distribution=counts))

    return out, py_unknown, sym_unknown


def _likely_typo_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None,
                          pyspell_checker_obj, py_reason,
                          py_unknown: set[str]) -> list[dict[str, Any]]:
    """Likely-typo rate, split by narration/dialogue.

    Eye dialect and dropped-g speech ("wonderin'", "bein'", "fust", "yus") is
    one edit-distance-1 hop from a common word almost by construction --
    that is exactly what phonetic dialogue spelling does on purpose -- so a
    blended rate would read a character's accent as a proofing error. Split
    by channel, the same way ``doubled_words``/``confusion_pairs`` are, so a
    dialogue-heavy rate is visible without being read as a narration problem.
    """

    words_total = analysis.word_count
    channel_ids = ("lexical.mechanical_likely_typo_rate_narration",
                  "lexical.mechanical_likely_typo_rate_dialogue")
    if pyspell_checker_obj is None:
        return [unavailable(mid, "Likely-typo rate", py_reason, family=FAMILY_LEXICAL)
               for mid in channel_ids] + [
            unavailable("lexical.mechanical_recurring_vocabulary_size",
                       "Recurring unknown-vocabulary size", py_reason, family=FAMILY_LEXICAL)]
    if not words_total:
        return [finding(mid, "Likely-typo rate", None, "per_1000_words", family=FAMILY_LEXICAL,
                        sample_size=0, min_sample=MIN_SAMPLE_WORDS, warning="no words in text")
               for mid in channel_ids] + [
            finding("lexical.mechanical_recurring_vocabulary_size",
                   "Recurring unknown-vocabulary size", None, "words", family=FAMILY_LEXICAL,
                   sample_size=0, min_sample=MIN_SAMPLE_WORDS, warning="no words in text")]

    occurrences = _word_occurrences(analysis)
    recurring_min_count = int(option(config, "recurring_min_count", DEFAULT_RECURRING_MIN_COUNT))
    max_typo_count = int(option(config, "likely_typo_max_count", DEFAULT_LIKELY_TYPO_MAX_COUNT))
    recurring = _recurring_vocabulary(occurrences, py_unknown, recurring_min_count)
    proper_nouns = _proper_noun_candidates(occurrences)

    def known(word: str) -> bool:
        return word in pyspell_checker_obj.word_frequency.dictionary

    regional = {word for word in py_unknown if _known_allowing_regional_variants(word, known)}
    excluded = proper_nouns | set(recurring) | regional

    doc_counts = Counter(item["lower"] for item in occurrences)
    channel_counts: dict[str, dict[str, int]] = {}
    channel_offset: dict[str, dict[str, int]] = {}
    for item, ch in zip(occurrences, _word_channels(analysis)):
        counts = channel_counts.setdefault(item["lower"], {"narration": 0, "dialogue": 0})
        counts[ch] += 1
        channel_offset.setdefault(item["lower"], {}).setdefault(ch, item["offset"])

    all_candidates = [word for word in py_unknown
                     if word not in excluded and doc_counts[word] <= max_typo_count
                     and word.isalpha() and len(word) > 2]
    # pyspellchecker's own correction() escalates to generating every
    # edit-distance-2 string whenever a word has no edit-distance-1 match in
    # the dictionary -- routine for archaic spellings, foreign names and
    # invented words alike, not just real typos -- and that escalation is
    # roughly 1000x slower per call (measured: ~0.8s/word with no ED1 match,
    # vs ~1ms/word for the ED1-only lookup used here). A 160,000-word novel
    # can have several hundred such words, which turned this loop into
    # minutes. Restricting this finding to edit-distance-1 corrections (this
    # module's own thin wrapper over the SAME dictionary, not a weaker one)
    # trades a little recall on two-edit typos for a bound that no longer
    # depends on how many odd words a book happens to contain -- see rule
    # "Bound O(n^2) work". The cap below is a second, independent bound in
    # case a future change makes even the ED1 path non-trivial.
    max_candidates = int(option(config, "likely_typo_max_candidates",
                                DEFAULT_LIKELY_TYPO_MAX_CANDIDATES))
    candidates = sorted(all_candidates, key=lambda w: -doc_counts[w])[:max_candidates]
    typo_evidence: dict[str, list] = {"narration": [], "dialogue": []}
    typo_count = {"narration": 0, "dialogue": 0}
    for word in candidates:
        near = pyspell_checker_obj.known(pyspell_checker_obj.edit_distance_1(word))
        if not near:
            continue
        correction = max(near, key=lambda w: pyspell_checker_obj.word_frequency[w])
        if correction == word:
            continue
        for ch in ("narration", "dialogue"):
            count = channel_counts[word][ch]
            if not count:
                continue
            typo_count[ch] += count
            if len(typo_evidence[ch]) < DEFAULT_MAX_REPORTED:
                typo_evidence[ch].append({"word": word, "suggested": correction, "count": count,
                                         "offset": channel_offset[word][ch]})

    capped = len(all_candidates) > max_candidates
    warning = (("this book has more likely-typo candidates than the "
               f"{max_candidates}-word bound checks; the rate is computed only from the "
               "most-repeated candidates within that bound, see likely_typo_max_candidates")
              if capped else None)
    out = []
    for ch, mid in zip(("narration", "dialogue"), channel_ids):
        view_words = analysis.narration.word_count if ch == "narration" else analysis.dialogue.word_count
        out.append(finding(
            mid, f"Likely-typo rate, {ch} only (unknown, non-recurring, one edit from a "
            f"common word)", rate(typo_count[ch], view_words, 1000.0) if view_words else None,
            "per_1000_words", family=FAMILY_LEXICAL, channel=ch, sample_size=view_words,
            min_sample=MIN_SAMPLE_WORDS,
            distribution={"candidates_found": len(all_candidates),
                         "candidates_checked": len(candidates),
                         "regional_spelling_variants_excluded": len(regional)},
            warning=warning if view_words else f"no {ch} found in this text",
            evidence=typo_evidence[ch]))

    excluded_evidence = sorted(recurring.items(), key=lambda kv: -kv[1])[:DEFAULT_MAX_REPORTED]
    out.append(finding(
        "lexical.mechanical_recurring_vocabulary_size",
        "Recurring unknown-vocabulary size (words unknown to the dictionary but "
        "repeated often enough to be treated as this manuscript's own vocabulary "
        "-- character names, places, coinages -- and excluded from every "
        "adjusted spelling rate)", len(recurring), "words", family=FAMILY_LEXICAL,
        sample_size=words_total, min_sample=MIN_SAMPLE_WORDS, unit_sensitive=True,
        evidence=[{"word": w, "count": c} for w, c in excluded_evidence]))
    return out


def _fused_token_findings(analysis: DocumentAnalysis,
                          config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Depends only on SymSpell: its own dictionary supplies both the
    "is this word unknown" candidate filter and the segmentation check, so
    this feature works whether or not pyspellchecker is installed."""

    words_total = analysis.word_count
    language = option(config, "language", DEFAULT_LANGUAGE)
    sym_spell, reason = _symspell(language)
    if sym_spell is None:
        return [unavailable("lexical.mechanical_fused_token_rate", "Fused-token rate",
                            reason, family=FAMILY_LEXICAL)]
    if not words_total:
        return [finding("lexical.mechanical_fused_token_rate", "Fused-token rate", None,
                        "per_1000_words", family=FAMILY_LEXICAL, sample_size=0,
                        min_sample=MIN_SAMPLE_WORDS, warning="no words in text")]

    min_length = int(option(config, "fused_min_length", DEFAULT_FUSED_MIN_LENGTH))
    max_candidates = int(option(config, "fused_max_candidates", DEFAULT_FUSED_MAX_CANDIDATES))
    occurrences = _word_occurrences(analysis)
    doc_counts = Counter(item["lower"] for item in occurrences)
    sym_unknown = {word for word in doc_counts if word not in sym_spell.words}
    candidates = sorted((w for w in sym_unknown if len(w) >= min_length and w.isalpha()),
                        key=lambda w: -doc_counts[w])[:max_candidates]
    flagged = []
    for word in candidates:
        try:
            result = sym_spell.word_segmentation(word, max_edit_distance=0)
        except Exception:  # pragma: no cover - segmentation internal failure
            continue
        parts = result.segmented_string.split()
        # ``distance_sum`` is not "how many letters were corrected" -- it
        # grows by about one for every space SymSpell inserts, so a clean
        # two-word split is reported as distance 1, not 0 (measured:
        # word_segmentation("helloworld", max_edit_distance=0) returns
        # segmented_string="hello world", distance_sum=1). The real
        # confidence check is that every segment it proposes is, on its own,
        # an exact dictionary word -- not a further correction.
        if len(parts) >= 2 and all(len(p) >= 2 and p in sym_spell.words for p in parts):
            flagged.append({"word": word, "segmented": result.segmented_string,
                           "count": doc_counts[word]})
    flagged_count = sum(item["count"] for item in flagged)
    return [finding(
        "lexical.mechanical_fused_token_rate",
        "Fused-token rate (an unrecognized word that SymSpell segments, with no "
        "edits, into two or more real words -- a likely missing space)",
        rate(flagged_count, words_total, 1000.0), "per_1000_words", family=FAMILY_LEXICAL,
        sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        distribution={"candidates_checked": len(candidates)},
        evidence=flagged[:DEFAULT_MAX_REPORTED])]


def _ocr_substitution_findings(analysis: DocumentAnalysis, known_word) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    if known_word is None:
        return [unavailable("lexical.mechanical_ocr_substitution_rate",
                            "Likely OCR-substitution rate",
                            "no spell checker was available to confirm a corrected reading",
                            family=FAMILY_LEXICAL)]
    if not words_total:
        return [finding("lexical.mechanical_ocr_substitution_rate",
                        "Likely OCR-substitution rate", None, "per_1000_words",
                        family=FAMILY_LEXICAL, sample_size=0, min_sample=MIN_SAMPLE_WORDS,
                        warning="no words in text")]

    flagged = []
    for match in _ALNUM_RUN_RE.finditer(analysis.text):
        token = match.group(0)
        if token.isalpha() or token.isdigit() or len(token) < 4:
            continue
        if known_word(token.lower()):
            continue
        candidate = "".join(_DIGIT_TO_LETTER.get(ch, ch) for ch in token)
        if candidate != token and candidate.isalpha() and known_word(candidate.lower()):
            flagged.append({"offset": match.start(), "original": token, "reading": candidate,
                           "pattern": "digit_for_letter"})
    for word in analysis.words:
        lower = word.lower()
        if "rn" not in lower or known_word(lower):
            continue
        candidate = lower.replace("rn", "m")
        if known_word(candidate):
            flagged.append({"original": word, "reading": candidate, "pattern": "rn_for_m"})

    return [finding(
        "lexical.mechanical_ocr_substitution_rate",
        "Likely OCR-substitution rate (digit-for-letter, e.g. w0rld->world, and "
        "rn-for-m confusions, confirmed only when the corrected reading is a "
        "known word and the original is not)",
        rate(len(flagged), words_total, 1000.0), "per_1000_words", family=FAMILY_LEXICAL,
        sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        evidence=flagged[:DEFAULT_MAX_REPORTED])]


# ------------------------------------------------ D. narration/dialogue rules

def _doubled_word_count(view: DocumentAnalysis) -> tuple[int, list[dict[str, Any]]]:
    """Consecutive repeated words, scoped to one sentence at a time.

    Scanning the flat word list would also catch two sentences that happen to
    end and begin on the same word ("...at Gimli. Gimli grunted..."), which is
    two different sentences, not a doubled word -- especially likely right
    where ``analysis.narration`` has spliced dialogue back out, juxtaposing
    text that was never adjacent in the original.
    """

    flagged = []
    for sentence in view.sentences:
        words = _WORD_RE.findall(sentence)
        for index in range(1, len(words)):
            if words[index].lower() == words[index - 1].lower() and words[index].isalpha():
                flagged.append({"word": words[index],
                               "context": " ".join(words[max(0, index - 3):index + 3])})
    return len(flagged), flagged


def _confusion_pair_count(view: DocumentAnalysis) -> tuple[int, list[dict[str, Any]]]:
    text = view.text.lower()
    flagged = []
    for phrase in CONFUSION_PAIRS:
        for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", text):
            flagged.append({"phrase": phrase, "offset": match.start()})
    return len(flagged), flagged


def _channel_findings(analysis: DocumentAnalysis, do_doubled: bool,
                      do_confusion: bool) -> list[dict[str, Any]]:
    out = []
    for channel_name, view in (("narration", analysis.narration), ("dialogue", analysis.dialogue)):
        words_total = view.word_count
        no_channel = None if words_total else f"no {channel_name} found in this text"
        if do_doubled:
            count, evidence = (_doubled_word_count(view) if words_total else (0, []))
            out.append(finding(
                f"lexical.mechanical_doubled_word_rate_{channel_name}",
                f"Doubled consecutive-word rate, {channel_name} only",
                rate(count, words_total, 1000.0) if words_total else None, "per_1000_words",
                family=FAMILY_LEXICAL, channel=channel_name, sample_size=words_total,
                min_sample=MIN_SAMPLE_WORDS, evidence=evidence[:10], warning=no_channel))
        if do_confusion:
            count, evidence = (_confusion_pair_count(view) if words_total else (0, []))
            out.append(finding(
                f"nlp.mechanical_confusion_pair_rate_{channel_name}",
                f"'Should of'-type confusion-pair rate, {channel_name} only "
                f"(a written slip, not a dialect spelling: the spoken contraction "
                f"is \"should've\", not \"should of\")",
                rate(count, words_total, 1000.0) if words_total else None, "per_1000_words",
                family=FAMILY_SYNTAX, channel=channel_name, sample_size=words_total,
                min_sample=MIN_SAMPLE_WORDS, evidence=evidence[:10], warning=no_channel))
    return out


def _sentence_summary_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None,
                               pyspell_known) -> list[dict[str, Any]]:
    """Combined rule-engine bookkeeping, replacing LanguageTool's own summary
    numbers: total rate, which single issue type dominates, how many distinct
    issue types fired, the worst single sentence, and a flagged-sentence rate
    -- computed separately per channel except for the single worst sentence,
    which is reported once for the whole document."""

    recurring_min_count = int(option(config, "recurring_min_count", DEFAULT_RECURRING_MIN_COUNT))
    out: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    combined_categories: Counter = Counter()

    for channel_name, view in (("narration", analysis.narration), ("dialogue", analysis.dialogue)):
        sentences = view.sentences
        if not sentences:
            out.append(finding(f"nlp.mechanical_flagged_sentence_rate_{channel_name}",
                               f"Share of sentences with a mechanical issue, {channel_name} only",
                               None, "percent", family=FAMILY_SYNTAX, channel=channel_name,
                               sample_size=0, min_sample=MIN_SAMPLE_SENTENCES,
                               warning=f"no {channel_name} found in this text"))
            out.append(finding(f"nlp.mechanical_total_issue_rate_{channel_name}",
                               f"Total mechanical-issue rate, {channel_name} only", None,
                               "per_1000_words", family=FAMILY_SYNTAX, channel=channel_name,
                               sample_size=0, min_sample=MIN_SAMPLE_WORDS,
                               warning=f"no {channel_name} found in this text"))
            continue

        py_unknown_local: set[str] = set()
        if pyspell_known is not None:
            occ = [{"lower": w.lower().replace("’", "'")} for w in view.words]
            counts = Counter(item["lower"] for item in occ)
            py_unknown_local = {w for w in counts if w.isalpha() and not pyspell_known(w)}
            recurring = {w for w, c in counts.items() if c >= recurring_min_count}
            py_unknown_local -= recurring

        category_totals: Counter = Counter()
        flagged_sentences = 0
        for sentence in sentences:
            issues = 0
            words = re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*", sentence)
            for index in range(1, len(words)):
                if words[index].lower() == words[index - 1].lower():
                    issues += 1
                    category_totals["doubled_word"] += 1
            lowered = sentence.lower()
            for phrase in CONFUSION_PAIRS:
                found = lowered.count(phrase)
                issues += found
                category_totals["confusion_pair"] += found
            if py_unknown_local:
                unknown_here = sum(1 for w in words if w.lower() in py_unknown_local)
                issues += unknown_here
                category_totals["unknown_word"] += unknown_here
            if issues:
                flagged_sentences += 1
            if issues and (worst is None or issues > worst["count"]):
                worst = {"count": issues, "channel": channel_name,
                        "text": sentence[:280]}

        out.append(finding(
            f"nlp.mechanical_flagged_sentence_rate_{channel_name}",
            f"Share of sentences with at least one mechanical issue, {channel_name} only",
            rate(flagged_sentences, len(sentences), 100.0), "percent", family=FAMILY_SYNTAX,
            channel=channel_name, sample_size=len(sentences), min_sample=MIN_SAMPLE_SENTENCES))
        total_issues = sum(category_totals.values())
        out.append(finding(
            f"nlp.mechanical_total_issue_rate_{channel_name}",
            f"Total mechanical-issue rate (doubled words + confusion pairs + "
            f"non-recurring unknown words), {channel_name} only",
            rate(total_issues, view.word_count, 1000.0), "per_1000_words", family=FAMILY_SYNTAX,
            channel=channel_name, sample_size=view.word_count, min_sample=MIN_SAMPLE_WORDS,
            distribution=dict(category_totals)))
        combined_categories.update(category_totals)

    out.append(finding(
        "nlp.mechanical_max_errors_in_sentence",
        "Most mechanical issues found in any one sentence",
        worst["count"] if worst else 0, "issues", family=FAMILY_SYNTAX,
        sample_size=analysis.sentence_count, min_sample=MIN_SAMPLE_SENTENCES,
        evidence=[worst] if worst else [],
        warning=None if worst else "no mechanical issues were found in any sentence"))

    combined_total = sum(combined_categories.values())
    top_category, top_count = (combined_categories.most_common(1)[0]
                               if combined_categories else (None, 0))
    out.append(finding(
        "nlp.mechanical_rule_concentration",
        "Share of all flagged mechanical issues attributable to the single most "
        "common issue type (doubled words / confusion pairs / unknown words)",
        rate(top_count, combined_total, 100.0) if combined_total else None, "percent",
        family=FAMILY_SYNTAX, sample_size=combined_total, min_sample=MIN_SAMPLE_SENTENCES,
        evidence=[{"category": name, "count": n} for name, n in combined_categories.most_common()],
        warning=None if combined_total else "no mechanical issues were found"))
    out.append(finding(
        "nlp.mechanical_unique_issue_type_count",
        "Distinct mechanical issue types triggered at least once (out of "
        "doubled_word / confusion_pair / unknown_word)",
        sum(1 for count in combined_categories.values() if count), "types",
        family=FAMILY_SYNTAX, sample_size=combined_total, min_sample=0,
        distribution=dict(combined_categories)))
    return out


# ------------------------------------------------------------------- measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    if _feature(config, "typography"):
        out.extend(_typography_findings(analysis))
    if _feature(config, "encoding"):
        out.extend(_encoding_findings(analysis))
    if _feature(config, "encoding_ftfy"):
        out.extend(_ftfy_findings(analysis))
    if _feature(config, "confusables"):
        out.extend(_confusable_findings(analysis))

    pyspell_known, py_reason, symspell_known, sym_reason = _known_word_functions(config)
    known_word = pyspell_known or symspell_known

    if _feature(config, "hyphenation"):
        out.extend(_hyphenation_findings(analysis, known_word))

    do_pyspell = _feature(config, "spelling_pyspellchecker")
    do_symspell = _feature(config, "spelling_symspell")
    do_disagreement = _feature(config, "spelling_disagreement")
    do_likely_typos = _feature(config, "likely_typos")
    # pyspellchecker's unknown-word set is shared by the raw/adjusted rate
    # findings, likely-typo scoring and the recurring-vocabulary report, so it
    # is computed at most once regardless of how many of those are enabled.
    py_unknown: set[str] = set()
    if (do_pyspell or do_disagreement or do_likely_typos) and pyspell_known is not None:
        occurrences = _word_occurrences(analysis)
        py_unknown = {item["lower"] for item in occurrences if not pyspell_known(item["lower"])}

    if do_pyspell or do_symspell or do_disagreement:
        spelling_out, _py_unknown_again, _sym_unknown = _spelling_findings(
            analysis, config, pyspell_known, py_reason, symspell_known, sym_reason,
            do_pyspell or do_disagreement, do_symspell or do_disagreement, do_disagreement)
        # A checker fetched only to satisfy disagreement, not requested on its
        # own, should not publish its unknown-word-rate findings.
        wanted_prefixes = []
        if do_pyspell:
            wanted_prefixes.append("lexical.mechanical_unknown_word_rate_pyspellchecker")
        if do_symspell:
            wanted_prefixes.append("lexical.mechanical_unknown_word_rate_symspell")
        if do_disagreement:
            wanted_prefixes.append("lexical.mechanical_checker_disagreement_rate")
        out.extend(item for item in spelling_out
                  if any(item["metric_id"].startswith(p) for p in wanted_prefixes))

    if do_likely_typos:
        checker_obj, _ = _pyspellchecker(option(config, "language", DEFAULT_LANGUAGE))
        out.extend(_likely_typo_findings(analysis, config, checker_obj, py_reason, py_unknown))

    if _feature(config, "fused_tokens"):
        out.extend(_fused_token_findings(analysis, config))

    if _feature(config, "ocr_substitution"):
        out.extend(_ocr_substitution_findings(analysis, known_word))

    do_doubled = _feature(config, "doubled_words")
    do_confusion = _feature(config, "confusion_pairs")
    if do_doubled or do_confusion:
        out.extend(_channel_findings(analysis, do_doubled, do_confusion))

    if _feature(config, "sentence_summary"):
        out.extend(_sentence_summary_findings(analysis, config, pyspell_known))

    return out
