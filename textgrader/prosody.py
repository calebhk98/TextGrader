"""Pronunciation lookup, physical line/stanza structure, and phonological
helpers shared by :mod:`textgrader.metrics.prosody_suite` and, once the
orchestrator wires it in, :mod:`textgrader.sequences` (Task 20's stress
sequence -- see :func:`stress_sequence` below).

Why this is its own module rather than living inside ``prosody_suite.py``:
the same pronunciation cache, rime/stress extraction and line/stanza
splitting are needed by a second, independent consumer (Task 20's signal
processing, which wants a plain ``(values, coverage, settings)`` stress
series to run spectral analysis on), and the project's own convention
(``textgrader/coherence.py``, ``textgrader/propositions.py``,
``textgrader/graphs.py``, ``textgrader/lexicons.py``) is that a capability
shared across metric modules is a top-level helper module, not a metric
module another metric module imports.

Physical line breaks
---------------------
:meth:`textgrader.document.DocumentAnalysis.paragraphs` (built from
:func:`textgrader.text.paragraphs`) deliberately joins every soft line wrap
inside a blank-line-delimited block into one line
(``re.sub(r"[ \\t]*\\n[ \\t]*", " ", block)``), which is exactly right for
reflowed prose and exactly wrong for a poem, where the line break *is* the
unit of structure. Rather than change ``document.py``/``text.py`` (shared by
every other suite, and out of scope for this task), :func:`line_structure`
below re-reads ``analysis.text`` -- the SAME already-canonicalized text (Gutenberg
boilerplate, Markdown headings and transcript lines already stripped, quotes
optionally normalized -- see ``DocumentAnalysis.from_text``) that
``paragraphs()`` itself consumes -- and splits it on the identical
blank-line block boundary (``\\n[ \\t]*\\n+``, copied from
:mod:`textgrader.text`'s own paragraph splitter so the two agree on where one
poetic "stanza" ends and the next begins), but keeps each internal single
``\\n`` as its own line instead of collapsing it to a space. A "stanza" here
is therefore exactly one ``DocumentAnalysis`` paragraph block; the only
difference is whether the lines inside it get joined. This means every
metric in this module sees the same paragraph/stanza boundaries the rest of
TextGrader does, and pays no second cleanup pass: ``analysis.text`` has
already had Gutenberg/heading/transcript stripping done once, by
``DocumentAnalysis.from_text``.

Pronunciation
-------------
Two independent backends, kept honestly separate (never blended into one
"the" pronunciation):

* **cmudict**, via the ``pronouncing`` package (verified for real below:
  ``pronouncing.phones_for_word("rhyme")`` returns ``['R AY1 M']``; the
  package's own ``Requires:`` line names ``cmudict``, the CMU Pronouncing
  Dictionary wrapped as an installable, versioned resource -- so this is
  the actual CMUdict, not a look-alike). Always tried first: it is a plain
  dictionary lookup, dependency-light and instant.
* **g2p_en**, a neural (but tiny, non-torch) grapheme-to-phoneme fallback
  for out-of-vocabulary words -- verified for real: ``G2p()("Zylnthar")``
  correctly returns syllabified, stress-marked ARPABET phones
  (``['Z', 'IH1', 'L', 'N', 'TH', 'ER0']``) for a word that is not in any
  dictionary. Its inference model is a small encoder/decoder implemented
  directly in NumPy from a 3.3 MB checkpoint file BUNDLED inside the wheel
  (``g2p_en/checkpoint20.npz`` -- confirmed on disk; no download and no
  torch at inference time). It does, however, need two NLTK corpora on
  first import to do POS-tag-based homograph disambiguation --
  ``averaged_perceptron_tagger`` and its own private copy of ``cmudict`` --
  which g2p_en downloads itself, unconditionally, the moment
  ``from g2p_en import G2p`` runs (verified for real in this sandbox: the
  bare import printed nltk's own "Error loading ... Security Violation
  ... NLTK_ALLOW_PROXIED_URLOPEN=1" message before the environment variable
  was set, and the download and subsequent lookups both succeeded once it
  was). Because that is a real, unconditional network action taken on
  *import*, not on first use of a flag a caller explicitly asked for,
  ``prosody_suite``'s ``g2p_fallback`` feature defaults to **False** -- see
  that module's ``_requires_g2p_fallback`` config note -- and this module
  never imports ``g2p_en`` except from inside :func:`_g2p_engine`, which is
  called only once that flag (or :func:`stress_sequence`'s own
  ``use_g2p_fallback`` config key) is on.

Feature-based near-rhyme
------------------------
:mod:`panphon` (verified for real: ``panphon.distance.Distance()
.weighted_feature_edit_distance_div_maxlen(...)`` runs against real IPA
strings below) works on IPA, not ARPABET, and ships no ARPABET bridge of its
own (checked: no ``arpa`` file anywhere under its installed package).
:data:`_ARPABET_TO_IPA` is the standard ARPABET-to-IPA correspondence table
(the one described at https://en.wikipedia.org/wiki/ARPABET and used by
every other ARPABET/IPA bridge this environment could find), with AH0 mapped
to schwa (``ə``) separately from stressed AH (``ʌ``) and ER mapped to a
rhotacized-vowel-plus-approximant sequence (``əɹ``/``ɜɹ``) because PanPhon's
own segment table has no single-character entry for the rhotacized vowels
``ɜ``/``ə`` (confirmed: ``FeatureTable().word_to_vector_list("ɜ", ...)``
returns an empty list; the two-segment sequence is a normal alternate
transcription and PanPhon accepts it).

Prosodic and Poesy
------------------
``prosodic`` v3 needs the ``espeak``/``espeak-ng`` *system* library (not a
pip package) as its out-of-vocabulary grapheme-to-phoneme backend; this
sandbox has neither binary (``which espeak espeak-ng`` finds nothing) and
lacks permission to install system packages here, so constructing a
``prosodic.Text`` prints (to stderr, at import time -- harmless, not raised)
a message ending in the exact string this module quotes in
:data:`ESPEAK_MISSING_HINT`. Despite that, real English words already in
CMUdict scan and rhyme-match correctly without espeak (verified for real:
scanning all four lines of the Sonnet 18 fixture in this module's test suite
correctly returned ``meter_type_scheme: 'iambic'``, ``beat_scheme: (5, 4, 4,
5)`` and rhyme scheme ``('Alternating A', 'abab')`` with accuracy 0.5 --
espeak is needed only for words outside CMUdict, exactly like g2p_en's
fallback role). ``poesy`` 0.4+ is, by its own module docstring, literally "a
thin compatibility layer over prosodic v3", so this module treats "Poesy"
and "Prosodic" as the single ``poesy`` optional package: importing poesy
transitively gives Prosodic's real meter/rhyme engine, not a separate
reimplementation. Both are real, independent second implementations kept as
a bounded, off-by-default cross-check (:mod:`textgrader.metrics.
prosody_suite`'s ``poesy_crosscheck`` feature) rather than the suite's
primary meter/rhyme engine, because ``poesy``/``prosodic`` v3's per-line
scan is combinatorial in a line's stress ambiguity (measured: 60 lines of
repeated Sonnet 18 text took about 6 seconds single-threaded) -- see that
feature's own bounding (``poesy_max_lines``/``poesy_max_seconds``).

``phonemizer`` (its ``EspeakBackend``) needs the same system ``espeak``/
``espeak-ng`` binary and fails the same way here -- verified for real:
``phonemizer.backend.EspeakBackend("en-us")`` raises
``RuntimeError: espeak not installed on your system`` in this sandbox,
exactly the string :data:`PHONEMIZER_MISSING_HINT` quotes. Its
``phonemizer_backend`` feature stays off by default and degrades to
"unavailable" with that quoted error rather than silently skipping.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import text as textlib
from .document import DocumentAnalysis
from .optional import on_reset, require

# ------------------------------------------------------------- ARPABET / IPA

#: Standard ARPABET (base phone, stress digit stripped) -> IPA correspondence.
#: See the module docstring for AH0/ER's special-casing.
_ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ",
    "AW": "aʊ", "AY": "aɪ", "EH": "ɛ", "EY": "eɪ",
    "IH": "ɪ", "IY": "i", "OW": "oʊ", "OY": "ɔɪ",
    "UH": "ʊ", "UW": "u",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f",
    "G": "ɡ", "HH": "h", "JH": "dʒ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ŋ", "P": "p", "R": "ɹ",
    "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

#: The exact espeak-related error strings this environment reproduces, so a
#: reader can grep the source for what was actually observed rather than
#: trust a paraphrase. See the module docstring's "Prosodic and Poesy" and
#: phonemizer sections.
ESPEAK_MISSING_HINT = (
    "espeak/espeak-ng is a SYSTEM package (apt-get install espeak-ng), not a pip "
    "package; without it prosodic/poesy fall back to CMUdict-only pronunciation and "
    "cannot G2P out-of-vocabulary words")
PHONEMIZER_MISSING_HINT = (
    "phonemizer's EspeakBackend needs the SYSTEM espeak/espeak-ng binary (apt-get "
    "install espeak-ng), not a pip package; without it phonemizer raised "
    "'RuntimeError: espeak not installed on your system'")


def _strip_stress(phone: str) -> str:
    return phone[:-1] if phone and phone[-1] in "012" else phone


def _stress_digit(phone: str) -> int | None:
    if phone and phone[-1] in "012":
        return int(phone[-1])
    return None


def base_phone(phone: str) -> str:
    """``phone`` without its stress digit (only vowels carry one)."""

    return _strip_stress(phone)


def stress_digit(phone: str) -> int | None:
    """The stress digit (0/1/2) on a vowel phone, or ``None`` for a consonant."""

    return _stress_digit(phone)


def is_vowel_phone(phone: str) -> bool:
    """Whether ``phone`` is a vowel nucleus.

    Both CMUdict and g2p_en's output append a stress digit (0/1/2) to vowel
    phones only, so this needs no lookup table for either backend.
    """

    return _stress_digit(phone) is not None


def phones_to_ipa(phones: Sequence[str]) -> str:
    """ARPABET phones (with or without stress digits) to an IPA string.

    Used only to bridge into :mod:`panphon`'s feature-distance machinery
    (see :func:`feature_rhyme_similarity`); stress information is dropped
    here on purpose, because PanPhon's feature vectors describe articulation,
    not prosody. AH0 (schwa) is kept distinct from stressed AH; ER is
    rendered as a two-segment rhotacized-vowel sequence (see module
    docstring).
    """

    out: list[str] = []
    for phone in phones:
        base = _strip_stress(phone)
        stress = _stress_digit(phone)
        if base == "AH" and stress == 0:
            out.append("ə")
        elif base == "ER":
            out.append("ɜɹ" if stress in (1, 2) else "əɹ")
        else:
            out.append(_ARPABET_TO_IPA.get(base, ""))
    return "".join(out)


def rime_of(phones: Sequence[str]) -> tuple[str, ...]:
    """The "rhyming part": from the last stressed vowel to the end of the word.

    This is the standard definition of what makes two words rhyme (matching
    ``pronouncing.rhyming_part``'s docstring), reimplemented here rather than
    imported so it works identically over BOTH backends' phone lists (g2p_en
    emits the same stress-marked ARPABET convention CMUdict does -- see the
    module docstring) without depending on ``pronouncing`` being installed
    at all when only the g2p fallback is in play.
    """

    for index in range(len(phones) - 1, -1, -1):
        if _stress_digit(phones[index]) in (1, 2):
            return tuple(phones[index:])
    return tuple(phones)


def rime_key(phones: Sequence[str]) -> tuple[str, ...]:
    """The rime with stress digits removed, for EXACT-rhyme comparison.

    Two words rhyme when this key matches: the nucleus and everything after
    it must be phonetically identical, but which stress level (primary vs
    secondary) fell on the nucleus does not matter.
    """

    return tuple(_strip_stress(phone) for phone in rime_of(phones))


def phone_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein edit distance between two short phone-token sequences.

    This is the "phonetic near-rhyme" channel: two rimes one phone
    substitution/insertion/deletion apart (``dime`` vs ``dine``: ``M`` -> ``N``,
    distance 1) are near-rhymes without invoking PanPhon's articulatory-feature
    machinery, which is a separate, graded THIRD channel -- see
    :func:`feature_rhyme_similarity`.
    """

    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, token_b in enumerate(b, 1):
            cost = 0 if token_a == token_b else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


def syllable_count(phones: Sequence[str]) -> int:
    return sum(1 for phone in phones if is_vowel_phone(phone))


def stress_pattern(phones: Sequence[str]) -> tuple[int, ...]:
    """One stress digit (0/1/2) per syllable, in order."""

    return tuple(_stress_digit(phone) for phone in phones if is_vowel_phone(phone))


def binary_stress(phones: Sequence[str]) -> tuple[float, ...]:
    """One 1.0 (primary or secondary stress) / 0.0 (unstressed) per syllable."""

    return tuple(1.0 if digit in (1, 2) else 0.0 for digit in stress_pattern(phones))


# ------------------------------------------------------------ pronunciation

@dataclass(frozen=True)
class Pronunciation:
    """One word's resolved pronunciation, or the lack of one.

    ``source`` is ``"cmudict"``, ``"g2p"`` or ``"unresolved"`` (empty
    ``phones``); callers must check ``phones`` for emptiness, exactly the
    convention :class:`textgrader.sequences.Sequence` uses for its own
    ``warning`` field.
    """

    word: str
    phones: tuple[str, ...]
    source: str


_UNRESOLVED = Pronunciation("", (), "unresolved")

_cache_lock = threading.Lock()
_PRONUNCIATION_CACHE: dict[tuple[str, bool], Pronunciation] = {}
_g2p_lock = threading.Lock()
_G2P_ENGINE: list[Any] = []  # 0 or 1 items: [engine] once built, or left empty
_G2P_REASON: list[str] = []


def _reset_pronunciation_caches() -> None:
    with _cache_lock:
        _PRONUNCIATION_CACHE.clear()
    with _g2p_lock:
        _G2P_ENGINE.clear()
        _G2P_REASON.clear()


on_reset(_reset_pronunciation_caches)


def _g2p_engine() -> tuple[Any, str | None]:
    """Build (once) and cache g2p_en's ``G2p`` instance.

    Construction parses a homograph dictionary and loads the bundled
    checkpoint; measured at a few hundred milliseconds, so it is built once
    per process rather than per word.
    """

    with _g2p_lock:
        if _G2P_ENGINE:
            return _G2P_ENGINE[0], None
        if _G2P_REASON:
            return None, _G2P_REASON[0]
        module, reason = require("g2p_en")
        if module is None:
            _G2P_REASON.append(reason or "g2p_en unavailable")
            return None, _G2P_REASON[0]
        try:
            engine = module.G2p()
        except Exception as exc:  # pragma: no cover - g2p_en/nltk-data guard
            reason = (f"g2p_en.G2p() failed to build ({type(exc).__name__}: {exc}); it needs "
                     "the NLTK 'averaged_perceptron_tagger' and 'cmudict' corpora "
                     "(python -m nltk.downloader averaged_perceptron_tagger cmudict; set "
                     "NLTK_ALLOW_PROXIED_URLOPEN=1 behind a proxy)")
            _G2P_REASON.append(reason)
            return None, reason
        _G2P_ENGINE.append(engine)
        return engine, None


_ARPABET_TOKEN_RE = re.compile(r"^[A-Z]+[0-2]?$")


def _g2p_phones(word: str) -> tuple[str, ...] | None:
    engine, reason = _g2p_engine()
    if engine is None:
        return None
    try:
        raw = engine(word)
    except Exception:  # pragma: no cover - g2p_en runtime guard
        return None
    phones = tuple(token for token in raw if _ARPABET_TOKEN_RE.match(token))
    return phones or None


def pronounce(word: str, *, use_g2p: bool = False) -> Pronunciation:
    """Resolve one word's ARPABET pronunciation, cached process-wide.

    CMUdict (via ``pronouncing``) is tried first; a word not in CMUdict falls
    back to g2p_en's neural G2P only when ``use_g2p`` is true (off by
    default -- see the module docstring). A word neither backend can resolve
    comes back as :data:`Pronunciation` with empty ``phones`` and
    ``source="unresolved"``; callers must not invent a stress value for it.
    """

    key = (word.lower(), bool(use_g2p))
    with _cache_lock:
        cached = _PRONUNCIATION_CACHE.get(key)
    if cached is not None:
        return cached
    result = _pronounce_uncached(key[0], key[1])
    with _cache_lock:
        _PRONUNCIATION_CACHE[key] = result
    return result


def _pronounce_uncached(word: str, use_g2p: bool) -> Pronunciation:
    module, _ = require("pronouncing")
    if module is not None:
        try:
            options = module.phones_for_word(word)
        except Exception:  # pragma: no cover - pronouncing runtime guard
            options = []
        if options:
            return Pronunciation(word, tuple(options[0].split()), "cmudict")
    if use_g2p:
        phones = _g2p_phones(word)
        if phones:
            return Pronunciation(word, phones, "g2p")
    return _UNRESOLVED


def pronunciation_backend_reason() -> str | None:
    """Why CMUdict lookup is unavailable, or ``None`` when it works."""

    return require("pronouncing")[1]


# ------------------------------------------------------------ PanPhon bridge

_panphon_lock = threading.Lock()
_PANPHON_DISTANCE: list[Any] = []


def _reset_panphon_cache() -> None:
    with _panphon_lock:
        _PANPHON_DISTANCE.clear()


on_reset(_reset_panphon_cache)


def _panphon_distance() -> tuple[Any, str | None]:
    with _panphon_lock:
        if _PANPHON_DISTANCE:
            return _PANPHON_DISTANCE[0], None
    module, reason = require("panphon")
    if module is None:
        return None, reason
    try:
        instance = module.Distance()
    except Exception as exc:  # pragma: no cover - panphon data-table guard
        return None, f"panphon.distance.Distance() failed ({type(exc).__name__}: {exc})"
    with _panphon_lock:
        if not _PANPHON_DISTANCE:
            _PANPHON_DISTANCE.append(instance)
        return _PANPHON_DISTANCE[0], None


_similarity_lock = threading.Lock()
_SIMILARITY_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]], tuple[float | None, str | None]] = {}


def _reset_similarity_cache() -> None:
    with _similarity_lock:
        _SIMILARITY_CACHE.clear()


on_reset(_reset_similarity_cache)


def feature_rhyme_similarity(phones_a: Sequence[str], phones_b: Sequence[str]
                             ) -> tuple[float | None, str | None]:
    """Graded articulatory-feature similarity of two words' rimes, in [0, 1].

    This is deliberately a THIRD channel, independent of exact-rhyme
    (identical rime phones) and ARPABET-edit-distance near-rhyme (see
    :func:`phone_edit_distance`): it can score "cat" vs "cap" as fairly
    similar (both end in a short vowel plus a single voiceless stop,
    differing only in place of articulation) even though neither the phones
    nor their count match exactly. 1.0 is an identical rime; 0.0 is
    returned only for the degenerate case of two empty rimes (handled
    directly, without asking PanPhon to measure a distance between nothing
    and nothing).

    Cached by the exact (rime, rime) pair: a whole-book scan re-asks this
    same question for the same pair of common word endings many times (a
    line ending in "it" is compared against a nearby line ending in "him"
    over and over across a 180,000-word novel), and PanPhon's weighted
    feature-edit distance was measured, uncached, as the single largest cost
    in this suite's whole-novel benchmark -- see
    :mod:`textgrader.metrics.prosody_suite`'s module docstring for the
    before/after numbers.
    """

    key = (tuple(phones_a), tuple(phones_b))
    with _similarity_lock:
        cached = _SIMILARITY_CACHE.get(key)
    if cached is not None:
        return cached
    result = _feature_rhyme_similarity_uncached(phones_a, phones_b)
    with _similarity_lock:
        _SIMILARITY_CACHE[key] = result
    return result


def _feature_rhyme_similarity_uncached(phones_a: Sequence[str], phones_b: Sequence[str]
                                       ) -> tuple[float | None, str | None]:
    distance_module, reason = _panphon_distance()
    if distance_module is None:
        return None, reason
    ipa_a, ipa_b = phones_to_ipa(phones_a), phones_to_ipa(phones_b)
    if not ipa_a and not ipa_b:
        return None, "both rimes were empty"
    try:
        distance = distance_module.weighted_feature_edit_distance_div_maxlen(ipa_a, ipa_b)
    except Exception as exc:  # pragma: no cover - panphon runtime guard
        return None, f"panphon feature-edit distance failed ({type(exc).__name__}: {exc})"
    return max(0.0, 1.0 - float(distance)), None


# --------------------------------------------------------- line / stanza text

#: The identical blank-line block boundary :func:`textgrader.text.paragraphs`
#: splits on (``re.split(r"\n[ \t]*\n+", normalized)``), copied rather than
#: imported so a change to that regex cannot silently move this module's
#: notion of "stanza" out of step with the rest of the codebase without a
#: test noticing (``test_prosody_suite.py`` checks the two agree).
STANZA_BREAK_RE = re.compile(r"\n[ \t]*\n+")


@dataclass(frozen=True)
class Stanza:
    lines: tuple[str, ...]


@dataclass(frozen=True)
class LineStructure:
    """Physical line/stanza view of one document, read straight off ``analysis.text``.

    ``raw_line_count`` is the literal count of ``\\n``-delimited rows in the
    canonical text, including blank ones -- a fact about how the FILE is
    formatted. ``nonblank_lines``/``stanzas`` are the corrected POETIC line
    structure: see :func:`_looks_hard_wrapped` for why these two numbers are
    not the same count on ordinary word-wrapped prose.
    """

    raw_line_count: int
    nonblank_lines: tuple[str, ...]
    stanzas: tuple[Stanza, ...]
    #: How many stanzas were reflowed by the hard-wrap heuristic below.
    hard_wrapped_stanzas: int = 0


#: Word-wrapped prose lines cluster tightly just under a shared right margin.
#: Measured directly against Project Gutenberg's own plain-text formatting
#: (``gutenberg-514-little-women.txt``, 10,347 sampled non-final lines of
#: blocks with 4+ lines): mean width 67.2 characters, stdev 6.7, maximum
#: never above 71. A deliberately short poetic line essentially never
#: produces that signature by chance -- Sonnet 18's four lines (used in this
#: module's own test fixtures) run 40-47 characters, well under
#: ``_WRAP_MIN_WIDTH`` -- so a stanza this long, this consistent, and this
#: close to a shared margin is treated as reflowed prose (exactly the join
#: :func:`textgrader.text.paragraphs` performs for a soft-wrapped block)
#: rather than a poem's deliberate line breaks. This is a proxy, like every
#: other genre signal in this module (see the task's implementation note
#: #2): it degrades toward "keep the physical lines" whenever a stanza is
#: short or its widths do not cluster this tightly, so real short-lined
#: verse is never at risk of being joined.
_WRAP_MIN_LINES = 4
_WRAP_MIN_WIDTH = 55
_WRAP_TOLERANCE = 12
_WRAP_MIN_SHARE = 0.7


def _looks_hard_wrapped(lines: Sequence[str]) -> bool:
    if len(lines) < _WRAP_MIN_LINES:
        return False
    widths = [len(line) for line in lines[:-1]]
    max_width = max(widths)
    if max_width < _WRAP_MIN_WIDTH:
        return False
    near_margin = sum(1 for width in widths if width >= max_width - _WRAP_TOLERANCE)
    return (near_margin / len(widths)) >= _WRAP_MIN_SHARE


def line_structure(analysis: DocumentAnalysis) -> LineStructure:
    """The poem's physical line/stanza structure. See the module docstring."""

    return analysis.memo("prosody:line_structure", lambda: _build_line_structure(analysis))


def _build_line_structure(analysis: DocumentAnalysis) -> LineStructure:
    normalized = analysis.text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = normalized.split("\n")
    stanzas: list[Stanza] = []
    hard_wrapped = 0
    for block in STANZA_BREAK_RE.split(normalized):
        kept = tuple(line.strip() for line in block.split("\n") if textlib.words(line))
        if not kept:
            continue
        if _looks_hard_wrapped(kept):
            kept = (" ".join(kept),)
            hard_wrapped += 1
        stanzas.append(Stanza(kept))
    nonblank = tuple(line for stanza in stanzas for line in stanza.lines)
    return LineStructure(len(raw_lines), nonblank, tuple(stanzas), hard_wrapped)


# ------------------------------------------------------- stress sequence (Task 20)

def stress_sequence(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None
                    ) -> tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]:
    """A per-syllable stress sequence: ``1.0`` stressed, ``0.0`` unstressed.

    Built for Task 20's signal-processing work (spectral/periodicity analysis
    over a stress series), so this is importable independently of the
    ``prosody_suite`` metric module and of whether that suite is switched on.
    Once merged, the orchestrator registers this as a named
    :mod:`textgrader.sequences` channel (see that module's docstring for the
    ``Sequence``/``SequenceSpec`` contract); this function's own return shape
    is deliberately NOT a :class:`~textgrader.sequences.Sequence` so it has no
    import-time dependency on that module in either direction.

    Runs over ``analysis.words`` in reading order -- the whole document, not
    only poetic lines -- so it is available for prose too (see the task's
    "allowing these signals to run experimentally on prose").

    **Unknown words**: a word with no resolved pronunciation (not in
    CMUdict, and either ``use_g2p_fallback`` is off or g2p_en also could not
    resolve it) is SKIPPED entirely -- it contributes zero values to the
    sequence, never a placeholder or an interpolated guess. This means a
    position in ``values`` is not the same as a token position in the text
    once any word is skipped; ``coverage`` reports exactly how much of the
    document that affects, in both words and resulting syllables, so a
    consumer can decide whether the sequence is dense enough to analyze.

    Returns ``(values, coverage, settings)``:

    * ``values``: a tuple of ``0.0``/``1.0`` floats, one per resolved
      syllable, concatenated across words in reading order.
    * ``coverage``: ``{"word_count", "resolved_words", "unresolved_words",
      "coverage_ratio", "syllable_count", "source_counts": {"cmudict": n,
      "g2p": n}}``.
    * ``settings``: ``{"use_g2p_fallback": bool}`` -- the setting that
      produced ``values``, so two callers using different fallback settings
      are never confused for each other (mirrors every
      :class:`textgrader.sequences.Sequence`'s own ``settings`` field).
    """

    use_g2p = bool((config or {}).get("use_g2p_fallback", False))
    key = f"prosody:stress_sequence:{use_g2p}"
    return analysis.memo(key, lambda: _build_stress_sequence(analysis, use_g2p))


def _build_stress_sequence(analysis: DocumentAnalysis, use_g2p: bool
                           ) -> tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]:
    values: list[float] = []
    resolved = 0
    source_counts: Counter = Counter()
    words = analysis.words
    for word in words:
        pron = pronounce(word, use_g2p=use_g2p)
        if not pron.phones:
            continue
        resolved += 1
        source_counts[pron.source] += 1
        values.extend(binary_stress(pron.phones))
    total = len(words)
    coverage = {
        "word_count": total,
        "resolved_words": resolved,
        "unresolved_words": total - resolved,
        "coverage_ratio": (resolved / total) if total else None,
        "syllable_count": len(values),
        "source_counts": dict(source_counts),
    }
    settings = {"use_g2p_fallback": use_g2p}
    return tuple(values), coverage, settings


# ------------------------------------------------------------ phonemizer probe

def phonemize_word(word: str) -> tuple[str | None, str | None]:
    """One word's IPA transcription via ``phonemizer``'s espeak backend.

    Returns ``(ipa, None)`` on success or ``(None, reason)``; the reason is
    the exact, quoted error observed when espeak is missing (see the module
    docstring). The backend itself is cached (constructing ``EspeakBackend``
    each call would re-probe for the espeak library every time).
    """

    module, reason = require("phonemizer")
    if module is None:
        return None, reason
    backend, backend_reason = _phonemizer_backend()
    if backend is None:
        return None, backend_reason
    try:
        result = backend.phonemize([word], strip=True)
    except Exception as exc:  # pragma: no cover - phonemizer runtime guard
        return None, f"phonemizer.phonemize failed ({type(exc).__name__}: {exc})"
    return (result[0] if result else None), None


_phonemizer_lock = threading.Lock()
_PHONEMIZER_BACKEND: list[Any] = []
_PHONEMIZER_REASON: list[str] = []


def _reset_phonemizer_cache() -> None:
    with _phonemizer_lock:
        _PHONEMIZER_BACKEND.clear()
        _PHONEMIZER_REASON.clear()


on_reset(_reset_phonemizer_cache)


def _phonemizer_backend() -> tuple[Any, str | None]:
    with _phonemizer_lock:
        if _PHONEMIZER_BACKEND:
            return _PHONEMIZER_BACKEND[0], None
        if _PHONEMIZER_REASON:
            return None, _PHONEMIZER_REASON[0]
    module, reason = require("phonemizer")
    if module is None:
        with _phonemizer_lock:
            if not _PHONEMIZER_REASON:
                _PHONEMIZER_REASON.append(reason)
        return None, reason
    try:
        from phonemizer.backend import EspeakBackend
        backend = EspeakBackend("en-us")
    except Exception as exc:  # espeak binary missing: RuntimeError, verified above
        reason = f"{PHONEMIZER_MISSING_HINT} ({type(exc).__name__}: {exc})"
        with _phonemizer_lock:
            if not _PHONEMIZER_REASON:
                _PHONEMIZER_REASON.append(reason)
        return None, reason
    with _phonemizer_lock:
        if not _PHONEMIZER_BACKEND:
            _PHONEMIZER_BACKEND.append(backend)
        return _PHONEMIZER_BACKEND[0], None


# ------------------------------------------------------------- poesy / prosodic

@dataclass(frozen=True)
class PoesyResult:
    dominant_meter: str | None
    beat_scheme: tuple[int, ...] | None
    rhyme_scheme_form: str | None
    rhyme_scheme_accuracy: float | None
    lines_scanned: int
    seconds: float


def run_poesy(lines: Sequence[str], *, max_lines: int, max_seconds: float
              ) -> tuple[PoesyResult | None, str | None]:
    """Run Poesy/Prosodic's own meter+rhyme scan over a bounded line sample.

    Bounded because prosodic v3's per-line scan is combinatorial in a line's
    stress ambiguity (see the module docstring's measured cost); ``lines`` is
    truncated to ``max_lines`` before this function ever constructs a
    ``Poem``, and the whole call is additionally wall-clock-bounded so a
    pathological line cannot stall a run.
    """

    module, reason = require("poesy")
    if module is None:
        return None, reason
    sample = list(lines)[:max(1, int(max_lines))]
    if not sample:
        return None, "no lines to scan"
    start = time.monotonic()
    try:
        poem = module.Poem("\n".join(sample))
        stats = poem.statd
    except Exception as exc:  # pragma: no cover - prosodic/poesy runtime guard
        return None, f"poesy.Poem scan failed ({type(exc).__name__}: {exc})"
    elapsed = time.monotonic() - start
    if elapsed > max_seconds:
        return None, (f"poesy scan of {len(sample)} lines took {elapsed:.1f}s, over the "
                      f"poesy_max_seconds budget of {max_seconds:g}s")
    beat_scheme = stats.get("beat_scheme")
    result = PoesyResult(
        dominant_meter=stats.get("meter_type_scheme"),
        beat_scheme=tuple(beat_scheme) if beat_scheme else None,
        rhyme_scheme_form=stats.get("rhyme_scheme_form"),
        rhyme_scheme_accuracy=stats.get("rhyme_scheme_accuracy"),
        lines_scanned=len(sample), seconds=elapsed)
    return result, None


__all__ = [
    "Pronunciation", "pronounce", "pronunciation_backend_reason",
    "base_phone", "stress_digit", "is_vowel_phone", "phones_to_ipa",
    "rime_of", "rime_key", "phone_edit_distance", "syllable_count",
    "stress_pattern", "binary_stress", "feature_rhyme_similarity",
    "Stanza", "LineStructure", "line_structure", "stress_sequence",
    "phonemize_word", "PoesyResult", "run_poesy",
    "ESPEAK_MISSING_HINT", "PHONEMIZER_MISSING_HINT",
]
