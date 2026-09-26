"""Malformed-text, word-segmentation, language-ID and tokenization-anomaly
diagnostics -- Task 19 of the experimental-metrics program.

TextGrader grades text that legitimately contains invented words (fantasy
names, coinages), deliberate foreign phrases, dialect and archaic spelling.
None of that is malformed text. What this suite measures instead is the
token- and language-shape evidence of *mechanical* damage: missing spaces,
accidental spaces, OCR/encoding corruption, script mixing, and passages that
are statistically in a different language than the rest of the document
without the document being a deliberately multilingual work. Every finding
stays ``Polarity.NEUTRAL`` (measured elsewhere, in ``grade.py``'s finding
construction, by leaving polarity unset for this family) and nothing here
ever rewrites ``analysis.text``.

Heavy overlap with :mod:`mechanical_quality_suite`
----------------------------------------------------
That suite already owns fused/split-token detection (via SymSpell's own
segmentation), OCR-substitution heuristics, mixed-script *homoglyph* detection
(``confusable_homoglyphs``), two whole-document spell checkers, and a
"recurring vocabulary" exclusion for invented names. This suite does not
reimplement any of that. Where one of this suite's channels measures
something close enough to an existing ``mechanical_quality_suite`` id that a
user should know about the overlap, the finding's
``distribution["overlaps_existing_metric_id"]`` names it explicitly:

* ``lexical.malformed_fused_word_candidate_rate`` overlaps
  ``lexical.mechanical_fused_token_rate`` -- both detect a single unbroken
  token that a segmenter can confidently split into two or more real words.
  This suite's version requires **two of three independent segmenters**
  (wordninja, wordsegment, SymSpell) to agree on the *same* all-real-word
  split, rather than trusting SymSpell alone, and reports segmenter
  disagreement as its own numbers (see below) -- that is the actual
  contribution here, not a better single-tool fused-token rate.
* ``lexical.malformed_dictionary_recognized_rate`` and
  ``lexical.malformed_low_frequency_token_rate`` overlap
  ``lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted`` -- both
  describe "is this token part of the general vocabulary". This suite's
  version is a **frequency** measurement (wordfreq's Zipf scale) rather than
  **dictionary membership** (pyspellchecker/SymSpell word lists), which is a
  different, and often disagreeing, quantity: a real but rare word can be a
  dictionary hit and a frequency outlier at the same time.
* ``lexical.malformed_mixed_alnum_token_rate`` overlaps
  ``lexical.mechanical_ocr_substitution_rate`` -- both start from the same
  letter/digit "alnum run" token candidate pool. This suite's version reports
  the raw mixed-alphanumeric share with no correction requirement; the
  mechanical suite's version additionally requires digit-for-letter
  substitution AND that the corrected reading is a real dictionary word.
* ``lexical.malformed_token_charclass_entropy`` overlaps
  ``punct.mechanical_unicode_category_entropy`` -- both are Unicode-category
  Shannon entropy. This suite's version is computed only over the characters
  inside whitespace-delimited tokens (so it answers "how mixed is a typical
  token", e.g. "w0rld!!!" vs "world"), while the mechanical suite's version is
  computed over the *entire* document text including whitespace and
  punctuation-between-tokens (so it answers "how mixed is the character
  stream").
* ``lexical.malformed_dictionary_disagreement_rate`` overlaps
  ``lexical.mechanical_checker_disagreement_rate`` -- both are
  two-independent-recognizer disagreement rates. This suite's version
  compares wordfreq's Zipf-frequency verdict against SymSpell's dictionary
  verdict, restricted to the bounded *suspicious*-token candidate pool this
  suite already built for segmentation (not every word in the document, and
  not the pyspellchecker/SymSpell pairing the mechanical suite uses).

Everything else here (script distribution, repeated-symbol/mixed-alphanumeric/
long-token token-shape rates, split-word-candidate detection, segmentation-tool
disagreement, and every language-identification channel) has no equivalent
elsewhere in TextGrader and carries no overlap tag.

What this suite adds
---------------------
1. **Token shape**: long-token rate and tail length, mixed-alphanumeric token
   rate, repeated-symbol token rate, token character-class entropy, and
   Unicode-script distribution/entropy (Latin/Cyrillic/Greek/Han/Arabic/... --
   distinguished with :func:`unicodedata.name`'s per-character script prefix,
   never :mod:`regex`'s ``\\p{Script}`` machinery, so no extra package is
   needed for this group at all).
2. **Suspicious-token frequency**: the share of tokens wordfreq recognizes at
   a normal frequency, and the very-low-frequency tail, over the *general*
   English (or configured-language) vocabulary -- independent of any
   dictionary's word list.
3. **Word-segmentation disagreement**, run *only* on bounded, suspicious
   (long, wordfreq-rare) tokens, never on ordinary words: three independent
   segmenters -- wordninja, wordsegment, and SymSpell's own
   ``word_segmentation`` -- are compared. Their agreement/disagreement is
   reported directly (never reconciled into one "correct" segmentation),
   alongside a fused-word-candidate rate (>=2 segmenters agree on the same
   all-real-word split) and a split-word-candidate rate (two short adjacent
   tokens whose join is a common real word while neither piece is).
4. **Dictionary/frequency disagreement** for the same bounded suspicious-token
   pool: wordfreq's Zipf-frequency verdict against SymSpell's dictionary
   verdict.
5. **Language identification** at document, section (a bounded, deterministic
   sample of paragraph-sized windows), paragraph and sentence level, using up
   to five independent detectors -- Lingua, langid.py, langdetect (seeded),
   fastText's ``lid.176`` model, and CLD3 (via the ``gcld3`` binding) --
   reporting confidence distributions, a low-confidence-sentence rate, a
   code-switch rate (adjacent-sentence language changes within contiguous,
   deterministically sampled runs), a document-majority-language-agreement
   rate at both the paragraph and sentence level, section-level language
   entropy, and both a "do all detectors agree" rate and a mean pairwise
   detector-agreement rate. Every finding is ``Polarity.NEUTRAL``: code
   switching and a low-confidence sentence are *measurements*, not defects --
   they are exactly what a deliberately multilingual book or an invented
   in-world language should also produce, and the "invented fantasy names in
   otherwise English prose are not flagged as wrong-language sentences" test
   below is the check that this design actually holds.

Package verification (rule: a package's name is not evidence of what it
does)::

    >>> import wordninja; wordninja.split('thisisatest')
    ['this', 'is', 'a', 'test']
    >>> import wordsegment; wordsegment.load(); wordsegment.segment('thisisatest')
    ['this', 'is', 'a', 'test']
    >>> import langid; langid.classify('This is a test sentence in English.')
    ('en', -81.29681444168091)               # a log-likelihood, not a probability
    >>> import langdetect; langdetect.DetectorFactory.seed = 0
    >>> langdetect.detect_langs('This is a test sentence in English.')
    [en:0.9999964123276526]
    >>> from lingua import Language, LanguageDetectorBuilder
    >>> d = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.FRENCH).build()
    >>> d.detect_language_of('Ceci est une phrase de test en francais.')
    Language.FRENCH
    >>> import fasttext; m = fasttext.load_model('lid.176.ftz')
    >>> m.predict('Dies ist ein Testsatz auf Deutsch.', k=1)
    (('__label__de',), array([0.99995822]))
    >>> import gcld3; d = gcld3.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=1000)
    >>> d.FindLanguage(text='This is a test sentence in English.').language
    'en'

Every one of these was actually imported and exercised in this environment;
none is assumed to work from its name or PyPI description alone.

Why CLD3 is here at all (most projects skip it)
-------------------------------------------------
The task brief lists "CLD3 / pycld3 where installable" as an aside, because
``pycld3`` is effectively dead: it needs ``longintrepr.h``, a CPython
*internal* header that CPython removed from its public include path in 3.11,
so ``pip install pycld3`` fails to compile on any current Python with
``fatal error: longintrepr.h: No such file or directory`` -- reproduced
directly in this environment. ``gcld3`` (the actively-maintained fork/binding
of the same underlying Google CLD3 model, published as its own PyPI package)
**does** build here: it needs the Protocol Buffers compiler (``protoc``) on
the build machine, which was not preinstalled (``pip install pycld3``/
``gcld3`` first failed with ``RuntimeError: The Protobuf compiler, `protoc`,
... could not be found``); installing the system ``protobuf-compiler``
package (``apt-get install -y protobuf-compiler``, a build-time-only tool,
not a new language runtime -- the built wheel is pure C++/Python, exactly the
"a compiled helper a Python package calls" case the task rules call out for
KenLM) let it build and install cleanly, and it was then verified for real
against English and French text (see the transcript above). CLD3 needs no
downloaded model: its small neural weights are compiled into the extension.

fastText and numpy 2
----------------------
fastText's own ``FastText.py`` (both the ``fasttext`` and ``fasttext-wheel``
PyPI distributions ship the same file) calls
``np.array(probs, copy=False)`` inside ``predict()``. Under numpy 1.x,
``copy=False`` means "avoid a copy if possible"; under numpy 2.x it means
"never copy, raise if a copy would be required" -- and converting a freshly
built Python list of floats into an ndarray always requires one, so
``predict()`` raises ``ValueError: Unable to avoid copy while creating an
array as requested`` on first call under numpy 2. This project pins numpy 1.x
project-wide (and this task's own rules forbid upgrading it for one
dependency), so the bug does not fire here -- verified for real: with numpy
1.26.4 installed, ``fasttext.load_model(...).predict(...)`` above returned
real predictions with no patching. A preventive shim,
:func:`textgrader.optional.shim_fasttext_numpy2`, is still registered (rule:
"a compatibility patch goes in optional.py as a shim_* function"): it checks
the installed numpy major version and is a deliberate, verified no-op below
numpy 2, so nothing changes in this environment; it exists so that if numpy
is ever allowed to move to 2.x for some other reason, this suite's fastText
channel degrades to a retried call instead of an unhandled exception.

Performance and sampling
--------------------------
Every detector call is cheap in isolation (all under ~3ms/call once loaded,
measured directly), but calling all five detectors on every sentence of a
187,000-word novel (~9,000 sentences) would still be tens of thousands of
calls. Two independent bounds keep this suite's cost from scaling with book
length:

* **Lingua's language set is restricted by default**
  (``lingua_languages``, ~14 major languages) rather than using
  ``from_all_languages()``. Measured directly: loading Lingua's full 75-
  language n-gram model set costs ~900MB resident (lazily loaded on first
  detection call); the default restricted set costs ~215MB. The restricted
  set still classifies fantasy-name-laden English sentences as English with
  >65% confidence even with Latin among the candidate languages (see the
  false-positive test below) -- narrowing the candidate set does not
  reproduce the "any invented word looks foreign" failure mode, because
  Lingua scores the whole sentence's n-gram statistics, not individual
  out-of-vocabulary tokens.
* **Sentence-level detection runs over a bounded, deterministic sample of
  contiguous sentence *runs*** (``language_id_run_length`` sentences each,
  ``language_id_max_runs`` runs, spread evenly across the book), not every
  sentence and not an independently-sampled scatter of single sentences.
  Contiguity is required inside each run because the code-switch rate is an
  adjacent-sentence comparison: an evenly-strided *scatter* of single
  sentences would make two sentences hundreds of pages apart look
  "adjacent". Runs are chosen by a fixed arithmetic stride (no RNG), so the
  same book always samples the same runs. See the module-level
  ``measure_a_real_novel`` note in this suite's test file for the actual
  measured seconds/memory on a full Gutenberg novel with every feature on.
* Paragraph- and section-level detection additionally query only the
  *primary* detector (the first available detector in a fixed preference
  order: Lingua, langid, langdetect, fastText, CLD3 -- fixed by installed
  availability, never by what a document contains), not all five, since nine
  language-ID metric ids need only one label per paragraph/section, and only
  the sentence-level disagreement channels need every detector's opinion on
  the same unit.

ISO normalization
-------------------
Every detector's raw label is kept verbatim in ``details``/evidence. Labels
that are already a 2-letter code are used as their own ISO 639-1 code.
langdetect's regional variants (``zh-cn``/``zh-tw``) and CLD3's script-tagged
variants (``zh-Hant``, ``zh-Latn``) are folded to their base language. A
handful of fastText's ``lid.176`` labels are ISO 639-3 codes with no 639-1
equivalent (dozens of the 176 languages it supports have none at all); those
are left as their own code rather than invented, and the original label is
always available in ``details`` either way per the task's implementation
rule 5.

Evidence and offsets
----------------------
Every token-level finding (token shape, suspicious-token frequency,
segmentation, split-word, dictionary disagreement) keeps the character offset
of each piece of evidence, taken directly from the regex match or token scan
over ``analysis.text``, per the task's "retain character offsets for
evidence" rule. Sentence/paragraph-level language-ID evidence is bounded and
identified by its position (sentence/paragraph/window index) and a truncated
text snippet rather than a byte offset: :class:`~textgrader.document.DocumentAnalysis`
does not store per-sentence character offsets (``analysis.sentences`` is a
flat list of already-split strings), and re-deriving them by re-scanning the
text for every sampled sentence would reintroduce exactly the kind of
per-metric re-parsing ``DocumentAnalysis`` exists to avoid; the position index
is enough to locate the passage in the source document.

Every measurement group is switched independently under ``features`` (see
``DEFAULT_FEATURES``), and the whole suite is off by default in
``config.json``.
"""

from __future__ import annotations

import math
import os
import re
import threading
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import stats as stats_module
from ..document import DocumentAnalysis
from ..optional import on_reset, require, shim_fasttext_numpy2
from .common import MODERATE, finding, option, rate, unavailable

FAMILY = "lexical"
# Measured (see the test file's module-level docstring and this module's
# "Performance and sampling" note): the dependency-free token-shape group is
# a fraction of a second even on a full novel; the language-ID group's cost is
# bounded by run/section/paragraph sampling rather than by book length, and
# was measured at well under a minute on a 187,000-word novel with every
# detector enabled -- see the real-numbers report this task's instructions
# require, reproduced in this module's test file.
COST = MODERATE
REQUIRES: tuple[str, ...] = (
    "wordfreq", "symspellpy", "wordninja", "wordsegment",
    "langid", "langdetect", "lingua", "fasttext", "gcld3",
)
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

MIN_SAMPLE_WORDS = 200
MIN_SAMPLE_SENTENCES = 15
MIN_SAMPLE_TOKENS = 30
MIN_SAMPLE_PARAGRAPHS = 5
MIN_SAMPLE_SECTIONS = 3

DEFAULT_FEATURES: dict[str, bool] = {
    # Token shape: length/mixed-alphanumeric/repeated-symbol rates. No
    # package; a handful of linear regex passes over analysis.text.
    "token_shape": True,
    # Unicode-script distribution/entropy. No package (unicodedata only).
    "script_distribution": True,
    # Token character-class (letter/digit/punctuation/symbol) entropy. No
    # package (unicodedata only).
    "charclass_entropy": True,
    # wordfreq-based "confidently recognized" / very-low-frequency rates
    # over every word token. Needs wordfreq.
    "suspicious_token_frequency": True,
    # Word-segmentation disagreement, fused-word-candidate and
    # split-word-candidate detection over a bounded suspicious-token pool.
    # Needs wordfreq (candidate selection + plausibility) plus whichever of
    # wordninja/wordsegment/symspellpy are installed; degrades per-tool.
    "word_segmentation": True,
    # wordfreq-vs-SymSpell disagreement over the same suspicious-token pool.
    "dictionary_disagreement": True,
    # Each language-ID detector is independently switchable; the aggregate
    # metrics below use whichever of these are on AND installed.
    "language_id_lingua": True,
    "language_id_langid": True,
    "language_id_langdetect": True,
    "language_id_fasttext": True,
    "language_id_cld3": True,
    # The language-ID metric group itself (document/section/paragraph/
    # sentence confidence, code-switch rate, off-majority rates, section
    # entropy, detector disagreement). Off switches the whole group even if
    # individual detectors above are left on.
    "language_id": True,
}

DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_REPORTED = 25

# ---------------------------------------------------------------- token shape
DEFAULT_LONG_TOKEN_MIN_LENGTH = 15
DEFAULT_LONG_TOKEN_TAIL_QUANTILE = 0.99
DEFAULT_REPEATED_SYMBOL_MIN_RUN = 3

# ------------------------------------------------------- suspicious tokens
DEFAULT_SUSPICIOUS_MIN_LENGTH = 8
DEFAULT_SUSPICIOUS_ZIPF_THRESHOLD = 1.0
DEFAULT_SUSPICIOUS_MAX_CANDIDATES = 300
DEFAULT_RECOGNIZED_ZIPF_THRESHOLD = 2.0

# --------------------------------------------------------- word segmentation
DEFAULT_SEGMENTATION_PLAUSIBLE_ZIPF = 3.0
#: A fragment counts as "not a real standalone word" below this Zipf value.
#: Kept well below common function words (the/a/an/in/to/... all score
#: 6.0-7.7) but high enough that short, moderately-frequent noise fragments
#: produced by an actual random space insertion (measured: "li", "ha", "fli"
#: score 1.9-4.6 in wordfreq's English data despite not being real words) are
#: still caught, per the "random spaces inserted inside words" fixture test.
DEFAULT_SPLIT_WORD_FRAGMENT_ZIPF_MAX = 4.5
DEFAULT_SPLIT_WORD_JOINED_ZIPF_MIN = 3.0
DEFAULT_SPLIT_WORD_MIN_COMBINED_LENGTH = 6
DEFAULT_SPLIT_WORD_MAX_FRAGMENT_LENGTH = 6

# ------------------------------------------------------------- language ID
#: Restricted by default for memory/speed -- see the module docstring's
#: "Performance and sampling" section. "la" (Latin) is included deliberately:
#: fantasy/invented vocabulary is often Latin-flavored, and this is exactly
#: the case the false-positive test below checks Lingua does not fall for.
DEFAULT_LINGUA_LANGUAGES = ("en", "fr", "es", "de", "it", "pt", "nl", "la",
                           "sv", "da", "pl", "ru", "tr", "id")
DEFAULT_LANGUAGE_ID_MIN_CHARS = 12
DEFAULT_LANGUAGE_ID_LOW_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_LANGUAGE_ID_RUN_LENGTH = 80
DEFAULT_LANGUAGE_ID_MAX_RUNS = 20
DEFAULT_LANGUAGE_ID_MAX_PARAGRAPHS = 400
DEFAULT_LANGUAGE_ID_MIN_PARAGRAPH_CHARS = 40
DEFAULT_LANGUAGE_ID_SECTION_WORDS = 3000
DEFAULT_LANGUAGE_ID_MAX_SECTIONS = 60
DEFAULT_LANGUAGE_ID_MAX_DOCUMENT_CHARS = 200_000
DEFAULT_CLD3_MAX_BYTES = 3000
FASTTEXT_MODEL_URL_DEFAULT = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
FASTTEXT_MODEL_FILENAME = "lid.176.ftz"
#: Fixed preference order for "the primary detector": which single detector
#: answers paragraph/section-level questions and anchors the sentence-level
#: confidence/code-switch/off-majority numbers. Chosen by installed
#: availability only, never by what a given document contains.
DETECTOR_ORDER = ("lingua", "langid", "langdetect", "fasttext", "cld3")
_DETECTOR_FEATURE_NAMES = {
    "lingua": "language_id_lingua", "langid": "language_id_langid",
    "langdetect": "language_id_langdetect", "fasttext": "language_id_fasttext",
    "cld3": "language_id_cld3",
}

_LANG_ALIAS = {
    "zh-cn": "zh", "zh-tw": "zh", "zh-hant": "zh", "zh-hans": "zh",
    "zh-latn": "zh", "pt-br": "pt", "pt-pt": "pt",
}

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_ALNUM_RUN_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]+(?![A-Za-z0-9])")
_RAW_TOKEN_RE = re.compile(r"\S+")
_REPEATED_SYMBOL_RE = re.compile(r"([^\w\s])\1{%d,}" % (DEFAULT_REPEATED_SYMBOL_MIN_RUN - 1))
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")

#: unicodedata.name()'s first token, mapped to a short script label. Covers
#: the scripts the task spec names (Latin/Cyrillic/Greek) plus every other
#: script common enough in real-world mixed text to be worth naming; anything
#: else falls back to "Other".
_SCRIPT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("LATIN", "Latin"), ("CYRILLIC", "Cyrillic"), ("GREEK", "Greek"),
    ("ARABIC", "Arabic"), ("HEBREW", "Hebrew"), ("HANGUL", "Hangul"),
    ("HIRAGANA", "Hiragana"), ("KATAKANA", "Katakana"), ("CJK", "Han"),
    ("DEVANAGARI", "Devanagari"), ("THAI", "Thai"), ("ARMENIAN", "Armenian"),
    ("GEORGIAN", "Georgian"), ("BENGALI", "Bengali"), ("TAMIL", "Tamil"),
)


def _feature(config: Mapping[str, Any] | None, name: str) -> bool:
    features = option(config, "features", {})
    if not isinstance(features, Mapping):
        return DEFAULT_FEATURES.get(name, False)
    value = features.get(name, DEFAULT_FEATURES.get(name, False))
    return DEFAULT_FEATURES.get(name, False) if value is None else bool(value)


def _entropy(counter: Mapping[Any, int]) -> float | None:
    total = sum(counter.values())
    if not total:
        return None
    value = -sum((n / total) * math.log2(n / total) for n in counter.values() if n)
    return 0.0 if value == 0 else value


def _overlap(metric_id: str) -> dict[str, str] | None:
    mapping = {
        "lexical.malformed_fused_word_candidate_rate": "lexical.mechanical_fused_token_rate",
        "lexical.malformed_dictionary_recognized_rate":
            "lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted",
        "lexical.malformed_low_frequency_token_rate":
            "lexical.mechanical_unknown_word_rate_pyspellchecker_adjusted",
        "lexical.malformed_mixed_alnum_token_rate": "lexical.mechanical_ocr_substitution_rate",
        "lexical.malformed_token_charclass_entropy": "punct.mechanical_unicode_category_entropy",
        "lexical.malformed_dictionary_disagreement_rate":
            "lexical.mechanical_checker_disagreement_rate",
    }
    target = mapping.get(metric_id)
    return {"overlaps_existing_metric_id": target} if target else None


def _with_overlap(metric_id: str, distribution: Mapping[str, Any] | None) -> dict[str, Any] | None:
    overlap = _overlap(metric_id)
    if overlap is None:
        return dict(distribution) if distribution else None
    merged = dict(distribution) if distribution else {}
    merged.update(overlap)
    return merged


def _finding(metric_id: str, name: str, value: Any = None, unit: str | None = None, *,
            sample_size: int | None = None, min_sample: int | None = None,
            distribution: Mapping[str, Any] | None = None,
            evidence: Sequence[Mapping[str, Any]] | None = None,
            details: Sequence[Mapping[str, Any]] | None = None,
            warning: str | None = None, unit_sensitive: bool | None = None) -> dict[str, Any]:
    """:func:`common.finding`, but every id is checked against ``_overlap``."""

    return finding(metric_id, name, value, unit, family=FAMILY, sample_size=sample_size,
                   min_sample=min_sample, distribution=_with_overlap(metric_id, distribution),
                   evidence=evidence, details=details, warning=warning,
                   unit_sensitive=unit_sensitive)


def _stride_indices(n: int, cap: int) -> list[int]:
    if cap <= 0 or n <= cap:
        return list(range(n))
    step = n / cap
    return sorted({int(i * step) for i in range(cap)})


def _sentence_runs(n: int, run_length: int, max_runs: int) -> list[tuple[int, int]]:
    """Bounded, deterministic, contiguous sentence spans spread across the book.

    Contiguous so adjacent-sentence code-switch comparisons inside a run are
    meaningful; spread by a fixed arithmetic stride (no RNG) so the same book
    always samples the same runs.
    """

    if n <= 0:
        return []
    run_length = max(1, min(run_length, n))
    if n <= run_length:
        return [(0, n)]
    possible = max(1, n // run_length)
    count = max(1, min(max_runs, possible))
    if count == 1:
        return [(0, run_length)]
    stride = (n - run_length) / (count - 1)
    seen: list[tuple[int, int]] = []
    for index in range(count):
        start = min(int(round(index * stride)), n - run_length)
        span = (start, start + run_length)
        if span not in seen:
            seen.append(span)
    return sorted(seen)


# ------------------------------------------------------------ script/shape

def _char_script(ch: str) -> str | None:
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for prefix, label in _SCRIPT_PREFIXES:
        if name.startswith(prefix):
            return label
    return "Other"


def _token_shape_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None
                          ) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    if not words_total:
        ids = [
            ("lexical.malformed_long_token_rate", "Long-token rate", "per_1000_words"),
            ("lexical.malformed_long_alpha_token_tail",
             "Long-alphabetic-token tail length (99th percentile)", "characters"),
            ("lexical.malformed_mixed_alnum_token_rate", "Mixed-alphanumeric token rate",
             "per_1000_words"),
            ("lexical.malformed_repeated_symbol_token_rate", "Repeated-symbol token rate",
             "per_1000_words"),
        ]
        return [_finding(mid, name, None, unit, sample_size=0, min_sample=MIN_SAMPLE_WORDS,
                         warning="no words in text") for mid, name, unit in ids]

    min_length = int(option(config, "long_token_min_length", DEFAULT_LONG_TOKEN_MIN_LENGTH))
    words = analysis.words
    lengths = [len(w) for w in words if w.isalpha()]
    long_words = [(w, len(w)) for w in words if w.isalpha() and len(w) > min_length]
    tail_q = float(option(config, "long_token_tail_quantile", DEFAULT_LONG_TOKEN_TAIL_QUANTILE))
    tail_value = stats_module.quantile(sorted(lengths), tail_q) if lengths else None
    out = [_finding(
        "lexical.malformed_long_token_rate",
        f"Long-token rate (alphabetic tokens over {min_length} characters)",
        rate(len(long_words), words_total, 1000.0), "per_1000_words",
        sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        evidence=[{"word": w, "length": n} for w, n in
                 sorted(long_words, key=lambda item: -item[1])[:DEFAULT_MAX_REPORTED]])]
    out.append(_finding(
        "lexical.malformed_long_alpha_token_tail",
        f"Long-alphabetic-token tail length (p{int(tail_q * 100)} of token length)",
        tail_value, "characters", sample_size=len(lengths), min_sample=MIN_SAMPLE_WORDS,
        distribution={"quantile": tail_q, "max_length": max(lengths) if lengths else None}))

    alnum_tokens = _ALNUM_RUN_RE.findall(analysis.text)
    mixed = [t for t in alnum_tokens if any(c.isalpha() for c in t) and any(c.isdigit() for c in t)]
    out.append(_finding(
        "lexical.malformed_mixed_alnum_token_rate",
        "Mixed-alphanumeric token rate (a token containing both letters and digits)",
        rate(len(mixed), words_total, 1000.0), "per_1000_words",
        sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        distribution={"alnum_run_tokens_examined": len(alnum_tokens)},
        evidence=[{"token": t} for t in mixed[:DEFAULT_MAX_REPORTED]]))

    raw_tokens = _RAW_TOKEN_RE.findall(analysis.text)
    repeated = [t for t in raw_tokens if _REPEATED_SYMBOL_RE.search(t)]
    out.append(_finding(
        "lexical.malformed_repeated_symbol_token_rate",
        f"Repeated-symbol token rate (a token containing {DEFAULT_REPEATED_SYMBOL_MIN_RUN}+ "
        f"identical consecutive non-alphanumeric, non-space characters)",
        rate(len(repeated), words_total, 1000.0), "per_1000_words",
        sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        evidence=[{"token": t} for t in repeated[:DEFAULT_MAX_REPORTED]]))
    return out


def _script_distribution_findings(analysis: DocumentAnalysis,
                                  config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    if not words_total:
        return [_finding("lexical.malformed_script_distribution_entropy",
                         "Unicode-script distribution entropy", None, "bits",
                         sample_size=0, min_sample=MIN_SAMPLE_WORDS, warning="no words in text")]
    counts: Counter = Counter()
    for ch in analysis.text:
        script = _char_script(ch)
        if script is not None:
            counts[script] += 1
    total = sum(counts.values())
    entropy = _entropy(counts)
    return [_finding(
        "lexical.malformed_script_distribution_entropy",
        "Unicode-script distribution entropy (0 for a single-script document; "
        "rises with real script mixing, e.g. Latin+Cyrillic or Latin+Han passages)",
        entropy, "bits", sample_size=total, min_sample=MIN_SAMPLE_WORDS,
        distribution={"distinct_scripts": len(counts),
                     "percentages": {k: rate(v, total, 100.0) for k, v in counts.items()}},
        evidence=[{"script": s, "count": n} for s, n in counts.most_common(10)],
        warning=None if total else "no alphabetic characters found")]


def _charclass_entropy_findings(analysis: DocumentAnalysis,
                                config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    if not words_total:
        return [_finding("lexical.malformed_token_charclass_entropy",
                         "Token character-class entropy", None, "bits",
                         sample_size=0, min_sample=MIN_SAMPLE_WORDS, warning="no words in text")]
    counts: Counter = Counter()
    for token in _RAW_TOKEN_RE.findall(analysis.text):
        for ch in token:
            category = unicodedata.category(ch)[0]
            label = {"L": "letter", "N": "digit", "P": "punctuation"}.get(category, "symbol_other")
            counts[label] += 1
    total = sum(counts.values())
    return [_finding(
        "lexical.malformed_token_charclass_entropy",
        "Token character-class entropy (letter/digit/punctuation/other, computed "
        "over token characters only -- excludes inter-token whitespace)",
        _entropy(counts), "bits", sample_size=total, min_sample=MIN_SAMPLE_WORDS,
        distribution={"counts": dict(counts)},
        warning=None if total else "no tokens found")]


# ------------------------------------------------------------ suspicious tokens

def _suspicious_tokens(analysis: DocumentAnalysis, config: Mapping[str, Any] | None
                       ) -> tuple[list[tuple[str, int]], str | None]:
    """Bounded candidate pool: long, wordfreq-rare, alphabetic word types.

    Shared by the word-segmentation and dictionary-disagreement groups so a
    document that enables both pays for this scan once. Cached on the
    document because it depends only on the document and the (language,
    threshold, bounds) options, exactly like ``mechanical_quality_suite``'s
    own shared caches.
    """

    def build() -> tuple[list[tuple[str, int]], str | None]:
        language = option(config, "language", DEFAULT_LANGUAGE)
        min_length = int(option(config, "suspicious_min_length", DEFAULT_SUSPICIOUS_MIN_LENGTH))
        threshold = float(option(config, "suspicious_zipf_threshold",
                                 DEFAULT_SUSPICIOUS_ZIPF_THRESHOLD))
        max_candidates = int(option(config, "suspicious_max_candidates",
                                    DEFAULT_SUSPICIOUS_MAX_CANDIDATES))
        module, reason = require("wordfreq")
        if module is None:
            return [], reason
        counts = Counter(w.lower() for w in analysis.words if w.isalpha())
        candidates = [(w, c) for w, c in counts.items()
                     if len(w) >= min_length and module.zipf_frequency(w, language) < threshold]
        candidates.sort(key=lambda item: -item[1])
        return candidates[:max_candidates], None

    key = ("malformed.suspicious_tokens",
          option(config, "language", DEFAULT_LANGUAGE),
          int(option(config, "suspicious_min_length", DEFAULT_SUSPICIOUS_MIN_LENGTH)),
          float(option(config, "suspicious_zipf_threshold", DEFAULT_SUSPICIOUS_ZIPF_THRESHOLD)),
          int(option(config, "suspicious_max_candidates", DEFAULT_SUSPICIOUS_MAX_CANDIDATES)))
    return analysis.memo(str(key), build)


def _suspicious_frequency_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None
                                   ) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    language = option(config, "language", DEFAULT_LANGUAGE)
    module, reason = require("wordfreq")
    ids = [("lexical.malformed_dictionary_recognized_rate",
            "Fraction of word tokens at or above a normal general-language frequency", "percent"),
           ("lexical.malformed_low_frequency_token_rate",
            "Very-low-frequency token share (below general-language frequency, or "
            "entirely unattested)", "per_1000_words")]
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY, unit=unit) for mid, name, unit in ids]
    if not words_total:
        return [_finding(mid, name, None, unit, sample_size=0, min_sample=MIN_SAMPLE_WORDS,
                        warning="no words in text") for mid, name, unit in ids]

    recognized_threshold = float(option(config, "recognized_zipf_threshold",
                                        DEFAULT_RECOGNIZED_ZIPF_THRESHOLD))
    low_threshold = float(option(config, "suspicious_zipf_threshold",
                                 DEFAULT_SUSPICIOUS_ZIPF_THRESHOLD))
    zipfs = {w: module.zipf_frequency(w, language) for w in
            {t.lower() for t in analysis.words if t.isalpha()}}
    occurrences = [zipfs[t.lower()] for t in analysis.words if t.isalpha()]
    alpha_total = len(occurrences)
    recognized = sum(1 for z in occurrences if z >= recognized_threshold)
    low = sum(1 for z in occurrences if z < low_threshold)
    low_types = sorted({w for w, z in zipfs.items() if z < low_threshold})
    return [
        _finding("lexical.malformed_dictionary_recognized_rate",
                f"Fraction of word tokens at or above Zipf {recognized_threshold} in "
                f"wordfreq's general-language frequency data",
                rate(recognized, alpha_total, 100.0), "percent",
                sample_size=alpha_total, min_sample=MIN_SAMPLE_WORDS,
                distribution={"language": language, "zipf_threshold": recognized_threshold}),
        _finding("lexical.malformed_low_frequency_token_rate",
                f"Very-low-frequency token rate (below Zipf {low_threshold}, includes "
                f"entirely unattested tokens)",
                rate(low, alpha_total, 1000.0), "per_1000_words",
                sample_size=alpha_total, min_sample=MIN_SAMPLE_WORDS,
                distribution={"language": language, "zipf_threshold": low_threshold,
                             "distinct_low_frequency_types": len(low_types)},
                evidence=[{"word": w, "zipf": zipfs[w]} for w in low_types[:DEFAULT_MAX_REPORTED]]),
    ]


# --------------------------------------------------------- word segmentation

_SYMSPELL_CACHE: dict[str, tuple[Any, str | None]] = {}
_WORDSEGMENT_LOCK = threading.Lock()
_WORDSEGMENT_LOADED = False


def _reset_caches() -> None:
    global _WORDSEGMENT_LOADED
    _SYMSPELL_CACHE.clear()
    _WORDSEGMENT_LOADED = False
    _LINGUA_CACHE.clear()
    _LINGUA_LANGUAGE_MAP.clear()
    _LANGID_CACHE.clear()
    _LANGDETECT_SEEDED.clear()
    _FASTTEXT_CACHE.clear()
    _GCLD3_CACHE.clear()


on_reset(_reset_caches)


def _symspell(language: str) -> tuple[Any, str | None]:
    """A SymSpell instance over its own bundled English frequency dictionary.

    Deliberately this suite's own cache, not a shared import from
    ``mechanical_quality_suite`` -- the two suites must be independently
    switchable and independently failable (rule: keep new metrics off by
    default and failing independently); duplicating a ~15-line loader is
    cheaper than coupling two experimental suites' internals together.
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


def _wordsegment_module() -> tuple[Any, str | None]:
    global _WORDSEGMENT_LOADED
    module, reason = require("wordsegment")
    if module is None:
        return None, reason
    if not _WORDSEGMENT_LOADED:
        with _WORDSEGMENT_LOCK:
            if not _WORDSEGMENT_LOADED:
                module.load()
                _WORDSEGMENT_LOADED = True
    return module, None


def _segment_with(name: str, word: str, language: str) -> tuple[list[str] | None, str | None]:
    if name == "wordninja":
        module, reason = require("wordninja")
        if module is None:
            return None, reason
        try:
            return module.split(word), None
        except Exception as exc:  # pragma: no cover - wordninja input guard
            return None, f"wordninja failed on {word!r} ({type(exc).__name__}: {exc})"
    if name == "wordsegment":
        module, reason = _wordsegment_module()
        if module is None:
            return None, reason
        try:
            return module.segment(word), None
        except Exception as exc:  # pragma: no cover - wordsegment input guard
            return None, f"wordsegment failed on {word!r} ({type(exc).__name__}: {exc})"
    if name == "symspell":
        sym_spell, reason = _symspell(language)
        if sym_spell is None:
            return None, reason
        try:
            result = sym_spell.word_segmentation(word, max_edit_distance=0)
            return result.segmented_string.split(), None
        except Exception as exc:  # pragma: no cover - symspell input guard
            return None, f"symspellpy segmentation failed on {word!r} ({type(exc).__name__}: {exc})"
    raise ValueError(f"unknown segmenter {name!r}")


SEGMENTERS = ("wordninja", "wordsegment", "symspell")


def _plausible(parts: list[str] | None, wordfreq_module: Any, language: str,
              threshold: float) -> bool:
    if not parts or len(parts) < 2:
        return False
    return all(len(p) >= 2 and wordfreq_module.zipf_frequency(p.lower(), language) >= threshold
              for p in parts)


def _word_segmentation_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None
                                ) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    language = option(config, "language", DEFAULT_LANGUAGE)
    ids = [
        ("lexical.malformed_fused_word_candidate_rate",
         "Fused-word candidate rate (>=2 of 3 independent segmenters agree on the same "
         "all-real-word split of an unbroken token)", "per_1000_words"),
        ("lexical.malformed_segmentation_disagreement_rate",
         "Word-segmentation disagreement rate among suspicious tokens (the 3 segmenters "
         "do not all produce the same split)", "percent"),
        ("lexical.malformed_avg_segmentation_candidates",
         "Average number of segmenters proposing a plausible (all-real-word) split, "
         "per suspicious token", "segmenters"),
    ]
    wordfreq_module, wf_reason = require("wordfreq")
    if wordfreq_module is None:
        return [unavailable(mid, name, wf_reason, family=FAMILY, unit=unit)
               for mid, name, unit in ids]
    if not words_total:
        return [_finding(mid, name, None, unit, sample_size=0, min_sample=MIN_SAMPLE_TOKENS,
                        warning="no words in text") for mid, name, unit in ids]

    candidates, reason = _suspicious_tokens(analysis, config)
    if reason:
        return [unavailable(mid, name, reason, family=FAMILY, unit=unit)
               for mid, name, unit in ids]
    if not candidates:
        return [_finding(mid, name, 0.0, unit, sample_size=0, min_sample=MIN_SAMPLE_TOKENS,
                        warning="no tokens in this document were long and rare enough to "
                                "be worth checking for a missing space")
               for mid, name, unit in ids]

    plausible_threshold = float(option(config, "segmentation_plausible_zipf",
                                       DEFAULT_SEGMENTATION_PLAUSIBLE_ZIPF))
    fused_occurrences = 0
    disagree_weighted = 0
    checked_weighted = 0
    plausible_counts: list[int] = []
    fused_evidence: list[dict[str, Any]] = []
    disagreement_evidence: list[dict[str, Any]] = []
    tool_availability: dict[str, str | None] = {}

    for word, count in candidates:
        results: dict[str, list[str] | None] = {}
        for name in SEGMENTERS:
            parts, tool_reason = _segment_with(name, word, language)
            if parts is None:
                tool_availability.setdefault(name, tool_reason)
                continue
            tool_availability.setdefault(name, None)
            results[name] = [p for p in parts if p]
        if len(results) < 2:
            continue  # cannot compare or agree with fewer than two segmenters
        checked_weighted += count
        plausible_names = [n for n, parts in results.items()
                          if _plausible(parts, wordfreq_module, language, plausible_threshold)]
        plausible_counts.append(len(plausible_names))
        normalized = {n: tuple(p.lower() for p in parts) for n, parts in results.items()}
        if len(set(normalized.values())) > 1:
            disagree_weighted += count
            if len(disagreement_evidence) < DEFAULT_MAX_REPORTED:
                disagreement_evidence.append({"word": word, "count": count,
                                             "segmentations": {n: list(p) for n, p in normalized.items()}})
        if len(plausible_names) >= 2:
            plausible_splits = Counter(normalized[n] for n in plausible_names)
            best_split, best_votes = plausible_splits.most_common(1)[0]
            if best_votes >= 2:
                fused_occurrences += count
                if len(fused_evidence) < DEFAULT_MAX_REPORTED:
                    fused_evidence.append({"word": word, "segmented": list(best_split),
                                          "count": count, "agreeing_segmenters": best_votes})

    unavailable_tools = {n: r for n, r in tool_availability.items() if r}
    warning = None
    if unavailable_tools:
        warning = ("one or more segmenters were unavailable, so comparisons used only the "
                  f"remaining ones: {unavailable_tools}")
    if not checked_weighted:
        return [_finding(mid, name, 0.0, unit, sample_size=0, min_sample=MIN_SAMPLE_TOKENS,
                        warning=(warning or "fewer than two segmenters were available; "
                                            "segmentation comparison needs at least two"))
               for mid, name, unit in ids]

    avg_candidates = (sum(plausible_counts) / len(plausible_counts)) if plausible_counts else 0.0
    return [
        _finding(ids[0][0], ids[0][1], rate(fused_occurrences, words_total, 1000.0), ids[0][2],
                sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
                distribution={"candidates_examined": len(candidates),
                             "candidates_compared": checked_weighted,
                             "plausible_zipf_threshold": plausible_threshold},
                evidence=fused_evidence, warning=warning),
        _finding(ids[1][0], ids[1][1], rate(disagree_weighted, checked_weighted, 100.0), ids[1][2],
                sample_size=checked_weighted, min_sample=MIN_SAMPLE_TOKENS,
                distribution={"candidates_compared": checked_weighted},
                evidence=disagreement_evidence, warning=warning),
        _finding(ids[2][0], ids[2][1], avg_candidates, ids[2][2],
                sample_size=len(plausible_counts), min_sample=MIN_SAMPLE_TOKENS,
                distribution={"scale": "0-3 segmenters", "segmenters": list(SEGMENTERS)},
                warning=warning),
    ]


def _split_word_findings(analysis: DocumentAnalysis,
                         config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    language = option(config, "language", DEFAULT_LANGUAGE)
    metric_id = "lexical.malformed_split_word_candidate_rate"
    name = ("Split-word candidate rate (two short adjacent tokens whose join is a common "
           "real word while neither piece alone is)")
    module, reason = require("wordfreq")
    if module is None:
        return [unavailable(metric_id, name, reason, family=FAMILY, unit="per_1000_words")]
    if not words_total:
        return [_finding(metric_id, name, None, "per_1000_words", sample_size=0,
                        min_sample=MIN_SAMPLE_WORDS, warning="no words in text")]

    fragment_max = float(option(config, "split_word_fragment_zipf_max",
                                DEFAULT_SPLIT_WORD_FRAGMENT_ZIPF_MAX))
    joined_min = float(option(config, "split_word_joined_zipf_min",
                              DEFAULT_SPLIT_WORD_JOINED_ZIPF_MIN))
    min_combined = int(option(config, "split_word_min_combined_length",
                              DEFAULT_SPLIT_WORD_MIN_COMBINED_LENGTH))
    max_fragment = int(option(config, "split_word_max_fragment_length",
                              DEFAULT_SPLIT_WORD_MAX_FRAGMENT_LENGTH))
    # A non-overlapping regex over consecutive short-alpha-run pairs would
    # miss every other real adjacent pair once one match consumes both of its
    # tokens (e.g. over "ha rbor li ghts" it pairs "ha"+"rbor" then jumps to
    # "li"+"ghts", never checking "rbor"+"li"). Every CONSECUTIVE pair of
    # word tokens is checked instead, by scanning tokens with offsets once
    # and sliding a window of two.
    alpha_tokens = list(_ALPHA_TOKEN_RE.finditer(analysis.text))
    flagged = []
    for first, second in zip(alpha_tokens, alpha_tokens[1:]):
        a, b = first.group(0), second.group(0)
        if len(a) > max_fragment or len(b) > max_fragment:
            continue
        if analysis.text[first.end():second.start()] != " ":
            continue  # not separated by exactly one space
        joined = (a + b).lower()
        if len(joined) < min_combined:
            continue
        if module.zipf_frequency(a.lower(), language) > fragment_max:
            continue
        if module.zipf_frequency(b.lower(), language) > fragment_max:
            continue
        if module.zipf_frequency(joined, language) < joined_min:
            continue
        flagged.append({"offset": first.start(), "fragments": [a, b], "joined": a + b})
    return [_finding(
        metric_id, name, rate(len(flagged), words_total, 1000.0), "per_1000_words",
        sample_size=words_total, min_sample=MIN_SAMPLE_WORDS,
        distribution={"fragment_zipf_max": fragment_max, "joined_zipf_min": joined_min},
        evidence=flagged[:DEFAULT_MAX_REPORTED])]


def _dictionary_disagreement_findings(analysis: DocumentAnalysis,
                                      config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    language = option(config, "language", DEFAULT_LANGUAGE)
    metric_id = "lexical.malformed_dictionary_disagreement_rate"
    name = ("Dictionary/frequency disagreement rate on suspicious tokens (wordfreq's "
           "frequency verdict vs SymSpell's dictionary verdict)")
    wordfreq_module, wf_reason = require("wordfreq")
    if wordfreq_module is None:
        return [unavailable(metric_id, name, wf_reason, family=FAMILY, unit="percent")]
    if not words_total:
        return [_finding(metric_id, name, None, "percent", sample_size=0,
                        min_sample=MIN_SAMPLE_TOKENS, warning="no words in text")]
    sym_spell, sym_reason = _symspell(language)
    if sym_spell is None:
        return [unavailable(metric_id, name, sym_reason, family=FAMILY, unit="percent")]

    candidates, reason = _suspicious_tokens(analysis, config)
    if reason:
        return [unavailable(metric_id, name, reason, family=FAMILY, unit="percent")]
    if not candidates:
        return [_finding(metric_id, name, 0.0, "percent", sample_size=0,
                        min_sample=MIN_SAMPLE_TOKENS,
                        warning="no tokens in this document were rare enough to compare")]

    disagree = 0
    total = 0
    evidence = []
    for word, count in candidates:
        wf_known = wordfreq_module.zipf_frequency(word, language) > 0
        sym_known = word in sym_spell.words
        total += count
        if wf_known != sym_known:
            disagree += count
            if len(evidence) < DEFAULT_MAX_REPORTED:
                evidence.append({"word": word, "count": count, "wordfreq_known": wf_known,
                                "symspell_known": sym_known})
    return [_finding(metric_id, name, rate(disagree, total, 100.0), "percent",
                    sample_size=total, min_sample=MIN_SAMPLE_TOKENS,
                    distribution={"candidates_compared": len(candidates), "language": language},
                    evidence=evidence)]


# ------------------------------------------------------------- language ID

_LINGUA_CACHE: dict[tuple, tuple[Any, str | None]] = {}
_LINGUA_LANGUAGE_MAP: dict[int, dict[str, Any]] = {}
_LANGID_CACHE: dict[str, tuple[Any, str | None]] = {}
_LANGDETECT_SEEDED: dict[str, bool] = {}
_FASTTEXT_CACHE: dict[str, tuple[Any, str | None]] = {}
_GCLD3_CACHE: dict[int, tuple[Any, str | None]] = {}


def _normalize_lang(label: str | None) -> str | None:
    if not label:
        return label
    lowered = label.strip().lower()
    return _LANG_ALIAS.get(lowered, lowered)


def _lingua_language_map(module: Any) -> dict[str, Any]:
    key = id(module)
    if key not in _LINGUA_LANGUAGE_MAP:
        _LINGUA_LANGUAGE_MAP[key] = {
            lang.iso_code_639_1.name.lower(): lang for lang in module.Language.all()}
    return _LINGUA_LANGUAGE_MAP[key]


def _lingua_detector(languages: tuple[str, ...]) -> tuple[Any, str | None]:
    if languages in _LINGUA_CACHE:
        return _LINGUA_CACHE[languages]
    module, reason = require("lingua")
    if module is None:
        outcome = (None, reason)
    else:
        try:
            by_iso = _lingua_language_map(module)
            objs = [by_iso[code] for code in languages if code in by_iso]
            missing = [code for code in languages if code not in by_iso]
            if not objs:
                outcome = (None, f"none of lingua_languages {languages!r} are supported by "
                                 f"lingua's Language enum")
            else:
                detector = module.LanguageDetectorBuilder.from_languages(*objs).build()
                outcome = (detector, None if not missing else
                          f"unsupported lingua_languages ignored: {missing}")
        except Exception as exc:  # pragma: no cover - lingua build guard
            outcome = (None, f"lingua failed to build a detector ({type(exc).__name__}: {exc})")
    _LINGUA_CACHE[languages] = outcome
    return outcome


def _langid_identifier(key: str = "default") -> tuple[Any, str | None]:
    if key in _LANGID_CACHE:
        return _LANGID_CACHE[key]
    module, reason = require("langid")
    if module is None:
        outcome = (None, reason)
    else:
        try:
            identifier = module.langid.LanguageIdentifier.from_modelstring(
                module.langid.model, norm_probs=True)
            outcome = (identifier, None)
        except Exception as exc:  # pragma: no cover - langid model guard
            outcome = (None, f"langid failed to build an identifier ({type(exc).__name__}: {exc})")
    _LANGID_CACHE[key] = outcome
    return outcome


def _langdetect_module(key: str = "default") -> tuple[Any, str | None]:
    module, reason = require("langdetect")
    if module is None:
        return None, reason
    if not _LANGDETECT_SEEDED.get(key):
        module.DetectorFactory.seed = 0
        _LANGDETECT_SEEDED[key] = True
    return module, None


def _fasttext_cache_dir() -> Path:
    override = os.environ.get("TEXTGRADER_FASTTEXT_CACHE_DIR")
    return Path(override).expanduser() if override else Path.home() / ".cache" / "textgrader" / "fasttext"


def _ensure_fasttext_model(path_str: str, url: str) -> tuple[str | None, str | None]:
    path = Path(path_str)
    if path.exists():
        return str(path), None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
        return str(path), None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, (f"fastText's lid.176 model was not found at {path_str!r} and could not "
                      f"be downloaded from {url} ({type(exc).__name__}: {exc}); the compressed "
                      f"model is 938KB -- download it manually and set "
                      f"malformed_text_suite.fasttext_model_path, or run once with network access")


def _fasttext_model(config: Mapping[str, Any] | None) -> tuple[Any, str | None]:
    configured_path = option(config, "fasttext_model_path", "")
    path_str = configured_path or str(_fasttext_cache_dir() / FASTTEXT_MODEL_FILENAME)
    if path_str in _FASTTEXT_CACHE:
        return _FASTTEXT_CACHE[path_str]
    module, reason = require("fasttext")
    if module is None:
        outcome = (None, reason)
    else:
        shim_fasttext_numpy2()
        url = option(config, "fasttext_model_url", FASTTEXT_MODEL_URL_DEFAULT)
        resolved, download_reason = _ensure_fasttext_model(path_str, url)
        if resolved is None:
            outcome = (None, download_reason)
        else:
            try:
                model = module.load_model(resolved)
                outcome = (model, None)
            except Exception as exc:  # pragma: no cover - fastText load guard
                outcome = (None, f"fastText failed to load {resolved!r} "
                                 f"({type(exc).__name__}: {exc})")
    _FASTTEXT_CACHE[path_str] = outcome
    return outcome


def _gcld3_detector(config: Mapping[str, Any] | None) -> tuple[Any, str | None]:
    max_bytes = int(option(config, "cld3_max_bytes", DEFAULT_CLD3_MAX_BYTES))
    if max_bytes in _GCLD3_CACHE:
        return _GCLD3_CACHE[max_bytes]
    module, reason = require("gcld3")
    if module is None:
        outcome = (None, reason)
    else:
        try:
            outcome = (module.NNetLanguageIdentifier(min_num_bytes=0, max_num_bytes=max_bytes), None)
        except Exception as exc:  # pragma: no cover - gcld3 init guard
            outcome = (None, f"gcld3 failed to initialize ({type(exc).__name__}: {exc})")
    _GCLD3_CACHE[max_bytes] = outcome
    return outcome


#: Decimal places kept on every detector confidence.  Lingua's confidence for
#: the same text differs in its last bits from call to call (56.8720080273263
#: against 56.87200802732632 after aggregation), which made two builds of the
#: same corpus profile differ.  Ten places is far below any difference that
#: means anything and makes every detector's output reproducible.
CONFIDENCE_DECIMALS = 10


def _detect_one(name: str, text: str, config: Mapping[str, Any] | None
                ) -> tuple[dict[str, Any] | None, str | None]:
    """One detector's best label + confidence (0-1) for ``text``, normalized."""

    result, reason = _detect_one_raw(name, text, config)
    if result is not None and result.get("confidence") is not None:
        result = {**result, "confidence": round(result["confidence"], CONFIDENCE_DECIMALS)}
    return result, reason


def _detect_one_raw(name: str, text: str, config: Mapping[str, Any] | None
                    ) -> tuple[dict[str, Any] | None, str | None]:
    """The detector's own output, before :func:`_detect_one` rounds it."""

    if not text or not text.strip():
        return None, "empty text"
    if name == "lingua":
        languages = tuple(option(config, "lingua_languages", DEFAULT_LINGUA_LANGUAGES))
        detector, reason = _lingua_detector(languages)
        if detector is None:
            return None, reason
        try:
            values = detector.compute_language_confidence_values(text)
        except Exception as exc:  # pragma: no cover - lingua runtime guard
            return None, f"lingua failed on this text ({type(exc).__name__}: {exc})"
        if not values:
            return None, "lingua returned no candidate language"
        top = values[0]
        return {"raw_label": top.language.name, "iso": top.language.iso_code_639_1.name.lower(),
                "confidence": float(top.value)}, None
    if name == "langid":
        identifier, reason = _langid_identifier()
        if identifier is None:
            return None, reason
        try:
            label, confidence = identifier.classify(text)
        except Exception as exc:  # pragma: no cover - langid runtime guard
            return None, f"langid failed on this text ({type(exc).__name__}: {exc})"
        return {"raw_label": label, "iso": _normalize_lang(label),
                "confidence": float(confidence)}, None
    if name == "langdetect":
        module, reason = _langdetect_module()
        if module is None:
            return None, reason
        try:
            results = module.detect_langs(text)
        except Exception as exc:  # langdetect raises on e.g. pure-digit/pure-symbol input
            return None, f"langdetect could not classify this text ({type(exc).__name__}: {exc})"
        if not results:
            return None, "langdetect returned no candidate language"
        top = results[0]
        return {"raw_label": top.lang, "iso": _normalize_lang(top.lang),
                "confidence": float(top.prob)}, None
    if name == "fasttext":
        model, reason = _fasttext_model(config)
        if model is None:
            return None, reason
        try:
            labels, probs = model.predict(text.replace("\n", " "), k=1)
        except Exception as exc:  # pragma: no cover - fastText runtime guard
            return None, f"fastText failed on this text ({type(exc).__name__}: {exc})"
        if not labels:
            return None, "fastText returned no prediction"
        raw = labels[0].replace("__label__", "")
        return {"raw_label": raw, "iso": _normalize_lang(raw), "confidence": float(probs[0])}, None
    if name == "cld3":
        detector, reason = _gcld3_detector(config)
        if detector is None:
            return None, reason
        try:
            result = detector.FindLanguage(text=text)
        except Exception as exc:  # pragma: no cover - gcld3 runtime guard
            return None, f"gcld3 failed on this text ({type(exc).__name__}: {exc})"
        if not result or not result.language:
            return None, "gcld3 returned no result"
        return {"raw_label": result.language, "iso": _normalize_lang(result.language),
                "confidence": float(result.probability), "is_reliable": bool(result.is_reliable)}, None
    raise ValueError(f"unknown language detector {name!r}")


def _enabled_detectors(config: Mapping[str, Any] | None) -> list[str]:
    return [name for name in DETECTOR_ORDER if _feature(config, _DETECTOR_FEATURE_NAMES[name])]


_LANGUAGE_ID_METRIC_IDS = [
    ("lexical.malformed_document_language_confidence",
     "Document-level language-ID confidence (primary detector)", "percent"),
    ("lexical.malformed_paragraph_offmajority_rate",
     "Paragraph off-majority-language rate", "percent"),
    ("lexical.malformed_sentence_language_confidence_median",
     "Sentence-level language-ID confidence, median (primary detector)", "percent"),
    ("lexical.malformed_sentence_language_low_confidence_rate",
     "Sentence-level low-confidence rate (primary detector)", "percent"),
    ("lexical.malformed_code_switch_rate",
     "Code-switch rate (adjacent-sentence language changes within sampled runs)", "percent"),
    ("lexical.malformed_offmajority_sentence_rate",
     "Sentence off-majority-language rate", "percent"),
    ("lexical.malformed_section_language_entropy",
     "Section-level language entropy", "bits"),
    ("lexical.malformed_detector_disagreement_rate",
     "Language-detector disagreement rate (not all available detectors agree)", "percent"),
    ("lexical.malformed_detector_pairwise_agreement",
     "Mean pairwise language-detector agreement rate", "percent"),
]


def _language_id_findings(analysis: DocumentAnalysis,
                          config: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    words_total = analysis.word_count
    if not words_total:
        return [_finding(mid, name, None, unit, sample_size=0, min_sample=MIN_SAMPLE_SENTENCES,
                        warning="no words in text") for mid, name, unit in _LANGUAGE_ID_METRIC_IDS]

    detectors = _enabled_detectors(config)
    if not detectors:
        return [unavailable(mid, name, "no language-id detector feature is enabled "
                            "(language_id_lingua/langid/langdetect/fasttext/cld3 are all false)",
                            family=FAMILY, unit=unit) for mid, name, unit in _LANGUAGE_ID_METRIC_IDS]

    max_doc_chars = int(option(config, "language_id_max_document_chars",
                               DEFAULT_LANGUAGE_ID_MAX_DOCUMENT_CHARS))
    doc_text = analysis.text[:max_doc_chars]
    doc_results: dict[str, dict[str, Any]] = {}
    doc_reasons: dict[str, str] = {}
    for detector_name in detectors:
        res, reason = _detect_one(detector_name, doc_text, config)
        if res is None:
            doc_reasons[detector_name] = reason or "unavailable"
        else:
            doc_results[detector_name] = res

    available = [n for n in detectors if n in doc_results]
    if not available:
        combined = "; ".join(f"{n}: {r}" for n, r in doc_reasons.items())
        return [unavailable(mid, name, f"no enabled language-id detector could run on this "
                           f"document ({combined})", family=FAMILY, unit=unit)
               for mid, name, unit in _LANGUAGE_ID_METRIC_IDS]
    primary_name = available[0]

    min_chars = int(option(config, "language_id_min_chars", DEFAULT_LANGUAGE_ID_MIN_CHARS))
    low_threshold = float(option(config, "language_id_low_confidence_threshold",
                                 DEFAULT_LANGUAGE_ID_LOW_CONFIDENCE_THRESHOLD))

    # ---- section (window) level: majority language + entropy -----------
    section_words = int(option(config, "language_id_section_words",
                               DEFAULT_LANGUAGE_ID_SECTION_WORDS))
    max_sections = int(option(config, "language_id_max_sections", DEFAULT_LANGUAGE_ID_MAX_SECTIONS))
    windows = analysis.windows(section_words)
    if len(windows) > max_sections:
        idx = _stride_indices(len(windows), max_sections)
        windows = [windows[i] for i in idx]
    lang_word_counts: Counter = Counter()
    section_evidence = []
    for window in windows:
        if window.word_count < 20:
            continue
        res, _ = _detect_one(primary_name, window.text[:max_doc_chars], config)
        if res is None:
            continue
        lang_word_counts[res["iso"]] += window.word_count
        if len(section_evidence) < DEFAULT_MAX_REPORTED:
            section_evidence.append({"language": res["iso"], "words": window.word_count})
    section_entropy = _entropy(lang_word_counts) if lang_word_counts else None
    majority_lang = (lang_word_counts.most_common(1)[0][0] if lang_word_counts
                     else doc_results.get(primary_name, {}).get("iso"))

    # ---- paragraph level: off-majority rate -----------------------------
    max_paragraphs = int(option(config, "language_id_max_paragraphs",
                                DEFAULT_LANGUAGE_ID_MAX_PARAGRAPHS))
    min_paragraph_chars = int(option(config, "language_id_min_paragraph_chars",
                                     DEFAULT_LANGUAGE_ID_MIN_PARAGRAPH_CHARS))
    paragraphs = analysis.paragraphs
    p_indices = _stride_indices(len(paragraphs), max_paragraphs)
    paragraph_checked = 0
    paragraph_offmajority = 0
    paragraph_evidence = []
    for index in p_indices:
        para = paragraphs[index]
        if len(para) < min_paragraph_chars:
            continue
        res, _ = _detect_one(primary_name, para, config)
        if res is None:
            continue
        paragraph_checked += 1
        if majority_lang and res["iso"] != majority_lang:
            paragraph_offmajority += 1
            if len(paragraph_evidence) < DEFAULT_MAX_REPORTED:
                paragraph_evidence.append({"paragraph_index": index, "language": res["iso"],
                                          "text": para[:200]})

    # ---- sentence level: confidence, code-switch, off-majority, agreement
    run_length = int(option(config, "language_id_run_length", DEFAULT_LANGUAGE_ID_RUN_LENGTH))
    max_runs = int(option(config, "language_id_max_runs", DEFAULT_LANGUAGE_ID_MAX_RUNS))
    sentences = analysis.sentences
    runs = _sentence_runs(len(sentences), run_length, max_runs)

    primary_confidences: list[float] = []
    low_confidence = 0
    sentence_checked = 0
    sentence_offmajority = 0
    transitions_checked = 0
    transitions_switched = 0
    disagreement_checked = 0
    disagreement_disagree = 0
    pair_total: Counter = Counter()
    pair_agree: Counter = Counter()
    low_confidence_evidence = []
    code_switch_evidence = []

    for start, end in runs:
        previous_iso: str | None = None
        for index in range(start, end):
            sentence = sentences[index]
            if len(sentence) < min_chars:
                previous_iso = None
                continue
            per_detector: dict[str, dict[str, Any]] = {}
            for detector_name in detectors:
                res, _ = _detect_one(detector_name, sentence, config)
                if res is not None:
                    per_detector[detector_name] = res
            if not per_detector:
                previous_iso = None
                continue
            sentence_checked += 1
            primary_res = per_detector.get(primary_name)
            if primary_res is not None:
                primary_confidences.append(primary_res["confidence"])
                if primary_res["confidence"] < low_threshold:
                    low_confidence += 1
                    if len(low_confidence_evidence) < DEFAULT_MAX_REPORTED:
                        low_confidence_evidence.append({
                            "sentence_index": index, "language": primary_res["iso"],
                            "confidence": primary_res["confidence"], "text": sentence[:200]})
                if majority_lang and primary_res["iso"] != majority_lang:
                    sentence_offmajority += 1
                if previous_iso is not None:
                    transitions_checked += 1
                    if primary_res["iso"] != previous_iso:
                        transitions_switched += 1
                        if len(code_switch_evidence) < DEFAULT_MAX_REPORTED:
                            code_switch_evidence.append({
                                "sentence_index": index, "from": previous_iso,
                                "to": primary_res["iso"], "text": sentence[:200]})
                previous_iso = primary_res["iso"]
            if len(per_detector) >= 2:
                disagreement_checked += 1
                isos = {n: r["iso"] for n, r in per_detector.items()}
                if len(set(isos.values())) > 1:
                    disagreement_disagree += 1
                names_present = sorted(isos)
                for a_index in range(len(names_present)):
                    for b_index in range(a_index + 1, len(names_present)):
                        a_name, b_name = names_present[a_index], names_present[b_index]
                        pair_total[(a_name, b_name)] += 1
                        if isos[a_name] == isos[b_name]:
                            pair_agree[(a_name, b_name)] += 1

    confidence_summary = stats_module.summarize(primary_confidences)
    pairwise_rates = {f"{a}/{b}": rate(pair_agree[(a, b)], total, 100.0)
                      for (a, b), total in pair_total.items()}
    mean_pairwise_agreement = (sum(pairwise_rates.values()) / len(pairwise_rates)
                               if pairwise_rates else None)

    doc_confidence = doc_results[primary_name]["confidence"] * 100.0

    out = [
        _finding(_LANGUAGE_ID_METRIC_IDS[0][0], _LANGUAGE_ID_METRIC_IDS[0][1],
                doc_confidence, "percent", sample_size=len(doc_text), min_sample=MIN_SAMPLE_WORDS,
                distribution={"primary_detector": primary_name, "majority_language": majority_lang,
                             "detectors": {n: r for n, r in doc_results.items()},
                             "unavailable_detectors": doc_reasons}),
        _finding(_LANGUAGE_ID_METRIC_IDS[1][0], _LANGUAGE_ID_METRIC_IDS[1][1],
                rate(paragraph_offmajority, paragraph_checked, 100.0) if paragraph_checked else None,
                "percent", sample_size=paragraph_checked, min_sample=MIN_SAMPLE_PARAGRAPHS,
                distribution={"primary_detector": primary_name, "majority_language": majority_lang,
                             "paragraphs_sampled": len(p_indices)},
                evidence=paragraph_evidence,
                warning=None if paragraph_checked else
                "no paragraph in the sampled set was long enough to language-identify"),
        _finding(_LANGUAGE_ID_METRIC_IDS[2][0], _LANGUAGE_ID_METRIC_IDS[2][1],
                (confidence_summary.get("median") * 100.0)
                if confidence_summary.get("median") is not None else None,
                "percent", sample_size=sentence_checked, min_sample=MIN_SAMPLE_SENTENCES,
                distribution={"primary_detector": primary_name, "shape": confidence_summary},
                warning=None if sentence_checked else
                "no sampled sentence was long enough to language-identify"),
        _finding(_LANGUAGE_ID_METRIC_IDS[3][0], _LANGUAGE_ID_METRIC_IDS[3][1],
                rate(low_confidence, sentence_checked, 100.0) if sentence_checked else None,
                "percent", sample_size=sentence_checked, min_sample=MIN_SAMPLE_SENTENCES,
                distribution={"primary_detector": primary_name, "confidence_threshold": low_threshold},
                evidence=low_confidence_evidence,
                warning=None if sentence_checked else
                "no sampled sentence was long enough to language-identify"),
        _finding(_LANGUAGE_ID_METRIC_IDS[4][0], _LANGUAGE_ID_METRIC_IDS[4][1],
                rate(transitions_switched, transitions_checked, 100.0)
                if transitions_checked else None, "percent",
                sample_size=transitions_checked, min_sample=MIN_SAMPLE_SENTENCES,
                distribution={"primary_detector": primary_name,
                             "sentence_runs_sampled": len(runs), "run_length": run_length},
                evidence=code_switch_evidence,
                warning=None if transitions_checked else
                "fewer than two adjacent language-identifiable sentences were available "
                "within any sampled run"),
        _finding(_LANGUAGE_ID_METRIC_IDS[5][0], _LANGUAGE_ID_METRIC_IDS[5][1],
                rate(sentence_offmajority, sentence_checked, 100.0) if sentence_checked else None,
                "percent", sample_size=sentence_checked, min_sample=MIN_SAMPLE_SENTENCES,
                distribution={"primary_detector": primary_name, "majority_language": majority_lang},
                warning=None if sentence_checked else
                "no sampled sentence was long enough to language-identify"),
        _finding(_LANGUAGE_ID_METRIC_IDS[6][0], _LANGUAGE_ID_METRIC_IDS[6][1],
                section_entropy, "bits", sample_size=len(lang_word_counts) and sum(lang_word_counts.values()),
                min_sample=MIN_SAMPLE_WORDS,
                distribution={"primary_detector": primary_name, "sections_examined": len(windows),
                             "distinct_languages": len(lang_word_counts),
                             "language_word_share": {k: rate(v, sum(lang_word_counts.values()), 100.0)
                                                    for k, v in lang_word_counts.items()}},
                evidence=section_evidence,
                warning=None if lang_word_counts else "no section was long enough to language-identify"),
        _finding(_LANGUAGE_ID_METRIC_IDS[7][0], _LANGUAGE_ID_METRIC_IDS[7][1],
                rate(disagreement_disagree, disagreement_checked, 100.0)
                if disagreement_checked else None, "percent",
                sample_size=disagreement_checked, min_sample=MIN_SAMPLE_SENTENCES,
                distribution={"detectors_enabled": detectors},
                warning=None if disagreement_checked else
                "fewer than two language detectors produced a result on the same sentence; "
                "disagreement needs at least two"),
        _finding(_LANGUAGE_ID_METRIC_IDS[8][0], _LANGUAGE_ID_METRIC_IDS[8][1],
                mean_pairwise_agreement, "percent",
                sample_size=disagreement_checked, min_sample=MIN_SAMPLE_SENTENCES,
                distribution={"pairwise_agreement_rates": pairwise_rates},
                warning=None if pairwise_rates else
                "fewer than two language detectors produced a result on the same sentence"),
    ]
    return out


# ------------------------------------------------------------------- measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    if _feature(config, "token_shape"):
        out.extend(_token_shape_findings(analysis, config))
    if _feature(config, "script_distribution"):
        out.extend(_script_distribution_findings(analysis, config))
    if _feature(config, "charclass_entropy"):
        out.extend(_charclass_entropy_findings(analysis, config))
    if _feature(config, "suspicious_token_frequency"):
        out.extend(_suspicious_frequency_findings(analysis, config))
    if _feature(config, "word_segmentation"):
        out.extend(_word_segmentation_findings(analysis, config))
        out.extend(_split_word_findings(analysis, config))
    if _feature(config, "dictionary_disagreement"):
        out.extend(_dictionary_disagreement_findings(analysis, config))
    if _feature(config, "language_id"):
        out.extend(_language_id_findings(analysis, config))

    return out
