"""Randomness, gibberish, language-likeness, compression and information theory.

Different kinds of "noise" fail at different levels of text, and a single
gibberish score cannot see all of them at once. Random ASCII fails a character
model but says nothing about syntax. Word salad ("furiously sleep green ideas
the") can pass a spelling check and even a unigram frequency check while
failing every word-order model. Sentence-shuffled prose keeps every word,
every sentence and every parse tree intact while destroying only discourse
order, which a character or word n-gram model is structurally blind to.
Heavily repeated template text can look almost normal on a frequency table
while compressing far harder than any of the above. This module keeps every
one of those channels separate, on purpose: agreement or disagreement between
them is the diagnostic, not a number that averages them away.

Everything here is off by default (``randomness_suite.enabled = false``), and
every measurement group below that switch has its own ``features.<name>``
toggle plus its own named tunables, because a suite this broad has no single
"reasonable" configuration for every use.

**Language models are trained on a held-out split of the document being
graded, never on the whole thing - except the one channel that says
otherwise.** Every n-gram channel here (char, byte, word, POS, dependency,
punctuation), the PPM channel, and the KenLM channel work the same way: hold
back a suffix of the document's own sentences, fit a model on everything
before it (additive smoothing for the from-scratch n-gram channels; an
adaptive context model for PPM; a real Kneser-Ney-smoothed n-gram model,
trained by KenLM's own ``lmplz``, for the KenLM channel), and score the
held-out suffix. This is a legitimate, testable, leave-one-out measurement of
how internally predictable a text is at several linguistic levels; it is NOT
a claim about English in general, and every one of these findings says so in
its ``distribution``. The corruption-baseline group reuses the exact same
trained model to score a corrupted version of the same held-out text, which
is what turns "predictable to itself" into "sensitive to word order /
character order / sentence order", per the brief. The single exception is
``features.neural_language_model`` (off by default): now that torch and
transformers are installed, that one channel scores text under a real
pretrained model (distilgpt2) instead of one fit on the document, which IS a
claim about general English, with its own caveats spelled out in "Judgment
calls" below and in the channel's own docstring.

**Sample-size discipline is the organizing constraint of this whole module.**
Compression ratio, every entropy family, and every perplexity number move with
how much text produced them: a compressor's dictionary is still filling in on
the first kilobyte, and entropy over any heavy-tailed category set (word
types, PPMd contexts) rises with more distinct openings even when the
underlying process has not changed at all (see the discussion of this exact
failure mode in ``common.py``). Every finding here is
``sample_size_sensitive=True``, records the amount of text it saw
(``sample_size``, plus the raw character/token count in ``distribution``), and
several are additionally computed on a fixed-size block specifically so a
50,000-word chapter and a 300,000-word novel produce comparable numbers rather
than a number that mostly encodes their difference in length.

**Judgment calls made here, stated plainly:**

* Character-level channels (Shannon/byte entropy, char n-gram models, LZ
  complexity) lower-case the canonical text first. Case variation is a real
  stylistic signal (handled by other metrics) but roughly doubles the
  apparent alphabet size here for no informational gain about randomness.
* The permutation/spectral/SVD/approximate/sample-entropy family (all of
  which assume a real-valued dynamical series, not a symbol stream) is
  evaluated on the sentence-length sequence, the finest-grained numeric series
  every document already has without deriving a new one. This reads as
  "how patterned is the book's rhythm as a sequence", a different question
  from "how patterned is a character", and is a deliberate scope choice
  (see Deferred).
* Corruption baselines shuffle a tail slice of the document's own sentences
  (never the whole book: O(n) shuffles of a 300,000-word text are wasted work
  when a bounded, seeded sample answers the same question) and are always
  compared against a model trained on a disjoint, earlier slice of the same
  document, so nothing is scored against text it was fit on.
* "Letter-frequency divergence from English" uses single-letter frequencies
  from a small, well-established 26-value table (Lewand 2000). Letter-*bigram*
  divergence needed a reference table too, and this codebase ships no raw
  reference corpus to build one from (see NCD, below) - but it does ship
  ``data/prose_reference.json``, whose ``word_frequency`` field is a
  53,000-type word-count table pooled from the reference corpus at build
  time. Weighting every consecutive a-z letter pair inside every word by that
  word's corpus count turns a word-frequency table nobody built for this
  purpose into a legitimate bigram frequency table (the top results - "th",
  "he", "in", "er", "an" - match published English bigram-frequency lists,
  which is the actual check that it is legitimate, not a table pulled from
  thin air). ``distribution.reference`` on the finding names this exact
  source so nobody mistakes it for a licensed table.
* The "unpronounceable cluster" and "known-word rate" channels are
  deliberately weak, independent sensors, not a spellchecker: a heuristic
  consonant-run count will flag real words ("rhythm", "strengths") and a
  wordfreq lookup will miss real names. That is fine; they are one vote each,
  reported for what they are.
* The neural causal-LM perplexity channel (``features.neural_language_model``,
  off by default) is the one deliberate exception to "no bundled or
  downloaded reference language model": torch and transformers are now
  installed, so a real pretrained model (distilgpt2 by default) is available
  and answers a question the from-scratch n-gram channels structurally
  cannot - how well-formed does this read against general English, not just
  against itself. It is scored for what it is worth and no further: subword
  (BPE) tokenization fragments unfamiliar strings into short, individually
  unsurprising pieces, so this channel is NOT a gibberish detector and can
  rank purely random letters as *more* fluent than grammatically-scrambled
  real words (measured on this exact model: a real sentence scores ~243
  perplexity, the same sentence word-shuffled scores ~10,053, and random
  letters of the same length score ~303 - lower than the shuffled real
  sentence). See ``_group_neural_language_model``'s docstring and the
  finding's own ``distribution`` note for the full explanation; the character
  n-gram and compression channels above remain this module's gibberish
  sensors.
* The KenLM channel (``features.kenlm_language_model``, off by default)
  needs a compiled ``lmplz`` binary that ``pip install kenlm`` does not and
  cannot provide - that package is query-time bindings only. This channel
  never assumes a build toolchain's own temporary path (a specific
  environment's ``/tmp/kenlm-src/build/bin`` is not durable and is not this
  environment's ``/tmp``): it looks for ``lmplz`` at a configured
  ``kenlm_lmplz_path`` first, then on ``PATH``, and degrades to a single
  "unavailable" finding naming exactly that build step (source, Boost, and
  everywhere it looked) when neither is found, rather than pretending KenLM
  itself is unavailable. Training always writes to a fresh
  ``tempfile.TemporaryDirectory()``, never anywhere inside this repository or
  a fixed path, and that directory (the trained ``.arpa`` model included) is
  removed before the channel returns: a trained language model is exactly
  the kind of large, environment-specific artifact this codebase does not
  check in.
* The ``textdescriptives`` cross-check (``features.textdescriptives_cross_check``,
  off by default) reports that package's ``information_theory`` component
  numbers (entropy/perplexity from spaCy's static lexeme-probability table)
  next to this module's own wordfreq-based ``unknown_word_rate``, on purpose,
  without reconciling them: they use different reference tables and different
  formulas (textdescriptives sums per-token -p*log(p) over the whole
  document rather than normalizing by token count first), so the numbers
  disagree in scale, and the disagreement itself is reported rather than
  hidden.
* The ``gibberish-detector`` package (``features.gibberish_detector_package``,
  off by default) is trained fresh on this document's own held-out split,
  never on the package's own bundled reference file, exactly like every
  from-scratch n-gram channel above - see "Judgment calls" and
  ``_group_gibberish_detector_package``'s own docstring for what it actually
  measures and the two ways it disagrees with ``consonant_cluster_rate``:
  it is close to blind to word order (its own n-gram iterator strips spaces
  before taking letter bigrams) and it scores a heavily repeated template as
  LESS gibberish than varied real prose, because it measures predictability,
  not meaning.
* NCD against the reference corpus (``features.ncd_against_corpus``, off by
  default) is now implemented for real, reading the corpus folder directly
  at grading time behind its own switch - see "Deferred" below for why this
  one measurement, alone in this suite, touches the filesystem instead of a
  cached profile. Its default ``ncd_corpus_algorithm`` is ``lzma``, not
  ``zlib``: NCD needs ``compress(x + y)`` to see back across the WHOLE
  concatenation, and zlib/gzip's DEFLATE window is a fixed 32768 bytes,
  smaller than 2x this suite's own default byte cap - a real, measured
  failure (an identical document and an unrelated one both scored ~0.97 NCD
  under zlib at 100 KB), not a property of how the compared text was
  produced. ``_group_ncd_against_corpus``'s docstring has the full numbers
  and the guard that now refuses a number a windowed compressor cannot
  actually compute, rather than reporting a meaningless one.
* ``features.gibberish_detector_package`` and ``features.ncd_against_corpus``
  join ``features.kenlm_language_model`` and ``features.neural_language_model``
  under the same gating rule stated above for KenLM and the neural LM: both
  are off by default specifically so that corpus profiling - which runs this
  suite's ``MetricSpec.defaults`` over every reference book - never trains a
  gibberish-detector model or opens a corpus folder as a side effect of
  profiling being on. Verified the same way: a test patches ``require`` (for
  gibberish-detector) and asserts ``ncd_corpus_dirs`` defaults to ``[]`` (for
  NCD-against-corpus) and neither code path runs under default settings.

**Deferred** (named, not silently skipped):

* MALLET is not used: it is a Java toolchain, and TextGrader is Python only
  by decision.  Every language-model and compression channel here is either
  the standard library or a Python package.
* Every compressor named in the original brief - zlib, gzip, bz2, lzma,
  zstandard, brotli, lz4, snappy, pyppmd - is installed, exercised, and used
  for real, including pyppmd as an actual PPM predictive-model channel
  (``ppm_cross_entropy``), not only a compression ratio. ``snappy`` needed
  the system ``libsnappy-dev`` package in an earlier environment; that
  package is installed now, ``python-snappy`` imports and compresses, and it
  is wired up as a real channel like every other compressor, not kept
  artificially unavailable to demonstrate the degradation path -- that path
  is still exercised for real by ``TEXTGRADER_DISABLE_OPTIONAL=all`` and by
  whichever of these packages a given install genuinely lacks.
* KenLM is now used for real (``features.kenlm_language_model``, off by
  default): the ``kenlm`` Python wheel provides query-time bindings
  (``kenlm.Model``), and this channel trains a real Kneser-Ney-smoothed
  n-gram model on the document's own held-out split, exactly like the
  from-scratch Lidstone-smoothed channels above, then scores the held-out
  suffix with it. What pip cannot provide is the *trainer*: ``lmplz`` is a
  separate C++ binary (needs Boost) that has to be built from KenLM's
  source, so this channel looks for it via ``kenlm_lmplz_path`` or ``PATH``
  and reports a clear, actionable "unavailable" naming exactly that build
  step when it cannot find it -- it does not claim KenLM is impossible, only
  that the trainer is not installed in a given environment, which is a
  different and much narrower claim. See ``_group_kenlm_language_model``'s
  docstring for the discovery order and the honesty constraint this channel
  is held to.
* The NLTK language-model module is not used. NLTK's own smoothing classes
  were skipped in favor of one Lidstone-smoothed implementation shared by
  every from-scratch n-gram channel here (char, byte, word, POS, dependency,
  punctuation), so all six report comparably smoothed numbers instead of six
  different smoothing behaviours.
* ``gibberish-detector`` (PyPI, 0.1.1) is now installed and used for real
  (``features.gibberish_detector_package``, off by default - see the gating
  note above): it is on PyPI, ``pip install gibberish-detector`` works in
  this environment, and it is exercised for real against the corruption
  ladder (see ``_group_gibberish_detector_package``'s docstring for the
  measured numbers), not kept out for being unavailable. It is a second,
  independent channel beside the consonant-cluster heuristic, not a
  replacement for it - the two disagree on a heavily repeated template
  (this channel scores it as LESS gibberish than real prose; the heuristic
  scores it at 0%, "no long consonant runs") and on word order (this
  channel is nearly blind to it; the heuristic is unaffected by construction
  either way, since it only looks inside each word), and both disagreements
  are reported rather than reconciled, exactly this module's stated
  philosophy.
* AntroPy, EntropyHub and dit are not added as dependencies. This is a
  design choice, not an availability one: all of the specific measures the
  task names from them (permutation, spectral, SVD, approximate, sample
  entropy, LZ complexity) have short, well-known closed-form definitions,
  and implementing them directly on top of NumPy means they degrade to
  "NumPy unavailable" instead of "one more third-party package unavailable",
  and are easy to unit-test against the textbook formulas - the same
  reasoning that keeps the from-scratch n-gram channels off NLTK's smoothing
  classes, above. Nothing here is deferred for being uninstalled; all three
  are installable from PyPI and simply are not the better implementation for
  this codebase.
* Normalized Compression Distance against a whole reference corpus is now
  implemented (``features.ncd_against_corpus``, off by default, gated a
  second time on ``ncd_corpus_dirs`` actually pointing at reachable
  ``.txt``/``.md`` files - see the gating note above). The reason this was
  deferred through three earlier passes is still true of the corpus
  *profile*: nothing reaches this module from ``corpus.py`` except one
  document's ``DocumentAnalysis`` and, optionally, a profile of scalar
  distributions and pooled counts, never raw text, so a profile alone still
  cannot serve real NCD. What changed is the design, not the profile: a
  measurement that needs raw reference text and cannot be cached reads the
  corpus folder directly, at grading time, behind its own switch, off by
  default - the same pattern ``stylometry_suite``'s
  ``_ncd_against_corpus_findings`` implements (see that function's
  docstring), mirrored here with this suite's own multi-algorithm
  ``_compress`` and its own ``ncd_corpus_max_reference_documents`` /
  ``ncd_corpus_max_bytes`` bounds rather than reusing stylometry's. The
  letter-bigram table above, built from a pooled word-frequency count, is
  still the right (and much cheaper) tool for a frequency-table divergence
  and is unaffected by this addition. NCD between a document and seeded
  corruptions of itself (word/char/sentence shuffles, always on) remains a
  different, self-contained question - how much does compressibility change
  when a specific kind of structure is destroyed - from NCD against a real
  reference book - how much does this document already share, structurally,
  with one - and both are kept as separate findings rather than merged.
* A ``profile_vector(analysis, config)`` for this suite - which would get a
  per-book vector cached under ``feature_profiles["randomness_suite"]`` the
  next time the shipped corpus profile is rebuilt - was considered and not
  added. Two things would have to both be true for it to earn its keep: some
  channel here needs a genuine per-book *distribution* (not a scalar, which
  already has a slot in ``distributions``) to compare against, and this
  worktree would need to be able to rebuild ``data/prose_reference.json`` to
  actually ship it, which it cannot (that file is out of scope for this
  pass, and rebuilding it needs the raw reference books, not present here).
  Neither channel added in this pass wants one anyway: a compression ratio
  and a KenLM cross-entropy are both properties of one document against
  itself (or a corruption of itself), not a distribution over books the way
  ``feature_profiles["function_words"]`` is. The letter-bigram reference
  table already gets its corpus-wide table from a *pooled* count
  (``word_frequency``), which is the right shape for "one frequency table
  for the whole corpus" and does not need a per-book vector either.
* Byte n-gram cross-entropy is reported at one configurable order rather
  than swept across orders like the character channel: for UTF-8 English
  prose the byte and character streams are almost identical past order 2,
  so a full sweep would mostly restate the character sweep at higher cost.
* POS and dependency-label n-gram perplexity, and the ``textdescriptives``
  cross-check, are real, implemented measures, but sit behind
  ``features.pos_dependency`` and ``features.textdescriptives_cross_check``
  (both off by default) because both need a spaCy parse - the shared one for
  the former, a second, differently-configured pipeline (with
  textdescriptives' components attached) for the latter, since those
  components must be present before parsing, not applied after. Either one
  would otherwise turn a "moderate"-cost metric into a "parse"-cost one for
  every user regardless of whether they wanted that particular channel.
* The neural causal-LM channel needs torch and transformers, both of which
  are large, and a model download or a warm local cache the first time a
  given ``neural_lm_model`` name is used; ``features.neural_language_model``
  is off by default for that cost, not for a lack of a working
  implementation (see "Judgment calls" above), and the code path that would
  import either package is never reached unless that flag is explicitly on
  (verified by a test that patches ``optional.require`` and asserts neither
  name is ever requested under the default configuration).
"""

from __future__ import annotations

import bz2
import gzip
import importlib
import json
import lzma
import math
import os
import random
import shutil
import subprocess
import tempfile
import threading
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import optional
from ..document import DocumentAnalysis
from ..optional import require
from ..paths import PROSE_REFERENCE
from ..stats import shannon_entropy as numeric_shannon_entropy
from .common import MODERATE, finding, option, rate, unavailable
from .punctuation_profile import MARK_RE

FAMILY = "lexical"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("wordfreq",)
MIN_SAMPLE = 1000
UNIT_SENSITIVE = False

ID = "style.randomness_"

# ------------------------------------------------------------- configuration

DEFAULT_FEATURES: dict[str, bool] = {
    "char_entropy": True,
    "compression": True,
    "complexity_measures": True,
    "language_model": True,
    "punctuation_sequence": True,
    "pos_dependency": False,  # needs the shared spaCy parse; opt-in.
    "corruption_baselines": True,
    "lexical_gibberish": True,
    "ppm_language_model": True,  # pyppmd is a real predictive model, not just a ratio.
    "letter_bigram_divergence": True,  # derived reference table; see module docstring.
    "kenlm_language_model": False,  # needs a compiled lmplz binary; opt-in, see module docstring.
    "neural_language_model": False,  # downloads/runs a pretrained model; opt-in, see module docstring.
    "textdescriptives_cross_check": False,  # needs its own spaCy pipeline; opt-in for parse cost.
    "gibberish_detector_package": False,  # trains a fresh model per document; opt-in, see gating note below.
    "ncd_against_corpus": False,  # reads the corpus folder at grading time; opt-in, see gating note below.
}

#: Every tunable this module reads via ``option()``. Mirrored, key for key,
#: in ``config.json`` so the file documents the whole surface (task rule #2).
DEFAULTS: dict[str, Any] = {
    "features": DEFAULT_FEATURES,
    "language": "en",
    "char_ngram_orders": [2, 3, 4, 5, 6],
    "byte_ngram_order": 3,
    "word_ngram_orders": [1, 2, 3, 4],
    "punct_ngram_order": 3,
    "pos_ngram_order": 3,
    "dependency_ngram_order": 2,
    "lm_train_fraction": 0.7,
    "lm_smoothing_alpha": 0.5,
    "lm_max_chars": 150_000,
    "lm_max_tokens": 60_000,
    "entropy_token_cap": 100_000,
    "renyi_orders": [0.5, 2.0],
    "tsallis_orders": [0.5, 2.0],
    "excess_entropy_max_order": 4,
    "mi_lags": [1, 2, 3],
    "mi_max_tokens": 20_000,
    "complexity_quadratic_cap": 1500,
    "permutation_order": 3,
    "lz_max_chars": 20_000,
    "compression_algorithms": ["zlib", "gzip", "bz2", "lzma", "zstd", "brotli",
                                "lz4", "snappy", "ppmd"],
    "compression_level": 6,
    "compression_block_chars": 20_000,
    "ncd_algorithm": "zlib",
    "corruption_seed": 1337,
    "corruption_permutations": 3,
    "corruption_sentence_fraction": 0.3,
    "corruption_char_order": 4,
    "corruption_word_order": 2,
    "corruption_max_chars": 20_000,
    "consonant_cluster_min": 4,
    "ppm_max_order": 6,
    "kenlm_lmplz_path": "",  # empty = search PATH for lmplz; see module docstring.
    "kenlm_order": 3,
    "kenlm_memory": "50M",  # lmplz's -S sorting-memory cap; its own default (80%) is wasteful here.
    "kenlm_max_train_chars": 200_000,
    "kenlm_timeout_seconds": 30,
    "neural_lm_model": "distilgpt2",
    "neural_lm_max_chars": 6_000,
    "textdescriptives_max_chars": 50_000,
    "gibberish_detector_charset": "abcdefghijklmnopqrstuvwxyz",
    "ncd_corpus_dirs": [],  # empty = disabled even if the feature flag is on; see module docstring.
    "ncd_corpus_max_reference_documents": 10,
    "ncd_corpus_max_bytes": 100_000,
    # lzma, not zlib: NCD needs compress(x+y) to see back across the whole
    # concatenation, and zlib/gzip's DEFLATE window (32768 bytes, fixed) is
    # smaller than 2 * ncd_corpus_max_bytes by default. See
    # _group_ncd_against_corpus's docstring and its window guard.
    "ncd_corpus_algorithm": "lzma",
}

VOWELS = set("aeiou")

#: Lewand (2000)/Cornell letter-frequency table, English, percent -> proportion.
#: A widely used approximation, not a corpus TextGrader ships or controls.
ENGLISH_LETTER_FREQ: dict[str, float] = {
    "e": 12.702, "t": 9.056, "a": 8.167, "o": 7.507, "i": 6.966, "n": 6.749,
    "s": 6.327, "h": 6.094, "r": 5.987, "d": 4.253, "l": 4.025, "c": 2.782,
    "u": 2.758, "m": 2.406, "w": 2.360, "f": 2.228, "g": 2.015, "y": 1.974,
    "p": 1.929, "b": 1.492, "v": 0.978, "k": 0.772, "j": 0.153, "x": 0.150,
    "q": 0.095, "z": 0.074,
}
_LETTER_TOTAL = sum(ENGLISH_LETTER_FREQ.values())
ENGLISH_LETTER_FREQ = {k: v / _LETTER_TOTAL for k, v in ENGLISH_LETTER_FREQ.items()}

_ASCII_LOWERCASE = set("abcdefghijklmnopqrstuvwxyz")

_bigram_reference_lock = threading.Lock()
_bigram_reference_cache: tuple[dict[str, float] | None, int, str | None] | None = None

#: Loaded (tokenizer, model) pairs for ``features.neural_language_model``,
#: keyed by model name. A model is process-level, reusable state, not
#: something derived from one document, so it does not belong on
#: ``analysis.memo``; caching it here means grading several documents in one
#: process loads each named model once.
_lm_model_lock = threading.Lock()
_lm_model_cache: dict[str, tuple[Any, Any] | None] = {}

#: Loaded spaCy+textdescriptives pipelines for ``features.textdescriptives_cross_check``,
#: keyed by spaCy model name. Same reasoning as ``_lm_model_cache``: this is a
#: second, differently-configured spaCy pipeline (textdescriptives'
#: components must be added before parsing), not the one ``analysis.nlp``
#: already builds and caches per document.
_td_pipeline_lock = threading.Lock()
_td_pipeline_cache: dict[str, Any] = {}


def _bigram_reference_table() -> tuple[dict[str, float] | None, int, str | None]:
    """A letter-bigram frequency table derived from the shipped corpus profile.

    No raw reference corpus ships with this codebase (see the module
    docstring's "Deferred" section on NCD), but ``data/prose_reference.json``
    does ship a pooled ``word_frequency`` table: every distinct word type the
    reference corpus contained, with its total count across every book. That
    is enough to build a bigram table nobody built for this purpose: weight
    every consecutive a-z letter pair inside every reference word by that
    word's corpus count. Computed once per process and cached, since the
    source file is a couple of megabytes of JSON.
    """

    global _bigram_reference_cache
    with _bigram_reference_lock:
        if _bigram_reference_cache is not None:
            return _bigram_reference_cache
    try:
        payload = json.loads(PROSE_REFERENCE.read_text(encoding="utf-8"))
        word_frequency = payload["word_frequency"]
        if not isinstance(word_frequency, dict) or not word_frequency:
            raise ValueError("word_frequency is missing or empty")
        counts: Counter = Counter()
        for word, count in word_frequency.items():
            letters = [char for char in str(word).lower() if char in _ASCII_LOWERCASE]
            for a, b in zip(letters, letters[1:]):
                counts[a + b] += count
        total = sum(counts.values())
        if not total:
            raise ValueError("no a-z letter pairs found in word_frequency")
        result = ({pair: count / total for pair, count in counts.items()}, total, None)
    except Exception as exc:  # missing file, unreadable JSON, unexpected shape
        result = (None, 0, f"could not derive a letter-bigram reference table from "
                            f"{PROSE_REFERENCE.name} ({type(exc).__name__}: {exc}); rebuild the "
                            f"shipped corpus profile or run python3 build_corpus.py")
    with _bigram_reference_lock:
        _bigram_reference_cache = result
    return result


def _reset_randomness_suite_caches() -> None:
    """Forget every process-level cache this module keeps, for tests."""

    global _bigram_reference_cache
    with _bigram_reference_lock:
        _bigram_reference_cache = None
    with _lm_model_lock:
        _lm_model_cache.clear()
    with _td_pipeline_lock:
        _td_pipeline_cache.clear()


optional.on_reset(_reset_randomness_suite_caches)


def _features(config: Mapping[str, Any] | None) -> dict[str, bool]:
    """Merge configured feature flags onto the defaults, key by key.

    ``metric_options`` in ``grade.py`` merges the whole ``randomness_suite``
    options dict onto ``MetricSpec.defaults`` with a plain ``dict.update``, so
    a user who configures only ``features.compression`` would otherwise lose
    every other default flag when that shallow merge replaces the whole
    ``features`` key. Re-merging here, one level deeper, keeps a partial
    override partial.
    """

    configured = option(config, "features", {}) or {}
    return {**DEFAULT_FEATURES, **configured}


# ---------------------------------------------------------------- pure math

def _proportions(counts: Mapping[Any, int]) -> list[float]:
    total = sum(counts.values())
    return [n / total for n in counts.values() if n] if total else []


def _shannon_from_counts(counts: Mapping[Any, int]) -> float | None:
    ps = _proportions(counts)
    return -sum(p * math.log2(p) for p in ps) if ps else None


def _renyi_from_counts(counts: Mapping[Any, int], alpha: float) -> float | None:
    ps = _proportions(counts)
    if not ps:
        return None
    if abs(alpha - 1.0) < 1e-9:
        return _shannon_from_counts(counts)
    total = sum(p ** alpha for p in ps)
    if total <= 0:
        return None
    return math.log2(total) / (1 - alpha)


def _tsallis_from_counts(counts: Mapping[Any, int], q: float) -> float | None:
    ps = _proportions(counts)
    if not ps:
        return None
    if abs(q - 1.0) < 1e-9:
        return _shannon_from_counts(counts)
    total = sum(p ** q for p in ps)
    return (1 - total) / (q - 1)


def _normalized(entropy: float | None, distinct: int) -> float | None:
    if entropy is None:
        return None
    if distinct <= 1:
        return 0.0
    return entropy / math.log2(distinct)


def _conditional_entropy(seq: Sequence[Any]) -> tuple[float | None, int]:
    """H(X_i | X_{i-1}) for a categorical sequence, from its own bigrams."""

    if len(seq) < 2:
        return None, len(seq)
    h1 = _shannon_from_counts(Counter(seq))
    h2 = _shannon_from_counts(Counter(zip(seq, seq[1:])))
    if h1 is None or h2 is None:
        return None, len(seq)
    return max(h2 - h1, 0.0), len(seq) - 1


def _mutual_information(seq: Sequence[Any], lag: int) -> float | None:
    if len(seq) <= lag:
        return None
    pairs = list(zip(seq, seq[lag:]))
    total = len(pairs)
    if not total:
        return None
    joint = Counter(pairs)
    marg_x = Counter(a for a, _ in pairs)
    marg_y = Counter(b for _, b in pairs)
    mi = 0.0
    for (a, b), count in joint.items():
        pxy = count / total
        px = marg_x[a] / total
        py = marg_y[b] / total
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log2(pxy / (px * py))
    return max(mi, 0.0)


def _js_divergence(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Jensen-Shannon divergence in bits; 0 = identical, 1 = disjoint support."""

    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}

    def _kl(a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return sum(a[k] * math.log2(a[k] / b[k]) for k in keys
                   if a.get(k, 0.0) > 0 and b.get(k, 0.0) > 0)

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


# ------------------------------------------------------------- n-gram model

def _ngram_counts(seq: Sequence[Any], order: int) -> tuple[Counter, Counter]:
    context, ngram = Counter(), Counter()
    for i in range(len(seq) - order + 1):
        context[tuple(seq[i:i + order - 1])] += 1
        ngram[tuple(seq[i:i + order])] += 1
    return context, ngram


def _score_ngram(test_seq: Sequence[Any], context: Counter, ngram: Counter,
                  order: int, vocab: int, alpha: float) -> tuple[float | None, int]:
    """Cross-entropy in bits/symbol of ``test_seq`` under a Lidstone-smoothed model."""

    if len(test_seq) < order:
        return None, 0
    total, n = 0.0, 0
    for i in range(len(test_seq) - order + 1):
        ctx = tuple(test_seq[i:i + order - 1])
        full = tuple(test_seq[i:i + order])
        p = (ngram.get(full, 0) + alpha) / (context.get(ctx, 0) + alpha * vocab)
        total += -math.log2(p)
        n += 1
    return (total / n, n) if n else (None, 0)


def _lm_channel(seq: Sequence[Any], order: int, alpha: float, train_fraction: float,
                cap: int | None = None) -> dict[str, Any] | None:
    """Train on a prefix, score the held-out suffix. One order, one channel."""

    items = list(seq[:cap]) if cap else list(seq)
    n = len(items)
    if n < max(order * 6, 20):
        return None
    cut = max(order, int(n * train_fraction))
    train, test = items[:cut], items[cut:]
    if len(test) < order:
        return None
    context, ngram = _ngram_counts(train, order)
    vocab = len(set(train)) or 1
    ce, used = _score_ngram(test, context, ngram, order, vocab, alpha)
    if ce is None:
        return None
    return {"cross_entropy_bits": ce, "perplexity": 2 ** ce, "test_symbols": used,
            "train_symbols": len(train), "vocab_size": vocab, "order": order,
            "smoothing_alpha": alpha}


# ---------------------------------------------------------- corruption text

def _corruption_texts(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> dict[str, Any] | None:
    """A train/test split of the document's own sentences, plus seeded shuffles.

    The tail fraction of sentences is the "test" text; everything before it is
    "train". Corruption baselines score a model trained on ``train`` against
    ``test`` and against seeded corruptions of ``test``, so nothing is ever
    scored against text it was fit on, and the corruption itself never touches
    more than ``corruption_max_chars`` of text.

    Four groups in this module now want exactly this split (corruption
    baselines, NCD, the n-gram low-likelihood-sentence channel, and the new
    PPM cross-entropy channel), so it is cached on ``analysis.memo`` rather
    than rebuilt once per group: two metrics that ask the same document the
    same question should get the same answer from the same work, not four
    independent re-derivations of it.
    """

    fraction = option(opts, "corruption_sentence_fraction", 0.3)
    cap = option(opts, "corruption_max_chars", 20_000)
    key = f"randomness.corruption_texts:{fraction}:{cap}"
    return analysis.memo(key, lambda: _build_corruption_texts(analysis, fraction, cap))


def _build_corruption_texts(analysis: DocumentAnalysis, fraction: float, cap: int) -> dict[str, Any] | None:
    sentences = analysis.sentences
    if len(sentences) < 12:
        return None
    split = max(1, int(len(sentences) * (1 - fraction)))
    train_sents, test_sents = sentences[:split], sentences[split:]
    if not test_sents:
        return None
    train_text = " ".join(train_sents).lower()[:cap]
    test_text = " ".join(test_sents).lower()[:cap]
    if len(train_text) < 200 or len(test_text) < 50:
        return None
    # ``train_sentences`` is the un-joined, un-capped list of training
    # sentences (original case), kept alongside ``train_text`` because
    # KenLM trains on one sentence per line, not on one undifferentiated
    # blob of text the way the from-scratch Lidstone n-gram channels do.
    return {"train_text": train_text, "test_text": test_text, "test_sentences": test_sents,
            "train_sentences": train_sents}


def _shuffled(rng: random.Random, items: Sequence[Any]) -> list[Any]:
    out = list(items)
    rng.shuffle(out)
    return out


# --------------------------------------------------------------- compression

def _effective_setting(name: str, level: int) -> tuple[str, int | None]:
    """The (parameter name, clamped value) each codec actually receives.

    Every codec here shares the one ``compression_level`` knob, but each has
    its own valid range and its own name for it, and getting the clamp wrong
    is not cosmetic: an out-of-range value raises inside the codec (``zstd``
    rejects any level above 22; ``pyppmd``'s default variant rejects a
    negative ``max_order``) and used to be swallowed by ``_compress``'s
    blanket ``except Exception``, silently turning a misconfiguration into an
    "unavailable" finding instead of a working result. Clamping here, once,
    keeps ``_compress`` and the finding's own ``distribution`` in agreement
    about what was actually used.
    """

    if name == "bz2":
        return "level", max(1, min(level, 9))
    if name == "lzma":
        return "preset", max(0, min(level, 9))
    if name == "brotli":
        return "quality", max(0, min(level, 11))
    if name == "zstd":
        return "level", max(1, min(level, 22))
    if name == "lz4":
        # 0-16; values above 16 are already treated as 16 by the library, and
        # values below 0 are a documented "fast acceleration" mode, so both
        # ends are left unclamped here.
        return "compression_level", level
    if name == "ppmd":
        # pyppmd's "I" variant (the default) only accepts max_order 2-16;
        # compression_level's default of 6 happens to fall inside that range,
        # but nothing enforced it before, so a configured level outside it
        # crashed straight into _compress's except-and-degrade path instead
        # of compressing.
        return "max_order", max(2, min(level, 16))
    if name == "snappy":
        # python-snappy's compress() takes no level/quality parameter at
        # all -- Snappy trades ratio for speed by design and simply has no
        # such knob. Reporting the configured ``compression_level`` here
        # would misrepresent it as having been used when it was not.
        return "level", None
    return "level", level  # zlib, gzip: pass the configured level through.


#: Fixed, format-mandated back-reference windows for the compressors that
#: genuinely have a small one, in bytes. zlib/gzip's DEFLATE window is 32768
#: bytes (2**15) by construction and is NOT affected by ``compression_level``
#: -- level changes match-finding effort, not the window itself. LZ4's raw
#: block format encodes match offsets in 16 bits, so 65535 (rounded up to
#: 65536 here) is the largest distance a match can ever reference, again
#: regardless of level. python-snappy has no documented fixed limit, but
#: measuring it directly (compressing two 100 KB real-book excerpts,
#: concatenated, at several sizes) shows the same breakdown pattern between
#: roughly 64 KB and 96 KB combined input that LZ4 shows, so it is treated
#: the same, conservatively, at 65536.
#:
#: This matters because NCD needs ``compress(x + y)`` to be able to reference
#: BACK INTO x while encoding y: if ``len(x) + len(y)`` exceeds this window,
#: the compressor cannot see the earlier copy at all, so C(x+y) comes out
#: close to C(x) + C(y) even when x and y are identical -- NCD(x, x) and
#: NCD(x, y) for an unrelated y then both land near 1, and the measurement
#: cannot discriminate anything.  ``_ncd_corpus_algorithm``'s default is
#: ``lzma`` specifically to avoid this; see ``_group_ncd_against_corpus``.
_WINDOWED_COMPRESSOR_BYTES: dict[str, int] = {
    "zlib": 32_768, "gzip": 32_768, "lz4": 65_536, "snappy": 65_536,
}

#: liblzma's preset -> dictionary size table (XZ Utils' documented values;
#: not affected by anything else this module configures). Used only to
#: report a real number in ``distribution``, never to gate anything -- at
#: preset 0 the dictionary is already 256 KiB, comfortably above every byte
#: cap this suite uses by default, so lzma is never a windowing risk here.
_LZMA_PRESET_DICT_BYTES = [262_144, 1_048_576, 2_097_152, 4_194_304, 4_194_304,
                          8_388_608, 8_388_608, 16_777_216, 33_554_432, 67_108_864]


def _compressor_window_bytes(name: str, level: int) -> int | None:
    """The compressor's back-reference window or dictionary size in bytes,
    for ``distribution.compressor_window_bytes`` on every NCD finding, or
    ``None`` when the codec has no fixed distance limit at all (PPMd builds
    a full statistical model of the whole input; it has an order, not a
    window).

    Only ``_WINDOWED_COMPRESSOR_BYTES``' four entries are ever small enough,
    at this suite's byte caps, to change what an NCD finding means; the rest
    are reported for transparency, not because any of them has been observed
    to cause the failure this documents (see ``_group_ncd_against_corpus``'s
    docstring for the measured numbers behind the four that are).
    """

    if name in _WINDOWED_COMPRESSOR_BYTES:
        return _WINDOWED_COMPRESSOR_BYTES[name]
    if name == "bz2":
        # bz2's block size is exactly compresslevel * 100_000 bytes (Python's
        # own bz2 docs); a match cannot be found across a block boundary.
        return max(1, min(level, 9)) * 100_000
    if name == "lzma":
        preset = max(0, min(level, 9))
        return _LZMA_PRESET_DICT_BYTES[preset]
    if name == "zstd":
        # zstandard's default window scales with level; ~2 MiB (window log
        # 21) is the documented default around this suite's mid-range
        # levels and only grows from there, comfortably above every byte
        # cap this suite uses by default.
        return 2_097_152
    if name == "brotli":
        # Brotli's default window (lgwin) is 22 bits = 4 MiB unless a
        # caller sets lgwin explicitly, which this module does not.
        return 4_194_304
    if name == "ppmd":
        return None  # a context-order model, not a sliding window.
    return None


def _compress(name: str, data: bytes, level: int) -> tuple[bytes | None, str | None]:
    try:
        if name == "zlib":
            return zlib.compress(data, level), None
        if name == "gzip":
            return gzip.compress(data, compresslevel=level), None
        if name == "bz2":
            _, setting = _effective_setting(name, level)
            return bz2.compress(data, compresslevel=setting), None
        if name == "lzma":
            _, setting = _effective_setting(name, level)
            return lzma.compress(data, preset=setting), None
        if name == "zstd":
            module, reason = require("zstandard")
            if module is None:
                return None, reason
            _, setting = _effective_setting(name, level)
            return module.ZstdCompressor(level=setting).compress(data), None
        if name == "brotli":
            module, reason = require("brotli")
            if module is None:
                return None, reason
            _, setting = _effective_setting(name, level)
            return module.compress(data, quality=setting), None
        if name == "lz4":
            module, reason = require("lz4")
            if module is None:
                return None, reason
            _, setting = _effective_setting(name, level)
            return module.compress(data, compression_level=setting), None
        if name == "snappy":
            module, reason = require("snappy")
            if module is None:
                return None, reason
            return module.compress(data), None
        if name == "ppmd":
            module, reason = require("pyppmd")
            if module is None:
                return None, reason
            _, setting = _effective_setting(name, level)
            return module.compress(data, max_order=setting), None
        return None, f"unknown compressor {name!r}"
    except Exception as exc:  # pragma: no cover - defensive; a codec may reject input
        return None, f"{name} failed ({type(exc).__name__}: {exc})"


#: Module actually imported for each short compressor name, for both
#: ``require()`` (optional.PACKAGES is keyed by this) and version lookup.
_COMPRESSOR_MODULE_NAME = {"zstd": "zstandard", "brotli": "brotli", "lz4": "lz4",
                          "snappy": "snappy", "ppmd": "pyppmd"}


def _library_version(name: str) -> str | None:
    if name == "zlib":
        return zlib.ZLIB_VERSION
    module_name = _COMPRESSOR_MODULE_NAME.get(name)
    if module_name is None:
        return None
    module, _ = require(module_name)
    if module is None:
        return None
    version = getattr(module, "__version__", None)
    if version is not None:
        return version
    if name == "lz4":
        # require("lz4") returns the lz4.frame submodule (PACKAGES imports
        # "lz4.frame" directly, since that is what the compressor needs), and
        # that submodule carries no __version__ of its own - only the
        # top-level lz4 package does. Without this fallback every lz4 finding
        # silently recorded library_version=None even though the package
        # (and its version) were right there; require("lz4") having already
        # succeeded means the top-level package is importable too.
        try:
            return getattr(importlib.import_module("lz4"), "__version__", None)
        except Exception:  # pragma: no cover - defensive only
            return None
    return None


# ------------------------------------------------------------------- groups

def _group_char_entropy(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = analysis.text.lower()
    char_cap = option(opts, "entropy_token_cap", 100_000) * 6  # generous; O(n) work only
    chars = text[:char_cap]
    out: list[dict[str, Any]] = []

    char_counts = Counter(chars)
    distinct_chars = len(char_counts)
    char_h = _shannon_from_counts(char_counts)
    out.append(finding(f"{ID}char_entropy", "Character Shannon entropy", char_h, "bits",
                       family=FAMILY, sample_size=len(chars), min_sample=MIN_SAMPLE,
                       sample_size_sensitive=True,
                       distribution={"distinct_characters": distinct_chars,
                                    "characters_seen": len(chars), "lowercased": True},
                       warning=None if chars else "no characters to measure"))
    out.append(finding(f"{ID}char_entropy_normalized", "Character entropy, normalized for alphabet size",
                       _normalized(char_h, distinct_chars), "ratio", family=FAMILY,
                       sample_size=len(chars), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"distinct_characters": distinct_chars, "raw_entropy_bits": char_h}))

    byte_data = chars.encode("utf-8", errors="ignore")
    byte_counts = Counter(byte_data)
    distinct_bytes = len(byte_counts)
    byte_h = _shannon_from_counts(byte_counts)
    out.append(finding(f"{ID}byte_entropy", "Byte Shannon entropy", byte_h, "bits", family=FAMILY,
                       sample_size=len(byte_data), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"distinct_bytes": distinct_bytes, "encoding": "utf-8"},
                       warning=None if byte_data else "no bytes to measure"))
    out.append(finding(f"{ID}byte_entropy_normalized", "Byte entropy, normalized for alphabet size",
                       _normalized(byte_h, distinct_bytes), "ratio", family=FAMILY,
                       sample_size=len(byte_data), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"distinct_bytes": distinct_bytes, "raw_entropy_bits": byte_h}))

    lengths = [len(word) for word in analysis.tokens]
    length_counts = Counter(lengths)
    word_len_h = numeric_shannon_entropy(lengths) if lengths else None
    out.append(finding(f"{ID}word_length_entropy", "Word-length entropy", word_len_h, "bits",
                       family=FAMILY, sample_size=len(lengths), min_sample=MIN_SAMPLE,
                       sample_size_sensitive=True,
                       distribution={"distinct_lengths": len(length_counts)},
                       warning=None if lengths else "no words to measure"))
    out.append(finding(f"{ID}word_length_entropy_normalized", "Word-length entropy, normalized",
                       _normalized(word_len_h, len(length_counts)), "ratio", family=FAMILY,
                       sample_size=len(lengths), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"distinct_lengths": len(length_counts), "raw_entropy_bits": word_len_h}))

    token_cap = option(opts, "entropy_token_cap", 100_000)
    tokens = analysis.tokens[:token_cap]
    token_counts = Counter(tokens)
    token_h = _shannon_from_counts(token_counts)
    out.append(finding(
        f"{ID}token_type_entropy", "Word-type (token) entropy", token_h, "bits", family=FAMILY,
        sample_size=len(tokens), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"distinct_types": len(token_counts), "tokens_seen": len(tokens),
                     "note": "entropy over a heavy-tailed (Zipfian) category set; rises with "
                             "sample size even for a fixed process, see common.py"},
        warning=None if tokens else "no words to measure"))
    out.append(finding(f"{ID}token_type_entropy_normalized", "Word-type entropy, normalized for vocabulary size",
                       _normalized(token_h, len(token_counts)), "ratio", family=FAMILY,
                       sample_size=len(tokens), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"distinct_types": len(token_counts), "raw_entropy_bits": token_h}))

    cond_h, cond_n = _conditional_entropy(tokens)
    out.append(finding(f"{ID}token_conditional_entropy", "Conditional entropy H(word | previous word)",
                       cond_h, "bits", family=FAMILY, sample_size=cond_n, min_sample=MIN_SAMPLE,
                       sample_size_sensitive=True, distribution={"tokens_seen": len(tokens)},
                       warning=None if cond_h is not None else "not enough words to measure"))

    mi_cap = option(opts, "mi_max_tokens", 20_000)
    mi_tokens = analysis.tokens[:mi_cap]
    mi_lags = option(opts, "mi_lags", [1, 2, 3])
    mi_by_lag = {lag: _mutual_information(mi_tokens, lag) for lag in mi_lags}
    headline_lag = mi_lags[0] if mi_lags else 1
    out.append(finding(f"{ID}mutual_information", f"Mutual information between words at lag {headline_lag}",
                       mi_by_lag.get(headline_lag), "bits", family=FAMILY, sample_size=len(mi_tokens),
                       min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"by_lag": mi_by_lag, "tokens_seen": len(mi_tokens)},
                       warning=None if mi_tokens else "no words to measure"))

    max_order = min(option(opts, "excess_entropy_max_order", 4), 6)
    order_entropies = {}
    for order in range(1, max_order + 1):
        if len(chars) < order:
            order_entropies[order] = None
            continue
        block_counts = Counter(chars[i:i + order] for i in range(len(chars) - order + 1))
        order_entropies[order] = _shannon_from_counts(block_counts)
    conds = []
    prev = None
    for order in sorted(order_entropies):
        current = order_entropies[order]
        if current is None or (order > 1 and prev is None):
            conds.append(None)
        else:
            conds.append(current if order == 1 else current - prev)
        prev = current
    valid = [c for c in conds if c is not None]
    rate_estimate = valid[-1] if valid else None
    out.append(finding(f"{ID}entropy_rate_char", "Estimated character entropy rate (block-entropy proxy)",
                       rate_estimate, "bits/char", family=FAMILY, sample_size=len(chars),
                       min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"conditional_entropies_by_order": dict(zip(sorted(order_entropies), conds)),
                                    "max_order": max_order},
                       warning=None if valid else "not enough text to estimate an entropy rate"))
    excess = sum((c - rate_estimate) for c in valid) if valid and rate_estimate is not None else None
    out.append(finding(f"{ID}excess_entropy_proxy",
                       "Excess-entropy proxy (sum of order-1..N conditional entropies above the rate estimate)",
                       excess, "bits", family=FAMILY, sample_size=len(chars), min_sample=MIN_SAMPLE,
                       sample_size_sensitive=True,
                       distribution={"max_order": max_order,
                                    "note": "a small-N approximation of Crutchfield/Feldman excess entropy, "
                                            "not the N->infinity limit"},
                       warning=None if valid else "not enough text to estimate excess entropy"))

    renyi_orders = option(opts, "renyi_orders", [0.5, 2.0])
    renyi_by_order = {a: _renyi_from_counts(token_counts, a) for a in renyi_orders}
    headline_alpha = renyi_orders[0] if renyi_orders else 2.0
    out.append(finding(f"{ID}renyi_entropy", f"Renyi entropy of word types, order {headline_alpha}",
                       renyi_by_order.get(headline_alpha), "bits", family=FAMILY, sample_size=len(tokens),
                       min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"by_order": renyi_by_order},
                       warning=None if tokens else "no words to measure"))

    tsallis_orders = option(opts, "tsallis_orders", [0.5, 2.0])
    tsallis_by_q = {q: _tsallis_from_counts(token_counts, q) for q in tsallis_orders}
    headline_q = tsallis_orders[0] if tsallis_orders else 2.0
    out.append(finding(f"{ID}tsallis_entropy", f"Tsallis entropy of word types, q={headline_q}",
                       tsallis_by_q.get(headline_q), "dimensionless", family=FAMILY, sample_size=len(tokens),
                       min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                       distribution={"by_q": tsallis_by_q},
                       warning=None if tokens else "no words to measure"))
    return out


def _group_complexity(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Dynamical-systems complexity measures over the sentence-length series,
    plus Lempel-Ziv complexity over the (lowercased, capped) character stream."""

    numpy, reason = require("numpy")
    lengths = analysis.sentence_lengths
    ids_names = (
        (f"{ID}permutation_entropy", "Permutation entropy of sentence-length sequence"),
        (f"{ID}spectral_entropy", "Spectral entropy of sentence-length sequence"),
        (f"{ID}svd_entropy", "SVD entropy of sentence-length sequence"),
        (f"{ID}approximate_entropy", "Approximate entropy (ApEn) of sentence-length sequence"),
        (f"{ID}sample_entropy", "Sample entropy (SampEn) of sentence-length sequence"),
    )
    out: list[dict[str, Any]] = []
    if numpy is None:
        out.extend(unavailable(mid, name, reason, family=FAMILY) for mid, name in ids_names)
    elif len(lengths) < 12:
        out.extend(finding(mid, name, None, "bits" if "entropy" in mid else None, family=FAMILY,
                           sample_size=len(lengths), min_sample=MIN_SAMPLE,
                           warning="fewer than 12 sentences; a dynamical-systems measure needs a series")
                  for mid, name in ids_names)
    else:
        order = option(opts, "permutation_order", 3)
        pe = _permutation_entropy(numpy, lengths, order)
        se = _spectral_entropy(numpy, lengths)
        svd = _svd_entropy(numpy, lengths, order)
        cap = option(opts, "complexity_quadratic_cap", 1500)
        clipped = lengths[:cap]
        apen = _approximate_entropy(numpy, clipped)
        sampen = _sample_entropy(numpy, clipped)
        values = [pe, se, svd, apen, sampen]
        extra = [{}, {}, {}, {"clipped_to": len(clipped)}, {"clipped_to": len(clipped)}]
        for (mid, name), value, distribution_extra in zip(ids_names, values, extra):
            out.append(finding(mid, name, value, "bits" if "entropy" in mid.rsplit("_", 1)[0] else "ratio",
                               family=FAMILY, sample_size=len(lengths), min_sample=MIN_SAMPLE,
                               sample_size_sensitive=True,
                               distribution={"embedding_order": order, "series": "sentence_lengths",
                                            **distribution_extra}))

    text = analysis.text.lower()
    lz_cap = option(opts, "lz_max_chars", 20_000)
    symbols = text[:lz_cap]
    if len(symbols) < 50:
        out.append(finding(f"{ID}lz_complexity", "Lempel-Ziv incremental-parsing complexity", None,
                           "phrases", family=FAMILY, sample_size=len(symbols), min_sample=MIN_SAMPLE,
                           warning="fewer than 50 characters to parse"))
        out.append(finding(f"{ID}lz_complexity_normalized", "LZ complexity, normalized by n/log2(n)",
                           None, "ratio", family=FAMILY, sample_size=len(symbols), min_sample=MIN_SAMPLE,
                           warning="fewer than 50 characters to parse"))
    else:
        phrases = _lz_complexity(symbols)
        n = len(symbols)
        expected = n / math.log2(n) if n > 1 else 1.0
        out.append(finding(f"{ID}lz_complexity", "Lempel-Ziv incremental-parsing complexity", phrases,
                           "phrases", family=FAMILY, sample_size=n, min_sample=MIN_SAMPLE,
                           sample_size_sensitive=True,
                           distribution={"characters_parsed": n, "lowercased": True}))
        out.append(finding(f"{ID}lz_complexity_normalized", "LZ complexity, normalized by n/log2(n)",
                           phrases / expected if expected else None, "ratio", family=FAMILY,
                           sample_size=n, min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                           distribution={"characters_parsed": n,
                                        "note": "normalizes against the asymptotic phrase count of a "
                                                "random sequence of this length; ~1.0 reads as "
                                                "structurally random, well below 1.0 as repetitive"}))
    return out


def _permutation_entropy(numpy, values: Sequence[float], order: int, delay: int = 1) -> float | None:
    n = len(values)
    if n < order * delay + 1:
        return None
    patterns: Counter = Counter()
    count = 0
    for i in range(n - (order - 1) * delay):
        window = values[i:i + order * delay:delay]
        patterns[tuple(numpy.argsort(window))] += 1
        count += 1
    if not count:
        return None
    ps = numpy.array(list(patterns.values()), dtype=float) / count
    pe = float(-numpy.sum(ps * numpy.log2(ps)))
    max_pe = math.log2(math.factorial(order))
    return pe / max_pe if max_pe > 0 else 0.0


def _spectral_entropy(numpy, values: Sequence[float]) -> float | None:
    x = numpy.asarray(values, dtype=float)
    if len(x) < 8:
        return None
    x = x - x.mean()
    spectrum = numpy.abs(numpy.fft.rfft(x)) ** 2
    total = spectrum.sum()
    if total <= 0:
        return None
    psd = spectrum[spectrum > 0] / total
    se = float(-numpy.sum(psd * numpy.log2(psd)))
    max_se = math.log2(len(psd)) if len(psd) > 1 else 0.0
    return se / max_se if max_se > 0 else 0.0


def _svd_entropy(numpy, values: Sequence[float], order: int, delay: int = 1) -> float | None:
    x = numpy.asarray(values, dtype=float)
    n = len(x)
    m = n - (order - 1) * delay
    if m < order + 1:
        return None
    embedded = numpy.array([x[i:i + order * delay:delay] for i in range(m)])
    try:
        singular = numpy.linalg.svd(embedded, compute_uv=False)
    except Exception:  # pragma: no cover - numerical edge case
        return None
    singular = singular[singular > 0]
    if len(singular) == 0:
        return None
    p = singular / singular.sum()
    se = float(-numpy.sum(p * numpy.log2(p)))
    max_se = math.log2(len(p)) if len(p) > 1 else 0.0
    return se / max_se if max_se > 0 else 0.0


def _phi(numpy, x, m: int, r: float) -> float | None:
    n = len(x)
    count = n - m + 1
    if count <= 0:
        return None
    templates = numpy.array([x[i:i + m] for i in range(count)])
    matches = numpy.array([numpy.sum(numpy.max(numpy.abs(templates - templates[i]), axis=1) <= r)
                           for i in range(count)], dtype=float)
    matches = matches / count
    matches = matches[matches > 0]
    return float(numpy.mean(numpy.log(matches))) if len(matches) else None


def _approximate_entropy(numpy, values: Sequence[float], m: int = 2, r_ratio: float = 0.2) -> float | None:
    x = numpy.asarray(values, dtype=float)
    if len(x) < m + 2:
        return None
    r = r_ratio * numpy.std(x)
    if r <= 0:
        return None
    phi_m = _phi(numpy, x, m, r)
    phi_m1 = _phi(numpy, x, m + 1, r)
    if phi_m is None or phi_m1 is None:
        return None
    return float(phi_m - phi_m1)


def _sample_entropy(numpy, values: Sequence[float], m: int = 2, r_ratio: float = 0.2) -> float | None:
    x = numpy.asarray(values, dtype=float)
    n = len(x)
    if n < m + 2:
        return None
    r = r_ratio * numpy.std(x)
    if r <= 0:
        return None

    def _matches(mm: int) -> int:
        count = n - mm + 1
        if count <= 1:
            return 0
        templates = numpy.array([x[i:i + mm] for i in range(count)])
        total = 0
        for i in range(count):
            dist = numpy.max(numpy.abs(templates - templates[i]), axis=1)
            total += int(numpy.sum(dist <= r)) - 1  # exclude self-match
        return total

    b, a = _matches(m), _matches(m + 1)
    if b <= 0 or a <= 0:
        return None
    return float(-math.log(a / b))


def _lz_complexity(symbols: str) -> int:
    """Incremental-parsing (LZ78-style) phrase count.

    Each phrase is the shortest prefix of the remaining text not already seen
    as an earlier phrase. Repetitive text needs few phrases to describe;
    structurally random text needs close to one phrase per few characters.
    This is a standard, easily verified complexity count; it is deliberately
    not the Kaspar-Schuster (1976) LZ76 window-matching variant, which is more
    error-prone to reimplement correctly and answers a closely related but not
    identical question.
    """

    n = len(symbols)
    if not n:
        return 0
    seen: set[str] = set()
    phrases = 0
    i = 0
    while i < n:
        j = i + 1
        while j <= n and symbols[i:j] in seen:
            j += 1
        seen.add(symbols[i:j])
        phrases += 1
        i = j
    return phrases


def _group_compression(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = analysis.text.lower()
    level = option(opts, "compression_level", 6)
    block_chars = option(opts, "compression_block_chars", 20_000)
    block = text[:block_chars]
    block_bytes = block.encode("utf-8")
    algorithms = option(opts, "compression_algorithms", DEFAULTS["compression_algorithms"])
    out: list[dict[str, Any]] = []

    if not block_bytes:
        for name in algorithms:
            out.append(finding(f"{ID}compression_ratio_{name}", f"Compression ratio ({name}, fixed block)",
                               None, "ratio", family=FAMILY, min_sample=MIN_SAMPLE, sample_size=0,
                               warning="no text to compress"))
        return out

    for name in algorithms:
        compressed, error = _compress(name, block_bytes, level)
        if compressed is None:
            out.append(unavailable(f"{ID}compression_ratio_{name}", f"Compression ratio ({name}, fixed block)",
                                   error or f"{name} unavailable", family=FAMILY))
            continue
        original = len(block_bytes)
        compressed_len = len(compressed) or 1
        ratio = original / compressed_len
        bits_per_char = 8.0 * compressed_len / len(block)
        setting_name, setting_value = _effective_setting(name, level)
        out.append(finding(
            f"{ID}compression_ratio_{name}", f"Compression ratio ({name}, fixed block)", ratio, "ratio",
            family=FAMILY, sample_size=len(block), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
            distribution={"algorithm": name, "level": level, "block_chars": len(block),
                         "original_bytes": original, "compressed_bytes": compressed_len,
                         "bits_per_char": bits_per_char, "library_version": _library_version(name),
                         "encoding": "utf-8, lowercased canonical text",
                         # ``level`` above is the one shared config knob; this codec's own
                         # parameter (name and clamped value) may differ from it, e.g.
                         # ppmd's "max_order" or zstd's range-clamped "level".
                         "effective_setting": {setting_name: setting_value}}))

    zlib_compressed, _ = _compress("zlib", block_bytes, level)
    if zlib_compressed is not None:
        full_bytes = text.encode("utf-8")
        full_compressed, _ = _compress("zlib", full_bytes, level)
        if full_compressed:
            out.append(finding(
                f"{ID}compression_ratio_full", "Compression ratio (zlib, whole document)",
                len(full_bytes) / (len(full_compressed) or 1), "ratio", family=FAMILY,
                sample_size=len(full_bytes), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                distribution={"algorithm": "zlib", "level": level, "original_bytes": len(full_bytes),
                             "compressed_bytes": len(full_compressed),
                             "note": "grows with document length; compare only against the fixed-block "
                                     "ratio of a similarly-sized text, not against compression_ratio_zlib"}))
        block_bits_per_char = 8.0 * len(zlib_compressed) / len(block)
        char_h = _shannon_from_counts(Counter(block))
        residual = (block_bits_per_char - char_h) if char_h is not None else None
        out.append(finding(
            f"{ID}compression_shannon_residual",
            "Compressed bits/char (zlib) minus character Shannon entropy bits/char",
            residual, "bits/char", family=FAMILY, sample_size=len(block), min_sample=MIN_SAMPLE,
            sample_size_sensitive=True,
            distribution={"algorithm": "zlib", "level": level, "block_chars": len(block),
                         "compressed_bits_per_char": block_bits_per_char, "char_shannon_bits_per_char": char_h,
                         "note": "large positive residual: the compressor could not exploit structure "
                                 "Shannon entropy alone would miss (e.g. long-range repeats) or the "
                                 "block is short enough that container overhead dominates; large "
                                 "negative residual is not expected and would flag a bug"}))

    ncd = _group_ncd(analysis, opts)
    out.extend(ncd)
    return out


def _group_ncd(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    prep = _corruption_texts(analysis, opts)
    algo = option(opts, "ncd_algorithm", "zlib")
    level = option(opts, "compression_level", 6)
    seed = option(opts, "corruption_seed", 1337)
    ids_names = (
        (f"{ID}ncd_word_shuffle", "Normalized compression distance: original vs. word-shuffled"),
        (f"{ID}ncd_char_shuffle", "Normalized compression distance: original vs. character-shuffled"),
        (f"{ID}ncd_sentence_shuffle", "Normalized compression distance: original vs. sentence-shuffled"),
    )
    if prep is None:
        return [unavailable(mid, name, "fewer than 12 sentences to build a train/held-out split",
                            family=FAMILY) for mid, name in ids_names]

    test_text = prep["test_text"]
    rng = random.Random(seed)
    word_shuffled = " ".join(_shuffled(rng, test_text.split()))
    char_shuffled = "".join(_shuffled(rng, list(test_text)))
    sentence_shuffled = " ".join(_shuffled(rng, prep["test_sentences"])).lower()

    def _len(data: bytes) -> int | None:
        compressed, _ = _compress(algo, data, level)
        return len(compressed) if compressed is not None else None

    original_bytes = test_text.encode("utf-8")
    cx = _len(original_bytes)
    variants = {"word_shuffle": word_shuffled, "char_shuffle": char_shuffled,
               "sentence_shuffle": sentence_shuffled}
    out = []
    for (mid, name), (key, variant_text) in zip(ids_names, variants.items()):
        variant_bytes = variant_text.encode("utf-8")
        cy = _len(variant_bytes)
        cxy = _len(original_bytes + variant_bytes)
        if cx is None or cy is None or cxy is None or max(cx, cy) == 0:
            out.append(unavailable(mid, name, f"{algo} unavailable or produced empty output", family=FAMILY))
            continue
        ncd = (cxy - min(cx, cy)) / max(cx, cy)
        out.append(finding(mid, name, ncd, "ratio", family=FAMILY, sample_size=len(test_text),
                           min_sample=MIN_SAMPLE, sample_size_sensitive=True,
                           distribution={"algorithm": algo, "level": level, "seed": seed,
                                        "compressed_original": cx, "compressed_variant": cy,
                                        "compressed_concatenation": cxy,
                                        "compressor_window_bytes": _compressor_window_bytes(algo, level),
                                        "note": "near 0 means compression barely notices this "
                                                "corruption; near 1 means the corrupted text shares "
                                                "almost nothing compressible with the original"}))
    return out


def _group_language_model(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    train_fraction = option(opts, "lm_train_fraction", 0.7)
    alpha = option(opts, "lm_smoothing_alpha", 0.5)
    char_cap = option(opts, "lm_max_chars", 150_000)
    token_cap = option(opts, "lm_max_tokens", 60_000)

    chars = list(analysis.text.lower())
    char_orders = option(opts, "char_ngram_orders", [2, 3, 4, 5, 6])
    by_order = {order: _lm_channel(chars, order, alpha, train_fraction, char_cap) for order in char_orders}
    headline_order = char_orders[len(char_orders) // 2] if char_orders else 4
    headline = by_order.get(headline_order)
    out.append(finding(
        f"{ID}char_ngram_cross_entropy", f"Character {headline_order}-gram cross-entropy (held-out)",
        headline["cross_entropy_bits"] if headline else None, "bits/char", family=FAMILY,
        sample_size=headline["test_symbols"] if headline else len(chars), min_sample=MIN_SAMPLE,
        sample_size_sensitive=True,
        distribution={"by_order": by_order, "smoothing": "lidstone", "smoothing_alpha": alpha,
                     "train_fraction": train_fraction, "lowercased": True},
        warning=None if headline else "not enough text for a held-out character n-gram split"))

    byte_order = option(opts, "byte_ngram_order", 3)
    byte_stream = list(analysis.text.lower().encode("utf-8", errors="ignore")[:char_cap])
    byte_result = _lm_channel(byte_stream, byte_order, alpha, train_fraction)
    out.append(finding(
        f"{ID}byte_ngram_cross_entropy", f"Byte {byte_order}-gram cross-entropy (held-out)",
        byte_result["cross_entropy_bits"] if byte_result else None, "bits/byte", family=FAMILY,
        sample_size=byte_result["test_symbols"] if byte_result else len(byte_stream), min_sample=MIN_SAMPLE,
        sample_size_sensitive=True,
        distribution={"order": byte_order, "smoothing_alpha": alpha, "train_fraction": train_fraction,
                     "perplexity": byte_result["perplexity"] if byte_result else None},
        warning=None if byte_result else "not enough text for a held-out byte n-gram split"))

    tokens = analysis.tokens[:token_cap]
    word_orders = option(opts, "word_ngram_orders", [1, 2, 3, 4])
    word_by_order = {order: _lm_channel(tokens, order, alpha, train_fraction) for order in word_orders}
    headline_word_order = word_orders[len(word_orders) // 2] if word_orders else 2
    headline_word = word_by_order.get(headline_word_order)
    out.append(finding(
        f"{ID}word_ngram_cross_entropy", f"Word {headline_word_order}-gram cross-entropy (held-out)",
        headline_word["cross_entropy_bits"] if headline_word else None, "bits/word", family=FAMILY,
        sample_size=headline_word["test_symbols"] if headline_word else len(tokens), min_sample=MIN_SAMPLE,
        sample_size_sensitive=True,
        distribution={"by_order": word_by_order, "smoothing": "lidstone", "smoothing_alpha": alpha,
                     "train_fraction": train_fraction},
        warning=None if headline_word else "not enough words for a held-out word n-gram split"))

    char_bits = headline["cross_entropy_bits"] if headline else None
    word_bits = headline_word["cross_entropy_bits"] if headline_word else None
    gap = None
    if char_bits is not None and word_bits is not None and analysis.word_count:
        mean_word_chars = len(analysis.text) / max(analysis.word_count, 1)
        gap = (word_bits / mean_word_chars) - char_bits
    out.append(finding(
        f"{ID}word_char_surprisal_gap",
        "Word-level surprisal minus character-level surprisal, both per character",
        gap, "bits/char", family=FAMILY, sample_size=min(len(chars), len(tokens)), min_sample=MIN_SAMPLE,
        sample_size_sensitive=True,
        distribution={"char_order": headline_order, "word_order": headline_word_order,
                     "note": "positive: word choice is more surprising than local character "
                             "structure predicts (candidate word-salad signature); negative: "
                             "character transitions are the more surprising channel"},
        warning=None if gap is not None else "needs both a character and a word cross-entropy result"))

    prep = _corruption_texts(analysis, opts)
    if prep is None or headline_word is None:
        out.append(unavailable(f"{ID}low_likelihood_sentence_share",
                               "Share of held-out sentences with below-typical likelihood",
                               "not enough text for a held-out split", family=FAMILY))
        out.append(unavailable(f"{ID}lowest_likelihood_sentence",
                               "Lowest-likelihood held-out sentence", "not enough text for a held-out split",
                               family=FAMILY))
        return out

    train_tokens = prep["train_text"].split()
    order = option(opts, "corruption_word_order", 2)
    context, ngram = _ngram_counts(train_tokens, order)
    vocab = len(set(train_tokens)) or 1
    per_sentence = []
    for sentence in prep["test_sentences"]:
        sent_tokens = sentence.lower().split()
        ce, used = _score_ngram(sent_tokens, context, ngram, order, vocab, alpha)
        if ce is not None:
            per_sentence.append((ce, sentence, used))
    if not per_sentence:
        out.append(unavailable(f"{ID}low_likelihood_sentence_share",
                               "Share of held-out sentences with below-typical likelihood",
                               "no held-out sentence was long enough to score", family=FAMILY))
        out.append(unavailable(f"{ID}lowest_likelihood_sentence",
                               "Lowest-likelihood held-out sentence",
                               "no held-out sentence was long enough to score", family=FAMILY))
        return out

    scores = [ce for ce, _, _ in per_sentence]
    median = sorted(scores)[len(scores) // 2]
    mad = sorted(abs(s - median) for s in scores)[len(scores) // 2] or 1e-9
    threshold = median + 1.5 * mad
    below = sum(1 for ce in scores if ce > threshold)  # higher cross-entropy = lower likelihood
    out.append(finding(
        f"{ID}low_likelihood_sentence_share",
        "Share of held-out sentences noticeably less predictable than this document's own median",
        rate(below, len(scores)), "%", family=FAMILY, sample_size=len(scores), min_sample=MIN_SAMPLE,
        sample_size_sensitive=True,
        distribution={"threshold_bits_per_word": threshold, "median_bits_per_word": median,
                     "note": "threshold is self-referential (median + 1.5x MAD of this document's "
                             "own held-out sentences), because no external reference model is "
                             "available without overfitting to the document being graded"}))
    worst_ce, worst_sentence, worst_n = max(per_sentence, key=lambda item: item[0])
    excerpt = worst_sentence.strip()
    if len(excerpt) > 200:
        excerpt = excerpt[:200] + "..."
    out.append(finding(
        f"{ID}lowest_likelihood_sentence", "Lowest-likelihood held-out sentence (bits/word under this document's own model)",
        worst_ce, "bits/word", family=FAMILY, sample_size=worst_n, min_sample=1,
        evidence=[{"sentence": excerpt}],
        distribution={"order": order, "held_out_sentences_scored": len(per_sentence)}))
    return out


def _group_ppm_language_model(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """PPM (pyppmd) cross-entropy of the held-out suffix, as a real predictive model.

    ``compression_ratio_ppmd`` (in ``_group_compression``) answers "how much
    does PPMd shrink this text"; this answers a different question, "how
    surprised is PPMd's own adaptive model by text it has not seen", which is
    what makes it a language-model channel rather than another compression
    ratio. PPM (prediction by partial matching) already works by building an
    adaptive predictive model as it goes, so its cross-entropy on held-out
    text can be read straight off the compressed size: compressing
    ``train + test`` costs ``compress(train)``'s bits for the ``train``
    prefix (approximately - the codec's own small per-call header is the
    only thing that does not cancel exactly) plus however many additional
    bits the model, now primed on ``train``, needed to describe ``test``. The
    marginal cost *is* the cross-entropy, in the same units and against the
    same held-out split every other n-gram channel here uses, without this
    module needing to reimplement PPM's context-mixing itself.
    """

    mid = f"{ID}ppm_cross_entropy"
    name = "PPM (pyppmd) cross-entropy of held-out text"
    module, reason = require("pyppmd")
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY)]

    prep = _corruption_texts(analysis, opts)
    if prep is None:
        return [unavailable(mid, name, "fewer than 12 sentences to build a train/held-out split",
                            family=FAMILY)]

    order = max(2, min(option(opts, "ppm_max_order", 6), 16))
    train_bytes = prep["train_text"].encode("utf-8")
    test_bytes = prep["test_text"].encode("utf-8")
    test_chars = len(prep["test_text"])
    if test_chars < 50:
        return [unavailable(mid, name, "held-out text too short to score", family=FAMILY)]

    try:
        train_compressed = module.compress(train_bytes, max_order=order)
        combined_compressed = module.compress(train_bytes + test_bytes, max_order=order)
    except Exception as exc:
        return [unavailable(mid, name, f"pyppmd failed ({type(exc).__name__}: {exc})", family=FAMILY)]

    delta_bits = 8.0 * (len(combined_compressed) - len(train_compressed))
    if delta_bits <= 0:
        return [unavailable(mid, name,
                            "PPM model produced a non-positive held-out cost (held-out text "
                            "compressed for free against the trained prefix); too little held-out "
                            "text for the per-call header to cancel out", family=FAMILY)]

    cross_entropy = delta_bits / test_chars
    return [finding(
        mid, name, cross_entropy, "bits/char", family=FAMILY, sample_size=test_chars,
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"algorithm": "ppmd", "max_order": order, "perplexity": 2 ** cross_entropy,
                     "train_chars": len(prep["train_text"]), "test_chars": test_chars,
                     "library_version": _library_version("ppmd"),
                     "method": "8 * (len(compress(train+test)) - len(compress(train))) / "
                               "len(test_chars): the marginal compressed cost of the held-out "
                               "suffix under a model already primed on the training prefix",
                     "note": "a real PPM predictive-model cross-entropy, not the compression "
                             "ratio reported by compression_ratio_ppmd; compare it against "
                             "char_ngram_cross_entropy (same held-out split, additive-smoothed "
                             "fixed-order model) rather than against any compression_ratio_* "
                             "finding"})]


# ------------------------------------------------------------- KenLM language model

def _find_lmplz_binary(configured: str | None) -> tuple[str | None, list[str]]:
    """Where this process looked for KenLM's ``lmplz`` trainer, and what it found.

    Returns ``(path, checked)``: ``path`` is a usable, executable binary or
    ``None``; ``checked`` is every location actually tried, in the order
    tried, so an "unavailable" reason can name them instead of gesturing at
    "somewhere". ``pip install kenlm`` supplies only the query-time Python
    bindings (``kenlm.Model``); ``lmplz`` is a separate C++ program built
    from KenLM's own source tree (a C++ compiler and Boost, at minimum) that
    nothing in this codebase can install, so finding it is a config-or-PATH
    search, never an assumption about where a particular build happened to
    put it - a build toolchain's own temporary directory is not a location
    this function, or any other user's environment, can rely on.
    """

    checked: list[str] = []
    if configured:
        checked.append(configured)
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate), checked
    checked.append("PATH")
    found = shutil.which("lmplz")
    return found, checked


def _group_kenlm_language_model(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cross-entropy of the held-out suffix under a real, KenLM-trained,
    modified-Kneser-Ney-smoothed n-gram model.

    Every from-scratch n-gram channel elsewhere in this module (char, byte,
    word, POS, dependency, punctuation) shares one Lidstone-smoothed
    implementation on purpose (see the module docstring's "Deferred" section
    on why NLTK's smoothing classes were skipped in favor of that). This
    channel is different in kind, not just in smoothing constant: it trains
    an actual KenLM model - the modified Kneser-Ney estimator used by
    production speech-recognition and machine-translation systems - on the
    document's own held-out training split (one sentence per line, exactly
    like KenLM expects, not the single undifferentiated blob the from-scratch
    channels train on), then scores the held-out suffix with it. That makes
    it directly comparable to ``word_ngram_cross_entropy`` (same held-out
    split, same unit, different smoothing) without being the same
    measurement restated: the two are expected to differ by whatever
    Kneser-Ney's back-off buys over flat additive smoothing, and a large gap
    between them is itself informative.

    **Honesty about what is and is not installable.** ``pip install kenlm``
    gives only the query-time Python bindings (``kenlm.Model``); it does not
    and cannot give the trainer, ``lmplz``, which is a separate C++ program
    built from KenLM's own source (see https://github.com/kpu/kenlm#compiling).
    This channel looks for that binary at a configured ``kenlm_lmplz_path``
    first, then on ``PATH`` (see :func:`_find_lmplz_binary`), and if neither
    has it, reports a clear, actionable "unavailable" naming exactly the
    missing build step and everywhere it looked - never a bare "kenlm
    unavailable", which would send someone straight to (and no further than)
    a ``pip install`` that cannot fix this. Training always writes to a
    freshly created, then removed, ``tempfile.TemporaryDirectory()`` -
    never this repository, never a path a build toolchain happened to use -
    so no trained model ever lands anywhere durable, let alone in git.
    """

    mid = f"{ID}kenlm_cross_entropy"
    name = "KenLM (modified Kneser-Ney) cross-entropy of held-out text"
    module, reason = require("kenlm")
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY)]

    configured_path = option(opts, "kenlm_lmplz_path", "") or None
    lmplz_path, checked = _find_lmplz_binary(configured_path)
    if lmplz_path is None:
        return [unavailable(
            mid, name,
            f"KenLM's 'lmplz' trainer was not found (checked: {', '.join(checked)}). "
            "'pip install kenlm' provides only the query-time Python bindings, not the "
            "trainer: build KenLM from source (a C++ compiler and Boost; see "
            "https://github.com/kpu/kenlm#compiling) and either put the resulting 'lmplz' "
            "on PATH or set metrics.randomness_suite.kenlm_lmplz_path to its full path.",
            family=FAMILY)]

    prep = _corruption_texts(analysis, opts)
    if prep is None:
        return [unavailable(mid, name, "fewer than 12 sentences to build a train/held-out split",
                            family=FAMILY)]

    order = max(2, min(option(opts, "kenlm_order", 3), 6))
    memory = option(opts, "kenlm_memory", "50M")
    timeout = option(opts, "kenlm_timeout_seconds", 30)
    max_train_chars = option(opts, "kenlm_max_train_chars", 200_000)

    train_lines: list[str] = []
    total_chars = 0
    for sentence in prep["train_sentences"]:
        line = sentence.lower().strip()
        if not line:
            continue
        train_lines.append(line)
        total_chars += len(line)
        if total_chars >= max_train_chars:
            break
    test_sentences = [sentence.lower().strip() for sentence in prep["test_sentences"] if sentence.strip()]
    if len(train_lines) < 20 or not test_sentences:
        return [unavailable(mid, name, "not enough held-out sentences to train and score a "
                            "KenLM model", family=FAMILY)]

    try:
        with tempfile.TemporaryDirectory(prefix="textgrader-kenlm-") as tmp_dir:
            train_path = Path(tmp_dir) / "train.txt"
            arpa_path = Path(tmp_dir) / "model.arpa"
            train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
            command = [lmplz_path, "-o", str(order), "-S", str(memory), "-T", tmp_dir,
                      "--discount_fallback"]
            with train_path.open("rb") as stdin_file, arpa_path.open("wb") as stdout_file:
                completed = subprocess.run(command, stdin=stdin_file, stdout=stdout_file,
                                           stderr=subprocess.PIPE, timeout=timeout, check=False)
            if completed.returncode != 0:
                stderr_tail = completed.stderr.decode("utf-8", errors="replace").strip()[-400:]
                return [unavailable(mid, name, f"lmplz exited {completed.returncode}: "
                                    f"{stderr_tail or 'no error output'}", family=FAMILY)]
            # build_binary is optional polish, not a prerequisite: it converts the
            # ARPA text file KenLM just wrote into its compact binary format, which
            # loads faster and without KenLM's own "reading ARPA" progress notice on
            # stderr. Looked for next to lmplz (the common case: one build tree) and
            # then on PATH; the ARPA file itself is a perfectly usable model on its
            # own, so a missing build_binary is never reported as "unavailable".
            model_path, model_format = arpa_path, "arpa"
            build_binary = Path(lmplz_path).with_name("build_binary")
            if not (build_binary.is_file() and os.access(build_binary, os.X_OK)):
                found = shutil.which("build_binary")
                build_binary = Path(found) if found else None
            if build_binary is not None:
                binary_path = Path(tmp_dir) / "model.binary"
                converted = subprocess.run([str(build_binary), str(arpa_path), str(binary_path)],
                                           capture_output=True, timeout=timeout, check=False)
                if converted.returncode == 0 and binary_path.is_file():
                    model_path, model_format = binary_path, "binary"
            model = module.Model(str(model_path))
            total_log10, total_words = 0.0, 0
            for sentence in test_sentences:
                total_log10 += model.score(sentence, bos=True, eos=True)
                total_words += len(sentence.split()) + 1  # +1 for </s>, matching kenlm's own convention
    except subprocess.TimeoutExpired:
        return [unavailable(mid, name, f"lmplz did not finish within {timeout}s; try a smaller "
                            "kenlm_max_train_chars or a lower kenlm_order", family=FAMILY)]
    except OSError as exc:
        return [unavailable(mid, name, f"could not run lmplz at {lmplz_path!r} "
                            f"({type(exc).__name__}: {exc})", family=FAMILY)]
    except Exception as exc:  # kenlm.Model raises its own RuntimeError/OSError subclasses
        return [unavailable(mid, name, f"KenLM training or scoring failed "
                            f"({type(exc).__name__}: {exc})", family=FAMILY)]

    if total_words <= 0:
        return [unavailable(mid, name, "no held-out words to score", family=FAMILY)]

    perplexity = 10 ** (-total_log10 / total_words)
    if not math.isfinite(perplexity) or perplexity <= 0:
        return [unavailable(mid, name, "KenLM produced a non-finite perplexity", family=FAMILY)]
    cross_entropy_bits = math.log2(perplexity)

    return [finding(
        mid, name, cross_entropy_bits, "bits/word", family=FAMILY, sample_size=total_words,
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"algorithm": "kenlm", "order": order, "smoothing": "modified Kneser-Ney",
                     "memory": memory, "discount_fallback": True, "lmplz_path": lmplz_path,
                     "model_format": model_format,
                     "train_sentences": len(train_lines), "train_chars": total_chars,
                     "test_sentences": len(test_sentences), "perplexity": perplexity,
                     "method": "trained lmplz on the document's own held-out training split "
                               "(one sentence per line), scored the held-out suffix with "
                               "kenlm.Model.score(..., bos=True, eos=True), converted the "
                               "resulting corpus perplexity to bits/word",
                     "note": "a real, externally-implemented Kneser-Ney n-gram model, not this "
                             "module's own additive-smoothed n-gram channels; compare it "
                             "against word_ngram_cross_entropy (same held-out split, different "
                             "smoothing) rather than expecting the two to agree"})]


def _group_letter_bigram_divergence(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Jensen-Shannon divergence of this text's letter-bigram frequencies from
    a reference table derived from the shipped corpus profile.

    See the module docstring's "Judgment calls" section for where the
    reference table comes from and why weighting bigrams by a pooled
    word-frequency table is a legitimate substitute for a licensed one.
    """

    mid = f"{ID}letter_bigram_divergence"
    name = "Jensen-Shannon divergence of this text's letter-bigram frequencies from a corpus-derived reference"
    reference, total_ref, reason = _bigram_reference_table()
    if reference is None:
        return [unavailable(mid, name, reason, family=FAMILY)]

    tokens = analysis.tokens
    if not tokens:
        return [unavailable(mid, name, "no words to measure", family=FAMILY)]

    bigrams: Counter = Counter()
    for word in tokens:
        letters = [char for char in word.lower() if char in ENGLISH_LETTER_FREQ]
        for a, b in zip(letters, letters[1:]):
            bigrams[a + b] += 1
    total = sum(bigrams.values())
    if not total:
        return [unavailable(mid, name, "no a-z letter pairs found", family=FAMILY)]

    observed = {pair: count / total for pair, count in bigrams.items()}
    divergence = _js_divergence(observed, reference)
    return [finding(
        mid, name, divergence, "bits", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
        sample_size_sensitive=True,
        distribution={"reference": f"derived from {PROSE_REFERENCE.name}'s word_frequency table "
                                   f"({total_ref:,} reference letter-pair observations): every "
                                   "consecutive a-z letter pair inside every reference word, "
                                   "weighted by that word's corpus count",
                     "distinct_bigrams_observed": len(bigrams),
                     "distinct_bigrams_in_reference": len(reference)})]


def _group_punctuation_sequence(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    marks = [match.lastgroup for match in MARK_RE.finditer(analysis.text)]
    order = option(opts, "punct_ngram_order", 3)
    alpha = option(opts, "lm_smoothing_alpha", 0.5)
    train_fraction = option(opts, "lm_train_fraction", 0.7)
    result = _lm_channel(marks, order, alpha, train_fraction)
    out = [finding(
        f"{ID}punct_ngram_cross_entropy", f"Punctuation-sequence {order}-gram cross-entropy (held-out)",
        result["cross_entropy_bits"] if result else None, "bits/mark", family=FAMILY,
        sample_size=result["test_symbols"] if result else len(marks), min_sample=MIN_SAMPLE,
        sample_size_sensitive=True,
        distribution={"order": order, "smoothing_alpha": alpha, "marks_seen": len(marks),
                     "perplexity": result["perplexity"] if result else None},
        warning=None if result else "not enough punctuation marks for a held-out n-gram split")]
    cond_h, cond_n = _conditional_entropy(marks)
    out.append(finding(f"{ID}punct_conditional_entropy", "Conditional entropy H(mark | previous mark)",
                       cond_h, "bits", family=FAMILY, sample_size=cond_n, min_sample=MIN_SAMPLE,
                       sample_size_sensitive=True, distribution={"marks_seen": len(marks)},
                       warning=None if cond_h is not None else "not enough punctuation marks to measure"))
    return out


def _group_pos_dependency(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids_names = (
        (f"{ID}pos_ngram_cross_entropy", "POS-tag n-gram cross-entropy (held-out)"),
        (f"{ID}dependency_ngram_cross_entropy", "Dependency-label n-gram cross-entropy (held-out)"),
        (f"{ID}pos_conditional_entropy", "Conditional entropy H(POS tag | previous POS tag)"),
        (f"{ID}pos_shuffle_ratio", "POS-tag shuffle degradation ratio"),
    )
    if analysis.nlp_unavailable:
        return [unavailable(mid, name, analysis.nlp_unavailable, family=FAMILY) for mid, name in ids_names]

    token_cap = option(opts, "lm_max_tokens", 60_000)
    pos_tags: list[str] = []
    dep_labels: list[str] = []
    for _, doc in analysis.spacy_docs():
        for token in doc:
            pos_tags.append(token.pos_)
            dep_labels.append(token.dep_)
            if len(pos_tags) >= token_cap:
                break
        if len(pos_tags) >= token_cap:
            break

    alpha = option(opts, "lm_smoothing_alpha", 0.5)
    train_fraction = option(opts, "lm_train_fraction", 0.7)
    pos_order = option(opts, "pos_ngram_order", 3)
    dep_order = option(opts, "dependency_ngram_order", 2)

    pos_result = _lm_channel(pos_tags, pos_order, alpha, train_fraction)
    dep_result = _lm_channel(dep_labels, dep_order, alpha, train_fraction)
    cond_h, cond_n = _conditional_entropy(pos_tags)

    out = [
        finding(f"{ID}pos_ngram_cross_entropy", f"POS-tag {pos_order}-gram cross-entropy (held-out)",
               pos_result["cross_entropy_bits"] if pos_result else None, "bits/tag", family=FAMILY,
               sample_size=pos_result["test_symbols"] if pos_result else len(pos_tags), min_sample=MIN_SAMPLE,
               sample_size_sensitive=True,
               distribution={"order": pos_order, "smoothing_alpha": alpha, "tags_seen": len(pos_tags)},
               warning=None if pos_result else "not enough parsed tokens for a held-out POS n-gram split"),
        finding(f"{ID}dependency_ngram_cross_entropy",
               f"Dependency-label {dep_order}-gram cross-entropy (held-out)",
               dep_result["cross_entropy_bits"] if dep_result else None, "bits/label", family=FAMILY,
               sample_size=dep_result["test_symbols"] if dep_result else len(dep_labels), min_sample=MIN_SAMPLE,
               sample_size_sensitive=True,
               distribution={"order": dep_order, "smoothing_alpha": alpha, "labels_seen": len(dep_labels)},
               warning=None if dep_result else "not enough parsed tokens for a held-out dependency n-gram split"),
        finding(f"{ID}pos_conditional_entropy", "Conditional entropy H(POS tag | previous POS tag)",
               cond_h, "bits", family=FAMILY, sample_size=cond_n, min_sample=MIN_SAMPLE,
               sample_size_sensitive=True, distribution={"tags_seen": len(pos_tags)},
               warning=None if cond_h is not None else "not enough parsed tokens to measure"),
    ]

    seed = option(opts, "corruption_seed", 1337)
    n = len(pos_tags)
    if n < pos_order * 12:
        out.append(unavailable(f"{ID}pos_shuffle_ratio", "POS-tag shuffle degradation ratio",
                               "not enough parsed tokens for a shuffle baseline", family=FAMILY))
        return out
    cut = int(n * train_fraction)
    train_tags, test_tags = pos_tags[:cut], pos_tags[cut:]
    context, ngram = _ngram_counts(train_tags, pos_order)
    vocab = len(set(train_tags)) or 1
    ce_original, _ = _score_ngram(test_tags, context, ngram, pos_order, vocab, alpha)
    permutations = option(opts, "corruption_permutations", 3)
    shuffled_scores = []
    for i in range(permutations):
        rng = random.Random(seed + i)
        ce_shuffled, _ = _score_ngram(_shuffled(rng, test_tags), context, ngram, pos_order, vocab, alpha)
        if ce_shuffled is not None:
            shuffled_scores.append(ce_shuffled)
    if ce_original is None or not shuffled_scores:
        out.append(unavailable(f"{ID}pos_shuffle_ratio", "POS-tag shuffle degradation ratio",
                               "could not score the held-out POS sequence", family=FAMILY))
    else:
        mean_shuffled = sum(shuffled_scores) / len(shuffled_scores)
        ratio = mean_shuffled / ce_original if ce_original else None
        out.append(finding(
            f"{ID}pos_shuffle_ratio", "POS-tag shuffle degradation ratio (shuffled / original cross-entropy)",
            ratio, "ratio", family=FAMILY, sample_size=len(test_tags), min_sample=MIN_SAMPLE,
            sample_size_sensitive=True,
            distribution={"order": pos_order, "permutations": permutations, "seed": seed,
                         "original_bits_per_tag": ce_original, "shuffled_bits_per_tag_mean": mean_shuffled,
                         "note": ">1 means shuffling POS-tag order made this document's own POS "
                                 "model more surprised, i.e. tag order carried real structure"}))
    return out


def _group_corruption_baselines(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    ids_names = (
        (f"{ID}word_shuffle_ratio", "Word-shuffle degradation ratio (shuffled / original word bigram cross-entropy)"),
        (f"{ID}char_shuffle_ratio", "Character-shuffle degradation ratio (shuffled / original char n-gram cross-entropy)"),
        (f"{ID}sentence_shuffle_ratio", "Sentence-shuffle degradation ratio (shuffled / original char n-gram cross-entropy)"),
    )
    prep = _corruption_texts(analysis, opts)
    if prep is None:
        return [unavailable(mid, name, "fewer than 12 sentences to build a train/held-out split",
                            family=FAMILY) for mid, name in ids_names]

    alpha = option(opts, "lm_smoothing_alpha", 0.5)
    seed = option(opts, "corruption_seed", 1337)
    permutations = option(opts, "corruption_permutations", 3)
    word_order = option(opts, "corruption_word_order", 2)
    char_order = option(opts, "corruption_char_order", 4)
    train_words = prep["train_text"].split()
    test_words = prep["test_text"].split()
    train_chars = list(prep["train_text"])
    test_chars = list(prep["test_text"])

    def _ratio(train_seq, test_seq, order, shuffle_fn) -> tuple[float | None, dict[str, Any]]:
        if len(test_seq) < order * 4 or len(train_seq) < order * 4:
            return None, {}
        context, ngram = _ngram_counts(train_seq, order)
        vocab = len(set(train_seq)) or 1
        ce_original, _ = _score_ngram(test_seq, context, ngram, order, vocab, alpha)
        if ce_original is None or ce_original == 0:
            return None, {}
        scores = []
        for i in range(permutations):
            rng = random.Random(seed + i)
            ce_shuffled, _ = _score_ngram(shuffle_fn(rng, test_seq), context, ngram, order, vocab, alpha)
            if ce_shuffled is not None:
                scores.append(ce_shuffled)
        if not scores:
            return None, {}
        mean_shuffled = sum(scores) / len(scores)
        return mean_shuffled / ce_original, {"order": order, "original_bits": ce_original,
                                             "shuffled_bits_mean": mean_shuffled,
                                             "permutations": permutations, "seed": seed}

    out = []
    word_ratio, word_info = _ratio(train_words, test_words, word_order, _shuffled)
    out.append(finding(
        ids_names[0][0], ids_names[0][1], word_ratio, "ratio", family=FAMILY,
        sample_size=len(test_words), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution=word_info or None,
        warning=None if word_ratio is not None else "not enough held-out words for a shuffle baseline"))

    def _char_shuffle(rng, seq):
        return _shuffled(rng, seq)

    char_ratio, char_info = _ratio(train_chars, test_chars, char_order, _char_shuffle)
    out.append(finding(
        ids_names[1][0], ids_names[1][1], char_ratio, "ratio", family=FAMILY,
        sample_size=len(test_chars), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution=char_info or None,
        warning=None if char_ratio is not None else "not enough held-out characters for a shuffle baseline"))

    sentences = prep["test_sentences"]
    if len(sentences) < 3:
        out.append(unavailable(ids_names[2][0], ids_names[2][1],
                               "fewer than 3 held-out sentences to reorder", family=FAMILY))
        return out
    context, ngram = _ngram_counts(train_chars, char_order)
    vocab = len(set(train_chars)) or 1
    ce_original, _ = _score_ngram(test_chars, context, ngram, char_order, vocab, alpha)
    scores = []
    for i in range(permutations):
        rng = random.Random(seed + i)
        reordered = list(" ".join(_shuffled(rng, sentences)).lower())
        ce_shuffled, _ = _score_ngram(reordered, context, ngram, char_order, vocab, alpha)
        if ce_shuffled is not None:
            scores.append(ce_shuffled)
    if ce_original is None or ce_original == 0 or not scores:
        out.append(unavailable(ids_names[2][0], ids_names[2][1],
                               "could not score the held-out sentence order", family=FAMILY))
    else:
        mean_shuffled = sum(scores) / len(scores)
        out.append(finding(
            ids_names[2][0], ids_names[2][1], mean_shuffled / ce_original, "ratio", family=FAMILY,
            sample_size=len(test_chars), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
            distribution={"order": char_order, "permutations": permutations, "seed": seed,
                         "original_bits": ce_original, "shuffled_bits_mean": mean_shuffled,
                         "note": "a character n-gram model is local, so this ratio is expected to "
                                 "stay close to 1 even when sentence order is destroyed; a "
                                 "discourse-level shuffle signal needs the semantic-similarity "
                                 "metrics elsewhere in this codebase, not a compression/n-gram one"}))
    return out


def _group_lexical_gibberish(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    tokens = analysis.tokens
    out: list[dict[str, Any]] = []

    language = option(opts, "language", "en")
    module, reason = require("wordfreq")
    if module is None or not tokens:
        out.append(unavailable(f"{ID}unknown_word_rate", "Share of word types unknown to general-language frequencies",
                               reason if module is None else "no words to measure", family=FAMILY))
    else:
        counts = Counter(tokens)
        unknown_types = sum(1 for word in counts if module.zipf_frequency(word, language) <= 0.0)
        unknown_occurrences = sum(count for word, count in counts.items()
                                  if module.zipf_frequency(word, language) <= 0.0)
        out.append(finding(
            f"{ID}unknown_word_rate", "Share of word occurrences unknown to general-language frequencies",
            rate(unknown_occurrences, len(tokens)), "%", family=FAMILY, sample_size=len(tokens),
            min_sample=MIN_SAMPLE, sample_size_sensitive=True,
            distribution={"language": language, "distinct_unknown_types": unknown_types,
                         "distinct_types": len(counts),
                         "note": "a word is 'unknown' when wordfreq has never seen it for this "
                                 "language; proper nouns, coinages and typos all score the same way"}))

    letters = Counter(char for word in tokens for char in word if char in ENGLISH_LETTER_FREQ)
    total_letters = sum(letters.values())
    if total_letters:
        observed = {letter: count / total_letters for letter, count in letters.items()}
        divergence = _js_divergence(observed, ENGLISH_LETTER_FREQ)
        out.append(finding(
            f"{ID}letter_frequency_divergence",
            "Jensen-Shannon divergence of this text's letter frequencies from general English",
            divergence, "bits", family=FAMILY, sample_size=total_letters, min_sample=MIN_SAMPLE,
            sample_size_sensitive=True,
            distribution={"reference": "Lewand (2000) English single-letter frequency table",
                         "distinct_letters_observed": len(letters)}))
    else:
        out.append(unavailable(f"{ID}letter_frequency_divergence",
                               "Jensen-Shannon divergence of this text's letter frequencies from general English",
                               "no a-z letters found", family=FAMILY))

    cluster_min = option(opts, "consonant_cluster_min", 4)
    if tokens:
        flagged = 0
        for word in tokens:
            run = 0
            best = 0
            for char in word:
                if char.isalpha() and char not in VOWELS:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            if best >= cluster_min:
                flagged += 1
        out.append(finding(
            f"{ID}consonant_cluster_rate", f"Share of words containing a run of {cluster_min}+ consonants",
            rate(flagged, len(tokens)), "%", family=FAMILY, sample_size=len(tokens), min_sample=MIN_SAMPLE,
            sample_size_sensitive=True,
            distribution={"consonant_cluster_min": cluster_min,
                         "note": "a weak independent sensor, not a spellchecker; real words like "
                                 "'rhythm' or 'strengths' are flagged along with genuine gibberish"}))
    else:
        out.append(unavailable(f"{ID}consonant_cluster_rate",
                               f"Share of words containing a run of {cluster_min}+ consonants",
                               "no words to measure", family=FAMILY))

    if len(tokens) >= 2:
        repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
        out.append(finding(
            f"{ID}immediate_repeat_rate", "Rate of a word immediately repeating itself",
            rate(repeats, len(tokens) - 1), "%", family=FAMILY, sample_size=len(tokens) - 1,
            min_sample=MIN_SAMPLE, sample_size_sensitive=True,
            distribution={"note": "consecutive duplicate tokens only ('the the'); longer-range "
                                  "repeated phrases are style.repeated_ngrams's job, not this one's"}))
    else:
        out.append(unavailable(f"{ID}immediate_repeat_rate", "Rate of a word immediately repeating itself",
                               "fewer than two words", family=FAMILY))

    from .. import stats as stats_module
    char_cap = option(opts, "entropy_token_cap", 100_000)
    chars = list(analysis.text.lower()[:char_cap])
    if len(chars) >= 20:
        runs_by_label = stats_module.run_lengths(chars)
        all_runs = [length for lengths in runs_by_label.values() for length in lengths]
        run_h = numeric_shannon_entropy(all_runs) if all_runs else None
        longest = max(all_runs) if all_runs else 0
        longest_label = next((label for label, lengths in runs_by_label.items() if longest in lengths), None)
        out.append(finding(
            f"{ID}char_run_length_entropy", "Entropy of same-character run lengths",
            run_h, "bits", family=FAMILY, sample_size=len(all_runs), min_sample=MIN_SAMPLE,
            sample_size_sensitive=True,
            distribution={"characters_seen": len(chars), "longest_run": longest,
                         "longest_run_character": longest_label,
                         "note": "low entropy with a long longest run points at padding/template "
                                 "characters; near-uniform run lengths of 1 point at character-level "
                                 "randomness"}))
    else:
        out.append(unavailable(f"{ID}char_run_length_entropy", "Entropy of same-character run lengths",
                               "fewer than 20 characters", family=FAMILY))
    return out


# ------------------------------------------------- gibberish-detector package

def _group_gibberish_detector_package(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``gibberish-detector`` PyPI package, as an independent sensor beside
    this module's own hand-rolled consonant-cluster heuristic.

    ``gibberish-detector`` ships with a trainer, not a fixed pretrained
    model, and this channel uses it exactly the way it deliberately avoids
    using its own bundled example corpus: trained fresh on this document's
    own held-out split (:func:`_corruption_texts`, the same train/held-out
    convention every from-scratch n-gram channel in this module follows),
    never on a file shipped in the repo or in the package. That is a
    meaningfully different model from the one its own CLI trains (on a large
    curated English word list), so its raw score is reported on its own
    terms rather than compared against the package's own default
    ``is_gibberish`` threshold (4.0), which was calibrated for that other
    corpus and would silently misrepresent a document-trained model as
    "more confident" than it has any right to be.

    **What this channel actually measures, and why it disagrees with
    ``consonant_cluster_rate``:** the package's own n-gram iterator
    (``gibberish_detector.util.NGramIterator``) strips every character
    outside its charset - by default and here, whitespace and punctuation
    included - before taking adjacent-letter bigrams. That makes it
    materially blind to word order: shuffling word order changes only the
    handful of bigrams that cross a (now-removed) word boundary, leaving
    every bigram inside every word untouched, so ``shuffled_words`` scores
    only marginally higher than real prose on this channel even though a
    reader immediately sees scrambled nonsense - the same
    "structurally blind to this exact corruption" phenomenon this module's
    own docstring already documents for a local character n-gram model and
    sentence-order shuffling. Character-order and random-letter corruption,
    which do change adjacent-letter bigrams directly, register strongly.
    See the module docstring's "Judgment calls" for the measured numbers.

    A second, genuinely counter-intuitive result worth naming directly: a
    heavily repeated template (``the cat sat on the mat...`` on a loop)
    scores LOWER (less "gibberish") than ordinary varied prose, because
    this channel measures predictability, not meaning, and nothing is more
    predictable to a model trained on its own opening than its own
    unchanging refrain repeated back to it. That is not a bug in the
    channel; it is the same caveat this module already gives compression
    ratio and cross-entropy on repetitive text, restated for a third,
    independent measure that reaches the same conclusion by a different
    route -- which is exactly the kind of agreement-across-channels this
    module is built to surface.
    """

    mid = f"{ID}gibberish_detector_score"
    name = "gibberish-detector package score of held-out text (document-trained model)"
    module, reason = require("gibberish_detector")
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY)]

    prep = _corruption_texts(analysis, opts)
    if prep is None:
        return [unavailable(mid, name, "fewer than 12 sentences to build a train/held-out split",
                            family=FAMILY)]

    test_text = prep["test_text"]
    if len(test_text) < 50:
        return [unavailable(mid, name, "held-out text too short to score", family=FAMILY)]

    charset = str(option(opts, "gibberish_detector_charset", DEFAULTS["gibberish_detector_charset"]))
    try:
        # ``gibberish_detector``'s __init__.py imports nothing, so its
        # trainer/detector submodules are not attributes of the top-level
        # package `require()` already confirmed importable; they need their
        # own (equally safe, pure-Python, no-op if already imported) import.
        from gibberish_detector import trainer as gd_trainer, detector as gd_detector
        trained_model = gd_trainer.train_on_content(prep["train_text"], charset)
        det = gd_detector.Detector(trained_model, threshold=float("inf"))
        score = det.calculate_probability_of_being_gibberish(test_text)
    except Exception as exc:  # pragma: no cover - defensive; package is small and stable
        return [unavailable(mid, name, f"gibberish_detector failed ({type(exc).__name__}: {exc})",
                            family=FAMILY)]

    filtered_chars = sum(1 for char in test_text if char in charset)
    try:
        from gibberish_detector.__version__ import VERSION as _gd_version
    except Exception:  # pragma: no cover - defensive only
        _gd_version = None
    return [finding(
        mid, name, score, "score", family=FAMILY, sample_size=max(filtered_chars - 1, 0),
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"charset": charset, "train_chars": len(prep["train_text"]),
                     "test_chars": len(test_text), "filtered_bigrams_scored": max(filtered_chars - 1, 0),
                     "library": "gibberish-detector", "library_version": _gd_version,
                     "method": "a character-bigram model trained on this document's own held-out "
                               "training split (never on the package's bundled reference file), "
                               "scored against the held-out suffix; higher means less like the "
                               "training text, not a calibrated gibberish/not-gibberish threshold",
                     "note": "whitespace and punctuation are stripped before bigrams are taken "
                             "(the package's own design), so this channel is close to blind to "
                             "word order and should be read beside consonant_cluster_rate, not "
                             "in place of it -- see this function's docstring for the measured "
                             "corruption-ladder numbers"})]


# ------------------------------------------------------ neural language model

def _load_causal_lm(model_name: str) -> tuple[Any, Any, str | None]:
    """Load and cache a pretrained tokenizer+model pair. Imports nothing by itself.

    Callers must already have confirmed ``torch`` and ``transformers`` are
    importable (via ``optional.require``); this function only turns a model
    name into a ready-to-score (tokenizer, model) pair, or a reason it could
    not.
    """

    with _lm_model_lock:
        cached = _lm_model_cache.get(model_name)
    if cached is not None:
        return cached

    transformers, reason = require("transformers")
    if transformers is None:
        result = (None, None, reason)
    else:
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            result = (tokenizer, model, None)
        except Exception as exc:
            result = (None, None, f"could not load causal LM {model_name!r} "
                                  f"({type(exc).__name__}: {exc}); check the model name, network "
                                  f"access, or the Hugging Face cache")
    with _lm_model_lock:
        _lm_model_cache[model_name] = result
    return result


def _group_neural_language_model(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Perplexity under a pretrained causal LM (distilgpt2 by default).

    Off by default (``features.neural_language_model``); neither ``torch``
    nor ``transformers`` is imported (``optional.require`` is not even
    called) unless this group actually runs, so corpus profiling - which
    runs this suite's other, cheap channels over every reference book with
    ``MetricSpec.defaults`` - never triggers a model load. See
    ``test_neural_language_model_imports_nothing_under_default_config`` for
    the check that this stays true.

    **This is not a gibberish detector, and reads the wrong way on one of
    this suite's own test corruptions.** A causal LM's tokenizer splits text
    into subword (BPE) pieces from a fixed vocabulary; a string with no
    recognisable word in it does not fail to tokenize, it just fragments into
    many short, individually common pieces, each of which the model finds
    unsurprising on its own. A real sentence with its *word order* scrambled
    keeps every whole-word token (the vocabulary knows every word) but now in
    an order the model finds bizarre - and that surprise compounds one token
    at a time, order after order. So this channel can and does rank
    character-random letters as more "fluent" than word-order-scrambled real
    prose, even though the character n-gram and compression channels
    elsewhere in this module correctly rank random letters as the more
    random of the two. Measured on this exact model and default settings:
    "The lighthouse keeper walked down to the shore at dawn." scores ~243
    perplexity; the same sentence word-shuffled scores ~10,053; a same-length
    span of random lowercase letters scores ~303 - below the shuffled real
    sentence. Use this channel to ask whether a span reads as fluent,
    in-distribution English phrasing; use the character n-gram or
    compression channels above to detect gibberish.
    """

    mid = f"{ID}neural_lm_perplexity"
    name = "Causal-LM perplexity (pretrained, length-capped span)"
    torch, torch_reason = require("torch")
    if torch is None:
        return [unavailable(mid, name, torch_reason, family=FAMILY)]
    transformers, tf_reason = require("transformers")
    if transformers is None:
        return [unavailable(mid, name, tf_reason, family=FAMILY)]

    model_name = option(opts, "neural_lm_model", "distilgpt2")
    char_cap = option(opts, "neural_lm_max_chars", 6_000)
    text = analysis.text.strip()[:char_cap]
    if not text:
        return [unavailable(mid, name, "no text to score", family=FAMILY)]

    tokenizer, model, reason = _load_causal_lm(model_name)
    if model is None:
        return [unavailable(mid, name, reason, family=FAMILY)]

    try:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] < 4:
            return [unavailable(mid, name, "fewer than 4 model tokens to score", family=FAMILY)]
        with torch.no_grad():
            outcome = model(input_ids, labels=input_ids)
        loss_nats = float(outcome.loss)
    except Exception as exc:
        return [unavailable(mid, name, f"{model_name} failed ({type(exc).__name__}: {exc})",
                            family=FAMILY)]

    if not math.isfinite(loss_nats):
        return [unavailable(mid, name, f"{model_name} produced a non-finite loss", family=FAMILY)]

    n_tokens = int(input_ids.shape[1])
    perplexity = math.exp(loss_nats)
    return [finding(
        mid, name, perplexity, "perplexity", family=FAMILY, sample_size=n_tokens,
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"model": model_name, "framework": "transformers",
                     "cross_entropy_nats_per_token": loss_nats,
                     "cross_entropy_bits_per_token": loss_nats / math.log(2),
                     "tokens_scored": n_tokens, "characters_scored": len(text),
                     "tokenizer": "model-specific subword BPE vocabulary",
                     "warning_not_a_gibberish_detector": (
                         "subword tokenization can score purely random character strings as MORE "
                         "fluent (lower perplexity) than grammatically-scrambled real words; see "
                         "this group's docstring for a worked example and reasoning. Use the "
                         "character n-gram or compression channels above for gibberish detection.")})]


# --------------------------------------------------- textdescriptives cross-check

def _load_textdescriptives_pipeline(model_name: str) -> tuple[Any, str | None]:
    """Build and cache a spaCy pipeline with textdescriptives' information-theory
    component attached. A second, differently-configured pipeline from
    ``analysis.nlp``: textdescriptives' components must be present before
    parsing, so an already-parsed ``Doc`` from the shared pipeline cannot be
    reused here.
    """

    with _td_pipeline_lock:
        cached = _td_pipeline_cache.get(model_name)
    if cached is not None:
        return cached

    spacy, reason = require("spacy")
    if spacy is None:
        result = (None, reason)
    else:
        textdescriptives, td_reason = require("textdescriptives")
        if textdescriptives is None:
            result = (None, td_reason)
        else:
            try:
                nlp = spacy.load(model_name, disable=["ner"])
                nlp.add_pipe("textdescriptives/information_theory")
                result = (nlp, None)
            except Exception as exc:
                result = (None, f"could not build a textdescriptives pipeline on {model_name!r} "
                                f"({type(exc).__name__}: {exc}); run "
                                f"python -m spacy download {model_name}")
    with _td_pipeline_lock:
        _td_pipeline_cache[model_name] = result
    return result


def _group_textdescriptives_cross_check(analysis: DocumentAnalysis,
                                        opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """textdescriptives' word log-probability entropy/perplexity, reported next
    to (never reconciled with) this suite's own ``unknown_word_rate``.

    Off by default (``features.textdescriptives_cross_check``) because it
    needs its own spaCy parse - a second one, not the shared
    ``analysis.spacy_docs()`` parse ``features.pos_dependency`` uses, since
    textdescriptives' component has to be in the pipeline before parsing, not
    added to an already-parsed ``Doc``. Both channels ask "how unusual are
    this text's words against general English", but from different
    references (spaCy's static lexeme-probability table here, wordfreq's Zipf
    frequencies for ``unknown_word_rate``) and different formulas
    (textdescriptives sums ``-p * log(p)`` per token across the whole
    document rather than normalizing by token count, so its numbers do not
    live on a 0-1 or bits-per-symbol scale the way this suite's other entropy
    findings do). They are expected to disagree in scale; that disagreement
    is the reported result, not an error to fix.
    """

    mid = f"{ID}textdescriptives_word_perplexity"
    name = "Per-word log-probability perplexity (textdescriptives cross-check)"
    model_name = analysis.nlp_settings.model
    char_cap = option(opts, "textdescriptives_max_chars", 50_000)
    text = analysis.text[:char_cap]
    if len(text.split()) < 20:
        return [unavailable(mid, name, "fewer than 20 words to score", family=FAMILY)]

    nlp, reason = _load_textdescriptives_pipeline(model_name)
    if nlp is None:
        return [unavailable(mid, name, reason, family=FAMILY)]

    try:
        doc = nlp(text)
    except Exception as exc:
        return [unavailable(mid, name, f"textdescriptives parse failed "
                            f"({type(exc).__name__}: {exc})", family=FAMILY)]

    entropy = getattr(doc._, "entropy", None)
    perplexity = getattr(doc._, "perplexity", None)
    per_word = getattr(doc._, "per_word_perplexity", None)
    if per_word is None or not math.isfinite(per_word):
        return [unavailable(mid, name, "textdescriptives could not compute a per-word perplexity "
                            "for this span (missing lexeme-probability table for this model?)",
                            family=FAMILY)]

    return [finding(
        mid, name, per_word, "perplexity/word", family=FAMILY, sample_size=len(doc),
        min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        distribution={"source": "textdescriptives/information_theory (Doc._.entropy / "
                                "_.perplexity / _.per_word_perplexity, from spaCy token.prob "
                                "static lexeme frequencies)",
                     "spacy_model": model_name, "characters_scored": len(text),
                     "tokens_scored": len(doc),
                     "entropy_nats_raw_sum": entropy if entropy is not None and math.isfinite(entropy) else None,
                     "perplexity_raw": perplexity if perplexity is not None and math.isfinite(perplexity) else None,
                     "cross_check_of": f"{ID}unknown_word_rate",
                     "note": "a different reference table and a different formula from "
                             "unknown_word_rate (see this group's docstring); expect a different "
                             "scale, not agreement, and treat disagreement as the finding"})]


# ------------------------------------------------- NCD against reference documents

def _ncd_corpus_reference_texts(corpus_dirs: Sequence[str], max_documents: int,
                                max_bytes: int) -> tuple[list[tuple[str, bytes]], str | None, int]:
    """Up to ``max_documents`` reference texts, each truncated to
    ``max_bytes``, read fresh from ``corpus_dirs`` at GRADING time.

    Mirrors :func:`textgrader.metrics.stylometry_suite._ncd_reference_texts`
    exactly, for the same reason stated there: true NCD needs C(x), C(y) AND
    C(x+y), so both raw texts must exist in memory together at the moment of
    comparison, which is precisely what a corpus *profile* (no raw text
    retained; see this module's docstring's "Deferred" section and
    :mod:`textgrader.corpus`) cannot serve. Returns ``(rows, None,
    available)`` on success or ``([], reason, available)`` naming exactly
    what was missing -- no directories configured, a configured directory
    absent on disk, or one with no ``.txt``/``.md`` files -- with
    ``available`` the number of candidate files found BEFORE
    ``max_documents`` truncation.
    """

    if not corpus_dirs:
        return [], ("no corpus directory configured for NCD-against-corpus (set "
                    "metrics.randomness_suite.ncd_corpus_dirs to one or more directories of "
                    "reference .txt/.md files)"), 0
    found: list[tuple[str, Path]] = []
    missing: list[str] = []
    for raw_dir in corpus_dirs:
        directory = Path(str(raw_dir)).expanduser()
        if not directory.is_dir():
            missing.append(str(raw_dir))
            continue
        for pattern in ("*.txt", "*.md"):
            for path in directory.rglob(pattern):
                if path.is_file():
                    found.append((path.relative_to(directory).as_posix(), path))
    if not found:
        if missing and len(missing) == len(corpus_dirs):
            return [], f"none of the configured ncd_corpus_dirs exist on disk: {', '.join(missing)}", 0
        return [], ("the configured ncd_corpus_dirs exist but contain no .txt/.md reference "
                    "files" + (f" (also missing: {', '.join(missing)})" if missing else "")), 0
    found.sort(key=lambda item: item[0])
    available = len(found)
    rows: list[tuple[str, bytes]] = []
    for name, path in found[:max_documents]:
        try:
            data = path.read_bytes()[:max_bytes]
        except OSError:
            continue
        if data:
            rows.append((name, data))
    if not rows:
        return [], "reference documents were found but none could be read", available
    return rows, None, available


def _group_ncd_against_corpus(analysis: DocumentAnalysis, opts: Mapping[str, Any]) -> list[dict[str, Any]]:
    """True Normalized Compression Distance against real reference documents
    read from disk (see :func:`_ncd_corpus_reference_texts`), as distinct
    from ``ncd_word_shuffle``/``ncd_char_shuffle``/``ncd_sentence_shuffle``
    above, which are NCD against seeded corruptions of THIS document (no
    filesystem access, always on by default).

    This is the design settled on for the gap the module docstring used to
    call permanently out of scope ("nothing reaches this module except one
    document's DocumentAnalysis and, optionally, a corpus profile of scalar
    distributions and pooled counts, never raw text"): a measurement that
    genuinely needs raw reference text and cannot be served from a cached
    profile reads the corpus folder directly, at grading time, behind its
    own feature switch (``features.ncd_against_corpus``, off by default) so
    it never runs as a side effect of enabling this suite, and it is a
    second time gated on ``ncd_corpus_dirs`` actually pointing somewhere
    reachable. ``stylometry_suite``'s ``_ncd_against_corpus_findings``
    implements the identical pattern (same off-by-default feature name, same
    ``ncd_corpus_dirs`` option name) for its own compressor options; this is
    that same design, mirrored here with THIS suite's own multi-algorithm
    ``_compress`` (any of ``compression_algorithms``, not only zlib/lzma) and
    its own bounds (``ncd_corpus_max_reference_documents``,
    ``ncd_corpus_max_bytes``) so the two suites' identically-named
    ``ncd_corpus_dirs`` can point at the same folder without the two
    measurements sharing a cache key or a byte budget.

    Every finding records the algorithm, the byte caps and exactly how many
    reference documents were compared in its ``distribution``, so a reader
    can tell a real "nothing looked similar" from "only three reference
    documents were readable".

    **``ncd_corpus_algorithm`` defaults to ``lzma``, not ``zlib``, and this is
    not a stylistic choice.** NCD needs ``compress(x + y)`` to be able to
    reference back into ``x`` while it encodes ``y``. zlib/gzip's DEFLATE
    window is a fixed 32768 bytes (LZ4 and, empirically, snappy break down
    a little past 65536); this suite's byte caps default to
    ``ncd_corpus_max_bytes = 100_000`` per side, so a ``zlib`` run here
    concatenates roughly 200,000 bytes into a compressor that can only ever
    see the last 32768 of them. The result is not merely noisy, it is
    *unable to discriminate at all*: measured directly on two 100 KB real
    book excerpts, ``zlib`` gave NCD(x, x) = 0.97 and NCD(x, y) = 0.98 for an
    unrelated y -- an identical document and a different one score the same,
    because the compressor genuinely cannot see the earlier copy, not
    because of anything about how the text was produced or normalized. The
    same pair scores NCD(x, x) = 0.002 and NCD(x, y) = 0.95 under ``lzma``
    (dictionary >= 256 KiB at every preset), which is why it is the default.
    ``zstd``, ``brotli`` and ``bz2`` (block size = ``compression_level`` *
    100,000 bytes, so >= 100,000 at this suite's default level 6) are large
    enough to work at these byte caps too; only ``zlib``, ``gzip``, ``lz4``
    and ``snappy`` have a small enough window to matter here, and the guard
    below refuses to report a number for exactly those four when the
    configured byte caps exceed their window, rather than silently returning
    a "different" score for an identical pair. See
    ``_compressor_window_bytes`` for the exact figures and their sources,
    and ``distribution.compressor_window_bytes`` on every finding here (and
    on ``ncd_word_shuffle``/``ncd_char_shuffle``/``ncd_sentence_shuffle``
    above) for the number that applied to it.
    """

    ids = (
        (f"{ID}ncd_corpus_nearest_reference",
         "Normalized Compression Distance to the nearest reference document"),
        (f"{ID}ncd_corpus_reference_mean",
         "Mean Normalized Compression Distance across sampled reference documents"),
    )

    def _unavailable(warning: str, size: int) -> list[dict[str, Any]]:
        return [finding(mid, name, None, "ratio", family=FAMILY, sample_size=size, min_sample=200,
                        warning=warning) for mid, name in ids]

    word_count = analysis.word_count
    if word_count < 200:
        return _unavailable(f"only {word_count} words; need at least 200 to measure compressibility",
                            word_count)

    algorithm = str(option(opts, "ncd_corpus_algorithm", DEFAULTS["ncd_corpus_algorithm"]))
    level = option(opts, "compression_level", DEFAULTS["compression_level"])
    max_documents = max(1, int(option(opts, "ncd_corpus_max_reference_documents",
                                      DEFAULTS["ncd_corpus_max_reference_documents"])))
    max_bytes = max(1000, int(option(opts, "ncd_corpus_max_bytes", DEFAULTS["ncd_corpus_max_bytes"])))
    raw_dirs = option(opts, "ncd_corpus_dirs", DEFAULTS["ncd_corpus_dirs"])
    corpus_dirs = list(raw_dirs) if isinstance(raw_dirs, (list, tuple)) else ([raw_dirs] if raw_dirs else [])

    doc_bytes = analysis.text.encode("utf-8")[:max_bytes]
    window = _compressor_window_bytes(algorithm, level)

    # Refuse to report a number this compressor cannot actually compute,
    # rather than silently returning "different" for an identical pair (see
    # this function's docstring for the measured numbers behind this guard).
    # Sized against the CONFIGURED max_bytes, not the actual reference byte
    # count, so the same configuration always gives the same answer whether
    # a given reference document happens to be shorter than the cap or not.
    if window is not None and len(doc_bytes) + max_bytes > window:
        safe_max_bytes = max(1000, window // 2)
        return _unavailable(
            f"{algorithm} has a fixed {window}-byte compression window, but the configured "
            f"ncd_corpus_max_bytes ({max_bytes}) lets a document/reference pair reach "
            f"{len(doc_bytes) + max_bytes} bytes combined -- {algorithm} cannot see back that "
            f"far, so it would report every reference as maximally different, including this "
            f"document compared against itself. Lower ncd_corpus_max_bytes to at most "
            f"{safe_max_bytes}, or set ncd_corpus_algorithm to lzma (the default), whose "
            f"dictionary is megabytes rather than kilobytes.",
            word_count)

    references, reason, available = _ncd_corpus_reference_texts(corpus_dirs, max_documents, max_bytes)
    if not references:
        return _unavailable(reason, word_count)

    def _len(data: bytes) -> int | None:
        compressed, _ = _compress(algorithm, data, level)
        return len(compressed) if compressed is not None else None

    c_doc = _len(doc_bytes)
    if c_doc is None:
        return _unavailable(f"{algorithm} unavailable or produced empty output", word_count)

    distances: list[tuple[str, float]] = []
    for ref_name, ref_bytes in references:
        c_ref = _len(ref_bytes)
        c_joint = _len(doc_bytes + ref_bytes)
        if c_ref is None or c_joint is None:
            continue
        denominator = max(c_doc, c_ref)
        if denominator:
            distances.append((ref_name, (c_joint - min(c_doc, c_ref)) / denominator))
    if not distances:
        return _unavailable("no reference document produced a comparable compressed size", word_count)

    distances.sort(key=lambda item: item[1])
    nearest_name, nearest_ncd = distances[0]
    mean_ncd = sum(d for _, d in distances) / len(distances)
    common = {"algorithm": algorithm, "level": level, "reference_documents_compared": len(distances),
             "reference_documents_read": len(references), "reference_documents_available": available,
             "max_reference_documents": max_documents, "max_bytes_per_document": max_bytes,
             "document_bytes_compared": len(doc_bytes), "compressor_window_bytes": window,
             "note": "near 0 means this document compresses almost as well jointly with a "
                     "reference book as either does alone (shares a lot of compressible "
                     "structure with it); near 1 means the two share almost nothing"}
    evidence = [{"reference": rname, "ncd": d} for rname, d in distances[:25]]
    return [
        finding(ids[0][0], ids[0][1], nearest_ncd, "ratio", family=FAMILY, sample_size=word_count,
                min_sample=200, sample_size_sensitive=True, distribution=common,
                evidence=[{"reference": nearest_name, "ncd": nearest_ncd}]),
        finding(ids[1][0], ids[1][1], mean_ncd, "ratio", family=FAMILY, sample_size=word_count,
                min_sample=200, sample_size_sensitive=True, distribution=common, evidence=evidence),
    ]


# ---------------------------------------------------------------------- main

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    features = _features(config)
    out: list[dict[str, Any]] = []

    if features.get("char_entropy", True):
        out.extend(_group_char_entropy(analysis, config or {}))
    if features.get("complexity_measures", True):
        out.extend(_group_complexity(analysis, config or {}))
    if features.get("compression", True):
        out.extend(_group_compression(analysis, config or {}))
    if features.get("language_model", True):
        out.extend(_group_language_model(analysis, config or {}))
    if features.get("ppm_language_model", True):
        out.extend(_group_ppm_language_model(analysis, config or {}))
    if features.get("punctuation_sequence", True):
        out.extend(_group_punctuation_sequence(analysis, config or {}))
    if features.get("pos_dependency", False):
        out.extend(_group_pos_dependency(analysis, config or {}))
    if features.get("corruption_baselines", True):
        out.extend(_group_corruption_baselines(analysis, config or {}))
    if features.get("lexical_gibberish", True):
        out.extend(_group_lexical_gibberish(analysis, config or {}))
    if features.get("gibberish_detector_package", False):
        out.extend(_group_gibberish_detector_package(analysis, config or {}))
    if features.get("letter_bigram_divergence", True):
        out.extend(_group_letter_bigram_divergence(analysis, config or {}))
    if features.get("kenlm_language_model", False):
        out.extend(_group_kenlm_language_model(analysis, config or {}))
    if features.get("neural_language_model", False):
        out.extend(_group_neural_language_model(analysis, config or {}))
    if features.get("textdescriptives_cross_check", False):
        out.extend(_group_textdescriptives_cross_check(analysis, config or {}))
    if features.get("ncd_against_corpus", False):
        out.extend(_group_ncd_against_corpus(analysis, config or {}))

    return out
