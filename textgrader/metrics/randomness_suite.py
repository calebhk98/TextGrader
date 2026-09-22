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
graded, never on the whole thing.** There is no bundled or downloaded
reference language model in this codebase (no KenLM binary, no pretrained
weights), so "reference likelihood" here means: hold back a suffix of the
document's own sentences, fit an n-gram model with additive smoothing on
everything before it, and score the held-out suffix. This is a legitimate,
testable, leave-one-out measurement of how internally predictable a text is at
several linguistic levels; it is NOT a claim about English in general, and
every finding says so in its ``distribution``. The corruption-baseline group
reuses the exact same trained model to score a corrupted version of the same
held-out text, which is what turns "predictable to itself" into "sensitive to
word order / character order / sentence order", per the brief.

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
  only (a small, well-established 26-value table). A letter-bigram reference
  table would need either a licensed frequency list or a general corpus this
  codebase does not ship, so bigram divergence is not implemented (see
  Deferred) rather than approximated from something unverifiable.
* The "unpronounceable cluster" and "known-word rate" channels are
  deliberately weak, independent sensors, not a spellchecker: a heuristic
  consonant-run count will flag real words ("rhythm", "strengths") and a
  wordfreq lookup will miss real names. That is fine; they are one vote each,
  reported for what they are.

**Deferred** (named, not silently skipped):

* KenLM, the NLTK language-model module, and MALLET are not used. KenLM needs
  a compiled binary and a trained ARPA model this environment cannot build or
  download; shipping a from-scratch character n-gram trainer (above) is the
  stdlib-only substitute the task brief explicitly allows. NLTK's own
  smoothing classes were skipped in favor of one Lidstone-smoothed
  implementation shared by every n-gram channel here (char, byte, word, POS,
  dependency, punctuation), so all six report comparably smoothed numbers
  instead of six different smoothing behaviours.
* ``gibberish-detector`` is not installed and is not added as a dependency;
  the consonant-cluster heuristic here is this suite's independent weak
  sensor for the same phenomenon, implemented so it degrades to nothing
  rather than to an import error.
* AntroPy, EntropyHub and dit are not added as dependencies. None is
  installed in this environment, all of the specific measures the task names
  from them (permutation, spectral, SVD, approximate, sample entropy, LZ
  complexity) have short, well-known closed-form definitions, and
  implementing them directly on top of NumPy means they degrade to "NumPy
  unavailable" instead of "one more third-party package unavailable", and are
  easy to unit-test against the textbook formulas.
* Normalized Compression Distance against a whole reference corpus is not
  implemented. Nothing reaches this module except one document's
  ``DocumentAnalysis`` and (optionally) a corpus *profile* of scalar
  distributions; there is no representative-document store to compare
  against without extending the corpus-builder pipeline, which is out of
  this task's file scope. What IS implemented is NCD between a document and
  seeded corruptions of itself (word/char/sentence shuffles), which answers
  a related, self-contained question: how much does compressibility change
  when this specific kind of structure is destroyed.
* Byte n-gram cross-entropy is reported at one configurable order rather
  than swept across orders like the character channel: for UTF-8 English
  prose the byte and character streams are almost identical past order 2,
  so a full sweep would mostly restate the character sweep at higher cost.
* POS and dependency-label n-gram perplexity are real, implemented measures,
  but sit behind ``features.pos_dependency`` (off by default) because they
  need the shared spaCy parse, which is the one part of this suite that would
  otherwise turn a "moderate"-cost metric into a "parse"-cost one for every
  user regardless of whether they wanted this particular channel.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
import math
import random
import zlib
from collections import Counter
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..optional import require
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
    """

    sentences = analysis.sentences
    if len(sentences) < 12:
        return None
    fraction = option(opts, "corruption_sentence_fraction", 0.3)
    split = max(1, int(len(sentences) * (1 - fraction)))
    train_sents, test_sents = sentences[:split], sentences[split:]
    if not test_sents:
        return None
    cap = option(opts, "corruption_max_chars", 20_000)
    train_text = " ".join(train_sents).lower()[:cap]
    test_text = " ".join(test_sents).lower()[:cap]
    if len(train_text) < 200 or len(test_text) < 50:
        return None
    return {"train_text": train_text, "test_text": test_text, "test_sentences": test_sents}


def _shuffled(rng: random.Random, items: Sequence[Any]) -> list[Any]:
    out = list(items)
    rng.shuffle(out)
    return out


# --------------------------------------------------------------- compression

def _compress(name: str, data: bytes, level: int) -> tuple[bytes | None, str | None]:
    try:
        if name == "zlib":
            return zlib.compress(data, level), None
        if name == "gzip":
            return gzip.compress(data, compresslevel=level), None
        if name == "bz2":
            return bz2.compress(data, compresslevel=max(1, min(level, 9))), None
        if name == "lzma":
            return lzma.compress(data, preset=max(0, min(level, 9))), None
        if name == "zstd":
            module, reason = require("zstandard")
            if module is None:
                return None, reason
            return module.ZstdCompressor(level=level).compress(data), None
        if name == "brotli":
            module, reason = require("brotli")
            if module is None:
                return None, reason
            return module.compress(data, quality=max(0, min(level, 11))), None
        if name == "lz4":
            module, reason = require("lz4")
            if module is None:
                return None, reason
            return module.compress(data, compression_level=level), None
        if name == "snappy":
            module, reason = require("snappy")
            if module is None:
                return None, reason
            return module.compress(data), None
        if name == "ppmd":
            module, reason = require("pyppmd")
            if module is None:
                return None, reason
            return module.compress(data), None
        return None, f"unknown compressor {name!r}"
    except Exception as exc:  # pragma: no cover - defensive; a codec may reject input
        return None, f"{name} failed ({type(exc).__name__}: {exc})"


def _library_version(name: str) -> str | None:
    versions = {"zlib": zlib.ZLIB_VERSION}
    if name in versions:
        return versions[name]
    module_name = {"zstd": "zstandard", "brotli": "brotli", "lz4": "lz4",
                   "snappy": "snappy", "ppmd": "pyppmd"}.get(name)
    if module_name is None:
        return None
    module, _ = require(module_name)
    return getattr(module, "__version__", None) if module is not None else None


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
        out.append(finding(
            f"{ID}compression_ratio_{name}", f"Compression ratio ({name}, fixed block)", ratio, "ratio",
            family=FAMILY, sample_size=len(block), min_sample=MIN_SAMPLE, sample_size_sensitive=True,
            distribution={"algorithm": name, "level": level, "block_chars": len(block),
                         "original_bytes": original, "compressed_bytes": compressed_len,
                         "bits_per_char": bits_per_char, "library_version": _library_version(name),
                         "encoding": "utf-8, lowercased canonical text"}))

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
    if features.get("punctuation_sequence", True):
        out.extend(_group_punctuation_sequence(analysis, config or {}))
    if features.get("pos_dependency", False):
        out.extend(_group_pos_dependency(analysis, config or {}))
    if features.get("corruption_baselines", True):
        out.extend(_group_corruption_baselines(analysis, config or {}))
    if features.get("lexical_gibberish", True):
        out.extend(_group_lexical_gibberish(analysis, config or {}))

    return out
