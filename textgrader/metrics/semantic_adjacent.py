"""Sentence-to-sentence semantic similarity, and the shared machinery for the
whole ``semantic_repetition`` family.

Restating the same idea in different words is invisible to every lexical
repetition metric in this codebase: ``repeated_ngrams`` and
``lexical_repetition_distance`` both require the same words to come back, and
a writer who pads a scene by paraphrasing a sentence a moment after saying it
uses none. Embedding the sentences and measuring their cosine similarity
catches that directly: two sentences that mean almost the same thing embed
close together whether or not they share a single word.

``sentence-transformers`` is an optional, heavy dependency (it pulls in
PyTorch) and is not always installed. Every metric in this family therefore
has two backends, and a finding always says which one produced its value
because the two measure different things and must never be read as
interchangeable:

``embedding``
    cosine similarity of ``sentence-transformers`` sentence embeddings. This
    is the real target: distributional meaning, robust to paraphrase.
``lexical`` (the fallback used in this environment)
    cosine similarity of TF-IDF-weighted content-word vectors, with the IDF
    built from the document's own sentences. This is a lexical-overlap proxy
    only: it will flag a sentence restated with mostly the same nouns and
    verbs, but it will not catch a paraphrase that swaps out the vocabulary
    (a real embedding would), and it will over-flag two unrelated sentences
    that happen to share a rare content word. Every fallback finding says so
    in its ``warning`` and tags its evidence with ``"backend": "lexical"`` so
    a reader can never mistake one quantity for the other.

Shared here, and imported by ``semantic_window``, ``semantic_paragraph`` and
``semantic_clusters``, so that:

* the model loads at most once per process (``_MODEL_CACHE``), because
  loading ``SentenceTransformer`` is the single most expensive step and four
  metrics enabled together would otherwise pay for it four times;
* a document's sentence and paragraph vectors are computed at most once per
  document (``_VECTOR_CACHE``), keyed on ``id(analysis)`` rather than on the
  (unhashable, mutable-by-default) ``DocumentAnalysis`` itself. A bounded,
  small ``OrderedDict`` is used rather than ``functools.lru_cache`` because
  ``id()`` can be recycled once an object is garbage collected; each entry
  therefore also keeps a ``weakref`` to the analysis it was computed from and
  a lookup is only a hit if that weakref still resolves to the same object.
"""

from __future__ import annotations

import math
import weakref
from collections import Counter, OrderedDict
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from ..optional import on_reset, require
from ..stats import summarize
from .common import MODERATE, finding, option, tokens as tokenize

FAMILY = "semantic_repetition"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("sentence_transformers",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

DEFAULT_MODEL = "all-MiniLM-L6-v2"
EVIDENCE_SNIPPET_CHARS = 110

# A short function-word list for the fallback's content-word vectors. It is
# deliberately small and closed-class rather than an attempt at a full stop
# list; the goal is only to keep "the", "and", "was" from dominating every
# sentence's vector, not to build a linguistically complete filter.
STOPWORDS = frozenset("""
a an the and or but if of at by for with about against between into through
during before after above below to from up down in out on off over under
again further then once here there when where why how all any both each few
more most other some such no nor not only own same so than too very s t can
will just don should now is am are was were be been being have has had
having do does did doing would could might must shall this that these those
it its it's he she they them his her their our your my me him us we you i
as which who whom what also because while until whether either neither
""".split())


def truncate(text: str, limit: int = EVIDENCE_SNIPPET_CHARS) -> str:
    """Collapse whitespace and cut to ``limit`` chars for a quotable evidence row."""

    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


# ---------------------------------------------------------------- model cache

_MODEL_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_model_cache() -> None:
    """Drop the loaded model and any recorded "unavailable" answer."""

    _MODEL_CACHE.clear()


on_reset(_reset_model_cache)


def _load_model(model_name: str) -> tuple[Any, str | None]:
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    module, reason = require("sentence_transformers")
    if module is None:
        _MODEL_CACHE[model_name] = (None, reason)
        return _MODEL_CACHE[model_name]
    try:
        model = module.SentenceTransformer(model_name)
        outcome: tuple[Any, str | None] = (model, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"sentence-transformers model {model_name!r} unavailable "
                         f"({type(exc).__name__}: {exc})")
    _MODEL_CACHE[model_name] = outcome
    return outcome


def embed_texts(texts: Sequence[str], model_name: str) -> tuple[Any, str | None]:
    """``(embeddings, None)`` as a normalized 2D array, or ``(None, reason)``."""

    model, reason = _load_model(model_name)
    if model is None:
        return None, reason
    try:
        vectors = model.encode(list(texts), batch_size=64, show_progress_bar=False,
                               convert_to_numpy=True, normalize_embeddings=True)
        return vectors, None
    except Exception as exc:  # pragma: no cover - runtime/OOM failure
        return None, f"sentence-transformers encoding failed ({type(exc).__name__}: {exc})"


# ------------------------------------------------------------ lexical fallback

def _content_tokens(text: str) -> list[str]:
    return [word for word in tokenize(text) if len(word) > 2 and word not in STOPWORDS]


def lexical_vectors(units: Sequence[str]) -> list[dict[str, float]]:
    """TF-IDF content-word vectors, L2-normalized, IDF built from ``units`` itself.

    Building the IDF from the document being measured (rather than a general
    corpus) is what keeps this dependency-free: a word only counts as
    "distinctive" relative to the rest of this text, which is exactly the
    comparison a repetition metric needs.
    """

    tokenized = [_content_tokens(unit) for unit in units]
    doc_freq: Counter[str] = Counter()
    for words in tokenized:
        doc_freq.update(set(words))
    n = len(tokenized)
    idf = {word: math.log((1 + n) / (1 + count)) + 1.0 for word, count in doc_freq.items()}
    vectors: list[dict[str, float]] = []
    for words in tokenized:
        weights: dict[str, float] = {}
        for word, count in Counter(words).items():
            weights[word] = count * idf[word]
        norm = math.sqrt(sum(value * value for value in weights.values()))
        vectors.append({word: value / norm for word, value in weights.items()} if norm else {})
    return vectors


def _sparse_dot(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(word, 0.0) for word, value in a.items())


def similarity_at(backend: str, vectors: Any, i: int, j: int) -> float:
    """Cosine similarity between units ``i`` and ``j`` under ``backend``'s vectors."""

    if backend == "embedding":
        return float(vectors[i].dot(vectors[j]))
    return _sparse_dot(vectors[i], vectors[j])


def backend_note(backend: str, model_name: str, reason: str | None) -> str:
    """The sentence every finding in this family uses to name its backend.

    Deliberately placed in ``warning`` even on the success path (not only when
    something went wrong): an embedding similarity and a lexical-overlap
    similarity are not the same quantity, and a reader must never infer one
    from a number that is actually the other.
    """

    if backend == "embedding":
        return (f"backend=embedding: cosine similarity of {model_name!r} "
                f"sentence-transformers embeddings")
    detail = f" ({reason})" if reason else ""
    # Concrete, because "proxy" is easy to skim past. On the pair
    # "The dog was extremely happy to see her." / "The canine was overjoyed at
    # her arrival." the embedding backend scores 0.79 and this one scores 0.00,
    # which is the whole job of the metric going undone.
    return (f"backend=lexical: sentence-transformers unavailable{detail}; used a "
            f"dependency-free TF-IDF content-word cosine fallback instead. This is a "
            f"lexical-overlap proxy, NOT semantic similarity. A paraphrase that changes "
            f"its content words scores near zero here and near 0.8 with embeddings, so a "
            f"low value from this backend is not evidence of no repetition. "
            f"Install sentence-transformers (pip install -r requirements-embeddings.txt) "
            f"for the measurement this metric is for")


# ------------------------------------------------------- per-document caching

class _CacheEntry:
    __slots__ = ("ref", "value")

    def __init__(self, analysis: DocumentAnalysis, value: Any) -> None:
        self.ref = weakref.ref(analysis)
        self.value = value


_VECTOR_CACHE: "OrderedDict[tuple[int, str, str], _CacheEntry]" = OrderedDict()
_VECTOR_CACHE_MAX = 8


def _cache_get(analysis: DocumentAnalysis, kind: str, model_name: str) -> Any:
    key = (id(analysis), kind, model_name)
    entry = _VECTOR_CACHE.get(key)
    if entry is None or entry.ref() is not analysis:
        return None
    _VECTOR_CACHE.move_to_end(key)
    return entry.value


def _cache_put(analysis: DocumentAnalysis, kind: str, model_name: str, value: Any) -> None:
    key = (id(analysis), kind, model_name)
    _VECTOR_CACHE[key] = _CacheEntry(analysis, value)
    _VECTOR_CACHE.move_to_end(key)
    while len(_VECTOR_CACHE) > _VECTOR_CACHE_MAX:
        _VECTOR_CACHE.popitem(last=False)


def _unit_vectors(analysis: DocumentAnalysis, kind: str, units: Sequence[str],
                  model_name: str) -> tuple[str, Any, str]:
    cached = _cache_get(analysis, kind, model_name)
    if cached is not None:
        return cached
    if not units:
        result = ("lexical", [], "no text to compare")
    else:
        vectors, reason = embed_texts(units, model_name)
        if vectors is not None:
            result = ("embedding", vectors, backend_note("embedding", model_name, None))
        else:
            result = ("lexical", lexical_vectors(units), backend_note("lexical", model_name, reason))
    _cache_put(analysis, kind, model_name, result)
    return result


def get_sentence_vectors(analysis: DocumentAnalysis, model_name: str) -> tuple[str, Any, str]:
    """``(backend, vectors, backend_note)`` for ``analysis.sentences``, cached."""

    return _unit_vectors(analysis, "sentence", analysis.sentences, model_name)


def get_paragraph_vectors(analysis: DocumentAnalysis, model_name: str) -> tuple[str, Any, str]:
    """``(backend, vectors, backend_note)`` for ``analysis.paragraphs``, cached."""

    return _unit_vectors(analysis, "paragraph", analysis.paragraphs, model_name)


# -------------------------------------------------------------------- metric

HIGH_THRESHOLDS = (0.8, 0.9)

_IDS_NAMES = (
    ("semantic.adjacent_sentence_similarity", "Adjacent-sentence semantic similarity"),
    ("semantic.adjacent_similarity_high_share", "Share of adjacent sentence pairs restating"
                                                " the previous one"),
)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Cosine similarity between each sentence and the one right after it.

    Catches restating the same idea in different words one beat later - the
    kind of padding that leaves no lexical trace for an n-gram or word-reuse
    metric to find. It says nothing about repetition that skips a sentence or
    more; see ``semantic_window`` for that.
    """

    model_name = option(config, "model", DEFAULT_MODEL)
    sentences = analysis.sentences
    pair_count = max(0, len(sentences) - 1)
    if pair_count == 0:
        warning = f"needs at least two sentences; this text has {len(sentences)}"
        return [finding(mid, name, None, "cosine" if "share" not in mid else "%",
                        family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning)
                for mid, name in _IDS_NAMES]

    backend, vectors, note = get_sentence_vectors(analysis, model_name)
    values = [similarity_at(backend, vectors, i, i + 1) for i in range(pair_count)]

    distribution = summarize(values)
    ranked = sorted(range(pair_count), key=lambda i: -values[i])[:15]
    evidence = [{
        "backend": backend,
        "sentence_index": i,
        "similarity": values[i],
        "sentence_a": truncate(sentences[i]),
        "sentence_b": truncate(sentences[i + 1]),
    } for i in ranked]

    high_shares = {
        threshold: 100.0 * sum(1 for value in values if value >= threshold) / pair_count
        for threshold in HIGH_THRESHOLDS
    }

    return [
        finding("semantic.adjacent_sentence_similarity", "Adjacent-sentence semantic similarity",
                distribution.get("median"), "cosine", family=FAMILY, sample_size=pair_count,
                distribution=distribution, evidence=evidence, min_sample=MIN_SAMPLE,
                warning=note),
        finding("semantic.adjacent_similarity_high_share",
                "Share of adjacent sentence pairs restating the previous one",
                high_shares[HIGH_THRESHOLDS[0]], "%", family=FAMILY, sample_size=pair_count,
                min_sample=MIN_SAMPLE, warning=note,
                details=[{"backend": backend, "threshold": threshold, "share_percent": share}
                         for threshold, share in high_shares.items()]),
    ]
