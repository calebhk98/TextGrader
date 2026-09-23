"""Named, ordered numeric channels: the raw material for time-series metrics.

Every metric in this repository until now has treated a document's sentence
lengths, punctuation counts, parse depths and so on as an unordered sample: a
mean, a median, a distribution shape.  ``rhythm_autocorrelation`` and its
siblings are the one exception, and only for a single channel (sentence
length).  This module generalizes that idea: it names, builds and caches
every ordered numeric channel a metric might want to run classical
time-series analysis over, so that :mod:`textgrader.metrics.timeseries_suite`
(and any future metric) can ask for ``"sentence_words"`` or
``"window_pronoun_rate"`` by a stable name instead of re-deriving it.

A :class:`Sequence` is four things, matching the "Implementation details"
section of the task this module was built for: a **name** (the stable key
below), **values** (floats, in reading order -- position in the list *is* the
time axis), a **unit**, a **sample unit** (``sentence``, ``paragraph`` or
``window`` -- what one position in the series corresponds to), and
**provenance** (the settings that produced it, so a corpus profile can tell a
sequence built from 2,000-word windows apart from one built from 500-word
windows).  Nothing here computes a time-series *feature* -- no autocorrelation,
no trend, no entropy.  This module's only job is to answer "what ordered
channels does this document expose, and how do I get one", cheaply and
repeatably.

Every builder reuses :class:`~textgrader.document.DocumentAnalysis` and never
re-tokenizes, re-splits or re-parses: sentence and paragraph channels read
``analysis.sentences``/``analysis.sentence_lengths``/``analysis.paragraphs``
directly, parse-dependent channels read the one shared spaCy parse via
``analysis.spacy_docs()``, and window channels read ``analysis.windows()``,
the same paragraph-aligned slicing the ``book_drift`` family already shares
between three metrics.  Every sequence is memoized through
``analysis.memo(...)`` (which is exactly ``DocumentAnalysis._shared``, the
cache the task's implementation notes ask for) keyed on its name and its
settings, so two features that want the same sequence under the same settings
pay for it once, and two windowings of the same sequence are never confused
for each other.

Sequences that need an optional package (spaCy for the parse-derived
channels, ``wordfreq`` for rarity, ``sentence_transformers`` for embeddings)
degrade to an empty :class:`Sequence` carrying a ``warning`` that names the
reason, exactly like every other optional measurement in this codebase --
never an exception, never a fabricated channel.  The two embedding-backed
sequences (:data:`sentence_similarity_prev`, :data:`sentence_distance_centroid`)
reuse :mod:`textgrader.metrics.semantic_adjacent`'s model loading, per-document
vector cache and TF-IDF lexical fallback rather than re-implementing the
embedding/no-embedding split a third time in this codebase; that module
already carries the full explanation of why the fallback is a lexical-overlap
proxy and not a substitute measurement, and this module's findings pass its
``backend`` note through unchanged. Both sequences have now run end to end on
the real ``sentence-transformers`` backend, not only the fallback: on a short
paraphrase-heavy passage, the embedding backend read continuous, graded
similarity across almost every adjacent sentence pair while the TF-IDF
fallback read a near-flat zero except where two sentences happened to share
literal words -- the exact gap the fallback's own warning describes, now
confirmed on real text rather than only asserted in prose.

**Why the embedding sequences are never on by default anywhere.**
:mod:`textgrader.metrics.timeseries_suite` is ``cost="moderate"`` with
``requires=()``, which is what keeps it in the corpus builder's per-book
profiling pass (``textgrader/corpus.py``'s ``needs_parse``/``needs_model``
gate looks only at ``cost`` and ``requires``, not at what a sequence *this*
module builds might load). That gate cannot see that
``sentence_similarity_prev``/``sentence_distance_centroid`` load a
sentence-embedding model, so nothing in ``timeseries_suite`` may select
either one by default: doing so would make an ostensibly cheap, always-on
metric silently download and run a neural model over every reference book a
corpus is built from. Both sequences are reachable only when a caller's
``sequences`` setting names them explicitly (``timeseries_suite``'s own
``DEFAULT_SEQUENCES`` never does), and no code path in this module imports
:mod:`textgrader.metrics.semantic_adjacent` until one of them is actually
requested.

**Sentiment, emotion and topic sequences.**  An earlier pass left these three
out of the "Source sequences" list on reasoning that did not hold up once
actually checked, per this project's rule that "not installed" is never a
reason to defer something without proof: it said sentiment/emotion scoring
needed a lexicon this environment could not download (true only of
``nltk``'s VADER data, which this module never uses) and that topic
transitions needed a *categorical* channel outside this module's numeric
contract (true of the topic label itself, but not of the numeric features a
suite can compute from it). All three are now real:

* :data:`sentence_sentiment_compound` scores each sentence with
  ``vaderSentiment``'s rule-based compound polarity in ``[-1, 1]``. Unlike
  ``nltk``'s VADER integration, ``vaderSentiment`` ships its own lexicon
  inside the wheel and needs no download or network call at all -- the
  actual blocker named by the earlier pass simply does not apply to this
  package.
* :data:`sentence_emotion_valence` scores each sentence with the NRC
  emotion lexicon (``nrclex``) as a positive-minus-negative affect balance
  in ``[-1, 1]`` (0.0 when no sentence word is in the lexicon). ``nrclex``
  likewise bundles its lexicon as a JSON file read via
  ``importlib.resources``, so -- like ``vaderSentiment`` -- no download or
  network call happens at runtime, even though the package declares
  ``nltk``/``textblob`` as install-time dependencies for its optional
  ``load_raw_text()`` convenience method; this module never calls that
  method, tokenizing sentences itself and calling ``load_token_list()``
  instead, so neither of those two packages' own data needs ever come into
  play.
* :data:`window_topic_id` fits one ``scikit-learn`` NMF (default) or LDA
  topic model, seeded (``random_state``) for determinism, over every
  document window's bag-of-words and reports each window's *dominant* topic
  as a float index. That index is honestly a nominal label, not an ordered
  quantity -- averaging or autocorrelating it directly is not generally
  meaningful, which is exactly the "categorical channel" the earlier pass's
  reasoning pointed at -- but three purpose-built numeric feature groups in
  :mod:`textgrader.metrics.timeseries_suite` (``topic_transition_rate``,
  ``topic_transition_entropy``, ``topic_dwell_time``) read this sequence and
  report only well-defined numeric quantities derived from it (how often the
  topic changes, how much information the previous topic gives about the
  next one, how long a topic run lasts), never the raw nominal label as if
  it meant something on its own. Nothing stops a caller from also pointing
  this suite's generic ``acf``/``trend``/etc. feature groups at
  ``window_topic_id`` -- that follows directly from this suite's own
  documented design of running the same battery over every sequence in the
  registry -- but the label's own docstring below says plainly that it is
  nominal, so a reader is not misled into treating a difference between
  topic 2 and topic 0 as a distance.

All three stay out of every default the same way the two embedding sequences
do (see "Why the embedding sequences are never on by default anywhere" above)
and none is reachable unless a caller's ``sequences`` setting names it
explicitly -- ``window_topic_id`` in particular never fits a topic model
during corpus profiling for exactly that reason (see
``timeseries_suite``'s own module docstring for the corpus-profiling gate
this relies on).

Recurrence-quantification sequences (``PyRQA``) are not here because
recurrence quantification is its own task (Task 21), not a gap in this
task's own "Source sequences" list -- left untouched rather than folded in
here.
"""

from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from . import text as textlib
from .document import DocumentAnalysis
from .optional import require

# ------------------------------------------------------------------ the type

@dataclass(frozen=True)
class Sequence:
    """One ordered numeric channel.  ``values[i]`` happened before ``values[i+1]``."""

    name: str
    values: tuple[float, ...]
    unit: str
    #: What one position in the series is: ``sentence``, ``paragraph`` or ``window``.
    sample_unit: str
    description: str
    #: The settings that produced these values (window size, model name, ...).
    #: Corpus profiles must record this alongside every derived scalar.
    settings: Mapping[str, Any] = field(default_factory=dict)
    #: Set when the sequence is empty, or was built on a degraded backend
    #: (e.g. the TF-IDF fallback instead of a real embedding) despite having
    #: values.  Callers must check ``values`` for emptiness independently:
    #: a non-empty sequence can still carry a warning about its backend.
    warning: str | None = None

    @property
    def length(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    sample_unit: str
    unit: str
    #: Which metric family a feature built on this sequence should belong to.
    #: Sentence/paragraph-level channels are a fact about *rhythm*; window
    #: channels are windows of a whole book and belong to *drift*, the same
    #: split the task's own IDs (``rhythm.timeseries_`` / ``drift.timeseries_``)
    #: are named after.
    family: str
    requires: tuple[str, ...]
    description: str
    build: Callable[[DocumentAnalysis, Mapping[str, Any]], "Sequence"]


def _empty(spec_name: str, unit: str, sample_unit: str, reason: str,
          settings: Mapping[str, Any]) -> Sequence:
    return Sequence(spec_name, (), unit, sample_unit, reason, dict(settings), warning=reason)


def _rate(count: float, total: float, scale: float = 100.0) -> float:
    return scale * count / total if total else 0.0


# ------------------------------------------------------------- small helpers

#: A closed, deliberately small personal-pronoun list, matching the ones
#: ``pov_pronouns``/``pov_entity_ratio`` use for the same purpose. Duplicated
#: rather than imported: this is a foundation module several metrics build on,
#: and it must not depend on any one of them.
PRONOUNS = frozenset({
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
})

# A small closed stoplist for "content word" rarity, independent of (and
# smaller than) semantic_adjacent's -- this only needs to exclude the handful
# of words frequent enough to dominate a per-sentence median otherwise.
_RARITY_STOPWORDS = frozenset("""
a an the and or but if of at by for with about against between into through
during before after above below to from up down in out on off over under is
am are was were be been being have has had having do does did doing would
could might must shall this that these those it he she they them his her
their as which who what
""".split())

PUNCT_RE = re.compile(r"[.,;:!?\"'()\[\]{}—–…]")


def _categorical_entropy(labels: list[str]) -> float:
    """Entropy in bits of a small categorical sample, given as raw labels."""

    total = len(labels)
    if not total:
        return 0.0
    counts = Counter(labels)
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


# --------------------------------------------------------------- non-parse builders

def _build_sentence_words(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    values = [float(n) for n in analysis.sentence_lengths]
    return Sequence("sentence_words", tuple(values), "words", "sentence",
                    "Word count of each sentence, in reading order.", settings)


def _build_sentence_chars(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    values = [float(len(sentence)) for sentence in analysis.sentences]
    return Sequence("sentence_chars", tuple(values), "characters", "sentence",
                    "Character count of each sentence, in reading order.", settings)


def _build_paragraph_words(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    values = [float(n) for n in analysis.paragraph_lengths]
    return Sequence("paragraph_words", tuple(values), "words", "paragraph",
                    "Word count of each paragraph, in reading order.", settings)


def _build_paragraph_sentences(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    values = [float(n) for n in analysis.paragraph_sentence_counts]
    return Sequence("paragraph_sentences", tuple(values), "sentences", "paragraph",
                    "Sentence count of each paragraph, in reading order.", settings)


def _build_sentence_punctuation(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    values = [float(len(PUNCT_RE.findall(sentence))) for sentence in analysis.sentences]
    return Sequence("sentence_punctuation", tuple(values), "marks", "sentence",
                    "Punctuation-mark count of each sentence, in reading order.", settings)


def _build_sentence_char_entropy(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    values: list[float] = []
    for sentence in analysis.sentences:
        letters = [ch for ch in sentence.lower() if ch.isalpha()]
        if not letters:
            continue
        counts = Counter(letters)
        total = len(letters)
        values.append(-sum((n / total) * math.log2(n / total) for n in counts.values() if n))
    return Sequence("sentence_char_entropy", tuple(values), "bits", "sentence",
                    "Shannon entropy of the letter distribution inside each sentence.", settings)


def _build_window_dialogue_fraction(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    window_words = max(200, int(settings.get("window_words", 2000)))
    windows = analysis.windows(window_words)
    values = [float(view.dialogue_word_share or 0.0) for view in windows]
    return Sequence("window_dialogue_fraction", tuple(values), "%", "window",
                    "Share of each window's words that fall inside dialogue.",
                    {"window_words": window_words})


def _build_window_pronoun_rate(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    window_words = max(200, int(settings.get("window_words", 2000)))
    windows = analysis.windows(window_words)
    values = []
    for view in windows:
        tokens = view.tokens
        count = sum(1 for token in tokens if token in PRONOUNS)
        values.append(_rate(count, len(tokens), 1000.0))
    return Sequence("window_pronoun_rate", tuple(values), "per 1,000 words", "window",
                    "Personal-pronoun rate of each window.", {"window_words": window_words})


# ------------------------------------------------------------------ parse builders

def _build_sentence_parse_depth(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    if analysis.nlp_unavailable:
        return _empty("sentence_parse_depth", "levels", "sentence", analysis.nlp_unavailable, settings)
    values: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            tokens = list(sent)
            if tokens:
                values.append(float(max(sum(1 for _ in token.ancestors) for token in tokens)))
    return Sequence("sentence_parse_depth", tuple(values), "levels", "sentence",
                    "Maximum dependency-tree depth of each sentence.", settings)


def _build_sentence_dependency_distance(analysis: DocumentAnalysis,
                                         settings: Mapping[str, Any]) -> Sequence:
    if analysis.nlp_unavailable:
        return _empty("sentence_dependency_distance", "tokens", "sentence",
                      analysis.nlp_unavailable, settings)
    values: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            distances = [abs(token.i - token.head.i) for token in sent if token.dep_ != "ROOT"]
            if distances:
                values.append(statistics.fmean(distances))
    return Sequence("sentence_dependency_distance", tuple(values), "tokens", "sentence",
                    "Mean dependency distance of each sentence.", settings)


# Dependents of a finite verb that are not themselves a separate clause head.
# Matches syntax_finite_clauses' definition; kept as its own small copy rather
# than an import so this module never depends on a metric module for its
# only spaCy-derived count.
_CLAUSE_HELPER_DEPS = {"aux", "auxpass", "cop"}


def _is_finite_clause_head(token: Any) -> bool:
    if token.pos_ not in ("VERB", "AUX"):
        return False
    if token.dep_ in _CLAUSE_HELPER_DEPS:
        return False
    if token.morph.get("Tense"):
        return True
    return "Fin" in token.morph.get("VerbForm")


def _build_sentence_clause_count(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    if analysis.nlp_unavailable:
        return _empty("sentence_clause_count", "clauses", "sentence", analysis.nlp_unavailable, settings)
    values: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            values.append(float(sum(1 for token in sent if _is_finite_clause_head(token))))
    return Sequence("sentence_clause_count", tuple(values), "clauses", "sentence",
                    "Finite-clause count of each sentence.", settings)


def _build_sentence_pos_entropy(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    if analysis.nlp_unavailable:
        return _empty("sentence_pos_entropy", "bits", "sentence", analysis.nlp_unavailable, settings)
    values: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            tags = [token.pos_ for token in sent if token.is_alpha]
            if tags:
                values.append(_categorical_entropy(tags))
    return Sequence("sentence_pos_entropy", tuple(values), "bits", "sentence",
                    "Entropy of the part-of-speech mix inside each sentence.", settings)


def _build_sentence_entity_count(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    if analysis.nlp_unavailable:
        return _empty("sentence_entity_count", "entities", "sentence", analysis.nlp_unavailable, settings)
    if "ner" in analysis.nlp_settings.disable:
        return _empty("sentence_entity_count", "entities", "sentence",
                      "the shared spaCy pipeline has 'ner' in nlp.disable; this sequence needs "
                      "named-entity recognition, so nlp.disable must not include 'ner'", settings)
    pipe_names = list(getattr(analysis.nlp, "pipe_names", []) or [])
    if "ner" not in pipe_names:
        return _empty("sentence_entity_count", "entities", "sentence",
                      "the loaded spaCy pipeline has no 'ner' component despite nlp.disable "
                      "allowing it", settings)
    values: list[float] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            values.append(float(len(sent.ents)))
    return Sequence("sentence_entity_count", tuple(values), "entities", "sentence",
                    "Named-entity count of each sentence.", settings)


# --------------------------------------------------------------- rarity builder

def _build_sentence_content_rarity(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    module, reason = require("wordfreq")
    if module is None:
        return _empty("sentence_content_rarity", "zipf", "sentence", reason, settings)
    language = settings.get("language", "en")
    values: list[float] = []
    for sentence in analysis.sentences:
        words = [word.lower() for word in textlib.words(sentence)
                if len(word) > 2 and word.lower() not in _RARITY_STOPWORDS]
        if not words:
            continue
        scores = [module.zipf_frequency(word, language) for word in words]
        values.append(statistics.median(scores))
    return Sequence("sentence_content_rarity", tuple(values), "zipf", "sentence",
                    "Median content-word Zipf rarity of each sentence (lower is rarer).",
                    {"language": language})


# ------------------------------------------------------------- sentiment / emotion builders

def _build_sentence_sentiment_compound(analysis: DocumentAnalysis,
                                        settings: Mapping[str, Any]) -> Sequence:
    module, reason = require("vaderSentiment")
    if module is None:
        return _empty("sentence_sentiment_compound", "compound score", "sentence", reason, settings)
    analyzer = module.SentimentIntensityAnalyzer()
    values = [float(analyzer.polarity_scores(sentence)["compound"])
             for sentence in analysis.sentences]
    return Sequence("sentence_sentiment_compound", tuple(values), "compound score", "sentence",
                    "VADER rule-based compound sentiment polarity of each sentence, in "
                    "[-1, 1] (negative to positive; 0 is neutral). The lexicon ships inside "
                    "the vaderSentiment package itself, so this needs no download.", settings)


def _build_sentence_emotion_valence(analysis: DocumentAnalysis,
                                     settings: Mapping[str, Any]) -> Sequence:
    module, reason = require("nrclex")
    if module is None:
        return _empty("sentence_emotion_valence", "affect balance", "sentence", reason, settings)
    values: list[float] = []
    for sentence in analysis.sentences:
        tokens = [word.lower() for word in textlib.words(sentence)]
        lexicon = module.NRCLex()
        lexicon.load_token_list(tokens)
        frequencies = lexicon.affect_frequencies
        values.append(float(frequencies.get("positive", 0.0) - frequencies.get("negative", 0.0)))
    return Sequence("sentence_emotion_valence", tuple(values), "affect balance", "sentence",
                    "NRC emotion-lexicon positive-minus-negative affect share of each sentence, "
                    "in [-1, 1] (0.0 when no word in the sentence is in the lexicon). The "
                    "lexicon ships inside the nrclex package itself, so this needs no download.",
                    settings)


# ------------------------------------------------------------- topic-transition builder

#: Bounds on the one-time NMF/LDA fit this sequence needs: a small, capped
#: topic count and vocabulary keep the fit fast and its cost independent of
#: how long the book is (only the number of windows grows with length, and
#: that is already the same window count every other ``window_*`` sequence
#: pays for). Overridable per the settings this function receives, but always
#: re-clamped into this range so a caller cannot accidentally ask for a
#: combinatorial topic-model fit.
_TOPIC_MIN_TOPICS, _TOPIC_MAX_TOPICS = 2, 20
_TOPIC_MIN_VOCAB, _TOPIC_MAX_VOCAB = 50, 5000


def _build_window_topic_id(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    window_words = max(200, int(settings.get("window_words", 2000)))
    n_topics = min(_TOPIC_MAX_TOPICS, max(_TOPIC_MIN_TOPICS, int(settings.get("n_topics", 4))))
    model_name = str(settings.get("topic_model", "nmf")).lower()
    random_state = int(settings.get("random_state", 42))
    max_features = min(_TOPIC_MAX_VOCAB, max(_TOPIC_MIN_VOCAB, int(settings.get("max_features", 2000))))
    recorded_settings = {"window_words": window_words, "n_topics": n_topics,
                         "topic_model": model_name, "random_state": random_state,
                         "max_features": max_features}

    module, reason = require("sklearn")
    if module is None:
        return _empty("window_topic_id", "topic index", "window", reason, recorded_settings)
    if model_name not in ("nmf", "lda"):
        return _empty("window_topic_id", "topic index", "window",
                      f"unknown topic_model {model_name!r}; use 'nmf' or 'lda'", recorded_settings)

    windows = analysis.windows(window_words)
    # A topic model needs enough documents (windows) to have something to
    # tell topics apart with; below twice the topic count it is not fitting
    # topics so much as memorizing windows.
    min_windows = n_topics * 2
    if len(windows) < min_windows:
        return _empty("window_topic_id", "topic index", "window",
                      f"needs at least {min_windows} {window_words}-word windows to fit "
                      f"{n_topics} topics, found {len(windows)}", recorded_settings)

    texts = [" ".join(view.tokens) for view in windows]
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
        counts = vectorizer.fit_transform(texts)
        if counts.shape[1] == 0:
            return _empty("window_topic_id", "topic index", "window",
                          "no content words survived stop-word removal across the windows; "
                          "too little distinct vocabulary to fit a topic model", recorded_settings)
        if model_name == "lda":
            from sklearn.decomposition import LatentDirichletAllocation
            model = LatentDirichletAllocation(n_components=n_topics, random_state=random_state,
                                              max_iter=50)
        else:
            from sklearn.decomposition import NMF
            model = NMF(n_components=n_topics, random_state=random_state, max_iter=300,
                       init="nndsvda")
        doc_topic = model.fit_transform(counts)
        labels = tuple(float(int(row.argmax())) for row in doc_topic)
    except Exception as exc:  # pragma: no cover - library/runtime guard
        return _empty("window_topic_id", "topic index", "window",
                      f"{model_name} topic-model fit failed ({type(exc).__name__}: {exc})",
                      recorded_settings)
    return Sequence("window_topic_id", labels, "topic index", "window",
                    f"Dominant {model_name.upper()} topic index of each {window_words}-word "
                    f"window, fit once over all of this document's windows with "
                    f"n_topics={n_topics}, random_state={random_state}. A nominal label, not "
                    f"an ordered quantity: topic 3 is not 'more' than topic 1.",
                    recorded_settings)


# ------------------------------------------------------------- semantic builders

def _sentence_vectors(analysis: DocumentAnalysis, model_name: str):
    # Imported lazily so importing this module never eagerly imports the
    # semantic_repetition family, and so a circular import can never form
    # (semantic_adjacent does not, and must not, import this module).
    from .metrics import semantic_adjacent
    return semantic_adjacent.get_sentence_vectors(analysis, model_name)


def _build_sentence_similarity_prev(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    model_name = settings.get("model", "all-MiniLM-L6-v2")
    from .metrics import semantic_adjacent
    sentences = analysis.sentences
    if len(sentences) < 2:
        return _empty("sentence_similarity_prev", "cosine", "sentence",
                      "fewer than two sentences", {"model": model_name})
    backend, vectors, note = _sentence_vectors(analysis, model_name)
    values = [semantic_adjacent.similarity_at(backend, vectors, i, i + 1)
             for i in range(len(sentences) - 1)]
    return Sequence("sentence_similarity_prev", tuple(float(v) for v in values), "cosine", "sentence",
                    "Cosine similarity of each sentence to the one right before it.",
                    {"model": model_name, "backend": backend}, warning=note)


def _centroid_distance(backend: str, vectors: Any) -> list[float]:
    if backend == "embedding":
        numpy, _ = require("numpy")
        centroid = vectors.mean(axis=0)
        norm = float(numpy.linalg.norm(centroid)) if numpy is not None else 0.0
        if norm > 0:
            centroid = centroid / norm
        return [float(1.0 - vector.dot(centroid)) for vector in vectors]
    total: dict[str, float] = {}
    for vector in vectors:
        for word, weight in vector.items():
            total[word] = total.get(word, 0.0) + weight
    n = len(vectors) or 1
    mean_vector = {word: weight / n for word, weight in total.items()}
    norm = math.sqrt(sum(weight * weight for weight in mean_vector.values()))
    centroid = {word: weight / norm for word, weight in mean_vector.items()} if norm else {}
    out = []
    for vector in vectors:
        keys = vector if len(vector) < len(centroid) else centroid
        other = centroid if keys is vector else vector
        dot = sum(weight * other.get(word, 0.0) for word, weight in keys.items())
        out.append(1.0 - dot)
    return out


def _build_sentence_distance_centroid(analysis: DocumentAnalysis, settings: Mapping[str, Any]) -> Sequence:
    model_name = settings.get("model", "all-MiniLM-L6-v2")
    sentences = analysis.sentences
    if len(sentences) < 3:
        return _empty("sentence_distance_centroid", "cosine distance", "sentence",
                      "fewer than three sentences", {"model": model_name})
    backend, vectors, note = _sentence_vectors(analysis, model_name)
    values = _centroid_distance(backend, vectors)
    return Sequence("sentence_distance_centroid", tuple(float(v) for v in values), "cosine distance",
                    "sentence", "Cosine distance of each sentence from the document's centroid "
                                "(mean) sentence vector.", {"model": model_name, "backend": backend},
                    warning=note)


# ---------------------------------------------------------------------- registry

def _spec(name: str, sample_unit: str, unit: str, family: str, requires: tuple[str, ...],
         description: str, build) -> tuple[str, SequenceSpec]:
    item = SequenceSpec(name, sample_unit, unit, family, requires, description, build)
    return name, item


SEQUENCES: dict[str, SequenceSpec] = dict([
    _spec("sentence_words", "sentence", "words", "sentence_rhythm", (),
          "Word count per sentence.", _build_sentence_words),
    _spec("sentence_chars", "sentence", "characters", "sentence_rhythm", (),
          "Character count per sentence.", _build_sentence_chars),
    _spec("paragraph_words", "paragraph", "words", "sentence_rhythm", (),
          "Word count per paragraph.", _build_paragraph_words),
    _spec("paragraph_sentences", "paragraph", "sentences", "sentence_rhythm", (),
          "Sentence count per paragraph.", _build_paragraph_sentences),
    _spec("sentence_punctuation", "sentence", "marks", "sentence_rhythm", (),
          "Punctuation-mark count per sentence.", _build_sentence_punctuation),
    _spec("sentence_char_entropy", "sentence", "bits", "sentence_rhythm", (),
          "Character-entropy per sentence.", _build_sentence_char_entropy),
    _spec("window_dialogue_fraction", "window", "%", "book_drift", (),
          "Dialogue word-share per fixed-size window.", _build_window_dialogue_fraction),
    _spec("window_pronoun_rate", "window", "per 1,000 words", "book_drift", (),
          "Personal-pronoun rate per fixed-size window.", _build_window_pronoun_rate),
    _spec("sentence_parse_depth", "sentence", "levels", "sentence_rhythm", ("spacy",),
          "Maximum parse-tree depth per sentence.", _build_sentence_parse_depth),
    _spec("sentence_dependency_distance", "sentence", "tokens", "sentence_rhythm", ("spacy",),
          "Mean dependency distance per sentence.", _build_sentence_dependency_distance),
    _spec("sentence_clause_count", "sentence", "clauses", "sentence_rhythm", ("spacy",),
          "Finite-clause count per sentence.", _build_sentence_clause_count),
    _spec("sentence_pos_entropy", "sentence", "bits", "sentence_rhythm", ("spacy",),
          "Part-of-speech-mix entropy per sentence.", _build_sentence_pos_entropy),
    _spec("sentence_entity_count", "sentence", "entities", "sentence_rhythm", ("spacy",),
          "Named-entity count per sentence (needs nlp.disable without 'ner').",
          _build_sentence_entity_count),
    _spec("sentence_content_rarity", "sentence", "zipf", "sentence_rhythm", ("wordfreq",),
          "Median content-word Zipf rarity per sentence.", _build_sentence_content_rarity),
    _spec("sentence_similarity_prev", "sentence", "cosine", "sentence_rhythm",
          ("sentence_transformers",), "Semantic similarity to the previous sentence.",
          _build_sentence_similarity_prev),
    _spec("sentence_distance_centroid", "sentence", "cosine distance", "sentence_rhythm",
          ("sentence_transformers",), "Semantic distance from the document's centroid sentence.",
          _build_sentence_distance_centroid),
    _spec("sentence_sentiment_compound", "sentence", "compound score", "sentence_rhythm",
          ("vaderSentiment",), "VADER compound sentiment polarity per sentence.",
          _build_sentence_sentiment_compound),
    _spec("sentence_emotion_valence", "sentence", "affect balance", "sentence_rhythm",
          ("nrclex",), "NRC emotion-lexicon positive-minus-negative affect balance per sentence.",
          _build_sentence_emotion_valence),
    _spec("window_topic_id", "window", "topic index", "book_drift", ("sklearn",),
          "Dominant NMF/LDA topic index per fixed-size window (nominal label).",
          _build_window_topic_id),
])


def get_sequence(analysis: DocumentAnalysis, name: str,
                 settings: Mapping[str, Any] | None = None) -> Sequence:
    """Build (or fetch a cached) named sequence for ``analysis``.

    Memoized on ``analysis`` through :meth:`DocumentAnalysis.memo`, keyed on
    the sequence's name and settings, so a second feature asking for the same
    sequence under the same settings never re-derives it -- and a request
    under *different* settings (a different window size, a different
    embedding model) is never served the wrong cached answer.
    """

    spec = SEQUENCES.get(name)
    if spec is None:
        raise KeyError(f"unknown sequence {name!r}; known sequences are {sorted(SEQUENCES)}")
    settings = dict(settings or {})
    key = "sequences:" + name + ":" + ",".join(f"{k}={settings[k]}" for k in sorted(settings))
    return analysis.memo(key, lambda: spec.build(analysis, settings))


def list_sequences() -> list[str]:
    return sorted(SEQUENCES)


__all__ = ["Sequence", "SequenceSpec", "SEQUENCES", "get_sequence", "list_sequences", "PRONOUNS"]
