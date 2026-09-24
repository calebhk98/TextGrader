"""Conversational dynamics and speaker interaction, on top of the existing
dialogue channel: coordination/entrainment, turn-taking, question-response
behaviour, speaker separability and the conversation as a graph.

**A sensor, not an opinion.** Characters can converge, stay distinct, dominate
a scene, or interrupt each other for good reasons the text itself supplies.
Every finding here is ``Polarity.NEUTRAL``; nothing in this module decides
that convergence or dominance is good or bad prose.

**Reuse, never re-parse.** This module never touches a quotation mark itself.
Every turn boundary and every speaker label comes from code that already
exists:

* ``dialogue_attribution.classify_turns`` -- the paragraph-bounded,
  neighbour-bounded speech-tag/action-beat/untagged classification, with the
  same capitalized-name-next-to-a-speech-verb heuristic used everywhere else
  in this codebase. :func:`_ordered_events` calls it directly and keeps its
  ordering and offsets; it never re-derives which characters sit next to a
  quotation.
* ``dialogue_speaker_style.identify_speakers``'s transcript path
  (``Name: message`` chat lines) is mirrored, not reimplemented: the same
  ``textlib.transcript_lines`` call, the same ``TRANSCRIPT_MIN_LINES``/
  two-distinct-speaker gate, so a document that qualifies for the transcript
  path there qualifies for it here, with the actual match objects kept for
  their offsets (which that module's dict-of-lists return discards).
* ``function_words.FUNCTION`` -- the fixed 51-word function-word list -- backs
  the coordination categories below, imported rather than copied, so a
  category can never quietly drift from the list ``character_voice`` and
  ``dialogue_speaker_function_words`` already compare speakers on.
* ``dialogue_attribution.SPEECH_VERBS`` filters speech verbs ("said",
  "replied", ...) out of the content-word vocabulary lexical entrainment and
  response relevance draw on, so two turns are never scored as "on topic"
  because they both happen to use "said".
* ``sequences``'s two sentiment/emotion scorers (``vaderSentiment``'s
  rule-based compound polarity, the NRC lexicon's positive-minus-negative
  affect balance) are the only sentiment engines this module uses --
  :func:`_vader_scores`/:func:`_nrc_scores` call the *same* two packages
  through the *same* ``textgrader.optional.require`` gate ``sequences.py``
  uses, scored per spoken TURN rather than per sentence (a turn is often
  several sentences and a coupling measure needs one score per turn, not
  one per sentence-inside-a-turn), so no second sentiment engine is added
  anywhere in this module. A sibling task owns adding new sentiment engines;
  this one only reuses what already loads.
* ``semantic_adjacent.lexical_vectors``/``embed_texts`` back response
  relevance's two backends (TF-IDF lexical, on by default; real
  sentence-transformers embeddings, an explicit off-by-default upgrade),
  reusing that family's model cache and lexical fallback rather than writing
  a third TF-IDF-or-embedding split.
* ``dialogue_speaker_style.CONTRACTION_RE`` is the exact contraction pattern
  (closed suffix set, no bare ``'s``) this module's contraction-convergence
  marker reuses.

Where a measure here overlaps an existing dialogue metric, the finding's
``distribution`` says so by name (``dialogue.turn_words``,
``dialogue.speaker_function_word_distance``, ``style.character_voice_distance``)
and states what is actually new: this module never reports the same number
twice under two ids.

**ConvoKit.** The task names ConvoKit for coordination and politeness.
``pip install --dry-run convokit`` shows it pins ``numpy>=2.0.0`` while this
project's registry pins ``numpy`` at 1.26.4 (transformers/torch/scipy/spacy
are all built against it here) -- ``pip install --dry-run convokit`` reports::

    Collecting numpy>=2.0.0 (from convokit)
      Using cached numpy-2.4.6-...whl.metadata (6.6 kB)
    ...
    Would install ... numpy-2.4.6 ...

a genuine shared-package upgrade this project's rules forbid. ``--no-deps``
avoids that (``pip install --no-deps convokit`` succeeds and installs only
``convokit`` itself), but importing anything from it -- even just
``Coordination`` or ``PolitenessStrategies`` -- immediately needs
``convokit.model.Corpus``, which imports ``pymongo`` for its MongoDB-backed
storage backend (``ModuleNotFoundError: No module named 'bson'``); installing
``pymongo`` too (no numpy conflict there) then fails one import deeper,
inside ``pymongo``'s own TLS stack::

    File ".../pymongo/pyopenssl_context.py", line 31, in <module>
        import cryptography.x509 as x509
    ...
    pyo3_runtime.PanicException: Python API call failed

a broken ``cryptography``/``cffi`` binding in this container, unrelated to
anything this task is allowed to touch. Chasing a MongoDB-shaped corpus
object graph just to reach two feature-extraction classes is also the wrong
shape for this codebase's own document model, so both the coordination score
and the politeness-strategy detector below are the published formulas,
implemented directly and labelled as such (:func:`_coordination`, matching
Danescu-Niculescu-Mizil et al. 2012's "Echoes of power" linguistic-style
coordination measure; :data:`POLITENESS_PATTERNS`, a lexical/regex
approximation of a subset of Danescu-Niculescu-Mizil & Lee 2013's politeness
strategies), never routed through ConvoKit. ``convokit``/``pymongo``/
``dnspython`` were uninstalled again after this check; nothing in
``requirements.txt`` names them.

**Attribution coverage gates every genuinely per-speaker claim** (rule: low
coverage must suppress speaker-specific findings, mirroring
``dialogue_speaker_style``'s own gate). :func:`_attribution_state` computes
the same two numbers that module does -- named-speaker coverage and how many
speakers clear ``min_turns_per_speaker`` -- and every metric that needs to
know WHO said which of two adjacent turns (dominance, alternation,
coordination, entrainment, style separability, sentiment differentiation,
the interaction graph) reports ``insufficient_data``/``unavailable`` rather
than a confident number below that gate. Metrics that only need turn
ADJACENCY, not speaker identity (question-response rate, backchannel rate,
response relevance, politeness rates), are not gated on it, because they
make no claim about who did what.

**Aggregation, stated once.** Every "coordination"/"convergence"/
"accommodation" measure below (turn-length, sentence-length, function-word,
contraction, question-mark, exclamation-mark) is computed by the *same*
generic primitive, :func:`_coordination`: the Danescu-Niculescu-Mizil
formula, per responding speaker, then averaged across qualifying speakers
(a speaker needs ``min_coordination_pairs`` "primed" replies -- replies whose
immediately preceding, different-speaker turn also carried the marker --
before their own coordination number counts). The headline is the mean of
those per-speaker numbers; the full per-speaker breakdown (baseline rate,
conditional rate, sample sizes) is the ``evidence``, and the pooled
distribution of those per-speaker numbers is the ``distribution``, per the
project's "say how every headline aggregates" rule. "Entrainment" measures
(shared/rare content-word reuse, response relevance) use a different,
equally explicit primitive, :func:`_shuffle_contrast`: the real adjacent-pair
rate minus the rate over the same replies paired with a *shuffled* (seeded,
deterministic) other turn instead of their real predecessor, which controls
for how much of any apparent "entrainment" is just two turns discussing the
same standing topic.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from ..document import DocumentAnalysis
from .. import text as textlib
from ..stats import run_lengths, summarize
from .common import MODERATE, cosine_distance, finding, option, rate
from .dialogue_attribution import SPEECH_VERBS, classify_turns
from .dialogue_speaker_style import CONTRACTION_RE, TRANSCRIPT_MIN_LINES
from .function_words import FUNCTION
from .semantic_adjacent import embed_texts, lexical_vectors
from ..optional import on_reset, require

FAMILY = "dialogue"
COST = MODERATE
# Always attempted with a graceful degrade when default features want them;
# never spaCy or sentence_transformers here (see the ``pos_convergence``/
# ``response_relevance_embedding`` features below and this codebase's
# gating rule: needs_parse/needs_model would pull this whole suite out of
# every corpus profile, and most of what is here needs neither).
REQUIRES: tuple[str, ...] = ("networkx", "vaderSentiment", "nrclex")
MIN_SAMPLE = 20
UNIT_SENSITIVE = False

EVIDENCE_LIMIT = 25
PREFIX = "dialogue.conversation_"

FUNCTION_SET = frozenset(FUNCTION)
FUNCTION_CATEGORIES: dict[str, frozenset[str]] = {
    "articles": frozenset({"a", "an", "the"}),
    "conjunctions": frozenset({"and", "but", "or", "nor", "if", "as", "so", "than"}),
    "prepositions": frozenset({"at", "by", "for", "from", "in", "of", "on", "to", "with"}),
    "pronouns": frozenset({"he", "her", "him", "his", "i", "it", "its", "me", "my", "our",
                           "she", "their", "them", "they", "us", "we", "you", "your",
                           "this", "that"}),
    "auxiliary": frozenset({"be", "been", "had", "has", "have", "is", "was", "were", "will"}),
}
assert all(word in FUNCTION_SET for words in FUNCTION_CATEGORIES.values() for word in words)


# ---------------------------------------------------------------- turn events

def _ordered_events(analysis: DocumentAnalysis) -> tuple[list[dict[str, Any]], str]:
    """Ordered ``(events, method)``: every spoken turn, speaker or ``None``.

    Mirrors ``dialogue_speaker_style.identify_speakers`` exactly (same
    transcript gate, same ``classify_turns`` fallback) but keeps every turn
    -- not just the named ones -- in document order with its offsets, which
    every adjacency-based measure in this module needs and that function's
    ``dict[name, list[str]]`` return cannot supply.
    """

    if not analysis.processing.strip_transcript:
        matches = textlib.transcript_lines(analysis.raw)
        if len(matches) >= TRANSCRIPT_MIN_LINES:
            events = []
            for match in matches:
                info = match.groupdict()
                name = (info.get("username") or "").strip()
                message = (info.get("message") or "").strip()
                if message:
                    events.append({"start": match.start(), "end": match.end(),
                                  "text": message, "speaker": name or None, "label": "named"})
            if len({event["speaker"] for event in events if event["speaker"]}) >= 2:
                return events, "transcript"

    events = [{"start": item["start"], "end": item["end"], "text": item["text"],
              "speaker": item["speaker"], "label": item["label"]}
             for item in classify_turns(analysis)]
    return events, "tag"


def _named_groups(events: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event["speaker"]:
            groups.setdefault(event["speaker"], []).append(event)
    return groups


def _attribution_state(events: Sequence[Mapping[str, Any]], method: str, min_turns: int,
                       min_coverage: float) -> dict[str, Any]:
    """Coverage, qualifying speakers and the sufficiency gate, in one place.

    ``sufficient`` is what every per-speaker metric below checks before
    reporting a number. It requires at least two speakers with
    ``min_turns`` turns each, and -- for the ``tag`` method only, where
    attribution is a biased, incomplete heuristic rather than an explicit
    label -- named-turn coverage of at least ``min_coverage`` percent of all
    spoken turns, exactly ``dialogue_speaker_style``'s own gate.
    """

    total = len(events)
    named = sum(1 for event in events if event["speaker"])
    groups = _named_groups(events)
    qualifying = {name: turns for name, turns in groups.items() if len(turns) >= min_turns}
    coverage = 100.0 * named / total if total else None
    sufficient = len(qualifying) >= 2 and (
        method == "transcript" or (coverage is not None and coverage >= min_coverage))
    return {"total_turns": total, "named_turns": named, "coverage": coverage,
            "groups": groups, "qualifying": qualifying, "method": method,
            "sufficient": sufficient,
            "distinct_speakers": len(groups), "qualifying_speakers": len(qualifying)}


def _insufficient_reason(state: Mapping[str, Any], min_turns: int, min_coverage: float) -> str:
    if state["total_turns"] == 0:
        return "no spoken turns found"
    if state["qualifying_speakers"] < 2:
        return (f"found {state['distinct_speakers']} named speaker(s) ({state['method']} "
                f"attribution), {state['qualifying_speakers']} with at least {min_turns} "
                f"turns; need at least two to make a per-speaker claim")
    coverage = state["coverage"]
    return (f"only {coverage:.0f}% of {state['total_turns']} spoken turns could be attributed "
            f"to a speaker (speech tag with a nearby capitalized name), below the "
            f"{min_coverage:.0f}% these per-speaker measures need to be comparable")


# ------------------------------------------------------------------ numerics

def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _gini(counts: Sequence[float]) -> float | None:
    """Gini coefficient of a turn-count vector, 0 (every speaker talks
    equally often) to just under 1 (one speaker holds nearly every turn)."""

    values = sorted(float(c) for c in counts if c is not None)
    n = len(values)
    if n < 2 or sum(values) <= 0:
        return None
    cumulative = sum((index + 1) * value for index, value in enumerate(values))
    return (2 * cumulative) / (n * sum(values)) - (n + 1) / n


def _content_words(text: str) -> set[str]:
    tokens = (word.lower().replace("’", "'") for word in textlib.words(text))
    return {token for token in tokens
           if len(token) > 3 and token not in FUNCTION_SET and token not in SPEECH_VERBS}


# ------------------------------------------------------- reply-pair machinery

def _reply_pairs(events: Sequence[Mapping[str, Any]]) -> list[tuple[dict, dict]]:
    """Adjacent turns whose speaker is known to differ: the only pairs any
    directional/per-speaker measure below may treat as "B replying to A"."""

    out = []
    for i in range(1, len(events)):
        prev, nxt = events[i - 1], events[i]
        if prev["speaker"] and nxt["speaker"] and prev["speaker"] != nxt["speaker"]:
            out.append((prev, nxt))
    return out


def _adjacent_pairs(events: Sequence[Mapping[str, Any]]) -> list[tuple[dict, dict]]:
    """Every adjacent spoken-turn pair, speaker known or not -- for measures
    (question-response, backchannel, response relevance) that read the
    exchange structure itself rather than who is in it."""

    return [(events[i - 1], events[i]) for i in range(1, len(events))]


def _coordination(reply_pairs: Sequence[tuple[Mapping, Mapping]],
                  marker: Callable[[str], bool], min_primed: int) -> tuple[list[dict], int]:
    """The Danescu-Niculescu-Mizil (2012) coordination score, per responder.

    For each speaker B who replies to a different speaker at least
    ``min_primed`` times where THAT preceding turn carried ``marker``:
    ``coordination = P(B's turn carries marker | A's preceding turn did) -
    P(B's turn carries marker | B is replying to someone), i.e. B's own
    reply-position baseline)``. Returns the per-speaker rows and the total
    reply-pair count they were drawn from.
    """

    by_responder: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for prev, nxt in reply_pairs:
        by_responder[nxt["speaker"]].append((marker(prev["text"]), marker(nxt["text"])))
    rows = []
    for speaker, pairs in sorted(by_responder.items()):
        total = len(pairs)
        baseline = sum(1 for _, hit in pairs if hit) / total if total else None
        primed = [hit for primer, hit in pairs if primer]
        if baseline is None or len(primed) < min_primed:
            continue
        conditional = sum(primed) / len(primed)
        rows.append({"speaker": speaker, "coordination": 100 * (conditional - baseline),
                    "baseline_rate": 100 * baseline, "conditional_rate": 100 * conditional,
                    "n_primed": len(primed), "n_total": total})
    return rows, len(reply_pairs)


def _shuffle_contrast(pairs: Sequence[tuple[Any, Any]],
                      score: Callable[[Any, Any], float], seed: int) -> dict[str, Any] | None:
    """Real adjacent-pair score minus the same score against a shuffled
    partner: how much of an apparent link is adjacency itself, versus two
    turns that would look alike wherever they sat in the book.

    The shuffle is a seeded derangement-or-best-effort of the "prior turn"
    side only (`items[i]` -- see call sites): every response keeps its own
    text; only which OTHER turn it is compared against changes, so the
    control condition asks exactly one question -- "how similar would this
    reply be to a random other turn instead of the one right before it".
    """

    n = len(pairs)
    if n < 2:
        return None
    priors = [item[0] for item in pairs]
    replies = [item[1] for item in pairs]
    real = [score(priors[i], replies[i]) for i in range(n)]
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    for i in range(n):
        if order[i] == i:
            j = (i + 1) % n
            order[i], order[j] = order[j], order[i]
    control = [score(priors[order[i]], replies[i]) for i in range(n)]
    return {"real_rate": statistics.fmean(real), "control_rate": statistics.fmean(control),
           "contrast": statistics.fmean(real) - statistics.fmean(control), "n_pairs": n}


# -------------------------------------------------------------- text markers

def _has_contraction(text: str) -> bool:
    tokens = (word.lower().replace("’", "'") for word in textlib.words(text))
    return any(CONTRACTION_RE.search(token) for token in tokens)


def _ends_with(mark: str) -> Callable[[str], bool]:
    return lambda text: text.rstrip().rstrip("”’\"')").endswith(mark)


def _category_marker(category: frozenset[str]) -> Callable[[str], bool]:
    def marker(text: str) -> bool:
        tokens = {word.lower().replace("’", "'") for word in textlib.words(text)}
        return bool(tokens & category)
    return marker


def _length_marker(threshold: float) -> Callable[[str], bool]:
    return lambda text: len(textlib.words(text)) > threshold


def _sentence_count_marker(analysis: DocumentAnalysis, threshold: float) -> Callable[[str], bool]:
    return lambda text: analysis.derive(text, "turn").sentence_count > threshold


# ------------------------------------------------------------- sentiment/NRC

def _vader_scores(texts: Sequence[str]) -> tuple[list[float] | None, str | None]:
    module, reason = require("vaderSentiment")
    if module is None:
        return None, reason
    analyzer = module.SentimentIntensityAnalyzer()
    return [float(analyzer.polarity_scores(text)["compound"]) for text in texts], None


def _nrc_scores(texts: Sequence[str]) -> tuple[list[float] | None, str | None]:
    module, reason = require("nrclex")
    if module is None:
        return None, reason
    values = []
    for text in texts:
        lexicon = module.NRCLex()
        lexicon.load_token_list([word.lower() for word in textlib.words(text)])
        frequencies = lexicon.affect_frequencies
        values.append(float(frequencies.get("positive", 0.0) - frequencies.get("negative", 0.0)))
    return values, None


# ---------------------------------------------------------------- politeness

# A lexical/regex approximation of a subset of Danescu-Niculescu-Mizil & Lee
# (2013)'s politeness strategies -- see the module docstring for why this is
# implemented directly rather than through ConvoKit's PolitenessStrategies.
# Dependency-free (no spaCy), so it runs at this suite's default cost.
POLITENESS_PATTERNS: dict[str, re.Pattern] = {
    "gratitude": re.compile(r"\b(thanks?|thank you|thankful|appreciate[ds]?)\b", re.I),
    "please": re.compile(r"\bplease\b", re.I),
    "apology": re.compile(r"\b(sorry|apologi[sz]e[ds]?|forgive me|my fault|my apologies)\b",
                          re.I),
    "greeting": re.compile(r"^\s*(hi|hello|hey|good morning|good evening|good afternoon)\b",
                           re.I),
    "hedge": re.compile(r"\b(i think|i guess|i suppose|perhaps|maybe|sort of|kind of|"
                        r"probably|it seems|i believe)\b", re.I),
    "indirect_request": re.compile(r"\b(could you|would you|would you mind|might you)\b", re.I),
    "direct_question": re.compile(r"^\s*(why|what|where|when|who|how|do|does|did|is|are|"
                                  r"can|will)\b.*\?\s*$", re.I),
    "first_person_plural": re.compile(r"\b(we|us|our|ours)\b", re.I),
    "second_person": re.compile(r"\b(you|your|yours)\b", re.I),
}


def _politeness_hits(text: str) -> set[str]:
    return {name for name, pattern in POLITENESS_PATTERNS.items() if pattern.search(text)}


# ---------------------------------------------------------------- separability

def _turn_vector(text: str, categories: Mapping[str, frozenset[str]]) -> list[float]:
    tokens = [word.lower().replace("’", "'") for word in textlib.words(text)]
    total = len(tokens) or 1
    vector = [len(tokens) / 20.0,
              1.0 if _has_contraction(text) else 0.0,
              1.0 if text.rstrip().endswith("?") else 0.0,
              1.0 if text.rstrip().endswith("!") else 0.0]
    for words_set in categories.values():
        vector.append(sum(1 for token in tokens if token in words_set) / total)
    return vector


def _euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _separability_accuracy(labeled: Sequence[tuple[str, list[float]]]) -> dict[str, Any] | None:
    """Leave-one-out nearest-centroid accuracy of predicting a turn's speaker
    from its own style vector: how separable the cast actually sounds, not
    merely how far apart two averages are (``style.character_voice_distance``
    and ``dialogue.speaker_function_word_distance`` both report averages)."""

    n = len(labeled)
    speakers = sorted({name for name, _ in labeled})
    if n < 6 or len(speakers) < 2:
        return None
    correct = 0
    for index, (true_name, vector) in enumerate(labeled):
        sums: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(vector))
        counts: Counter[str] = Counter()
        for other_index, (name, other_vector) in enumerate(labeled):
            if other_index == index:
                continue
            counts[name] += 1
            sums[name] = [a + b for a, b in zip(sums[name], other_vector)]
        centroids = {name: [value / counts[name] for value in total]
                    for name, total in sums.items() if counts[name]}
        if len(centroids) < 2:
            continue
        predicted = min(centroids, key=lambda name: _euclid(vector, centroids[name]))
        correct += int(predicted == true_name)
    accuracy = correct / n
    chance = 1.0 / len(speakers)
    return {"accuracy": accuracy, "chance": chance, "excess": accuracy - chance,
           "n_turns": n, "n_speakers": len(speakers)}


# --------------------------------------------------------------------- graph

def _graph_stats(reply_pairs: Sequence[tuple[Mapping, Mapping]]) -> dict[str, Any] | None:
    edges: Counter[tuple[str, str]] = Counter()
    for prev, nxt in reply_pairs:
        edges[(prev["speaker"], nxt["speaker"])] += 1
    nodes = {node for edge in edges for node in edge}
    if len(nodes) < 2 or not edges:
        return None
    possible = len(nodes) * (len(nodes) - 1)
    density = len(edges) / possible if possible else None
    reciprocated = sum(1 for (a, b) in edges if (b, a) in edges)
    reciprocity = reciprocated / len(edges)
    out = {"nodes": len(nodes), "edges": len(edges), "density": density,
          "reciprocity": reciprocity,
          "top_edges": [{"from": a, "to": b, "weight": w}
                        for (a, b), w in edges.most_common(EVIDENCE_LIMIT)]}
    module, reason = require("networkx")
    if module is not None:
        graph = module.DiGraph()
        for (a, b), weight in edges.items():
            graph.add_edge(a, b, weight=weight)
        try:
            out["networkx_density"] = module.density(graph)
            out["networkx_reciprocity"] = module.reciprocity(graph)
        except Exception as exc:  # pragma: no cover - defensive only
            out["networkx_note"] = f"networkx cross-check failed ({type(exc).__name__}: {exc})"
    else:
        out["networkx_note"] = reason
    return out


# ------------------------------------------------------- dialogue-act model

# Off by default (``features.dialogue_act``). Verified for real in this
# environment: ``pipeline("text-classification",
# model="WSHAPER/distilbert-multilingual-dialogue-act-classifier")`` loaded
# and scored three sample sentences correctly ("What is the timeline?" ->
# question 0.998; "Send the report." -> directive 0.976; "The meeting went
# well." -> inform 0.995) -- see the config note for the model card
# (4-class: commissive/directive/inform/question, distilbert-base-
# multilingual-cased, Apache-2.0). Cached per process per model name, the
# same pattern ``semantic_adjacent`` uses for its sentence-transformers model.
_DIALOGUE_ACT_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_dialogue_act_cache() -> None:
    _DIALOGUE_ACT_CACHE.clear()


on_reset(_reset_dialogue_act_cache)


def _load_dialogue_act_pipeline(model_name: str) -> tuple[Any, str | None]:
    if model_name in _DIALOGUE_ACT_CACHE:
        return _DIALOGUE_ACT_CACHE[model_name]
    module, reason = require("transformers")
    if module is None:
        _DIALOGUE_ACT_CACHE[model_name] = (None, reason)
        return _DIALOGUE_ACT_CACHE[model_name]
    try:
        pipe = module.pipeline("text-classification", model=model_name, top_k=1)
        outcome: tuple[Any, str | None] = (pipe, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"transformers pipeline for {model_name!r} unavailable "
                         f"({type(exc).__name__}: {exc})")
    _DIALOGUE_ACT_CACHE[model_name] = outcome
    return outcome


def _transition_entropy(labels: Sequence[str]) -> dict[str, Any] | None:
    """Normalized Shannon entropy of the (label[t], label[t+1]) joint
    distribution -- the same construction ``timeseries_suite``'s
    ``topic_transition_entropy`` feature uses for a nominal sequence."""

    pairs = list(zip(labels, labels[1:]))
    if len(pairs) < 2:
        return None
    counts = Counter(pairs)
    total = len(pairs)
    raw_entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    distinct = len(counts)
    max_entropy = math.log2(distinct) if distinct > 1 else 1.0
    return {"normalized": raw_entropy / max_entropy if max_entropy > 0 else 0.0,
           "raw_entropy_bits": raw_entropy, "distinct_transitions": distinct,
           "distinct_labels": len(set(labels)), "n_transitions": total}


# ------------------------------------------------------------- findings: core

def _coordination_finding(metric_id: str, name: str, rows: list[dict], n_pairs: int,
                          min_primed: int, note: str | None = None) -> dict[str, Any]:
    """Every coordination/convergence finding shares this shape: the headline
    is the MEAN of the qualifying (speaker[, category]) rows' coordination
    scores, in percentage points; ``distribution`` is the shape of those same
    rows (not a different, larger sample); ``evidence`` is the rows
    themselves, largest-sample first."""

    if not rows:
        return finding(metric_id, name, None, "percentage points", family=FAMILY,
                       sample_size=n_pairs, min_sample=min_primed,
                       warning=(f"no responding speaker (or speaker/category pair) had at "
                                f"least {min_primed} replies to a preceding different-speaker "
                                f"turn that carried the marker ({n_pairs} cross-speaker reply "
                                f"pairs available in total)"))
    values = [row["coordination"] for row in rows]
    summary = summarize(values)
    evidence = ([{"note": note}] if note else []) + sorted(
        rows, key=lambda row: -row["n_total"])[:EVIDENCE_LIMIT]
    return finding(metric_id, name, statistics.fmean(values), "percentage points", family=FAMILY,
                  sample_size=len(rows), distribution=summary, min_sample=2, evidence=evidence)


def _finding_attribution(events: list[dict], state: Mapping[str, Any], min_turns: int,
                         min_coverage: float) -> list[dict]:
    coverage_id, gap_id = PREFIX + "attribution_coverage", PREFIX + "attribution_tagged_unnamed_gap"
    total = state["total_turns"]
    if total == 0:
        warning = "no spoken turns found"
        return [finding(coverage_id, "Named-speaker turn coverage", None, "percent",
                        family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE, warning=warning),
                finding(gap_id, "Speech-tagged turns with no resolvable speaker name",
                        None, "percentage points", family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE, warning=warning)]
    coverage = state["coverage"]
    evidence = [{"method": state["method"], "distinct_speakers": state["distinct_speakers"],
                "qualifying_speakers": state["qualifying_speakers"]}]
    coverage_finding = finding(
        coverage_id, "Named-speaker turn coverage (share of spoken turns a speaker name could "
        "be resolved for)", coverage, "percent", family=FAMILY, sample_size=total,
        min_sample=MIN_SAMPLE, evidence=evidence,
        warning=None if state["sufficient"] else _insufficient_reason(state, min_turns, min_coverage))
    if state["method"] == "transcript":
        gap_finding = finding(gap_id, "Speech-tagged turns with no resolvable speaker name", 0.0,
                              "percentage points", family=FAMILY, sample_size=total,
                              min_sample=MIN_SAMPLE,
                              warning="transcript-format speakers are already explicit; this "
                                      "cross-check only applies to the speech-tag method")
    else:
        tagged = sum(1 for event in events if event.get("label") == "speech_tag")
        gap = rate(tagged, total) - coverage
        gap_finding = finding(
            gap_id, "Confidence cross-check: speech-tagged turns whose tag carried no "
            "resolvable capitalized name (e.g. a pronoun tag), as percentage points of all "
            "turns -- the gap between 'this turn was tagged at all' and 'this turn's speaker "
            "is named'", gap, "percentage points", family=FAMILY, sample_size=total,
            min_sample=MIN_SAMPLE,
            evidence=[{"speech_tagged_rate_percent": rate(tagged, total),
                      "named_rate_percent": coverage}],
            warning=None if gap < 50.0 else
            "over half of tagged turns could not be resolved to a name; per-speaker findings "
            "below draw on a small, name-biased slice of the tagged turns")
    return [coverage_finding, gap_finding]


def _finding_dominance(state: Mapping[str, Any], min_turns: int, min_coverage: float) -> list[dict]:
    gini_id, top_id = PREFIX + "speaker_dominance_gini", PREFIX + "top_speaker_turn_share"
    if not state["sufficient"]:
        reason = _insufficient_reason(state, min_turns, min_coverage)
        return [finding(gini_id, "Speaker turn-count dominance (Gini coefficient)", None,
                        "gini (0=equal, 1=one speaker)", family=FAMILY,
                        sample_size=state["qualifying_speakers"], min_sample=2, warning=reason),
                finding(top_id, "Top speaker's share of named turns", None, "percent",
                        family=FAMILY, sample_size=state["qualifying_speakers"], min_sample=2,
                        warning=reason)]
    counts = {name: len(turns) for name, turns in state["qualifying"].items()}
    values = list(counts.values())
    gini = _gini(values)
    top_name, top_count = max(counts.items(), key=lambda item: item[1])
    top_share = rate(top_count, sum(values))
    rows = sorted(({"speaker": name, "turns": count} for name, count in counts.items()),
                 key=lambda row: -row["turns"])
    summary = summarize(values)
    return [
        finding(gini_id, "Speaker turn-count dominance (Gini coefficient, pooled over every "
                "qualifying speaker's turn count -- a single distributional statistic, not a "
                "median of anything)", gini, "gini (0=equal, 1=one speaker)", family=FAMILY,
                sample_size=len(values), distribution=summary, min_sample=2,
                evidence=rows[:EVIDENCE_LIMIT]),
        finding(top_id, "Top speaker's share of named turns", top_share, "percent",
                family=FAMILY, sample_size=len(values), min_sample=2,
                evidence=[{"speaker": top_name, "turns": top_count}]),
    ]


def _finding_alternation(events: list[dict], state: Mapping[str, Any], min_turns: int,
                         min_coverage: float) -> list[dict]:
    rate_id, run_id = PREFIX + "speaker_alternation_rate", PREFIX + "same_speaker_run_length"
    if not state["sufficient"]:
        reason = _insufficient_reason(state, min_turns, min_coverage)
        return [finding(rate_id, "Share of adjacent named turns with a different speaker", None,
                        "percent", family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                        warning=reason),
                finding(run_id, "Consecutive same-speaker run length among named turns", None,
                        "turns", family=FAMILY, sample_size=0, min_sample=5, warning=reason)]
    named_sequence = [event["speaker"] for event in events if event["speaker"]]
    changes = sum(1 for i in range(1, len(named_sequence))
                 if named_sequence[i] != named_sequence[i - 1])
    alternation_rate = rate(changes, len(named_sequence) - 1)
    lengths_by_speaker = run_lengths(named_sequence)
    all_lengths = sorted(length for lengths in lengths_by_speaker.values() for length in lengths)
    run_summary = summarize(all_lengths)
    return [
        finding(rate_id, "Share of adjacent NAMED turns (ignoring untagged turns in between) "
                "with a different speaker than the one before", alternation_rate, "percent",
                family=FAMILY, sample_size=len(named_sequence) - 1, min_sample=MIN_SAMPLE,
                evidence=[{"named_turns": len(named_sequence)}]),
        finding(run_id, "Consecutive same-speaker run length among named turns (median; full "
                "shape in distribution)", run_summary.get("median"), "turns", family=FAMILY,
                sample_size=len(all_lengths), distribution=run_summary, min_sample=5,
                evidence=[{"speaker": name, "runs": lengths} for name, lengths in
                         sorted(lengths_by_speaker.items(),
                               key=lambda item: -max(item[1]))[:EVIDENCE_LIMIT]]),
    ]


QA_MIN_SAMPLE = 10
BACKCHANNEL_MIN_SAMPLE = 15


def _finding_turn_taking(events: list[dict]) -> list[dict]:
    response_id = PREFIX + "question_response_rate"
    unanswered_id = PREFIX + "unanswered_question_rate"
    backchannel_id = PREFIX + "backchannel_rate"
    total_turns = len(events)
    if total_turns == 0:
        warning = "no spoken turns found"
        return [finding(response_id, "Question-response rate", None, "percent", family=FAMILY,
                        sample_size=0, min_sample=QA_MIN_SAMPLE, warning=warning),
                finding(unanswered_id, "Unanswered-question rate", None, "percent",
                        family=FAMILY, sample_size=0, min_sample=QA_MIN_SAMPLE, warning=warning),
                finding(backchannel_id, "Backchannel/short-response rate", None, "percent",
                        family=FAMILY, sample_size=0, min_sample=BACKCHANNEL_MIN_SAMPLE,
                        warning=warning)]

    def different_or_unknown(cur: Mapping, nxt: Mapping) -> bool:
        return not (cur["speaker"] and nxt["speaker"] and cur["speaker"] == nxt["speaker"])

    question_indices = [i for i, event in enumerate(events) if event["text"].rstrip().endswith("?")]
    q_total = len(question_indices)
    if q_total:
        answered = sum(1 for i in question_indices
                      if i + 1 < total_turns and different_or_unknown(events[i], events[i + 1]))
        response_rate = rate(answered, q_total)
        unanswered_rate = rate(q_total - answered, q_total)
        response_finding = finding(
            response_id, "Share of question-ending turns immediately followed by another turn "
            "(a different or unknown speaker, not the same speaker continuing) -- an immediate-"
            "reply rate, not a judgement that the reply actually answered the question",
            response_rate, "percent", family=FAMILY, sample_size=q_total,
            min_sample=QA_MIN_SAMPLE, evidence=[{"questions": q_total, "immediately_followed": answered}])
        unanswered_finding = finding(
            unanswered_id, "Share of question-ending turns with no immediate reply (last turn, "
            "or the same speaker continues)", unanswered_rate, "percent", family=FAMILY,
            sample_size=q_total, min_sample=QA_MIN_SAMPLE)
    else:
        no_q_warning = "no turn in this document ends in a question mark"
        response_finding = finding(response_id, "Question-response rate", None, "percent",
                                   family=FAMILY, sample_size=0, min_sample=QA_MIN_SAMPLE,
                                   warning=no_q_warning)
        unanswered_finding = finding(unanswered_id, "Unanswered-question rate", None, "percent",
                                     family=FAMILY, sample_size=0, min_sample=QA_MIN_SAMPLE,
                                     warning=no_q_warning)

    reply_like = [(events[i - 1], events[i]) for i in range(1, total_turns)
                 if different_or_unknown(events[i - 1], events[i])]
    if reply_like:
        backchannel_hits = sum(1 for _, nxt in reply_like if len(textlib.words(nxt["text"])) <= 2)
        backchannel_finding = finding(
            backchannel_id, "Share of reply-like turns (adjacent, not the same known speaker "
            "continuing) that are two words or shorter -- 'Yes.', 'No.', 'Really?'",
            rate(backchannel_hits, len(reply_like)), "percent", family=FAMILY,
            sample_size=len(reply_like), min_sample=BACKCHANNEL_MIN_SAMPLE,
            evidence=[{"reply_like_pairs": len(reply_like), "backchannel": backchannel_hits}])
    else:
        backchannel_finding = finding(backchannel_id, "Backchannel/short-response rate", None,
                                      "percent", family=FAMILY, sample_size=0,
                                      min_sample=BACKCHANNEL_MIN_SAMPLE,
                                      warning="no adjacent reply-like turn pairs found")
    return [response_finding, unanswered_finding, backchannel_finding]


# --------------------------------------------------------- findings: coordination

def _finding_function_word_coordination(reply_pairs: list[tuple[dict, dict]],
                                        min_primed: int) -> list[dict]:
    metric_id = PREFIX + "function_word_coordination"
    all_rows = []
    for category, words_set in FUNCTION_CATEGORIES.items():
        rows, _ = _coordination(reply_pairs, _category_marker(words_set), min_primed)
        all_rows.extend({"category": category, **row} for row in rows)
    return [_coordination_finding(
        metric_id, "Function-word coordination, pooled over five categories (articles, "
        "conjunctions, prepositions, pronouns, auxiliary/be-verbs -- the same FUNCTION list "
        "character_voice/dialogue_speaker_function_words compare speakers on): does a "
        "responder's own use of a category rise right after the other speaker used it, above "
        "the responder's own baseline rate in reply position? Distinct from "
        "dialogue.speaker_function_word_distance, which reports a static pairwise distance "
        "with no notion of order or response.",
        all_rows, len(reply_pairs), min_primed)]


def _finding_contraction_convergence(reply_pairs: list[tuple[dict, dict]],
                                     min_primed: int) -> list[dict]:
    rows, n_pairs = _coordination(reply_pairs, _has_contraction, min_primed)
    return [_coordination_finding(
        PREFIX + "contraction_convergence", "Contraction-use coordination: does a responder's "
        "own contraction rate rise right after the other speaker used one? (uses the same "
        "closed-suffix contraction rule as dialogue_speaker_style, never bare 's)",
        rows, n_pairs, min_primed)]


def _finding_punctuation_convergence(reply_pairs: list[tuple[dict, dict]],
                                     min_primed: int) -> list[dict]:
    question_rows, n_pairs = _coordination(reply_pairs, _ends_with("?"), min_primed)
    exclaim_rows, _ = _coordination(reply_pairs, _ends_with("!"), min_primed)
    return [
        _coordination_finding(
            PREFIX + "question_mark_convergence", "Question-ending coordination: does a "
            "responder end their own turn in '?' more often right after the other speaker "
            "did?", question_rows, n_pairs, min_primed),
        _coordination_finding(
            PREFIX + "exclamation_convergence", "Exclamation-ending coordination: does a "
            "responder end their own turn in '!' more often right after the other speaker "
            "did?", exclaim_rows, n_pairs, min_primed),
    ]


def _finding_length_accommodation(analysis: DocumentAnalysis, events: list[dict],
                                  reply_pairs: list[tuple[dict, dict]],
                                  min_primed: int) -> list[dict]:
    word_id = PREFIX + "turn_length_accommodation"
    sentence_id = PREFIX + "sentence_count_accommodation"
    named_texts = [event["text"] for event in events if event["speaker"]]
    if len(reply_pairs) < 2 or len(named_texts) < 2:
        warning = f"only {len(reply_pairs)} cross-speaker reply pairs available"
        return [finding(word_id, "Turn-length accommodation", None, "percentage points",
                        family=FAMILY, sample_size=len(reply_pairs), min_sample=min_primed,
                        warning=warning),
                finding(sentence_id, "Sentence-count accommodation", None, "percentage points",
                        family=FAMILY, sample_size=len(reply_pairs), min_sample=min_primed,
                        warning=warning)]
    word_threshold = statistics.median(len(textlib.words(text)) for text in named_texts)
    word_rows, n_pairs = _coordination(reply_pairs, _length_marker(word_threshold), min_primed)
    sentence_threshold = statistics.median(
        analysis.derive(text, "turn").sentence_count for text in named_texts)
    sentence_rows, _ = _coordination(
        reply_pairs, _sentence_count_marker(analysis, sentence_threshold), min_primed)
    return [
        _coordination_finding(
            word_id, f"Turn-length accommodation: binary long/short split at the document's own "
            f"median named-turn length ({word_threshold:.0f} words) -- does a responder's own "
            f"turn cross that split more often right after the other speaker's did? Overlaps "
            f"dialogue.turn_words' length distribution; this measures the cross-speaker ADJACENT "
            f"coordination that distribution does not.", word_rows, n_pairs, min_primed),
        _coordination_finding(
            sentence_id, f"Sentence-count accommodation: same construction, split at "
            f"{sentence_threshold:.0f} sentences per turn.", sentence_rows, n_pairs, min_primed),
    ]


def _finding_entrainment(reply_pairs: list[tuple[dict, dict]], min_pairs: int, seed: int,
                         rare_zipf_threshold: float) -> list[dict]:
    lexical_id, rare_id = PREFIX + "lexical_entrainment", PREFIX + "rare_word_entrainment"
    if len(reply_pairs) < min_pairs:
        warning = (f"only {len(reply_pairs)} cross-speaker reply pairs; need at least "
                  f"{min_pairs}")
        return [finding(lexical_id, "Lexical entrainment", None, "rate difference (real minus "
                        "shuffled-control)", family=FAMILY, sample_size=len(reply_pairs),
                        min_sample=min_pairs, warning=warning),
                finding(rare_id, "Rare-word entrainment", None, "rate difference (real minus "
                        "shuffled-control)", family=FAMILY, sample_size=len(reply_pairs),
                        min_sample=min_pairs, warning=warning)]

    text_pairs = [(prev["text"], nxt["text"]) for prev, nxt in reply_pairs]

    def overlap(prev_text: str, nxt_text: str) -> float:
        return 1.0 if _content_words(prev_text) & _content_words(nxt_text) else 0.0

    lexical = _shuffle_contrast(text_pairs, overlap, seed)
    wordfreq, wf_reason = require("wordfreq")
    if wordfreq is not None:
        def is_rare(word: str) -> bool:
            zipf = wordfreq.zipf_frequency(word, "en")
            return 0 < zipf < rare_zipf_threshold
        rare_warning = None
    else:
        def is_rare(word: str) -> bool:
            return len(word) >= 8
        rare_warning = (f"wordfreq unavailable ({wf_reason}); used a length>=8-character proxy "
                        f"for 'rare' instead of a real corpus-frequency threshold")

    def rare_overlap(prev_text: str, nxt_text: str) -> float:
        prev_rare = {word for word in _content_words(prev_text) if is_rare(word)}
        nxt_rare = {word for word in _content_words(nxt_text) if is_rare(word)}
        return 1.0 if prev_rare & nxt_rare else 0.0

    rare = _shuffle_contrast(text_pairs, rare_overlap, seed)
    lexical_finding = finding(
        lexical_id, "Lexical entrainment: real adjacent-pair shared-content-word rate minus the "
        "same rate against a seeded shuffled partner (controls for two turns simply discussing "
        "the same standing topic)", lexical["contrast"],
        "rate difference (real minus shuffled-control)", family=FAMILY,
        sample_size=lexical["n_pairs"], min_sample=min_pairs,
        distribution={"real_rate_percent": 100 * lexical["real_rate"],
                     "control_rate_percent": 100 * lexical["control_rate"], "seed": seed})
    rare_finding = finding(
        rare_id, f"Rare-word entrainment: same construction, restricted to content words with "
        f"a wordfreq zipf frequency under {rare_zipf_threshold} (English)", rare["contrast"],
        "rate difference (real minus shuffled-control)", family=FAMILY,
        sample_size=rare["n_pairs"], min_sample=min_pairs,
        distribution={"real_rate_percent": 100 * rare["real_rate"],
                     "control_rate_percent": 100 * rare["control_rate"], "seed": seed},
        warning=rare_warning)
    return [lexical_finding, rare_finding]


def _finding_response_relevance(events: list[dict], min_pairs: int, seed: int,
                                use_embedding: bool, embedding_model: str) -> list[dict]:
    metric_id = PREFIX + "response_relevance"
    reply_like = [(events[i - 1], events[i]) for i in range(1, len(events))
                 if not (events[i - 1]["speaker"] and events[i]["speaker"]
                        and events[i - 1]["speaker"] == events[i]["speaker"])]
    if len(reply_like) < min_pairs:
        return [finding(metric_id, "Response relevance", None, "similarity difference (real "
                        "minus shuffled-control)", family=FAMILY, sample_size=len(reply_like),
                        min_sample=min_pairs,
                        warning=f"only {len(reply_like)} adjacent reply-like turn pairs; need "
                                f"at least {min_pairs}")]

    prev_texts = [prev["text"] for prev, _ in reply_like]
    next_texts = [nxt["text"] for _, nxt in reply_like]
    backend = "lexical"
    prev_vectors = lexical_vectors(prev_texts)
    next_vectors = lexical_vectors(next_texts)

    def lexical_similarity(prev_vector: Mapping[str, float], next_vector: Mapping[str, float]) -> float:
        distance = cosine_distance(prev_vector, next_vector)
        return 1.0 - distance if distance is not None else 0.0

    pairs: list[tuple[Any, Any]] = list(zip(prev_vectors, next_vectors))
    score_fn: Callable[[Any, Any], float] = lexical_similarity
    note = ("backend=lexical: TF-IDF cosine similarity, IDF built from this document's own "
           "turns -- a topical-overlap proxy, not a semantic-relevance measure (see "
           "semantic_adjacent's module docstring); it will miss a paraphrased reply that "
           "shares no vocabulary with what it answers")

    if use_embedding:
        vectors, reason = embed_texts(prev_texts + next_texts, embedding_model)
        if vectors is not None:
            split = len(prev_texts)
            embedded_prev, embedded_next = vectors[:split], vectors[split:]
            pairs = list(zip(embedded_prev, embedded_next))
            score_fn = lambda a, b: float((a * b).sum())
            backend, note = "embedding", None
        else:
            note = f"backend=lexical (embedding requested but unavailable: {reason})"

    contrast = _shuffle_contrast(pairs, score_fn, seed)
    return [finding(
        metric_id, "Response relevance: real adjacent reply-like pair similarity minus the same "
        "similarity against a seeded shuffled partner", contrast["contrast"],
        "similarity difference (real minus shuffled-control)", family=FAMILY,
        sample_size=contrast["n_pairs"], min_sample=min_pairs,
        distribution={"backend": backend, "real_similarity": contrast["real_rate"],
                     "control_similarity": contrast["control_rate"],
                     "embedding_model": embedding_model if backend == "embedding" else None,
                     "seed": seed},
        warning=note)]


def _finding_politeness(events: list[dict]) -> list[dict]:
    metric_id = PREFIX + "politeness_strategy_rate"
    total = len(events)
    if total == 0:
        return [finding(metric_id, "Politeness-strategy rate", None, "percent", family=FAMILY,
                        sample_size=0, min_sample=MIN_SAMPLE, warning="no spoken turns found")]
    per_strategy: Counter[str] = Counter()
    any_hit = 0
    for event in events:
        hits = _politeness_hits(event["text"])
        per_strategy.update(hits)
        any_hit += bool(hits)
    return [finding(
        metric_id, "Share of turns containing at least one recognized politeness strategy "
        "(regex approximation of a subset of Danescu-Niculescu-Mizil & Lee 2013's politeness "
        "strategies -- gratitude, please, apology, greeting, hedge, indirect request, direct "
        "question, first-person-plural, second-person; not their published classifier, see "
        "module docstring for why ConvoKit's own could not be reached here)",
        rate(any_hit, total), "percent", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
        distribution={"per_strategy_percent": {name: rate(count, total)
                                              for name, count in per_strategy.items()}},
        evidence=[{"strategy": name, "count": count, "rate_percent": rate(count, total)}
                 for name, count in per_strategy.most_common(EVIDENCE_LIMIT)])]


def _finding_sentiment(events: list[dict], reply_pairs: list[tuple[dict, dict]],
                       state: Mapping[str, Any], min_turns: int, min_coverage: float) -> list[dict]:
    sentiment_id = PREFIX + "sentiment_coupling"
    emotion_id = PREFIX + "emotion_coupling"
    spread_id = PREFIX + "speaker_sentiment_spread"
    if not state["sufficient"]:
        reason = _insufficient_reason(state, min_turns, min_coverage)
        return [finding(sentiment_id, "VADER sentiment coupling", None, "pearson r",
                        family=FAMILY, sample_size=0, min_sample=5, warning=reason),
                finding(emotion_id, "NRC emotion coupling", None, "pearson r", family=FAMILY,
                        sample_size=0, min_sample=5, warning=reason),
                finding(spread_id, "Spread of per-speaker mean sentiment", None,
                        "compound score", family=FAMILY, sample_size=0, min_sample=2,
                        warning=reason)]

    named_events = [event for event in events if event["speaker"]]
    named_texts = [event["text"] for event in named_events]
    vader_values, vader_reason = _vader_scores(named_texts)
    nrc_values, nrc_reason = _nrc_scores(named_texts)

    def coupling(metric_id: str, name: str, values: list[float] | None,
                reason: str | None) -> dict[str, Any]:
        if values is None:
            return finding(metric_id, name, None, "pearson r", family=FAMILY, sample_size=0,
                          min_sample=5, warning=reason)
        by_id = {id(event): value for event, value in zip(named_events, values)}
        xs = [by_id[id(prev)] for prev, nxt in reply_pairs
             if id(prev) in by_id and id(nxt) in by_id]
        ys = [by_id[id(nxt)] for prev, nxt in reply_pairs
             if id(prev) in by_id and id(nxt) in by_id]
        r = _pearson(xs, ys)
        return finding(metric_id, name, r, "pearson r", family=FAMILY, sample_size=len(xs),
                      min_sample=5, distribution={"n_pairs": len(xs)},
                      warning=None if r is not None else
                      "not enough cross-speaker reply pairs with a usable sentiment score")

    sentiment_finding = coupling(
        sentiment_id, "Pearson correlation between one turn's VADER compound sentiment and the "
        "immediately following cross-speaker turn's -- the same scorer sequences.py's "
        "sentence_sentiment_compound sequence uses, applied per spoken turn rather than per "
        "sentence", vader_values, vader_reason)
    emotion_finding = coupling(
        emotion_id, "Pearson correlation between one turn's NRC positive-minus-negative affect "
        "balance and the immediately following cross-speaker turn's -- the same scorer "
        "sequences.py's sentence_emotion_valence sequence uses, applied per spoken turn",
        nrc_values, nrc_reason)

    if vader_values is not None:
        by_speaker: dict[str, list[float]] = defaultdict(list)
        for event, value in zip(named_events, vader_values):
            by_speaker[event["speaker"]].append(value)
        qualifying_rows = [{"speaker": name, "mean_sentiment": statistics.fmean(values),
                            "n_turns": len(values)}
                          for name, values in by_speaker.items() if name in state["qualifying"]]
        means = [row["mean_sentiment"] for row in qualifying_rows]
        spread_summary = summarize(means)
        qualifying_rows.sort(key=lambda row: -row["n_turns"])
        spread_finding = finding(
            spread_id, "Spread of per-speaker mean VADER compound sentiment across qualifying "
            "speakers (headline is the MEDIAN of those per-speaker means; the shape of the "
            "spread across speakers is in distribution, mirroring "
            "dialogue.speaker_question_rate's 'spread of per-speaker rate' convention)",
            spread_summary.get("median"), "compound score", family=FAMILY, sample_size=len(means),
            distribution=spread_summary, min_sample=2, evidence=qualifying_rows[:EVIDENCE_LIMIT])
    else:
        spread_finding = finding(spread_id, "Spread of per-speaker mean sentiment", None,
                                 "compound score", family=FAMILY, sample_size=0, min_sample=2,
                                 warning=vader_reason)
    return [sentiment_finding, emotion_finding, spread_finding]


def _finding_separability(state: Mapping[str, Any], min_turns: int, min_coverage: float,
                          max_turns: int, seed: int) -> list[dict]:
    metric_id = PREFIX + "speaker_separability_accuracy"
    if not state["sufficient"]:
        reason = _insufficient_reason(state, min_turns, min_coverage)
        return [finding(metric_id, "Speaker style separability", None, "accuracy above chance",
                        family=FAMILY, sample_size=0, min_sample=6, warning=reason)]
    labeled = [(name, turn["text"]) for name, turns in state["qualifying"].items()
              for turn in turns]
    if len(labeled) > max_turns:
        labeled = random.Random(seed).sample(labeled, max_turns)
    vectors = [(name, _turn_vector(text, FUNCTION_CATEGORIES)) for name, text in labeled]
    result = _separability_accuracy(vectors)
    if result is None:
        return [finding(metric_id, "Speaker style separability", None, "accuracy above chance",
                        family=FAMILY, sample_size=len(labeled), min_sample=6,
                        warning="not enough named turns across at least two speakers for a "
                                "leave-one-out classification")]
    return [finding(
        metric_id, "Leave-one-out nearest-centroid speaker-identification accuracy, above chance "
        "(1/n_speakers): a broader per-turn vector (word count, contraction/question/"
        "exclamation-ending markers, five function-word category rates) than "
        "dialogue.speaker_function_word_distance's function-words-only pairwise averages, and a "
        "classification question ('can a turn be attributed to its speaker') rather than a "
        "distance between two averages", result["excess"], "accuracy above chance",
        family=FAMILY, sample_size=result["n_turns"], min_sample=6, distribution=result)]


def _finding_graph(reply_pairs: list[tuple[dict, dict]], min_edges: int) -> list[dict]:
    density_id, reciprocity_id = PREFIX + "graph_density", PREFIX + "graph_reciprocity"
    stats_ = _graph_stats(reply_pairs)
    n_edges = stats_["edges"] if stats_ else 0
    if stats_ is None or n_edges < min_edges:
        warning = (f"only {n_edges} distinct speaker-pair edges; need at least {min_edges} for "
                  f"a stable graph statistic")
        return [finding(density_id, "Conversation-graph density", None,
                        "edges / possible edges", family=FAMILY, sample_size=n_edges,
                        min_sample=min_edges, warning=warning),
                finding(reciprocity_id, "Reciprocity of turn exchange", None,
                        "share of edges reciprocated", family=FAMILY, sample_size=n_edges,
                        min_sample=min_edges, warning=warning)]
    return [
        finding(density_id, "Conversation-graph density: speakers as nodes, an edge per ordered "
                "(speaker-replied-to, replying-speaker) pair that occurred at least once",
                stats_["density"], "edges / possible edges", family=FAMILY,
                sample_size=stats_["edges"], min_sample=min_edges, evidence=stats_["top_edges"],
                distribution={"nodes": stats_["nodes"], "edges": stats_["edges"],
                             "networkx_density": stats_.get("networkx_density"),
                             "networkx_note": stats_.get("networkx_note")}),
        finding(reciprocity_id, "Reciprocity of turn exchange: share of directed reply edges "
                "that go both ways (A replies to B AND B replies to A somewhere in the "
                "document)", stats_["reciprocity"], "share of edges reciprocated",
                family=FAMILY, sample_size=stats_["edges"], min_sample=min_edges,
                evidence=stats_["top_edges"],
                distribution={"networkx_reciprocity": stats_.get("networkx_reciprocity"),
                             "networkx_note": stats_.get("networkx_note")}),
    ]


def _finding_pos_convergence(analysis: DocumentAnalysis, reply_pairs: list[tuple[dict, dict]],
                             min_primed: int, max_pairs: int) -> list[dict]:
    metric_id = PREFIX + "pos_pattern_convergence"
    if analysis.nlp_unavailable:
        return [finding(metric_id, "POS-pattern convergence", None, "percentage points",
                        family=FAMILY, sample_size=0, min_sample=min_primed,
                        warning=f"spaCy parse unavailable ({analysis.nlp_unavailable})")]
    if len(reply_pairs) < 2:
        return [finding(metric_id, "POS-pattern convergence", None, "percentage points",
                        family=FAMILY, sample_size=len(reply_pairs), min_sample=min_primed,
                        warning="not enough cross-speaker reply pairs")]
    sample = reply_pairs[:max_pairs]
    pos_cache: dict[str, Counter] = {}

    def pos_counts(text: str) -> Counter:
        if text not in pos_cache:
            counter: Counter = Counter()
            for _, doc in analysis.derive(text, "turn").spacy_docs():
                counter.update(token.pos_ for token in doc if not token.is_space)
            pos_cache[text] = counter
        return pos_cache[text]

    def dominant_tag(text: str) -> str | None:
        counts = pos_counts(text)
        return counts.most_common(1)[0][0] if counts else None

    tag_frequency: Counter = Counter()
    for prev, nxt in sample:
        tag_frequency.update(pos_counts(prev["text"]).keys())
        tag_frequency.update(pos_counts(nxt["text"]).keys())
    top_tags = [tag for tag, _ in tag_frequency.most_common(6)]
    if not top_tags:
        return [finding(metric_id, "POS-pattern convergence", None, "percentage points",
                        family=FAMILY, sample_size=len(sample), min_sample=min_primed,
                        warning="no POS tags recovered from the sampled turns")]
    all_rows = []
    for tag in top_tags:
        rows, _ = _coordination(sample, lambda text, tag=tag: dominant_tag(text) == tag,
                                min_primed)
        all_rows.extend({"pos_tag": tag, **row} for row in rows)
    return [_coordination_finding(
        metric_id, f"POS-pattern convergence: does a responder's own turn's DOMINANT coarse "
        f"POS tag match the other speaker's preceding turn's more often than the responder's "
        f"own baseline? Top {len(top_tags)} tags observed, sampled from the first "
        f"{len(sample)} of {len(reply_pairs)} cross-speaker reply pairs (capped at {max_pairs} "
        f"for parse cost).", all_rows, len(sample), min_primed)]


def _finding_dialogue_act(events: list[dict], model_name: str, max_turns: int) -> list[dict]:
    distribution_id = PREFIX + "dialogue_act_distribution"
    entropy_id = PREFIX + "dialogue_act_transition_entropy"
    texts = [event["text"] for event in events]
    if len(texts) < 2:
        warning = "no spoken turns found"
        return [finding(distribution_id, "Dialogue-act label distribution", None,
                        "percent (top label share)", family=FAMILY, sample_size=0,
                        min_sample=10, warning=warning),
                finding(entropy_id, "Dialogue-act transition entropy", None, "ratio",
                        family=FAMILY, sample_size=0, min_sample=10, warning=warning)]
    pipe, reason = _load_dialogue_act_pipeline(model_name)
    if pipe is None:
        return [finding(distribution_id, "Dialogue-act label distribution", None,
                        "percent (top label share)", family=FAMILY, sample_size=0,
                        min_sample=10, warning=reason),
                finding(entropy_id, "Dialogue-act transition entropy", None, "ratio",
                        family=FAMILY, sample_size=0, min_sample=10, warning=reason)]
    sample = texts[:max_turns]
    try:
        raw = pipe(sample, truncation=True)
    except Exception as exc:  # pragma: no cover - runtime/OOM failure
        failure = f"dialogue-act inference failed ({type(exc).__name__}: {exc})"
        return [finding(distribution_id, "Dialogue-act label distribution", None,
                        "percent (top label share)", family=FAMILY, sample_size=0,
                        min_sample=10, warning=failure),
                finding(entropy_id, "Dialogue-act transition entropy", None, "ratio",
                        family=FAMILY, sample_size=0, min_sample=10, warning=failure)]
    labels = [(row[0] if isinstance(row, list) else row)["label"] for row in raw]
    counts = Counter(labels)
    total = len(labels)
    distribution_finding = finding(
        distribution_id, f"Dialogue-act label distribution ({model_name}, versioned via "
        f"transformers' pinned model revision; labels and per-label shares in distribution)",
        rate(counts.most_common(1)[0][1], total), "percent (top label share)", family=FAMILY,
        sample_size=total, min_sample=10,
        distribution={"labels_percent": {label: rate(count, total)
                                        for label, count in counts.items()}, "model": model_name},
        evidence=[{"label": label, "count": count} for label, count in counts.most_common()])
    transitions = _transition_entropy(labels)
    if transitions is None:
        entropy_finding = finding(entropy_id, "Dialogue-act transition entropy", None, "ratio",
                                  family=FAMILY, sample_size=total, min_sample=10,
                                  warning="not enough consecutive turns for a transition")
    else:
        entropy_finding = finding(
            entropy_id, f"Dialogue-act transition entropy, normalized by log2(distinct "
            f"transitions observed) -- the same construction as timeseries_suite's "
            f"topic_transition_entropy feature ({model_name})", transitions["normalized"],
            "ratio", family=FAMILY, sample_size=transitions["n_transitions"], min_sample=10,
            distribution={key: value for key, value in transitions.items() if key != "normalized"})
    return [distribution_finding, entropy_finding]


def _finding_scene_drift(analysis: DocumentAnalysis, min_turns: int, min_coverage: float,
                         window_words: int) -> list[dict]:
    metric_id = PREFIX + "scene_style_drift"
    windows = analysis.windows(window_words)
    if len(windows) < 3:
        return [finding(metric_id, "Scene-to-scene drift in speaker dominance", None,
                        "percentage points (std across windows)", family=FAMILY,
                        sample_size=len(windows), min_sample=3,
                        warning=f"only {len(windows)} scene window(s) at {window_words} words "
                                f"each; need at least 3 to measure drift")]
    per_window = []
    for index, window in enumerate(windows):
        window_events, window_method = _ordered_events(window)
        window_state = _attribution_state(window_events, window_method, min_turns, min_coverage)
        if not window_state["sufficient"]:
            continue
        counts = {name: len(turns) for name, turns in window_state["qualifying"].items()}
        per_window.append({"window": index, "turns": window_state["total_turns"],
                          "top_speaker_turn_share": rate(max(counts.values()), sum(counts.values()))})
    if len(per_window) < 3:
        return [finding(metric_id, "Scene-to-scene drift in speaker dominance", None,
                        "percentage points (std across windows)", family=FAMILY,
                        sample_size=len(per_window), min_sample=3,
                        warning=f"only {len(per_window)} of {len(windows)} windows had "
                                f"sufficient speaker attribution on their own; need at least 3 "
                                f"to measure drift")]
    values = [row["top_speaker_turn_share"] for row in per_window]
    summary = summarize(values)
    return [finding(
        metric_id, "Scene-to-scene drift in speaker dominance: standard deviation, across "
        "windows, of each window's own top-speaker turn share (each window re-runs the exact "
        "attribution/coverage gate the whole-document dominance metric uses, independently)",
        summary.get("std"), "percentage points (std across windows)", family=FAMILY,
        sample_size=len(per_window), distribution=summary, min_sample=3,
        evidence=per_window[:EVIDENCE_LIMIT])]


# --------------------------------------------------------------------- wiring

#: Every group is independently switchable (rule: "every measurement
#: individually switchable"). The heavy/model-backed ones -- pos_convergence
#: (spaCy), dialogue_act (transformers/torch), response_relevance_embedding
#: (sentence_transformers) -- default to False and load nothing under the
#: default config; see config.json's "_requires_*" notes for exactly what
#: each needs and what happens without it. scene_drift defaults to False
#: because it re-runs the whole attribution pipeline per window (bounded, but
#: extra cost a reader has to opt into).
DEFAULT_FEATURES = {
    "attribution": True,
    "dominance": True,
    "alternation": True,
    "length_accommodation": True,
    "function_word_coordination": True,
    "contraction_convergence": True,
    "punctuation_convergence": True,
    "lexical_entrainment": True,
    "turn_taking": True,
    "response_relevance": True,
    "politeness": True,
    "sentiment_coupling": True,
    "speaker_separability": True,
    "graph": True,
    "pos_convergence": False,
    "dialogue_act": False,
    "response_relevance_embedding": False,
    "scene_drift": False,
}

DEFAULT_DIALOGUE_ACT_MODEL = "WSHAPER/distilbert-multilingual-dialogue-act-classifier"
DEFAULT_RESPONSE_RELEVANCE_MODEL = "all-MiniLM-L6-v2"


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    features = {**DEFAULT_FEATURES, **dict(option(config, "features", {}))}
    min_turns = int(option(config, "min_turns_per_speaker", 8))
    min_coverage = float(option(config, "min_attribution_coverage", 25.0))
    min_primed = int(option(config, "min_coordination_pairs", 5))
    min_entrainment_pairs = int(option(config, "min_entrainment_pairs", 10))
    min_graph_edges = int(option(config, "min_graph_edges", 3))
    rare_zipf_threshold = float(option(config, "rare_word_zipf_threshold", 3.0))
    seed = int(option(config, "shuffle_seed", 0))
    separability_max_turns = int(option(config, "separability_max_turns", 400))
    response_relevance_model = str(option(config, "response_relevance_model",
                                          DEFAULT_RESPONSE_RELEVANCE_MODEL))
    dialogue_act_model = str(option(config, "dialogue_act_model", DEFAULT_DIALOGUE_ACT_MODEL))
    dialogue_act_max_turns = int(option(config, "dialogue_act_max_turns", 200))
    pos_convergence_max_pairs = int(option(config, "pos_convergence_max_pairs", 300))
    scene_window_words = int(option(config, "scene_window_words", 6000))

    events, method = _ordered_events(analysis)
    state = _attribution_state(events, method, min_turns, min_coverage)
    reply_pairs = _reply_pairs(events)

    out: list[dict[str, Any]] = []
    if features.get("attribution", True):
        out += _finding_attribution(events, state, min_turns, min_coverage)
    if features.get("dominance", True):
        out += _finding_dominance(state, min_turns, min_coverage)
    if features.get("alternation", True):
        out += _finding_alternation(events, state, min_turns, min_coverage)
    if features.get("turn_taking", True):
        out += _finding_turn_taking(events)
    if features.get("length_accommodation", True):
        out += _finding_length_accommodation(analysis, events, reply_pairs, min_primed)
    if features.get("function_word_coordination", True):
        out += _finding_function_word_coordination(reply_pairs, min_primed)
    if features.get("contraction_convergence", True):
        out += _finding_contraction_convergence(reply_pairs, min_primed)
    if features.get("punctuation_convergence", True):
        out += _finding_punctuation_convergence(reply_pairs, min_primed)
    if features.get("lexical_entrainment", True):
        out += _finding_entrainment(reply_pairs, min_entrainment_pairs, seed, rare_zipf_threshold)
    if features.get("response_relevance", True):
        out += _finding_response_relevance(events, min_entrainment_pairs, seed,
                                           bool(features.get("response_relevance_embedding", False)),
                                           response_relevance_model)
    if features.get("politeness", True):
        out += _finding_politeness(events)
    if features.get("sentiment_coupling", True):
        out += _finding_sentiment(events, reply_pairs, state, min_turns, min_coverage)
    if features.get("speaker_separability", True):
        out += _finding_separability(state, min_turns, min_coverage, separability_max_turns, seed)
    if features.get("graph", True):
        out += _finding_graph(reply_pairs, min_graph_edges)
    if features.get("pos_convergence", False):
        out += _finding_pos_convergence(analysis, reply_pairs, min_primed, pos_convergence_max_pairs)
    if features.get("dialogue_act", False):
        out += _finding_dialogue_act(events, dialogue_act_model, dialogue_act_max_turns)
    if features.get("scene_drift", False):
        out += _finding_scene_drift(analysis, min_turns, min_coverage, scene_window_words)
    return out
