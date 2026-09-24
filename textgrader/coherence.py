"""Shared entity-grid, lexical-chain and permutation utilities for the
``coherence_suite`` metric family.

Nothing here is a metric.  This module holds the pieces that are genuinely
shared between several coherence measurements - a Jaccard overlap, a lexical
chain builder, an entity-grid role classifier, a deterministic permutation
scorer - so that :mod:`textgrader.metrics.coherence_suite` reads as a set of
small ``measure``-shaped functions rather than repeating the same windowed
loop five times with slightly different bookkeeping.

Judgement calls made once, here, rather than re-litigated per metric:

* **Entity identity is surface identity, by default, and always as its own
  finding.**  Two mentions are "the same entity" if the lemma of a noun
  chunk's syntactic head matches, case-folded.  This is not coreference: it
  will not link "the old woman" to "she" three sentences later, and it will
  wrongly merge two different rooms both called "the room".  A real
  coreference backend (``fastcoref``) is available as an opt-in feature (see
  :mod:`textgrader.metrics.coherence_suite`'s ``coreference`` flag) and, when
  turned on, resolves exactly the "she" case above.  The surface channel is
  never replaced by it: the two are reported side by side under distinct
  metric ids, tagged with which backend produced each number, because a
  coreference model disagreeing with the lemma heuristic is itself evidence
  (about pronoun density, about how much surface identity alone was hiding),
  not noise to collapse away.
* **Roles are a closed four-way schema**: subject (``S``), object (``O``),
  other mention (``X``), or absent (``-``).  This is deliberately the
  minimal Barzilay/Lapata-style schema, not the fuller PropBank-style role
  set some entity-grid implementations use, because deriving reliable
  finer-grained roles from a dependency parse alone (no semantic role
  labeler is installed) invites more precision than the input supports.
* **Permutations reorder, they never resample.**  Every permutation test in
  this module shuffles the exact list of sentences or paragraphs the document
  already has; it never invents, drops or rewords a unit.  That is what makes
  the result a measurement of order sensitivity rather than of vocabulary.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics.semantic_adjacent import STOPWORDS
from .optional import on_reset, require, shim_fastcoref_transformers
from .text import words as split_words

ROLE_SCHEMA_VERSION = "sxo-v1"

SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass", "expl", "agent"})
OBJECT_DEPS = frozenset({"dobj", "obj", "iobj", "pobj", "dative", "attr", "oprd"})


# --------------------------------------------------------------- lexical

def content_words(text: str, min_len: int = 3) -> list[str]:
    """Lower-cased, stop-word-filtered tokens; the unit every overlap measure
    in this module compares.  Reuses the closed-class stop list the semantic
    similarity family already curated, instead of a second copy of it."""

    return [word.lower() for word in split_words(text)
            if len(word) >= min_len and word.lower() not in STOPWORDS]


def jaccard(a: Iterable[str], b: Iterable[str], default: float | None = None) -> float | None:
    """Jaccard overlap of two content-word sets, or ``default`` if both are empty.

    ``default=None`` (the reporting default) makes an empty-vs-empty pair
    honestly undefined rather than silently perfect or silently zero.
    ``default=0.0`` is used by the permutation scorers, which need a total
    order over shuffles and cannot skip a pair.
    """

    set_a, set_b = set(a), set(b)
    union = set_a | set_b
    if not union:
        return default
    return len(set_a & set_b) / len(union)


def adjacent_overlap(units: Sequence[str], min_len: int = 3) -> list[float]:
    """Jaccard content-word overlap between each unit and the one after it.

    Pairs where both units contribute no content words are left out of the
    result entirely (rather than scored 0 or 1), since "no overlap" and
    "nothing to overlap" are different findings.
    """

    token_sets = [set(content_words(unit, min_len)) for unit in units]
    out: list[float] = []
    for a, b in zip(token_sets, token_sets[1:]):
        value = jaccard(a, b)
        if value is not None:
            out.append(value)
    return out


def _identity(word: str) -> str:
    return word


def lexical_chains(units: Sequence[str], *, gap: int = 3, min_len: int = 3,
                   key_fn: Callable[[str], str] = _identity) -> list[list[int]]:
    """Chains of unit indices connected by a shared content-word concept.

    A chain is a maximal run of units in which each member shares at least
    one content-word concept with a member no more than ``gap`` units
    earlier.  This is the classic lexical-chain idea (Morris & Hirst) reduced
    to its cheapest honest form: with the default ``key_fn`` (exact word
    identity) it is exact repetition, not a thesaurus walk, so it never
    claims a semantic link it cannot support.  Cost is linear in the number
    of content-word occurrences: one dict lookup per token, no pairwise unit
    comparison.

    ``key_fn`` maps a content word to whatever "the same concept" means for
    this chain: the identity function for the default surface chain, or
    :func:`wordnet_concept_key` for the synonym-aware one built on top of it.
    Both walk the exact same algorithm, so the two chain counts on one
    document are directly comparable, and where they disagree is exactly the
    sentences whose repeated *idea* is not a repeated *word*.
    """

    open_chains: dict[str, list[int]] = {}
    finished: list[list[int]] = []
    for index, unit in enumerate(units):
        seen_this_unit: set[str] = set()
        for word in content_words(unit, min_len):
            concept = key_fn(word)
            if concept in seen_this_unit:
                continue
            seen_this_unit.add(concept)
            chain = open_chains.get(concept)
            if chain is not None and index - chain[-1] <= gap:
                chain.append(index)
            else:
                if chain is not None:
                    finished.append(chain)
                open_chains[concept] = [index]
    finished.extend(open_chains.values())
    return finished


# ------------------------------------------------------- WordNet lexical cohesion

_WORDNET_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_wordnet_cache() -> None:
    _WORDNET_CACHE.clear()


on_reset(_reset_wordnet_cache)


def require_wordnet() -> tuple[Any, str | None]:
    """``(wordnet_module, None)`` or ``(None, reason)``; never raises.

    Two independent things can be missing: the ``nltk`` package itself (an
    ordinary :func:`textgrader.optional.require` check), and the ``wordnet``
    corpus data, which ``nltk`` does not bundle and which has its own,
    separate download step.  A cheap lookup (``synsets("the")``) is the only
    reliable way to tell whether the corpus is really there; ``import``ing
    ``nltk.corpus.wordnet`` alone does not touch disk.
    """

    if "wordnet" in _WORDNET_CACHE:
        return _WORDNET_CACHE["wordnet"]
    nltk, reason = require("nltk")
    if nltk is None:
        _WORDNET_CACHE["wordnet"] = (None, reason)
        return _WORDNET_CACHE["wordnet"]
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("the")  # forces the corpus reader to actually touch disk
        outcome: tuple[Any, str | None] = (wn, None)
    except LookupError as exc:
        outcome = (None, f"nltk is installed but the 'wordnet' corpus is not "
                         f"({exc}); run: python -m nltk.downloader wordnet")
    except Exception as exc:  # pragma: no cover - defensive
        outcome = (None, f"wordnet unavailable ({type(exc).__name__}: {exc})")
    _WORDNET_CACHE["wordnet"] = outcome
    return outcome


def wordnet_concept_key(word: str, wn_module: Any) -> str:
    """A WordNet concept id for ``word``, or ``word`` itself if WordNet has
    no entry for it (so a chain never loses a repetition WordNet cannot see;
    it can only gain the synonym links WordNet does see).

    No word-sense disambiguation is attempted: this uses ``synsets(word)[0]``,
    which NLTK returns ordered by how frequent each sense is in a sense-tagged
    corpus.  That is a reasonable default for common words and a real source
    of error for one used in a rare sense - "bank" the riverbank and "bank"
    the financial institution both resolve to whichever sense is more common
    overall, not to the one this document meant.  This is exactly why the
    WordNet-backed chain is reported as its own finding next to the identity
    one rather than replacing it: two implementations of "the same idea
    recurs" disagreeing is the data, not a defect.
    """

    try:
        synsets = wn_module.synsets(word)
    except Exception:  # pragma: no cover - defensive against a corrupt corpus
        synsets = []
    return synsets[0].name() if synsets else word


def wordnet_best_synset(word: str, wn_module: Any) -> Any | None:
    """The same first-sense call as :func:`wordnet_concept_key`, but returning
    the synset object itself rather than its name, for callers (see
    :func:`hypernym_lexical_chains`) that need to measure a DISTANCE between
    two words' senses rather than test them for exact equality."""

    try:
        synsets = wn_module.synsets(word)
    except Exception:  # pragma: no cover - defensive against a corrupt corpus
        synsets = []
    return synsets[0] if synsets else None


def hypernym_lexical_chains(units: Sequence[str], wn_module: Any, *, gap: int = 3,
                            min_len: int = 3, max_distance: int = 3
                           ) -> list[dict[str, Any]]:
    """Chains linked by WordNet hypernym-tree PROXIMITY, not identity.

    :func:`lexical_chains` (identity) and its ``key_fn=wordnet_concept_key``
    mode (synonymy) both require two mentions to resolve to the exact same
    concept before they count as "the same idea recurring" - a chain of "car"
    and "sedan" only forms if both happen to share their very first WordNet
    sense.  This chain is deliberately looser: two words link if their
    first-sense synsets sit within ``max_distance`` hops of each other in
    WordNet's hypernym tree (``car`` -> ``motor vehicle`` <- ``truck`` is
    distance 2), which catches "the same broad idea keeps coming up" even
    when no single word or sense is ever repeated. That looseness is exactly
    why this is reported as its OWN finding next to the identity- and
    synonym-based ones rather than replacing either: a document can score
    high here and low on both of the stricter chains, and that gap is itself
    the finding (repetition of a *category*, not of a word or a sense).

    Cost is bounded the same way :func:`lexical_chains` bounds it: only
    chains still inside the ``gap``-sentence window are ever compared
    against, so a chain that fell silent is moved to ``finished`` and stops
    costing anything, keeping this roughly linear in the number of content
    words actually in play at once rather than in the whole document's
    length. ``shortest_path_distance`` is still the expensive part per
    comparison (a bounded hypernym-tree walk), which is the reason this
    channel is opt-in (``features.lexical_wordnet_hypernym``) rather than
    folded into the identity chain that always runs.

    Returns a list of ``{"indices": [...], "distances": [...]}`` dicts (one
    per chain; ``distances`` has one entry per link after the first mention),
    not the bare index lists :func:`lexical_chains` returns, because a
    hypernym chain's own distance is data a caller needs to report (how
    loose was the concept this chain tracked), not something implied by
    membership the way exact-match chains imply distance zero.
    """

    open_chains: list[dict[str, Any]] = []
    finished: list[dict[str, Any]] = []
    synset_cache: dict[str, Any] = {}
    for index, unit in enumerate(units):
        still_open = []
        for chain in open_chains:
            (finished if index - chain["last_index"] > gap else still_open).append(chain)
        open_chains = still_open
        seen_this_unit: set[str] = set()
        for word in content_words(unit, min_len):
            if word in seen_this_unit:
                continue
            seen_this_unit.add(word)
            if word not in synset_cache:
                synset_cache[word] = wordnet_best_synset(word, wn_module)
            synset = synset_cache[word]
            if synset is None:
                continue
            best_chain, best_distance = None, None
            for chain in open_chains:
                if chain["last_index"] == index:
                    continue  # already extended by another word this unit
                try:
                    distance = synset.shortest_path_distance(chain["synset"])
                except Exception:  # pragma: no cover - cross-POS/disconnected synsets
                    distance = None
                if distance is None or distance > max_distance:
                    continue
                if best_distance is None or distance < best_distance:
                    best_chain, best_distance = chain, distance
            if best_chain is not None:
                best_chain["indices"].append(index)
                best_chain["last_index"] = index
                best_chain["distances"].append(best_distance)
            else:
                open_chains.append({"synset": synset, "last_index": index,
                                    "indices": [index], "distances": []})
    finished.extend(open_chains)
    return [{"indices": chain["indices"], "distances": chain["distances"]} for chain in finished]


# ----------------------------------------------------------------- entities

def chunk_role(dep: str) -> str:
    if dep in SUBJECT_DEPS:
        return "S"
    if dep in OBJECT_DEPS:
        return "O"
    return "X"

_ROLE_RANK = {"S": 3, "O": 2, "X": 1}


def entity_mentions_by_sentence(sents_channels: Iterable[tuple[Any, str, int]]
                                ) -> tuple[list[dict[str, str]], list[str]]:
    """``(mentions_per_sentence, channel_per_sentence)`` from noun chunks.

    ``mentions_per_sentence[i]`` maps an entity key (a lemma) to the single
    best role ``spaCy`` evidenced for it in sentence ``i`` (subject beats
    object beats other, so a name used as both subject and object of the same
    sentence is not double counted). Only chunks headed by a common or proper
    noun become entities; a chunk headed by a pronoun is real evidence of
    *something* being talked about but, without coreference, this module has
    no way to say what, so pronoun-headed chunks are not tracked as entities
    here (see :func:`coref_mentions_by_sentence` for the backend that can).

    ``sents_channels`` is ``DocumentAnalysis.spacy_sents_by_channel()``'s
    ``(sentence, channel, offset)`` triples rather than a bare sentence
    iterable, so that a per-channel (dialogue/narration) entity-grid metric
    costs nothing beyond the one walk of the shared parse the full-document
    metrics already pay for - re-parsing ``analysis.narration``/``.dialogue``
    separately was the cost the suite's docstring used to justify skipping
    this split entirely.
    """

    per_sentence: list[dict[str, str]] = []
    channels: list[str] = []
    for sent, channel, _offset in sents_channels:
        roles: dict[str, str] = {}
        try:
            chunks = list(sent.noun_chunks)
        except Exception:  # pragma: no cover - defensive against parser edge cases
            chunks = []
        for chunk in chunks:
            root = chunk.root
            if root.pos_ not in ("NOUN", "PROPN"):
                continue
            key = root.lemma_.lower().strip()
            if not key:
                continue
            role = chunk_role(root.dep_)
            if _ROLE_RANK.get(role, 0) > _ROLE_RANK.get(roles.get(key, ""), 0):
                roles[key] = role
        per_sentence.append(roles)
        channels.append(channel)
    return per_sentence, channels


def channel_indices(channels: Sequence[str], wanted: str) -> list[int]:
    """Indices of ``channels`` equal to ``wanted`` ("narration"/"dialogue").

    A sentence spaCy's own boundary detection put half in quotes and half out
    (``"mixed"``, from :meth:`DocumentAnalysis.spacy_sents_by_channel`) counts
    toward neither channel, the same rule the lexical overlap channels use.
    """

    return [index for index, channel in enumerate(channels) if channel == wanted]


def project_rows(per_sentence: Sequence[Mapping[str, str]], indices: Sequence[int]
                 ) -> list[dict[str, str]]:
    """The subsequence of ``per_sentence`` at ``indices``, reindexed 0..n-1.

    This is what makes a channel split of the entity grid meaningful rather
    than a shuffle: "reintroduced 3 sentences later" has to mean 3 sentences
    *of that channel*, not 3 sentences of the interleaved original with the
    other channel's sentences silently still counted in the gap.  Entity
    *identity* is still whatever the caller tracked over the whole document
    (so a chain spanning a channel gap is not severed); only which
    observations count toward a channel's own rates is filtered here.
    """

    return [dict(per_sentence[index]) for index in indices]


def entity_frequency(per_sentence: Sequence[Mapping[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for roles in per_sentence:
        for key in roles:
            counts[key] = counts.get(key, 0) + 1
    return counts


def grid_rows(per_sentence: Sequence[Mapping[str, str]], entities: Sequence[str]
             ) -> dict[str, list[str]]:
    """Dense role sequence (including ``"-"`` for absent) per tracked entity."""

    return {key: [roles.get(key, "-") for roles in per_sentence] for key in entities}


def transition_counts(rows: Mapping[str, Sequence[str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for sequence in rows.values():
        for a, b in zip(sequence, sequence[1:]):
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1
    return counts


#: The closed four-way role schema's 16 ordered transitions, as the fixed,
#: JSON-safe string keys :func:`transition_frequency_vector` always emits (one
#: entry per pair, present even at 0.0), so two documents' vectors -- or a
#: document's and a corpus's pooled mean -- are always directly comparable
#: term by term without either side guessing which keys the other has.
ROLES: tuple[str, ...] = ("S", "O", "X", "-")
TRANSITION_KEYS: tuple[str, ...] = tuple(f"{a}->{b}" for a in ROLES for b in ROLES)


def transition_frequency_vector(counts: Mapping[tuple[str, str], int]) -> dict[str, float]:
    """The 16-way entity-grid transition table as a probability vector.

    This is the table :func:`transition_counts` already produces, reshaped
    into the one form a corpus profile can actually cache: a flat, fixed-key
    ``dict[str, float]`` summing to 1.0, suitable for
    :func:`textgrader.corpus.build_profile`'s ``profile_vector`` hook (which
    stores one such row per book) exactly the way
    :func:`textgrader.metrics.function_words.vector` already does for
    function-word rates.  Returns ``{}`` (falsy, so a caller can treat it the
    same as "nothing to cache") when there were no transitions at all, rather
    than a vector of sixteen zeros that would silently pull a corpus mean
    toward zero for a book that simply had no entity grid.
    """

    total = sum(counts.values())
    if not total:
        return {}
    return {f"{a}->{b}": counts.get((a, b), 0) / total for a in ROLES for b in ROLES}


def entropy_of_counts(counts: Mapping[Any, int]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def build_entity_graph(module: Any, rows: Mapping[str, Sequence[str]], window: int):
    """A ``networkx.Graph`` linking entities that co-occur within ``window``
    sentences of each other at least once.  ``module`` is the already
    ``optional.require``-d ``networkx`` module; this function never imports it
    itself, so a caller without the package never pays for the attempt."""

    graph = module.Graph()
    graph.add_nodes_from(rows)
    keys = list(rows)
    # Sentence -> the entities present in it, built once instead of an
    # all-pairs scan of (entity, entity, sentence).
    length = len(next(iter(rows.values()))) if rows else 0
    present_at: list[list[str]] = [[] for _ in range(length)]
    for key, sequence in rows.items():
        for index, role in enumerate(sequence):
            if role != "-":
                present_at[index].append(key)
    for start in range(length):
        window_entities: set[str] = set()
        for offset in range(window + 1):
            index = start + offset
            if index >= length:
                break
            window_entities.update(present_at[index])
        window_entities_list = sorted(window_entities)
        for i, a in enumerate(window_entities_list):
            for b in window_entities_list[i + 1:]:
                graph.add_edge(a, b)
    return graph


def graph_stats(module: Any, graph: Any) -> dict[str, Any]:
    nodes = graph.number_of_nodes()
    if nodes == 0:
        return {"nodes": 0, "edges": 0, "density": None, "average_degree": None,
                "average_clustering": None, "connected_components": 0,
                "largest_component_share": None}
    components = list(module.connected_components(graph))
    largest = max((len(component) for component in components), default=0)
    degrees = [degree for _, degree in graph.degree()]
    return {
        "nodes": nodes, "edges": graph.number_of_edges(),
        "density": module.density(graph),
        "average_degree": sum(degrees) / nodes if nodes else None,
        "average_clustering": module.average_clustering(graph) if nodes else None,
        "connected_components": len(components),
        "largest_component_share": largest / nodes if nodes else None,
    }


# ----------------------------------------------------------- real coreference

#: Default fastcoref model: small (~90M parameters), CPU-friendly, and the
#: package's own default, so a user who only sets ``features.coreference`` to
#: true without also naming a model gets the one fastcoref itself considers
#: standard.
DEFAULT_COREF_MODEL = "biu-nlp/f-coref"

_COREF_MODEL_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_coref_model_cache() -> None:
    _COREF_MODEL_CACHE.clear()


on_reset(_reset_coref_model_cache)


def _shim_transformers_tied_weights() -> None:
    """Kept as a name other modules and docstrings already reference; the
    implementation lives in :func:`textgrader.optional.shim_fastcoref_transformers`
    so this suite and the logic suite cannot disagree about whether
    coreference loads.
    """

    shim_fastcoref_transformers()


def _load_coref_model(model_name: str) -> tuple[Any, str | None]:
    if model_name in _COREF_MODEL_CACHE:
        return _COREF_MODEL_CACHE[model_name]
    module, reason = require("fastcoref")
    if module is None:
        _COREF_MODEL_CACHE[model_name] = (None, reason)
        return _COREF_MODEL_CACHE[model_name]
    try:
        _shim_transformers_tied_weights()
        model = module.FCoref(model_name_or_path=model_name, device="cpu",
                              enable_progress_bar=False)
        outcome: tuple[Any, str | None] = (model, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"fastcoref model {model_name!r} unavailable "
                         f"({type(exc).__name__}: {exc}); pip install fastcoref and a "
                         f"compatible transformers version")
    _COREF_MODEL_CACHE[model_name] = outcome
    return outcome


def mention_role(doc: Any, start: int, end: int) -> str:
    """S/O/X for the character span ``doc.text[start:end]``, by the syntactic
    dependency of its head token.

    Unlike :func:`chunk_role`'s noun-chunk callers, a coreference mention span
    can be a single pronoun, a bare proper name, or a span spaCy's own
    noun-chunk detector never proposed; ``Doc.char_span`` with
    ``alignment_mode="expand"`` recovers a token span for all three by
    widening to the nearest token boundaries, and ``"X"`` (other mention) is
    the honest answer on the rare span it still can't align, rather than
    guessing a role or dropping the mention.
    """

    try:
        span = doc.char_span(start, end, alignment_mode="expand")
    except Exception:  # pragma: no cover - defensive against parser edge cases
        span = None
    if span is None or span.root is None:
        return "X"
    return chunk_role(span.root.dep_)


def _cluster_label(doc_text: str, mentions: Sequence[tuple[int, int]], used: set[str]) -> str:
    """A human-readable, unique key for one coreference cluster: the longest
    surface form among its mentions (usually the fullest proper name),
    case-folded, so "Alice", "she" and "her" are reported under "alice" and
    not under whichever mention happened to occur first."""

    best = max(mentions, key=lambda span: span[1] - span[0])
    label = " ".join(doc_text[best[0]:best[1]].split()).lower() or "entity"
    candidate = label
    suffix = 2
    while candidate in used:
        candidate = f"{label}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def coref_mentions_by_sentence(doc: Any, sentence_bounds: Sequence[tuple[int, int]],
                               clusters: Sequence[Sequence[tuple[int, int]]]
                              ) -> list[dict[str, str]]:
    """Per-sentence best role per coreference cluster, mirroring
    :func:`entity_mentions_by_sentence`'s output shape so every downstream
    entity-grid function (frequency, transitions, entropy, the graph) works
    unchanged on either backend.

    ``sentence_bounds`` and every mention span in ``clusters`` share one
    coordinate space: character offsets into ``doc.text``.  A singleton
    cluster (fastcoref found no second mention to link) contributes nothing -
    one mention is not continuity, and the surface channel already covers
    a single occurrence of a name.
    """

    per_sentence: list[dict[str, str]] = [{} for _ in sentence_bounds]
    used_labels: set[str] = set()
    for mentions in clusters:
        if len(mentions) < 2:
            continue
        label = _cluster_label(doc.text, mentions, used_labels)
        for start, end in mentions:
            index = _bisect_sentence(sentence_bounds, start)
            if index is None:
                continue
            role = mention_role(doc, start, end)
            if _ROLE_RANK.get(role, 0) > _ROLE_RANK.get(per_sentence[index].get(label, ""), 0):
                per_sentence[index][label] = role
    return per_sentence


def _bisect_sentence(bounds: Sequence[tuple[int, int]], position: int) -> int | None:
    """The index of the sentence containing character ``position``, or the
    nearest one before it (a mention can straddle a boundary spaCy drew
    differently than fastcoref's own tokenizer would have); ``None`` only if
    there are no sentences at all."""

    for index, (start, end) in enumerate(bounds):
        if start <= position < end:
            return index
    for index, (_start, end) in enumerate(bounds):
        if position < end:
            return index
    return len(bounds) - 1 if bounds else None


def resolve_coreference(analysis: Any, model_name: str, max_words: int
                        ) -> tuple[list[dict[str, str]], list[str], dict[str, Any], str | None]:
    """Real, fastcoref-backed entity mentions, windowed to bound cost.

    Returns ``(per_sentence_roles, channels, settings, note)``:

    * ``per_sentence_roles``/``channels`` are the same shape
      :func:`entity_mentions_by_sentence` returns, but only over a PREFIX of
      the document (see ``settings["sentences_considered"]``) - coreference
      is expensive enough that running it on a 300,000-word novel by default
      would be a serious performance regression, so this only ever resolves
      the shared parse's first chunk, up to ``max_words`` words, cut at a
      sentence boundary.  For the vast majority of real documents (a chunk is
      up to ``nlp.max_chars_per_chunk`` characters, far larger than any
      sensible coreference cap) that prefix sits entirely inside one chunk;
      the function stops early rather than continue into a second chunk, so
      it never silently resolves coreference across a chunk seam it cannot
      verify spaCy's own boundary detection agreed on.
    * ``settings`` records the backend, model, and window actually used, for
      the caller to fold into every affected finding's ``distribution``.
    * ``note`` is always set: either why coreference could not run at all, or
      a description of the window that was used, because a finding produced
      this way must always say which backend and how much of the document
      produced it.
    """

    model, reason = _load_coref_model(model_name)
    settings = {"backend": "coreference", "model": model_name}
    if model is None:
        return [], [], settings, reason

    doc0 = offset0 = None
    bounds: list[tuple[int, int]] = []
    channels: list[str] = []
    word_total = 0
    for sent, channel, offset in analysis.spacy_sents_by_channel():
        if doc0 is None:
            doc0, offset0 = sent.doc, offset
        elif sent.doc is not doc0:
            break  # do not cross a chunk seam; see the docstring above
        bounds.append((sent.start_char, sent.end_char))
        channels.append(channel)
        word_total += len(sent.text.split())
        if word_total >= max_words:
            break
    if not bounds:
        return [], [], settings, "no sentences in the shared parse to resolve coreference over"

    window_text = doc0.text[:bounds[-1][1]]
    try:
        results = model.predict(texts=[window_text])
    except Exception as exc:  # pragma: no cover - runtime/OOM failure
        return [], [], settings, f"fastcoref inference failed ({type(exc).__name__}: {exc})"

    clusters = results[0].get_clusters(as_strings=False)
    per_sentence = coref_mentions_by_sentence(doc0, bounds, clusters)
    chain_clusters = [cluster for cluster in clusters if len(cluster) >= 2]
    settings.update({
        "sentences_considered": len(bounds), "window_words": word_total,
        "window_word_cap": max_words, "clusters_found": len(clusters),
        "chains_with_two_or_more_mentions": len(chain_clusters),
    })
    note = (f"backend=coreference: fastcoref ({model_name!r}) resolved over the first "
           f"{len(bounds)} sentence(s) (~{word_total:,} words, cap {max_words:,}) of the "
           f"shared parse's first chunk; entities introduced later in a longer document are "
           f"not covered by this channel, unlike the surface-lemma channel, which covers the "
           f"whole document")
    return per_sentence, channels, settings, note


# -------------------------------------------------------------------- RST

#: The rstdt checkpoint of isanlp_rst's v3 model: English RST Discourse
#: Treebank relations, the right inventory for English prose (as opposed to
#: e.g. a de/ru-trained checkpoint the same package also ships).
DEFAULT_RST_MODEL = "tchewik/isanlp_rst_v3"
DEFAULT_RST_MODEL_VERSION = "rstdt"

_RST_PARSER_CACHE: dict[tuple[str, str], tuple[Any, str | None]] = {}


def _reset_rst_parser_cache() -> None:
    _RST_PARSER_CACHE.clear()


on_reset(_reset_rst_parser_cache)


def _load_rst_parser(model_name: str, model_version: str) -> tuple[Any, str | None]:
    """Cached across every call in this process, exactly like
    :func:`_load_coref_model`: constructing ``isanlp_rst.parser.Parser``
    downloads ~5 GB of checkpoints on first use and holds ~5-6 GB resident
    once loaded, so it must never happen more than once per process no
    matter how many documents or findings ask for it."""

    key = (model_name, model_version)
    if key in _RST_PARSER_CACHE:
        return _RST_PARSER_CACHE[key]
    module, reason = require("isanlp_rst")
    if module is None:
        _RST_PARSER_CACHE[key] = (None, reason)
        return _RST_PARSER_CACHE[key]
    try:
        parser = module.Parser(hf_model_name=model_name, hf_model_version=model_version,
                               cuda_device=-1)
        outcome: tuple[Any, str | None] = (parser, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"isanlp_rst parser {model_name!r} (version {model_version!r}) "
                         f"unavailable ({type(exc).__name__}: {exc}); pip install isanlp-rst "
                         f"and its isanlp dependency "
                         f"(pip install git+https://github.com/iinemo/isanlp.git)")
    _RST_PARSER_CACHE[key] = outcome
    return outcome


def rst_tree_summary(node: Any) -> dict[str, Any]:
    """Depth, EDU count, per-EDU word length, and internal-node
    relation/nuclearity counts for one ``isanlp_rst`` ``DiscourseUnit`` tree.

    Walked iteratively (an explicit stack, not recursion) so a pathological
    tree cannot hit Python's recursion limit.  Depth is edge count from the
    root to the deepest leaf (a single-EDU "tree" has depth 0), which is the
    convention that makes trees of different sizes comparable the way this
    suite's other shape-based metrics already are.  Every attribute access is
    defensive (``getattr`` with a default, broad ``except``) because this
    reads exactly the fields the task's own worked example showed
    (``.relation``, ``.nuclearity``, ``.left``, ``.right``, leaves carrying
    ``relation == "elementary"``), not a documented, versioned API contract;
    a future isanlp_rst release that renames or drops one of them can only
    make this summary emptier, never raise.
    """

    relation_counts: Counter = Counter()
    nuclearity_counts: Counter = Counter()
    leaf_word_counts: list[int] = []
    max_depth = 0
    stack: list[tuple[Any, int]] = [(node, 0)]
    while stack:
        current, depth = stack.pop()
        if current is None:
            continue
        try:
            relation = getattr(current, "relation", None)
            left = getattr(current, "left", None)
            right = getattr(current, "right", None)
        except Exception:  # pragma: no cover - defensive against a malformed node
            continue
        is_leaf = relation == "elementary" or (left is None and right is None)
        if is_leaf:
            max_depth = max(max_depth, depth)
            try:
                text = getattr(current, "text", "") or ""
            except Exception:  # pragma: no cover - defensive
                text = ""
            leaf_word_counts.append(len(text.split()))
            continue
        if relation:
            relation_counts[relation] += 1
        try:
            nuclearity = getattr(current, "nuclearity", None)
        except Exception:  # pragma: no cover - defensive
            nuclearity = None
        if nuclearity:
            nuclearity_counts[nuclearity] += 1
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    return {"depth": max_depth, "leaf_count": len(leaf_word_counts),
           "leaf_word_counts": leaf_word_counts, "relation_counts": relation_counts,
           "nuclearity_counts": nuclearity_counts}


def sample_rst_passages(sentences: Sequence[str], num_passages: int, passage_sentences: int,
                        max_sentences: int, seed: int
                       ) -> tuple[list[tuple[int, int, list[str]]], dict[str, Any]]:
    """A deterministic, spread-across-the-book sample of passages, never a
    whole-document parse: at ~2 seconds a sentence, parsing a 300,000-word
    novel sentence by sentence would take hours (see the module's callers'
    docstring), so this is the entire cost-control mechanism for the RST
    channel, exactly the way :func:`resolve_coreference` bounds fastcoref to
    one windowed prefix instead of the whole book -- except a fixed prefix
    would never see the book's ending, so this samples spread evenly across
    it instead.

    The document is split into ``num_passages`` equal-width contiguous
    stripes by sentence index, and one ``passage_sentences``-sentence window
    is drawn from a random position within each stripe using
    ``random.Random(seed)``, so the same document and seed always yield the
    same passages (deterministic) while every stripe of the book gets a
    chance to be sampled (spread) rather than always the opening pages.
    ``num_passages`` is silently reduced (never raised) so that
    ``num_passages * passage_sentences`` never exceeds ``max_sentences`` -- a
    hard cap on total parser cost that config options alone cannot be set to
    exceed.

    Returns ``(passages, sample_info)``: ``passages`` is a list of
    ``(start_index, end_index, sentence_list)`` triples (0 or more, fewer
    than requested only when the hard cap or a short document forced it);
    ``sample_info`` records exactly what was asked for and what was actually
    sampled (document size, passage size and count requested vs. used, the
    seed), so a caller can put it in every finding's ``distribution`` and a
    reader can never mistake a 10-passage sample for a whole-book parse.
    """

    total = len(sentences)
    base_info = {"total_sentences_in_document": total, "passages_requested": max(0, num_passages),
                "passage_sentences_target": max(1, passage_sentences),
                "max_sentences_cap": max(1, max_sentences), "seed": seed}
    if total == 0 or num_passages <= 0 or passage_sentences <= 0:
        return [], {**base_info, "passages_sampled": 0, "total_sentences_sampled": 0}

    passage_len = max(1, min(passage_sentences, total))
    cap_passages = max(1, max_sentences // passage_len)
    n_passages = max(1, min(num_passages, cap_passages))

    rng = random.Random(seed)
    edges = [round(i * total / n_passages) for i in range(n_passages + 1)]
    passages: list[tuple[int, int, list[str]]] = []
    for i in range(n_passages):
        bin_start, bin_end = edges[i], edges[i + 1]
        bin_size = max(1, bin_end - bin_start)
        slack = max(0, bin_size - passage_len)
        start = bin_start + (rng.randint(0, slack) if slack > 0 else 0)
        start = max(0, min(start, total - passage_len))
        end = start + passage_len
        passages.append((start, end, list(sentences[start:end])))

    sampled_sentences = sum(end - start for start, end, _ in passages)
    return passages, {**base_info, "passages_sampled": len(passages),
                      "total_sentences_sampled": sampled_sentences}


def resolve_rst(analysis: Any, model_name: str, model_version: str, num_passages: int,
                passage_sentences: int, max_sentences: int, max_seconds: float, seed: int
               ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    """Real, isanlp_rst-backed discourse-tree summaries over a bounded sample.

    Returns ``(summaries, settings, note)``: ``summaries`` is one dict per
    successfully parsed passage (from :func:`rst_tree_summary`, plus its
    sentence range), ``settings`` carries the backend/model/sampling
    metadata every RST finding folds into its own ``distribution`` (see
    :func:`sample_rst_passages`), and ``note`` is always set -- either why no
    tree could be produced at all, or a plain-language description of what
    was actually sampled and how long it took, because a finding produced
    this way must always say it is a sample, never a whole-book parse.

    ``max_seconds`` is a hard wall-clock budget checked between passages (not
    only a target): once it is exceeded, sampling stops and every finding
    still reports on whatever passages parsed before the cutoff, rather than
    running an unbounded number of two-second-a-sentence parses on a slow
    machine.
    """

    parser, reason = _load_rst_parser(model_name, model_version)
    settings: dict[str, Any] = {"backend": "rst", "model": model_name,
                                "model_version": model_version}
    if parser is None:
        return [], settings, reason

    passages, sample_info = sample_rst_passages(analysis.sentences, num_passages,
                                                passage_sentences, max_sentences, seed)
    settings.update(sample_info)
    if not passages:
        return [], settings, "no sentences available to sample for RST parsing"

    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    stopped_early = False
    for start, end, passage in passages:
        if time.monotonic() - started > max_seconds:
            stopped_early = True
            break
        text = " ".join(passage)
        try:
            result = parser(text)
            tree = ((result or {}).get("rst") or [None])[0]
        except Exception:  # pragma: no cover - a single bad passage should not sink the sample
            continue
        if tree is None:
            continue
        summaries.append({"start_sentence": start, "end_sentence": end,
                          "sentence_count": end - start, **rst_tree_summary(tree)})
    elapsed = time.monotonic() - started
    settings.update({"passages_parsed": len(summaries), "elapsed_seconds": round(elapsed, 1),
                     "max_seconds_cap": max_seconds, "stopped_early_on_time_cap": stopped_early})
    note = (f"backend=rst: isanlp_rst ({model_name!r}, version {model_version!r}) parsed "
           f"{len(summaries)} of {sample_info['passages_sampled']} sampled passage(s) "
           f"({sample_info['total_sentences_sampled']} of "
           f"{sample_info['total_sentences_in_document']:,} sentences in the document, "
           f"seed={seed}) spread across the document in {elapsed:.1f}s; this is a SAMPLE, not a "
           f"whole-book parse" + (" (stopped early: time cap reached)" if stopped_early else ""))
    if not summaries:
        return [], settings, note + "; no sampled passage produced a usable tree"
    return summaries, settings, note


# -------------------------------------------------------------- permutation

def permutation_percentile(real_items: Sequence[Any], score_fn: Callable[[Sequence[Any]], float],
                           permutations: int, rng: random.Random) -> tuple[float | None, float]:
    """Where the real order's score ranks among ``permutations`` reshuffles.

    Returns ``(percentile, real_score)``.  The percentile is the share of the
    ``permutations + 1`` samples (the real order plus every shuffle) that
    score at or below the real order, so ``100.0`` means the real order beat
    every shuffle tried and ``50.0`` means it looked like a typical shuffle.
    Reshuffling never changes the multiset of items, only their order, so a
    high percentile is evidence about arrangement, not about vocabulary.
    """

    if len(real_items) < 2:
        return None, 0.0
    real_score = score_fn(real_items)
    samples = [real_score]
    pool = list(real_items)
    for _ in range(max(0, permutations)):
        rng.shuffle(pool)
        samples.append(score_fn(pool))
    beaten_or_tied = sum(1 for value in samples if value <= real_score)
    return 100.0 * beaten_or_tied / len(samples), real_score


def order_score_from_overlap(units: Sequence[str], min_len: int = 3) -> float:
    """Sum of adjacent content-word overlap; the default, dependency-free
    scoring function for both permutation tests.  Empty-vs-empty pairs score
    0.0 here (not left out, unlike :func:`adjacent_overlap`'s reporting mode)
    because a permutation scorer needs one total order over every shuffle."""

    token_sets = [set(content_words(unit, min_len)) for unit in units]
    return sum(jaccard(a, b, default=0.0) for a, b in zip(token_sets, token_sets[1:]))


__all__ = [
    "ROLE_SCHEMA_VERSION", "content_words", "jaccard", "adjacent_overlap",
    "lexical_chains", "require_wordnet", "wordnet_concept_key", "wordnet_best_synset",
    "hypernym_lexical_chains", "chunk_role",
    "entity_mentions_by_sentence", "channel_indices", "project_rows", "entity_frequency",
    "grid_rows", "transition_counts", "entropy_of_counts", "ROLES", "TRANSITION_KEYS",
    "transition_frequency_vector", "build_entity_graph",
    "graph_stats", "DEFAULT_COREF_MODEL", "mention_role", "coref_mentions_by_sentence",
    "resolve_coreference", "DEFAULT_RST_MODEL", "DEFAULT_RST_MODEL_VERSION",
    "rst_tree_summary", "sample_rst_passages", "resolve_rst",
    "permutation_percentile", "order_score_from_overlap",
]
