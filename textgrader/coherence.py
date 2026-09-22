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
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics.semantic_adjacent import STOPWORDS
from .optional import on_reset, require
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
    """Work around a version-skew crash between ``fastcoref`` and a newer
    ``transformers``, without guessing at anything fastcoref itself computes.

    ``fastcoref==2.1.6``'s model class does not go through the tied-weight
    bookkeeping newer ``transformers`` releases expect every ``PreTrainedModel``
    subclass to have set up before ``from_pretrained`` finishes, so loading can
    raise ``AttributeError: ... has no attribute 'all_tied_weights_keys'``
    before a single weight is loaded - confirmed against transformers 5.17 in
    this environment.  The coref head fastcoref adds has no tied weights of
    its own (it is a span classifier, not a model with an input/output
    embedding to tie), so an empty mapping is the correct value here, not a
    guess standing in for one: this only supplies the default the class
    itself would have set if fastcoref's code called the newer init path, and
    only when the attribute is missing in the first place, so an
    already-compatible transformers install is never touched.
    """

    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:  # pragma: no cover - transformers itself unavailable
        return
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


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
    "lexical_chains", "require_wordnet", "wordnet_concept_key", "chunk_role",
    "entity_mentions_by_sentence", "channel_indices", "project_rows", "entity_frequency",
    "grid_rows", "transition_counts", "entropy_of_counts", "build_entity_graph",
    "graph_stats", "DEFAULT_COREF_MODEL", "mention_role", "coref_mentions_by_sentence",
    "resolve_coreference", "permutation_percentile", "order_score_from_overlap",
]
