"""Shallow subject-predicate-object triples, extracted once and cached.

This is the "OpenIE" mentioned in the logic-suite spec, sized to what is
actually available here: no Stanford OpenIE, no AllenNLP SRL, no coreference
resolver. What spaCy's dependency parse gives for free is a shallow
proposition per clause -- its grammatical subject, its verb, and (if one
exists) an object, attribute or prepositional complement -- and that is all
this module extracts. A sentence yields one proposition for its main clause
and one more for each clausal complement or subordinate clause it carries
(``ccomp``: "she said [the lamp was lit]"; ``advcl``: "when he arrived, [it
was dark]"; verb-headed ``conj``: "the lamp was dark and [the reservoir was
full]") that has its own subject; see ``_CLAUSE_DEPS`` for exactly which
dependency labels count and why ``xcomp``/relative clauses do not. This
matters more than it sounds: without it, a reported or subordinate claim like
"the historian said the lighthouse was built in 1861" would only ever surface
as a proposition about the historian *saying* something, and the actual claim
inside the quote -- the one worth checking against other mentions of the
lighthouse -- would never exist as data at all. It is still not semantic-role
labelling: "give" in "she gave him the book" and "give" in "she gave up"
produce the same predicate lemma here, and a verb's oblique arguments beyond
the first object are dropped entirely. Treat every :class:`Proposition` as
"roughly what one clause asserted, if it asserted one simple thing," not as a
logical form.

Two judgement calls shape everything downstream:

**No coreference, by default.** A pronoun subject ("he", "she", "it", "they")
cannot be linked to the noun phrase it refers to without a coreference
resolver. Every pronoun-subject proposition gets an empty ``subject_key`` and
is excluded from every cross-sentence bucket this module builds unless a
caller opts into :func:`resolve_coreference`, which is off by default. That
default exclusion is a real recall loss on its own -- most contradiction pairs
in ordinary prose are exactly "Alice was tired... She wasn't, though" -- and
it is the reason :func:`resolve_coreference` exists: ``fastcoref`` is now
installed and loads and predicts cleanly here, through
:func:`textgrader.optional.shim_fastcoref_transformers`, and a pronoun's
resolved antecedent is keyed the same way a directly-named subject would be
(see :func:`_coref_representative`), so "she" and "Alice" land in the same
bucket -- exercised against the real model, not only a fake one, on exactly
that "Alice was tired... She wasn't" case; recall on it goes from 0 candidates
with the feature off to 1 with it on. It stays opt-in because it is another
neural model on top of the spaCy parse this module already pays for, its
resolution is only sampled over the first ``coreference_max_chars`` characters
(not the whole book), and every resolution it produces is the model's own
judgement, never a verified reading -- not because it fails to load. See
``_load_coref_model`` for how a genuine load or runtime failure (a missing
package, a future incompatible ``transformers`` release, ...) still degrades
to an actionable ``unavailable`` reason rather than a crash, and
:mod:`textgrader.metrics.logic_suite` for how that reason surfaces in a
finding's warning.

**No named-entity recognition required.** The shared spaCy pipeline disables
``ner`` by default (:class:`textgrader.document.NlpSettings`) because it
roughly doubles parse time, and only one existing metric (``pov_entity_ratio``)
asks for it. Rather than add a second metric family with an unusual NLP
requirement, subject/object identity here is keyed off proper-noun tokens
(``PROPN``, from part-of-speech tagging, which the default pipeline always
runs) with a fall-back to the lemma of common nouns. If ``ner`` happens to be
enabled, the entity label is folded into the key for a sharper match and is
also kept on the ``Proposition`` for evidence; nothing here requires it.

**WordNet, now available.** :func:`wordnet_antonym_conflict` and
:func:`wordnet_hypernym_related` add a lexical-relation signal that is
independent of both the surface-heuristic tests above and of any NLI model:
the former flags a same-subject+predicate pair whose object readings are
direct WordNet antonyms ("open"/"shut"), the latter flags one where the two
objects are actually in an is-a relationship ("dog"/"poodle") and so are
probably not a real conflict despite :func:`attribute_conflict` calling them
``property``-different. Both check every WordNet sense of each lemma, not
just the one used in this sentence, so a false positive from an unusual sense
is possible and each finding built from these says so.

Extraction is bounded by ``cap`` (propositions kept) and runs over
:meth:`DocumentAnalysis.spacy_sents_by_channel`, the one shared parse, so
enabling this alongside ``tense_consistency`` or any other ``parse``-cost
metric costs nothing extra. The result is cached on ``analysis`` via
:meth:`DocumentAnalysis.memo`, keyed by ``cap``, so every metric in
``logic_suite`` that wants propositions gets the same list.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Callable

from .document import DocumentAnalysis
from .optional import on_reset, require, shim_fastcoref_transformers

#: Occurrences of one (subject, predicate) pair beyond this are sampled down
#: to the first this-many (in document order) before pairing.  A hard safety
#: valve independent of the user-facing ``max_pairs``/``max_comparisons``
#: options: it exists so a pathological input (the same short clause repeated
#: thousands of times) cannot make candidate generation itself quadratic
#: before either option gets a chance to apply.
_BUCKET_SAMPLE_CAP = 300

_NUMBER_RE = re.compile(r"^-?\d[\d,]*\.?\d*$")
_YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-9]{2})$")

_OBJECT_DEPS = ("attr", "dobj", "acomp", "oprd", "dative")
_NEGATING_SUBJECTS = {"no", "none", "nobody", "nothing", "neither"}


@dataclass(frozen=True)
class Proposition:
    """One sentence's shallow (subject, predicate, object) reading."""

    sentence_index: int
    paragraph_index: int
    offset: int
    text: str
    channel: str
    subject_text: str
    #: Bucket key for cross-sentence matching, or ``""`` when the subject is a
    #: pronoun and so cannot be safely linked to anything without coreference.
    subject_key: str
    subject_is_pronoun: bool
    predicate_lemma: str
    negated: bool
    object_text: str | None
    object_key: str | None
    object_is_numeric: bool
    object_number: float | None
    entity_labels: tuple[str, ...]
    #: Character span of the subject token within :attr:`DocumentAnalysis.text`
    #: (document-wide, not clause-relative), or ``None`` if it could not be
    #: computed. Used only by :func:`resolve_coreference` to match a pronoun
    #: subject against a coreference resolver's mention spans; every other
    #: consumer of a ``Proposition`` should keep using ``subject_key``.
    subject_char_span: tuple[int, int] | None = None
    #: True once :func:`resolve_coreference` has replaced this pronoun
    #: subject's key with a resolved antecedent's. False for every proposition
    #: produced directly by :func:`extract`.
    subject_resolved_via_coref: bool = False


@dataclass(frozen=True)
class Extraction:
    propositions: list[Proposition]
    sentences_scanned: int
    truncated: bool
    ner_available: bool
    spacy_model: str
    spacy_version: str


def _truncate(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


def _number(text: str) -> float | None:
    cleaned = text.replace(",", "")
    if not _NUMBER_RE.match(cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _looks_like_year(text: str | None) -> bool:
    return bool(text) and bool(_YEAR_RE.match(text.strip()))


def _paragraph_spans(analysis: DocumentAnalysis) -> list[tuple[int, int]]:
    """Character spans of each paragraph within ``analysis.text``, in order.

    Paragraphs are exact substrings of the canonical text (only whitespace
    around them is trimmed elsewhere), so locating each one with a forward
    ``str.find`` from the end of the previous match is exact and O(n) over the
    text. A paragraph that cannot be relocated (should not happen, but a
    metric must not crash if a future cleanup step ever makes one) is simply
    skipped; sentences inside it fall back to the previous paragraph's index.
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    text = analysis.text
    for paragraph in analysis.paragraphs:
        start = text.find(paragraph, cursor)
        if start == -1:
            continue
        end = start + len(paragraph)
        spans.append((start, end))
        cursor = end
    return spans


def _paragraph_index(starts: list[int], offset: int) -> int:
    if not starts:
        return 0
    return max(bisect_right(starts, offset) - 1, 0)


def _normalize_key(token: Any) -> str:
    if token.ent_type_:
        return f"ent:{token.ent_type_}:{token.text.lower()}"
    if token.pos_ == "PROPN":
        return f"propn:{token.text.lower()}"
    return f"lemma:{token.lemma_.lower()}"


def _find(children: Any, deps: tuple[str, ...]) -> Any:
    return next((child for child in children if child.dep_ in deps), None)


def _object_of(root: Any) -> Any:
    direct = _find(root.children, _OBJECT_DEPS)
    if direct is not None:
        return direct
    prep = _find(root.children, ("prep",))
    if prep is not None:
        pobj = _find(prep.children, ("pobj",))
        if pobj is not None:
            return pobj
    return None


def _negated(root: Any, subject: Any) -> bool:
    if any(child.dep_ == "neg" for child in root.children):
        return True
    return subject.lemma_.lower() in _NEGATING_SUBJECTS


#: Dependency labels whose token is itself treated as a clause "root" worth a
#: proposition of its own, beyond the sentence's grammatical ``ROOT``.
#: ``ccomp`` ("she said [the lamp was lit]") and ``advcl`` ("when he arrived,
#: [it was dark]") are where most of the reported-speech and subordinate
#: claims that matter for a contradiction scan actually live -- a sentence
#: whose only extracted proposition was its matrix clause ("she said...")
#: would make every quoted or reported claim in the document invisible to
#: this whole module. ``conj`` is included only when the conjunct is itself
#: verb-like, to catch "the lamp was dark and the reservoir was full" without
#: also firing on ordinary noun-phrase coordination ("cats and dogs").
#: ``xcomp`` and ``relcl`` are deliberately left out: an ``xcomp`` shares its
#: subject with the matrix clause through control, which this module does not
#: resolve, and a relative clause's "subject" is often the relativizer itself
#: ("the man who lied"), which would produce a meaningless bucket key.
_CLAUSE_DEPS = ("ccomp", "advcl")

#: An ``advcl`` introduced by one of these is a hypothetical or a rule, not an
#: assertion: "if the lamp had been lit..." does not claim the lamp was lit.
#: Found by hand-checking this module's own candidate output against a
#: constructed example ("If the lamp had truly been lit at dusk, the
#: reservoir should have been lower...") that produced exactly this false
#: contradiction candidate before the filter was added. A concessive marker
#: ("although", "though") is deliberately NOT filtered: "Although the lamp
#: was lit, the room stayed dark" does assert that the lamp was lit.
_HYPOTHETICAL_MARKERS = frozenset({"if", "unless", "provided"})


def _is_hypothetical(token: Any) -> bool:
    return any(child.dep_ == "mark" and child.lemma_.lower() in _HYPOTHETICAL_MARKERS
              for child in token.children)


def _clause_roots(sent: Any) -> list[Any]:
    roots = [sent.root]
    for token in sent:
        if token.dep_ == "advcl":
            if not _is_hypothetical(token):
                roots.append(token)
        elif token.dep_ == "ccomp":
            roots.append(token)
        elif token.dep_ == "conj" and token.pos_ in ("VERB", "AUX") and token.head is sent.root:
            roots.append(token)
    return roots


def _clause_text(root: Any) -> str:
    """The contiguous span covering just this clause, not the whole sentence.

    ``root.left_edge``/``right_edge`` are the leftmost/rightmost tokens in the
    subtree rooted at ``root``, so slicing the parent ``Doc`` between them is
    the standard spaCy idiom for "the text of this clause". For an embedded
    ``ccomp`` this reads far better as evidence than the enclosing sentence
    ("the lamp was not lit at all" rather than the full "the inspector
    reported that ...").
    """

    return _truncate(root.doc[root.left_edge.i:root.right_edge.i + 1].text)


def _proposition(root: Any, *, sentence_index: int, paragraph_index: int, start_char: int,
                 doc_offset: int, channel: str, entity_labels: tuple[str, ...]) -> "Proposition | None":
    subject = _find(root.children, ("nsubj", "nsubjpass"))
    if subject is None:
        return None
    obj = _object_of(root)
    return Proposition(
        sentence_index=sentence_index, paragraph_index=paragraph_index,
        offset=start_char, text=_clause_text(root), channel=channel,
        subject_text=subject.text,
        subject_key="" if subject.pos_ == "PRON" else _normalize_key(subject),
        subject_is_pronoun=subject.pos_ == "PRON",
        predicate_lemma=root.lemma_.lower(), negated=_negated(root, subject),
        object_text=obj.text if obj is not None else None,
        object_key=_normalize_key(obj) if obj is not None else None,
        object_is_numeric=obj is not None and (obj.like_num or _number(obj.text) is not None),
        object_number=_number(obj.text) if obj is not None else None,
        entity_labels=entity_labels,
        subject_char_span=(doc_offset + subject.idx, doc_offset + subject.idx + len(subject.text)))


def _extract(analysis: DocumentAnalysis, cap: int) -> Extraction:
    if analysis.nlp_unavailable:
        return Extraction([], 0, False, False, "", analysis.nlp_unavailable)
    spans = _paragraph_spans(analysis)
    starts = [start for start, _ in spans]
    props: list[Proposition] = []
    scanned = 0
    truncated = False
    ner_available = "ner" in (getattr(analysis.nlp, "pipe_names", None) or [])
    for sent, channel, offset in analysis.spacy_sents_by_channel():
        scanned += 1
        if len(props) >= cap:
            truncated = True
            continue
        start_char = offset + sent.start_char
        paragraph_index = _paragraph_index(starts, start_char)
        entity_labels = tuple(sorted({ent.label_ for ent in sent.ents}))
        for root in _clause_roots(sent):
            if len(props) >= cap:
                truncated = True
                break
            prop = _proposition(root, sentence_index=scanned - 1, paragraph_index=paragraph_index,
                                start_char=start_char, doc_offset=offset, channel=channel,
                                entity_labels=entity_labels)
            if prop is not None:
                props.append(prop)
    meta = getattr(analysis.nlp, "meta", {}) or {}
    spacy_module, _reason = require("spacy")
    library_version = getattr(spacy_module, "__version__", "unknown")
    return Extraction(
        props, scanned, truncated, ner_available,
        f"{analysis.nlp_settings.model}:{meta.get('name', '')}",
        f"spacy={library_version}; model={meta.get('version', 'unknown')}")


def extract(analysis: DocumentAnalysis, cap: int) -> Extraction:
    """Cached shallow-proposition extraction, shared by every logic-suite metric."""

    return analysis.memo(f"logic_propositions:{int(cap)}", lambda: _extract(analysis, int(cap)))


@dataclass(frozen=True)
class PairScan:
    """Bounded output of :func:`bucketed_pairs`: what was found, and how hard it looked."""

    pairs: list[tuple[Proposition, Proposition, str]]
    comparisons: int
    pairs_capped: bool
    buckets_sampled: bool


def bucketed_pairs(propositions: list[Proposition], *, window_sentences: int,
                   max_pairs: int, max_comparisons: int,
                   test: Callable[[Proposition, Proposition], str | None]) -> PairScan:
    """Candidate pairs sharing a subject and predicate, scored by ``test``.

    This is the one candidate-pair generator behind every contradiction-shaped
    metric in ``logic_suite``. Propositions are bucketed by
    ``(subject_key, predicate_lemma)`` -- the shared-entity-and-topic signal
    the spec asks for -- so two sentences are ever compared only if they
    already agree on who is doing what. Within a bucket, a pair is scanned
    only if the two sentences are within ``window_sentences`` of each other or
    share a paragraph; ``test(a, b)`` then decides, from whatever the caller
    considers a conflict (opposite polarity, a differing value, ...), whether
    it is worth keeping and what to call it.

    Three independent caps bound the cost on a book: ``_BUCKET_SAMPLE_CAP``
    (module-level, not configurable) limits any one bucket before pairing
    starts; ``max_comparisons`` stops the scan outright once that many pairs
    have been examined, matched or not; ``max_pairs`` stops it once that many
    have matched. Nothing here is O(propositions^2): every pair considered
    shares a bucket, and every bucket is capped before its pairs are counted.

    Buckets are visited smallest-first, not in document order. A pathological
    document with one enormous, never-matching bucket early on (the same
    harmless sentence repeated thousands of times, for instance) used to
    exhaust the whole comparison budget before a later, much smaller bucket
    that actually contains a conflict was ever reached -- found by
    hand-testing this function against a deliberately repetitive synthetic
    book, not merely suspected (see
    ``test_bucket_ordering_reaches_a_small_conflict_before_a_huge_uninteresting_bucket``).
    A small bucket is the cheap case regardless of what it contains: it is
    fully compared (or ruled out by the window/paragraph check) in only a few
    comparisons, so visiting every small bucket before spending the budget on
    a large one costs almost nothing and buys every document a real chance at
    its small buckets, no matter where they sit in the text. Ties (equal
    bucket size) keep their first-seen order, via a stable sort on the
    dict's own insertion order, so the result is still deterministic run to
    run for the same input. This does change which pairs a large, heavily
    capped document surfaces relative to visiting buckets in document order --
    a real, intentional behaviour change, not a side effect -- because the
    old order's only property was "whichever bucket the text happens to fill
    first," which is not a property worth preserving over actually finding
    the small, informative buckets a book-length ``max_comparisons`` would
    otherwise never reach.
    """

    buckets: dict[tuple[str, str], list[Proposition]] = defaultdict(list)
    for prop in propositions:
        if prop.subject_key:
            buckets[(prop.subject_key, prop.predicate_lemma)].append(prop)

    ordered_buckets = sorted(buckets.values(), key=len)

    pairs: list[tuple[Proposition, Proposition, str]] = []
    comparisons = 0
    buckets_sampled = False
    window = max(0, int(window_sentences))
    for items in ordered_buckets:
        if len(items) < 2:
            continue
        if len(items) > _BUCKET_SAMPLE_CAP:
            items = items[:_BUCKET_SAMPLE_CAP]
            buckets_sampled = True
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if comparisons >= max_comparisons:
                    return PairScan(pairs, comparisons, True, buckets_sampled)
                a, b = items[i], items[j]
                comparisons += 1
                same_paragraph = a.paragraph_index == b.paragraph_index
                if not same_paragraph and abs(a.sentence_index - b.sentence_index) > window:
                    continue
                label = test(a, b)
                if not label:
                    continue
                pairs.append((a, b, label))
                if len(pairs) >= max_pairs:
                    return PairScan(pairs, comparisons, True, buckets_sampled)
    return PairScan(pairs, comparisons, False, buckets_sampled)


def negation_conflict(a: Proposition, b: Proposition) -> str | None:
    return "negation" if a.negated != b.negated else None


def attribute_conflict(a: Proposition, b: Proposition) -> str | None:
    """A same subject+predicate pair asserting two incompatible values.

    Skips anything the negation scan already owns (a plain polarity flip is
    not also an attribute conflict) and anything without two comparable
    object values. Numeric objects are compared by value; a pair of
    four-digit numbers that both look like years is labelled ``temporal``
    rather than ``numeric`` (or via an explicit ``DATE`` entity label, when
    ``ner`` happens to be enabled) because "built in 1990" against "built in
    2004" is a date disagreement, not an arithmetic one, even though nothing
    here actually orders or parses the dates -- it only notices they differ.
    """

    if a.negated or b.negated:
        return None
    if a.object_key is None or b.object_key is None:
        return None
    if a.object_is_numeric and b.object_is_numeric:
        if a.object_number is None or b.object_number is None or a.object_number == b.object_number:
            return None
        is_temporal = (_looks_like_year(a.object_text) and _looks_like_year(b.object_text)) or (
            "DATE" in a.entity_labels and "DATE" in b.entity_labels)
        return "temporal" if is_temporal else "numeric"
    if a.object_key == b.object_key:
        return None
    return "property"


# ----------------------------------------------------------- WordNet relations
#
# nltk's wordnet corpus data is now downloaded and cached in this environment
# (it previously was not -- see logic_suite's module docstring). Importing
# ``nltk`` itself never fails, so availability is only known once the corpus
# data is actually touched; ``load_wordnet`` does that once, eagerly, and
# caches whichever answer it gets so every later call is free and every
# metric that wants WordNet sees the same "available"/"unavailable" verdict.

_WORDNET_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_wordnet_cache() -> None:
    _WORDNET_CACHE.clear()


on_reset(_reset_wordnet_cache)


def load_wordnet() -> tuple[Any, str | None]:
    """``(wordnet_module, None)`` or ``(None, reason)``.  Never raises."""

    if "wn" in _WORDNET_CACHE:
        return _WORDNET_CACHE["wn"]
    module, reason = require("nltk")
    if module is None:
        _WORDNET_CACHE["wn"] = (None, reason)
        return _WORDNET_CACHE["wn"]
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")  # forces the corpus-data LookupError now, not on first real use
        outcome: tuple[Any, str | None] = (wn, None)
    except LookupError as exc:
        outcome = (None, f"nltk wordnet corpus data unavailable ({exc}); run "
                         f"python -c \"import nltk; nltk.download('wordnet')\"")
    except Exception as exc:  # pragma: no cover - unexpected nltk failure
        outcome = (None, f"nltk wordnet unavailable ({type(exc).__name__}: {exc})")
    _WORDNET_CACHE["wn"] = outcome
    return outcome


def _object_lemma(object_key: str | None) -> str | None:
    """The bare lemma inside an ``object_key``, or ``None`` for a proper noun/entity key.

    Only a plain ``lemma:...`` key (a common noun or adjectival complement) is
    something WordNet's antonym/hypernym graph can meaningfully be asked
    about; a name has no synset.
    """

    if object_key and object_key.startswith("lemma:"):
        return object_key.split(":", 1)[1]
    return None


def _is_antonym_of(wn: Any, lemma_a: str, lemma_b: str) -> bool:
    for pos in (wn.ADJ, wn.ADJ_SAT, wn.VERB, wn.NOUN, wn.ADV):
        for synset in wn.synsets(lemma_a, pos=pos)[:4]:
            for lemma_obj in synset.lemmas():
                if lemma_obj.name().lower() != lemma_a:
                    continue
                for antonym in lemma_obj.antonyms():
                    if antonym.name().lower() == lemma_b:
                        return True
    return False


def wordnet_antonym_conflict(a: Proposition, b: Proposition) -> str | None:
    """Same subject+predicate pair whose objects are direct WordNet antonyms.

    Independent of :func:`attribute_conflict`: that test only notices the two
    object readings differ ("open" vs. "shut" AND "open" vs. "ajar" both
    read as ``property``); this one asks WordNet whether they are lexical
    opposites specifically, a narrower and independently-sourced signal.
    Every sense of each lemma is checked -- WordNet is not sense-
    disambiguated against this sentence's context -- so an uncommon sense can
    still produce a false positive; the finding built from this warns about
    exactly that, in its own words, not just here.
    """

    if a.negated or b.negated:
        return None
    lemma_a, lemma_b = _object_lemma(a.object_key), _object_lemma(b.object_key)
    if not lemma_a or not lemma_b or lemma_a == lemma_b:
        return None
    wn, reason = load_wordnet()
    if wn is None:
        return None
    if _is_antonym_of(wn, lemma_a, lemma_b) or _is_antonym_of(wn, lemma_b, lemma_a):
        return "antonym"
    return None


def wordnet_hypernym_related(a: Proposition, b: Proposition) -> bool:
    """True when ``a`` and ``b``'s object lemmas sit in a WordNet hypernym/hyponym relation.

    Meant as a cross-check on :func:`attribute_conflict`'s ``property``
    subtype, never a change to it: "the animal was a poodle" and "the animal
    was a dog" differ in object text but are not a contradiction, because a
    poodle is a dog. This function only answers the question; the caller
    decides what, if anything, to do with a ``True`` -- ``attribute_conflict``
    itself is left exactly as it was, so its candidate count stays stable and
    comparable across runs regardless of whether WordNet is installed.
    """

    lemma_a, lemma_b = _object_lemma(a.object_key), _object_lemma(b.object_key)
    if not lemma_a or not lemma_b or lemma_a == lemma_b:
        return False
    wn, reason = load_wordnet()
    if wn is None:
        return False
    synsets_a = wn.synsets(lemma_a, pos=wn.NOUN)[:3]
    synsets_b = wn.synsets(lemma_b, pos=wn.NOUN)[:3]
    for synset_a in synsets_a:
        ancestors_a = {node.name() for node in synset_a.closure(lambda s: s.hypernyms())}
        for synset_b in synsets_b:
            if synset_b.name() in ancestors_a:
                return True
            ancestors_b = {node.name() for node in synset_b.closure(lambda s: s.hypernyms())}
            if synset_a.name() in ancestors_b:
                return True
    return False


# --------------------------------------------------------------- coreference
#
# fastcoref is now installed and, through
# ``textgrader.optional.shim_fastcoref_transformers``, loads and predicts
# cleanly against the installed transformers release -- verified by exercising
# the real model, not assumed (see tests/test_logic_suite.py's
# ``test_coreference_resolution_against_a_real_fastcoref_model``). It stays
# optional and off by default anyway, because loading it is expensive (another
# neural model on top of the spaCy parse this module already pays for) and its
# resolution is only sampled over a document's first ``coreference_max_chars``
# characters, not a claim that it is unusable. ``_load_coref_model``'s own
# try/except still turns a genuine load or runtime failure -- a missing
# package, a future incompatible transformers release -- into an actionable
# "unavailable" reason rather than a crash.

_PRONOUN_WORDS = frozenset({
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "this", "that", "these", "those",
})

_COREF_MODEL_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_coref_cache() -> None:
    _COREF_MODEL_CACHE.clear()


on_reset(_reset_coref_cache)


def _load_coref_model() -> tuple[Any, str | None]:
    if "model" in _COREF_MODEL_CACHE:
        return _COREF_MODEL_CACHE["model"]
    module, reason = require("fastcoref")
    if module is None:
        _COREF_MODEL_CACHE["model"] = (None, reason)
        return _COREF_MODEL_CACHE["model"]
    try:
        shim_fastcoref_transformers()
        model = module.FCoref(device="cpu", enable_progress_bar=False)
        outcome: tuple[Any, str | None] = (model, None)
    except Exception as exc:  # pragma: no cover - model download/runtime/version failure
        outcome = (None, f"fastcoref model unavailable ({type(exc).__name__}: {exc}); "
                         f"install fastcoref and a compatible transformers, or see "
                         f"textgrader.optional.shim_fastcoref_transformers, which covers "
                         f"the tied-weight skew against transformers 5.x")
    _COREF_MODEL_CACHE["model"] = outcome
    return outcome


def _coref_representative(nlp: Any, text: str) -> tuple[str, str] | None:
    """``(bucket_key, display_text)`` for a cluster's antecedent mention text.

    Keyed with :func:`_normalize_key` on the mention's own head token -- the
    same function a directly-mentioned subject is keyed with -- so a
    coreference-resolved "she" lands in the *same* subject+predicate bucket as
    a sentence that names the character outright, rather than a bucket only
    pronouns ever reach.

    This reparses the short mention text through the shared spaCy pipeline
    rather than lower-casing its last word directly, which is worth the extra
    (very small) parse: a plural or inflected mention keeps its surface form
    ("the men", "the children") while a directly-named subject is keyed off
    spaCy's lemma ("man", "child"). Found wrong by exercising this against a
    real fastcoref model, not assumed -- "The men were tired. They were not
    tired at all" resolved the pronoun (``resolved_count`` was 1) but the
    resolved key was ``lemma:men`` against the directly-extracted sentence's
    ``lemma:man``, so the two never shared a bucket and coreference bought
    nothing on exactly the case it exists for. Tagging a two-or-three word
    phrase in isolation, with no surrounding sentence, is itself an
    approximation -- POS-tagging a bare noun phrase is easier than a full
    sentence and rarely wrong, but not guaranteed to match how the same words
    would have been tagged in context.
    """

    stripped = text.strip()
    if not stripped:
        return None
    tokens = [token for token in nlp(stripped) if not token.is_space and not token.is_punct]
    if not tokens:
        return None
    return _normalize_key(tokens[-1]), stripped


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


@dataclass(frozen=True)
class CorefResolution:
    """What :func:`resolve_coreference` did, whether or not it could resolve anything."""

    propositions: list[Proposition]
    resolved_count: int
    available: bool
    reason: str | None
    chars_used: int
    truncated: bool
    model_name: str


def _resolve_coreference(analysis: DocumentAnalysis, propositions: list[Proposition],
                         max_chars: int) -> CorefResolution:
    model_name = "biu-nlp/f-coref"
    eligible = [p for p in propositions if p.subject_is_pronoun and p.subject_char_span
               and p.subject_char_span[1] <= max_chars]
    if not eligible:
        return CorefResolution(propositions, 0, False,
                               "no pronoun-subject proposition within coreference_max_chars",
                               0, False, model_name)
    model, reason = _load_coref_model()
    if model is None:
        return CorefResolution(propositions, 0, False, reason, 0, False, model_name)

    text = analysis.text[:max_chars]
    truncated = len(analysis.text) > max_chars
    try:
        result = model.predict(texts=text)
        clusters = result.get_clusters(as_strings=False)
    except Exception as exc:  # pragma: no cover - runtime failure
        return CorefResolution(propositions, 0, False,
                               f"fastcoref prediction failed ({type(exc).__name__}: {exc})",
                               len(text), truncated, model_name)

    # For each cluster, the longest mention that is not itself a bare pronoun
    # becomes every pronoun mention's resolved antecedent. A cluster with no
    # such mention (an isolated "he"/"she" fastcoref could not anchor to a
    # name or noun phrase) resolves nothing, on purpose.
    representatives: list[tuple[tuple[int, int], str, str]] = []
    for cluster in clusters:
        best: tuple[int, int, str] | None = None
        for start, end in cluster:
            mention = text[start:end]
            if mention.strip().lower() in _PRONOUN_WORDS:
                continue
            if best is None or (end - start) > (best[1] - best[0]):
                best = (start, end, mention)
        if best is None:
            continue
        keyed = _coref_representative(analysis.nlp, best[2])
        if keyed is None:
            continue
        key, display = keyed
        for span in cluster:
            representatives.append((span, key, display))

    resolved_count = 0
    out: list[Proposition] = []
    for prop in propositions:
        match = None
        if prop.subject_is_pronoun and prop.subject_char_span and prop.subject_char_span[1] <= max_chars:
            for span, key, display in representatives:
                if _spans_overlap(prop.subject_char_span, span):
                    match = (key, display)
                    break
        if match is None:
            out.append(prop)
            continue
        key, display = match
        out.append(replace(prop, subject_key=key, subject_text=display, subject_resolved_via_coref=True))
        resolved_count += 1
    return CorefResolution(out, resolved_count, True, None, len(text), truncated, model_name)


def resolve_coreference(analysis: DocumentAnalysis, extraction: "Extraction",
                        max_chars: int) -> CorefResolution:
    """Cached: resolve pronoun subjects to a named antecedent via fastcoref, if available.

    Off by default and deliberately bounded to the first ``max_chars`` of the
    document -- coreference over a whole novel is neither validated nor
    affordable in this pass (see the module docstring). A pronoun proposition
    whose span falls after that cut, or that fastcoref places in a cluster
    with no non-pronoun mention, is left exactly as :func:`extract` produced
    it: excluded from every cross-sentence bucket, never guessed at.
    """

    return analysis.memo(f"logic_coref:{int(max_chars)}",
                         lambda: _resolve_coreference(analysis, extraction.propositions, int(max_chars)))


__all__ = ["Proposition", "Extraction", "PairScan", "CorefResolution", "extract", "bucketed_pairs",
          "negation_conflict", "attribute_conflict", "load_wordnet", "wordnet_antonym_conflict",
          "wordnet_hypernym_related", "resolve_coreference"]
