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

**No coreference.** A pronoun subject ("he", "she", "it", "they") cannot be
linked to the noun phrase it refers to without a coreference resolver, and
none is available in this environment (neuralcoref and its successors need
either an old spaCy pipeline or a model download this environment cannot
fetch). Rather than guess, every pronoun-subject proposition gets an empty
``subject_key`` and is excluded from every cross-sentence bucket this module
builds. That is a real loss -- most contradiction pairs in ordinary prose are
exactly "Alice was tired... She wasn't, though" -- and it is recorded, not
hidden: see the ``Deferred`` note in :mod:`textgrader.metrics.logic_suite`.

**No named-entity recognition required.** The shared spaCy pipeline disables
``ner`` by default (:class:`textgrader.document.NlpSettings`) because it
roughly doubles parse time, and only one existing metric (``pov_entity_ratio``)
asks for it. Rather than add a second metric family with an unusual NLP
requirement, subject/object identity here is keyed off proper-noun tokens
(``PROPN``, from part-of-speech tagging, which the default pipeline always
runs) with a fall-back to the lemma of common nouns. If ``ner`` happens to be
enabled, the entity label is folded into the key for a sharper match and is
also kept on the ``Proposition`` for evidence; nothing here requires it.

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
from dataclasses import dataclass
from typing import Any, Callable

from .document import DocumentAnalysis
from .optional import require

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
                 channel: str, entity_labels: tuple[str, ...]) -> "Proposition | None":
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
        entity_labels=entity_labels)


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
                                start_char=start_char, channel=channel, entity_labels=entity_labels)
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
    """

    buckets: dict[tuple[str, str], list[Proposition]] = defaultdict(list)
    for prop in propositions:
        if prop.subject_key:
            buckets[(prop.subject_key, prop.predicate_lemma)].append(prop)

    pairs: list[tuple[Proposition, Proposition, str]] = []
    comparisons = 0
    buckets_sampled = False
    window = max(0, int(window_sentences))
    for items in buckets.values():
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


__all__ = ["Proposition", "Extraction", "PairScan", "extract", "bucketed_pairs",
          "negation_conflict", "attribute_conflict"]
