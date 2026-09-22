"""Logic, consistency, entailment and argument-structure sensors -- the
honest subset.

The spec this module implements from (``docs/experimental-tasks/03-...``)
is written against a toolchain this environment does not have: no NLI
cross-encoder, no OpenIE service, no argument-mining model, no GPU, no
network access to fetch one. Rather than stub those behind an
``unavailable`` branch nobody ran, this module ships only what a
dependency-free pass and the installed spaCy pipeline can genuinely compute,
and names every one of them for what it actually measures. See ``Deferred``
at the end of this docstring for what was left out and why.

Nothing here checks whether a claim is *true*. Two sentences that share a
subject and a verb but disagree about the object are a **contradiction
candidate** -- worth a human's attention, not a verdict. A "therefore" that
connects two sentences with no shared vocabulary is a **low lexical-overlap
premise/conclusion pair** -- it might still be a perfectly good inference
dressed in different words, or it might be a non sequitur; this module
cannot tell you which, only that the surface signal a reader would lean on is
absent. Every finding below says this in its name or its docstring, not just
once here.

Four independently switchable measurement groups, under ``features`` in this
suite's config block:

``negation_and_quantifiers`` (stdlib, cost ``fast``)
    Surface negation density and universal/existential ("all", "never", "no
    one") quantifier density -- the raw vocabulary a contradiction or an
    unfalsifiable claim is usually built from.

``connective_relations`` (stdlib, cost ``fast``)
    For "therefore/thus/hence", "however/but/nevertheless", "because/since"
    and "if/unless", the lexical overlap between the clause the connective
    introduces and the clause it follows, plus how often the two disagree on
    negation, plus a structural completeness check for conditionals, plus a
    connective-chain-length proxy for argument structure built from nothing
    but where these markers sit relative to each other.

``propositions`` (spaCy dependency parse, cost ``parse``)
    Shallow subject-predicate-object triples from :mod:`textgrader.propositions`,
    bucketed by shared subject and predicate to find negation-flip and
    entity/property/numeric/temporal conflict *candidates*, a paragraph-scoped
    version of the same scan, an exact-repeated-proposition rate, and a proxy
    for claims introduced about a brand-new named entity with no connective
    linking them to what came before.

``modal_argument_position`` (stdlib, cost ``fast``)
    Whether hedges and modals cluster around the sentences that carry an
    argumentative connective, compared with the rest of the text.

Every scalar this module reports is a **candidate rate or a lexical-overlap
score**, not a truth value, and every one records the settings it was
computed under (``window_sentences``, ``max_pairs``, the spaCy pipeline
name/version) in its ``distribution``, because two runs with different caps
or a different spaCy model are not the same measurement.

Deferred
--------

Left out entirely, rather than shipped as an ``unavailable`` branch this
environment could never exercise:

* **Pairwise NLI (entailment/neutral/contradiction probabilities).** Needs a
  cross-encoder NLI model (``sentence-transformers`` CrossEncoder or
  ``transformers``); neither package is installed here, there is no GPU, and
  downloading one is not something a metric module should attempt silently.
  This is the single biggest gap against the spec: "Adjacent-sentence
  entailment probability distribution", "Adjacent-sentence contradiction
  probability distribution", and the connective *entailment/support scores*
  the spec asks for are all, properly, NLI's job. The lexical-overlap scores
  in ``connective_relations`` are a much weaker proxy for the same intuition
  and say so in their own docstrings.
* **OpenIE / Stanford CoreNLP / AllenNLP SRL.** No such service or legacy
  model is installed or reachable. ``textgrader/propositions.py`` extracts a
  dependency-parse proxy instead (documented there), which is narrower:
  single subject, single object, no semantic roles, no nested clauses.
* **Coreference resolution.** No coreference resolver is installed. Every
  pronoun-subject proposition is excluded from cross-sentence matching
  rather than guessed at, which means most of the ordinary "Alice was
  tired... She wasn't, though" contradictions in real prose are invisible to
  this suite. This is a real, significant recall loss, not a rounding error.
* **WordNet / ConceptNet / VerbNet / FrameNet / PropBank antonymy and
  commonsense checks.** ``nltk`` is installed but its ``wordnet`` corpus data
  is not present in this environment and cannot be downloaded here
  (confirmed: ``nltk.corpus.wordnet`` raises ``LookupError`` on import); no
  ConceptNet/ VerbNet/FrameNet/PropBank wrapper is installed either.
  Shipping an antonym-based contradiction check without ever having run it
  against real WordNet data is exactly the untested-behind-``unavailable``
  trap the task brief warns against, so it is left out rather than guessed.
* **Argument mining (claim/premise/support/attack extraction).** No
  argument-mining model or toolkit is installed. ``connective_chain_length``
  is offered as a much narrower proxy -- built only from where "therefore"
  and "because" sit relative to each other -- and is named and documented as
  exactly that, never as claim/premise/support/attack labels from a model.
* **Temporal-order reasoning.** No date parser (``dateparser``/``dateutil``)
  is added; ``entity_attribute_conflict_candidates`` can only notice that two
  explicit numbers/years attached to the same subject+predicate differ, never
  which one comes first or whether the difference is even meaningful (a
  birth year and a death year for the same person are supposed to differ).
* **Semantic-role pattern consistency / argument omission rates.** These need
  real SRL (PropBank-style ARG0/ARG1/ARGM labels), which needs AllenNLP or an
  equivalent model this environment does not have; the shallow dependency
  triples here are not semantic roles and are not offered as a substitute.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping

from .. import text as textlib
from ..document import DocumentAnalysis
from .. import propositions as prop_lib
from ..stats import run_lengths, summarize
from .common import PARSE, finding, option, rate, unavailable
from .discourse_hedges import HEDGES, MODALS
from .semantic_adjacent import STOPWORDS

FAMILY = "discourse"
# The suite's most expensive group needs the shared spaCy parse; the three
# stdlib groups cost nothing extra once it is paid, and pay nothing at all
# when propositions is switched off (COST/REQUIRES only tell the runner what
# to warn about before it runs, not what actually executes).
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy",)
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

# ------------------------------------------------------------- vocabularies

NEGATION_WORDS = frozenset({
    "not", "never", "no", "none", "nobody", "nothing", "nowhere", "neither",
    "nor", "without",
})

ABSOLUTE_WORDS = frozenset({
    "all", "every", "always", "never", "none", "nobody", "everybody",
    "everyone", "everything", "nothing", "entirely", "solely", "invariably",
    "universally",
})
ABSOLUTE_PHRASES: dict[str, tuple[str, ...]] = {
    "no one": ("no", "one"), "without exception": ("without", "exception"),
    "in every case": ("in", "every", "case"), "each and every": ("each", "and", "every"),
}

THEREFORE_PATTERN = re.compile(r"\b(?:therefore|thus|hence|consequently)\b", re.I)
CONTRAST_PATTERN = re.compile(
    r"\b(?:however|but|nevertheless|nonetheless|yet|although|though)\b", re.I)
BECAUSE_PATTERN = re.compile(r"\b(?:because|since)\b", re.I)
CONDITIONAL_PATTERN = re.compile(r"\b(?:if|unless|provided that)\b", re.I)
ANY_CONNECTIVE_PATTERN = re.compile(
    r"\b(?:therefore|thus|hence|consequently|however|but|nevertheless|nonetheless|"
    r"yet|although|though|because|since|if|unless|provided that)\b", re.I)

_METRIC_NAMES = {
    "discourse.logic_negation_rate": "Negation-cue rate",
    "discourse.logic_absolute_claim_rate": "Universal/existential quantifier ('absolute claim') rate",
    "discourse.logic_therefore_overlap": "Premise/conclusion lexical overlap around therefore/thus/hence",
    "discourse.logic_contrast_overlap": "Clause lexical overlap around however/but/nevertheless",
    "discourse.logic_because_overlap": "Clause lexical overlap around because/since",
    "discourse.logic_conditional_clause_shape_rate": "Conditional sentences with a recognizable antecedent+consequent shape",
    "discourse.logic_connective_chain_length": "Connective-linked sentence chain length",
    "discourse.logic_negation_flip_candidates": "Negation-flip contradiction candidates",
    "discourse.logic_paragraph_contradiction_rate": "Same-paragraph negation-flip candidates",
    "discourse.logic_entity_attribute_conflict_candidates": "Entity/property/numeric/temporal conflict candidates",
    "discourse.logic_repeated_assertion_rate": "Exact-repeated-proposition rate",
    "discourse.logic_new_entity_claim_rate": "Unlinked new-named-entity introduction rate",
    "discourse.logic_modal_density_near_connectives": "Modal/hedge density near argumentative connectives vs. elsewhere",
}


def _snippet(text: str, limit: int = 140) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit - 1].rstrip() + "…"


def _content_words(words: list[str]) -> set[str]:
    return {w.lower() for w in words if len(w) > 2 and w.lower() not in STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float | None:
    union = a | b
    return len(a & b) / len(union) if union else None


def _has_negation(words: list[str]) -> bool:
    return any(w.lower() in NEGATION_WORDS or w.lower().endswith("n't") for w in words)


def _by_length(table: Mapping[str, tuple[str, ...]]) -> dict[int, dict[tuple[str, ...], str]]:
    out: dict[int, dict[tuple[str, ...], str]] = {}
    for name, phrase in table.items():
        out.setdefault(len(phrase), {})[phrase] = name
    return out


def _settings(**extra: Any) -> dict[str, Any]:
    """Settings worth recording alongside a number so two runs can be compared."""

    return dict(extra)


# -------------------------------------------------------- negation/quantifiers

def _negation_and_quantifiers(analysis: DocumentAnalysis, max_evidence: int) -> list[dict[str, Any]]:
    tokens = analysis.tokens
    words_total = analysis.word_count

    neg_counts: Counter = Counter()
    for token in tokens:
        if token in NEGATION_WORDS or token.endswith("n't"):
            neg_counts[token] += 1
    neg_total = sum(neg_counts.values())

    abs_counts: Counter = Counter()
    for token in tokens:
        if token in ABSOLUTE_WORDS:
            abs_counts[token] += 1
    by_len = _by_length(ABSOLUTE_PHRASES)
    for index in range(len(tokens)):
        for length, table in by_len.items():
            if index + length > len(tokens):
                continue
            name = table.get(tuple(tokens[index:index + length]))
            if name:
                abs_counts[name] += 1
    abs_total = sum(abs_counts.values())

    no_words = "no words to measure"
    return [
        finding("discourse.logic_negation_rate", _METRIC_NAMES["discourse.logic_negation_rate"],
                rate(neg_total, words_total, 1000.0), "per 1,000 words", family=FAMILY,
                sample_size=words_total, min_sample=MIN_SAMPLE,
                evidence=[{"cue": name, "count": count} for name, count in neg_counts.most_common(max_evidence)],
                warning=None if words_total else no_words),
        finding("discourse.logic_absolute_claim_rate", _METRIC_NAMES["discourse.logic_absolute_claim_rate"],
                rate(abs_total, words_total, 1000.0), "per 1,000 words", family=FAMILY,
                sample_size=words_total, min_sample=MIN_SAMPLE,
                evidence=[{"cue": name, "count": count} for name, count in abs_counts.most_common(max_evidence)],
                warning=(None if words_total else no_words) or (
                    "shares vocabulary with discourse.booster_rate ('always', 'never', 'every'); this "
                    "rate isolates the quantifier reading as a candidate-premise signal, not a "
                    "rhetorical-intensity one" if abs_total else None)),
    ]


# ----------------------------------------------------------- connective relations

def _relation_scores(analysis: DocumentAnalysis, pattern: re.Pattern, min_words: int,
                     max_evidence: int) -> tuple[list[float], list[dict[str, Any]], int, int]:
    """``(overlap_values, evidence, negation_mismatch_count, occurrences)``.

    Only the first marker match in a sentence is scored (a second "but" in
    the same sentence is rare and not worth a second, overlapping clause
    split). When nothing precedes the marker in its own sentence, the
    previous sentence stands in as the premise/context clause; otherwise the
    text within the same sentence on each side of the marker is used, which
    is what makes this one scorer work for both "X. Therefore, Y." and
    "X, but Y" without treating them as different cases.
    """

    sentences = analysis.sentences
    values: list[float] = []
    evidence: list[dict[str, Any]] = []
    mismatches = 0
    occurrences = 0
    for index, sentence in enumerate(sentences):
        match = pattern.search(sentence)
        if not match:
            continue
        occurrences += 1
        left_text = sentence[:match.start()]
        right_text = sentence[match.end():]
        left_words = textlib.words(left_text)
        cross_sentence = not left_words
        if cross_sentence:
            if index == 0:
                continue
            left_text = sentences[index - 1]
            left_words = textlib.words(left_text)
        right_words = textlib.words(right_text)
        left_set, right_set = _content_words(left_words), _content_words(right_words)
        if len(left_set) < min_words or len(right_set) < min_words:
            continue
        overlap = _jaccard(left_set, right_set)
        if overlap is None:
            continue
        values.append(overlap)
        left_neg, right_neg = _has_negation(left_words), _has_negation(right_words)
        mismatch = left_neg != right_neg
        mismatches += int(mismatch)
        if len(evidence) < max_evidence:
            evidence.append({
                "sentence_index": index, "marker": match.group(0).lower(),
                "premise_from_previous_sentence": cross_sentence,
                "overlap": round(overlap, 3), "negation_mismatch": mismatch,
                "premise": _snippet(left_text), "conclusion": _snippet(right_text),
            })
    return values, evidence, mismatches, occurrences


def _relation_finding(metric_id: str, values: list[float], evidence: list[dict[str, Any]],
                      mismatches: int, occurrences: int, min_words: int) -> dict[str, Any]:
    if not values:
        warning = ("no qualifying occurrences: either the marker never appears, or every "
                   "occurrence fell below connective_min_words content words on one side")
        return finding(metric_id, _METRIC_NAMES[metric_id], None, "jaccard overlap",
                       family=FAMILY, sample_size=occurrences, min_sample=5, warning=warning)
    summary = summarize(values)
    return finding(
        metric_id, _METRIC_NAMES[metric_id], summary.get("median"), "jaccard overlap (0-1)",
        family=FAMILY, sample_size=summary.get("count"), min_sample=5,
        distribution={**summary, "occurrences_seen": occurrences,
                     "negation_mismatch_count": mismatches,
                     "negation_mismatch_rate_percent": rate(mismatches, len(values), 100.0),
                     "settings": _settings(connective_min_words=min_words)},
        evidence=evidence,
        warning="lexical overlap is a surface proxy for support/contrast/cause; it is not "
                "entailment and was not scored by any NLI model (see module Deferred notes)")


def _conditional_shape(analysis: DocumentAnalysis, max_evidence: int) -> dict[str, Any]:
    sentences = analysis.sentences
    occurrences = 0
    complete = 0
    incomplete_evidence: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        match = CONDITIONAL_PATTERN.search(sentence)
        if not match:
            continue
        occurrences += 1
        left_words = textlib.words(sentence[:match.start()])
        right_words = textlib.words(sentence[match.end():])
        shaped = bool(left_words) or len(right_words) >= 3
        if shaped:
            complete += 1
        elif len(incomplete_evidence) < max_evidence:
            incomplete_evidence.append({"sentence_index": index, "text": _snippet(sentence)})
    metric_id = "discourse.logic_conditional_clause_shape_rate"
    if not occurrences:
        return finding(metric_id, _METRIC_NAMES[metric_id], None, "percent", family=FAMILY,
                       sample_size=0, min_sample=5,
                       warning="no if/unless/provided-that occurrences found")
    return finding(
        metric_id, _METRIC_NAMES[metric_id], rate(complete, occurrences, 100.0), "percent",
        family=FAMILY, sample_size=occurrences, min_sample=5,
        distribution={"complete": complete, "occurrences": occurrences},
        evidence=incomplete_evidence,
        warning="structural proxy only: a sentence counts as 'shaped' whenever material "
                "appears on both sides of the marker, regardless of whether that material "
                "is actually a coherent antecedent and consequent")


def _connective_chain_length(analysis: DocumentAnalysis, max_evidence: int) -> dict[str, Any]:
    sentences = analysis.sentences
    metric_id = "discourse.logic_connective_chain_length"
    if len(sentences) < 2:
        return finding(metric_id, _METRIC_NAMES[metric_id], None, "sentences", family=FAMILY,
                       sample_size=len(sentences), min_sample=5,
                       warning="fewer than two sentences to link")

    def leads_with(pattern: re.Pattern, sentence: str) -> bool:
        match = pattern.match(sentence.lstrip())
        return bool(match)

    labels = ["plain"]
    for sentence in sentences[1:]:
        if leads_with(THEREFORE_PATTERN, sentence) or leads_with(BECAUSE_PATTERN, sentence):
            labels.append("support")
        elif leads_with(CONTRAST_PATTERN, sentence):
            labels.append("contrast")
        else:
            labels.append("plain")

    runs = run_lengths(labels)
    support_runs = runs.get("support", [])
    chain_lengths = [length + 1 for length in support_runs]
    contrast_edges = sum(runs.get("contrast", []))

    if not chain_lengths:
        return finding(metric_id, _METRIC_NAMES[metric_id], None, "sentences", family=FAMILY,
                       sample_size=len(sentences), min_sample=5,
                       distribution={"support_edges": 0, "contrast_edges": contrast_edges},
                       warning="no sentence opened on a therefore/because-style connective; "
                               "no chain to measure")

    summary = summarize(chain_lengths)
    evidence: list[dict[str, Any]] = []

    # ``run_lengths`` only returns lengths, not where each run started, and
    # evidence needs the start to quote the chain; recomputed here directly
    # rather than changing the shared ``stats.run_lengths`` return shape that
    # other metrics already depend on.
    starts: list[tuple[int, int]] = []
    current_label, start = labels[0], 0
    for index in range(1, len(labels)):
        if labels[index] != current_label:
            if current_label == "support":
                starts.append((start - 1, index - start + 1))
            current_label, start = labels[index], index
    if current_label == "support":
        starts.append((start - 1, len(labels) - start + 1))
    for chain_start, length in sorted(starts, key=lambda item: -item[1])[:max_evidence]:
        evidence.append({
            "chain_length_sentences": length, "start_sentence_index": chain_start,
            "text": _snippet(" ".join(sentences[chain_start:chain_start + min(length, 4)])),
        })

    return finding(
        metric_id, _METRIC_NAMES[metric_id], summary.get("median"), "sentences", family=FAMILY,
        sample_size=summary.get("count"), min_sample=5,
        distribution={**summary, "support_edges": sum(support_runs), "contrast_edges": contrast_edges},
        evidence=evidence, sample_size_sensitive=True,
        warning="built only from which sentences open on a therefore/because- or however/but-style "
                "marker; this is not argument-mining claim/premise/support/attack extraction, only "
                "a connective-adjacency proxy for it (see module Deferred notes)")


def _connective_relations(analysis: DocumentAnalysis, min_words: int, max_evidence: int) -> list[dict[str, Any]]:
    out = []
    for metric_id, pattern in (
        ("discourse.logic_therefore_overlap", THEREFORE_PATTERN),
        ("discourse.logic_contrast_overlap", CONTRAST_PATTERN),
        ("discourse.logic_because_overlap", BECAUSE_PATTERN),
    ):
        values, evidence, mismatches, occurrences = _relation_scores(
            analysis, pattern, min_words, max_evidence)
        out.append(_relation_finding(metric_id, values, evidence, mismatches, occurrences, min_words))
    out.append(_conditional_shape(analysis, max_evidence))
    out.append(_connective_chain_length(analysis, max_evidence))
    return out


# --------------------------------------------------------------- propositions

def _proposition_findings(analysis: DocumentAnalysis, *, proposition_cap: int, window_sentences: int,
                          max_pairs: int, max_comparisons: int, max_evidence: int,
                          repeated_assertion_min_words: int) -> list[dict[str, Any]]:
    ids = ["discourse.logic_negation_flip_candidates", "discourse.logic_paragraph_contradiction_rate",
          "discourse.logic_entity_attribute_conflict_candidates", "discourse.logic_repeated_assertion_rate",
          "discourse.logic_new_entity_claim_rate"]
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], analysis.nlp_unavailable, family=FAMILY)
                for metric_id in ids]

    extraction = prop_lib.extract(analysis, proposition_cap)
    props = extraction.propositions
    settings = _settings(window_sentences=window_sentences, max_pairs=max_pairs,
                         proposition_cap=proposition_cap, spacy_model=extraction.spacy_model,
                         spacy_version=extraction.spacy_version, ner_available=extraction.ner_available)
    base_warning = (
        "candidates only: matched on shared surface subject and predicate, not confirmed "
        "coreference or verified meaning; pronoun-subject sentences are excluded entirely "
        "because no coreference resolver is available (see module Deferred notes)")
    if extraction.truncated:
        base_warning += f"; proposition extraction stopped at the proposition_cap of {proposition_cap:,}"

    if not props:
        no_data = "no sentence yielded a usable (subject, predicate) proposition"
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], no_data, family=FAMILY)
                for metric_id in ids]

    def pair_evidence(pairs, limit) -> list[dict[str, Any]]:
        rows = []
        for a, b, label in pairs[:limit]:
            rows.append({
                "type": label, "subject": a.subject_text, "predicate": a.predicate_lemma,
                "sentence_a": {"index": a.sentence_index, "offset": a.offset, "text": a.text,
                              "negated": a.negated, "object": a.object_text},
                "sentence_b": {"index": b.sentence_index, "offset": b.offset, "text": b.text,
                              "negated": b.negated, "object": b.object_text},
            })
        return rows

    negation_scan = prop_lib.bucketed_pairs(
        props, window_sentences=window_sentences, max_pairs=max_pairs,
        max_comparisons=max_comparisons, test=prop_lib.negation_conflict)
    negation_settings = {**settings, "comparisons_examined": negation_scan.comparisons,
                        "pairs_capped": negation_scan.pairs_capped,
                        "buckets_sampled": negation_scan.buckets_sampled}
    out = [finding(
        "discourse.logic_negation_flip_candidates", _METRIC_NAMES["discourse.logic_negation_flip_candidates"],
        rate(len(negation_scan.pairs), len(props), 1000.0), "candidates per 1,000 propositions",
        family=FAMILY, sample_size=len(props), min_sample=100, sample_size_sensitive=True,
        distribution={"candidate_count": len(negation_scan.pairs), "settings": negation_settings},
        evidence=pair_evidence(negation_scan.pairs, max_evidence), warning=base_warning)]

    same_paragraph = [p for p in negation_scan.pairs if p[0].paragraph_index == p[1].paragraph_index]
    paragraph_total = analysis.paragraph_count
    out.append(finding(
        "discourse.logic_paragraph_contradiction_rate",
        _METRIC_NAMES["discourse.logic_paragraph_contradiction_rate"],
        rate(len(same_paragraph), paragraph_total, 100.0) if paragraph_total else None,
        "candidates per 100 paragraphs", family=FAMILY, sample_size=paragraph_total,
        min_sample=20, sample_size_sensitive=True,
        distribution={"candidate_count": len(same_paragraph), "settings": settings},
        evidence=pair_evidence(same_paragraph, max_evidence),
        warning=base_warning if paragraph_total else "no paragraphs to measure"))

    attribute_scan = prop_lib.bucketed_pairs(
        props, window_sentences=window_sentences, max_pairs=max_pairs,
        max_comparisons=max_comparisons, test=prop_lib.attribute_conflict)
    subtype_counts = Counter(label for _, _, label in attribute_scan.pairs)
    attribute_settings = {**settings, "comparisons_examined": attribute_scan.comparisons,
                         "pairs_capped": attribute_scan.pairs_capped,
                         "buckets_sampled": attribute_scan.buckets_sampled}
    out.append(finding(
        "discourse.logic_entity_attribute_conflict_candidates",
        _METRIC_NAMES["discourse.logic_entity_attribute_conflict_candidates"],
        rate(len(attribute_scan.pairs), len(props), 1000.0), "candidates per 1,000 propositions",
        family=FAMILY, sample_size=len(props), min_sample=100, sample_size_sensitive=True,
        distribution={"candidate_count": len(attribute_scan.pairs), "by_type": dict(subtype_counts),
                     "settings": attribute_settings},
        evidence=pair_evidence(attribute_scan.pairs, max_evidence),
        warning=base_warning + "; 'temporal' means two differing explicit years/dates were found, "
                "not that either was verified or that an order was established"))

    eligible = [p for p in props if p.subject_key and p.object_key
               and len(textlib.words(p.text)) >= repeated_assertion_min_words]
    groups: dict[tuple[str, str, str, bool], list[prop_lib.Proposition]] = defaultdict(list)
    for p in eligible:
        groups[(p.subject_key, p.predicate_lemma, p.object_key, p.negated)].append(p)
    duplicate_groups = [items for items in groups.values() if len(items) > 1]
    excess = sum(len(items) - 1 for items in duplicate_groups)
    dup_evidence = [
        {"subject": items[0].subject_text, "predicate": items[0].predicate_lemma,
        "object": items[0].object_text, "count": len(items),
        "occurrences": [{"sentence_index": p.sentence_index, "offset": p.offset, "text": p.text}
                        for p in items[:5]]}
        for items in sorted(duplicate_groups, key=lambda g: -len(g))[:max_evidence]]
    out.append(finding(
        "discourse.logic_repeated_assertion_rate", _METRIC_NAMES["discourse.logic_repeated_assertion_rate"],
        rate(excess, len(eligible), 1000.0) if eligible else None,
        "excess repeats per 1,000 eligible propositions", family=FAMILY, sample_size=len(eligible),
        min_sample=100, sample_size_sensitive=True,
        distribution={"excess_repeat_count": excess, "duplicate_group_count": len(duplicate_groups),
                     "settings": {**settings, "repeated_assertion_min_words": repeated_assertion_min_words}},
        evidence=dup_evidence,
        warning="exact structural duplication only (same subject, predicate and object reading); "
                "a paraphrase using different words is invisible here and would need a semantic-"
                "similarity model (see module Deferred notes)" if eligible else
                "no proposition had both a non-pronoun subject and an object"))

    out.append(_new_entity_claim_finding(analysis, max_evidence, extraction))
    return out


def _new_entity_claim_finding(analysis: DocumentAnalysis, max_evidence: int,
                              extraction: "prop_lib.Extraction") -> dict[str, Any]:
    metric_id = "discourse.logic_new_entity_claim_rate"
    seen: set[str] = set()
    introductions = 0
    unlinked = 0
    evidence: list[dict[str, Any]] = []
    sentence_index = -1
    # Plain ``spacy_docs()`` rather than ``spacy_sents_by_channel()``: this
    # scan has no use for the dialogue/narration split, and the channel
    # classifier's per-sentence scan over every quotation span in the
    # document is not a cost worth paying twice (propositions.extract already
    # pays it once, for the metrics that do need the channel).
    for offset, doc in analysis.spacy_docs():
        for sent in doc.sents:
            sentence_index += 1
            run: list[Any] = []
            new_here = False
            for token in list(sent) + [None]:
                is_propn = token is not None and token.pos_ == "PROPN"
                if is_propn:
                    run.append(token)
                    continue
                if run:
                    phrase = " ".join(t.text for t in run).lower()
                    if phrase not in seen:
                        seen.add(phrase)
                        new_here = True
                    if len(run) > 1:
                        # A leading capitalized common word ("Old Mara") is
                        # often mistagged PROPN by this small model and folds
                        # into the run; registering the last token alone too
                        # means a later bare "Mara" is recognized as the
                        # same, already-seen name rather than flagged as a
                        # second, brand-new entity.
                        seen.add(run[-1].text.lower())
                    run = []
            if not new_here:
                continue
            introductions += 1
            linked = bool(ANY_CONNECTIVE_PATTERN.search(sent.text)) or sentence_index == 0
            if not linked:
                unlinked += 1
                if len(evidence) < max_evidence:
                    evidence.append({"sentence_index": sentence_index,
                                     "offset": offset + sent.start_char,
                                     "text": _snippet(sent.text)})
    if not introductions:
        return unavailable(metric_id, _METRIC_NAMES[metric_id],
                           "no proper-noun phrase (PROPN) was found anywhere in the document",
                           family=FAMILY)
    return finding(
        metric_id, _METRIC_NAMES[metric_id], rate(unlinked, introductions, 100.0), "percent",
        family=FAMILY, sample_size=introductions, min_sample=10, sample_size_sensitive=True,
        distribution={"introductions": introductions, "unlinked": unlinked,
                     "settings": {"spacy_model": extraction.spacy_model}},
        evidence=evidence,
        warning="'new entity' means a proper-noun phrase (POS tag PROPN) not seen earlier in the "
                "document, not a resolved named-entity identity; 'linked' means the sentence "
                "contains any of this module's own connective markers, not that the claim is "
                "actually supported. A document's opening sentence is never counted as unlinked. "
                "This is a proxy for 'a claim about someone/something new arrives with no visible "
                "logical handle', nothing stronger")


# ----------------------------------------------------------- modal/hedge position

_STANCE_TABLE = _by_length({**HEDGES, **MODALS})


def _stance_hits(words: list[str]) -> int:
    tokens = [w.lower() for w in words]
    hits = 0
    for index in range(len(tokens)):
        for length, table in _STANCE_TABLE.items():
            if index + length > len(tokens):
                continue
            if tuple(tokens[index:index + length]) in table:
                hits += 1
    return hits


def _modal_density_near_connectives(analysis: DocumentAnalysis) -> dict[str, Any]:
    metric_id = "discourse.logic_modal_density_near_connectives"
    arg_words = arg_hits = rest_words = rest_hits = 0
    for sentence in analysis.sentences:
        words = textlib.words(sentence)
        hits = _stance_hits(words)
        if ANY_CONNECTIVE_PATTERN.search(sentence):
            arg_words += len(words)
            arg_hits += hits
        else:
            rest_words += len(words)
            rest_hits += hits
    arg_rate = rate(arg_hits, arg_words, 1000.0)
    rest_rate = rate(rest_hits, rest_words, 1000.0)
    if arg_words < 200 or rest_words < 200:
        return finding(metric_id, _METRIC_NAMES[metric_id], None, "ratio", family=FAMILY,
                       sample_size=arg_words, min_sample=200,
                       distribution={"argumentative_sentence_words": arg_words,
                                    "other_words": rest_words},
                       warning="insufficient_data: fewer than 200 words in the argumentative-"
                               "connective sentences, the baseline sentences, or both")
    ratio = (arg_rate / rest_rate) if rest_rate else None
    return finding(
        metric_id, _METRIC_NAMES[metric_id], ratio, "ratio (>1 = denser near connectives)",
        family=FAMILY, sample_size=arg_words, min_sample=200,
        distribution={"argumentative_rate_per_1000_words": arg_rate,
                     "other_rate_per_1000_words": rest_rate,
                     "argumentative_sentence_words": arg_words, "other_words": rest_words},
        warning=None if rest_rate else "no hedge or modal found outside argumentative-connective "
                                       "sentences; ratio is undefined")


# --------------------------------------------------------------------- measure

FEATURE_METRICS: dict[str, tuple[str, ...]] = {
    "negation_and_quantifiers": (
        "discourse.logic_negation_rate", "discourse.logic_absolute_claim_rate"),
    "connective_relations": (
        "discourse.logic_therefore_overlap", "discourse.logic_contrast_overlap",
        "discourse.logic_because_overlap", "discourse.logic_conditional_clause_shape_rate",
        "discourse.logic_connective_chain_length"),
    "propositions": (
        "discourse.logic_negation_flip_candidates", "discourse.logic_paragraph_contradiction_rate",
        "discourse.logic_entity_attribute_conflict_candidates", "discourse.logic_repeated_assertion_rate",
        "discourse.logic_new_entity_claim_rate"),
    "modal_argument_position": ("discourse.logic_modal_density_near_connectives",),
}


def _disabled(feature: str) -> list[dict[str, Any]]:
    return [unavailable(metric_id, _METRIC_NAMES[metric_id],
                        f"disabled by config (features.{feature}=false)", family=FAMILY)
            for metric_id in FEATURE_METRICS[feature]]


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    features = option(config, "features", {})
    window_sentences = int(option(config, "window_sentences", 6))
    max_pairs = int(option(config, "max_pairs", 200))
    max_comparisons = int(option(config, "max_comparisons", 50_000))
    max_evidence = int(option(config, "max_evidence", 20))
    proposition_cap = int(option(config, "proposition_cap", 20_000))
    connective_min_words = int(option(config, "connective_min_words", 4))
    repeated_assertion_min_words = int(option(config, "repeated_assertion_min_words", 5))

    def on(name: str) -> bool:
        value = features.get(name, True) if isinstance(features, Mapping) else True
        return value is not False

    out: list[dict[str, Any]] = []
    out.extend(_negation_and_quantifiers(analysis, max_evidence) if on("negation_and_quantifiers")
              else _disabled("negation_and_quantifiers"))
    out.extend(_connective_relations(analysis, connective_min_words, max_evidence)
              if on("connective_relations") else _disabled("connective_relations"))
    out.extend(_proposition_findings(
        analysis, proposition_cap=proposition_cap, window_sentences=window_sentences,
        max_pairs=max_pairs, max_comparisons=max_comparisons, max_evidence=max_evidence,
        repeated_assertion_min_words=repeated_assertion_min_words)
        if on("propositions") else _disabled("propositions"))
    out.extend([_modal_density_near_connectives(analysis)] if on("modal_argument_position")
              else _disabled("modal_argument_position"))
    return out
