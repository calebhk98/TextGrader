"""Richer syntactic complexity: T-unit/clause ratios, phrasal elaboration,
dependency topology, real constituency structure, and syntactic surprisal.

The existing ``syntax_*`` modules already cover mean dependency distance
(``syntax_dependency_distance.py``), real per-sentence tree depth
(``syntax_parse_depth.py``), four clause-type rates
(``syntax_clause_types.py``), coordination-vs-subordination
(``syntax_coordination.py``), finite-clause counts (``syntax_finite_clauses.py``)
and sentence-opening shapes (``syntax_openings.py``). This suite does not
recompute any of those; where a measure here is adjacent to one of them, the
finding's ``distribution`` names the existing metric id rather than silently
duplicating it.

Everything is switched on or off independently through ``features``:

``tunit_clause`` (on by default, spaCy only)
    An L2SCA-*style* ratio family -- mean length of sentence/T-unit/clause,
    clauses per T-unit, dependent clauses per clause/T-unit, coordinate
    phrases per clause/T-unit, complex nominals per clause/T-unit, verb
    phrases per T-unit, and a finite/nonfinite ratio.  **This is an
    approximation, not L2SCA itself.**  L2SCA (Lu 2010) and TAASSC (Kyle
    2016) identify T-units and clauses with Tregex patterns over a
    *constituency* parse. TAASSC is not a Python package (no PyPI
    distribution under ``taassc``/``TAASSC`` exists; it ships as a
    standalone tool historically paired with Stanford CoreNLP, which is out
    of scope here -- see the module-level "Stanford CoreNLP" note below).
    Every metric id in this feature carries an explicit ``l2sca`` infix and
    every finding's ``warning``/``distribution`` says the counts come from
    spaCy's *dependency* labels instead: a T-unit head is a predicate
    (``VERB``/``AUX``, excluding ``aux``/``auxpass``/``cop`` helpers) whose
    ``dep_`` is ``ROOT``/``parataxis``, or a ``conj`` chain that resolves up
    to one of those; every other predicate head is a dependent clause. A
    verbless fragment still counts as one (trivial) T-unit/clause, the
    convention automated T-unit counters use so the ratios stay defined.
    See :func:`_is_main_clause_head` for the exact, disclosed rule.

``phrasal_elaboration`` (on by default, spaCy only)
    Noun-phrase length and internal depth (from ``doc.noun_chunks``),
    pre-/post-modifier counts, postmodifier-type diversity, PP-attachment
    density, appositive rate and participial-modifier rate.

``dependency_topology`` (on by default, spaCy only)
    Branching factor, tree imbalance, head-direction, long-dependency rate,
    non-projective (crossing-arc) sentence rate, root POS entropy, subtree-size
    distribution, and dependency-label entropy/transition entropy.  Overlaps
    with ``syntax_parse_depth``/``syntax_dependency_distance`` are disclosed in
    each finding's ``distribution`` rather than recomputed.

``constituency`` (**off by default even when the suite is on**)
    Real constituency-tree measures -- tree depth, phrase-type entropy,
    production-rule entropy, distinct-subtree rate, and sentence-template
    top-share/diversity -- from `benepar <https://github.com/nikitakit/self-attentive-parser>`_,
    a real neural constituency parser, not an approximation. Verified in this
    environment: ``import benepar; benepar.__doc__`` reads
    ``"benepar: Berkeley Neural Parser"``; ``benepar.download('benepar_en3')``
    fetched a real 260MB pretrained model
    (``~/nltk_data/models/benepar_en3``), and parsing "The hypothesis that the
    committee, which had convened after the report was published, would
    reject the proposal was disproved." produced a real bracketed tree
    (``(S (NP (DT The) (NN hypothesis) (SBAR ...`` -- an ``SBAR``-nested
    relative clause inside the subject, exactly as expected). Loading a
    ``benepar_en3`` pipeline onto ``en_core_web_sm`` hit a real
    ``transformers``/``benepar`` version-skew bug: with ``transformers``
    5.17.0, benepar's retokenizer calls
    ``T5Tokenizer.build_inputs_with_special_tokens``, an attribute the
    rewritten (``TokenizersBackend``) tokenizer classes in ``transformers>=5``
    no longer define (``AttributeError: T5Tokenizer has no attribute
    build_inputs_with_special_tokens``). :func:`_shim_t5_retokenizer` restores
    exactly that one pre-5.x T5 method (append one EOS id; two EOS ids for a
    pair), the same one-attribute, defensive-``hasattr`` pattern
    :func:`textgrader.coherence._shim_transformers_tied_weights` already uses
    for an analogous fastcoref/transformers skew. With the shim, loading took
    5.6s and 1.24GB resident (measured with ``resource.getrusage``), and
    parsing averaged 1.5s/sentence on CPU across 20 sentences of varying
    length -- in the same ballpark as ``coherence_suite``'s ``isanlp_rst``
    feature (1.9-6.7s/sentence), so it is gated exactly the same way: off by
    default, a bounded deterministic sample (reusing
    :func:`textgrader.coherence.sample_rst_passages` with
    ``passage_sentences=1`` to draw individual sentences spread across the
    book), a wall-clock cap that never counts model-load time, and
    ``sample_size``/``min_sample`` set so a cap-truncated sample is
    ``insufficient_data`` rather than silently compared as complete --
    see :func:`_resolve_constituency`.

    ``stanza`` also ships a constituency parser and was the spec's other
    named option; ``benepar`` was chosen instead because it installs and
    integrates as an ordinary spaCy pipeline component sharing this
    project's existing spaCy-centric plumbing (:func:`_load_benepar` adds it
    to a normal ``spacy.load(...)`` pipeline), whereas stanza runs its own,
    separate pipeline and tokenization; since benepar was verified working
    end-to-end above, a second constituency backend was not built -- keeping
    scope bounded rather than shipping two half-verified ones. Independent
    dependency parses (SuPar) were not added for the same reason: this suite
    already reuses the shared spaCy dependency parse for everything that
    does not need constituency structure, and a second dependency parser
    would be a disagreement channel, not a new measurement -- worth
    revisiting, not blocking this pass.

``syntactic_surprisal`` (on by default; needs a corpus profile to produce a
    value, never trains on the document it scores)
    POS-bigram and dependency-label-bigram cross-entropy against a
    corpus-trained model, plus a per-sentence surprisal distribution with
    top/bottom evidence sentences. "Train syntactic n-gram models from the
    corpus, never from the test document alone" (the task spec) is enforced
    structurally: :func:`profile_vector` is the only place this suite ever
    builds a POS-bigram/dependency-bigram table, it is called once per book
    while *building* a corpus profile, and :func:`measure` only ever reads
    that pooled table back through ``profile`` -- there is no code path that
    fits a model from the current document. This suite's ``cost`` is
    ``"parse"``, so :func:`textgrader.corpus._metric_names` drops it from
    corpus profiling unless the profile is built with ``--parse-metrics``
    -- enabling ``syntax_complexity_suite`` in ``metrics`` is not enough on
    its own, the same two-part requirement ``coherence_suite``'s
    ``_requires_transition_corpus_delta`` note documents for its own
    ``profile_vector`` hook (that one also needs ``--model-metrics``; this
    one does not, since nothing here needs ``sentence_transformers``).
    Without a profile built with BOTH flags, every finding in this feature
    reports ``unavailable`` naming that requirement, exactly like
    ``coherence_suite``'s ``discourse.coherence_entity_grid_transition_corpus_delta``
    does with no corpus at all. Constituency production-rule cross-entropy
    (the spec's fourth surprisal item) is deferred: wiring it in would mean
    ``profile_vector`` runs the same expensive, sampled benepar pass as the
    ``constituency`` feature during ordinary corpus profiling once a user
    turns ``features.constituency`` on for profiling, which is a real cost
    this pass chooses not to default into place without being asked; the
    within-document production-rule entropy the ``constituency`` feature
    already reports is the substitute for now.

Stanford CoreNLP is not used anywhere in this module: it is a Java tool and
TextGrader is Python only, per the task spec's own note and this project's
cross-task rule 6.

Sentence boundaries and this suite's honesty about them
---------------------------------------------------------
Every per-sentence ratio here rides on :class:`textgrader.document.Segmenter`
(recently fixed: pySBD used to return an entire multi-paragraph quoted speech
as one "sentence", up to 1,293 words, which would have made every L2SCA-style
ratio in this module a near-meaningless outlier average). Every shape-based
finding here uses :func:`textgrader.metrics.common.shape`, whose headline
value is the *median*, with the full quantile curve in ``distribution`` --
the same defence :mod:`textgrader.stats` already gives every other per-
sentence metric in this codebase, and the reason
``test_shape_metrics_are_not_dominated_by_one_huge_sentence`` in the test
module asserts it directly, with one deliberately oversized sentence mixed
into otherwise-short ones.

No polarity judgement is made anywhere in this module: children's prose and
dense expository prose both have a "normal" range for these measures, in
opposite directions, and neither is scored as better. Every finding here is
``Polarity.NEUTRAL`` by the same default every other optional finding in this
codebase already gets.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, Mapping, Sequence

from .. import coherence as coh
from ..document import DocumentAnalysis
from ..optional import on_reset, require
from .common import PARSE, finding, option, rate, shape, summarize, unavailable

FAMILY = "syntax"
COST = PARSE
REQUIRES: tuple[str, ...] = ("spacy", "benepar")
MIN_SAMPLE = 30
UNIT_SENSITIVE = False

#: features.constituency is the ONLY thing standing between a book and a
#: benepar/transformers load: this suite's cost is "parse", so
#: MetricSpec.needs_model (which only checks for "sentence_transformers" in
#: REQUIRES) does not exclude it from corpus profiling the way needs_model
#: would -- exactly the same gating note logic_suite's transformers-backed
#: features already carry. Every other feature is spaCy-only, which the
#: whole suite already requires through its "parse" cost, so those default on.
DEFAULT_FEATURES: dict[str, bool] = {
    "tunit_clause": True,
    "phrasal_elaboration": True,
    "dependency_topology": True,
    "syntactic_surprisal": True,
    "constituency": False,
}


def _features(config: Mapping[str, Any] | None) -> dict[str, bool]:
    merged = dict(DEFAULT_FEATURES)
    merged.update(option(config, "features", {}) or {})
    return merged


# ---------------------------------------------------------- clause/T-unit approximation

#: Dependents of a predicate that are not themselves a separate clause head
#: (an auxiliary, passive auxiliary, or copula sitting on the real predicate).
#: Mirrors syntax_finite_clauses._HELPER_DEPS's reasoning; kept as an
#: independent copy so this module has no import-time coupling to that one's
#: private names (both are small, stable, and documented in place).
_HELPER_DEPS = {"aux", "auxpass", "cop"}
_MAIN_CLAUSE_DEPS = {"ROOT", "parataxis"}


def _is_predicate_head(token: Any) -> bool:
    return token.pos_ in ("VERB", "AUX") and token.dep_ not in _HELPER_DEPS


def _is_main_clause_head(token: Any) -> bool:
    """Whether a predicate head forms its own T-unit (an independent clause).

    Walks a ``conj`` chain up to its ultimate ancestor: two clauses joined by
    "and"/"but"/"or" at the top level ("She left and he stayed.") are two
    T-units, but a clause coordinated *under* a subordinator ("because she
    left and he stayed") is not -- the whole thing is still one T-unit with
    one dependent clause plus its internal coordination. ``advcl``,
    ``relcl``, ``ccomp``, ``xcomp``, ``acl`` and ``csubj``/``csubjpass``
    (anything other than ``ROOT``/``parataxis``/a ``conj`` chain back to one
    of those) return ``False`` immediately: the predicate is a dependent
    clause. This is exactly the distinction Lu (2010)'s L2SCA drew with
    Tregex over constituency trees; this is the dependency-label
    approximation of the same distinction.
    """

    current = token
    seen: set[int] = set()
    while current.i not in seen:
        seen.add(current.i)
        if current.dep_ in _MAIN_CLAUSE_DEPS:
            return True
        if current.dep_ == "conj":
            current = current.head
            continue
        return False
    return False  # pragma: no cover - defensive cycle guard; spaCy trees are acyclic


def _complex_nominal_indices(tokens: Sequence[Any]) -> set[int]:
    """Noun heads elaborated by a relative/participial clause, a possessive,
    an appositive, or a prepositional-phrase postmodifier, plus clausal
    subjects (``csubj``/``csubjpass``) -- L2SCA's "complex nominal", each
    counted once even when a noun qualifies on more than one ground."""

    out: set[int] = set()
    for token in tokens:
        if token.pos_ not in ("NOUN", "PROPN", "PRON"):
            continue
        for child in token.children:
            if child.dep_ in ("relcl", "acl", "poss", "appos"):
                out.add(token.i)
                break
            if child.dep_ == "prep" and any(gc.dep_ in ("pobj", "pcomp") for gc in child.children):
                out.add(token.i)
                break
    out.update(token.i for token in tokens if token.dep_ in ("csubj", "csubjpass"))
    return out


def _collect_clause_stats(analysis: DocumentAnalysis) -> dict[str, Any]:
    tunit_length: list[float] = []
    clause_length: list[float] = []
    clauses_per_tunit: list[float] = []
    dependent_per_clause: list[float] = []
    dependent_per_tunit: list[float] = []
    coordinate_per_clause: list[float] = []
    coordinate_per_tunit: list[float] = []
    complex_nominal_per_clause: list[float] = []
    complex_nominal_per_tunit: list[float] = []
    verb_phrases_per_tunit: list[float] = []
    sentence_words: list[float] = []
    tunits_per_sentence: list[float] = []
    # Raw per-sentence counts (as opposed to the per-sentence RATIOS above),
    # kept so the L2SCA findings below can pool them (sum of numerators /
    # sum of denominators over the whole sampled text) rather than averaging
    # or medianing the per-sentence ratios -- see _pooled()'s docstring for
    # why those are not the same quantity.
    clause_counts: list[float] = []
    dependent_counts: list[float] = []
    coordinate_counts: list[float] = []
    complex_nominal_counts: list[float] = []
    finite_total = 0
    nonfinite_total = 0
    sentence_count = 0

    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            tokens = list(sent)
            if not tokens:
                continue
            sentence_count += 1
            n_words = sum(1 for token in tokens if not token.is_punct)
            sentence_words.append(float(n_words))

            predicate_heads = [token for token in tokens if _is_predicate_head(token)]
            predicate_idx = {token.i for token in predicate_heads}
            n_tunit_raw = sum(1 for token in predicate_heads if _is_main_clause_head(token))
            n_predicate = len(predicate_heads)
            n_dependent = n_predicate - n_tunit_raw
            n_tunit_eff = max(1, n_tunit_raw)
            n_clause_eff = n_tunit_eff + n_dependent
            tunits_per_sentence.append(float(n_tunit_eff))

            coordinate_phrases = sum(1 for token in tokens
                                     if token.dep_ == "conj" and token.i not in predicate_idx)
            n_complex_nominal = len(_complex_nominal_indices(tokens))

            for token in predicate_heads:
                if token.morph.get("Tense") or "Fin" in token.morph.get("VerbForm"):
                    finite_total += 1
                else:
                    nonfinite_total += 1

            tunit_length.append(n_words / n_tunit_eff)
            clause_length.append(n_words / n_clause_eff)
            clauses_per_tunit.append(n_clause_eff / n_tunit_eff)
            dependent_per_clause.append(n_dependent / n_clause_eff)
            dependent_per_tunit.append(n_dependent / n_tunit_eff)
            coordinate_per_clause.append(coordinate_phrases / n_clause_eff)
            coordinate_per_tunit.append(coordinate_phrases / n_tunit_eff)
            complex_nominal_per_clause.append(n_complex_nominal / n_clause_eff)
            complex_nominal_per_tunit.append(n_complex_nominal / n_tunit_eff)
            # Verb phrases share this implementation's predicate-head
            # detection with clauses (documented limitation: L2SCA's Tregex
            # patterns distinguish "verb phrase" from "clause" more finely
            # than a dependency-label proxy can without more machinery than
            # this pass adds), so this ratio is identical to
            # clauses_per_tunit above under this approximation.
            verb_phrases_per_tunit.append(n_predicate / n_tunit_eff)
            clause_counts.append(float(n_clause_eff))
            dependent_counts.append(float(n_dependent))
            coordinate_counts.append(float(coordinate_phrases))
            complex_nominal_counts.append(float(n_complex_nominal))

    return {
        "sentence_count": sentence_count,
        "sentence_words": sentence_words,
        "tunits_per_sentence": tunits_per_sentence,
        "tunit_length": tunit_length,
        "clause_length": clause_length,
        "clauses_per_tunit": clauses_per_tunit,
        "dependent_per_clause": dependent_per_clause,
        "dependent_per_tunit": dependent_per_tunit,
        "coordinate_per_clause": coordinate_per_clause,
        "coordinate_per_tunit": coordinate_per_tunit,
        "complex_nominal_per_clause": complex_nominal_per_clause,
        "complex_nominal_per_tunit": complex_nominal_per_tunit,
        "verb_phrases_per_tunit": verb_phrases_per_tunit,
        "clause_counts": clause_counts,
        "dependent_counts": dependent_counts,
        "coordinate_counts": coordinate_counts,
        "complex_nominal_counts": complex_nominal_counts,
        "finite_total": finite_total,
        "nonfinite_total": nonfinite_total,
    }


_TUNIT_IDS = {
    "syntax.complexity_l2sca_mean_sentence_length": "L2SCA-style mean sentence length (approximation)",
    "syntax.complexity_l2sca_tunits_per_sentence": "L2SCA-style T-units per sentence (approximation)",
    "syntax.complexity_l2sca_mean_tunit_length": "L2SCA-style mean T-unit length (approximation)",
    "syntax.complexity_l2sca_clauses_per_tunit": "L2SCA-style clauses per T-unit (approximation)",
    "syntax.complexity_l2sca_mean_clause_length": "L2SCA-style mean clause length (approximation)",
    "syntax.complexity_l2sca_dependent_clauses_per_clause":
        "L2SCA-style dependent clauses per clause (approximation)",
    "syntax.complexity_l2sca_dependent_clauses_per_tunit":
        "L2SCA-style dependent clauses per T-unit (approximation)",
    "syntax.complexity_l2sca_coordinate_phrases_per_clause":
        "L2SCA-style coordinate phrases per clause (approximation)",
    "syntax.complexity_l2sca_coordinate_phrases_per_tunit":
        "L2SCA-style coordinate phrases per T-unit (approximation)",
    "syntax.complexity_l2sca_complex_nominals_per_clause":
        "L2SCA-style complex nominals per clause (approximation)",
    "syntax.complexity_l2sca_complex_nominals_per_tunit":
        "L2SCA-style complex nominals per T-unit (approximation)",
    "syntax.complexity_l2sca_verb_phrases_per_tunit":
        "L2SCA-style verb phrases per T-unit (approximation; identical to clauses/T-unit here)",
    "syntax.complexity_finite_nonfinite_clause_ratio": "Finite-to-nonfinite clause ratio",
}

_L2SCA_WARNING = ("approximation of L2SCA, computed from spaCy dependency labels rather than "
                  "the Tregex/constituency patterns L2SCA and TAASSC define it with -- see the "
                  "module docstring's 'tunit_clause' section")


def _join_warnings(*parts: str | None) -> str | None:
    joined = "; ".join(part for part in parts if part)
    return joined or None


def _pooled(metric_id: str, numerator_total: float, denominator_total: float,
           per_sentence_values: Sequence[float], unit: str, *,
           cross_ref: str | None = None, approximation: bool = True) -> dict[str, Any]:
    """A single L2SCA-style ratio, headlined as L2SCA and TAASSC actually
    define it: the POOLED, text-level ratio of totals (sum of numerators /
    sum of denominators over every sentence/T-unit/clause sampled), not the
    median or mean of each sentence's own small ratio.

    Those two are genuinely different quantities whenever sentences differ
    in size (a ratio of sums is not the same as a mean of ratios -- the
    same reason a batting average is computed hits/at-bats, not the mean of
    each game's own average). On real books this matters: complex-nominal
    and coordinate-phrase counts per clause are small integers on most
    sentences, so the MEDIAN or MEAN of the per-sentence ratio collapses
    toward the value the *majority* of (short, simple) sentences take,
    while the pooled ratio reflects the whole sampled text and keeps the
    resolution a corpus percentile needs. The per-sentence shape is not
    thrown away -- it is still reported, under
    ``distribution["per_sentence_shape"]`` -- but it no longer headlines.

    ``sample_size`` is the denominator total (the number of sentences,
    T-units or clauses actually pooled over), per the same "how much did
    this rate rest on" convention every other rate metric in this codebase
    already uses.
    """

    pooled_value = numerator_total / denominator_total if denominator_total else None
    per_sentence_shape = summarize(per_sentence_values) if per_sentence_values else {"count": 0}
    distribution: dict[str, Any] = {
        "aggregation": "pooled",
        "numerator_total": numerator_total,
        "denominator_total": denominator_total,
        "per_sentence_shape": per_sentence_shape,
    }
    if cross_ref:
        distribution["overlaps_existing_metric_id"] = cross_ref
    warning = None if denominator_total else "no units to pool this ratio over"
    if approximation:
        warning = _join_warnings(warning, _L2SCA_WARNING)
    return finding(metric_id, _TUNIT_IDS[metric_id], pooled_value, unit, family=FAMILY,
                   sample_size=int(denominator_total), min_sample=MIN_SAMPLE,
                   distribution=distribution, warning=warning)


def _tunit_clause(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    stats = analysis.memo("syntax_complexity_clause_stats", lambda: _collect_clause_stats(analysis))
    sentence_count = stats["sentence_count"]
    no_data = "no sentences to compute T-unit/clause ratios over"
    if not sentence_count:
        return [unavailable(metric_id, name, no_data, family=FAMILY)
                for metric_id, name in _TUNIT_IDS.items()]

    total_words = sum(stats["sentence_words"])
    total_tunit = sum(stats["tunits_per_sentence"])
    total_clause = sum(stats["clause_counts"])
    total_dependent = sum(stats["dependent_counts"])
    total_coordinate = sum(stats["coordinate_counts"])
    total_complex_nominal = sum(stats["complex_nominal_counts"])
    total_predicate = float(stats["finite_total"] + stats["nonfinite_total"])

    out: list[dict[str, Any]] = [
        _pooled("syntax.complexity_l2sca_mean_sentence_length", total_words, sentence_count,
               stats["sentence_words"], "words",
               cross_ref="style.sentence_words_p50 draws its median from this same per-sentence "
                        "word-count population; this headlines the pooled MEAN (total words / "
                        "total sentences) instead, matching L2SCA's MLS definition -- the two "
                        "differ in aggregation (mean vs. median), not in what they count"),
        _pooled("syntax.complexity_l2sca_tunits_per_sentence", total_tunit, sentence_count,
               stats["tunits_per_sentence"], "T-units"),
        _pooled("syntax.complexity_l2sca_mean_tunit_length", total_words, total_tunit,
               stats["tunit_length"], "words"),
        _pooled("syntax.complexity_l2sca_clauses_per_tunit", total_clause, total_tunit,
               stats["clauses_per_tunit"], "ratio"),
        _pooled("syntax.complexity_l2sca_mean_clause_length", total_words, total_clause,
               stats["clause_length"], "words"),
        _pooled("syntax.complexity_l2sca_dependent_clauses_per_clause", total_dependent,
               total_clause, stats["dependent_per_clause"], "ratio",
               cross_ref="syntax.subordination_rate (a rate per 100 sentences, also a "
                        "document-level pooled quantity; this pools dependent clauses over "
                        "clauses instead of over sentences)"),
        _pooled("syntax.complexity_l2sca_dependent_clauses_per_tunit", total_dependent,
               total_tunit, stats["dependent_per_tunit"], "ratio"),
        _pooled("syntax.complexity_l2sca_coordinate_phrases_per_clause", total_coordinate,
               total_clause, stats["coordinate_per_clause"], "ratio",
               cross_ref="syntax.coordination_rate (a rate per 100 sentences, also a "
                        "document-level pooled quantity; this excludes clause-level "
                        "coordination, already reflected in the T-unit count above, and pools "
                        "over clauses instead of over sentences)"),
        _pooled("syntax.complexity_l2sca_coordinate_phrases_per_tunit", total_coordinate,
               total_tunit, stats["coordinate_per_tunit"], "ratio"),
        _pooled("syntax.complexity_l2sca_complex_nominals_per_clause", total_complex_nominal,
               total_clause, stats["complex_nominal_per_clause"], "ratio"),
        _pooled("syntax.complexity_l2sca_complex_nominals_per_tunit", total_complex_nominal,
               total_tunit, stats["complex_nominal_per_tunit"], "ratio"),
        _pooled("syntax.complexity_l2sca_verb_phrases_per_tunit", total_predicate, total_tunit,
               stats["verb_phrases_per_tunit"], "ratio"),
    ]

    finite, nonfinite = stats["finite_total"], stats["nonfinite_total"]
    total_predicates = finite + nonfinite
    ratio = finite / nonfinite if nonfinite else (None if not finite else float("inf"))
    ratio_warning = None
    if not total_predicates:
        ratio_warning = "no predicate heads found to classify as finite or nonfinite"
    elif not nonfinite:
        ratio_warning = ("no nonfinite predicate found; finite-to-nonfinite ratio is undefined "
                         "rather than infinite")
        ratio = None
    out.append(finding(
        "syntax.complexity_finite_nonfinite_clause_ratio", _TUNIT_IDS["syntax.complexity_finite_nonfinite_clause_ratio"],
        ratio, "ratio", family=FAMILY, sample_size=total_predicates, min_sample=MIN_SAMPLE,
        distribution={"aggregation": "pooled", "finite_count": finite, "nonfinite_count": nonfinite,
                     "overlaps_existing_metric_id": "syntax.finite_clauses_per_sentence "
                     "(a per-sentence median count via shape(); this ratio pools the finite and "
                     "nonfinite totals across the whole document and compares them directly)"},
        warning=ratio_warning))
    return out


# -------------------------------------------------------------- phrasal elaboration

def _collect_phrasal_stats(analysis: DocumentAnalysis) -> dict[str, Any]:
    np_lengths: list[float] = []
    np_depths: list[float] = []
    premodifiers: list[float] = []
    postmodifiers: list[float] = []
    postmodifier_types: Counter = Counter()
    prep_count = 0
    appos_count = 0
    participial_count = 0
    total_words = 0
    sentence_count = 0
    noun_chunks_supported = True

    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            sentence_count += 1
        for token in doc:
            if not token.is_punct:
                total_words += 1
            if token.dep_ == "prep":
                prep_count += 1
            elif token.dep_ == "appos":
                appos_count += 1
            elif token.dep_ == "acl" and "Part" in token.morph.get("VerbForm"):
                participial_count += 1
        if noun_chunks_supported:
            try:
                chunks = list(doc.noun_chunks)
            except Exception:  # pragma: no cover - a pipeline without noun-chunk support
                noun_chunks_supported = False
                continue
            for chunk in chunks:
                tokens = list(chunk)
                words = [token for token in tokens if not token.is_punct]
                if not words:
                    continue
                np_lengths.append(float(len(words)))
                root = chunk.root
                depth = 0
                for token in tokens:
                    depth = max(depth, sum(1 for ancestor in token.ancestors
                                           if chunk.start <= ancestor.i < chunk.end))
                np_depths.append(float(depth))
                premodifiers.append(float(sum(1 for token in tokens if token.i < root.i)))
                post_children = [child for child in root.children
                                if child.i > root.i and child.dep_ in ("prep", "relcl", "acl", "appos")]
                postmodifiers.append(float(len(post_children)))
                for child in post_children:
                    postmodifier_types[child.dep_] += 1

    return {
        "np_lengths": np_lengths, "np_depths": np_depths, "premodifiers": premodifiers,
        "postmodifiers": postmodifiers, "postmodifier_types": postmodifier_types,
        "prep_count": prep_count, "appos_count": appos_count,
        "participial_count": participial_count, "total_words": total_words,
        "sentence_count": sentence_count, "noun_chunks_supported": noun_chunks_supported,
    }


_PHRASAL_IDS = {
    "syntax.complexity_noun_phrase_length": "Noun-phrase length",
    "syntax.complexity_noun_phrase_depth": "Noun-phrase internal depth",
    "syntax.complexity_premodifiers_per_np": "Premodifiers per noun phrase",
    "syntax.complexity_postmodifiers_per_np": "Postmodifiers per noun phrase",
    "syntax.complexity_pp_attachment_density": "Prepositional-phrase attachment density",
    "syntax.complexity_appositive_rate": "Appositive rate",
    "syntax.complexity_participial_modifier_rate": "Participial modifier rate",
    "syntax.complexity_nominal_postmodifier_diversity": "Nominal postmodifier-type diversity",
}


def _phrasal_elaboration(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    stats = analysis.memo("syntax_complexity_phrasal_stats", lambda: _collect_phrasal_stats(analysis))
    no_data = "no sentences to compute phrasal-elaboration measures over"
    if not stats["sentence_count"]:
        return [unavailable(metric_id, name, no_data, family=FAMILY)
                for metric_id, name in _PHRASAL_IDS.items()]

    out: list[dict[str, Any]] = []
    if stats["np_lengths"]:
        out.extend(shape("syntax.complexity_noun_phrase_length", _PHRASAL_IDS["syntax.complexity_noun_phrase_length"],
                         stats["np_lengths"], "words", family=FAMILY, min_sample=MIN_SAMPLE))
        out.extend(shape("syntax.complexity_noun_phrase_depth", _PHRASAL_IDS["syntax.complexity_noun_phrase_depth"],
                         stats["np_depths"], "levels", family=FAMILY, min_sample=MIN_SAMPLE))
        out.extend(shape("syntax.complexity_premodifiers_per_np", _PHRASAL_IDS["syntax.complexity_premodifiers_per_np"],
                         stats["premodifiers"], "modifiers", family=FAMILY, min_sample=MIN_SAMPLE))
        out.extend(shape("syntax.complexity_postmodifiers_per_np", _PHRASAL_IDS["syntax.complexity_postmodifiers_per_np"],
                         stats["postmodifiers"], "modifiers", family=FAMILY, min_sample=MIN_SAMPLE))
        post_types = stats["postmodifier_types"]
        total_post = sum(post_types.values())
        entropy = coh.entropy_of_counts(post_types) if total_post else None
        out.append(finding(
            "syntax.complexity_nominal_postmodifier_diversity",
            _PHRASAL_IDS["syntax.complexity_nominal_postmodifier_diversity"], entropy, "bits",
            family=FAMILY, sample_size=total_post, min_sample=MIN_SAMPLE,
            sample_size_sensitive=True,
            evidence=[{"postmodifier_type": key, "count": count}
                     for key, count in post_types.most_common(10)],
            warning=None if total_post else "no noun-phrase postmodifier found"))
    else:
        no_np = ("no noun chunk was found" if stats["noun_chunks_supported"]
                else "this spaCy pipeline does not support doc.noun_chunks")
        for metric_id in ("syntax.complexity_noun_phrase_length", "syntax.complexity_noun_phrase_depth",
                          "syntax.complexity_premodifiers_per_np", "syntax.complexity_postmodifiers_per_np",
                          "syntax.complexity_nominal_postmodifier_diversity"):
            out.append(unavailable(metric_id, _PHRASAL_IDS[metric_id], no_np, family=FAMILY))

    words = stats["total_words"]
    sentences = stats["sentence_count"]
    out.append(finding(
        "syntax.complexity_pp_attachment_density", _PHRASAL_IDS["syntax.complexity_pp_attachment_density"],
        rate(stats["prep_count"], words, 1000.0), "per 1,000 words", family=FAMILY,
        sample_size=words, min_sample=200,
        warning=None if words else "no words to compute PP density over"))
    out.append(finding(
        "syntax.complexity_appositive_rate", _PHRASAL_IDS["syntax.complexity_appositive_rate"],
        rate(stats["appos_count"], sentences), "per 100 sentences", family=FAMILY,
        sample_size=sentences, min_sample=MIN_SAMPLE,
        warning=None if sentences else no_data))
    out.append(finding(
        "syntax.complexity_participial_modifier_rate",
        _PHRASAL_IDS["syntax.complexity_participial_modifier_rate"],
        rate(stats["participial_count"], sentences), "per 100 sentences", family=FAMILY,
        sample_size=sentences, min_sample=MIN_SAMPLE,
        warning=None if sentences else no_data))
    return out


# ------------------------------------------------------------- dependency topology

def _collect_topology_stats(analysis: DocumentAnalysis, long_dep_threshold: int) -> dict[str, Any]:
    branching: list[float] = []
    subtree_sizes: list[float] = []
    tree_imbalance: list[float] = []
    root_pos: Counter = Counter()
    dep_labels: Counter = Counter()
    dep_transitions: Counter = Counter()
    rightward = 0
    total_arcs = 0
    long_dep = 0
    nonprojective_sentences = 0
    sentence_count = 0
    skipped_long_sentences = 0
    #: A sentence longer than this is skipped for the O(sentence_length) or
    #: O(sentence_length^2) scans below (subtree size, branching, crossing
    #: arcs). Ordinary prose sentences are far shorter than this even before
    #: the pySBD dialogue-splitting fix; this exists only so a single
    #: pathological outlier cannot blow up per-document cost.
    max_scan_tokens = 300

    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            tokens = list(sent)
            if not tokens:
                continue
            sentence_count += 1
            children_count: dict[int, int] = {}
            pos_seq: list[str] = []
            dep_seq: list[str] = []
            root = None
            for token in tokens:
                pos_seq.append(token.pos_)
                dep_seq.append(token.dep_)
                dep_labels[token.dep_] += 1
                if token.dep_ == "ROOT":
                    root = token
                else:
                    total_arcs += 1
                    if abs(token.i - token.head.i) > long_dep_threshold:
                        long_dep += 1
                    if token.i > token.head.i:
                        rightward += 1
                    children_count[token.head.i] = children_count.get(token.head.i, 0) + 1
            for a, b in zip(dep_seq, dep_seq[1:]):
                dep_transitions[(a, b)] += 1
            if root is not None:
                root_pos[root.pos_] += 1
            if len(tokens) > max_scan_tokens:
                skipped_long_sentences += 1
                continue
            branching.extend(float(count) for count in children_count.values())
            depths = [sum(1 for _ in token.ancestors) for token in tokens]
            max_depth = max(depths) if depths else 0
            node_count = len(tokens)
            subtree_sizes.extend(float(len(list(token.subtree))) for token in tokens)
            tree_imbalance.append(max_depth / math.log2(node_count + 1) if node_count > 1 else 0.0)
            if node_count > 1:
                arcs = [(min(token.i, token.head.i), max(token.i, token.head.i))
                       for token in tokens if token.dep_ != "ROOT"]
                crossed = False
                for i in range(len(arcs)):
                    a1, b1 = arcs[i]
                    for j in range(i + 1, len(arcs)):
                        a2, b2 = arcs[j]
                        if a1 < a2 < b1 < b2 or a2 < a1 < b2 < b1:
                            crossed = True
                            break
                    if crossed:
                        break
                if crossed:
                    nonprojective_sentences += 1

    return {
        "sentence_count": sentence_count, "branching": branching, "subtree_sizes": subtree_sizes,
        "tree_imbalance": tree_imbalance, "root_pos": root_pos, "dep_labels": dep_labels,
        "dep_transitions": dep_transitions, "rightward": rightward, "total_arcs": total_arcs,
        "long_dep": long_dep, "nonprojective_sentences": nonprojective_sentences,
        "skipped_long_sentences": skipped_long_sentences,
    }


_TOPOLOGY_IDS = {
    "syntax.complexity_branching_factor": "Dependency branching factor",
    "syntax.complexity_tree_imbalance": "Dependency-tree imbalance index",
    "syntax.complexity_dependency_rightward_share": "Dependency head-direction (rightward share)",
    "syntax.complexity_long_dependency_rate": "Long-dependency rate",
    "syntax.complexity_nonprojective_sentence_rate": "Non-projective (crossing-dependency) sentence rate",
    "syntax.complexity_root_pos_entropy": "Sentence-root POS entropy",
    "syntax.complexity_subtree_size": "Dependency subtree size",
    "syntax.complexity_dependency_label_entropy": "Dependency-label entropy",
    "syntax.complexity_dependency_transition_entropy": "Dependency-label transition entropy",
}


def _dependency_topology(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    threshold = int(option(config, "long_dependency_threshold", 10))
    stats = analysis.memo(f"syntax_complexity_topology_stats_{threshold}",
                          lambda: _collect_topology_stats(analysis, threshold))
    sentence_count = stats["sentence_count"]
    no_data = "no sentences to compute dependency-topology measures over"
    if not sentence_count:
        return [unavailable(metric_id, name, no_data, family=FAMILY)
                for metric_id, name in _TOPOLOGY_IDS.items()]

    out: list[dict[str, Any]] = []
    out.extend(shape("syntax.complexity_branching_factor", _TOPOLOGY_IDS["syntax.complexity_branching_factor"],
                     stats["branching"], "children", family=FAMILY, min_sample=MIN_SAMPLE))
    imbalance_finding = shape("syntax.complexity_tree_imbalance",
                              _TOPOLOGY_IDS["syntax.complexity_tree_imbalance"],
                              stats["tree_imbalance"], "ratio", family=FAMILY, min_sample=MIN_SAMPLE)[0]
    imbalance_finding["distribution"] = {
        **(imbalance_finding["distribution"] or {}),
        "overlaps_existing_metric_id": "syntax.max_parse_depth/syntax.mean_max_parse_depth "
        "(raw per-sentence tree depth; this normalizes depth by log2(node_count+1) instead "
        "of reporting it directly, so long AND deep sentences are distinguished from short "
        "AND deep ones)",
        "skipped_sentences_over_300_tokens": stats["skipped_long_sentences"]}
    out.append(imbalance_finding)

    out.append(finding(
        "syntax.complexity_dependency_rightward_share",
        _TOPOLOGY_IDS["syntax.complexity_dependency_rightward_share"],
        rate(stats["rightward"], stats["total_arcs"], 100.0), "%", family=FAMILY,
        sample_size=stats["total_arcs"], min_sample=100,
        distribution={"note": "share of non-root dependents whose head appears earlier in the "
                              "sentence (a rightward/left-headed attachment); a value near 50% "
                              "is a balanced mix, not evidence of anything on its own"},
        warning=None if stats["total_arcs"] else "no non-root dependency arc to classify"))
    out.append(finding(
        "syntax.complexity_long_dependency_rate", _TOPOLOGY_IDS["syntax.complexity_long_dependency_rate"],
        rate(stats["long_dep"], stats["total_arcs"], 100.0), "%", family=FAMILY,
        sample_size=stats["total_arcs"], min_sample=100,
        distribution={"threshold_tokens": threshold,
                     "overlaps_existing_metric_id": "syntax.dependency_distance_mean/"
                     "syntax.sentence_dependency_distance (per-sentence mean distance; this is "
                     "the share of ALL individual dependency arcs across the document that "
                     "exceed a fixed threshold, a different unit of analysis)"},
        warning=None if stats["total_arcs"] else "no non-root dependency arc to classify"))
    out.append(finding(
        "syntax.complexity_nonprojective_sentence_rate",
        _TOPOLOGY_IDS["syntax.complexity_nonprojective_sentence_rate"],
        rate(stats["nonprojective_sentences"], sentence_count - stats["skipped_long_sentences"], 100.0),
        "%", family=FAMILY, sample_size=sentence_count - stats["skipped_long_sentences"],
        min_sample=MIN_SAMPLE,
        distribution={"skipped_sentences_over_300_tokens": stats["skipped_long_sentences"]},
        warning=None if sentence_count > stats["skipped_long_sentences"] else
        "every sentence exceeded the 300-token scan cap"))

    root_pos_total = sum(stats["root_pos"].values())
    out.append(finding(
        "syntax.complexity_root_pos_entropy", _TOPOLOGY_IDS["syntax.complexity_root_pos_entropy"],
        coh.entropy_of_counts(stats["root_pos"]) if root_pos_total else None, "bits", family=FAMILY,
        sample_size=root_pos_total, min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        evidence=[{"root_pos": key, "count": count} for key, count in stats["root_pos"].most_common(10)],
        warning=None if root_pos_total else no_data))

    if stats["subtree_sizes"]:
        subtree_finding = shape("syntax.complexity_subtree_size", _TOPOLOGY_IDS["syntax.complexity_subtree_size"],
                                stats["subtree_sizes"], "tokens", family=FAMILY, min_sample=MIN_SAMPLE)[0]
        subtree_finding["distribution"] = {**(subtree_finding["distribution"] or {}),
                                           "skipped_sentences_over_300_tokens": stats["skipped_long_sentences"]}
        out.append(subtree_finding)
    else:
        out.append(unavailable("syntax.complexity_subtree_size", _TOPOLOGY_IDS["syntax.complexity_subtree_size"],
                               "every sentence exceeded the 300-token scan cap", family=FAMILY))

    dep_label_total = sum(stats["dep_labels"].values())
    out.append(finding(
        "syntax.complexity_dependency_label_entropy",
        _TOPOLOGY_IDS["syntax.complexity_dependency_label_entropy"],
        coh.entropy_of_counts(stats["dep_labels"]) if dep_label_total else None, "bits",
        family=FAMILY, sample_size=dep_label_total, min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        evidence=[{"dependency_label": key, "count": count}
                 for key, count in stats["dep_labels"].most_common(10)],
        warning=None if dep_label_total else no_data))

    dep_trans_total = sum(stats["dep_transitions"].values())
    out.append(finding(
        "syntax.complexity_dependency_transition_entropy",
        _TOPOLOGY_IDS["syntax.complexity_dependency_transition_entropy"],
        coh.entropy_of_counts(stats["dep_transitions"]) if dep_trans_total else None, "bits",
        family=FAMILY, sample_size=dep_trans_total, min_sample=MIN_SAMPLE, sample_size_sensitive=True,
        warning=None if dep_trans_total else no_data))
    return out


# --------------------------------------------------------------------- constituency

_BENEPAR_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_benepar_cache() -> None:
    _BENEPAR_CACHE.clear()


on_reset(_reset_benepar_cache)


def _shim_t5_retokenizer() -> None:
    """Restore ``T5Tokenizer(Fast).build_inputs_with_special_tokens``.

    benepar's retokenizer (``benepar/retokenization.py``) calls this method
    to locate a T5 tokenizer's special-token positions. ``transformers``
    5.17.0's rewritten tokenizer classes (``TokenizersBackend``) no longer
    define it at all, so loading ``benepar_en3`` (a T5-based checkpoint)
    raises ``AttributeError: T5Tokenizer has no attribute
    build_inputs_with_special_tokens`` -- reproduced directly in this
    environment before this shim existed. The restored method is exactly
    T5's own pre-5.x rule: a single sequence gets one trailing EOS id, a
    pair gets EOS after each sequence. Installed only when the attribute is
    actually missing (``hasattr`` guard), so a future transformers release
    that restores it is left alone -- the same one-attribute, defensive
    pattern :func:`textgrader.coherence._shim_transformers_tied_weights`
    already uses for an analogous fastcoref/transformers version skew.
    """

    try:
        from transformers import T5Tokenizer, T5TokenizerFast
    except Exception:  # pragma: no cover - transformers itself unavailable
        return

    def _build(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    for cls in (T5Tokenizer, T5TokenizerFast):
        if not hasattr(cls, "build_inputs_with_special_tokens"):
            cls.build_inputs_with_special_tokens = _build


def _load_benepar(model_name: str) -> tuple[Any, str | None]:
    """A cached spaCy pipeline with ``benepar`` attached, built once per
    process exactly like :func:`textgrader.coherence._load_rst_parser`."""

    if model_name in _BENEPAR_CACHE:
        return _BENEPAR_CACHE[model_name]
    spacy_module, reason = require("spacy")
    if spacy_module is None:
        _BENEPAR_CACHE[model_name] = (None, reason)
        return _BENEPAR_CACHE[model_name]
    benepar_module, reason = require("benepar")
    if benepar_module is None:
        _BENEPAR_CACHE[model_name] = (None, reason)
        return _BENEPAR_CACHE[model_name]
    _shim_t5_retokenizer()
    try:
        nlp = spacy_module.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        nlp.add_pipe("benepar", config={"model": model_name})
        outcome: tuple[Any, str | None] = (nlp, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"benepar model {model_name!r} unavailable ({type(exc).__name__}: {exc}); "
                         f"pip install benepar and run python -c \"import benepar; "
                         f"benepar.download({model_name!r})\"")
    _BENEPAR_CACHE[model_name] = outcome
    return outcome


def _resolve_constituency(analysis: DocumentAnalysis, model_name: str, num_sentences: int,
                          max_sentences_cap: int, max_seconds: float, seed: int
                         ) -> tuple[list[str], dict[str, Any], str | None]:
    """Bracketed constituency-tree strings for a bounded, deterministic
    sample of individual sentences, spread across the document.

    Reuses :func:`textgrader.coherence.sample_rst_passages` with
    ``passage_sentences=1``: that function already does exactly the sampling
    this needs (deterministic, seeded, spread across the book, hard-capped),
    one sentence per "passage" instead of RST's multi-sentence windows.
    """

    nlp, reason = _load_benepar(model_name)
    settings: dict[str, Any] = {"backend": "benepar", "model": model_name}
    if nlp is None:
        return [], settings, reason

    passages, sample_info = coh.sample_rst_passages(analysis.sentences, num_sentences, 1,
                                                     max_sentences_cap, seed)
    settings.update(sample_info)
    if not passages:
        return [], settings, "no sentences available to sample for constituency parsing"

    started = time.monotonic()  # after the (cached, one-time) parser load above
    stopped_early = False
    trees: list[str] = []
    for _start, _end, sentence_texts in passages:
        if time.monotonic() - started > max_seconds:
            stopped_early = True
            break
        text = sentence_texts[0] if sentence_texts else ""
        if not text.strip():
            continue
        try:
            doc = nlp(text)
            for sent in doc.sents:
                trees.append(sent._.parse_string)
        except Exception:  # pragma: no cover - a single bad sentence should not sink the sample
            continue

    settings.update({"sentences_parsed": len(trees), "elapsed_seconds": time.monotonic() - started,
                     "max_seconds_cap": max_seconds, "stopped_early_on_time_cap": stopped_early})
    note = (f"backend=benepar model={model_name}: parsed {len(trees)} of "
           f"{sample_info['passages_sampled']} sampled sentence(s)"
           + (" (stopped early: time cap reached)" if stopped_early else ""))
    return trees, settings, note


def _tree_metrics(trees: Sequence[str]) -> dict[str, Any] | None:
    nltk_module, reason = require("nltk")
    if nltk_module is None:
        return None
    from nltk import Tree

    depths: list[float] = []
    phrase_types: Counter = Counter()
    productions: Counter = Counter()
    root_templates: Counter = Counter()
    for text in trees:
        try:
            tree = Tree.fromstring(text)
        except Exception:  # pragma: no cover - a malformed bracket string should not sink the sample
            continue
        depths.append(float(tree.height() - 1))
        for subtree in tree.subtrees():
            is_preterminal = len(subtree) == 1 and isinstance(subtree[0], str)
            if not is_preterminal:
                phrase_types[subtree.label()] += 1
        productions_here = tree.productions()
        for production in productions_here:
            if not production.is_lexical():
                productions[str(production)] += 1
        if productions_here:
            root_templates[str(productions_here[0])] += 1
    return {"depths": depths, "phrase_types": phrase_types, "productions": productions,
           "root_templates": root_templates}


_CONSTITUENCY_IDS = {
    "syntax.complexity_constituency_tree_depth": "Constituency tree depth (benepar sample)",
    "syntax.complexity_constituency_phrase_type_entropy": "Constituency phrase-type entropy (benepar sample)",
    "syntax.complexity_constituency_production_rule_entropy":
        "Constituency production-rule entropy (benepar sample)",
    "syntax.complexity_constituency_distinct_subtree_rate":
        "Distinct constituency production-rule rate (benepar sample)",
    "syntax.complexity_constituency_template_top_share":
        "Top sentence-template (root production) share (benepar sample)",
    "syntax.complexity_constituency_template_diversity":
        "Sentence-template (root production) diversity (benepar sample)",
}


def _constituency(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    model = option(config, "constituency_model", "benepar_en3")
    num_sentences = int(option(config, "constituency_sample_sentences", 30))
    max_sentences_cap = int(option(config, "constituency_max_sentences", 60))
    max_seconds = float(option(config, "constituency_max_seconds", 180.0))
    seed = int(option(config, "constituency_seed", 0))

    trees, settings, note = analysis.memo(
        f"syntax_complexity_constituency_{model}_{num_sentences}_{max_sentences_cap}_{seed}",
        lambda: _resolve_constituency(analysis, model, num_sentences, max_sentences_cap,
                                      max_seconds, seed))
    if not trees:
        return [unavailable(metric_id, name, note or "no constituency parse produced", family=FAMILY)
               for metric_id, name in _CONSTITUENCY_IDS.items()]

    metrics = _tree_metrics(trees)
    if metrics is None:
        reason = "nltk is required to read benepar's bracketed trees but is unavailable"
        return [unavailable(metric_id, name, reason, family=FAMILY)
               for metric_id, name in _CONSTITUENCY_IDS.items()]

    passages_parsed = settings.get("sentences_parsed", len(trees))
    passages_sampled = settings.get("passages_sampled", passages_parsed)

    out: list[dict[str, Any]] = []
    depth_finding = shape("syntax.complexity_constituency_tree_depth",
                          _CONSTITUENCY_IDS["syntax.complexity_constituency_tree_depth"],
                          metrics["depths"], "levels", family=FAMILY, min_sample=passages_sampled)[0]
    depth_finding["distribution"] = {**settings, **(depth_finding["distribution"] or {})}
    depth_finding["warning"] = note
    depth_finding["sample_size"] = passages_parsed
    out.append(depth_finding)

    phrase_total = sum(metrics["phrase_types"].values())
    out.append(finding(
        "syntax.complexity_constituency_phrase_type_entropy",
        _CONSTITUENCY_IDS["syntax.complexity_constituency_phrase_type_entropy"],
        coh.entropy_of_counts(metrics["phrase_types"]) if phrase_total else None, "bits",
        family=FAMILY, sample_size=passages_parsed, min_sample=passages_sampled,
        sample_size_sensitive=True,
        distribution={**settings, "phrase_nodes_sampled": phrase_total},
        evidence=[{"phrase_type": key, "count": count}
                 for key, count in metrics["phrase_types"].most_common(15)],
        warning=note if phrase_total else f"{note}; no phrase-level node found"))

    production_total = sum(metrics["productions"].values())
    out.append(finding(
        "syntax.complexity_constituency_production_rule_entropy",
        _CONSTITUENCY_IDS["syntax.complexity_constituency_production_rule_entropy"],
        coh.entropy_of_counts(metrics["productions"]) if production_total else None, "bits",
        family=FAMILY, sample_size=passages_parsed, min_sample=passages_sampled,
        sample_size_sensitive=True,
        distribution={**settings, "non_lexical_productions_sampled": production_total},
        evidence=[{"production": key, "count": count}
                 for key, count in metrics["productions"].most_common(15)],
        warning=note if production_total else f"{note}; no non-lexical production found"))

    distinct_rate = (100.0 * len(metrics["productions"]) / production_total
                    if production_total else None)
    out.append(finding(
        "syntax.complexity_constituency_distinct_subtree_rate",
        _CONSTITUENCY_IDS["syntax.complexity_constituency_distinct_subtree_rate"], distinct_rate, "%",
        family=FAMILY, sample_size=passages_parsed, min_sample=passages_sampled,
        distribution={**settings, "distinct_productions": len(metrics["productions"]),
                     "total_productions_observed": production_total},
        warning=note if production_total else f"{note}; no non-lexical production found"))

    template_total = sum(metrics["root_templates"].values())
    top_template, top_count = (metrics["root_templates"].most_common(1)[0]
                               if metrics["root_templates"] else (None, 0))
    out.append(finding(
        "syntax.complexity_constituency_template_top_share",
        _CONSTITUENCY_IDS["syntax.complexity_constituency_template_top_share"],
        rate(top_count, template_total) if template_total else None, "%", family=FAMILY,
        sample_size=passages_parsed, min_sample=passages_sampled,
        distribution={**settings},
        evidence=[{"template": key, "count": count}
                 for key, count in metrics["root_templates"].most_common(10)],
        warning=note if template_total else f"{note}; no sentence template found"))
    out.append(finding(
        "syntax.complexity_constituency_template_diversity",
        _CONSTITUENCY_IDS["syntax.complexity_constituency_template_diversity"],
        coh.entropy_of_counts(metrics["root_templates"]) if template_total else None, "bits",
        family=FAMILY, sample_size=passages_parsed, min_sample=passages_sampled,
        sample_size_sensitive=True, distribution={**settings},
        warning=note if template_total else f"{note}; no sentence template found"))
    return out


# ---------------------------------------------------------------- syntactic surprisal

_PROFILE_KEY = "syntax_complexity_suite"


def _pos_bigrams_by_sentence(analysis: DocumentAnalysis) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    """One entry per sentence: its text (trimmed for evidence) and its
    ordered POS-tag bigrams, for both :func:`profile_vector` (pooled into a
    corpus-wide frequency table) and the per-sentence surprisal reading in
    :func:`_syntactic_surprisal` (scored against that pooled table)."""

    out: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            tags = [token.pos_ for token in sent]
            bigrams = tuple(zip(tags, tags[1:]))
            out.append((sent.text.strip()[:160], bigrams))
    return out


def _dep_bigrams_by_sentence(analysis: DocumentAnalysis) -> list[tuple[tuple[str, str], ...]]:
    out: list[tuple[tuple[str, str], ...]] = []
    for _, doc in analysis.spacy_docs():
        for sent in doc.sents:
            labels = [token.dep_ for token in sent]
            out.append(tuple(zip(labels, labels[1:])))
    return out


def _freq_vector(counts: Counter, prefix: str, max_keys: int = 400) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {f"{prefix}{a}>{b}": count / total
           for (a, b), count in counts.most_common(max_keys)}


def profile_vector(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None
                   ) -> dict[str, float] | None:
    """The per-book POS-bigram/dependency-label-bigram frequency table
    :mod:`textgrader.corpus` caches for this suite, one row per book under
    ``feature_profiles["syntax_complexity_suite"]`` -- exactly the mechanism
    ``coherence_suite.profile_vector`` already uses for its entity-grid
    transition table. This is the ONLY place this suite fits a syntactic
    n-gram model; :func:`measure` only ever reads the pooled result back
    through ``profile``, never from the document being scored -- see the
    module docstring's "syntactic_surprisal" section.
    """

    if analysis.nlp_unavailable:
        return None
    pos_bigrams: Counter = Counter()
    for _, bigrams in _pos_bigrams_by_sentence(analysis):
        pos_bigrams.update(bigrams)
    dep_bigrams: Counter = Counter()
    for bigrams in _dep_bigrams_by_sentence(analysis):
        dep_bigrams.update(bigrams)
    vector = {**_freq_vector(pos_bigrams, "pos2:"), **_freq_vector(dep_bigrams, "dep2:")}
    return vector or None


def _pooled_model(profile: Mapping[str, Any] | None, prefix: str) -> dict[str, float] | None:
    rows = ((profile or {}).get("feature_profiles") or {}).get(_PROFILE_KEY) or []
    rows = [row for row in rows if row]
    if not rows:
        return None
    keys = {key for row in rows for key in row if key.startswith(prefix)}
    if not keys:
        return None
    return {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys}


def _cross_entropy(counts: Counter, prefix: str, model: Mapping[str, float]
                   ) -> tuple[float | None, int]:
    """``-sum p(x) * log2(q(x))`` for this document's bigram distribution
    ``p`` against the pooled corpus model ``q``. An unseen bigram is floored
    at half the model's smallest observed probability (never zero, so a
    single novel bigram cannot make the whole cross-entropy undefined)."""

    total = sum(counts.values())
    if not total:
        return None, 0
    floor = min(model.values()) / 2 if model else 1e-6
    entropy = 0.0
    for (a, b), count in counts.items():
        q = model.get(f"{prefix}{a}>{b}", floor)
        entropy += (count / total) * -math.log2(max(q, 1e-12))
    return entropy, total


_SURPRISAL_IDS = {
    "syntax.complexity_pos_bigram_cross_entropy": "POS-bigram cross-entropy against the corpus",
    "syntax.complexity_dependency_label_bigram_cross_entropy":
        "Dependency-label-bigram cross-entropy against the corpus",
    "syntax.complexity_sentence_syntactic_surprisal": "Per-sentence syntactic surprisal",
}
_NO_CORPUS_MODEL = ("no corpus-trained POS/dependency n-gram model available; build a profile "
                    "with --parse-metrics AND syntax_complexity_suite enabled "
                    "(features.syntactic_surprisal on, the default) to populate one -- this "
                    "suite's cost is 'parse', so textgrader.corpus._metric_names drops it from "
                    "profiling unless --parse-metrics is passed, even with the suite enabled; "
                    "see profile_vector in this module")


def _syntactic_surprisal(analysis: DocumentAnalysis, config: Mapping[str, Any],
                         profile: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    pos_model = _pooled_model(profile, "pos2:")
    if pos_model is None:
        out = [unavailable(metric_id, name, _NO_CORPUS_MODEL, family=FAMILY)
              for metric_id, name in _SURPRISAL_IDS.items()]
        return out

    per_sentence = _pos_bigrams_by_sentence(analysis)
    pos_counts: Counter = Counter()
    for _, bigrams in per_sentence:
        pos_counts.update(bigrams)
    pos_entropy, pos_total = _cross_entropy(pos_counts, "pos2:", pos_model)

    out = [finding(
        "syntax.complexity_pos_bigram_cross_entropy", _SURPRISAL_IDS["syntax.complexity_pos_bigram_cross_entropy"],
        pos_entropy, "bits", family=FAMILY, sample_size=pos_total, min_sample=200,
        distribution={"corpus_books": len(((profile or {}).get("feature_profiles") or {})
                                          .get(_PROFILE_KEY, []))},
        warning=None if pos_total else "no POS bigram found in this document")]

    dep_model = _pooled_model(profile, "dep2:")
    if dep_model is None:
        out.append(unavailable("syntax.complexity_dependency_label_bigram_cross_entropy",
                               _SURPRISAL_IDS["syntax.complexity_dependency_label_bigram_cross_entropy"],
                               _NO_CORPUS_MODEL, family=FAMILY))
    else:
        dep_counts: Counter = Counter()
        for bigrams in _dep_bigrams_by_sentence(analysis):
            dep_counts.update(bigrams)
        dep_entropy, dep_total = _cross_entropy(dep_counts, "dep2:", dep_model)
        out.append(finding(
            "syntax.complexity_dependency_label_bigram_cross_entropy",
            _SURPRISAL_IDS["syntax.complexity_dependency_label_bigram_cross_entropy"],
            dep_entropy, "bits", family=FAMILY, sample_size=dep_total, min_sample=200,
            warning=None if dep_total else "no dependency-label bigram found in this document"))

    per_sentence_surprisal: list[tuple[str, float]] = []
    floor = min(pos_model.values()) / 2 if pos_model else 1e-6
    for text, bigrams in per_sentence:
        if not bigrams:
            continue
        values = [-math.log2(max(pos_model.get(f"pos2:{a}>{b}", floor), 1e-12)) for a, b in bigrams]
        per_sentence_surprisal.append((text, sum(values) / len(values)))

    if per_sentence_surprisal:
        values = [value for _, value in per_sentence_surprisal]
        ordered = sorted(per_sentence_surprisal, key=lambda item: item[1])
        evidence = (
            [{"sentence": text, "mean_surprisal_bits": round(value, 2), "extreme": "lowest"}
            for text, value in ordered[:5]]
            + [{"sentence": text, "mean_surprisal_bits": round(value, 2), "extreme": "highest"}
              for text, value in ordered[-5:]])
        surprisal_finding = shape("syntax.complexity_sentence_syntactic_surprisal",
                                  _SURPRISAL_IDS["syntax.complexity_sentence_syntactic_surprisal"],
                                  values, "bits", family=FAMILY, min_sample=MIN_SAMPLE,
                                  evidence=evidence)[0]
        out.append(surprisal_finding)
    else:
        out.append(unavailable("syntax.complexity_sentence_syntactic_surprisal",
                               _SURPRISAL_IDS["syntax.complexity_sentence_syntactic_surprisal"],
                               "no sentence had a POS bigram to score", family=FAMILY))
    return out


# ---------------------------------------------------------------------------- entry

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    if analysis.nlp_unavailable:
        all_ids = {**_TUNIT_IDS, **_PHRASAL_IDS, **_TOPOLOGY_IDS}
        features = _features(config)
        if features.get("constituency", False):
            all_ids = {**all_ids, **_CONSTITUENCY_IDS}
        if features.get("syntactic_surprisal", True):
            all_ids = {**all_ids, **_SURPRISAL_IDS}
        return [unavailable(metric_id, name, analysis.nlp_unavailable, family=FAMILY)
               for metric_id, name in all_ids.items()]

    features = _features(config)
    out: list[dict[str, Any]] = []
    if features.get("tunit_clause", True):
        out.extend(_tunit_clause(analysis, config))
    if features.get("phrasal_elaboration", True):
        out.extend(_phrasal_elaboration(analysis, config))
    if features.get("dependency_topology", True):
        out.extend(_dependency_topology(analysis, config))
    if features.get("constituency", False):
        out.extend(_constituency(analysis, config))
    if features.get("syntactic_surprisal", True):
        out.extend(_syntactic_surprisal(analysis, config, profile))
    return out
