"""Logic, consistency, entailment and argument-structure sensors.

The first pass through this module (see git history) was written against a
toolchain this environment did not have: no NLI cross-encoder, no OpenIE
service, no argument-mining model, no GPU, no network access to fetch one.
It shipped only what a dependency-free pass and the installed spaCy pipeline
could genuinely compute, and named every one of them for what it actually
measured. A second pass installed an NLI model, ``fastcoref``, downloaded
WordNet corpus data and ``python-dateutil``, closing most of what that first
pass's own ``Deferred`` section named. This fourth pass closed the rest of it
for real, by actually trying the tools the first two passes had assumed away
rather than repeating the assumption:

* **AllenNLP** (the obvious source for real SRL and real OpenIE) was
  installed, not merely cited as unavailable -- and it does not fit here, on
  hard evidence, not a shrug. ``pip install --dry-run "allennlp==2.10.1"``
  (the last release, bundling ``allennlp-models``' SRL and OpenIE predictors)
  fails outright in this environment::

      ERROR: Could not find a version that satisfies the requirement
      torch<1.13.0,>=1.10.0 (from allennlp) (from versions: 1.13.0, 1.13.1,
      2.0.0, 2.0.1, ..., 2.14.0)
      ERROR: No matching distribution found for torch<1.13.0,>=1.10.0

  AllenNLP pins ``torch<1.13.0``; PyPI's oldest ``torch`` wheel for this
  environment's Python (3.11) is ``1.13.0`` itself, one version too new. There
  is no torch this environment can install that satisfies AllenNLP's own
  ceiling -- not "not installed," genuinely not installable without a Python
  this project does not run on. This closes AllenNLP for both SRL and OpenIE
  at once (both predictors ship in ``allennlp-models``, gated by the same
  ``allennlp`` core dependency), permanently, and is why it is not attempted
  again below.
* **Semantic role labelling** is real anyway: ``cu-kairos/propbank_srl_seq2seq_t5_small``,
  a ``transformers``-native seq2seq model (no new optional package -- the
  same ``transformers`` install ``nli_entailment`` already needs), backs the
  new ``semantic_role_labeling`` feature. See
  :mod:`textgrader.propositions`'s "semantic role labelling" section for how
  it is called (its own model card's ``pipeline("text2text-generation", ...)``
  does not work against this environment's ``transformers`` 5.17.0 --
  ``KeyError: "Unknown task text2text-generation, ...`` on the actual call,
  not assumed -- so this module drives the model and tokenizer directly) and
  for a real, hand-checked example of exactly the failure mode a shallow
  dependency-parse subject cannot see: SRL reads "he" as ARG-1 (patient) in
  "He was arrested by the police" and would read the same referent as ARG-0
  (agent) in an active rewording, a voice-invariant reading a grammatical
  subject/object split cannot give.
* **Open information extraction**, honestly, could not be closed as asked:
  Stanford OpenIE is Java (out by the Python-only decision) and AllenNLP's
  OpenIE predictor is blocked by the identical dry-run failure quoted above.
  What is offered instead, under its own honest name -- ``relation_extraction``,
  never "openie" -- is ``Babelscape/rebel-large``, a real, transformers-native
  model, but a **closed-schema** relation extractor (roughly 200 Wikidata-style
  relation types such as "capital of" or "spouse"), not an open-domain,
  arbitrary-predicate extractor. That is a genuine difference in kind, not
  degree, and every finding built from it says so. Hand-checked, not assumed,
  on real sentences: "Marie Curie was born in Warsaw" correctly comes back
  ``{'head': 'Marie Curie', 'type': 'place of birth', 'tail': 'Warsaw'}``, but
  "Alice was tired after her long journey through the old town" -- a plain
  fictional sentence asserting nothing about where Alice lives -- generated
  ``{'head': 'Alice', 'type': 'residence', 'tail': 'old town'}`` anyway: a
  plausible-sounding, schema-compatible triple the seq2seq format's own
  pressure to always emit something invented rather than one the sentence
  actually states. So this is not simply "fiction scores near zero because
  the schema does not cover it" -- it can just as easily score a false
  positive, and every finding built from this channel says a triple is this
  model's own generated reading, never a verified fact, for exactly that
  reason.
* **Argument mining** is real: ``raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L``,
  a RoBERTa-large Argument Relation Identification classifier trained on the
  US2016 and QT30 debate corpora, backs the new ``argument_mining`` feature,
  labelling this suite's own therefore/because/however-adjacent clause pairs
  No-Relation / Inference (support) / Conflict (attack) / Rephrase. The
  existing ``connective_chain_length`` proxy is unchanged and kept under its
  own honest name -- a marker-adjacency proxy for argument structure, never
  claim/premise/support/attack labels from a model -- exactly as the module
  docstring said before this pass, now sitting beside the real thing instead
  of standing in for it.

See ``Deferred`` at the end of this docstring for what remains out of scope
and why -- every remaining item is now either excluded by the Python-only
decision or backed by the dry-run evidence quoted above, not an unchecked
assumption -- and see the "Gating" section below before touching any
off-by-default group's defaults, old or new.

Nothing here checks whether a claim is *true*, including the groups backed by
a real model. Two sentences that share a subject and a verb but disagree
about the object are a **contradiction candidate** -- worth a human's
attention, not a verdict. A "therefore" that connects two sentences with no
shared vocabulary is a **low lexical-overlap premise/conclusion pair** -- it
might still be a perfectly good inference dressed in different words, or it
might be a non sequitur. An NLI model's "contradiction" label is a **model
score on a sentence pair stripped of context**, not a fact about the text --
fiction legitimately contains contradictions (unreliable narrators, lies,
hypotheticals, quoted falsehoods), so a high contradiction share is never
reported as an error count. Every finding below says this in its name or its
docstring, not just once here.

Fourteen independently switchable measurement groups, under ``features`` in
this suite's config block. The first four were the original pass and default
to *on*; the other ten each need an installed model or resource and default to
*off* regardless of their own cost (see "Gating"):

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
    linking them to what came before. These candidate counts are unchanged by
    every group added since: WordNet and NLI report their own readings of the
    same pairs alongside them rather than editing them, so this group's
    numbers stay comparable across a run with or without the new groups on.

``modal_argument_position`` (stdlib, cost ``fast``)
    Whether hedges and modals cluster around the sentences that carry an
    argumentative connective, compared with the rest of the text.

``nli_entailment`` (``transformers`` cross-encoder, off by default)
    Real entailment/neutral/contradiction scores, from
    ``cross-encoder/nli-deberta-v3-small`` by default, over a candidate pool
    built the same way as ``propositions``' heuristic scans (shared subject
    and predicate, capped separately and much lower via ``nli_max_pairs``
    because this is the most expensive thing in the module). Reports the
    label distribution with the model name and version recorded alongside it,
    the strongest contradictions as bounded evidence, and a confusion-matrix
    cross-check of how often this suite's own negation/attribute-conflict
    heuristics agree or disagree with the model's label on the same pairs.

``coreference_resolution`` (``fastcoref``, off by default; modifies
``propositions``/``nli_entailment`` rather than adding metrics of its own)
    Resolves a pronoun subject ("she") to a named antecedent ("Alice") over
    the first ``coreference_max_chars`` characters, so it can enter the same
    subject+predicate buckets a directly-named subject would. ``fastcoref``
    now loads and predicts cleanly here, through
    :func:`textgrader.optional.shim_fastcoref_transformers`, and has been
    exercised against the real model, not only a fake one (see
    ``tests/test_logic_suite.py``'s
    ``test_coreference_resolution_against_a_real_fastcoref_model``) -- this is
    no longer off by default because it is broken. It stays off by default
    because it is another neural model on top of the spaCy parse this module
    already pays for, its resolution is only sampled over the first
    ``coreference_max_chars`` characters rather than the whole book, and every
    resolution is the model's own judgement, never a verified reading; see
    ``textgrader/propositions.py`` for exactly how it resolves and degrades.

``lexical_opposition`` (``nltk`` + downloaded WordNet corpus data, off by
default)
    A WordNet antonym channel independent of both the surface heuristics and
    the NLI model (``discourse.logic_wordnet_antonym_candidates``), plus an
    informational hypernym/hyponym cross-check on ``propositions``'
    ``property``-subtype candidates that flags likely false positives
    ("dog"/"poodle" is not a contradiction) without ever changing that
    metric's own count.

``temporal_ordering`` (``python-dateutil``, off by default)
    For ``propositions``' ``temporal``-subtype candidates (two differing
    explicit years/dates on the same subject+predicate), which one is
    chronologically earlier and whether that matches narrative order --
    explicitly framed as an observation, not an error, since flashbacks and
    non-chronological narration produce this legitimately.

``semantic_role_labeling`` (``transformers`` seq2seq SRL, off by default)
    Real PropBank-style ARG-0..ARG-4 (+ ARGM-*) role frames from
    ``cu-kairos/propbank_srl_seq2seq_t5_small`` by default, over clause-level
    predicates capped by ``srl_max_predicates`` (see
    :func:`textgrader.propositions.extract_srl_frames`). Two channels built
    from those frames: ``discourse.logic_srl_role_pattern_consistency`` --
    whether the same predicate's core-argument role signature (which ARG-N
    slots got filled) holds steady across its mentions in the document, a
    voice-invariant check a shallow subject/object split cannot make -- and
    ``discourse.logic_srl_argument_omission_rate``, the share of frames the
    model filled with fewer than two core arguments. Both are candidate
    signals, explicitly not errors: legitimate voice alternation, ellipsis and
    verb-sense variation all produce the same pattern a genuine inconsistency
    would.

``relation_extraction`` (``transformers`` seq2seq relation extraction, off by
default)
    ``Babelscape/rebel-large`` by default, over sentences capped by
    ``relation_extraction_max_sentences`` (see
    :func:`textgrader.propositions.extract_relations`). CLOSED-schema (~200
    Wikidata-style relation types), not open-domain OpenIE -- see this
    docstring's opening section for why an open-domain extractor could not be
    closed this pass, and why that makes this a genuinely different, narrower
    tool than the one the task named. Reports a triple rate and how often a
    triple's head/tail also appear as a subject/object pair in
    ``propositions``' own dependency-parse proxy -- the two extractors
    disagreeing, or one finding nothing the other also finds nothing on, is
    reported as data, never as either channel being wrong.

``argument_mining`` (``transformers`` argument-relation classifier, off by
default)
    ``raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L`` by default, applied to
    this suite's own therefore/because/however-adjacent clause pairs (capped
    by ``argument_mining_max_pairs``), labelling each pair No-Relation /
    Inference (support) / Conflict (attack) / Rephrase. Reports the label
    distribution and a confusion-matrix cross-check of the connective's own
    implied polarity (support: therefore/because; contrast: however/but)
    against the model's label. ``connective_chain_length`` (under
    ``connective_relations``) is unchanged and kept under its own honest
    name -- see this docstring's opening section.

``propbank_argument_structure`` (``transformers`` SRL + ``nltk`` PropBank
corpus, off by default)
    Cross-checks ``semantic_role_labeling``'s own SRL frames (re-extracted
    through the same cached :func:`textgrader.propositions.extract_srl_frames`
    call -- never a second model run when both features share the same
    ``srl_model``/``srl_max_predicates``) against PropBank's own downloaded
    frame files via :func:`textgrader.propositions.propbank_best_roleset`.
    Where ``discourse.logic_srl_argument_omission_rate`` only asks "did the
    model fill fewer than two role slots", this asks the more principled
    version the task named: "did THIS predicate's own PropBank roleset
    (``give.01``'s giver/thing-given/recipient, not a generic threshold)
    expect an argument the model never filled" -- ``discourse.logic_
    propbank_argument_omission_rate``. Which roleset applies is picked
    heuristically (most numbered-argument overlap with what was already
    filled), not from a gold sense tag, so a rarer sense can be missed; the
    finding's own warning says so, and says a fragment or an elliptical line
    of dialogue produces exactly the same pattern as a genuinely dropped
    argument.

``verbnet_class_consistency`` (``nltk`` VerbNet corpus, off by default)
    For two DIFFERENT verbs used with the SAME two participants (shared
    non-pronoun subject and non-pronoun object, within ``window_sentences`` or
    a paragraph), whether they share any VerbNet class at all --
    ``discourse.logic_verbnet_class_conflict_candidates``. "The workers built
    the tower... the workers destroyed the tower" shares no VerbNet class and
    is flagged; "she opened the door... she closed the door" shares one
    (``other_cos-45.4``) and is not, which is itself a finding -- VerbNet's
    classing, not this module's guess, decided the ordinary same-object scene
    beat is not a mismatch while the incompatible one is. A verb absent from
    VerbNet's inventory makes the pair unjudgeable, never a conflict.

``framenet_frame_consistency`` (``nltk`` FrameNet corpus, off by default)
    The same "same participants, different verb" check as
    ``verbnet_class_consistency``, against FrameNet's frame inventory instead
    of VerbNet's classes -- ``discourse.logic_framenet_frame_conflict_
    candidates``. Independent lexical resource, independent partition of the
    verb lexicon, deliberately kept as its own channel rather than merged with
    the VerbNet one: the two agreeing (as they do on both worked examples
    above) is corroboration, not redundancy, and either one flagging alone
    without the other is data worth keeping visible, per this project's
    standing rule to treat library disagreement as data rather than force one
    reading. The first FrameNet lookup in a process costs several seconds
    (``nltk`` builds an internal lemma index on first use); every later one is
    a dict access.

Every scalar this module reports is a **candidate rate, a lexical-overlap
score, or a labelled model score**, never a truth value on its own, and every
one records the settings it was computed under (``window_sentences``,
``max_pairs``, the spaCy pipeline name/version, the NLI model name, whether
coreference resolution ran) in its ``distribution``, because two runs with
different caps, a different spaCy model, or a different NLI checkpoint are
not the same measurement.

Gating
------

``textgrader/corpus.py`` decides which metrics it profiles over every
reference book by excluding ``needs_parse`` and ``needs_model`` metrics, and
``MetricSpec.needs_model`` is defined as ``"sentence_transformers" in
self.requires`` -- nothing else. Every model-backed feature in this suite
(``nli_entailment``, the SRL/relation-extraction/argument-mining trio, and
coreference resolution) is backed by ``transformers`` or ``fastcoref``, never
``sentence_transformers``, so that guard **cannot see any of them**: a corpus
build run with ``include_model_metrics=True`` would not exclude this suite on
that basis alone. Two things currently stand between this and a book-length
pass of five transformer models (``nli_entailment``, coreference resolution,
``semantic_role_labeling``, ``relation_extraction`` and ``argument_mining`` --
``lexical_opposition`` and ``temporal_ordering`` are ``nltk``/``dateutil``,
not transformer models, but default to off for the same reason) nobody asked
for, and both must stay true:

1. This suite's registry ``cost`` is ``"parse"``, so ``needs_parse`` already
   excludes it from corpus profiling regardless of ``needs_model`` -- do not
   make this suite cheaper without re-checking this.
2. ``features.nli_entailment``, ``coreference_resolution``,
   ``lexical_opposition``, ``temporal_ordering``, ``semantic_role_labeling``,
   ``relation_extraction``, ``argument_mining``, ``propbank_argument_structure``,
   ``verbnet_class_consistency`` and ``framenet_frame_consistency`` all default
   to ``False`` in both ``MetricSpec.defaults`` and ``config.json``, and the
   ``on()`` helper in :func:`measure` falls back to ``False`` for every one of
   these ten when a caller's ``features`` mapping omits them -- unlike the
   original four groups, which fall back to ``True`` for backward
   compatibility. Getting that fallback backwards for a new group would turn
   on a transformer-model or corpus-lookup pass for every caller that passes
   ``config=None`` or a partial ``features`` mapping, most of which are tests.
   ``_ON_BY_DEFAULT`` is the one place this is decided; a new feature group
   must never be added to it without the same scrutiny this note asks for.

``_load_nli_pipeline``, ``_load_argument_mining_pipeline``,
``prop_lib._load_coref_model``, ``prop_lib._load_srl_model`` and
``prop_lib._load_relation_model`` are the only five places this module (and
the ``propositions`` module it uses) ever import ``transformers`` or
``fastcoref``, and every one is called only from inside its own feature's
finding function -- never at import time, never unless the owning feature is
on. ``prop_lib.load_propbank``, ``prop_lib.load_verbnet`` and
``prop_lib.load_framenet`` are the equivalent three places for the corpora
this pass added; the same rule applies even though none of the three needs a
GPU-class dependency -- a book-length pass of ``nltk``'s FrameNet lemma-index
build (several seconds the first time; see that feature's own note above) is
still not something a default config should ever pay for.
``tests/test_logic_suite.py``'s ``test_default_config_never_loads_nli_or_coref_model``
asserts this directly for the five model loaders by monkeypatching them to
raise and running ``measure()`` under the default config; a second test,
``test_default_config_never_touches_propbank_verbnet_or_framenet``, does the
identical check for the three corpus loaders this pass added.

Measured cost, CPU only, ``cross-encoder/nli-deberta-v3-small``: loading the
pipeline is about 2.5s; scoring is roughly 30ms/pair batched (measured on an
otherwise idle core), so the default ``nli_max_pairs=60`` costs on the order
of 4-5s end to end on top of whatever ``propositions`` already paid for the
parse and the two heuristic bucket scans, which this group does not repeat.
That per-pair figure rose sharply (tens of seconds for the same 60 pairs) when
measured on a machine with four other CPU-bound processes contending for the
same cores -- a real property of shared hardware, not of this code -- so
timing this channel is only meaningful on an otherwise-quiet machine.
Proposition extraction and the heuristic bucket scans stay ``parse``-class
and are entirely unaffected by whether this group is on.

Measured cost, CPU only, the three seq2seq/classification channels added this
pass: each is one model *generation* (SRL, relation extraction) or
classification (argument mining) call per candidate, batched, and each is
markedly more expensive per item than ``nli_entailment``'s classification
call -- a batch of 6 short SRL inputs measured at roughly 44s/item and a
single ``Babelscape/rebel-large`` load at over 130s on this environment's
otherwise-idle-in-theory CPU, both far worse in practice whenever other
CPU-bound work shares the same cores (exactly the contention effect the NLI
paragraph above already documents, reproduced here on every new channel).
This is why ``srl_max_predicates``, ``relation_extraction_max_sentences`` and
``argument_mining_max_pairs`` all default to a much smaller 40 rather than
``nli_max_pairs``'s 60, and why every real-model test for these three
features in ``tests/test_logic_suite.py`` caps its own run far below even
that, the same way the existing NLI tests already do with ``nli_max_pairs``.

Measured cost, CPU only, the three lexical-resource channels this pass added:
``propbank_argument_structure`` pays nothing beyond ``semantic_role_labeling``'s
own SRL model cost when both are on with the same settings (the SRL frames are
the same cached call; the PropBank roleset lookup itself is a dict/XML-element
walk, sub-millisecond per frame once ``load_propbank`` has run once).
``verbnet_class_consistency`` is similarly cheap once ``load_verbnet`` has run
(``vn.classids`` is a fast in-memory lookup). ``framenet_frame_consistency`` is
the one with a real, one-time cost: the first ``fn.frames_by_lemma`` call in a
process measured at roughly 6s (building ``nltk``'s internal FrameNet lemma
index), after which every further lookup in the same process is a dict access
under 1ms -- see :func:`textgrader.propositions.load_framenet`'s own docstring.

No corpus-reference channel
----------------------------

This module deliberately does not define ``profile_vector`` (the hook
``textgrader.corpus.build_profile`` looks for to cache a per-book feature
vector under ``feature_profiles``). Every scalar this suite reports is
already captured the ordinary way, one number per book, by whatever calls
``measure`` during profiling -- a distribution over reference books is not
missing, just not this module's job to build twice. What ``profile_vector``
is *for* is a vector too rich for a single scalar (a frequency table, an
embedding); nothing here is that shape, and this suite's numbers are
candidate rates and model-label shares that this module is emphatic, in
every finding's own warning, are not meant to imply a normative "acceptable
contradiction rate" a manuscript should be graded against -- building a
corpus-comparison channel on top of them would cut directly against that.
This suite's ``cost`` also stays ``"parse"`` specifically so it is excluded
from default corpus profiling (see "Gating" below); a ``profile_vector`` here
would be built, at real cost, for reference books nobody asked to profile.

Deferred
--------

Permanently out of scope, each for a reason checked this pass, not assumed:

* **Stanford CoreNLP / Stanford OpenIE.** Not used: both are Java, and
  TextGrader is Python only by decision. This is the one item in this section
  excluded by project decision rather than by a capability check, and it
  would stay excluded even if a Python wrapper existed.
* **AllenNLP SRL / OpenIE.** Actually installed, not merely cited as
  unavailable, and rejected on its own dry-run failure: see this docstring's
  opening section for the quoted ``pip install --dry-run "allennlp==2.10.1"``
  output. AllenNLP pins ``torch<1.13.0``; no such wheel exists for this
  environment's Python (3.11). ``semantic_role_labeling`` and
  ``relation_extraction`` above are real, working replacements for what this
  bullet used to defer -- a different model each, from the Hugging Face hub
  through the ``transformers`` this codebase already depends on -- not a
  second attempt at AllenNLP itself.
* **ConceptNet.** Not shipped via ``nltk``; using it means either a network
  call to its public API (a runtime dependency this module does not want) or
  a separate multi-hundred-megabyte downloaded database (``conceptnet-lite``
  or similar). Neither was attempted this pass -- a genuine scope decision,
  not a claim that either is impossible.

Closed across this and earlier passes, with real limitations of their own
(each documented at its own feature group above and, for coreference,
WordNet, SRL, relation extraction, PropBank, VerbNet and FrameNet, in
``textgrader/propositions.py``): pairwise NLI, coreference resolution,
WordNet antonymy, temporal ordering, semantic role labelling, closed-schema
relation extraction, argument mining, PropBank argument-structure omission,
VerbNet class-consistency and FrameNet frame-consistency.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from .. import text as textlib
from ..document import DocumentAnalysis
from ..optional import on_reset, require
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
    "discourse.logic_nli_label_distribution": "NLI-scored candidate-pair label distribution (entailment/neutral/contradiction)",
    "discourse.logic_nli_heuristic_agreement": "Heuristic-candidate vs. NLI-label agreement",
    "discourse.logic_wordnet_antonym_candidates": "WordNet antonym-based lexical-opposition candidates",
    "discourse.logic_wordnet_hypernym_downgrade_rate": "Property-conflict candidates WordNet marks as hypernym/hyponym-related, not opposed",
    "discourse.logic_temporal_order_candidates": "Temporal-conflict pairs whose parsed date order contradicts narrative order",
    "discourse.logic_srl_role_pattern_consistency": "SRL role-pattern (in)consistency across mentions of the same predicate",
    "discourse.logic_srl_argument_omission_rate": "SRL frames with fewer than two core arguments filled ('thin' predication rate)",
    "discourse.logic_relation_extraction_triple_rate": "Closed-schema relation-extraction triple rate",
    "discourse.logic_relation_extraction_proposition_overlap_rate": "Relation-extraction triples corroborated by the propositions dependency-parse proxy",
    "discourse.logic_argument_relation_label_distribution": "Argument-mining relation-label distribution over connective-linked clause pairs",
    "discourse.logic_argument_relation_connective_agreement": "Connective-polarity vs. argument-mining-model label agreement",
    "discourse.logic_propbank_argument_omission_rate": "PropBank-roleset-expected core arguments the SRL model left unfilled",
    "discourse.logic_verbnet_class_conflict_candidates": "Same-participants, different-verb pairs sharing no VerbNet class",
    "discourse.logic_framenet_frame_conflict_candidates": "Same-participants, different-verb pairs evoking no shared FrameNet frame",
}

#: Reference date for :func:`dateutil.parser.parse` when a parsed string omits
#: a field (a bare year has no month/day). Fixed rather than "now" so the same
#: input parses to the same value on every run, which a comparable metric needs.
_TEMPORAL_PARSE_DEFAULT = datetime(2000, 1, 1)


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


# --------------------------------------------------------------- NLI backend
#
# The one place in this module that imports a heavy ML dependency, and it
# does so lazily, through ``require``, only from inside ``_load_nli_pipeline``
# -- never at module import time and never unless ``features.nli_entailment``
# is explicitly on. See the module docstring's "Gating" section for why that
# matters more here than for any other metric in this codebase: this suite's
# cost class is ``parse``, not ``model``, so the corpus builder's
# ``include_model_metrics`` guard does not see it -- ``features`` being off by
# default is the only thing standing between this and a book-length NLI pass
# nobody asked for.

_NLI_MODEL_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_nli_cache() -> None:
    _NLI_MODEL_CACHE.clear()


on_reset(_reset_nli_cache)


def _load_nli_pipeline(model_name: str) -> tuple[Any, str | None]:
    if model_name in _NLI_MODEL_CACHE:
        return _NLI_MODEL_CACHE[model_name]
    module, reason = require("transformers")
    if module is None:
        _NLI_MODEL_CACHE[model_name] = (None, reason)
        return _NLI_MODEL_CACHE[model_name]
    try:
        pipeline_obj = module.pipeline("text-classification", model=model_name, device=-1, top_k=None)
        outcome: tuple[Any, str | None] = (pipeline_obj, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"transformers NLI model {model_name!r} unavailable "
                         f"({type(exc).__name__}: {exc}); pip install transformers torch")
    _NLI_MODEL_CACHE[model_name] = outcome
    return outcome


def _nli_label_scores(pipeline_obj: Any, pairs: list[tuple[str, str]],
                      batch_size: int) -> list[tuple[str, dict[str, float]]]:
    """``(top_label, {label: score, ...})`` per pair, batched through the pipeline."""

    inputs = [{"text": a, "text_pair": b} for a, b in pairs]
    raw = pipeline_obj(inputs, batch_size=batch_size, truncation=True)
    out = []
    for rows in raw:
        scores = {row["label"].lower(): float(row["score"]) for row in rows}
        top_label = max(scores, key=scores.get) if scores else "neutral"
        out.append((top_label, scores))
    return out


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

@dataclass
class _PropContext:
    """Everything the propositions/NLI/WordNet/temporal groups share.

    Built once per :func:`measure` call so that enabling several of these
    groups together pays for extraction, coreference resolution and the two
    heuristic bucket scans exactly once -- the same sharing discipline
    :meth:`DocumentAnalysis.spacy_sents_by_channel` already gives the parse
    itself.
    """

    analysis: DocumentAnalysis
    unavailable_reason: str | None
    extraction: "prop_lib.Extraction | None"
    props: list  # list[prop_lib.Proposition]; coreference-resolved when that feature is on
    coref_meta: dict[str, Any]
    negation_scan: "prop_lib.PairScan | None"
    attribute_scan: "prop_lib.PairScan | None"
    settings: dict[str, Any]
    window_sentences: int
    max_pairs: int
    max_comparisons: int


def _build_proposition_context(analysis: DocumentAnalysis, *, proposition_cap: int, window_sentences: int,
                               max_pairs: int, max_comparisons: int, coreference_enabled: bool,
                               coreference_max_chars: int) -> _PropContext:
    empty = lambda reason: _PropContext(  # noqa: E731 - small, local, and only used twice
        analysis, reason, None, [], {"enabled": coreference_enabled}, None, None, {},
        window_sentences, max_pairs, max_comparisons)
    if analysis.nlp_unavailable:
        return empty(analysis.nlp_unavailable)

    extraction = prop_lib.extract(analysis, proposition_cap)
    props = extraction.propositions
    if not props:
        return empty("no sentence yielded a usable (subject, predicate) proposition")

    coref_meta: dict[str, Any] = {"enabled": coreference_enabled}
    if coreference_enabled:
        resolution = prop_lib.resolve_coreference(analysis, extraction, coreference_max_chars)
        coref_meta.update(applied=resolution.available, reason=resolution.reason,
                          resolved_count=resolution.resolved_count, chars_used=resolution.chars_used,
                          truncated=resolution.truncated, model=resolution.model_name)
        props = resolution.propositions

    settings = _settings(window_sentences=window_sentences, max_pairs=max_pairs,
                         proposition_cap=proposition_cap, spacy_model=extraction.spacy_model,
                         spacy_version=extraction.spacy_version, ner_available=extraction.ner_available,
                         coreference_resolution=coref_meta)
    negation_scan = prop_lib.bucketed_pairs(
        props, window_sentences=window_sentences, max_pairs=max_pairs,
        max_comparisons=max_comparisons, test=prop_lib.negation_conflict)
    attribute_scan = prop_lib.bucketed_pairs(
        props, window_sentences=window_sentences, max_pairs=max_pairs,
        max_comparisons=max_comparisons, test=prop_lib.attribute_conflict)
    return _PropContext(analysis, None, extraction, props, coref_meta, negation_scan, attribute_scan,
                        settings, window_sentences, max_pairs, max_comparisons)


def _coref_note(coref_meta: Mapping[str, Any]) -> str:
    if not coref_meta.get("enabled"):
        return ("pronoun-subject sentences are excluded entirely because "
                "features.coreference_resolution is off (see module docstring)")
    if not coref_meta.get("applied"):
        return ("features.coreference_resolution is on but unavailable this run "
                f"({coref_meta.get('reason')}); pronoun-subject sentences are excluded entirely, "
                "exactly as when the feature is off")
    return (f"features.coreference_resolution resolved {coref_meta.get('resolved_count', 0):,} "
           f"pronoun-subject proposition(s) against a named antecedent within the first "
           f"{coref_meta.get('chars_used', 0):,} characters"
           f"{' (document is longer; the rest was not scanned)' if coref_meta.get('truncated') else ''}"
           "; every resolution is the model's own judgement, not a verified reading")


def _proposition_findings(context: _PropContext, *, max_evidence: int,
                          repeated_assertion_min_words: int) -> list[dict[str, Any]]:
    ids = ["discourse.logic_negation_flip_candidates", "discourse.logic_paragraph_contradiction_rate",
          "discourse.logic_entity_attribute_conflict_candidates", "discourse.logic_repeated_assertion_rate",
          "discourse.logic_new_entity_claim_rate"]
    if context.unavailable_reason:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], context.unavailable_reason, family=FAMILY)
                for metric_id in ids]

    extraction, props, settings = context.extraction, context.props, context.settings
    base_warning = ("candidates only: matched on shared surface subject and predicate, not "
                    "confirmed identity or verified meaning; " + _coref_note(context.coref_meta) +
                    ". Where features.nli_entailment is also on, "
                    "discourse.logic_nli_heuristic_agreement cross-checks these same candidates "
                    "against a real entailment model")
    if extraction.truncated:
        base_warning += f"; proposition extraction stopped at the proposition_cap of {settings['proposition_cap']:,}"

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

    negation_scan, attribute_scan = context.negation_scan, context.attribute_scan
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
    paragraph_total = context.analysis.paragraph_count
    out.append(finding(
        "discourse.logic_paragraph_contradiction_rate",
        _METRIC_NAMES["discourse.logic_paragraph_contradiction_rate"],
        rate(len(same_paragraph), paragraph_total, 100.0) if paragraph_total else None,
        "candidates per 100 paragraphs", family=FAMILY, sample_size=paragraph_total,
        min_sample=20, sample_size_sensitive=True,
        distribution={"candidate_count": len(same_paragraph), "settings": settings},
        evidence=pair_evidence(same_paragraph, max_evidence),
        warning=base_warning if paragraph_total else "no paragraphs to measure"))

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
                "not that either was verified or ordered (enable features.temporal_ordering for that)"))

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
                "similarity model (see discourse.logic_nli_label_distribution's entailment share "
                "for the closest thing this suite offers to that, over a different, capped pool)"
                if eligible else "no proposition had both a non-pronoun subject and an object"))

    out.append(_new_entity_claim_finding(context.analysis, max_evidence, extraction))
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


# ------------------------------------------------------------- NLI adjudication

def _nli_findings(context: _PropContext, *, nli_model: str, nli_max_pairs: int, nli_batch_size: int,
                  max_evidence: int) -> list[dict[str, Any]]:
    """Real entailment/neutral/contradiction scores over a capped candidate pool.

    The candidate pool is the same shared-subject-and-predicate bucketing
    :func:`prop_lib.bucketed_pairs` already uses for the heuristic scans, run
    again with its own, separate ``nli_max_pairs`` cap -- deliberately not
    reusing the heuristic scans' pairs, so this channel also sees pairs
    neither heuristic flagged (a real, if narrow, recall gain: two readings
    that differ in a way ``attribute_conflict``'s exact-value-mismatch test
    cannot see, e.g. genuine paraphrase, can still score ``contradiction`` or
    ``entailment`` here).
    """

    ids = list(FEATURE_METRICS["nli_entailment"])
    if context.unavailable_reason:
        return [unavailable(mid, _METRIC_NAMES[mid], context.unavailable_reason, family=FAMILY) for mid in ids]
    props = context.props
    if not props:
        return [unavailable(mid, _METRIC_NAMES[mid], "no usable propositions", family=FAMILY) for mid in ids]

    def any_distinct_pair(a: prop_lib.Proposition, b: prop_lib.Proposition) -> str | None:
        return None if a.text.strip() == b.text.strip() else "pair"

    # ``max_comparisons`` is the suite-wide search-budget option (cheap, pure
    # Python, no model call), reused as-is; ``nli_max_pairs`` is the separate,
    # much smaller cap on how many of the pairs that search finds are ever
    # actually sent to the model -- the only one of the two that matters for
    # this channel's wall-clock cost. Deriving the search budget FROM
    # nli_max_pairs instead (an earlier version of this code did) starves the
    # search on any document whose matching bucket is large enough to need
    # ``_BUCKET_SAMPLE_CAP`` sampling, before it ever reaches a real pair --
    # found by benchmarking this function on a synthetic book, not assumed.
    scan = prop_lib.bucketed_pairs(
        props, window_sentences=context.window_sentences, max_pairs=nli_max_pairs,
        max_comparisons=context.max_comparisons, test=any_distinct_pair)
    if not scan.pairs:
        warning = ("no eligible candidate pair (two propositions sharing a subject and predicate, "
                   "with distinct text, within window_sentences of each other or the same paragraph)")
        return [unavailable(mid, _METRIC_NAMES[mid], warning, family=FAMILY) for mid in ids]

    pipeline_obj, reason = _load_nli_pipeline(nli_model)
    if pipeline_obj is None:
        return [unavailable(mid, _METRIC_NAMES[mid], reason, family=FAMILY) for mid in ids]

    pairs_text = [(a.text, b.text) for a, b, _ in scan.pairs]
    try:
        labels = _nli_label_scores(pipeline_obj, pairs_text, nli_batch_size)
    except Exception as exc:  # pragma: no cover - runtime failure
        warning = f"NLI scoring failed ({type(exc).__name__}: {exc})"
        return [unavailable(mid, _METRIC_NAMES[mid], warning, family=FAMILY) for mid in ids]

    heuristic_labels = [prop_lib.negation_conflict(a, b) or prop_lib.attribute_conflict(a, b) or "none"
                        for a, b, _ in scan.pairs]
    label_counts = Counter(label for label, _ in labels)
    total = len(labels)
    settings = _settings(window_sentences=context.window_sentences, nli_max_pairs=nli_max_pairs,
                         nli_model=nli_model, nli_batch_size=nli_batch_size,
                         comparisons_examined=scan.comparisons, pairs_capped=scan.pairs_capped,
                         buckets_sampled=scan.buckets_sampled, coreference_resolution=context.coref_meta)
    ranked = sorted(range(total), key=lambda i: -labels[i][1].get("contradiction", 0.0))

    top_evidence = []
    for i in ranked[:max_evidence]:
        a, b, _ = scan.pairs[i]
        label, scores = labels[i]
        top_evidence.append({
            "nli_label": label, "scores": {k: round(v, 4) for k, v in scores.items()},
            "heuristic_label": heuristic_labels[i],
            "sentence_a": {"index": a.sentence_index, "offset": a.offset, "text": a.text},
            "sentence_b": {"index": b.sentence_index, "offset": b.offset, "text": b.text},
        })
    out = [finding(
        "discourse.logic_nli_label_distribution", _METRIC_NAMES["discourse.logic_nli_label_distribution"],
        rate(label_counts.get("contradiction", 0), total, 100.0), "percent labeled contradiction",
        family=FAMILY, sample_size=total, min_sample=5, sample_size_sensitive=True,
        distribution={"counts": dict(label_counts),
                     "percentages": {label: rate(count, total, 100.0) for label, count in label_counts.items()},
                     "model": nli_model, "settings": settings},
        evidence=top_evidence,
        warning=(f"model judgement from {nli_model!r} on each sentence pair in isolation, NOT a "
                "fact about the text: fiction legitimately contains contradictions (unreliable "
                "narrators, lies, hypotheticals, quoted falsehoods), so a high contradiction share "
                "is not evidence of an error in the writing. Evidence is sorted by contradiction "
                "score; see discourse.logic_nli_heuristic_agreement for how these labels compare "
                "with the surface-heuristic candidates"))]

    confusion: dict[str, Counter] = defaultdict(Counter)
    for h, (label, _) in zip(heuristic_labels, labels):
        confusion[h][label] += 1
    flagged = sum(count for h, counts in confusion.items() if h != "none" for count in counts.values())
    corroborated = sum(counts.get("contradiction", 0) for h, counts in confusion.items() if h != "none")
    disagreements = []
    for i in ranked:
        h = heuristic_labels[i]
        label, scores = labels[i]
        if h != "none" and label != "contradiction":
            a, b, _ = scan.pairs[i]
            disagreements.append({
                "heuristic_label": h, "nli_label": label,
                "scores": {k: round(v, 4) for k, v in scores.items()},
                "sentence_a": {"index": a.sentence_index, "text": a.text},
                "sentence_b": {"index": b.sentence_index, "text": b.text},
            })
        if len(disagreements) >= max_evidence:
            break
    out.append(finding(
        "discourse.logic_nli_heuristic_agreement", _METRIC_NAMES["discourse.logic_nli_heuristic_agreement"],
        rate(corroborated, flagged, 100.0),
        "percent of heuristic candidates the NLI model also labels contradiction",
        family=FAMILY, sample_size=flagged, min_sample=5, sample_size_sensitive=True,
        distribution={"confusion": {h: dict(c) for h, c in confusion.items()}, "settings": settings},
        evidence=disagreements,
        warning=(
            "cross-tabulates this suite's own negation-flip/attribute-conflict heuristic labels "
            "against the same pairs' NLI labels (rows: heuristic label including 'none' for a pair "
            "neither heuristic flagged; columns: NLI label); neither channel is ground truth, so a "
            "disagreement means the two signals disagree, not that either is wrong -- a heuristic "
            "flag the model calls 'entailment' or 'neutral' is still worth a second look, not "
            "dismissed, and this project keeps both readings visible rather than picking one") if flagged
            else "no pair in this NLI batch was already flagged by the negation or attribute-conflict "
            "heuristics, so there is nothing to cross-check"))
    return out


# --------------------------------------------------------- WordNet lexical relations

def _wordnet_findings(context: _PropContext, *, max_evidence: int) -> list[dict[str, Any]]:
    ids = list(FEATURE_METRICS["lexical_opposition"])
    if context.unavailable_reason:
        return [unavailable(mid, _METRIC_NAMES[mid], context.unavailable_reason, family=FAMILY) for mid in ids]
    props = context.props
    if not props:
        return [unavailable(mid, _METRIC_NAMES[mid], "no usable propositions", family=FAMILY) for mid in ids]
    wn, reason = prop_lib.load_wordnet()
    if wn is None:
        return [unavailable(mid, _METRIC_NAMES[mid], reason, family=FAMILY) for mid in ids]

    antonym_scan = prop_lib.bucketed_pairs(
        props, window_sentences=context.window_sentences, max_pairs=context.max_pairs,
        max_comparisons=context.max_comparisons, test=prop_lib.wordnet_antonym_conflict)
    settings = _settings(window_sentences=context.window_sentences, max_pairs=context.max_pairs,
                         comparisons_examined=antonym_scan.comparisons,
                         pairs_capped=antonym_scan.pairs_capped, coreference_resolution=context.coref_meta)

    def pair_evidence(pairs, limit) -> list[dict[str, Any]]:
        return [{"type": label, "subject": a.subject_text, "predicate": a.predicate_lemma,
                "sentence_a": {"index": a.sentence_index, "text": a.text, "object": a.object_text},
                "sentence_b": {"index": b.sentence_index, "text": b.text, "object": b.object_text}}
               for a, b, label in pairs[:limit]]

    out = [finding(
        "discourse.logic_wordnet_antonym_candidates", _METRIC_NAMES["discourse.logic_wordnet_antonym_candidates"],
        rate(len(antonym_scan.pairs), len(props), 1000.0), "candidates per 1,000 propositions",
        family=FAMILY, sample_size=len(props), min_sample=100, sample_size_sensitive=True,
        distribution={"candidate_count": len(antonym_scan.pairs), "settings": settings},
        evidence=pair_evidence(antonym_scan.pairs, max_evidence),
        warning="candidates only: WordNet antonymy is checked across every sense of each lemma, not "
                "disambiguated to this sentence's meaning, so an uncommon sense can produce a false "
                "positive; and most true opposites in ordinary prose are not encoded as a direct "
                "WordNet antonym pair at all, so this channel under-reports far more than it "
                "over-reports. Independent of both the negation/attribute heuristics above and of "
                "any NLI model -- a separate, lexical-resource-based signal, not a stronger version "
                "of either")]

    attribute_pairs = context.attribute_scan.pairs if context.attribute_scan else []
    property_pairs = [(a, b) for a, b, label in attribute_pairs if label == "property"]
    downgrade_evidence: list[dict[str, Any]] = []
    downgrade_count = 0
    for a, b in property_pairs:
        if prop_lib.wordnet_hypernym_related(a, b):
            downgrade_count += 1
            if len(downgrade_evidence) < max_evidence:
                downgrade_evidence.append({
                    "subject": a.subject_text, "predicate": a.predicate_lemma,
                    "sentence_a": {"index": a.sentence_index, "text": a.text, "object": a.object_text},
                    "sentence_b": {"index": b.sentence_index, "text": b.text, "object": b.object_text}})
    out.append(finding(
        "discourse.logic_wordnet_hypernym_downgrade_rate",
        _METRIC_NAMES["discourse.logic_wordnet_hypernym_downgrade_rate"],
        rate(downgrade_count, len(property_pairs), 100.0) if property_pairs else None, "percent",
        family=FAMILY, sample_size=len(property_pairs), min_sample=5, sample_size_sensitive=True,
        distribution={"downgrade_count": downgrade_count, "property_candidate_count": len(property_pairs)},
        evidence=downgrade_evidence,
        warning=(
            "informational only: never subtracts from discourse.logic_entity_attribute_conflict_"
            "candidates itself, which is kept exactly as it was for stability regardless of whether "
            "WordNet is installed. Flags this suite's own 'property'-subtype attribute-conflict "
            "candidates whose two object readings sit in a WordNet hypernym/hyponym relation "
            "('dog'/'poodle'), which is usually not a real contradiction; a human should still look, "
            "since an is-a relation does not rule out the sentence meaning something else here")
            if property_pairs else "no 'property'-subtype attribute-conflict candidates to check "
            "(needs features.propositions on, and at least one such pair)"))
    return out


# -------------------------------------------------- VerbNet / FrameNet consistency
#
# Both ask the same shape of question -- "two DIFFERENT verbs used for the
# SAME two participants: do they belong to any shared semantic class/frame at
# all" -- against a different lexical resource, via
# :func:`prop_lib.participant_key`'s (subject_key, object_key) bucketing
# rather than :func:`prop_lib.bucketed_pairs`' default (subject_key,
# predicate_lemma) one. See :mod:`textgrader.propositions`' own section
# comment for the worked build/destroy (flagged, no shared class or frame) and
# open/close (not flagged, one shared class and one shared frame) examples
# this design was checked against, not assumed from either resource's own
# documentation.

def _verbnet_findings(context: _PropContext, *, max_evidence: int) -> list[dict[str, Any]]:
    metric_id = "discourse.logic_verbnet_class_conflict_candidates"
    if context.unavailable_reason:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], context.unavailable_reason, family=FAMILY)]
    props = context.props
    if not props:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], "no usable propositions", family=FAMILY)]
    vn_module, reason = prop_lib.load_verbnet()
    if vn_module is None:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], reason, family=FAMILY)]

    scan = prop_lib.bucketed_pairs(
        props, window_sentences=context.window_sentences, max_pairs=context.max_pairs,
        max_comparisons=context.max_comparisons, test=prop_lib.verbnet_class_conflict,
        key=prop_lib.participant_key)
    settings = _settings(window_sentences=context.window_sentences, max_pairs=context.max_pairs,
                         comparisons_examined=scan.comparisons, pairs_capped=scan.pairs_capped,
                         buckets_sampled=scan.buckets_sampled, coreference_resolution=context.coref_meta)
    evidence = [{
        "subject": a.subject_text, "object": a.object_text,
        "verb_a": a.predicate_lemma, "verb_b": b.predicate_lemma,
        "verbnet_classes_a": sorted(prop_lib.verbnet_classes(vn_module, a.predicate_lemma)),
        "verbnet_classes_b": sorted(prop_lib.verbnet_classes(vn_module, b.predicate_lemma)),
        "sentence_a": {"index": a.sentence_index, "text": a.text},
        "sentence_b": {"index": b.sentence_index, "text": b.text},
    } for a, b, _label in scan.pairs[:max_evidence]]
    return [finding(
        metric_id, _METRIC_NAMES[metric_id], rate(len(scan.pairs), len(props), 1000.0),
        "candidates per 1,000 propositions", family=FAMILY, sample_size=len(props), min_sample=100,
        sample_size_sensitive=True,
        distribution={"candidate_count": len(scan.pairs), "settings": settings},
        evidence=evidence,
        warning=(
            "candidates only: flags two DIFFERENT verbs used for the SAME two participants (a shared "
            "non-pronoun subject and non-pronoun object) whose VerbNet classes share nothing at all -- "
            "a class-level 'this does not read as the same kind of event' signal, not a contradiction "
            "verdict. A verb VerbNet has no entry for makes a pair unjudgeable, never flagged, so this "
            "under-reports at least as often as it over-reports; two verbs from unrelated classes can "
            "still describe a perfectly coherent sequence of events (a character can build something "
            "and later, truthfully, destroy it), so a human should read the actual sentences, not treat "
            "a candidate here as an error"))]


def _framenet_findings(context: _PropContext, *, max_evidence: int) -> list[dict[str, Any]]:
    metric_id = "discourse.logic_framenet_frame_conflict_candidates"
    if context.unavailable_reason:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], context.unavailable_reason, family=FAMILY)]
    props = context.props
    if not props:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], "no usable propositions", family=FAMILY)]
    fn_module, reason = prop_lib.load_framenet()
    if fn_module is None:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], reason, family=FAMILY)]

    scan = prop_lib.bucketed_pairs(
        props, window_sentences=context.window_sentences, max_pairs=context.max_pairs,
        max_comparisons=context.max_comparisons, test=prop_lib.framenet_frame_conflict,
        key=prop_lib.participant_key)
    settings = _settings(window_sentences=context.window_sentences, max_pairs=context.max_pairs,
                         comparisons_examined=scan.comparisons, pairs_capped=scan.pairs_capped,
                         buckets_sampled=scan.buckets_sampled, coreference_resolution=context.coref_meta)
    evidence = [{
        "subject": a.subject_text, "object": a.object_text,
        "verb_a": a.predicate_lemma, "verb_b": b.predicate_lemma,
        "framenet_frames_a": sorted(prop_lib.framenet_frames(fn_module, a.predicate_lemma)),
        "framenet_frames_b": sorted(prop_lib.framenet_frames(fn_module, b.predicate_lemma)),
        "sentence_a": {"index": a.sentence_index, "text": a.text},
        "sentence_b": {"index": b.sentence_index, "text": b.text},
    } for a, b, _label in scan.pairs[:max_evidence]]
    return [finding(
        metric_id, _METRIC_NAMES[metric_id], rate(len(scan.pairs), len(props), 1000.0),
        "candidates per 1,000 propositions", family=FAMILY, sample_size=len(props), min_sample=100,
        sample_size_sensitive=True,
        distribution={"candidate_count": len(scan.pairs), "settings": settings},
        evidence=evidence,
        warning=(
            "candidates only: flags two DIFFERENT verbs used for the SAME two participants (a shared "
            "non-pronoun subject and non-pronoun object) whose FrameNet lexical units evoke no frame in "
            "common -- an independent lexical resource asking the same shape of question as "
            "discourse.logic_verbnet_class_conflict_candidates, kept as its own channel rather than "
            "merged with it because the two agreeing is corroboration and either one flagging alone is "
            "still data worth keeping visible (see module docstring). A verb FrameNet has no lexical "
            "unit for makes a pair unjudgeable, never flagged; a candidate here is a 'read the actual "
            "sentences' prompt, not a verdict -- describing the same participants through two "
            "semantically distant verbs is not automatically an inconsistency"))]


# ------------------------------------------------------------- temporal ordering

def _temporal_findings(context: _PropContext, *, max_evidence: int) -> list[dict[str, Any]]:
    """Which of two differing dates on the same subject+predicate is earlier.

    Reuses :attr:`_PropContext.attribute_scan`'s ``temporal``-subtype pairs
    rather than generating its own -- those are already exactly "two
    propositions, same subject and predicate, differing years/dates"; this
    group only adds ordering on top, via ``dateutil``, plus a check against
    where each proposition sits in the narrative.
    """

    metric_id = "discourse.logic_temporal_order_candidates"
    if context.unavailable_reason:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], context.unavailable_reason, family=FAMILY)]
    attribute_pairs = context.attribute_scan.pairs if context.attribute_scan else []
    temporal_pairs = [(a, b) for a, b, label in attribute_pairs if label == "temporal"]
    if not temporal_pairs:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id],
                            "no 'temporal'-subtype attribute-conflict candidates to order (needs "
                            "features.propositions on, and at least one such pair)", family=FAMILY)]
    parser_module, reason = require("dateutil")
    if parser_module is None:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], reason, family=FAMILY)]

    def parse(text: str):
        try:
            return parser_module.parse(text, default=_TEMPORAL_PARSE_DEFAULT)
        except (ValueError, OverflowError, TypeError):
            return None

    reversed_count = 0
    parsed_count = 0
    evidence: list[dict[str, Any]] = []
    for a, b in temporal_pairs:
        date_a, date_b = parse(a.object_text or ""), parse(b.object_text or "")
        if date_a is None or date_b is None or date_a == date_b:
            continue
        parsed_count += 1
        if date_a < date_b:
            earlier, later, earlier_date, later_date = a, b, date_a, date_b
        else:
            earlier, later, earlier_date, later_date = b, a, date_b, date_a
        if earlier.sentence_index > later.sentence_index:
            reversed_count += 1
            if len(evidence) < max_evidence:
                evidence.append({
                    "subject": earlier.subject_text, "predicate": earlier.predicate_lemma,
                    "earlier": {"sentence_index": earlier.sentence_index, "text": earlier.text,
                                "date_text": earlier.object_text,
                                "parsed_date": earlier_date.date().isoformat()},
                    "later": {"sentence_index": later.sentence_index, "text": later.text,
                             "date_text": later.object_text, "parsed_date": later_date.date().isoformat()},
                })
    if not parsed_count:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id],
                            "dateutil could not parse either value in any temporal-conflict pair",
                            family=FAMILY)]
    return [finding(
        metric_id, _METRIC_NAMES[metric_id], rate(reversed_count, parsed_count, 100.0), "percent",
        family=FAMILY, sample_size=parsed_count, min_sample=5, sample_size_sensitive=True,
        distribution={"reversed_count": reversed_count, "temporal_pairs_parsed": parsed_count,
                     "temporal_pairs_seen": len(temporal_pairs)},
        evidence=evidence,
        warning=(
            "'reversed' means the chronologically earlier date's sentence appears later in the text "
            "than the chronologically later date's sentence -- a narrative-order observation, NOT an "
            "error: flashbacks, foreshadowing, frame stories and other non-chronological narration "
            "all produce this legitimately and often deliberately. Dates are parsed with dateutil "
            "from a single object token (a bare year or short date string) with no sentence context, "
            "so an unrelated number could occasionally be mis-read as a date"))]


# ----------------------------------------------------------- semantic role labelling

def _srl_findings(analysis: DocumentAnalysis, *, srl_model: str, srl_max_predicates: int,
                  max_evidence: int) -> list[dict[str, Any]]:
    """Real PropBank-style SRL, and two channels the spec asks for on top of it.

    Independent of ``propositions``: this re-walks the shared spaCy parse for
    its own clause-level predicates (see :func:`prop_lib.extract_srl_frames`)
    rather than reusing ``context.props``, so it works whether or not
    ``features.propositions`` is on and its own ``srl_max_predicates`` cap is
    never distorted by ``proposition_cap``.
    """

    ids = list(FEATURE_METRICS["semantic_role_labeling"])
    extraction = prop_lib.extract_srl_frames(analysis, srl_max_predicates, srl_model)
    if not extraction.available:
        return [unavailable(mid, _METRIC_NAMES[mid], extraction.reason, family=FAMILY) for mid in ids]
    frames = extraction.frames
    settings = _settings(srl_model=srl_model, srl_max_predicates=srl_max_predicates,
                         candidates_seen=extraction.candidates_seen, truncated=extraction.truncated,
                         sentences_scanned=extraction.sentences_scanned)

    by_predicate: dict[str, list[prop_lib.SRLFrame]] = defaultdict(list)
    for f in frames:
        by_predicate[f.predicate_lemma].append(f)
    eligible = {lemma: fs for lemma, fs in by_predicate.items() if len(fs) >= 2}
    inconsistent = 0
    consistency_evidence: list[dict[str, Any]] = []
    for lemma, fs in eligible.items():
        signatures = Counter(prop_lib.srl_core_roles(f.roles) for f in fs)
        _majority_sig, majority_count = signatures.most_common(1)[0]
        if majority_count < len(fs):
            inconsistent += 1
            if len(consistency_evidence) < max_evidence:
                consistency_evidence.append({
                    "predicate": lemma, "occurrences": len(fs),
                    "signatures": [{"core_roles": sorted(sig), "count": count}
                                  for sig, count in signatures.items()],
                    "examples": [{"sentence_index": f.sentence_index, "text": f.text, "roles": dict(f.roles)}
                                for f in fs[:4]],
                })
    out = [finding(
        "discourse.logic_srl_role_pattern_consistency", _METRIC_NAMES["discourse.logic_srl_role_pattern_consistency"],
        rate(inconsistent, len(eligible), 100.0) if eligible else None, "percent",
        family=FAMILY, sample_size=len(eligible), min_sample=5, sample_size_sensitive=True,
        distribution={"inconsistent_predicate_count": inconsistent, "predicate_groups_examined": len(eligible),
                     "settings": settings},
        evidence=consistency_evidence,
        warning=(
            f"candidate signal, not an error: 'inconsistent' means {srl_model!r}'s own generated "
            "core-argument role signature (which ARG-0..ARG-4 slots it filled) differs across two or "
            "more mentions of the same predicate lemma in this document. Active/passive alternation "
            "(the same referent surfacing as ARG-0 in one mention and being omitted, or the sentence "
            "eliding an argument, in another), a different sense of the same verb, or a real "
            "inconsistency in what the text asserts can all produce this -- the finding cannot tell "
            "them apart, only surface the disagreement. Every role is this model's own generated "
            "reading, never a verified PropBank annotation" if eligible else
            "no predicate lemma had two or more SRL-scored occurrences to compare within "
            "srl_max_predicates; raise it to see repeats in a longer document"))]

    thin_frames = [f for f in frames if len(prop_lib.srl_core_roles(f.roles)) < 2]
    omission_evidence = [{"sentence_index": f.sentence_index, "text": f.text, "predicate": f.predicate_text,
                          "roles": dict(f.roles)} for f in thin_frames[:max_evidence]]
    out.append(finding(
        "discourse.logic_srl_argument_omission_rate", _METRIC_NAMES["discourse.logic_srl_argument_omission_rate"],
        rate(len(thin_frames), len(frames), 100.0) if frames else None, "percent",
        family=FAMILY, sample_size=len(frames), min_sample=10, sample_size_sensitive=True,
        distribution={"thin_frame_count": len(thin_frames), "frames_scored": len(frames), "settings": settings},
        evidence=omission_evidence,
        warning=(
            "candidate signal, not an error: counts SRL frames the model filled with fewer than two "
            "core arguments (ARG-0..ARG-4), a 'thin' predication. Many verbs are legitimately "
            "intransitive, passive with an unstated agent, or elliptical in dialogue, so a high rate "
            "is not itself a defect in the writing. No PropBank per-verb frame file is consulted, so "
            f"this never claims a specific argument was 'required' and skipped -- only that "
            f"{srl_model!r} filled fewer than two role slots for that predicate" if frames else
            "no SRL frame was scored (no extractable predicate within srl_max_predicates)")))
    return out


# ---------------------------------------------- PropBank roleset argument structure

def _propbank_findings(analysis: DocumentAnalysis, *, srl_model: str, srl_max_predicates: int,
                       max_evidence: int) -> list[dict[str, Any]]:
    """The principled version of ``discourse.logic_srl_argument_omission_rate``:
    instead of a fixed "fewer than two roles" threshold, compares each SRL
    frame against the specific PropBank roleset (``give.01``'s giver/thing-
    given/recipient, not a generic count) that best explains what the model
    already filled, via :func:`prop_lib.propbank_best_roleset`.

    Re-extracts SRL frames through the same
    :func:`prop_lib.extract_srl_frames` cache ``semantic_role_labeling`` uses
    -- with the same ``srl_model``/``srl_max_predicates`` (this feature reuses
    those two options rather than defining its own), the two features share
    one model call, never two, whether one or both are on.

    The PropBank corpus check runs FIRST, before the (far more expensive) SRL
    model is ever loaded: there is no point paying for a model generation call
    whose output cannot be interpreted without the roleset data, and a missing
    corpus should degrade instantly regardless of whether transformers is even
    installed here.
    """

    metric_id = "discourse.logic_propbank_argument_omission_rate"
    pb_module, reason = prop_lib.load_propbank()
    if pb_module is None:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], reason, family=FAMILY)]
    extraction = prop_lib.extract_srl_frames(analysis, srl_max_predicates, srl_model)
    if not extraction.available:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id], extraction.reason, family=FAMILY)]
    frames = extraction.frames
    if not frames:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id],
                            "no SRL frame was scored (no extractable predicate within "
                            "srl_max_predicates)", family=FAMILY)]

    settings = _settings(srl_model=srl_model, srl_max_predicates=srl_max_predicates,
                         candidates_seen=extraction.candidates_seen, truncated=extraction.truncated,
                         sentences_scanned=extraction.sentences_scanned)
    matched = 0
    omitted = 0
    evidence: list[dict[str, Any]] = []
    for f in frames:
        filled = prop_lib.srl_core_roles(f.roles)
        match = prop_lib.propbank_best_roleset(pb_module, f.predicate_lemma, filled)
        if match is None:
            continue
        matched += 1
        roleset_id, expected = match
        filled_numbers = {number for number in (prop_lib.arg_number(role) for role in filled)
                          if number is not None}
        missing = expected - filled_numbers
        if missing:
            omitted += 1
        evidence.append({
            "sentence_index": f.sentence_index, "text": f.text, "predicate": f.predicate_text,
            "roleset": roleset_id, "expected_core_roles": sorted(expected),
            "filled_roles": dict(f.roles), "missing_core_roles": sorted(missing),
        })
    if not matched:
        return [unavailable(metric_id, _METRIC_NAMES[metric_id],
                            "no scored predicate's lemma had a matching PropBank roleset (this "
                            "environment's propbank frame files have no entry for any verb SRL scored "
                            "within srl_max_predicates)", family=FAMILY)]
    evidence.sort(key=lambda row: -len(row["missing_core_roles"]))
    return [finding(
        metric_id, _METRIC_NAMES[metric_id], rate(omitted, matched, 100.0), "percent",
        family=FAMILY, sample_size=matched, min_sample=5, sample_size_sensitive=True,
        distribution={"omitted_count": omitted, "matched_frames": matched, "frames_scored": len(frames),
                     "settings": settings},
        evidence=evidence[:max_evidence],
        warning=(
            f"candidate signal, not an error: for each SRL frame, the PropBank roleset (from nltk's "
            f"downloaded propbank corpus) whose declared numbered arguments overlap most with what "
            f"{srl_model!r} already filled is picked as this occurrence's roleset -- a heuristic match, "
            "not real word-sense disambiguation, so a rarer sense can be missed. 'Omitted' means that "
            "roleset declares a numbered core argument (ARG-0..ARG-4) the model's frame does not fill. "
            "A deliberate stylistic ellipsis ('He gave.' as a sentence fragment, elliptical dialogue, a "
            "reader-inferred argument) produces exactly the same pattern as a genuinely dropped "
            "argument, so a high rate is not itself a defect in the writing -- and because the roleset "
            "match is itself picked from the same filled-role overlap being measured, an 'omission' can "
            "also just mean a different, still-valid roleset would have matched with less missing"))]


# ------------------------------------------------- closed-schema relation extraction

def _relation_extraction_findings(analysis: DocumentAnalysis, context: "_PropContext | None", *,
                                  relation_extraction_model: str, relation_extraction_max_sentences: int,
                                  max_evidence: int) -> list[dict[str, Any]]:
    """A REBEL-backed cross-check, kept beside (never merged into)
    ``propositions``' own dependency-parse triples -- see
    :func:`prop_lib.extract_relations` for why REBEL is closed-schema relation
    extraction, not open-domain OpenIE, and why that distinction matters here.
    """

    ids = list(FEATURE_METRICS["relation_extraction"])
    extraction = prop_lib.extract_relations(analysis, relation_extraction_max_sentences, relation_extraction_model)
    if not extraction.available:
        return [unavailable(mid, _METRIC_NAMES[mid], extraction.reason, family=FAMILY) for mid in ids]

    settings = _settings(relation_extraction_model=relation_extraction_model,
                         relation_extraction_max_sentences=relation_extraction_max_sentences,
                         sentences_scanned=extraction.sentences_scanned,
                         sentences_scored=extraction.sentences_scored, truncated=extraction.truncated)
    relations = extraction.relations
    evidence = [{"sentence_index": r.sentence_index, "text": r.text, "head": r.head,
                "relation_type": r.relation_type, "tail": r.tail} for r in relations[:max_evidence]]
    out = [finding(
        "discourse.logic_relation_extraction_triple_rate",
        _METRIC_NAMES["discourse.logic_relation_extraction_triple_rate"],
        rate(len(relations), extraction.sentences_scored, 100.0) if extraction.sentences_scored else None,
        "triples per 100 sentences scored", family=FAMILY, sample_size=extraction.sentences_scored,
        min_sample=10, sample_size_sensitive=True,
        distribution={"triple_count": len(relations), "settings": settings}, evidence=evidence,
        warning=(
            f"{relation_extraction_model!r} is CLOSED-schema relation extraction (roughly 200 "
            "Wikidata-style relation types such as 'capital of' or 'spouse'), not open-domain "
            "arbitrary-predicate OpenIE the way Stanford OpenIE or AllenNLP's OpenIE predictor work "
            "(neither is available here -- see module Deferred notes for the quoted evidence). This "
            "is not simply a channel that stays quiet on fiction: hand-checked on real text, an "
            "invented sentence stating nothing about where a character lives still generated a "
            "plausible-sounding 'residence' triple the text never asserts, a schema-compatible "
            "guess invented under the seq2seq format's own pressure to always emit something. So "
            "this rate is never comparable to a 'should have relations' expectation in either "
            "direction, and it is never combined with discourse.logic_entity_attribute_conflict_"
            "candidates' own count. Every triple is this model's own generated reading, never a "
            "verified fact"))]

    props = context.props if (context is not None and not context.unavailable_reason) else []
    if not relations:
        out.append(unavailable(
            "discourse.logic_relation_extraction_proposition_overlap_rate",
            _METRIC_NAMES["discourse.logic_relation_extraction_proposition_overlap_rate"],
            "no relation triple was extracted to cross-check", family=FAMILY))
    elif not props:
        out.append(unavailable(
            "discourse.logic_relation_extraction_proposition_overlap_rate",
            _METRIC_NAMES["discourse.logic_relation_extraction_proposition_overlap_rate"],
            "needs features.propositions on, for a comparison pool of subject/object pairs", family=FAMILY))
    else:
        subject_object_pairs = {(p.subject_text.lower(), p.object_text.lower())
                                for p in props if p.object_text}

        def corroborated(head: str, tail: str) -> bool:
            head_l, tail_l = head.lower(), tail.lower()
            return any((head_l in s or s in head_l) and (tail_l in o or o in tail_l)
                      for s, o in subject_object_pairs)

        overlap_evidence = []
        hits = 0
        for r in relations:
            is_hit = corroborated(r.head, r.tail)
            hits += int(is_hit)
            if len(overlap_evidence) < max_evidence:
                overlap_evidence.append({"sentence_index": r.sentence_index, "head": r.head,
                                         "relation_type": r.relation_type, "tail": r.tail,
                                         "corroborated_by_propositions": is_hit})
        out.append(finding(
            "discourse.logic_relation_extraction_proposition_overlap_rate",
            _METRIC_NAMES["discourse.logic_relation_extraction_proposition_overlap_rate"],
            rate(hits, len(relations), 100.0), "percent", family=FAMILY, sample_size=len(relations),
            min_sample=5, sample_size_sensitive=True,
            distribution={"corroborated_count": hits, "triple_count": len(relations), "settings": settings},
            evidence=overlap_evidence,
            warning=(
                "percent of this model's relation triples whose head and tail both loosely match "
                "(case-insensitive substring, not identity) a subject/object pair "
                "discourse.propositions' dependency-parse proxy also extracted in this document -- "
                "two independently-built extractors agreeing is a corroboration signal, not proof of "
                "either being correct; disagreeing is data too (different sentence coverage, "
                "different schema), never treated as either channel being wrong")))
    return out


# --------------------------------------------------------------- argument mining

_ARGMIN_MODEL_CACHE: dict[str, tuple[Any, str | None]] = {}


def _reset_argmin_cache() -> None:
    _ARGMIN_MODEL_CACHE.clear()


on_reset(_reset_argmin_cache)


def _load_argument_mining_pipeline(model_name: str) -> tuple[Any, str | None]:
    if model_name in _ARGMIN_MODEL_CACHE:
        return _ARGMIN_MODEL_CACHE[model_name]
    module, reason = require("transformers")
    if module is None:
        _ARGMIN_MODEL_CACHE[model_name] = (None, reason)
        return _ARGMIN_MODEL_CACHE[model_name]
    try:
        pipeline_obj = module.pipeline("text-classification", model=model_name, device=-1, top_k=None)
        outcome: tuple[Any, str | None] = (pipeline_obj, None)
    except Exception as exc:  # pragma: no cover - model download/runtime failure
        outcome = (None, f"argument-mining model {model_name!r} unavailable ({type(exc).__name__}: {exc}); "
                         f"pip install transformers torch")
    _ARGMIN_MODEL_CACHE[model_name] = outcome
    return outcome


def _argument_candidate_pairs(analysis: DocumentAnalysis, min_words: int,
                              max_pairs: int) -> list[tuple[str, int, str, str, str]]:
    """``(polarity, sentence_index, premise_text, conclusion_text, marker)`` for
    the same therefore/because/however-style connectives ``connective_relations``
    scores, filtered the same way (``min_words`` content words on each side).

    Deliberately its own scan rather than a refactor of :func:`_relation_scores`:
    ``connective_relations``' surface candidate counts are pinned stable
    (see the module docstring) across every group added since, and this
    channel needs its own, independently-capped pool regardless of whether
    ``connective_relations`` itself is even on.
    """

    sentences = analysis.sentences
    pairs: list[tuple[str, int, str, str, str]] = []
    for polarity, pattern in (("support", THEREFORE_PATTERN), ("support", BECAUSE_PATTERN),
                             ("contrast", CONTRAST_PATTERN)):
        for index, sentence in enumerate(sentences):
            if len(pairs) >= max_pairs:
                return pairs
            match = pattern.search(sentence)
            if not match:
                continue
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
            if len(_content_words(left_words)) < min_words or len(_content_words(right_words)) < min_words:
                continue
            pairs.append((polarity, index, left_text.strip(), right_text.strip(), match.group(0).lower()))
    return pairs


def _argument_mining_findings(analysis: DocumentAnalysis, *, argument_mining_model: str,
                              argument_mining_max_pairs: int, connective_min_words: int,
                              max_evidence: int) -> list[dict[str, Any]]:
    """A real argument-relation classifier over this suite's own connective-linked
    clause pairs -- claim/premise/support/attack from a model, not a proxy.

    ``connective_chain_length`` (see ``connective_relations``) is unchanged and
    kept under its own honest, narrower name: a marker-adjacency proxy for
    argument structure built from nothing but sentence position. This group
    is the real thing it stood in for, offered alongside it, never replacing it.
    """

    ids = list(FEATURE_METRICS["argument_mining"])
    pairs = _argument_candidate_pairs(analysis, connective_min_words, argument_mining_max_pairs)
    if not pairs:
        warning = ("no therefore/because/however-style connective pair passed "
                   "connective_min_words content words on both sides")
        return [unavailable(mid, _METRIC_NAMES[mid], warning, family=FAMILY) for mid in ids]

    pipeline_obj, reason = _load_argument_mining_pipeline(argument_mining_model)
    if pipeline_obj is None:
        return [unavailable(mid, _METRIC_NAMES[mid], reason, family=FAMILY) for mid in ids]

    pairs_text = [(premise, conclusion) for _polarity, _index, premise, conclusion, _marker in pairs]
    try:
        labels = _nli_label_scores(pipeline_obj, pairs_text, 8)
    except Exception as exc:  # pragma: no cover - runtime failure
        warning = f"argument-mining scoring failed ({type(exc).__name__}: {exc})"
        return [unavailable(mid, _METRIC_NAMES[mid], warning, family=FAMILY) for mid in ids]

    label_counts = Counter(label for label, _scores in labels)
    total = len(labels)
    related = total - label_counts.get("no-relation", 0)
    settings = _settings(argument_mining_model=argument_mining_model,
                         argument_mining_max_pairs=argument_mining_max_pairs,
                         connective_min_words=connective_min_words)
    top_evidence = [{
        "marker": marker, "connective_polarity": polarity, "argmin_label": label,
        "scores": {k: round(v, 4) for k, v in scores.items()}, "premise": premise, "conclusion": conclusion,
    } for (polarity, _index, premise, conclusion, marker), (label, scores)
      in list(zip(pairs, labels))[:max_evidence]]

    out = [finding(
        "discourse.logic_argument_relation_label_distribution",
        _METRIC_NAMES["discourse.logic_argument_relation_label_distribution"],
        rate(related, total, 100.0),
        "percent of connective-linked pairs the model labels as some argument relation (not No-Relation)",
        family=FAMILY, sample_size=total, min_sample=5, sample_size_sensitive=True,
        distribution={"counts": dict(label_counts),
                     "percentages": {label: rate(count, total, 100.0) for label, count in label_counts.items()},
                     "model": argument_mining_model, "settings": settings},
        evidence=top_evidence,
        warning=(
            f"{argument_mining_model!r} is a real argument-relation classifier (RoBERTa-large, "
            "trained on the US2016 and QT30 debate corpora for Argument Relation Identification) "
            "applied here to this suite's own therefore/because/however-adjacent clause pairs, "
            "labelling each pair No-Relation / Inference (support) / Conflict (attack) / Rephrase -- "
            "genuine claim/premise support-attack labelling, not the connective_chain_length proxy "
            "(kept, unchanged, as its own metric; see module Deferred notes). A model label is a "
            "judgement about these two clauses in isolation, never a verified logical relationship, "
            "and which clause is 'premise' vs. 'conclusion' is this suite's own structural guess from "
            "which side of the connective it sits on, not confirmed by the model or anything else"))]

    confusion: dict[str, Counter] = defaultdict(Counter)
    expected_match = 0
    for (polarity, *_rest), (label, _scores) in zip(pairs, labels):
        confusion[polarity][label] += 1
        if (polarity == "support" and label == "inference") or (polarity == "contrast" and label == "conflict"):
            expected_match += 1
    out.append(finding(
        "discourse.logic_argument_relation_connective_agreement",
        _METRIC_NAMES["discourse.logic_argument_relation_connective_agreement"],
        rate(expected_match, total, 100.0),
        "percent where the connective's implied polarity (support: therefore/because; "
        "contrast: however/but) matches the model's label (Inference/Conflict respectively)",
        family=FAMILY, sample_size=total, min_sample=5, sample_size_sensitive=True,
        distribution={"confusion": {polarity: dict(counts) for polarity, counts in confusion.items()},
                     "settings": settings},
        evidence=top_evidence,
        warning=(
            "cross-tabulates this suite's own connective-polarity guess (rows: 'support' for "
            "therefore/because, 'contrast' for however/but) against the argument-mining model's "
            "label (columns); neither is ground truth, so a disagreement means the two signals "
            "disagree, not that either is wrong -- 'however' often separates two independent claims "
            "without one attacking the other, which the model is free to call No-Relation or Rephrase")))
    return out


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
    "nli_entailment": (
        "discourse.logic_nli_label_distribution", "discourse.logic_nli_heuristic_agreement"),
    "lexical_opposition": (
        "discourse.logic_wordnet_antonym_candidates", "discourse.logic_wordnet_hypernym_downgrade_rate"),
    "temporal_ordering": ("discourse.logic_temporal_order_candidates",),
    "semantic_role_labeling": (
        "discourse.logic_srl_role_pattern_consistency", "discourse.logic_srl_argument_omission_rate"),
    "relation_extraction": (
        "discourse.logic_relation_extraction_triple_rate",
        "discourse.logic_relation_extraction_proposition_overlap_rate"),
    "argument_mining": (
        "discourse.logic_argument_relation_label_distribution",
        "discourse.logic_argument_relation_connective_agreement"),
    "propbank_argument_structure": ("discourse.logic_propbank_argument_omission_rate",),
    "verbnet_class_consistency": ("discourse.logic_verbnet_class_conflict_candidates",),
    "framenet_frame_consistency": ("discourse.logic_framenet_frame_conflict_candidates",),
}

#: Feature groups that were part of the module's original, dependency-free
#: pass and stay on unless a config explicitly turns them off. Every group
#: added since (NLI, coreference, WordNet, temporal ordering) either loads a
#: heavy optional model or, per the task that added it, must be opt-in
#: regardless of cost -- so it defaults to *off* when a ``features`` mapping
#: does not mention it, not to *on* like the four groups below. This is the
#: one thing standing between an NLI model and a book nobody asked to run it
#: against (see the module docstring's "Gating" section): getting it backwards
#: would turn on a transformer-model pass for anyone who calls this module's
#: ``measure`` with ``config=None`` or with a partial ``features`` mapping.
_ON_BY_DEFAULT = frozenset({
    "negation_and_quantifiers", "connective_relations", "propositions", "modal_argument_position",
})

#: ``features`` keys that are read directly as plain booleans (no metric ids
#: of their own, so they are not in ``FEATURE_METRICS``/``_disabled``): they
#: only change what feeds the groups above. Off by default, like every new
#: group; listed here purely so ``measure`` and the tests have one place that
#: enumerates every key a config's ``features`` mapping may set.
_MODIFIER_FEATURES = frozenset({"coreference_resolution"})


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
    coreference_max_chars = int(option(config, "coreference_max_chars", 20_000))
    nli_model = str(option(config, "nli_model", "cross-encoder/nli-deberta-v3-small"))
    nli_max_pairs = int(option(config, "nli_max_pairs", 60))
    nli_batch_size = int(option(config, "nli_batch_size", 16))
    srl_model = str(option(config, "srl_model", "cu-kairos/propbank_srl_seq2seq_t5_small"))
    srl_max_predicates = int(option(config, "srl_max_predicates", 40))
    relation_extraction_model = str(option(config, "relation_extraction_model", "Babelscape/rebel-large"))
    relation_extraction_max_sentences = int(option(config, "relation_extraction_max_sentences", 40))
    argument_mining_model = str(option(config, "argument_mining_model",
                                       "raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L"))
    argument_mining_max_pairs = int(option(config, "argument_mining_max_pairs", 40))

    def on(name: str) -> bool:
        fallback = name in _ON_BY_DEFAULT
        value = features.get(name, fallback) if isinstance(features, Mapping) else fallback
        return value is not False

    out: list[dict[str, Any]] = []
    out.extend(_negation_and_quantifiers(analysis, max_evidence) if on("negation_and_quantifiers")
              else _disabled("negation_and_quantifiers"))
    out.extend(_connective_relations(analysis, connective_min_words, max_evidence)
              if on("connective_relations") else _disabled("connective_relations"))

    # Built once, whether one or several of {propositions, nli_entailment,
    # lexical_opposition, temporal_ordering} are on, so extraction, optional
    # coreference resolution and the two heuristic bucket scans are each paid
    # for at most once per document. Never built at all -- so never even
    # inspecting analysis.nlp_unavailable's spaCy-parse cost -- when none of
    # the four groups that need it is enabled.
    needs_props = (on("propositions") or on("nli_entailment") or on("lexical_opposition")
                  or on("temporal_ordering") or on("verbnet_class_consistency")
                  or on("framenet_frame_consistency"))
    context = _build_proposition_context(
        analysis, proposition_cap=proposition_cap, window_sentences=window_sentences,
        max_pairs=max_pairs, max_comparisons=max_comparisons,
        coreference_enabled=on("coreference_resolution"),
        coreference_max_chars=coreference_max_chars) if needs_props else None

    out.extend(_proposition_findings(
        context, max_evidence=max_evidence, repeated_assertion_min_words=repeated_assertion_min_words)
        if on("propositions") else _disabled("propositions"))
    out.extend([_modal_density_near_connectives(analysis)] if on("modal_argument_position")
              else _disabled("modal_argument_position"))
    out.extend(_nli_findings(context, nli_model=nli_model, nli_max_pairs=nli_max_pairs,
                             nli_batch_size=nli_batch_size, max_evidence=max_evidence)
              if on("nli_entailment") else _disabled("nli_entailment"))
    out.extend(_wordnet_findings(context, max_evidence=max_evidence)
              if on("lexical_opposition") else _disabled("lexical_opposition"))
    out.extend(_temporal_findings(context, max_evidence=max_evidence)
              if on("temporal_ordering") else _disabled("temporal_ordering"))
    out.extend(_srl_findings(analysis, srl_model=srl_model, srl_max_predicates=srl_max_predicates,
                             max_evidence=max_evidence)
              if on("semantic_role_labeling") else _disabled("semantic_role_labeling"))
    out.extend(_relation_extraction_findings(
        analysis, context, relation_extraction_model=relation_extraction_model,
        relation_extraction_max_sentences=relation_extraction_max_sentences, max_evidence=max_evidence)
        if on("relation_extraction") else _disabled("relation_extraction"))
    out.extend(_argument_mining_findings(
        analysis, argument_mining_model=argument_mining_model,
        argument_mining_max_pairs=argument_mining_max_pairs, connective_min_words=connective_min_words,
        max_evidence=max_evidence)
        if on("argument_mining") else _disabled("argument_mining"))
    out.extend(_propbank_findings(analysis, srl_model=srl_model, srl_max_predicates=srl_max_predicates,
                                  max_evidence=max_evidence)
              if on("propbank_argument_structure") else _disabled("propbank_argument_structure"))
    out.extend(_verbnet_findings(context, max_evidence=max_evidence)
              if on("verbnet_class_consistency") else _disabled("verbnet_class_consistency"))
    out.extend(_framenet_findings(context, max_evidence=max_evidence)
              if on("framenet_frame_consistency") else _disabled("framenet_frame_consistency"))
    return out
