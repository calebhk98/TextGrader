"""Lexical sophistication and psycholinguistic norm analysis.

Word choice can be measured against many external normative dimensions
besides frequency: how early a word is learned, how concrete or perceptible
it is, how it rates on valence/arousal/dominance, how strongly it grounds in
a sensory or motor modality, how widely it disperses across contexts. Two
texts with identical mean word length or Zipf frequency can differ sharply on
every one of these, and the differences are measurable independently of
whether "more sophisticated" or "more concrete" is the right direction for a
given manuscript -- a picture book *should* score low on age of acquisition
and high on concreteness; an expert text may not. **This suite measures, it
does not judge**: every finding here is ``Polarity.NEUTRAL`` (the default;
see :mod:`textgrader.results`), off by default, and never feeds the existing
maturity aggregate.

Where the norm tables come from
--------------------------------

:mod:`textgrader.lexicons` is the single loader for every downloaded norm
table (own that module's docstring for the full provenance story, including
which two named resources -- the English Lexicon Project and CELEX -- could
not be obtained, and exactly why). This module only asks that module for
tables and scores; it never parses a resource file itself, and if a table is
unavailable (not yet downloaded, no ``openpyxl`` for one of the ``.xlsx``
resources, or a resource this project has no licence path for), only that
resource's own findings degrade to ``unavailable`` -- everything else in the
suite is unaffected. Run ``python -m textgrader.lexicons download all``
once to populate the cache (a few tens of MB), or point
``resource_paths.<name>`` at a locally supplied file.

How a norm becomes findings
----------------------------

Every enabled norm *dimension* (Brysbaert concreteness; Kuperman AoA, surface
and lemma-aggregated separately; Warriner valence/arousal/dominance; NRC
valence/arousal/dominance, kept as an independent resource from Warriner even
though both measure VAD; Lancaster perceptual/action sensorimotor strength;
SUBTLEX-US Zipf frequency and contextual diversity; all nine Glasgow Norms
scales; all six rated MRC dimensions) becomes exactly **two** findings,
``lexical.norm_<key>_token_mean`` and ``lexical.norm_<key>_type_mean``,
because token-weighted and type-weighted are a different question (rule 4 of
the task's cross-task checklist) and a mean that silently mixed them would
answer neither. The *headline* value on the token-weighted finding is the
token-weighted mean over covered tokens -- a ratio of totals (pooled), not a
mean of per-sentence or per-paragraph ratios, which the task's own
retrospective flagged as a way to lose all resolution (a median of per-
sentence ratios can snap to a handful of values). Its ``distribution`` then
carries everything a mean cannot show on its own:

- ``coverage_pct`` / ``matched_tokens`` / ``total_tokens`` -- always present,
  and always beside the mean, so a mean over 12% of the text cannot be
  mistaken for one over 93% (rule 3).
- ``median``, ``std``, ``mad``, ``p10``/``p25``/``p75``/``p90``.
- ``low_tail_share`` / ``high_tail_share``, measured against
  ``low_tail_reference`` / ``high_tail_reference`` -- the **norm table's
  own** p10/p90 (see :func:`textgrader.lexicons.reference_quantiles`), not an
  invented per-metric cutoff. A word at or below the 10th percentile of the
  *lexicon's* own concreteness values is "low-tail" regardless of what this
  one document's own p10 happens to be.
- ``sentence_level`` / ``paragraph_level`` -- the distribution (via
  :func:`textgrader.stats.summarize`) of each sentence's/paragraph's own
  mean over its covered tokens, plus how many sentences/paragraphs had zero
  coverage at all.
- ``between_paragraph_variance`` -- the variance of that per-paragraph mean
  list.
- ``early_vs_late_drift`` -- token-weighted mean over the first half of the
  token stream vs the second half, each with its own coverage.
- ``dialogue_vs_narration`` -- token-weighted mean over
  ``analysis.dialogue``'s tokens vs ``analysis.narration``'s tokens, each
  with its own coverage; ``None`` when either channel has no covered tokens
  (most non-fiction, or fiction with no quoted dialogue).
- ``resource``, ``version``, ``source_url``, ``sha256``, ``entry_count`` --
  every one of :func:`textgrader.lexicons.norm_table`'s metadata fields, so a
  corpus profile built against one resource version is never silently held
  against a report built against another (rule 9; the whole point of
  :mod:`textgrader.lexicons` owning versioning).

The type-weighted finding is lighter (coverage, median, quantiles) since a
type has no sentence position, paragraph, or dialogue channel of its own.

Surface vs lemma
-----------------

Kuperman's AoA ships both a surface-form rating and one aggregated to each
word's dominant lemma; both are kept, as ``aoa`` and ``aoa_lemma``, per rule
2. ``aoa_lemma`` is looked up by the *same* surface tokens as every other
channel here, though -- it is the lemma-*aggregated rating value* attached to
whichever surface form a word takes, not a rating reached by first
lemmatizing this document's own tokens. A true lemma-normalized lookup (this
document's tokens run through a lemmatizer before the table lookup) is kept
behind ``features.lemma_lookup`` (off by default) for exactly the reason
``stylometry_suite``'s ``pos_dependency`` and ``coherence_suite``'s
``coreference`` are off by default: it would force the shared spaCy parse
(tens of seconds on a novel) inside a suite whose declared ``COST`` is
``moderate``, not ``parse``. When it is on, it reuses whatever parse is
already cached (:meth:`DocumentAnalysis.spacy_docs`) rather than requesting
its own.

Cross-checks, kept separate rather than merged
------------------------------------------------

- **wordfreq** already has its own comprehensive suite
  (:mod:`textgrader.metrics.lexical_zipf`, ``lexical.word_zipf`` and
  friends); this module does not duplicate it as a fourth Zipf channel.
  Instead it adds ``lexical.norm_frequency_source_agreement``: for the words
  both wordfreq and SUBTLEX-US Zipf cover, their Pearson correlation and mean
  absolute/signed difference -- two independent frequency estimates *should*
  agree closely, and by how much they do not is itself worth keeping (rule
  18), not something to average away.
- **lexicalrichness cross-checks MTLD/HD-D/MATTR** already inside
  ``stylometry_suite``/``lexical_mtld``/``lexical_hdd``/``mattr``. This suite
  adds two more, independently-installed implementations, per the task
  spec's explicit "cross-check, don't replace" instruction:
  ``lexical_diversity`` (PyPI ``lexical-diversity``, Kristopher Kyle's own
  lightweight package: confirmed by import -- ``lexical_diversity.lex_div``
  exposes ``mtld``/``hdd``/``mattr``/``msttr``/``ttr``/``root_ttr``/
  ``maas_ttr`` functions taking a plain token list) and **TAALED itself**
  (PyPI ``taaled``; confirmed by its own docstring, "Underlying code for
  TAALED (second generation)", author ``kristopherkyle`` -- the same
  researcher who wrote ``lexical_diversity`` above, and TAALED is the actual
  tool the task spec names). Both compute over a length-capped prefix of the
  document (``diversity_crosscheck_max_tokens``, default 50,000) because
  neither library is written for book-length input, and the cap is recorded
  on every finding it applies to.
- **LFTK** (PyPI ``lftk``; confirmed by import -- ``lftk.Extractor`` takes a
  spaCy ``Doc`` and ``lftk.search_features`` lists genuine linguistic feature
  keys, including ``a_kup_pw``/``a_bry_pw``/``a_subtlex_us_zipf_pw``, i.e.
  LFTK bundles its own copies of Kuperman AoA, a Brysbaert-sourced AoA, and
  SUBTLEX-US Zipf). Reported as ``features.lftk_crosscheck`` (off by
  default, like ``lemma_lookup``, because it needs a spaCy ``Doc`` and this
  suite's cost class is ``moderate``): three per-word averages from LFTK's
  own bundled tables, beside (never replacing) this suite's own
  independently-downloaded numbers for the same concepts.
- **TAALES** itself -- as opposed to TAALED above -- is a Java desktop
  application (a downloadable GUI tool, not a library or a data file one
  installs with pip); nothing in this codebase can drive a desktop app, and
  none of TAALES's own output files are redistributable data this module
  could load instead. Its acceptance criteria (lexical sophistication,
  frequency/range, n-gram and contextual-distinctiveness measures) are what
  the resource-based channels above and the ``lexical_diversity``/``taaled``
  cross-checks already cover from the Python side.

Deferred, with evidence
-------------------------

- **English Lexicon Project** and **CELEX**: see
  :mod:`textgrader.lexicons`'s module docstring for the exact HTTP/licensing
  finding for each (registered as resources with ``download_url=None`` so a
  user with a licensed/local copy can still point ``resource_paths`` at it).
- **Morphological family size**: the task spec names CELEX as the source;
  without it (see above) there is no independent, freely licensed source of
  morphological family size counts this task located, so no
  ``morphological_family_size`` channel is emitted. (WordNet-derived
  morphological relations exist in ``coherence_suite``'s optional
  ``lexical_wordnet`` feature for a different purpose -- synonymy chains, not
  family-size counts -- and are not repurposed here to avoid quietly
  presenting a different measurement under this name.)
- **Academic/general vocabulary bands** (e.g. an AWL-style list): the task
  spec names no specific source, and no royalty-free, precisely-versioned
  academic word list was located with confidence during this task; adding
  one without a citable, versioned source would violate rule 9 (a metric
  whose "version" is "whichever list I found" is not reproducible). Left out
  rather than guessed at.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .. import lexicons
from ..document import DocumentAnalysis
from ..optional import require
from ..stats import summarize
from .common import MODERATE, finding, option, rate, unavailable

FAMILY = "lexical"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("wordfreq", "openpyxl", "lexical_diversity", "taaled", "lftk")
MIN_SAMPLE = 50
MIN_SAMPLE_TYPES = 20
UNIT_SENSITIVE = False

DEFAULT_FEATURES: dict[str, bool] = {
    "concreteness": True, "age_of_acquisition": True, "warriner_vad": True, "nrc_vad": True,
    "sensorimotor": True, "subtlex": True, "glasgow": True, "mrc": True,
    "frequency_source_agreement": True,
    "lexdiv_crosscheck": True, "taaled_crosscheck": True,
    # Off by default: each forces the shared spaCy parse, which this suite's
    # "moderate" cost class does not otherwise pay for -- see the module
    # docstring's "Surface vs lemma" / "Cross-checks" sections.
    "lemma_lookup": False, "lftk_crosscheck": False,
}


def _features(config: Mapping[str, Any] | None) -> dict[str, bool]:
    merged = dict(DEFAULT_FEATURES)
    merged.update(option(config, "features", {}) or {})
    return merged


def _resource_paths(config: Mapping[str, Any] | None) -> dict[str, str]:
    return dict(option(config, "resource_paths", {}) or {})


# ------------------------------------------------------------------- channels

@dataclass(frozen=True)
class Channel:
    key: str
    resource: str
    dimension: str
    label: str
    feature: str
    lookup: str = "surface"


CHANNELS: tuple[Channel, ...] = (
    Channel("concreteness", "brysbaert_concreteness", "concreteness",
           "Concreteness (Brysbaert, Warriner & Kuperman 2014)", "concreteness"),
    Channel("aoa", "kuperman_aoa", "aoa",
           "Age of acquisition (Kuperman et al. 2012, surface form)", "age_of_acquisition"),
    Channel("aoa_lemma", "kuperman_aoa", "aoa_lemma",
           "Age of acquisition (Kuperman et al. 2012, lemma-aggregated rating)",
           "age_of_acquisition"),
    Channel("warriner_valence", "warriner_vad", "valence",
           "Valence (Warriner, Kuperman & Brysbaert 2013)", "warriner_vad"),
    Channel("warriner_arousal", "warriner_vad", "arousal",
           "Arousal (Warriner, Kuperman & Brysbaert 2013)", "warriner_vad"),
    Channel("warriner_dominance", "warriner_vad", "dominance",
           "Dominance (Warriner, Kuperman & Brysbaert 2013)", "warriner_vad"),
    Channel("nrc_valence", "nrc_vad", "valence", "Valence (NRC VAD Lexicon v2.1)", "nrc_vad"),
    Channel("nrc_arousal", "nrc_vad", "arousal", "Arousal (NRC VAD Lexicon v2.1)", "nrc_vad"),
    Channel("nrc_dominance", "nrc_vad", "dominance", "Dominance (NRC VAD Lexicon v2.1)", "nrc_vad"),
    Channel("sensorimotor_perceptual", "lancaster_sensorimotor", "perceptual_strength",
           "Dominant-modality perceptual strength (Lancaster Sensorimotor Norms)",
           "sensorimotor"),
    Channel("sensorimotor_action", "lancaster_sensorimotor", "action_strength",
           "Dominant-effector action strength (Lancaster Sensorimotor Norms)", "sensorimotor"),
    Channel("subtlex_zipf", "subtlex_us", "zipf",
           "Subtitle frequency, Zipf scale (SUBTLEX-US)", "subtlex"),
    Channel("subtlex_contextual_diversity", "subtlex_us", "contextual_diversity",
           "Contextual diversity: share of films/episodes containing the word (SUBTLEX-US)",
           "subtlex"),
    Channel("glasgow_arousal", "glasgow_norms", "arousal", "Arousal (Glasgow Norms)", "glasgow"),
    Channel("glasgow_valence", "glasgow_norms", "valence", "Valence (Glasgow Norms)", "glasgow"),
    Channel("glasgow_dominance", "glasgow_norms", "dominance", "Dominance (Glasgow Norms)",
           "glasgow"),
    Channel("glasgow_concreteness", "glasgow_norms", "concreteness",
           "Concreteness (Glasgow Norms)", "glasgow"),
    Channel("glasgow_imageability", "glasgow_norms", "imageability",
           "Imageability (Glasgow Norms)", "glasgow"),
    Channel("glasgow_familiarity", "glasgow_norms", "familiarity",
           "Familiarity (Glasgow Norms)", "glasgow"),
    Channel("glasgow_aoa", "glasgow_norms", "age_of_acquisition",
           "Age of acquisition, banded self-report (Glasgow Norms)", "glasgow"),
    Channel("glasgow_semantic_size", "glasgow_norms", "semantic_size",
           "Semantic size (Glasgow Norms)", "glasgow"),
    Channel("glasgow_gender", "glasgow_norms", "gender_association",
           "Gender association (Glasgow Norms)", "glasgow"),
    Channel("mrc_familiarity", "mrc", "familiarity", "Familiarity (MRC Psycholinguistic Database)",
           "mrc"),
    Channel("mrc_concreteness", "mrc", "concreteness",
           "Concreteness (MRC Psycholinguistic Database)", "mrc"),
    Channel("mrc_imageability", "mrc", "imageability",
           "Imageability (MRC Psycholinguistic Database)", "mrc"),
    Channel("mrc_meaningfulness_colorado", "mrc", "meaningfulness_colorado",
           "Meaningfulness, Colorado norms (MRC Psycholinguistic Database)", "mrc"),
    Channel("mrc_meaningfulness_paivio", "mrc", "meaningfulness_paivio",
           "Meaningfulness, Paivio norms (MRC Psycholinguistic Database)", "mrc"),
    Channel("mrc_aoa", "mrc", "aoa", "Age of acquisition (MRC Psycholinguistic Database)", "mrc"),
)


def _unit_for(channel: Channel) -> str:
    spec = lexicons.resource_info(channel.resource)
    if spec is None:
        return ""
    for dim in spec.dimensions:
        if dim.key == channel.dimension:
            return dim.unit
    return ""


# --------------------------------------------------------------- shared math

def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _covered(words: Sequence[str], table: Mapping[str, Mapping[str, float]], dimension: str
            ) -> list[float]:
    out = []
    for word in words:
        entry = table.get(word)
        if entry is not None and dimension in entry:
            out.append(entry[dimension])
    return out


def _unit_summary(values: Sequence[float]) -> dict[str, Any]:
    summary = summarize(values, shape=False)
    return {key: summary.get(key) for key in ("median", "std", "mad", "p10", "p25", "p75", "p90")}


def _chunk_means(scores: Sequence[float | None], lengths: Sequence[int]) -> list[float]:
    """Mean of each contiguous, length-delimited chunk of ``scores`` that has
    at least one non-``None`` value.

    ``scores`` is the whole document's per-token lookup, computed once;
    ``lengths`` is ``analysis.sentence_lengths`` or ``.paragraph_lengths``,
    which this suite reuses rather than re-tokenizing and re-looking-up each
    sentence/paragraph's own text against the table a second (and third...)
    time. The two are guaranteed to partition ``scores`` because both come
    from the same canonical token stream (``sum(sentence_lengths) ==
    sum(paragraph_lengths) == len(analysis.tokens)`` by construction -- see
    :class:`textgrader.document.DocumentAnalysis`); this is what makes
    slicing safe instead of re-deriving the boundaries from the sentence/
    paragraph text itself.
    """

    out = []
    index = 0
    for length in lengths:
        covered = [v for v in scores[index:index + length] if v is not None]
        index += length
        if covered:
            out.append(_mean(covered))
    return out


def _channel_findings(channel: Channel, tokens: Sequence[str], type_counts: Counter,
                      sentence_lengths: Sequence[int], paragraph_lengths: Sequence[int],
                      dialogue_words: Sequence[str], narration_words: Sequence[str],
                      resource_paths: Mapping[str, str]) -> list[dict[str, Any]]:
    token_id = f"lexical.norm_{channel.key}_token_mean"
    type_id = f"lexical.norm_{channel.key}_type_mean"
    unit = _unit_for(channel)
    override = resource_paths.get(channel.resource)
    table, reason, meta = lexicons.norm_table(channel.resource, override)
    resource_meta = {"resource": meta.get("resource"), "version": meta.get("version"),
                     "source_url": meta.get("source_url"), "sha256": meta.get("sha256"),
                     "entry_count": meta.get("entry_count"), "homepage": meta.get("homepage"),
                     "license": meta.get("license")}
    if table is None:
        return [unavailable(token_id, f"{channel.label}: token-weighted mean", reason,
                            family=FAMILY, unit=unit),
                unavailable(type_id, f"{channel.label}: type-weighted mean", reason,
                           family=FAMILY, unit=unit)]

    total_tokens = len(tokens)
    # One O(tokens) pass builds the whole document's aligned score array;
    # every other view below (sentence/paragraph means, drift) slices this
    # array instead of re-tokenizing and re-looking-up its own text a second
    # time, which is what keeps this "moderate" rather than the suite's own
    # earlier profiling bottleneck (this loop, called 300,000+ times over 28
    # channels' sentence/paragraph breakdowns, measured as this suite's
    # single largest cost on a 150,000-word novel before the fix).
    scores: list[float | None] = [(table.get(word) or {}).get(channel.dimension)
                                  for word in tokens]
    values = [value for value in scores if value is not None]
    matched = len(values)
    reference = (meta.get("dimension_reference") or {}).get(channel.dimension)
    low_ref = reference.get("p10") if reference else None
    high_ref = reference.get("p90") if reference else None

    if matched == 0:
        no_match = f"none of the document's {total_tokens} tokens matched {channel.resource}"
        distribution = {"coverage_pct": 0.0, "matched_tokens": 0, "total_tokens": total_tokens,
                        **resource_meta}
        out = [finding(token_id, f"{channel.label}: token-weighted mean", None, unit,
                       family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                       distribution=distribution, warning=no_match)]
    else:
        sentence_means = _chunk_means(scores, sentence_lengths)
        paragraph_means = _chunk_means(scores, paragraph_lengths)

        half = total_tokens // 2
        early_values = [value for value in scores[:half] if value is not None]
        late_values = [value for value in scores[half:] if value is not None]
        drift = None
        if early_values and late_values:
            drift = {
                "early_mean": _mean(early_values), "late_mean": _mean(late_values),
                "delta": _mean(late_values) - _mean(early_values),
                "early_coverage_pct": rate(len(early_values), half) if half else None,
                "late_coverage_pct": rate(len(late_values), total_tokens - half),
            }

        dialogue_values = _covered(dialogue_words, table, channel.dimension)
        narration_values = _covered(narration_words, table, channel.dimension)
        dialogue_diff = None
        if dialogue_values and narration_values:
            dialogue_diff = {
                "dialogue_mean": _mean(dialogue_values), "narration_mean": _mean(narration_values),
                "delta": _mean(dialogue_values) - _mean(narration_values),
                "dialogue_coverage_pct": rate(len(dialogue_values), len(dialogue_words)),
                "narration_coverage_pct": rate(len(narration_values), len(narration_words)),
            }

        low_share = (rate(sum(1 for v in values if v <= low_ref), matched)
                    if low_ref is not None else None)
        high_share = (rate(sum(1 for v in values if v >= high_ref), matched)
                     if high_ref is not None else None)

        sentence_summary = summarize(sentence_means) if sentence_means else None
        paragraph_summary = summarize(paragraph_means) if paragraph_means else None
        between_paragraph_variance = (paragraph_summary["std"] ** 2
                                      if paragraph_summary and paragraph_summary.get("std") is not None
                                      else None)

        distribution = {
            "coverage_pct": rate(matched, total_tokens), "matched_tokens": matched,
            "total_tokens": total_tokens, "weighting": "token-weighted mean over covered tokens "
            "(a pooled ratio of totals, not a mean of per-sentence ratios)",
            **_unit_summary(values),
            "low_tail_share": low_share, "high_tail_share": high_share,
            "low_tail_reference": low_ref, "high_tail_reference": high_ref,
            "tail_reference_source": "the norm table's own p10/p90 across every word it rates, "
                                     "not a per-document threshold",
            "sentence_level": sentence_summary,
            "sentences_with_coverage": len(sentence_means),
            "sentences_total": len(sentence_lengths),
            "paragraph_level": paragraph_summary,
            "paragraphs_with_coverage": len(paragraph_means),
            "paragraphs_total": len(paragraph_lengths),
            "between_paragraph_variance": between_paragraph_variance,
            "early_vs_late_drift": drift,
            "dialogue_vs_narration": dialogue_diff,
            **resource_meta,
        }
        out = [finding(token_id, f"{channel.label}: token-weighted mean", _mean(values), unit,
                       family=FAMILY, sample_size=matched, min_sample=MIN_SAMPLE,
                       distribution=distribution)]

    type_values = []
    for word, _count in type_counts.items():
        entry = table.get(word)
        if entry is not None and channel.dimension in entry:
            type_values.append(entry[channel.dimension])
    type_total = len(type_counts)
    type_matched = len(type_values)
    type_distribution = {
        "coverage_pct": rate(type_matched, type_total), "matched_types": type_matched,
        "total_types": type_total,
        "weighting": "unweighted mean over distinct covered word types (each word counted once, "
                    "regardless of how often it recurs)",
        **(_unit_summary(type_values) if type_values else {}),
        **resource_meta,
    }
    out.append(finding(type_id, f"{channel.label}: type-weighted mean",
                       _mean(type_values) if type_values else None, unit, family=FAMILY,
                       sample_size=type_matched, min_sample=MIN_SAMPLE_TYPES,
                       distribution=type_distribution,
                       warning=None if type_values else
                       f"none of the document's {type_total} distinct word types matched "
                       f"{channel.resource}"))
    return out


# -------------------------------------------------------------- lemma lookup

def _lemma_channel_findings(analysis: DocumentAnalysis, table: Mapping[str, Mapping[str, float]],
                            dimension: str, label: str, unit: str,
                            resource_meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A lighter surface-vs-lemma cross-check: this document's own tokens are
    lemmatized (via the shared spaCy parse) before the same table lookup,
    rather than looked up as-is. Coverage + token/type means only -- not the
    full sentence/paragraph/drift/dialogue treatment above -- to avoid a
    second full parse-dependent breakdown for a feature that is off by
    default specifically to bound cost (see the module docstring)."""

    metric_id = f"lexical.norm_{dimension}_true_lemma_mean"
    if analysis.nlp_unavailable:
        return [unavailable(metric_id, label, analysis.nlp_unavailable, family=FAMILY, unit=unit)]
    lemmas = [token.lemma_.lower() for token in analysis.spacy_tokens()
             if not token.is_space and not token.is_punct]
    values = _covered(lemmas, table, dimension)
    total = len(lemmas)
    if not values:
        return [finding(metric_id, label, None, unit, family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE,
                        distribution={"coverage_pct": 0.0, "total_lemmas": total, **resource_meta},
                        warning=f"none of the document's {total} lemmas matched the table")]
    return [finding(metric_id, label, _mean(values), unit, family=FAMILY, sample_size=len(values),
                    min_sample=MIN_SAMPLE,
                    distribution={"coverage_pct": rate(len(values), total),
                                 "matched_lemmas": len(values), "total_lemmas": total,
                                 **_unit_summary(values), **resource_meta})]


# ------------------------------------------------------- frequency agreement

def _frequency_agreement(analysis: DocumentAnalysis, config: Mapping[str, Any],
                         resource_paths: Mapping[str, str]) -> dict[str, Any]:
    metric_id = "lexical.norm_frequency_source_agreement"
    name = "Agreement between wordfreq and SUBTLEX-US Zipf frequency"
    wordfreq_module, wf_reason = require("wordfreq")
    table, subtlex_reason, meta = lexicons.norm_table("subtlex_us", resource_paths.get("subtlex_us"))
    if wordfreq_module is None:
        return unavailable(metric_id, name, wf_reason, family=FAMILY)
    if table is None:
        return unavailable(metric_id, name, subtlex_reason, family=FAMILY)
    language = option(config, "language", "en")
    types = sorted(set(analysis.tokens))
    wf_scores, sx_scores = [], []
    for word in types:
        entry = table.get(word)
        if entry is None or "zipf" not in entry:
            continue
        wf = wordfreq_module.zipf_frequency(word, language)
        if wf <= 0:
            continue
        wf_scores.append(wf)
        sx_scores.append(entry["zipf"])
    if len(wf_scores) < 5:
        return finding(metric_id, name, None, "pearson r", family=FAMILY, sample_size=len(wf_scores),
                       min_sample=5, warning=f"only {len(wf_scores)} distinct words scored by both "
                       f"sources; need at least 5 to compute a correlation")
    correlation = statistics.correlation(wf_scores, sx_scores)
    diffs = [a - b for a, b in zip(wf_scores, sx_scores)]
    return finding(
        metric_id, name, correlation, "pearson r", family=FAMILY, sample_size=len(wf_scores),
        distribution={
            "compared_word_types": len(wf_scores),
            "mean_signed_difference_wordfreq_minus_subtlex": _mean(diffs),
            "mean_absolute_difference": _mean([abs(d) for d in diffs]),
            "wordfreq_summary": _unit_summary(wf_scores),
            "subtlex_summary": _unit_summary(sx_scores),
            "note": "two independent general-language/subtitle frequency estimates on the same "
                    "Zipf scale; disagreement is reported, not resolved (rule 18)",
            **meta,
        })


# ------------------------------------------------------------- diversity crosschecks

DEFAULT_DIVERSITY_MAX_TOKENS = 50_000
DEFAULT_DIVERSITY_WINDOW = 50


def _diversity_prefix(analysis: DocumentAnalysis, cap: int) -> tuple[list[str], bool]:
    tokens = analysis.tokens
    if len(tokens) <= cap:
        return list(tokens), False
    return list(tokens[:cap]), True


def _lexdiv_crosscheck(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    module, reason = require("lexical_diversity")
    cap = int(option(config, "diversity_crosscheck_max_tokens", DEFAULT_DIVERSITY_MAX_TOKENS))
    window = int(option(config, "diversity_crosscheck_window", DEFAULT_DIVERSITY_WINDOW))
    ids = (("lexical.norm_lexdiv_mtld", "MTLD (lexical_diversity package cross-check)"),
          ("lexical.norm_lexdiv_hdd", "HD-D (lexical_diversity package cross-check)"),
          ("lexical.norm_lexdiv_mattr", "MATTR (lexical_diversity package cross-check)"),
          ("lexical.norm_lexdiv_msttr", "MSTTR (lexical_diversity package cross-check)"),
          ("lexical.norm_lexdiv_root_ttr", "Root TTR / Guiraud's R (lexical_diversity cross-check)"),
          ("lexical.norm_lexdiv_maas", "Maas's a^2 (lexical_diversity package cross-check)"))
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY) for mid, name in ids]
    tokens, truncated = _diversity_prefix(analysis, cap)
    if len(tokens) < 2:
        return [finding(mid, name, None, family=FAMILY, sample_size=len(tokens), min_sample=MIN_SAMPLE,
                        warning=f"needs at least a couple dozen tokens; this text has {len(tokens)}")
               for mid, name in ids]
    note = (f"computed over the first {cap} of {analysis.word_count} tokens" if truncated else
           "computed over the whole document")
    try:
        values = {
            "lexical.norm_lexdiv_mtld": module.mtld(tokens),
            "lexical.norm_lexdiv_hdd": module.hdd(tokens),
            "lexical.norm_lexdiv_mattr": module.mattr(tokens, window),
            "lexical.norm_lexdiv_msttr": module.msttr(tokens, window),
            "lexical.norm_lexdiv_root_ttr": module.root_ttr(tokens),
            "lexical.norm_lexdiv_maas": module.maas_ttr(tokens),
        }
    except Exception as exc:  # pragma: no cover - third-party failure mode
        return [unavailable(mid, name, f"lexical_diversity raised {type(exc).__name__}: {exc}",
                            family=FAMILY) for mid, name in ids]
    return [finding(mid, name, values[mid], family=FAMILY, sample_size=len(tokens),
                    min_sample=MIN_SAMPLE, distribution={"window": window, "truncated": truncated,
                                                          "note": note})
           for mid, name in ids]


def _taaled_crosscheck(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """TAALED's own MTLD/HD-D/MSTTR/RTTR/Maas, called directly on its
    ``lexdiv`` class rather than through its eager constructor.

    ``taaled.ld.lexdiv(text)`` computes a couple dozen statistics up front,
    including a *second* sliding-window MATTR at ``window_length=11``
    besides the one this suite asks for. Its MATTR implementation rebuilds a
    fresh ``set()`` at every window position (exactly the cost
    :mod:`textgrader.metrics.mattr`'s own docstring describes fixing with a
    running ``Counter``): measured here at 4.5s for a 50-word window over
    5,000 tokens, i.e. roughly O(tokens * window). At this suite's default
    ``diversity_crosscheck_max_tokens`` (50,000) that alone is ~45 seconds,
    and constructing the full object (which pays that cost twice, once per
    window size, plus a second forward/backward MTLD pass at a different
    threshold) measured well over two minutes and was cut off rather than
    timed to completion. Calling the individual methods on a bypassed
    instance (``lexdiv.__new__(lexdiv)``, skipping ``__init__``) computes
    only what is asked for and dropping just the MATTR call -- which
    :func:`_lexdiv_crosscheck` already cross-checks via the ``lexical_diversity``
    package's own, much cheaper MATTR -- brought the whole cross-check for a
    50,000-token sample under 0.3 seconds in this environment.
    """

    module, reason = require("taaled")
    cap = int(option(config, "diversity_crosscheck_max_tokens", DEFAULT_DIVERSITY_MAX_TOKENS))
    window = int(option(config, "diversity_crosscheck_window", DEFAULT_DIVERSITY_WINDOW))
    attrs = (("MTLD", "lexical.norm_taaled_mtld", "MTLD (TAALED package cross-check)", ()),
            ("HDD", "lexical.norm_taaled_hdd", "HD-D (TAALED package cross-check)", ()),
            ("MSTTR", "lexical.norm_taaled_msttr", "MSTTR (TAALED package cross-check)", (window,)),
            ("RTTR", "lexical.norm_taaled_rttr", "Root TTR / Guiraud's R (TAALED cross-check)", ()),
            ("MAAS", "lexical.norm_taaled_maas", "Maas's a^2 (TAALED package cross-check)", ()))
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY) for _attr, mid, name, _args in attrs]
    tokens, truncated = _diversity_prefix(analysis, cap)
    if len(tokens) < 2:
        return [finding(mid, name, None, family=FAMILY, sample_size=len(tokens), min_sample=MIN_SAMPLE,
                        warning=f"needs at least a couple dozen tokens; this text has {len(tokens)}")
               for _attr, mid, name, _args in attrs]
    note = (f"computed over the first {cap} of {analysis.word_count} tokens" if truncated else
           "computed over the whole document")
    try:
        instance = module.lexdiv.__new__(module.lexdiv)
        values = {attr: getattr(instance, attr)(tokens, *args) for attr, _mid, _name, args in attrs}
    except Exception as exc:  # pragma: no cover - third-party failure mode
        return [unavailable(mid, name, f"taaled raised {type(exc).__name__}: {exc}", family=FAMILY)
               for _attr, mid, name, _args in attrs]
    return [finding(mid, name, values[attr], family=FAMILY, sample_size=len(tokens),
                    min_sample=MIN_SAMPLE, distribution={"window": window, "truncated": truncated,
                                                          "note": note})
           for attr, mid, name, _args in attrs]


DEFAULT_LFTK_MAX_CHARS = 200_000


def _lftk_crosscheck(analysis: DocumentAnalysis, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    module, reason = require("lftk")
    ids = (("lexical.norm_lftk_kuperman_aoa_pw", "Kuperman AoA, per word (LFTK's own bundled table)"),
          ("lexical.norm_lftk_brysbaert_aoa_pw",
           "Brysbaert-sourced AoA, per word (LFTK's own bundled table)"),
          ("lexical.norm_lftk_subtlex_zipf_pw",
           "SUBTLEX-US Zipf, per word (LFTK's own bundled table)"))
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY) for mid, name in ids]
    if analysis.nlp_unavailable:
        return [unavailable(mid, name, analysis.nlp_unavailable, family=FAMILY) for mid, name in ids]
    cap = int(option(config, "lftk_max_chars", DEFAULT_LFTK_MAX_CHARS))
    text = analysis.text[:cap]
    truncated = len(analysis.text) > cap
    try:
        doc = analysis.nlp(text)
        extractor = module.Extractor(docs=doc)
        result = extractor.extract(features=["a_kup_pw", "a_bry_pw", "a_subtlex_us_zipf_pw"])
    except Exception as exc:  # pragma: no cover - third-party failure mode
        return [unavailable(mid, name, f"lftk raised {type(exc).__name__}: {exc}", family=FAMILY)
               for mid, name in ids]
    mapping = {
        "lexical.norm_lftk_kuperman_aoa_pw": "a_kup_pw",
        "lexical.norm_lftk_brysbaert_aoa_pw": "a_bry_pw",
        "lexical.norm_lftk_subtlex_zipf_pw": "a_subtlex_us_zipf_pw",
    }
    note = f"computed over the first {cap} of {len(analysis.text)} characters" if truncated else \
        "computed over the whole document"
    return [finding(mid, name, result.get(mapping[mid]), family=FAMILY,
                    sample_size=len(text.split()), min_sample=MIN_SAMPLE,
                    distribution={"truncated": truncated, "note": note})
           for mid, name in ids]


# --------------------------------------------------------------- resource summary

def _resource_summary(resource_paths: Mapping[str, str]) -> dict[str, Any]:
    per_resource = {}
    available_count = 0
    for name in lexicons.list_resources():
        _table, reason, meta = lexicons.norm_table(name, resource_paths.get(name))
        ok = reason is None
        available_count += int(ok)
        per_resource[name] = {
            "available": ok, "reason": None if ok else reason,
            "version": meta.get("version"), "source_url": meta.get("source_url"),
            "sha256": meta.get("sha256"), "entry_count": meta.get("entry_count"),
        }
    return finding(
        "lexical.norm_resource_versions", "Norm resources available to this run",
        available_count, "resources", family=FAMILY,
        distribution={"resources": per_resource,
                     "note": "a corpus profile built with different resource versions/sha256 "
                             "here must not be compared against this report without warning "
                             "(rule 9); see textgrader.lexicons for how to (re)download"})


# ------------------------------------------------------------------------ entry

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    config = config or {}
    features = _features(config)
    resource_paths = _resource_paths(config)

    tokens = analysis.tokens
    type_counts = Counter(tokens)
    # ``sentence_lengths``/``paragraph_lengths`` (already cached word counts
    # per unit) partition the same token stream ``tokens`` does, so every
    # channel below slices one precomputed per-token score array instead of
    # re-tokenizing and re-scoring each sentence/paragraph's own text --
    # see ``_chunk_means``'s docstring for why that is safe and why it matters.
    sentence_lengths = analysis.sentence_lengths
    paragraph_lengths = analysis.paragraph_lengths
    dialogue_words = analysis.dialogue.tokens
    narration_words = analysis.narration.tokens

    out: list[dict[str, Any]] = [_resource_summary(resource_paths)]

    for channel in CHANNELS:
        if not features.get(channel.feature, True):
            continue
        out.extend(_channel_findings(channel, tokens, type_counts, sentence_lengths,
                                     paragraph_lengths, dialogue_words, narration_words,
                                     resource_paths))

    if features.get("lemma_lookup", False):
        table, reason, meta = lexicons.norm_table("kuperman_aoa", resource_paths.get("kuperman_aoa"))
        resource_meta = {"resource": meta.get("resource"), "version": meta.get("version"),
                         "source_url": meta.get("source_url"), "sha256": meta.get("sha256")}
        if table is None:
            out.append(unavailable("lexical.norm_aoa_true_lemma_mean",
                                   "Age of acquisition, true lemma-normalized lookup", reason,
                                   family=FAMILY))
        else:
            out.extend(_lemma_channel_findings(
                analysis, table, "aoa",
                "Age of acquisition, true lemma-normalized lookup (Kuperman et al.)", "years",
                resource_meta))

    if features.get("frequency_source_agreement", True):
        out.append(_frequency_agreement(analysis, config, resource_paths))

    if features.get("lexdiv_crosscheck", True):
        out.extend(_lexdiv_crosscheck(analysis, config))
    if features.get("taaled_crosscheck", True):
        out.extend(_taaled_crosscheck(analysis, config))
    if features.get("lftk_crosscheck", False):
        out.extend(_lftk_crosscheck(analysis, config))

    return out
