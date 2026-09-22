"""The metric registry: one row per switch in ``config.json``.

Importing this module must stay cheap and must never import a third-party
package, because ``grade.py`` reads the registry to decide what to load.  Every
metric module is imported lazily, one at a time, so a broken optional
dependency can disable exactly one measurement.

Each row records what the metric costs on a book-length text and which optional
packages it wants, which is what lets the runner warn an agent before it spends
forty seconds on a parse it did not ask for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class MetricSpec:
    #: Config switch and registry key.
    name: str
    #: Module under ``textgrader.metrics``.
    module: str
    #: Style family reported with every finding.
    family: str
    #: ``fast``, ``moderate``, ``parse`` or ``model``.  Measured, not guessed:
    #: run ``python3 benchmark.py`` to reproduce the README's figures.
    #: ``fast`` is under half a second on a 300,000-word novel, ``moderate`` is
    #: up to a few seconds, ``parse`` needs the shared spaCy parse, and
    #: ``model`` encodes the text with a sentence-embedding model.
    cost: str = "fast"
    #: Optional packages the metric uses; absence degrades it, not the run.
    requires: tuple[str, ...] = ()
    #: Default options merged under the config switch.
    defaults: Mapping[str, Any] = field(default_factory=dict)
    #: One line for the README table and ``--list-metrics``.
    summary: str = ""

    @property
    def needs_parse(self) -> bool:
        return self.cost == "parse" or "spacy" in self.requires

    @property
    def needs_model(self) -> bool:
        """True when the metric downloads and runs a neural model.

        Such a metric works without the model, on a clearly-labelled fallback,
        so it is not "unavailable"; it is expensive enough that nothing should
        turn it on without being asked, least of all the corpus builder
        profiling forty books.
        """

        return "sentence_transformers" in self.requires


def _spec(*args: Any, **kwargs: Any) -> tuple[str, MetricSpec]:
    item = MetricSpec(*args, **kwargs)
    return item.name, item


REGISTRY: dict[str, MetricSpec] = dict([
    # ---------------------------------------------------------- established
    _spec("repeated_ngrams", "repeated_ngrams", "repetition", "moderate",
          defaults={"sizes": [3, 4, 5, 6], "min_count": 2, "max_reported": 40},
          summary="Repeated word sequences, scored by excess occurrences rather than by type count."),
    _spec("sentence_openings", "sentence_openings", "repetition",
          defaults={"words": 3},
          summary="How often a sentence starts with the same few words as another."),
    _spec("local_repetition", "local_repetition", "repetition", "moderate",
          defaults={"windows": [50, 100, 250]},
          summary="Content-word reuse inside sliding windows."),
    _spec("function_words", "function_words", "authorial",
          summary="Burrows's Delta against the corpus function-word profiles."),
    _spec("mattr", "mattr", "lexical", "moderate", defaults={"window": 100},
          summary="Moving-average type-token ratio, length-resistant lexical diversity."),
    _spec("punctuation", "punctuation", "punctuation",
          summary="Rate of each punctuation mark per 1,000 words."),
    _spec("length_quantiles", "length_quantiles", "sentence_rhythm",
          summary="Sentence and paragraph length quantiles."),
    _spec("pov_pronouns", "pov_pronouns", "pov", defaults={"block_words": 500},
          summary="Person-marking pronoun rates and block-by-block POV evidence."),
    _spec("dialogue_contractions", "dialogue_contractions", "dialogue",
          summary="Contraction rate inside spoken text, excluding possessives."),
    _spec("dialogue_tags", "dialogue_tags", "dialogue",
          summary="Speech-tag density and how elaborate the tags are."),
    _spec("sentence_segmentation", "sentence_segmentation", "sentence_rhythm",
          requires=("pysbd",),
          summary="Which segmenter was used, and how much it disagrees with the built-in one."),
    _spec("passive_voice", "passive_voice", "syntax", "parse", ("spacy",),
          summary="Share of clauses in the passive voice."),
    _spec("clause_structure", "clause_structure", "syntax", "parse", ("spacy",),
          summary="Mean token depth in the dependency tree (not per-sentence tree depth)."),
    _spec("pos_distribution", "pos_distribution", "syntax", "parse", ("spacy",),
          summary="Share of each open-class part of speech."),
    _spec("tense_consistency", "tense_consistency", "syntax", "parse", ("spacy",),
          summary="Rate of sentence-to-sentence tense changes in narration."),
    _spec("nominalizations", "nominalizations", "lexical", "parse", ("spacy",),
          summary="Suffix-matched nominalization density; a proxy, not a parse of derivation."),
    _spec("character_voice", "character_voice", "dialogue",
          summary="Pairwise distance between transcript speakers' function-word profiles."),

    _spec("distribution_shape", "distribution_shape", "distribution_shape",
          defaults={"bands": 5},
          summary="Holds the text's sentence, paragraph, word and turn distributions against "
                  "the corpus's pooled ones, so a chapter compares with a shelf of novels."),

    # -------------------------------------------------------------- rhythm
    _spec("sentence_length_autocorrelation", "rhythm_autocorrelation", "sentence_rhythm",
          defaults={"lags": [1, 2, 3]},
          summary="Lag-1..n autocorrelation of sentence length: catches metronomic prose."),
    _spec("sentence_length_deltas", "rhythm_deltas", "sentence_rhythm",
          summary="Distribution of the change in length between adjacent sentences."),
    _spec("sentence_run_lengths", "rhythm_runs", "sentence_rhythm",
          defaults={"short_max": 8, "long_min": 25},
          summary="Run-length distribution of short/medium/long sentence bands."),
    _spec("sentence_length_entropy", "rhythm_entropy", "sentence_rhythm",
          summary="Entropy of the sentence-length distribution, normalized for range."),
    _spec("paragraph_rhythm", "paragraph_rhythm", "paragraph_rhythm",
          summary="Paragraph length dispersion, quantiles and autocorrelation."),
    _spec("single_sentence_paragraph_runs", "paragraph_single_runs", "paragraph_rhythm",
          summary="Runs of consecutive one-sentence paragraphs."),

    # -------------------------------------------------------------- syntax
    _spec("dependency_distance", "syntax_dependency_distance", "syntax", "parse", ("spacy",),
          summary="Mean and SD of dependency distance, a real syntactic-complexity measure."),
    _spec("parse_depth", "syntax_parse_depth", "syntax", "parse", ("spacy",),
          summary="Per-sentence maximum dependency-tree depth."),
    _spec("finite_clauses", "syntax_finite_clauses", "syntax", "parse", ("spacy",),
          summary="Finite clauses per sentence."),
    _spec("clause_types", "syntax_clause_types", "syntax", "parse", ("spacy",),
          summary="Relative, adverbial and complement clause rates."),
    _spec("coordination_ratio", "syntax_coordination", "syntax", "parse", ("spacy",),
          summary="Coordination against subordination."),
    _spec("opening_patterns", "syntax_openings", "syntax", "parse", ("spacy",),
          defaults={"depth": 2, "max_reported": 25},
          summary="POS/dependency sentence-opening shapes, with no hard-coded vocabulary."),

    # ------------------------------------------------------------- lexical
    _spec("mtld", "lexical_mtld", "lexical", "moderate",
          requires=("lexicalrichness",), defaults={"threshold": 0.72},
          summary="Measure of Textual Lexical Diversity."),
    _spec("hdd", "lexical_hdd", "lexical", "moderate",
          requires=("lexicalrichness",), defaults={"sample": 42},
          summary="HD-D, a hypergeometric length-resistant diversity measure."),
    _spec("word_rarity", "lexical_zipf", "lexical", "moderate", ("wordfreq",),
          defaults={"language": "en"},
          summary="Zipf word-rarity distribution from general-language frequencies."),
    _spec("repetition_distance", "lexical_repetition_distance", "repetition", "moderate",
          defaults={"min_length": 4},
          summary="How soon a content word is reused, in tokens."),
    _spec("lemma_repetition", "lexical_lemma_repetition", "repetition", "parse", ("spacy",),
          summary="Repetition measured over lemmas, so walk/walked/walking cannot hide."),

    # ------------------------------------------------------------ semantic
    _spec("adjacent_sentence_similarity", "semantic_adjacent", "semantic_repetition",
          "model", ("sentence_transformers",), defaults={"model": "all-MiniLM-L6-v2"},
          summary="Embedding similarity between neighbouring sentences."),
    _spec("local_similarity_window", "semantic_window", "semantic_repetition",
          "model", ("sentence_transformers",),
          defaults={"model": "all-MiniLM-L6-v2", "windows": [3, 5]},
          summary="Similarity to the previous three and five sentences."),
    _spec("paragraph_similarity", "semantic_paragraph", "semantic_repetition",
          "model", ("sentence_transformers",), defaults={"model": "all-MiniLM-L6-v2"},
          summary="Paragraph-to-paragraph semantic similarity."),
    _spec("duplicate_sentence_clusters", "semantic_clusters", "semantic_repetition",
          "model", ("sentence_transformers",),
          defaults={"model": "all-MiniLM-L6-v2", "threshold": 0.85, "max_reported": 30},
          summary="Clusters of sentences that restate one another."),

    # ------------------------------------------------------------ dialogue
    _spec("dialogue_channels", "dialogue_channels", "dialogue", "moderate",
          summary="Every core shape measured separately for dialogue and for narration."),
    _spec("dialogue_turn_lengths", "dialogue_turn_lengths", "dialogue", "moderate",
          summary="Distribution of spoken turn lengths in words and sentences."),
    _spec("speaker_style", "dialogue_speaker_style", "dialogue",
          defaults={"min_turns": 8},
          summary="Questions, exclamations and contractions per identified speaker."),
    _spec("dialogue_attribution", "dialogue_attribution", "dialogue",
          summary="Speech tag against action beat against untagged turn."),
    _spec("dialogue_runs", "dialogue_runs", "dialogue",
          summary="Consecutive spoken turns with no narration between them."),
    _spec("speaker_function_words", "dialogue_speaker_function_words", "dialogue",
          defaults={"min_turns": 8},
          summary="Per-speaker function-word profile and pairwise distance."),

    # ----------------------------------------------------------- discourse
    _spec("sentence_initial_connectives", "discourse_connectives", "discourse",
          summary="Rate of sentences opening on However, Indeed, Moreover and the like."),
    _spec("causal_connectives", "discourse_causal", "discourse",
          summary="Causal and explanatory connective rates."),
    _spec("hedges_boosters", "discourse_hedges", "discourse",
          summary="Hedge, booster and modal rates."),
    _spec("rhetorical_constructions", "discourse_constructions", "discourse", "moderate",
          defaults={"max_reported": 30, "min_count": 3},
          summary="Repeated rhetorical templates, discovered rather than listed."),

    # ----------------------------------------------------------------- pov
    _spec("narration_pov", "pov_narration", "pov",
          summary="Person-marking rates measured in narration only, free of dialogue."),
    _spec("pov_block_confidence", "pov_block_confidence", "pov",
          defaults={"block_words": 1000},
          summary="Per-block POV call with an explicit evidence count; no evidence means no call."),
    _spec("entity_pronoun_ratio", "pov_entity_ratio", "pov", "parse", ("spacy",),
          defaults={"enable_ner": True},
          summary="Named entities against pronouns: over-naming or pronoun saturation."),

    # --------------------------------------------------------- punctuation
    _spec("punctuation_profile", "punctuation_profile", "punctuation", "moderate",
          summary="Full per-mark distribution, per sentence and per 1,000 words."),
    _spec("punctuation_entropy", "punctuation_entropy", "punctuation", "moderate",
          summary="Entropy of the punctuation mix, a regularity signal."),
    _spec("punctuation_patterns", "punctuation_patterns", "punctuation", "moderate",
          defaults={"max_reported": 25},
          summary="Repeated punctuation shapes across consecutive sentences and paragraphs."),

    # ---------------------------------------------------------- book drift
    _spec("chapter_zscores", "drift_chapter_zscores", "book_drift", "moderate",
          defaults={"window_words": 2500},
          summary="Which section looks unlike the rest of this book, and on which measures."),
    _spec("rolling_drift", "drift_rolling", "book_drift", "moderate",
          defaults={"window_words": 2500},
          summary="Gradual style drift from the opening to the close."),
    _spec("change_points", "drift_change_points", "book_drift", "moderate",
          requires=("ruptures",), defaults={"window_words": 2500, "penalty": 3.0},
          summary="Where the style changes abruptly."),

    # ---------------------------------------------------------- experimental
    _spec("coherence_suite", "coherence_suite", "discourse", "parse",
          requires=("spacy", "sentence_transformers", "networkx"),
          defaults={
              "features": {"lexical": True, "semantic": True, "entity": True,
                          "connectives": True, "order_permutation": True},
              "min_word_len": 3,
              "chain_gap": 3,
              "chain_min_length": 2,
              "semantic_model": "all-MiniLM-L6-v2",
              "semantic_low_tail_threshold": 0.15,
              "entity_lookback_sentences": 10,
              "entity_max_tracked": 150,
              "entity_graph_window_sentences": 3,
              "entity_min_mentions_for_graph": 2,
              "connective_max_reported": 25,
              "permutations": 50,
              "seed": 0,
              "order_min_sentences_per_paragraph": 4,
              "order_max_sentences_per_paragraph": 40,
              "order_max_paragraphs_sampled": 30,
              "order_max_paragraphs_for_doc": 60,
          },
          summary="Experimental discourse coherence/cohesion: lexical and semantic adjacency, "
                  "a surface-based entity grid and graph, connective-family rates, and "
                  "sentence/paragraph order-permutation baselines. Off by default; each group "
                  "toggles independently under 'features'."),
    _spec("logic_suite", "logic_suite", "discourse", "parse", ("spacy",),
          defaults={
              "features": {
                  "negation_and_quantifiers": True,
                  "connective_relations": True,
                  "propositions": True,
                  "modal_argument_position": True,
              },
              "window_sentences": 6,
              "max_pairs": 200,
              "max_comparisons": 50_000,
              "max_evidence": 20,
              "proposition_cap": 20_000,
              "connective_min_words": 4,
              "repeated_assertion_min_words": 5,
          },
          summary="Candidate contradictions, connective-relation overlap and proposition "
                  "structure; no NLI/OpenIE model is available here, so every value is a "
                  "surface-heuristic candidate, never a truth or entailment claim."),
    _spec("randomness_suite", "randomness_suite", "lexical", "moderate",
          requires=("wordfreq",),
          defaults={
              "features": {
                  "char_entropy": True, "compression": True, "complexity_measures": True,
                  "language_model": True, "punctuation_sequence": True,
                  "pos_dependency": False, "corruption_baselines": True,
                  "lexical_gibberish": True,
              },
              "language": "en",
              "char_ngram_orders": [2, 3, 4, 5, 6], "byte_ngram_order": 3,
              "word_ngram_orders": [1, 2, 3, 4], "punct_ngram_order": 3,
              "pos_ngram_order": 3, "dependency_ngram_order": 2,
              "lm_train_fraction": 0.7, "lm_smoothing_alpha": 0.5,
              "lm_max_chars": 150000, "lm_max_tokens": 60000, "entropy_token_cap": 100000,
              "renyi_orders": [0.5, 2.0], "tsallis_orders": [0.5, 2.0],
              "excess_entropy_max_order": 4, "mi_lags": [1, 2, 3], "mi_max_tokens": 20000,
              "complexity_quadratic_cap": 1500, "permutation_order": 3, "lz_max_chars": 20000,
              "compression_algorithms": ["zlib", "gzip", "bz2", "lzma", "zstd", "brotli",
                                         "lz4", "snappy", "ppmd"],
              "compression_level": 6, "compression_block_chars": 20000, "ncd_algorithm": "zlib",
              "corruption_seed": 1337, "corruption_permutations": 3,
              "corruption_sentence_fraction": 0.3, "corruption_char_order": 4,
              "corruption_word_order": 2, "corruption_max_chars": 20000,
              "consonant_cluster_min": 4,
          },
          summary="Character/word/POS/punctuation language-likeness, multi-algorithm "
                  "compression ratios and NCD, entropy/complexity families (Shannon, "
                  "Renyi, Tsallis, permutation, spectral, SVD, ApEn/SampEn, LZ), and "
                  "seeded shuffle-corruption baselines. Off by default; experimental."),
    _spec("timeseries_suite", "timeseries_suite", "sentence_rhythm", "moderate",
          defaults={
              "sequences": ["sentence_words", "paragraph_words", "sentence_punctuation"],
              "feature_groups": ["dispersion", "acf", "trend", "turning_points", "runs"],
              "lags": [1, 2, 3],
              "pacf_max_lag": 5,
              "window_words": 2000,
              "rolling_window": 10,
              "change_point_penalty": 2.0,
              "page_hinkley_delta": 0.005,
              "page_hinkley_lambda": 3.0,
              "permutation_entropy_order": 3,
              "permutation_entropy_delay": 1,
              "dfa_min_box": 4,
              "piecewise_segments": 2,
              "detrend": False,
              "embedding_model": "all-MiniLM-L6-v2",
              "language": "en",
              "wavelet_name": "db4",
              "wavelet_max_level": 5,
              "min_lengths": {},
              "max_findings": 200,
          },
          summary="Autocorrelation/trend/spectral/nonlinear features over named linguistic "
                  "sequences (sentence length, punctuation, parse depth, embedding-based "
                  "sentence similarity, ...), plus optional catch22 (22 canonical features), "
                  "wavelet energy/entropy and a textdescriptives dependency-distance "
                  "cross-check; each sequence and feature group is independently selectable "
                  "and off by default."),
    _spec("stylometry_suite", "stylometry_suite", "authorial", "moderate",
          defaults={
              "features": {
                  "character_ngrams": True, "byte_ngrams": True, "word_ngrams": True,
                  "function_word_ngrams": True, "punctuation_shape": True, "word_shape": True,
                  "affixes": True, "sentence_openings": True, "contractions_capitalization": True,
                  "lexical_richness": True, "vocabulary_growth": True, "section_stability": True,
                  "compression": True, "corpus_language_model": True, "corpus_reference": True,
                  # Off by default: forces the shared spaCy parse (tens of seconds on a novel).
                  "pos_dependency": False,
              },
              "char_ngram_orders": [2, 3, 4, 5, 6], "byte_ngram_orders": [2],
              "word_ngram_orders": [1, 2, 3], "pos_ngram_orders": [1, 2, 3, 4],
              "dependency_ngram_order": 2, "punctuation_ngram_order": 2,
              "affix_length": 3, "min_affix_word_length": 5, "max_reported": 25,
              "max_chars_for_ngrams": 500000, "section_window_words": 3000,
              "section_shift_threshold": 2.5, "k_neighbors": 5,
              "distance_metrics": ["cosine", "euclidean", "manhattan", "jensen_shannon"],
              "primary_distance": "cosine", "compression_algorithm": "zlib",
              "min_corpus_documents": 4, "min_documents_per_author": 2,
              "outlier_threshold": 3.5, "seed": 42,
          },
          summary="Stylometry/authorship suite: character/word/POS/punctuation n-gram entropy, "
                  "lexical-richness statistics, Heaps/Zipf fits, section-to-section style "
                  "stability, compression-based measures, and nearest/centroid/OOD distances "
                  "against a reference corpus. Every measurement group is independently "
                  "switchable; see the module docstring for what was deferred."),
])

#: Config-name -> module-name, kept for older callers.
MODULES = {name: spec.module for name, spec in REGISTRY.items()}

#: Metrics that need a spaCy parse.  They share one parse per document.
NLP_METRICS = {name for name, spec in REGISTRY.items() if spec.needs_parse}

#: Metrics that will load a sentence-embedding model when one is installed.
MODEL_METRICS = {name for name, spec in REGISTRY.items() if spec.needs_model}

#: Metrics whose value scales with how much text you supply.
FAMILIES = sorted({spec.family for spec in REGISTRY.values()})


def specs_by_family() -> dict[str, list[MetricSpec]]:
    out: dict[str, list[MetricSpec]] = {}
    for spec in REGISTRY.values():
        out.setdefault(spec.family, []).append(spec)
    return out
