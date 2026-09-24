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
          requires=("ruptures",), defaults={"window_words": 2500, "penalty": 2.0},
          summary="Where the style changes abruptly."),

    # ---------------------------------------------------------- experimental
    _spec("coherence_suite", "coherence_suite", "discourse", "parse",
          requires=("spacy", "sentence_transformers", "networkx", "fastcoref", "nltk",
                   "isanlp_rst"),
          defaults={
              "features": {"lexical": True, "lexical_wordnet": False,
                          "lexical_wordnet_hypernym": False, "semantic": True,
                          "entity": True, "coreference": False, "rst": False,
                          "connectives": True, "order_permutation": True},
              "min_word_len": 3,
              "chain_gap": 3,
              "chain_min_length": 2,
              "chain_hypernym_max_distance": 3,
              "semantic_model": "all-MiniLM-L6-v2",
              "semantic_low_tail_threshold": 0.15,
              "entity_lookback_sentences": 10,
              "entity_max_tracked": 150,
              "entity_graph_window_sentences": 3,
              "entity_min_mentions_for_graph": 2,
              "coreference_model": "biu-nlp/f-coref",
              "coreference_max_words": 4000,
              "rst_model": "tchewik/isanlp_rst_v3",
              "rst_model_version": "rstdt",
              "rst_passages": 8,
              "rst_passage_sentences": 6,
              "rst_max_sentences": 60,
              "rst_max_seconds": 420.0,
              "rst_seed": 0,
              "connective_max_reported": 25,
              "permutations": 50,
              "seed": 0,
              "order_min_sentences_per_paragraph": 4,
              "order_max_sentences_per_paragraph": 40,
              "order_max_paragraphs_sampled": 30,
              "order_max_paragraphs_for_doc": 60,
          },
          summary="Experimental discourse coherence/cohesion: lexical, WordNet-synonym and "
                  "WordNet-hypernym-proximity lexical chains, embedding-based semantic "
                  "adjacency, a surface-based entity grid and graph (with narration/dialogue "
                  "channel splits) and a corpus-referenced transition-frequency delta, plus an "
                  "optional real-coreference (fastcoref) backend reported side by side with the "
                  "surface grid, a sampled real-RST-parse (isanlp_rst) channel (tree depth, "
                  "segment length, nuclearity balance, relation-family entropy), "
                  "connective-family rates, and sentence/paragraph order-permutation baselines. "
                  "Off by default; each group toggles independently under 'features', and the "
                  "coreference/WordNet/RST groups stay off even when the rest of the suite is "
                  "enabled."),
    _spec("logic_suite", "logic_suite", "discourse", "parse",
          ("spacy", "transformers", "fastcoref", "nltk", "dateutil"),
          defaults={
              "features": {
                  "negation_and_quantifiers": True,
                  "connective_relations": True,
                  "propositions": True,
                  "modal_argument_position": True,
                  # Off by default -- see the module docstring's "Gating" note:
                  # this suite's cost class ("parse") means MetricSpec.needs_model
                  # (which only checks for "sentence_transformers") does NOT
                  # exclude this from corpus profiling, so these flags are the
                  # only thing standing between a transformer model and a book
                  # nobody asked to run one against.
                  "nli_entailment": False,
                  "coreference_resolution": False,
                  "lexical_opposition": False,
                  "temporal_ordering": False,
                  "semantic_role_labeling": False,
                  "relation_extraction": False,
                  "argument_mining": False,
                  "propbank_argument_structure": False,
                  "verbnet_class_consistency": False,
                  "framenet_frame_consistency": False,
              },
              "window_sentences": 6,
              "max_pairs": 200,
              "max_comparisons": 50_000,
              "max_evidence": 20,
              "proposition_cap": 20_000,
              "connective_min_words": 4,
              "repeated_assertion_min_words": 5,
              "coreference_max_chars": 20_000,
              "nli_model": "cross-encoder/nli-deberta-v3-small",
              "nli_max_pairs": 60,
              "nli_batch_size": 16,
              "srl_model": "cu-kairos/propbank_srl_seq2seq_t5_small",
              "srl_max_predicates": 40,
              "relation_extraction_model": "Babelscape/rebel-large",
              "relation_extraction_max_sentences": 40,
              "argument_mining_model": "raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L",
              "argument_mining_max_pairs": 40,
          },
          summary="Candidate contradictions, connective-relation overlap and proposition "
                  "structure from surface heuristics (on by default), plus seven off-by-default "
                  "channels that need an installed model or resource: real NLI entailment/"
                  "contradiction scoring over the same candidate pairs with a heuristic-vs-model "
                  "agreement readout, fastcoref coreference resolution so pronoun subjects can "
                  "enter the candidate pool, WordNet antonym/hypernym lexical relations, "
                  "dateutil-based temporal ordering, PropBank-style semantic role labelling (role-"
                  "pattern consistency and argument-omission rate), closed-schema relation "
                  "extraction (REBEL) kept beside the dependency-parse proxy, and a real argument-"
                  "relation (claim/premise support/attack) classifier over connective-linked clause "
                  "pairs, kept beside the honestly-named connective_chain_length proxy. Every "
                  "heuristic value stays a candidate, never a truth claim; every model value is a "
                  "labelled model score, never a fact about the text -- see the module docstring."),
    _spec("randomness_suite", "randomness_suite", "lexical", "moderate",
          requires=("wordfreq",),
          defaults={
              "features": {
                  "char_entropy": True, "compression": True, "complexity_measures": True,
                  "language_model": True, "punctuation_sequence": True,
                  "pos_dependency": False, "corruption_baselines": True,
                  "lexical_gibberish": True, "ppm_language_model": True,
                  "letter_bigram_divergence": True, "kenlm_language_model": False,
                  "neural_language_model": False, "textdescriptives_cross_check": False,
                  "gibberish_detector_package": False, "ncd_against_corpus": False,
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
              "consonant_cluster_min": 4, "ppm_max_order": 6,
              "kenlm_lmplz_path": "", "kenlm_order": 3, "kenlm_memory": "50M",
              "kenlm_max_train_chars": 200000, "kenlm_timeout_seconds": 30,
              "neural_lm_model": "distilgpt2", "neural_lm_max_chars": 6000,
              "textdescriptives_max_chars": 50000,
              "gibberish_detector_charset": "abcdefghijklmnopqrstuvwxyz",
              "ncd_corpus_dirs": [], "ncd_corpus_max_reference_documents": 10,
              "ncd_corpus_max_bytes": 100000, "ncd_corpus_algorithm": "lzma",
          },
          summary="Character/word/POS/punctuation language-likeness, multi-algorithm "
                  "compression ratios (zlib/gzip/bz2/lzma/zstd/brotli/lz4/snappy/pyppmd) and "
                  "self-corruption NCD, a real PPM predictive-model cross-entropy, "
                  "entropy/complexity families (Shannon, Renyi, Tsallis, permutation, spectral, "
                  "SVD, ApEn/SampEn, LZ), single-letter and corpus-derived letter-bigram "
                  "divergence from English, seeded shuffle-corruption baselines, and five opt-in "
                  "external cross-checks (a real KenLM Kneser-Ney n-gram model, a pretrained "
                  "causal-LM perplexity, textdescriptives, a gibberish-detector package model "
                  "trained on the document's own held-out split, and true NCD against real "
                  "reference documents read from disk at grading time). Off by default; "
                  "experimental."),
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
              "tsfresh_feature_set": "minimal",
              "tsfresh_max_features": 25,
              "adwin_delta": 0.002,
              "topic_n_topics": 4,
              "topic_model": "nmf",
              "topic_random_state": 42,
              "topic_max_features": 2000,
              "min_lengths": {},
              "max_findings": 200,
          },
          summary="Autocorrelation/trend/spectral/nonlinear features over named linguistic "
                  "sequences (sentence length, punctuation, parse depth, embedding-based "
                  "sentence similarity, VADER sentiment, NRC emotion valence, NMF/LDA topic "
                  "id, ...), plus optional catch22/catch24 (24 canonical features), wavelet "
                  "energy/entropy, a river ADWIN drift detector beside the suite's own "
                  "Page-Hinkley detector, topic-transition rate/entropy/dwell-time, a "
                  "configurable/capped tsfresh feature set and a textdescriptives "
                  "dependency-distance cross-check; each sequence and feature group is "
                  "independently selectable and off by default."),
    _spec("stylometry_suite", "stylometry_suite", "authorial", "moderate",
          # Deliberately NOT ("lexicalrichness", "sentence_transformers"): the
          # latter would flip needs_model for this WHOLE suite and drop it out
          # of the corpus builder's default profiling pass (see the module
          # docstring's "The critical gating rule"). lexicalrichness alone is
          # safe -- it affects neither needs_parse nor needs_model -- and is
          # used opportunistically by lexicalrichness_crosscheck (default on).
          requires=("lexicalrichness",),
          defaults={
              "features": {
                  "character_ngrams": True, "byte_ngrams": True, "word_ngrams": True,
                  "function_word_ngrams": True, "punctuation_shape": True, "word_shape": True,
                  "affixes": True, "sentence_openings": True, "contractions_capitalization": True,
                  "lexical_richness": True, "vocabulary_growth": True, "section_stability": True,
                  "compression": True, "corpus_language_model": True, "corpus_reference": True,
                  # Off by default: forces the shared spaCy parse (tens of seconds on a novel).
                  "pos_dependency": False,
                  "author_language_model": True, "word_frequency_distance": True,
                  "lexicalrichness_crosscheck": True,
                  # Off by default: loads a sentence-transformers model; the ONLY
                  # feature in this suite that can (see "The critical gating rule").
                  "embedding_style": False,
                  # Off by default: a heavier, more specialized analysis, only
                  # meaningful with author-labelled corpus data.
                  "impostors": False,
                  # Off by default: fits a REAL classifier (scikit-learn) per
                  # iteration, unlike "impostors" above (a distance
                  # comparison). Never during corpus profiling -- see the
                  # module docstring's section K2.
                  "impostors_classifier": False,
                  # On by default: cheap, hand-implemented arithmetic over the
                  # already-cached function-word profile -- see the module
                  # docstring's section P (the "PyDelta" gap).
                  "delta_family": True,
                  # Off by default: like ncd_against_corpus, reads real
                  # reference documents from disk at grading time and needs
                  # the pystylometry package -- see the module docstring's
                  # section Q.
                  "pystylometry_reference": False,
                  # Off by default: needs a corpus profile built with
                  # embedding_style ALSO enabled at profiling time (so
                  # feature_profiles['stylometry_suite'] holds per-book
                  # embedding vectors) plus sentence-transformers at grading
                  # time. See the module docstring's "How to get embedding
                  # vectors into a profile".
                  "embedding_reference": False,
                  # Off by default: the one measurement in this suite that
                  # reads the corpus FOLDER (not the cached profile) at
                  # grading time; needs ncd_corpus_dirs configured.
                  "ncd_against_corpus": False,
              },
              "char_ngram_orders": [2, 3, 4, 5, 6], "byte_ngram_orders": [2],
              "word_ngram_orders": [1, 2, 3], "pos_ngram_orders": [1, 2, 3, 4],
              "dependency_ngram_order": 2, "punctuation_ngram_order": 2,
              "affix_length": 3, "min_affix_word_length": 5, "max_reported": 25,
              "max_chars_for_ngrams": 500000, "section_window_words": 3000,
              "section_shift_threshold": 2.5, "k_neighbors": 5,
              "distance_metrics": ["cosine", "euclidean", "manhattan", "jensen_shannon"],
              # lzma, not zlib: zlib's fixed 32 KiB window cannot reliably
              # compute NCD once either side is much bigger than half that
              # (ordinary for this suite's book-length inputs) -- see
              # stylometry_suite's COMPRESSOR_DICTIONARY_BYTES/
              # _ncd_window_guard for the measured evidence.
              "primary_distance": "cosine", "compression_algorithm": "lzma",
              "min_corpus_documents": 4, "min_documents_per_author": 2,
              "outlier_threshold": 3.5, "seed": 42,
              "word_frequency_vocab_cap": 3000,
              "embedding_model": "all-MiniLM-L6-v2", "embedding_primary_distance": "cosine",
              "impostors_k": 10, "impostors_iterations": 25,
              "impostors_feature_fraction": 0.5, "impostors_min_authors": 2,
              "impostors_target_author": None,
              # "function_words" (default) or "embedding" -- see the module
              # docstring's "Impostors-style verification".
              "impostors_representation": "function_words",
              # impostors_classifier: iteration count kept lower than
              # impostors_iterations because each one fits a real classifier.
              "impostors_classifier_iterations": 15,
              "impostors_classifier_type": "logistic_regression",
              # ncd_against_corpus: directories of reference .txt/.md files,
              # read fresh from disk at grading time (see the module
              # docstring's "True NCD against reference documents"), plus how
              # many of them and how many bytes of each to bound the cost.
              "ncd_corpus_dirs": [], "ncd_max_reference_documents": 10,
              "ncd_max_bytes": 100000,
              # pystylometry_reference: reuses ncd_corpus_dirs (same real
              # reference text, read the same way) with its own document/byte
              # caps and most-frequent-word count.
              "pystylometry_max_reference_documents": 5,
              "pystylometry_max_bytes": 200000, "pystylometry_mfw": 200,
          },
          summary="Stylometry/authorship suite: character/word/POS/punctuation n-gram entropy, "
                  "lexical-richness statistics (plus a lexicalrichness cross-check), Heaps/Zipf "
                  "fits, section-to-section style stability, compression-based measures, "
                  "nearest/centroid/OOD distances against a reference corpus (both function-word "
                  "and, once a profile caches per-book embedding vectors, sentence-embedding), a "
                  "word-frequency distance family, per-author unigram cross-entropy, a bounded "
                  "distance-based impostors approximation AND (off by default) a real "
                  "classifier-refit-per-iteration general-impostors score (function-word or "
                  "embedding representation), a hand-implemented Delta family (Burrows/Argamon "
                  "quadratic/Eder's/cosine, on by default) plus an off-by-default pystylometry "
                  "cross-check of the same family against real reference text (also giving Zeta "
                  "and Kilgarriff's chi-squared, which nothing else in this suite computes), an "
                  "off-by-default sentence-embedding section-drift representation, and an "
                  "off-by-default true NCD against real reference documents read from disk at "
                  "grading time. Every measurement group is independently switchable; see the "
                  "module docstring for what remains deferred."),
    _spec("anomaly_suite", "anomaly_suite", "distribution_shape", "moderate",
          requires=("sklearn",),
          defaults={
              "features": {
                  "isolation_forest": True, "lof": True, "one_class_svm": True,
                  "elliptic_envelope": True, "mahalanobis": True, "knn": True, "pca": True,
                  "gmm": True,
                  # Off only when PyOD/hdbscan are absent -- these three flags default to
                  # True the same as every other detector; see config.json's
                  # "_features_requires" note for exactly what each needs installed.
                  "hbos": True, "ecod": True, "copod": True, "abod": True, "kde": True,
                  "sos": True, "hdbscan": True,
                  "consensus_count": True, "disagreement": True,
              },
              "columns": None, "min_coverage": 0.7, "max_features": 20,
              "min_corpus_documents": 20, "seed": 42, "loo_reference_size": 15,
              "consensus_percentile": 90.0, "evidence_features": 5,
              "isolation_forest_n_estimators": 50, "one_class_svm_nu": 0.1,
              "pca_components": None, "gmm_components": 2, "hdbscan_min_cluster_size": 5,
          },
          summary="Fifteen independent multivariate anomaly detectors (Isolation Forest, LOF, "
                  "One-Class SVM, Elliptic Envelope, Mahalanobis, kNN distance, PCA "
                  "reconstruction error and Gaussian-mixture likelihood via scikit-learn; HBOS, "
                  "ECOD, COPOD, ABOD, KDE and SOS via PyOD; a GLOSH density score via hdbscan) "
                  "over each document's core-prose-metric feature vector against the reference "
                  "corpus, fit fresh at grading time with leave-one-out corpus validation, plus "
                  "a consensus count and a cross-detector disagreement score. Off by default; "
                  "experimental."),
    _spec("distribution_distance_suite", "distribution_distance_suite", "distribution_shape",
          "moderate", requires=("scipy",),
          defaults={
              # Mirrors textgrader.corpus.ITEM_SOURCES; not imported from there, to keep
              # this module's own import free of anything beyond the standard library.
              "sources": ["sentence_words", "paragraph_words", "paragraph_sentences",
                          "word_characters", "sentence_commas", "turn_words"],
              "features": {
                  "wasserstein": True, "energy": True, "ks": True, "cramer_vonmises": True,
                  "anderson_darling": True, "jensen_shannon": True, "kl_divergence": True,
                  "hellinger": True, "bhattacharyya": True, "total_variation": True,
                  "mmd": True, "quantile_vector": True, "tail": True,
                  "optimal_transport": True, "distance_correlation": True,
              },
              "histogram_bins": 16, "histogram_smoothing": 0.5,
              "quantile_vector_points": [0.10, 0.25, 0.50, 0.75, 0.90],
              "tail_quantile": 0.10, "mmd_max_sample": 500,
              "distance_correlation_max_sample": 300,
              "distance_correlation_pairs": {
                  "sentence": ["sentence_words", "sentence_commas"],
                  "paragraph": ["paragraph_words", "paragraph_sentences"],
              },
              "min_distinct_values": 5,
          },
          summary="A full two-sample battery (Wasserstein, energy distance, Kolmogorov-Smirnov, "
                  "Cramer-von Mises, Anderson-Darling, Jensen-Shannon, KL both ways and "
                  "symmetrized, Hellinger, Bhattacharyya, total variation, MMD, a quantile-vector "
                  "distance, lower/upper tail mismatch and a closed-form 1D optimal-transport "
                  "cost) against the corpus's pooled sentence/paragraph/word/turn distributions, "
                  "plus a within-document distance-correlation channel between naturally paired "
                  "sequences. Every distance is its own finding, kept separate from any p-value; "
                  "every family is independently switchable. Off by default; experimental."),
    _spec("mechanical_quality_suite", "mechanical_quality_suite", "lexical", "moderate",
          requires=("pyspellchecker", "symspellpy", "ftfy", "confusable_homoglyphs"),
          defaults={
              "features": {
                  "typography": True, "encoding": True, "encoding_ftfy": True,
                  "confusables": True, "hyphenation": True,
                  "spelling_pyspellchecker": True, "spelling_symspell": True,
                  "spelling_disagreement": True, "likely_typos": True,
                  "doubled_words": True, "fused_tokens": True, "ocr_substitution": True,
                  "confusion_pairs": True, "sentence_summary": True,
              },
              "language": "en",
              "recurring_min_count": 3,
              "likely_typo_max_count": 2,
              "likely_typo_max_candidates": 500,
              "fused_min_length": 8,
              "fused_max_candidates": 400,
              "max_reported": 25,
          },
          summary="Experimental grammar/spelling/typography/encoding mechanical-quality "
                  "diagnostics (LanguageTool replaced by Python-only checks -- see the module "
                  "docstring): mixed quote/apostrophe/dash/ellipsis style, control/zero-width/"
                  "replacement/private-use character rates, a diagnostic (non-mutating) ftfy "
                  "repair estimate plus an independent mojibake pattern scan, mixed-script "
                  "homoglyph detection, confirmed broken line-wrap hyphenation, two independent "
                  "dictionary spell checkers (pyspellchecker, SymSpell) with a disagreement "
                  "rate, likely-typo scoring that excludes a reported per-document 'recurring "
                  "vocabulary' of invented names/places, fused- and split-token detection, "
                  "OCR-substitution heuristics, and a small dialect-safe grammar rule set -- "
                  "every mechanical rate that risks reading dialect as an error is reported "
                  "separately for narration and dialogue. Off by default; experimental."),
    _spec("syntax_complexity_suite", "syntax_complexity_suite", "syntax", "parse",
          requires=("spacy", "benepar"),
          defaults={
              "features": {
                  "tunit_clause": True, "phrasal_elaboration": True,
                  "dependency_topology": True, "syntactic_surprisal": True,
                  "constituency": False,
              },
              "long_dependency_threshold": 10,
              "constituency_model": "benepar_en3",
              "constituency_sample_sentences": 30,
              "constituency_max_sentences": 60,
              "constituency_max_seconds": 180.0,
              "constituency_seed": 0,
          },
          summary="Experimental richer syntactic-complexity analysis beyond mean parse depth: "
                  "an L2SCA-style T-unit/clause ratio family approximated from spaCy dependency "
                  "labels (mean length of sentence/T-unit/clause, clauses and dependent clauses "
                  "per T-unit/clause, coordinate phrases and complex nominals per T-unit/clause, "
                  "verb phrases per T-unit, finite/nonfinite ratio -- every finding says it is an "
                  "approximation, not real L2SCA/TAASSC, which used Tregex over constituency "
                  "trees), phrasal elaboration (noun-phrase length/depth, pre/postmodifier "
                  "counts and diversity, PP-attachment density, appositive and participial-"
                  "modifier rates), dependency topology (branching factor, tree imbalance, head "
                  "direction, long-dependency rate, non-projective sentence rate, root-POS and "
                  "dependency-label/transition entropy, subtree-size distribution), and "
                  "corpus-trained syntactic surprisal (POS-bigram and dependency-label-bigram "
                  "cross-entropy plus a per-sentence surprisal distribution with top/bottom "
                  "evidence sentences -- never fit on the document being scored; needs a corpus "
                  "profile built with BOTH this suite enabled AND --parse-metrics, since this "
                  "suite's cost is 'parse'). A fifth, off-by-default feature adds "
                  "real constituency-tree measures (tree depth, phrase-type and production-rule "
                  "entropy, distinct-subtree rate, sentence-template diversity) from a real "
                  "neural constituency parser (benepar), over a bounded, seeded, deterministic "
                  "sample of sentences -- never a whole-book parse. Off by default; "
                  "experimental."),
    _spec("parser_consensus", "parser_consensus", "syntax", "parse",
          requires=("spacy", "pysbd", "nltk", "syntok", "stanza", "benepar"),
          defaults={
              "features": {
                  "segmentation": True, "tokenization": True, "pos": True, "dependency": True,
                  "chunks": True,
                  # Off even when the suite is on: each loads an extra model
                  # beyond the en_core_web_sm pipeline every cost="parse"
                  # metric already shares -- see the module docstring.
                  "spacy_md": False, "spacy_lg": False, "stanza": False,
                  "constituency_benepar": False,
              },
              "segmenters": ["builtin", "pysbd", "pysbd_quote_aware", "nltk_punkt",
                            "spacy_sentencizer", "spacy_parser", "syntok"],
              "long_sentence_words": 40,
              "max_paragraphs": 200,
              "max_words": 20000,
              "max_sentences_for_parse": 300,
              "max_seconds_parse": 180.0,
              "max_reported": 10,
              "seed": 0,
              "spacy_md_model": "en_core_web_md",
              "spacy_lg_model": "en_core_web_lg",
              "stanza_processors": "tokenize,mwt,pos,lemma,depparse",
              "benepar_model": "benepar_en3",
          },
          summary="Experimental parser/segmenter disagreement suite: up to seven independent "
                  "sentence segmenters (built-in, plain pySBD, the pipeline's quote-aware pySBD, "
                  "NLTK Punkt, spaCy's rule-based sentencizer, spaCy's parser-derived sentences, "
                  "syntok) compared by long-sentence-share and max-sentence-length -- the tail, "
                  "not the mean, so a plain-pySBD-style quote collapse is flagged even when it "
                  "barely moves a book's average -- plus pairwise boundary precision/recall/F1 "
                  "and all/majority/one-only boundary consensus; independent tokenizer "
                  "comparison (token-count/boundary-F1/special-token disagreement); and, once a "
                  "second parser is enabled (spaCy md/lg or stanza; a single parser alone "
                  "correctly reports 'insufficient consensus' rather than comparing one thing to "
                  "itself), POS/morphology agreement, UAS/LAS/root/label/depth/dependency-"
                  "distance agreement between the first two available parsers with an explicit "
                  "alignment-coverage figure, and a noun-phrase span F1 between spaCy's "
                  "dependency-derived chunks and a real constituency parse (stanza, or benepar "
                  "where it loads). Every comparison is over a deterministic, seeded, "
                  "spread-across-the-book sample, never the whole book. Off by default; "
                  "experimental."),
    _spec("conversation_suite", "conversation_suite", "dialogue", "moderate",
          requires=("networkx", "vaderSentiment", "nrclex"),
          defaults={
              "features": {
                  "attribution": True, "dominance": True, "alternation": True,
                  "length_accommodation": True, "function_word_coordination": True,
                  "contraction_convergence": True, "punctuation_convergence": True,
                  "lexical_entrainment": True, "turn_taking": True,
                  "response_relevance": True, "politeness": True,
                  "sentiment_coupling": True, "speaker_separability": True, "graph": True,
                  # Off even when the suite is on: each needs a heavier optional
                  # dependency (spaCy, a transformers model download, a
                  # sentence-transformers model) or extra per-window cost --
                  # see config.json's "_requires_*" notes.
                  "pos_convergence": False, "dialogue_act": False,
                  "response_relevance_embedding": False, "scene_drift": False,
              },
              "min_turns_per_speaker": 8,
              "min_attribution_coverage": 25.0,
              "min_coordination_pairs": 5,
              "min_entrainment_pairs": 10,
              "min_graph_edges": 3,
              "min_coupling_pairs": 30,
              "rare_word_zipf_threshold": 3.0,
              "shuffle_seed": 0,
              "separability_max_turns": 400,
              "response_relevance_model": "all-MiniLM-L6-v2",
              "dialogue_act_model": "WSHAPER/distilbert-multilingual-dialogue-act-classifier",
              "dialogue_act_max_turns": 200,
              "pos_convergence_max_pairs": 300,
              "scene_window_words": 6000,
          },
          summary="Experimental conversational-dynamics suite built on the existing dialogue "
                  "extraction/attribution: Danescu-Niculescu-Mizil-style directional "
                  "coordination (turn-length, sentence-count, function-word category, "
                  "contraction, question-mark, exclamation-mark -- does a responder's own rate "
                  "rise right after the other speaker's did, above the responder's own "
                  "reply-position baseline?); shuffled-partner-controlled lexical and rare-word "
                  "entrainment and response relevance (lexical by default, a real "
                  "sentence-transformers embedding backend as an off-by-default upgrade); "
                  "speaker turn-count dominance (Gini) and alternation/run-length; "
                  "question-response/unanswered-question/backchannel rates; a leave-one-out "
                  "nearest-centroid speaker-separability accuracy beyond the existing "
                  "function-word-distance metrics; a regex approximation of "
                  "Danescu-Niculescu-Mizil & Lee's politeness strategies; VADER/NRC "
                  "sentiment/emotion response coupling and per-speaker sentiment "
                  "differentiation (reusing sequences.py's two scorers, never a new sentiment "
                  "engine); a conversation graph (density, reciprocity, cross-checked against "
                  "networkx where installed); and, off by default, POS-pattern convergence "
                  "(needs spaCy), a versioned dialogue-act classifier's label distribution and "
                  "transition entropy (needs transformers/torch), and scene-window style "
                  "drift. Every genuinely per-speaker claim is suppressed below "
                  "min_attribution_coverage/min_turns_per_speaker, mirroring "
                  "dialogue_speaker_style's own gate. Off by default; experimental."),
    _spec("graph_suite", "graph_suite", "discourse", "parse",
          requires=("spacy", "networkx", "fastcoref", "booknlp"),
          defaults={
              "features": {
                  "lexical_chain": True, "surface_name": True, "entity_cooccurrence": True,
                  "character_cooccurrence": True, "paragraph_entity_overlap": True,
                  "quote_speaker": True, "topic_transition": True, "temporal_drift": True,
                  "sentence_semantic": False, "paragraph_semantic": False,
                  "coreference_entity": False, "dependency_relation": False, "booknlp": False,
              },
              "min_word_len": 3,
              "lexical_chain_gap": 3,
              "lexical_chain_min_length": 2,
              "surface_name_window_sentences": 3,
              "entity_max_tracked": 150,
              "entity_graph_window_sentences": 3,
              "character_max_tracked": 150,
              "character_graph_window_sentences": 3,
              "min_mentions_for_graph": 2,
              "paragraph_overlap_window": 5,
              "topic_n_topics": 6,
              "topic_max_paragraphs": 2000,
              "semantic_model": "all-MiniLM-L6-v2",
              "semantic_graph_k": 5,
              "semantic_graph_min_similarity": 0.5,
              "semantic_graph_max_units": 1500,
              "coreference_model": "biu-nlp/f-coref",
              "coreference_max_words": 4000,
              "booknlp_model": "small",
              "booknlp_pipeline": "entity,quote,supersense,event,coref",
              "booknlp_max_words": 3000,
              "seed": 0,
          },
          summary="Experimental network-science suite: named-entity/character/surface-name "
                  "co-occurrence graphs, a paragraph entity-overlap graph, a quote-speaker "
                  "interaction graph, a TF-IDF/k-means topic-transition graph, a lexical-chain "
                  "graph, an early-vs-late-half temporal drift channel (edge-set change and "
                  "community persistence), plus (off by default) sentence/paragraph "
                  "embedding-similarity graphs, a real-coreference-resolved entity graph, a "
                  "dependency-relation (POS-pair) aggregate graph, and real BookNLP "
                  "character/quote extraction. Every graph reports the same battery: node/edge "
                  "counts, density, degree shape, components, clustering, assortativity, bounded "
                  "path statistics, seeded Louvain community structure, PageRank/betweenness/"
                  "degree centralization, edge-weight entropy and node-reappearance distance, "
                  "plus an independent igraph cross-check kept as a disagreement channel where "
                  "igraph is installed. Off by default; experimental."),
    _spec("lexical_norms_suite", "lexical_norms_suite", "lexical", "moderate",
          requires=("wordfreq", "openpyxl", "lexical_diversity", "taaled", "lftk"),
          defaults={
              "features": {
                  "concreteness": True, "age_of_acquisition": True, "warriner_vad": True,
                  "nrc_vad": True, "sensorimotor": True, "subtlex": True, "glasgow": True,
                  "mrc": True, "frequency_source_agreement": True, "lexdiv_crosscheck": True,
                  "taaled_crosscheck": True,
                  # Off by default: each forces the shared spaCy parse, which this suite's
                  # "moderate" cost class does not otherwise pay for -- see the module
                  # docstring's "Surface vs lemma" / "Cross-checks" sections.
                  "lemma_lookup": False, "lftk_crosscheck": False,
              },
              "language": "en",
              "resource_paths": {},
              "diversity_crosscheck_max_tokens": 50_000,
              "diversity_crosscheck_window": 50,
              "lftk_max_chars": 200_000,
          },
          summary="Experimental lexical sophistication and psycholinguistic norm analysis: "
                  "concreteness (Brysbaert), age of acquisition (Kuperman, surface and "
                  "lemma-aggregated), Warriner and NRC valence/arousal/dominance (kept as "
                  "independent resources), Lancaster sensorimotor perceptual/action strength, "
                  "SUBTLEX-US Zipf frequency and contextual diversity, all nine Glasgow Norms "
                  "scales and all six rated MRC Psycholinguistic Database dimensions -- each as "
                  "a token-weighted and a type-weighted finding, both carrying coverage, "
                  "quantiles, table-referenced low/high tail shares, sentence- and "
                  "paragraph-level distributions, between-paragraph variance, early-vs-late "
                  "drift and a dialogue-vs-narration difference. Plus a wordfreq-vs-SUBTLEX-US "
                  "frequency agreement check, two independent lexical-diversity cross-checks "
                  "(the lexical_diversity package and TAALED itself, alongside -- never "
                  "replacing -- mattr.py/lexical_mtld.py/lexical_hdd.py), and two off-by-default, "
                  "parse-requiring extras (a true lemma-normalized AoA lookup, and an LFTK "
                  "cross-check against its own bundled Kuperman/Brysbaert/SUBTLEX-US tables). "
                  "Every norm resource is downloaded, cached and versioned by "
                  "textgrader.lexicons (see config.json's _requires_ notes and that module's "
                  "docstring), never bundled in the repository; a resource that is not cached "
                  "or not licensable (English Lexicon Project, CELEX) degrades only its own "
                  "findings. Off by default; experimental."),
])

#: Config-name -> module-name, kept for older callers.
MODULES = {name: spec.module for name, spec in REGISTRY.items()}

#: Metrics that need a spaCy parse.  They share one parse per document.
NLP_METRICS = {name for name, spec in REGISTRY.items() if spec.needs_parse}

#: Metrics that will load a sentence-embedding model when one is installed.
MODEL_METRICS = {name for name, spec in REGISTRY.items() if spec.needs_model}

#: Metrics whose value scales with how much text you supply.
FAMILIES = sorted({spec.family for spec in REGISTRY.values()})



def is_enabled(metric_config: Mapping[str, Any] | None, name: str) -> bool:
    """Whether ``name`` is switched on in a ``metrics`` config mapping.

    Lives here rather than in ``grade.py`` because the corpus builder needs the
    same answer: a profile that precomputes a different set of metrics than the
    run it is compared against is the kind of mismatch this project refuses
    everywhere else.
    """

    setting = (metric_config or {}).get(name)
    if setting is None:
        return False
    return setting is True or (isinstance(setting, dict) and bool(setting.get("enabled", False)))


def specs_by_family() -> dict[str, list[MetricSpec]]:
    out: dict[str, list[MetricSpec]] = {}
    for spec in REGISTRY.values():
        out.setdefault(spec.family, []).append(spec)
    return out
