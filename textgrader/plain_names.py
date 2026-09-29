"""Plain-language descriptions for measurements whose names a writer cannot act on.

A name like "Spectral centroid of sentence words" is accurate and useless to
someone revising prose: an agent handed it did not know which sentences to
change or in which direction.  :func:`describe` returns, for those metrics
only, what the number tracks in the text and what moves it, in the writer's
terms.  Metric ids and their original names are unchanged; this is shown
beside them.

Most obscure names come from a few families that apply one feature to
several sequences (sentence length, paragraph length, punctuation per
sentence), so those are generated; the rest are listed one by one.  A metric
with no entry returns ``None`` and keeps only its own name.
"""

from __future__ import annotations

import re

#: What each sequence is, in the words a writer would use.
SEQUENCES = {
    "sentence_words": "sentence length",
    "paragraph_words": "paragraph length",
    "paragraph_sentences": "sentences per paragraph",
    "sentence_punctuation": "punctuation marks per sentence",
    "sentence_commas": "commas per sentence",
    "word_characters": "word length",
    "turn_words": "dialogue turn length",
    "turn_sentences": "sentences per dialogue turn",
}

SIGNAL = {
    "zero_crossing_rate": "how often {s} flips between above and below its average from one to "
                          "the next; high = long and short alternate, low = similar values come in runs",
    "peaks": "how many lone spikes {s} has (one value above both neighbours), per 100; "
             "high = frequent single long ones between shorter ones",
    "spectral_centroid": "how fast {s} changes; high = quick back-and-forth from one to the next, "
                         "low = slow swells and lulls over many in a row",
    "spectral_bandwidth": "how mixed the speeds of change in {s} are; high = quick alternation and "
                          "slow swells together, low = one steady kind of rhythm",
    "spectral_flatness": "how patternless {s} is; high = no rhythm to the order, low = a regular, "
                         "repeating rhythm",
    "spectral_rolloff": "how much of the change in {s} is quick back-and-forth; high = mostly quick "
                        "alternation, low = mostly slow swells",
    "band_energy": "share of the variation in {s} that comes from slow swells across many in a "
                   "row; high = runs of similar values that shift gradually, low = quick alternation",
    "welch_spectral": "how evenly the variation in {s} is spread over fast and slow change; "
                      "high = no dominant rhythm",
}

RQA = {
    "recurrence_rate": "how often two stretches of {s} anywhere in the text look alike",
    "determinism": "how often a stretch of {s} repeats the pattern of an earlier stretch; "
                   "high = recurring rhythmic patterns",
    "laminarity": "how often {s} stays at a similar level several times in a row; high = runs of "
                  "similar values, low = constant alternation",
    "trapping_time": "average length of a run of similar {s} values",
    "longest_vertical_line": "the longest run of similar {s} values",
    "longest_diagonal_line": "the longest stretch where the {s} pattern repeats an earlier stretch",
    "avg_diagonal_length": "average length of a repeated {s} pattern",
    "diagonal_entropy": "variety in how long repeated {s} patterns last; low = repeats are all "
                        "brief, high = a mix of brief and long repeats",
    "recurrence_time": "average gap before a similar {s} pattern comes back",
    "threshold": "how close two {s} values must be to count as similar; set from this text's own "
                 "spread, so high = {s} varies widely",
    "trend": "whether similar {s} patterns cluster together (drift through the text) or spread "
             "evenly; high = the rhythm changes as the text goes on",
}

TIMESERIES = {
    "acf": "how much each {s} predicts the next; high = similar values follow each other",
    "dispersion": "how widely {s} varies around its middle",
    "runs": "the longest unbroken run of {s} on one side of its median, as a share of the text",
    "trend": "whether {s} rises or falls from the start of the text to the end",
    "turning_points": "how often {s} changes direction (up then down or down then up), per 100",
}

STATS = {
    "cv": "how much {s} varies relative to its average (standard deviation / mean)",
    "iqr": "spread of the middle half of {s} (75th minus 25th percentile)",
    "kurtosis": "how extreme the outliers in {s} are; high = rare very short and very long ones",
    "mad": "typical distance of {s} from its median",
    "range": "longest minus shortest {s}",
    "skewness": "how lopsided {s} is; high = mostly low values with a few very high ones",
    "mean": "average {s}",
    "median": "median {s}",
    "std": "standard deviation of {s}",
}

DISTRIBUTION = {
    "tail_lower": "how different the low end (10th percentile) of {s} is from the corpus's",
    "tail_upper": "how different the high end (90th percentile) of {s} is from the corpus's",
    "quantile_vector": "average gap between this text's percentiles of {s} and the corpus's",
}

EXACT = {
    "style.randomness_char_run_length_entropy":
        "how varied runs of the same character are: mostly how often words with doubled letters "
        "appear ('all', 'look', 'really', 'little'); typed '--' dashes add a little",
    "style.randomness_consonant_cluster_rate":
        "share of words with four or more consonants in a row ('strengths', 'rhythm')",
    "rhythm.prosody_stress_entropy":
        "how evenly strongly stressed, lightly stressed and unstressed syllables are mixed; "
        "high = more polysyllabic words with secondary stress",
    "rhythm.prosody_meter_conformity_rate":
        "how regularly each line's syllables fall into one repeating stress pattern "
        "(da-DUM da-DUM); high = more metrical sentences",
    "rhythm.prosody_consonance_density":
        "how often nearby content words end in the same consonant sound ('back ... luck')",
    "syntax.complexity_l2sca_tunits_per_sentence":
        "independent clauses per sentence; high = more sentences joined with 'and', 'but', 'so' "
        "or semicolons",
    "discourse.coherence_entity_grid_transition_entropy_dialogue":
        "how varied the roles of recurring people and things are from one dialogue line to the "
        "next (subject, object, mentioned, absent); low = lines keep the same focus",
    "discourse.coherence_entity_dangling_rate_dialogue":
        "share of people and things introduced in dialogue that are never mentioned again",
    "dialogue.conversation_response_relevance":
        "how much replies reuse words from the line they answer, beyond what any other line "
        "would share with them",
    "dialogue.conversation_lexical_entrainment":
        "how much speakers pick up each other's words from one turn to the next, beyond chance",
    "dialogue.conversation_rare_word_entrainment":
        "how much speakers pick up each other's uncommon words from one turn to the next, "
        "beyond chance",
    "semantic.structure_bm25_centroid_relatedness":
        "how strongly each sentence ties to the whole text's vocabulary; high = the text keeps "
        "circling the same words",
    "style.autofeature_manifest": "bookkeeping: how many automatic features were computed",
    "rhythm.signal_sentence_words_x_sentence_punctuation_coherence":
        "how closely sentence length and punctuation per sentence rise and fall together; "
        "high = longer sentences reliably carry more punctuation",
    "rhythm.signal_sentence_words_x_sentence_punctuation_cross_correlation":
        "how closely sentence length and punctuation per sentence move together from one "
        "sentence to the next",
}

#: Two-sample tests between this text's values and the corpus's pooled ones.
DISTANCE_TESTS = ("anderson_darling", "cvm", "ks", "energy", "hellinger", "jensen_shannon",
                  "kl_forward", "kl_reverse", "kl_symmetric", "mmd", "optimal_transport",
                  "total_variation", "wasserstein", "bhattacharyya")

_FAMILIES = [
    (re.compile(r"^rhythm\.signal_(?P<seq>[a-z_]+?)_(?P<feat>" + "|".join(SIGNAL) + r")$"), SIGNAL),
    (re.compile(r"^drift\.nonlinear_(?P<seq>[a-z_]+?)_rqa_(?P<feat>" + "|".join(RQA) + r")$"), RQA),
    (re.compile(r"^rhythm\.timeseries_(?P<seq>[a-z_]+?)_(?P<feat>" + "|".join(TIMESERIES) + r")$"),
     TIMESERIES),
    (re.compile(r"^style\.autofeature_(?P<seq>[a-z_]+?)_stats_(?P<feat>" + "|".join(STATS) + r")$"),
     STATS),
    (re.compile(r"^style\.distribution_(?P<seq>[a-z_]+?)_(?P<feat>" + "|".join(DISTRIBUTION) + r")$"),
     DISTRIBUTION),
]
_DISTANCE = re.compile(r"^style\.distribution_(?P<seq>[a-z_]+?)_(?P<test>"
                       + "|".join(DISTANCE_TESTS) + r")(?P<p>_pvalue)?$")
_QUANTILE = re.compile(r"^style\.autofeature_(?P<seq>[a-z_]+?)_stats_quantile_p(?P<p>\d+)$")
_DISAGREEMENT = re.compile(r"^nlp\.readability_(?P<formula>[a-z_]+?)_disagreement$")


def _sequence(key: str) -> str:
    return SEQUENCES.get(key, key.replace("_", " "))


def describe(metric_id: str) -> str | None:
    """What ``metric_id`` tracks in the prose, or ``None`` if its name already says."""

    if metric_id in EXACT:
        return EXACT[metric_id]
    match = _QUANTILE.match(metric_id)
    if match:
        return (f"the {int(match['p'])}th-percentile {_sequence(match['seq'])}: "
                f"{int(match['p'])}% of them are this value or lower")
    for pattern, table in _FAMILIES:
        match = pattern.match(metric_id)
        if match:
            return table[match["feat"]].format(s=_sequence(match["seq"]))
    match = _DISTANCE.match(metric_id)
    if match:
        test, seq = match["test"].replace("_", " "), _sequence(match["seq"])
        if match["p"]:
            return (f"chance that this text's spread of {seq} came from the corpus's own spread "
                    f"({test} test); low = shaped differently")
        return (f"how differently {seq} is spread than across the corpus ({test}); "
                f"high = a different mix of low, typical and high values")
    match = _DISAGREEMENT.match(metric_id)
    if match:
        formula = match["formula"].replace("_", " ")
        return (f"how much different implementations of the {formula} readability formula "
                f"disagree on this text; driven by how each counts sentences and syllables, so "
                f"it moves with unusual punctuation, fragments and dialogue formatting")
    return None
