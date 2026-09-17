"""Burrows's Delta over a fixed function-word list, against a corpus profile.

Function words (articles, pronouns, prepositions, conjunctions) are a
well-established authorship signal because they are used unconsciously and
far more frequently than content words, so their per-1,000-word rates form a
stable per-author fingerprint.  Delta is the mean absolute z-score of this
document's rates against the corpus mean and standard deviation for each
word: how many standard deviations away, on average, this document's
function-word habits sit from the corpus's.

``FUNCTION`` (the fixed word list) and ``vector()`` (the rate calculator) are
imported directly by other modules (``character_voice``,
``dialogue_speaker_function_words``, ``discourse_constructions``) that need
the same per-speaker or per-text function-word profile, so both keep their
exact names and signatures here.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from .common import FAST, finding
from .common import tokens as tokenize

FAMILY = "authorial"
COST = FAST
REQUIRES: tuple[str, ...] = ()
MIN_SAMPLE = 200
UNIT_SENSITIVE = False

FUNCTION = ("a an and are as at be been but by for from had has have he her him his i if in "
           "is it its me my nor not of on or our she so than that the their them then there "
           "they this to us was we were what when which who will with you your").split()


def vector(text: str) -> dict[str, float]:
    """Function-word rate per 1,000 words, keyed by word.  A standalone
    function of raw text so callers can profile a speaker's lines or a
    transcript turn without building a full :class:`DocumentAnalysis`."""

    tokens = tokenize(text)
    if not tokens:
        return {}
    counts = Counter(tokens)
    return {word: 1000 * counts[word] / len(tokens) for word in FUNCTION}


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    total = analysis.word_count
    if total == 0:
        return [finding("style.function_word_delta", "Function-word Burrows Delta", None, "delta",
                        family=FAMILY, sample_size=0, min_sample=MIN_SAMPLE,
                        warning="no words to measure function-word rates over")]

    counts = Counter(analysis.tokens)
    document_rates = {word: 1000 * counts[word] / total for word in FUNCTION}

    rows = (profile or {}).get("feature_profiles", {}).get("function_words", [])
    if not rows:
        return [finding("style.function_word_delta", "Function-word Burrows Delta", None, "delta",
                        family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE,
                        warning="no corpus function-word profiles available")]

    means = {word: sum(row.get(word, 0) for row in rows) / len(rows) for word in FUNCTION}
    sds = {word: (sum((row.get(word, 0) - means[word]) ** 2 for row in rows) / len(rows)) ** 0.5
          for word in FUNCTION}
    z_scores = {word: abs(document_rates.get(word, 0) - means[word]) / sds[word]
               for word in FUNCTION if sds[word]}
    delta = sum(z_scores.values()) / len(z_scores) if z_scores else None

    evidence = [{"word": word, "z_score": z} for word, z in
                sorted(z_scores.items(), key=lambda item: -item[1])[:15]]
    return [finding(
        "style.function_word_delta", "Function-word Burrows Delta", delta, "delta",
        family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE, evidence=evidence,
        distribution={"corpus_size": len(rows), "words_with_variance": len(z_scores)},
        warning=None if z_scores else "corpus function-word profile had zero variance for every word")]
