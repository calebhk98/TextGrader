"""Word rarity from live general-language frequencies, not a bundled word list.

TextGrader used to ship a static ``word_frequency.json`` snapshot; a stale or
contaminated snapshot (built, for instance, from a corpus that overlaps with
the very manuscripts being graded) quietly inflates or deflates every rarity
call made against it, and there is no way for a reader of the report to tell.
``wordfreq`` computes Zipf frequencies from its own maintained, cited corpora
at call time, so this module replaces the bundled list outright rather than
falling back to it.

The Zipf scale (Van Heuven et al. 2014) is log-based and lands most English
text between about 1 (vanishingly rare) and 7 (extremely common, "the"-class);
a drop of one point is a roughly ten-fold change in how often the word is
used. This is a lexicon-frequency proxy for "how easy is this word", not a
readability or difficulty score: a rare proper noun and a rare technical term
score the same low Zipf value for the same reason (wordfreq has rarely seen
either), even though only one of them is a vocabulary demand on the reader.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

from ..document import DocumentAnalysis
from ..optional import require
from ..stats import histogram, summarize
from .common import MODERATE, finding, option, unavailable

FAMILY = "lexical"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("wordfreq",)
MIN_SAMPLE = 50
UNIT_SENSITIVE = False

RARE_ZIPF = 3.0
VERY_RARE_ZIPF = 2.0
HISTOGRAM_EDGES = (2.0, 3.0, 4.0, 5.0, 6.0)

_IDS_NAMES = (
    ("lexical.word_zipf", "Word rarity (Zipf frequency)"),
    ("lexical.rare_word_share", "Share of rare words (Zipf < 3)"),
    ("lexical.very_rare_word_share", "Share of very rare words (Zipf < 2)"),
)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    language = option(config, "language", "en")
    tokens = analysis.tokens
    if not tokens:
        return [unavailable(mid, name, "no words to score", family=FAMILY)
                for mid, name in _IDS_NAMES]

    module, reason = require("wordfreq")
    if module is None:
        return [unavailable(mid, name, reason, family=FAMILY) for mid, name in _IDS_NAMES]

    counts = Counter(tokens)
    # zipf_frequency is a lookup into wordfreq's own tables plus a language
    # normalisation step; scoring it per distinct type and broadcasting to
    # occurrences, instead of per token, is what keeps a 400,000-word novel
    # (repeating a few thousand distinct words hundreds of thousands of times)
    # out of FAST-metric-costs-a-minute territory.
    zipf_by_word = {word: module.zipf_frequency(word, language) for word in counts}
    scores = [zipf_by_word[word] for word, count in counts.items() for _ in range(count)]

    distribution = summarize(scores)
    distribution["histogram"] = histogram(scores, HISTOGRAM_EDGES)
    total = len(scores)
    rare_share = 100.0 * sum(1 for value in scores if value < RARE_ZIPF) / total
    very_rare_share = 100.0 * sum(1 for value in scores if value < VERY_RARE_ZIPF) / total

    rarest = sorted(zipf_by_word.items(), key=lambda item: (item[1], item[0]))[:20]
    evidence = [{"word": word, "zipf": score, "count": counts[word]} for word, score in rarest]

    return [
        finding("lexical.word_zipf", "Word rarity (Zipf frequency)", distribution.get("median"),
                "zipf", family=FAMILY, sample_size=total, distribution=distribution,
                evidence=evidence, min_sample=MIN_SAMPLE),
        finding("lexical.rare_word_share", "Share of rare words (Zipf < 3)", rare_share, "%",
                family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE),
        finding("lexical.very_rare_word_share", "Share of very rare words (Zipf < 2)",
                very_rare_share, "%", family=FAMILY, sample_size=total, min_sample=MIN_SAMPLE),
    ]
