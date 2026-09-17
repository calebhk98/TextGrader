"""TextGrader's core prose measurements.

These are the measurements every run makes: reading grade, sentence and
paragraph shape, lexical diversity, the clause-cue proxies and the size counts.
They lived in ``measures/prose_grade.py`` and were loaded by file path with
``importlib.util.spec_from_file_location``, because ``measures/`` was a folder
of scripts rather than a package.  Two callers did that, each getting its own
copy of the module and its own caches.  They are library code, so they live in
the library.

Nothing here assumes a manuscript, a reader age or a maturity target.  The
regex-labelled clause measurements are honestly named lexical proxies: they
match cue words and do not parse English grammar.  The ``clause_types`` metric
is the parse-based replacement.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path

from .document import DocumentAnalysis
from .paths import DATA_DIR

ROOT = Path(__file__).resolve().parents[1]

def in_twenty_to_thirty_five(word_count):
    return 20 <= word_count <= 35
REFERENCE = DATA_DIR / "prose_reference.json"

SUBORDINATOR = (r"\b(because|although|though|while|whereas|since|unless|until|after|before"
                r"|if|when|whenever|as|so that|even though|rather than|whether)\b")
RELATIVE = r"\b(who|whom|whose|which|that)\b"

# A sentence that opens on a subordinate clause and closes it with a comma.
# An opening quote is allowed because dialogue may use this construction too.

# The narrative negative: a sentence whose beat is an action not taken or a
# reaction that did not happen. "He doesn't look up." "Nobody says anything."
# "She reads it twice and doesn't add to the thread." One of these is a device.
# This is a lexical proxy, not a judgement about the purpose of the negative.
NEGATIVE = re.compile(
    r"\b(?:doesn't|does not|didn't|did not|don't|do not|isn't|is not|wasn't"
    r"|was not|aren't|weren't|never)\s+[a-z]"
    r"|\b(?:nobody|no one|nothing|none of them)\b"
    r"|\bwithout \w+ing\b", re.I)

FRONTLOAD = re.compile(
    r"""^["']?(?:Because|Since|While|Though|Although|When|Whenever|As|After"""
    r"""|Before|If|Once|Until|Unless|Whether|Where)\b[^,.!?]{3,60},""")

# The 100 commonest English words. A high share of these means a small
# working vocabulary, which is one of the two things that reads young.
TOP100 = set("""the be to of and a in that have i it for not on with he as you do at this
but his by from they we say her she or an will my one all would there their what so up out
if about who get which go me when make can like time no just him know take people into year
your good some could them see other than then now look only come its over think also back
after use two how our work first well way even new want because any these give day most us""".split())

# metric key -> (label, higher_is_more_mature)
METRICS = {
    "fk":        ("reading grade (Flesch-Kincaid)", True),
    "wps":       ("words per sentence", True),
    "sttr":      ("lexical diversity (sTTR)", True),
    "commas":    ("commas per sentence", True),
    "subord":    ("sentences with a subordinate clause %", True),
    "relcl":     ("sentences with a relative clause %", True),
    "b2035":     ("sentences of 20-35 words %", True),
    "long7":     ("words of 7+ characters %", True),
    "u10":       ("sentences under 10 words %", False),
    "simple":    ("sentences with no subordinate/relative clause %", False),
    "shortruns": ("sentences inside a run of 3+ short ones %", False),
    "top100":    ("words from the commonest 100 %", False),
    "slcv":      ("sentence length variation (CV %)", True),
    "wpp":       ("words per paragraph", True),
    "spp":       ("sentences per paragraph", True),
    "wlen":      ("mean word length (characters)", True),
    "ari":       ("reading grade (ARI)", True),
    "lexile":    ("Lexile (approximate, see note)", True),
}

# Historical CLI summary group, kept for output compatibility.
CORE12 = ("fk wps sttr commas subord relcl b2035 long7 u10 simple shortruns "
          "top100").split()

# Measured, reported against the corpus range, deliberately not graded.
MONITOR = {
    "front":   "sentences opening on a subordinate clause %",
    "and2":    'sentences with two or more "and" %',
    "andrate": '"and" as a share of all words %',
    "negative": "sentences whose beat is a negative %",
}


def syllables(word):
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


WORDFREQ = DATA_DIR / "word_frequency.json"
_FREQ_CACHE = {}

#: The bundled table is off by default and stays in the repository on purpose.
#: It was built from Gutenberg text that still contained the licence boilerplate,
#: which inflates words like "project", "gutenberg", "ebook" and "copyright", and
#: the coefficients in :func:`lexile` were fitted against that contaminated table
#: rather than against a published Lexile corpus.  Deleting it would silently
#: change every historical number; enabling it silently would present an
#: untrustworthy one.  So it is a choice the configuration has to make.
FREQUENCY_SOURCES = ("none", "bundled", "wordfreq")


def word_frequency(source="none"):
    """Return ``{word: per-million frequency}`` for the requested source.

    ``none``
        the default.  Approximate Lexile is then unavailable rather than wrong.
    ``bundled``
        ``word_frequency.json`` as shipped, Gutenberg boilerplate included.
        This reproduces every number this repository ever printed.
    ``wordfreq``
        general-language frequencies from the ``wordfreq`` package, which are
        far cleaner but are NOT what the Lexile coefficients were fitted to, so
        the resulting scale is different again.  Both facts are reported.
    """

    if source in _FREQ_CACHE:
        return _FREQ_CACHE[source]
    table = {}
    if source == "bundled" and WORDFREQ.is_file():
        # The file is ``{"tokens": ..., "freq_per_million": {word: rate}}``, so
        # the rates are one level in.  Reading the outer object instead leaves
        # every lookup missing and every word counted as rare, which does not
        # fail loudly: it returns a Lexile four hundred points high.
        try:
            table = json.loads(WORDFREQ.read_text(encoding="utf-8"))["freq_per_million"]
        except Exception:
            # Deliberately broad.  This loader feeds one optional metric, and
            # its caller is inside the core analysis, so anything raised here
            # costs all eighteen core measurements rather than just Lexile.
            # An empty table makes ``lexile`` return None, which is the
            # honest outcome: the number is unavailable, the rest still run.
            table = {}
        if not isinstance(table, dict):
            table = {}
    _FREQ_CACHE[source] = table
    return table


def _wordfreq_lookup():
    from .optional import require
    module, _ = require("wordfreq")
    if module is None:
        return None
    return lambda word: 10 ** (module.zipf_frequency(word, "en") - 6) * 1_000_000


def lexile(sent_lengths, lower_words, source="none"):
    """Approximate Lexile: sentence length against word familiarity.

    Stenner's two-variable shape — long sentences push the number up, common
    words pull it down, so a page of long sentences made of familiar words and
    a page of short sentences made of rare ones can land in the same place.

    Two things make this an approximation rather than a Lexile measure. The
    frequency table is built from the 23-book reference corpus instead of the
    proprietary Lexile corpus, and the map from the raw score onto the Lexile
    scale is fitted here rather than published. The fit is a least-squares line
    through twelve children's classics in that corpus whose Lexile measures are
    published (Alice 850L, Five Children and It 810L, Peter Pan 920L, The
    Railway Children 970L, A Little Princess 970L, The Secret Garden 970L, Tom
    Sawyer 950L, Anne of Green Gables 990L, Black Beauty 1010L, Treasure Island
    1080L, The Wind in the Willows 1140L, Little Women 1300L). Mean absolute
    error against those twelve is 77L and the worst is 199L, on Little Women.

    So: read it to +/- 100L, and use it to watch a chapter move relative to the
    others rather than to claim a chapter is exactly 1000L. Recalibrate here if
    the corpus changes.

    Words the corpus has never seen count as rare rather than being dropped,
    since dropping them would make invented and technical vocabulary free.
    """
    if source not in FREQUENCY_SOURCES or source == "none":
        return None
    if not sent_lengths or not lower_words:
        return None
    lookup = None
    freq = {}
    if source == "wordfreq":
        lookup = _wordfreq_lookup()
        if lookup is None:
            return None
    else:
        freq = word_frequency(source)
        if not freq:
            return None
    RARE = 0.05  # per million: about one appearance across a twenty-book shelf
    seen = {}
    def rate(word):
        if word not in seen:
            seen[word] = lookup(word) if lookup else freq.get(word, RARE)
        return seen[word]
    log_word_frequencies = [math.log10(max(rate(item), RARE) * 5) for item in lower_words]
    raw = 9.82247 * math.log(statistics.fmean(sent_lengths)) - 2.14634 * statistics.fmean(log_word_frequencies)
    return 46.45 * raw + 102.45


def analysis_for(text, config=None):
    """Build the shared pipeline object these metrics measure."""

    if isinstance(text, DocumentAnalysis):
        return text
    from .document import NlpSettings, TextProcessing
    config = config or {}
    return DocumentAnalysis.from_text(
        text,
        processing=TextProcessing.from_config(config.get("text_processing")),
        nlp_settings=NlpSettings.from_config(config.get("nlp")),
        comparison_unit=config.get("analysis", {}).get("comparison_unit", "unknown"))


def measure(text, floor=40, config=None, lexile_source="none"):
    """Every core metric for one text, computed from the shared pipeline.

    ``text`` may be a string or an already-built
    :class:`textgrader.document.DocumentAnalysis`.  Passing the analysis is what
    guarantees that these numbers and the optional metrics in ``grade.py``
    describe the same document: cleanup, tokenization and sentence segmentation
    happen once, in ``textgrader/document.py``, not once per caller.

    Returns ``None`` under ``floor`` sentences.  The floor exists so a
    percentile is not read off four sentences; ``grade.py`` measures below it on
    purpose and then withholds the corpus comparison instead, which is more
    useful than returning nothing.

    ``sttr`` comes back ``None`` when the text is shorter than one 1000-word
    window.  Type-token ratio falls as a text gets longer, so measuring it over
    a 600-word chapter and comparing that with a 70,000-word book flatters the
    chapter by twenty points or more.  The sampled form exists to remove that
    dependence and cannot do so without a full window.
    """

    analysis = analysis_for(text, config)
    sentences = analysis.sentences
    sentence_lengths = analysis.sentence_lengths
    keep = [index for index, length in enumerate(sentence_lengths) if length]
    sentences = [sentences[index] for index in keep]
    sentence_lengths = [sentence_lengths[index] for index in keep]
    if len(sentences) < floor:
        return None
    body = analysis.text
    word_tokens = analysis.words
    long_words = analysis.tokens
    sentence_count, word_count = len(sentence_lengths), len(word_tokens)
    if not word_count:
        return None

    runs = short_run_sentences = 0
    for value in sentence_lengths:
        if value < 8:
            runs += 1
        else:
            short_run_sentences += runs if runs >= 3 else 0
            runs = 0
    short_run_sentences += runs if runs >= 3 else 0

    type_token_windows = [len(set(long_words[index:index + 1000])) / 1000
                          for index in range(0, len(long_words) - 999, 1000)]
    words_per_sentence = statistics.fmean(sentence_lengths)

    paragraph_word_counts = [count for count in analysis.paragraph_lengths if count]
    paragraph_sentence_counts = [count for count, words_in in
                                 zip(analysis.paragraph_sentence_counts,
                                     analysis.paragraph_lengths) if words_in and count]
    multiple_and_count = sum(1 for sentence in sentences
                             if len(re.findall(r"\band\b", sentence.lower())) >= 2)
    negative = sum(1 for sentence in sentences if NEGATIVE.search(sentence))
    character_count = sum(len(item) for item in word_tokens)

    return {
        "fk": 0.39 * words_per_sentence + 11.8 * (sum(syllables(item) for item in word_tokens) / word_count) - 15.59,
        "ari": 4.71 * (character_count / word_count) + 0.5 * words_per_sentence - 21.43,
        "lexile": lexile(sentence_lengths, long_words, lexile_source),
        "wps": words_per_sentence,
        "sttr": 100 * statistics.fmean(type_token_windows) if type_token_windows else None,
        "commas": body.count(",") / sentence_count,
        "subord": 100 * sum(1 for sentence in sentences if re.search(SUBORDINATOR, sentence, re.I)) / sentence_count,
        "relcl": 100 * sum(1 for sentence in sentences if re.search(RELATIVE, sentence, re.I)) / sentence_count,
        "b2035": 100 * sum(1 for value in sentence_lengths if in_twenty_to_thirty_five(value)) / sentence_count,
        "long7": 100 * sum(1 for item in word_tokens if len(item) >= 7) / word_count,
        "u10": 100 * sum(1 for value in sentence_lengths if value < 10) / sentence_count,
        "simple": 100 * sum(1 for sentence in sentences
                            if not re.search(SUBORDINATOR, sentence, re.I)
                            and not re.search(RELATIVE, sentence, re.I)) / sentence_count,
        "shortruns": 100 * short_run_sentences / sentence_count,
        "top100": 100 * sum(1 for item in long_words if item in TOP100) / word_count,
        "slcv": 100 * statistics.stdev(sentence_lengths) / words_per_sentence if len(sentence_lengths) > 1 else 0.0,
        "wpp": statistics.fmean(paragraph_word_counts) if paragraph_word_counts else 0.0,
        "spp": statistics.fmean(paragraph_sentence_counts) if paragraph_sentence_counts else 0.0,
        "wlen": statistics.fmean([len(item) for item in word_tokens]),
        "front": 100 * sum(1 for sentence in sentences if FRONTLOAD.match(sentence.strip())) / sentence_count,
        "and2": 100 * multiple_and_count / sentence_count,
        "negative": 100 * negative / sentence_count,
        "andrate": 100 * long_words.count("and") / word_count,
        "_words": word_count,
        # The chapter as written, transcript included. Every graded measure
        # excludes chat lines, which is right for reading grade and wrong for
        # length: it made all five chat chapters look hundreds of words short
        # of the floor when only one of them is. Judge length on this.
        "_words_all": analysis.raw_word_count,
        "_sentences": sentence_count,
        "_paragraphs": len(paragraph_word_counts),
        "_transcript": analysis.transcript_share,
    }


