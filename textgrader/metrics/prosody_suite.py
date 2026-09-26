"""Poetry/prosody: line & stanza structure, meter/stress, rhyme and
phonological patterning (experimental).

This suite measures; it never judges. Free verse with no detectable meter is
not a failure and is not force-fit into one (see the acceptance criteria in
``docs/experimental-tasks/16-add-poetry-prosody-meter-rhyme-phonological-
pattern-and-line-structure-analysis.md``); a prose paragraph scored with the
same channels is not treated as a malformed poem either -- these signals are
explicitly allowed to run on prose too, as an experimental cadence/sound-
texture probe (advertising copy, slogans, speeches). Every finding stays
``Polarity.NEUTRAL`` and every metric is off by default.

Physical line breaks
---------------------
See :mod:`textgrader.prosody`'s module docstring for the full explanation.
In short: :meth:`DocumentAnalysis.paragraphs` joins every soft line wrap
inside a blank-line block into one line, which erases exactly the structure
a poem needs, so this suite never reads ``analysis.paragraphs`` or
``analysis.sentences`` for line/stanza structure. Instead
:func:`textgrader.prosody.line_structure` re-splits ``analysis.text`` -- the
SAME already-canonicalized text ``paragraphs()`` itself consumes -- on the
identical blank-line boundary, keeping each internal newline as its own
line. ``document.py``/``text.py`` are untouched.

Library survey (see the task document's table)
------------------------------------------------
* **CMU Pronouncing Dictionary**, via the ``pronouncing`` package (verified
  for real, see :mod:`textgrader.prosody`'s docstring) -- the primary
  pronunciation backend for every channel below.
* **g2p_en** -- an out-of-vocabulary grapheme-to-phoneme fallback, verified
  for real, but off by default (``g2p_fallback``) because importing it
  triggers a real, unconditional NLTK-corpus download the first time --
  see ``textgrader/prosody.py``'s docstring and this suite's
  ``_requires_g2p_fallback`` config note.
* **PanPhon** -- feature-based near-rhyme (``near_rhyme_feature``), on by
  default: lightweight, bundled data tables, no network call. Verified for
  real against real IPA strings (see ``textgrader/prosody.py``).
* **phonemizer** -- a second, independent pronunciation backend
  (``phonemizer_backend``), off by default. Its ``EspeakBackend`` needs the
  SYSTEM ``espeak``/``espeak-ng`` binary, which this sandbox does not have
  and cannot install (``which espeak espeak-ng`` finds nothing; no
  permission to run ``apt-get`` here); constructing it raises the exact,
  quoted ``RuntimeError: espeak not installed on your system`` this suite's
  finding reports back verbatim when the feature is turned on anyway.
* **Prosodic** / **Poesy** -- a second, independent meter+rhyme-scheme
  engine (``poesy_crosscheck``), off by default and bounded
  (``poesy_max_lines``/``poesy_max_seconds``) because its per-line scan is
  combinatorial in stress ambiguity. As of ``poesy>=0.4`` this package is,
  by its own module docstring, "a thin compatibility layer over prosodic
  v3", so this suite treats "Poesy" and "Prosodic" as the one working
  import path -- see ``textgrader/prosody.py`` for the real Sonnet-18 scan
  this environment reproduced WITHOUT espeak (CMUdict-only pronunciation is
  enough for real English vocabulary; espeak is prosodic's own
  out-of-vocabulary fallback, exactly like g2p_en's role for this suite's
  own primary engine).

Three separate rhyme channels, never merged
--------------------------------------------
Per the task's own "Implementation details" #4: exact rhyme (identical
stress-stripped rime phones), phonetic near-rhyme (ARPABET edit distance of
1 between rimes) and PanPhon feature-based similarity (a graded [0, 1]
articulatory-distance score) are three DIFFERENT questions and are reported
as three different metric families here (``rhyme`` for the first two,
``near_rhyme_feature`` for the third), never averaged into one "rhyme
score".

Enjambment is a proxy
----------------------
Per the task's #5, ``rhythm.prosody_enjambment_rate`` is explicitly labeled
a surface proxy (lines lacking terminal punctuation) in its own finding's
``warning``; no syntactic-closure analysis backs it.

Pronunciation coverage is always reported
-------------------------------------------
Whenever any phonology-dependent feature runs, ``rhythm.prosody_
pronunciation_coverage`` is emitted alongside it (see the task's #3 and
acceptance criteria), broken down by backend (``cmudict``/``g2p``) in its
``distribution``, so a proper-noun-heavy or invented-vocabulary text is
never silently compared against a corpus of ordinary prose on channels that
resolved a much smaller share of its words.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .. import prosody
from .. import stats as stats_module
from .. import text as textlib
from ..document import DocumentAnalysis
from .common import MODERATE, finding, option, rate, shape, top, unavailable

FAMILY = "prosody"
COST = MODERATE
REQUIRES: tuple[str, ...] = ("pronouncing", "g2p_en", "panphon", "phonemizer", "poesy")
MIN_SAMPLE = 6
UNIT_SENSITIVE = False

#: Sample-size floors, chosen for poetry's scale (a single sonnet has 14
#: lines; a haiku has 3) rather than the ~30-sentence floor prose metrics use.
LINE_MIN_SAMPLE = 6
PAIR_MIN_SAMPLE = 5
PHONE_MIN_SAMPLE = 30

DEFAULT_FEATURES: dict[str, bool] = {
    "line_stanza_structure": True,
    "meter_stress": True,
    "rhyme": True,
    "near_rhyme_feature": True,
    "phonological_patterning": True,
    # Off by default: see textgrader/prosody.py's module docstring for the
    # real, unconditional NLTK-corpus download this triggers on first import.
    "g2p_fallback": False,
    # Off by default: needs the SYSTEM espeak/espeak-ng binary; see the
    # module docstring's library survey.
    "phonemizer_backend": False,
    # Off by default: combinatorial per-line cost; bounded when enabled.
    "poesy_crosscheck": False,
}

#: A small, closed function-word list -- independent of every other suite's
#: copy (``sequences.py``'s ``_RARITY_STOPWORDS``, ``mechanical_quality_
#: suite``'s, ...) on purpose, matching this project's own stated convention
#: (see ``sequences.py``): this only needs to exclude the handful of words
#: frequent enough to dominate a sound-pattern window otherwise.
_STOPWORDS = frozenset("""
a an the and or but if of at by for with about against between into through
during before after above below to from up down in out on off over under is
am are was were be been being have has had having do does did doing would
could might must shall this that these those it he she they them his her
their as which who what i you we my your our
""".split())

#: Canonical English feet, as (stress pattern) tuples. Used for this suite's
#: OWN hand-rolled meter scan -- a fast, linear, single-reading approximation
#: (see ``_best_foot``'s docstring) kept deliberately separate from Poesy/
#: Prosodic's real combinatorial parse (``poesy_crosscheck``).
_FOOT_PATTERNS: dict[str, tuple[int, ...]] = {
    "iambic": (0, 1), "trochaic": (1, 0), "spondaic": (1, 1),
    "anapestic": (0, 0, 1), "dactylic": (1, 0, 0),
}

_TRAILERS = "\"'”’)]"
_TERMINAL_PUNCT = ".!?;:—…,"

#: A rough IPA-vowel character set for the phonemizer cross-check's syllable
#: proxy (counting nucleus characters in an espeak IPA transcription). Not
#: exhaustive of every IPA vowel in existence; sized for the vowels espeak's
#: en-us backend actually emits.
_IPA_VOWEL_CHARS = frozenset("aeiouɑæʌɔɛɪiʊuɜəɒyøœɐɨʉɯɤʏ")


def _feature(config: Mapping[str, Any] | None, name: str) -> bool:
    features = option(config, "features", {})
    if not isinstance(features, Mapping):
        return DEFAULT_FEATURES.get(name, False)
    value = features.get(name, DEFAULT_FEATURES.get(name, False))
    return DEFAULT_FEATURES.get(name, False) if value is None else bool(value)


def _entropy_from_counts(counts: Mapping[Any, int]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def _has_terminal_punctuation(line: str) -> bool:
    trimmed = line.rstrip().rstrip(_TRAILERS)
    return bool(trimmed) and trimmed[-1] in _TERMINAL_PUNCT


def _onset_consonant(phones: Sequence[str]) -> str | None:
    if not phones or prosody.is_vowel_phone(phones[0]):
        return None
    return prosody.base_phone(phones[0])


def _coda_consonant(phones: Sequence[str]) -> str | None:
    if not phones or prosody.is_vowel_phone(phones[-1]):
        return None
    return prosody.base_phone(phones[-1])


def _stressed_nucleus(phones: Sequence[str]) -> str | None:
    first_vowel = None
    for phone in phones:
        if prosody.is_vowel_phone(phone):
            if first_vowel is None:
                first_vowel = phone
            if prosody.stress_digit(phone) == 1:
                return prosody.base_phone(phone)
    return prosody.base_phone(first_vowel) if first_vowel else None


# --------------------------------------------------------------- shared pass

@dataclass(frozen=True)
class _LineWord:
    word: str
    pronunciation: "prosody.Pronunciation"


def _stride_sample_by_count(stanzas: tuple["prosody.Stanza", ...], target_lines: int
                            ) -> tuple[tuple["prosody.Stanza", ...], bool]:
    """An evenly spread subset of ``stanzas`` totalling at most ``target_lines`` lines.

    Rather than truncate to the opening (which would bias every distribution
    toward chapter one's style), lines are STRIDE-sampled at even spacing
    across the whole document -- the same "spread across the book" principle
    ``coherence_suite``'s RST sampling and ``syntax_complexity_suite``'s
    constituency sampling already use -- and whole stanzas are kept together
    wherever possible so lines/stanza and rhyme-scheme grouping stay
    meaningful.
    """

    total = sum(len(stanza.lines) for stanza in stanzas)
    if total <= target_lines or target_lines <= 0:
        return stanzas, False
    positions = [(s_index, l_index) for s_index, stanza in enumerate(stanzas)
                for l_index in range(len(stanza.lines))]
    step = len(positions) / target_lines
    kept: set[tuple[int, int]] = set()
    next_take = 0.0
    for index, position in enumerate(positions):
        if index >= next_take:
            kept.add(position)
            next_take += step
    sampled: list["prosody.Stanza"] = []
    for s_index, stanza in enumerate(stanzas):
        kept_lines = tuple(line for l_index, line in enumerate(stanza.lines)
                           if (s_index, l_index) in kept)
        if kept_lines:
            sampled.append(prosody.Stanza(kept_lines))
    return tuple(sampled), True


def _sample_stanzas(stanzas: tuple["prosody.Stanza", ...], max_lines: int, max_words: int
                    ) -> tuple[tuple["prosody.Stanza", ...], bool]:
    """Bound both line COUNT and total WORD volume before any lookup runs.

    A poem is never affected (the defaults, 2,000 lines / 30,000 words, sit
    far above any real poem's size); a whole novel is. Line count alone is
    not enough: after :func:`textgrader.prosody._looks_hard_wrapped` rejoins
    a hard-wrapped block into one line (see that function's docstring), a
    "line" is a whole paragraph, so 2,000 of them can still be tens of
    thousands of words -- exactly what a first version of this bound missed,
    caught by re-measuring the same benchmark novel after the hard-wrap fix
    landed (it rose back to about 13.7s).

    The word budget is converted into an EQUIVALENT line-count target (this
    document's mean words/line, divided into ``max_words``) and applied with
    the same by-count stride sampler, rather than tried as a cumulative
    word-weighted stride directly: a first version did that and failed
    silently, because a stride step smaller than one line's own word count
    (56,599 words already spread over exactly 2,000 lines, budgeted at
    30,000 -- a step of 1.9 "words" per slot) can never skip a whole line,
    so it kept every single one. Converting to a line-count target first
    avoids that failure mode entirely. Applying the line cap before the word
    cap keeps a pathological many-tiny-lines input and a pathological
    few-huge-lines input both bounded by the SAME two numbers. Every
    per-word pronunciation lookup, PanPhon feature-distance call and
    phonological-pattern pass scales with total line/word count; see
    :func:`_document_prosody`'s docstring for the measured benchmark.
    """

    by_lines, sampled_lines = _stride_sample_by_count(stanzas, max_lines)
    total_words = sum(len(textlib.words(line)) for stanza in by_lines for line in stanza.lines)
    total_lines = sum(len(stanza.lines) for stanza in by_lines)
    if total_words <= max_words or total_lines == 0:
        return by_lines, sampled_lines
    mean_words_per_line = total_words / total_lines
    target_lines = max(1, int(max_words / mean_words_per_line))
    by_words, sampled_words = _stride_sample_by_count(by_lines, target_lines)
    return by_words, sampled_lines or sampled_words


def _document_prosody(analysis: DocumentAnalysis, use_g2p: bool, max_lines: int, max_words: int):
    """Per-line pronunciations plus aggregate coverage, cached on ``analysis``.

    Every feature below reads this SAME pass (memoized via
    ``DocumentAnalysis.memo``, per this project's rule 5), so turning on
    ``meter_stress``, ``rhyme`` and ``phonological_patterning`` together
    costs one set of pronunciation lookups, not three -- and
    :func:`prosody.pronounce` itself caches every individual word lookup
    process-wide on top of that.

    **Measured benchmark** (``gutenberg-514-little-women.txt``, 186,000
    words): unbounded, this suite's default config took 49.6s wall-clock,
    driven mostly by PanPhon's feature-based near-rhyme channel
    re-computing the same common word-ending pairs tens of thousands of
    times. Adding a 2,000-LINE stride-sampled cap plus
    ``near_rhyme_feature_max_pairs`` and a process-wide PanPhon similarity
    cache brought that to 3.2s -- but only before
    :func:`textgrader.prosody._looks_hard_wrapped` existed; once hard-wrapped
    Gutenberg paragraphs started being correctly rejoined into one "line"
    each (a real correctness fix -- see that function's docstring), the same
    2,000-line cap silently became a 2,000-PARAGRAPH cap, tens of thousands
    of words, and the measured time rose back to 13.7s. The additional
    30,000-WORD cap this function's ``max_words`` applies fixes that: the
    same novel, same config, now measures about 3.0s.
    """

    key = f"prosody_suite:document:{use_g2p}:{max_lines}:{max_words}"
    return analysis.memo(key, lambda: _build_document_prosody(analysis, use_g2p, max_lines,
                                                              max_words))


def _build_document_prosody(analysis: DocumentAnalysis, use_g2p: bool, max_lines: int,
                            max_words: int):
    full_structure = prosody.line_structure(analysis)
    sampled_stanzas, was_sampled = _sample_stanzas(full_structure.stanzas, max_lines, max_words)
    if was_sampled:
        sampled_nonblank = tuple(line for stanza in sampled_stanzas for line in stanza.lines)
        structure = prosody.LineStructure(full_structure.raw_line_count, sampled_nonblank,
                                          sampled_stanzas)
    else:
        structure = full_structure
    lines_words: list[list[_LineWord]] = []
    stanza_line_indices: list[list[int]] = []
    resolved = 0
    total = 0
    source_counts: Counter = Counter()
    index = 0
    for stanza in structure.stanzas:
        indices = []
        for line in stanza.lines:
            entries = []
            for word in textlib.words(line):
                total += 1
                pron = prosody.pronounce(word, use_g2p=use_g2p)
                if pron.phones:
                    resolved += 1
                    source_counts[pron.source] += 1
                entries.append(_LineWord(word, pron))
            lines_words.append(entries)
            indices.append(index)
            index += 1
        stanza_line_indices.append(indices)
    coverage = {
        "word_count": total, "resolved_words": resolved,
        "unresolved_words": total - resolved,
        "coverage_ratio": (100.0 * resolved / total) if total else None,
        "source_counts": dict(source_counts),
    }
    return structure, full_structure, was_sampled, lines_words, stanza_line_indices, coverage


def _end_word(entries: list[_LineWord]) -> _LineWord | None:
    return entries[-1] if entries else None


# ------------------------------------------------------------ line / stanza

_LINE_STANZA_EMPTY_IDS = [
    ("rhythm.prosody_words_per_line", "Words per line", "words"),
    ("rhythm.prosody_line_length_cv", "Line-length coefficient of variation", "percent"),
    ("rhythm.prosody_line_length_entropy", "Line-length entropy", "bits"),
    ("rhythm.prosody_stanza_count", "Stanza count", "stanzas"),
    ("rhythm.prosody_lines_per_stanza", "Lines per stanza", "lines"),
    ("rhythm.prosody_stanza_symmetry", "Share of stanzas at the modal stanza length", "percent"),
    ("rhythm.prosody_repeated_line_length_rate",
     "Share of lines at the single most common line length", "percent"),
    ("rhythm.prosody_enjambment_rate",
     "Lines lacking terminal punctuation (enjambment proxy)", "percent"),
]


def _line_stanza_findings(structure: "prosody.LineStructure",
                          full_structure: "prosody.LineStructure",
                          sampled: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    nonblank = structure.nonblank_lines
    nonblank_count = len(nonblank)
    sample_note = (f"based on an evenly spread sample of {nonblank_count} of "
                  f"{len(full_structure.nonblank_lines)} nonblank lines (see "
                  "max_lines_analyzed)") if sampled else None
    # The two raw counts are always the TRUE whole-document facts, never the
    # sampled subset -- sampling exists to bound cost on everything ELSE.
    out.append(finding("rhythm.prosody_line_count", "Physical line count",
                       float(full_structure.raw_line_count), "lines", family=FAMILY,
                       sample_size=full_structure.raw_line_count, unit_sensitive=True))
    out.append(finding("rhythm.prosody_nonblank_line_count", "Nonblank poetic-line count",
                       float(len(full_structure.nonblank_lines)), "lines", family=FAMILY,
                       sample_size=len(full_structure.nonblank_lines), unit_sensitive=True))
    sampled_section_start = len(out)
    if not nonblank_count:
        warning = "no physical lines with words were found in the analyzed text"
        for metric_id, name, unit in _LINE_STANZA_EMPTY_IDS:
            out.append(finding(metric_id, name, None, unit, family=FAMILY, sample_size=0,
                               min_sample=LINE_MIN_SAMPLE, warning=warning))
        return out

    word_lengths = [len(textlib.words(line)) for line in nonblank]
    out.extend(shape("rhythm.prosody_words_per_line", "Words per line", word_lengths, "words",
                     family=FAMILY, min_sample=LINE_MIN_SAMPLE, headline="mean"))
    summary = stats_module.summarize(word_lengths)
    out.append(finding("rhythm.prosody_line_length_cv", "Line-length coefficient of variation",
                       summary.get("cv"), "percent", family=FAMILY, sample_size=nonblank_count,
                       min_sample=LINE_MIN_SAMPLE, distribution=summary))
    out.append(finding("rhythm.prosody_line_length_entropy", "Line-length entropy",
                       stats_module.shannon_entropy(word_lengths), "bits", family=FAMILY,
                       sample_size=nonblank_count, min_sample=LINE_MIN_SAMPLE,
                       sample_size_sensitive=True))

    stanzas = structure.stanzas
    stanza_lens = [len(item.lines) for item in stanzas]
    out.append(finding("rhythm.prosody_stanza_count", "Stanza count", float(len(stanzas)),
                       "stanzas", family=FAMILY, sample_size=len(stanzas), unit_sensitive=True))
    if stanza_lens:
        out.extend(shape("rhythm.prosody_lines_per_stanza", "Lines per stanza", stanza_lens,
                         "lines", family=FAMILY, min_sample=2, headline="mean"))
        modal_len, modal_count = Counter(stanza_lens).most_common(1)[0]
        out.append(finding("rhythm.prosody_stanza_symmetry",
                           "Share of stanzas at the modal stanza length",
                           rate(modal_count, len(stanza_lens)), "percent", family=FAMILY,
                           sample_size=len(stanza_lens), min_sample=2,
                           distribution={"modal_lines_per_stanza": modal_len}))
    else:
        out.append(unavailable("rhythm.prosody_lines_per_stanza", "Lines per stanza",
                               "no stanzas", family=FAMILY))
        out.append(unavailable("rhythm.prosody_stanza_symmetry",
                               "Share of stanzas at the modal stanza length", "no stanzas",
                               family=FAMILY))

    length_counts = Counter(word_lengths)
    _, top_count = length_counts.most_common(1)[0]
    out.append(finding("rhythm.prosody_repeated_line_length_rate",
                       "Share of lines at the single most common line length",
                       rate(top_count, nonblank_count), "percent", family=FAMILY,
                       sample_size=nonblank_count, min_sample=LINE_MIN_SAMPLE,
                       evidence=[{"length_words": length, "count": count}
                                 for length, count in length_counts.most_common(10)]))

    eligible = nonblank[:-1] if nonblank_count > 1 else ()
    enjambed = sum(1 for line in eligible if not _has_terminal_punctuation(line))
    out.append(finding("rhythm.prosody_enjambment_rate",
                       "Lines lacking terminal punctuation (enjambment proxy)",
                       rate(enjambed, len(eligible)) if eligible else None, "percent",
                       family=FAMILY, sample_size=len(eligible), min_sample=LINE_MIN_SAMPLE,
                       distribution={"enjambed_count": enjambed},
                       warning=("a surface proxy: no syntactic-closure analysis is performed, "
                                "per the task's own labeling requirement") if eligible else
                               "fewer than two lines; enjambment needs a following line"))
    if sample_note:
        for item in out[sampled_section_start:]:
            if not item.get("warning"):
                item["warning"] = sample_note
    return out


def _pronunciation_coverage_finding(coverage: Mapping[str, Any], sampled: bool,
                                    lines_analyzed: int, total_lines: int,
                                    use_g2p: bool) -> dict[str, Any]:
    total = coverage["word_count"]
    warning = None
    if not total:
        warning = "no words to look up"
    elif prosody.pronunciation_backend_reason():
        warning = prosody.pronunciation_backend_reason()
    elif sampled:
        warning = (f"every OTHER finding in this suite that depends on pronunciation is also "
                  f"computed from this same evenly spread sample of {lines_analyzed} of "
                  f"{total_lines} nonblank lines (see max_lines_analyzed)")
    # Per the task's "Corpus/profile requirements": record the pronunciation
    # backend/version and language every phonology-dependent finding relied
    # on, so a corpus profile built with g2p_fallback on is never silently
    # compared against one built without it.
    backend = "cmudict+g2p_en" if use_g2p else "cmudict"
    return finding("rhythm.prosody_pronunciation_coverage",
                   "Share of line words with a resolved pronunciation",
                   coverage["coverage_ratio"], "percent", family=FAMILY, sample_size=total,
                   min_sample=LINE_MIN_SAMPLE,
                   distribution={"resolved_words": coverage["resolved_words"],
                                 "unresolved_words": coverage["unresolved_words"],
                                 "source_counts": coverage["source_counts"],
                                 "lines_analyzed": lines_analyzed, "lines_total": total_lines,
                                 "sampled": sampled, "backend": backend, "language": "en"},
                   warning=warning)


# ---------------------------------------------------------------- meter/stress

_METER_EMPTY_IDS = [
    ("rhythm.prosody_dominant_meter_confidence",
     "Share of scored lines matching the dominant foot pattern", "percent"),
    ("rhythm.prosody_meter_conformity_rate",
     "Per-line stress-match rate against that line's best-fit foot", "percent"),
    ("rhythm.prosody_meter_deviation",
     "Line-to-line variation (SD) in meter conformity", "percent"),
    ("rhythm.prosody_feet_per_line",
     "Feet per line (line syllables divided by best-fit foot length)", "feet"),
]


def _line_stress_digits(entries: list[_LineWord]) -> tuple[int, ...]:
    digits: list[int] = []
    for entry in entries:
        digits.extend(prosody.stress_pattern(entry.pronunciation.phones))
    return tuple(digits)


def _scansion_stress(entries: list[_LineWord]) -> tuple[int, ...]:
    """Binary stress for METER SCANNING only (see ``_best_foot``'s docstring).

    A monosyllabic function word (this module's small stopword list) is
    demoted to unstressed regardless of its CMUdict citation-form stress
    digit -- matching how English scansion conventionally treats them: a
    dictionary marks "not", "is", "a" with a stand-alone stress that does
    not survive inside a sentence, and without this adjustment nearly every
    monosyllabic function word reads as stressed, which flattens the
    difference between a metrical line and an unmetrical one almost to
    nothing (measured directly while developing this module: without the
    demotion, a free-verse fixture's lines matched as many different best-fit
    feet, one per line, as a real iambic-pentameter fixture's did). A
    polysyllabic word, or any word outside the stopword list, keeps its
    dictionary stress unchanged. This heuristic is used ONLY for foot-fitting
    and periodicity; ``rhythm.prosody_stress_entropy`` and Task 20's
    :func:`textgrader.prosody.stress_sequence` stay pure dictionary stress,
    since demoting function words is a scansion CONVENTION about how a line
    is meant to be read aloud, not a fact about any one word's pronunciation.
    """

    out: list[int] = []
    for entry in entries:
        digits = prosody.stress_pattern(entry.pronunciation.phones)
        if len(digits) == 1 and entry.word.lower() in _STOPWORDS:
            out.append(0)
        else:
            out.extend(1 if digit else 0 for digit in digits)
    return tuple(out)


def _best_foot(stress_binary: Sequence[int]) -> tuple[str, int, float] | None:
    """Best-fitting foot for one line's binary stress sequence.

    A GREEDY, single-reading approximation: the stress sequence -- one
    reading per word, resolved by dictionary/G2P lookup, no ambiguity search
    -- is tiled against each candidate foot starting at position 0, and the
    pattern with the highest match rate wins. This is explicitly NOT a real
    metrical parse (a real scansion allows substitution and resolves
    ambiguous words by context); Poesy/Prosodic's real, combinatorial parse
    (``poesy_crosscheck``, off by default) is kept as a second, independent
    implementation on purpose, never reconciled with this one -- see this
    module's docstring.
    """

    if len(stress_binary) < 2:
        return None
    best: tuple[str, int, float] | None = None
    for name, pattern in _FOOT_PATTERNS.items():
        matches = sum(1 for i, value in enumerate(stress_binary)
                      if value == pattern[i % len(pattern)])
        conformity = matches / len(stress_binary)
        if best is None or conformity > best[2]:
            best = (name, len(pattern), conformity)
    return best


def _meter_findings(lines_words: list[list[_LineWord]]) -> tuple[list[dict[str, Any]], str | None]:
    out: list[dict[str, Any]] = []
    syllables_per_line: list[int] = []
    all_digits: list[int] = []
    all_binary: list[float] = []
    per_line: list[tuple[str, int, float, int]] = []
    for entries in lines_words:
        digits = _line_stress_digits(entries)
        syllables_per_line.append(len(digits))
        all_digits.extend(digits)
        scansion = _scansion_stress(entries)
        all_binary.extend(float(value) for value in scansion)
        best = _best_foot(scansion)
        if best is not None:
            per_line.append((best[0], best[1], best[2], len(scansion)))

    if any(syllables_per_line):
        out.extend(shape("rhythm.prosody_syllables_per_line", "Syllables per line",
                         syllables_per_line, "syllables", family=FAMILY, min_sample=LINE_MIN_SAMPLE,
                         headline="mean"))
    else:
        out.append(unavailable("rhythm.prosody_syllables_per_line", "Syllables per line",
                               "no resolved syllables", family=FAMILY))

    if all_digits:
        out.append(finding("rhythm.prosody_stress_entropy", "Stress-level entropy (0/1/2)",
                           stats_module.shannon_entropy(all_digits), "bits", family=FAMILY,
                           sample_size=len(all_digits), min_sample=PHONE_MIN_SAMPLE,
                           sample_size_sensitive=True))
    else:
        out.append(unavailable("rhythm.prosody_stress_entropy", "Stress-level entropy (0/1/2)",
                               "no resolved syllables", family=FAMILY))

    if len(all_binary) > 3:
        out.append(finding("rhythm.prosody_stress_periodicity",
                           "Lag-2 autocorrelation of the binary stress sequence (2 syllables: "
                           "the modal English foot length, fixed regardless of the document)",
                           stats_module.autocorrelation(all_binary, 2), "ratio", family=FAMILY,
                           sample_size=len(all_binary), min_sample=PHONE_MIN_SAMPLE))
    else:
        out.append(unavailable("rhythm.prosody_stress_periodicity",
                               "Lag-2 autocorrelation of the binary stress sequence",
                               "too few resolved syllables", family=FAMILY))

    dominant_foot = None
    if per_line:
        foot_counts = Counter(item[0] for item in per_line)
        dominant_foot, dominant_count = foot_counts.most_common(1)[0]
        out.append(finding("rhythm.prosody_dominant_meter_confidence",
                           "Share of scored lines matching the dominant foot pattern",
                           rate(dominant_count, len(per_line)), "percent", family=FAMILY,
                           sample_size=len(per_line), min_sample=LINE_MIN_SAMPLE,
                           distribution={"dominant_meter": dominant_foot,
                                         "foot_counts": dict(foot_counts)}))
        conformities = [item[2] * 100 for item in per_line]
        out.extend(shape("rhythm.prosody_meter_conformity_rate",
                         "Per-line stress-match rate against that line's best-fit foot",
                         conformities, "percent", family=FAMILY, min_sample=LINE_MIN_SAMPLE))
        out.append(finding("rhythm.prosody_meter_deviation",
                           "Line-to-line variation (SD) in meter conformity",
                           stats_module.summarize(conformities).get("std"), "percent",
                           family=FAMILY, sample_size=len(per_line), min_sample=LINE_MIN_SAMPLE))
        feet_per_line = [item[3] / item[1] for item in per_line if item[1]]
        out.extend(shape("rhythm.prosody_feet_per_line",
                         "Feet per line (line syllables divided by best-fit foot length)",
                         feet_per_line, "feet", family=FAMILY, min_sample=LINE_MIN_SAMPLE,
                         headline="mean"))
    else:
        for metric_id, name, unit in _METER_EMPTY_IDS:
            out.append(unavailable(metric_id, name, "no line had two or more resolved "
                                   "syllables to score a foot against", family=FAMILY, unit=unit))
    return out, dominant_foot


# --------------------------------------------------------------------- rhyme

_RHYME_EMPTY_IDS = [
    ("rhythm.prosody_end_rhyme_density", "Share of eligible lines that rhyme with a nearby "
     "earlier line", "percent"),
    ("rhythm.prosody_perfect_rhyme_rate", "Among matched lines, the share whose match was an "
     "exact rhyme", "percent"),
    ("rhythm.prosody_near_rhyme_rate", "Among matched lines, the share whose best match was a "
     "near/slant rhyme only", "percent"),
    ("rhythm.prosody_rhyme_class_entropy", "Entropy of the end-rhyme class distribution", "bits"),
    ("rhythm.prosody_internal_rhyme_rate", "Share of lines containing an internal rhyme", "percent"),
    ("rhythm.prosody_rhyme_recurrence_distance", "Lines between successive occurrences of the "
     "same end-rhyme class", "lines"),
    ("rhythm.prosody_rhyme_scheme_regularity", "Share of same-length line groups matching the "
     "modal rhyme scheme", "percent"),
]


def _scheme_string(keys: Sequence[tuple[str, ...] | None]) -> str:
    labels: dict[tuple[str, ...], str] = {}
    out: list[str] = []
    for key in keys:
        if key is None:
            out.append("-")
            continue
        label = labels.get(key)
        if label is None:
            label = chr(ord("a") + len(labels)) if len(labels) < 26 else "?"
            labels[key] = label
        out.append(label)
    return "".join(out)


def _stanza_groups(stanza_indices: list[list[int]], total_lines: int,
                   default_size: int) -> list[list[int]]:
    real_stanzas = [indices for indices in stanza_indices if len(indices) >= 2]
    if len(stanza_indices) > 1 and real_stanzas:
        return real_stanzas
    size = max(2, int(default_size))
    return [list(range(start, min(start + size, total_lines)))
            for start in range(0, total_lines, size)
            if min(start + size, total_lines) - start >= 2]


def _rhyme_findings(config: Mapping[str, Any] | None, lines_words: list[list[_LineWord]],
                    stanza_indices: list[list[int]]) -> list[dict[str, Any]]:
    lookback = max(1, int(option(config, "rhyme_lookback_lines", 4)))
    min_internal_len = int(option(config, "internal_rhyme_min_word_length", 3))
    group_size = int(option(config, "rhyme_scheme_group_size", 4))

    end_keys: list[tuple[str, ...] | None] = []
    for entries in lines_words:
        end = _end_word(entries)
        if end is not None and end.pronunciation.phones:
            end_keys.append(prosody.rime_key(end.pronunciation.phones))
        else:
            end_keys.append(None)

    resolved_indices = [i for i, key in enumerate(end_keys) if key is not None]
    out: list[dict[str, Any]] = []
    if len(resolved_indices) < 2:
        reason = "fewer than two line endings had a resolved pronunciation"
        for metric_id, name, unit in _RHYME_EMPTY_IDS:
            out.append(unavailable(metric_id, name, reason, family=FAMILY, unit=unit))
        return out

    eligible = matched = perfect_only = near_only = 0
    for i in resolved_indices:
        candidates = [j for j in range(max(0, i - lookback), i) if end_keys[j] is not None]
        if not candidates:
            continue
        eligible += 1
        best = None
        for j in candidates:
            if end_keys[i] == end_keys[j]:
                best = "exact"
                break
            if prosody.phone_edit_distance(end_keys[i], end_keys[j]) <= 1:
                best = best or "near"
        if best:
            matched += 1
            if best == "exact":
                perfect_only += 1
            else:
                near_only += 1

    if eligible:
        out.append(finding("rhythm.prosody_end_rhyme_density",
                           "Share of eligible lines that rhyme with a nearby earlier line",
                           rate(matched, eligible), "percent", family=FAMILY, sample_size=eligible,
                           min_sample=LINE_MIN_SAMPLE, distribution={"matched_lines": matched}))
    else:
        out.append(unavailable("rhythm.prosody_end_rhyme_density",
                               "Share of eligible lines that rhyme with a nearby earlier line",
                               "no line had an earlier resolved line within the lookback window",
                               family=FAMILY))
    if matched:
        out.append(finding("rhythm.prosody_perfect_rhyme_rate",
                           "Among matched lines, the share whose match was an exact rhyme",
                           rate(perfect_only, matched), "percent", family=FAMILY,
                           sample_size=matched, min_sample=PAIR_MIN_SAMPLE))
        out.append(finding("rhythm.prosody_near_rhyme_rate",
                           "Among matched lines, the share whose best match was a near/slant "
                           "rhyme only", rate(near_only, matched), "percent", family=FAMILY,
                           sample_size=matched, min_sample=PAIR_MIN_SAMPLE))
    else:
        out.append(unavailable("rhythm.prosody_perfect_rhyme_rate",
                               "Among matched lines, the share whose match was an exact rhyme",
                               "no matched lines", family=FAMILY))
        out.append(unavailable("rhythm.prosody_near_rhyme_rate",
                               "Among matched lines, the share whose best match was a near/"
                               "slant rhyme only", "no matched lines", family=FAMILY))

    classes = Counter(end_keys[i] for i in resolved_indices)
    out.append(finding("rhythm.prosody_rhyme_class_entropy",
                       "Entropy of the end-rhyme class distribution",
                       _entropy_from_counts(classes), "bits", family=FAMILY,
                       sample_size=len(resolved_indices), min_sample=LINE_MIN_SAMPLE,
                       sample_size_sensitive=True,
                       distribution={"distinct_classes": len(classes)}))

    groups: dict[tuple[str, ...], list[int]] = {}
    for i in resolved_indices:
        groups.setdefault(end_keys[i], []).append(i)
    gaps = [float(b - a) for members in groups.values() if len(members) >= 2
           for a, b in zip(members, members[1:])]
    if gaps:
        out.extend(shape("rhythm.prosody_rhyme_recurrence_distance",
                         "Lines between successive occurrences of the same end-rhyme class",
                         gaps, "lines", family=FAMILY, min_sample=PAIR_MIN_SAMPLE))
    else:
        out.append(unavailable("rhythm.prosody_rhyme_recurrence_distance",
                               "Lines between successive occurrences of the same end-rhyme class",
                               "no end-rhyme class recurred", family=FAMILY))

    internal_hits = internal_eligible = 0
    for entries in lines_words:
        candidates = [prosody.rime_key(entry.pronunciation.phones) for entry in entries
                     if len(entry.word) >= min_internal_len and entry.pronunciation.phones]
        if len(candidates) < 2:
            continue
        internal_eligible += 1
        if len(set(candidates)) < len(candidates):
            internal_hits += 1
    if internal_eligible:
        out.append(finding("rhythm.prosody_internal_rhyme_rate",
                           "Share of lines containing an internal rhyme",
                           rate(internal_hits, internal_eligible), "percent", family=FAMILY,
                           sample_size=internal_eligible, min_sample=LINE_MIN_SAMPLE))
    else:
        out.append(unavailable("rhythm.prosody_internal_rhyme_rate",
                               "Share of lines containing an internal rhyme",
                               "fewer than two resolvable words on any single line",
                               family=FAMILY))

    groups_by_length = _stanza_groups(stanza_indices, len(end_keys), group_size)
    schemes = [_scheme_string([end_keys[i] for i in indices]) for indices in groups_by_length]
    if schemes:
        length_counts = Counter(len(scheme) for scheme in schemes)
        modal_length, _ = length_counts.most_common(1)[0]
        same_length = [scheme for scheme in schemes if len(scheme) == modal_length]
        scheme_counts = Counter(same_length)
        modal_scheme, modal_scheme_count = scheme_counts.most_common(1)[0]
        out.append(finding("rhythm.prosody_rhyme_scheme_regularity",
                           "Share of same-length line groups matching the modal rhyme scheme",
                           rate(modal_scheme_count, len(same_length)), "percent", family=FAMILY,
                           sample_size=len(same_length), min_sample=2,
                           distribution={"modal_scheme": modal_scheme,
                                         "group_count": len(schemes)},
                           evidence=[{"scheme": scheme} for scheme in schemes[:10]]))
    else:
        out.append(unavailable("rhythm.prosody_rhyme_scheme_regularity",
                               "Share of same-length line groups matching the modal rhyme scheme",
                               "fewer than two lines in any group", family=FAMILY))
    return out


def _feature_rhyme_findings(config: Mapping[str, Any] | None,
                            lines_words: list[list[_LineWord]]) -> list[dict[str, Any]]:
    lookback = max(1, int(option(config, "rhyme_lookback_lines", 4)))
    # PanPhon's weighted feature-edit distance is the single most expensive
    # per-pair operation in this suite (see this module's docstring's
    # benchmark table); capped independently of max_lines_analyzed because
    # even a 2,000-line sample times lookback pairs is more comparisons than
    # this one channel needs for a stable median.
    max_pairs = max(50, int(option(config, "near_rhyme_feature_max_pairs", 3000)))
    ends = [_end_word(entries) for entries in lines_words]
    similarities: list[float] = []
    reason: str | None = None
    pairs_seen = 0
    for i, end in enumerate(ends):
        if len(similarities) >= max_pairs:
            break
        if end is None or not end.pronunciation.phones:
            continue
        rime_i = prosody.rime_of(end.pronunciation.phones)
        for j in range(max(0, i - lookback), i):
            other = ends[j]
            if other is None or not other.pronunciation.phones:
                continue
            rime_j = prosody.rime_of(other.pronunciation.phones)
            similarity, why = prosody.feature_rhyme_similarity(rime_i, rime_j)
            pairs_seen += 1
            if similarity is not None:
                similarities.append(similarity * 100)
            elif reason is None:
                reason = why
            if len(similarities) >= max_pairs:
                break
    if similarities:
        return shape("rhythm.prosody_feature_rhyme_similarity",
                     "PanPhon feature-based similarity of nearby line-ending rimes",
                     similarities, "percent", family=FAMILY, min_sample=PAIR_MIN_SAMPLE,
                     # Most nearby line endings share no rime at all, so the
                     # median was 0 on all 50 reference books (and on a sonnet
                     # quatrain); the mean keeps the resolution.
                     headline="mean")
    return [unavailable("rhythm.prosody_feature_rhyme_similarity",
                        "PanPhon feature-based similarity of nearby line-ending rimes",
                        reason or "no comparable line-ending pairs", family=FAMILY)]


# ---------------------------------------------------------- phonological patterning

_PHONOLOGICAL_PAIR_EMPTY_IDS = [
    ("rhythm.prosody_alliteration_density",
     "Share of nearby content-word pairs sharing an onset consonant", "percent"),
    ("rhythm.prosody_assonance_density",
     "Share of nearby content-word pairs sharing a stressed vowel", "percent"),
    ("rhythm.prosody_consonance_density",
     "Share of nearby content-word pairs sharing a coda consonant", "percent"),
    ("rhythm.prosody_line_sound_recurrence",
     "Sound-pattern hits (alliteration + assonance + consonance) per line", "hits"),
]


def _phonological_findings(config: Mapping[str, Any] | None,
                           lines_words: list[list[_LineWord]]) -> list[dict[str, Any]]:
    window = max(2, int(option(config, "sound_pattern_window_words", 3)))
    min_len = int(option(config, "content_min_word_length", 3))

    phone_counter: Counter = Counter()
    onset_counter: Counter = Counter()
    coda_counter: Counter = Counter()
    phoneme_stream: list[str] = []
    vowel_count = consonant_count = 0
    for entries in lines_words:
        for entry in entries:
            phones = entry.pronunciation.phones
            if not phones:
                continue
            for phone in phones:
                base = prosody.base_phone(phone)
                phone_counter[base] += 1
                phoneme_stream.append(base)
                if prosody.is_vowel_phone(phone):
                    vowel_count += 1
                else:
                    consonant_count += 1
            onset = _onset_consonant(phones)
            if onset:
                onset_counter[onset] += 1
            coda = _coda_consonant(phones)
            if coda:
                coda_counter[coda] += 1

    out: list[dict[str, Any]] = []
    total_phones = vowel_count + consonant_count
    if total_phones:
        out.append(finding("rhythm.prosody_phoneme_entropy", "Phoneme entropy",
                           _entropy_from_counts(phone_counter), "bits", family=FAMILY,
                           sample_size=total_phones, min_sample=PHONE_MIN_SAMPLE,
                           sample_size_sensitive=True,
                           distribution={"distinct_phonemes": len(phone_counter)},
                           evidence=top(phone_counter, 15, key="phoneme")))
        out.append(finding("rhythm.prosody_vowel_consonant_balance",
                           "Share of resolved phonemes that are vowels",
                           rate(vowel_count, total_phones), "percent", family=FAMILY,
                           sample_size=total_phones, min_sample=PHONE_MIN_SAMPLE,
                           distribution={"vowel_count": vowel_count,
                                         "consonant_count": consonant_count}))
    else:
        out.append(unavailable("rhythm.prosody_phoneme_entropy", "Phoneme entropy",
                               "no resolved phonemes", family=FAMILY))
        out.append(unavailable("rhythm.prosody_vowel_consonant_balance",
                               "Share of resolved phonemes that are vowels",
                               "no resolved phonemes", family=FAMILY))

    if len(phoneme_stream) > 3:
        bigrams = list(zip(phoneme_stream, phoneme_stream[1:]))
        bigram_counts = Counter(bigrams)
        repeats = sum(count - 1 for count in bigram_counts.values() if count > 1)
        out.append(finding("rhythm.prosody_phoneme_bigram_repetition_rate",
                           "Share of phoneme bigrams that repeat one seen earlier",
                           rate(repeats, len(bigrams)), "percent", family=FAMILY,
                           sample_size=len(bigrams), min_sample=PHONE_MIN_SAMPLE,
                           distribution={"distinct_bigrams": len(bigram_counts)},
                           evidence=[{"bigram": " ".join(pair), "count": count}
                                     for pair, count in bigram_counts.most_common(10)
                                     if count > 1]))
    else:
        out.append(unavailable("rhythm.prosody_phoneme_bigram_repetition_rate",
                               "Share of phoneme bigrams that repeat one seen earlier",
                               "fewer than four resolved phonemes", family=FAMILY))

    alliteration_hits = alliteration_pairs = 0
    assonance_hits = assonance_pairs = 0
    consonance_hits = consonance_pairs = 0
    per_line_hits: list[float] = []
    for entries in lines_words:
        content = [entry for entry in entries if len(entry.word) >= min_len
                  and entry.word.lower() not in _STOPWORDS and entry.pronunciation.phones]
        line_hits = 0
        for i in range(len(content)):
            for j in range(i + 1, min(i + window, len(content))):
                phones_i, phones_j = content[i].pronunciation.phones, content[j].pronunciation.phones
                onset_i, onset_j = _onset_consonant(phones_i), _onset_consonant(phones_j)
                alliteration_pairs += 1
                if onset_i and onset_i == onset_j:
                    alliteration_hits += 1
                    line_hits += 1
                nucleus_i, nucleus_j = _stressed_nucleus(phones_i), _stressed_nucleus(phones_j)
                assonance_pairs += 1
                if nucleus_i and nucleus_i == nucleus_j:
                    assonance_hits += 1
                    line_hits += 1
                coda_i, coda_j = _coda_consonant(phones_i), _coda_consonant(phones_j)
                consonance_pairs += 1
                if coda_i and coda_i == coda_j:
                    consonance_hits += 1
                    line_hits += 1
        per_line_hits.append(float(line_hits))

    if alliteration_pairs:
        out.append(finding("rhythm.prosody_alliteration_density",
                           "Share of nearby content-word pairs sharing an onset consonant",
                           rate(alliteration_hits, alliteration_pairs), "percent", family=FAMILY,
                           sample_size=alliteration_pairs, min_sample=PAIR_MIN_SAMPLE,
                           evidence=top(onset_counter, 10, key="onset")))
        out.append(finding("rhythm.prosody_assonance_density",
                           "Share of nearby content-word pairs sharing a stressed vowel",
                           rate(assonance_hits, assonance_pairs), "percent", family=FAMILY,
                           sample_size=assonance_pairs, min_sample=PAIR_MIN_SAMPLE))
        out.append(finding("rhythm.prosody_consonance_density",
                           "Share of nearby content-word pairs sharing a coda consonant",
                           rate(consonance_hits, consonance_pairs), "percent", family=FAMILY,
                           sample_size=consonance_pairs, min_sample=PAIR_MIN_SAMPLE,
                           evidence=top(coda_counter, 10, key="coda")))
        out.extend(shape("rhythm.prosody_line_sound_recurrence",
                         "Sound-pattern hits (alliteration + assonance + consonance) per line",
                         per_line_hits, "hits", family=FAMILY, min_sample=LINE_MIN_SAMPLE))
    else:
        for metric_id, name, unit in _PHONOLOGICAL_PAIR_EMPTY_IDS:
            out.append(unavailable(metric_id, name, "fewer than two nearby resolved content "
                                   "words on any single line", family=FAMILY, unit=unit))
    return out


# -------------------------------------------------------------- phonemizer cross-check

def _phonemizer_finding(config: Mapping[str, Any] | None,
                        lines_words: list[list[_LineWord]]) -> dict[str, Any]:
    sample_size = max(1, int(option(config, "phonemizer_sample_words", 40)))
    sample: list[_LineWord] = []
    for entries in lines_words:
        for entry in entries:
            if entry.pronunciation.phones:
                sample.append(entry)
                if len(sample) >= sample_size:
                    break
        if len(sample) >= sample_size:
            break

    metric_id = "rhythm.prosody_phonemizer_backend_agreement"
    name = ("Syllable-count agreement between phonemizer/espeak and the primary "
           "pronunciation backend")
    if not sample:
        return unavailable(metric_id, name, "no resolved words to sample", family=FAMILY)

    agree = compared = 0
    reason = None
    for entry in sample:
        ipa, why = prosody.phonemize_word(entry.word)
        if ipa is None:
            reason = reason or why
            continue
        compared += 1
        phonemizer_syllables = sum(1 for char in ipa if char in _IPA_VOWEL_CHARS)
        if phonemizer_syllables == prosody.syllable_count(entry.pronunciation.phones):
            agree += 1
    if not compared:
        return unavailable(metric_id, name, reason or prosody.PHONEMIZER_MISSING_HINT,
                           family=FAMILY)
    return finding(metric_id, name, rate(agree, compared), "percent", family=FAMILY,
                   sample_size=compared, min_sample=10,
                   distribution={"words_compared": compared, "words_sampled": len(sample)})


# ------------------------------------------------------------------- poesy cross-check

def _poesy_findings(config: Mapping[str, Any] | None, lines: Sequence[str],
                    own_dominant_meter: str | None) -> list[dict[str, Any]]:
    max_lines = max(1, int(option(config, "poesy_max_lines", 40)))
    max_seconds = float(option(config, "poesy_max_seconds", 20.0))
    result, reason = prosody.run_poesy(lines, max_lines=max_lines, max_seconds=max_seconds)
    if result is None:
        return [
            unavailable("rhythm.prosody_poesy_meter_agreement",
                       "Whether Poesy/Prosodic's meter classification agrees with this suite's "
                       "own dominant-foot classification", reason, family=FAMILY),
            unavailable("rhythm.prosody_poesy_rhyme_scheme_accuracy",
                       "Poesy/Prosodic's own rhyme-scheme-fit accuracy", reason, family=FAMILY),
        ]
    agreement = None
    if own_dominant_meter and result.dominant_meter:
        agreement = 100.0 if own_dominant_meter in result.dominant_meter.lower() else 0.0
    out = [finding("rhythm.prosody_poesy_meter_agreement",
                   "Whether Poesy/Prosodic's meter classification agrees with this suite's own "
                   "dominant-foot classification", agreement, "percent", family=FAMILY,
                   sample_size=1, min_sample=1,
                   distribution={"own_dominant_meter": own_dominant_meter,
                                 "poesy_meter_type_scheme": result.dominant_meter,
                                 "poesy_beat_scheme": (list(result.beat_scheme)
                                                       if result.beat_scheme else None),
                                 "lines_scanned": result.lines_scanned,
                                 "seconds": result.seconds},
                   warning="a single whole-document categorical comparison, not a distribution "
                           "over independent observations" if agreement is not None else
                           "this suite's own scan or Poesy's both need a resolvable dominant "
                           "meter for agreement to be defined")]
    accuracy = result.rhyme_scheme_accuracy
    out.append(finding("rhythm.prosody_poesy_rhyme_scheme_accuracy",
                       "Poesy/Prosodic's own rhyme-scheme-fit accuracy",
                       accuracy * 100 if accuracy is not None else None, "percent",
                       family=FAMILY, sample_size=result.lines_scanned, min_sample=4,
                       distribution={"rhyme_scheme_form": result.rhyme_scheme_form,
                                     "lines_scanned": result.lines_scanned}))
    return out


# --------------------------------------------------------------------- entrypoint

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    use_g2p = _feature(config, "g2p_fallback")
    max_lines = max(50, int(option(config, "max_lines_analyzed", 2000)))
    max_words = max(1000, int(option(config, "max_words_analyzed", 30000)))
    structure, full_structure, sampled, lines_words, stanza_indices, coverage = (
        _document_prosody(analysis, use_g2p, max_lines, max_words))
    lines = [line for stanza in structure.stanzas for line in stanza.lines]

    if _feature(config, "line_stanza_structure"):
        out.extend(_line_stanza_findings(structure, full_structure, sampled))

    needs_pronunciation = (_feature(config, "meter_stress") or _feature(config, "rhyme")
                          or _feature(config, "near_rhyme_feature")
                          or _feature(config, "phonological_patterning"))
    if needs_pronunciation:
        out.append(_pronunciation_coverage_finding(coverage, sampled, len(lines),
                                                   len(full_structure.nonblank_lines), use_g2p))

    own_dominant_meter = None
    if _feature(config, "meter_stress"):
        meter_out, own_dominant_meter = _meter_findings(lines_words)
        out.extend(meter_out)

    if _feature(config, "rhyme"):
        out.extend(_rhyme_findings(config, lines_words, stanza_indices))

    if _feature(config, "near_rhyme_feature"):
        out.extend(_feature_rhyme_findings(config, lines_words))

    if _feature(config, "phonological_patterning"):
        out.extend(_phonological_findings(config, lines_words))

    if _feature(config, "phonemizer_backend"):
        out.append(_phonemizer_finding(config, lines_words))

    if _feature(config, "poesy_crosscheck"):
        out.extend(_poesy_findings(config, lines, own_dominant_meter))

    return out
