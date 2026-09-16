#!/usr/bin/env python3
"""Grade prose maturity against a reference corpus, not against invented targets.

style_report.py checks a chapter against the numbers written into
STYLE_GUIDES.md. Several of those have no source behind them, and some have
floors that reward choppier writing: chapter 8, the most grown-up chapter in
the book, fails there partly for not having ENOUGH short sentences.

This grades differently. Every measure is compared against 23 real books, and
reported as a percentile in that field plus a straight type_token_windows/loss against one
named book. The default benchmark is Peter Pan, which is below the target
audience, so losing to it on a measure is a clear signal rather than a
judgement call.

    python3 prose_grade.py MANUSCRIPT.md
    python3 prose_grade.py chapters/*.md --brief
    python3 prose_grade.py draft.md --benchmark treasure_island
    python3 prose_grade.py --list-benchmarks
    python3 prose_grade.py chapters/*.md --summary   # every measure, one row per file
    python3 prose_grade.py --build-reference DIR1 DIR2   # regenerate the corpus file

Sixteen measures are graded. Twelve cover the sentence and the word: length,
subordination, relative clauses, clause-free sentences, short runs, commas,
reading grade, lexical diversity, long words, common words, and the two bands
at either end. Four more were added later to cover what those twelve miss and
what a reader actually notices: sentence-length variation, words per paragraph,
sentences per paragraph, and mean word length.

A further group is measured but not graded, in MONITOR. These are patterns
with no maturity direction, or ones under a house ban, where the useful output
is the corpus range rather than a percentile: how often a sentence opens on a
subordinate clause, how often it carries two "and"s, and the plain "and" rate.
Grading them would reward moving a number that is not a maturity signal.

Reference data lives in prose_reference.json next to this script. The corpus
texts themselves are not in the repository; the JSON is the durable artifact,
and rebuilding it needs the book files again.
"""

import argparse
import math
import json
import re
import statistics as statistics
import sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from textgrader.text import (paragraphs, sentences as sents,
                             remove_markdown_headings, strip_gutenberg,
                             strip_transcript, words)


def in_twenty_to_thirty_five(word_count):
    return 20 <= word_count <= 35
REFERENCE = HERE / "prose_reference.json"

SUBORDINATOR = (r"\b(because|although|though|while|whereas|since|unless|until|after|before"
                r"|if|when|whenever|as|so that|even though|rather than|whether)\b")
RELATIVE = r"\b(who|whom|whose|which|that)\b"

# A sentence that opens on a subordinate clause and closes it with a comma:
# "Because the room was cold, she kept her coat on." Front-loading is the
# natural replacement for a banned trailing clause, so it is worth watching
# for the same overuse that got the trailing form banned. An opening quote is
# allowed for, since dialogue does this too.
# A line of the group chat: "ruth: what percentage of americans" and so on.
# These are not prose and must not be measured as prose. Chapter 32 is 76%
# transcript, chapter 24 is 47%, and grading those files as if the transcript
# were narration says the writing is simple when what it is is a chat log.
TRANSCRIPT = re.compile(r"^[a-z][a-z0-9_]{1,9}: ")

# The narrative negative: a sentence whose beat is an action not taken or a
# reaction that did not happen. "He doesn't look up." "Nobody says anything."
# "She reads it twice and doesn't add to the thread." One of these is a device.
# One in six sentences is a tic, and the book has been running above every book
# in the corpus on it.
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

# The twelve the book was first graded on. Kept so the headline number stays
# comparable with every earlier measurement in the notes.
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


WORDFREQ = HERE / "word_frequency.json"
_FREQ = None


def word_frequency():
    """Return no implicit frequency table.

    The historical bundled table included Gutenberg boilerplate and its fitted
    coefficients could not be reproduced.  Approximate Lexile is therefore
    unavailable until an explicit, versioned calibration is supplied rather
    than silently presenting an untrustworthy value.
    """
    global _FREQ
    if _FREQ is None:
        _FREQ = {}
    return _FREQ


def lexile(sent_lengths, lower_words):
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
    freq = word_frequency()
    if not freq or not sent_lengths or not lower_words:
        return None
    RARE = 0.05  # per million: about one appearance across a twenty-book shelf
    log_word_frequencies = [math.log10(max(freq.get(word_count, RARE), RARE) * 5) for word_count in lower_words]
    raw = 9.82247 * math.log(statistics.fmean(sent_lengths)) - 2.14634 * statistics.fmean(log_word_frequencies)
    return 46.45 * raw + 102.45


def measure(text, floor=40):
    """Every metric for one text. Returns None under `floor` sentences.

    sttr comes back None when the text is shorter than one 1000-word window.
    Type-token ratio falls as a text gets longer, so measuring it over a
    600-word chapter and comparing that with a 70,000-word book flatters the
    chapter by twenty points or more. The sampled form exists to remove that
    dependence, and it cannot do so without a full window.
    """
    text = remove_markdown_headings(strip_gutenberg(text))
    full_words = len(words(text))
    text, transcript_share = strip_transcript(text)
    paras = [paragraph for paragraph in paragraphs(text) if paragraph.strip() != "---"]
    sentence_data = [(sentence, len(words(sentence))) for paragraph in paras for sentence in sents(paragraph)]
    sentence_data = [(sentence, word_count) for sentence, word_count in sentence_data if word_count]
    if len(sentence_data) < floor:
        return None
    sentences = [sentence for sentence, _ in sentence_data]
    sentence_lengths = [word_count for _, word_count in sentence_data]
    word_tokens = words(text)
    long_words = [length.lower() for length in word_tokens]
    sentence_count, word_count = len(sentence_lengths), len(word_tokens)

    runs = short_run_sentences = 0
    for value in sentence_lengths:
        if value < 8:
            runs += 1
        else:
            short_run_sentences += runs if runs >= 3 else 0
            runs = 0
    short_run_sentences += runs if runs >= 3 else 0

    type_token_windows = [len(set(long_words[index:index + 1000])) / 1000 for index in range(0, len(long_words) - 999, 1000)]
    words_per_sentence = statistics.fmean(sentence_lengths)

    # Paragraph shape. A paragraph with no words in it is a stray marker line,
    # not a paragraph, and would drag both averages down.
    paragraph_sentence_counts, paragraph_word_counts = [], []
    for paragraph in paras:
        paragraph_sentences = [sentence for sentence in sents(paragraph) if words(sentence)]
        if paragraph_sentences:
            paragraph_sentence_counts.append(len(paragraph_sentences))
            paragraph_word_counts.append(len(words(paragraph)))
    multiple_and_count = sum(1 for sentence in sentences if len(re.findall(r"\band\b", sentence.lower())) >= 2)
    negative = sum(1 for sentence in sentences if NEGATIVE.search(sentence))

    character_count = sum(len(length) for length in word_tokens)

    return {
        "fk": 0.39 * words_per_sentence + 11.8 * (sum(syllables(length) for length in word_tokens) / word_count) - 15.59,
        "ari": 4.71 * (character_count / word_count) + 0.5 * words_per_sentence - 21.43,
        "lexile": lexile(sentence_lengths, long_words),
        "wps": words_per_sentence,
        "sttr": 100 * statistics.fmean(type_token_windows) if type_token_windows else None,
        "commas": text.count(",") / sentence_count,
        "subord": 100 * sum(1 for sentence in sentences if re.search(SUBORDINATOR, sentence, re.I)) / sentence_count,
        "relcl": 100 * sum(1 for sentence in sentences if re.search(RELATIVE, sentence, re.I)) / sentence_count,
        "b2035": 100 * sum(1 for value in sentence_lengths if in_twenty_to_thirty_five(value)) / sentence_count,
        "long7": 100 * sum(1 for length in word_tokens if len(length) >= 7) / word_count,
        "u10": 100 * sum(1 for value in sentence_lengths if value < 10) / sentence_count,
        "simple": 100 * sum(1 for sentence in sentences
                            if not re.search(SUBORDINATOR, sentence, re.I)
                            and not re.search(RELATIVE, sentence, re.I)) / sentence_count,
        "shortruns": 100 * short_run_sentences / sentence_count,
        "top100": 100 * sum(1 for length in long_words if length in TOP100) / word_count,
        "slcv": 100 * statistics.stdev(sentence_lengths) / words_per_sentence if len(sentence_lengths) > 1 else 0.0,
        "wpp": statistics.fmean(paragraph_word_counts) if paragraph_word_counts else 0.0,
        "spp": statistics.fmean(paragraph_sentence_counts) if paragraph_sentence_counts else 0.0,
        "wlen": statistics.fmean([len(length) for length in word_tokens]),
        "front": 100 * sum(1 for sentence in sentences if FRONTLOAD.match(sentence.strip())) / sentence_count,
        "and2": 100 * multiple_and_count / sentence_count,
        "negative": 100 * negative / sentence_count,
        "andrate": 100 * long_words.count("and") / word_count,
        "_words": word_count,
        # The chapter as written, transcript included. Every graded measure
        # excludes chat lines, which is right for reading grade and wrong for
        # length: it made all five chat chapters look hundreds of words short
        # of the floor when only one of them is. Judge length on this.
        "_words_all": full_words,
        "_sentences": sentence_count,
        "_paragraphs": len(paragraph_word_counts),
        "_transcript": transcript_share,
    }


def build_reference(dirs, out):
    books = {}
    for directory in dirs:
        for text_path in sorted(Path(directory).rglob("*.txt")):
            if "stripped" in text_path.stem:
                continue
            got = measure(strip_gutenberg(text_path.read_text(encoding="utf-8", errors="replace")))
            if got:
                books[text_path.stem] = got
                print(f"  measured {text_path.stem} ({got['_words']:,} words)")
    out.write_text(json.dumps(books, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {len(books)} books to {out}")


def percentile(values, candidate):
    return 100 * sum(1 for value in values if value < candidate) / len(values) if values else None


def grade(path, ref, benchmark, brief):
    # Strip Gutenberg boilerplate here too, or a reference book graded against
    # itself scores differently from its own stored entry.
    text = strip_gutenberg(Path(path).read_text(encoding="utf-8"))
    got = measure(text)
    if not got:
        print(f"{Path(path).name}: too short to grade (needs 40+ sentences)")
        return None

    if not ref:
        print(f"{Path(path).name}: corpus comparison unavailable (empty reference)")
        return None
    bench = ref.get(benchmark) if benchmark else None
    losses, pcts, core, skipped = [], [], [], []
    lines = []
    for key, (label, higher) in METRICS.items():
        if got[key] is None:
            skipped.append(label)
            continue
        vals = [benchmark_value[key] for benchmark_value in ref.values()
                if benchmark_value.get(key) is not None]
        if not vals:
            skipped.append(label)
            continue
        metric_percentile = percentile(vals, got[key])
        if not higher:
            metric_percentile = 100 - metric_percentile
        pcts.append(metric_percentile)
        if key in CORE12:
            core.append(metric_percentile)
        # >= / <= so a book tied with the benchmark, including the benchmark
        # itself, is not scored as a loss.
        beat = None if bench is None or bench.get(key) is None else (
            got[key] >= bench[key] if higher else got[key] <= bench[key])
        # Distance from the benchmark in corpus standard deviations, so gaps on
        # measures with different units can be ranked against each other.
        standard_deviation = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gap = ((bench[key] - got[key]) / standard_deviation * (1 if higher else -1)
               if bench is not None and standard_deviation else 0.0)
        if beat is False:
            losses.append((-gap, label, got[key], bench[key], gap))
        lines.append((metric_percentile, label, got[key], bench.get(key) if bench else None, beat))

    print("=" * 78)
    print(f"{Path(path).name}  |  {got['_words']:,} words  |  "
          f"benchmark: {benchmark or 'none'}  |  corpus: {len(ref)} books")
    metric_count = len(METRICS) - len(skipped)
    print(f"\nstyle percentile (descriptive median of {metric_count} measures): {statistics.median(pcts):.0f}"
          f"      lost to benchmark on {len(losses)} of {metric_count}")
    print(f"  on the original 12 measures: {statistics.median(core):.0f}")
    if got["_transcript"] >= 1:
        print(f"  {got['_transcript']:.0f}% of this chapter is chat transcript, "
              f"measured separately and excluded above")
    if not brief:
        print(f"\n  {'measure':46}{'this':>8}{'bench':>8}{'pct':>6}")
        for path, label, mine, theirs, beat in sorted(lines):
            theirs_text = f"{theirs:8.2f}" if theirs is not None else f"{'—':>8}"
            print(f"  {label:46}{mine:>8.2f}{theirs_text}{path:>5.0f}%  "
                  f"{'<-- differs' if beat is False else ''}")
    if skipped:
        print(f"  not measurable in a text this short: {', '.join(skipped)}")
    if not brief:
        print(f"\n  monitored, not graded (corpus low / median / high):")
        for key, label in MONITOR.items():
            vals = sorted(benchmark_value[key] for benchmark_value in ref.values() if key in benchmark_value)
            if not vals:
                continue
            band = f"{vals[0]:.2f} / {statistics.median(vals):.2f} / {vals[-1]:.2f}"
            over = "  above every book in the corpus" if got[key] > vals[-1] else ""
            print(f"  {label:46}{got[key]:>8.2f}   corpus {band}{over}")

    if losses:
        print(f"\n  fix first, by size of the gap to {benchmark} "
              f"(in corpus standard deviations):")
        for _, label, mine, theirs, gap in sorted(losses)[:4]:
            print(f"    {label:46}{mine:>8.2f} vs {theirs:>7.2f}   {gap:>4.1f} sd")
    return statistics.median(pcts), len(losses)


# Short column headings for the summary table, in print order.
SUMMARY_COLS = [
    ("_words", "words"), ("_paragraphs", "paras"), ("_sentences", "sents"),
    ("wps", "w/sent"), ("slcv", "sl CV"), ("wpp", "w/para"), ("spp", "s/para"),
    ("wlen", "w len"), ("long7", "7+ch"), ("sttr", "sTTR"), ("top100", "top100"),
    ("fk", "F-K"), ("ari", "ARI"), ("lexile", "Lexile"), ("commas", "commas"), ("subord", "subord"), ("relcl", "relcl"),
    ("simple", "simple"), ("u10", "u10"), ("b2035", "20-35"),
    ("shortruns", "runs"), ("front", "front"), ("and2", "and2"),
    ("andrate", "and%"), ("negative", "neg%"), ("_transcript", "chat%"),
]


def summary(paths, ref):
    """One row per file, every measure, plus the corpus for comparison.

    The per-file report answers "is this chapter mature". This answers "which
    measure is the book losing on, and in which chapters", which is the
    question a revision pass actually starts from.
    """
    rows = []
    for path in paths:
        # No sentence floor here. The floor exists so a percentile is not read
        # off four sentences; a descriptive row is still worth having.
        got = measure(strip_gutenberg(Path(path).read_text(encoding="utf-8")), floor=1)
        if got is None:
            print(f"  {Path(path).stem}: no sentences")
            continue
        rows.append((Path(path).stem, got))
    if not rows:
        return

    head = f"{'file':<22}" + "".join(f"{header:>8}" for _, header in SUMMARY_COLS)
    print(head)
    print("-" * len(head))
    for stem, got in rows:
        cells = []
        for key, _ in SUMMARY_COLS:
            value = got[key]
            cells.append("       -" if value is None else
                         f"{value:>8,}" if key == "_words" else
                         f"{value:>8.0f}" if key.startswith("_") else f"{value:>8.1f}")
        print(f"{stem[:21]:<22}" + "".join(cells))

    print("-" * len(head))
    for label, pick in (("book median", statistics.median), ("corpus median", None)):
        if pick:
            vals = {key: pick([grade_value[key] for _, grade_value in rows if grade_value[key] is not None] or [0])
                    for key, _ in SUMMARY_COLS}
        else:
            vals = {key: (statistics.median([benchmark[key] for benchmark in ref.values() if key in benchmark])
                        if any(key in benchmark for benchmark in ref.values()) else 0.0)
                    for key, _ in SUMMARY_COLS}
        cells = "".join(f"{vals[key]:>8,.0f}" if key.startswith("_") else
                        f"{vals[key]:>8.1f}" for key, _ in SUMMARY_COLS)
        print(f"{label:<22}" + cells)
    print("\ncorpus word and paragraph counts are whole books, so the first three "
          "columns\nonly compare like with like between chapters. A dash under sTTR "
          "means the\nchapter is under 1000 words, which is shorter than the sampling "
          "window.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--benchmark", help="optional named-book diagnostic")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--brief", action="store_true", help="summary line per file only")
    parser.add_argument("--summary", action="store_true",
                    help="one row per file with every measure, no grading")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--build-reference", nargs="+", metavar="DIR")
    args = parser.parse_args()

    if args.build_reference:
        return build_reference(args.build_reference, args.reference)
    if not args.reference.is_file():
        sys.exit(f"error: no reference corpus at {args.reference}; "
                 f"rebuild with --build-reference DIR")
    ref = json.loads(args.reference.read_text(encoding="utf-8"))

    if args.list_benchmarks:
        print(f"{len(ref)} books in {args.reference.name}, by reading grade:")
        for name in sorted(ref, key=lambda k: ref[k]["fk"]):
            print(f"  {ref[name]['fk']:5.1f}  {name}")
        return
    if not args.paths:
        sys.exit("error: give one or more files to grade")
    if args.benchmark and args.benchmark not in ref:
        sys.exit(f"error: no book named {args.benchmark}; try --list-benchmarks")

    if args.summary:
        return summary(args.paths, ref)

    results = []
    for path in args.paths:
        got = grade(path, ref, args.benchmark, args.brief)
        if got:
            results.append((path.name, *got))
        print()
    if len(results) > 1:
        print("=" * 78)
        print(f"{'file':30}{'percentile':>12}{'losses':>12}")
        for name, pct, lost in sorted(results, key=lambda r: r[1]):
            print(f"{name[:29]:30}{pct:>11.0f}%{lost:>12}")


# Run from grade.py, not on its own. Every measure in measures/ reports one
# slice; the scorecard is the whole picture and it is the thing that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --show and the per-file
# arguments are for, and it is never how a pass gets judged.
def _solo_notice():
    import sys, os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [one measure of thirteen. the scorecard is: python3 grade.py]",
          file=sys.stderr)

if __name__ == "__main__":
    _solo_notice()
    main()
