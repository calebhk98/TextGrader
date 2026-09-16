#!/usr/bin/env python3
"""Grade prose maturity against a reference corpus, not against invented targets.

style_report.py checks a chapter against the numbers written into
STYLE_GUIDES.md. Several of those have no source behind them, and some have
floors that reward choppier writing: chapter 8, the most grown-up chapter in
the book, fails there partly for not having ENOUGH short sentences.

This grades differently. Every measure is compared against 23 real books, and
reported as a percentile in that field plus a straight win/loss against one
named book. The default benchmark is Peter Pan, which is below the target
audience, so losing to it on a measure is a clear signal rather than a
judgement call.

    python3 prose_grade.py HALSTEAD.md
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
import statistics as st
import sys
from pathlib import Path

from style_report import paragraphs, sents, words

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
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


def syllables(w):
    w = w.lower()
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def strip_transcript(text):
    """Remove chat-transcript lines, and report what share of the words they were."""
    lines = text.split("\n")
    kept = [l for l in lines if not TRANSCRIPT.match(l.strip())]
    total = len(re.findall(r"[A-Za-z][A-Za-z']*", text))
    left = len(re.findall(r"[A-Za-z][A-Za-z']*", "\n".join(kept)))
    share = 100 * (total - left) / total if total else 0.0
    return "\n".join(kept), share


WORDFREQ = HERE / "word_frequency.json"
_FREQ = None


def word_frequency():
    """Occurrences per million, built from the 23-book reference corpus.

    The real Lexile measure uses a proprietary frequency corpus that is not
    available here. This is the same shape of input drawn from the same books
    the rest of the script grades against, which is the best substitute on hand
    and is why the Lexile column is labelled approximate.
    """
    global _FREQ
    if _FREQ is None:
        try:
            _FREQ = json.loads(WORDFREQ.read_text())["freq_per_million"]
        except Exception:
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
    lwf = [math.log10(max(freq.get(w, RARE), RARE) * 5) for w in lower_words]
    raw = 9.82247 * math.log(st.fmean(sent_lengths)) - 2.14634 * st.fmean(lwf)
    return 46.45 * raw + 102.45


def measure(text, floor=40):
    """Every metric for one text. Returns None under `floor` sentences.

    sttr comes back None when the text is shorter than one 1000-word window.
    Type-token ratio falls as a text gets longer, so measuring it over a
    600-word chapter and comparing that with a 70,000-word book flatters the
    chapter by twenty points or more. The sampled form exists to remove that
    dependence, and it cannot do so without a full window.
    """
    full_words = len(words(text))
    text, transcript_share = strip_transcript(text)
    paras = [p for p in paragraphs(text) if p.strip() != "---"]
    sl_all = [(s, len(words(s))) for p in paras for s in sents(p)]
    sl_all = [(s, n) for s, n in sl_all if n]
    if len(sl_all) < floor:
        return None
    ss = [s for s, _ in sl_all]
    sl = [n for _, n in sl_all]
    w = words(text)
    lw = [x.lower() for x in w]
    n_s, n_w = len(sl), len(w)

    runs = inrun = 0
    for v in sl:
        if v < 8:
            runs += 1
        else:
            inrun += runs if runs >= 3 else 0
            runs = 0
    inrun += runs if runs >= 3 else 0

    win = [len(set(lw[i:i + 1000])) / 1000 for i in range(0, len(lw) - 1000, 1000)]
    wps = st.fmean(sl)

    # Paragraph shape. A paragraph with no words in it is a stray marker line,
    # not a paragraph, and would drag both averages down.
    para_s, para_w = [], []
    for p in paras:
        n = [x for x in sents(p) if words(x)]
        if n:
            para_s.append(len(n))
            para_w.append(len(words(p)))
    and2 = sum(1 for s in ss if len(re.findall(r"\band\b", s.lower())) >= 2)
    negative = sum(1 for s in ss if NEGATIVE.search(s))

    n_chars = sum(len(x) for x in w)

    return {
        "fk": 0.39 * wps + 11.8 * (sum(syllables(x) for x in w) / n_w) - 15.59,
        "ari": 4.71 * (n_chars / n_w) + 0.5 * wps - 21.43,
        "lexile": lexile(sl, lw),
        "wps": wps,
        "sttr": 100 * st.fmean(win) if win else None,
        "commas": text.count(",") / n_s,
        "subord": 100 * sum(1 for s in ss if re.search(SUBORDINATOR, s, re.I)) / n_s,
        "relcl": 100 * sum(1 for s in ss if re.search(RELATIVE, s, re.I)) / n_s,
        "b2035": 100 * sum(1 for v in sl if 20 < v <= 35) / n_s,
        "long7": 100 * sum(1 for x in w if len(x) >= 7) / n_w,
        "u10": 100 * sum(1 for v in sl if v < 10) / n_s,
        "simple": 100 * sum(1 for s in ss
                            if not re.search(SUBORDINATOR, s, re.I)
                            and not re.search(RELATIVE, s, re.I)) / n_s,
        "shortruns": 100 * inrun / n_s,
        "top100": 100 * sum(1 for x in lw if x in TOP100) / n_w,
        "slcv": 100 * st.stdev(sl) / wps if len(sl) > 1 else 0.0,
        "wpp": st.fmean(para_w) if para_w else 0.0,
        "spp": st.fmean(para_s) if para_s else 0.0,
        "wlen": st.fmean([len(x) for x in w]),
        "front": 100 * sum(1 for s in ss if FRONTLOAD.match(s.strip())) / n_s,
        "and2": 100 * and2 / n_s,
        "negative": 100 * negative / n_s,
        "andrate": 100 * lw.count("and") / n_w,
        "_words": n_w,
        # The chapter as written, transcript included. Every graded measure
        # excludes chat lines, which is right for reading grade and wrong for
        # length: it made all five chat chapters look hundreds of words short
        # of the floor when only one of them is. Judge length on this.
        "_words_all": full_words,
        "_sentences": n_s,
        "_paragraphs": len(para_w),
        "_transcript": transcript_share,
    }


def strip_gutenberg(t):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*(.*?)\*\*\* ?END OF", t, re.S)
    return m.group(1) if m else t


def build_reference(dirs, out):
    books = {}
    for d in dirs:
        for f in sorted(Path(d).rglob("*.txt")):
            if "stripped" in f.stem:
                continue
            got = measure(strip_gutenberg(f.read_text(encoding="utf-8", errors="replace")))
            if got:
                books[f.stem] = got
                print(f"  measured {f.stem} ({got['_words']:,} words)")
    out.write_text(json.dumps(books, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {len(books)} books to {out}")


def percentile(values, x):
    return 100 * sum(1 for v in values if v < x) / len(values)


def grade(path, ref, benchmark, brief):
    # Strip Gutenberg boilerplate here too, or a reference book graded against
    # itself scores differently from its own stored entry.
    text = strip_gutenberg(Path(path).read_text(encoding="utf-8"))
    got = measure(text)
    if not got:
        print(f"{Path(path).name}: too short to grade (needs 40+ sentences)")
        return None

    bench = ref[benchmark]
    losses, pcts, core, skipped = [], [], [], []
    lines = []
    for key, (label, higher) in METRICS.items():
        if got[key] is None:
            skipped.append(label)
            continue
        vals = [b[key] for b in ref.values()]
        p = percentile(vals, got[key])
        if not higher:
            p = 100 - p
        pcts.append(p)
        if key in CORE12:
            core.append(p)
        # >= / <= so a book tied with the benchmark, including the benchmark
        # itself, is not scored as a loss.
        beat = got[key] >= bench[key] if higher else got[key] <= bench[key]
        # Distance from the benchmark in corpus standard deviations, so gaps on
        # measures with different units can be ranked against each other.
        sd = st.stdev(vals)
        gap = (bench[key] - got[key]) / sd * (1 if higher else -1)
        if not beat:
            losses.append((-gap, label, got[key], bench[key], gap))
        lines.append((p, label, got[key], bench[key], beat))

    print("=" * 78)
    print(f"{Path(path).name}  |  {got['_words']:,} words  |  "
          f"benchmark: {benchmark}  |  corpus: {len(ref)} books")
    n_m = len(METRICS) - len(skipped)
    print(f"\nmaturity percentile (median of {n_m} measures): {st.median(pcts):.0f}"
          f"      lost to benchmark on {len(losses)} of {n_m}")
    print(f"  on the original 12 measures: {st.median(core):.0f}")
    if got["_transcript"] >= 1:
        print(f"  {got['_transcript']:.0f}% of this chapter is chat transcript, "
              f"measured separately and excluded above")
    if not brief:
        print(f"\n  {'measure':46}{'this':>8}{'bench':>8}{'pct':>6}")
        for p, label, mine, theirs, beat in sorted(lines):
            print(f"  {label:46}{mine:>8.2f}{theirs:>8.2f}{p:>5.0f}%  "
                  f"{'' if beat else '<-- loses'}")
    if skipped:
        print(f"  not measurable in a text this short: {', '.join(skipped)}")
    if not brief:
        print(f"\n  monitored, not graded (corpus low / median / high):")
        for key, label in MONITOR.items():
            vals = sorted(b[key] for b in ref.values() if key in b)
            if not vals:
                continue
            band = f"{vals[0]:.2f} / {st.median(vals):.2f} / {vals[-1]:.2f}"
            over = "  above every book in the corpus" if got[key] > vals[-1] else ""
            print(f"  {label:46}{got[key]:>8.2f}   corpus {band}{over}")

    if losses:
        print(f"\n  fix first, by size of the gap to {benchmark} "
              f"(in corpus standard deviations):")
        for _, label, mine, theirs, gap in sorted(losses)[:4]:
            print(f"    {label:46}{mine:>8.2f} vs {theirs:>7.2f}   {gap:>4.1f} sd")
    return st.median(pcts), len(losses)


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

    head = f"{'file':<22}" + "".join(f"{h:>8}" for _, h in SUMMARY_COLS)
    print(head)
    print("-" * len(head))
    for stem, got in rows:
        cells = []
        for key, _ in SUMMARY_COLS:
            v = got[key]
            cells.append("       -" if v is None else
                         f"{v:>8,}" if key == "_words" else
                         f"{v:>8.0f}" if key.startswith("_") else f"{v:>8.1f}")
        print(f"{stem[:21]:<22}" + "".join(cells))

    print("-" * len(head))
    for label, pick in (("book median", st.median), ("corpus median", None)):
        if pick:
            vals = {k: pick([g[k] for _, g in rows if g[k] is not None] or [0])
                    for k, _ in SUMMARY_COLS}
        else:
            vals = {k: (st.median([b[k] for b in ref.values() if k in b])
                        if any(k in b for b in ref.values()) else 0.0)
                    for k, _ in SUMMARY_COLS}
        cells = "".join(f"{vals[k]:>8,.0f}" if k.startswith("_") else
                        f"{vals[k]:>8.1f}" for k, _ in SUMMARY_COLS)
        print(f"{label:<22}" + cells)
    print("\ncorpus word and paragraph counts are whole books, so the first three "
          "columns\nonly compare like with like between chapters. A dash under sTTR "
          "means the\nchapter is under 1000 words, which is shorter than the sampling "
          "window.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--benchmark", default="peter_pan")
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    ap.add_argument("--brief", action="store_true", help="summary line per file only")
    ap.add_argument("--summary", action="store_true",
                    help="one row per file with every measure, no grading")
    ap.add_argument("--list-benchmarks", action="store_true")
    ap.add_argument("--build-reference", nargs="+", metavar="DIR")
    a = ap.parse_args()

    if a.build_reference:
        return build_reference(a.build_reference, a.reference)
    if not a.reference.is_file():
        sys.exit(f"error: no reference corpus at {a.reference}; "
                 f"rebuild with --build-reference DIR")
    ref = json.loads(a.reference.read_text(encoding="utf-8"))

    if a.list_benchmarks:
        print(f"{len(ref)} books in {a.reference.name}, by reading grade:")
        for name in sorted(ref, key=lambda k: ref[k]["fk"]):
            print(f"  {ref[name]['fk']:5.1f}  {name}")
        return
    if not a.paths:
        sys.exit("error: give one or more files to grade")
    if a.benchmark not in ref:
        sys.exit(f"error: no book named {a.benchmark}; try --list-benchmarks")

    if a.summary:
        return summary(a.paths, ref)

    results = []
    for p in a.paths:
        got = grade(p, ref, a.benchmark, a.brief)
        if got:
            results.append((p.name, *got))
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
