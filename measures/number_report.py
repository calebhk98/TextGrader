#!/usr/bin/env python3
"""Count the numbers in the manuscript and say whether the same few keep coming back.

The suspicion this answers is that the book reaches for the same handful of
numbers over and over. That is measurable two ways, and the second is the one
that matters:

  rate      how many numbers per thousand words. A dense book is not a problem
            by itself; this manuscript counts things because its narrator counts
            things.
  spread    how much of that total sits on the few commonest values. A writer
            with a tic uses four and eleven for everything; a writer without one
            spreads the same number of numbers across more values.

Both are printed against the 23-book corpus, so "too many" means more than real
books do rather than more than felt right on the day.

    python3 number_report.py                     the book, against the corpus
    python3 number_report.py --chapters          one row per chapter
    python3 number_report.py --value four        every use of one number
"""

import argparse
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
CORPUS = [
    Path("/tmp/claude-0/-home-user-test/e98b5ab4-e37f-5614-9ff3-15e67e5c0180/scratchpad/agent_gutenberg/raw"),
    Path("/tmp/claude-0/-home-user-test/e98b5ab4-e37f-5614-9ff3-15e67e5c0180/scratchpad/agent_modern/texts"),
]

WORDS = ("one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty "
         "fifty sixty seventy eighty ninety hundred thousand million").split()
# "One" as a pronoun ("the one who", "one of them") is not a count, and neither
# is "a hundred" used as "lots". Those are left in: stripping them needs a parser,
# and they land in every text in the corpus equally, so the comparison holds.
NUMBER = re.compile(r"\b(?:\d[\d,]*|" + "|".join(WORDS) + r")\b", re.I)
WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def numbers(text):
    text = re.sub(r"(?m)^#.*$", "", text)
    text = re.sub(r"(?m)^\*[A-Z][a-z]+ \d{4}.*\*$", "", text)   # the date lines
    return [m.group(0).lower().replace(",", "") for m in NUMBER.finditer(text)]


def profile(text):
    n = numbers(text)
    w = len(WORD.findall(text))
    c = Counter(n)
    total = len(n) or 1
    return {
        "words": w,
        "count": len(n),
        "rate": 1000 * len(n) / w if w else 0,
        "distinct": len(c),
        "top5": 100 * sum(v for _, v in c.most_common(5)) / total,
        "top1": 100 * c.most_common(1)[0][1] / total if c else 0,
        "counter": c,
    }


def strip_gutenberg(t):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*", t, re.S)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF", t)
    return t[: m.start()] if m else t


def corpus_profiles():
    out = []
    for d in CORPUS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.txt")):
            if "stripped" in f.stem:
                continue
            out.append((f.stem, profile(strip_gutenberg(
                f.read_text(encoding="utf-8", errors="replace")))))
    return out


def band(label, value, vals, higher_is_worse=True):
    lo, med, hi = min(vals), st.median(vals), max(vals)
    flag = ""
    if higher_is_worse and value > hi:
        flag = "   above every book in the corpus"
    elif not higher_is_worse and value < lo:
        flag = "   below every book in the corpus"
    print(f"  {label:34}{value:8.1f}   corpus {lo:.1f} / {med:.1f} / {hi:.1f}{flag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", type=Path, default=HERE / "HALSTEAD.md")
    ap.add_argument("--chapters", action="store_true", help="one row per chapter")
    ap.add_argument("--value", help="print every line containing this number")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    if a.value:
        pat = re.compile(rf"\b{re.escape(a.value)}\b", re.I)
        for f in sorted((HERE / "chapters").glob("*.md")):
            for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
                if pat.search(line):
                    for m in pat.finditer(line):
                        s = max(0, m.start() - 45)
                        print(f"{f.stem}:{i}  ...{line[s:m.end() + 45]}...")
        return

    if a.chapters:
        print(f"{'chapter':24}{'nums':>6}{'/1k':>7}{'distinct':>10}{'top5 %':>8}  commonest")
        for f in sorted((HERE / "chapters").glob("*.md")):
            p = profile(f.read_text(encoding="utf-8"))
            top = ", ".join(f"{k} x{v}" for k, v in p["counter"].most_common(3))
            print(f"{f.stem[:23]:24}{p['count']:>6}{p['rate']:>7.1f}"
                  f"{p['distinct']:>10}{p['top5']:>8.1f}  {top}")
        return

    if not a.path.is_file():
        sys.exit(f"error: no such file {a.path}")
    me = profile(a.path.read_text(encoding="utf-8"))
    ref = corpus_profiles()

    print(f"{a.path.name}: {me['count']:,} numbers in {me['words']:,} words, "
          f"{me['distinct']} distinct values")
    if not ref:
        print("\nno corpus texts available, printing the book's own figures only")
    else:
        print("\n  measure                             this   corpus low / median / high")
        band("numbers per 1000 words", me["rate"], [p["rate"] for _, p in ref])
        band("share on the commonest 5 values %", me["top5"], [p["top5"] for _, p in ref])
        band("share on the single commonest %", me["top1"], [p["top1"] for _, p in ref])
        band("distinct values used", me["distinct"],
             [p["distinct"] for _, p in ref], higher_is_worse=False)

    print(f"\n  the {a.top} commonest, as a share of all numbers, against the corpus:")
    print(f"    {'value':<10}{'uses':>6}{'share':>8}{'corpus med':>12}{'corpus max':>12}")
    for k, v in me["counter"].most_common(a.top):
        share = 100 * v / me["count"]
        if ref:
            others = [100 * p["counter"].get(k, 0) / max(p["count"], 1) for _, p in ref]
            med, hi = st.median(others), max(others)
            flag = "   <-- above every book" if share > hi else ""
            print(f"    {k:<10}{v:>6}{share:>7.1f}%{med:>11.1f}%{hi:>11.1f}%{flag}")
        else:
            print(f"    {k:<10}{v:>6}{share:>7.1f}%")


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
