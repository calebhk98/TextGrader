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
import statistics as statistics
import sys
from collections import Counter
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from project_config import CHAPTERS_DIR, CORPUS_DIRS, MANUSCRIPT

CORPUS = CORPUS_DIRS

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
    return [match.group(0).lower().replace(",", "") for match in NUMBER.finditer(text)]


def profile(text):
    values = numbers(text)
    words = len(WORD.findall(text))
    counter = Counter(values)
    total = len(values) or 1
    return {
        "words": words,
        "count": len(values),
        "rate": 1000 * len(values) / words if words else 0,
        "distinct": len(counter),
        "distinct_rate": 1000 * len(counter) / words if words else 0,
        "top5": 100 * sum(count for _, count in counter.most_common(5)) / total,
        "top1": 100 * counter.most_common(1)[0][1] / total if counter else 0,
        "counter": counter,
    }


def strip_gutenberg(text):
    match = re.search(r"\*\*\* ?START OF.*?\*\*\*", text, re.S)
    if match:
        text = text[match.end():]
    match = re.search(r"\*\*\* ?END OF", text)
    return text[: match.start()] if match else text


def corpus_profiles():
    out = []
    for directory in CORPUS:
        if not directory.is_dir():
            continue
        for text_path in sorted(directory.glob("*.txt")):
            if "stripped" in text_path.stem:
                continue
            out.append((text_path.stem, profile(strip_gutenberg(
                text_path.read_text(encoding="utf-8", errors="replace")))))
    return out


def band(label, value, vals, higher_is_worse=True):
    minimum, med, maximum = min(vals), statistics.median(vals), max(vals)
    flag = ""
    if higher_is_worse and value > maximum:
        flag = "   above every book in the corpus"
    elif not higher_is_worse and value < minimum:
        flag = "   below every book in the corpus"
    print(f"  {label:34}{value:8.1f}   corpus {minimum:.1f} / {med:.1f} / {maximum:.1f}{flag}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path, default=MANUSCRIPT)
    parser.add_argument("--chapters", action="store_true", help="one row per chapter")
    parser.add_argument("--value", help="print every line containing this number")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    if args.value:
        pat = re.compile(rf"\b{re.escape(args.value)}\b", re.I)
        for chapter_path in sorted(CHAPTERS_DIR.glob("*.md")):
            for line_number, line in enumerate(chapter_path.read_text(encoding="utf-8").split("\n"), 1):
                if pat.search(line):
                    for match in pat.finditer(line):
                        start = max(0, match.start() - 45)
                        print(f"{chapter_path.stem}:{line_number}  ...{line[start:match.end() + 45]}...")
        return

    if args.chapters:
        print(f"{'chapter':24}{'nums':>6}{'/1k':>7}{'distinct':>10}{'top5 %':>8}  commonest")
        for chapter_path in sorted(CHAPTERS_DIR.glob("*.md")):
            chapter_profile = profile(chapter_path.read_text(encoding="utf-8"))
            top = ", ".join(f"{value} x{count}" for value, count in chapter_profile["counter"].most_common(3))
            print(f"{chapter_path.stem[:23]:24}{chapter_profile['count']:>6}{chapter_profile['rate']:>7.1f}"
                  f"{chapter_profile['distinct']:>10}{chapter_profile['top5']:>8.1f}  {top}")
        return

    if not args.path.is_file():
        sys.exit(f"error: no such file {args.path}")
    book_profile = profile(args.path.read_text(encoding="utf-8"))
    ref = corpus_profiles()

    print(f"{args.path.name}: {book_profile['count']:,} numbers in {book_profile['words']:,} words, "
          f"{book_profile['distinct']} distinct values")
    if not ref:
        print("\nno corpus texts available, printing the book's own figures only")
    else:
        print("\n  measure                             this   corpus low / median / high")
        band("numbers per 1000 words", book_profile["rate"], [chapter_profile["rate"] for ignored, chapter_profile in ref])
        band("share on the commonest 5 values %", book_profile["top5"], [chapter_profile["top5"] for ignored, chapter_profile in ref])
        band("share on the single commonest %", book_profile["top1"], [chapter_profile["top1"] for ignored, chapter_profile in ref])
        band("distinct values per 1000 words", book_profile["distinct_rate"],
             [chapter_profile["distinct_rate"] for ignored, chapter_profile in ref], higher_is_worse=False)

    print(f"\n  the {args.top} commonest, as a share of all numbers, against the corpus:")
    print(f"    {'value':<10}{'uses':>6}{'share':>8}{'corpus med':>12}{'corpus max':>12}")
    for value, count in book_profile["counter"].most_common(args.top):
        share = 100 * count / book_profile["count"]
        if ref:
            others = [100 * chapter_profile["counter"].get(value, 0) / max(chapter_profile["count"], 1) for ignored, chapter_profile in ref]
            med, highest = statistics.median(others), max(others)
            flag = "   <-- above every book" if share > highest else ""
            print(f"    {value:<10}{count:>6}{share:>7.1f}%{med:>11.1f}%{highest:>11.1f}%{flag}")
        else:
            print(f"    {value:<10}{count:>6}{share:>7.1f}%")


# Run from grade.py, not on its own. Each script in measures/ reports one
# diagnostic; grade.py assembles enabled checks and is the interface that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --show and the per-file
# arguments are for, and it is never how a pass gets judged.
def _solo_notice():
    import sys, os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [bundled diagnostic; use grade.py for the structured report]",
          file=sys.stderr)

if __name__ == "__main__":
    _solo_notice()
    main()
