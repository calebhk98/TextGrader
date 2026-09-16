#!/usr/bin/env python3
"""Build HALSTEAD.md from chapters/. One direction only.

chapters/ is the source of truth. Every edit is made there, one file per
chapter, so agents and people can work on different chapters without
colliding. This concatenates them into the single readable manuscript.

Nothing goes the other way any more. The script that re-cut chapters/ from a
manuscript file has been deleted, because running it discarded work: it reverts
every chapter to whatever the manuscript last held, and the manuscript is
always the stale copy.

    python3 build_manuscript.py            write HALSTEAD.md
    python3 build_manuscript.py --check    say what would change, write nothing

**Order comes from the filenames and nothing else.** chapters/NN_slug.md sorts
into reading order, so there is no list to keep updated and no ordering to
remember. To move a chapter, rename the files; the build follows.

The script checks, before writing, that the numbers run 01..NN with no gaps or
duplicates and that each file's own heading matches its filename's number. A
chapter renamed without its heading being changed is the mistake this catches.
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHAPTERS = HERE / "chapters"
OUT = HERE / "HALSTEAD.md"

WORDS = ("One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve "
         "Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen Twenty"
         ).split()


def number_word(n):
    """1 -> 'One', 21 -> 'Twenty-One'. The manuscript spells chapter numbers."""
    if n <= 20:
        return WORDS[n - 1]
    tens, ones = divmod(n, 10)
    stem = {2: "Twenty", 3: "Thirty", 4: "Forty"}[tens]
    return stem if ones == 0 else f"{stem}-{WORDS[ones - 1]}"


def chapters():
    """Every chapters/NN_*.md in filename order, which is reading order."""
    found = []
    for p in sorted(CHAPTERS.glob("*.md")):
        m = re.match(r"(\d+)_", p.name)
        if m:
            found.append((int(m.group(1)), p))
    return found


def check(found):
    """Numbering gaps, duplicates, and headings that disagree with filenames."""
    problems = []
    nums = [n for n, _ in found]
    for i, n in enumerate(nums, start=1):
        if n != i:
            problems.append(f"numbering: expected {i:02d}, found {n:02d} "
                            f"({found[i-1][1].name})")
            break
    for n, p in found:
        head = p.read_text(encoding="utf-8").split("\n", 1)[0]
        want = f"## Chapter {number_word(n)}:"
        if not head.startswith(want):
            problems.append(f"{p.name}: heading is {head!r}, expected it to "
                            f"start {want!r}")
    return problems


def build(found):
    return "\n\n\n".join(p.read_text(encoding="utf-8").strip()
                         for _, p in found) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what would change, write nothing")
    a = ap.parse_args()

    found = chapters()
    if not found:
        sys.exit(f"no chapters found in {CHAPTERS}")

    problems = check(found)
    if problems:
        print("chapters/ is inconsistent:\n")
        for p in problems:
            print(f"  {p}")
        sys.exit("\nfix the filenames or the headings, then run again.")

    new = build(found)
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    words = len(new.split())

    if new == old:
        print(f"{OUT.name} already current: {len(found)} chapters, {words:,} words")
        return
    if a.check:
        d = words - len(old.split())
        print(f"{OUT.name} would change: {len(found)} chapters, "
              f"{words:,} words ({d:+,})")
        return
    OUT.write_text(new, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(found)} chapters, {words:,} words")


if __name__ == "__main__":
    main()
