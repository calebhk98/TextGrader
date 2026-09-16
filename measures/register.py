#!/usr/bin/env python3
"""Chapters that have drifted more formal than the rest of the book.

Readers found this before any script did. Two of them read chapters 25 and 26
and said the prose had gone formal in a way the first twenty-four had not:
"possesses a voice built for open ground", "in the manner of somebody settling
in", "the particular variety of cold that reaches the fingers a considerable
time before it reaches anything else". They were right, and the thing they were
hearing turned out to be countable.

The count is words of nine letters or more as a share of all words. It is crude
and it works, because the formality in this book arrives as vocabulary rather
than as syntax: a plain observation puts on a long coat. Chapter 25 came in at
5.55% against a book median of 2.54%, the highest of all thirty-six chapters
and more than double the middle of the book.

The judgement is against the book's own median rather than against the corpus,
which is the point. The question is not whether the prose is ornate for a
novel, it is whether a chapter reads as though a different person wrote it. A
chapter is flagged at more than 1.6 times the median, which lets the later
chapters sit where they already sit, since the book does get denser as Chloe
gets older and that rise is wanted.

    python3 register.py            report
    python3 register.py --words N  the long words in the worst chapters
"""
import argparse, glob, re, statistics as st, sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent

WORD = re.compile(r"[A-Za-z']+")
LONG = 9
# A chapter may sit this far above the book's own middle before it reads as
# somebody else's prose. Set from the observed spread: with 25 excluded the
# rest of the book tops out around 1.6, so this flags the outlier and leaves
# the ordinary late-chapter rise alone.
RATIO = 1.6

# A chapter whose long words are its subject rather than its manner. The count
# below cannot tell ornament from vocabulary, and one chapter in the book is
# about paperwork: its long words are registrar, signature, percentage,
# contracts, coordinator, documents, translates. That is what a translation
# office is called, not a costume the prose has put on, and plaining it would
# mean renaming the job.
#
# A sharper measure was tried and is worse. Counting only words that are long
# AND rare in the book flags nine to twelve chapters instead of one, because it
# detects "a later chapter" rather than "an ornate chapter" - the vocabulary
# widens as Chloe ages, which is wanted. Do not reach for that again.
EXEMPT = {
    "28_nineteen": "long words are the job: registrar, signature, contracts, coordinator",
}


def profile(path):
    text = re.sub(r"^#.*$", "", Path(path).read_text(), flags=re.M)
    words = WORD.findall(text)
    if not words:
        return 0.0, [], 0
    long_words = [w for w in words if len(w) >= LONG]
    return len(long_words) / len(words) * 100, long_words, len(words)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--words", type=int, metavar="N",
                    help="print the N commonest long words in each flagged chapter")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(HERE / "chapters" / "*.md"))):
        pct, longs, n = profile(f)
        rows.append((Path(f).stem, pct, longs, n))
    if not rows:
        print("  no chapters found")
        return 0

    median = st.median(r[1] for r in rows)
    ceiling = median * RATIO
    over = [r for r in rows if r[1] > ceiling and r[0] not in EXEMPT]
    excused = [r for r in rows if r[1] > ceiling and r[0] in EXEMPT]

    print(f"  long words are {LONG} letters or more, as a share of all words")
    print(f"  book median {median:.2f}%, flag above {ceiling:.2f}% "
          f"({RATIO} times the median)\n")
    for stem, pct, _, _ in rows:
        if pct > ceiling and stem in EXEMPT:
            mark = f"  <-- over, and excused: {EXEMPT[stem]}"
        elif pct > ceiling:
            mark = "  <-- formal for this book"
        else:
            mark = ""
        print(f"  {stem[:24]:26s}{pct:6.2f}%{mark}")

    if a.words:
        from collections import Counter
        for stem, pct, longs, _ in over:
            common = Counter(w.lower() for w in longs).most_common(a.words)
            print(f"\n  {stem}: " + ", ".join(f"{w} ({c})" for w, c in common))

    if over:
        names = ", ".join(f"{r[0][:2]} ({r[1]:.2f}%)" for r in over)
        # One verdict line, matching how the scorecard reads every other
        # measure. The per-chapter rows above are the detail to fix from; the
        # scorecard has been burned once already by matching a detail line.
        print(f"\n  FAIL  {len(over)} chapter(s) over: {names}")
        print("  A chapter here reads as though a different person wrote it.\n")
        return 1
    if excused:
        print(f"\n  {len(excused)} over the line and excused by name in this "
              f"script, with the reason on the row above.")
    print(f"\n  none over {ceiling:.2f}% unexcused: pass\n")
    return 0


# Run from grade.py, not on its own. Every measure in measures/ reports one
# slice; the scorecard is the whole picture and it is the thing that says
# whether a pass helped.
def _solo_notice():
    import sys, os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [one measure of thirteen. the scorecard is: python3 grade.py]",
          file=sys.stderr)


if __name__ == "__main__":
    _solo_notice()
    sys.exit(main() or 0)
