#!/usr/bin/env python3
"""Phrases the author has ruled out, checked against the chapters.

This file exists because of a failure, and the failure is worth stating. The
author flagged *"and I want to be clear about that before anything else"* every
single time he came across it, was told a check had been added, and found it
still in chapter 1 several passes later. No check had been added. `prose_check.py`
scans the character sheets and has never looked at a chapter.

A ruling that lives only in a conversation gets lost. A ruling with a row in a
script does not. Anything the author rules out by name goes in here, with the
reason, and `grade.py` runs it.

    python3 banned_phrases.py            report
    python3 banned_phrases.py --show N   print the full line for entry N
"""
import argparse, glob, re, sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent

# (pattern, what it is, why it is out)
BANNED = [
    (r"\b(?:say|said|read|reads) (?:it|that|them) again[^.!?\"]{0,30}\bslower\b"
     r"|\bagain,? slower\b",
     "say that again, slower",
     "The author, on finding three of these still in the manuscript after being "
     "told they were gone: bad writing. A character asking for a repeat at "
     "reduced speed is a stock beat that stands in for a reaction. Removed from "
     "chapters 11, 27 and 29."),

    (r"\bI want to be clear about (?:that|this)\b",
     "I want to be clear about that",
     "A speaker announcing the importance of what they are about to say. The "
     "author has flagged this one on sight every time it has appeared."),

    (r"\bbefore anything else\b(?=[^.!?]*\bclear\b)|\bclear about that before anything else\b",
     "clear about that before anything else",
     "The same construction with its tail attached."),

    (r"\blet me be clear\b",
     "let me be clear",
     "Same move, different opening."),

    (r"\bthe (?:simple|honest|plain) truth is\b",
     "the simple truth is",
     "Announcing that what follows is true, which is the narrator or a speaker "
     "putting a thumb on the scale."),

    (r"\bwhat (?:he|she|they) (?:did not|didn't) (?:know|realise|realize) (?:was|then)\b",
     "what she did not know was",
     "Withholding announced to the reader. Rule 1."),

    (r"\bit (?:would|will) be (?:years|a long time) before\b",
     "it would be years before",
     "Narrator stepping outside the scene to flag a future the camera cannot see."),
]


def scan(paths):
    hits = []
    for f in paths:
        for i, line in enumerate(Path(f).read_text(encoding="utf-8").split("\n"), 1):
            for rx, name, why in BANNED:
                for m in re.finditer(rx, line, re.I):
                    hits.append((Path(f).stem, i, name, why, line, m.start()))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int)
    a = ap.parse_args()

    paths = sorted(glob.glob(str(HERE / "chapters" / "*.md")))
    hits = scan(paths)

    if a.show is not None:
        if 1 <= a.show <= len(hits):
            stem, ln, name, why, line, _ = hits[a.show - 1]
            print(f"{stem}:{ln}\n\n{line}\n\n  [{name}] {why}")
        else:
            print(f"no entry {a.show}; there are {len(hits)}")
        return

    print(f"\n  {len(BANNED)} ruled-out phrases, {len(paths)} chapters\n")
    if not hits:
        print("  none present.\n")
        return 0

    for n, (stem, ln, name, why, line, col) in enumerate(hits, 1):
        s = max(0, col - 60)
        print(f"  {n}. {stem}:{ln}  [{name}]")
        print(f"     ...{line[s:col + 80]}...")
    print(f"\n  {len(hits)} present. Each one is a ruling already made.\n")
    return 1


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
    sys.exit(main() or 0)
