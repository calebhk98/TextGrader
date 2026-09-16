#!/usr/bin/env python3
"""Flag banned constructions, negative framing, and book references in the sheets.

PROSE_RULES.md rules 1 and 2 ban two families of construction. Rule 1 covers
sentences whose work is a verdict on correctness or ranking. Rule 2 covers
sentences that define a thing by what it is not, or undercut a claim in the
breath that makes it.

    python3 prose_check.py
    python3 prose_check.py characters/RUTH.md --show
    python3 prose_check.py --rule 2

A first version of this check used loose patterns and reported "the sentence
that is correcting them" as a verdict phrase and "narrow rather than broad" as
negation. Both patterns below are written to avoid that: the verdict list
requires a word boundary after the adjective, and the negation list looks for
a denial of an interpretation rather than any ordinary contrast.
"""

import argparse
import re
import sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
SKIP = {"_TEMPLATE.md", "_CALIBRATION.md", "_ALLOCATIONS.md", "_DIFFERENTIATION.md",
        "_SHEET_RULES.md", "_APPEARANCES.md"}

# Rule 1. A verdict standing in for a reason. The trailing boundary keeps
# "that is correcting" and "this is rightly famous for" out of the results.
VERDICT = [
    (r"\b(?:that|this|it|which|he|she|they)(?:'s| is| was)\s+(?:the\s+)?"
     r"(?:correct|right|wrong|best|worst|strongest|weakest)\b(?!\w)", "verdict"),
    (r"\bI (?:was|am) wrong\b", "verdict"),
    (r"\b(?:you|you're|youre)\s+right\b", "verdict"),
    (r"\bthat is a better\b", "verdict"),
    (r"\bthe (?:correct|right) (?:answer|way|version|reading|call)\b", "verdict"),
    (r"\bwhich is the good part\b", "verdict"),
    (r"\b(?:correctly|rightly)\s+(?:identifies|reads|sees|says|notes|calls)\b", "verdict"),
]

# Rule 2. A denial of a reading nobody asserted, or a claim undercut as it lands.
NEGATION = [
    (r"\bnever as\b", "denies a reading"),
    (r"\bnot (?:as )?(?:modesty|arrogance|stoicism|cruelty|kindness|pity|weakness)\b",
     "denies a reading"),
    (r"\bit (?:is|isn't|is not|was|wasn't)\s+n?o?t?\s*(?:modesty|stoicism|arrogance)\b",
     "denies a reading"),
    (r"\bthat (?:reads|looks|sounds) (?:like|as) \w+ from outside\.? It (?:isn't|is not)\b",
     "denies a reading"),
    (r"\brather than (?:as )?(?:modesty|humility|arrogance|stoicism|a performance)\b",
     "denies a reading"),
    (r"\bnot because \w+,? but because\b", "not-A-but-B"),
    (r"\bnot (?:a|an|the) \w+ (?:but|so much as) (?:a|an|the)\b", "not-A-but-B"),
    (r"\b(?:which|that) is not (?:to say|the same as)\b", "undercut"),
    (r"\bwithout (?:being|reading as) \w+ about it\b", "undercut"),
]


# Rule 3. A sheet is about the person, not about this book. The author: "you
# can't use characters/DAVE.md in a ghost book, if it mentions this book." A
# surname or a birth month survives being moved to a haunted house; a chapter
# number does not. Twenty sheets carried a navigation block naming chapters
# before this check existed, permitted by an older version of _SHEET_RULES.md
# and caught by nothing. They are in _APPEARANCES.md now.
BOOK_REFERENCE = [
    (r"\bchapters?\s+\d", "chapter reference"),
    (r"\bchapters?\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|"
     r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
     r"nineteen|twenty)\b", "chapter reference"),
    (r"\bappears in\b", "navigation"),
    (r"\bthe (?:camp|school) chapters\b", "chapter reference"),
    (r"\b(?:his|her|their) viewpoint chapters?\b", "chapter reference"),
]


# Quoted text is the manuscript's, or a character's, so the sheet's own prose
# rules do not reach it. Sam saying "This is the worst mistake of my life" is
# dialogue, not the sheet handing the reader a verdict.
QUOTED = re.compile(r'"[^"]*"|\u201c[^\u201d]*\u201d')

# The verdict family only fires when the sheet is passing judgement. These
# lead-ins turn the same words into ordinary description: somebody telling her
# when she is wrong, a section asking what she is wrong about.
DESCRIBES = re.compile(
    r"\b(?:when|whether|if|tell(?:s|ing)?|told|admit(?:s|ting)?|about|knows?|"
    r"knew|say(?:s|ing)?|said|thinks?|believes?|assumes?|being|been|prove|"
    r"proved|turns? out)\s+(?:\w+\s+){0,3}$")

# _TEMPLATE.md asks every sheet for a "What they are wrong about" section, so
# the heading and its opening clause are the structure rather than a verdict.
SECTION = re.compile(r"^-?\s*What (?:he|she|they)(?:'s| is| are)? wrong about\b", re.I)


def sentences(text):
    """Yield (line_no, sentence). Sheets are markdown, so strip the furniture."""
    for i, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith(("|", "#", "```")):
            continue
        clean = re.sub(r"\*+|_+|`+", "", line)
        clean = QUOTED.sub(" ", clean)
        for s in re.split(r"(?<=[.!?])\s+", clean):
            if s.strip():
                yield i, s.strip()


def scan(path, rules):
    hits = []
    for line_no, s in sentences(path.read_text(encoding="utf-8")):
        for pattern, label in rules:
            m = re.search(pattern, s, re.I)
            if m and label == "verdict" and DESCRIBES.search(s[:m.start()]):
                continue
            if m and label == "verdict" and SECTION.match(s):
                continue
            if m:
                hits.append((line_no, label, m.group(0), s))
                break
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheets", nargs="*", type=Path)
    ap.add_argument("--root", type=Path, default=HERE)
    ap.add_argument("--rule", type=int, choices=(1, 2, 3), help="check only this rule")
    ap.add_argument("--show", action="store_true", help="print the whole sentence")
    a = ap.parse_args()

    rules = VERDICT + NEGATION + BOOK_REFERENCE
    if a.rule == 1:
        rules = VERDICT
    elif a.rule == 2:
        rules = NEGATION
    elif a.rule == 3:
        rules = BOOK_REFERENCE

    files = a.sheets or sorted((a.root / "characters").glob("*.md"))
    total = 0
    for f in files:
        if f.name in SKIP or not f.is_file():
            continue
        hits = scan(f, rules)
        if not hits:
            continue
        total += len(hits)
        print(f"\n{f.name}")
        for line_no, label, found, sentence in hits:
            print(f"  {line_no:>4} [{label}] {found}")
            if a.show:
                print(f"       {sentence[:150]}")

    print(f"\n{total} flagged across {len(files)} files.")
    return 1 if total else 0


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
    sys.exit(main())
