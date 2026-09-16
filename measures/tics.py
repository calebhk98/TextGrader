#!/usr/bin/env python3
"""Count the repeated constructions outside readers flagged, against the corpus.

Reviewers reported a set of habits by feel. Most of them are countable, and a
count settles whether a habit is a voice or a tic. Rates are per 100,000 words
so the book and a 70,000-word novel compare directly.

    python3 tics.py                 the manuscript against the corpus
    python3 tics.py --show PATTERN  print every hit for one pattern
"""
import argparse, glob, json, re, statistics as st
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
CORPUS = [
    "/tmp/claude-0/-home-user-test/e98b5ab4-e37f-5614-9ff3-15e67e5c0180/scratchpad/agent_gutenberg/raw",
    "/tmp/claude-0/-home-user-test/e98b5ab4-e37f-5614-9ff3-15e67e5c0180/scratchpad/agent_modern/texts",
]

# The target is a reasonable rate, not corpus parity. The corpus is 23 books and
# a voice is made of repetition; the author's instruction is to cut roughly to a
# third of the current rate on the worst offenders and stop there. Several of
# these will still sit far above every reference book afterwards, on purpose.
# Every number word from one to fifteen sits above the highest-numbered book in
# the corpus, so this is a habit of precision rather than three unlucky numbers.
# The target for each is that corpus maximum: no more numerous than the most
# number-heavy of twenty-three published novels. Substituting one number for
# another does not help and only moves the problem, so the fix is to delete the
# precision wherever it is not doing work.
NUMBER_TARGET = {
    "one": 515.7, "two": 282.1, "three": 172.7, "four": 119.3, "five": 112.4,
    "six": 57.3, "seven": 45.9, "eight": 45.9, "nine": 31.2, "ten": 73.4,
    "eleven": 10.3, "twelve": 18.9, "thirteen": 9.7, "fourteen": 11.3,
    "fifteen": 14.5, "forty": 30.0,
}
TARGETS = {
    "habit stated for the reader": 0.0,
    "reads it twice": 3.5,

    "the word 'same'": 125.0,
    "sentence opens She/He + verb": 120.0,


    "the word 'flat'": 35.0,

    "'the way you/she would'": 20.0,
    "hedged exact (about N)": 18.0,
    "eyes on / eyes down": 12.0,
    "hand(s) flat": 10.0,
    # Same target as its sibling gesture above, on the docstring's rule of
    # roughly a third of the current rate. 37 uses across 22 chapters, 31.4 per
    # 100,000, against a corpus median of zero and a maximum of 7.2: the book
    # uses it four times as often as the most hand-heavy of 23 novels, and the
    # median novel never uses it at all.
    "both hands": 10.0,
    # A cap at a quarter of the rate it was found at, on the author's call.
    # That lands under the corpus maximum of 3.3 as well, in a corpus where the
    # median novel never uses the formula once. The bare word 'rather' is
    # deliberately left untargeted - at 95.8 against a corpus median of 72.6 it
    # is ordinary English, and it is the formula rather than the word that the
    # author can hear.
    "'I'd rather X than Y'": 2.0,
    # 142.8 per 100,000 against a corpus median of 0.0 and a maximum of 50.5:
    # nearly three times the most explanation-heavy of 23 novels, in a corpus
    # where the median novel never does it. Target is a third of the measured
    # rate, per the rule above, which lands just under that maximum.
    "', because' inside speech": 48.0,

    # Declining to act. These three are CAPS, not targets: the author's ruling
    # is "about 2x more passive than they should be, and I don't want to
    # increase it," so each is set at half the rate it was found at. Do not
    # read a number under the cap as room to add more. The other three shapes
    # that were measured came back fine and are not tracked: says nothing ran
    # at a fifth of the corpus maximum, so the characters are not quiet - it is
    # the physical and decision beats that had narrowed, not the talking.
    "cap: leaves it / lets it go": 13.3,
    "cap: puts it back down": 29.4,
    "cap: does X instead": 53.5,
    "'the whole/rest of it'": 10.0,

    # The author found this one by reading, not by measuring, which is the
    # usual order in this project. The book ran 28 instances at 22.1 per
    # 100,000 against a corpus median of 1.4 and a maximum of 8.8, so it was
    # sixteen times the median and two and a half times the most in-order-
    # heavy book in the reference set. It is the narrator's tell for
    # competence: everybody in this book recounts things in order, gives the
    # figures by name and in order, reads the column in order. The target is
    # the corpus maximum, as everywhere else here.
    "'in order' (not 'in order to')": 8.8,
    "turning an object": 6.0,
    "announced withholding": 3.0,
    "'that's not X, that's Y'": 2.0,
}

PATTERNS = {
    # The author, on "He says it the way he says everything": this is done
    # constantly, for different people for different actions, and it is talking
    # to the reader. It tells you an action is characteristic instead of letting
    # the repetition do it. Target 0: there is no good instance of it.
    # Thirteen instances at 10.5 per 100,000, against a corpus median of zero
    # and a maximum of 1.58. A whole-book reader stopped counting past twenty
    # and called it a house tell rather than observed behaviour. Target is
    # roughly twice the corpus maximum, which leaves room for the beats where
    # re-reading is the point.
    "reads it twice":
        r"read(?:s|)\s+(?:it|them|the\s+\w+)\s+(?:twice|a\s+(?:second|third)\s+time)",
    "habit stated for the reader":
        r"\b(?:the way|as)\s+(?:he|she|they)\s+"
        r"(?:(?:always|normally|usually|invariably|generally)\s+\w+"
        r"|(?:says?|said|does?|did|answers?|handles?|takes?|reads?)\s+"
        r"(?:everything|anything|it all|all of it|every\s+\w+|each\s+\w+)"
        r"|does\s+with\b)",

    "hedged exact (about N)": r"\babout (?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|a hundred)\b",
    "hand(s) flat":          r"\bhands? (?:flat|pressed flat)\b|\bflat on the (?:table|counter|desk|bench|wall|floor)\b",
    # The sibling gesture. hand(s) flat got a row and a target and was cut back
    # to it; this one was never measured and stands at 37 uses across 22
    # chapters. Nothing measures anything unless it is in this report.
    "both hands":            r"\bboth hands\b",
    # The author, by ear: "I'd rather tell you that plainly than X. IDK how to
    # search for that, but everytime I see it, I can tell." It is a stated
    # preference with the rejected alternative attached, and it is spread across
    # speakers, which is what makes it read as house voice rather than anyone's
    # character. Counted two ways: the whole word, and the formula itself.
    "the word 'rather'":     r"\brather\b",
    # Characters explaining themselves inside their own speech. Found
    # independently by three scene-audit agents in three different chapters,
    # each naming it a top-three item, and shared across the mother, the
    # librarian, a teacher, Sam and Chloe - which is house voice reaching into
    # dialogue rather than anybody's character. The narration-only tic scan in
    # style_report.py could never see it: it strips quoted spans first, which
    # is correct for rule 1 and blind to exactly this.
    # Matches both punctuations on purpose: splitting a compound question turns
    # ", because" into "? Because" and moved five instances out of sight of this
    # row without removing one word of the habit.
    #
    # It also requires a CLOSING quote with no quote mark in between, so the
    # because has to be inside one spoken span. Without that the pattern ran
    # from any quote character forward and swept up narration after a short line
    # of dialogue: it read 124 where the hand catalogue, reading each one,
    # counted 78. The row was reporting the tic plus a slice of house rule 1.
    "', because' inside speech":
        r'"[^"]{10,400}?[,?]\s+[Bb]ecause\b[^"]{0,400}?"',
    "cap: leaves it / lets it go":
        r"\b(?:leaves?|left)\s+(?:it|that|them|the\s+\w+)\s+(?:there|alone|where|at\s+that|be)\b"
        r"|\blets?\s+(?:it|that|them)\s+(?:go|sit|stand|lie|drop|rest)\b"
        r"|\blet\s+(?:it|that|them)\s+(?:go|sit|stand|lie|drop|rest)\b",
    "cap: puts it back down":
        r"\b(?:puts?|put|sets?|set|lays?|laid|places?)\s+(?:it|them|the\s+\w+|his\s+\w+|her\s+\w+)"
        r"\s+(?:back\s+)?down\b"
        r"|\b(?:puts?|put|sets?|set)\s+(?:it|them|the\s+\w+)\s+back\b",
    "cap: does X instead":   r"\binstead\b",
    "'I'd rather X than Y'": r"\b(?:I'?d|I would|he'?d|she'?d|they'?d)\s+rather\b[^.;!?]{0,80}\bthan\b",
    "the word 'flat'":       r"\bflat(?:ly)?\b",
    "'the way you/she would'": r"\bthe way (?:you|she|he|they|somebody|a person|anybody)\b",
    "announced withholding": r"\bkeeps? (?:it|that|the rest of it|them) to (?:him|her|them)sel(?:f|ves)\b|\bkept (?:it|that) to (?:him|her)self\b",
    "'that's not X, that's Y'": r"\b(?:that|this|it)'s not [^.,;!?]{1,40}, (?:that|it)'s\b",
    "'the whole/rest of it'": r"\bthe (?:whole|rest|entire) of it\b",
    "'in order' (not 'in order to')": r"\bin order\b(?!\s+to\b)",
    "the word 'same'":       r"\bsame\b",
    "'though' at clause end": r",\s+though\b",
    "'except' as connector": r",\s+except\b",
    "turning an object":     r"\bturn(?:s|ing|ed)? (?:it|the \w+) over\b",
    "eyes on / eyes down":   r"\beyes (?:on|down|still on)\b|\bnot looking up\b",
    "sentence opens She/He + verb": r"(?:^|(?<=[.!?]\s))(?:She|He) [a-z]+s\b",
}

def words(t): return re.findall(r"[A-Za-z']+", t)

def strip_gut(t):
    m = re.search(r"\*\*\* ?START OF.*?\*\*\*(.*?)\*\*\* ?END OF", t, re.S)
    return m.group(1) if m else t

# Every pattern is matched case-insensitively except the sentence-opening one,
# which needs the capital to find a sentence start. Without this the counter
# silently missed every instance that began a sentence: the "that's not X,
# that's Y" row read six book-wide when the true figure was nine.
CASE_SENSITIVE = {"sentence opens She/He + verb"}

for _n, _t in NUMBER_TARGET.items():
    PATTERNS[f"number: {_n}"] = r"\b" + _n + r"\b"
    TARGETS[f"number: {_n}"] = _t


def measure(text):
    n = len(words(text))
    out = {}
    for k, p in PATTERNS.items():
        flags = re.M if k in CASE_SENSITIVE else re.M | re.I
        out[k] = 100000 * len(re.findall(p, text, flags)) / n
    return out, n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show")
    a = ap.parse_args()

    book = "\n".join(Path(f).read_text() for f in sorted(glob.glob(str(HERE/"chapters"/"*.md"))))
    if a.show:
        pat = PATTERNS.get(a.show) or a.show
        flags = 0 if a.show in CASE_SENSITIVE else re.I
        for f in sorted(glob.glob(str(HERE/"chapters"/"*.md"))):
            for i, line in enumerate(Path(f).read_text().split("\n"), 1):
                for m in re.finditer(pat, line, flags):
                    s = max(0, m.start()-60)
                    print(f"{Path(f).stem}:{i}  ...{line[s:m.end()+60]}...")
        return

    bk, bn = measure(book)
    ref = []
    for d in CORPUS:
        for f in sorted(Path(d).rglob("*.txt")):
            if "stripped" in f.stem: continue
            r, n = measure(strip_gut(f.read_text(encoding="utf-8", errors="replace")))
            if n > 20000: ref.append(r)

    print(f"\n{len(ref)} reference books. Rates per 100,000 words.\n")
    print(f"  {'construction':<32}{'book':>8}{'target':>9}{'corpus med':>12}{'corpus max':>12}{'':>3}")
    print("  " + "-"*78)
    rows, over = [], 0
    for k in PATTERNS:
        vals = sorted(r[k] for r in ref)
        med, mx = st.median(vals), max(vals)
        t = TARGETS.get(k)
        excess = bk[k]/t if t else 0
        rows.append((excess, k, bk[k], t, med, mx))
    for excess, k, b, t, med, mx in sorted(rows, reverse=True):
        if t is None:
            print(f"  {k:<32}{b:>8.1f}{'-':>9}{med:>12.1f}{mx:>12.1f}")
            continue
        ok = b <= t
        over += 0 if ok else 1
        print(f"  {k:<32}{b:>8.1f}{t:>9.1f}{med:>12.1f}{mx:>12.1f}"
              f"{'  pass' if ok else '  CUT ' + f'{100*(1-t/b):.0f}%'}")
    print(f"\n  {over} of {len(TARGETS)} still over target.")
    print("  The target is a reasonable rate, not the corpus. Several of these stay")
    print("  above every reference book after the cut, which is intended: a voice is")
    print("  made of repetition and the corpus is only twenty-three books.")

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
