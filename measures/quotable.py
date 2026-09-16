#!/usr/bin/env python3
"""How often a speech ends on a maxim, against the corpus.

A reader at chapter 9 put it this way: the book has one gear for climactic
dialogue and everybody in it speaks in that gear. Every character, at the
moment a scene turns, resolves the beat into a tight symmetrical line that
would look at home on a poster.

    "You can hand somebody a month. You cannot hand somebody a street."

The countable part of that is not the symmetry, which turns out to be rare.
It is that the last sentence of a speech stops being about the people in the
room and becomes a general statement about how the world works: present
tense, no proper nouns, an indefinite subject, a copula. A particular becomes
a maxim, and the scene signs off on it.

That shape is measurable. Ten speeches in the book ended on a maxim, 1.42% of
the multi-sentence speeches, against a corpus running 0.0% to 1.41%, median
0.64%. Level with the most maxim-heavy book in the reference set and past none
of it, which is why this is not wired into grade.py: a target set there would
be a target set on noise, and it is the same verdict agency.py got before that
script was deleted.

The list is what earned its keep. Ten lines is small enough to judge one at a
time, and the author did, keeping three and cutting seven:

    kept    14  "You cannot track it, and shooting one down in the air is
                 beyond everybody on this field."
    kept    15  "Whoever calls the count is the one whose bad afternoon
                 everybody else has to have."
    kept    30  "Who paid for the steel is somebody else's question."

The seven cut were rewritten to end on a particular instead of a general
claim: Priya's cousin still thinks a pony is a baby horse, the teacher will
show Chloe the room on the way out, the bank man has nothing on his desk that
changes it. Chapter 2's is still standing because chapters 1 and 2 are locked.
The book now sits at 0.55%, under the corpus median.

What the detector cannot see is the symmetry itself - "you can hand somebody a
month, you cannot hand somebody a street" - because in this book it is rare. A
clause-alignment scan over the whole manuscript found three instances, inside
the corpus range too. If the complaint outlives these cuts, it is that scan
that needs building, not this one.

    python3 quotable.py            report
    python3 quotable.py --corpus   the per-book corpus table
"""
import argparse, glob, re, sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent

CORPUS_DIRS = (
    "/tmp/claude-0/-home-user-test/e98b5ab4-e37f-5614-9ff3-15e67e5c0180/scratchpad/agent_gutenberg/raw",
    "/tmp/claude-0/-home-user-test/e98b5ab4-e37f-5614-9ff3-15e67e5c0180/scratchpad/agent_modern/texts",
)

WORD = re.compile(r"[A-Za-z']+")
QUOTE = re.compile(r'["“]([^"“”]{25,1500})["”]')

# An indefinite subject, or a quantifier that makes the claim general.
GENERIC = re.compile(r"\b(nobody|no one|everybody|everyone|somebody|someone|"
                     r"anybody|anyone|nothing|everything|anything|people|"
                     r"always|never|every|a person|most people)\b", re.I)

# A maxim asserts. It needs a present-tense verb of state or of what happens.
COPULA = re.compile(r"\b(is|isn't|are|aren't|means|matters|counts|works|"
                    r"happens|costs|takes|comes down to)\b", re.I)

# Any past tense at all anchors the sentence to this scene, which is the
# opposite of the shape being counted.
PAST = re.compile(r"\b(was|were|had|did|said|went|came|took|got|saw|knew|"
                  r"told|made|felt|looked|turned|walked|asked|gave)\b"
                  r"|\b\w+ed\b", re.I)

# A name makes it a statement about a person rather than about the world.
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")

# "You" used for anybody at all, rather than for the person being spoken to.
IMPERSONAL_YOU = re.compile(r"\byou (can|can't|cannot|don't|do|have to|get|"
                            r"give|need|end up|start|stop|say|ask|know)\b", re.I)

# A speaker undertaking to do something is talking about this scene, however
# many indefinite pronouns the sentence happens to contain.
PROMISE = re.compile(r"\b(I'll|I will|we'll|we will|I'm going to|"
                     r"we're going to)\b", re.I)


def closers(text):
    """The last sentence of every speech long enough to have arrived at one."""
    out = []
    for spoken in QUOTE.findall(text):
        parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', spoken) if s.strip()]
        if len(parts) >= 2:
            out.append(parts[-1])
    return out


def is_maxim(s):
    # A question asks; it does not pronounce. Four of the first fourteen hits
    # were questions, and dropping them is what moved this from a rate the
    # corpus could match to one it cannot.
    if s.rstrip().endswith("?"):
        return False
    n = len(WORD.findall(s))
    if not 4 <= n <= 22:
        return False
    if PAST.search(s):
        return False
    # The opening word is capitalised because it opens; that is not a name.
    if PROPER.search(s[0].lower() + s[1:] if s else s):
        return False
    if PROMISE.search(s):
        return False
    # A list of particulars is not a general claim, whatever its last clause
    # does. Two or more commas with no verb before the first one is a list.
    head = s.split(",")[0]
    if s.count(",") >= 2 and not COPULA.search(head):
        return False
    if not COPULA.search(s):
        return False
    return bool(GENERIC.search(s) or IMPERSONAL_YOU.search(s))


def measure(paths):
    total = 0
    hits = []
    for p in paths:
        text = Path(p).read_text(errors="ignore")
        for c in closers(text):
            total += 1
            if is_maxim(c):
                hits.append((Path(p).stem, c))
    return hits, total


def corpus_rates():
    rows = []
    for d in CORPUS_DIRS:
        for f in sorted(glob.glob(d + "/*")):
            # The corpus carries a stripped copy of each modern book; counting
            # both would weight those authors twice.
            if "strip" in Path(f).name:
                continue
            hits, total = measure([f])
            if total >= 50:
                rows.append((Path(f).stem, len(hits), total, len(hits) / total * 100))
    rows.sort(key=lambda r: r[3])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="store_true", help="the per-book corpus table")
    a = ap.parse_args()

    rows = corpus_rates()
    if not rows:
        print("  corpus texts not readable; nothing to compare against")
        return 0
    ceiling = max(r[3] for r in rows)
    median = rows[len(rows) // 2][3]

    if a.corpus:
        for stem, h, t, pct in rows:
            print(f"  {stem[:38]:38s} {h:3d}/{t:5d}  {pct:5.2f}%")
        print(f"\n  median {median:.2f}%   maximum {ceiling:.2f}%   {len(rows)} books")
        return 0

    hits, total = measure(sorted(glob.glob(str(HERE / "chapters" / "*.md"))))
    pct = len(hits) / total * 100 if total else 0.0

    print(f"  {len(hits)} of {total} multi-sentence speeches end on a maxim: "
          f"{pct:.2f}%")
    print(f"  corpus median {median:.2f}%, corpus maximum {ceiling:.2f}% "
          f"across {len(rows)} books\n")

    for stem, line in hits:
        print(f"  {stem[:22]:22s} {line}")

    chapters = len({s for s, _ in hits})
    print(f"\n  spread across {chapters} chapters. The habit belongs to no one "
          f"character,")
    print(f"  which is the reader's actual complaint: one gear, everybody in it.")

    if pct > ceiling:
        allowed = int(ceiling * total / 100)
        print(f"\n  {pct:.2f}% is over corpus max of {ceiling:.2f}%. "
              f"CUT to {allowed} or fewer.\n")
        return 1
    print(f"\n  {pct:.2f}% is at or under the corpus maximum: pass\n")
    return 0


# Run from grade.py, not on its own. Every measure in measures/ reports one
# slice; the scorecard is the whole picture and it is the thing that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --corpus is for, and it is never
# how a pass gets judged.
def _solo_notice():
    import sys, os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [one measure of thirteen. the scorecard is: python3 grade.py]",
          file=sys.stderr)


if __name__ == "__main__":
    _solo_notice()
    sys.exit(main() or 0)
