#!/usr/bin/env python3
"""Measure how distinguishable each character's dialogue is from the others'.

The swap test - would anyone notice if this line moved to another character? -
is a judgement call. This is the measurable version: extract each character's
lines, compute the same statistics for each speaker, and show how far apart
they sit. Characters whose numbers coincide have no voice yet.

    python3 voice_separation.py            # both channels
    python3 voice_separation.py --chat     # group chat only
    python3 voice_separation.py --prose    # tagged prose dialogue only

Two channels, measured separately because they are attributable with very
different confidence.

CHAT is exact. Messages are "name: text", so every line has a known speaker
and none are missed.

PROSE IS A BIASED SAMPLE and its numbers must be read with that in mind. Only
lines carrying an explicit "<name> says" tag can be attributed, and the book
drops the tag once a two-hander is established, which is the style guide's own
advice. Tagged lines therefore skew towards the openings of exchanges, which
run short. Treat the prose figures as comparable BETWEEN characters, since the
bias hits every character alike, and not as an estimate of that character's
true average line length.
"""

import argparse
import collections
import re
import statistics as st
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent

# The group chat is not in files of its own. It is written inside the ordinary
# chapters, as name-prefixed lines, across eleven of them from chapter 24 on.
# This was an empty list, so collect_chat() iterated over nothing and the group
# chat reported "nothing with at least 8 lines" every time it was run - several
# hundred lines of the back third of the book, unmeasured, in the one instrument
# built to answer whether the cast sounds alike.
CHAT_FILES = []
# Only these seven ever post in the group chat.
CAST = ["chloe", "ruth", "sam", "kavi", "nadia", "eli", "theo"]

# Everyone who is ever tagged as speaking in prose. This list started as the
# seven chat names plus the parents, which made every other character invisible
# to the measurement: Priya speaks in ten scenes across six chapters and scored
# nothing at all. Titles are matched separately so "Mrs. Aldana says" is found.
SPEAKERS = ["Chloe", "Ruth", "Sam", "Kavi", "Nadia", "Eli", "Theo", "Odile", "Priya",
            "Fen", "Owen", "Kayleigh", "Bryce", "Marisol",
            "Aldana", "Vance", "Prahl", "Baptiste", "Bell", "Hearn", "Kowalczyk",
            "Doyle", "Pruitt", "Sinclair", "Amberg", "Sandoval",
            "Prentice", "Ammons", "Whitaker", "Deb", "Ruiz"]
TAGGED = r"(?:Mrs\.? |Mr\.? |Ms\.? |Dr\.? |Coach |Sergeant )?(%s|her mom|her mother|her dad|her father)" % "|".join(SPEAKERS)
# The bias is directional, not just short, and one character sheet was built on
# the wrong end of it. In a two-hander the tag usually rides on the ANSWER, so
# the question is the turn most likely to go untagged. Counting questions off
# tagged lines therefore undercounts them systematically. RUTH.md carried
# "0% questions, never a question mark" as a law of the character on that
# basis; reading her turns by hand gives 41 questions in 191 turns, about 21%,
# with 16 spoken turns ending in a question mark. Never take a zero from this
# script as a fact about a person. Compare speakers with each other, and read
# the scenes before writing any rule into a sheet.

SAYS = r"(?:says|said|asks|asked|tells|told|shouts|screams|whispers)"
HEDGE = r"\b(i think|maybe|probably|idk|i dont know|i don't know|kind of|sort of|i guess)\b"


def words(t):
    return re.findall(r"[A-Za-z][A-Za-z']*", t)


def normalise(t):
    return t.replace('“', '"').replace('”', '"')


def chat_sources(root):
    """Every file the chat is actually written in."""
    files = sorted((root / "chapters").glob("*.md"))
    files += [root / n for n in CHAT_FILES if (root / n).is_file()]
    return files


def collect_chat(root):
    out = collections.defaultdict(list)
    for f in chat_sources(root):
        if not f.is_file():
            continue
        for line in f.read_text(encoding="utf-8").split("\n"):
            m = re.match(r"^\s*(%s):\s*(.+)$" % "|".join(CAST), line.strip(), re.I)
            if m:
                out[m.group(1).lower()].append(m.group(2).strip())
    return out


def collect_prose(root):
    """Lines with an explicit speaker tag. See the module docstring on bias."""
    out = collections.defaultdict(list)
    files = sorted((root / "chapters").glob("*.md")) + [root / n for n in CHAT_FILES]
    # The book attributes far more often with an action beat than with a speech
    # verb: '"Chloe." Mrs. Aldana is standing at the end of her desk.' Matching
    # only "<name> says" missed most of the cast, so the first pattern accepts
    # any sentence that opens with the speaker's name straight after a quote.
    patterns = [
        (rf'"([^"]+)"[,.!?]?\s+{TAGGED}\b', 0, 1),
        (rf'"([^"]+)"[,]?\s+{SAYS}\s+{TAGGED}', 0, 1),
        (rf'{TAGGED}\s+{SAYS}[,:]?\s+"([^"]+)"', 1, 0),
    ]
    for f in files:
        if not f.is_file():
            continue
        for line in normalise(f.read_text(encoding="utf-8")).split("\n"):
            for pat, ti, wi in patterns:
                for m in re.finditer(pat, line):
                    g = m.groups()
                    who = g[wi].lower().replace("her ", "")
                    who = {"mother": "mom", "father": "dad"}.get(who, who)
                    out[who].append(g[ti])
    return out


def profile(lines):
    tokens = [words(x) for x in lines]
    flat = [w.lower() for t in tokens for w in t]
    if not flat:
        return None
    return {
        "n": len(lines),
        "words": len(flat),
        "wpl": len(flat) / len(lines),
        "ttr": 100 * len(set(flat)) / len(flat),
        "q": 100 * sum(1 for x in lines if "?" in x) / len(lines),
        "short": 100 * sum(1 for t in tokens if len(t) <= 3) / len(lines),
        "long": 100 * sum(1 for t in tokens if len(t) > 15) / len(lines),
        "hedge": 100 * sum(1 for x in lines if re.search(HEDGE, x, re.I)) / len(lines),
    }


def show(title, data, floor, note):
    rows = {k: p for k, v in data.items() if (p := profile(v)) and p["n"] >= floor}
    if not rows:
        print(f"\n{title}: nothing with at least {floor} lines")
        return
    print(f"\n{title}   ({note})")
    print(f"  {'speaker':10}{'lines':>7}{'words':>7}{'w/line':>8}{'TTR%':>7}"
          f"{'quest%':>8}{'1-3w%':>7}{'>15w%':>7}{'hedge%':>8}")
    for k, p in sorted(rows.items(), key=lambda kv: -kv[1]["wpl"]):
        print(f"  {k:10}{p['n']:>7}{p['words']:>7}{p['wpl']:>8.1f}{p['ttr']:>7.1f}"
              f"{p['q']:>8.0f}{p['short']:>7.0f}{p['long']:>7.0f}{p['hedge']:>8.0f}")
    for key, label in (("wpl", "words per line"), ("short", "1-3 word share")):
        vals = [p[key] for p in rows.values()]
        lo, hi = min(vals), max(vals)
        spread = (hi - lo) / st.fmean(vals)
        print(f"  {label:16} {lo:.1f} to {hi:.1f}   "
              f"spread {100 * spread:.0f}% of the mean"
              f"{'   <- speakers barely differ' if spread < 0.5 else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=HERE)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--prose", action="store_true")
    ap.add_argument("--min-lines", type=int, default=8)
    a = ap.parse_args()
    both = not (a.chat or a.prose)
    if a.chat or both:
        show("GROUP CHAT", collect_chat(a.root), a.min_lines,
             "exact attribution, every message counted")
    if a.prose or both:
        show("PROSE DIALOGUE", collect_prose(a.root), a.min_lines,
             "tagged lines only, biased short AND against questions - never read a 0 as real")


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
