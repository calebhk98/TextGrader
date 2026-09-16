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


def words(text):
    return re.findall(r"[A-Za-z][A-Za-z']*", text)


def normalise(text):
    return text.replace('“', '"').replace('”', '"')


def chat_sources(root):
    """Every file the chat is actually written in."""
    files = sorted((root / "chapters").glob("*.md"))
    files += [root / name for name in CHAT_FILES if (root / name).is_file()]
    return files


def collect_chat(root):
    out = collections.defaultdict(list)
    for source_path in chat_sources(root):
        if not source_path.is_file():
            continue
        for line in source_path.read_text(encoding="utf-8").split("\n"):
            match = re.match(r"^\s*(%s):\s*(.+)$" % "|".join(CAST), line.strip(), re.I)
            if match:
                out[match.group(1).lower()].append(match.group(2).strip())
    return out


def collect_prose(root):
    """Lines with an explicit speaker tag. See the module docstring on bias."""
    out = collections.defaultdict(list)
    files = sorted((root / "chapters").glob("*.md")) + [root / name for name in CHAT_FILES]
    # The book attributes far more often with an action beat than with a speech
    # verb: '"Chloe." Mrs. Aldana is standing at the end of her desk.' Matching
    # only "<name> says" missed most of the cast, so the first pattern accepts
    # any sentence that opens with the speaker's name straight after a quote.
    patterns = [
        (rf'"([^"]+)"[,.!?]?\s+{TAGGED}\b', 0, 1),
        (rf'"([^"]+)"[,]?\s+{SAYS}\s+{TAGGED}', 0, 1),
        (rf'{TAGGED}\s+{SAYS}[,:]?\s+"([^"]+)"', 1, 0),
    ]
    for source_path in files:
        if not source_path.is_file():
            continue
        for line in normalise(source_path.read_text(encoding="utf-8")).split("\n"):
            for pat, tag_index, words_index in patterns:
                for match in re.finditer(pat, line):
                    groups = match.groups()
                    who = groups[words_index].lower().replace("her ", "")
                    who = {"mother": "mom", "father": "dad"}.get(who, who)
                    out[who].append(groups[tag_index])
    return out


def profile(lines):
    tokens = [words(line) for line in lines]
    flat = [word.lower() for tokens in tokens for word in tokens]
    if not flat:
        return None
    # Mean segmental TTR compares equal-size samples rather than rewarding a
    # speaker merely for having fewer total words. Fifty words is deliberately
    # small enough for dialogue samples; incomplete trailing windows are kept.
    window = 50
    segments = [flat[index:index + window] for index in range(0, len(flat), window)]
    msttr = st.fmean(len(set(segment)) / len(segment) for segment in segments)
    return {
        "n": len(lines),
        "words": len(flat),
        "wpl": len(flat) / len(lines),
        "ttr": 100 * msttr,
        "q": 100 * sum(1 for line in lines if "?" in line) / len(lines),
        "short": 100 * sum(1 for tokens in tokens if len(tokens) <= 3) / len(lines),
        "long": 100 * sum(1 for tokens in tokens if len(tokens) > 15) / len(lines),
        "hedge": 100 * sum(1 for line in lines if re.search(HEDGE, line, re.I)) / len(lines),
    }


def show(title, data, floor, note):
    rows = {speaker: speaker_profile for speaker, lines in data.items() if (speaker_profile := profile(lines)) and speaker_profile["n"] >= floor}
    if not rows:
        print(f"\n{title}: nothing with at least {floor} lines")
        return
    print(f"\n{title}   ({note})")
    print(f"  {'speaker':10}{'lines':>7}{'words':>7}{'w/line':>8}{'TTR%':>7}"
          f"{'quest%':>8}{'1-3w%':>7}{'>15w%':>7}{'hedge%':>8}")
    for speaker, speaker_profile in sorted(rows.items(), key=lambda kv: -kv[1]["wpl"]):
        print(f"  {speaker:10}{speaker_profile['n']:>7}{speaker_profile['words']:>7}{speaker_profile['wpl']:>8.1f}{speaker_profile['ttr']:>7.1f}"
              f"{speaker_profile['q']:>8.0f}{speaker_profile['short']:>7.0f}{speaker_profile['long']:>7.0f}{speaker_profile['hedge']:>8.0f}")
    for key, label in (("wpl", "words per line"), ("short", "1-3 word share")):
        vals = [speaker_profile[key] for speaker_profile in rows.values()]
        minimum, maximum = min(vals), max(vals)
        mean = st.fmean(vals)
        spread = (maximum - minimum) / mean if mean else 0.0
        print(f"  {label:16} {minimum:.1f} to {maximum:.1f}   "
              f"spread {100 * spread:.0f}% of the mean"
              f"{'   <- speakers barely differ' if spread < 0.5 else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--prose", action="store_true")
    parser.add_argument("--min-lines", type=int, default=8)
    args = parser.parse_args()
    both = not (args.chat or args.prose)
    if args.chat or both:
        show("GROUP CHAT", collect_chat(args.root), args.min_lines,
             "exact attribution, every message counted")
    if args.prose or both:
        show("PROSE DIALOGUE", collect_prose(args.root), args.min_lines,
             "tagged lines only, biased short AND against questions - never read a 0 as real")


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
