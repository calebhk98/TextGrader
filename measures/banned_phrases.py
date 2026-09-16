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
sys.path.insert(0, str(HERE))
from project_config import BANNED_CONSTRUCTIONS, CHAPTERS_DIR

# (pattern, what it is, why it is out)
BANNED = []

# Personal bans are disabled by default and supplied only through configuration.
if BANNED_CONSTRUCTIONS is not None:
    BANNED = [
        (item["pattern"], item["name"], item.get("reason", "Configured rule."))
        for item in BANNED_CONSTRUCTIONS
    ]


def scan(paths):
    hits = []
    for chapter_path in paths:
        for line_number, line in enumerate(Path(chapter_path).read_text(encoding="utf-8").split("\n"), 1):
            for pattern, name, why in BANNED:
                for match in re.finditer(pattern, line, re.I):
                    hits.append((Path(chapter_path).stem, line_number, name, why, line, match.start()))
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int)
    args = parser.parse_args()

    paths = sorted(glob.glob(str(CHAPTERS_DIR / "*.md")))
    hits = scan(paths)

    if args.show is not None:
        if 1 <= args.show <= len(hits):
            stem, line_number, name, why, line, _ = hits[args.show - 1]
            print(f"{stem}:{line_number}\n\n{line}\n\n  [{name}] {why}")
        else:
            print(f"no entry {args.show}; there are {len(hits)}")
        return

    print(f"\n  {len(BANNED)} ruled-out phrases, {len(paths)} chapters\n")
    if not hits:
        print("  none present.\n")
        return 0

    for hit_number, (stem, line_number, name, why, line, col) in enumerate(hits, 1):
        sentence = max(0, col - 60)
        print(f"  {hit_number}. {stem}:{line_number}  [{name}]")
        print(f"     ...{line[sentence:col + 80]}...")
    print(f"\n  {len(hits)} present. Each one is a ruling already made.\n")
    return 1


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
    sys.exit(main() or 0)
