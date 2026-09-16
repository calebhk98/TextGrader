#!/usr/bin/env python3
"""Check edited chapters against the invariants a line edit must not break.

Ten agents edited chapters/01..10 in place. Each fix is a judgement call and
cannot be checked mechanically, but the things that would quietly wreck a
chapter can be:

  - two-trailing-space hard line breaks, which the book no longer uses. Every
    chapter separates paragraphs with a blank line; chapters 1-6 were the last
    holdouts and are being converted. Any trailing-space line is now a defect.
  - the heading, the blank line, and the generated date line at the top
  - em dashes, banned everywhere including dialogue
  - curly quotes, in a manuscript written with straight ones
  - the chapter still being there at all

Compares each chapter against the same file at a git revision, so "how many
trailing-space lines were there before" is answered by the repository rather
than by a number typed into this script.

    python3 check_edits.py                     compare against HEAD
    python3 check_edits.py --since <rev>        compare against another revision
    python3 check_edits.py --chapters 01 02     only these
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from project_config import CHAPTERS_DIR
from textgrader.chapters import chapter_number
DATELINE = re.compile(r"^\*[A-Z][a-z]+ \d{4}(?:\s*[–-]\s*[A-Z][a-z]+ \d{4})?\*$")
WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def git_root(path):
    """Find the repository containing *path* without assuming its layout."""
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                            capture_output=True, text=True, cwd=path)
    return Path(result.stdout.strip()) if result.returncode == 0 else None


def at_revision(rev, path):
    repo = git_root(path.parent)
    if repo is None:
        return None
    try:
        relpath = path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None
    result = subprocess.run(["git", "show", f"{rev}:{relpath}"],
                            capture_output=True, text=True, cwd=repo)
    return result.stdout if result.returncode == 0 else None


def counts(text):
    lines = text.split("\n")
    return {
        "hard breaks": sum(1 for line in lines if line.endswith("  ") and line.strip()),
        "em dashes": text.count("—"),
        "curly quotes": len(re.findall(r"[“”‘’]", text)),
        "words": len(WORD.findall(text)),
        "lines": len(lines),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="HEAD", help="revision to compare against")
    parser.add_argument("--chapters", nargs="*",
                    help="number prefixes, e.g. 01 02 or 01,02")
    args = parser.parse_args()

    files = sorted(CHAPTERS_DIR.glob("*.md"))
    if args.chapters:
        # Both "--chapters 01 02" and "--chapters 01,02" have to work. The
        # comma form used to match nothing, print an empty table and report
        # "0 problem(s)", so every caller who wrote it got a silent pass. Two
        # agents and this script's own caller were misled by that before it
        # was found; an argument matching no chapter is now an error.
        want = {int(number.strip())
                for arg in args.chapters for number in arg.split(",") if number.strip()}
        files = [path for path in files if chapter_number(path) in want]
        missing = want - {chapter_number(path) for path in files}
        if missing or not files:
            sys.exit(f"no chapter matches: {', '.join(map(str, sorted(missing))) or '(none given)'}")

    problems = 0
    print(f"{'chapter':<24}{'words':>8}{'hard breaks':>14}{'em dash':>9}{'curly':>7}")
    for chapter_path in files:
        now = chapter_path.read_text(encoding="utf-8")
        old = at_revision(args.since, chapter_path)
        current_counts, old_counts = counts(now), counts(old) if old else None

        note = []
        lines = now.split("\n")
        if not lines[0].startswith("## "):
            note.append("no heading on line 1"); problems += 1
        if len(lines) < 3 or lines[1].strip() or not DATELINE.match(lines[2].strip()):
            note.append("date line is not on line 3"); problems += 1
        if current_counts["em dashes"]:
            note.append(f"{current_counts['em dashes']} em dash"); problems += 1
        if old_counts and current_counts["curly quotes"] > old_counts["curly quotes"]:
            note.append(f"curly quotes up {old_counts['curly quotes']}->{current_counts['curly quotes']}")
            problems += 1
        # The book is on one convention now: blank lines between paragraphs.
        if current_counts["hard breaks"]:
            note.append(f"{current_counts['hard breaks']} trailing-space line(s), convert to "
                        f"blank-line paragraphs"); problems += 1

        delta = f"{current_counts['words'] - old_counts['words']:+d}" if old_counts else "new"
        hard_break_change = f"{current_counts['hard breaks']}" + (f" (was {old_counts['hard breaks']})" if old_counts else "")
        print(f"{chapter_path.stem:<24}{current_counts['words']:>8}{hard_break_change:>14}{current_counts['em dashes']:>9}"
              f"{current_counts['curly quotes']:>7}   {delta:>6}  {'; '.join(note)}")

    print(f"\n{problems} problem(s).")
    return 1 if problems else 0


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
