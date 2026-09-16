#!/usr/bin/env python3
"""Build the configured manuscript file from chapters/. One direction only.

chapters/ is the source of truth. Every edit is made there, one file per
chapter, so agents and people can work on different chapters without
colliding. This concatenates them into the single readable manuscript.

Nothing goes the other way any more. The script that re-cut chapters/ from a
manuscript file has been deleted, because running it discarded work: it reverts
every chapter to whatever the manuscript last held, and the manuscript is
always the stale copy.

    python3 build_manuscript.py            write the configured manuscript
    python3 build_manuscript.py --check    say what would change, write nothing

**Order comes from the filenames and nothing else.** chapters/NN_slug.md sorts
into reading order, so there is no list to keep updated and no ordering to
remember. To move a chapter, rename the files; the build follows.

The script checks, before writing, that the chapter numbers run in sequence
with no gaps or duplicates, and that each file's own heading matches the
descriptive slug in its filename (``NN_slug.md`` -> a heading reading
"Slug"). A chapter renamed without its heading being changed, or renumbered
without the run being closed up, is the mistake this catches.
"""

import argparse
import re
import sys
from pathlib import Path

from project_config import CHAPTERS_DIR, MANUSCRIPT
from textgrader.chapters import chapter_number

HERE = Path(__file__).resolve().parent
CHAPTERS = CHAPTERS_DIR
OUT = MANUSCRIPT

ONES = ("Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve "
        "Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen").split()
TENS = ("", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy",
        "Eighty", "Ninety")


def number_word(number):
    """Return a spelled cardinal number suitable for a chapter heading."""
    if not 0 < number < 1000:
        raise ValueError("chapter numbers must be between 1 and 999")
    if number < 20:
        return ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return TENS[tens] if not ones else f"{TENS[tens]}-{ONES[ones]}"
    hundreds, rest = divmod(number, 100)
    stem = f"{ONES[hundreds]} Hundred"
    return stem if not rest else f"{stem} {number_word(rest)}"


def chapters():
    """Markdown files in natural numeric order, with lexical fallbacks."""
    paths = list(CHAPTERS.glob("*.md"))
    paths.sort(key=lambda path: (chapter_number(path) is None,
                                chapter_number(path) or 0, path.name.casefold()))
    return [(chapter_number(path), path) for path in paths]


def slug_of(path):
    """The descriptive part of a chapter filename, after its leading number."""
    stem = Path(path).stem
    match = re.match(r"^\d+[_\-\s]*(.*)$", stem)
    slug = match.group(1) if match else stem
    return re.sub(r"[_\-]+", " ", slug).strip()


def _normalise(text):
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def heading_of(text):
    """The text of a level-1..6 markdown heading on the file's first line, or None."""
    first_line = text.split("\n", 1)[0]
    match = re.match(r"^#{1,6}\s+(.*)$", first_line.strip())
    return match.group(1).strip() if match else None


def check(found):
    """Reject ambiguous duplicate numbers, sequence gaps, and a heading that
    no longer names what its filename says it is.

    A chapter renamed without its heading being changed - or renumbered
    without the neighbouring files shifting to close the gap - is the mistake
    this exists to catch; see the module docstring.
    """
    problems = []
    numbers = [number for number, _ in found if number is not None]
    duplicates = sorted({number for number in numbers if numbers.count(number) > 1})
    if duplicates:
        problems.append("duplicate chapter number(s): " + ", ".join(map(str, duplicates)))

    if numbers:
        gaps = sorted(set(range(min(numbers), max(numbers) + 1)) - set(numbers))
        if gaps:
            problems.append("missing chapter number(s) in the sequence: " + ", ".join(map(str, gaps)))

    for _, path in found:
        text = path.read_text(encoding="utf-8")
        heading = heading_of(text)
        if heading is None:
            problems.append(f"{path.name}: no markdown heading on its first line")
            continue
        slug = slug_of(path)
        if slug and _normalise(heading) != _normalise(slug):
            problems.append(f"{path.name}: heading {heading!r} does not match "
                            f"the filename's {slug!r}")
    return problems


def build(found):
    return "\n\n\n".join(path.read_text(encoding="utf-8").strip()
                         for _, path in found) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                    help="report what would change, write nothing")
    args = parser.parse_args()

    found = chapters()
    if not found:
        sys.exit(f"no chapters found in {CHAPTERS}")

    problems = check(found)
    if problems:
        print("chapters/ is inconsistent:\n")
        for problem in problems:
            print(f"  {problem}")
        sys.exit("\nfix the filenames or the headings, then run again.")

    new = build(found)
    old = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    words = len(new.split())

    if new == old:
        print(f"{OUT.name} already current: {len(found)} chapters, {words:,} words")
        return
    if args.check:
        word_difference = words - len(old.split())
        print(f"{OUT.name} would change: {len(found)} chapters, "
              f"{words:,} words ({word_difference:+,})")
        return
    OUT.write_text(new, encoding="utf-8")
    print(f"wrote {OUT.name}: {len(found)} chapters, {words:,} words")


if __name__ == "__main__":
    main()
