#!/usr/bin/env python3
"""Check that quotations in the character sheets actually appear in the manuscript.

A sheet writer working from memory can produce a quotation that reads exactly like
the book, attach a real file and line number to it, and mark it [text]. That is worse
than an honest gap: the author revises toward a line that was never written.

This finds them. For every quoted string in characters/*.md that is long enough to be
a real citation, it searches the manuscript for the text. Anything not found is
reported with the sheet and line it came from.

    python3 verify_citations.py
    python3 verify_citations.py --min-words 4
    python3 verify_citations.py characters/RUTH.md

Matching is deliberately loose - case, curly quotes, and internal whitespace are
normalised - so a hit means the words really are in the book and a miss is worth
looking at by hand. Ellipses inside a quotation split it into fragments, each checked
separately.
"""

import argparse
import re
import sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from project_config import CHARACTERS_DIR, CHAPTERS_DIR, MANUSCRIPT

# Quotes in these are about the sheets themselves, not about the manuscript.
SKIP_FILES = {"_TEMPLATE.md", "_DIFFERENTIATION.md", "_ALLOCATIONS.md",
              "CHARACTER_SHEETS.md"}


def norm(text):
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'").replace("—", "-")
    return re.sub(r"\s+", " ", text).lower().strip()


def load_manuscript(root):
    root = Path(root)
    chapters_dir = root / CHAPTERS_DIR.relative_to(HERE)
    manuscript = root / MANUSCRIPT.relative_to(HERE)
    sources = [*sorted(chapters_dir.glob("*.md")), manuscript]
    blob = []
    for source_path in sources:
        if source_path.is_file():
            blob.append(source_path.read_text(encoding="utf-8", errors="replace"))
    if not blob:
        sys.exit(f"error: no manuscript files under {root}")
    return norm(" ".join(blob))


# A quotation only counts as a citation if the sheet points at a manuscript
# location on the same line. Everything else is the sheet's own prose, a
# proposal, or a line lifted from a reference document, and flagging those
# buried the real problems in noise.
CITES = re.compile(r"(chapters/\w+\.md|CHAPTERS_\d+_\d+_v2\.md|MANUSCRIPT_FULL\.md)")


def quotations(text):
    """Yield (line_no, fragment) for quoted text sitting next to a citation."""
    for line_number, line in enumerate(text.split("\n"), 1):
        if not CITES.search(line):
            continue
        for match in re.finditer(r'"([^"]{12,400})"|“([^”]{12,400})”', line):
            quotation = match.group(1) or match.group(2)
            quotation = re.sub(r"\*+|_+|`+", "", quotation)          # markdown emphasis inside quotes
            # An assembled exchange is several real quotes joined; check each.
            for part in re.split(r"'\s+[A-Z][a-z]+\s+(?:says|said)[,.]?\s+'|\"\s+\"", quotation):
                yield line_number, part


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheets", nargs="*", type=Path)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--min-words", type=int, default=5,
                    help="ignore quotations shorter than this (default 5)")
    args = parser.parse_args()

    book = load_manuscript(args.root)
    files = args.sheets or sorted((args.root / CHARACTERS_DIR.relative_to(HERE)).glob("*.md"))
    files = [path if path.is_absolute() else args.root / path for path in files]

    checked = missing = 0
    bad = {}
    for sheet_path in files:
        if sheet_path.name in SKIP_FILES:
            continue
        for line_no, quote in quotations(sheet_path.read_text(encoding="utf-8")):
            # An ellipsis joins two separate fragments; check each on its own.
            for frag in re.split(r"\.\.\.|…", quote):
                frag = frag.strip(" ,.;:-\u2019\u2018'\"")
                if len(frag.split()) < args.min_words:
                    continue
                checked += 1
                if norm(frag) not in book:
                    missing += 1
                    bad.setdefault(sheet_path.name, []).append((line_no, frag))

    for name in sorted(bad):
        print(f"\n{name}")
        for line_no, frag in bad[name]:
            print(f"  line {line_no}: {frag[:150]}")

    print(f"\n{checked} quotations checked, {missing} not found in the manuscript.")
    if missing:
        print("A miss is not proof of fabrication - it may be a paraphrase, a proposal,")
        print("or a line quoted from a reference document. Each one needs a human look.")
    return 1 if missing else 0


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
