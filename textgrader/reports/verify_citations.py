#!/usr/bin/env python3
"""Check that quotations in the character sheets actually appear in the manuscript.

A sheet writer working from memory can produce a quotation that reads exactly
like the book, attach a real file and line number to it, and mark it [text].
That is worse than an honest gap: revision then proceeds toward a line that was
never written.

This finds them. For every quoted string in a character sheet that is long
enough to be a real citation - sitting on a line that also names a source
file, per ``project_measures.verify_citations.citation_patterns`` - it
searches the manuscript for the text. Anything not found is reported with
the sheet and line it came from.

ARGUMENT CONTRACT (read this before wiring up a caller):

    python3 verify_citations.py
    python3 verify_citations.py --min-words 4
    python3 verify_citations.py --sheets characters/SOME_CHARACTER.md
    python3 verify_citations.py --manuscript MANUSCRIPT.md

There is no bare positional argument. A caller checking one manuscript file's
citations must say so with ``--manuscript``; a caller checking specific
sheets must say so with ``--sheets``. The two used to share one ambiguous
positional list, and grade.py was handing the manuscript path to it as if it
were a sheet, which searched the manuscript for citations of itself instead
of checking any actual character sheet.

Matching is deliberately loose - case, curly quotes, and internal whitespace
are normalised - so a hit means the words really are in the book and a miss
is worth looking at by hand. Ellipses inside a quotation split it into
fragments, each checked separately.

Config (``project_measures.verify_citations``):

    "citation_patterns": [...],  # regexes; a line must match one to be read
                                  # as a citation. Default: "chapters/<file>.md"
                                  # -style paths, plus the configured
                                  # manuscript's own filename.
    "skip_files": [],             # sheet filenames to never scan
    "min_words": 5                # ignore quotations shorter than this
"""

import argparse
import re
import sys
from pathlib import Path

from . import solo_notice
from .. import project as project_config


def norm(text):
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'").replace("—", "-")
    return re.sub(r"\s+", " ", text).lower().strip()


def load_manuscript(chapters_dir, manuscript_path):
    sources = []
    if chapters_dir and chapters_dir.is_dir():
        sources.extend(sorted(chapters_dir.glob("*.md")))
    if manuscript_path:
        sources.append(manuscript_path)
    blob = [source_path.read_text(encoding="utf-8", errors="replace")
           for source_path in sources if source_path.is_file()]
    if not blob:
        sys.exit(f"error: no manuscript files found (chapters_dir={chapters_dir}, "
                 f"manuscript={manuscript_path})")
    return norm(" ".join(blob))


def default_citation_patterns(manuscript_path):
    patterns = [r"chapters/\w+\.md"]
    if manuscript_path:
        patterns.append(re.escape(manuscript_path.name))
    return patterns


def quotations(text, cites_pattern):
    """Yield (line_no, fragment) for quoted text sitting next to a citation."""
    for line_number, line in enumerate(text.split("\n"), 1):
        if not cites_pattern.search(line):
            continue
        for match in re.finditer(r'"([^"]{12,400})"|“([^”]{12,400})”', line):
            quotation = match.group(1) or match.group(2)
            quotation = re.sub(r"\*+|_+|`+", "", quotation)          # markdown emphasis inside quotes
            # An assembled exchange is several real quotes joined; check each.
            for part in re.split(r"'\s+[A-Z][a-z]+\s+(?:says|said)[,.]?\s+'|\"\s+\"", quotation):
                yield line_number, part


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheets", nargs="*", type=Path,
                    help="character-sheet files to check; default: every *.md "
                         "in the configured characters directory")
    parser.add_argument("--manuscript", type=Path,
                    help="the manuscript file to search for citations; default: "
                         "the configured manuscript, alongside the configured chapters directory")
    parser.add_argument("--root", type=Path, default=None,
                        help="project root; defaults to the configuration's directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--min-words", type=int, default=None,
                    help="ignore quotations shorter than this (default: configured, or 5)")
    args = parser.parse_args(argv)
    # The project root defaults to wherever the configuration lives, so a
    # --config elsewhere moves the whole project with it.
    if args.root is None:
        args.root = Path(project_config.load_config(getattr(args, "config", None))["_config_dir"])


    config = project_config.load_config(args.config)
    settings = project_config.measure_settings("verify_citations", config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    default_manuscript = project_config.project_path("manuscript", "MANUSCRIPT.md", config)
    manuscript_path = args.manuscript if args.manuscript else default_manuscript
    if manuscript_path and not manuscript_path.is_absolute():
        manuscript_path = args.root / manuscript_path

    skip_files = set(settings.get("skip_files", []))
    min_words = args.min_words if args.min_words is not None else settings.get("min_words", 5)
    patterns = settings.get("citation_patterns") or default_citation_patterns(manuscript_path)
    cites_pattern = re.compile("(" + "|".join(patterns) + ")")

    book = load_manuscript(chapters_dir, manuscript_path)
    characters_dir = project_config.project_path("characters_dir", "characters", config)
    files = args.sheets if args.sheets else (
        sorted(characters_dir.glob("*.md")) if characters_dir and characters_dir.is_dir() else [])
    files = [path if path.is_absolute() else args.root / path for path in files]

    checked = missing = 0
    bad = {}
    for sheet_path in files:
        if sheet_path.name in skip_files or not sheet_path.is_file():
            continue
        for line_no, quote in quotations(sheet_path.read_text(encoding="utf-8"), cites_pattern):
            # An ellipsis joins two separate fragments; check each on its own.
            for frag in re.split(r"\.\.\.|…", quote):
                frag = frag.strip(" ,.;:-’‘'\"")
                if len(frag.split()) < min_words:
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


# Run from grade.py, not on its own. Each script in measures/ reports one
# diagnostic; grade.py assembles enabled checks and is the interface that says
# whether a pass helped.

if __name__ == "__main__":
    solo_notice()
    sys.exit(main())
