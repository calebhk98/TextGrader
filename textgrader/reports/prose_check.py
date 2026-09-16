#!/usr/bin/env python3
"""Flag configured banned constructions in character sheets or a manuscript.

Every construction this scans for is project prose policy, not something
this script should know on its own, so it comes entirely from
``project_measures.prose_check`` in the user's config. With nothing
configured there is nothing to flag, and the script says so and exits 0.

Config (``project_measures.prose_check``), all default to ``[]``/``{}``:

    "verdict_patterns":        [{"pattern": "...", "label": "verdict"}, ...]
    "negation_patterns":       [{"pattern": "...", "label": "..."}, ...]
    "book_reference_patterns": [{"pattern": "...", "label": "..."}, ...]
    "skip_files": ["_TEMPLATE.md", ...]

Each pattern entry may also be written as a two-element ``[pattern, label]``
list for brevity.

ARGUMENT CONTRACT (read this before wiring up a caller):

    python3 prose_check.py                       every *.md in the configured
                                                   characters directory
    python3 prose_check.py characters/RUTH.md     explicit character sheet(s)
    python3 prose_check.py --manuscript FILE.md   scan one manuscript/chapter
                                                   file's prose instead of any
                                                   character sheet
    python3 prose_check.py --rule 2               only the negation patterns

A positional argument is always a CHARACTER SHEET. Passing the manuscript
path positionally used to be silently accepted and scanned as if it were a
sheet, which is wrong: a manuscript chapter is prose, not a character
description, and grade.py was doing exactly that. A positional path that
resolves inside the configured chapters directory is now a hard error
instead, naming ``--manuscript`` as the fix. Callers that want to check the
manuscript's own prose (e.g. grade.py) must pass ``--manuscript PATH``.
"""

import argparse
import re
import sys
from pathlib import Path

from .. import project as project_config

# Quoted text is the manuscript's, or a character's, so the sheet's own prose
# rules do not reach it. Sam saying "This is the worst mistake of my life" is
# dialogue, not the sheet handing the reader a verdict. This guard, and the
# two below, are generic to the "verdict" rule shape rather than facts about
# any manuscript, so they stay fixed rather than moving into config.
QUOTED = re.compile(r'"[^"]*"|“[^”]*”')

# The verdict family only fires when the sheet is passing judgement. These
# lead-ins turn the same words into ordinary description: somebody telling
# her when she is wrong, a section asking what she is wrong about.
DESCRIBES = re.compile(
    r"\b(?:when|whether|if|tell(?:s|ing)?|told|admit(?:s|ting)?|about|knows?|"
    r"knew|say(?:s|ing)?|said|thinks?|believes?|assumes?|being|been|prove|"
    r"proved|turns? out)\s+(?:\w+\s+){0,3}$")

# A heading like "What she is wrong about" is the sheet's own structure, not
# a verdict, whatever pattern happens to fire on its opening clause.
SECTION = re.compile(r"^-?\s*What (?:he|she|they)(?:'s| is| are)? wrong about\b", re.I)


def _rule_list(raw):
    """Normalise configured rules to a list of (pattern, label) pairs."""
    if not isinstance(raw, list):
        return []
    rules = []
    for entry in raw:
        if isinstance(entry, dict) and "pattern" in entry:
            rules.append((entry["pattern"], entry.get("label", "rule")))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            rules.append((entry[0], entry[1]))
    return rules


def sentences(text):
    """Yield (line_no, sentence). Sheets are markdown, so strip the furniture."""
    for start, line in enumerate(text.split("\n"), 1):
        if line.lstrip().startswith(("|", "#", "```")):
            continue
        clean = re.sub(r"\*+|_+|`+", "", line)
        clean = QUOTED.sub(" ", clean)
        for sentence in re.split(r"(?<=[.!?])\s+", clean):
            if sentence.strip():
                yield start, sentence.strip()


def scan(path, rules):
    hits = []
    for line_no, sentence in sentences(path.read_text(encoding="utf-8")):
        for pattern, label in rules:
            match = re.search(pattern, sentence, re.I)
            if match and label == "verdict" and DESCRIBES.search(sentence[:match.start()]):
                continue
            if match and label == "verdict" and SECTION.match(sentence):
                continue
            if match:
                hits.append((line_no, label, match.group(0), sentence))
                break
    return hits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sheets", nargs="*", type=Path,
                    help="character-sheet files (never the manuscript; use --manuscript for that)")
    parser.add_argument("--manuscript", type=Path,
                    help="scan this one manuscript/chapter file's prose instead of a character sheet")
    parser.add_argument("--root", type=Path, default=None,
                        help="project root; defaults to the configuration's directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--rule", type=int, choices=(1, 2, 3), help="check only this rule")
    parser.add_argument("--show", action="store_true", help="print the whole sentence")
    args = parser.parse_args(argv)
    # The project root defaults to wherever the configuration lives, so a
    # --config elsewhere moves the whole project with it.
    if args.root is None:
        args.root = Path(project_config.load_config(getattr(args, "config", None))["_config_dir"])


    config = project_config.load_config(args.config)
    settings = project_config.measure_settings("prose_check", config)
    characters_dir = project_config.project_path("characters_dir", "characters", config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    skip_files = set(settings.get("skip_files", []))

    verdict = _rule_list(settings.get("verdict_patterns", []))
    negation = _rule_list(settings.get("negation_patterns", []))
    book_reference = _rule_list(settings.get("book_reference_patterns", []))

    if not (verdict or negation or book_reference):
        print("no project_measures.prose_check patterns configured "
              "(verdict_patterns, negation_patterns, book_reference_patterns); nothing to flag.")
        return 0

    rules = verdict + negation + book_reference
    if args.rule == 1:
        rules = verdict
    elif args.rule == 2:
        rules = negation
    elif args.rule == 3:
        rules = book_reference

    files = []
    if args.manuscript:
        files = [args.manuscript if args.manuscript.is_absolute() else args.root / args.manuscript]
    elif args.sheets:
        for sheet_path in args.sheets:
            resolved = sheet_path if sheet_path.is_absolute() else args.root / sheet_path
            if chapters_dir and chapters_dir.is_dir() and resolved.resolve().is_relative_to(chapters_dir.resolve()):
                sys.exit(f"error: {sheet_path} is a manuscript chapter, not a character sheet; "
                         f"pass it with --manuscript instead")
            files.append(resolved)
    else:
        files = sorted(characters_dir.glob("*.md")) if characters_dir and characters_dir.is_dir() else []

    total = 0
    for sheet_path in files:
        if sheet_path.name in skip_files or not sheet_path.is_file():
            continue
        hits = scan(sheet_path, rules)
        if not hits:
            continue
        total += len(hits)
        print(f"\n{sheet_path.name}")
        for line_no, label, found, sentence in hits:
            print(f"  {line_no:>4} [{label}] {found}")
            if args.show:
                print(f"       {sentence[:150]}")

    print(f"\n{total} flagged across {len(files)} file(s).")
    return 1 if total else 0


# Run from grade.py, not on its own. Each script in measures/ reports one
# diagnostic; grade.py assembles enabled checks and is the interface that says
# whether a pass helped.
def _solo_notice():
    import sys, os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [bundled diagnostic; use grade.py for the structured report]",
          file=sys.stderr)

if __name__ == "__main__":
    _solo_notice()
    sys.exit(main())
