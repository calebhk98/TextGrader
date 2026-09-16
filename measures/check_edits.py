#!/usr/bin/env python3
"""Check chapters against a project's own formatting invariants.

Nothing here is a formatting rule until it is configured under
``project_measures.check_edits``. Every check below defaults to off, so
running this with no configuration finds nothing to report and exits 0
rather than silently enforcing somebody else's manuscript policy.

Config (``project_measures.check_edits``), all optional:

    "require_heading_prefix": null,   # e.g. "## " to require a heading on line 1
    "dateline_pattern": null,         # a regex the dateline line must fully match
    "dateline_line": null,            # 1-based line number the dateline must sit on
    "forbid_em_dash": false,
    "quote_style": null,              # null | "straight" | "curly"
    "forbid_hard_line_breaks": false  # markdown two-trailing-space breaks

Compares each chapter against the same file at a git revision when a
revision comparison is possible, so "how many curly quotes were there
before" is answered by the repository rather than by a number typed into
this script.

    python3 check_edits.py                     the configured chapters dir, compared to HEAD
    python3 check_edits.py chapters/01.md       one explicit chapter
    python3 check_edits.py --since <rev>        compare against another revision
    python3 check_edits.py --chapters 01 02     only these (needs a chapter number in the filename)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import project_config
from textgrader.chapters import chapter_number

WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def resolve_paths(explicit, default_dir):
    if not explicit:
        return sorted(default_dir.glob("*.md")) if default_dir and default_dir.is_dir() else []
    paths = []
    for item in explicit:
        item = Path(item)
        if item.is_dir():
            paths.extend(sorted(item.glob("*.md")))
        elif item.is_file():
            paths.append(item)
    return paths


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories; default: the configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--since", default="HEAD", help="revision to compare against")
    parser.add_argument("--chapters", nargs="*",
                    help="number prefixes, e.g. 01 02 or 01,02")
    args = parser.parse_args(argv)

    config = project_config.load_config(args.config)
    settings = project_config.measure_settings("check_edits", config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)

    heading_prefix = settings.get("require_heading_prefix")
    dateline_pattern = settings.get("dateline_pattern")
    dateline_line = settings.get("dateline_line")
    forbid_em_dash = bool(settings.get("forbid_em_dash", False))
    quote_style = settings.get("quote_style")
    forbid_hard_line_breaks = bool(settings.get("forbid_hard_line_breaks", False))
    dateline_regex = re.compile(dateline_pattern) if dateline_pattern else None

    if not any([heading_prefix, dateline_pattern, forbid_em_dash, quote_style, forbid_hard_line_breaks]):
        print("no project_measures.check_edits policy is configured "
              "(require_heading_prefix, dateline_pattern, forbid_em_dash, quote_style, "
              "forbid_hard_line_breaks); nothing to enforce.")

    files = resolve_paths(args.paths, chapters_dir)
    if args.chapters:
        # Both "--chapters 01 02" and "--chapters 01,02" have to work.
        want = {int(number.strip())
                for arg in args.chapters for number in arg.split(",") if number.strip()}
        files = [path for path in files if chapter_number(path) in want]
        missing = want - {chapter_number(path) for path in files}
        if missing or not files:
            sys.exit(f"no chapter matches: {', '.join(map(str, sorted(missing))) or '(none given)'}")

    if not files:
        print("no chapters found")
        return 0

    problems = 0
    print(f"{'chapter':<24}{'words':>8}{'hard breaks':>14}{'em dash':>9}{'curly':>7}")
    for chapter_path in files:
        now = chapter_path.read_text(encoding="utf-8")
        old = at_revision(args.since, chapter_path)
        current_counts, old_counts = counts(now), counts(old) if old else None

        note = []
        lines = now.split("\n")
        if heading_prefix and not lines[0].startswith(heading_prefix):
            note.append(f"no heading (expected to start with {heading_prefix!r}) on line 1")
            problems += 1
        if dateline_regex and dateline_line:
            index = dateline_line - 1
            # A dateline on line 3 implies a blank line 2 between it and the
            # heading; that was baked into the original hard-coded check, so
            # it stays true for any configured line rather than only line 3.
            blank_before = index == 0 or (index - 1 < len(lines) and not lines[index - 1].strip())
            on_line = index < len(lines) and dateline_regex.match(lines[index].strip())
            if not (blank_before and on_line):
                note.append(f"date line is not on line {dateline_line}")
                problems += 1
        if forbid_em_dash and current_counts["em dashes"]:
            note.append(f"{current_counts['em dashes']} em dash"); problems += 1
        if quote_style == "straight" and old_counts and current_counts["curly quotes"] > old_counts["curly quotes"]:
            note.append(f"curly quotes up {old_counts['curly quotes']}->{current_counts['curly quotes']}")
            problems += 1
        elif quote_style == "straight" and not old_counts and current_counts["curly quotes"]:
            note.append(f"{current_counts['curly quotes']} curly quote(s), quote_style is straight")
            problems += 1
        if forbid_hard_line_breaks and current_counts["hard breaks"]:
            note.append(f"{current_counts['hard breaks']} trailing-space line(s), convert to "
                        f"blank-line paragraphs"); problems += 1

        delta = f"{current_counts['words'] - old_counts['words']:+d}" if old_counts else "new"
        hard_break_change = f"{current_counts['hard breaks']}" + (f" (was {old_counts['hard breaks']})" if old_counts else "")
        print(f"{chapter_path.stem:<24}{current_counts['words']:>8}{hard_break_change:>14}{current_counts['em dashes']:>9}"
              f"{current_counts['curly quotes']:>7}   {delta:>6}  {'; '.join(note)}")

    print(f"\n{problems} problem(s).")
    return 1 if problems else 0


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
