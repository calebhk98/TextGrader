#!/usr/bin/env python3
"""Phrases a project has ruled out, checked against the chapters.

A ruling that lives only in a conversation gets lost. Someone decides a phrase
is out, says so, believes a check now exists, and finds the phrase still in
chapter 1 several passes later - because the check was never written, or
because the check that exists scans something else. This is the row in a
script that a ruling needs in order to survive.

Nothing is banned by default and there is nothing project-specific in this
file. Every pattern comes from ``project_rules.banned_phrases`` in the user's
own configuration, with the reason recorded beside it, and ``grade.py`` runs
it as a structured rule.

    python3 banned_phrases.py            report
    python3 banned_phrases.py --show N   print the full line for entry N
"""
import argparse, re, sys
from pathlib import Path

from . import solo_notice
from .. import project as project_config


def banned_from(config):
    """(pattern, what it is, why it is out), from project_rules.banned_phrases.

    Personal bans are disabled by default and supplied only through
    configuration; there is nothing manuscript-specific in this file itself.
    """
    constructions = config.get("project_rules", {}).get("banned_phrases", [])
    return [(item["pattern"], item["name"], item.get("reason", "Configured rule."))
           for item in constructions] if constructions else []


def scan(paths, banned):
    hits = []
    for chapter_path in paths:
        for line_number, line in enumerate(Path(chapter_path).read_text(encoding="utf-8").split("\n"), 1):
            for pattern, name, why in banned:
                for match in re.finditer(pattern, line, re.I):
                    hits.append((Path(chapter_path).stem, line_number, name, why, line, match.start()))
    return hits


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories; default: the configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--show", type=int)
    args = parser.parse_args()

    config = project_config.load_config(args.config)
    banned = banned_from(config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    paths = resolve_paths(args.paths, chapters_dir)
    if not banned:
        print("no project_rules.banned_phrases configured; nothing is banned, so nothing to scan for.")
        return 0
    hits = scan(paths, banned)

    if args.show is not None:
        if 1 <= args.show <= len(hits):
            stem, line_number, name, why, line, _ = hits[args.show - 1]
            print(f"{stem}:{line_number}\n\n{line}\n\n  [{name}] {why}")
        else:
            print(f"no entry {args.show}; there are {len(hits)}")
        return

    print(f"\n  {len(banned)} ruled-out phrases, {len(paths)} chapters\n")
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

if __name__ == "__main__":
    solo_notice()
    sys.exit(main() or 0)
