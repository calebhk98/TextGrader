#!/usr/bin/env python3
"""Chapters that have drifted more formal than the rest of the book.

The count is words of nine letters or more as a share of all words. It is a
crude proxy for register, useful when a manuscript's formality arrives as
vocabulary rather than as syntax: a plain observation puts on a long coat.

The judgement, when one is configured, is against the book's own median
rather than against a corpus, which is the point for this measure: the
question is not whether the prose is ornate for a novel, it is whether a
chapter reads as though a different person wrote it.

Every number below is descriptive until a policy is configured. With no
``project_measures.register.ratio`` set there is no basis to fail a chapter
for its distance from the book's own median, so this prints the table and
exits 0.

    python3 register.py                    report, configured chapters dir
    python3 register.py chapters/*.md      report, explicit files
    python3 register.py --words N          the long words in the worst chapters
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import project_config
from textgrader.chapters import chapter_number

WORD = re.compile(r"[A-Za-z']+")
DEFAULT_LONG_WORD_LETTERS = 9


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


def profile(path, long_word_letters):
    text = re.sub(r"^#.*$", "", Path(path).read_text(encoding="utf-8"), flags=re.M)
    tokens = WORD.findall(text)
    if not tokens:
        return 0.0, [], 0
    long_words = [word for word in tokens if len(word) >= long_word_letters]
    return len(long_words) / len(tokens) * 100, long_words, len(tokens)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories; default: the configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--words", type=int, metavar="N",
                    help="print the N commonest long words in each flagged chapter")
    args = parser.parse_args(argv)

    config = project_config.load_config(args.config)
    settings = project_config.measure_settings("register", config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    long_word_letters = settings.get("long_word_letters", DEFAULT_LONG_WORD_LETTERS)
    ratio = settings.get("ratio")
    exempt = settings.get("exempt", {}) if isinstance(settings.get("exempt", {}), dict) else {}

    files = resolve_paths(args.paths, chapters_dir)
    if not files:
        print("  no chapters found")
        return 0

    rows = []
    for path in files:
        pct, longs, word_count = profile(path, long_word_letters)
        rows.append((path.stem, pct, longs, word_count))

    median = statistics.median(row[1] for row in rows)
    # When the overall median is zero, compare against the typical non-zero
    # chapter instead of making every occurrence an automatic outlier.
    positive = [row[1] for row in rows if row[1] > 0]
    baseline = median if median > 0 else (statistics.median(positive) if positive else 0.0)

    print(f"  long words are {long_word_letters} letters or more, as a share of all words")
    print(f"  book median {median:.2f}%")

    if ratio is None:
        print("  project_measures.register.ratio is not set, so no chapter is flagged as "
              "formal-drift; the table below is descriptive only.\n")
        for stem, pct, _, _ in rows:
            print(f"  {stem[:24]:26s}{pct:6.2f}%")
        if args.words:
            from collections import Counter
            for stem, pct, longs, _ in rows:
                if longs:
                    common = Counter(word.lower() for word in longs).most_common(args.words)
                    print(f"\n  {stem}: " + ", ".join(f"{word} ({count})" for word, count in common))
        return 0

    ceiling = baseline * ratio
    over = [row for row in rows if row[1] > ceiling and row[0] not in exempt]
    excused = [row for row in rows if row[1] > ceiling and row[0] in exempt]

    print(f"  flag above {ceiling:.2f}% ({ratio} times the median)\n")
    for stem, pct, _, _ in rows:
        if pct > ceiling and stem in exempt:
            mark = f"  <-- over, and excused: {exempt[stem]}"
        elif pct > ceiling:
            mark = "  <-- formal for this book"
        else:
            mark = ""
        print(f"  {stem[:24]:26s}{pct:6.2f}%{mark}")

    if args.words:
        from collections import Counter
        for stem, pct, longs, _ in over:
            common = Counter(word.lower() for word in longs).most_common(args.words)
            print(f"\n  {stem}: " + ", ".join(f"{word} ({count})" for word, count in common))

    if over:
        names = ", ".join(f"{chapter_number(row[0])} ({row[1]:.2f}%)" for row in over)
        # One verdict line, matching how the scorecard reads every other
        # measure. The per-chapter rows above are the detail to fix from.
        print(f"\n  FAIL  {len(over)} chapter(s) over: {names}")
        print("  A chapter here reads as though a different person wrote it.\n")
        return 1
    if excused:
        print(f"\n  {len(excused)} over the line and excused by name in "
              f"project_measures.register.exempt, with the reason on the row above.")
    print(f"\n  none over {ceiling:.2f}% unexcused: pass\n")
    return 0


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
    sys.exit(main() or 0)
