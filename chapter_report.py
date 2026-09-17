#!/usr/bin/env python3
"""Grade a manuscript chapter by chapter against a reference corpus.

A whole-book number tells a revising agent that something is wrong somewhere in
two hundred thousand words.  A per-chapter table tells it which chapter to open,
which is the difference between a diagnosis and a chore.

Each chapter is compared with the corpus, not with the rest of the book.  That
distinction matters: comparing chapters with each other finds the chapter unlike
its neighbours, and is blind to a book that is uniformly weak, which is exactly
the case worth catching.  The corpus is the outside reference.

The corpus does not have to be built from chapters.  Rates do not care how much
text produced them, and the distribution metrics compare populations of
sentences and paragraphs rather than per-document averages, so a shelf of whole
novels is a perfectly good yardstick for one chapter.  Only the raw size counts
are withheld across a mismatch, and those say how long a chapter is rather than
how it is written.

    python3 chapter_report.py chapters/ --corpus corpus.json
    python3 chapter_report.py book.md --corpus corpus.json      # split on headings
    python3 chapter_report.py chapters/ --corpus corpus.json --enable-all
    python3 chapter_report.py chapters/ --corpus corpus.json --json-out report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import grade
from textgrader.document import DocumentAnalysis, NlpSettings, TextProcessing
from textgrader.project import load_config
from textgrader.results import Action, StatusType


def chapter_sources(target: Path, min_words: int) -> list[tuple[str, Path | str]]:
    """``(name, path-or-text)`` per chapter, from a directory or one file."""

    if target.is_dir():
        found = sorted(path for pattern in ("*.md", "*.txt")
                       for path in target.glob(pattern))
        return [(path.stem, path) for path in found]
    document = DocumentAnalysis.from_path(target)
    sections = [(title or f"section {index + 1}", view)
                for index, (title, view) in enumerate(document.sections)
                if view.word_count >= min_words]
    if len(sections) < 2:
        raise SystemExit(f"error: {target} has no chapter headings to split on; "
                         f"pass a directory of chapter files instead")
    return [(title, view.raw) for title, view in sections]


def grade_chapter(name: str, source: Path | str, config: dict) -> dict:
    report = (grade.analyze(source, config) if isinstance(source, Path)
              else grade.analyze_text(source, config, name))
    flagged, compared = [], 0
    for item in report.results:
        if item.corpus and item.corpus.get("outlier") is not None:
            compared += 1
        if item.status_type is StatusType.CORPUS_OUTLIER:
            flagged.append({"metric_id": item.metric_id, "family": item.family,
                            "value": item.value, "unit": item.unit,
                            "direction": item.direction, "severity": item.severity,
                            "corpus_median": (item.corpus or {}).get("corpus_median")})
    flagged.sort(key=lambda row: -(row["severity"] or 0))
    document = report.document or {}
    return {"chapter": name, "words": document.get("analyzed_words"),
            "sentences": document.get("sentences"),
            "dialogue_share": document.get("dialogue_word_share"),
            "compared": compared, "flagged": len(flagged),
            "severity_total": round(sum(row["severity"] or 0 for row in flagged), 1),
            "outliers": flagged,
            "errors": [(item.metric_id, item.error) for item in report.results
                       if item.status_type is StatusType.INTERNAL_ERROR]}


def render(rows: list[dict], top: int = 12) -> None:
    if not rows:
        print("no chapters found")
        return
    counts = [row["flagged"] for row in rows]
    print(f"{len(rows)} chapters, {statistics.median(row['compared'] for row in rows):.0f} "
          f"metrics compared each")
    print(f"flags per chapter: median {statistics.median(counts):.0f}, "
          f"mean {statistics.fmean(counts):.1f}, worst {max(counts)}, "
          f"clean {sum(1 for value in counts if value == 0)}/{len(rows)}\n")

    ranked = sorted(rows, key=lambda row: (-row["flagged"], -row["severity_total"]))
    print(f"{'chapter':<34}{'words':>8}{'dlg%':>6}{'flags':>7}{'severity':>10}  worst finding")
    for row in ranked[:top]:
        worst = row["outliers"][0]["metric_id"] if row["outliers"] else "-"
        share = row["dialogue_share"]
        print(f"  {row['chapter'][:31]:<32}{row['words']:>8,}"
              f"{('-' if share is None else f'{share:.0f}'):>6}"
              f"{row['flagged']:>7}{row['severity_total']:>10}  {worst}")
    if len(ranked) > top:
        print(f"  ... {len(ranked) - top} more")
        best = ranked[-1]
        print(f"\ncleanest: {best['chapter']} ({best['flagged']} flags, "
              f"{best['words']:,} words)")

    everywhere = Counter(row["metric_id"] for chapter in rows for row in chapter["outliers"])
    print(f"\nmetrics flagged in the most chapters (a book-wide habit, not one chapter's)")
    for metric_id, count in everywhere.most_common(10):
        print(f"  {metric_id:<48}{count:>3}/{len(rows)} chapters")
    errors = [error for row in rows for error in row["errors"]]
    if errors:
        print(f"\n{len(errors)} internal errors; first: {errors[0]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="a directory of chapter files, or one manuscript")
    parser.add_argument("--corpus", required=True, help="corpus profile to compare against")
    parser.add_argument("--config", default=None)
    parser.add_argument("--enable", action="append", metavar="METRIC")
    parser.add_argument("--enable-family", action="append", metavar="FAMILY")
    parser.add_argument("--enable-all", action="store_true")
    parser.add_argument("--min-words", type=int, default=500,
                        help="skip sections shorter than this when splitting one file")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = grade.apply_switches(config, args.enable, None, args.enable_family,
                                  args.enable_all)
    for name in grade.BUNDLED_MEASURES:
        config.setdefault("metrics", {})[name] = False
    config["corpus_profile"] = str(Path(args.corpus).resolve())
    config["_config_dir"] = "/"
    config.setdefault("analysis", {})["comparison_unit"] = "chapter"

    rows = [grade_chapter(name, source, config)
            for name, source in chapter_sources(Path(args.target), args.min_words)]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1, default=str) + "\n",
                                       encoding="utf-8")
    render(rows, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
