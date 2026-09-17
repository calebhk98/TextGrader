#!/usr/bin/env python3
"""Command-line front end for the core prose measurements.

The measurements themselves are library code and live in
``textgrader/core_metrics.py``.  This file is the historical CLI, kept because
it is the only way to read the pre-2.0 ``data/prose_reference.json`` layout and
to print the per-file summary table a revision pass starts from.

    python -m textgrader.reports.prose_grade MANUSCRIPT.md
    python -m textgrader.reports.prose_grade chapters/*.md --brief
    python -m textgrader.reports.prose_grade draft.md --benchmark <source id>
    python -m textgrader.reports.prose_grade --list-benchmarks
    python -m textgrader.reports.prose_grade chapters/*.md --summary
    python -m textgrader.reports.prose_grade --build-reference DIR1 DIR2

``grade.py`` is the supported entry point and produces the structured report.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from ..core_metrics import CORE12, METRICS, MONITOR, REFERENCE, measure
from ..text import strip_gutenberg

__all__ = ["measure", "build_reference", "grade", "summary", "main"]

def build_reference(dirs, out):
    books = {}
    for directory in dirs:
        for text_path in sorted(Path(directory).rglob("*.txt")):
            if "stripped" in text_path.stem:
                continue
            got = measure(strip_gutenberg(text_path.read_text(encoding="utf-8", errors="replace")))
            if got:
                books[text_path.stem] = got
                print(f"  measured {text_path.stem} ({got['_words']:,} words)")
    out.write_text(json.dumps(books, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {len(books)} books to {out}")


def percentile(values, candidate):
    return 100 * sum(1 for value in values if value < candidate) / len(values) if values else None


def grade(path, ref, benchmark, brief):
    # Strip Gutenberg boilerplate here too, or a reference book graded against
    # itself scores differently from its own stored entry.
    text = strip_gutenberg(Path(path).read_text(encoding="utf-8"))
    got = measure(text)
    if not got:
        print(f"{Path(path).name}: too short to grade (needs 40+ sentences)")
        return None

    if not ref:
        print(f"{Path(path).name}: corpus comparison unavailable (empty reference)")
        return None
    bench = ref.get(benchmark) if benchmark else None
    losses, pcts, core, skipped = [], [], [], []
    lines = []
    for key, (label, higher) in METRICS.items():
        if got[key] is None:
            skipped.append(label)
            continue
        vals = [benchmark_value[key] for benchmark_value in ref.values()
                if benchmark_value.get(key) is not None]
        if not vals:
            skipped.append(label)
            continue
        metric_percentile = percentile(vals, got[key])
        if not higher:
            metric_percentile = 100 - metric_percentile
        pcts.append(metric_percentile)
        if key in CORE12:
            core.append(metric_percentile)
        # >= / <= so a book tied with the benchmark, including the benchmark
        # itself, is not scored as a loss.
        beat = None if bench is None or bench.get(key) is None else (
            got[key] >= bench[key] if higher else got[key] <= bench[key])
        # Distance from the benchmark in corpus standard deviations, so gaps on
        # measures with different units can be ranked against each other.
        standard_deviation = statistics.stdev(vals) if len(vals) > 1 else 0.0
        gap = ((bench[key] - got[key]) / standard_deviation * (1 if higher else -1)
               if bench is not None and standard_deviation else 0.0)
        if beat is False:
            losses.append((-gap, label, got[key], bench[key], gap))
        lines.append((metric_percentile, label, got[key], bench.get(key) if bench else None, beat))

    print("=" * 78)
    print(f"{Path(path).name}  |  {got['_words']:,} words  |  "
          f"benchmark: {benchmark or 'none'}  |  corpus: {len(ref)} books")
    metric_count = len(METRICS) - len(skipped)
    print(f"\nstyle percentile (descriptive median of {metric_count} measures): {statistics.median(pcts):.0f}"
          f"      lost to benchmark on {len(losses)} of {metric_count}")
    print(f"  on the original 12 measures: {statistics.median(core):.0f}")
    if got["_transcript"] >= 1:
        print(f"  {got['_transcript']:.0f}% of this chapter is chat transcript, "
              f"measured separately and excluded above")
    if not brief:
        print(f"\n  {'measure':46}{'this':>8}{'bench':>8}{'pct':>6}")
        for path, label, mine, theirs, beat in sorted(lines):
            theirs_text = f"{theirs:8.2f}" if theirs is not None else f"{'—':>8}"
            print(f"  {label:46}{mine:>8.2f}{theirs_text}{path:>5.0f}%  "
                  f"{'<-- differs' if beat is False else ''}")
    if skipped:
        print(f"  not measurable in a text this short: {', '.join(skipped)}")
    if not brief:
        print(f"\n  monitored, not graded (corpus low / median / high):")
        for key, label in MONITOR.items():
            vals = sorted(benchmark_value[key] for benchmark_value in ref.values() if key in benchmark_value)
            if not vals:
                continue
            band = f"{vals[0]:.2f} / {statistics.median(vals):.2f} / {vals[-1]:.2f}"
            over = "  above every book in the corpus" if got[key] > vals[-1] else ""
            print(f"  {label:46}{got[key]:>8.2f}   corpus {band}{over}")

    if losses:
        print(f"\n  fix first, by size of the gap to {benchmark} "
              f"(in corpus standard deviations):")
        for _, label, mine, theirs, gap in sorted(losses)[:4]:
            print(f"    {label:46}{mine:>8.2f} vs {theirs:>7.2f}   {gap:>4.1f} sd")
    return statistics.median(pcts), len(losses)


# Short column headings for the summary table, in print order.
SUMMARY_COLS = [
    ("_words", "words"), ("_paragraphs", "paras"), ("_sentences", "sents"),
    ("wps", "w/sent"), ("slcv", "sl CV"), ("wpp", "w/para"), ("spp", "s/para"),
    ("wlen", "w len"), ("long7", "7+ch"), ("sttr", "sTTR"), ("top100", "top100"),
    ("fk", "F-K"), ("ari", "ARI"), ("lexile", "Lexile"), ("commas", "commas"), ("subord", "subord"), ("relcl", "relcl"),
    ("simple", "simple"), ("u10", "u10"), ("b2035", "20-35"),
    ("shortruns", "runs"), ("front", "front"), ("and2", "and2"),
    ("andrate", "and%"), ("negative", "neg%"), ("_transcript", "chat%"),
]


def summary(paths, ref):
    """One row per file, every measure, plus the corpus for comparison.

    The per-file report answers "is this chapter mature". This answers "which
    measure is the book losing on, and in which chapters", which is the
    question a revision pass actually starts from.
    """
    rows = []
    for path in paths:
        # No sentence floor here. The floor exists so a percentile is not read
        # off four sentences; a descriptive row is still worth having.
        got = measure(strip_gutenberg(Path(path).read_text(encoding="utf-8")), floor=1)
        if got is None:
            print(f"  {Path(path).stem}: no sentences")
            continue
        rows.append((Path(path).stem, got))
    if not rows:
        return

    head = f"{'file':<22}" + "".join(f"{header:>8}" for _, header in SUMMARY_COLS)
    print(head)
    print("-" * len(head))
    for stem, got in rows:
        cells = []
        for key, _ in SUMMARY_COLS:
            value = got[key]
            cells.append("       -" if value is None else
                         f"{value:>8,}" if key == "_words" else
                         f"{value:>8.0f}" if key.startswith("_") else f"{value:>8.1f}")
        print(f"{stem[:21]:<22}" + "".join(cells))

    print("-" * len(head))
    for label, pick in (("book median", statistics.median), ("corpus median", None)):
        if pick:
            vals = {key: pick([grade_value[key] for _, grade_value in rows if grade_value[key] is not None] or [0])
                    for key, _ in SUMMARY_COLS}
        else:
            vals = {key: (statistics.median([benchmark[key] for benchmark in ref.values() if key in benchmark])
                        if any(key in benchmark for benchmark in ref.values()) else 0.0)
                    for key, _ in SUMMARY_COLS}
        cells = "".join(f"{vals[key]:>8,.0f}" if key.startswith("_") else
                        f"{vals[key]:>8.1f}" for key, _ in SUMMARY_COLS)
        print(f"{label:<22}" + cells)
    print("\ncorpus word and paragraph counts are whole books, so the first three "
          "columns\nonly compare like with like between chapters. A dash under sTTR "
          "means the\nchapter is under 1000 words, which is shorter than the sampling "
          "window.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--benchmark", help="optional named-book diagnostic")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--brief", action="store_true", help="summary line per file only")
    parser.add_argument("--summary", action="store_true",
                    help="one row per file with every measure, no grading")
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--build-reference", nargs="+", metavar="DIR")
    args = parser.parse_args()

    if args.build_reference:
        return build_reference(args.build_reference, args.reference)
    if not args.reference.is_file():
        sys.exit(f"error: no reference corpus at {args.reference}; "
                 f"rebuild with --build-reference DIR")
    ref = json.loads(args.reference.read_text(encoding="utf-8"))

    if args.list_benchmarks:
        print(f"{len(ref)} books in {args.reference.name}, by reading grade:")
        for name in sorted(ref, key=lambda k: ref[k]["fk"]):
            print(f"  {ref[name]['fk']:5.1f}  {name}")
        return
    if not args.paths:
        sys.exit("error: give one or more files to grade")
    if args.benchmark and args.benchmark not in ref:
        sys.exit(f"error: no book named {args.benchmark}; try --list-benchmarks")

    if args.summary:
        return summary(args.paths, ref)

    results = []
    for path in args.paths:
        got = grade(path, ref, args.benchmark, args.brief)
        if got:
            results.append((path.name, *got))
        print()
    if len(results) > 1:
        print("=" * 78)
        print(f"{'file':30}{'percentile':>12}{'losses':>12}")
        for name, pct, lost in sorted(results, key=lambda r: r[1]):
            print(f"{name[:29]:30}{pct:>11.0f}%{lost:>12}")


def solo_notice():
    """One line reminding a human that grade.py is the supported entry point."""

    import os

    from . import VIA_GRADE_ENV_VAR
    if os.environ.get(VIA_GRADE_ENV_VAR):
        return
    print("  [one report; use grade.py for the structured report]", file=sys.stderr)


if __name__ == "__main__":
    solo_notice()
    main()
