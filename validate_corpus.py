#!/usr/bin/env python3
"""Hold out one corpus text, rebuild the reference from the rest, grade it.

This is the calibration check the tool could not previously make of itself.
Every book in a reference corpus belongs to the population that corpus
describes, so grading one against the others should flag almost nothing.  What
it flags anyway is a false positive, and counting them per metric across the
whole corpus says which measurements are trustworthy and which are noise.

    python3 validate_corpus.py corpus/books
    python3 validate_corpus.py corpus/books --enable-family sentence_rhythm
    python3 validate_corpus.py corpus/books --json-out validation.json

For each text: drop it, build a profile from the remaining ones, grade the
dropped text against that profile, and record every metric that came back as a
corpus outlier.  Then aggregate.

Reading the result:

``false positive rate``
    the share of the corpus a metric flags.  A metric using a modified-z
    threshold of 3.5 on a roughly normal distribution should flag a few per
    cent.  Much above that and the metric's distribution is not shaped the way
    the threshold assumes, which is a fact about the metric, not the books.
``never flagged``
    either well behaved or not discriminating at all.  Read it with the
    metric's spread: a measurement identical in every book cannot flag
    anything, and cannot tell you anything either.
``severity``
    how far outside the metric put the book.  A metric that flags rarely but at
    enormous severity is behaving differently from one that flags often at 3.6.

A high rate is not automatically a defect.  A corpus of genuinely varied books
SHOULD contain outliers, and a metric that finds them is working.  What this
separates is "this metric flags a few unusual books" from "this metric flags
half the corpus", and only the second is a calibration problem.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import grade
from textgrader.corpus import build_profile, write_profile
from textgrader.metrics import REGISTRY
from textgrader.project import load_config
from textgrader.results import Action, StatusType


def holdout_profile(profile: dict, source_id: str) -> dict:
    """The profile that building from every text except ``source_id`` gives.

    Distributions are the pooled per-text values, so removing a text is exactly
    removing its value from each list: there is no need to re-measure the other
    twenty-nine books once per hold-out, which would make this quadratic in
    parsing rather than in arithmetic.  The identity is worth stating because it
    is what makes the check affordable, and it holds only for distributions
    built this way.
    """

    kept = [book for book in profile["books"] if book["source_id"] != source_id]
    dropped = next(book for book in profile["books"] if book["source_id"] == source_id)
    index = profile["books"].index(dropped)
    out = dict(profile)
    out["books"] = kept
    out["book_count"] = len(kept)
    out["corpus_name"] = f"{profile.get('corpus_name', 'corpus')} minus {source_id}"

    distributions = {}
    for key, entry in profile["distributions"].items():
        values = [book[key] for book in kept
                  if isinstance(book.get(key), (int, float))
                  and not isinstance(book.get(key), bool)]
        if not values and isinstance(entry, dict) and entry.get("values"):
            # An alias distribution whose per-book column has another name.
            continue
        from textgrader.stats import summarize
        summary = summarize(values)
        summary["values"] = sorted(values)
        distributions[key] = summary
    out["distributions"] = distributions

    features = profile.get("feature_profiles", {})
    out["feature_profiles"] = {
        name: [row for position, row in enumerate(rows) if position != index]
        for name, rows in features.items()}
    return out


def grade_holdout(text_path: Path, profile: dict, config: dict, workspace: Path) -> dict:
    """Grade one text against a profile that does not contain it."""

    profile_path = workspace / "holdout.json"
    write_profile(profile, profile_path)
    run_config = {**config, "_config_dir": str(workspace),
                  "_config_path": str(workspace / "config.json"),
                  "corpus_profile": "holdout.json"}
    report = grade.analyze(text_path, run_config)
    outliers, compared, errors = [], 0, []
    for item in report.results:
        if item.status_type is StatusType.INTERNAL_ERROR:
            errors.append({"metric_id": item.metric_id, "error": item.error})
        if item.corpus and item.corpus.get("outlier") is not None:
            compared += 1
        if item.status_type is StatusType.CORPUS_OUTLIER:
            outliers.append({
                "metric_id": item.metric_id, "family": item.family,
                "value": item.value, "direction": item.direction,
                "severity": item.severity,
                "corpus_median": (item.corpus or {}).get("corpus_median"),
                "method": (item.corpus or {}).get("method"),
            })
    outliers.sort(key=lambda row: -(row["severity"] or 0))
    return {"compared": compared, "outliers": outliers, "errors": errors,
            "words": (report.document or {}).get("analyzed_words")}


def validate(sources, config, *, limit=None, quiet=False, parse_metrics=False,
             model_metrics=False):
    started = time.monotonic()
    if not quiet:
        print("measuring every text once to build the full reference ...")
    full = build_profile(sources, corpus_name="validation corpus",
                         text_processing=config.get("text_processing"),
                         nlp=config.get("nlp"), metrics=config.get("metrics", {}),
                         comparison_unit="book", include_parse_metrics=parse_metrics,
                         include_model_metrics=model_metrics,
                         progress=None if quiet else
                         (lambda name, book: print(f"  {name} ({book['word_count']:,} words)")))
    books = full["books"][:limit] if limit else full["books"]
    if len(full["books"]) < 3:
        raise SystemExit("error: leave-one-out needs at least 3 texts")

    rows = []
    with tempfile.TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        for position, book in enumerate(books, 1):
            path = Path(book["source_absolute_path"]) if "source_absolute_path" in book else None
            if path is None or not path.is_file():
                path = _locate(sources, book)
            if path is None:
                continue
            outcome = grade_holdout(path, holdout_profile(full, book["source_id"]),
                                    config, workspace)
            rows.append({"source_id": book["source_id"],
                         "name": book["source_path"], **outcome})
            if not quiet:
                print(f"  [{position}/{len(books)}] {book['source_path'][:44]:<46}"
                      f"{outcome['compared']:>4} compared "
                      f"{len(outcome['outliers']):>3} flagged")
    return {"corpus_name": full["corpus_name"], "book_count": len(full["books"]),
            "held_out": len(rows), "seconds": round(time.monotonic() - started, 1),
            "books": rows, "metrics": summarize_metrics(rows, len(rows)),
            "distribution_spread": spread(full)}


def _locate(sources, book):
    for source in sources:
        candidate = Path(source)
        if candidate.is_file() and candidate.name == book["source_filename"]:
            return candidate
        if candidate.is_dir():
            found = candidate / book["source_path"]
            if found.is_file():
                return found
    return None


def summarize_metrics(rows, total):
    """Per-metric false-positive rate across the corpus."""

    flags = Counter()
    severities = defaultdict(list)
    directions = defaultdict(Counter)
    families = {}
    for row in rows:
        for item in row["outliers"]:
            flags[item["metric_id"]] += 1
            if item["severity"] is not None:
                severities[item["metric_id"]].append(item["severity"])
            directions[item["metric_id"]][item["direction"]] += 1
            families[item["metric_id"]] = item["family"]
    out = []
    for metric_id, count in flags.most_common():
        values = severities[metric_id]
        out.append({
            "metric_id": metric_id, "family": families.get(metric_id),
            "flagged": count, "of": total, "rate": count / total if total else None,
            "median_severity": statistics.median(values) if values else None,
            "max_severity": max(values) if values else None,
            "directions": dict(directions[metric_id]),
        })
    return out


def spread(profile):
    """How much each measured quantity actually varies across the corpus.

    A metric that never flags anything may be well behaved or may be constant.
    The coefficient of variation separates the two.
    """

    out = {}
    for key, entry in profile["distributions"].items():
        if not isinstance(entry, dict) or not entry.get("count"):
            continue
        out[key] = {"median": entry.get("median"), "cv": entry.get("cv"),
                    "count": entry.get("count")}
    return out


def render(result, top=25):
    print("\n" + "=" * 78)
    print(f"{result['held_out']} of {result['book_count']} texts held out, "
          f"{result['seconds']}s")
    per_book = [len(row["outliers"]) for row in result["books"]]
    compared = [row["compared"] for row in result["books"]]
    if per_book:
        print(f"metrics compared per text: median {statistics.median(compared):.0f}")
        print(f"false positives per text:  median {statistics.median(per_book):.0f}, "
              f"mean {statistics.fmean(per_book):.1f}, max {max(per_book)}")
        clean = sum(1 for count in per_book if count == 0)
        print(f"texts with no false positive at all: {clean} of {len(per_book)}")

    print("\nnoisiest metrics (a metric flagging its own population)")
    print(f"  {'metric':<44}{'flagged':>8}{'rate':>8}{'med sev':>9}  direction")
    for row in result["metrics"][:top]:
        direction = ", ".join(f"{key} {value}" for key, value in sorted(row["directions"].items()))
        severity = f"{row['median_severity']:.1f}" if row["median_severity"] is not None else "-"
        print(f"  {row['metric_id']:<44}{row['flagged']:>5}/{row['of']:<3}"
              f"{100 * row['rate']:>7.0f}%{severity:>9}  {direction}")
    if not result["metrics"]:
        print("  none: no metric flagged any held-out text")

    errors = [error for row in result["books"] for error in row["errors"]]
    if errors:
        print(f"\n{len(errors)} internal errors during validation:")
        for error in errors[:10]:
            print(f"  {error['metric_id']}: {error['error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", help="corpus .txt/.md files or directories")
    parser.add_argument("--config", default=None)
    parser.add_argument("--limit", type=int, help="hold out only the first N texts")
    parser.add_argument("--enable", action="append", metavar="METRIC")
    parser.add_argument("--enable-family", action="append", metavar="FAMILY")
    parser.add_argument("--enable-all", action="store_true")
    parser.add_argument("--parse-metrics", action="store_true",
                        help="profile and grade the spaCy metrics too (slow)")
    parser.add_argument("--model-metrics", action="store_true",
                        help="profile and grade the semantic metrics too (slow)")
    parser.add_argument("--json-out")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config = grade.apply_switches(config, args.enable, None, args.enable_family,
                                  args.enable_all)
    # The reports are about one project's house policy and have nothing to say
    # about a reference corpus.
    for name in grade.BUNDLED_MEASURES:
        config.setdefault("metrics", {})[name] = False
    config.setdefault("analysis", {})["comparison_unit"] = "book"

    result = validate([Path(item) for item in args.sources], config,
                      limit=args.limit, quiet=args.quiet,
                      parse_metrics=args.parse_metrics, model_metrics=args.model_metrics)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, default=str) + "\n",
                                       encoding="utf-8")
    render(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
