"""Build and load self-contained, reproducible reference-corpus profiles.

This module deliberately does not acquire books.  It accepts local UTF-8 text
files (or directories containing them) and emits portable JSON containing source
provenance and the distributions TextGrader uses at run time.

Three things recorded here exist because leaving them out produced silently
wrong comparisons:

``text_processing``
    exactly how the corpus text was cleaned and segmented.  A profile built
    without stripping Gutenberg boilerplate does not describe the same kind of
    document as a manuscript that had it stripped, and ``grade.py`` now says so.

``comparison_unit``
    whether each observation is a book, a chapter or a scene.  A chapter graded
    against a shelf of novels predictably "fails" document length, which is a
    fact about how books are divided, not about the chapter.

the parse-based metrics
    the builder used to skip everything in ``NLP_METRICS`` outright, so passive
    voice, tense, POS and the rest could be measured on a manuscript but could
    never be compared with anything.  They are profiled now, behind a flag,
    because they are slow rather than because they are unwanted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 2
PARSER_VERSION = "2"
METRIC_DEFINITION_VERSION = "2"

from .core_metrics import measure as core_measure
from .document import COMPARISON_UNITS, DocumentAnalysis, NlpSettings, TextProcessing
from .metrics import MODEL_METRICS, REGISTRY
from .stats import summarize

CORE_METRIC_KEYS = (
    "fk", "ari", "wps", "slcv", "wpp", "spp", "wlen", "long7", "sttr",
    "top100", "commas", "subord", "relcl", "simple", "u10", "b2035",
    "shortruns", "front", "and2", "andrate", "negative", "_words",
    "_sentences", "_paragraphs",
)

#: Metric ids that are counts rather than rates.  ``grade.py`` refuses to
#: compare these across different ``comparison_unit`` values.
SCALE_DEPENDENT = {"_words", "_sentences", "_paragraphs",
                   "word_count", "sentence_count", "paragraph_count"}


def _source_files(inputs: Iterable[str | Path]) -> list[tuple[int, Path, str]]:
    found: list[tuple[int, Path, str]] = []
    for input_index, item in enumerate(inputs):
        path = Path(item).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() not in (".txt", ".md"):
                raise ValueError(f"corpus source is not a .txt or .md file: {path}")
            found.append((input_index, path, path.name))
        elif path.is_dir():
            for pattern in ("*.txt", "*.md"):
                for source in path.rglob(pattern):
                    found.append((input_index, source, source.relative_to(path).as_posix()))
        else:
            raise FileNotFoundError(f"corpus input does not exist: {path}")
    return sorted(found, key=lambda row: (row[0], row[2].casefold(), row[2]))


def _manifest_entries(manifest: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not manifest:
        return {}
    entries = manifest.get("sources", manifest)
    if isinstance(entries, list):
        return {str(entry.get("filename", entry.get("path"))): entry for entry in entries}
    if isinstance(entries, dict):
        return entries
    raise ValueError("manifest 'sources' must be an object or list")


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    """One corpus distribution, stored as its values plus its whole shape.

    The raw values are kept because robust comparison needs the empirical
    distribution, not a summary: a median and a MAD cannot answer "how far
    outside the observed range is this", and MAD collapses to zero on discrete
    metrics where the quantiles still carry information.
    """

    ordered = sorted(value for value in values if value is not None)
    summary = summarize(ordered)
    summary["values"] = ordered
    summary["q1"], summary["q3"] = summary.get("p25"), summary.get("p75")
    return summary


def _timestamp(value: str | None) -> str:
    if value:
        return value
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime.now(timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


def _metric_names(include_parse: bool, include_model: bool) -> list[str]:
    """Which registered metrics to precompute for every corpus text.

    Everything cheap is profiled whether or not it is enabled for grading,
    because a distribution is only useful if it already exists on the day
    somebody turns a metric on.  Two groups are opt-in because of what they
    cost per book, not because they are unwanted: the spaCy metrics are tens of
    seconds each, and the semantic ones download and run an embedding model.
    """

    return [name for name, spec in REGISTRY.items()
            if (include_parse or not spec.needs_parse)
            and (include_model or not spec.needs_model)]


def build_profile(inputs: Iterable[str | Path], *, corpus_name: str = "local corpus",
                  manifest: Mapping[str, Any] | None = None, built_at: str | None = None,
                  preprocessing: Mapping[str, Any] | None = None,
                  text_processing: Mapping[str, Any] | None = None,
                  nlp: Mapping[str, Any] | None = None,
                  metrics: Mapping[str, Any] | None = None,
                  comparison_unit: str = "book",
                  include_core_metrics: bool = True,
                  include_parse_metrics: bool = False,
                  include_model_metrics: bool = False,
                  progress=None) -> dict[str, Any]:
    """Profile local text files without retaining or later requiring raw books."""

    files = _source_files(inputs)
    if not files:
        raise ValueError("corpus contains no .txt or .md files")
    if comparison_unit not in COMPARISON_UNITS:
        raise ValueError(f"comparison_unit must be one of {', '.join(COMPARISON_UNITS)}")
    entries = _manifest_entries(manifest)
    # ``preprocessing`` is the pre-2.0 name and is accepted so old callers keep
    # working; ``text_processing`` is what the runtime configuration calls it.
    settings = dict(preprocessing or {})
    settings.update(text_processing or {})
    processing = TextProcessing.from_config(settings)
    nlp_settings = NlpSettings.from_config(nlp)
    metric_settings = dict(metrics or {})
    wanted = _metric_names(include_parse_metrics, include_model_metrics)

    books: list[dict[str, Any]] = []
    used_ids: Counter[str] = Counter()
    frequency: Counter[str] = Counter()
    feature_profiles: dict[str, list[dict[str, float]]] = {"function_words": []}
    metric_errors: dict[str, str] = {}

    for _, path, relative_name in files:
        raw_bytes = path.read_bytes()
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corpus source is not UTF-8: {path}") from exc
        digest = hashlib.sha256(raw_bytes).hexdigest()
        analysis = DocumentAnalysis.from_text(raw, processing=processing,
                                              nlp_settings=nlp_settings,
                                              source=relative_name,
                                              comparison_unit=comparison_unit)
        item_meta = entries.get(relative_name, entries.get(path.name, {}))
        if not isinstance(item_meta, dict):
            raise ValueError(f"manifest entry for {relative_name!r} must be an object")
        base_id = str(item_meta.get("id") or f"{path.stem}-{digest[:12]}")
        used_ids[base_id] += 1
        source_id = base_id if used_ids[base_id] == 1 else f"{base_id}-{used_ids[base_id]}"
        frequency.update(analysis.tokens)
        book = {
            "source_id": source_id, "source_filename": path.name,
            "source_path": relative_name, "source_hash": f"sha256:{digest}",
            "word_count": analysis.word_count,
            "sentence_count": analysis.sentence_count,
            "paragraph_count": analysis.paragraph_count,
            "mean_sentence_words": (statistics.fmean(analysis.sentence_lengths)
                                    if analysis.sentence_lengths else None),
            "mean_paragraph_words": (statistics.fmean(analysis.paragraph_lengths)
                                     if analysis.paragraph_lengths else None),
            "mean_word_characters": (statistics.fmean(map(len, analysis.words))
                                     if analysis.words else None),
            "metadata": {key: value for key, value in item_meta.items()
                         if key not in {"id", "filename", "path"}},
        }
        if include_core_metrics:
            core = core_measure(analysis, floor=1)
            if core:
                book.update({key: core[key] for key in CORE_METRIC_KEYS
                             if core.get(key) is not None})
        for name in wanted:
            spec = REGISTRY[name]
            setting = metric_settings.get(name, {})
            options = dict(spec.defaults)
            if isinstance(setting, Mapping):
                options.update({key: value for key, value in setting.items() if key != "enabled"})
            try:
                module = importlib.import_module(f"textgrader.metrics.{spec.module}")
                findings = module.measure(analysis, config=options, profile=None)
            except Exception as exc:
                metric_errors[name] = f"{type(exc).__name__}: {exc}"
                continue
            for finding in findings or []:
                value = finding.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    book[finding["metric_id"]] = value
        function_words = importlib.import_module("textgrader.metrics.function_words")
        feature_profiles["function_words"].append(function_words.vector(analysis.text))
        books.append(book)
        if progress:
            progress(relative_name, book)

    base_keys = ("word_count", "sentence_count", "paragraph_count", "mean_sentence_words",
                 "mean_paragraph_words", "mean_word_characters")
    distributions = {key: _distribution([book.get(key) for book in books])
                     for key in base_keys}
    # Stable analysis IDs used by the runtime report.  Aliases keep the
    # provenance-friendly long names in each book while avoiding copied
    # thresholds or raw-book access during grading.
    distributions.update({
        "wps": distributions["mean_sentence_words"],
        "wpp": distributions["mean_paragraph_words"],
        "wlen": distributions["mean_word_characters"],
        "_words": distributions["word_count"],
        "_sentences": distributions["sentence_count"],
        "_paragraphs": distributions["paragraph_count"],
    })
    measured = sorted({key for book in books for key in book
                       if key not in base_keys and isinstance(book.get(key), (int, float))
                       and not isinstance(book.get(key), bool)})
    distributions.update({key: _distribution([book[key] for book in books if key in book])
                          for key in measured})
    effective = {name: {**REGISTRY[name].defaults,
                        **{key: value for key, value in
                           (metric_settings.get(name) or {}).items() if key != "enabled"}}
                 for name in wanted if isinstance(metric_settings.get(name, {}), Mapping)}
    return {
        "schema_version": SCHEMA_VERSION,
        "textgrader_version": "0.2.0",
        "parser_version": PARSER_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "corpus_name": corpus_name,
        "comparison_unit": comparison_unit,
        "build_timestamp": _timestamp(built_at),
        "text_processing": processing.fingerprint(),
        # Retained under the pre-2.0 name so older readers still find it.
        "preprocessing": processing.fingerprint(),
        "core_metrics": include_core_metrics,
        "parse_metrics": include_parse_metrics,
        "model_metrics": include_model_metrics,
        "metric_settings": effective,
        "metric_errors": metric_errors,
        "book_count": len(books), "books": books,
        "distributions": distributions,
        "word_frequency": {word: frequency[word] for word in sorted(frequency)},
        "word_frequency_total": sum(frequency.values()),
        "feature_profiles": feature_profiles,
    }


def write_profile(profile: Mapping[str, Any], destination: str | Path) -> None:
    Path(destination).write_text(
        json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")


def load_profile(path: str | Path) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") not in (1, SCHEMA_VERSION):
        raise ValueError(f"unsupported corpus profile schema: {profile.get('schema_version')!r}")
    if not isinstance(profile.get("books"), list) or not isinstance(profile.get("distributions"), dict):
        raise ValueError("invalid corpus profile")
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a TextGrader corpus profile from local text files")
    parser.add_argument("inputs", nargs="+",
                        help="local .txt/.md files or directories (searched recursively)")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--name", default="local corpus")
    parser.add_argument("--manifest", type=Path, help="optional JSON metadata manifest")
    parser.add_argument("--config", type=Path,
                        help="project config whose text_processing and metric options "
                             "must be reproduced")
    parser.add_argument("--comparison-unit", choices=COMPARISON_UNITS, default="book",
                        help="what one input file is: a whole book, a chapter, a scene")
    parser.add_argument("--no-core-metrics", action="store_true",
                        help="omit the default core prose distributions")
    parser.add_argument("--parse-metrics", action="store_true",
                        help="also profile the spaCy-parse metrics (tens of seconds per book)")
    parser.add_argument("--model-metrics", action="store_true",
                        help="also profile the semantic metrics, which download and run a "
                             "sentence-embedding model over every book")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}

    def progress(name, book):
        if not args.quiet:
            print(f"  measured {name} ({book['word_count']:,} words)")

    profile = build_profile(
        args.inputs, corpus_name=args.name, manifest=manifest,
        text_processing=config.get("text_processing"), nlp=config.get("nlp"),
        metrics=config.get("metrics", {}), comparison_unit=args.comparison_unit,
        include_core_metrics=not args.no_core_metrics,
        include_parse_metrics=args.parse_metrics,
        include_model_metrics=args.model_metrics, progress=progress)
    write_profile(profile, args.output)
    if not args.quiet:
        print(f"\nwrote {profile['book_count']} {args.comparison_unit}(s) to {args.output}")
        if profile["metric_errors"]:
            print("metrics that could not be profiled:")
            for name, reason in sorted(profile["metric_errors"].items()):
                print(f"  {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
