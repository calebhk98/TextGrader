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
from .metrics import MODEL_METRICS, REGISTRY, is_enabled
from .stats import quantile_curve, summarize

CORE_METRIC_KEYS = (
    "fk", "ari", "wps", "slcv", "wpp", "spp", "wlen", "long7", "sttr",
    "top100", "commas", "subord", "relcl", "simple", "u10", "b2035",
    "shortruns", "front", "and2", "andrate", "negative", "_words",
    "_sentences", "_paragraphs",
)

#: Quantities measured once per sentence, paragraph, word or spoken turn rather
#: than once per text.  These are pooled across the whole corpus and stored as
#: one quantile curve each, so the yardstick is built from every sentence in
#: every book rather than from thirty per-book averages.  That is what lets a
#: chapter be compared with it: the unit of observation on both sides is the
#: sentence, not the document, so how the corpus happens to be divided into
#: files stops mattering.
def _item_values(analysis) -> dict[str, list[float]]:
    body = analysis.text
    sentences = analysis.sentences
    return {
        "sentence_words": [float(n) for n in analysis.sentence_lengths],
        "paragraph_words": [float(n) for n in analysis.paragraph_lengths],
        "paragraph_sentences": [float(n) for n in analysis.paragraph_sentence_counts],
        "word_characters": [float(len(word)) for word in analysis.words],
        "sentence_commas": [float(sentence.count(",")) for sentence in sentences],
        "turn_words": [float(len(part.split())) for part in analysis.turns],
    }


ITEM_SOURCES = ("sentence_words", "paragraph_words", "paragraph_sentences",
                "word_characters", "sentence_commas", "turn_words")

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


def _metric_names(include_parse: bool, include_model: bool,
                  metric_config: Mapping[str, Any] | None = None,
                  selection: str = "enabled") -> list[str]:
    """Which registered metrics to precompute for every corpus text.

    ``selection="enabled"`` (the default) profiles exactly the metrics switched
    on in the ``metrics`` config, because computing a distribution nobody asked
    for is work nobody asked for -- and the expensive suites are expensive
    enough that "profile everything cheap" stopped being cheap.  Pass
    ``selection="all"`` to precompute every registered metric regardless, which
    is what you want when building a reference profile to ship, so a
    distribution already exists on the day somebody turns a metric on.

    ``include_parse``/``include_model`` remain a separate axis: they say
    whether the spaCy and embedding-model metrics may run at all, and they gate
    both selections.
    """

    if selection == "auto":
        # No config supplied at all means the caller has not said what it
        # wants, so precompute everything; an explicit (possibly empty)
        # metrics mapping is a statement about what is wanted, so honour it.
        selection = "all" if metric_config is None else "enabled"
    if selection not in ("enabled", "all"):
        raise ValueError(
            f"unknown metric selection {selection!r}; use 'enabled', 'all' or 'auto'")
    return [name for name, spec in REGISTRY.items()
            if (include_parse or not spec.needs_parse)
            and (include_model or not spec.needs_model)
            and (selection == "all" or is_enabled(metric_config, name))]


def build_profile(inputs: Iterable[str | Path], *, corpus_name: str = "local corpus",
                  manifest: Mapping[str, Any] | None = None, built_at: str | None = None,
                  preprocessing: Mapping[str, Any] | None = None,
                  text_processing: Mapping[str, Any] | None = None,
                  nlp: Mapping[str, Any] | None = None,
                  metrics: Mapping[str, Any] | None = None,
                  comparison_unit: str = "book",
                  split_sections: bool = False,
                  min_section_words: int = 500,
                  include_core_metrics: bool = True,
                  lexile_frequency_source: str = "none",
                  include_parse_metrics: bool = False,
                  include_model_metrics: bool = False,
                  metric_selection: str = "auto",
                  progress=None) -> dict[str, Any]:
    """Precompute, once, what every grading run would otherwise recompute.

    A profile is a cache.  Measuring forty reference books is slow and the
    answer does not change between runs, so it is done once here and the
    result is what ``grade.py`` compares a manuscript against.  The corpus
    itself stays reproducible: ``build_corpus.py`` downloads it and this
    function reads whatever text files are in the corpus folder, so a profile
    can always be rebuilt rather than being a artifact nobody can regenerate.

    What lands in the profile is therefore a question of what is worth caching.
    Per-book scalars and the feature vectors under ``feature_profiles`` are,
    because they are small and every run needs them.  Raw book text is not: it
    is large, and re-reading the corpus folder gets it back.  A measurement
    that genuinely needs both texts at once (a true compression distance
    between a manuscript and a specific reference book, say) cannot be served
    from the cache at all and has to read the corpus at grading time, which is
    why those channels carry their own config switch.

    ``metric_selection`` decides which metrics are precomputed: ``"enabled"``
    follows the ``metrics`` config, ``"all"`` precomputes everything the
    parse/model flags allow, and ``"auto"`` (the default) means ``"enabled"``
    when a ``metrics`` mapping was supplied and ``"all"`` when it was not.
    Building a profile to ship wants ``"all"``; a project profiling its own
    corpus for its own enabled metrics wants ``"enabled"`` and should not pay
    for suites it has switched off.
    """

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
    wanted = _metric_names(include_parse_metrics, include_model_metrics,
                           metrics, metric_selection)

    books: list[dict[str, Any]] = []
    used_ids: Counter[str] = Counter()
    frequency: Counter[str] = Counter()
    # Per-author unigram frequency, alongside the corpus-wide ``frequency``
    # above. Populated only for books whose manifest entry gives an ``author``;
    # a corpus built without author metadata gets an empty dict, and a reader
    # of an OLDER profile (built before this table existed) finds the key
    # simply absent -- both are the same "no per-author table" case to
    # ``textgrader.metrics.stylometry_suite``'s ``author_language_model``
    # group, which reads this with ``.get(..., {})`` rather than requiring it.
    author_frequency: dict[str, Counter[str]] = {}
    feature_profiles: dict[str, list[dict[str, float]]] = {"function_words": []}
    metric_errors: dict[str, str] = {}
    skipped: list[str] = []
    pooled: dict[str, list[float]] = {name: [] for name in ITEM_SOURCES}

    for _, path, relative_name in files:
        raw_bytes = path.read_bytes()
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corpus source is not UTF-8: {path}") from exc
        digest = hashlib.sha256(raw_bytes).hexdigest()
        document = DocumentAnalysis.from_text(raw, processing=processing,
                                              nlp_settings=nlp_settings,
                                              source=relative_name,
                                              comparison_unit=comparison_unit)
        # One observation per section rather than per file, so a chapter can be
        # compared with chapters.  Comparing a 4,000-word chapter against whole
        # novels measures how books are divided rather than how they are
        # written, which ``units_comparable`` refuses; splitting is how you
        # build a corpus it will accept.
        parts = [(relative_name, document)]
        if split_sections:
            parts = [(f"{relative_name}#{title or index + 1}", view)
                     for index, (title, view) in enumerate(document.sections)
                     if view.word_count >= min_section_words]
            # A file with no detectable chapter heading must be skipped, not
            # fall back to itself: adding a whole novel to a corpus of chapters
            # is the unit contamination this option exists to avoid, and it is
            # invisible afterwards because the profile records one unit name.
            if len(parts) < 2:
                skipped.append(relative_name)
                continue
        for relative_name, analysis in parts:
            item_meta = entries.get(relative_name, entries.get(path.name, {}))
            if not isinstance(item_meta, dict):
                raise ValueError(f"manifest entry for {relative_name!r} must be an object")
            base_id = str(item_meta.get("id") or f"{path.stem}-{digest[:12]}")
            used_ids[base_id] += 1
            source_id = base_id if used_ids[base_id] == 1 else f"{base_id}-{used_ids[base_id]}"
            frequency.update(analysis.tokens)
            author = item_meta.get("author")
            if author:
                author_frequency.setdefault(str(author), Counter()).update(analysis.tokens)
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
                core = core_measure(analysis, floor=1,
                                    lexile_source=lexile_frequency_source)
                if core:
                    book.update({key: core[key] for key in CORE_METRIC_KEYS
                                 if core.get(key) is not None})
                    # Lexile is off unless a frequency source is configured, so
                    # it is written only when one was. Without this the metric
                    # could be enabled on a manuscript and never had a corpus
                    # to compare against: measurable, never comparable.
                    if core.get("lexile") is not None:
                        book["lexile"] = core["lexile"]
            for name in wanted:
                spec = REGISTRY[name]
                setting = metric_settings.get(name, {})
                options = dict(spec.defaults)
                if isinstance(setting, Mapping):
                    options.update({key: value for key, value in setting.items()
                                    if key != "enabled" and not key.startswith("_")})
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
                # A metric that needs more than one number per book -- an
                # embedding, a transition table, a frequency vector -- caches it
                # by exposing profile_vector(analysis, config).  Only a scalar
                # fits in a book row, which is what kept those measurements
                # out of the profile and forced them to be within-document.
                builder = getattr(module, "profile_vector", None)
                if callable(builder):
                    try:
                        vector = builder(analysis, options)
                    except Exception as exc:
                        metric_errors[f"{name}.profile_vector"] = f"{type(exc).__name__}: {exc}"
                        vector = None
                    if vector:
                        rows = feature_profiles.setdefault(name, [])
                        # Row i must describe books[i].  A book whose vector was
                        # skipped or failed gets an empty placeholder, because
                        # appending only on success silently shifts every later
                        # row and turns "nearest reference book" into a wrong
                        # answer rather than a missing one.
                        while len(rows) < len(books):
                            rows.append({})
                        rows.append(vector)
            for name, values in _item_values(analysis).items():
                pooled[name].extend(values)
            function_words = importlib.import_module("textgrader.metrics.function_words")
            feature_profiles["function_words"].append(function_words.vector(analysis.text))
            books.append(book)
        if progress:
            progress(relative_name, book)

    for name, rows in feature_profiles.items():
        while len(rows) < len(books):
            rows.append({})

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
        # Which frequency table the corpus Lexiles came from. The three
        # sources are three different scales, so a corpus built with one and a
        # manuscript measured with another are not comparable, and grade.py
        # withholds the comparison rather than printing a percentile off two
        # different rulers.
        "lexile_frequency_source": lexile_frequency_source,
        "parse_metrics": include_parse_metrics,
        "model_metrics": include_model_metrics,
        "metric_settings": effective,
        "metric_errors": metric_errors,
        "skipped_sources": skipped,
        "book_count": len(books), "books": books,
        "distributions": distributions,
        "word_frequency": {word: frequency[word] for word in sorted(frequency)},
        # The population, not a summary of it: every sentence, paragraph, word
        # and spoken turn in the corpus, as one quantile curve each.
        "item_distributions": {
            name: {"quantiles": quantile_curve(values), "count": len(values),
                   "sources": len(books)}
            for name, values in pooled.items() if values},
        "word_frequency_total": sum(frequency.values()),
        # Per-author unigram frequency tables, additive to the schema: empty
        # when no book's manifest entry carried an "author", and absent
        # entirely from any profile written before this field existed, which
        # is why every reader of these two keys uses .get(..., {}) rather
        # than assuming they are present. See CORPUS_LANGUAGE_MODEL /
        # AUTHOR_LANGUAGE_MODEL in textgrader.metrics.stylometry_suite.
        "author_word_frequency": {
            author: {word: counter[word] for word in sorted(counter)}
            for author, counter in sorted(author_frequency.items())},
        "author_word_frequency_total": {
            author: sum(counter.values()) for author, counter in sorted(author_frequency.items())},
        "feature_profiles": feature_profiles,
    }


def without_source(profile: dict, source_id: str) -> dict:
    """The profile that building from every text except ``source_id`` gives.

    Distributions are the pooled per-text values, so removing a text is exactly
    removing its value from each list: there is no need to re-measure the other
    other texts once per hold-out, which would make this quadratic in
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
        summary = summarize(values)
        summary["values"] = sorted(values)
        distributions[key] = summary
    out["distributions"] = distributions

    features = profile.get("feature_profiles", {})
    out["feature_profiles"] = {
        name: [row for position, row in enumerate(rows) if position != index]
        for name, rows in features.items()}
    return out



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
                        help="what ONE OBSERVATION is: a whole book, a chapter, a scene")
    parser.add_argument("--split-sections", action="store_true",
                        help="profile each chapter of each file separately, so a chapter "
                             "can be compared with chapters; set --comparison-unit to match")
    parser.add_argument("--min-section-words", type=int, default=500,
                        help="sections shorter than this are skipped when splitting")
    parser.add_argument("--no-core-metrics", action="store_true",
                        help="omit the default core prose distributions")
    parser.add_argument("--metric-selection", choices=("enabled", "all", "auto"), default=None,
                        help="which registered metrics to precompute: 'enabled' follows the "
                             "metrics config, 'all' precomputes everything the parse/model "
                             "flags allow, 'auto' picks 'enabled' when a config supplies "
                             "metrics. Defaults to corpus_builder.metric_selection in the "
                             "config, else 'auto'.")
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
        lexile_frequency_source=(config.get("analysis", {}) or {}).get(
            "lexile_frequency_source", "none"),
        metrics=config.get("metrics", {}), comparison_unit=args.comparison_unit,
        metric_selection=args.metric_selection or
        (config.get("corpus_builder", {}) or {}).get("metric_selection", "auto"),
        split_sections=args.split_sections, min_section_words=args.min_section_words,
        include_core_metrics=not args.no_core_metrics,
        include_parse_metrics=args.parse_metrics,
        include_model_metrics=args.model_metrics, progress=progress)
    write_profile(profile, args.output)
    if not args.quiet:
        print(f"\nwrote {profile['book_count']} {args.comparison_unit}(s) to {args.output}")
        if profile["skipped_sources"]:
            print(f"skipped {len(profile['skipped_sources'])} file(s) with no detectable "
                  f"sections to split on:")
            for name in profile["skipped_sources"][:10]:
                print(f"  {name}")
        if profile["metric_errors"]:
            print("metrics that could not be profiled:")
            for name, reason in sorted(profile["metric_errors"].items()):
                print(f"  {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
