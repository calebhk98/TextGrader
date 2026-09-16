"""Build and load self-contained, reproducible reference-corpus profiles.

This module deliberately does not acquire books.  It accepts local UTF-8 text
files (or directories containing them) and emits portable JSON which contains
source provenance and the distributions used by TextGrader at run time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
PARSER_VERSION = "1"
METRIC_DEFINITION_VERSION = "1"

from .text import paragraphs, sentences, strip_gutenberg, words


def _words(text: str) -> list[str]:
    return [word.lower().replace("’", "'") for word in words(text)]


def _sentences(text: str) -> list[str]:
    return sentences(text)


def _paragraphs(text: str) -> list[str]:
    return paragraphs(text)


def _source_files(inputs: Iterable[str | Path]) -> list[tuple[int, Path, str]]:
    found: list[tuple[int, Path, str]] = []
    for input_index, item in enumerate(inputs):
        path = Path(item).expanduser().resolve()
        if path.is_file():
            if path.suffix.lower() != ".txt":
                raise ValueError(f"corpus source is not a .txt file: {path}")
            found.append((input_index, path, path.name))
        elif path.is_dir():
            for source in path.rglob("*.txt"):
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
    ordered = sorted(values)
    if not ordered:
        return {"values": [], "count": 0, "median": None, "mad": None, "q1": None, "q3": None}
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    # inclusive quartiles behave sensibly for small corpora.
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [ordered[0]] * 3
    return {"values": ordered, "count": len(ordered), "median": median,
            "mad": statistics.median(deviations), "q1": quartiles[0], "q3": quartiles[2]}


def _timestamp(value: str | None) -> str:
    if value:
        return value
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    moment = datetime.fromtimestamp(int(epoch), timezone.utc) if epoch else datetime.now(timezone.utc)
    return moment.isoformat().replace("+00:00", "Z")


def build_profile(inputs: Iterable[str | Path], *, corpus_name: str = "local corpus",
                  manifest: Mapping[str, Any] | None = None, built_at: str | None = None,
                  preprocessing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Profile local text files without retaining or later requiring raw books."""
    files = _source_files(inputs)
    if not files:
        raise ValueError("corpus contains no .txt files")
    entries = _manifest_entries(manifest)
    preprocessing_settings = {"strip_gutenberg": True, **(preprocessing or {})}
    books: list[dict[str, Any]] = []
    used_ids: Counter[str] = Counter()
    frequency: Counter[str] = Counter()

    for input_index, path, relative_name in files:
        raw_bytes = path.read_bytes()
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"corpus source is not UTF-8: {path}") from exc
        digest = hashlib.sha256(raw_bytes).hexdigest()
        clean = strip_gutenberg(raw) if preprocessing_settings["strip_gutenberg"] else raw
        words = _words(clean)
        sentences = _sentences(clean)
        paragraphs = _paragraphs(clean)
        sentence_lengths = [len(_words(sentence)) for sentence in sentences]
        paragraph_lengths = [len(_words(paragraph)) for paragraph in paragraphs]
        item_meta = entries.get(relative_name, entries.get(path.name, {}))
        if not isinstance(item_meta, dict):
            raise ValueError(f"manifest entry for {relative_name!r} must be an object")
        base_id = str(item_meta.get("id") or f"{path.stem}-{digest[:12]}")
        used_ids[base_id] += 1
        source_id = base_id if used_ids[base_id] == 1 else f"{base_id}-{used_ids[base_id]}"
        frequency.update(words)
        books.append({
            "source_id": source_id, "source_filename": path.name,
            "source_path": relative_name, "source_hash": f"sha256:{digest}",
            "word_count": len(words), "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "mean_sentence_words": statistics.fmean(sentence_lengths) if sentence_lengths else None,
            "mean_paragraph_words": statistics.fmean(paragraph_lengths) if paragraph_lengths else None,
            "mean_word_characters": statistics.fmean(map(len, words)) if words else None,
            "metadata": {key: value for key, value in item_meta.items()
                         if key not in {"id", "filename", "path"}},
        })

    metric_keys = ("word_count", "sentence_count", "paragraph_count", "mean_sentence_words",
                   "mean_paragraph_words", "mean_word_characters")
    distributions = {key: _distribution([book[key] for book in books if book[key] is not None])
                     for key in metric_keys}
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
    return {
        "schema_version": SCHEMA_VERSION,
        "textgrader_version": "0.1.0",
        "parser_version": PARSER_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "corpus_name": corpus_name,
        "build_timestamp": _timestamp(built_at),
        "preprocessing": preprocessing_settings,
        "book_count": len(books), "books": books,
        "distributions": distributions,
        "word_frequency": {word: frequency[word] for word in sorted(frequency)},
        "word_frequency_total": sum(frequency.values()),
    }


def write_profile(profile: Mapping[str, Any], destination: str | Path) -> None:
    Path(destination).write_text(json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                 encoding="utf-8")


def load_profile(path: str | Path) -> dict[str, Any]:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported corpus profile schema: {profile.get('schema_version')!r}")
    if not isinstance(profile.get("books"), list) or not isinstance(profile.get("distributions"), dict):
        raise ValueError("invalid corpus profile")
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a TextGrader corpus profile from local .txt files")
    parser.add_argument("inputs", nargs="+", help="local .txt files or directories (searched recursively)")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--name", default="local corpus")
    parser.add_argument("--manifest", type=Path, help="optional JSON metadata manifest")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    write_profile(build_profile(args.inputs, corpus_name=args.name, manifest=manifest), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
