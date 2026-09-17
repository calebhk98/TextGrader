#!/usr/bin/env python3
"""Build an auditable text corpus from one or more public-library providers."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence

from textgrader.corpus_builder.builder import BuildError, build, provider_health
from textgrader.corpus_builder.classifiers import classify_pov, classify_tense, narration_without_quotes, word_count
from textgrader.corpus_builder.config import ConfigError, Settings, from_mapping, load, validate
from textgrader.corpus_builder.providers import GENRE_TERMS, PROVIDER_TYPES
from textgrader.corpus_builder.providers.gutenberg import strip_boilerplate as strip_gutenberg

VERSION = "2.0.0"
CorpusError = (ConfigError, BuildError)
build_corpus = build
settings_from_dict = from_mapping
validate_settings = validate


def _merge_cli(settings: Settings, args: argparse.Namespace) -> Settings:
    values = dataclasses.asdict(settings)
    overrides = {
        "output_dir": args.output,
        "count": args.count,
        "minimum_count": args.minimum_count,
        "providers": args.provider,
        "metadata_providers": args.metadata_provider,
        "work_types": args.work_type,
        "genres": args.genre,
        "include_authors": args.include_author,
        "max_authors": args.max_authors,
        "pov": args.pov,
        "tense": args.tense,
        "min_year": args.min_year,
        "max_year": args.max_year,
        "year_policy": args.year_policy,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_books_per_author": args.max_books_per_author,
        "seed": args.seed,
        "openlibrary_email": args.openlibrary_email,
        "google_books_api_key": args.google_books_api_key,
        "refresh_catalog": True if args.refresh_catalog else None,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    return from_mapping(values)


def write_example_config(path: Path) -> None:
    settings = Settings(genres=["science_fiction"], pov="third", tense="past")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(settings), indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local text corpus from public digital libraries.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, help="standalone settings JSON, or a project config containing corpus_builder")
    parser.add_argument("--write-example-config", type=Path, metavar="PATH")
    parser.add_argument("--list-genres", action="store_true")
    parser.add_argument("--list-providers", action="store_true")
    parser.add_argument("--healthcheck", action="store_true", help="check configured provider endpoints and exit")
    parser.add_argument("--output")
    parser.add_argument("--count", type=int)
    parser.add_argument("--minimum-count", type=int)
    parser.add_argument("--provider", action="append", choices=sorted(PROVIDER_TYPES), help="source provider; repeat to combine sources")
    parser.add_argument("--metadata-provider", action="append", choices=["openlibrary", "google_books"], help="publication-year fallback; repeat to define the fallback order")
    parser.add_argument("--work-type", action="append", choices=["any", "fiction", "short_story", "essay", "speech", "poetry", "drama"], help="accepted form; repeat for several")
    parser.add_argument("--genre", action="append", choices=sorted(GENRE_TERMS))
    parser.add_argument("--include-author", action="append", help="author-name substring; repeat to allow several")
    parser.add_argument("--max-authors", type=int, help="maximum distinct primary authors; omit for unlimited")
    parser.add_argument("--max-books-per-author", type=int)
    parser.add_argument("--pov", choices=["any", "first", "third"])
    parser.add_argument("--tense", choices=["any", "past", "present"])
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--max-year", type=int)
    parser.add_argument("--year-policy", choices=["ignore", "best_effort", "strict"])
    parser.add_argument("--min-words", type=int)
    parser.add_argument("--max-words", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--openlibrary-email")
    parser.add_argument("--google-books-api-key")
    parser.add_argument("--refresh-catalog", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_genres:
        print("\n".join(sorted(GENRE_TERMS)))
        return 0
    if args.list_providers:
        print("\n".join(sorted(PROVIDER_TYPES)))
        return 0
    if args.write_example_config:
        write_example_config(args.write_example_config)
        print(f"Wrote {args.write_example_config}")
        return 0
    try:
        settings = load(args.config) if args.config else Settings()
        settings = _merge_cli(settings, args)
        if settings.count < 30:
            print("WARNING: target count is below 30; this may produce unstable distributions.", file=sys.stderr)
        if args.healthcheck:
            statuses = provider_health(settings)
            print(json.dumps(statuses, indent=2))
            return 0 if all(item["available"] for item in statuses.values()) else 1
        manifest, status = build(settings)
        summary = manifest["summary"]
        print(f"Selected {summary['selected']} of {summary['requested']} documents from {', '.join(settings.providers)}")
        print(f"Manifest: {Path(settings.output_dir) / 'manifest.json'}")
        return status
    except KeyboardInterrupt:
        print("Interrupted. Partial downloads and caches are preserved.", file=sys.stderr)
        return 130
    except (ConfigError, BuildError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
