#!/usr/bin/env python3
"""Count configured repeated constructions against a reference corpus.

Every construction this script looks for, and every target rate, comes from
``project_measures.tics`` in the user's own config. There is nothing to scan
for until that is populated: a habit worth tracking is specific to one
manuscript, and hard-coding somebody else's tics here would just make this
script silently measure the wrong book. Rates are per 100,000 words so a
70,000-word novel and a 300,000-word one compare directly.

Config (``project_measures.tics`` in config.json):

    "patterns": {
        "<label>": {"pattern": "<regex>", "target_per_100k": <number-or-null>}
    },
    "number_words": false,
    "number_targets": {"<word>": <target_per_100k>}

``patterns`` is checked against every reference book too, so a target of
``null`` still reports the book's own rate and the corpus spread; only a
number turns a row into a pass/fail. ``number_words``/``number_targets`` add
one row per tracked number word, built the same way, so a project that wants
to watch "one" through "fifteen" does not have to spell out the regex for
each.

Every user regex is compiled once, at load time, so a typo in a pattern fails
loudly with the label attached instead of quietly matching nothing. When the
optional ``regex`` package is installed, matching runs under a timeout
(``regex.timeout_seconds`` in config, default 2.0) so a catastrophic pattern
cannot hang the run; without it, matching falls back to the standard library
``re`` module and a warning is printed once, since ``re`` has no timeout.

    python3 tics.py                 the manuscript against the corpus
    python3 tics.py --show LABEL    print every hit for one configured pattern
"""
import argparse
import re as std_re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import project_config
from textgrader.optional import require

DEFAULT_TIMEOUT_SECONDS = 2.0


def words(text):
    return std_re.findall(r"[A-Za-z']+", text)


def strip_gutenberg(text):
    match = std_re.search(r"\*\*\* ?START OF.*?\*\*\*(.*?)\*\*\* ?END OF", text, std_re.S)
    return match.group(1) if match else text


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


def corpus_dirs_of(config):
    return tuple((Path(config["_config_dir"]) / value).resolve()
                for value in config.get("corpus_dirs", []))


def pick_engine(config):
    """Return (engine module, whether it supports timeout=, timeout seconds).

    A user-supplied pattern is arbitrary regex and arbitrary regex can be
    catastrophically slow to backtrack. The ``regex`` package's timeout
    argument turns that into a warning instead of a hang; plain ``re`` has no
    such argument, so a pattern that hangs under ``re`` still hangs, and that
    is why "regex" is preferred whenever it is installed.
    """
    regex_config = config.get("regex", {}) if isinstance(config.get("regex"), dict) else {}
    timeout = regex_config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    requested = regex_config.get("engine", "auto")
    if requested in ("auto", "regex"):
        module, reason = require("regex")
        if module is not None:
            return module, True, timeout
        if requested == "regex":
            print(f"  warning: regex.engine=\"regex\" requested but unavailable ({reason}); "
                  f"falling back to re, which cannot time out a pattern", file=sys.stderr)
        else:
            print("  warning: the optional 'regex' package is not installed; falling back to "
                  "re, which cannot time out a pattern. A hung user pattern will hang this run.",
                  file=sys.stderr)
    return std_re, False, timeout


def build_pattern_config(settings):
    """Merge configured construction patterns with the number-word rows.

    Returns a plain ``{label: {"pattern": ..., "target_per_100k": ..., "case_sensitive": ...}}``
    mapping, still uncompiled, so callers can validate every entry themselves.
    """
    patterns = dict(settings.get("patterns", {})) if isinstance(settings.get("patterns"), dict) else {}
    number_targets = settings.get("number_targets", {})
    if settings.get("number_words") and isinstance(number_targets, dict):
        for word, target in number_targets.items():
            patterns[f"number: {word}"] = {"pattern": rf"\b{std_re.escape(str(word))}\b",
                                           "target_per_100k": target}
    return patterns


def compile_patterns(patterns_config, engine):
    """Compile every configured pattern, failing loudly on the first bad one."""
    ignorecase, multiline = engine.IGNORECASE, engine.MULTILINE
    compiled, targets = {}, {}
    for label, entry in patterns_config.items():
        if not isinstance(entry, dict) or "pattern" not in entry:
            sys.exit(f"project_measures.tics.patterns[{label!r}] must be an object with "
                     f"a 'pattern' key")
        raw_pattern = entry["pattern"]
        flags = multiline | (0 if entry.get("case_sensitive") else ignorecase)
        try:
            compiled[label] = engine.compile(raw_pattern, flags)
        except Exception as exc:
            sys.exit(f"project_measures.tics.patterns[{label!r}]: invalid regex "
                     f"{raw_pattern!r}: {exc}")
        targets[label] = entry.get("target_per_100k")
    return compiled, targets


def count_matches(compiled, text, use_timeout, timeout, label):
    try:
        if use_timeout:
            return sum(1 for _ in compiled.finditer(text, timeout=timeout))
        return sum(1 for _ in compiled.finditer(text))
    except Exception as exc:
        print(f"  warning: pattern {label!r} failed or timed out ({exc}); counted as 0",
              file=sys.stderr)
        return 0


def measure(text, compiled, use_timeout, timeout):
    word_count = len(words(text))
    out = {}
    for label, pattern in compiled.items():
        hits = count_matches(pattern, text, use_timeout, timeout, label)
        out[label] = 100000 * hits / word_count if word_count else 0.0
    return out, word_count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories; default: the configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--show", help="a configured label, or a literal regex")
    args = parser.parse_args(argv)

    config = project_config.load_config(args.config)
    settings = project_config.measure_settings("tics", config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    corpus_dirs = corpus_dirs_of(config)

    patterns_config = build_pattern_config(settings)
    if not patterns_config:
        print("no project_measures.tics.patterns configured (and number_words is off).")
        print("set project_measures.tics.patterns, or number_words + number_targets, "
              "to enable this report.")
        return 0

    engine, use_timeout, timeout = pick_engine(config)
    compiled, targets = compile_patterns(patterns_config, engine)

    paths = resolve_paths(args.paths, chapters_dir)
    if not paths:
        print("no chapters found")
        return 0
    book_text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    if args.show:
        pattern_source = patterns_config.get(args.show, {}).get("pattern") if args.show in patterns_config else args.show
        case_sensitive = patterns_config.get(args.show, {}).get("case_sensitive", False)
        flags = engine.MULTILINE | (0 if case_sensitive else engine.IGNORECASE)
        try:
            show_pattern = engine.compile(pattern_source, flags)
        except Exception as exc:
            sys.exit(f"invalid regex {pattern_source!r}: {exc}")
        for chapter_path in paths:
            for line_number, line in enumerate(chapter_path.read_text(encoding="utf-8").split("\n"), 1):
                for match in show_pattern.finditer(line):
                    start = max(0, match.start() - 60)
                    print(f"{chapter_path.stem}:{line_number}  ...{line[start:match.end() + 60]}...")
        return 0

    book_rates, book_words = measure(book_text, compiled, use_timeout, timeout)
    ref = []
    for directory in corpus_dirs:
        if not directory.is_dir():
            continue
        for chapter_path in sorted(directory.rglob("*.txt")):
            if "stripped" in chapter_path.stem:
                continue
            cleaned = strip_gutenberg(chapter_path.read_text(encoding="utf-8", errors="replace"))
            row, word_count = measure(cleaned, compiled, use_timeout, timeout)
            if word_count > 20000:
                ref.append(row)

    print(f"\n{len(ref)} reference books. Rates per 100,000 words.\n")
    if not ref:
        print("  No readable reference books found. Add .txt files to a configured")
        print("  corpus directory or set corpus_dirs in config.json. Book rates and")
        print("  configured targets are still shown below.\n")
    print(f"  {'construction':<32}{'book':>8}{'target':>9}{'corpus med':>12}{'corpus max':>12}{'':>3}")
    print("  " + "-" * 78)
    rows, over = [], 0
    for label in compiled:
        vals = sorted(row[label] for row in ref)
        med = st.median(vals) if vals else None
        corpus_maximum = max(vals) if vals else None
        target = targets.get(label)
        excess = book_rates[label] / target if target else 0
        rows.append((excess, label, book_rates[label], target, med, corpus_maximum))
    for excess, label, benchmark, target, med, corpus_maximum in sorted(rows, reverse=True):
        med_text = f"{med:.1f}" if med is not None else "n/a"
        max_text = f"{corpus_maximum:.1f}" if corpus_maximum is not None else "n/a"
        if target is None:
            print(f"  {label:<32}{benchmark:>8.1f}{'not set':>9}{med_text:>12}{max_text:>12}")
            continue
        within_range = benchmark <= target
        over += 0 if within_range else 1
        print(f"  {label:<32}{benchmark:>8.1f}{target:>9.1f}{med_text:>12}{max_text:>12}"
              f"{'  pass' if within_range else '  CUT ' + f'{100 * (1 - target / benchmark):.0f}%'}")

    configured_targets = sum(1 for target in targets.values() if target is not None)
    if configured_targets:
        print(f"\n  {over} of {configured_targets} configured target(s) still over target.")
    else:
        print("\n  no target_per_100k values configured; every row above is descriptive only.")
        print("  set project_measures.tics.patterns[label].target_per_100k to enable pass/fail.")
    return 1 if over else 0


# Run from grade.py, not on its own. Each script in measures/ reports one
# diagnostic; grade.py assembles enabled checks and is the interface that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --show and the per-file
# arguments are for, and it is never how a pass gets judged.
def _solo_notice():
    import sys, os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [bundled diagnostic; use grade.py for the structured report]",
          file=sys.stderr)

if __name__ == "__main__":
    _solo_notice()
    sys.exit(main() or 0)
