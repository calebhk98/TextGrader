#!/usr/bin/env python3
"""How many sentences a character gets to say before the quotation marks close.

Two related shapes are measured: how many sentences a quotation runs to
before it closes, and how many words. Chat transcripts are excluded: they
follow their own unpunctuated convention, so counting sentences in them is
meaningless.

Nothing here is a target until ``project_measures.quote_length`` supplies
one. With no targets configured this prints the book's own numbers, against
whatever peer-book corpus rows are configured, and does not fail.

Config (``project_measures.quote_length``):

    "peer_books": ["<corpus source id>", "<corpus source id>", ...],
    "targets": {
        "sentences_per_quotation_min": <number>,
        "three_plus_sentences_min": <number>,
        "quotation_word_mean": [low, high],
        "quotation_word_cv": [low, high],
        "short_quotation_share": [low, high],
        "short_quotation_hard_max": <number>,
        "long_quotation_share": [low, high]
    }

READ THIS BEFORE CHANGING ANY BAND: ``short_quotation_share`` and
``long_quotation_share`` are two of three buckets that must sum with the
unnamed middle bucket to 100% (every quotation falls in exactly one). Picking
independent percentile targets for each column can produce bands whose
implied middle bucket is negative or absurdly large - a statistically
impossible book. ``_check_bands_are_possible()`` catches that arithmetic
error whenever bands are configured; it does not judge whether the bands are
otherwise a good idea.
"""
import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import project_config

import re


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


def check_bands_are_possible(targets):
    """A bucket target that cannot sum to 100 is the error this exists to catch.

    Raised rather than asserted: ``assert`` disappears under ``python -O``,
    which would silently let an impossible book back in.
    """
    short_bounds = targets.get("short_quotation_share")
    long_bounds = targets.get("long_quotation_share")
    if not (isinstance(short_bounds, (list, tuple)) and isinstance(long_bounds, (list, tuple))):
        return
    lower_bound = short_bounds[0] + long_bounds[0]
    upper_bound = short_bounds[1] + long_bounds[1]
    if not (lower_bound < 100 and upper_bound < 100):
        raise ValueError(
            "project_measures.quote_length.targets: short_quotation_share + "
            "long_quotation_share leaves no room for the middle bucket")


def quotations(text):
    """Spoken turns, with parse failures excluded.

    A regex over straight quotes cannot tell a closing mark from an
    apostrophe or an unmatched one, so a single span can swallow pages of
    narration between two stray marks in an unedited source text, wrecking
    any standard deviation computed from it. A real spoken turn does not
    cross a paragraph break: a genuine multi-paragraph speech reopens the
    quotation mark on each paragraph. So spans containing a blank line are
    dropped, on both sides of any comparison.
    """
    text = re.sub(r"(?m)^#.*$", "", text)
    text = re.sub(r"(?m)^[a-z]+: .*$", "", text)   # chat transcript lines
    text = text.replace("“", '"').replace("”", '"')
    return [quotation for quotation in re.findall(r'"([^"]{2,})"', text) if "\n\n" not in quotation]


def sentences(quote):
    return max(1, len(re.findall(r"[.!?]+(?:\s|$)", quote.strip())) or 1)


def profile(text):
    quotes = quotations(text)
    counts = [sentences(quotation) for quotation in quotes]
    if not counts:
        return None
    word_lengths = [len(re.findall(r"[A-Za-z']+", quotation)) for quotation in quotes]
    mean_w = sum(word_lengths) / len(word_lengths)
    standard_deviation = statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0.0
    return {
        "quotes": len(counts),
        "mean": sum(counts) / len(counts),
        "one": sum(1 for count in counts if count == 1) / len(counts) * 100,
        "three": sum(1 for count in counts if count >= 3) / len(counts) * 100,
        "wmean": mean_w,
        "wmed": statistics.median(word_lengths),
        "wcv": 100 * standard_deviation / mean_w if mean_w else 0.0,
        "short": sum(1 for value in word_lengths if value <= 4) / len(word_lengths) * 100,
        "long": sum(1 for value in word_lengths if value >= 30) / len(word_lengths) * 100,
    }


def corpus(corpus_dirs, peer_books):
    rows = []
    for directory in corpus_dirs:
        if not directory.is_dir():
            continue
        for name in sorted(p.name for p in directory.iterdir()):
            try:
                text = (directory / name).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            text_profile = profile(text)
            if text_profile and text_profile["quotes"] >= 200:
                rows.append((name.replace("_stripped", "")[:24], text_profile))
    if peer_books:
        # Named peer books, not percentiles: comparing against a chosen set of
        # real books is deliberate (see the module docstring), so a name that
        # matches nothing in the corpus is dropped rather than silently
        # widening the comparison back out to the whole corpus.
        rows = [row for row in rows if row[0].split(".")[0] in peer_books]
    return rows


def band(value, bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return "not set"
    lower_bound, upper_bound = bounds
    return "ok" if lower_bound <= value <= upper_bound else "FAIL"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories; default: the configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    args = parser.parse_args(argv)

    config = project_config.load_config(args.config)
    settings = project_config.measure_settings("quote_length", config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    corpus_dirs = corpus_dirs_of(config)
    peer_books = settings.get("peer_books", [])
    targets = settings.get("targets", {})
    targets = targets if isinstance(targets, dict) else {}

    if targets:
        check_bands_are_possible(targets)
    else:
        print("no project_measures.quote_length.targets configured; the measurements below "
              "are descriptive only, and no dialogue-length policy is enforced.\n")

    files = resolve_paths(args.paths, chapters_dir)
    if not files:
        print(f"no chapters found")
        return 0
    book = profile("\n".join(path.read_text(encoding="utf-8") for path in files))
    if not book:
        print("no quotations found; nothing to measure")
        return 0
    ref = corpus(corpus_dirs, peer_books)

    print(f"\n  {'chapter':<26}{'quotes':>7}{'mean':>7}{'1 sent %':>10}{'3+ %':>7}")
    print("  " + "-" * 57)
    for chapter_path in files:
        chapter_profile = profile(chapter_path.read_text(encoding="utf-8"))
        if not chapter_profile or chapter_profile["quotes"] < 15:
            continue
        flag = "  <-- clipped" if chapter_profile["mean"] < 1.25 else ""
        print(f"  {chapter_path.stem[:26]:<26}{chapter_profile['quotes']:>7}"
              f"{chapter_profile['mean']:>7.2f}{chapter_profile['one']:>10.1f}{chapter_profile['three']:>7.1f}{flag}")

    print("  " + "-" * 57)
    print(f"  {'BOOK':<26}{book['quotes']:>7}{book['mean']:>7.2f}"
          f"{book['one']:>10.1f}{book['three']:>7.1f}")
    if ref:
        med = statistics.median(chapter_profile["mean"] for _, chapter_profile in ref)
        med_one = statistics.median(chapter_profile["one"] for _, chapter_profile in ref)
        med_three = statistics.median(chapter_profile["three"] for _, chapter_profile in ref)
        lower_bound = min(ref, key=lambda r: r[1]["mean"])
        upper_bound = max(ref, key=lambda r: r[1]["mean"])
        print(f"  {'corpus median':<26}{'':>7}{med:>7.2f}{med_one:>10.1f}"
              f"{med_three:>7.1f}   ({len(ref)} books)")
        print(f"  {'corpus low  ' + lower_bound[0]:<26}{'':>7}{lower_bound[1]['mean']:>7.2f}")
        print(f"  {'corpus high ' + upper_bound[0]:<26}{'':>7}{upper_bound[1]['mean']:>7.2f}")

    sentence_target = targets.get("sentences_per_quotation_min")
    three_plus_target = targets.get("three_plus_sentences_min")
    if sentence_target is None:
        print(f"\n  sentences per quotation: mean {book['mean']:.2f}, target not set")
    else:
        print(f"\n  sentences per quotation: target mean {sentence_target}, "
              f"target 3+ {three_plus_target if three_plus_target is not None else 'not set'}%: "
              f"{'ok' if book['mean'] >= sentence_target else 'UNDER'}")

    word_mean_bounds = targets.get("quotation_word_mean")
    word_cv_bounds = targets.get("quotation_word_cv")
    short_bounds = targets.get("short_quotation_share")
    long_bounds = targets.get("long_quotation_share")
    short_hard_max = targets.get("short_quotation_hard_max")
    mid = 100 - book["short"] - book["long"]

    print("\n  words per quotation" + (f", against {', '.join(peer_books)}" if peer_books else ""))
    if not any([word_mean_bounds, word_cv_bounds, short_bounds, long_bounds]):
        print("  no word-length bands configured; figures below are descriptive only.")
    word_mean_text = f"{word_mean_bounds[0]:.0f}-{word_mean_bounds[1]:.0f}" if word_mean_bounds else "not set"
    print(f"    {'mean':<22}{book['wmean']:>7.2f}   target {word_mean_text:<10}    {band(book['wmean'], word_mean_bounds)}")
    print(f"    {'median':<22}{book['wmed']:>7.0f}")
    cv_text = f"{word_cv_bounds[0]:.0f}-{word_cv_bounds[1]:.0f}" if word_cv_bounds else "not set"
    print(f"    {'variation (CV %)':<22}{book['wcv']:>7.0f}    target {cv_text:<10}  {band(book['wcv'], word_cv_bounds)}")
    short_text = f"{short_bounds[0]:.0f}-{short_bounds[1]:.0f}%" if short_bounds else "not set"
    hard_max_note = ""
    if short_hard_max is not None and book["short"] > short_hard_max:
        hard_max_note = "  OVER HARD MAX"
    print(f"    {'4 words or under':<22}{book['short']:>7.1f}%   target {short_text:<10}   "
          f"{band(book['short'], short_bounds)}{hard_max_note}")
    print(f"    {'5 to 29 words':<22}{mid:>7.1f}%")
    long_text = f"{long_bounds[0]:.0f}-{long_bounds[1]:.0f}%" if long_bounds else "not set"
    print(f"    {'30 words or over':<22}{book['long']:>7.1f}%   target {long_text:<10}   "
          f"{band(book['long'], long_bounds)}")
    print(f"    {'buckets sum to':<22}"
          f"{book['short'] + mid + book['long']:>7.1f}%")
    print("\n  The three bucket rows are coupled: every turn lands in exactly one")
    print("  and they sum to 100. Do not target them independently.\n")

    failures = [band(book["wmean"], word_mean_bounds), band(book["wcv"], word_cv_bounds),
               band(book["short"], short_bounds), band(book["long"], long_bounds)]
    below_sentence_target = sentence_target is not None and book["mean"] < sentence_target
    return 1 if ("FAIL" in failures or below_sentence_target) else 0


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
