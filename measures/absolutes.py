#!/usr/bin/env python3
"""Absolute statements, counted and benchmarked against the reference corpus.

Two things are measured, and the second matters more than the first.

**Rate.** How often the prose reaches for never / always / nobody / everything.
Every novel uses these; the question is whether this one leans on them harder
than twenty-three published books do.

**The flat absolute with an exception behind it.** A sentence states something
universal, and within the next few sentences the text supplies a case it does
not cover.

    "Nobody has ever written to her."  ... then a letter arrives.
    "She tells him everything."        ... then she declines to say what one
                                           of her projects was about.

The absolute reads as authorial fact rather than as a character's belief, so
the exception lands as a contradiction instead of as a turn. Where the turn is
deliberate the fix is usually to soften the absolute, not to drop the
exception.

    python3 absolutes.py                  the manuscript, chapter by chapter
    python3 absolutes.py --corpus DIR...  rebuild the corpus benchmark
"""

import argparse
import importlib.util
import json
import re
import statistics as statistics
import sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import project_config

spec = importlib.util.spec_from_file_location(
    "pg", Path(__file__).resolve().parent / "prose_grade.py")
prose_grade = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prose_grade)

REF = HERE / "absolutes_reference.json"

# Strong universals only. "all" and "every" on their own are far too common in
# ordinary prose to separate signal from noise, so they count only in the
# constructions that actually assert a universal.
ABSOLUTE = re.compile(r"""
    \b(?:
        all | always | never | every | everybody | everyone | everything
      | none | nobody | no\s+one | nothing | not\s+once | not\s+ever
      | any | anybody | anyone | anything | anywhere | everywhere | nowhere
      | entire | entirely | completely | totally | absolutely | utterly
      | whole | forever | invariably | constantly | perfectly
      | without\s+exception | in\s+every\s+case | each
    )\b
""", re.I | re.X)

# Counted in every context, dialogue included, because the word is the thing
# being counted and not the construction around it. The narration and dialogue
# splits below are reported separately as well, since a character speaking in
# absolutes and a narrator asserting them are different problems, but neither
# split is a filter: the headline figure is every occurrence in the chapter.
QUOTED = re.compile(r'"[^"]*"')


def denarrate(text):
    """Drop everything inside quotation marks."""
    return QUOTED.sub(" ", text)


# What an exception looks like arriving behind one.
EXCEPTION = re.compile(r"""
    \b(?:
        except | apart\s+from | other\s+than | aside\s+from | save\s+for
      | with\s+the\s+exception | but\s+for\s+the
      | the\s+one\s+(?:time|thing|exception|person)
      | the\s+first\s+time\s+(?:she|he|they|it)
    )\b
""", re.I | re.X)

WINDOW = 3   # sentences after the absolute in which an exception counts


def measure(text, side="all"):
    text, _ = prose_grade.strip_transcript(text)
    if side == "narration":
        text = QUOTED.sub(" ", text)
    elif side == "spoken":
        text = " ".join(match.group(0).strip('"') for match in QUOTED.finditer(text))
    paras = [paragraph for paragraph in prose_grade.paragraphs(text) if paragraph.strip() != "---"]
    sentences = [sentence for paragraph in paras for sentence in prose_grade.sents(paragraph) if prose_grade.words(sentence)]
    if len(sentences) < 10:
        return None if side == "all" else {
            "words": len(prose_grade.words(text)), "sentences": len(sentences),
            "absolutes": 0, "per1000": 0.0, "share": 0.0,
            "pairs": [], "lines": []}
    word_count = len(prose_grade.words(text))
    hits = [hit_index for hit_index, sentence in enumerate(sentences) if ABSOLUTE.search(sentence)]
    pairs = []
    for hit_index in hits:
        for candidate_index in range(hit_index + 1, min(hit_index + 1 + WINDOW, len(sentences))):
            if EXCEPTION.search(sentences[candidate_index]):
                pairs.append((sentences[hit_index], sentences[candidate_index]))
                break
    return {
        "words": word_count,
        "sentences": len(sentences),
        "absolutes": len(hits),
        "per1000": 1000 * len(hits) / word_count,
        "share": 100 * len(hits) / len(sentences),
        "pairs": pairs,
        "lines": [sentences[hit_index] for hit_index in hits],
    }


def build(dirs):
    out = {}
    for directory in dirs:
        for text_path in sorted(Path(directory).rglob("*.txt")):
            if "stripped" in text_path.stem:
                continue
            text = text_path.read_text(encoding="utf-8", errors="replace")
            text = prose_grade.strip_gutenberg(text) if hasattr(prose_grade, "strip_gutenberg") else text
            measurements = measure(text)
            if measurements:
                out[text_path.stem] = {key: measurements[key] for key in ("per1000", "share", "words")}
                out[text_path.stem]["pairs"] = len(measurements["pairs"])
                for side in ("narration", "spoken"):
                    side_measurements = measure(text, side=side)
                    out[text_path.stem][side + "_per1000"] = side_measurements["per1000"] if side_measurements else 0.0
                print(f"  {text_path.stem:<36}{measurements['per1000']:6.2f} per 1000")
    REF.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {len(out)} books to {REF.name}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories; default: the configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--corpus", nargs="*")
    parser.add_argument("--pairs", action="store_true",
                    help="print every absolute-then-exception pair in full")
    parser.add_argument("--list", metavar="CHAPTER",
                    help="print every narration absolute in one chapter")
    args = parser.parse_args()
    if args.corpus:
        return build(args.corpus)

    config = project_config.load_config(args.config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    chapters = resolve_paths(args.paths, chapters_dir)

    if args.list:
        measurements = measure(Path(args.list).read_text(encoding="utf-8"))
        for line in measurements["lines"]:
            print("  " + ABSOLUTE.sub(lambda x: x.group(0).upper(),
                                      line.strip())[:300])
        print(f"\n  {len(measurements['lines'])} sentences carry an absolute in "
              f"{Path(args.list).stem}")
        return

    if not chapters:
        print("no chapters found")
        return 0

    ref = json.loads(REF.read_text()) if REF.exists() else {}
    rates = sorted(reference_values["per1000"] for reference_values in ref.values())
    shares = sorted(reference_values["share"] for reference_values in ref.values())

    print(f"\n{'chapter':<24}{'words':>7}{'absolutes':>11}{'per 1000':>10}"
          f"{'% of sents':>12}{'exceptions':>12}")
    print("-" * 76)
    total_absolutes = total_words = total_pairs = 0
    all_pairs = []
    for chapter_path in chapters:
        measurements = measure(chapter_path.read_text(encoding="utf-8"))
        if not measurements:
            continue
        total_absolutes += measurements["absolutes"]; total_words += measurements["words"]; total_pairs += len(measurements["pairs"])
        all_pairs += [(chapter_path.stem, absolute_sentence, exception_sentence) for absolute_sentence, exception_sentence in measurements["pairs"]]
        flag = "  <--" if rates and measurements["per1000"] > max(rates) else ""
        print(f"{chapter_path.stem:<24}{measurements['words']:>7}{measurements['absolutes']:>11}"
              f"{measurements['per1000']:>10.2f}{measurements['share']:>12.1f}{len(measurements['pairs']):>12}{flag}")

    book = 1000 * total_absolutes / total_words if total_words else 0
    print("-" * 76)
    print(f"{'BOOK':<24}{total_words:>7}{total_absolutes:>11}{book:>10.2f}{'':>12}{total_pairs:>12}")
    if rates:
        pct = 100 * sum(1 for rate in rates if rate < book) / len(rates)
        print(f"\n  corpus of {len(rates)} books:  low {min(rates):.2f}   "
              f"median {statistics.median(rates):.2f}   high {max(rates):.2f}")
        print(f"  this book {book:.2f} per 1000 words, "
              f"at the {pct:.0f}th percentile of the corpus")
        corpus_pair_rates = sorted(reference_values["pairs"] / (reference_values["words"] / 1000) for reference_values in ref.values())
        book_pair_rate = 1000 * total_pairs / total_words if total_words else 0.0
        print(f"\n  absolute followed within {WINDOW} sentences by an exception:")
        print(f"  corpus  low {min(corpus_pair_rates):.3f}   median {statistics.median(corpus_pair_rates):.3f}   "
              f"high {max(corpus_pair_rates):.3f}   per 1000 words")
        print(f"  this book {book_pair_rate:.3f}")

    for side, label in (("narration", "narration only"), ("spoken", "dialogue only")):
        word_count = absolute_count = 0
        for chapter_path in chapters:
            side_measurements = measure(chapter_path.read_text(encoding="utf-8"), side=side)
            if side_measurements:
                word_count += side_measurements["words"]; absolute_count += side_measurements["absolutes"]
        if not (word_count and ref):
            continue
        side_rates = [reference_values.get(side + "_per1000", 0) for reference_values in ref.values()]
        # Zero is a real observation (a book with no uses), not missing data.
        side_rates = sorted(side_rates)
        book_rate = 1000 * absolute_count / word_count
        percentile = 100 * sum(1 for rate in side_rates if rate < book_rate) / len(side_rates)
        print(f"\n  {label}: this book {book_rate:.2f} per 1000, at the {percentile:.0f}th "
              f"percentile")
        print(f"  corpus  low {min(side_rates):.2f}   median {statistics.median(side_rates):.2f}   "
              f"high {max(side_rates):.2f}")

    if args.pairs and all_pairs:
        print(f"\n{'=' * 76}\nEVERY ABSOLUTE WITH AN EXCEPTION BEHIND IT\n{'=' * 76}")
        for stem, absolute_sentence, exception_sentence in all_pairs:
            print(f"\n{stem}")
            print(f"  absolute : {absolute_sentence.strip()[:300]}")
            print(f"  exception: {exception_sentence.strip()[:300]}")
    elif all_pairs:
        print(f"\n  {len(all_pairs)} pairs found. Run with --pairs to read them.")


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
    main()
