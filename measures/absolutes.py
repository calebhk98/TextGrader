#!/usr/bin/env python3
"""Absolute statements, counted and benchmarked against the reference corpus.

Two things are measured, and the second matters more than the first.

**Rate.** How often the prose reaches for never / always / nobody / everything.
Every novel uses these; the question is whether this one leans on them harder
than twenty-three published books do.

**The flat absolute with an exception behind it.** The defect pattern found by
hand in this manuscript: a sentence states something universal, and within the
next few sentences the text supplies a case it does not cover.

    "Nobody has ever sent Chloe anything."  ... then a letter arrives for her.
    "She tells him everything."             ... then she declines to say what
                                                one project was about.

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
import statistics as st
import sys
from pathlib import Path

# The measures live in measures/; the manuscript is a level up.
HERE = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pg", HERE / "prose_grade.py")
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

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
    text, _ = pg.strip_transcript(text)
    if side == "narration":
        text = QUOTED.sub(" ", text)
    elif side == "spoken":
        text = " ".join(m.group(0).strip('"') for m in QUOTED.finditer(text))
    paras = [p for p in pg.paragraphs(text) if p.strip() != "---"]
    ss = [s for p in paras for s in pg.sents(p) if pg.words(s)]
    if len(ss) < 10:
        return None if side == "all" else {
            "words": len(pg.words(text)), "sentences": len(ss),
            "absolutes": 0, "per1000": 0.0, "share": 0.0,
            "pairs": [], "lines": []}
    n_w = len(pg.words(text))
    hits = [i for i, s in enumerate(ss) if ABSOLUTE.search(s)]
    pairs = []
    for i in hits:
        for j in range(i + 1, min(i + 1 + WINDOW, len(ss))):
            if EXCEPTION.search(ss[j]):
                pairs.append((ss[i], ss[j]))
                break
    return {
        "words": n_w,
        "sentences": len(ss),
        "absolutes": len(hits),
        "per1000": 1000 * len(hits) / n_w,
        "share": 100 * len(hits) / len(ss),
        "pairs": pairs,
        "lines": [ss[i] for i in hits],
    }


def build(dirs):
    out = {}
    for d in dirs:
        for f in sorted(Path(d).rglob("*.txt")):
            if "stripped" in f.stem:
                continue
            t = f.read_text(encoding="utf-8", errors="replace")
            t = pg.strip_gutenberg(t) if hasattr(pg, "strip_gutenberg") else t
            m = measure(t)
            if m:
                out[f.stem] = {k: m[k] for k in ("per1000", "share", "words")}
                out[f.stem]["pairs"] = len(m["pairs"])
                for sd in ("narration", "spoken"):
                    q = measure(t, side=sd)
                    out[f.stem][sd + "_per1000"] = q["per1000"] if q else 0.0
                print(f"  {f.stem:<36}{m['per1000']:6.2f} per 1000")
    REF.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {len(out)} books to {REF.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="*")
    ap.add_argument("--pairs", action="store_true",
                    help="print every absolute-then-exception pair in full")
    ap.add_argument("--list", metavar="CHAPTER",
                    help="print every narration absolute in one chapter")
    a = ap.parse_args()
    if a.corpus:
        return build(a.corpus)

    if a.list:
        m = measure(Path(a.list).read_text(encoding="utf-8"))
        for line in m["lines"]:
            print("  " + ABSOLUTE.sub(lambda x: x.group(0).upper(),
                                      line.strip())[:300])
        print(f"\n  {len(m['lines'])} sentences carry an absolute in "
              f"{Path(a.list).stem}")
        return

    ref = json.loads(REF.read_text()) if REF.exists() else {}
    rates = sorted(v["per1000"] for v in ref.values())
    shares = sorted(v["share"] for v in ref.values())

    print(f"\n{'chapter':<24}{'words':>7}{'absolutes':>11}{'per 1000':>10}"
          f"{'% of sents':>12}{'exceptions':>12}")
    print("-" * 76)
    tot_a = tot_w = tot_p = 0
    all_pairs = []
    for p in sorted((HERE / "chapters").glob("*.md")):
        m = measure(p.read_text(encoding="utf-8"))
        if not m:
            continue
        tot_a += m["absolutes"]; tot_w += m["words"]; tot_p += len(m["pairs"])
        all_pairs += [(p.stem, x, y) for x, y in m["pairs"]]
        flag = "  <--" if rates and m["per1000"] > max(rates) else ""
        print(f"{p.stem:<24}{m['words']:>7}{m['absolutes']:>11}"
              f"{m['per1000']:>10.2f}{m['share']:>12.1f}{len(m['pairs']):>12}{flag}")

    book = 1000 * tot_a / tot_w if tot_w else 0
    print("-" * 76)
    print(f"{'BOOK':<24}{tot_w:>7}{tot_a:>11}{book:>10.2f}{'':>12}{tot_p:>12}")
    if rates:
        pct = 100 * sum(1 for r in rates if r < book) / len(rates)
        print(f"\n  corpus of {len(rates)} books:  low {min(rates):.2f}   "
              f"median {st.median(rates):.2f}   high {max(rates):.2f}")
        print(f"  this book {book:.2f} per 1000 words, "
              f"at the {pct:.0f}th percentile of the corpus")
        cp = sorted(v["pairs"] / (v["words"] / 1000) for v in ref.values())
        bp = 1000 * tot_p / tot_w
        print(f"\n  absolute followed within {WINDOW} sentences by an exception:")
        print(f"  corpus  low {min(cp):.3f}   median {st.median(cp):.3f}   "
              f"high {max(cp):.3f}   per 1000 words")
        print(f"  this book {bp:.3f}")

    for sd, label in (("narration", "narration only"), ("spoken", "dialogue only")):
        w = c = 0
        for p in sorted((HERE / "chapters").glob("*.md")):
            q = measure(p.read_text(encoding="utf-8"), side=sd)
            if q:
                w += q["words"]; c += q["absolutes"]
        if not (w and ref):
            continue
        rs = [v.get(sd + "_per1000", 0) for v in ref.values()]
        rs = sorted(r for r in rs if r)
        b = 1000 * c / w
        pc = 100 * sum(1 for r in rs if r < b) / len(rs)
        print(f"\n  {label}: this book {b:.2f} per 1000, at the {pc:.0f}th "
              f"percentile")
        print(f"  corpus  low {min(rs):.2f}   median {st.median(rs):.2f}   "
              f"high {max(rs):.2f}")

    if a.pairs and all_pairs:
        print(f"\n{'=' * 76}\nEVERY ABSOLUTE WITH AN EXCEPTION BEHIND IT\n{'=' * 76}")
        for stem, x, y in all_pairs:
            print(f"\n{stem}")
            print(f"  absolute : {x.strip()[:300]}")
            print(f"  exception: {y.strip()[:300]}")
    elif all_pairs:
        print(f"\n  {len(all_pairs)} pairs found. Run with --pairs to read them.")


# Run from grade.py, not on its own. Every measure in measures/ reports one
# slice; the scorecard is the whole picture and it is the thing that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --show and the per-file
# arguments are for, and it is never how a pass gets judged.
def _solo_notice():
    import sys, os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [one measure of thirteen. the scorecard is: python3 grade.py]",
          file=sys.stderr)

if __name__ == "__main__":
    _solo_notice()
    main()
