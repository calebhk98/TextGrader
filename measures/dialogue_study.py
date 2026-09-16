#!/usr/bin/env python3
"""Which reference books hold a high reading level while still talking a lot?

The claim this settles: that turning summary into dialogue must lower the
reading level. Plenty of real novels are heavily spoken and still read well
above this manuscript, so the constraint is not dialogue itself — it is how the
dialogue is built. This finds the books that do both, and the ones that do
neither, so somebody can read them side by side and see the difference.

    python3 dialogue_study.py CORPUS_DIR [CORPUS_DIR ...]

For each book: share of words inside quotation marks, mean length of a spoken
sentence, mean length of a narrated one, and the reading measures. Sorted by
reading grade among the books that are at least a quarter dialogue.
"""

import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib.util
spec = importlib.util.spec_from_file_location("pg", Path(__file__).resolve().parent / "prose_grade.py")
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

QUOTE = re.compile(
    "\u201c([^\u201c\u201d]{2,600})\u201d"   # curly, the Gutenberg default
    "|\"([^\"]{2,600})\""                     # straight
)

# Books that mark speech with single quotes (the British convention) are
# skipped rather than guessed at: an apostrophe and a closing quote are the
# same character, so any pattern that catches the speech also catches every
# possessive between two of them. Two of the twenty-three go this way.


def spoken_and_narrated(text):
    """Split into what is inside quotation marks and what is outside.

    Quoted spans are joined into utterances, not into one string. A split
    quote ("A," she says, "B.") is one utterance and must be rejoined; two
    utterances ("A," Ruth says. "B," Sam says.) must not be. The difference is
    whether the narration between them closes a sentence. Joining every span
    in the chapter, as this did until now, glued separate speakers together
    and inflated every spoken mean the dialogue pass was steered by: chapter
    14 measured 16.7 and was really 12.9, chapter 18 measured 16.3 and was
    really 13.5. Utterances are separated by a blank line here so that the
    caller's paragraph-based sentence splitter cannot run them together
    either.
    """
    utterances, current, last, out = [], [], 0, []
    for m in QUOTE.finditer(text):
        gap = text[last:m.start()]
        out.append(gap)
        if current and re.search(r"[.!?]", gap):
            utterances.append(" ".join(current))
            current = []
        current.append(m.group(1) or m.group(2) or "")
        last = m.end()
    if current:
        utterances.append(" ".join(current))
    out.append(text[last:])
    return "\n\n".join(utterances), " ".join(out)


def sent_lengths(t):
    return [len(pg.words(s)) for p in pg.paragraphs(t) for s in pg.sents(p)
            if pg.words(s)]


def main():
    dirs = [Path(d) for d in sys.argv[1:]]
    if not dirs:
        sys.exit(__doc__)
    rows = []
    for d in dirs:
        for f in sorted(d.iterdir()):
            if not f.is_file() or "stripped" in f.name:
                continue
            try:
                t = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            t = pg.strip_gutenberg(t) if hasattr(pg, "strip_gutenberg") else t
            m = pg.measure(t)
            if not m:
                continue
            sp, na = spoken_and_narrated(t)
            sw, nw = len(pg.words(sp)), len(pg.words(na))
            if not sw:
                continue
            sl, nl = sent_lengths(sp), sent_lengths(na)
            rows.append({
                "name": f.stem[:34],
                "quoted": 100 * sw / (sw + nw),
                "spoken": st.fmean(sl) if sl else 0,
                "narr": st.fmean(nl) if nl else 0,
                "fk": m["fk"], "lexile": m["lexile"], "wps": m["wps"],
                "slcv": m["slcv"], "u10": m["u10"],
            })

    talky = [r for r in rows if r["quoted"] >= 25]
    talky.sort(key=lambda r: -r["fk"])

    def show(title, rs):
        print(f"\n{title}")
        print(f"  {'book':<36}{'quoted':>8}{'spoken':>8}{'narr':>7}"
              f"{'w/sent':>8}{'slCV':>7}{'u10':>7}{'F-K':>6}{'Lexile':>8}")
        for r in rs:
            print(f"  {r['name']:<36}{r['quoted']:7.1f}%{r['spoken']:8.1f}"
                  f"{r['narr']:7.1f}{r['wps']:8.1f}{r['slcv']:7.1f}"
                  f"{r['u10']:7.1f}{r['fk']:6.1f}{r['lexile']:8.0f}")

    print(f"{len(rows)} books, {len(talky)} of them at least a quarter dialogue.")
    show("HIGHEST reading grade among the talky books", talky[:4])
    show("LOWEST reading grade among the talky books", talky[-4:])

    print("\n  quoted  share of words inside quotation marks")
    print("  spoken  mean words in a spoken sentence")
    print("  narr    mean words in a narrated sentence")
    print("  u10     sentences under ten words, as a percentage")
    print("\nRead the top group against the bottom group. Both are talking;")
    print("only one of them is doing it in whole sentences.")


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
