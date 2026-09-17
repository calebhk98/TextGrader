#!/usr/bin/env python3
"""Which reference books hold a high reading level while still talking a lot?

The claim this settles: that turning summary into dialogue must lower the
reading level. Plenty of real novels are heavily spoken and still read high, so
the constraint is not dialogue itself - it is how the dialogue is built. This
finds the books that do both, and the ones that do neither, so somebody can
read them side by side and see the difference.

    python3 dialogue_study.py CORPUS_DIR [CORPUS_DIR ...]

For each book: share of words inside quotation marks, mean length of a spoken
sentence, mean length of a narrated one, and the reading measures. Sorted by
reading grade among the books that are at least a quarter dialogue.
"""

import re
import statistics as statistics
import sys
from pathlib import Path

from . import solo_notice
from ..core_metrics import measure
from ..text import paragraphs, sentences as sents, strip_gutenberg, strip_transcript, words

QUOTE = re.compile(
    "\u201c([^\u201c\u201d]{2,600})\u201d"   # curly, the Gutenberg default
    "|\"([^\"]{2,600})\""                     # straight
)

# Books that mark speech with single quotes (the British convention) are
# skipped rather than guessed at: an apostrophe and a closing quote are the
# same character, so any pattern that catches the speech also catches every
# possessive between two of them. Expect a few skips in any real corpus.


def spoken_and_narrated(text):
    """Split into what is inside quotation marks and what is outside.

    Quoted spans are joined into utterances, not into one string. A split
    quote ("A," she says, "B.") is ONE utterance and must be rejoined; two
    consecutive speeches ("A," he says. "B," she says.) are TWO and must not
    be. The difference is whether the narration between them closes a
    sentence.

    Joining every span in a chapter, as this did until it was fixed, glues
    separate speakers into one utterance and inflates every spoken mean that
    comes out of it - by three or four words a sentence on ordinary dialogue,
    which is enough to make a talky chapter look as though it were written in
    long speeches. Utterances are separated by a blank line here so that the
    caller's paragraph-based sentence splitter cannot run them together
    either.
    """
    utterances, current, last, out = [], [], 0, []
    for match in QUOTE.finditer(text):
        gap = text[last:match.start()]
        out.append(gap)
        if current and re.search(r"[.!?]", gap):
            utterances.append(" ".join(current))
            current = []
        current.append(match.group(1) or match.group(2) or "")
        last = match.end()
    if current:
        utterances.append(" ".join(current))
    out.append(text[last:])
    return "\n\n".join(utterances), " ".join(out)


def sent_lengths(text):
    return [len(words(sentence)) for paragraph in paragraphs(text) for sentence in sents(paragraph)
            if words(sentence)]


def sources(arguments):
    """Accept files as readily as directories.

    ``grade.py`` used to hand this program the manuscript's PARENT DIRECTORY,
    as if any sibling of the manuscript were a corpus text. It now passes the
    manuscript itself, so a single file has to be a legitimate input.
    """

    found = []
    for item in arguments:
        path = Path(item)
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(child for child in path.rglob("*") if child.is_file()))
        else:
            print(f"  no such file or directory: {path}", file=sys.stderr)
    return found


def main():
    paths = sources(sys.argv[1:])
    if not paths:
        sys.exit(__doc__)
    rows = []
    if True:
        for chapter_path in paths:
            if "stripped" in chapter_path.name:
                continue
            try:
                text = chapter_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text = strip_gutenberg(text)
            match = measure(text)
            if not match:
                continue
            spoken, narrated = spoken_and_narrated(text)
            spoken_words, narrated_words = len(words(spoken)), len(words(narrated))
            if not spoken_words:
                continue
            spoken_lengths, narrated_lengths = sent_lengths(spoken), sent_lengths(narrated)
            rows.append({
                "name": chapter_path.stem[:34],
                "quoted": 100 * spoken_words / (spoken_words + narrated_words),
                "spoken": statistics.fmean(spoken_lengths) if spoken_lengths else 0,
                "narr": statistics.fmean(narrated_lengths) if narrated_lengths else 0,
                "fk": match["fk"], "lexile": match.get("lexile"), "wps": match["wps"],
                "slcv": match["slcv"], "u10": match["u10"],
            })

    talky = [result for result in rows if result["quoted"] >= 25]
    talky.sort(key=lambda row: -row["fk"])

    def show(title, rows_to_show):
        print(f"\n{title}")
        print(f"  {'book':<36}{'quoted':>8}{'spoken':>8}{'narr':>7}"
              f"{'w/sent':>8}{'slCV':>7}{'u10':>7}{'F-K':>6}{'Lexile':>8}")
        for row in rows_to_show:
            # Approximate Lexile is off unless a frequency source is configured,
            # so it is routinely None. Formatting None as a float raised a
            # TypeError that killed the whole report; a dash is the honest cell.
            lexile = f"{row['lexile']:8.0f}" if row["lexile"] is not None else f"{'-':>8}"
            print(f"  {row['name']:<36}{row['quoted']:7.1f}%{row['spoken']:8.1f}"
                  f"{row['narr']:7.1f}{row['wps']:8.1f}{row['slcv']:7.1f}"
                  f"{row['u10']:7.1f}{row['fk']:6.1f}{lexile}")

    print(f"{len(rows)} books, {len(talky)} of them at least a quarter dialogue.")
    show("HIGHEST reading grade among the talky books", talky[:4])
    show("LOWEST reading grade among the talky books", talky[-4:])

    print("\n  quoted  share of words inside quotation marks")
    print("  spoken  mean words in a spoken sentence")
    print("  narr    mean words in a narrated sentence")
    print("  u10     sentences under ten words, as a percentage")
    print("\nRead the top group against the bottom group. Both are talking;")
    print("only one of them is doing it in whole sentences.")


# Run from grade.py, not on its own. Each script in measures/ reports one
# diagnostic; grade.py assembles enabled checks and is the interface that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --show and the per-file
# arguments are for, and it is never how a pass gets judged.

if __name__ == "__main__":
    solo_notice()
    main()
