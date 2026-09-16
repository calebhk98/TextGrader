#!/usr/bin/env python3
"""How many sentences a character gets to say before the quotation marks close.

The book was written with a strong one-line-per-turn habit, and the habit is
measurable: 82% of its quotations were a single sentence against a corpus
median of 76%, and only 6% ran to three or more against a median of 12%. That
puts it at the clipped end of the corpus, beside Peter Pan and Alice, when the
prose everywhere else is at the other end. Characters should be able to say
"What? No. We can't do that, it hurts us. We should do X instead." in one
breath, and mostly they cannot.

Chat transcripts are excluded: they are lowercase, unpunctuated and follow
their own convention, so counting sentences in them is meaningless.
"""
import glob
import os
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from project_config import CHAPTERS_DIR, CORPUS_DIRS, DIALOGUE_TARGETS

# ---------------------------------------------------------------------------
# TARGETS. Read this before changing any number below.
#
# These bands come from FOUR NAMED BOOKS, not from percentiles, and that is
# deliberate. A reviewer caught the repository building an impossible target
# out of column-wise percentiles, twice:
#
#   "You cannot combine the 75th percentile of mean (20.4), the 75th percentile
#    of <=4w (35.9%), the 75th percentile of 5-29w (60.7%) and the 75th
#    percentile of >=30w (18.8%). If you sum those bucket percentages you get
#    115.4%. It creates a statistically impossible book, because the columns
#    are coupled: every turn falls in exactly one bin and they sum to 100."
#
# Worse, the coupling runs the OPPOSITE way from what mixing the columns
# assumes. Across the corpus, a longer mean goes with FEWER short lines, not
# more: Black Beauty has the longest mean at 37.3 and one of the lowest short
# shares at 14.5%, while Men Without Women has the shortest mean at 6.5 and the
# highest short share at 43.7%. Aiming at p75 of the mean AND p75 of the short
# share aims at two different books at once.
#
# **The rule for anyone editing this file: pick real books whose profile you
# want and use their whole rows. Never mix percentiles across these columns.**
#
# The author's chosen references, and their actual measured rows:
#
#   book                mean   CV%   <=4w   5-29w   >=30w    sum
#   tom_sawyer          17.0   148   26.2    58.4    15.4   100.0
#   treasure_island     22.2   148   25.6    52.1    22.3   100.0
#   little_women        23.4   113   13.4    61.1    25.5   100.0
#   wind_in_willows     27.8   152   20.4    50.8    28.8   100.0
#
# and his stated aims within that group: a mean "of like 20ish, plus or minus a
# bit", a CV "of like 125, or like 115-135", and short replies at "a max of
# ~30, goal closer to 20-25". The bands below are those, widened only to the
# edges of the peer group where he did not specify.
PEER_BOOKS = ("tom_sawyer", "treasure_island", "little_women", "wind_in_willows")

TARGET_MEAN = DIALOGUE_TARGETS["sentences_per_quotation_min"]
TARGET_THREE_PLUS = DIALOGUE_TARGETS["three_plus_sentences_min"]
TARGET_WORD_MEAN = tuple(DIALOGUE_TARGETS["quotation_word_mean"])
TARGET_WORD_CV = tuple(DIALOGUE_TARGETS["quotation_word_cv"])
TARGET_SHORT_SHARE = tuple(DIALOGUE_TARGETS["short_quotation_share"])
TARGET_SHORT_HARD_MAX = DIALOGUE_TARGETS["short_quotation_hard_max"]
TARGET_LONG_SHARE = tuple(DIALOGUE_TARGETS["long_quotation_share"])


def _check_bands_are_possible():
    """A bucket target that cannot sum to 100 is the error this file exists to
    prevent, so it is asserted rather than trusted."""
    lower_bound = TARGET_SHORT_SHARE[0] + TARGET_LONG_SHARE[0]
    upper_bound = TARGET_SHORT_SHARE[1] + TARGET_LONG_SHARE[1]
    assert lower_bound < 100 and upper_bound < 100, "short+long bands leave no room for the middle"
    assert 40 <= 100 - upper_bound and 100 - lower_bound <= 70, (
        f"implied middle band {100 - upper_bound:.0f}-{100 - lower_bound:.0f}% is outside the "
        f"peer range of 50.8-61.1%")


_check_bands_are_possible()


def quotations(text):
    """Spoken turns, with parse failures excluded.

    A regex over straight quotes cannot tell a closing mark from an apostrophe
    or an unmatched one, so in a Gutenberg text a single span can swallow pages
    of narration between two stray marks. Hemingway produced a 2,993-word
    "quotation" that way and This Side of Paradise a 1,899-word one, and those
    wrecked the standard deviation the corpus targets were built on. A real
    spoken turn does not cross a paragraph break: a genuine multi-paragraph
    speech reopens the quotation mark on each paragraph. So spans containing a
    blank line are dropped, on both sides of the comparison.
    """
    text = re.sub(r"(?m)^#.*$", "", text)
    text = re.sub(r"(?m)^[a-z]+: .*$", "", text)   # chat transcript lines
    text = text.replace("\u201c", '"').replace("\u201d", '"')
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


def corpus():
    rows = []
    for directory in CORPUS_DIRS:
        if not os.path.isdir(directory):
            continue
        for name in sorted(os.listdir(directory)):
            try:
                text = open(os.path.join(directory, name), encoding="utf-8",
                            errors="ignore").read()
            except OSError:
                continue
            text_path = profile(text)
            if text_path and text_path["quotes"] >= 200:
                rows.append((name.replace("_stripped", "")[:24], text_path))
    return rows


def main():
    files = sorted(glob.glob(str(CHAPTERS_DIR / "*.md")))
    if not files:
        sys.exit(f"no chapters found in {CHAPTERS_DIR}")
    book = profile("\n".join(open(chapter_path).read() for chapter_path in files))
    ref = corpus()

    print(f"\n  {'chapter':<26}{'quotes':>7}{'mean':>7}{'1 sent %':>10}{'3+ %':>7}")
    print("  " + "-" * 57)
    for chapter_path in files:
        chapter_profile = profile(open(chapter_path).read())
        if not chapter_profile or chapter_profile["quotes"] < 15:
            continue
        flag = "  <-- clipped" if chapter_profile["mean"] < 1.25 else ""
        print(f"  {os.path.basename(chapter_path)[:-3]:<26}{chapter_profile['quotes']:>7}"
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
    print(f"\n  sentences per quotation: target mean {TARGET_MEAN}, "
          f"target 3+ {TARGET_THREE_PLUS}%: "
          f"{'ok' if book['mean'] >= TARGET_MEAN else 'UNDER'}")

    lower_bound, upper_bound = TARGET_WORD_MEAN
    cvlo, cvhi = TARGET_WORD_CV
    slo, shi = TARGET_SHORT_SHARE
    llo, lhi = TARGET_LONG_SHARE
    mid = 100 - book["short"] - book["long"]

    def band(value, lower_bound, upper_bound):
        return "ok" if lower_bound <= value <= upper_bound else "FAIL"

    print("\n  words per quotation, against " + ", ".join(PEER_BOOKS))
    print("  bands are whole rows from those four books, never mixed percentiles")
    # Two decimals, not one. At 17.947 against a floor of 18.0 the one-decimal
    # form printed "18.0 ... target 18-24 ... FAIL", which reads as a broken
    # script rather than as a near miss.
    print(f"    {'mean':<22}{book['wmean']:>7.2f}   peers 17.0-27.8   "
          f"target {lower_bound:.0f}-{upper_bound:.0f}    {band(book['wmean'], lower_bound, upper_bound)}")
    print(f"    {'median':<22}{book['wmed']:>7.0f}")
    print(f"    {'variation (CV %)':<22}{book['wcv']:>7.0f}    peers 113-152     "
          f"target {cvlo:.0f}-{cvhi:.0f}  {band(book['wcv'], cvlo, cvhi)}")
    print(f"    {'4 words or under':<22}{book['short']:>7.1f}%   peers 13.4-26.2%  "
          f"target {slo:.0f}-{shi:.0f}%   {band(book['short'], slo, shi)}"
          + ("  OVER HARD MAX" if book["short"] > TARGET_SHORT_HARD_MAX else ""))
    print(f"    {'5 to 29 words':<22}{mid:>7.1f}%   peers 50.8-61.1%")
    print(f"    {'30 words or over':<22}{book['long']:>7.1f}%   peers 15.4-28.8%  "
          f"target {llo:.0f}-{lhi:.0f}%   {band(book['long'], llo, lhi)}")
    print(f"    {'buckets sum to':<22}"
          f"{book['short'] + mid + book['long']:>7.1f}%")
    print("\n  The three bucket rows are coupled: every turn lands in exactly one")
    print("  and they sum to 100. Do not target them independently.\n")


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
