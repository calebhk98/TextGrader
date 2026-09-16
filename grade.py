#!/usr/bin/env python3
"""One command for the whole picture: readability, style, numbers, voice, integrity.

The measurements live in five scripts that grew separately and each answer one
question. This runs all of them, in a fixed order, with the arguments that make
them comparable, so there is one thing to run and one report to read.

    python3 grade.py                 the whole report
    python3 grade.py --table         just the per-chapter table
    python3 grade.py --targets       just the table plus the distance to target
    python3 grade.py --brief         drop the slow sections (voice, citations)

What comes from where:

  prose_grade.py       sixteen measures graded as percentiles against 23 real
                       books, plus Flesch-Kincaid, ARI and an approximate
                       Lexile. This is the spine of the report.
  style_report.py      the two things prose_grade does not measure: how much of
                       a chapter is dialogue, and how far spoken and narrated
                       sentence lengths sit apart. Its PASS/FAIL thresholds are
                       invented and are deliberately not reproduced here.
  number_report.py     whether the same handful of numbers keeps recurring.
  voice_separation.py  whether each character's dialogue is distinguishable
                       from everyone else's.
  verify_citations.py  quotations in the character sheets that are not in the
                       manuscript.
  prose_check.py       PROSE_RULES violations in the character sheets.
  quote_length.py      sentences per quotation, against the corpus.
  banned_phrases.py    phrases the author has ruled out by name.
  style_report.py      conjunction rates and the narrator tic scan.
  check_edits.py       em dashes and the hard-line-break convention.

Targets are the author's: Flesch-Kincaid about 7, Lexile about 1000. The
corpus median is shown beside them because a target with nothing to compare it
against is a number somebody made up.
"""

import argparse
import os
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

from project_config import CHAPTERS_DIR, DIALOGUE_TARGETS, MANUSCRIPT, READING_TARGETS

HERE = Path(__file__).resolve().parent
CHAPTERS = sorted(CHAPTERS_DIR.glob("*.md"))
BOOK = MANUSCRIPT

LEXILE_TARGET = READING_TARGETS["lexile"]

# The reading grade rises with Chloe's age instead of sitting flat across the
# book. Each band is (first chapter, last chapter, floor, ceiling-of-intent).
# The floor is what the band should reach; the second number is where the band
# stops being worth pushing. A chapter already above its band is left alone.
# Floors by band, and one ceiling for the whole book. The author revised these
# downward after checking what the formulas actually mean: "my 9th grade target
# is actually too high based on the formulas. A better target is apparently
# like 7th-8th grade."
#
# The corpus agrees. Measured F-K across the 23 reference books: median 6.07,
# and only one book in the set clears 9 (Age of Innocence at 9.37). The books
# the author has named as the register he wants sit at Treasure Island 6.72,
# Wind in the Willows 7.67, Black Beauty 8.00, Little Women 8.08. This book is
# at 7.21, above the corpus median and between Treasure Island and Wind in the
# Willows, which is where it should be.
FK_BANDS = [tuple(band) for band in READING_TARGETS["chapter_bands"]]
FK_MAX = READING_TARGETS["fk_max"]
FK_BOOK_TARGET = READING_TARGETS["fk_book"]


def fk_band(stem):
    """The (floor, ceiling) this chapter's number is judged against."""
    try:
        chapter_number = int(stem[:2])
    except ValueError:
        return None
    for first_chapter, last_chapter, floor, ceil in FK_BANDS:
        if first_chapter <= chapter_number <= last_chapter:
            return floor, ceil
    return None


def load(name):
    # measures/ has to be importable too: prose_grade imports style_report from
    # alongside it, and dialogue_study loads prose_grade the same way.
    module = HERE / "measures"
    if str(module) not in sys.path:
        sys.path.insert(0, str(module))
    spec = importlib.util.spec_from_file_location(name[:-3], module / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(script, *args, quiet_rc=True):
    """Run one of the measures in measures/ and return its stdout.

    Several of them exit non-zero when they find something, which is correct
    behaviour and not a failure, so the return code is ignored by default.
    """
    env = dict(os.environ, HALSTEAD_VIA_GRADE="1")
    result = subprocess.run([sys.executable, str(HERE / "measures" / script), *map(str, args)],
                       capture_output=True, text=True, env=env)
    if result.returncode and not quiet_rc:
        return f"({script} exited {result.returncode})\n{result.stdout}{result.stderr}"
    return result.stdout.rstrip("\n")


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")



# metric -> (label, goal, comparison, source of the goal)
#   ">=" pass at or above, "<=" pass at or below, "~" pass within the band
GOALS = [
    # Ceiling raised to 5,000 so a long chapter is not a failure, but the
    # book is meant to stay around 300 pages: the target is still 2,000-3,000
    # and MEAN_WORD_TARGET below is what the book average is judged on.
    ("_words_all", "word count",                 (2000, 5000), "~",  "author, transcript included"),
    ("fk",       "reading grade (Flesch-Kincaid)", FK_MAX,      ">=", "configured, see band"),
    ("lexile",   "Lexile (approx)",               LEXILE_TARGET, ">=", "configured"),
    ("ari",      "reading grade (ARI)",           FK_MAX,      ">=", "configured, tracks F-K"),
    ("wps",      "words per sentence",            14.3,        ">=", "corpus median"),
    ("slcv",     "sentence-length variation CV",  85.3,        ">=", "corpus median"),
    ("wpp",      "words per paragraph",           (22, 48),    "~",  "corpus median 33.7"),
    ("spp",      "sentences per paragraph",       (1.8, 3.2),  "~",  "corpus median 2.4"),
    ("wlen",     "mean word length",              4.1,         ">=", "corpus median"),
    ("long7",    "words of 7+ characters %",      13.2,        ">=", "corpus median"),
    ("sttr",     "lexical diversity sTTR",        40.7,        ">=", "corpus median"),
    ("top100",   "commonest-100 words %",         48.1,        "<=", "corpus median"),
    ("commas",   "commas per sentence",           0.9,         ">=", "corpus median"),
    ("subord",   "sentences with subordination %", 22.2,       ">=", "corpus median"),
    ("relcl",    "sentences with a relative clause %", (14, 26), "~", "corpus median 19.5"),
    ("simple",   "clause-free sentences %",       65.9,        "<=", "corpus median"),
    ("u10",      "sentences under 10 words %",    45.7,        "<=", "corpus median"),
    ("b2035",    "sentences of 20-35 words %",    15.9,        ">=", "corpus median"),
    ("shortruns","sentences in a run of 3+ short %", 17.1,     "<=", "corpus median"),
    ("negative", "negative-space sentences %",    5.0,         "<=", "author"),
    ("front",    "sentences opening on a subordinate clause %", (0.5, 3.0), "~", "corpus median 1.6"),
    ("andrate",  '"and" as a share of words %',   3.3,         "<=", "corpus median"),
]


def dialogue_share(path):
    """Share of words inside quotation marks, and the spoken-sentence count.

    The spoken mean is blind to the worst failure available. A chapter that
    narrates every exchange - "the sergeant tells him", "what the debrief tells
    him" - has no quoted sentences to measure, so it scores nothing at all, and
    a chapter with five spoken sentences can post a better mean than a chapter
    with two hundred. Chapter 26 measured 17.0 off five sentences while being
    four per cent dialogue. Always read the share and the count beside the mean.
    """
    import statistics as _st
    spec2 = importlib.util.spec_from_file_location(
        "ds", HERE / "measures" / "dialogue_study.py")
    dialogue_module = importlib.util.module_from_spec(spec2)
    import sys as _sys
    _argv, _sys.argv = _sys.argv, ["x"]
    try:
        spec2.loader.exec_module(dialogue_module)
    finally:
        _sys.argv = _argv
    text = Path(path).read_text(encoding="utf-8")
    text, _ = dialogue_module.pg.strip_transcript(text)
    spoken, narrated = dialogue_module.spoken_and_narrated(text)
    spoken_lengths = dialogue_module.sent_lengths(spoken)
    spoken_words, narrated_words = len(dialogue_module.pg.words(spoken)), len(dialogue_module.pg.words(narrated))
    if not (spoken_words + narrated_words):
        return None
    return {"quoted": 100 * spoken_words / (spoken_words + narrated_words),
            "spoken": _st.fmean(spoken_lengths) if spoken_lengths else None,
            "n": len(spoken_lengths),
            "short": 100 * sum(1 for length in spoken_lengths if length <= 3) / len(spoken_lengths) if spoken_lengths else None}


def one_chapter(path):
    """One chapter down the page instead of across it.

    The wide table has twenty-seven columns and adjacent ones are easy to
    confuse; an agent working chapter 22 read the and-rate column as the
    negative-space column and reported a fall where there had been a rise.
    This format has one metric per line with its goal beside it, so there is
    nothing to miscount.
    """
    prose_grader = load("prose_grade.py")
    text = Path(path).read_text(encoding="utf-8")
    measurements = prose_grader.measure(text, floor=10)
    if not measurements:
        sys.exit(f"{path}: too short to measure")

    name = Path(path).stem
    print("\n" + name + "\n" + "-" * len(name) + "\n")
    print(f"  {'metric':<40}{'value':>9}{'goal':>14}  {'':<6}{'goal from'}")
    print("  " + "-" * 84)

    band = fk_band(name)
    failed = []
    for key, label, goal, comparison, src in GOALS:
        if band and key in ("fk", "ari"):
            goal, src = band[0], f"band {band[0]:g}-{band[1]:g} for this chapter"
        value = measurements.get(key)
        if value is None:
            print(f"  {label:<40}{'-':>9}{'':>14}  {'':<6}{src}")
            continue
        if comparison == "~":
            lower_bound, upper_bound = goal
            at_goal = lower_bound <= value <= upper_bound
            goal_text = f"{lower_bound:g} to {upper_bound:g}"
        elif comparison == ">=":
            at_goal = value >= goal
            goal_text = f">= {goal:g}"
        else:
            at_goal = value <= goal
            goal_text = f"<= {goal:g}"
        mark = "pass" if at_goal else "FAIL"
        if not at_goal:
            failed.append(label)
        print(f"  {label:<40}{value:9.1f}{goal_text:>14}  {mark:<6}{src}")

    dialogue = dialogue_share(path)
    prose_grade_measurements = load("prose_grade.py")
    tshare = (prose_grade_measurements.measure(Path(path).read_text(encoding="utf-8"), floor=10)
              or {}).get("_transcript", 0)
    if dialogue and tshare >= 25:
        # A chat chapter carries its dialogue as transcript lines, which are
        # stripped before measurement, so the quoted-share and spoken-mean
        # gates below are measuring the handful of lines that happen to sit in
        # quotation marks. Chapter 32 scored 0.2% quoted and a spoken mean of
        # 2.0 while being mostly people talking to each other.
        print("  " + "-" * 84)
        print(f"  dialogue measures skipped: {tshare:.0f}% of this chapter is chat")
        print("  transcript, which is stripped before measurement, so quoted share")
        print("  and spoken mean would describe a few stray lines rather than the")
        print("  chapter. Read the transcript itself.")
        dialogue = None
    if dialogue:
        quoted_min = DIALOGUE_TARGETS["quoted_word_share_min"]
        spoken_min = DIALOGUE_TARGETS["spoken_sentence_words_min"]
        short_max = DIALOGUE_TARGETS["short_spoken_lines_max"]
        print("  " + "-" * 84)
        print(f"  {'dialogue, share of words quoted %':<40}{dialogue['quoted']:9.1f}"
              f"{f'>= {quoted_min:g}':>14}  "
              f"{'pass' if dialogue['quoted'] >= quoted_min else 'FAIL':<6}configured")
        if dialogue["spoken"] is not None:
            print(f"  {'mean spoken sentence, words':<40}{dialogue['spoken']:9.1f}"
                  f"{f'>= {spoken_min:g}':>14}  "
                  f"{'pass' if dialogue['spoken'] >= spoken_min else 'FAIL':<6}configured")
            print(f"  {'spoken lines of 1-3 words %':<40}{dialogue['short']:9.1f}"
                  f"{f'<= {short_max:g}':>14}  "
                  f"{'pass' if dialogue['short'] <= short_max else 'FAIL':<6}configured")
        if dialogue["n"] < 20:
            print(f"\n  ** Only {dialogue['n']} spoken sentences in this chapter. The mean above is")
            print("     measured over almost nothing and means almost nothing. A chapter this")
            print("     quiet has usually narrated its dialogue instead of writing it; check")
            print("     the narration for 'he tells her', 'she asks him whether', and the like.")
    print("  " + "-" * 84)
    if band:
        print(f"\n  Reading grade is judged against this chapter's band, not one")
        print(f"  book-wide number: floor {band[0]:g}, and past {band[1]:g} it stops being")
        print(f"  worth pushing. A chapter already above its band is left alone.")
    print("\n  %d of %d at goal." % (len(GOALS) - len(failed), len(GOALS)))
    if failed:
        print("  short on: " + ", ".join(failed))
    print("\n  A FAIL is a prompt to look, not an instruction to change the number.")
    print("  Several of these goals are the corpus median, which half of 23 real")
    print("  books sit below. A chapter can be right and still fail three of them.")


def table():
    rule("1. EVERY MEASURE, ONE ROW PER CHAPTER")
    print(run("prose_grade.py", "--summary", *CHAPTERS))


# The per-chapter ceiling is 5,000, which is deliberately loose: one long
# chapter costs the book nothing. What the book cannot afford is every chapter
# drifting up, so the average is what is actually judged. The book currently
# runs about 300 printed pages, which is where the author wants it, and he
# will accept fifty either way. At roughly 380 words to the page that puts
# the ceiling at a 3,600 mean, and it refuses the 600-page version outright.
MEAN_WORD_TARGET = (2600, 3600)


def book_length(prose_grader):
    """The book average, which is the number that keeps the page count honest."""
    counts = [prose_grader.measure(chapter_path.read_text(encoding="utf-8"), floor=10)["_words_all"]
              for chapter_path in CHAPTERS]
    counts = [word_count for word_count in counts if word_count]
    total, mean = sum(counts), sum(counts) / len(counts)
    minimum_words, maximum_words = MEAN_WORD_TARGET
    verdict = "ok" if minimum_words <= mean <= maximum_words else ("under" if mean < minimum_words else "OVER")
    print(f"  book length   {total:,} words over {len(counts)} chapters, "
          f"mean {mean:,.0f}   target {minimum_words:,}-{maximum_words:,}   {verdict}")
    over = [(chapter_path.stem, word_count) for chapter_path, word_count in zip(CHAPTERS, counts) if word_count > 5000]
    if over:
        print("  past the 5,000 ceiling: "
              + ", ".join(f"{chapter_name} ({word_count:,})" for chapter_name, word_count in over))
    longest = sorted(zip([chapter_path.stem for chapter_path in CHAPTERS], counts),
                     key=lambda r: -r[1])[:3]
    print("  longest: " + ", ".join(f"{chapter_name} {word_count:,}" for chapter_name, word_count in longest))


def targets():
    """Distance to the author's targets, per band and for the book."""
    prose_grader = load("prose_grade.py")
    rule("2. AGAINST TARGET (reading grade by band, Lexile %d)" % LEXILE_TARGET)
    book_length(prose_grader)

    rows = []
    for chapter_path in CHAPTERS:
        measurements = prose_grader.measure(chapter_path.read_text(encoding="utf-8"), floor=10)
        if measurements:
            # Also measure the chapter as a reader meets it, transcript
            # included. The graded figure strips chat, which is right for
            # judging the prose, but in the last five chapters chat is a
            # quarter to two fifths of the words and the reader reads it.
            orig = prose_grader.strip_transcript
            prose_grader.strip_transcript = lambda x: (x, 0.0)
            try:
                as_read = prose_grader.measure(chapter_path.read_text(encoding="utf-8"), floor=10)
            finally:
                prose_grader.strip_transcript = orig
            rows.append((chapter_path.stem, measurements["fk"], measurements["lexile"],
                         as_read["fk"] if as_read else measurements["fk"]))
    whole = prose_grader.measure(BOOK.read_text(encoding="utf-8"))

    print(f"\n  book       F-K {whole['fk']:5.1f}   Lexile {whole['lexile']:7.1f}")
    print(f"  target     F-K {FK_BOOK_TARGET:5.1f}   Lexile {LEXILE_TARGET:7.1f}")
    print(f"  gap        F-K {whole['fk'] - FK_BOOK_TARGET:+5.1f}   "
          f"Lexile {whole['lexile'] - LEXILE_TARGET:+7.1f}")

    print("\n  The reading grade is meant to climb with Chloe's age rather than sit")
    print("  flat, so each band is judged on its own average, not against one number.\n")
    print(f"  {'band':<12}{'chapters':<12}{'floor':>7}{'average':>9}{'':>3}{'under floor'}")
    print("  " + "-" * 62)

    for first_chapter, last_chapter, floor, ceil in FK_BANDS:
        got = [row for row in rows if first_chapter <= int(row[0][:2]) <= last_chapter]
        if not got:
            continue
        avg = sum(row[1] for row in got) / len(got)
        low = sorted((row for row in got if row[1] < floor), key=lambda r: r[1])
        high = sorted((row for row in got if row[3] > ceil), key=lambda r: -r[3])
        mark = "" if avg >= floor else "  <-- band under floor"
        # Three decimals in the failing list. At 5.496 against a floor of 5.5
        # both the one- and two-decimal forms printed "5.5" under a floor
        # printed as 5.5, which reads as a broken script rather than a near
        # miss. A chapter named here should show why it was named.
        names = ", ".join(f"{row[0][:2]} ({row[1]:.3f})" for row in low) or "none"
        print(f"  {str(first_chapter) + '-' + str(last_chapter):<12}{len(got):<12}{floor:>7.1f}{avg:>9.2f}{mark}")
        print(f"  {'':<12}{'':<12}{'':>7}{'':>9}   {names}")
        if high:
            over = ", ".join(f"{row[0][:2]} ({row[3]:.1f})" for row in high)
            print(f"  {'':<12}{'':<12}{'':>7}{'':>9}   over the "
                  f"{ceil:.0f} ceiling: {over}")

    print("\n  A chapter already above its band is left where it is. The bands are")
    print("  floors for the band average, not per-chapter quotas, and a chapter with")
    print("  a structural reason to sit low (chapter 20's protected fight holds 57%")
    print("  of its sentences) is allowed to.")


def grade_vs_corpus():
    rule("3. GRADED AGAINST THE CORPUS")
    print(run("prose_grade.py", BOOK))


def dialogue():
    """Dialogue share and the spoken/narration split, per chapter.

    style_report prints these alongside PASS/FAIL against thresholds that have
    no source behind them. The numbers are worth having; the verdicts are not,
    so only the numbers are pulled through.
    """
    rule("4. DIALOGUE SHARE AND THE SPOKEN / NARRATION SPLIT")
    print(run("style_report.py", "--summary"))
    print(run("quote_length.py"))
    print(f"\n  {'chapter':<24}{'quoted':>8}{'spoken':>9}{'narration':>11}{'gap':>7}")
    print("  " + "-" * 57)
    quoted_values, gaps = [], []
    for chapter_path in CHAPTERS:
        out = run("style_report.py", chapter_path)
        quoted_match = re.search(r"QUOTED\s+([\d.]+)%", out)
        spoken_match = re.search(r"spoken\s+mean\s+([\d.]+) words", out)
        narrated_match = re.search(r"narration\s+mean\s+([\d.]+) words", out)
        if not (quoted_match and spoken_match and narrated_match):
            continue
        quoted_value, spoken_value, narrated_value = float(quoted_match.group(1)), float(spoken_match.group(1)), float(narrated_match.group(1))
        quoted_values.append(quoted_value)
        gaps.append(narrated_value - spoken_value)
        print(f"  {chapter_path.stem:<24}{quoted_value:7.1f}%{spoken_value:9.1f}{narrated_value:11.1f}{narrated_value - spoken_value:7.1f}")
    if quoted_values:
        import statistics as st
        print("  " + "-" * 57)
        print(f"  {'median':<24}{st.median(quoted_values):7.1f}%"
              f"{'':9}{'':11}{st.median(gaps):7.1f}")
    print("\n  quoted    share of words inside quotation marks")
    print("  gap       narration mean minus spoken mean, in words. A wide gap is")
    print("            ordinary; a gap near zero means the narration has gone as")
    print("            clipped as the dialogue.")


def numbers():
    rule("5. NUMBERS")
    print(run("number_report.py", BOOK))


def constructions():
    rule("6. REPEATED CONSTRUCTIONS, AGAINST THE CORPUS")
    print(run("tics.py"))


def absolutes():
    rule("7. ABSOLUTE WORDS")
    print(run("absolutes.py"))


def banned():
    rule("8. RULED-OUT PHRASES")
    print(run("banned_phrases.py"))


def register():
    rule("9. REGISTER DRIFT BETWEEN CHAPTERS")
    print(run("register.py"))


def voice():
    rule("10. VOICE SEPARATION")
    print(run("voice_separation.py", "--prose"))
    print(run("voice_separation.py", "--chat"))


def integrity():
    rule("11. INTEGRITY")
    for label, script, args in (
            ("Character-sheet quotations not found in the manuscript",
             "verify_citations.py", ()),
            ("PROSE_RULES violations in the character sheets",
             "prose_check.py", ()),
            ("Em dashes and the hard-line-break convention",
             "check_edits.py", ())):
        out = run(script, *args)
        lines = [line for line in out.split("\n") if line.strip()]
        print(f"\n  {label}")
        for line in lines:
            print(f"    {line}")


class Tee:
    """Print a section and keep it, so the scorecard can read what failed."""

    def __init__(self):
        self.lines = []

    def run(self, function):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            function()
        out = buf.getvalue()
        print(out, end="")
        self.lines.extend(out.split("\n"))


# A measure has failed if one of these appears on its line, and passed if one
# of the PASS_MARKERS does. Each script says so in its own words, which is why
# these are lists rather than one pattern. Anything matching neither is prose.
FAIL_MARKERS = (
    r"\bFAIL\b",
    r":\s*(?:UNDER|OVER)\b",
    # The same verdict without a colon in front of it. `prose_grade.py` ends
    # the book-length line with a bare OVER or UNDER, so a book one word past
    # its mean ceiling matched neither list: the line stopped saying "ok",
    # nothing said "FAIL", and the scorecard quietly went from 59 measures to
    # 58 and reported all 58 passing. A measure that can leave the count is
    # worse than a measure that fails.
    r"\s(?:UNDER|OVER)\s*$",
    r"\bCUT \d+%",
    # The band table's verdict, not its notes. Section 2 prints two things: a
    # band average against the band floor, which is the judgement, and beneath
    # it the individual chapters sitting under that floor, which is a list to
    # read while fixing. This used to match the list, so a band whose average
    # cleared its floor still failed the scorecard on the strength of its own
    # footnote, and with chapters 1 and 2 locked the section could never pass
    # however the book changed. The docstring below the table has said all
    # along that these are "floors for the band average, not per-chapter
    # quotas". This now matches what the table itself marks.
    r"<-- band under floor",
    r"[1-9]\d* problem\(s\)",
    r"[1-9]\d* not found in the manuscript",
    r"^\s*[1-9]\d* flagged across",
    r"\bover corpus max\b",
)

PASS_MARKERS = (
    r"\bpass\b",
    r"\bok\s*$",
    r"\bok\s{2,}",
    r"^\s*0 problem\(s\)",
    r"^\s*0 flagged across",
    r"0 quotations checked, 0 not found",
    r"^\s*none$",
)

SKIP = (r"^\s*0 problem", r"^\s*0 flagged", r"0 not found", r"still over target")


def scorecard(seen):
    """What passed, what did not, and the ratio.

    The author, on an earlier version that printed only the failures: "Seeing
    only you failed 5 metrics, when there are 300, hides that you passed 295."
    A hundred per cent was never the goal; ninety to ninety-five is.
    """
    rule("SCORECARD")
    bad, good, context = [], 0, ""
    for line in seen:
        stripped_line = line.rstrip()
        if not stripped_line.strip():
            continue
        if re.match(r"^\d+\. [A-Z]", stripped_line.strip()):
            context = stripped_line.strip()
            continue
        if any(re.search(pattern, stripped_line) for pattern in FAIL_MARKERS) and \
                not any(re.search(pattern, stripped_line) for pattern in SKIP):
            bad.append((context, stripped_line.strip()))
        elif any(re.search(pattern, stripped_line, re.I) for pattern in PASS_MARKERS):
            good += 1

    total = good + len(bad)
    if total:
        print(f"\n  {good} of {total} measures passing "
              f"({100 * good / total:.0f}%)\n")
    if not bad:
        print("  nothing failing.\n")
        return
    print(f"  the {len(bad)} not passing:\n")
    last = None
    for ctx, line in bad:
        if ctx != last:
            print(f"    {ctx or 'unsectioned'}")
            last = ctx
        print(f"      {line}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", action="store_true", help="only the per-chapter table")
    parser.add_argument("--targets", action="store_true", help="only the table and targets")
    parser.add_argument("--brief", action="store_true",
                    help="skip voice separation and citation checking")
    parser.add_argument("--one", metavar="CHAPTER",
                    help="one chapter, down the page, each metric against its goal")
    args = parser.parse_args()

    if args.one:
        return one_chapter(args.one)

    if not BOOK.exists():
        sys.exit(f"{BOOK.name} is missing; run build_manuscript.py first")

    tee = Tee()
    tee.run(table)
    if args.table:
        return
    tee.run(targets)
    if args.targets:
        return
    tee.run(grade_vs_corpus)
    tee.run(dialogue)
    tee.run(numbers)
    tee.run(constructions)
    tee.run(absolutes)
    tee.run(banned)
    tee.run(register)
    if not args.brief:
        tee.run(voice)
    tee.run(integrity)
    scorecard(tee.lines)


if __name__ == "__main__":
    main()
