#!/usr/bin/env python3
"""Measure how distinguishable each character's dialogue is from the others'.

The swap test - would anyone notice if this line moved to another character? -
is a judgement call. This is the measurable version: extract each character's
lines, compute the same statistics for each speaker, and show how far apart
they sit. Characters whose numbers coincide have no voice yet.

    python3 voice_separation.py                    both channels, configured chapters dir
    python3 voice_separation.py MANUSCRIPT.md       both channels, one explicit file
    python3 voice_separation.py --chat chapters/    group chat only, an explicit directory

Two channels, measured separately because they are attributable with very
different confidence.

CHAT is exact when a cast list is configured. Messages are "name: text", so
every line has a known speaker and none are missed.

PROSE IS A BIASED SAMPLE and its numbers must be read with that in mind. Only
lines carrying an explicit "<name> says" tag can be attributed, and a book
that drops the tag once a two-hander is established will skew tagged lines
towards the openings of exchanges, which run short. Treat the prose figures
as comparable BETWEEN characters, since the bias hits every character alike,
and not as an estimate of that character's true average line length.

Every fact about who is in the cast lives in
``project_measures.voice_separation.speakers`` in the user's own config, not
in this file. With no cast configured, speakers are DISCOVERED instead:
chat speakers from generic "name: message" lines (the same shape
``textgrader.text.transcript_lines`` already knows how to find), and prose
speakers from a capitalised token sitting next to a speech verb. Discovery is
a much cruder net than a named cast and will over- and under-generate on
ambiguous prose; supplying ``speakers`` explicitly is the reliable path.
"""

import argparse
import collections
import re
import statistics as st
import sys
from pathlib import Path

from .. import project as project_config
from ..text import TranscriptConfig, transcript_lines

# Titles, speech verbs and hedge phrases are generic English, not facts about
# any one manuscript, so they stay as module defaults. A project can still
# override them (a formal book might add "Professor", a slangier one more
# hedges) through the same config keys.
DEFAULT_TITLES = ["Mrs.", "Mr.", "Ms.", "Dr.", "Coach", "Sergeant"]
DEFAULT_SPEECH_VERBS = ["says", "said", "asks", "asked", "tells", "told",
                        "shouts", "screams", "whispers"]
DEFAULT_HEDGES = ["i think", "maybe", "probably", "idk", "i dont know",
                  "i don't know", "kind of", "sort of", "i guess"]
DEFAULT_MIN_LINES = 8

# A capitalised run of one or two words, used only when no explicit speaker
# list is configured. It is deliberately loose: discovery trades precision
# for not requiring the user to type out a cast list first.
GENERIC_NAME = r"[A-Z][a-z']+(?:\s[A-Z][a-z']+)?"


def words(text):
    return re.findall(r"[A-Za-z][A-Za-z']*", text)


def normalise(text):
    return text.replace('“', '"').replace('”', '"')


def resolve_paths(explicit, default_dir):
    """Expand file/dir positional arguments into a sorted list of files.

    An explicit argument may be a single manuscript file (what grade.py
    passes) or a directory of chapter files. Only with nothing explicit do
    we fall back to the configured chapters directory, so this script never
    scans a directory the caller did not ask for.
    """
    if not explicit:
        if default_dir and default_dir.is_dir():
            return sorted(default_dir.glob("*.md"))
        return []
    paths = []
    for item in explicit:
        item = Path(item)
        if item.is_dir():
            paths.extend(sorted(item.glob("*.md")))
        elif item.is_file():
            paths.append(item)
    return paths


def _title_part(titles):
    if not titles:
        return ""
    group = "|".join(re.escape(title) for title in titles)
    return rf"(?:(?:{group})\s*)?"


def tagged_pattern(speakers, titles):
    """The name-matching half of a speaker tag.

    With ``speakers`` configured this only ever matches that cast, which is
    exact. Empty, it falls back to any capitalised token, which is the
    discovery mode described in the module docstring.
    """
    name_group = "|".join(re.escape(name) for name in speakers) if speakers else GENERIC_NAME
    return rf"{_title_part(titles)}({name_group}|her mom|her mother|her dad|her father)"


def collect_chat(paths, speakers):
    out = collections.defaultdict(list)
    if speakers:
        pattern = re.compile(r"^\s*(%s):\s*(.+)$" % "|".join(re.escape(name) for name in speakers), re.I)
        for source_path in paths:
            for line in source_path.read_text(encoding="utf-8").split("\n"):
                match = pattern.match(line.strip())
                if match:
                    out[match.group(1).lower()].append(match.group(2).strip())
        return out
    # Discovery: any generic "name: message" line. This is the same shape a
    # real chat transcript has, so it costs nothing to look for it everywhere
    # rather than only in files the user has separately told us are chat.
    config = TranscriptConfig(ignore_case=True)
    for source_path in paths:
        text = source_path.read_text(encoding="utf-8")
        for match in transcript_lines(text, config):
            username = match.group("username").strip()
            message = match.group("message").strip()
            if username and message:
                out[username.lower()].append(message)
    return out


def collect_prose(paths, speakers, titles, speech_verbs):
    """Lines with an explicit speaker tag. See the module docstring on bias."""
    out = collections.defaultdict(list)
    tagged = tagged_pattern(speakers, titles)
    says = "(?:%s)" % "|".join(re.escape(verb) for verb in speech_verbs)
    # Much fiction attributes with an action beat more often than with a
    # speech verb: '"Wait." Mrs. Hale is standing at the end of the desk.'
    # Matching only "<name> says" misses most tagged dialogue, so the first
    # pattern also accepts any sentence that opens with the speaker's name
    # straight after a quote.
    patterns = [
        (rf'"([^"]+)"[,.!?]?\s+{tagged}\b', 0, 1),
        (rf'"([^"]+)"[,]?\s+{says}\s+{tagged}', 0, 1),
        (rf'{tagged}\s+{says}[,:]?\s+"([^"]+)"', 1, 0),
    ]
    for source_path in paths:
        for line in normalise(source_path.read_text(encoding="utf-8")).split("\n"):
            for pat, tag_index, words_index in patterns:
                for match in re.finditer(pat, line):
                    groups = match.groups()
                    who = groups[words_index].lower().replace("her ", "")
                    who = {"mother": "mom", "father": "dad"}.get(who, who)
                    out[who].append(groups[tag_index])
    return out


def profile(lines, hedge_pattern=None):
    if hedge_pattern is None:
        hedge_pattern = r"\b(%s)\b" % "|".join(re.escape(hedge) for hedge in DEFAULT_HEDGES)
    tokens = [words(line) for line in lines]
    flat = [word.lower() for tokens in tokens for word in tokens]
    if not flat:
        return None
    # Mean segmental TTR compares equal-size samples rather than rewarding a
    # speaker merely for having fewer total words. Fifty words is deliberately
    # small enough for dialogue samples.
    #
    # The trailing window is dropped, because keeping it puts back the bias
    # MSTTR exists to remove. A four-word tail scores at or near 100% and is
    # then averaged with the SAME WEIGHT as a full fifty-word window, so its
    # pull on the mean is 1/segments: largest for the speakers with least
    # text, which is the direction of the plain-TTR artefact. Measured on
    # tagged dialogue, a 94-word speaker read 7.3 points high, enough to move
    # them from mid-pack to narrowest vocabulary in the book.
    #
    # ``metrics/dialogue_channels._window_ttr`` drops only tails under half a
    # window, which is right there: it reports a median over the windows, so a
    # short one distorts little. The mean here weighs every segment equally,
    # so this needs the stricter rule.
    window = 50
    segments = [flat[index:index + window] for index in range(0, len(flat), window)]
    if len(segments) > 1 and len(segments[-1]) < window:
        segments.pop()
    # Under one full window there is no equal-size comparison to make, so the
    # number is withheld rather than computed from a single short sample and
    # printed as though it meant the same as the others.
    complete = len(flat) >= window
    msttr = st.fmean(len(set(segment)) / len(segment) for segment in segments) if complete else None
    return {
        "n": len(lines),
        "words": len(flat),
        "wpl": len(flat) / len(lines),
        # Plain type-token ratio, reported beside MSTTR rather than replaced by
        # it. It is a poor vocabulary measure: it falls mechanically as a
        # sample grows, and across this project's tagged dialogue it ranked
        # speakers at Spearman rho -0.976 against their word counts, which is
        # very nearly a pure inverse ranking of who talks most. It is kept
        # because it is the diagnostic for its own replacement (that rho is
        # what demonstrates the confound), because existing work is calibrated
        # against it, and because a speaker whose two ranks diverge is a
        # speaker whose apparent vocabulary was an artefact of line volume.
        # What it must not do is wear MSTTR's name, or the column silently
        # changes scale under readers holding the old numbers.
        "ttr": 100 * len(set(flat)) / len(flat),
        "msttr": 100 * msttr if msttr is not None else None,
        "msttr_segments": len(segments) if complete else 0,
        "q": 100 * sum(1 for line in lines if "?" in line) / len(lines),
        "short": 100 * sum(1 for tokens in tokens if len(tokens) <= 3) / len(lines),
        "long": 100 * sum(1 for tokens in tokens if len(tokens) > 15) / len(lines),
        "hedge": 100 * sum(1 for line in lines if re.search(hedge_pattern, line, re.I)) / len(lines),
    }


def show(title, data, floor, note, hedges=None):
    hedges = DEFAULT_HEDGES if hedges is None else hedges
    hedge_pattern = r"\b(%s)\b" % "|".join(re.escape(hedge) for hedge in hedges)
    rows = {speaker: speaker_profile for speaker, lines in data.items()
           if (speaker_profile := profile(lines, hedge_pattern)) and speaker_profile["n"] >= floor}
    if not rows:
        print(f"\n{title}: nothing with at least {floor} lines")
        return
    print(f"\n{title}   ({note})")
    print(f"  {'speaker':10}{'lines':>7}{'words':>7}{'w/line':>8}{'TTR%':>7}"
          f"{'MSTTR%':>8}{'seg':>5}{'quest%':>8}{'1-3w%':>7}{'>15w%':>7}{'hedge%':>8}")
    for speaker, speaker_profile in sorted(rows.items(), key=lambda kv: -kv[1]["wpl"]):
        # The segment count travels with MSTTR because the value alone cannot
        # say whether it averaged two windows or twenty, and those are not the
        # same evidence. A dash means the speaker has under one full window.
        msttr = speaker_profile["msttr"]
        msttr_column = f"{msttr:>8.1f}" if msttr is not None else f"{'-':>8}"
        segments = speaker_profile["msttr_segments"]
        print(f"  {speaker:10}{speaker_profile['n']:>7}{speaker_profile['words']:>7}{speaker_profile['wpl']:>8.1f}{speaker_profile['ttr']:>7.1f}"
              f"{msttr_column}{(segments or '-'):>5}"
              f"{speaker_profile['q']:>8.0f}{speaker_profile['short']:>7.0f}{speaker_profile['long']:>7.0f}{speaker_profile['hedge']:>8.0f}")
    for key, label in (("wpl", "words per line"), ("short", "1-3 word share")):
        vals = [speaker_profile[key] for speaker_profile in rows.values()]
        minimum, maximum = min(vals), max(vals)
        mean = st.fmean(vals)
        spread = (maximum - minimum) / mean if mean else 0.0
        print(f"  {label:16} {minimum:.1f} to {maximum:.1f}   "
              f"spread {100 * spread:.0f}% of the mean"
              f"{'   <- speakers barely differ' if spread < 0.5 else ''}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories to scan; default: the "
                         "configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--prose", action="store_true")
    parser.add_argument("--min-lines", type=int, default=None)
    args = parser.parse_args(argv)

    config = project_config.load_config(args.config)
    settings = project_config.measure_settings("voice_separation", config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    paths = resolve_paths(args.paths, chapters_dir)

    speakers = [name for name in settings.get("speakers", []) if isinstance(name, str)]
    # The two channels have different casts, and merging them misattributes.
    # A book's chat participants are usually a handful of the prose cast, so a
    # prose-only name in the chat regex turns any line shaped "Name: text" -
    # which ordinary prose produces - into a chat message from someone who
    # never posts. Measured on a real manuscript, that credited 46 chat lines
    # to a character with zero. chat_speakers falls back to speakers, so a
    # project with one cast configures one list and nothing changes for it.
    chat_speakers = [name for name in settings.get("chat_speakers", speakers)
                     if isinstance(name, str)]
    titles = settings.get("titles", DEFAULT_TITLES)
    speech_verbs = settings.get("speech_verbs", DEFAULT_SPEECH_VERBS)
    hedges = settings.get("hedges", DEFAULT_HEDGES)
    min_lines = args.min_lines if args.min_lines is not None else settings.get("min_lines", DEFAULT_MIN_LINES)

    if not paths:
        print("no chapter files found; pass file/directory arguments or set chapters_dir")
        return 0

    note_suffix = "" if speakers else ("  (no project_measures.voice_separation.speakers "
                                       "configured; speakers were auto-discovered and may be noisy)")
    # An EMPTY chat_speakers is not "nothing to attribute": collect_chat falls
    # back to discovery whenever its list is empty, however it got that way,
    # so the note has to say discovery rather than silence.
    discovery_note = ("  (speakers were auto-discovered from generic 'name: message' "
                      "lines and may be noisy)")
    if settings.get("chat_speakers") is None:
        chat_note_suffix = note_suffix
    else:
        chat_note_suffix = "" if chat_speakers else discovery_note
    both = not (args.chat or args.prose)
    if args.chat or both:
        show("GROUP CHAT", collect_chat(paths, chat_speakers), min_lines,
             "exact attribution, every message counted" + chat_note_suffix, hedges)
    if args.prose or both:
        show("PROSE DIALOGUE", collect_prose(paths, speakers, titles, speech_verbs), min_lines,
             "tagged lines only, biased short AND against questions - never read a 0 as real" + note_suffix,
             hedges)
    if not speakers:
        print("\n  project_measures.voice_separation.speakers would turn discovery into exact "
              "attribution for both channels above.")
    elif settings.get("chat_speakers") is None and (args.chat or both):
        print("\n  the group chat above was attributed with the full prose cast. If only some "
              "of them post, set project_measures.voice_separation.chat_speakers: a prose-only "
              "name matches any line shaped 'Name: text' and credits messages to someone who "
              "never posts.")
    return 0


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
