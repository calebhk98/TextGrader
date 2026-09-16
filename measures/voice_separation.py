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

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import project_config
from textgrader.text import TranscriptConfig, transcript_lines

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
    # small enough for dialogue samples; incomplete trailing windows are kept.
    window = 50
    segments = [flat[index:index + window] for index in range(0, len(flat), window)]
    msttr = st.fmean(len(set(segment)) / len(segment) for segment in segments)
    return {
        "n": len(lines),
        "words": len(flat),
        "wpl": len(flat) / len(lines),
        "ttr": 100 * msttr,
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
          f"{'quest%':>8}{'1-3w%':>7}{'>15w%':>7}{'hedge%':>8}")
    for speaker, speaker_profile in sorted(rows.items(), key=lambda kv: -kv[1]["wpl"]):
        print(f"  {speaker:10}{speaker_profile['n']:>7}{speaker_profile['words']:>7}{speaker_profile['wpl']:>8.1f}{speaker_profile['ttr']:>7.1f}"
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
    titles = settings.get("titles", DEFAULT_TITLES)
    speech_verbs = settings.get("speech_verbs", DEFAULT_SPEECH_VERBS)
    hedges = settings.get("hedges", DEFAULT_HEDGES)
    min_lines = args.min_lines if args.min_lines is not None else settings.get("min_lines", DEFAULT_MIN_LINES)

    if not paths:
        print("no chapter files found; pass file/directory arguments or set chapters_dir")
        return 0

    note_suffix = "" if speakers else ("  (no project_measures.voice_separation.speakers "
                                       "configured; speakers were auto-discovered and may be noisy)")
    both = not (args.chat or args.prose)
    if args.chat or both:
        show("GROUP CHAT", collect_chat(paths, speakers), min_lines,
             "exact attribution, every message counted" + note_suffix, hedges)
    if args.prose or both:
        show("PROSE DIALOGUE", collect_prose(paths, speakers, titles, speech_verbs), min_lines,
             "tagged lines only, biased short AND against questions - never read a 0 as real" + note_suffix,
             hedges)
    if not speakers:
        print("\n  project_measures.voice_separation.speakers would turn discovery into exact "
              "attribution for both channels above.")
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
