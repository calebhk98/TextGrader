#!/usr/bin/env python3
"""How often a speech ends on a maxim, against the corpus.

The complaint this makes countable is that a book has one gear for climactic
dialogue: every character, at the moment a scene turns, resolves the beat into
a tight symmetrical line that would look at home on a poster. Readers describe
it as everyone sounding like the same aphorist.

The symmetry itself is hard to count and turns out to be rare in most prose.
What is countable is the move underneath it: the last sentence of a speech
stops being about the people in the room and becomes a general statement about
how the world works - present tense, no proper nouns, an indefinite subject, a
copula. A particular becomes a maxim and the scene signs off on it.

The rate is reported against the corpus, and the flagged lines are listed
because the list is what earns its keep. The shape is a heuristic, not a
judgement: a maxim can be exactly the right ending, and only a reader can say
which ones are. A handful of lines is small enough to judge one at a time,
which is the intended use. The usual fix for a line that should not have been
a maxim is to end it on a particular rather than a general claim - name the
specific thing or person, rather than asserting what people in general do.

Corpus rates for this are low and the spread between books is narrow, so a
document sitting near the top of the range is not far from ordinary. Read it
as "go and look at these ten lines", not as a target to drive to zero.

    python3 quotable.py            report
    python3 quotable.py --corpus   the per-book corpus table
"""
import argparse, glob, re, statistics, sys
from pathlib import Path

from . import solo_notice
from .. import project as project_config
from ..text import strip_gutenberg

WORD = re.compile(r"[A-Za-z']+")
QUOTE = re.compile(r'["“]([^"“”]{25,1500})["”]')

# An indefinite subject, or a quantifier that makes the claim general.
GENERIC = re.compile(r"\b(nobody|no one|everybody|everyone|somebody|someone|"
                     r"anybody|anyone|nothing|everything|anything|people|"
                     r"always|never|every|a person|most people)\b", re.I)

# A maxim asserts. It needs a present-tense verb of state or of what happens.
COPULA = re.compile(r"\b(is|isn't|are|aren't|means|matters|counts|works|"
                    r"happens|costs|takes|comes down to)\b", re.I)

# Any past tense at all anchors the sentence to this scene, which is the
# opposite of the shape being counted.
PAST = re.compile(r"\b(was|were|had|did|said|went|came|took|got|saw|knew|"
                  r"told|made|felt|looked|turned|walked|asked|gave)\b"
                  r"|\b\w+ed\b", re.I)

# A name makes it a statement about a person rather than about the world.
PROPER = re.compile(r"\b[A-Z][a-z]{2,}\b")

# "You" used for anybody at all, rather than for the person being spoken to.
IMPERSONAL_YOU = re.compile(r"\byou (can|can't|cannot|don't|do|have to|get|"
                            r"give|need|end up|start|stop|say|ask|know)\b", re.I)

# A speaker undertaking to do something is talking about this scene, however
# many indefinite pronouns the sentence happens to contain.
PROMISE = re.compile(r"\b(I'll|I will|we'll|we will|I'm going to|"
                     r"we're going to)\b", re.I)


def closers(text):
    """The last sentence of every speech long enough to have arrived at one."""
    out = []
    for spoken in QUOTE.findall(text):
        parts = [sentence.strip() for sentence in re.split(r'(?<=[.!?])\s+', spoken) if sentence.strip()]
        if len(parts) >= 2:
            out.append(parts[-1])
    return out


def is_maxim(sentence):
    # A question asks; it does not pronounce. Four of the first fourteen hits
    # were questions, and dropping them is what moved this from a rate the
    # corpus could match to one it cannot.
    if sentence.rstrip().endswith("?"):
        return False
    word_count = len(WORD.findall(sentence))
    if not 4 <= word_count <= 22:
        return False
    if PAST.search(sentence):
        return False
    # The opening word is capitalised because it opens; that is not a name.
    if PROPER.search(sentence[0].lower() + sentence[1:] if sentence else sentence):
        return False
    if PROMISE.search(sentence):
        return False
    # A list of particulars is not a general claim, whatever its last clause
    # does. Two or more commas with no verb before the first one is a list.
    head = sentence.split(",")[0]
    if sentence.count(",") >= 2 and not COPULA.search(head):
        return False
    if not COPULA.search(sentence):
        return False
    return bool(GENERIC.search(sentence) or IMPERSONAL_YOU.search(sentence))


def measure(paths):
    total = 0
    hits = []
    for paragraph in paths:
        text = Path(paragraph).read_text(errors="ignore")
        for candidate in closers(text):
            total += 1
            if is_maxim(candidate):
                hits.append((Path(paragraph).stem, candidate))
    return hits, total


def measure_text(text, source="text"):
    candidates = closers(text)
    hits = [(source, candidate) for candidate in candidates if is_maxim(candidate)]
    return hits, len(candidates)


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


def corpus_rates(corpus_dirs):
    rows = []
    for directory in corpus_dirs:
        for text_path in sorted(glob.glob(str(directory / "*"))):
            # The corpus carries a stripped copy of each modern book; counting
            # both would weight those authors twice.
            if "strip" in Path(text_path).name:
                continue
            # Corpus files can include distribution boilerplate which is not prose.
            cleaned = strip_gutenberg(Path(text_path).read_text(encoding="utf-8", errors="replace"))
            hits, total = measure_text(cleaned, Path(text_path).stem)
            if total >= 50:
                rows.append((Path(text_path).stem, len(hits), total, len(hits) / total * 100))
    rows.sort(key=lambda r: r[3])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path,
                    help="chapter files or directories; default: the configured chapters directory")
    parser.add_argument("--config", help="path to a config.json "
                    "(default: $TEXTGRADER_CONFIG, or the repo's own)")
    parser.add_argument("--corpus", action="store_true", help="the per-book corpus table")
    args = parser.parse_args()

    config = project_config.load_config(args.config)
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)
    corpus_dirs = tuple((Path(config["_config_dir"]) / value).resolve()
                        for value in config.get("corpus_dirs", []))
    chapters = resolve_paths(args.paths, chapters_dir)

    rows = corpus_rates(corpus_dirs)
    if not rows:
        print("  corpus texts not readable; nothing to compare against")
        return 0
    ceiling = max(row[3] for row in rows)
    median = statistics.median(row[3] for row in rows)

    if args.corpus:
        for stem, hits, total, pct in rows:
            print(f"  {stem[:38]:38s} {hits:3d}/{total:5d}  {pct:5.2f}%")
        print(f"\n  median {median:.2f}%   maximum {ceiling:.2f}%   {len(rows)} books")
        return 0

    if not chapters:
        print("  no chapters found")
        return 0
    hits, total = measure(chapters)
    pct = len(hits) / total * 100 if total else 0.0

    print(f"  {len(hits)} of {total} multi-sentence speeches end on a maxim: "
          f"{pct:.2f}%")
    print(f"  corpus median {median:.2f}%, corpus maximum {ceiling:.2f}% "
          f"across {len(rows)} books\n")

    for stem, line in hits:
        print(f"  {stem[:22]:22s} {line}")

    chapters = len({sentence for sentence, _ in hits})
    print(f"\n  spread across {chapters} chapters. The habit belongs to no one "
          f"character,")
    print(f"  which is the reader's actual complaint: one gear, everybody in it.")

    if pct > ceiling:
        allowed = int(ceiling * total / 100)
        print(f"\n  {pct:.2f}% is over corpus max of {ceiling:.2f}%. "
              f"CUT to {allowed} or fewer.\n")
        return 1
    print(f"\n  {pct:.2f}% is at or under the corpus maximum: pass\n")
    return 0


# Run from grade.py, not on its own. Each script in measures/ reports one
# diagnostic; grade.py assembles enabled checks and is the interface that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --corpus is for, and it is never
# how a pass gets judged.


if __name__ == "__main__":
    solo_notice()
    sys.exit(main() or 0)
