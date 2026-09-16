#!/usr/bin/env python3
"""Prose statistics and tic scan for one or more chapters.

Every band or target printed as PASS/FAIL below comes from
``project_measures.style_report.targets`` in the user's config. With no
targets configured, every column is measured and printed and marked
"not set" instead of PASS or FAIL, since there is no basis to judge a book
against nobody's stated preference.

    python3 style_report.py chapters/01_before.md
    python3 style_report.py chapters/01_before.md "Ch1 draft 3"
    python3 style_report.py chapters/*.md          # one report per file
    python3 style_report.py --summary chapters/*.md

Config (``project_measures.style_report.targets``), all optional:

    "sentence_mean": [11, 18],
    "sentence_cv": [68, 100],
    "sentence_bands": {"under_10": [low, high], "10_20": [...],
                       "20_35": [...], "over_35": [...]},
    "quoted_share": [26, 34],
    "spoken_mean": [8, 10],
    "narration_mean": [14, 17],
    "narration_under_10_max": 40,
    "paragraph_sentence_mean": [2.9, 3.5],
    "paragraph_sentence_cv_min": 100,
    "paragraph_distribution_reference": {"1": 36.7, "2": 24.0, ...},
    "conjunctions": {"and": [2.5, 3.5], "but": [0.25, 0.75], ...},
    "multi_and_max": 10,
    "section_breaks": [2, 6]

Only deliberate change from the script this began as: numpy is replaced with
the standard library. np.median, np.mean and np.std(ddof=1) are
statistics.median, fmean and stdev exactly, so the numbers are unaffected and
the script has no third-party dependency.
"""

import re
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, median, stdev

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import project_config

ABBR = r'(?:Mrs|Mr|Ms|Dr|St|Jr|Sr|vs|etc|[A-Z])'

# The four sentence-length buckets and the two paragraph-count buckets are
# structural: every book has a mode, a median, a mean and some spread of
# paragraph lengths, whatever they turn out to be. Only the bands judged
# against are project policy, so only the bands move into config.
SENTENCE_BUCKETS = [
    ("under_10", lambda value: value < 10),
    ("10_20", lambda value: 10 <= value < 20),
    ("20_35", lambda value: 20 <= value <= 35),
    ("over_35", lambda value: value > 35),
]
PARAGRAPH_BUCKETS = [
    ("1", lambda value: value == 1), ("2", lambda value: value == 2),
    ("3", lambda value: value == 3), ("4", lambda value: value == 4),
    ("5", lambda value: value == 5), ("6-7", lambda value: 6 <= value <= 7),
    ("8-9", lambda value: 8 <= value <= 9), ("10+", lambda value: value >= 10),
]
DEFAULT_CONJUNCTIONS = ("and", "but", "so", "because", "which")


def in_ten_to_twenty(word_count):
    return 10 <= word_count < 20


def in_twenty_to_thirty_five(word_count):
    return 20 <= word_count <= 35


def load(path):
    return open(path, encoding='utf-8').read()


def hard_breaks(text):
    """True if the text uses markdown hard line breaks to end paragraphs."""
    return any(line.endswith('  ') for line in text.split('\n') if line.strip())


def _clean(blk):
    blk = blk.strip()
    if not blk or blk == '---':
        return []
    blk = blk.replace('---', ' ').strip()
    return [re.sub(r'\s+', ' ', blk)] if blk else []


def paragraphs(text):
    """Split into paragraphs on blank lines AND on markdown hard line breaks.

    Splitting inside each blank-line block, rather than picking one
    convention for the whole file, keeps a file that mixes both conventions
    honest. A plain newline is a soft wrap and does not end a paragraph, so
    conventionally wrapped text still measures correctly.
    """
    text = re.sub(r'^#.*$', '', text, flags=re.M)
    out = []
    for blk in re.split(r'\n\s*\n', text):
        current = []
        for line in blk.split('\n'):
            current.append(line)
            if line.endswith('  '):
                out.extend(_clean(' '.join(current)))
                current = []
        if current:
            out.extend(_clean(' '.join(current)))
    return out


def sents(text):
    text = re.sub(rf'\b({ABBR})\.', r'\1<DOT>', text)
    text = re.sub(r'([.!?])(["”\']?)\s+', r'\1\2|', text)
    return [part.strip().replace('<DOT>', '.') for part in text.split('|') if part.strip()]


def words(text):
    return re.findall(r"[A-Za-z][A-Za-z']*", text)


def quoted_words(body):
    """Count words inside double quotes, counting curly quotes as quotes."""
    body = body.replace('“', '"').replace('”', '"')
    return sum(len(words(match))
               for para in body.split('\n')
               for match in re.findall(r'"[^"]*"', para))


def split_speech(paras):
    """Split into (spoken, narration) sentence word-counts.

    Anything inside double quotes is spoken; everything else, including the
    "she says" tags, is narration.
    """
    spoken, narration = [], []
    for paragraph in paras:
        paragraph = paragraph.replace('“', '"').replace('”', '"')
        pos, told, utterances, current = 0, [], [], []
        for match in re.finditer(r'"[^"]*"', paragraph):
            gap = paragraph[pos:match.start()]
            told.append(gap)
            # A split quote ("A," she says, "B.") is one utterance and must be
            # rejoined. Two utterances ("A," Ruth says. "B," Sam says.) must
            # not be. The difference is whether the narration between them
            # closes a sentence.
            if current and re.search(r'[.!?]', gap):
                utterances.append(' '.join(current))
                current = []
            current.append(match.group(0).strip('"'))
            pos = match.end()
        if current:
            utterances.append(' '.join(current))
        told.append(paragraph[pos:])
        for chunk in utterances:
            spoken.extend(length for length in (len(words(sentence)) for sentence in sents(chunk)) if length)
        narration.extend(
            length for length in (len(words(sentence)) for sentence in sents(' '.join(told))) if length)
    return spoken, narration


def _range(targets, key):
    value = targets.get(key)
    return tuple(value) if isinstance(value, (list, tuple)) and len(value) == 2 else None


def _status(bounds, value):
    if bounds is None:
        return 'not set'
    lower_bound, upper_bound = bounds
    return 'PASS' if lower_bound <= value <= upper_bound else 'FAIL'


def _max_status(maximum, value, below=True):
    if maximum is None:
        return 'not set'
    return 'PASS' if (value < maximum if below else value >= maximum) else 'FAIL'


def report(path, label, targets):
    text = load(path)
    paras = paragraphs(text)
    sentences_per_paragraph = [len(sents(paragraph)) for paragraph in paras]
    sentence_lengths = [len(words(sentence)) for paragraph in paras for sentence in sents(paragraph)]
    word_lengths = [len(word) for paragraph in paras for word in words(paragraph)]
    body = re.sub(r'^#.*$', '', text, flags=re.M).replace('---', ' ')
    total_words = len(words(body))
    quoted_word_count = quoted_words(body)

    if not paras or not sentence_lengths or not total_words:
        print("=" * 66)
        print(f"{label} | {total_words} words | {len(paras)} paras | {len(sentence_lengths)} sentences")
        print("\nnot enough text to report on")
        return

    quoted_share = 100 * quoted_word_count / total_words
    mode = Counter(sentence_lengths).most_common(1)[0][0]
    sentence_length_deviation = stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    sentence_length_variation = 100 * sentence_length_deviation / fmean(sentence_lengths)
    sentence_mean_bounds = _range(targets, "sentence_mean")
    sentence_cv_bounds = _range(targets, "sentence_cv")

    print("=" * 66)
    print(f"{label} | {total_words} words | {len(paras)} paras | {len(sentence_lengths)} sentences"
          f"{'  [paragraphs split on hard line breaks]' if hard_breaks(text) else ''}")
    print(f"\nWORDS/SENTENCE  mode {mode}  median {median(sentence_lengths):.0f}  mean {fmean(sentence_lengths):.2f}  "
          f"sd {sentence_length_deviation:.2f}  CV {sentence_length_variation:.1f}%  max {max(sentence_lengths)}")
    print(f"  mode<median<mean  {'PASS' if mode < median(sentence_lengths) < fmean(sentence_lengths) else 'FAIL'}     "
          f"mean {'-'.join(map(str, sentence_mean_bounds)) if sentence_mean_bounds else 'not set'}  "
          f"{_status(sentence_mean_bounds, fmean(sentence_lengths))}     "
          f"CV {'-'.join(map(str, sentence_cv_bounds)) if sentence_cv_bounds else 'not set'}%  "
          f"{_status(sentence_cv_bounds, sentence_length_variation)}")
    sentence_bands = targets.get("sentence_bands", {}) if isinstance(targets.get("sentence_bands"), dict) else {}
    for name, predicate in SENTENCE_BUCKETS:
        pct = 100 * sum(1 for value in sentence_lengths if predicate(value)) / len(sentence_lengths)
        bounds = _range(sentence_bands, name)
        target_text = f"{bounds[0]}-{bounds[1]}%" if bounds else "not set"
        print(f"  {name:>8}: {pct:5.1f}%  target {target_text:<10}  {_status(bounds, pct)}")

    quoted_bounds = _range(targets, "quoted_share")
    target_text = f"{quoted_bounds[0]}-{quoted_bounds[1]}%" if quoted_bounds else "not set"
    print(f"\nQUOTED  {quoted_share:.1f}%  target {target_text}  {_status(quoted_bounds, quoted_share)}")

    # Spoken and narration measured separately, since a combined number hides
    # which half is broken.
    spoken, narration = split_speech(paras)
    if spoken and narration:
        spoken_mean, narration_mean = fmean(spoken), fmean(narration)
        short_narration_share = 100 * sum(1 for value in narration if value < 10) / len(narration)
        spoken_bounds = _range(targets, "spoken_mean")
        narration_bounds = _range(targets, "narration_mean")
        narration_under_10_max = targets.get("narration_under_10_max")
        print(f"\nSPOKEN vs NARRATION")
        print(f"  spoken     mean {spoken_mean:5.2f} words  ({len(spoken):4d} sentences)  "
              f"target {'-'.join(map(str, spoken_bounds)) if spoken_bounds else 'not set'}   "
              f"{_status(spoken_bounds, spoken_mean)}")
        print(f"  narration  mean {narration_mean:5.2f} words  ({len(narration):4d} sentences)  "
              f"target {'-'.join(map(str, narration_bounds)) if narration_bounds else 'not set'}  "
              f"{_status(narration_bounds, narration_mean)}")
        print(f"  narration under 10 words  {short_narration_share:.1f}%   "
              f"target {'<' + str(narration_under_10_max) + '%' if narration_under_10_max is not None else 'not set'}  "
              f"{_max_status(narration_under_10_max, short_narration_share)}")

    paragraph_length_deviation = stdev(sentences_per_paragraph) if len(sentences_per_paragraph) > 1 else 0.0
    variation = 100 * paragraph_length_deviation / fmean(sentences_per_paragraph)
    paragraph_mean_bounds = _range(targets, "paragraph_sentence_mean")
    paragraph_cv_min = targets.get("paragraph_sentence_cv_min")
    print(f"\nSENT/PARA  mean {fmean(sentences_per_paragraph):.2f}  median {median(sentences_per_paragraph):.0f}  "
          f"sd {paragraph_length_deviation:.2f}  CV {variation:.1f}%  max {max(sentences_per_paragraph)}")
    print(f"  mean {'-'.join(map(str, paragraph_mean_bounds)) if paragraph_mean_bounds else 'not set'} "
          f"{_status(paragraph_mean_bounds, fmean(sentences_per_paragraph))}   "
          f"CV>={paragraph_cv_min if paragraph_cv_min is not None else 'not set'}% "
          f"{_max_status(paragraph_cv_min, variation, below=False)}")
    reference = targets.get("paragraph_distribution_reference", {})
    reference = reference if isinstance(reference, dict) else {}
    print(f"  bucket  this   {'reference' if reference else '(no reference configured)'}")
    for name, predicate in PARAGRAPH_BUCKETS:
        pct = 100 * sum(1 for value in sentences_per_paragraph if predicate(value)) / len(sentences_per_paragraph)
        reference_text = f"{reference[name]:6.1f}%" if name in reference else "    n/a"
        print(f"   {name:>4} {pct:6.1f}% {reference_text}")

    print(f"\nWORD LEN  mean {fmean(word_lengths):.2f}   "
          f">=7 chars {100 * sum(1 for length in word_lengths if length >= 7) / len(word_lengths):.1f}%")

    lowercase_words = [word.lower() for word in words(body)]
    word_counts = Counter(lowercase_words)
    total_words = len(lowercase_words)
    conjunctions = targets.get("conjunctions", {}) if isinstance(targets.get("conjunctions"), dict) else {}
    print("\nCONJUNCTIONS (% of all words)")
    for word in conjunctions or DEFAULT_CONJUNCTIONS:
        pct = 100 * word_counts[word] / total_words
        bounds = _range(conjunctions, word)
        target_text = f"{bounds[0]}-{bounds[1]}%" if bounds else "not set"
        print(f"  {word:<8} {word_counts[word]:4d}  {pct:5.2f}%   target {target_text:<10}   {_status(bounds, pct)}")
    multiple_and_count = sum(1 for sentence in [candidate for paragraph in paras for candidate in sents(paragraph)]
                 if len(re.findall(r'\band\b', sentence.lower())) >= 2)
    multi_and_share = 100 * multiple_and_count / len(sentence_lengths)
    multi_and_max = targets.get("multi_and_max")
    print(f"  sentences with 2+ 'and': {multi_and_share:.1f}%   "
          f"target {'<' + str(multi_and_max) + '%' if multi_and_max is not None else 'not set'}  "
          f"{_max_status(multi_and_max, multi_and_share)}")

    raw_text = load(path)
    section_breaks = len([length for length in raw_text.split('\n') if set(length.strip()) == {'_'}])
    breaks_bounds = _range(targets, "section_breaks")
    target_text = f"{breaks_bounds[0]}-{breaks_bounds[1]}" if breaks_bounds else "not set"
    print(f"\nSECTION BREAKS  {section_breaks}   target {target_text}   {_status(breaks_bounds, section_breaks)}")

    # Rule 1 governs the narrator, so quoted spans come out before matching,
    # exactly as summarise() does it. Without this a fixing agent is handed
    # every character's own explanatory clause as a violation.
    print("\nTIC SCAN, narration only (quoted spans removed)")
    tic_hits = 0
    for paragraph in paras:
        for sentence in sents(narration_of(paragraph)):
            for pattern, name in TICS:
                if re.search(pattern, sentence, re.I):
                    print(f"  [{name}] {sentence[:110]}")
                    tic_hits += 1
    if not tic_hits:
        print("  none found")


def narration_of(para):
    """A paragraph with everything spoken taken out of it.

    Quoted speech is stripped at paragraph level rather than sentence level,
    because a speech that runs to several sentences leaves its interior
    sentences carrying no quote mark at all. Group-chat lines
    (``name: message``) carry no quote marks either, so they are dropped the
    same way transcript detection elsewhere in the project does it.
    """
    para = re.sub(r'"[^"]*"', ' ', para)
    kept = [line for line in para.split('\n')
            if not re.match(r'^[a-z][a-z0-9_]*: ', line.strip())]
    return '\n'.join(kept)


# Generic narrator-voice tics: an evaluative aside, a superlative, a
# not-X-but-Y construction, an em dash. These are craft heuristics rather
# than facts about any one manuscript, so unlike the target bands above they
# stay as module defaults; there is no numeric threshold to configure, only
# a count to read.
TICS = [
    (r'which is (?:the|what|how|why|a|his|her|its)\b', 'narrator evaluation'),
    (r',\s+which is ', 'trailing explanatory clause'),
    (r',\s+(?:so|because|since)\s+(?:whoever|anybody|anyone|everybody|everyone|'
     r'nobody|the rest|people|you|it|the)\b',
     'so/because tail (triage, read each)'),
    (r'\b(best|worst|funniest|nicest|only time|never once|for the first time)\b',
     'superlative'),
    (r"it'?s not \w+[,;] it'?s ", 'not-X-but-Y'),
    (r'—', 'em dash'),
]


def summarise(paths, targets):
    """Every measure in this script, over the whole book, in one block."""
    text = '\n'.join(load(path) for path in paths)
    body = '\n'.join(line for line in text.split('\n') if not line.startswith('#'))
    all_paras = paras_of(body)
    sents_all = [sentence for paragraph in all_paras for sentence in sents(paragraph)]
    sentence_lengths = [len(words(sentence)) for sentence in sents_all]
    sentences_per_paragraph = [len(sents(paragraph)) for paragraph in all_paras]
    word_lengths = [len(word) for word in words(body)]
    lowercase_words = [word.lower() for word in words(body)]
    word_count = len(lowercase_words)
    counts = Counter(lowercase_words)

    print(f"\n  {len(lowercase_words):,} words   {len(all_paras):,} paragraphs   "
          f"{len(sents_all):,} sentences")

    mode = Counter(sentence_lengths).most_common(1)[0][0]
    median_value, mean_value, standard_deviation = median(sentence_lengths), fmean(sentence_lengths), stdev(sentence_lengths)
    variation = 100 * standard_deviation / mean_value
    sentence_mean_bounds = _range(targets, "sentence_mean")
    sentence_cv_bounds = _range(targets, "sentence_cv")
    print("\n  WORDS PER SENTENCE")
    print(f"    mode {mode}   median {median_value:g}   mean {mean_value:.2f}   sd {standard_deviation:.2f}   "
          f"CV {variation:.1f}%   max {max(sentence_lengths)}")
    print(f"    mode<median<mean {'PASS' if mode < median_value < mean_value else 'FAIL':<6}"
          f"mean {_status(sentence_mean_bounds, mean_value):<6}"
          f"CV {_status(sentence_cv_bounds, variation)}")
    sentence_bands = targets.get("sentence_bands", {}) if isinstance(targets.get("sentence_bands"), dict) else {}
    for name, predicate in SENTENCE_BUCKETS:
        pct = 100 * sum(1 for value in sentence_lengths if predicate(value)) / len(sentence_lengths)
        bounds = _range(sentence_bands, name)
        target_text = f"{bounds[0]}-{bounds[1]}%" if bounds else "not set"
        print(f"    {name:<10}{pct:>6.1f}%   target {target_text:<12}   {_status(bounds, pct)}")

    quoted_word_count = sum(len(words(match)) for match in re.findall(r'"([^"]+)"', body))
    quoted_share = 100 * quoted_word_count / word_count
    quoted_bounds = _range(targets, "quoted_share")
    target_text = f"{quoted_bounds[0]}-{quoted_bounds[1]}%" if quoted_bounds else "not set"
    print(f"\n  QUOTED  {quoted_share:.1f}%   target {target_text}   {_status(quoted_bounds, quoted_share)}")

    spoken_lengths = [len(words(sentence)) for sentence in sents_all if sentence.lstrip().startswith('"')]
    narrated_lengths = [len(words(sentence)) for sentence in sents_all if not sentence.lstrip().startswith('"')]
    print("\n  SPOKEN vs NARRATION")
    if spoken_lengths:
        print(f"    spoken     mean {fmean(spoken_lengths):5.2f} words  ({len(spoken_lengths):5d} sentences)")
    if narrated_lengths:
        print(f"    narration  mean {fmean(narrated_lengths):5.2f} words  ({len(narrated_lengths):5d} sentences)")
        short = 100 * sum(1 for value in narrated_lengths if value < 10) / len(narrated_lengths)
        narration_under_10_max = targets.get("narration_under_10_max")
        print(f"    narration under 10 words  {short:.1f}%   "
              f"target {'<' + str(narration_under_10_max) + '%' if narration_under_10_max is not None else 'not set'}   "
              f"{_max_status(narration_under_10_max, short)}")

    print(f"\n  SENTENCES PER PARAGRAPH  mean {fmean(sentences_per_paragraph):.2f}   "
          f"median {median(sentences_per_paragraph):g}   max {max(sentences_per_paragraph)}")
    reference = targets.get("paragraph_distribution_reference", {})
    reference = reference if isinstance(reference, dict) else {}
    print(f"    {'bucket':>8}{'this':>8}{'reference' if reference else '(none)':>10}")
    for name, predicate in PARAGRAPH_BUCKETS:
        pct = 100 * sum(1 for value in sentences_per_paragraph if predicate(value)) / len(sentences_per_paragraph)
        reference_text = f"{reference[name]:>9.1f}%" if name in reference else "      n/a"
        print(f"    {name:>8}{pct:>7.1f}%{reference_text}")

    print(f"\n  WORD LENGTH  mean {fmean(word_lengths):.2f}   "
          f"7+ chars {100 * sum(1 for word in word_lengths if word >= 7) / len(word_lengths):.1f}%")

    conjunctions = targets.get("conjunctions", {}) if isinstance(targets.get("conjunctions"), dict) else {}
    print("\n  CONJUNCTIONS")
    for word in conjunctions or DEFAULT_CONJUNCTIONS:
        pct = 100 * counts[word] / word_count
        bounds = _range(conjunctions, word)
        target_text = f"{bounds[0]}-{bounds[1]}%" if bounds else "not set"
        print(f"    {word:<10}{counts[word]:>7}{pct:>9.2f}%   target {target_text:<10}   "
              f"{_status(bounds, pct):<6}")
    multi = sum(1 for sentence in sents_all
                if len(re.findall(r'\band\b', sentence.lower())) >= 2)
    multiple_and_share = 100 * multi / len(sents_all)
    multi_and_max = targets.get("multi_and_max")
    print(f"    {'2+ and':<10}{multi:>7}{multiple_and_share:>9.1f}%   "
          f"target {'<' + str(multi_and_max) + '%' if multi_and_max is not None else 'not set':<10}   "
          f"{_max_status(multi_and_max, multiple_and_share)}")

    breaks = len([word for word in text.split('\n') if set(word.strip()) == {'_'}])
    breaks_bounds = _range(targets, "section_breaks")
    target_text = f"{breaks_bounds[0]}-{breaks_bounds[1]} each" if breaks_bounds else "not set"
    print(f"\n  SECTION BREAKS  {breaks} across {len(paths)} chapters   target {target_text}")

    narration = [sentence for paragraph in all_paras for sentence in sents(narration_of(paragraph))]
    print("\n  TIC SCAN, narration only (quoted spans removed)")
    for pattern, name in TICS:
        hits = sum(1 for sentence in narration if re.search(pattern, sentence, re.I))
        if hits:
            print(f"    {name:<40}{hits:>5}")


def paras_of(body):
    return [paragraph for paragraph in re.split(r'\n\s*\n', body) if paragraph.strip()]


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


# Run from grade.py, not on its own. Each script in measures/ reports one
# diagnostic; grade.py assembles enabled checks and is the interface that says
# whether a pass helped. Running one of these alone is for reading the
# individual hits during a fix, which is what --show and the per-file
# arguments are for, and it is never how a pass gets judged.
def _solo_notice():
    import os
    if os.environ.get("HALSTEAD_VIA_GRADE"):
        return
    print("  [bundled diagnostic; use grade.py for the structured report]",
          file=sys.stderr)


def main(argv):
    config_path = None
    if "--config" in argv:
        index = argv.index("--config")
        config_path = argv[index + 1]
        argv = argv[:index] + argv[index + 2:]

    config = project_config.load_config(config_path)
    settings = project_config.measure_settings("style_report", config)
    targets = settings.get("targets", {})
    targets = targets if isinstance(targets, dict) else {}
    if not targets:
        print("no project_measures.style_report.targets configured; every band below "
              "is measured and printed, and marked 'not set' rather than PASS/FAIL.\n")
    chapters_dir = project_config.project_path("chapters_dir", "chapters", config)

    if not argv:
        sys.exit(f"usage: {sys.argv[0]} <chapter.md> [label]\n"
                 f"       {sys.argv[0]} --summary [chapters/*.md]")
    if argv[0] == '--summary':
        paths = resolve_paths([Path(item) for item in argv[1:]], chapters_dir)
        if not paths:
            sys.exit("no chapters found; pass file/directory arguments or set chapters_dir")
        summarise(paths, targets)
        return 0
    # One path plus an optional label, as the original took. More than one
    # path and they are all treated as files, so a glob reports on each in turn.
    if len(argv) == 2 and not argv[1].endswith('.md'):
        report(argv[0], argv[1], targets)
    else:
        for item in argv:
            report(item, item.split('/')[-1], targets)
    return 0


if __name__ == '__main__':
    _solo_notice()
    sys.exit(main(sys.argv[1:]) or 0)
