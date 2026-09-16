#!/usr/bin/env python3
"""Prose statistics and tic scan for a Halstead chapter.

This is the script embedded at the end of STYLE_GUIDES.md, reconstructed into a
runnable file. That copy was pasted through a markdown escaper, which put
backslashes in front of most operators ("t \\= load(path)") and mangled the
regex character classes, so it could not run as printed. Thresholds, output
format and metric definitions are unchanged from it.

Only deliberate change: numpy is replaced with the standard library. np.median,
np.mean and np.std(ddof=1) are statistics.median, fmean and stdev exactly, so
the numbers are identical and the script has no dependencies.

    python3 style_report.py chapters/01_before.md
    python3 style_report.py chapters/01_before.md "Ch1 draft 3"
    python3 style_report.py chapters/*.md          # one report per file
"""

import re
import sys
from collections import Counter
from statistics import fmean, median, stdev

ABBR = r'(?:Mrs|Mr|Ms|Dr|St|Jr|Sr|vs|etc|[A-Z])'


def in_ten_to_twenty(word_count):
    """Return whether a sentence belongs to the 10–under-20 bucket."""
    return 10 <= word_count < 20


def in_twenty_to_thirty_five(word_count):
    """Return whether a sentence belongs to the inclusive 20–35 bucket."""
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

    The original script split on blank lines only. Chapters 1-6 of the
    manuscript separate paragraphs with two trailing spaces and a single
    newline instead, so on those it saw one paragraph per chapter and every
    per-paragraph number was meaningless (chapter 1: "1 paras", 321
    sentences/paragraph). Splitting inside each blank-line block, rather than
    picking one convention per file, also keeps a mixed file honest, which
    The original manuscript is: chapters 1-6 hard-break, 7-20 blank-line.

    A plain newline is a soft wrap and does not end a paragraph, so
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
    """Count words inside double quotes, counting curly quotes as quotes.

    The original scanned the whole file with '"[^"]*"'. That regex cannot see
    the 12 manuscript paragraphs that use curly quotes, so straight
    quotes on either side of them paired across the intervening narration and
    counted it as dialogue. The manuscript came out at 38.2% quoted against a
    ~30% target, a FAIL, when the real figure is 28.1%, a PASS.

    Normalising the curly quotes fixes that. Restarting the scan at each
    paragraph keeps any future unbalanced quote from inverting dialogue and
    narration for the whole rest of the file.
    """
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
            # rejoined. Two utterances ("A," Ruth says. "B," Sam says.) must not
            # be. The difference is whether the narration between them closes a
            # sentence. Joining every quoted span in a paragraph, as this did
            # until now, glued separate speakers together and inflated the
            # spoken mean wherever a chapter put several voices in one
            # paragraph: chapter 6 had single "sentences" 83 words long made of
            # three people talking.
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


def report(path, label):
    text = load(path)
    paras = paragraphs(text)
    sentences_per_paragraph = [len(sents(paragraph)) for paragraph in paras]
    sentence_lengths = [len(words(sentence)) for paragraph in paras for sentence in sents(paragraph)]
    word_lengths = [len(word) for paragraph in paras for word in words(paragraph)]
    body = re.sub(r'^#.*$', '', text, flags=re.M).replace('---', ' ')
    total_words = len(words(body))
    quoted_word_count = quoted_words(body)

    # Guards the original lacked: every statistic below divides by one of these.
    if not paras or not sentence_lengths or not total_words:
        print("=" * 66)
        print(f"{label} | {total_words} words | {len(paras)} paras | {len(sentence_lengths)} sentences")
        print("\nnot enough text to report on")
        return

    quoted_share = 100 * quoted_word_count / total_words
    mode = Counter(sentence_lengths).most_common(1)[0][0]
    sentence_length_deviation = stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    sentence_length_variation = 100 * sentence_length_deviation / fmean(sentence_lengths)

    def status(condition):
        return 'PASS' if condition else 'FAIL'

    print("=" * 66)
    print(f"{label} | {total_words} words | {len(paras)} paras | {len(sentence_lengths)} sentences"
          f"{'  [paragraphs split on hard line breaks]' if hard_breaks(text) else ''}")
    print(f"\nWORDS/SENTENCE  mode {mode}  median {median(sentence_lengths):.0f}  mean {fmean(sentence_lengths):.2f}  "
          f"sd {sentence_length_deviation:.2f}  CV {sentence_length_variation:.1f}%  max {max(sentence_lengths)}")
    print(f"  mode<median<mean  {status(mode < median(sentence_lengths) < fmean(sentence_lengths))}     "
          f"mean 11-18  {status(11 <= fmean(sentence_lengths) <= 18)}     "
          f"CV 68-100%  {status(68 <= sentence_length_variation <= 100)}")
    for name, predicate, lower_bound, upper_bound in [('<10', lambda value: value < 10, 35, 45),
                            ('10-20', in_ten_to_twenty, 30, 35),
                            ('20-35', in_twenty_to_thirty_five, 15, 20),
                            ('>35', lambda value: value > 35, 0, 5)]:
        pct = 100 * sum(1 for value in sentence_lengths if predicate(value)) / len(sentence_lengths)
        print(f"  {name:>6}: {pct:5.1f}%  target {lower_bound}-{upper_bound}%  {status(lower_bound <= pct <= upper_bound)}")

    print(f"\nQUOTED  {quoted_share:.1f}%  target ~30%  {status(26 <= quoted_share <= 34)}")

    # Spoken and narration measured separately. STYLE_GUIDES.md section 6 calls
    # this "the most useful diagnostic in the whole spec" because the combined
    # number hides which half is broken, but the script never implemented it.
    spoken, narration = split_speech(paras)
    if spoken and narration:
        spoken_mean, narration_mean = fmean(spoken), fmean(narration)
        short_narration_share = 100 * sum(1 for value in narration if value < 10) / len(narration)
        print(f"\nSPOKEN vs NARRATION")
        print(f"  spoken     mean {spoken_mean:5.2f} words  ({len(spoken):4d} sentences)  "
              f"target 8-10   {status(8 <= spoken_mean <= 10)}")
        print(f"  narration  mean {narration_mean:5.2f} words  ({len(narration):4d} sentences)  "
              f"target 14-17  {status(14 <= narration_mean <= 17)}")
        print(f"  narration under 10 words  {short_narration_share:.1f}%   target <40%  {status(short_narration_share < 40)}")

    paragraph_length_deviation = stdev(sentences_per_paragraph) if len(sentences_per_paragraph) > 1 else 0.0
    variation = 100 * paragraph_length_deviation / fmean(sentences_per_paragraph)
    print(f"\nSENT/PARA  mean {fmean(sentences_per_paragraph):.2f}  median {median(sentences_per_paragraph):.0f}  "
          f"sd {paragraph_length_deviation:.2f}  CV {variation:.1f}%  max {max(sentences_per_paragraph)}")
    print(f"  mean 2.9-3.5 {status(2.9 <= fmean(sentences_per_paragraph) <= 3.5)}   CV>=100% {status(variation >= 100)}")
    print("  bucket  this   YA")
    for name, predicate, young_adult_rate in [('1', lambda value: value == 1, 36.7),
                        ('2', lambda value: value == 2, 24.0),
                        ('3', lambda value: value == 3, 15.1),
                        ('4', lambda value: value == 4, 8.0),
                        ('5', lambda value: value == 5, 5.9),
                        ('6-7', lambda value: 6 <= value <= 7, 5.3),
                        ('8-9', lambda value: 8 <= value <= 9, 2.2),
                        ('10+', lambda value: value >= 10, 2.8)]:
        print(f"   {name:>4} {100 * sum(1 for value in sentences_per_paragraph if predicate(value)) / len(sentences_per_paragraph):6.1f}% {young_adult_rate:6.1f}%")

    print(f"\nWORD LEN  mean {fmean(word_lengths):.2f}   "
          f">=7 chars {100 * sum(1 for length in word_lengths if length >= 7) / len(word_lengths):.1f}%")

    # conjunction rates
    lowercase_words = [length.lower() for length in words(body)]
    word_counts = Counter(lowercase_words)
    total_words = len(lowercase_words)
    print("\nCONJUNCTIONS (% of all words)")
    for word, lower_bound, upper_bound in [('and', 2.5, 3.5), ('but', 0.25, 0.75), ('so', 0.2, 0.6),
                      ('because', 0.2, 0.7), ('which', 0.1, 0.4)]:
        pct = 100 * word_counts[word] / total_words
        flag = 'PASS' if lower_bound <= pct <= upper_bound else 'FAIL'
        print(f"  {word:<8} {word_counts[word]:4d}  {pct:5.2f}%   target {lower_bound}-{upper_bound}%   {flag}")
    multiple_and_count = sum(1 for sentence in [candidate for paragraph in paras for candidate in sents(paragraph)]
                 if len(re.findall(r'\band\b', sentence.lower())) >= 2)
    print(f"  sentences with 2+ 'and': {100 * multiple_and_count / len(sentence_lengths):.1f}%   target <10%")

    # section breaks
    raw_text = load(path)
    # The book was unified onto the underscore rule; the old '---' form is gone,
    # so this counted zero everywhere and reported PASS on every file.
    section_breaks = len([length for length in raw_text.split('\n') if set(length.strip()) == {'_'}])
    print(f"\nSECTION BREAKS  {section_breaks}   target 2-6   "
          f"{'PASS' if 2 <= section_breaks <= 6 else 'FAIL'}")

    # tic scan
    # Rule 1 governs the narrator, so quoted spans come out before matching,
    # exactly as summarise() does it. Without this the per-chapter report hands
    # a fixing agent every character's own explanatory clause as a violation,
    # which is how a character's line gets "corrected" into something flatter
    # than the author wrote.
    print("\nTIC SCAN, narration only (quoted spans removed)")
    tic_patterns = [
        (r'which is (?:the|what|how|why|a|his|her|its)\b', 'narrator evaluation'),
        (r',\s+which is ', 'trailing explanatory clause'),
        (r'\b(best|worst|funniest|nicest|only time|never once|for the first time)\b', 'superlative'),
        (r"it'?s not \w+[,;] it'?s ", 'not-X-but-Y'),
        (r'—', 'em dash'),
    ]
    tic_hits = 0
    for paragraph in paras:
        # Strip at paragraph level, not sentence level. A speech that runs to
        # several sentences leaves its interior sentences carrying no quote
        # mark at all, so a per-sentence strip cannot see they are dialogue.
        for sentence in sents(narration_of(paragraph)):
            for pattern, name in tic_patterns:
                if re.search(pattern, sentence, re.I):
                    print(f"  [{name}] {sentence[:110]}")
                    tic_hits += 1
    if not tic_hits:
        print("  none found")




def narration_of(para):
    """A paragraph with everything spoken taken out of it.

    Rule 1 governs the narrator, so both forms of speech come out before any
    tic is matched. Two things bite here and both have sent fixing agents at
    lines the narrator never wrote:

    Quoted speech is stripped at paragraph level rather than sentence level,
    because a speech that runs to several sentences leaves its interior
    sentences carrying no quote mark at all.

    Group-chat lines (``priya: does anyone know ...``) carry no quote marks
    either, and there are several hundred of them across the late chapters.
    Scanned raw they read as narration, which is how a character's own chat
    message gets reported as the narrator addressing the reader.
    """
    para = re.sub(r'"[^"]*"', ' ', para)
    kept = [line for line in para.split('\n')
            if not re.match(r'^[a-z][a-z0-9_]*: ', line.strip())]
    return '\n'.join(kept)


TICS = [
    (r'which is (?:the|what|how|why|a|his|her|its)\b', 'narrator evaluation'),
    (r',\s+which is ', 'trailing explanatory clause'),
    # Triage, not a verdict: this catches the author's radiator example
    # ("...at the far end, so whoever gets there early sits down that end")
    # along with legitimate causal narration. Read every hit before cutting.
    (r',\s+(?:so|because|since)\s+(?:whoever|anybody|anyone|everybody|everyone|'
     r'nobody|the rest|people|you|it|the)\b',
     'so/because tail (triage, read each)'),
    (r'\b(best|worst|funniest|nicest|only time|never once|for the first time)\b',
     'superlative'),
    (r"it'?s not \w+[,;] it'?s ", 'not-X-but-Y'),
    (r'\u2014', 'em dash'),
]

CONJ_TARGET = [('and', 2.5, 3.5), ('but', 0.25, 0.75), ('so', 0.2, 0.6),
               ('because', 0.2, 0.7), ('which', 0.1, 0.4)]

# Measured over the 23-book reference corpus, for the two that keep failing.
CORPUS_NOTE = {'and': 'corpus 2.29 to 5.09, median 3.34',
               'but': 'corpus 0.19 to 0.78, median 0.39'}


def summarise(paths):
    """Every measure in this script, over the whole book, in one block.

    report() prints the same measures for one chapter and then lists every tic
    hit, which runs to hundreds of lines across thirty-six chapters. This is
    the same content with the tics counted rather than listed, so grade.py can
    carry it.
    """
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

    def verdict(condition):
        return 'ok' if condition else 'FAIL'

    print(f"\n  {len(lowercase_words):,} words   {len(all_paras):,} paragraphs   "
          f"{len(sents_all):,} sentences")

    mode = Counter(sentence_lengths).most_common(1)[0][0]
    median_value, mean_value, standard_deviation = median(sentence_lengths), fmean(sentence_lengths), stdev(sentence_lengths)
    variation = 100 * standard_deviation / mean_value
    print("\n  WORDS PER SENTENCE")
    print(f"    mode {mode}   median {median_value:g}   mean {mean_value:.2f}   sd {standard_deviation:.2f}   "
          f"CV {variation:.1f}%   max {max(sentence_lengths)}")
    print(f"    mode<median<mean {verdict(mode < median_value < mean_value):<6}"
          f"mean 11-18 {verdict(11 <= mean_value <= 18):<6}"
          f"CV 68-100% {verdict(68 <= variation <= 100)}")
    # The four bands were carried over from a YA style guide written for a
    # simpler book, and two of them were wrong here. It set the over-35 share
    # at 0-5%, but the corpus median is 6.9% and the three reference books
    # with the highest reading grade are the three with the MOST long
    # sentences: Black Beauty 18.9%, Wind in the Willows 13.8%, Little Women
    # 13.7%. Hitting 0-5% would have driven the book toward Peter Pan, which
    # the author has ruled out. These are the observed range across the
    # eleven reference novels, with the median printed beside them.
    for name, predicate, lower_bound, upper_bound, med in [('under 10', lambda value: value < 10, 21.7, 50.7, 42.7),
                                  ('10-20', in_ten_to_twenty, 26.1, 38.8, 31.0),
                                  ('20-35', in_twenty_to_thirty_five, 15.2, 30.3, 19.9),
                                  ('over 35', lambda value: value > 35, 4.9, 18.9, 6.9)]:
        pct = 100 * sum(1 for value in sentence_lengths if predicate(value)) / len(sentence_lengths)
        print(f"    {name:<10}{pct:>6.1f}%   corpus {lower_bound}-{upper_bound}%  median {med}%   "
              f"{verdict(lower_bound <= pct <= upper_bound)}")

    quoted_words = sum(len(words(match)) for match in re.findall(r'"([^"]+)"', body))
    print(f"\n  QUOTED  {100 * quoted_words / word_count:.1f}%   target ~30%")

    spoken_lengths = [len(words(sentence)) for sentence in sents_all if sentence.lstrip().startswith('"')]
    narrated_lengths = [len(words(sentence)) for sentence in sents_all if not sentence.lstrip().startswith('"')]
    print("\n  SPOKEN vs NARRATION")
    if spoken_lengths:
        print(f"    spoken     mean {fmean(spoken_lengths):5.2f} words  ({len(spoken_lengths):5d} sentences)")
    if narrated_lengths:
        print(f"    narration  mean {fmean(narrated_lengths):5.2f} words  ({len(narrated_lengths):5d} sentences)")
        short = 100 * sum(1 for value in narrated_lengths if value < 10) / len(narrated_lengths)
        print(f"    narration under 10 words  {short:.1f}%   target <40%   "
              f"{verdict(short < 40)}")

    print(f"\n  SENTENCES PER PARAGRAPH  mean {fmean(sentences_per_paragraph):.2f}   "
          f"median {median(sentences_per_paragraph):g}   max {max(sentences_per_paragraph)}")
    print(f"    {'bucket':>8}{'this':>8}{'YA':>8}")
    for name, predicate, young_adult_rate in [('1', lambda value: value == 1, 36.7), ('2', lambda value: value == 2, 24.0),
                         ('3', lambda value: value == 3, 15.1), ('4', lambda value: value == 4, 8.0),
                         ('5', lambda value: value == 5, 5.9),
                         ('6-7', lambda value: 6 <= value <= 7, 5.3),
                         ('8-9', lambda value: 8 <= value <= 9, 2.2),
                         ('10+', lambda value: value >= 10, 2.8)]:
        print(f"    {name:>8}{100 * sum(1 for value in sentences_per_paragraph if predicate(value)) / len(sentences_per_paragraph):>7.1f}%"
              f"{young_adult_rate:>7.1f}%")

    print(f"\n  WORD LENGTH  mean {fmean(word_lengths):.2f}   "
          f"7+ chars {100 * sum(1 for word in word_lengths if word >= 7) / len(word_lengths):.1f}%")

    print("\n  CONJUNCTIONS")
    for word, lower_bound, upper_bound in CONJ_TARGET:
        pct = 100 * counts[word] / word_count
        print(f"    {word:<10}{counts[word]:>7}{pct:>9.2f}%   target {lower_bound}-{upper_bound}%   "
              f"{verdict(lower_bound <= pct <= upper_bound):<6}{CORPUS_NOTE.get(word, '')}")
    multi = sum(1 for sentence in sents_all
                if len(re.findall(r'\band\b', sentence.lower())) >= 2)
    multiple_and_share = 100 * multi / len(sents_all)
    print(f"    {'2+ and':<10}{multi:>7}{multiple_and_share:>9.1f}%   target <10%    "
          f"{verdict(multiple_and_share < 10)}")

    breaks = len([word for word in text.split('\n') if set(word.strip()) == {'_'}])
    print(f"\n  SECTION BREAKS  {breaks} across {len(paths)} chapters   "
          f"target 2-6 each")

    # Rule 1 governs the narrator, so quoted spans come out before matching.
    # Strip at paragraph level, not sentence level. A speech that runs to
    # several sentences leaves its interior sentences carrying no quote mark at
    # all, so a per-sentence strip counts a character's own words as narration.
    narration = [sentence for paragraph in all_paras for sentence in sents(narration_of(paragraph))]
    print("\n  TIC SCAN, narration only (quoted spans removed)")
    for pattern, name in TICS:
        hits = sum(1 for sentence in narration if re.search(pattern, sentence, re.I))
        if hits:
            print(f"    {name:<40}{hits:>5}")


def paras_of(body):
    return [paragraph for paragraph in re.split(r'\n\s*\n', body) if paragraph.strip()]


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

if __name__ == '__main__':
    _solo_notice()
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <chapter.md> [label]\n"
                 f"       {sys.argv[0]} --summary chapters/*.md")
    if sys.argv[1] == '--summary':
        summarise(sys.argv[2:] or sorted(__import__('glob').glob('chapters/*.md')))
        sys.exit(0)
    # One path plus an optional label, as the original took. More than one path
    # and they are all treated as files, so a glob reports on each in turn.
    if len(sys.argv) == 3 and not sys.argv[2].endswith('.md'):
        report(sys.argv[1], sys.argv[2])
    else:
        for _p in sys.argv[1:]:
            report(_p, _p.split('/')[-1])
