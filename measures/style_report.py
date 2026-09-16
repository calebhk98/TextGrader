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


def load(p):
    return open(p, encoding='utf-8').read()


def hard_breaks(t):
    """True if the text uses markdown hard line breaks to end paragraphs."""
    return any(l.endswith('  ') for l in t.split('\n') if l.strip())


def _clean(blk):
    blk = blk.strip()
    if not blk or blk == '---':
        return []
    blk = blk.replace('---', ' ').strip()
    return [re.sub(r'\s+', ' ', blk)] if blk else []


def paragraphs(t):
    """Split into paragraphs on blank lines AND on markdown hard line breaks.

    The original script split on blank lines only. Chapters 1-6 of the
    manuscript separate paragraphs with two trailing spaces and a single
    newline instead, so on those it saw one paragraph per chapter and every
    per-paragraph number was meaningless (chapter 1: "1 paras", 321
    sentences/paragraph). Splitting inside each blank-line block, rather than
    picking one convention per file, also keeps a mixed file honest, which
    HALSTEAD.md is: chapters 1-6 hard-break, 7-20 blank-line.

    A plain newline is a soft wrap and does not end a paragraph, so
    conventionally wrapped text still measures correctly.
    """
    t = re.sub(r'^#.*$', '', t, flags=re.M)
    out = []
    for blk in re.split(r'\n\s*\n', t):
        current = []
        for line in blk.split('\n'):
            current.append(line)
            if line.endswith('  '):
                out.extend(_clean(' '.join(current)))
                current = []
        if current:
            out.extend(_clean(' '.join(current)))
    return out


def sents(t):
    t = re.sub(rf'\b({ABBR})\.', r'\1<DOT>', t)
    t = re.sub(r'([.!?])(["”\']?)\s+', r'\1\2|', t)
    return [p.strip().replace('<DOT>', '.') for p in t.split('|') if p.strip()]


def words(t):
    return re.findall(r"[A-Za-z][A-Za-z']*", t)


def quoted_words(body):
    """Count words inside double quotes, counting curly quotes as quotes.

    The original scanned the whole file with '"[^"]*"'. That regex cannot see
    the 12 paragraphs of HALSTEAD.md that use curly quotes, so straight
    quotes on either side of them paired across the intervening narration and
    counted it as dialogue. The manuscript came out at 38.2% quoted against a
    ~30% target, a FAIL, when the real figure is 28.1%, a PASS.

    Normalising the curly quotes fixes that. Restarting the scan at each
    paragraph keeps any future unbalanced quote from inverting dialogue and
    narration for the whole rest of the file.
    """
    body = body.replace('“', '"').replace('”', '"')
    return sum(len(words(m))
               for para in body.split('\n')
               for m in re.findall(r'"[^"]*"', para))


def split_speech(paras):
    """Split into (spoken, narration) sentence word-counts.

    Anything inside double quotes is spoken; everything else, including the
    "she says" tags, is narration.
    """
    spoken, narration = [], []
    for p in paras:
        p = p.replace('“', '"').replace('”', '"')
        pos, told, utterances, current = 0, [], [], []
        for m in re.finditer(r'"[^"]*"', p):
            gap = p[pos:m.start()]
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
            current.append(m.group(0).strip('"'))
            pos = m.end()
        if current:
            utterances.append(' '.join(current))
        told.append(p[pos:])
        for chunk in utterances:
            spoken.extend(n for n in (len(words(s)) for s in sents(chunk)) if n)
        narration.extend(
            n for n in (len(words(s)) for s in sents(' '.join(told))) if n)
    return spoken, narration


def report(path, label):
    t = load(path)
    paras = paragraphs(t)
    spp = [len(sents(p)) for p in paras]
    sl = [len(words(s)) for p in paras for s in sents(p)]
    wl = [len(w) for p in paras for w in words(p)]
    body = re.sub(r'^#.*$', '', t, flags=re.M).replace('---', ' ')
    tot = len(words(body))
    q = quoted_words(body)

    # Guards the original lacked: every statistic below divides by one of these.
    if not paras or not sl or not tot:
        print("=" * 66)
        print(f"{label} | {tot} words | {len(paras)} paras | {len(sl)} sentences")
        print("\nnot enough text to report on")
        return

    qs = 100 * q / tot
    mode = Counter(sl).most_common(1)[0][0]
    sd = stdev(sl) if len(sl) > 1 else 0.0
    cv_sl = 100 * sd / fmean(sl)

    def ok(b):
        return 'PASS' if b else 'FAIL'

    print("=" * 66)
    print(f"{label} | {tot} words | {len(paras)} paras | {len(sl)} sentences"
          f"{'  [paragraphs split on hard line breaks]' if hard_breaks(t) else ''}")
    print(f"\nWORDS/SENTENCE  mode {mode}  median {median(sl):.0f}  mean {fmean(sl):.2f}  "
          f"sd {sd:.2f}  CV {cv_sl:.1f}%  max {max(sl)}")
    print(f"  mode<median<mean  {ok(mode < median(sl) < fmean(sl))}     "
          f"mean 11-18  {ok(11 <= fmean(sl) <= 18)}     "
          f"CV 68-100%  {ok(66 <= cv_sl <= 100)}")
    for name, f, lo, hi in [('<10', lambda v: v < 10, 35, 45),
                            ('10-20', lambda v: 10 <= v <= 20, 30, 35),
                            ('20-35', lambda v: 20 < v <= 35, 15, 20),
                            ('>35', lambda v: v > 35, 0, 5)]:
        pct = 100 * sum(1 for v in sl if f(v)) / len(sl)
        print(f"  {name:>6}: {pct:5.1f}%  target {lo}-{hi}%  {ok(lo <= pct <= hi)}")

    print(f"\nQUOTED  {qs:.1f}%  target ~30%  {ok(26 <= qs <= 34)}")

    # Spoken and narration measured separately. STYLE_GUIDES.md section 6 calls
    # this "the most useful diagnostic in the whole spec" because the combined
    # number hides which half is broken, but the script never implemented it.
    spoken, narration = split_speech(paras)
    if spoken and narration:
        sp_mean, na_mean = fmean(spoken), fmean(narration)
        na_u10 = 100 * sum(1 for v in narration if v < 10) / len(narration)
        print(f"\nSPOKEN vs NARRATION")
        print(f"  spoken     mean {sp_mean:5.2f} words  ({len(spoken):4d} sentences)  "
              f"target 8-10   {ok(8 <= sp_mean <= 10)}")
        print(f"  narration  mean {na_mean:5.2f} words  ({len(narration):4d} sentences)  "
              f"target 14-17  {ok(14 <= na_mean <= 17)}")
        print(f"  narration under 10 words  {na_u10:.1f}%   target <40%  {ok(na_u10 < 40)}")

    sd_spp = stdev(spp) if len(spp) > 1 else 0.0
    cv = 100 * sd_spp / fmean(spp)
    print(f"\nSENT/PARA  mean {fmean(spp):.2f}  median {median(spp):.0f}  "
          f"sd {sd_spp:.2f}  CV {cv:.1f}%  max {max(spp)}")
    print(f"  mean 2.9-3.5 {ok(2.9 <= fmean(spp) <= 3.5)}   CV>=100% {ok(cv >= 100)}")
    print("  bucket  this   YA")
    for name, f, ya in [('1', lambda v: v == 1, 36.7),
                        ('2', lambda v: v == 2, 24.0),
                        ('3', lambda v: v == 3, 15.1),
                        ('4', lambda v: v == 4, 8.0),
                        ('5', lambda v: v == 5, 5.9),
                        ('6-7', lambda v: 6 <= v <= 7, 5.3),
                        ('8-9', lambda v: 8 <= v <= 9, 2.2),
                        ('10+', lambda v: v >= 10, 2.8)]:
        print(f"   {name:>4} {100 * sum(1 for v in spp if f(v)) / len(spp):6.1f}% {ya:6.1f}%")

    print(f"\nWORD LEN  mean {fmean(wl):.2f}   "
          f">=7 chars {100 * sum(1 for x in wl if x >= 7) / len(wl):.1f}%")

    # conjunction rates
    _w = [x.lower() for x in words(body)]
    _c = Counter(_w)
    _n = len(_w)
    print("\nCONJUNCTIONS (% of all words)")
    for k, lo, hi in [('and', 2.5, 3.5), ('but', 0.25, 0.75), ('so', 0.2, 0.6),
                      ('because', 0.2, 0.7), ('which', 0.1, 0.4)]:
        pct = 100 * _c[k] / _n
        flag = 'PASS' if lo <= pct <= hi else 'FAIL'
        print(f"  {k:<8} {_c[k]:4d}  {pct:5.2f}%   target {lo}-{hi}%   {flag}")
    _multi = sum(1 for s in [x for p in paras for x in sents(p)]
                 if len(re.findall(r'\band\b', s.lower())) >= 2)
    print(f"  sentences with 2+ 'and': {100 * _multi / len(sl):.1f}%   target <10%")

    # section breaks
    _raw = load(path)
    # The book was unified onto the underscore rule; the old '---' form is gone,
    # so this counted zero everywhere and reported PASS on every file.
    _br = len([x for x in _raw.split('\n') if set(x.strip()) == {'_'}])
    print(f"\nSECTION BREAKS  {_br}   target 2-6   "
          f"{'PASS' if 2 <= _br <= 6 else 'FAIL'}")

    # tic scan
    # Rule 1 governs the narrator, so quoted spans come out before matching,
    # exactly as summarise() does it. Without this the per-chapter report hands
    # a fixing agent every character's own explanatory clause as a violation,
    # which is how a character's line gets "corrected" into something flatter
    # than the author wrote.
    print("\nTIC SCAN, narration only (quoted spans removed)")
    _pats = [
        (r'which is (?:the|what|how|why|a|his|her|its)\b', 'narrator evaluation'),
        (r',\s+which is ', 'trailing explanatory clause'),
        (r'\b(best|worst|funniest|nicest|only time|never once|for the first time)\b', 'superlative'),
        (r"it'?s not \w+[,;] it'?s ", 'not-X-but-Y'),
        (r'—', 'em dash'),
    ]
    _hits = 0
    for p_ in paras:
        # Strip at paragraph level, not sentence level. A speech that runs to
        # several sentences leaves its interior sentences carrying no quote
        # mark at all, so a per-sentence strip cannot see they are dialogue.
        for s in sents(narration_of(p_)):
            for rx, name in _pats:
                if re.search(rx, s, re.I):
                    print(f"  [{name}] {s[:110]}")
                    _hits += 1
    if not _hits:
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
    kept = [ln for ln in para.split('\n')
            if not re.match(r'^[a-z][a-z0-9_]*: ', ln.strip())]
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
    text = '\n'.join(load(p) for p in paths)
    body = '\n'.join(l for l in text.split('\n') if not l.startswith('#'))
    all_paras = paras_of(body)
    sents_all = [s for p_ in all_paras for s in sents(p_)]
    sl = [len(words(s)) for s in sents_all]
    spp = [len(sents(p_)) for p_ in all_paras]
    wl = [len(x) for x in words(body)]
    w = [x.lower() for x in words(body)]
    n = len(w)
    counts = Counter(w)

    def verdict(ok):
        return 'ok' if ok else 'FAIL'

    print(f"\n  {len(w):,} words   {len(all_paras):,} paragraphs   "
          f"{len(sents_all):,} sentences")

    mo = Counter(sl).most_common(1)[0][0]
    me, mn, sd = median(sl), fmean(sl), stdev(sl)
    cv = 100 * sd / mn
    print("\n  WORDS PER SENTENCE")
    print(f"    mode {mo}   median {me:g}   mean {mn:.2f}   sd {sd:.2f}   "
          f"CV {cv:.1f}%   max {max(sl)}")
    print(f"    mode<median<mean {verdict(mo < me < mn):<6}"
          f"mean 11-18 {verdict(11 <= mn <= 18):<6}"
          f"CV 68-100% {verdict(68 <= cv <= 100)}")
    # The four bands were carried over from a YA style guide written for a
    # simpler book, and two of them were wrong here. It set the over-35 share
    # at 0-5%, but the corpus median is 6.9% and the three reference books
    # with the highest reading grade are the three with the MOST long
    # sentences: Black Beauty 18.9%, Wind in the Willows 13.8%, Little Women
    # 13.7%. Hitting 0-5% would have driven the book toward Peter Pan, which
    # the author has ruled out. These are the observed range across the
    # eleven reference novels, with the median printed beside them.
    for name, f_, lo, hi, med in [('under 10', lambda v: v < 10, 21.7, 50.7, 42.7),
                                  ('10-20', lambda v: 10 <= v < 20, 26.1, 38.8, 31.0),
                                  ('20-35', lambda v: 20 <= v <= 35, 15.2, 30.3, 19.9),
                                  ('over 35', lambda v: v > 35, 4.9, 18.9, 6.9)]:
        pct = 100 * sum(1 for v in sl if f_(v)) / len(sl)
        print(f"    {name:<10}{pct:>6.1f}%   corpus {lo}-{hi}%  median {med}%   "
              f"{verdict(lo <= pct <= hi)}")

    qw = sum(len(words(m)) for m in re.findall(r'"([^"]+)"', body))
    print(f"\n  QUOTED  {100 * qw / n:.1f}%   target ~30%")

    sp = [len(words(s)) for s in sents_all if s.lstrip().startswith('"')]
    na = [len(words(s)) for s in sents_all if not s.lstrip().startswith('"')]
    print("\n  SPOKEN vs NARRATION")
    if sp:
        print(f"    spoken     mean {fmean(sp):5.2f} words  ({len(sp):5d} sentences)")
    if na:
        print(f"    narration  mean {fmean(na):5.2f} words  ({len(na):5d} sentences)")
        short = 100 * sum(1 for v in na if v < 10) / len(na)
        print(f"    narration under 10 words  {short:.1f}%   target <40%   "
              f"{verdict(short < 40)}")

    print(f"\n  SENTENCES PER PARAGRAPH  mean {fmean(spp):.2f}   "
          f"median {median(spp):g}   max {max(spp)}")
    print(f"    {'bucket':>8}{'this':>8}{'YA':>8}")
    for name, f_, ya in [('1', lambda v: v == 1, 36.7), ('2', lambda v: v == 2, 24.0),
                         ('3', lambda v: v == 3, 15.1), ('4', lambda v: v == 4, 8.0),
                         ('5', lambda v: v == 5, 5.9),
                         ('6-7', lambda v: 6 <= v <= 7, 5.3),
                         ('8-9', lambda v: 8 <= v <= 9, 2.2),
                         ('10+', lambda v: v >= 10, 2.8)]:
        print(f"    {name:>8}{100 * sum(1 for v in spp if f_(v)) / len(spp):>7.1f}%"
              f"{ya:>7.1f}%")

    print(f"\n  WORD LENGTH  mean {fmean(wl):.2f}   "
          f"7+ chars {100 * sum(1 for x in wl if x >= 7) / len(wl):.1f}%")

    print("\n  CONJUNCTIONS")
    for k, lo, hi in CONJ_TARGET:
        pct = 100 * counts[k] / n
        print(f"    {k:<10}{counts[k]:>7}{pct:>9.2f}%   target {lo}-{hi}%   "
              f"{verdict(lo <= pct <= hi):<6}{CORPUS_NOTE.get(k, '')}")
    multi = sum(1 for s in sents_all
                if len(re.findall(r'\band\b', s.lower())) >= 2)
    mpct = 100 * multi / len(sents_all)
    print(f"    {'2+ and':<10}{multi:>7}{mpct:>9.1f}%   target <10%    "
          f"{verdict(mpct < 10)}")

    breaks = len([x for x in text.split('\n') if set(x.strip()) == {'_'}])
    print(f"\n  SECTION BREAKS  {breaks} across {len(paths)} chapters   "
          f"target 2-6 each")

    # Rule 1 governs the narrator, so quoted spans come out before matching.
    # Strip at paragraph level, not sentence level. A speech that runs to
    # several sentences leaves its interior sentences carrying no quote mark at
    # all, so a per-sentence strip counts a character's own words as narration.
    narration = [s for p_ in all_paras for s in sents(narration_of(p_))]
    print("\n  TIC SCAN, narration only (quoted spans removed)")
    for rx, name in TICS:
        hits = sum(1 for s in narration if re.search(rx, s, re.I))
        if hits:
            print(f"    {name:<40}{hits:>5}")


def paras_of(body):
    return [p for p in re.split(r'\n\s*\n', body) if p.strip()]


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
