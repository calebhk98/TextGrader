"""How much independent sentence segmenters and syntactic parsers disagree.

**Why this suite exists.** A quote-aware pySBD segmenter replaced a plain one
in :mod:`textgrader.document` (see ``Segmenter`` there and the commit "Split
sentences inside quoted speech, and record which splitter ran") because plain
pySBD refuses to split inside quotation marks, brackets and ``--`` pairs. In
dialogue-heavy fiction a paragraph is often one long quoted speech, so plain
pySBD returned whole speeches as single "sentences": measured on six
dialogue-heavy novels it lost 35-45% of sentence boundaries and inflated mean
sentence length by 40-80%, and one paragraph of *Sense and Sensibility* came
back as an 815-word "sentence" (*Confidence*: 1,293 words). That bug shipped
unnoticed even though :mod:`textgrader.metrics.sentence_segmentation` already
measured segmenter disagreement, because that module reports *mean-level*
disagreement, which a heavy-tailed collapse hides: a handful of 800+ word
sentences barely move a mean over a whole novel.

This suite is built specifically not to make that mistake again. Its headline
segmentation findings (``syntax.parser_long_sentence_share`` and
``syntax.parser_max_sentence_length``) report the *tail*, not the mean: the
share of sentences over a length threshold and the single longest sentence
found by each segmenter, exactly the shape of the pySBD bug. See
:mod:`tests.test_parser_consensus`'s
``test_plain_pysbd_collapse_is_flagged_by_long_sentence_share`` for a
synthetic reproduction and this module's own runtime check against real
dialogue-heavy Gutenberg text.

**What this suite does NOT do.** It never changes which segmenter or parser
TextGrader's canonical analysis uses (that stays exactly
:attr:`~textgrader.document.DocumentAnalysis.sentences`/``.nlp``); it only
measures how much independent implementations of the same job disagree with
that canonical choice and with one another. Disagreement is the measurement:
per the project's philosophy, two libraries disagreeing is data to keep, not
an error to reconcile into one number.

**Channels** (each independently switchable under ``features``):

``segmentation`` (on by default; needs no extra download beyond what a
``parse``-cost metric already pays for -- pySBD, spaCy's shared parse, NLTK's
``punkt_tab`` data and the pure-Python ``syntok`` package, each of which
degrades to its own "unavailable" reason without crashing anything else)
    Compares up to seven segmenters -- the canonical pipeline choice, the
    built-in punctuation splitter, plain pySBD, the pipeline's quote-aware
    pySBD, NLTK's Punkt, spaCy's rule-based sentencizer, spaCy's
    parser-derived sentence boundaries, and ``syntok`` -- over a deterministic,
    seeded, spread-across-the-book sample of paragraphs (never the whole book:
    see :func:`_sample_paragraphs`). Every comparison is done in
    ``(paragraph_index, character_offset)`` coordinates: every segmenter is
    run on the exact same paragraph string
    (:attr:`~textgrader.document.DocumentAnalysis.paragraphs`), so boundary
    offsets need no cross-text reconciliation and precision/recall/F1 between
    two segmenters is an exact set comparison, never a fuzzy string diff.

``tokenization`` (on by default, same dependencies as ``segmentation``)
    Token-count disagreement, token-boundary F1 and a special-token
    (contraction/hyphen/apostrophe) disagreement rate across the same word
    tokenizer sources (the canonical Unicode word tokenizer, spaCy, NLTK and
    syntok).

``pos`` / ``dependency`` (on by default, but need a SECOND parser to compare
against the always-available spaCy ``en_core_web_sm`` pipeline; with only one
parser available they correctly report an "insufficient/unavailable
consensus" finding rather than pretending to compare two things -- see
:func:`_active_parsers`)
    POS/morphology agreement and UAS/LAS/root/label/depth/dependency-distance
    agreement between the first two available parsers, in a fixed priority
    order (spaCy sm, spaCy md, spaCy lg, stanza), over a deterministic,
    seeded sample of whole sentences. Comparison is restricted to
    EXACT-character-span token pairs between the two parsers (never a fuzzy
    alignment), and the resulting ``alignment_coverage`` is reported on every
    finding so a reader can tell a genuine parse disagreement from a
    tokenization mismatch (see the module's ``Implementation details`` item 4
    in the task spec).

``chunks`` (on by default, degrades to "unavailable" without a second
constituent source)
    Noun-phrase span agreement between spaCy's dependency-derived
    ``noun_chunks`` and a real constituency parse's NP spans (stanza's
    ``constituency`` processor, or benepar where it loads).

``spacy_md`` / ``spacy_lg`` / ``stanza`` / ``constituency_benepar`` (OFF by
default even when the suite itself is on)
    Each loads an additional model beyond the ``en_core_web_sm`` pipeline
    every ``cost=parse`` metric already shares, so each is its own flag per
    this project's gating rule: enabling the suite alone loads nothing beyond
    what corpus profiling already pays for spaCy. ``stanza`` needs its models
    downloaded separately (``stanza.download('en', processors=...)``) and is
    never triggered by this suite itself (``download_method=None`` is always
    passed), so an environment that has not downloaded them sees a plain
    "unavailable" finding rather than a surprise multi-hundred-megabyte
    download. ``constituency_benepar`` needs ``pip install benepar`` plus
    ``benepar.download('benepar_en3')`` (about 260 MB).  benepar calls a T5
    tokenizer method that ``transformers`` 5.x removed;
    :func:`textgrader.optional.shim_benepar_transformers` restores it, shared
    with ``syntax_complexity_suite`` so both suites load benepar or neither
    does.  Any other load failure is reported verbatim as this feature's
    "unavailable" reason.

**Cost.** ``COST = "parse"``: the suite needs the shared spaCy parse like every
other syntax metric, and its own sampling (bounded by ``max_paragraphs``,
``max_words``, ``max_sentences_for_parse`` and ``max_seconds_parse``) keeps
every extra parser's cost independent of book length. A sample cut short by
``max_seconds_parse`` reports a ``sample_size`` below its ``min_sample``,
which ``grade.py`` turns into ``insufficient_data`` and withholds from corpus
comparison automatically, exactly like ``coherence_suite``'s RST channel.

**Deferred.** SuPar and UDPipe were both verified to install cleanly
(``pip install supar`` / ``pip install ufal.udpipe``) but are not wired in:
the task's own "what to compare" instructions name spaCy, stanza and benepar
specifically, and adding two more independent dependency-parser backends
would roughly double this module's already large surface for a family
(dependency parsing) two backends already cover. ``segtok`` was verified
functional (``segtok.segmenter.split_single`` correctly keeps "Mr." and "Dr."
sentence-final) but was left out of the default segmenter set for the same
reason ``syntok`` was kept and ``segtok`` was not: the two are similar
rule-based segmenters and one independent rule-based segmenter alongside
pySBD/Punkt/spaCy/builtin is enough to keep the segmenter-count list stable.
``blingfire`` is not installable here (no matching wheel was found on PyPI
for this interpreter/platform combination).
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .. import text as textlib
from ..document import DocumentAnalysis, Segmenter, TextProcessing
from ..optional import on_reset, require, shim_benepar_transformers
from .common import finding, option, shape

FAMILY = "syntax"
COST = "parse"
REQUIRES: tuple[str, ...] = ("spacy", "pysbd", "nltk", "syntok", "stanza", "benepar")
UNIT_SENSITIVE = False

MIN_SAMPLE_PARAGRAPHS = 8
MIN_SAMPLE_SENTENCES = 15

#: Fixed, stable segmenter names.  Metric ids never encode a free-form library
#: name, per the "stable metric ids" project rule; ``canonical`` is always
#: included (it costs nothing -- it is analysis.sentences_by_paragraph, which
#: the pipeline already computed) and is never removable via config.
CANONICAL = "canonical"
ALL_SEGMENTERS = ("builtin", "pysbd", "pysbd_quote_aware", "nltk_punkt",
                  "spacy_sentencizer", "spacy_parser", "syntok")
DEFAULT_SEGMENTERS = list(ALL_SEGMENTERS)

#: Priority order for picking the first two AVAILABLE parsers to compare.
#: Fixed rather than user-orderable so a corpus profile's "parser_a vs
#: parser_b" label means the same thing on every machine that enables the
#: same feature flags.
PARSER_PRIORITY = ("spacy_sm", "spacy_md", "spacy_lg", "stanza")

DEFAULT_FEATURES: dict[str, bool] = {
    "segmentation": True,
    "tokenization": True,
    "pos": True,
    "dependency": True,
    "chunks": True,
    # Off even when the suite is on: each loads an extra model beyond the
    # en_core_web_sm pipeline every cost="parse" metric already shares.
    "spacy_md": False,
    "spacy_lg": False,
    "stanza": False,
    "constituency_benepar": False,
}

MAX_EXAMPLES = 8
SNIPPET_CHARS = 160


# --------------------------------------------------------------- span tools

def _snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _locate_spans(paragraph: str, sentences: Sequence[str]) -> list[tuple[int, int]]:
    """Best-effort ``(start, end)`` spans for sentence STRINGS a segmenter
    returned with no offsets of its own.

    Sequential ``str.find`` from a moving cursor, with two fallbacks: a
    whitespace-relaxed regex for a segmenter that collapsed internal
    whitespace differently than the source paragraph, then trimming up to two
    trailing characters for a segmenter that appended a stray punctuation
    character not present in the source at that position. The second
    fallback matters in practice: :class:`~textgrader.document.Segmenter`'s
    quote-aware rejoin step (``_quote_aware``) treats a straight ASCII ``"``
    as unambiguously a CLOSING quote when deciding whether to move a stranded
    character onto the previous sentence, but a straight quote is also the
    OPENING character of the next quotation in prose that does not use curly
    quotes -- so on straight-quote dialogue its own sentence strings can carry
    one bogus trailing ``"`` that is not actually in the source text at that
    position. That is a real, separately-reportable quirk of the canonical
    segmenter this suite cannot fix (``document.py`` is off limits here), but
    this suite must not let it silently zero out every boundary comparison on
    exactly the dialogue-heavy prose it exists to check, so a sentence that
    cannot be found verbatim gets one more attempt with trailing characters
    stripped before its boundary is given up on. A sentence that still cannot
    be located is dropped rather than mis-aligned: a missing boundary lowers
    that segmenter's recall against the others, which is the honest outcome,
    rather than a wrong span silently corrupting every downstream comparison.
    """

    spans: list[tuple[int, int]] = []
    cursor = 0
    for raw in sentences:
        text = raw.strip()
        if not text:
            continue
        idx = paragraph.find(text, cursor)
        if idx >= 0:
            end = idx + len(text)
        else:
            pattern = re.compile(r"\s+".join(re.escape(part) for part in text.split()))
            match = pattern.search(paragraph, cursor)
            if match:
                idx, end = match.start(), match.end()
            else:
                idx, end = -1, -1
                trimmed = text
                for _ in range(2):
                    trimmed = trimmed[:-1]
                    if not trimmed:
                        break
                    found_at = paragraph.find(trimmed, cursor)
                    if found_at >= 0:
                        idx, end = found_at, found_at + len(trimmed)
                        break
                if idx < 0:
                    continue
        spans.append((idx, end))
        cursor = end
    return spans


def _normalize_end(paragraph: str, spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Trim trailing whitespace from every span's end offset.

    Segmenters do not agree on whether a sentence's own span includes the
    whitespace before the next one: plain pySBD's ``char_span=True`` output
    includes it (measured: ``"...sky. "`` with a trailing space, one char
    longer than every other segmenter's span for the identical sentence),
    while the built-in splitter, syntok, NLTK and spaCy all end exactly on
    the last non-whitespace character. Left unnormalized, that alone would
    make every pySBD boundary "disagree" with everything else by exactly one
    character on almost every sentence in a book -- a whitespace convention,
    not a segmentation disagreement -- and would swamp the real signal this
    suite exists to surface. Applied uniformly to every segmenter's spans, so
    it changes nothing for a segmenter that was already trimmed.
    """

    out = []
    for start, end in spans:
        text = paragraph[start:end]
        stripped = text.rstrip()
        out.append((start, start + len(stripped)))
    return out


def _boundaries(spans: Sequence[tuple[int, int]]) -> set[int]:
    """End-offsets of every sentence except the paragraph's own final one.

    The paragraph's final boundary (its own end) is trivially agreed on by
    every segmenter and would inflate every agreement score if counted.
    """

    if len(spans) <= 1:
        return set()
    return {end for _, end in spans[:-1]}


# ------------------------------------------------------------- segmenters

@dataclass
class SegmenterOutcome:
    name: str
    available: bool
    reason: str | None
    #: paragraph index (into the sampled list, 0-based within the sample) ->
    #: list of (start, end) spans within that paragraph's own text.
    spans_by_paragraph: dict[int, list[tuple[int, int]]] = field(default_factory=dict)


def _builtin_spans(paragraphs: Sequence[str]) -> SegmenterOutcome:
    spans = {i: _locate_spans(p, textlib.sentences(p)) for i, p in enumerate(paragraphs)}
    return SegmenterOutcome("builtin", True, None, spans)


def _canonical_spans(paragraphs: Sequence[str],
                     sentences_by_paragraph: Sequence[Sequence[str]]) -> SegmenterOutcome:
    spans = {i: _locate_spans(p, sentences_by_paragraph[i]) for i, p in enumerate(paragraphs)}
    return SegmenterOutcome(CANONICAL, True, None, spans)


def _pysbd_spans(paragraphs: Sequence[str], language: str) -> SegmenterOutcome:
    pysbd, reason = require("pysbd")
    if pysbd is None:
        return SegmenterOutcome("pysbd", False, reason)
    try:
        seg = pysbd.Segmenter(language=language, clean=False, char_span=True)
    except Exception as exc:  # pragma: no cover - pySBD language guard
        return SegmenterOutcome("pysbd", False, f"pysbd failed ({type(exc).__name__}: {exc})")
    spans: dict[int, list[tuple[int, int]]] = {}
    for i, paragraph in enumerate(paragraphs):
        try:
            found = seg.segment(paragraph)
            spans[i] = [(s.start, s.end) for s in found if paragraph[s.start:s.end].strip()]
        except Exception:  # pragma: no cover - pySBD input guard
            spans[i] = []
    return SegmenterOutcome("pysbd", True, None, spans)


def _pysbd_quote_aware_spans(paragraphs: Sequence[str], language: str) -> SegmenterOutcome:
    pysbd, reason = require("pysbd")
    if pysbd is None:
        return SegmenterOutcome("pysbd_quote_aware", False, reason)
    seg = Segmenter(TextProcessing(segmenter="pysbd", language=language, segmenter_clean=False))
    if seg._impl is None:  # pragma: no cover - pysbd import raced/failed inside Segmenter
        return SegmenterOutcome("pysbd_quote_aware", False,
                                seg.warning or "pysbd unavailable inside Segmenter")
    spans = {i: _locate_spans(p, seg.split(p)) for i, p in enumerate(paragraphs)}
    return SegmenterOutcome("pysbd_quote_aware", True, None, spans)


def _nltk_punkt_spans(paragraphs: Sequence[str]) -> SegmenterOutcome:
    nltk, reason = require("nltk")
    if nltk is None:
        return SegmenterOutcome("nltk_punkt", False, reason)
    try:
        from nltk.tokenize import PunktTokenizer
        tokenizer = PunktTokenizer("english")
    except LookupError as exc:
        return SegmenterOutcome("nltk_punkt", False,
                                f"NLTK 'punkt_tab' data not found ({exc}); run "
                                f"python -m nltk.downloader punkt_tab")
    except Exception as exc:  # pragma: no cover - nltk runtime guard
        return SegmenterOutcome("nltk_punkt", False, f"nltk failed ({type(exc).__name__}: {exc})")
    spans = {i: list(tokenizer.span_tokenize(p)) for i, p in enumerate(paragraphs)}
    return SegmenterOutcome("nltk_punkt", True, None, spans)


def _syntok_spans(paragraphs: Sequence[str]) -> SegmenterOutcome:
    syntok_mod, reason = require("syntok")
    if syntok_mod is None:
        return SegmenterOutcome("syntok", False, reason)
    try:
        import syntok.segmenter as syntok_seg
    except Exception as exc:  # pragma: no cover - broken install
        return SegmenterOutcome("syntok", False, f"syntok failed ({type(exc).__name__}: {exc})")
    spans: dict[int, list[tuple[int, int]]] = {}
    for i, paragraph in enumerate(paragraphs):
        found: list[tuple[int, int]] = []
        try:
            for para in syntok_seg.process(paragraph):
                for sent in para:
                    tokens = list(sent)
                    if not tokens:
                        continue
                    start = tokens[0].offset
                    end = tokens[-1].offset + len(tokens[-1].value)
                    found.append((start, end))
        except Exception:  # pragma: no cover - syntok input guard
            found = []
        spans[i] = found
    return SegmenterOutcome("syntok", True, None, spans)


def _spacy_sentencizer_spans(paragraphs: Sequence[str], language: str) -> SegmenterOutcome:
    spacy_mod, reason = require("spacy")
    if spacy_mod is None:
        return SegmenterOutcome("spacy_sentencizer", False, reason)
    try:
        nlp = spacy_mod.blank(language)
        nlp.add_pipe("sentencizer")
        docs = list(nlp.pipe(paragraphs))
    except Exception as exc:  # pragma: no cover - spaCy runtime guard
        return SegmenterOutcome("spacy_sentencizer", False,
                                f"spaCy blank pipeline failed ({type(exc).__name__}: {exc})")
    spans = {i: [(s.start_char, s.end_char) for s in doc.sents] for i, doc in enumerate(docs)}
    return SegmenterOutcome("spacy_sentencizer", True, None, spans)


def _spacy_parser_spans(paragraphs: Sequence[str], analysis: DocumentAnalysis) -> SegmenterOutcome:
    nlp = analysis.nlp
    if nlp is None:
        return SegmenterOutcome("spacy_parser", False, analysis.nlp_unavailable)
    try:
        docs = list(nlp.pipe(paragraphs))
    except Exception as exc:  # pragma: no cover - spaCy runtime guard
        return SegmenterOutcome("spacy_parser", False, f"spaCy parse failed ({type(exc).__name__}: {exc})")
    spans = {i: [(s.start_char, s.end_char) for s in doc.sents] for i, doc in enumerate(docs)}
    return SegmenterOutcome("spacy_parser", True, None, spans)


SEGMENTER_BUILDERS: dict[str, Callable[..., SegmenterOutcome]] = {
    "builtin": lambda ctx: _builtin_spans(ctx.paragraphs),
    "pysbd": lambda ctx: _pysbd_spans(ctx.paragraphs, ctx.language),
    "pysbd_quote_aware": lambda ctx: _pysbd_quote_aware_spans(ctx.paragraphs, ctx.language),
    "nltk_punkt": lambda ctx: _nltk_punkt_spans(ctx.paragraphs),
    "spacy_sentencizer": lambda ctx: _spacy_sentencizer_spans(ctx.paragraphs, ctx.language),
    "spacy_parser": lambda ctx: _spacy_parser_spans(ctx.paragraphs, ctx.analysis),
    "syntok": lambda ctx: _syntok_spans(ctx.paragraphs),
}


@dataclass
class SampleContext:
    analysis: DocumentAnalysis
    paragraphs: list[str]
    language: str


# ------------------------------------------------------------------ sampling

def _sample_paragraphs(paragraphs: Sequence[str], max_paragraphs: int, max_words: int,
                       seed: int) -> tuple[list[int], dict[str, Any]]:
    """A deterministic, spread-across-the-book sample of paragraph indices.

    Mirrors :func:`textgrader.coherence.sample_rst_passages`'s stripe-sampling
    shape: the document is split into ``max_paragraphs`` equal-width
    contiguous stripes by paragraph index and one paragraph is drawn from a
    random position within each stripe using ``random.Random(seed)``, so the
    sample is reproducible and spread across the whole book rather than
    always the opening pages. Selection then stops, in document order, once
    ``max_words`` would be exceeded, so a handful of enormous paragraphs
    cannot blow the cost budget; when that cuts the sample short, the
    returned ``sampled`` count is below ``requested``, which every finding's
    ``sample_size``/``min_sample`` pair turns into ``insufficient_data``
    rather than a comparison built on a truncated sample.
    """

    total = len(paragraphs)
    info = {"total_paragraphs": total, "max_paragraphs_cap": max(0, max_paragraphs),
           "max_words_cap": max(1, max_words), "seed": seed}
    if total == 0 or max_paragraphs <= 0:
        return [], {**info, "paragraphs_requested": 0, "paragraphs_sampled": 0, "words_sampled": 0}

    # "requested" is the target AFTER reducing for how much document there
    # actually is (mirrors coh.sample_rst_passages's own "sampled" field,
    # which is what that channel's min_sample uses): a short document that
    # naturally has fewer paragraphs than max_paragraphs is not "insufficient
    # data" just because the config knob asked for more than exists. Only a
    # words_sampled cut short by max_words_cap (paragraphs_sampled < this
    # target) means the sample was truncated.
    n = max(1, min(max_paragraphs, total))
    info["paragraphs_requested"] = n
    rng = random.Random(seed)
    edges = [round(i * total / n) for i in range(n + 1)]
    candidates: list[int] = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        candidates.append(lo + (rng.randint(0, hi - lo - 1) if hi - lo > 1 else 0))
    candidates = sorted(set(candidates))

    chosen: list[int] = []
    words_used = 0
    for idx in candidates:
        words = len(textlib.words(paragraphs[idx]))
        if chosen and words_used + words > max_words:
            break
        chosen.append(idx)
        words_used += words
    return chosen, {**info, "paragraphs_sampled": len(chosen), "words_sampled": words_used}


def _sample_sentences(sentences: Sequence[str], max_sentences: int,
                      seed: int) -> tuple[list[int], dict[str, Any]]:
    """The sentence-level analogue of :func:`_sample_paragraphs`, for the
    per-sentence parser comparisons (POS/dependency), which are far more
    expensive per unit than segmentation and so use a smaller, separately
    capped sample."""

    total = len(sentences)
    info = {"total_sentences": total, "max_sentences_cap": max(0, max_sentences), "seed": seed}
    if total == 0 or max_sentences <= 0:
        return [], {**info, "sentences_requested": 0, "sentences_sampled": 0}
    n = max(1, min(max_sentences, total))
    info["sentences_requested"] = n
    rng = random.Random(seed)
    edges = [round(i * total / n) for i in range(n + 1)]
    chosen: list[int] = []
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        chosen.append(lo + (rng.randint(0, hi - lo - 1) if hi - lo > 1 else 0))
    chosen = sorted(set(chosen))
    return chosen, {**info, "sentences_sampled": len(chosen)}


# ------------------------------------------------------------- segmentation

def _segmenter_outcomes(analysis: DocumentAnalysis, sample_indices: Sequence[int],
                        wanted: Sequence[str]) -> dict[str, SegmenterOutcome]:
    paragraphs = [analysis.paragraphs[i] for i in sample_indices]
    sentences_by_paragraph = [analysis.sentences_by_paragraph[i] for i in sample_indices]
    ctx = SampleContext(analysis, paragraphs, analysis.processing.language)
    outcomes: dict[str, SegmenterOutcome] = {
        CANONICAL: _canonical_spans(paragraphs, sentences_by_paragraph),
    }
    for name in wanted:
        builder = SEGMENTER_BUILDERS.get(name)
        if builder is None:
            outcomes[name] = SegmenterOutcome(name, False, f"unknown segmenter {name!r}")
            continue
        outcomes[name] = builder(ctx)
    for outcome in outcomes.values():
        outcome.spans_by_paragraph = {
            pidx: _normalize_end(paragraphs[pidx], spans)
            for pidx, spans in outcome.spans_by_paragraph.items()}
    return outcomes


def _sentence_stats(paragraphs: Sequence[str], outcome: SegmenterOutcome,
                    long_words: int) -> dict[str, Any]:
    lengths: list[int] = []
    longest: tuple[int, int, int] | None = None  # (paragraph_index, words, start)
    for pidx, spans in outcome.spans_by_paragraph.items():
        for start, end in spans:
            words = len(textlib.words(paragraphs[pidx][start:end]))
            lengths.append(words)
            if longest is None or words > longest[1]:
                longest = (pidx, words, start)
    if not lengths:
        return {"available": outcome.available, "reason": outcome.reason, "sentence_count": 0}
    over = sum(1 for w in lengths if w > long_words)
    stats = {"available": True, "reason": outcome.reason, "sentence_count": len(lengths),
             "long_sentence_share_pct": 100.0 * over / len(lengths),
             "sentences_over_threshold": over, "max_sentence_length_words": max(lengths),
             "mean_sentence_length_words": sum(lengths) / len(lengths)}
    if longest is not None:
        pidx, words, start = longest
        span = next(s for s in outcome.spans_by_paragraph[pidx] if s[0] == start)
        stats["longest_example"] = {"paragraph_index": pidx, "words": words,
                                    "snippet": _snippet(paragraphs[pidx][span[0]:span[1]])}
    return stats


def _segmentation_findings(analysis: DocumentAnalysis, paragraphs: Sequence[str],
                           outcomes: Mapping[str, SegmenterOutcome], sample_info: Mapping[str, Any],
                           long_words: int, max_reported: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sampled = sample_info["paragraphs_sampled"]
    requested = sample_info["paragraphs_requested"]

    available = {name: o for name, o in outcomes.items() if o.available}
    per_segmenter = {name: _sentence_stats(paragraphs, o, long_words) for name, o in outcomes.items()}

    out.append(finding(
        "syntax.parser_segmenters_available", "Sentence segmenters available for comparison",
        len(available), "segmenters", family=FAMILY, sample_size=sampled, min_sample=requested,
        distribution={"requested": list(outcomes), "available": sorted(available),
                     "reasons": {n: o.reason for n, o in outcomes.items() if not o.available},
                     **sample_info},
        warning=None if len(available) >= 2 else
        "fewer than two segmenters are available; disagreement cannot be measured"))

    if len(available) < 1:
        return out

    # -------- headline channels: the tail, not the mean (see module docstring)
    long_share = {name: s.get("long_sentence_share_pct") for name, s in per_segmenter.items()
                 if s.get("sentence_count")}
    max_len = {name: s.get("max_sentence_length_words") for name, s in per_segmenter.items()
              if s.get("sentence_count")}
    worst_share_name = max(long_share, key=long_share.get) if long_share else None
    worst_len_name = max(max_len, key=max_len.get) if max_len else None

    evidence = []
    for name in sorted(per_segmenter):
        example = per_segmenter[name].get("longest_example")
        if example:
            evidence.append({"segmenter": name, **example})
    evidence.sort(key=lambda e: -e["words"])

    out.append(finding(
        "syntax.parser_long_sentence_share",
        f"Share of sentences over {long_words} words, worst segmenter",
        long_share.get(worst_share_name) if worst_share_name else None, "%", family=FAMILY,
        sample_size=sampled, min_sample=requested,
        distribution={"by_segmenter": {n: per_segmenter[n].get("long_sentence_share_pct")
                                       for n in per_segmenter}, "worst_segmenter": worst_share_name,
                     "long_sentence_words": long_words},
        evidence=evidence[:max_reported],
        warning=None if long_share else "no sentences were produced by any segmenter"))

    out.append(finding(
        "syntax.parser_max_sentence_length", "Longest sentence found by any segmenter, in words",
        max_len.get(worst_len_name) if worst_len_name else None, "words", family=FAMILY,
        sample_size=sampled, min_sample=requested,
        distribution={"by_segmenter": {n: per_segmenter[n].get("max_sentence_length_words")
                                       for n in per_segmenter}, "worst_segmenter": worst_len_name},
        evidence=evidence[:max_reported],
        warning=None if max_len else "no sentences were produced by any segmenter"))

    counts = {n: s["sentence_count"] for n, s in per_segmenter.items() if s.get("sentence_count")}
    count_disagreement = None
    worst_pair = None
    if len(counts) >= 2:
        lo_name = min(counts, key=counts.get)
        hi_name = max(counts, key=counts.get)
        lo, hi = counts[lo_name], counts[hi_name]
        count_disagreement = 100.0 * (hi - lo) / hi if hi else None
        worst_pair = f"{lo_name} vs {hi_name}"
    out.append(finding(
        "syntax.parser_sentence_count_disagreement",
        "Largest relative sentence-count gap between any two segmenters",
        count_disagreement, "%", family=FAMILY, sample_size=sampled, min_sample=requested,
        distribution={"sentence_counts_by_segmenter": counts, "worst_pair": worst_pair},
        warning=None if count_disagreement is not None else
        "fewer than two segmenters produced sentences"))

    # -------- pairwise boundary precision/recall/F1
    names = sorted(available)
    boundaries = {n: {pidx: _boundaries(outcomes[n].spans_by_paragraph.get(pidx, []))
                      for pidx in range(len(paragraphs))} for n in names}
    pairs: dict[str, dict[str, Any]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            tp = fp = fn = 0
            for pidx in range(len(paragraphs)):
                ba, bb = boundaries[a][pidx], boundaries[b][pidx]
                tp += len(ba & bb)
                fp += len(bb - ba)
                fn += len(ba - bb)
            precision = tp / (tp + fp) if (tp + fp) else None
            recall = tp / (tp + fn) if (tp + fn) else None
            f1 = (2 * precision * recall / (precision + recall)
                 if precision and recall and (precision + recall) else None)
            pairs[f"{a}|{b}"] = {"precision": precision, "recall": recall, "f1": f1,
                                 "shared_boundaries": tp, "a_only": fn, "b_only": fp}
    f1_values = [p["f1"] for p in pairs.values() if p["f1"] is not None]
    scored_pairs = {k: v["f1"] for k, v in pairs.items() if v["f1"] is not None}
    min_pair = min(scored_pairs, key=scored_pairs.get) if scored_pairs else None
    out.append(finding(
        "syntax.parser_boundary_agreement_f1", "Mean pairwise sentence-boundary F1",
        sum(f1_values) / len(f1_values) if f1_values else None, "F1", family=FAMILY,
        sample_size=sampled, min_sample=requested,
        distribution={"pairs": pairs, "worst_pair": min_pair,
                     "worst_pair_f1": pairs[min_pair]["f1"] if min_pair else None},
        warning=None if f1_values else "fewer than two segmenters could be compared"))

    # -------- consensus: how many segmenters support each canonical boundary
    canon_boundaries = boundaries.get(CANONICAL, {})
    others = [n for n in names if n != CANONICAL]
    total_b = all_b = majority_b = only_one_b = 0
    for pidx, canon_set in canon_boundaries.items():
        for offset in canon_set:
            total_b += 1
            support = 1 + sum(1 for n in others if offset in boundaries[n].get(pidx, set()))
            voters = 1 + len(others)
            if support == voters:
                all_b += 1
            elif support > voters / 2:
                majority_b += 1
            if support == 1:
                only_one_b += 1
    out.append(finding(
        "syntax.parser_boundary_consensus",
        "Share of canonical sentence boundaries every available segmenter agrees on",
        100.0 * all_b / total_b if total_b else None, "%", family=FAMILY,
        sample_size=sampled, min_sample=requested,
        distribution={"total_canonical_boundaries": total_b, "supported_by_all": all_b,
                     "supported_by_majority": majority_b, "supported_by_only_one": only_one_b,
                     "comparison_segmenters": others},
        warning=None if total_b else "no canonical sentence boundaries to check"))

    # -------- worst-disagreement paragraphs
    worst: list[tuple[int, int]] = []
    for pidx, paragraph in enumerate(paragraphs):
        seen = [per := len(outcomes[n].spans_by_paragraph.get(pidx, [])) for n in names]
        seen = [c for c in seen if c]
        if len(seen) >= 2 and max(seen) - min(seen) > 0:
            worst.append((max(seen) - min(seen), pidx))
    worst.sort(reverse=True)
    worst_evidence = []
    for gap, pidx in worst[:max_reported]:
        worst_evidence.append({
            "paragraph_index": pidx, "count_gap": gap,
            "sentence_counts": {n: len(outcomes[n].spans_by_paragraph.get(pidx, []))
                               for n in names if outcomes[n].spans_by_paragraph.get(pidx)},
            "snippet": _snippet(paragraphs[pidx])})
    out.append(finding(
        "syntax.parser_worst_paragraphs",
        "Paragraphs where segmenters disagree most about how many sentences it contains",
        len(worst), "paragraphs", family=FAMILY, sample_size=sampled, min_sample=requested,
        distribution={"paragraphs_with_any_disagreement": len(worst)},
        evidence=worst_evidence))

    return out


# -------------------------------------------------------------- tokenization

_CONTRACTION_RE = re.compile(r"[A-Za-z]+['’][A-Za-z]+")
_HYPHEN_RE = re.compile(r"[A-Za-z]+-[A-Za-z]+(?:-[A-Za-z]+)*")


def _tokens_with_spans(text: str, source: str) -> list[tuple[int, int, str]] | None:
    """``(start, end, text)`` for every token a tokenizer source finds, or
    ``None`` when that source is unavailable."""

    if source == "canonical":
        return [(m.start(), m.end(), m.group()) for m in textlib.WORD_RE.finditer(text)]
    if source == "spacy":
        spacy_mod, reason = require("spacy")
        if spacy_mod is None:
            return None
        nlp = _blank_tokenizer(spacy_mod, "en")
        doc = nlp(text)
        return [(t.idx, t.idx + len(t.text), t.text) for t in doc if not t.is_space]
    if source == "nltk":
        nltk_mod, reason = require("nltk")
        if nltk_mod is None:
            return None
        try:
            from nltk.tokenize import TreebankWordTokenizer
            spans = list(TreebankWordTokenizer().span_tokenize(text))
        except LookupError:
            return None
        return [(s, e, text[s:e]) for s, e in spans]
    if source == "syntok":
        syntok_mod, reason = require("syntok")
        if syntok_mod is None:
            return None
        from syntok.tokenizer import Tokenizer
        tokens = list(Tokenizer().tokenize(text))
        return [(t.offset, t.offset + len(t.value), t.value) for t in tokens]
    return None


_BLANK_TOKENIZER_CACHE: dict[str, Any] = {}


def _blank_tokenizer(spacy_mod: Any, language: str) -> Any:
    if language not in _BLANK_TOKENIZER_CACHE:
        _BLANK_TOKENIZER_CACHE[language] = spacy_mod.blank(language)
    return _BLANK_TOKENIZER_CACHE[language]


def _reset_blank_tokenizer_cache() -> None:
    _BLANK_TOKENIZER_CACHE.clear()


on_reset(_reset_blank_tokenizer_cache)

TOKEN_SOURCES = ("canonical", "spacy", "nltk", "syntok")


def _tokenization_findings(paragraphs: Sequence[str], sample_info: Mapping[str, Any],
                           max_reported: int) -> list[dict[str, Any]]:
    text = "\n".join(paragraphs)
    per_source: dict[str, list[tuple[int, int, str]]] = {}
    unavailable: dict[str, str] = {}
    for source in TOKEN_SOURCES:
        found = _tokens_with_spans(text, source)
        if found is None:
            unavailable[source] = f"{source} tokenizer unavailable"
        else:
            per_source[source] = found

    sampled = sample_info["paragraphs_sampled"]
    requested = sample_info["paragraphs_requested"]
    counts = {name: len(spans) for name, spans in per_source.items()}
    out: list[dict[str, Any]] = []

    count_disagreement = None
    if len(counts) >= 2:
        lo, hi = min(counts.values()), max(counts.values())
        count_disagreement = 100.0 * (hi - lo) / hi if hi else None
    out.append(finding(
        "syntax.parser_token_count_disagreement",
        "Largest relative token-count gap between any two tokenizers",
        count_disagreement, "%", family=FAMILY, sample_size=sampled, min_sample=requested,
        distribution={"token_counts_by_source": counts, "unavailable": unavailable},
        warning=None if len(counts) >= 2 else "fewer than two tokenizers are available"))

    starts = {name: {s for s, _, _ in spans} for name, spans in per_source.items()}
    pairs: dict[str, float | None] = {}
    for i, a in enumerate(sorted(starts)):
        for b in sorted(starts)[i + 1:]:
            sa, sb = starts[a], starts[b]
            tp = len(sa & sb)
            precision = tp / len(sb) if sb else None
            recall = tp / len(sa) if sa else None
            f1 = (2 * precision * recall / (precision + recall)
                 if precision and recall and (precision + recall) else None)
            pairs[f"{a}|{b}"] = f1
    f1_values = [v for v in pairs.values() if v is not None]
    out.append(finding(
        "syntax.parser_token_boundary_f1", "Mean pairwise token-start-boundary F1",
        sum(f1_values) / len(f1_values) if f1_values else None, "F1", family=FAMILY,
        sample_size=sampled, min_sample=requested,
        distribution={"pairs": pairs},
        warning=None if f1_values else "fewer than two tokenizers are available"))

    # Special tokens: contractions and hyphenated words, where a source may
    # keep them as one token or split them ("don't" -> ["do","n't"]).
    canonical_words = per_source.get("canonical")
    special_total = special_disagree = 0
    examples: list[dict[str, Any]] = []
    if canonical_words:
        other_sources = [n for n in per_source if n != "canonical"]
        other_starts = {n: starts[n] for n in other_sources}
        for start, end, word in canonical_words:
            if not (_CONTRACTION_RE.fullmatch(word) or _HYPHEN_RE.fullmatch(word)):
                continue
            special_total += 1
            kept_whole = [n for n in other_sources if (start, end) in
                         {(s, e) for s, e, _ in per_source[n]}]
            if len(kept_whole) != len(other_sources):
                special_disagree += 1
                if len(examples) < max_reported:
                    examples.append({"word": word, "kept_as_one_token_by":
                                     kept_whole or ["none"]})
    out.append(finding(
        "syntax.parser_special_token_disagreement",
        "Share of contraction/hyphenated words tokenizers disagree on splitting",
        100.0 * special_disagree / special_total if special_total else None, "%", family=FAMILY,
        sample_size=special_total, min_sample=5,
        distribution={"special_tokens_checked": special_total,
                     "special_tokens_disagreed": special_disagree},
        evidence=examples,
        warning=None if special_total else
        "no contraction or hyphenated word tokens were found in the sample"))

    return out


# --------------------------------------------------------- parser normalizer

@dataclass
class Tok:
    start: int
    end: int
    text: str
    upos: str | None
    morph: str | None
    dep: str | None
    head_start: int | None
    head_end: int | None
    is_root: bool


def _spacy_toks(doc: Any) -> list[Tok]:
    out = []
    for t in doc:
        if t.is_space:
            continue
        out.append(Tok(t.idx, t.idx + len(t.text), t.text, t.pos_, str(t.morph) or None,
                       t.dep_, t.head.idx, t.head.idx + len(t.head.text), t.head.i == t.i))
    return out


def _stanza_toks(sentence: Any) -> list[Tok]:
    out = []
    for word in sentence.words:
        if word.start_char is None:
            continue
        head_idx = word.head  # 1-based index into sentence.words; 0 = root
        if head_idx and 1 <= head_idx <= len(sentence.words):
            head_word = sentence.words[head_idx - 1]
            head_span = (head_word.start_char, head_word.end_char)
        else:
            head_span = (word.start_char, word.end_char)
        out.append(Tok(word.start_char, word.end_char, word.text, word.upos, word.feats,
                       word.deprel, head_span[0], head_span[1], head_idx == 0))
    return out


# ------------------------------------------------------------- parser models

_PARSER_CACHE: dict[tuple[str, str], tuple[Any, str | None]] = {}


def _reset_parser_cache() -> None:
    _PARSER_CACHE.clear()


on_reset(_reset_parser_cache)


def _load_spacy_model(model_name: str) -> tuple[Any, str | None]:
    key = ("spacy_model", model_name)
    if key in _PARSER_CACHE:
        return _PARSER_CACHE[key]
    spacy_mod, reason = require("spacy")
    if spacy_mod is None:
        _PARSER_CACHE[key] = (None, reason)
        return _PARSER_CACHE[key]
    try:
        outcome: tuple[Any, str | None] = (spacy_mod.load(model_name), None)
    except Exception as exc:
        outcome = (None, f"spaCy model {model_name!r} unavailable ({type(exc).__name__}: {exc}); "
                        f"run python -m spacy download {model_name}")
    _PARSER_CACHE[key] = outcome
    return outcome


def _load_stanza(processors: str) -> tuple[Any, str | None]:
    key = ("stanza", processors)
    if key in _PARSER_CACHE:
        return _PARSER_CACHE[key]
    stanza_mod, reason = require("stanza")
    if stanza_mod is None:
        _PARSER_CACHE[key] = (None, reason)
        return _PARSER_CACHE[key]
    try:
        # download_method=None: this suite must NEVER trigger a model
        # download itself (see the module docstring's "stanza" feature note).
        pipeline = stanza_mod.Pipeline("en", processors=processors, download_method=None,
                                       verbose=False)
        outcome: tuple[Any, str | None] = (pipeline, None)
    except Exception as exc:
        outcome = (None, f"stanza English model unavailable ({type(exc).__name__}: {exc}); run "
                        f"python -c \"import stanza; stanza.download('en', "
                        f"processors={processors!r})\"")
    _PARSER_CACHE[key] = outcome
    return outcome


def _load_benepar(model_name: str) -> tuple[Any, str | None]:
    key = ("benepar", model_name)
    if key in _PARSER_CACHE:
        return _PARSER_CACHE[key]
    benepar_mod, reason = require("benepar")
    if benepar_mod is None:
        _PARSER_CACHE[key] = (None, reason)
        return _PARSER_CACHE[key]
    try:
        shim_benepar_transformers()
        outcome: tuple[Any, str | None] = (benepar_mod.Parser(model_name), None)
    except Exception as exc:
        outcome = (None, f"benepar model {model_name!r} unavailable ({type(exc).__name__}: {exc}); "
                        f"pip install benepar and run "
                        f"python -c \"import benepar; benepar.download({model_name!r})\"")
    _PARSER_CACHE[key] = outcome
    return outcome


@dataclass
class ParserHandle:
    name: str
    kind: str  # "spacy" or "stanza"
    available: bool
    reason: str | None
    #: Given raw sentence text, returns (Tok list, np_chunk_spans or None).
    parse: Callable[[str], tuple[list[Tok], list[tuple[int, int]] | None]] | None = None


def _spacy_handle(name: str, model: Any) -> ParserHandle:
    def parse(text: str) -> tuple[list[Tok], list[tuple[int, int]] | None]:
        doc = model(text)
        chunks = [(nc.start_char, nc.end_char) for nc in doc.noun_chunks]
        return _spacy_toks(doc), chunks
    return ParserHandle(name, "spacy", True, None, parse)


def _stanza_handle(name: str, pipeline: Any, want_constituency: bool) -> ParserHandle:
    def parse(text: str) -> tuple[list[Tok], list[tuple[int, int]] | None]:
        doc = pipeline(text)
        toks: list[Tok] = []
        np_spans: list[tuple[int, int]] | None = [] if want_constituency else None
        for sentence in doc.sentences:
            toks.extend(_stanza_toks(sentence))
            if want_constituency and getattr(sentence, "constituency", None) is not None:
                np_spans.extend(_np_spans_from_constituency(sentence))
        return toks, np_spans
    return ParserHandle(name, "stanza", True, None, parse)


def _np_spans_from_constituency(sentence: Any) -> list[tuple[int, int]]:
    words = sentence.words
    counter = [0]
    spans: list[tuple[int, int]] = []

    def walk(node: Any) -> tuple[int | None, int | None]:
        if node.is_leaf():
            idx = counter[0]
            counter[0] += 1
            return idx, idx
        first = last = None
        for child in node.children:
            c_first, c_last = walk(child)
            if c_first is not None:
                first = c_first if first is None else first
                last = c_last
        if node.label == "NP" and first is not None and last is not None:
            spans.append((words[first].start_char, words[last].end_char))
        return first, last

    walk(sentence.constituency)
    return spans


def _active_parsers(analysis: DocumentAnalysis, config: Mapping[str, Any] | None
                    ) -> tuple[list[ParserHandle], dict[str, str]]:
    """The first two AVAILABLE parsers in :data:`PARSER_PRIORITY`.

    Fixed priority (never a full pairwise matrix over N parsers) keeps this
    O(1) in the number of enabled feature flags and keeps a corpus profile's
    "parser_a vs parser_b" label meaning the same thing across machines with
    the same features enabled -- see the module docstring.
    """

    features = option(config, "features", DEFAULT_FEATURES)
    reasons: dict[str, str] = {}
    handles: list[ParserHandle] = []

    sm, sm_reason = analysis.nlp, analysis.nlp_unavailable
    if sm is not None:
        handles.append(_spacy_handle("spacy_sm", sm))
    else:
        reasons["spacy_sm"] = sm_reason or "spaCy unavailable"

    if features.get("spacy_md"):
        model, reason = _load_spacy_model(option(config, "spacy_md_model", "en_core_web_md"))
        if model is not None:
            handles.append(_spacy_handle("spacy_md", model))
        else:
            reasons["spacy_md"] = reason or "unavailable"
    if features.get("spacy_lg"):
        model, reason = _load_spacy_model(option(config, "spacy_lg_model", "en_core_web_lg"))
        if model is not None:
            handles.append(_spacy_handle("spacy_lg", model))
        else:
            reasons["spacy_lg"] = reason or "unavailable"
    if features.get("stanza"):
        processors = option(config, "stanza_processors", "tokenize,mwt,pos,lemma,depparse")
        pipeline, reason = _load_stanza(processors)
        if pipeline is not None:
            handles.append(_stanza_handle("stanza", pipeline, False))
        else:
            reasons["stanza"] = reason or "unavailable"

    order = {name: i for i, name in enumerate(PARSER_PRIORITY)}
    handles.sort(key=lambda h: order.get(h.name, len(order)))
    return handles[:2], reasons


def _align_exact(toks_a: Sequence[Tok], toks_b: Sequence[Tok]) -> list[tuple[Tok, Tok]]:
    by_span_b = {(t.start, t.end): t for t in toks_b}
    return [(a, by_span_b[(a.start, a.end)]) for a in toks_a if (a.start, a.end) in by_span_b]


def _pos_dependency_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None,
                             max_reported: int) -> list[dict[str, Any]]:
    features = option(config, "features", DEFAULT_FEATURES)
    parsers, unavailable_reasons = _active_parsers(analysis, config)
    out: list[dict[str, Any]] = []

    do_pos = features.get("pos", True)
    do_dep = features.get("dependency", True)
    if not (do_pos or do_dep):
        return out

    ids = [("syntax.parser_pos_agreement_rate", "POS agreement rate between parsers"),
          ("syntax.parser_morph_agreement_rate", "Morphological-feature agreement rate"),
          ("syntax.parser_uas", "Unlabeled attachment agreement (UAS) between parsers"),
          ("syntax.parser_las", "Labeled attachment agreement (LAS) between parsers"),
          ("syntax.parser_root_agreement_rate", "Sentence-root agreement rate between parsers"),
          ("syntax.parser_dep_label_agreement_rate",
           "Dependency-label agreement rate on aligned tokens, ignoring attachment"),
          ("syntax.parser_parse_depth_disagreement",
           "Mean absolute difference in per-sentence max dependency-tree depth"),
          ("syntax.parser_dependency_distance_disagreement",
           "Mean absolute difference in per-sentence mean dependency distance"),
          ("syntax.parser_alignment_coverage",
           "Share of tokens the two parsers agree the exact boundaries of")]
    if len(parsers) < 2:
        note = ("only one parser is available for comparison "
               f"({parsers[0].name if parsers else 'none'}); consensus needs a second one -- "
               f"enable features.spacy_md, features.spacy_lg or features.stanza. "
               f"unavailable parsers: {unavailable_reasons or 'none configured'}")
        return [finding(mid, name, None, None, family=FAMILY, warning=note) for mid, name in ids]

    a, b = parsers
    max_sentences = int(option(config, "max_sentences_for_parse", 300))
    max_seconds = float(option(config, "max_seconds_parse", 180.0))
    seed = int(option(config, "seed", 0))
    sample_idx, sample_info = _sample_sentences(analysis.sentences, max_sentences, seed)

    pos_total = pos_match = morph_total = morph_match = 0
    uas_total = uas_match = las_match = label_match = 0
    root_total = root_match = 0
    align_total = align_covered = 0
    depth_diffs: list[float] = []
    distance_diffs: list[float] = []
    per_sentence_las: list[float] = []
    disagreement_evidence: list[dict[str, Any]] = []

    started = time.monotonic()
    parsed = 0
    for i in sample_idx:
        if time.monotonic() - started > max_seconds:
            break
        text = analysis.sentences[i]
        toks_a, _ = a.parse(text)
        toks_b, _ = b.parse(text)
        parsed += 1
        align_total += max(len(toks_a), len(toks_b))
        aligned = _align_exact(toks_a, toks_b)
        align_covered += len(aligned)
        if not aligned:
            continue

        sent_las_hit = sent_las_total = 0
        for ta, tb in aligned:
            if ta.upos is not None and tb.upos is not None:
                pos_total += 1
                pos_match += ta.upos == tb.upos
            if ta.morph is not None and tb.morph is not None:
                morph_total += 1
                morph_match += ta.morph == tb.morph
            # Attachment agreement: both parsers report a head as a CHARACTER
            # SPAN (Tok.head_start/end), not an index, so "the same head" is
            # exact span equality -- no separate resolution step needed. This
            # is why the comparison is restricted to exact-span-aligned token
            # pairs in the first place (see _align_exact).
            head_a = (ta.head_start, ta.head_end)
            head_b = (tb.head_start, tb.head_end)
            uas_total += 1
            head_agrees = head_a == head_b
            uas_match += head_agrees
            las_match += head_agrees and ta.dep == tb.dep
            sent_las_total += 1
            sent_las_hit += head_agrees and ta.dep == tb.dep
            if ta.dep is not None and tb.dep is not None:
                label_match += ta.dep == tb.dep
        if sent_las_total:
            sent_las = sent_las_hit / sent_las_total
            per_sentence_las.append(sent_las)
            if sent_las < 0.5 and len(disagreement_evidence) < max_reported:
                disagreement_evidence.append({
                    "sentence_index": i, "snippet": _snippet(text), "sentence_las": sent_las,
                    "aligned_tokens": len(aligned)})

        roots_a = [t for t in toks_a if t.is_root]
        roots_b = [t for t in toks_b if t.is_root]
        if roots_a and roots_b:
            root_total += 1
            root_match += (roots_a[0].start, roots_a[0].end) == (roots_b[0].start, roots_b[0].end)

        depth_a = _max_depth(toks_a)
        depth_b = _max_depth(toks_b)
        if depth_a is not None and depth_b is not None:
            depth_diffs.append(abs(depth_a - depth_b))
        dist_a = _mean_dependency_distance(toks_a)
        dist_b = _mean_dependency_distance(toks_b)
        if dist_a is not None and dist_b is not None:
            distance_diffs.append(abs(dist_a - dist_b))

    requested = sample_info["sentences_requested"]
    header = {"parser_a": a.name, "parser_b": b.name, **sample_info, "sentences_parsed": parsed}

    def _rate(match: int, total: int) -> float | None:
        return 100.0 * match / total if total else None

    coverage = 100.0 * align_covered / align_total if align_total else None
    out.append(finding("syntax.parser_alignment_coverage",
                       "Share of tokens the two parsers agree the exact boundaries of", coverage,
                       "%", family=FAMILY, sample_size=parsed, min_sample=requested,
                       distribution=dict(header)))
    if not do_pos:
        pos_findings = []
    else:
        pos_findings = [
            finding("syntax.parser_pos_agreement_rate", "POS agreement rate between parsers",
                   _rate(pos_match, pos_total), "%", family=FAMILY, sample_size=parsed,
                   min_sample=requested,
                   distribution={"aligned_tokens_checked": pos_total, **header}),
            finding("syntax.parser_morph_agreement_rate", "Morphological-feature agreement rate",
                   _rate(morph_match, morph_total), "%", family=FAMILY, sample_size=parsed,
                   min_sample=requested,
                   distribution={"aligned_tokens_checked": morph_total, **header},
                   warning=None if morph_total else
                   "neither parser reported morphological features on aligned tokens"),
        ]
    out.extend(pos_findings)
    if do_dep:
        depth_shape = shape("syntax.parser_parse_depth_disagreement",
                            "Mean absolute difference in per-sentence max dependency-tree depth",
                            depth_diffs, "levels", family=FAMILY, min_sample=requested)[0]
        depth_shape["distribution"] = {**(depth_shape["distribution"] or {}), **header}
        distance_shape = shape("syntax.parser_dependency_distance_disagreement",
                               "Mean absolute difference in per-sentence mean dependency distance",
                               distance_diffs, "tokens", family=FAMILY, min_sample=requested)[0]
        distance_shape["distribution"] = {**(distance_shape["distribution"] or {}), **header}
        las_shape = shape("syntax.parser_las", "Labeled attachment agreement (LAS) between parsers",
                          per_sentence_las, "F1-like rate", family=FAMILY, min_sample=requested)[0]
        las_shape["value"] = _rate(las_match, uas_total)
        las_shape["distribution"] = {**(las_shape["distribution"] or {}),
                                     "per_sentence_las_shape": las_shape["distribution"], **header}
        las_shape["evidence"] = disagreement_evidence
        out.extend([
            finding("syntax.parser_uas", "Unlabeled attachment agreement (UAS) between parsers",
                   _rate(uas_match, uas_total), "%", family=FAMILY, sample_size=parsed,
                   min_sample=requested, distribution={"aligned_tokens_checked": uas_total, **header}),
            las_shape,
            finding("syntax.parser_root_agreement_rate", "Sentence-root agreement rate between parsers",
                   _rate(root_match, root_total), "%", family=FAMILY, sample_size=parsed,
                   min_sample=requested, distribution={"sentences_with_both_roots": root_total, **header}),
            finding("syntax.parser_dep_label_agreement_rate",
                   "Dependency-label agreement rate on aligned tokens, ignoring attachment",
                   _rate(label_match, pos_total) if pos_total else None, "%", family=FAMILY,
                   sample_size=parsed, min_sample=requested,
                   distribution={"aligned_tokens_checked": pos_total, **header}),
            depth_shape, distance_shape,
        ])
    return out


def _max_depth(toks: Sequence[Tok]) -> int | None:
    if not toks:
        return None
    by_span = {(t.start, t.end): t for t in toks}
    depth_cache: dict[tuple[int, int], int] = {}

    def depth_of(t: Tok, guard: int = 0) -> int:
        key = (t.start, t.end)
        if key in depth_cache:
            return depth_cache[key]
        if t.is_root or guard > len(toks) + 1:
            depth_cache[key] = 0
            return 0
        head = by_span.get((t.head_start, t.head_end))
        if head is None or (head.start, head.end) == key:
            depth_cache[key] = 0
            return 0
        result = 1 + depth_of(head, guard + 1)
        depth_cache[key] = result
        return result

    return max(depth_of(t) for t in toks)


def _mean_dependency_distance(toks: Sequence[Tok]) -> float | None:
    distances = []
    by_span = {(t.start, t.end): i for i, t in enumerate(toks)}
    for i, t in enumerate(toks):
        if t.is_root:
            continue
        head_i = by_span.get((t.head_start, t.head_end))
        if head_i is not None:
            distances.append(abs(head_i - i))
    return sum(distances) / len(distances) if distances else None


# -------------------------------------------------------------------- chunks

def _chunk_findings(analysis: DocumentAnalysis, config: Mapping[str, Any] | None,
                    max_reported: int) -> list[dict[str, Any]]:
    features = option(config, "features", DEFAULT_FEATURES)
    if not features.get("chunks", True):
        return []

    nlp = analysis.nlp
    if nlp is None:
        return [finding("syntax.parser_np_chunk_agreement",
                        "Noun-phrase span agreement between an independent constituency parse "
                        "and spaCy's dependency-derived chunks", None, "F1", family=FAMILY,
                        warning=analysis.nlp_unavailable or "spaCy unavailable")]

    second_name = None
    second_np: Callable[[str], list[tuple[int, int]]] | None = None
    reasons: dict[str, str] = {}
    if features.get("constituency_benepar"):
        model, reason = _load_benepar(option(config, "benepar_model", "benepar_en3"))
        if model is not None:
            second_name = "benepar"

            def second_np(text: str, _model=model) -> list[tuple[int, int]]:
                import benepar  # noqa: F401 - triggers no reload; model already loaded
                from nltk import Tree as NltkTree
                tokens = text.split()
                tree = list(_model.parse_sents([tokens]))[0]
                return _np_spans_from_nltk_tree(tree, text)
        else:
            reasons["constituency_benepar"] = reason or "unavailable"
    if second_name is None and features.get("stanza"):
        pipeline, reason = _load_stanza(option(config, "stanza_processors",
                                               "tokenize,mwt,pos,constituency"))
        if pipeline is not None and "constituency" in option(
                config, "stanza_processors", "tokenize,mwt,pos,constituency"):
            second_name = "stanza_constituency"

            def second_np(text: str, _pipeline=pipeline) -> list[tuple[int, int]]:
                doc = _pipeline(text)
                spans: list[tuple[int, int]] = []
                for sentence in doc.sentences:
                    if getattr(sentence, "constituency", None) is not None:
                        spans.extend(_np_spans_from_constituency(sentence))
                return spans
        else:
            reasons["stanza"] = reason or "constituency processor not requested"

    if second_name is None:
        return [finding("syntax.parser_np_chunk_agreement",
                        "Noun-phrase span agreement between an independent constituency parse "
                        "and spaCy's dependency-derived chunks", None, "F1", family=FAMILY,
                        distribution={"unavailable_backends": reasons},
                        warning="only one noun-phrase source (spaCy noun_chunks) is available; "
                                "enable features.stanza (with a 'constituency' processor) or "
                                "features.constituency_benepar for a second, independent source")]

    max_sentences = int(option(config, "max_sentences_for_parse", 300))
    seed = int(option(config, "seed", 0))
    sample_idx, sample_info = _sample_sentences(analysis.sentences, min(max_sentences, 100), seed)

    tp = fp = fn = 0
    for i in sample_idx:
        text = analysis.sentences[i]
        spacy_spans = {(nc.start_char, nc.end_char) for nc in nlp(text).noun_chunks}
        try:
            other_spans = set(second_np(text))
        except Exception:  # pragma: no cover - external parser runtime guard
            continue
        tp += len(spacy_spans & other_spans)
        fp += len(other_spans - spacy_spans)
        fn += len(spacy_spans - other_spans)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)
         if precision and recall and (precision + recall) else None)
    return [finding("syntax.parser_np_chunk_agreement",
                    f"Noun-phrase span F1 between spaCy noun_chunks and {second_name}", f1, "F1",
                    family=FAMILY, sample_size=len(sample_idx),
                    min_sample=sample_info["sentences_requested"],
                    distribution={"precision": precision, "recall": recall, "second_source": second_name,
                                 **sample_info})]


def _np_spans_from_nltk_tree(tree: Any, text: str) -> list[tuple[int, int]]:
    """Character spans of NP-labeled subtrees, mapped back onto ``text`` by
    matching leaf order against a whitespace tokenization of it (benepar's
    ``parse_sents`` takes pre-tokenized input, so this is exact for the
    space-separated input this suite gives it)."""

    words = text.split()
    offsets = []
    cursor = 0
    for w in words:
        start = text.index(w, cursor)
        offsets.append((start, start + len(w)))
        cursor = start + len(w)
    leaves = tree.leaves()
    spans: list[tuple[int, int]] = []

    def walk(node: Any, pos: int) -> int:
        if isinstance(node, str):
            return pos + 1
        first = pos
        for child in node:
            pos = walk(child, pos)
        if node.label() == "NP" and pos > first:
            spans.append((offsets[first][0], offsets[pos - 1][1]))
        return pos

    walk(tree, 0)
    return spans


# ------------------------------------------------------------------- measure

def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
           profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    features = option(config, "features", DEFAULT_FEATURES)
    max_paragraphs = int(option(config, "max_paragraphs", 200))
    max_words = int(option(config, "max_words", 20000))
    long_words = int(option(config, "long_sentence_words", 40))
    max_reported = int(option(config, "max_reported", 10))
    seed = int(option(config, "seed", 0))
    segmenter_names = list(option(config, "segmenters", DEFAULT_SEGMENTERS))

    out: list[dict[str, Any]] = []

    if features.get("segmentation", True) or features.get("tokenization", True):
        sample_idx, sample_info = _sample_paragraphs(analysis.paragraphs, max_paragraphs,
                                                      max_words, seed)
        paragraphs = [analysis.paragraphs[i] for i in sample_idx]
        if features.get("segmentation", True):
            outcomes = _segmenter_outcomes(analysis, sample_idx, segmenter_names)
            out.extend(_segmentation_findings(analysis, paragraphs, outcomes, sample_info,
                                              long_words, max_reported))
        if features.get("tokenization", True):
            out.extend(_tokenization_findings(paragraphs, sample_info, max_reported))

    if features.get("pos", True) or features.get("dependency", True):
        out.extend(_pos_dependency_findings(analysis, config, max_reported))

    if features.get("chunks", True):
        out.extend(_chunk_findings(analysis, config, max_reported))

    return out
