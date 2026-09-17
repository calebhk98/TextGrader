#!/usr/bin/env python3
"""Time every metric on one text and print a cost table.

TextGrader is run repeatedly by authoring agents, so "how long does this take
on a novel" is part of the interface rather than trivia.  This measures it on
your machine and your text instead of asking you to trust a number in a README.

    python3 benchmark.py draft.md
    python3 benchmark.py draft.md --metric mattr --metric mtld
    python3 benchmark.py --words 300000          # generated text, no file needed

The shared pipeline is timed separately and reported first, because several
metrics that look expensive are really paying a one-off segmentation or parse
that every other metric then gets free.
"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import time
from pathlib import Path

from textgrader.project import load_config
from textgrader.document import DocumentAnalysis, NlpSettings, TextProcessing
from textgrader.metrics import REGISTRY

SLOW = 1.0


def generated(words: int) -> str:
    rng = random.Random(4)
    vocab = ("the quiet room held a long silence while she considered what had happened "
             "and whether anyone would notice however perhaps not because nobody asked").split()
    blocks, produced = [], 0
    while produced < words:
        sentences = []
        for _ in range(rng.randint(1, 5)):
            count = rng.randint(4, 30)
            body = " ".join(rng.choice(vocab) for _ in range(count))
            sentences.append(f'"{body.capitalize()}," she said.' if rng.random() < .25
                             else body.capitalize() + ".")
            produced += count
        blocks.append(" ".join(sentences))
    return "\n\n".join(blocks)


def timed(label, function):
    started = time.monotonic()
    outcome = function()
    return label, time.monotonic() - started, outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manuscript", nargs="?")
    parser.add_argument("--config", default=None)
    parser.add_argument("--words", type=int, default=300_000,
                        help="size of the generated text when no file is given")
    parser.add_argument("--metric", action="append", help="time only these; repeatable")
    parser.add_argument("--skip-parse", action="store_true",
                        help="skip the spaCy metrics, which dominate the total")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    text = (Path(args.manuscript).read_text(encoding="utf-8") if args.manuscript
            else generated(args.words))
    processing = TextProcessing.from_config(config.get("text_processing"))
    nlp_settings = NlpSettings.from_config(config.get("nlp"))

    rows = []
    _, build, analysis = timed("pipeline.build", lambda: DocumentAnalysis.from_text(
        text, processing=processing, nlp_settings=nlp_settings))
    rows.append(("pipeline.build", build))
    for label, getter in (("pipeline.words", lambda: analysis.words),
                          ("pipeline.paragraphs", lambda: analysis.paragraphs),
                          ("pipeline.sentences", lambda: analysis.sentences),
                          ("pipeline.quotations", lambda: analysis.quotation_spans),
                          ("pipeline.dialogue", lambda: analysis.dialogue.sentences),
                          ("pipeline.narration", lambda: analysis.narration.sentences)):
        rows.append(timed(label, getter)[:2])

    names = args.metric or list(REGISTRY)
    if args.skip_parse:
        names = [name for name in names if not REGISTRY[name].needs_parse]
    if any(REGISTRY[name].needs_parse for name in names):
        rows.append(timed("pipeline.spacy_parse", lambda: analysis.spacy_docs())[:2])

    for name in names:
        spec = REGISTRY[name]
        options = dict(spec.defaults)

        def run(spec=spec, options=options):
            module = importlib.import_module(f"textgrader.metrics.{spec.module}")
            return module.measure(analysis, config=options, profile=None)

        try:
            _, elapsed, findings = timed(name, run)
        except Exception as exc:
            rows.append((f"{name}  [FAILED: {type(exc).__name__}]", float("nan")))
            continue
        rows.append((f"{name} ({spec.cost}, {len(findings or [])} findings)", elapsed))

    if args.json:
        print(json.dumps({"words": analysis.word_count,
                          "sentences": analysis.sentence_count,
                          "seconds": {label: value for label, value in rows}}, indent=2))
        return 0
    print(f"{analysis.word_count:,} words, {analysis.sentence_count:,} sentences, "
          f"segmenter={analysis.segmenter}\n")
    print(f"{'step':<58}{'seconds':>9}")
    for label, elapsed in rows:
        flag = "  <- over a second" if elapsed > SLOW else ""
        print(f"{label:<58}{elapsed:>9.3f}{flag}")
    total = sum(value for _, value in rows if value == value)
    print(f"\n{'total':<58}{total:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
