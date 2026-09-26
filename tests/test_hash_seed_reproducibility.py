"""The same text must measure the same under every process hash seed.

Python randomizes string hashing per process, so iterating a set of words
visits them in a different order in each run.  A cap applied after such a
loop, or a sort whose key ties, then keeps different items on different
runs.  Tests that measure twice in one process share a seed and cannot see
this; these run each measurement in its own interpreter with its own seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from textgrader import optional

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ("1", "2", "3")

SCRIPT = """
import json, sys
from textgrader.document import DocumentAnalysis
from textgrader.metrics import REGISTRY
import importlib
suite, text, overrides = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
spec = REGISTRY[suite]
options = dict(spec.defaults)
options.update(overrides)
module = importlib.import_module(f"textgrader.metrics.{spec.module}")
analysis = DocumentAnalysis.from_text(text, comparison_unit="book")
out = {item["metric_id"]: {key: item.get(key) for key in ("value", "distribution", "evidence")}
       for item in module.measure(analysis, config=options, profile=None)}
print(json.dumps(out, sort_keys=True, default=str))
"""

# Twelve misspellings, each used once (a word used repeatedly counts as the
# book's own vocabulary and is excluded), so every one ties on frequency and
# a cap of three has to choose among them.
TYPO_TEXT = " ".join(
    f"She said the {word} was on the table by the door."
    for word in ("recieve", "wierd", "adress", "untill", "freind", "beleive", "calender",
                 "tommorow", "seperate", "occured", "neccessary", "goverment"))

# Candidate generation over tokens that recur with different strides: each
# item's candidates then spread over hundreds of indices.  A set of small
# integers iterates in sorted order whatever order they were added in, until
# its values outgrow the set's hash table and collide; only then does the
# (hash-seeded) order of the string tokens leak through to a capped loop.
CANDIDATE_SCRIPT = """
import json
from textgrader.metrics import reuse_algorithms as ra
sets = [frozenset({f"a{i % 7}", f"b{i % 11}", f"c{i % 13}", f"d{i % 17}", f"e{i % 19}"})
        for i in range(1500)]
graph, _ = ra.capped_inverted_candidates(sets, cap=40, max_candidates_per_item=2)
print(json.dumps(sorted((k, sorted(v)) for k, v in graph.items())))
"""


def _run(args: list[str], seed: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    done = subprocess.run([sys.executable, "-c", *args], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def _assert_seed_independent(suite: str, text: str, overrides: dict) -> None:
    outputs = {seed: _run([SCRIPT, suite, text, json.dumps(overrides)], seed) for seed in SEEDS}
    first = outputs[SEEDS[0]]
    for seed, output in outputs.items():
        if output != first:
            one, other = json.loads(first), json.loads(output)
            differing = sorted(key for key in one.keys() | other.keys()
                               if one.get(key) != other.get(key))
            pytest.fail(f"{suite} differs between PYTHONHASHSEED={SEEDS[0]} and {seed}: "
                        f"{differing}")


def test_mechanical_quality_is_hash_seed_independent():
    for package in ("pyspellchecker", "symspellpy"):
        module, reason = optional.require(package)
        if module is None:
            pytest.skip(reason)
    _assert_seed_independent("mechanical_quality_suite", TYPO_TEXT,
                             {"likely_typo_max_candidates": 3})


def test_reuse_candidate_generation_is_hash_seed_independent():
    outputs = {seed: _run([CANDIDATE_SCRIPT], seed) for seed in SEEDS}
    assert len(set(outputs.values())) == 1, "candidate graphs differ between hash seeds"
