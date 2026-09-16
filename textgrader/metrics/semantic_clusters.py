"""Clusters of sentences that restate one another.

The other three modules in this family report a *distribution* of pairwise
similarity. That answers "how much restatement is there" but not "how many
distinct things get restated, and how badly". This module turns
above-threshold similarity into groups: every sentence within ``threshold``
cosine similarity of some other member of the group (single-linkage) is one
cluster, and a cluster of size >= 2 is a duplicate.

**Algorithm and its complexity.** A full single-linkage agglomeration compares
every pair of sentences, O(n^2), which is 570 million comparisons at 24,000
sentences - too slow and too much memory (an n x n similarity matrix would be
2.3 GB of floats alone) for a "moderate"-cost metric. Instead this uses a
**blocked, incremental** approximation:

1. Each sentence is reduced to its content words (no closed-class words,
   length > 2).
2. A capped inverted index maps each content word to the most recent
   ``BLOCK_CAP`` sentence indices seen carrying it (a token that has occurred
   hundreds of times without forming a cluster is thereafter only compared
   against its most recent occurrences, not all of them - this trades a
   little recall on extremely common content words for a hard bound on cost).
3. Sentences are processed in order. Each sentence's *candidates* are the
   union of the postings for its own content words - i.e. only sentences that
   already share at least one non-trivial word with it. It is compared
   (real backend cosine, not the blocking heuristic) against each candidate;
   above ``threshold`` it is unioned into that candidate's cluster via a
   union-find structure, otherwise it starts a new singleton.

This makes the real cost O(n * average postings-list size), not O(n^2): a
sentence with no repeated vocabulary anywhere else in the document does one
cheap dictionary lookup and no similarity comparisons at all. Its cost is
**candidate-bound, not exhaustive**: two sentences that restate the same idea
in completely disjoint vocabulary (true paraphrase with no word overlap) will
never become candidates of each other and so will not be clustered, even
under the embedding backend, which could in principle have caught it. That is
the honest limitation of trading O(n^2) for tractability; it is why this
metric is named after "duplicate" clusters and not "paraphrase" clusters, and
why ``semantic_adjacent``/``semantic_window`` (each an O(n) or O(n*window)
scan with no blocking) remain the metrics to use for genuinely wide-vocabulary
paraphrase.

``max_sentences`` bounds this further and is essential on a 24,000-sentence
book: beyond it, the excess sentences are skipped with a warning rather than
processed, so the metric degrades to "clustering was done over the first N
sentences" instead of hanging. See the module-level default and its measured
justification below.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from ..document import DocumentAnalysis
from .common import finding, option, rate
from .semantic_adjacent import (
    COST, FAMILY, MIN_SAMPLE, REQUIRES, UNIT_SENSITIVE, DEFAULT_MODEL,
    EVIDENCE_SNIPPET_CHARS, STOPWORDS, get_sentence_vectors, similarity_at, truncate,
)
from .common import tokens as tokenize

__all__ = ["FAMILY", "COST", "REQUIRES", "MIN_SAMPLE", "UNIT_SENSITIVE", "measure"]

DEFAULT_THRESHOLD = 0.85
DEFAULT_MAX_REPORTED = 30
# Measured on big.txt (400k words, 23,920 sentences): the dependency-free
# lexical fallback clusters every one of them in about 8 seconds, well under
# the ~30s moderate-cost budget (see the module docstring's algorithm note
# for why: cost tracks shared-vocabulary candidates, not n^2). 30,000 leaves
# headroom above that measured book without letting a pathologically large
# document (a multi-book corpus fed in as one file) run unbounded. The real
# sentence-transformers backend is materially slower per comparison (dense
# 384-d dot products through a Python loop rather than sparse dict lookups)
# and was not measurable in this environment (the package is not installed),
# so this cap also protects that path from a document large enough to make
# even candidate-bound comparisons expensive.
DEFAULT_MAX_SENTENCES = 30000
# A content word seen this many times already is thereafter only compared
# against its most recent occurrences, bounding worst-case candidate lists.
BLOCK_CAP = 40


def _content_tokens(text: str) -> list[str]:
    return [word for word in tokenize(text) if len(word) > 2 and word not in STOPWORDS]


class _UnionFind:
    """Union-find over ``0..n-1`` with path compression and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _cluster(backend: str, vectors: Any, sentences: Sequence[str],
            threshold: float) -> _UnionFind:
    n = len(sentences)
    uf = _UnionFind(n)
    postings: dict[str, list[int]] = defaultdict(list)
    for i, sentence in enumerate(sentences):
        words = _content_tokens(sentence)
        candidates: set[int] = set()
        for word in words:
            candidates.update(postings.get(word, ()))
        for candidate in candidates:
            if uf.find(candidate) == uf.find(i):
                continue
            if similarity_at(backend, vectors, i, candidate) >= threshold:
                uf.union(i, candidate)
        for word in words:
            bucket = postings[word]
            bucket.append(i)
            if len(bucket) > BLOCK_CAP:
                del bucket[0]
    return uf


_IDS = (
    ("semantic.duplicate_cluster_count", "Number of duplicate-sentence clusters", "clusters"),
    ("semantic.clustered_sentence_share", "Share of sentences inside a duplicate cluster", "%"),
    ("semantic.largest_duplicate_cluster", "Largest duplicate-sentence cluster", "sentences"),
)


def measure(analysis: DocumentAnalysis, config: Mapping[str, Any] | None = None,
            profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    model_name = option(config, "model", DEFAULT_MODEL)
    threshold = float(option(config, "threshold", DEFAULT_THRESHOLD))
    max_reported = max(1, int(option(config, "max_reported", DEFAULT_MAX_REPORTED)))
    max_sentences = max(2, int(option(config, "max_sentences", DEFAULT_MAX_SENTENCES)))

    all_sentences = analysis.sentences
    if len(all_sentences) < 2:
        warning = f"needs at least two sentences; this text has {len(all_sentences)}"
        return [finding(mid, name, None, unit, family=FAMILY, sample_size=0,
                        min_sample=MIN_SAMPLE, warning=warning) for mid, name, unit in _IDS]

    truncated = len(all_sentences) > max_sentences
    sentences = all_sentences[:max_sentences]
    backend, vectors, note = get_sentence_vectors(analysis, model_name)

    uf = _cluster(backend, vectors, sentences, threshold)
    members: dict[int, list[int]] = defaultdict(list)
    for i in range(len(sentences)):
        members[uf.find(i)].append(i)
    clusters = [indices for indices in members.values() if len(indices) >= 2]
    clusters.sort(key=len, reverse=True)

    considered = len(sentences)
    clustered_count = sum(len(indices) for indices in clusters)
    largest = len(clusters[0]) if clusters else 0

    evidence = [{
        "backend": backend,
        "cluster_size": len(indices),
        "sentence_indices": indices[:10],
        "sample_sentences": [truncate(sentences[i], EVIDENCE_SNIPPET_CHARS) for i in indices[:4]],
    } for indices in clusters[:max_reported]]

    warning = note
    if truncated:
        warning = (f"{note}; this text has {len(all_sentences)} sentences, above "
                   f"max_sentences={max_sentences}; only the first {max_sentences} were "
                   f"clustered and the rest were skipped rather than risk hanging")

    return [
        finding("semantic.duplicate_cluster_count", "Number of duplicate-sentence clusters",
                len(clusters), "clusters", family=FAMILY, sample_size=considered,
                min_sample=MIN_SAMPLE, warning=warning,
                distribution={"threshold": threshold, "backend": backend}),
        finding("semantic.clustered_sentence_share", "Share of sentences inside a duplicate cluster",
                rate(clustered_count, considered), "%", family=FAMILY, sample_size=considered,
                min_sample=MIN_SAMPLE, warning=warning, evidence=evidence,
                distribution={"threshold": threshold, "backend": backend,
                              "clustered_sentences": clustered_count}),
        finding("semantic.largest_duplicate_cluster", "Largest duplicate-sentence cluster",
                largest, "sentences", family=FAMILY, sample_size=considered,
                min_sample=MIN_SAMPLE, warning=warning,
                evidence=evidence[:1] if evidence else None),
    ]
