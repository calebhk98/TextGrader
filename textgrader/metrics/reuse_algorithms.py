"""Candidate-bounded building blocks shared by :mod:`reuse_suite`.

Every function here answers one narrow question -- build shingles, find
candidate pairs, hash a block, score two strings -- and every candidate-
generation function is explicit about its own bound: **none of them perform
an all-pairs comparison**. A book with 20,000 sentences has 200 million
possible pairs; nothing in this module ever looks at more than a small,
bounded multiple of the unit count.

Two independent candidate-generation techniques are used, matching the task's
own list of acceptable ones:

``minhash_lsh_candidates``
    MinHash + :class:`datasketch.MinHashLSH` when the package is installed.
    Without it, this degrades to the same **capped inverted-shingle index**
    :mod:`semantic_clusters` already uses for its blocked single-linkage
    clustering -- a shingle seen more than ``cap`` times is thereafter only
    compared against its most recent occurrences, and each item's candidate
    list is hard-capped at ``max_candidates_per_item``. Either path is
    provably bounded: the datasketch path's candidate count depends on LSH
    bucket collisions (which the cap still bounds per item), and the fallback
    path is bounded by construction. See the module's own docstring test in
    ``tests/test_reuse_suite.py`` (``test_long_document_comparison_is_bounded``)
    for a measured pair count on a 20,000-sentence synthetic document.

``longest_approximate_repeated_run``
    windowed candidate generation: fixed-size, strided token windows are
    shingled and run through the same capped inverted index, then only the
    resulting candidates are scored with a real string-similarity function
    and greedily extended, one constant-size slice per step, with each
    diagonal's runs recorded so no stretch is extended twice. A raw 187,000-word novel produces on the order of
    tens of thousands of windows, which the inverted index handles in
    O(windows * average postings length), never O(windows^2).

Every wrapper around an optional third-party library goes through
:func:`textgrader.optional.require` and returns a plain, empty result (never
raises) when that library is unavailable, so a missing package degrades one
channel of :mod:`reuse_suite`, never the whole suite.
"""

from __future__ import annotations

import bisect
import difflib
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from ..optional import require

CandidateGraph = dict[int, set[int]]

#: TLSH's own sentinel for "too short/uniform to hash meaningfully" (its
#: ``hash()`` returns this literal string rather than raising). Treated
#: exactly like "below the configured minimum block length" -- see
#: ``tlsh_pairwise`` -- so a short unit can never reach this channel even if
#: a caller misconfigures the length floor.
TLSH_NULL = "TNULL"


# --------------------------------------------------------------- shingling

def shingles(tokens: Sequence[str], k: int) -> frozenset:
    """Overlapping ``k``-token shingles of ``tokens``, as a set of strings.

    A sequence shorter than ``k`` becomes its own single shingle (the whole
    thing), so a short unit still participates in candidate generation
    instead of contributing an empty set that can never collide with anything.
    """

    if not tokens:
        return frozenset()
    if len(tokens) < k:
        return frozenset({" ".join(tokens)})
    return frozenset(" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ---------------------------------------------------------- candidate pairs

def capped_inverted_candidates(shingle_sets: Sequence[frozenset], *, cap: int = 40,
                               max_candidates_per_item: int = 20) -> tuple[CandidateGraph, int]:
    """Dependency-free candidate generation: shared-shingle blocking.

    Mirrors :mod:`semantic_clusters`'s blocked postings-list approach. Cost is
    O(n * average postings-list length), which the ``cap`` bounds directly: a
    shingle that has already appeared in more than ``cap`` units is only
    compared against its most recent ``cap`` occurrences from then on.
    ``max_candidates_per_item`` additionally hard-bounds each item's own
    candidate list, so the total edge count returned can never exceed
    ``n * max_candidates_per_item``, linear in the unit count regardless of
    how repetitive the document is.

    Returns ``(graph, pair_count)``; ``pair_count`` is the number of
    undirected candidate edges, i.e. exactly how many real-similarity
    computations a caller must perform -- never a function of n^2.
    """

    postings: dict[str, list[int]] = defaultdict(list)
    graph: CandidateGraph = defaultdict(set)
    for i, shingle_set in enumerate(shingle_sets):
        candidates: set[int] = set()
        for token in shingle_set:
            candidates.update(postings.get(token, ()))
        # Sorted: a set's iteration order depends on its insertion history,
        # which follows the (per-process, hash-seeded) order of the string
        # shingles, and the cap below keeps whichever candidates come first.
        for j in sorted(candidates):
            if j == i or j in graph[i]:
                continue
            if len(graph[i]) >= max_candidates_per_item:
                break
            graph[i].add(j)
            graph[j].add(i)
        for token in shingle_set:
            bucket = postings[token]
            bucket.append(i)
            if len(bucket) > cap:
                del bucket[0]
    pair_count = sum(len(neighbors) for neighbors in graph.values()) // 2
    return dict(graph), pair_count


def minhash_lsh_candidates(shingle_sets: Sequence[frozenset], *, num_perm: int = 32,
                           threshold: float = 0.5, seed: int = 1,
                           max_candidates_per_item: int = 20, cap: int = 40,
                           ) -> tuple[CandidateGraph, dict[frozenset, float], int, str]:
    """MinHash/LSH candidate pairs, with a dependency-free fallback.

    Returns ``(graph, jaccard_by_pair, pair_count, backend)``. ``backend`` is
    ``"datasketch_minhash_lsh"`` when the real library ran, or
    ``"inverted_shingle_index"`` when it degraded to
    :func:`capped_inverted_candidates`. Either way the exact Jaccard is
    computed only for the returned candidate pairs, never for the full grid.
    MinHash's own permutation seed is fixed (``seed``) so two runs over the
    same document produce the same candidates and the same estimate.
    """

    n = len(shingle_sets)
    if n < 2:
        return {}, {}, 0, "no_candidates_possible"
    datasketch, reason = require("datasketch")
    if datasketch is None:
        graph, _ = capped_inverted_candidates(
            shingle_sets, cap=cap, max_candidates_per_item=max_candidates_per_item)
        jaccards: dict[frozenset, float] = {}
        for i, neighbors in graph.items():
            for j in neighbors:
                if i < j:
                    jaccards[frozenset((i, j))] = jaccard(shingle_sets[i], shingle_sets[j])
        return graph, jaccards, len(jaccards), "inverted_shingle_index"

    minhashes = []
    lsh = datasketch.MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, shingle_set in enumerate(shingle_sets):
        m = datasketch.MinHash(num_perm=num_perm, seed=seed)
        for token in shingle_set:
            m.update(token.encode("utf8"))
        minhashes.append(m)
        lsh.insert(i, m)
    graph: CandidateGraph = defaultdict(set)
    jaccards: dict[frozenset, float] = {}
    for i, m in enumerate(minhashes):
        found = 0
        for j in sorted(lsh.query(m)):  # query returns set order; the cap keeps the first
            if j == i:
                continue
            if found >= max_candidates_per_item:
                break
            found += 1
            graph[i].add(j)
            graph[j].add(i)
            key = frozenset((i, j))
            if key not in jaccards:
                jaccards[key] = jaccard(shingle_sets[i], shingle_sets[j])
    return dict(graph), jaccards, len(jaccards), "datasketch_minhash_lsh"


# ------------------------------------------------------------- transformed views

def word_shape(token: str) -> str:
    """Case/digit/punctuation shape of a token, with runs collapsed.

    ``"Hello,"`` -> ``"Xx,"``; ``"CAT-99"`` -> ``"X-d"``; ``"it's"`` -> ``"x'x"``.
    """

    out: list[str] = []
    for char in token:
        if char.isupper():
            kind = "X"
        elif char.islower():
            kind = "x"
        elif char.isdigit():
            kind = "d"
        else:
            kind = char
        if out and out[-1] == kind and kind in "Xxd":
            continue
        out.append(kind)
    return "".join(out)


# ------------------------------------------------------------ longest repeats

def longest_repeated_token_run(tokens: Sequence[str]) -> tuple[int, int, int] | None:
    """``(length, first_position, second_position)`` of the longest exact
    token run occurring at least twice, or ``None``.

    Binary search over the run length ``L`` (the property "some run of length
    L repeats" is monotone: if one of length L repeats, its length-(L-1)
    prefix repeats too), with one O(n) rolling-hash pass per probed length --
    O(n log n) total, never a pairwise scan of windows.  A 61-bit modulus
    keeps hash collisions rare, and every candidate hit is confirmed with a
    direct list-slice comparison, so a collision can only cost extra work,
    never a wrong answer.
    """

    n = len(tokens)
    if n < 2:
        return None
    ids: dict[str, int] = {}
    coded = [ids.setdefault(t, len(ids)) for t in tokens]
    mod = (1 << 61) - 1
    base = 131542391627 % mod
    prefix = [0] * (n + 1)
    power = [1] * (n + 1)
    for i in range(n):
        prefix[i + 1] = (prefix[i] * base + coded[i] + 1) % mod
        power[i + 1] = (power[i] * base) % mod

    def window_hash(start: int, length: int) -> int:
        return (prefix[start + length] - prefix[start] * power[length]) % mod

    def find_repeat(length: int) -> tuple[int, int] | None:
        if length <= 0 or length > n:
            return None
        seen: dict[int, int] = {}
        for start in range(n - length + 1):
            h = window_hash(start, length)
            prev = seen.get(h)
            if prev is not None:
                if tokens[prev:prev + length] == tokens[start:start + length]:
                    return prev, start
                continue
            seen[h] = start
        return None

    lo, hi, best = 1, n - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        found = find_repeat(mid)
        if found:
            best = (mid, found[0], found[1])
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _ratio(a: str, b: str) -> float:
    """0-100 similarity, RapidFuzz when available, difflib otherwise.

    difflib's ``SequenceMatcher.ratio`` is a real, dependency-free (stdlib)
    similarity measure -- never a stub -- so this channel produces a real
    number even with every optional package absent.
    """

    return _scorer()(a, b)


def _scorer():
    rapidfuzz, _ = require("rapidfuzz")
    if rapidfuzz is not None:
        return rapidfuzz.fuzz.ratio
    return lambda a, b: 100.0 * difflib.SequenceMatcher(None, a, b).ratio()


def _covering(intervals: list[list[int]], position: int) -> list[int] | None:
    """The interval in a sorted, disjoint ``[start, end)`` list holding ``position``."""

    index = bisect.bisect_right(intervals, [position, float("inf")]) - 1
    if index >= 0 and intervals[index][0] <= position < intervals[index][1]:
        return intervals[index]
    return None


def _cover(intervals: list[list[int]], start: int, end: int) -> None:
    """Add ``[start, end)`` to a sorted, disjoint interval list, merging overlaps."""

    index = bisect.bisect_left(intervals, [start, start])
    if index > 0 and intervals[index - 1][1] >= start:
        index -= 1
        start = intervals[index][0]
    stop = index
    while stop < len(intervals) and intervals[stop][0] <= end:
        end = max(end, intervals[stop][1])
        stop += 1
    intervals[index:stop] = [[start, end]]


def longest_approximate_repeated_run(
        tokens: Sequence[str], *, window: int = 8, stride: int = 4, shingle_k: int = 3,
        threshold: float = 80.0, max_windows: int = 20000, extend_step: int = 4,
        max_candidates_per_item: int = 10) -> dict[str, Any] | None:
    """The longest pair of (approximately) matching, non-overlapping windows.

    Candidate generation is windowed + a capped inverted shingle index (see
    the module docstring): fixed-size, strided windows of the token stream
    are shingled, and only windows that already share a shingle become
    candidates. A candidate pair whose RapidFuzz/difflib ratio clears
    ``threshold`` is then extended forward, ``extend_step`` tokens at a time,
    for as long as the trailing ``window`` tokens of each side still clear it
    and the two sides have not grown into each other.  The longest pair is
    returned with the similarity of its whole run.

    Two things keep this linear in the length of what repeats.  Each
    extension step scores a ``window``-sized slice, so a step costs the same
    at token 10 as at token 10,000; rescoring the whole run on every step
    made a long repeat quadratic, and a book whose chapters repeat, cubic.
    And a run is recorded against its diagonal (the offset between its two
    sides): a later candidate on the same diagonal inside a recorded run is
    the same repeat and is skipped, and an extension that reaches one jumps
    to its end instead of rescoring it.
    """

    n = len(tokens)
    if n < window * 2:
        return None
    starts = list(range(0, n - window + 1, max(1, stride)))
    if len(starts) > max_windows:
        step = len(starts) / max_windows
        starts = [starts[int(i * step)] for i in range(max_windows)]
    shingle_sets = [shingles(tokens[s:s + window], shingle_k) for s in starts]
    graph, pair_count = capped_inverted_candidates(
        shingle_sets, max_candidates_per_item=max_candidates_per_item)
    score = _scorer()

    def similar(lo: int, hi: int, length: int) -> float:
        return score(" ".join(tokens[lo:lo + length]), " ".join(tokens[hi:hi + length]))

    best: tuple[int, int, int] | None = None
    covered: dict[int, list[list[int]]] = defaultdict(list)
    for i in sorted(graph):
        for j in sorted(graph[i]):
            if j <= i:
                continue
            a_start, b_start = starts[i], starts[j]
            lo, hi = (a_start, b_start) if a_start < b_start else (b_start, a_start)
            if lo == hi:
                continue
            # The two candidate windows share a shingle but need not start at
            # the same offset within whatever repeated passage they belong to
            # (a fixed stride only samples every ``stride`` tokens). Search a
            # bounded neighborhood of alignments -- at most 2*window shifts,
            # a constant per candidate, never a function of document length
            # -- and keep the shift that scores best before extending.
            best_shift, top = 0, -1.0
            for shift in range(-window, window + 1):
                shifted_hi = hi + shift
                if shifted_hi < 0 or shifted_hi + window > n or shifted_hi <= lo:
                    continue
                candidate_score = similar(lo, shifted_hi, window)
                if candidate_score > top:
                    best_shift, top = shift, candidate_score
            hi = hi + best_shift
            if top < threshold:
                continue
            diagonal = covered[hi - lo]
            if _covering(diagonal, lo) is not None:
                continue
            length = window
            while True:
                known = _covering(diagonal, lo + length)
                if known is not None:
                    length = min(known[1] - lo, hi - lo, n - hi)
                new_length = length + extend_step
                if hi + new_length > n or lo + new_length > hi:
                    break
                tail = new_length - window
                if similar(lo + tail, hi + tail, window) < threshold:
                    break
                length = new_length
            _cover(diagonal, lo, lo + length)
            if best is None or length > best[0]:
                best = (length, lo, hi)
    if best is None:
        return None
    length, lo, hi = best
    return {"length": length, "first_position": lo, "second_position": hi,
            "similarity": similar(lo, hi, length), "candidate_pairs_examined": pair_count}


# ------------------------------------------------------------------- motifs

def motif_positions(tokens: Sequence[str], window: int) -> dict[tuple, list[int]]:
    """Positions of every length-``window`` motif in an (already abstracted)
    token sequence, e.g. a function-word or POS skeleton. One O(n) dict pass.
    """

    positions: dict[tuple, list[int]] = defaultdict(list)
    n = len(tokens)
    for i in range(n - window + 1):
        positions[tuple(tokens[i:i + window])].append(i)
    return positions


# --------------------------------------------------------------- fuzzy hashes

def simhash_nearest_neighbors(texts: Sequence[str], *, f: int = 64, shingle_k: int = 4,
                              index_k: int = 3,
                              extra_candidates: CandidateGraph | None = None
                              ) -> tuple[dict[int, int], str]:
    """Per-unit nearest-neighbor Hamming distance (bits, out of ``f``).

    Uses :class:`simhash.SimhashIndex`'s own permutation/bucket structure for
    candidate generation -- a unit is only ever compared against units that
    land in a shared bucket, never against every other unit. ``index_k`` (the
    number of permutation blocks SimhashIndex tolerates a difference in) is
    itself already a near-duplicate-oriented net, so ``extra_candidates`` (the
    same MinHash/LSH candidate graph the caller's other channels already
    built, run at a looser Jaccard threshold) is unioned in -- without it,
    this channel is self-selecting for near-ceiling similarity for the same
    reason described in :mod:`reuse_suite`'s ``DEFAULT_CANDIDATE_THRESHOLD``.
    """

    simhash_mod, reason = require("simhash")
    n = len(texts)
    if simhash_mod is None:
        return {}, reason or "simhash unavailable"
    if n < 2:
        return {}, "fewer than two units to compare"
    fingerprints = []
    for text in texts:
        words = text.split()
        pieces = [" ".join(words[i:i + shingle_k]) for i in range(max(1, len(words) - shingle_k + 1))]
        fingerprints.append(simhash_mod.Simhash(pieces or [text], f=f))
    objs = [(str(i), fp) for i, fp in enumerate(fingerprints)]
    index = simhash_mod.SimhashIndex(objs, f=f, k=index_k)
    nearest: dict[int, int] = {}
    for i, fp in enumerate(fingerprints):
        best = None
        candidates = {int(key) for key in index.get_near_dups(fp)}
        if extra_candidates:
            candidates.update(extra_candidates.get(i, ()))
        for j in candidates:
            if j == i:
                continue
            dist = fp.distance(fingerprints[j])
            if best is None or dist < best:
                best = dist
        if best is not None:
            nearest[i] = best
    return nearest, "ok"


def _eligible_candidate_pairs(texts: Sequence[str], eligible: Sequence[int], *,
                              shingle_k: int = 3, max_candidates_per_item: int = 20
                              ) -> list[tuple[int, int]]:
    """Candidate pairs generated FROM SCRATCH over only ``eligible`` indices.

    TLSH/ssdeep's own minimum-length floor means most candidate pairs a
    whole-population (sentence- or paragraph-wide) candidate graph finds
    involve at least one unit that is too short for either hash -- on a real
    novel, well under 1% of an unrestricted candidate graph's pairs had both
    members long enough. Re-running candidate generation over just the
    eligible subset (still the same capped inverted-shingle index, never
    all-pairs even within that subset) gives these two channels a real,
    adequately-sized sample instead of the handful of coincidental survivors
    an outside graph would hand them.
    """

    if len(eligible) < 2:
        return []
    local_sets = [shingles(texts[i].split(), shingle_k) for i in eligible]
    graph, _ = capped_inverted_candidates(local_sets, max_candidates_per_item=max_candidates_per_item)
    seen: set[tuple[int, int]] = set()
    for local_i, neighbors in graph.items():
        for local_j in neighbors:
            i, j = eligible[local_i], eligible[local_j]
            pair = (i, j) if i < j else (j, i)
            seen.add(pair)
    return sorted(seen)


def tlsh_pairwise(texts: Sequence[str], graph: CandidateGraph | None = None, *,
                  min_chars: int = 256) -> tuple[list[int], list[dict[str, Any]], str]:
    """TLSH diff scores (lower = more similar) over candidate pairs generated
    from scratch among units at or above ``min_chars`` (see
    :func:`_eligible_candidate_pairs`), skipping any unit under it entirely.

    TLSH itself refuses to hash text with too little length/complexity,
    returning the literal string ``"TNULL"``; that sentinel is treated
    exactly like "below the minimum" so a short unit can never contribute a
    number here even if a caller sets ``min_chars`` too low. ``graph`` is
    accepted for backward-compatible callers but is not used: candidate
    generation is redone over only the length-eligible units.
    """

    tlsh_mod, reason = require("tlsh")
    if tlsh_mod is None:
        return [], [], reason
    hashes: dict[int, str] = {}
    eligible: list[int] = []
    for i, text in enumerate(texts):
        if len(text) < min_chars:
            continue
        try:
            h = tlsh_mod.hash(text.encode("utf-8", "ignore"))
        except Exception:
            continue
        if h and h != TLSH_NULL:
            hashes[i] = h
            eligible.append(i)
    diffs: list[int] = []
    evidence: list[dict[str, Any]] = []
    for i, j in _eligible_candidate_pairs(texts, eligible):
        diff = tlsh_mod.diff(hashes[i], hashes[j])
        diffs.append(diff)
        evidence.append({"a": i, "b": j, "tlsh_diff": diff})
    reason = "ok" if hashes else f"no block reached the minimum length of {min_chars} characters"
    return diffs, evidence, reason


def ssdeep_pairwise(texts: Sequence[str], graph: CandidateGraph | None = None, *,
                    min_chars: int = 100) -> tuple[list[int], list[dict[str, Any]], str]:
    """ppdeep (ssdeep-compatible) similarity scores (0-100, higher = more
    similar) over candidate pairs generated from scratch among units at or
    above ``min_chars`` -- see :func:`tlsh_pairwise`'s docstring for why.
    """

    ppdeep_mod, reason = require("ppdeep")
    if ppdeep_mod is None:
        return [], [], reason
    hashes: dict[int, str] = {}
    eligible: list[int] = []
    for i, text in enumerate(texts):
        if len(text) < min_chars:
            continue
        try:
            hashes[i] = ppdeep_mod.hash(text)
        except Exception:
            continue
        eligible.append(i)
    scores: list[int] = []
    evidence: list[dict[str, Any]] = []
    for i, j in _eligible_candidate_pairs(texts, eligible):
        score = ppdeep_mod.compare(hashes[i], hashes[j])
        scores.append(score)
        evidence.append({"a": i, "b": j, "ssdeep_score": score})
    reason = "ok" if hashes else f"no block reached the minimum length of {min_chars} characters"
    return scores, evidence, reason


def edit_and_fuzzy_nearest_neighbors(
        texts: Sequence[str], graph: CandidateGraph) -> dict[str, tuple[list[float], str | None]]:
    """Per-unit nearest-neighbor similarity for four independent string
    metrics, all evaluated only over ``graph``'s candidate pairs.

    Returns a dict keyed by channel name (``levenshtein``, ``jaro_winkler``,
    ``token_set``, ``token_sort``), each mapped to ``(values, reason)`` where
    ``values`` is one best-neighbor similarity per unit that had a candidate,
    and ``reason`` names why a channel is empty when its package is missing.
    """

    levenshtein_mod, lev_reason = require("levenshtein")
    jellyfish_mod, jf_reason = require("jellyfish")
    rapidfuzz_mod, rf_reason = require("rapidfuzz")

    lev_best: dict[int, float] = {}
    jw_best: dict[int, float] = {}
    tset_best: dict[int, float] = {}
    tsort_best: dict[int, float] = {}
    for i, neighbors in graph.items():
        for j in neighbors:
            if j == i:
                continue
            a, b = texts[i], texts[j]
            if levenshtein_mod is not None:
                maxlen = max(len(a), len(b), 1)
                sim = 100.0 * (1.0 - levenshtein_mod.distance(a, b) / maxlen)
                lev_best[i] = max(lev_best.get(i, 0.0), sim)
            if jellyfish_mod is not None:
                sim = 100.0 * jellyfish_mod.jaro_winkler_similarity(a, b)
                jw_best[i] = max(jw_best.get(i, 0.0), sim)
            if rapidfuzz_mod is not None:
                tset_best[i] = max(tset_best.get(i, 0.0), rapidfuzz_mod.fuzz.token_set_ratio(a, b))
                tsort_best[i] = max(tsort_best.get(i, 0.0), rapidfuzz_mod.fuzz.token_sort_ratio(a, b))

    return {
        "levenshtein": (list(lev_best.values()), None if levenshtein_mod is not None else lev_reason),
        "jaro_winkler": (list(jw_best.values()), None if jellyfish_mod is not None else jf_reason),
        "token_set": (list(tset_best.values()), None if rapidfuzz_mod is not None else rf_reason),
        "token_sort": (list(tsort_best.values()), None if rapidfuzz_mod is not None else rf_reason),
    }


def textdistance_crosscheck(texts: Sequence[str], graph: CandidateGraph) -> tuple[list[float], str | None]:
    """A fifth, independent normalized-similarity cross-check (Sorensen-Dice
    over character bigrams), kept alongside -- never instead of -- the four
    channels in :func:`edit_and_fuzzy_nearest_neighbors`. Two independent
    edit-distance-family implementations disagreeing is data, not
    redundancy (see this project's rule on library disagreement).
    """

    textdistance_mod, reason = require("textdistance")
    if textdistance_mod is None:
        return [], reason
    best: dict[int, float] = {}
    for i, neighbors in graph.items():
        for j in neighbors:
            if j == i:
                continue
            sim = 100.0 * textdistance_mod.sorensen.normalized_similarity(texts[i], texts[j])
            best[i] = max(best.get(i, 0.0), sim)
    return list(best.values()), None


def occurrences_of(joined_text: str, needle: str) -> list[int]:
    """Every start offset of ``needle`` inside ``joined_text``, via
    Aho-Corasick (a single linear pass) rather than a second search per hit.
    Falls back to :meth:`str.find` scanning when pyahocorasick is unavailable
    -- still linear, just without the automaton.
    """

    if not needle:
        return []
    ahocorasick_mod, _ = require("ahocorasick")
    if ahocorasick_mod is not None:
        automaton = ahocorasick_mod.Automaton()
        automaton.add_word(needle, needle)
        automaton.make_automaton()
        return [end - len(needle) + 1 for end, _ in automaton.iter(joined_text)]
    out = []
    start = 0
    while True:
        found = joined_text.find(needle, start)
        if found < 0:
            break
        out.append(found)
        start = found + 1
    return out
