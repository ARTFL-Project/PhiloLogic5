"""Unit tests for the co-occurrence kernels of multi_word_search: each hit once, against a brute-force reference, and
in reasonable time for sentences with hundreds of hits of each group (a few such made searches time out)."""

import itertools
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import philologic.runtime.Query  # noqa: F401 (sets the Numba cache directory)
from philologic.runtime.multi_word_search import (
    _cooc_match_doc_two_groups,
    _find_common_sentences,
    _groups_overlap,
    _process_n_groups,
)


def make_hits(sentences, words):
    """The hits, sorted by byte, of words in sentences: lists of words, one sentence each, of document 1."""
    rows = []
    for sent, tokens in enumerate(sentences, 1):
        for pos, token in enumerate(tokens, 1):
            if token in words:
                rows.append([1, 1, 1, 1, 1, sent, 0, pos, sent * 100000 + pos * 10])
    return np.array(rows, dtype=np.uint32).reshape(-1, 9)


def positions(rows, n_groups):
    """Each hit as (sentence, positions of its words in text order)."""
    return [(int(r[5]), tuple(int(r[7 + 2 * g]) for g in range(n_groups))) for r in rows]


def reference(sentences, groups, ordered, max_distance=0, exact=False):
    """Every hit, once: the positions of one word of each group in a sentence, in query order if ordered, all
    different, within max_distance (or at exactly it)."""
    hits = set()
    for sent, tokens in enumerate(sentences, 1):
        choices = [[p for p, t in enumerate(tokens, 1) if t in group] for group in groups]
        for combo in itertools.product(*choices):
            if len(set(combo)) < len(combo) or (ordered and list(combo) != sorted(combo)):
                continue
            span = max(combo) - min(combo)
            if max_distance and (span != max_distance if exact else span > max_distance):
                continue
            hits.add((sent, tuple(sorted(combo))))
    return hits


def search(sentences, groups, ordered, max_distance=0, exact=False):
    dedup = not ordered and _groups_overlap(groups)
    hits = [make_hits(sentences, set(g)) for g in groups]
    if len(groups) == 2:
        rows = _cooc_match_doc_two_groups(hits[0], hits[1], 6, ordered, max_distance, exact, dedup)
    else:
        data = _find_common_sentences(hits, 6)
        if data is None:
            return []
        rows = _process_n_groups([d[0] for d in data], [d[1] for d in data], ordered, list(range(len(groups))),
                                 max_distance, exact, len(groups), dedup)
    return positions(rows, len(groups))


SENTENCES = [
    "de la de le la de et la de".split(),
    "la de de de la et le de".split(),
    "et le la".split(),
    "de de".split(),
]
GROUPS = [
    (["de"], ["la"]),
    (["de"], ["de"]),
    (["de", "la"], ["la", "le"]),
    (["de"], ["la"], ["et"]),
    (["de"], ["de"], ["la"]),
    (["de"], ["de"], ["de"]),
    (["de", "la"], ["la", "et"], ["et", "de"]),  # each group overlapping the next, in a cycle
    (["la"], ["de"], ["le"], ["et"]),
]


@pytest.mark.unit
class TestCooccurrences:
    @pytest.mark.parametrize("groups", GROUPS, ids=["+".join("|".join(g) for g in gs) for gs in GROUPS])
    @pytest.mark.parametrize("ordered", [True, False], ids=["ordered", "unordered"])
    @pytest.mark.parametrize("max_distance, exact", [(0, False), (3, False), (2, True)], ids=["sentence", "within 3", "exactly 2"])
    def test_each_hit_once(self, groups, ordered, max_distance, exact):
        found = search(SENTENCES, groups, ordered, max_distance, exact)
        assert len(found) == len(set(found))
        assert set(found) == reference(SENTENCES, groups, ordered, max_distance, exact)

    def test_groups_overlap(self):
        assert _groups_overlap([["de"], ["la", "de"]])
        assert not _groups_overlap([["de"], ["la"], ["le", "les"]])

    @pytest.mark.parametrize(
        "groups, ordered, expected",
        [
            ((["a"], ["b"], ["c"]), True, comb(100 + 2, 3)),  # the a, b and c of indices ia <= ib <= ic
            ((["a"], ["b"], ["c"]), False, 100**3),
            ((["a"], ["a"], ["b"]), False, comb(100, 2) * 100),
            ((["a"], ["a"]), False, comb(100, 2)),
        ],
    )
    def test_long_sentence(self, groups, ordered, expected):
        """One sentence of 100 times "a b c": up to a million hits, which each used to be compared with all earlier
        ones of the sentence (hours)."""
        start = time.time()
        found = search([["a", "b", "c"] * 100], groups, ordered)
        assert time.time() - start < 30
        assert len(found) == len(set(found)) == expected
