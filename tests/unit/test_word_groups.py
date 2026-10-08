"""Unit tests for the word groups of the word senses report (links and hub groups in
philologic.runtime.reports.word_senses), on made-up sentences."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from philologic.runtime.reports import word_senses as ws

TOPIC_A = [f"a{i}" for i in range(10)]
TOPIC_B = [f"b{i}" for i in range(10)]
NOISE = [f"w{i}" for i in range(40)]
VOCAB = TOPIC_A + TOPIC_B + ["shared"] + NOISE
INDEX = {w: i for i, w in enumerate(VOCAB)}


def sentences(specs, seed=0):
    """Sentences: specs is a list of topics, "a", "b" or None (noise only); each sentence has 4 noise words, 4 of its
    topic's and, one in two, the word both topics share (more frequent in either than in the sentences on nothing in
    particular). Returns sentences x words, 1 where a sentence has a word."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for s, topic in enumerate(specs):
        words = list(rng.choice(NOISE, 4, replace=False))
        if topic:
            words += list(rng.choice(TOPIC_A if topic == "a" else TOPIC_B, 4, replace=False))
            if s % 2 == 0:
                words.append("shared")
        rows += [s] * len(words)
        cols += [INDEX[w] for w in words]
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(specs), len(VOCAB)))


@pytest.mark.unit
class TestLinks:
    def test_words_of_a_topic_are_linked_and_others_not(self):
        X = sentences(["a", "b"] * 200)
        i, j, w, count = ws.links_of(X)
        pairs = {(VOCAB[a][0], VOCAB[b][0]) for a, b in zip(i, j)}
        assert ("a", "a") in pairs and ("b", "b") in pairs
        assert ("a", "b") not in pairs and ("b", "a") not in pairs  # never together
        assert np.all((w > 0) & (w <= 1))

    def test_a_pair_seen_once_is_no_link(self):
        X = csr_matrix(np.array([[1, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=float))
        i, j, _, _ = ws.links_of(X)
        assert not ((i == 0) & (j == 1)).any()

    def test_a_pair_repeated_in_one_work_is_no_link_but_one_across_works_is(self):
        # 20 works of 30 sentences, 6 words each; x and y together 12 times in work 0 only, u and v once in each of 12
        rng = np.random.default_rng(7)
        n_works, per_work, vocab = 20, 30, 60
        rows, cols = [], []
        for s in range(n_works * per_work):
            for word in rng.choice(np.arange(4, vocab), 6, replace=False):
                rows.append(s)
                cols.append(word)
        works = np.repeat(np.arange(n_works), per_work)
        for s in range(12):
            rows += [s, s]
            cols += [0, 1]  # x, y: one passage of work 0
        for d in range(12):
            rows += [d * per_work + 5] * 2
            cols += [2, 3]  # u, v: across works
        X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_works * per_work, vocab))
        X.data[:] = 1.0
        i, j, _, _ = ws.links_of(X)
        assert ((i == 0) & (j == 1)).any()  # by sentences alone, the passage's pair is a link
        i, j, _, _ = ws.links_of(X, works)
        assert not ((i == 0) & (j == 1)).any()
        assert ((i == 2) & (j == 3)).any()


@pytest.mark.unit
class TestHubGroups:
    def test_each_topic_has_its_hub_and_its_words(self):
        X = sentences(["a", "b", None] * 200)
        groups = ws.hub_groups(X, np.arange(600) // 20)  # 30 works
        assert len(groups) >= 2
        for hub, members in groups[:2]:
            kind = VOCAB[hub][0]
            other = "b" if kind == "a" else "a"
            words = {VOCAB[m] for m in members}
            assert len(words & set(TOPIC_A if kind == "a" else TOPIC_B)) >= 8  # its topic's words
            assert not words & set(TOPIC_A if other == "a" else TOPIC_B)  # none of the other's
        assert {VOCAB[h][0] for h, _ in groups[:2]} == {"a", "b"}

    def test_no_groups_without_links(self):
        assert ws.hub_groups(sentences([None] * 300), np.arange(300) // 20) == []
