"""Unit tests for the word senses report's method (philologic.runtime.reports.word_senses), on made-up sentences."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from philologic.runtime.reports import word_senses as ws

SHARED = [f"w{i}" for i in range(40)]  # words of every sense
SENSES = {name: [f"{name}{i}" for i in range(12)] for name in "abc"}
VOCAB = SHARED + [w for words in SENSES.values() for w in words]
INDEX = {w: i for i, w in enumerate(VOCAB)}


def sentences(specs, seed=0):
    """Sentences of works: specs is a list of (work, sense) for each sentence, sense "a", "b", "c" or None (shared
    words only). Each sentence has 8 shared words and 3 of its sense's. Returns the three channels and the works."""
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i, (_, sense) in enumerate(specs):
        words = list(rng.choice(SHARED, 8, replace=False))
        if sense:
            words += list(rng.choice(SENSES[sense], 3, replace=False))
        rows += [i] * len(words)
        cols += [INDEX[w] for w in words]
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(specs), len(VOCAB)))
    # The corpus: every word as frequent as any other, many other words besides
    p = np.full(len(VOCAB), 1.0 / (10 * len(VOCAB)))
    empty = csr_matrix((len(specs), 0))
    channels = [(X, p, 1.0 - p.sum()), (empty, np.empty(0), 1.0), (empty, np.empty(0), 1.0)]
    return channels, np.array([work for work, _ in specs])


def leaves(root):
    return [root] if not root.children else [leaf for child in root.children for leaf in leaves(child)]


def words_of(node, n=12):
    return {
        VOCAB[i]
        for i, *_ in ws.describe(node.counts[0], node.dists[0], node.parent_dists[0].p, node.parent_dists[0].p)[:n]
    }


@pytest.mark.unit
class TestSenses:
    def test_two_senses_used_across_works_are_found(self):
        specs = [(work, "a" if i % 2 else "b") for work in range(20) for i in range(30)]
        channels, works = sentences(specs)
        root = ws.induce_senses(channels, works)
        found = leaves(root)
        assert len(found) == 2
        described = [words_of(leaf) for leaf in found]
        assert any(d and d <= set(SENSES["a"]) for d in described)
        assert any(d and d <= set(SENSES["b"]) for d in described)
        assert root.test["z"] > ws.EVIDENCE
        assert root.test["sentences_won"] > 0.9 * root.test["sentences"]

    def test_sentences_go_to_their_sense(self):
        specs = [(work, "a" if i % 2 else "b") for work in range(20) for i in range(30)]
        channels, works = sentences(specs)
        root = ws.induce_senses(channels, works)
        nodes, logp, margin = ws.node_posteriors(root, [csr_matrix(X) for X, _, _ in channels])
        found = [node.id for node in nodes if not node.children]
        a_leaf = next(i for i in found if words_of(nodes[i]) <= set(SENSES["a"]))
        is_a = np.array([sense == "a" for _, sense in specs])
        assert np.mean((margin[:, a_leaf] > 0) == is_a) > 0.98
        assert np.allclose(np.exp(logp[:, found]).sum(axis=1), 1.0)  # each sentence is in one of the senses found
        assert np.all(np.isinf(margin[:, 0]))  # and in the whole query

    def test_no_senses_in_sentences_alike(self):
        specs = [(work, None) for work in range(20) for _ in range(30)]
        channels, works = sentences(specs)
        root = ws.induce_senses(channels, works)
        assert not root.children
        assert root.reason in ("no_gain", "inseparable")

    def test_senses_of_unequal_shares_are_found(self):
        specs = [(work, "a" if i % 3 == 0 else "b") for work in range(15) for i in range(20)]
        channels, works = sentences(specs)
        root = ws.induce_senses(channels, works)
        is_a = np.array([sense == "a" for _, sense in specs])
        assert sorted(set(is_a[child.units]) for child in root.children) == [{False}, {True}]

    def test_no_split_that_no_word_tells_apart(self):
        # A parent with no word that may mark a sense: the two senses would be the parent, twice
        specs = [(work, "a" if i % 2 else "b") for work in range(10) for i in range(20)]
        channels, works = sentences(specs)
        X = csr_matrix(channels[0][0])
        parent = ws.Dist(np.full(len(VOCAB), 1.0 / (2 * len(VOCAB))), 0.5, np.zeros(len(VOCAB), dtype=bool))
        r = np.column_stack([np.arange(200) % 2, 1 - np.arange(200) % 2]).astype(float)
        assert ws.fit_split([X], [parent], r, search=True) is None

    def test_senses_within_a_single_work_are_found(self):
        # Sentences are the units: one work's two senses are told apart, tested on its sentences left out in turn
        specs = [(0, "a" if i % 2 else "b") for i in range(300)]
        channels, works = sentences(specs)
        root = ws.induce_senses(channels, works)
        assert len(leaves(root)) == 2

    def test_importance_is_relative_to_the_main_distinction(self):
        # a against b and c, then b against c: the second distinction is within half the sentences, less important
        specs = [(work, "aabc"[i % 4]) for work in range(20) for i in range(40)]
        channels, works = sentences(specs)
        root = ws.induce_senses(channels, works)
        assert len(leaves(root)) == 3
        second = next(child for child in root.children if child.children)
        assert 0 < second.test["importance"] < root.test["importance"]
        # Distinctions less important than finest times the main one are not looked for
        coarse = ws.induce_senses(channels, works, finest=1.0)
        assert len(leaves(coarse)) == 2
        assert {child.reason for child in coarse.children} >= {"faint"}

    def test_the_same_senses_every_time(self):
        specs = [(work, "a" if i % 3 else "b") for work in range(15) for i in range(20)]
        channels, works = sentences(specs)
        Xall = [csr_matrix(X) for X, _, _ in channels]
        first = ws.node_posteriors(ws.induce_senses(channels, works), Xall)[1]
        second = ws.node_posteriors(ws.induce_senses(channels, works), Xall)[1]
        assert np.array_equal(first, second)

    def test_splits_start_from_a_word_of_a_sense(self):
        specs = [(work, "a" if i % 2 else "b") for work in range(20) for i in range(30)]
        channels, works = sentences(specs)
        X = csr_matrix(channels[0][0])
        collocates = [np.arange(len(VOCAB))]
        seeds = ws.seed_splits([X], collocates)
        assert 0 < len(seeds) <= ws._SEEDS_TRIED
        r, (group, word) = seeds[0]
        assert group == 0 and VOCAB[word][0] in "ab"
        assert np.array_equal(r[:, 0] > 0, np.asarray(X[:, word].todense()).ravel() > 0)


@pytest.mark.unit
class TestBreak:
    def test_the_largest_drop_decides_what_is_shown_at_first(self):
        révolution = [1.0, 0.75, 0.43, 0.32, 0.25, 0.23, 0.22, 0.16, 0.14, 0.13, 0.11]
        assert ws.natural_break(révolution) == 2  # 75% to 43%
        assert ws.natural_break([1.0, 0.31, 0.30, 0.18]) == 1

    def test_distinctions_all_alike_are_all_shown(self):
        assert ws.natural_break([1.0, 0.9, 0.85]) == 3  # the drop is to the floor of the search
        assert ws.natural_break([]) == 0


@pytest.mark.unit
class TestModel:
    def test_discount_from_words_seen_once_and_twice(self):
        assert ws.discount(np.array([1, 1, 1, 2, 5, 0])) == pytest.approx(3 / (3 + 2))
        assert ws.discount(np.array([3, 4])) == 1.0  # none in one sentence: as if one were
        assert ws.discount(np.array([2, 2, 3])) == pytest.approx(1 / (1 + 4))

    def test_probabilities_sum_to_one(self):
        parent = ws.Dist(np.array([0.1, 0.2, 0.3]), 0.4)
        dist = ws.fit_dist(np.array([5.0, 0.0, 1.0]), parent, 0.5)
        assert dist.p.sum() + dist.rest == pytest.approx(1.0)

    def test_frequent_words_keep_their_rate_and_unseen_ones_their_parents(self):
        parent = ws.Dist(np.array([0.01, 0.01, 0.01]), 0.97)
        dist = ws.fit_dist(np.array([1000.0, 0.0, 0.0]), parent, 0.5)
        assert dist.p[0] == pytest.approx(999.5 / 1000)
        # The unseen words share the discounted mass as the parent's probabilities do, the rest of the corpus included
        assert dist.p[1] / dist.rest == pytest.approx(0.01 / 0.97)

    def test_only_the_querys_collocates_mark_a_sense(self):
        markers = np.array([False, True, True])
        parent = ws.Dist(np.array([0.2, 0.2, 0.2]), 0.4, markers)
        # The first two far more frequent than in the parent, the third far less: only the collocates get rates of their
        # own, the other word keeps its parent's, scaled as the rest of the corpus is
        dist = ws.fit_sparse(np.array([300.0, 300.0, 40.0]), parent)
        assert dist.p[1] == pytest.approx(300 / 640)
        assert dist.p[2] == pytest.approx(40 / 640)
        assert dist.p[0] / parent.p[0] == pytest.approx(dist.rest / parent.rest)
        assert dist.p.sum() + dist.rest == pytest.approx(1.0)

    def test_a_word_of_the_other_sense_keeps_a_small_rate(self):
        # Absent from this sense of a split, it keeps a share of the discounted mass, not its parent's rate
        parent = ws.Dist(np.array([0.25, 0.25, 0.25]), 0.25, discount=0.5)
        dist = ws.fit_sparse(np.array([0.0, 100.0, 50.0]), parent, own=np.array([0, 1]))
        assert 0 < dist.p[0] < 0.01 * parent.p[0]
        assert dist.p[1] == pytest.approx(99.5 / 150, rel=0.01)
        assert dist.p.sum() + dist.rest == pytest.approx(1.0)

    def test_rarer_words_come_first_among_the_words_of_a_sense(self):
        parent_p = np.array([0.05, 0.05, 0.5])
        c = np.array([30.0, 30.0, 40.0])
        weights = np.array([0.5, 4.0, 1.0])  # the second word much rarer in the corpus
        dist = ws.fit_sparse(c, ws.Dist(parent_p, 0.9, None, weights))
        listed = ws.describe(c, dist, parent_p, parent_p)
        assert [i for i, *_ in listed] == [1, 0]
        # And weigh more in the evidence for a sense
        assert np.allclose(ws._evidence(dist), weights * np.log(dist.p))

    def test_words_new_to_a_sense_and_its_parent_are_not_discounted_twice(self):
        corpus = ws.Dist(np.array([0.001, 0.001]), 0.998)
        parent = ws.fit_dist(np.array([100.0, 0.0]), corpus, 0.5)
        child = ws.fit_dist(np.array([50.0, 0.0]), parent, 0.5)
        # A word in neither: the child gives it about what its parent does, not the parent's share of a share
        assert child.p[1] > 0.5 * parent.p[1]

    def test_works_stay_whole_in_folds_of_like_sizes(self):
        groups = np.repeat(np.arange(12), np.arange(1, 13))
        folds, n_folds = ws.work_folds(groups)
        assert n_folds == ws.N_FOLDS
        for g in range(12):
            assert len(set(folds[groups == g])) == 1
        sizes = np.bincount(folds)
        assert sizes.max() - sizes.min() <= 12

    def test_fewer_folds_than_works_when_there_are_few(self):
        folds, n_folds = ws.work_folds(np.array([7, 7, 9, 9, 9]))
        assert n_folds == 2
        assert ws.work_folds(np.array([3, 3]))[0] is None


def plain_em(Xs, parent, r, every_marker):
    """The EM of a split in plain numpy, as the compiled one must compute it."""
    n = Xs[0].shape[0]
    previous = None
    for _ in range(ws._MAX_ITER):
        weight = r.sum(axis=0)
        if weight.min() < 1.0:
            return None
        pi = weight / n
        counts = [X.T @ r for X in Xs]
        owns = [ws._split_words(c, parent[ci], every_marker) for ci, c in enumerate(counts)]
        if not any(len(own) for own in owns):
            return None
        dists = [
            [ws.fit_sparse(c[:, k], parent[ci], own) for k in range(2)] for ci, (c, own) in enumerate(zip(counts, owns))
        ]
        ll = ws._mixture_loglik(Xs, pi, dists)
        total = ws._logsumexp2(ll)[:, None]
        r = np.exp(ll - total)
        total = float(total.sum())
        if previous is not None and abs(total - previous) <= ws._TOL * abs(total):
            break
        previous = total
    if r.sum(axis=0).min() < 1.0:
        return None
    return pi, dists, r, counts


@pytest.mark.unit
class TestCompiled:
    def test_the_compiled_em_is_numpy_to_the_last_bit(self):
        specs = [(work, "abc"[i % 3]) for work in range(12) for i in range(25)]
        channels, works = sentences(specs, seed=4)
        X = csr_matrix(channels[0][0])
        rng = np.random.default_rng(5)
        weights = rng.uniform(0.5, 6.0, X.shape[1])
        # Two groups of words, the second a few of the first's columns, weighted, with markers
        Xs = [X, csr_matrix(X[:, :20])]
        parent = [
            ws.Dist(np.full(X.shape[1], 0.8 / X.shape[1]), 0.2, rng.uniform(size=X.shape[1]) > 0.2, weights, 0.6),
            ws.Dist(np.full(20, 0.9 / 20), 0.1, None, None, 0.3),
        ]
        r = rng.dirichlet([1.0, 1.0], X.shape[0])
        for every_marker in (True, False):
            mine, plain = ws._em(Xs, parent, r, every_marker), plain_em(Xs, parent, r, every_marker)
            assert np.array_equal(mine[0], plain[0]) and np.array_equal(mine[2], plain[2])
            for ds, dp in zip(mine[1], plain[1]):
                for a, b in zip(ds, dp):
                    assert np.array_equal(a.p, b.p) and a.rest == b.rest
            assert all(np.array_equal(a, b) for a, b in zip(mine[3], plain[3]))

    def test_sums_are_numpys_to_the_last_bit(self):
        rng = np.random.default_rng(6)
        for n in list(range(0, 300)) + [1000, 4096, 8193, 100003]:
            a = rng.lognormal(0, 3, size=(n, 2))
            assert ws._pairwise_sum(a[:, 1], 0, n) == float(a[:, 1].sum())
            assert ws._pairwise_sum(np.ascontiguousarray(a[:, 0]), 0, n) == float(a[:, 0].sum())


@pytest.mark.unit
class TestSmoothing:
    def test_lines_are_left_as_they_are(self):
        line = 0.1 + 0.02 * np.arange(20)
        for lam in (0.1, 1e3, 1e8):
            assert np.allclose(ws.whittaker(np.ones(20), line[None, :], lam)[0], line)

    def test_shares_that_sum_to_one_still_do(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, 30)
        shares = np.vstack([a, 1 - a])
        w = rng.integers(0, 5, 30).astype(float)
        w[:2] = 1.0
        smoothed = ws.whittaker(w, shares, 50.0)
        assert np.allclose(smoothed.sum(axis=0), 1.0)

    def test_cross_validation_smooths_noise_away(self):
        rng = np.random.default_rng(1)
        x = np.linspace(0, 3, 120)
        truth = 0.5 + 0.3 * np.sin(x)
        noisy = truth + rng.normal(0, 0.1, len(x))
        w = np.ones(len(x))
        lam = ws.choose_smoothness(w, noisy[None, :])
        assert ws._LAMBDAS[0] < lam < ws._LAMBDAS[-1]
        smoothed = ws.whittaker(w, noisy[None, :], lam)[0]
        assert np.mean((smoothed - truth) ** 2) < 0.3 * np.mean((noisy - truth) ** 2)

    def test_inverse_diagonal_is_exact(self):
        w = np.array([1.0, 0.0, 2.0, 1.0, 3.0, 0.5, 1.0])
        lam = 3.0
        n = len(w)
        D = np.diff(np.eye(n), 2, axis=0)
        A = np.diag(w) + lam * D.T @ D
        d, l1, l2 = ws._ldl(w, lam)
        assert np.allclose(ws._inverse_diagonal(d, l1, l2), np.diag(np.linalg.inv(A)))

    def test_course_over_time_with_bands(self):
        years = np.repeat(np.arange(1800, 1810), 20)
        works = np.repeat(np.arange(40), 5)
        P = np.column_stack([np.linspace(0.2, 0.8, len(years)), 1 - np.linspace(0.2, 0.8, len(years))])
        curves = ws.sense_curves(P, years, works, {y: 1000 for y in range(1800, 1810)})
        assert curves["start"] == 1800 and curves["end"] == 1809
        value = np.array(curves["share"]["value"])
        assert value[0, -1] > value[0, 0]  # the first sense rises
        low, high = np.array(curves["share"]["low"]), np.array(curves["share"]["high"])
        assert np.all(low <= value + 1e-9) and np.all(value <= high + 1e-9)
        assert curves["rate"] is not None

    def test_no_course_without_years(self):
        P = np.ones((10, 1))
        assert ws.sense_curves(P, np.zeros(10, dtype=int), np.arange(10)) is None
        assert ws.sense_curves(P, np.full(10, 1800), np.arange(10)) is None
