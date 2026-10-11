"""Unit tests for the helpers of the word trends reports (philologic.runtime.reports.word_trends), on made-up data."""

import json
from types import SimpleNamespace

import numpy as np
import orjson
from scipy.special import ndtr
from scipy.stats import mannwhitneyu

from philologic.runtime.reports import word_trends as wt


class TestRankTest:
    def test_it_is_mann_and_whitneys_test_with_ties(self):
        rng = np.random.default_rng(0)
        for shift in (0.0, 0.2, 0.5):
            before = np.round(rng.random(30), 1)  # rounded: many ties, as works sharing no word are all at 0
            after = np.round(rng.random(20) + shift, 1)
            z, share = wt._rank_test(before, after)
            expected = mannwhitneyu(after, before, alternative="greater", method="asymptotic", use_continuity=False)
            assert abs(float(ndtr(-z)) - expected.pvalue) < 1e-9
            assert abs(share - expected.statistic / (len(before) * len(after))) < 1e-12

    def test_later_works_all_closer_give_the_whole_share(self):
        z, share = wt._rank_test(np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.6, 0.7, 0.8]))
        assert share == 1.0 and z > 0

    def test_later_works_all_farther_give_none(self):
        z, share = wt._rank_test(np.array([0.5, 0.6, 0.7]), np.array([0.1, 0.2]))
        assert share == 0.0 and z < 0

    def test_all_alike_tell_nothing(self):
        z, share = wt._rank_test(np.zeros(5), np.zeros(4))
        assert z == 0.0 and share == 0.5


class TestDispersion:
    def test_sentences_drawn_alike_are_not_overdispersed(self):
        rng = np.random.default_rng(1)
        n = rng.integers(50, 200, 400).astype(np.float64)
        k = rng.binomial(n.astype(np.int64), 0.1).astype(np.float64)
        assert wt._dispersion(k, n) < 1.3

    def test_works_that_differ_are(self):
        rng = np.random.default_rng(2)
        n = rng.integers(50, 200, 400).astype(np.float64)
        p = rng.choice([0.01, 0.3], 400)  # some works use the word much more than others
        k = rng.binomial(n.astype(np.int64), p).astype(np.float64)
        assert wt._dispersion(k, n) > 5

    def test_never_below_one(self):
        n = np.full(10, 100.0)
        assert wt._dispersion(np.full(10, 10.0), n) == 1.0  # every work exactly at the rate


class TestAssociations:
    def test_words_always_together_go_together_and_independent_ones_do_not(self):
        n = 10000.0
        alone = np.array([500.0, 500.0, 1000.0])
        together = np.array(
            [
                [500.0, 500.0, 50.0],  # 0 and 1 are always in the same sentences
                [500.0, 500.0, 50.0],
                [50.0, 50.0, 1000.0],  # 2 is with them as often as chance would have it
            ]
        )
        goes_with, affinity = wt._associations(together, alone, n)
        assert goes_with[0, 1] and goes_with[1, 0]
        assert not goes_with[0, 2] and not goes_with[1, 2]
        assert not goes_with.diagonal().any()
        assert affinity[0, 1] > 0.99 and abs(affinity[0, 2]) < 1e-9

    def test_words_that_avoid_each_other_do_not_go_together(self):
        goes_with, affinity = wt._associations(np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.array([1000.0] * 2), 4000.0)
        assert not goes_with[0, 1] and affinity[0, 1] == -1.0


class TestCached:
    def test_a_fresh_report_reads_as_a_cached_one(self, tmp_path):
        # The first answer must serialize as the next ones do (orjson takes no numpy number)
        db = SimpleNamespace(hitlist_dir=str(tmp_path))
        calls = []

        def compute():
            calls.append(1)
            return {"z": np.float64(1.5), "n": np.int64(3).item(), "list": [round(np.float64(0.25), 2)]}

        first = wt._cached(db, "test", ["q"], compute)
        second = wt._cached(db, "test", ["q"], compute)
        assert len(calls) == 1
        assert orjson.dumps(first) == orjson.dumps(second)
        assert type(first["z"]) is float and type(first["list"][0]) is float
        assert json.loads(next(tmp_path.glob("*.test.json")).read_text())["z"] == 1.5
