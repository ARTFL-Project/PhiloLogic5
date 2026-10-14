"""Unit tests for the senses over time's periods and words (philologic.runtime.reports.sense_periods)."""

import numpy as np
from scipy.stats import chi2_contingency

from philologic.runtime.reports import sense_periods as sp
from philologic.runtime.reports import word_senses as ws


def sense(years_of_works, sentences_per_work=3):
    """A sense's sentences, sentences_per_work for each work, the works with these years, in year order."""
    s = object.__new__(sp._Sense)
    docs = np.repeat(np.arange(len(years_of_works)), sentences_per_work)
    s.docs = docs
    s.yr = np.asarray(years_of_works)[docs]
    s.rows = np.arange(len(docs))
    return s


class TestG2:
    def test_it_is_the_log_likelihood_ratio_of_the_2x2_table(self):
        for a, n1, c, n2 in ((74, 29902, 3, 56986), (164, 29902, 44, 56986), (5, 100, 5, 100)):
            expected = chi2_contingency([[a, n1 - a], [c, n2 - c]], correction=False, lambda_="log-likelihood")[0]
            assert abs(float(sp._g2(np.array([a]), n1, np.array([c]), n2)[0]) - expected) < 1e-6


class TestCut:
    def test_periods_of_as_many_works_in_year_order(self):
        s = sense([1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807])
        period, n = s.cut(4)
        assert n == 4
        works_per_period = [len(np.unique(s.docs[period == k])) for k in range(n)]
        assert works_per_period == [2, 2, 2, 2]
        assert all(s.yr[period == k].max() < s.yr[period == k + 1].min() for k in range(n - 1))

    def test_the_works_of_a_year_stay_together(self):
        s = sense([1800, 1800, 1800, 1801, 1802, 1802])
        period, n = s.cut(2)
        for year in (1800, 1802):
            assert len(np.unique(period[s.yr == year])) == 1

    def test_no_more_periods_than_years(self):
        s = sense([1800, 1800, 1801, 1801])
        assert s.cut(3)[1] == 2


class TestChoice:
    def test_the_most_periods_before_the_first_that_fails(self):
        s = sense(list(range(1800, 1820)))
        s.has_new = lambda period, k: int(period.max()) + 1 <= 4  # 2, 3 and 4 periods hold; 5 does not
        s.describe = lambda period: int(period.max()) + 1
        n, tried = s.periods()
        assert n == 4 and tried == [[2, True], [3, True], [4, True], [5, False]]

    def test_one_period_when_two_fail(self):
        s = sense(list(range(1800, 1820)))
        s.has_new = lambda period, k: False
        s.describe = lambda period: int(period.max()) + 1
        assert s.periods() == (1, [[2, False]])


class TestMembership:
    def test_the_senses_membership_reads_back_as_written(self, tmp_path):
        rng = np.random.default_rng(0)
        membership = rng.random((37, 5)) > 0.5  # sentences x senses, not a multiple of 8 sentences
        path = str(tmp_path / "x.membership.npz")
        ws._save_membership(path, membership)
        assert np.array_equal(ws._load_membership(path), membership)

    def test_a_missing_file_reads_as_none(self, tmp_path):
        assert ws._load_membership(str(tmp_path / "none.membership.npz")) is None
