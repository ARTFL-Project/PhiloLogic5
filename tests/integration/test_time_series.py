"""Integration tests for the time series report: hits and words of each period, of the objects its metadata selects."""

import os
import sys
import urllib.parse
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime.reports.time_series import generate_time_series


def time_series(db, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": root}, config)
    return generate_time_series(request, config)["results"]


def words_by_period(db, start, end, interval, where="1"):
    """The words of the documents of each period, from toms."""
    counts = {}
    for period in range(start, end + 1, interval):
        last = min(period + interval - 1, end)
        (words,) = db.dbh.execute(
            f"SELECT COALESCE(SUM(word_count), 0) FROM toms WHERE philo_type='doc' AND year BETWEEN ? AND ? AND {where}",
            (period, last),
        ).fetchone()
        counts[str(period)] = int(words)
    return counts


def hits_by_period(results):
    return {period: entry["count"] for period, entry in results["absolute_count"].items()}


@pytest.mark.integration
class TestTimeSeries:
    def test_whole_database(self, eltec_db):
        results = time_series(eltec_db, q="the", start_date=1841, end_date=1920, year_interval=20)
        assert results["date_count"] == words_by_period(eltec_db, 1841, 1920, 20)

    def test_metadata_filter(self, eltec_db):
        """Relative frequencies are of the words of the documents the metadata selects, not of all."""
        author = "Trollope, Anthony (1815-1882)"
        results = time_series(eltec_db, q="the", author=f'"{author}"', start_date=1841, end_date=1920, year_interval=20)
        expected = words_by_period(eltec_db, 1841, 1920, 20, where=f"author = '{author}'")
        assert results["date_count"] == expected
        assert any(expected.values()) and sum(expected.values()) < sum(words_by_period(eltec_db, 1841, 1920, 20).values())
        hits = eltec_db.query("the", "single_term", "0", author=f'"{author}"')
        hits.finish()
        assert sum(hits_by_period(results).values()) == len(hits)

    def test_metadata_only(self, eltec_db):
        """With no search term, the documents of each period the metadata selects."""
        author = "Trollope, Anthony (1815-1882)"
        results = time_series(eltec_db, author=f'"{author}"', start_date=1841, end_date=1920, year_interval=20)
        assert sum(hits_by_period(results).values()) == 3

    def test_last_period_cut(self, eltec_db):
        """The last period stops at the end date."""
        results = time_series(eltec_db, q="the", start_date=1841, end_date=1855, year_interval=10)
        assert results["date_count"] == words_by_period(eltec_db, 1841, 1855, 10)

    def test_year_filter_kept(self, eltec_db):
        results = time_series(eltec_db, q="the", year="1855", start_date=1841, end_date=1920, year_interval=20)
        hits = eltec_db.query("the", "single_term", "0", year="1855")
        hits.finish()
        assert sum(hits_by_period(results).values()) == len(hits) > 0

    @pytest.mark.parametrize("interval", ["0", "-10", "abc"])
    def test_bad_interval(self, eltec_db, interval):
        """The database's interval (10) instead: 0 made range() fail."""
        results = time_series(eltec_db, q="the", start_date=1841, end_date=1860, year_interval=interval)
        assert list(results["absolute_count"]) == ["1841", "1851"]
