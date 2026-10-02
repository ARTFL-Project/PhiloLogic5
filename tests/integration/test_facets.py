"""Integration tests for facets (get_frequency.py): relative frequencies over the words of the objects the metadata
filters select."""

import os
import sys
import urllib.parse
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime.reports.frequency import frequency_results

AUTHOR = "Trollope, Anthony (1815-1882)"


def facets(db, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": root}, config)
    return {r["label"]: r for r in frequency_results(request, config)["results"]}


@pytest.mark.integration
class TestFacets:
    def test_doc_field_with_filter(self, eltec_db):
        results = facets(eltec_db, q="the", author=f'"{AUTHOR}"', frequency_field="year")
        expected = dict(eltec_db.dbh.execute(
            "SELECT year, SUM(word_count) FROM toms WHERE philo_type='doc' AND author=? GROUP BY year", (AUTHOR,)))
        assert {label: r["total_word_count"] for label, r in results.items()} == {str(y): w for y, w in expected.items()}

    def test_div_field_with_doc_filter(self, eltec_db):
        """A division value's words are those of its divisions in the documents selected, not in all."""
        results = facets(eltec_db, q="the", author=f'"{AUTHOR}"', frequency_field="head")
        assert results
        docs = {row[0].split()[0] for row in eltec_db.dbh.execute(
            "SELECT philo_id FROM toms WHERE philo_type='doc' AND author=?", (AUTHOR,))}
        for label, result in list(results.items())[:10]:
            words = sum(int(w or 0) for philo_id, w in eltec_db.dbh.execute(
                "SELECT philo_id, word_count FROM toms WHERE philo_type IN ('div1','div2','div3') AND head=?", (label,))
                if philo_id.split()[0] in docs)
            assert result["total_word_count"] == words, label

    def test_null_bucket(self, eltec_db):
        """The objects with no value are a bucket of their own (they were asked for as the string "NULL")."""
        nulls = facets(eltec_db, q="the", frequency_field="publisher").get("NULL")
        (docs,) = eltec_db.dbh.execute("SELECT COUNT(*) FROM toms WHERE philo_type='doc' AND publisher IS NULL").fetchone()
        assert (nulls is not None) == (docs > 0)
