"""Integration tests for the table of contents report: the divisions of a document it lists."""

import os
import urllib.parse

import pytest

from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime.reports.table_of_contents import generate_toc_object


def toc_ids(db, philo_id):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    query = urllib.parse.urlencode({"philo_id": philo_id})
    request = WSGIHandler({"QUERY_STRING": query, "PHILOLOGIC_DBPATH": root}, config)
    return [entry["philo_id"] for entry in generate_toc_object(request, config)["toc"]]


@pytest.mark.integration
class TestTableOfContents:
    def test_no_divisions_without_words(self, eltec_db):
        """Divisions of no words open an empty page: toms has their word count as text, so "0" == 0 kept them."""
        empty = [
            row["philo_id"]
            for row in eltec_db.dbh.execute(
                "SELECT philo_id FROM toms WHERE philo_type IN ('div1', 'div2', 'div3') AND word_count = '0'"
            )
        ]
        assert empty
        for philo_id in empty:
            doc_id = philo_id.split()[0]
            depth = {"div1": 2, "div2": 3, "div3": 4}[eltec_db[philo_id].philo_type]
            ids = toc_ids(eltec_db, doc_id)
            assert ids and " ".join(philo_id.split()[:depth]) not in ids
