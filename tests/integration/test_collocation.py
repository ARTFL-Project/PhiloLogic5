"""Integration tests for collocation counting: around hits of one word, of co-occurrences, and of phrases."""

import io
import os
import sys
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import WebConfig, WSGIHandler
from philologic.runtime.reports.collocation import _vectorized_collocation, collocation_results


class Rows:
    """Hits as _vectorized_collocation reads them, from an array of hit rows."""

    def __init__(self, rows):
        self.rows = np.ascontiguousarray(rows, dtype=np.uint32)
        self.length = self.rows.shape[1]

    def finish(self):
        pass

    def open_raw(self):
        return io.BytesIO(self.rows.tobytes())


def collocation(db, q, distance=""):
    """The collocation report for q, counting collocates within distance words or in the sentence."""
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    query_string = urllib.parse.urlencode({"q": q, "colloc_filter_choice": "nofilter", "method_arg": distance})
    request = WSGIHandler({"QUERY_STRING": query_string, "PHILOLOGIC_DBPATH": root}, config)
    return collocation_results(request, config)


def total_hits(db, q, method, method_arg="0"):
    hits = db.query(q, method, method_arg)
    hits.finish()
    return len(hits)


@pytest.mark.integration
class TestCountsMatchConcordances:
    """The hits of the report, and the count of each collocate, are those of the concordances they link to: the
    query (and the collocate) in the same sentence, or within n words of each other, unordered. A phrase is one word:
    within n words of its first or last word."""

    @pytest.mark.parametrize("q", ["lord", "my lord", '"my lord"', '"my lord" king'])
    def test_in_sentence(self, shakespeare_db, q):
        report = collocation(shakespeare_db, q)
        assert report["results_length"] == total_hits(shakespeare_db, q, "sentence_unordered")
        for word, count in report["collocates"][:5]:
            assert count == total_hits(shakespeare_db, f'{q} "{word}"', "sentence_unordered"), word

    @pytest.mark.parametrize("q", ["lord", "my lord", '"my lord"', '"my lord" king'])
    def test_within_n_words(self, shakespeare_db, q):
        report = collocation(shakespeare_db, q, distance="3")
        assert report["results_length"] == total_hits(shakespeare_db, q, "proxy_unordered", "3")
        assert report["collocates"]
        for word, count in report["collocates"][:5]:
            assert count == total_hits(shakespeare_db, f'{q} "{word}"', "proxy_unordered", "3"), word


@pytest.mark.integration
class TestPhrase:
    def test_within_n_words(self, shakespeare_db):
        """_vectorized_collocation counts collocates within n words of all the query words, of a phrase too
        (collocation_results passes it n plus the words of the phrase after its first, for within n words of it)."""
        hits = shakespeare_db.query('"my lord"', "phrase_ordered", "0", raw_results=True)
        hits.finish()
        rows = hits.read_array()
        assert len(rows) > 0 and rows.shape[1] == 11
        words = {"my", "My", "MY", "lord", "Lord", "LORD"}

        def count(rows, distance):
            return _vectorized_collocation(shakespeare_db.path, Rows(rows), words, False, None, None, distance)

        # The two words already span 1: no collocate is within 1 word of both
        assert count(rows, 1) == Counter()
        # Within 2: the word before "my" and the word after "lord"
        expected = Counter(count(rows[:, :9], 1))
        expected.update(count(np.hstack([rows[:, :7], rows[:, 9:11]]), 1))
        assert count(rows, 2) == expected


def collocation_with(db, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": root}, config)
    return collocation_results(request, config), config


@pytest.mark.integration
class TestRequests:
    def test_missing_stopwords(self, shakespeare_db, monkeypatch):
        """A stopwords list that isn't found is said, rather than taken for a word to filter."""
        from philologic.runtime.reports import collocation as module

        real = module.build_filter_list
        report, config = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="stopwords")
        config.stopwords = "/nonexistent/stopwords.txt"
        assert real(type("R", (), {"colloc_filter_choice": "stopwords", "filter_frequency": ""})(), config, False) is None

    def test_bad_filter_frequency(self, shakespeare_db):
        from philologic.runtime.exceptions import BadRequest

        with pytest.raises(BadRequest, match="must be a number"):
            collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_frequency="1OO")

    def test_negative_distance(self, shakespeare_db):
        from philologic.runtime.exceptions import BadRequest

        with pytest.raises(BadRequest, match="negative"):
            collocation_with(shakespeare_db, q="lord", method_arg="-3")

    def test_map_field_not_metadata(self, shakespeare_db):
        from philologic.runtime.exceptions import BadRequest

        with pytest.raises(BadRequest, match="metadata field"):
            collocation_with(shakespeare_db, q="lord", map_field="word_count")


@pytest.mark.integration
class TestByMetadata:
    def test_div_field(self, shakespeare_db):
        """Collocates counted by a div field ("Similar Usage") are grouped by the values the hits' citations show. The
        words of a scene, in its implicit div3, had the div3's empty head instead: a single group, named ""."""
        from philologic.runtime.reports.collocation import load_map_field_cache

        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="nofilter", map_field="head")
        group_names = load_map_field_cache(report["file_path"])[3]
        hits = shakespeare_db.query("lord", "single_term", "0")
        hits.finish()
        assert len(group_names) > 1 and "" not in group_names
        assert set(group_names) == {hit["head"] for hit in hits} - {""}
