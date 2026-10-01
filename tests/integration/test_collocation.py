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
    """The top collocates of q, as the collocation report counts them, within distance words or in the sentence."""
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    query_string = urllib.parse.urlencode({"q": q, "colloc_filter_choice": "nofilter", "method_arg": distance})
    request = WSGIHandler({"QUERY_STRING": query_string, "PHILOLOGIC_DBPATH": root}, config)
    return collocation_results(request, config)["collocates"]


def total_hits(db, q, method, method_arg="0"):
    hits = db.query(q, method, method_arg)
    hits.finish()
    return len(hits)


@pytest.mark.integration
class TestCountsMatchConcordances:
    """The count of a collocate is the number of hits of the concordance it links to: the query and the collocate in
    the same sentence, or within n words of each other."""

    @pytest.mark.parametrize("q", ["lord", "my lord"])
    def test_in_sentence(self, shakespeare_db, q):
        for word, count in collocation(shakespeare_db, q)[:5]:
            assert count == total_hits(shakespeare_db, f'{q} "{word}"', "sentence_unordered"), word

    @pytest.mark.parametrize("q", ["lord", "my lord"])
    def test_within_n_words(self, shakespeare_db, q):
        collocates = collocation(shakespeare_db, q, distance="3")[:5]
        assert collocates
        for word, count in collocates:
            assert count == total_hits(shakespeare_db, f'{q} "{word}"', "proxy_unordered", "3"), word


@pytest.mark.integration
class TestPhrase:
    def test_within_n_words(self, shakespeare_db):
        """Within n words of a phrase: the phrase and the collocate within n words."""
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
