"""Integration tests for collocation counting around hits of more than one word: phrases and co-occurrences."""

import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.reports.collocation import _vectorized_collocation

QUERY_WORDS = {"my", "My", "MY", "lord", "Lord", "LORD"}


class Rows:
    """Hits as _vectorized_collocation reads them, from an array of hit rows."""

    def __init__(self, rows):
        self.rows = np.ascontiguousarray(rows, dtype=np.uint32)
        self.length = self.rows.shape[1]

    def finish(self):
        pass

    def open_raw(self):
        return io.BytesIO(self.rows.tobytes())


def collocates(db, rows, distance=None, per_sentence=False):
    return _vectorized_collocation(
        db.path, Rows(rows), QUERY_WORDS, False, None, None, distance, per_sentence=per_sentence
    )


def hit_rows(db, q, method):
    hits = db.query(q, method, "0", raw_results=True)
    hits.finish()
    return hits.read_array()


@pytest.mark.integration
class TestCooccurrence:
    def test_each_sentence_counts_once(self, shakespeare_db):
        """A sentence with several combinations of the query words' occurrences counts once."""
        rows = hit_rows(shakespeare_db, "my lord", "sentence_unordered")
        sentence_ids = rows[:, :6]
        firsts = np.flatnonzero(np.concatenate(([True], np.any(sentence_ids[1:] != sentence_ids[:-1], axis=1))))
        assert len(firsts) < len(rows), "the corpus should have sentences with several combinations"
        per_sentence = collocates(shakespeare_db, rows, per_sentence=True)
        assert per_sentence == collocates(shakespeare_db, rows[firsts])
        assert sum(per_sentence.values()) < sum(collocates(shakespeare_db, rows).values())


@pytest.mark.integration
class TestPhrase:
    def test_distance_from_both_ends(self, shakespeare_db):
        """Collocates within n words of a phrase are on either side of it, not only around its first word."""
        rows = hit_rows(shakespeare_db, '"my lord"', "phrase_ordered")
        assert len(rows) > 0 and rows.shape[1] == 11
        first_word = rows[:, :9]
        second_word = np.hstack([rows[:, :7], rows[:, 9:11]])
        # Within 1 word of the phrase: left of "my", right of "lord" (the words in between are the query's)
        around_phrase = collocates(shakespeare_db, rows, distance=1)
        expected = Counter(collocates(shakespeare_db, first_word, distance=1))
        expected.update(collocates(shakespeare_db, second_word, distance=1))
        assert around_phrase == expected
        assert around_phrase != collocates(shakespeare_db, first_word, distance=1)
