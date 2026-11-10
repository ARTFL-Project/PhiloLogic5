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
            return _vectorized_collocation(shakespeare_db.path, Rows(rows), words, False, distance)

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


@pytest.mark.integration
class TestFrequencyFilter:
    """The words in more than filter_df_pct percent of the sentences are filtered; the clients that give only
    filter_frequency filter the most frequent words."""

    QUERY_WORDS = {"lord"}  # the words of the query's .terms file, which are filtered too

    def df_words(self, db, pct):
        from philologic.runtime.reports.collocation import _df_filter_list

        return set(_df_filter_list(db.path, False, pct))

    def top_n(self, db, n):
        with open(os.path.join(db.path, "frequencies", "word_frequencies"), encoding="utf8") as f:
            return {line.split()[0] for line, _ in zip(f, range(n)) if line.strip()}

    def test_the_df_filter_lists_the_words_in_many_sentences(self, shakespeare_db):
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_df_pct="1")
        df = np.load(os.path.join(shakespeare_db.path, "collocations", "sentence_df_word.npy"))
        n_sentences = len(np.load(os.path.join(shakespeare_db.path, "collocations", "sent_offsets.npy"))) - 1
        vocab = np.load(os.path.join(shakespeare_db.path, "collocations", "vocab.npy"), allow_pickle=True)
        expected = {str(w) for w in vocab[df > n_sentences * 0.01]} | self.QUERY_WORDS
        assert set(report["filter_list"]) == expected
        assert "the" in expected

    @pytest.mark.parametrize("given, resolved", [("0.1", 0.5), ("0.5", 0.5), ("2.5", 2.5), ("50", 5.0), ("inf", 5.0)])
    def test_the_percentage_is_clamped(self, shakespeare_db, given, resolved):
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_df_pct=given)
        assert set(report["filter_list"]) == self.df_words(shakespeare_db, resolved) | self.QUERY_WORDS

    def test_no_parameter_at_all_is_the_default_percentage(self, shakespeare_db):
        expected = self.df_words(shakespeare_db, 1.0) | self.QUERY_WORDS
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency")
        assert set(report["filter_list"]) == expected
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="")
        assert set(report["filter_list"]) == expected

    @pytest.mark.parametrize("bad", ["abc", "1,5", "nan", "5%"])
    def test_a_percentage_that_is_not_a_number(self, shakespeare_db, bad):
        from philologic.runtime.exceptions import BadRequest

        with pytest.raises(BadRequest, match="must be a number"):
            collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_df_pct=bad)

    def test_a_client_that_gives_only_filter_frequency_gets_the_top_n(self, shakespeare_db):
        """The clients built before the df filter send filter_frequency=100 and no filter_df_pct."""
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_frequency="100")
        assert set(report["filter_list"]) == self.top_n(shakespeare_db, 100) | self.QUERY_WORDS
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_frequency="7")
        assert set(report["filter_list"]) == self.top_n(shakespeare_db, 7) | self.QUERY_WORDS

    def test_the_percentage_given_wins_over_filter_frequency(self, shakespeare_db):
        report, _ = collocation_with(
            shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_frequency="100", filter_df_pct="2"
        )
        assert set(report["filter_list"]) == self.df_words(shakespeare_db, 2.0) | self.QUERY_WORDS
