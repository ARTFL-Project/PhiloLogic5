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
    query (and the collocate) in the same sentence, or within n words of each other, unordered."""

    @pytest.mark.parametrize("q", ["lord", "my lord"])
    def test_in_sentence(self, shakespeare_db, q):
        report = collocation(shakespeare_db, q)
        assert report["results_length"] == total_hits(shakespeare_db, q, "sentence_unordered")
        for word, count in report["collocates"][:5]:
            assert count == total_hits(shakespeare_db, f'{q} "{word}"', "sentence_unordered"), word

    @pytest.mark.parametrize("q", ["lord", "my lord"])
    def test_within_n_words(self, shakespeare_db, q):
        report = collocation(shakespeare_db, q, distance="3")
        assert report["results_length"] == total_hits(shakespeare_db, q, "proxy_unordered", "3")
        assert report["collocates"]
        for word, count in report["collocates"][:5]:
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


def database_copy(db, tmp_path, without=()):
    """A database that links to the files of db, but for the collocation counts named in without: its own
    data/collocations has links to the arrays, so counts can be written there (and nowhere in db)."""
    root = tmp_path / "root"
    data = root / "data"
    data.mkdir(parents=True)
    for name in os.listdir(db.path):
        if name != "collocations":
            os.symlink(os.path.join(db.path, name), data / name)
    (data / "collocations").mkdir()
    for name in os.listdir(os.path.join(db.path, "collocations")):
        if name not in without:
            os.symlink(os.path.join(db.path, "collocations", name), data / "collocations" / name)
    return data.parent


def collocation_in(root, **params):
    config = WebConfig(str(root))
    request = WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": str(root)}, config)
    return collocation_results(request, config)


@pytest.mark.integration
class TestFrequencyFilter:
    """The words in more than filter_df_pct percent of the sentences are filtered; a database without their counts,
    and the clients that give only filter_frequency, filter the most frequent words."""

    COUNTS = ("sentence_df_word.npy", "sentence_df_lemma.npy", "token_counts_word.npy", "token_counts_lemma.npy")

    def top_n(self, db, n, lemmas=False):
        from philologic.runtime.reports.collocation import build_filter_list

        config = type("C", (), {"db_path": os.path.dirname(os.path.normpath(db.path)), "stopwords": ""})()
        request = type("R", (), {"colloc_filter_choice": "frequency", "filter_frequency": str(n), "filter_df_pct": ""})()
        return build_filter_list(request, config, lemmas)

    def test_the_df_filter_lists_the_words_in_many_sentences(self, shakespeare_db):
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_df_pct="1")
        assert report["frequency_filter"] == {"type": "df", "pct": 1.0}
        df = np.load(os.path.join(shakespeare_db.path, "collocations", "sentence_df_word.npy"))
        n_sentences = len(np.load(os.path.join(shakespeare_db.path, "collocations", "sent_offsets.npy"))) - 1
        vocab = np.load(os.path.join(shakespeare_db.path, "collocations", "vocab.npy"), allow_pickle=True)
        expected = {str(w) for w in vocab[df > n_sentences * 0.01]} | {"lord", "Lord", "LORD"}
        assert set(report["filter_list"]) == expected
        assert "the" in expected

    @pytest.mark.parametrize("given, resolved", [("0.1", 0.5), ("0.5", 0.5), ("2.5", 2.5), ("50", 5.0), ("inf", 5.0)])
    def test_the_percentage_is_clamped(self, shakespeare_db, given, resolved):
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_df_pct=given)
        assert report["frequency_filter"] == {"type": "df", "pct": resolved}

    def test_no_parameter_at_all_is_the_default_percentage(self, shakespeare_db):
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency")
        assert report["frequency_filter"] == {"type": "df", "pct": 1.0}
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="")
        assert report["frequency_filter"] == {"type": "df", "pct": 1.0}

    @pytest.mark.parametrize("bad", ["abc", "1,5", "nan", "5%"])
    def test_a_percentage_that_is_not_a_number(self, shakespeare_db, bad):
        from philologic.runtime.exceptions import BadRequest

        with pytest.raises(BadRequest, match="must be a number"):
            collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_df_pct=bad)

    def test_a_client_that_gives_only_filter_frequency_gets_the_top_n(self, shakespeare_db):
        """The clients built before the df filter send filter_frequency=100 and no filter_df_pct."""
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_frequency="100")
        assert report["frequency_filter"] == {"type": "top_n", "n": 100}
        assert set(report["filter_list"]) == set(self.top_n(shakespeare_db, 100)) | {"lord", "Lord", "LORD"}
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_frequency="7")
        assert report["frequency_filter"] == {"type": "top_n", "n": 7}
        assert len(report["filter_list"]) <= 7 + 3

    def test_the_percentage_given_wins_over_filter_frequency(self, shakespeare_db):
        report, _ = collocation_with(
            shakespeare_db, q="lord", colloc_filter_choice="frequency", filter_frequency="100", filter_df_pct="2"
        )
        assert report["frequency_filter"] == {"type": "df", "pct": 2.0}

    def test_no_filter_says_none(self, shakespeare_db):
        report, _ = collocation_with(shakespeare_db, q="lord", colloc_filter_choice="nofilter")
        assert "frequency_filter" not in report

    def test_stopwords_say_none(self, shakespeare_db, tmp_path):
        """A database with stopwords filters them, not the common words (one without has none to filter, and answers with
        the common words, as it always did)."""
        root = database_copy(shakespeare_db, tmp_path)
        data = root / "data"
        (data / "stopwords.txt").write_text("the\nof\n")
        (data / "web_config.cfg").unlink()  # a link to the original's, which isn't written
        (data / "web_config.cfg").write_text(
            open(os.path.join(shakespeare_db.path, "web_config.cfg")).read() + '\nstopwords = "stopwords.txt"\n'
        )
        report = collocation_in(root, q="lord", colloc_filter_choice="stopwords")
        assert "frequency_filter" not in report
        assert {"the", "of"} <= set(report["filter_list"]) and "a" not in report["filter_list"]

    def test_without_counts_the_top_n_words_are_filtered(self, shakespeare_db, tmp_path):
        root = database_copy(shakespeare_db, tmp_path, without=self.COUNTS)
        for params, n in (({"filter_df_pct": "1"}, 100), ({"filter_df_pct": "1", "filter_frequency": "30"}, 30), ({}, 100)):
            report = collocation_in(root, q="lord", colloc_filter_choice="frequency", **params)
            assert report["frequency_filter"] == {"type": "top_n", "n": n}
            assert set(report["filter_list"]) == set(self.top_n(shakespeare_db, n)) | {"lord", "Lord", "LORD"}

    def test_without_counts_a_bad_filter_frequency_is_still_refused(self, shakespeare_db, tmp_path):
        from philologic.runtime.exceptions import BadRequest

        root = database_copy(shakespeare_db, tmp_path, without=self.COUNTS)
        with pytest.raises(BadRequest, match="must be a number"):
            collocation_in(root, q="lord", colloc_filter_choice="frequency", filter_df_pct="1", filter_frequency="1OO")

    def test_counts_of_the_words_only_fall_back_for_lemmas(self, eltec_db, tmp_path):
        root = database_copy(eltec_db, tmp_path, without=("sentence_df_lemma.npy", "token_counts_lemma.npy"))
        words = collocation_in(root, q="house", colloc_filter_choice="frequency", filter_df_pct="1")
        assert words["frequency_filter"] == {"type": "df", "pct": 1.0}
        lemmas = collocation_in(root, q="lemma:house", colloc_filter_choice="frequency", filter_df_pct="1")
        assert lemmas["frequency_filter"] == {"type": "top_n", "n": 100}
        assert all(w.startswith("lemma:") for w in lemmas["filter_list"])
        extra = set(lemmas["filter_list"]) - set(self.top_n(eltec_db, 100, True))  # the query's own words
        assert set(self.top_n(eltec_db, 100, True)) <= set(lemmas["filter_list"]) and all("house" in w for w in extra)

    def test_the_cache_follows_the_filter_applied(self, shakespeare_db, tmp_path):
        """Once the counts are written, a request the top n was the answer to is answered with the df filter, and its
        results are not those cached before."""
        from philologic.loadtime.PostFilters import write_corpus_counts

        root = database_copy(shakespeare_db, tmp_path, without=self.COUNTS)
        params = {"q": "lord", "colloc_filter_choice": "frequency", "filter_df_pct": "1", "filter_frequency": "100"}
        before = collocation_in(root, **params)
        assert before["frequency_filter"]["type"] == "top_n"
        legacy = collocation_in(root, **{k: v for k, v in params.items() if k != "filter_df_pct"})
        assert legacy["file_path"] == before["file_path"]  # the same filter, the same cache
        write_corpus_counts(str(root / "data" / "collocations"))
        after = collocation_in(root, **params)
        assert after["frequency_filter"]["type"] == "df"
        assert after["file_path"] != before["file_path"]
        again = collocation_in(root, **{**params, "filter_df_pct": "1.0"})
        assert again["file_path"] == after["file_path"]
        assert collocation_in(root, **{**params, "filter_df_pct": "2"})["file_path"] != after["file_path"]
        assert collocation_in(root, **{**params, "filter_df_pct": "0.2"})["file_path"] == collocation_in(
            root, **{**params, "filter_df_pct": "0.5"}
        )["file_path"]
        assert collocation_in(root, **{k: v for k, v in params.items() if k != "filter_df_pct"})["file_path"] == before["file_path"]
