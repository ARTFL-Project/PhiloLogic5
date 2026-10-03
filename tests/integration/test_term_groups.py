"""Integration tests, through the web app, of get_term_groups: the query's groups, and the terms whose expansion the
cap on regexes with no literal start cut, which the results summary says."""

from pathlib import Path

import pytest

from philologic.runtime.term_expansion import REGEX_EXPANSION_CAP


@pytest.fixture(scope="module")
def term_groups(eltec_db_path):
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    corpus = Path(eltec_db_path).parent
    with web_app_client(corpus.parent) as client:
        yield lambda q, **params: client.simulate_get(
            f"/philologic5/{corpus.name}/scripts/get_term_groups.py", params={"q": q, **params}
        ).json


@pytest.mark.integration
class TestCutTerms:
    def test_cut(self, term_groups):
        result = term_groups(".*")
        assert result["cut_terms"] == [{"term": ".*", "not": False}]
        assert result["expansion_cap"] == REGEX_EXPANSION_CAP

    def test_cut_after_not(self, term_groups):
        assert term_groups("lov.* NOT .*")["cut_terms"] == [{"term": ".*", "not": True}]

    @pytest.mark.parametrize("q", ["love", "lov.*", ".*xyzq"])
    def test_not_cut(self, term_groups, q):
        assert term_groups(q)["cut_terms"] == []

    def test_cut_terms_lose_hits(self, eltec_db):
        """What the note says: uncut, .* would find every word of the corpus."""
        hits = eltec_db.query(".*", "single_term", "0")
        hits.finish()
        (words,) = eltec_db.dbh.execute("SELECT SUM(word_count) FROM toms WHERE philo_type = 'doc'").fetchone()
        assert 0 < len(hits) < int(words)


@pytest.mark.integration
class TestApproximateGroups:
    def test_folded(self, term_groups):
        """Each group of an approximate search is the similar words of a term typed, which the summary folds."""
        result = term_groups("love death", approximate="yes", approximate_ratio="80")
        assert [group["term"] for group in result["approximate_groups"]] == ["love", "death"]
        for group, words in zip(result["approximate_groups"], result["term_groups"]):
            assert group["variants"] == len(words.split("|")) > 1

    def test_not_approximate(self, term_groups):
        assert term_groups("love death")["approximate_groups"] == []
