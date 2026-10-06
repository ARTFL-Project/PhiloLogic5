"""Integration tests, through the web app, of the metadata grammar (docs/query_syntax.md): values are read on their
own terms, not by the word-search rules, which made "-" and "'" spaces."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def web(eltec_db_path):
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    corpus = Path(eltec_db_path).parent
    with web_app_client(corpus.parent) as client:
        yield lambda path, **params: client.simulate_get(f"/philologic5/{corpus.name}/{path}", params=params)


def docs(web, **metadata):
    response = web("reports/bibliography.py", results_per_page="1", **metadata)
    assert response.status_code == 200, response.text[:300]
    return response.json["results_length"]


def sql_count(db, where, params=()):
    return db.dbh.execute(f"SELECT COUNT(*) FROM toms WHERE philo_type = 'doc' AND {where}", params).fetchone()[0]


@pytest.mark.integration
class TestWords:
    def test_hyphenated_word(self, web):
        """Its parts side by side, in that order: as two words, farine-folle found Folle-Farine too."""
        assert docs(web, title="folle-farine") == 1
        assert docs(web, title="farine-folle") == 0

    @pytest.mark.parametrize("title", ["audley's", "audley’s"])
    def test_apostrophe(self, web, title):
        assert docs(web, title=title) == 1

    def test_not_hyphenated_word(self, web, eltec_db):
        """NOT applies to the whole word: the rules left NOT folle, then farine."""
        assert docs(web, title="NOT folle-farine") == sql_count(eltec_db, "1") - 1

    def test_regex_matches_whole_words(self, web, eltec_db):
        """As in word search: ick.* found Dickens, as the words containing ick."""
        assert docs(web, author="dick.*") == sql_count(eltec_db, "author LIKE 'Dickens%'") > 0
        assert docs(web, author="ick.*") == 0


@pytest.mark.integration
class TestValues:
    def test_quoted_value(self, web, eltec_db):
        author = "Dickens, Charles (1812-1870)"
        assert docs(web, author=f'"{author}"') == sql_count(eltec_db, "author = ?", (author,)) > 0

    def test_year_range(self, web, eltec_db):
        assert docs(web, year="1840-1850") == sql_count(eltec_db, "year BETWEEN 1840 AND 1850") > 0


@pytest.mark.integration
class TestNot:
    def test_not_keeps_objects_with_no_value(self, eltec_db):
        """NOT x is everything x doesn't select, divisions without a head included: SQL's NOT left them out."""

        def divs(head):
            hits = eltec_db.query(head=head)
            hits.finish()
            return len(hits)

        assert divs("chapter") + divs("NOT chapter") == divs("NULL") + divs("NOT NULL")
        assert divs("NOT chapter NOT NULL") == divs("NOT NULL") - divs("chapter")


@pytest.mark.integration
class TestPhiloType:
    """philo_type is no metadata field, and is ignored: alone, it made a metadata query with nothing to query (a 500)."""

    def test_search(self, web):
        response = web("scripts/get_total_results.py", q="love", philo_type="doc")
        assert response.status_code == 200, response.text[:300]
        assert response.json == web("scripts/get_total_results.py", q="love").json > 0

    def test_concordance(self, web):
        assert web("reports/concordance.py", q="love", philo_type="doc").status_code == 200

    def test_with_a_field(self, web):
        assert docs(web, title="folle-farine", philo_type="div1") == docs(web, title="folle-farine") == 1

    def test_query(self, eltec_db):
        hits = eltec_db.query(philo_type="doc")
        hits.finish()
        assert len(hits) == sql_count(eltec_db, "1")


@pytest.mark.integration
class TestNoValue:
    """A blank value, or one of operators alone, is no value, as "" is: it was a 500 (an SQL syntax error). The web
    client sends a value of spaces as it is."""

    @pytest.mark.parametrize("value", [" ", "  ", "|", "OR", " | "])
    def test_bibliography(self, web, value):
        assert docs(web, author=value) == docs(web)

    @pytest.mark.parametrize("value", [" ", "|"])
    def test_search(self, web, value):
        response = web("scripts/get_total_results.py", q="love", author=value)
        assert response.status_code == 200, response.text[:300]
        assert response.json == web("scripts/get_total_results.py", q="love").json

    def test_with_a_value(self, web):
        assert docs(web, author=" ", title="folle-farine") == docs(web, title="folle-farine") == 1


@pytest.mark.integration
class TestAutocomplete:
    def test_hyphenated_word(self, web):
        suggestions = web("scripts/autocomplete_metadata.py", term="folle-f", field="title").json
        assert any("Folle" in s for s in suggestions)

    @pytest.mark.parametrize("before", ["dickens |", '"Dickens, Charles (1812-1870)" |'])
    def test_rest_of_the_value_kept(self, web, before):
        suggestions = web("scripts/autocomplete_metadata.py", term=f"{before} trol", field="author").json
        assert suggestions and all(s.startswith(f"{before} CUTHERE ") for s in suggestions)
