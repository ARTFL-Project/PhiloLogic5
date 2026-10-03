"""Integration tests, through the web app, of requests with bad or missing parameters: a 400 or a 404 that says why,
where they gave a 500."""

from pathlib import Path

import pytest

Q = {"q": "love"}


@pytest.fixture(scope="module")
def web(eltec_db_path):
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    corpus = Path(eltec_db_path).parent
    with web_app_client(corpus.parent) as client:
        yield lambda path, **params: client.simulate_get(f"/philologic5/{corpus.name}/{path}", params=params)


@pytest.mark.integration
class TestBadParameters:
    @pytest.mark.parametrize(
        "path, params, status, reason",
        [
            ("reports/concordance.py", {**Q, "start": "abc"}, 400, "start must be a whole number"),
            ("reports/concordance.py", {**Q, "start": "1", "end": "1.5"}, 400, "end must be a whole number"),
            ("reports/kwic.py", {**Q, "results_per_page": "abc"}, 400, "results_per_page must be a whole number"),
            ("reports/concordance.py", {**Q, "start": str(2**64)}, 400, "start is too large"),
            ("reports/concordance.py", {**Q, "approximate": "yes", "approximate_ratio": "abc"}, 400, "approximate_ratio"),
            ("reports/bibliography.py", {"start": "abc"}, 400, "start must be a whole number"),
            ("scripts/get_total_results.py", {**Q, "start": "abc"}, 400, "start must be a whole number"),
            ("reports/concordance.py", {**Q, "sort_by": "philo_seq"}, 400, "can't be sorted by 'philo_seq'"),
            ("reports/bibliography.py", {"sort_by": "philo_id"}, 400, "can't be sorted by 'philo_id'"),
            ("reports/bibliography.py", {"sort_by": "foo"}, 400, "can't be sorted by 'foo'"),
            ("scripts/get_frequency.py", {**Q, "frequency_field": "philo_seq"}, 400, "no metadata field"),
            ("scripts/get_more_context.py", {**Q, "hit_num": "abc"}, 400, "hit_num must be a whole number"),
            ("scripts/get_more_context.py", {**Q, "hit_num": "-5"}, 400, "hit_num must be the number of a hit"),
            ("scripts/get_more_context.py", Q, 400, "hit_num must be the number of a hit"),
            ("scripts/get_more_context.py", {**Q, "hit_num": "99999999"}, 404, "No hit 99999999"),
            ("scripts/alignment_to_text.py", {}, 400, "A filename and a start_byte are required"),
            ("scripts/get_header.py", {}, 400, "A philo_id is required"),
            ("scripts/get_header.py", {"philo_id": "999999"}, 404, "No text object"),
            ("scripts/get_sorted_kwic.py", {"first_kwic_sorting_option": "title"}, 400, "A search term (q) is required"),
            (
                "scripts/get_landing_page_content.py",
                {"group_by_field": "title", "is_range": "true", "query": ""},
                400,
                "no range of initials",
            ),
            ("scripts/get_word_property_count.py", {**Q, "word_property": "nothing"}, 400, "no word property"),
            ("reports/navigation.py", {"philo_id": "1 2 0 0 0 0 0 0 999999"}, 404, "No page"),
        ],
    )
    def test_refused(self, web, path, params, status, reason):
        response = web(path, **params)
        assert response.status_code == status, response.text[:300]
        assert reason in response.json["description"]

    @pytest.mark.parametrize("term", ["", "  "])
    def test_blank_autocomplete(self, web, term):
        assert web("scripts/autocomplete_term.py", term=term).json == []
        assert web("scripts/autocomplete_metadata.py", term=term, field="author").json == []

    def test_empty_values_are_defaults(self, web):
        """An empty results_per_page, start or end is the default, as the client's links may leave them."""
        response = web("reports/concordance.py", **Q, results_per_page="", start="", end="")
        assert response.status_code == 200
        assert response.json["description"] == {"start": 1, "end": 25, "results_per_page": 25}

    def test_resolve_cite_without_abbreviations(self, web):
        """No abbrev column in toms: the database's home page, as when no citation matches (it was a 500)."""
        response = web("scripts/resolve_cite.py", q="Gen. 1.1")
        assert response.status_code == 302
        assert response.headers["location"].endswith(f"/{Path(response.headers['location']).name}/")
