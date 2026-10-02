"""Integration tests, through the web app, of the KWIC sorted by metadata and of the export of results."""

import json
import re
from pathlib import Path

import pytest

from philologic.runtime.HitList import sort_key

QUERY = {"q": "love", "method": "proxy", "method_arg": "", "results_per_page": "25"}


@pytest.fixture(scope="module")
def web(eltec_db_path):
    pytest.importorskip("falcon")
    from tests.fixtures.web_app import web_app_client

    corpus = Path(eltec_db_path).parent
    with web_app_client(corpus.parent) as client:
        yield lambda path, **params: client.simulate_get(f"/philologic5/{corpus.name}/{path}", params=params)


def last_json(response):
    assert response.status_code == 200, response.text[:300]
    return json.loads(response.text.splitlines()[-1])  # streamed responses end with the result, after progress lines


@pytest.mark.integration
class TestSortedKwic:
    def test_title_order_is_the_concordance_order(self, web, eltec_db):
        """Sorted by title, the KWIC lists its hits in the order the concordance sorts titles."""
        hits = eltec_db.query("love", "single_term", "0")  # what the web app resolves proxy to for one word
        hits.finish()
        params = {**QUERY, "first_kwic_sorting_option": "title", "start": "1", "end": str(len(hits))}
        titles = [r["metadata_fields"]["title"] for r in last_json(web("scripts/get_sorted_kwic.py", **params))["results"]]
        assert len(titles) == len(hits) and len(set(titles)) > 1
        assert titles == sorted(titles, key=lambda title: sort_key(title, eltec_db.locals.ascii_conversion))


@pytest.mark.integration
class TestExport:
    @pytest.mark.parametrize("report", ["concordance", "kwic"])
    def test_plain_text_json(self, web, report):
        """filter_html leaves the contexts of the JSON export without tags, as those of the CSV."""
        html = web("scripts/export_results.py", **QUERY, report=report, output_format="json", filter_html="false").json
        plain = web("scripts/export_results.py", **QUERY, report=report, output_format="json", filter_html="true").json
        assert len(plain) == len(html) == 25
        assert all("<" in result["context"] for result in html)
        assert all(not re.search(r"<[^>]+>", result["context"]) for result in plain)
        assert [r["philo_id"] for r in plain] == [r["philo_id"] for r in html]

    def test_time_series_links_to_concordance(self, web):
        results = web(
            "scripts/export_results.py", q="love", report="time_series", output_format="json",
            start_date="1840", end_date="1860", year_interval="10",
        ).json
        urls = [period["url"] for period in results["absolute_count"].values()]
        assert urls and all(url.startswith("/concordance?") and "year=" in url for url in urls)
