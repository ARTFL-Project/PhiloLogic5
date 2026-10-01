"""Unit tests for the metadata fields in the web config of a new database: the loader leaves out those the database
doesn't have, and the web config writer cites what is left as citations["name"]"""

import sqlite3
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.Config import NEW_WEB_CONFIG_CITATIONS, WEB_CONFIG_DEFAULTS, Config
from philologic.loadtime.Loader import Loader

pytestmark = pytest.mark.unit


def written(values):
    """The values of a web config written with values, as read back"""
    config = Config("", WEB_CONFIG_DEFAULTS)
    for key, value in values.items():
        config[key] = value
    source = str(config)
    read = {}
    exec(source, {}, read)
    return source, read


def names(citations):
    return [citation["field"] + "@" + citation["object_level"] for citation in citations]


@pytest.fixture
def cited_fields(tmp_path, monkeypatch):
    """cited_fields of a database with the given fields (in toms) and page fields (in pages, if any)"""

    def make(fields, page_fields=None):
        conn = sqlite3.connect(tmp_path / "toms.db")
        conn.execute(f"create table toms (philo_type, philo_id, {', '.join(fields)})")
        if page_fields is not None:
            conn.execute("create table pages (philo_type, philo_id, n, facs)")
            conn.execute("insert into pages values ('page', '1 0 0 0 0 0 0 0 1', ?, ?)", page_fields)
        # Fields which the parser looks for but the database doesn't have
        monkeypatch.setattr(Loader, "metadata_fields", [*fields, "pub_place", "speaker", "div_date"])
        monkeypatch.setattr(Loader, "metadata_fields_not_found", ["pub_place", "speaker", "div_date"])
        return Loader.cited_fields(conn.cursor())

    return make


def test_new_web_config_cites_as_before():
    source, read = written(NEW_WEB_CONFIG_CITATIONS)
    assert 'concordance_citation = [citations["author"], citations["title"], citations["year"]' in source
    assert all(read[key] == value for key, value in NEW_WEB_CONFIG_CITATIONS.items())


def test_fields_the_database_lacks_are_left_out(cited_fields):
    values = cited_fields(["filename", "author", "title", "year", "head"], page_fields=("12", ""))
    source, read = written(values)
    assert set(read["citations"]) == {"author", "title", "year", "div1_head", "div2_head", "div3_head", "page"}
    assert names(read["concordance_citation"]) == [
        "author@doc",
        "title@doc",
        "year@doc",
        "head@div1",
        "head@div2",
        "head@div3",
        "n@page",
    ]
    assert names(read["navigation_citation"]) == ["author@doc", "title@doc", "year@doc"]
    assert names(read["aggregation_config"][0]["break_up_field_citation"]) == ["title@doc", "year@doc"]
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    assert "pub_place" not in code and "speaker" not in code and "div_date" not in code
    assert "time_series" in read["search_reports"]


def test_without_pages_head_author_or_year(cited_fields):
    values = cited_fields(["filename", "title", "publisher"])
    _, read = written(values)
    # div1s are still cited: by their type and n, or as sections
    assert names(read["concordance_citation"]) == ["title@doc", "head@div1"]
    assert [entry["field"] for entry in read["aggregation_config"]] == ["title"]
    assert [entry["group_by_field"] for entry in read["default_landing_page_browsing"]] == ["title"]
    assert read["results_summary"] == [{"field": "title", "object_level": "doc"}]
    assert read["collocation_fields_to_compare"] == ["title"]
    # No time series without years
    assert "time_series" not in read["search_reports"] and read["time_series_year_field"] == ""


def test_aggregation_falls_back_to_filename(cited_fields):
    _, read = written(cited_fields(["filename", "year"]))
    # The web app needs an aggregation field
    assert [entry["field"] for entry in read["aggregation_config"]] == ["filename"]
    assert read["default_landing_page_browsing"] == []
