"""Unit tests for the landing page's browsing by initial (landing_page.group_by_range), on a small toms table."""

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.reports.landing_page import group_by_range

DOCS = [  # title, author, year
    ("Zoloé", "Sade", 1800),
    ("Élégies", "Parny", 1784),
    ("[L']Impasse", "Anon", 1900),
    ("“Ma vie”", "Anon", 1901),
    ("Oeuvres poétiques", "Labé", 1555),
    ("Oeuvres poétiques", "Ronsard", 1560),
    ("Bible", None, 1910),
    ("Bible", None, 1910),
    ("ballades", "Villon", 1489),
    ("Odes", "Ronsard", 1550),
    ("500 millions de la Bégum", "Verne", 1879),
]

CITATIONS = [
    {"field": "author", "link": False, "prefix": "", "suffix": ", ", "style": {}},
    {"field": "title", "link": True, "prefix": "", "suffix": "", "style": {}},
    {"field": "year", "link": False, "prefix": " [", "suffix": "]", "style": {}},
]


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("landing")
    (db_path / "data").mkdir()
    (db_path / "data" / "db.locals.py").write_text('metadata_fields = ["author", "title", "year"]\n')
    dbh = sqlite3.connect(db_path / "data" / "toms.db")
    dbh.execute("CREATE TABLE toms (philo_type, philo_id, title, author, year)")
    dbh.executemany(
        "INSERT INTO toms VALUES ('doc', ?, ?, ?, ?)", [(f"{i} 0 0 0 0 0 0", *doc) for i, doc in enumerate(DOCS, 1)]
    )
    dbh.commit()
    browsing = [
        {"group_by_field": "title", "citation": CITATIONS},
        {"group_by_field": "author", "citation": CITATIONS[:1]},
    ]
    return SimpleNamespace(db_path=str(db_path), default_landing_page_browsing=browsing)


def browse(config, field, letters):
    request = SimpleNamespace(group_by_field=field, display_count="false")
    content = group_by_range(list(letters.lower()), request, config)["content"]
    return {initial: [(r["metadata"][field], r["count"]) for r in group["results"]] for initial, group in content.items()}


@pytest.mark.unit
class TestGroupByRange:
    def test_titles(self, config):
        assert browse(config, "title", "az") == {
            "B": [("ballades", 1), ("Bible", 2)],  # the two volumes show alike
            "E": [("Élégies", 1)],
            "L": [("[L']Impasse", 1)],
            "M": [("“Ma vie”", 1)],
            "O": [("Odes", 1), ("Oeuvres poétiques", 1), ("Oeuvres poétiques", 1)],  # one each for Labé and Ronsard
            "Z": [("Zoloé", 1)],
        }

    def test_range(self, config):
        assert list(browse(config, "title", "ei")) == ["E"]

    def test_digits(self, config):
        """Single characters are initials: "0-9" were the years 0 to 9, so titles starting with a digit were in no tab."""
        assert browse(config, "title", "09") == {"5": [("500 millions de la Bégum", 1)]}

    def test_years(self, config):
        request = SimpleNamespace(group_by_field="year", display_count="false")
        result = group_by_range(["1550", "1560"], request, config)
        assert result["content_type"] == "date" and sorted(result["content"]) == ["1550", "1555", "1560"]

    def test_authors_counted(self, config):
        """Browsing by author, an author's documents still make one entry, with their count."""
        assert browse(config, "author", "pr") == {"P": [("Parny", 1)], "R": [("Ronsard", 2)]}
