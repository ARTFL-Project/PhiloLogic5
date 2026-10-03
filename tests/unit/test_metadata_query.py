"""Unit tests for metadata query syntax (MetadataQuery.make_grouped_sql_clause, QuerySyntax): ranges, OR, NOT and NULL,
run on a small toms table."""

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.exceptions import BadRequest
from philologic.runtime.MetadataQuery import make_grouped_sql_clause
from philologic.runtime.QuerySyntax import group_terms, parse_query

YEARS = [1650, 1700, 1720, 1750, 1789, 1800, 1850, 1900, None]


@pytest.fixture(scope="module")
def db():
    dbh = sqlite3.connect(":memory:")
    dbh.execute("CREATE TABLE toms (year int)")
    dbh.executemany("INSERT INTO toms VALUES (?)", [(y,) for y in YEARS])
    return SimpleNamespace(dbh=dbh)


def years(db, value):
    """The years a metadata value selects (its terms are values as they are, as expand_grouped_query makes them)."""
    grouped = group_terms(parse_query(value))
    expanded = []
    for group in grouped:  # as expand_grouped_query does: NOT starts a group, OR is dropped, terms become values
        current = []
        for kind, token in group:
            if kind == "NOT":
                if current:
                    expanded.append(current)
                current = [(kind, token)]
            elif kind == "TERM":
                current.append(("QUOTE", f'"{token}"'))
            elif kind != "OR":
                current.append((kind, token))
        expanded.append(current)
    clause = make_grouped_sql_clause(expanded, "year", db)
    return sorted((y for (y,) in db.dbh.execute(f"SELECT year FROM toms WHERE {clause}")), key=lambda y: (y is not None, y))


@pytest.mark.unit
class TestRanges:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("1700-1750", [1700, 1720, 1750]),
            ("-1700", [1650, 1700]),
            ("1850-", [1850, 1900]),
            ("NOT 1700-1800", [1650, 1850, 1900]),  # NOT, as in SQL, leaves out what has no value
            ("1700-1800 NOT 1720-1750", [1700, 1789, 1800]),
            ("1700-1720 | 1850-1900", [1700, 1720, 1850, 1900]),  # was AND, so nothing
            ("1650 | 1800-1850", [1650, 1800, 1850]),
            ("1789 | 1900", [1789, 1900]),
            ("NOT 1789", [1650, 1700, 1720, 1750, 1800, 1850, 1900]),
            ("NULL", [None]),
            ("NOT NULL", [1650, 1700, 1720, 1750, 1789, 1800, 1850, 1900]),
            ("1650 | NULL", [None, 1650]),
        ],
    )
    def test_years(self, db, value, expected):
        assert years(db, value) == sorted(expected, key=lambda y: (y is not None, y))

    @pytest.mark.parametrize("value", ["1789-07-14", "17-89-1"])
    def test_not_a_range(self, db, value):
        with pytest.raises(BadRequest, match="a range is two values"):
            years(db, value)

    def test_negative_years(self, db):
        db.dbh.execute("INSERT INTO toms VALUES (-450)")
        try:
            assert years(db, "-500--400") == [-450]
        finally:
            db.dbh.execute("DELETE FROM toms WHERE year = -450")


@pytest.mark.unit
class TestSyntax:
    @pytest.mark.parametrize(
        "query, tokens",
        [
            ("1700-1750", [("RANGE", "1700-1750")]),
            ("a-f", [("RANGE", "a-f")]),
            ("17[0-4].", [("TERM", "17[0-4].")]),  # a regex: the hyphen of a bracket expression makes no range
            ("[a-z]tat", [("TERM", "[a-z]tat")]),
            ("lord[", [("TERM", "lord[")]),  # no regex, a word (which matches nothing)
        ],
    )
    def test_tokens(self, query, tokens):
        assert parse_query(query) == tokens

    def test_ranges_in_or(self):
        assert group_terms(parse_query("1700-1750 | 1800-1850")) == [
            [("RANGE", "1700-1750"), ("OR", "|"), ("RANGE", "1800-1850")]
        ]

    def test_ranges_and(self):
        assert group_terms(parse_query("1700-1750 1800-1850")) == [[("RANGE", "1700-1750")], [("RANGE", "1800-1850")]]
