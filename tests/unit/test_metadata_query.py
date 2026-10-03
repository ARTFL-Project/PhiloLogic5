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
from philologic.runtime.QuerySyntax import group_terms, parse_metadata_query, quote_metadata_value, quoted_text

YEARS = [1650, 1700, 1720, 1750, 1789, 1800, 1850, 1900, None]


@pytest.fixture(scope="module")
def db():
    dbh = sqlite3.connect(":memory:")
    dbh.execute("CREATE TABLE toms (year int)")
    dbh.executemany("INSERT INTO toms VALUES (?)", [(y,) for y in YEARS])
    return SimpleNamespace(dbh=dbh)


def years(db, value):
    """The years a metadata value selects (its terms are values as they are, as expand_grouped_query makes them)."""
    grouped = group_terms(parse_metadata_query(value, "int"))
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
            ("NOT 1700-1800", [1650, 1850, 1900, None]),  # everything 1700-1800 doesn't select: SQL's NOT left out None
            ("1700-1800 NOT 1720-1750", [1700, 1789, 1800]),
            ("1700-1720 | 1850-1900", [1700, 1720, 1850, 1900]),  # was AND, so nothing
            ("1650 | 1800-1850", [1650, 1800, 1850]),
            ("1789 | 1900", [1789, 1900]),
            ("NOT 1789", [1650, 1700, 1720, 1750, 1800, 1850, 1900, None]),
            ("NULL", [None]),
            ("NOT NULL", [1650, 1700, 1720, 1750, 1789, 1800, 1850, 1900]),
            ("NOT 1789 NOT NULL", [1650, 1700, 1720, 1750, 1800, 1850, 1900]),  # what SQL's NOT gave
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
    """Metadata values have their own grammar (QuerySyntax.parse_metadata_query), not the word-search rules."""

    @pytest.mark.parametrize(
        "value, field_type, tokens",
        [
            ("1700-1750", "int", [("RANGE", "1700-1750")]),
            ("17[0-4].", "int", [("TERM", "17[0-4].")]),  # a regex: the hyphen of a bracket expression makes no range
            ("a-f", "text", [("TERM", "a-f")]),  # no string ranges: in text fields "-" is part of a word
            ("jean-jacques rousseau", "text", [("TERM", "jean-jacques"), ("TERM", "rousseau")]),
            ("d'autriche", "text", [("TERM", "d'autriche")]),  # the word rules made it d autriche
            ("rousseau NOT jean-baptiste", "text", [("TERM", "rousseau"), ("NOT", "NOT"), ("TERM", "jean-baptiste")]),
            ('zola | "Hugo, Victor, 1802-1885."', "text", [("TERM", "zola"), ("OR", "|"), ("QUOTE", '"Hugo, Victor, 1802-1885."')]),
            ("hugo OR zola", "text", [("TERM", "hugo"), ("OR", "|"), ("TERM", "zola")]),
            ("NOT NULL", "text", [("NOT", "NOT"), ("NULL", "NULL")]),
            ("\u201cHugo, Victor\u201d", "text", [("QUOTE", '"Hugo, Victor"')]),  # typographic quotes
            ("Hugo\uff5cZola\u3000\uff36\uff49\uff43", "text", [("TERM", "Hugo"), ("OR", "|"), ("TERM", "Zola"), ("TERM", "Vic")]),
            ('"Hugo, Vi', "text", [("QUOTE", '"Hugo, Vi"')]),  # unclosed: the rest
        ],
    )
    def test_tokens(self, value, field_type, tokens):
        assert parse_metadata_query(value, field_type) == tokens

    def test_ranges_in_or(self):
        assert group_terms(parse_metadata_query("1700-1750 | 1800-1850", "int")) == [
            [("RANGE", "1700-1750"), ("OR", "|"), ("RANGE", "1800-1850")]
        ]

    def test_ranges_and(self):
        assert group_terms(parse_metadata_query("1700-1750 1800-1850", "int")) == [
            [("RANGE", "1700-1750")],
            [("RANGE", "1800-1850")],
        ]

    @pytest.mark.parametrize("value", ['Les Révoltés de la "Bounty"', '"Bounty"', "a | b", ""])
    def test_quoted_values(self, value):
        """A quote inside a quoted value is doubled: values with quotes could not be matched."""
        tokens = parse_metadata_query(quote_metadata_value(value))
        assert [kind for kind, _ in tokens] == ["QUOTE"] and quoted_text(tokens[0][1]) == value


@pytest.fixture(scope="module")
def titles():
    dbh = sqlite3.connect(":memory:")
    dbh.execute("CREATE TABLE toms (title text)")
    dbh.executemany("INSERT INTO toms VALUES (?)", [('Les Révoltés de la "Bounty"',), ("Le Bounty",), (None,)])
    return SimpleNamespace(dbh=dbh)


@pytest.mark.unit
class TestQuotedValues:
    def test_quote_inside(self, titles):
        expanded = [[token] for token in parse_metadata_query('"Les Révoltés de la ""Bounty"""')]
        clause = make_grouped_sql_clause(expanded, "title", titles)
        assert [t for (t,) in titles.dbh.execute(f"SELECT title FROM toms WHERE {clause}")] == ['Les Révoltés de la "Bounty"']
