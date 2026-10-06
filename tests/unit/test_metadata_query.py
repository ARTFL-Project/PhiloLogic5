"""Unit tests for metadata query syntax (MetadataQuery.groups_clause, QuerySyntax): ranges, OR, NOT and NULL,
run on a small toms table; and for the values of div fields bulk_load_metadata finds for hits."""

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import MetadataQuery
from philologic.runtime.exceptions import BadRequest
from philologic.runtime.MetadataQuery import bulk_load_metadata, groups_clause, level_query, object_ids
from philologic.runtime.QuerySyntax import parse_metadata_query, quote_metadata_value, quoted_text, value_groups

YEARS = [1650, 1700, 1720, 1750, 1789, 1800, 1850, 1900, None]


@pytest.fixture(scope="module")
def db():
    dbh = sqlite3.connect(":memory:")
    dbh.execute("CREATE TABLE toms (year int)")
    dbh.executemany("INSERT INTO toms VALUES (?)", [(y,) for y in YEARS])
    return SimpleNamespace(dbh=dbh, path="/nonexistent/")


def years(db, value):
    """The years a metadata value selects (its terms the values they are, as the index of the years has them)."""
    with patch.object(MetadataQuery, "metadata_pattern_search", lambda term, *args: [term]):
        clause = groups_clause(db, "year", value_groups(value, "int"))
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
        assert value_groups("1700-1750 | 1800-1850", "int") == [
            (False, [("RANGE", "1700-1750"), ("RANGE", "1800-1850")])
        ]

    def test_ranges_and(self):
        assert value_groups("1700-1750 1800-1850", "int") == [
            (False, [("RANGE", "1700-1750")]),
            (False, [("RANGE", "1800-1850")]),
        ]

    @pytest.mark.parametrize(
        "value, groups",
        [
            ("hugo zola", [(False, ["hugo"]), (False, ["zola"])]),
            ("hugo | zola", [(False, ["hugo", "zola"])]),
            ("hugo NOT zola", [(False, ["hugo"]), (True, ["zola"])]),
            ("NOT hugo | zola", [(True, ["hugo", "zola"])]),
            ("hugo OR NOT zola", [(False, ["hugo"]), (True, ["zola"])]),  # NOT starts a group of its own
            ("NOT hugo zola", [(True, ["hugo"]), (False, ["zola"])]),
            ("| hugo", [(False, ["hugo"])]),
            ("hugo NOT", [(False, ["hugo"]), (True, [])]),  # a NOT of nothing, which selects everything
        ],
    )
    def test_groups(self, value, groups):
        assert [(negated, [t for _, t in tokens]) for negated, tokens in value_groups(value)] == groups

    def test_dates_join(self):
        """In date fields, dates join the group before them, as alternatives (only after a line break does the date
        grammar read a second one)."""
        assert value_groups("1789<=>1790\n1800", "date") == [
            (False, [("DATE_RANGE", "1789-01-01<=>1790-12-31"), ("DATE_RANGE", "1800-01-01<=>1800-12-31")])
        ]
        assert value_groups("NOT 1789", "date") == [(True, [("DATE_RANGE", "1789-01-01<=>1789-12-31")])]

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
        clause = groups_clause(titles, "title", value_groups('"Les Révoltés de la ""Bounty"""'))
        assert [t for (t,) in titles.dbh.execute(f"SELECT title FROM toms WHERE {clause}")] == ['Les Révoltés de la "Bounty"']


@pytest.fixture(scope="module")
def articles():
    """Divs as the Encyclopédie has them: an article (div1) with its head and author, whose words are in an implicit
    div2 and div3 without them; an article with a part (div2) of its own head; an article without an author."""
    dbh = sqlite3.connect(":memory:")
    dbh.execute("CREATE TABLE toms (philo_type text, philo_id text, head text, author text)")
    dbh.executemany(
        "INSERT INTO toms VALUES (?, ?, ?, ?)",
        [
            ("doc", "1 0 0 0 0 0 0", None, None),
            ("div1", "1 1 0 0 0 0 0", "ABEILLE", "Diderot"),
            ("div2", "1 1 1 0 0 0 0", None, None),
            ("div3", "1 1 1 1 0 0 0", None, None),
            ("div1", "1 2 0 0 0 0 0", "ABRI", "Jaucourt"),
            ("div2", "1 2 1 0 0 0 0", "Abri, en Marine", None),
            ("div3", "1 2 1 1 0 0 0", "", None),
            ("div1", "1 3 0 0 0 0 0", "ABSENCE", None),
            ("div2", "1 3 1 0 0 0 0", None, None),
            ("div3", "1 3 1 1 0 0 0", None, None),
        ],
    )
    return SimpleNamespace(dbh=dbh, locals=SimpleNamespace(metadata_types={"head": "div", "author": "div"}))


@pytest.mark.unit
class TestDivValues:
    def test_inherited(self, articles):
        """A div with no value of a div field has that of its div2, else of its div1, as a hit's citation shows it."""
        caches = bulk_load_metadata(articles, ["head", "author"], inherit=True)
        assert caches["head"] == (4, {
            (1, 1, 0, 0): "ABEILLE", (1, 1, 1, 0): "ABEILLE", (1, 1, 1, 1): "ABEILLE",
            (1, 2, 0, 0): "ABRI", (1, 2, 1, 0): "Abri, en Marine", (1, 2, 1, 1): "Abri, en Marine",
            (1, 3, 0, 0): "ABSENCE", (1, 3, 1, 0): "ABSENCE", (1, 3, 1, 1): "ABSENCE",
        })
        authors = caches["author"][1]
        assert authors[1, 1, 1, 1] == "Diderot" and authors[1, 2, 1, 1] == "Jaucourt" and authors[1, 3, 1, 1] == ""

    def test_own(self, articles):
        """Without inherit, each div has its own value: the frequency report sums each div's words once."""
        heads = bulk_load_metadata(articles, ["head"])["head"][1]
        assert heads[1, 1, 1, 1] == "" and heads[1, 2, 1, 0] == "Abri, en Marine"


class Locals(dict):
    """db.locals, whose entries are attributes too"""

    __getattr__ = dict.__getitem__


@pytest.fixture(scope="module")
def plays():
    """Documents by two authors, with acts (div1) and scenes (div2) whose heads repeat, and speeches (para)."""
    dbh = sqlite3.connect(":memory:")
    dbh.execute("CREATE TABLE toms (philo_type text, philo_id text, author text, head text, who text)")
    dbh.executemany(
        "INSERT INTO toms VALUES (?, ?, ?, ?, ?)",
        [
            ("doc", "1 0 0 0 0 0 0", "Racine", None, None),
            ("div1", "1 1 0 0 0 0 0", None, "Acte", None),
            ("div2", "1 1 1 0 0 0 0", None, "Scène", None),
            ("para", "1 1 1 1 1 0 0", None, None, "Phèdre"),
            ("div2", "1 1 2 0 0 0 0", None, "Acte", None),
            ("para", "1 1 2 1 1 0 0", None, None, "Phèdre"),
            ("div2", "1 1 3 0 0 0 0", None, "Scène", None),
            ("para", "1 1 3 1 1 0 0", None, None, "Phèdre"),
            ("div1", "1 2 0 0 0 0 0", None, "Prologue", None),
            ("para", "1 2 1 1 1 0 0", None, None, "Phèdre"),
            ("doc", "2 0 0 0 0 0 0", "Corneille", None, None),
            ("div1", "2 1 0 0 0 0 0", None, "Acte", None),
            ("para", "2 1 1 1 1 0 0", None, None, "Phèdre"),
        ],
    )
    return SimpleNamespace(
        dbh=dbh,
        path="/nonexistent/",
        locals=Locals(
            debug=False,
            metadata_fields=["author", "head", "who"],
            word_attributes=[],
            metadata_sql_types={},
        ),
    )


DOCS, DIVS, PARAS = ("doc",), ("div", "div1", "div2", "div3"), ("para",)


@pytest.mark.unit
class TestObjectIds:
    """The objects of the last level of a metadata query, within those of the levels before it."""

    def ids(self, db, *levels):
        return [" ".join(map(str, philo_id[:5])) for philo_id in object_ids(db, list(levels))]

    def test_one_level(self, plays):
        assert self.ids(plays, (DIVS, {"head": ['"Acte"']})) == ["1 1 0 0 0", "1 1 2 0 0", "2 1 0 0 0"]

    def test_within_documents(self, plays):
        assert self.ids(plays, (DOCS, {"author": ['"Racine"']}), (DIVS, {"head": ['"Acte"']})) == [
            "1 1 0 0 0",
            "1 1 2 0 0",
        ]

    def test_within_nested_divs(self, plays):
        """The last act found (1 1 2) is in the first (1 1): the speech after it in the first is in an act too."""
        assert self.ids(plays, (DIVS, {"head": ['"Acte"']}), (PARAS, {"who": ['"Phèdre"']})) == [
            "1 1 1 1 1",
            "1 1 2 1 1",
            "1 1 3 1 1",
            "2 1 1 1 1",
        ]
        assert self.ids(
            plays, (DOCS, {"author": ['"Racine"']}), (DIVS, {"head": ['"Acte"']}), (PARAS, {"who": ['"Phèdre"']})
        ) == ["1 1 1 1 1", "1 1 2 1 1", "1 1 3 1 1"]

    def test_none_above(self, plays):
        assert self.ids(plays, (DOCS, {"author": ['"Molière"']}), (PARAS, {"who": ['"Phèdre"']})) == []

    def test_any_type(self, plays):
        """A level of no type, as philo_id alone makes, selects objects of any."""
        assert self.ids(plays, (None, {"head": ['"Prologue"']})) == ["1 2 0 0 0"]


@pytest.mark.unit
class TestIndexes:
    """With no statistics, SQLite took the philo_type index, which selects every object of a type, over a field's, as
    soon as a value had 5 alternatives: head=chapitre went through all of frantext's divs, in 0.35 s, not 0.05."""

    def plan(self, db, philo_types, fields):
        executed = []

        class Recorder:
            def execute(self, query, params):
                executed.append((query, params))
                return db.dbh.execute(query, params)

        recorded = SimpleNamespace(dbh=Recorder(), path=db.path, locals=db.locals)
        level_query(recorded, philo_types, fields, True).fetchall()
        query, params = executed[-1]
        return " ".join(row[3] for row in db.dbh.execute("EXPLAIN QUERY PLAN " + query, params))

    def test_field_index(self, plays):
        plays.dbh.execute("CREATE INDEX IF NOT EXISTS toms_philo_type_index ON toms (philo_type)")
        plays.dbh.execute("CREATE INDEX IF NOT EXISTS toms_head_index ON toms (head)")
        heads = " | ".join(f'"{head}"' for head in ("Acte", "Scène", "Prologue", "Épilogue", "Intermède"))
        assert "toms_head_index" in self.plan(plays, DIVS, {"head": [heads]})
        assert "toms_philo_type_index" in self.plan(plays, DIVS, {"head": [f"NOT {heads}"]})  # most divs
