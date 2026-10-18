"""Unit tests for the TEI header functions of the loader, used by parse_tei_header and the load previews"""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Loader import Loader, tei_header, tei_header_metadata

pytestmark = pytest.mark.unit

TEXT = """<TEI><teiHeader><fileDesc><titleStmt><title>Le Titre</title><author></author></titleStmt>
<sourceDesc><bibl><author>Diderot</author><date>vers 1770</date></bibl></sourceDesc></fileDesc>
<profileDesc><creation><date>sans date</date></creation></profileDesc></teiHeader><text>Texte</text></TEI>"""

XPATHS = {
    "title": [".//titleStmt/title"],
    "author": [".//titleStmt/author", ".//sourceDesc/bibl/author"],
    "create_date": [".//profileDesc/creation/date", ".//sourceDesc/bibl/date/"],
    "publisher": [".//publicationStmt/publisher"],
}


def test_no_header():
    assert tei_header("<TEI><text>Texte</text></TEI>") is None


def test_header_metadata_and_matched_xpaths():
    matched = {}
    metadata = tei_header_metadata(tei_header(TEXT), XPATHS, {}, matched)
    # Empty elements and dates without digits are skipped, trailing slashes of xpaths ignored
    assert metadata == {"title": "Le Titre", "author": "Diderot", "create_date": "vers 1770"}
    assert matched == {
        "title": ".//titleStmt/title",
        "author": ".//sourceDesc/bibl/author",
        "create_date": ".//sourceDesc/bibl/date/",
    }
    assert Loader.create_year_field(metadata)["year"] == 1770


def test_header_metadata_sql_types():
    metadata = tei_header_metadata(tei_header(TEXT), XPATHS, {"create_date": "int", "title": "text"})
    # An int field takes the first value found, digits or not; a text type leaves the field out
    assert metadata == {"author": "Diderot", "create_date": None}


@pytest.mark.parametrize(
    "date, year",
    [
        ("Octob. 12. 1642.", 1642),  # the year, not the day
        ("2649 [i.e. 1649]", 1649),  # the correction of a misprint
        ("16461 [sic, i.e. 1646]", 1646),
        ("1733/4 [i.e. 1734]", 1734),
        ("MDCLXXX1X [1689]", 1689),
        ("1.5.5.0. [1550]", 1550),
        ("1,5,5,7 [i.e. 1557]", 1557),
        ("in the yeare of our Lorde God. 1.5.4.6]", 1546),  # digits separated by dots
        ("[1]596.", 1596),  # digits a cataloguer supplied
        ("1[651?]", 1651),
        ("1744[1743]", 1744),
        ("[1655-1680]", 1655),
        ("post-1640]", 1640),  # a hyphen inside a date joins
        ("--1798--", 1798),
        ("-0044", -44),  # a minus starting the date: before the common era
        ("-900 CE (900 BCE)", -900),
        ("44 BC", -44),
        ("99", 99),
        ("16uu", 1600),  # unknown digits (MARC, EDTF): the first year of the span
        ("169u", 1690),
        ("16XX", 1600),
        ("1uuu-1672?", 1672),  # only the millennium: no year
        ("--05-12", None),  # W3C's month and day, without a year
        ("[s. d.]", None),
    ],
)
def test_year_of_the_date(date, year):
    assert Loader.create_year_field({"create_date": date}).get("year") == year
