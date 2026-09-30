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
