"""Unit tests for XMLParser."""

import io
import json
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Parser import (
    XMLParser,
    DEFAULT_TAG_TO_OBJ_MAP,
    DEFAULT_METADATA_TO_PARSE,
    DEFAULT_DOC_XPATHS,
    TOKEN_REGEX,
)
from philologic.loadtime.PlainTextParser import PlainTextParser


@pytest.mark.unit
class TestXMLParserBasics:
    """Basic XMLParser functionality tests."""

    def test_parser_class_exists(self):
        """Test that XMLParser class is importable."""
        assert XMLParser is not None
        # Full initialization test is done via integration tests
        # since XMLParser requires many configuration parameters

    def test_default_tag_to_obj_map_structure(self):
        """Test that DEFAULT_TAG_TO_OBJ_MAP has expected structure."""
        assert isinstance(DEFAULT_TAG_TO_OBJ_MAP, dict)
        assert "div" in DEFAULT_TAG_TO_OBJ_MAP
        assert "p" in DEFAULT_TAG_TO_OBJ_MAP
        assert "pb" in DEFAULT_TAG_TO_OBJ_MAP
        assert DEFAULT_TAG_TO_OBJ_MAP["div"] == "div"
        assert DEFAULT_TAG_TO_OBJ_MAP["p"] == "para"
        assert DEFAULT_TAG_TO_OBJ_MAP["pb"] == "page"

    def test_default_doc_xpaths_structure(self):
        """Test that DEFAULT_DOC_XPATHS has expected structure."""
        assert isinstance(DEFAULT_DOC_XPATHS, dict)
        assert "author" in DEFAULT_DOC_XPATHS
        assert "title" in DEFAULT_DOC_XPATHS
        assert isinstance(DEFAULT_DOC_XPATHS["author"], list)
        assert len(DEFAULT_DOC_XPATHS["author"]) > 0

    def test_default_metadata_to_parse_structure(self):
        """Test that DEFAULT_METADATA_TO_PARSE has expected structure."""
        assert isinstance(DEFAULT_METADATA_TO_PARSE, dict)
        assert "div" in DEFAULT_METADATA_TO_PARSE
        assert "para" in DEFAULT_METADATA_TO_PARSE
        assert "page" in DEFAULT_METADATA_TO_PARSE
        assert "head" in DEFAULT_METADATA_TO_PARSE["div"]


@pytest.mark.unit
class TestTokenRegex:
    """Tests for token regex patterns."""

    def test_token_regex_matches_words(self):
        """Test that TOKEN_REGEX matches basic words."""
        import regex as re
        pattern = re.compile(TOKEN_REGEX)

        # Should match simple words
        assert pattern.search("hello") is not None
        assert pattern.search("world") is not None

    def test_token_regex_matches_unicode(self):
        """Test that TOKEN_REGEX matches unicode characters."""
        import regex as re
        pattern = re.compile(TOKEN_REGEX)

        # Should match accented characters
        assert pattern.search("café") is not None
        assert pattern.search("naïve") is not None
        assert pattern.search("résumé") is not None

    def test_token_regex_matches_numbers(self):
        """Test that TOKEN_REGEX matches numbers."""
        import regex as re
        pattern = re.compile(TOKEN_REGEX)

        # Should match numbers
        assert pattern.search("123") is not None
        assert pattern.search("1847") is not None

    def test_token_regex_matches_entities(self):
        """Test that TOKEN_REGEX handles entity patterns."""
        import regex as re
        pattern = re.compile(TOKEN_REGEX)

        # Should match entity-like patterns
        assert pattern.search("&amp;") is not None


@pytest.mark.unit
class TestParserTagHandling:
    """Tests for tag handling in parser."""

    def test_tag_to_obj_map_div_types(self):
        """Test that div tags are properly mapped."""
        assert DEFAULT_TAG_TO_OBJ_MAP.get("div1") == "div"
        assert DEFAULT_TAG_TO_OBJ_MAP.get("div2") == "div"
        assert DEFAULT_TAG_TO_OBJ_MAP.get("div3") == "div"
        assert DEFAULT_TAG_TO_OBJ_MAP.get("front") == "div"

    def test_tag_to_obj_map_para_types(self):
        """Test that paragraph-like tags are properly mapped."""
        para_tags = ["p", "sp", "lg", "note", "stage"]
        for tag in para_tags:
            assert DEFAULT_TAG_TO_OBJ_MAP.get(tag) == "para", f"{tag} should map to para"

    def test_tag_to_obj_map_special_types(self):
        """Test special tag mappings."""
        assert DEFAULT_TAG_TO_OBJ_MAP.get("pb") == "page"
        assert DEFAULT_TAG_TO_OBJ_MAP.get("ref") == "ref"
        assert DEFAULT_TAG_TO_OBJ_MAP.get("l") == "line"


@pytest.mark.unit
class TestMetadataExtraction:
    """Tests for metadata extraction patterns."""

    def test_author_xpaths(self):
        """Test that author XPaths are reasonable."""
        author_paths = DEFAULT_DOC_XPATHS["author"]
        assert any("titleStmt/author" in path for path in author_paths)
        assert any("sourceDesc" in path for path in author_paths)

    def test_title_xpaths(self):
        """Test that title XPaths are reasonable."""
        title_paths = DEFAULT_DOC_XPATHS["title"]
        assert any("titleStmt/title" in path for path in title_paths)
        assert any("sourceDesc" in path for path in title_paths)

    def test_date_xpaths(self):
        """Test that date XPaths exist."""
        assert "create_date" in DEFAULT_DOC_XPATHS
        assert "pub_date" in DEFAULT_DOC_XPATHS
        assert len(DEFAULT_DOC_XPATHS["pub_date"]) > 0

    def test_header_values_keep_xml_entities(self, tmp_path):
        """Test that &amp; and &lt; in a TEI header are decoded, without losing the text after them."""
        from philologic.loadtime.Loader import Loader

        (tmp_path / "text.xml").write_text(
            "<TEI><teiHeader><titleStmt><title>Letters &amp;c. Printed at Boston</title>"
            "<author>Becket &amp; DeHondt</author></titleStmt>"
            "<sourceDesc><bibl><pubPlace>1 &lt; 2 &eacute;</pubPlace></bibl></sourceDesc></teiHeader>"
            "<text><body><p>text</p></body></text></TEI>",
            encoding="utf8",
        )
        loader = Loader.__new__(Loader)
        loader.textdir = str(tmp_path)
        loader.deleted_files = []
        loader.debug = False
        loader.parser_config = {
            "doc_xpaths": {
                "title": [".//titleStmt/title"],
                "author": [".//titleStmt/author"],
                "pub_place": [".//sourceDesc/bibl/pubPlace"],
            },
            "metadata_sql_types": {},
        }
        [metadata] = loader.parse_tei_header(verbose=False)
        assert metadata["title"] == "Letters &c. Printed at Boston"
        assert metadata["author"] == "Becket & DeHondt"
        assert metadata["pub_place"] == "1 < 2 é"


def parse_div_head(head, div_tag='<div type="chapter">'):
    """Parse a div with the given head, and return the head the parser gives the div."""
    text = f"<TEI><text><body>{div_tag}\n{head}\n<p>Some text.</p>\n</div></body></text></TEI>\n"
    output = io.StringIO()
    parser = XMLParser(output, 1, len(text.encode()), known_metadata={"filename": "t.xml"}, metadata_sql_types={})
    parser.parse(io.StringIO(text))
    for line in output.getvalue().splitlines():
        if line.startswith("div1"):
            return json.loads(line.split("\t")[3])["head"]


@pytest.mark.unit
class TestDivHead:
    """Tests for the heads of divs, as table of contents labels."""

    @pytest.mark.parametrize(
        "head, expected",
        [
            ("<head>HISTOIRE<lb/>\nDES COLONIES</head>", "HISTOIRE DES COLONIES"),
            ("<head>HISTOIRE<lb/>DES COLONIES</head>", "HISTOIRE DES COLONIES"),
            ('<head>PROS<lb break="no"/>\nPERITE</head>', "PROSPERITE"),
            ("<head>CHAPITRE I.</head><head>Des Colonies</head>", "CHAPITRE I. Des Colonies"),
            ('<head><hi rend="initial">L</hi>ETTRES</head>', "LETTRES"),
            ('<head>1<hi rend="sup">re</hi> PARTIE</head>', "1re PARTIE"),
            ("<head></head>", "chapter"),
        ],
    )
    def test_head(self, head, expected):
        """Test that tags break words in heads as in the index, and that heads are kept apart."""
        assert parse_div_head(head) == expected

    @pytest.mark.parametrize(
        "div_tag, head, expected",
        [
            ('<div type="tome" n="2">', "", "tome 2"),
            ('<div n="2" type="tome">', "", "tome 2"),
            ('<div type="tome" n="2">', "<head>TOME SECOND</head>", "TOME SECOND"),
            ('<div type="tome" n="">', "", "tome"),
            ('<div type="tome" xml:n="2">', "", "tome"),
            ('<div n="2">', "", "[NA]"),
            ('<div type="chapitre" n="IV">', "", "chapitre IV"),
            ('<div type="scène" n="3bis">', "", "scène 3bis"),
            ('<div type="article" n="3.1">', "", "article 3.1"),
        ],
    )
    def test_head_from_type_and_n(self, div_tag, head, expected):
        """Test that a div with no head is named by its type and its n, if it has one."""
        assert parse_div_head(head, div_tag) == expected

    @pytest.mark.parametrize(
        "div_tag, head, expected",
        [
            ('<div type="poème" n="Le Vœu">', "", "Le Vœu"),
            ('<div type="partie" n="Première partie">', "", "Première partie"),
            ('<div type="journal" n="Octobre 1914">', "", "Octobre 1914"),
            ('<div n="Préface">', "", "Préface"),
            ('<div type="poème" n="Le Vœu">', "<head>LE VŒU</head>", "LE VŒU"),
        ],
    )
    def test_head_from_n_label(self, div_tag, head, expected):
        """Test that a div with no head whose n is a label, not a number, is named by its n alone."""
        assert parse_div_head(head, div_tag) == expected


@pytest.mark.unit
def test_long_s_is_indexed_as_s():
    """Test that words with a long s are indexed with an s, at the bytes of the long s form."""
    text = "<TEI><text><body><div><p>Ils ſont Chriſtiens.</p></div></body></text></TEI>\n"
    output = io.StringIO()
    parser = XMLParser(output, 1, len(text.encode()), known_metadata={"filename": "t.xml"}, metadata_sql_types={})
    parser.parse(io.StringIO(text))
    words = [line.split("\t") for line in output.getvalue().splitlines() if line.startswith("word")]
    assert [word[1] for word in words] == ["ils", "sont", "christiens"]
    sont = json.loads(words[1][3])
    assert text.encode()[sont["start_byte"] : sont["end_byte"]].decode() == "ſont"


@pytest.mark.unit
def test_long_s_is_indexed_as_s_in_plain_text():
    """Test that the plain text parser indexes words with a long s with an s too, at the bytes of the long s form."""
    text = "Ils ſont Chriſtiens.\n"
    output = io.StringIO()
    parser = PlainTextParser(output, 1, len(text.encode()), known_metadata={"filename": "t.txt"})
    parser.parse(io.StringIO(text))
    words = [line.split("\t") for line in output.getvalue().splitlines() if line.startswith("word")]
    assert [word[1] for word in words] == ["ils", "sont", "christiens"]
    sont = json.loads(words[1][3])
    assert text.encode()[sont["start_byte"] : sont["end_byte"]].decode() == "ſont"


def parse_words(text):
    """Parse the TEI text, and return the words indexed."""
    output = io.StringIO()
    parser = XMLParser(output, 1, len(text.encode()), known_metadata={"filename": "t.xml"}, metadata_sql_types={})
    parser.parse(io.StringIO(text))
    return [line.split("\t")[1] for line in output.getvalue().splitlines() if line.startswith("word")]


@pytest.mark.unit
def test_text_elements_in_header_do_not_start_the_text():
    """Test that <text> elements in the TEI header (empty or not, as in a source description) do not start the text:
    no header word is indexed, and the header's <date> is not handled as a date in a div."""
    text = (
        "<TEI><teiHeader><sourceDesc><sources><source><text/></source><source><text>**</text></source></sources>"
        "</sourceDesc>\n"
        '<profileDesc><creation><date type="created" when="1633">1633</date> Theatre</creation></profileDesc>\n'
        "</teiHeader>\n"
        "<text><body><div><p>Corps du texte.</p></div></body></text></TEI>\n"
    )
    assert parse_words(text) == ["corps", "du", "texte"]


@pytest.mark.unit
def test_date_outside_any_div_is_skipped():
    """Test that a <date> in the body before the first div, or after the last one, does not crash the parser."""
    text = (
        '<TEI><text><body><p>Avant <date when="1633">1633</date>.</p>\n'
        '<div><p>Dans.</p></div><p>Apres <date when="1700">1700</date>.</p></body></text></TEI>\n'
    )
    assert parse_words(text) == ["avant", "dans", "apres"]


def parse_divs(text):
    """Parse a TEI text of this front and body, and return its divs in document order: (object type, head, the words
    in it), the words found from their ids"""
    text = f"<TEI><text>{text}</text></TEI>\n"
    output = io.StringIO()
    parser = XMLParser(output, 1, len(text.encode()), known_metadata={"filename": "t.xml"}, metadata_sql_types={})
    parser.parse(io.StringIO(text))
    records = [line.split("\t") for line in output.getvalue().splitlines()]
    words = [(ident.split(), name) for kind, name, ident, _ in records if kind == "word"]
    divs = []
    for kind, name, ident, attrib in records:
        if kind.startswith("div") and name != "__philo_virtual":
            div_id = ident.split()[: int(kind[3]) + 1]
            div_words = " ".join(word for word_id, word in words if word_id[: len(div_id)] == div_id)
            divs.append(([int(i) for i in div_id], kind, json.loads(attrib).get("head", ""), div_words))
    return [div[1:] for div in sorted(divs)]


@pytest.mark.unit
class TestDivLevels:
    """Tests for the levels of divs and where they end."""

    @pytest.mark.parametrize("part, chapter", [("div", "div"), ("div1", "div2")])
    def test_div_ends_at_its_end_tag(self, part, chapter):
        """Test that a div ends at its end tag, numbered or not: what follows is in the div it was in."""
        text = (
            f'<body><{part} type="part"><head>Partie</head><{chapter} type="chapter"><head>Premier</head>'
            f'<p>Dedans.</p></{chapter}><p>Après.</p><{chapter} type="chapter"><head>Second</head><p>Ailleurs.</p>'
            f"</{chapter}></{part}></body>"
        )
        assert parse_divs(text) == [
            ("div1", "Partie", "partie premier dedans après second ailleurs"),
            ("div2", "Premier", "premier dedans"),
            ("div2", "Second", "second ailleurs"),
        ]

    def test_numbered_div_in_front_ends_at_its_end_tag(self):
        """Test that a numbered div in the front matter (a div1) ends at its end tag, whatever its number."""
        text = (
            '<front><div2 type="preface"><head>Préface</head><p>Dedans.</p></div2><p>Après.</p></front>'
            "<body><div1><head>Corps</head><p>Texte.</p></div1></body>"
        )
        assert parse_divs(text) == [
            ("div1", "", "préface dedans après"),
            ("div2", "Préface", "préface dedans"),
            ("div1", "Corps", "corps texte"),
        ]

    def test_empty_divs_and_div_gen_are_no_divs(self):
        """Test that self-closing divs and <divGen> make no div, and leave the levels of the next divs as they were."""
        text = (
            '<body><div type="part"><head>Une</head><divGen type="toc"/><div type="chapter"><head>Chapitre</head>'
            '<p>Texte.</p></div><div type="notes"/></div><div type="part"><head>Deux</head><p>Fin.</p></div></body>'
        )
        assert parse_divs(text) == [
            ("div1", "Une", "une chapitre texte"),
            ("div2", "Chapitre", "chapitre texte"),
            ("div1", "Deux", "deux fin"),
        ]

    def test_divs_after_a_fourth_level_div_keep_their_level(self):
        """Test that a div deeper than three levels takes the place of the div3, and the divs after it have the level
        of their depth."""
        text = (
            "<body><div><head>Un</head><div><head>Deux</head><div><head>Trois</head><div><head>Quatre</head>"
            "<p>Texte.</p></div></div><div><head>Troisbis</head><p>Texte.</p></div></div>"
            "<div><head>Deuxbis</head><p>Texte.</p></div></div></body>"
        )
        assert [(kind, head) for kind, head, _ in parse_divs(text)] == [
            ("div1", "Un"),
            ("div2", "Deux"),
            ("div3", "Trois"),
            ("div3", "Quatre"),
            ("div3", "Troisbis"),
            ("div2", "Deuxbis"),
        ]

    def test_numbered_divs_without_end_tags(self):
        """Test that a numbered div ends the divs of its level or deeper whose end tags are left out."""
        text = (
            "<body><div1><head>Un</head><p>A.</p><div2><head>Deux</head><p>B.</p><div2><head>Trois</head><p>C.</p>"
            "<div1><head>Quatre</head><p>D.</p></body>"
        )
        assert parse_divs(text) == [
            ("div1", "Un", "un a deux b trois c"),
            ("div2", "Deux", "deux b"),
            ("div2", "Trois", "trois c"),
            ("div1", "Quatre", "quatre d"),
        ]

    def test_mismatched_end_tags(self):
        """Test that a numbered end tag of another name than the div tags still open ends the innermost one of its
        level, and nothing if a div tag of its name was ended before it, as a div0 by its first div1, and that an
        unnumbered one ends nothing: the numbers of the divs give their levels."""
        text = (
            "<body><div><head>Un</head><p>A.</p></div1><div><head>Deux</head><p>B.</p></div1>"
            "<div1><head>Trois</head></div2><div2><head>Quatre</head><p>C.</p></div2></div1>"
            "<div0><head>Cinq</head><div1><head>Six</head><p>D.</p></div1></div0><div><head>Sept</head></div>"
            "<div1><head>Acte</head><p>E.</p></div><div2><head>Scène</head><p>F.</p></div></body>"
        )
        assert parse_divs(text) == [
            ("div1", "Un", "un a"),
            ("div1", "Deux", "deux b"),
            ("div1", "Trois", "trois quatre c"),
            ("div2", "Quatre", "quatre c"),
            ("div1", "Cinq", "cinq"),
            ("div1", "Six", "six d"),
            ("div1", "Sept", "sept"),
            ("div1", "Acte", "acte e scène f"),
            ("div2", "Scène", "scène f"),
        ]
