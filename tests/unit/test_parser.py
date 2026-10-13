"""Unit tests for XMLParser."""

import io
import json
import random
import sys
from pathlib import Path

import pytest
from orjson import dumps

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.OHCOVector import CompoundRecord
from philologic.loadtime.Parser import (
    XMLParser,
    control_char_re,
    DEFAULT_TAG_TO_OBJ_MAP,
    DEFAULT_METADATA_TO_PARSE,
    DEFAULT_DOC_XPATHS,
    TOKEN_REGEX,
)


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


def plain_parser(**parse_options):
    return XMLParser(io.StringIO(), 1, 1, metadata_sql_types={}, **parse_options)


@pytest.mark.unit
class TestTagExceptions:
    """cleanup_content only tries the tag exceptions regex where matches can start: same content as trying it
    everywhere (with tag_exceptions.sub)"""

    def test_used_with_default_options(self):
        assert plain_parser().tag_exception_starts is not None
        assert plain_parser(token_regex=TOKEN_REGEX).tag_exception_starts is not None
        assert plain_parser(token_regex=r"\w+").tag_exception_starts is None
        assert plain_parser(tag_exceptions=[r"(<hi>)"]).tag_exception_starts is None

    def test_same_content_as_regex_sub(self):
        pieces = ["a", "b", "é", "1", "&", ";", " ", "ͅ", "<hi>", "</hi>", "<i>", "<sup>", "<hi rend='x'>", "<"]
        pieces += [">", "</emph>", "<orig>", "<b>", "<lb/>"]
        rng = random.Random(5)
        fast, plain = plain_parser(), plain_parser()
        plain.tag_exception_starts = None
        for _ in range(20000):
            content = "".join(rng.choice(pieces) for _ in range(rng.randrange(1, 16)))
            fast.content = plain.content = content
            fast.cleanup_content()
            plain.cleanup_content()
            assert fast.content == plain.content, repr(content)


@pytest.mark.unit
def test_remove_control_chars():
    """Same as removing them with control_char_re, for every character"""
    parser = plain_parser()
    for code_point in range(0x110000):
        if not 0xD800 <= code_point <= 0xDFFF:
            text = f"a{chr(code_point)}b"
            assert parser.remove_control_chars(text) == control_char_re.sub("", text)


def compound_record_str(record):
    """CompoundRecord.__str__ as it was, to compare with"""
    print_id = record.id
    if 0 in record.id:
        parent_index = record.id.index(0) - 1
    else:
        parent_index = len(record.id) - 1
    parent_index = max(parent_index, 0)
    parent_id = record.id[:parent_index] + [0] * (len(record.id) - parent_index)
    print_id.append(record.attrib.get("start_byte", 0))
    print_id.append(record.attrib.get("page", 0))
    record.attrib["parent"] = " ".join(map(str, parent_id))
    clean_attrib = {}
    for k, v in record.attrib.items():
        value_type = type(v)
        if value_type is str:
            clean_attrib[k] = " ".join(v.split())
        elif value_type is int:
            clean_attrib[k] = v
        else:
            try:
                clean_attrib[k] = " ".join(v.split())
            except AttributeError:
                clean_attrib[k] = v
    return f"{record.type}\t{record.name}\t{' '.join(map(str, print_id))}\t{dumps(clean_attrib).decode('utf8')}"


@pytest.mark.unit
def test_compound_record_str():
    """Same line, and same record afterwards, as before"""
    values = [0, 1, 1234, "x", " a  b\t", "é f", 1.5, None, ["a"], True, "", "1 2 0 0"]
    for seed in range(20000):
        records = []
        for _ in range(2):
            rng = random.Random(seed)
            record = CompoundRecord(
                rng.choice(["word", "div1"]), rng.choice(["w", "a b"]), [rng.choice([0, 1, 37]) for _ in range(7)]
            )
            for key in rng.sample(["start_byte", "end_byte", "page", "parent", "head", "lemma"], rng.randrange(7)):
                record.attrib[key] = rng.choice(values)
            records.append(record)
        new, old = records
        for _ in range(2):  # the printed id grows at each call
            assert str(new) == compound_record_str(old)
            assert new.id == old.id and new.attrib == old.attrib


@pytest.mark.unit
@pytest.mark.parametrize("language", [None, "fr"])
def test_lowercase_after_exclamation_and_question_marks(language):
    """! and ? end a sentence, unless a word in lowercase follows: an interjection, or the words introducing what was
    said, go on with it"""
    xml = "<text><body><div><p>Ah! mon Dieu! Viens-tu ? » dit-il. Non? Oui!</p></div></body></text>"
    output = io.StringIO()
    options = {"language": language} if language else {}
    XMLParser(output, 1, len(xml.encode("utf8")), metadata_sql_types={}, **options).parse(io.StringIO(xml))
    sentences = {}
    for line in output.getvalue().split("\n")[:-1]:
        kind, name, philo_id, _ = line.split("\t")
        if kind == "word":
            sentences.setdefault(philo_id.split()[5], []).append(name)
    assert list(sentences.values()) == [["ah", "mon", "dieu"], ["viens", "tu", "dit", "il"], ["non"], ["oui"]]


@pytest.mark.unit
def test_lowercase_after_exclamation_and_question_marks_in_plain_text():
    from philologic.loadtime.PlainTextParser import PlainTextParser

    text = "Ah! mon Dieu! Viens-tu ? dit-il. Non? Oui!\n"
    output = io.StringIO()
    PlainTextParser(output, 1, len(text.encode("utf8"))).parse(io.StringIO(text))
    sentences = {}
    for line in output.getvalue().split("\n")[:-1]:
        kind, name, philo_id, _ = line.split("\t")
        if kind == "word":
            sentences.setdefault(philo_id.split()[5], []).append(name)
    assert list(sentences.values()) == [["ah", "mon", "dieu"], ["viens", "tu", "dit", "il"], ["non"], ["oui"]]


@pytest.mark.unit
@pytest.mark.parametrize("language", [None, "fr"])
def test_abbreviations_are_indexed_as_their_expansion(language):
    """The words of an expansion get the bytes of what they stand for in the file: of the word they expand, if the
    abbreviation has as many words, else all of it; a word joined across an abbreviation spans all its text"""
    xml = (
        '<text><body><div><p>avant <abbr expan="Valerius Maximus">Val. Max.</abbr> pres<abbr expan="en">&emacr;</abbr>t '
        '<abbr expan="que">q&#772;</abbr>dam <abbr expan="Saint Augustin">S. A. H.</abbr> <abbr>etc</abbr> '
        '<abbr expan="donatus es"><w>donatu</w>\'<w>s</w></abbr> <abbr expan="unus">u<pb n="2"/>n.</abbr></p></div>'
        "</body></text>"
    )
    output = io.StringIO()
    options = {"language": language} if language else {}
    XMLParser(output, 1, len(xml.encode("utf8")), metadata_sql_types={}, **options).parse(io.StringIO(xml))
    data = xml.encode("utf8")
    words = []
    for line in output.getvalue().split("\n")[:-1]:
        kind, name, _, attrib = line.split("\t")
        if kind == "word":
            attrib = json.loads(attrib)
            words.append((name, data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8")))
    assert words == [
        ("avant", "avant"),
        ("valerius", "Val."),
        ("maximus", "Max."),
        ("present", 'pres<abbr expan="en">&emacr;</abbr>t'),
        ("quedam", "q&#772;</abbr>dam"),
        ("saint", "S. A. H."),
        ("augustin", "S. A. H."),
        ("etc", "etc"),
        ("donatus", "donatu</w>'<w>s"),  # tags inside it, as <w>, dropped
        ("es", "donatu</w>'<w>s"),
        ("u", "u"),  # tags inside it which make objects (a page): left as it is
        ("n", "n"),
    ]
