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
@pytest.mark.parametrize(
    "body, records",
    [
        ("Ga<hi/>maliel</hi>; Act", [("word", "Ga<hi/>maliel"), ("punct", ";"), ("word", "Act")]),
        ('<hi>Laugh<hi rend="x"/>ter;</hi> x', [("word", 'Laugh<hi rend="x"/>ter'), ("punct", ";"), ("word", "x")]),
        ("Ga<hi/>;</hi> x", [("word", "Ga"), ("punct", ";"), ("word", "x")]),
        (
            "in-4<sup>o</sup>. Elle",
            [("word", "in"), ("punct", "-"), ("word", "4<sup>o"), ("punct", "."), ("word", "Elle")],
        ),
    ],
)
def test_words_joined_across_tags_before_a_semicolon(body, records):
    """A word joined across tag exceptions ends where its text does, and a semicolon right after its tags (which the
    entity alternative of TOKEN_REGEX let the join take) keeps its own bytes"""
    xml = f"<text><body><div><p>{body}</p></div></body></text>"
    output = io.StringIO()
    XMLParser(output, 1, len(xml.encode("utf8")), metadata_sql_types={}).parse(io.StringIO(xml))
    data = xml.encode("utf8")
    found = []
    for line in output.getvalue().split("\n")[:-1]:
        kind, _, _, attrib = line.split("\t")
        if kind in ("word", "punct"):
            attrib = json.loads(attrib)
            found.append((kind, data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8")))
    assert found == records


@pytest.mark.unit
@pytest.mark.parametrize(
    "body, words",
    [
        ('ex<g ref="char:EOLhyphen"/>ecution', [("execution", 'ex<g ref="char:EOLhyphen"/>ecution')]),
        ("co<g>\u0304</g>mo<g>\u0304</g>s", [("co\u0304mo\u0304s", "co<g>\u0304</g>mo<g>\u0304</g>s")]),  # (abbreviation strokes)
        ("x<hi>y</hi>z<hi>w</hi>", [("xyzw", "x<hi>y</hi>z<hi>w")]),
        ("wo<hi><i>r</i></hi>d", [("word", "wo<hi><i>r</i></hi>d")]),
        ("expedient<lb/>demourer", [("expedient", "expedient"), ("demourer", "demourer")]),
    ],
)
def test_words_joined_across_tags(body, words):
    """Tag exceptions don't split words, one or several, in a row or not: other tags do"""
    xml = f"<text><body><div><p>{body} end</p></div></body></text>"
    output = io.StringIO()
    XMLParser(output, 1, len(xml.encode("utf8")), metadata_sql_types={}).parse(io.StringIO(xml))
    data = xml.encode("utf8")
    found = []
    for line in output.getvalue().split("\n")[:-1]:
        kind, name, _, attrib = line.split("\t")
        if kind == "word":
            attrib = json.loads(attrib)
            found.append((name, data[attrib["start_byte"] : attrib["end_byte"]].decode("utf8")))
    assert found == words + [("end", "end")]


@pytest.mark.unit
def test_abbreviation_joined_across_tags():
    """The tag after a word joined across tags is taken out with it: the abbreviation M<hi>r</hi>. ends no sentence"""
    xml = '<text><body><div><p>M<hi rend="sup">r</hi>. Smith came.</p></div></body></text>'
    output = io.StringIO()
    XMLParser(output, 1, len(xml.encode("utf8")), metadata_sql_types={}).parse(io.StringIO(xml))
    sentences = [line.split("\t")[1] for line in output.getvalue().split("\n")[:-1] if line.startswith("sent\t")]
    assert sentences == ["."]


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


def parse_div_head(head):
    """Parse a div with the given head, and return the head the parser gives the div."""
    text = f'<TEI><text><body><div type="chapter">\n{head}\n<p>Some text.</p>\n</div></body></text></TEI>\n'
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "flatten_ligatures, forms, lemma",
    [
        (True, ["mon", "coeur", "oeil", "oeuvres", "fin", "caesar", "coeurs"], "coeur"),
        (False, ["mon", "cœur", "œil", "œuvres", "ﬁn", "cæsar", "cœurs"], "cœur"),
    ],
)
def test_ligatures_are_indexed_as_letters(flatten_ligatures, forms, lemma):
    """Test that with flatten_ligatures, words have their ligatures as letters, as they have their entities, at the
    bytes of the ligature form, and so do the lemmas of <w> tags."""
    text = (
        "<TEI><text><body><div><p>Mon cœur, &oelig;il, Œuvres, ﬁn, Cæsar, "
        '<w lemma="cœur">cœurs</w>.</p></div></body></text></TEI>\n'
    )
    if not flatten_ligatures:  # (the entity then stays one)
        text = text.replace("&oelig;", "œ")
    output = io.StringIO()
    parser = XMLParser(
        output,
        1,
        len(text.encode()),
        known_metadata={"filename": "t.xml"},
        metadata_sql_types={},
        flatten_ligatures=flatten_ligatures,
    )
    parser.parse(io.StringIO(text))
    words = [line.split("\t") for line in output.getvalue().splitlines() if line.startswith("word")]
    assert [word[1] for word in words] == forms
    coeur = json.loads(words[1][3])
    assert text.encode()[coeur["start_byte"] : coeur["end_byte"]].decode() == "cœur"
    assert json.loads(words[-1][3])["lemma"] == lemma


@pytest.mark.unit
@pytest.mark.parametrize("flatten_ligatures, forms", [(True, ["mon", "coeur", "fin"]), (False, ["mon", "cœur", "ﬁn"])])
def test_ligatures_are_indexed_as_letters_in_plain_text(flatten_ligatures, forms):
    """Test that the plain text parser indexes ligatures as their letters with flatten_ligatures too, and the lemmas of
    its words so."""
    text = "Mon Cœur, ﬁn.\n"
    output = io.StringIO()
    lemmas = {"coeur": "coeur", "cœur": "cœur"}  # as the loader has those of the lemma file
    parser = PlainTextParser(
        output,
        1,
        len(text.encode()),
        known_metadata={"filename": "t.txt"},
        lemmas=lemmas,
        flatten_ligatures=flatten_ligatures,
    )
    parser.parse(io.StringIO(text))
    words = [line.split("\t") for line in output.getvalue().splitlines() if line.startswith("word")]
    assert [word[1] for word in words] == forms
    coeur = json.loads(words[1][3])
    assert text.encode()[coeur["start_byte"] : coeur["end_byte"]].decode() == "Cœur"
    assert coeur["lemma"] == forms[1]
