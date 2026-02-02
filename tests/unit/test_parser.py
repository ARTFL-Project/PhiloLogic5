"""Unit tests for XMLParser."""

import io
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
