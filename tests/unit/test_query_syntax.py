"""Unit tests for QuerySyntax parsing."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.QuerySyntax import parse_query, group_terms, parse_date_query


@pytest.mark.unit
class TestParseQuery:
    """Tests for parse_query function."""

    def test_single_term(self):
        """Test parsing a single search term."""
        result = parse_query("hamlet")
        assert len(result) == 1
        assert result[0] == ("TERM", "hamlet")

    def test_multiple_terms(self):
        """Test parsing multiple search terms."""
        result = parse_query("to be")
        assert len(result) == 2
        assert result[0] == ("TERM", "to")
        assert result[1] == ("TERM", "be")

    def test_quoted_phrase(self):
        """Test parsing a quoted phrase."""
        result = parse_query('"to be or not to be"')
        assert len(result) == 1
        assert result[0][0] == "QUOTE"
        assert "to be or not to be" in result[0][1]

    def test_or_operator(self):
        """Test parsing OR operator."""
        result = parse_query("hamlet | macbeth")
        assert len(result) == 3
        assert result[0] == ("TERM", "hamlet")
        assert result[1] == ("OR", "|")
        assert result[2] == ("TERM", "macbeth")

    def test_not_operator(self):
        """Test parsing NOT operator."""
        result = parse_query("hamlet NOT ghost")
        assert len(result) == 3
        assert result[0] == ("TERM", "hamlet")
        assert result[1] == ("NOT", "NOT")
        assert result[2] == ("TERM", "ghost")

    def test_range_query(self):
        """Test parsing range query."""
        result = parse_query("1800-1850")
        assert len(result) == 1
        assert result[0][0] == "RANGE"
        assert result[0][1] == "1800-1850"

    def test_lemma_query(self):
        """Test parsing lemma search."""
        result = parse_query("lemma:be")
        assert len(result) == 1
        assert result[0][0] == "LEMMA"
        assert result[0][1] == "lemma:be"

    def test_attr_query(self):
        """Test parsing attribute search."""
        result = parse_query("pos:NOUN")
        assert len(result) == 1
        assert result[0][0] == "ATTR"
        assert result[0][1] == "pos:NOUN"

    def test_lemma_attr_query(self):
        """Test parsing combined lemma and attribute search."""
        result = parse_query("lemma:be:pos")
        assert len(result) == 1
        assert result[0][0] == "LEMMA_ATTR"

    def test_null_query(self):
        """Test parsing NULL value."""
        result = parse_query("NULL")
        assert len(result) == 1
        assert result[0] == ("NULL", "NULL")

    def test_complex_query(self):
        """Test parsing a complex query with multiple operators."""
        result = parse_query('"my lord" | hamlet NOT ghost')
        assert any(item[0] == "QUOTE" for item in result)
        assert any(item[0] == "OR" for item in result)
        assert any(item[0] == "NOT" for item in result)


@pytest.mark.unit
class TestGroupTerms:
    """Tests for group_terms function."""

    def test_group_single_term(self):
        """Test grouping a single term."""
        parsed = parse_query("hamlet")
        grouped = group_terms(parsed)
        assert len(grouped) == 1
        assert grouped[0] == [("TERM", "hamlet")]

    def test_group_or_terms(self):
        """Test grouping OR terms together."""
        parsed = parse_query("hamlet | macbeth")
        grouped = group_terms(parsed)
        # OR terms should be grouped together
        assert len(grouped) >= 1

    def test_group_not_terms(self):
        """Test grouping NOT terms."""
        parsed = parse_query("hamlet NOT ghost")
        grouped = group_terms(parsed)
        # NOT should be grouped with preceding term
        assert len(grouped) >= 1

    def test_group_multiple_independent_terms(self):
        """Test grouping multiple independent terms."""
        parsed = parse_query("love death")
        grouped = group_terms(parsed)
        # Two independent terms create separate groups
        assert len(grouped) == 2

    def test_group_empty_filtered(self):
        """Test that empty groups are filtered out."""
        parsed = parse_query("hamlet")
        grouped = group_terms(parsed)
        assert all(g != [] for g in grouped)


@pytest.mark.unit
class TestParseDateQuery:
    """Tests for parse_date_query function."""

    def test_year_query(self):
        """Test parsing a single year."""
        result = parse_date_query("1847")
        assert len(result) == 1
        # Year is expanded to a range
        assert result[0][0] == "DATE_RANGE"
        assert "1847-01-01" in result[0][1]
        assert "1847-12-31" in result[0][1]

    def test_year_month_query(self):
        """Test parsing year-month format."""
        result = parse_date_query("1847-06")
        assert len(result) == 1
        assert result[0][0] == "DATE_RANGE"

    def test_date_range_query(self):
        """Test parsing date range."""
        result = parse_date_query("1800<=>1850")
        assert len(result) == 1
        assert result[0][0] == "DATE_RANGE"
        assert "1800" in result[0][1]
        assert "1850" in result[0][1]

    def test_date_or_query(self):
        """Test parsing OR in date query."""
        result = parse_date_query("1800 | 1850")
        # Should contain OR operator
        assert any(item[0] == "OR" for item in result)


@pytest.mark.unit
class TestQueryEdgeCases:
    """Tests for edge cases in query parsing."""

    def test_empty_query(self):
        """Test parsing empty query."""
        result = parse_query("")
        assert result == []

    def test_whitespace_only(self):
        """Test parsing whitespace-only query."""
        result = parse_query("   ")
        assert result == []

    def test_special_characters(self):
        """Test handling of special characters."""
        # Ampersand should be handled
        result = parse_query("rock & roll")
        assert len(result) >= 2

    def test_unclosed_quote(self):
        """Test handling of unclosed quote."""
        result = parse_query('"to be')
        # Should still parse, treating unclosed quote as partial
        assert len(result) >= 1

    def test_unicode_term(self):
        """Test parsing unicode search term."""
        result = parse_query("café")
        assert len(result) == 1
        assert result[0] == ("TERM", "café")
