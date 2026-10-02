"""Unit tests for QuerySyntax parsing."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.exceptions import BadRequest
from philologic.runtime.Query import MAX_DISTANCE, check_method, check_phrases, query_parse, resolve_method, split_terms
from philologic.runtime.QuerySyntax import parse_query, group_terms, parse_date_query, quoted_text


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


@pytest.mark.unit
class TestCustomQueryPatterns:
    """Tests for custom query_patterns parameter."""

    def test_default_patterns_when_none(self):
        """Test that None query_patterns uses built-in defaults."""
        result = parse_query("hamlet", query_patterns=None)
        assert result == [("TERM", "hamlet")]

    def test_custom_patterns(self):
        """Test that custom patterns override defaults."""
        # Custom patterns that only recognize TERM (no RANGE, QUOTE, etc.)
        custom_patterns = [
            ("TERM", r'[^\s]+'),
        ]
        result = parse_query("1800-1850", query_patterns=custom_patterns)
        # With default patterns this would be RANGE, but custom treats it as TERM
        assert len(result) == 1
        assert result[0] == ("TERM", "1800-1850")

    def test_custom_patterns_new_token_type(self):
        """Test that custom patterns can introduce new token types."""
        custom_patterns = [
            ("WILDCARD", r'\*'),
            ("TERM", r'[^\s*]+'),
        ]
        result = parse_query("test*word", query_patterns=custom_patterns)
        assert ("WILDCARD", "*") in result
        assert ("TERM", "test") in result
        assert ("TERM", "word") in result


@pytest.mark.unit
class TestPhrasesAlone:
    """A quoted phrase is searched as the words of several groups, so it can only be a group of its own."""

    @pytest.mark.parametrize(
        "query",
        ['"la liberté" | "le roi"', '"la liberté" | roi de', 'roi NOT "le roi"', 'chine | "Empire du milieu" | pékin',
         '"sangue di drago" | "sangue di dragone"'],
    )
    def test_refused(self, query):
        with pytest.raises(BadRequest, match="quoted phrase"):
            check_phrases(group_terms(parse_query(query)))

    @pytest.mark.parametrize(
        "query", ['"la liberté"', '"la liberté" roi', "roi | reine", '"roi" | "reine"', 'roi NOT "rois"', '"la liberté', "roi"],
    )
    def test_allowed(self, query):
        check_phrases(group_terms(parse_query(query)))


@pytest.mark.unit
class TestQueryParserRules:
    """The database's query_parser_regex rules (frantext's, in part) rewrite queries, but not regex bracket expressions,
    nor quoted metadata values."""

    config = type("Config", (), {"query_parser_regex": [(" OR ", " | "), ("'", " "), (",", ""), ("-", " ")]})()

    @pytest.mark.parametrize(
        "query, rewritten",
        [
            ("peut-être", "peut être"),
            ("aujourd'hui OR demain", "aujourd hui | demain"),
            ('"peut-être"', '"peut être"'),  # quoted words of a search are split as the index splits them
            ("[a-z]tat", "[a-z]tat"),
            ("[a-z]tat-ci", "[a-z]tat ci"),
            ("-".join(["x"] * 41), " ".join(["x"] * 41)),  # every hyphen: re.U was passed as re.sub's count (32)
        ],
    )
    def test_search_terms(self, query, rewritten):
        assert query_parse(query, self.config) == rewritten

    @pytest.mark.parametrize(
        "value, rewritten",
        [
            ('zola | "Hugo, Victor, 1802-1885."', 'zola | "Hugo, Victor, 1802-1885."'),
            ('NOT "Hugo, Victor, 1802-1885."', 'NOT "Hugo, Victor, 1802-1885."'),
            ("rousseau, jean-jacques", "rousseau jean jacques"),
        ],
    )
    def test_metadata_values(self, value, rewritten):
        assert query_parse(value, self.config, keep_quoted=True) == rewritten


@pytest.mark.unit
class TestQuotedTerms:
    @pytest.mark.parametrize("token, text", [('"liberté"', "liberté"), ('"liberté', "liberté"), ('""', ""), ('"', "")])
    def test_quoted_text(self, token, text):
        """The closing quote may be missing: its last letter was dropped instead ("liberté searched libert)."""
        assert quoted_text(token) == text

    @pytest.mark.parametrize("query", ['"la liberté"', '"la  liberté"', '"la liberté'])
    def test_phrase_groups(self, query):
        assert split_terms(group_terms(parse_query(query))) == [(("QUOTE", '"la"'),), (("QUOTE", '"liberté"'),)]


@pytest.mark.unit
class TestSearchChecks:
    @pytest.mark.parametrize("query", ["NOT amour", "roi NOT reine", "a.* NOT abalone"])
    def test_not(self, query):
        grouped = group_terms(parse_query(query))
        if query.startswith("NOT"):
            with pytest.raises(BadRequest, match="NOT"):
                check_phrases(grouped)
        else:
            check_phrases(grouped)

    def test_methods(self):
        two_groups = split_terms(group_terms(parse_query("roi reine")))
        one_group = split_terms(group_terms(parse_query("roi")))
        with pytest.raises(BadRequest, match="Unknown search method"):
            check_method(two_groups, "cooc", "")
        with pytest.raises(BadRequest, match="negative"):
            check_method(two_groups, "proxy_unordered", "-3")
        check_method(one_group, "cooc", "-3")  # one group is searched by single_term whatever the method
        check_method(two_groups, "sentence_unordered", "6")
        check_method(two_groups, "", "")

    def test_huge_distance(self):
        assert resolve_method("roi reine", "proxy", "99999999999999999999", "no") == ("proxy_unordered", str(MAX_DISTANCE))
