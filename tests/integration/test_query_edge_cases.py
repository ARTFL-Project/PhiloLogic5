"""Integration tests for queries that used to fail or give wrong results: regex metacharacters, words in no index,
and hitlists written with another width than they are read with."""

import os
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime.exceptions import BadRequest


def finished(hits):
    """hits, once complete, checked to hold whole hits of the width it reads them with."""
    hits.finish()
    size = os.path.getsize(hits.filename)
    assert size == len(hits) * hits.hitsize, f"{size} bytes is no whole number of {hits.hitsize}-byte hits"
    return hits


def count(db, q, method="single_term", method_arg="0"):
    return len(finished(db.query(q, method, method_arg)))


@pytest.mark.integration
class TestWordsInNoIndex:
    """A query group that matches no word makes the whole query match nothing."""

    @pytest.mark.parametrize("method, method_arg", [
        ("phrase_ordered", "0"),
        ("phrase_unordered", "0"),
        ("proxy_unordered", "5"),
        ("proxy_ordered", "5"),
        ("sentence_unordered", "0"),
    ])
    def test_missing_word(self, shakespeare_db, method, method_arg):
        assert count(shakespeare_db, "lord", "single_term") > 0
        assert count(shakespeare_db, "my lord", method, method_arg) > 0
        assert count(shakespeare_db, "lord xqzwv", method, method_arg) == 0
        assert count(shakespeare_db, "xqzwv lord", method, method_arg) == 0
        assert count(shakespeare_db, "my xqzwv lord", method, method_arg) == 0

    def test_missing_quoted_word(self, shakespeare_db):
        assert count(shakespeare_db, '"my" "xqzwv"', "phrase_ordered") == 0


@pytest.mark.integration
class TestSingleTermGroups:
    """single_term searches merge the hits of every group into hits of one word."""

    def test_two_groups(self, shakespeare_db):
        both = finished(shakespeare_db.query('"hamlet" "lord"', "single_term", "0"))
        assert both.length == 9
        assert len(both) == count(shakespeare_db, '"hamlet"') + count(shakespeare_db, '"lord"')


@pytest.mark.integration
class TestRegexTokens:
    """Regex tokens match whole words, and tokens that are not valid regexes are words."""

    def test_optional_character(self, shakespeare_db):
        assert count(shakespeare_db, "lords?") == count(shakespeare_db, "lord") + count(shakespeare_db, "lords")
        # The prefix scanned from leaves out the character the quantifier makes optional
        assert count(shakespeare_db, "lordd?") == count(shakespeare_db, "lord")

    def test_whole_word(self, shakespeare_db):
        """"lor?" matches "lor" and "lo", not every word starting with them."""
        assert count(shakespeare_db, "lor?") == count(shakespeare_db, "lor") + count(shakespeare_db, "lo")

    def test_wildcard(self, shakespeare_db):
        assert count(shakespeare_db, "lord.*") >= count(shakespeare_db, "lord") + count(shakespeare_db, "lords")

    @pytest.mark.parametrize("q", ["lord[", "*lord*", "lord\\", '"(my lord)"'])
    def test_invalid_regex(self, shakespeare_db, q):
        """Searched for as words, which they are not: no hits, rather than a failed search."""
        assert count(shakespeare_db, q, "phrase_ordered") == 0

    def test_unmatched_parenthesis(self, shakespeare_db):
        """Refused with its reason: it was searched as the word "(lord", and found nothing, silently."""
        with pytest.raises(BadRequest, match="unmatched parenthesis"):
            count(shakespeare_db, "(lord", "phrase_ordered")

    def test_invalid_regex_in_metadata(self, shakespeare_db):
        hits = shakespeare_db.query("", title=["[hamlet"])
        hits.finish()
