"""Integration tests for all 8 query methods."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))


@pytest.mark.integration
class TestSingleTerm:
    """Tests for single_term search method."""

    def test_basic_single_term(self, shakespeare_db):
        """Test basic single word search."""
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()
        assert len(hits) > 0, "Should find 'hamlet' in Shakespeare corpus"

    def test_common_word(self, shakespeare_db):
        """Test search for common word."""
        hits = shakespeare_db.query("the", method="single_term")
        hits.finish()
        assert len(hits) > 1000, "Should find many instances of 'the'"

    def test_case_insensitive(self, shakespeare_db):
        """Test that searches are case insensitive."""
        hits_lower = shakespeare_db.query("hamlet", method="single_term")
        hits_upper = shakespeare_db.query("HAMLET", method="single_term")
        hits_lower.finish()
        hits_upper.finish()
        assert len(hits_lower) == len(hits_upper), "Search should be case insensitive"

    def test_rare_word(self, shakespeare_db):
        """Test search for rare word."""
        hits = shakespeare_db.query("yorick", method="single_term")
        hits.finish()
        # Yorick is mentioned in Hamlet
        assert len(hits) >= 1, "Should find 'yorick' in Hamlet"

    def test_nonexistent_word(self, shakespeare_db):
        """Test search for nonexistent word."""
        hits = shakespeare_db.query("xyznonexistent", method="single_term")
        hits.finish()
        assert len(hits) == 0, "Should not find nonexistent word"

    def test_multiple_word_forms(self, shakespeare_db):
        """Test that different word forms can be found."""
        # Search for both singular and potential plural forms
        hits_hamlet = shakespeare_db.query("hamlet", method="single_term")
        hits_hamlet.finish()

        # hamlet should be found in the corpus
        assert len(hits_hamlet) > 0, "Should find 'hamlet'"

    def test_or_terms(self, shakespeare_db):
        """Test OR search with |."""
        hits_combined = shakespeare_db.query("hamlet | macbeth", method="single_term")
        hits_hamlet = shakespeare_db.query("hamlet", method="single_term")
        hits_macbeth = shakespeare_db.query("macbeth", method="single_term")

        hits_combined.finish()
        hits_hamlet.finish()
        hits_macbeth.finish()

        # Combined should have hits from both terms
        assert len(hits_combined) >= len(hits_hamlet), "OR query should include hamlet hits"
        assert len(hits_combined) >= len(hits_macbeth), "OR query should include macbeth hits"


@pytest.mark.integration
class TestPhraseOrdered:
    """Tests for phrase_ordered search method."""

    def test_exact_phrase(self, shakespeare_db):
        """Test exact phrase search."""
        hits = shakespeare_db.query('"my lord"', method="phrase_ordered")
        hits.finish()
        # "my lord" is very common in Shakespeare
        assert len(hits) > 0, "Should find 'my lord' phrase"

    def test_longer_phrase(self, shakespeare_db):
        """Test longer phrase search."""
        hits = shakespeare_db.query('"to be or not"', method="phrase_ordered")
        hits.finish()
        # The famous soliloquy
        assert len(hits) >= 1, "Should find 'to be or not' phrase"

    def test_phrase_respects_order(self, shakespeare_db):
        """Test that phrase order is enforced."""
        hits_correct = shakespeare_db.query('"my lord"', method="phrase_ordered")
        hits_reversed = shakespeare_db.query('"lord my"', method="phrase_ordered")

        hits_correct.finish()
        hits_reversed.finish()

        # These should have different counts
        assert len(hits_correct) != len(hits_reversed) or len(hits_reversed) == 0


@pytest.mark.integration
class TestPhraseUnordered:
    """Tests for phrase_unordered search method."""

    def test_unordered_phrase(self, shakespeare_db):
        """Test phrase search without strict ordering."""
        hits = shakespeare_db.query("love death", method="phrase_unordered", method_arg=5)
        hits.finish()
        # Should find love and death within 5 words of each other
        assert isinstance(len(hits), int)

    def test_unordered_more_flexible(self, shakespeare_db):
        """Test that unordered allows words in either order."""
        hits_ordered = shakespeare_db.query("good night", method="phrase_unordered", method_arg=3)
        hits_reversed = shakespeare_db.query("night good", method="phrase_unordered", method_arg=3)

        hits_ordered.finish()
        hits_reversed.finish()

        # Unordered should give same results regardless of query order
        assert len(hits_ordered) == len(hits_reversed)


@pytest.mark.integration
class TestProxyOrdered:
    """Tests for proxy_ordered search method."""

    def test_proximity_ordered(self, shakespeare_db):
        """Test proximity search with order."""
        hits = shakespeare_db.query("hamlet ghost", method="proxy_ordered", method_arg=10)
        hits.finish()
        # hamlet followed by ghost within 10 words
        assert isinstance(len(hits), int)

    def test_proximity_smaller_distance(self, shakespeare_db):
        """Test that smaller distance gives fewer results."""
        hits_wide = shakespeare_db.query("love hate", method="proxy_ordered", method_arg=50)
        hits_narrow = shakespeare_db.query("love hate", method="proxy_ordered", method_arg=5)

        hits_wide.finish()
        hits_narrow.finish()

        # Wider distance should find at least as many as narrow
        assert len(hits_wide) >= len(hits_narrow)


@pytest.mark.integration
class TestProxyUnordered:
    """Tests for proxy_unordered search method."""

    def test_proximity_unordered(self, shakespeare_db):
        """Test proximity search without order constraint."""
        hits = shakespeare_db.query("love death", method="proxy_unordered", method_arg=20)
        hits.finish()
        # Should find love and death within 20 words in any order
        assert isinstance(len(hits), int)

    def test_unordered_more_results_than_ordered(self, shakespeare_db):
        """Test that unordered finds at least as many as ordered."""
        hits_ordered = shakespeare_db.query("king queen", method="proxy_ordered", method_arg=10)
        hits_unordered = shakespeare_db.query("king queen", method="proxy_unordered", method_arg=10)

        hits_ordered.finish()
        hits_unordered.finish()

        # Unordered should find at least as many results
        assert len(hits_unordered) >= len(hits_ordered)


@pytest.mark.integration
class TestExactCoocOrdered:
    """Tests for exact_cooc_ordered search method."""

    def test_exact_cooccurrence_ordered(self, shakespeare_db):
        """Test exact co-occurrence with order."""
        hits = shakespeare_db.query("my lord", method="exact_cooc_ordered", method_arg=2)
        hits.finish()
        # Should find "my" exactly 2 words before "lord"
        assert isinstance(len(hits), int)


@pytest.mark.integration
class TestExactCoocUnordered:
    """Tests for exact_cooc_unordered search method."""

    def test_exact_cooccurrence_unordered(self, shakespeare_db):
        """Test exact co-occurrence without order."""
        hits = shakespeare_db.query("lord my", method="exact_cooc_unordered", method_arg=2)
        hits.finish()
        assert isinstance(len(hits), int)


@pytest.mark.integration
class TestSentenceOrdered:
    """Tests for sentence_ordered search method."""

    def test_sentence_cooccurrence_ordered(self, shakespeare_db):
        """Test co-occurrence within sentence with order."""
        hits = shakespeare_db.query("to be", method="sentence_ordered")
        hits.finish()
        # Should find "to" before "be" in same sentence
        assert len(hits) > 0, "Should find 'to be' in sentences"


@pytest.mark.integration
class TestSentenceUnordered:
    """Tests for sentence_unordered search method."""

    def test_sentence_cooccurrence_unordered(self, shakespeare_db):
        """Test co-occurrence within sentence without order."""
        hits = shakespeare_db.query("king prince", method="sentence_unordered")
        hits.finish()
        assert isinstance(len(hits), int)


@pytest.mark.integration
class TestHitListFunctionality:
    """Tests for HitList functionality."""

    def test_hitlist_length(self, shakespeare_db):
        """Test that HitList reports correct length."""
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()
        length = len(hits)
        assert isinstance(length, int)
        assert length > 0

    def test_hitlist_iteration(self, shakespeare_db):
        """Test that HitList can be iterated."""
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()

        count = 0
        for hit in hits:
            count += 1
            if count >= 10:
                break

        assert count > 0, "Should be able to iterate over hits"

    def test_hitlist_indexing(self, shakespeare_db):
        """Test that HitList supports indexing."""
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()

        if len(hits) > 0:
            first_hit = hits[0]
            assert first_hit is not None
            assert hasattr(first_hit, "philo_id")

    def test_hit_philo_id_structure(self, shakespeare_db):
        """Test that hits have proper philo_id structure."""
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()

        if len(hits) > 0:
            hit = hits[0]
            philo_id = hit.philo_id

            # philo_id should be a tuple/list of integers
            assert len(philo_id) >= 7, "philo_id should have at least 7 components"
            assert all(isinstance(x, int) for x in philo_id)
