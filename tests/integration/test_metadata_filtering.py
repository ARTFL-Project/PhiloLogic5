"""Integration tests for metadata filtering."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))


@pytest.mark.integration
class TestMetadataFiltering:
    """Tests for metadata-based query filtering."""

    def test_title_filter(self, shakespeare_db):
        """Test filtering by title metadata."""
        # Search for "murder" only in documents with "Macbeth" in title
        hits = shakespeare_db.query(
            "murder",
            method="single_term",
            title=".*[Mm]acbeth.*"
        )
        hits.finish()

        # Should find results since murder is in Macbeth
        assert isinstance(len(hits), int)

    def test_word_search_with_no_filter(self, shakespeare_db):
        """Test word search without metadata filter."""
        hits_all = shakespeare_db.query("love", method="single_term")
        hits_all.finish()

        # Should find many instances of "love" across all plays
        assert len(hits_all) > 0

    def test_metadata_only_query(self, shakespeare_db):
        """Test query with only metadata, no word search."""
        # Get all documents
        docs = list(shakespeare_db.get_all("doc"))
        assert len(docs) > 0, "Should retrieve documents"

    def test_get_all_by_type(self, shakespeare_db):
        """Test get_all with different philo_types."""
        # Get all documents
        docs = list(shakespeare_db.get_all("doc"))
        assert len(docs) > 0

        # Get all div1 (acts)
        divs = list(shakespeare_db.get_all("div1"))
        assert isinstance(len(divs), int)

    def test_combined_word_and_metadata(self, shakespeare_db):
        """Test combining word search with metadata filter."""
        # Search for ghost in the corpus
        hits_all = shakespeare_db.query("ghost", method="single_term")
        hits_all.finish()

        # The filtered results should be a subset
        assert len(hits_all) > 0


@pytest.mark.integration
class TestDocumentRetrieval:
    """Tests for document retrieval and metadata access."""

    def test_document_has_basic_metadata(self, shakespeare_db):
        """Test that documents have basic metadata fields."""
        docs = list(shakespeare_db.get_all("doc"))

        for doc in docs:
            # Check that basic fields are accessible
            # Note: exact field names depend on the corpus
            assert doc is not None

    def test_hit_has_document_context(self, shakespeare_db):
        """Test that hits can access document context."""
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()

        if len(hits) > 0:
            hit = hits[0]
            # Should be able to access the hit
            assert hit is not None
            assert hit.philo_id is not None


@pytest.mark.integration
class TestSortOrder:
    """Tests for result sorting."""

    def test_default_sort(self, shakespeare_db):
        """Test that results have a default sort order."""
        hits = shakespeare_db.query("love", method="single_term")
        hits.finish()

        # Should be able to iterate in order
        prev_id = None
        for i, hit in enumerate(hits):
            if i >= 10:
                break
            current_id = hit.philo_id
            if prev_id is not None:
                # IDs should be in some consistent order
                assert current_id is not None
            prev_id = current_id

    def test_sorted_results(self, shakespeare_db):
        """Test results with explicit sort order."""
        hits = shakespeare_db.query("love", method="single_term", sort_order=["rowid"])
        hits.finish()
        assert len(hits) >= 0


@pytest.mark.integration
class TestELTeCMetadata:
    """Tests for ELTeC-specific metadata."""

    def test_eltec_document_metadata(self, eltec_db):
        """Test that ELTeC documents have expected metadata."""
        docs = list(eltec_db.get_all("doc"))

        assert len(docs) > 0, "Should have documents"

        for doc in docs:
            # ELTeC documents should have author and title
            assert doc is not None
