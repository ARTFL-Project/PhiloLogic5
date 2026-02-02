"""Integration tests for the full loader pipeline."""

import sqlite3
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))


@pytest.mark.integration
class TestShakespeareCorpusBuild:
    """Tests for Shakespeare corpus building (38 plays)."""

    def test_corpus_builds_successfully(self, shakespeare_db_path):
        """Test that corpus builds without errors."""
        db_path = Path(shakespeare_db_path)
        assert db_path.exists(), "Database directory should exist"
        assert (db_path / "toms.db").exists(), "toms.db should exist"
        assert (db_path / "words.lmdb").exists(), "words.lmdb should exist"

    def test_sql_tables_created(self, shakespeare_db_path):
        """Test that all required SQL tables exist."""
        db_path = Path(shakespeare_db_path) / "toms.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert "toms" in tables, "toms table should exist"
        conn.close()

    def test_document_count(self, shakespeare_db_path):
        """Test that all 38 plays were loaded."""
        db_path = Path(shakespeare_db_path) / "toms.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM toms WHERE philo_type='doc'")
        doc_count = cursor.fetchone()[0]

        assert doc_count == 38, f"Expected 38 documents, got {doc_count}"
        conn.close()

    def test_metadata_extracted(self, shakespeare_db):
        """Test that document metadata was properly extracted."""
        docs = list(shakespeare_db.get_all("doc"))

        assert len(docs) == 38, "Should have 38 documents"

        for doc in docs:
            assert doc["author"] is not None or doc["title"] is not None

    def test_word_index_has_content(self, shakespeare_db_path):
        """Test that word index has reasonable content."""
        import lmdb

        lmdb_path = Path(shakespeare_db_path) / "words.lmdb"
        env = lmdb.open(str(lmdb_path), readonly=True)

        with env.begin() as txn:
            common_words = [b"the", b"and", b"to", b"of", b"a"]
            found_count = sum(1 for w in common_words if txn.get(w) is not None)
            assert found_count == len(common_words), "Common words should be indexed"

        env.close()

    def test_lmdb_index_valid(self, shakespeare_db_path):
        """Test that LMDB word index is valid and queryable."""
        import lmdb

        lmdb_path = Path(shakespeare_db_path) / "words.lmdb"
        env = lmdb.open(str(lmdb_path), readonly=True)

        with env.begin() as txn:
            assert txn.get(b"the") is not None, "'the' should be indexed"
            assert txn.get(b"hamlet") is not None, "'hamlet' should be indexed"

        env.close()


@pytest.mark.integration
class TestELTeCCorpusBuild:
    """Tests for ELTeC corpus building (100 novels)."""

    @pytest.mark.timeout(900)  # 15 minutes for 100 novels
    def test_corpus_builds_successfully(self, eltec_db_path):
        """Test that ELTeC corpus builds without errors."""
        db_path = Path(eltec_db_path)
        assert db_path.exists(), "Database directory should exist"
        assert (db_path / "toms.db").exists(), "toms.db should exist"
        assert (db_path / "words.lmdb").exists(), "words.lmdb should exist"

    def test_document_count(self, eltec_db_path):
        """Test that all 100 novels were loaded."""
        db_path = Path(eltec_db_path) / "toms.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM toms WHERE philo_type='doc'")
        doc_count = cursor.fetchone()[0]

        assert doc_count == 100, f"Expected 100 documents, got {doc_count}"
        conn.close()

    def test_metadata_extracted(self, eltec_db):
        """Test that document metadata was properly extracted."""
        docs = list(eltec_db.get_all("doc"))

        assert len(docs) == 100, "Should have 100 documents"

        for doc in docs:
            assert doc["author"] is not None or doc["title"] is not None


@pytest.mark.integration
class TestCorpusStructure:
    """Tests for corpus structural integrity."""

    def test_div_hierarchy(self, shakespeare_db):
        """Test that div hierarchy is properly created."""
        div1s = list(shakespeare_db.get_all("div1"))
        assert len(div1s) > 0, "Should have div1 elements"

    def test_paragraph_objects(self, shakespeare_db):
        """Test that paragraph objects exist."""
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()

        if len(hits) > 0:
            hit = hits[0]
            philo_id = hit.philo_id
            assert len(philo_id) >= 5, "philo_id should have para component"
