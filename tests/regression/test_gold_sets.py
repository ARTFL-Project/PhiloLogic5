"""Gold set validation tests.

These tests validate query results against known-good gold sets.
Run after generating gold sets with: python tests/tools/generate_gold_sets.py
"""

import json
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

GOLD_SETS_DIR = Path(__file__).parent.parent / "gold_sets"


def load_gold_set(corpus: str, query_type: str) -> dict:
    """Load a gold set file.

    Args:
        corpus: Corpus name (shakespeare, eltec)
        query_type: Query type (single_term, phrase, proximity, etc.)

    Returns:
        Gold set dictionary or None if file doesn't exist
    """
    path = GOLD_SETS_DIR / corpus / f"{query_type}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_available_gold_sets(corpus: str) -> list:
    """Get list of available gold set files for a corpus.

    Args:
        corpus: Corpus name

    Returns:
        List of query types with available gold sets
    """
    corpus_dir = GOLD_SETS_DIR / corpus
    if not corpus_dir.exists():
        return []
    return [f.stem for f in corpus_dir.glob("*.json")]


@pytest.mark.gold_set
@pytest.mark.integration
class TestShakespeareGoldSets:
    """Validate queries against Shakespeare gold sets."""

    @pytest.fixture(scope="class")
    def db(self, shakespeare_db_path):
        """Database fixture for gold set tests."""
        from philologic.runtime.DB import DB
        return DB(str(shakespeare_db_path))

    def test_gold_sets_exist(self):
        """Verify that at least some gold sets exist."""
        available = get_available_gold_sets("shakespeare")
        # This test will skip if no gold sets have been generated
        if not available:
            pytest.skip("No Shakespeare gold sets found. Run generate_gold_sets.py first.")

    @pytest.mark.parametrize("query_type", ["single_term", "phrase", "proximity", "sentence"])
    def test_gold_set_validation(self, db, query_type):
        """Validate query results against gold set."""
        gold_set = load_gold_set("shakespeare", query_type)

        if gold_set is None:
            pytest.skip(f"Gold set for {query_type} not found")

        for test_case in gold_set["test_cases"]:
            query = test_case["query"]
            expected = test_case["expected"]

            hits = db.query(
                qs=query["qs"],
                method=query["method"],
                method_arg=query.get("method_arg", ""),
            )
            hits.finish()

            # Validate hit count
            assert len(hits) == expected["total_hits"], (
                f"Test {test_case['id']}: expected {expected['total_hits']} hits, "
                f"got {len(hits)}"
            )

            # Validate first N philo_ids if present
            if expected.get("first_10_philo_ids"):
                actual_first = [list(h.philo_id) for h in list(hits)[:10]]
                assert actual_first == expected["first_10_philo_ids"], (
                    f"Test {test_case['id']}: philo_ids mismatch"
                )


@pytest.mark.gold_set
@pytest.mark.integration
class TestELTeCGoldSets:
    """Validate queries against ELTeC gold sets."""

    @pytest.fixture(scope="class")
    def db(self, eltec_db_path):
        """Database fixture for gold set tests."""
        from philologic.runtime.DB import DB
        return DB(str(eltec_db_path))

    def test_gold_sets_exist(self):
        """Verify that at least some gold sets exist."""
        available = get_available_gold_sets("eltec")
        if not available:
            pytest.skip("No ELTeC gold sets found. Run generate_gold_sets.py first.")

    @pytest.mark.parametrize("query_type", ["single_term", "phrase", "proximity"])
    def test_gold_set_validation(self, db, query_type):
        """Validate query results against gold set."""
        gold_set = load_gold_set("eltec", query_type)

        if gold_set is None:
            pytest.skip(f"Gold set for {query_type} not found")

        for test_case in gold_set["test_cases"]:
            query = test_case["query"]
            expected = test_case["expected"]

            hits = db.query(
                qs=query["qs"],
                method=query["method"],
                method_arg=query.get("method_arg", ""),
            )
            hits.finish()

            # Validate hit count
            assert len(hits) == expected["total_hits"], (
                f"Test {test_case['id']}: expected {expected['total_hits']} hits, "
                f"got {len(hits)}"
            )


@pytest.mark.gold_set
class TestGoldSetIntegrity:
    """Tests for gold set file integrity."""

    def test_gold_set_schema(self):
        """Test that gold set files have correct schema."""
        for corpus in ["shakespeare", "eltec"]:
            available = get_available_gold_sets(corpus)
            for query_type in available:
                gold_set = load_gold_set(corpus, query_type)

                # Check required top-level keys
                assert "metadata" in gold_set, f"{corpus}/{query_type}: missing metadata"
                assert "test_cases" in gold_set, f"{corpus}/{query_type}: missing test_cases"

                # Check metadata structure
                metadata = gold_set["metadata"]
                assert "corpus" in metadata
                assert "generated_date" in metadata

                # Check test case structure
                for tc in gold_set["test_cases"]:
                    assert "id" in tc, f"{corpus}/{query_type}: test case missing id"
                    assert "query" in tc, f"{corpus}/{query_type}: test case missing query"
                    assert "expected" in tc, f"{corpus}/{query_type}: test case missing expected"

                    # Check query structure
                    query = tc["query"]
                    assert "qs" in query
                    assert "method" in query

                    # Check expected structure
                    expected = tc["expected"]
                    assert "total_hits" in expected
