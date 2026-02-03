"""Gold set validation tests.

These tests validate query results against known-good gold sets by comparing
the actual hitlist binary files byte-by-byte with stored reference files.

Run after generating gold sets with: python tests/tools/generate_gold_sets.py --all
"""

import filecmp
import json
import shutil
import struct
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

GOLD_SETS_DIR = Path(__file__).parent.parent / "gold_sets"

# Hitlist record size: 9 unsigned ints (7 philo_id + 2 word info) * 4 bytes
# This is for single-word queries; multi-word queries have larger records
DEFAULT_HIT_SIZE = 36  # 9 * 4 bytes


def load_gold_set_metadata(corpus: str, query_type: str) -> dict:
    """Load gold set metadata file.

    Args:
        corpus: Corpus name (shakespeare, eltec)
        query_type: Query type (single_term, phrase, proximity, etc.)

    Returns:
        Metadata dictionary or None if file doesn't exist
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


def clear_hitlists(db_path: Path):
    """Clear the hitlists directory."""
    hitlists_dir = db_path / "hitlists"
    if hitlists_dir.exists():
        shutil.rmtree(hitlists_dir)
        hitlists_dir.mkdir()


def compare_hitlists(gold_file: Path, test_file: Path, hit_size: int = DEFAULT_HIT_SIZE) -> tuple:
    """Compare two hitlist files.

    Fast path: uses filecmp for byte-by-byte comparison.
    On mismatch: finds the exact hit that differs.

    Args:
        gold_file: Path to the gold reference hitlist
        test_file: Path to the test hitlist
        hit_size: Size of each hit record in bytes

    Returns:
        Tuple of (matches: bool, error_info: dict or None)
        If matches is False, error_info contains details about the mismatch.
    """
    # Check file sizes first
    gold_size = gold_file.stat().st_size
    test_size = test_file.stat().st_size

    if gold_size != test_size:
        return False, {
            "error": "size_mismatch",
            "gold_size": gold_size,
            "test_size": test_size,
            "gold_hits": gold_size // hit_size,
            "test_hits": test_size // hit_size,
        }

    # Fast path: are they identical?
    if filecmp.cmp(gold_file, test_file, shallow=False):
        return True, None

    # Files differ - find where
    with open(gold_file, 'rb') as f1, open(test_file, 'rb') as f2:
        hit_num = 0
        while True:
            chunk1 = f1.read(hit_size)
            chunk2 = f2.read(hit_size)

            if not chunk1:
                break

            if chunk1 != chunk2:
                # Unpack to show the actual difference
                # Format: 9 unsigned ints (assuming single-word query)
                fmt = f'{hit_size // 4}I'
                gold_hit = struct.unpack(fmt, chunk1)
                test_hit = struct.unpack(fmt, chunk2)
                return False, {
                    "error": "content_mismatch",
                    "hit_num": hit_num,
                    "gold_philo_id": gold_hit[:7],
                    "test_philo_id": test_hit[:7],
                }

            hit_num += 1

    # Shouldn't reach here if filecmp said they differ
    return True, None


def format_mismatch_error(test_id: str, error_info: dict) -> str:
    """Format a detailed error message for a hitlist mismatch."""
    if error_info["error"] == "size_mismatch":
        return (
            f"Test {test_id}: hitlist size mismatch\n"
            f"  Gold: {error_info['gold_hits']} hits ({error_info['gold_size']} bytes)\n"
            f"  Test: {error_info['test_hits']} hits ({error_info['test_size']} bytes)"
        )
    elif error_info["error"] == "content_mismatch":
        return (
            f"Test {test_id}: hitlist content mismatch at hit #{error_info['hit_num']}\n"
            f"  Gold philo_id: {error_info['gold_philo_id']}\n"
            f"  Test philo_id: {error_info['test_philo_id']}"
        )
    else:
        return f"Test {test_id}: {error_info}"


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
        if not available:
            pytest.skip("No Shakespeare gold sets found. Run generate_gold_sets.py first.")

    @pytest.mark.parametrize("query_type", ["single_term", "phrase", "proximity", "sentence"])
    def test_gold_set_validation(self, db, query_type):
        """Validate query results against gold set by comparing hitlist files."""
        metadata = load_gold_set_metadata("shakespeare", query_type)

        if metadata is None:
            pytest.skip(f"Gold set for {query_type} not found")

        errors = []

        for test_case in metadata["test_cases"]:
            query = test_case["query"]
            expected = test_case["expected"]
            test_id = test_case["id"]

            # Clear hitlists before query
            clear_hitlists(Path(db.path))

            # Run the query
            hits = db.query(
                qs=query["qs"],
                method=query["method"],
                method_arg=query.get("method_arg", ""),
            )
            hits.finish()

            # Quick check: hit count
            if len(hits) != expected["total_hits"]:
                errors.append(
                    f"Test {test_id}: expected {expected['total_hits']} hits, got {len(hits)}"
                )
                continue

            # Full validation: compare hitlist files
            gold_file = GOLD_SETS_DIR / "shakespeare" / expected["hitlist_file"]
            test_file = Path(hits.filename)

            if not gold_file.exists():
                errors.append(f"Test {test_id}: gold hitlist file not found: {gold_file}")
                continue

            matches, error_info = compare_hitlists(gold_file, test_file)
            if not matches:
                errors.append(format_mismatch_error(test_id, error_info))

        if errors:
            pytest.fail("\n".join(errors))


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
        """Validate query results against gold set by comparing hitlist files."""
        metadata = load_gold_set_metadata("eltec", query_type)

        if metadata is None:
            pytest.skip(f"Gold set for {query_type} not found")

        errors = []

        for test_case in metadata["test_cases"]:
            query = test_case["query"]
            expected = test_case["expected"]
            test_id = test_case["id"]

            # Clear hitlists before query
            clear_hitlists(Path(db.path))

            # Run the query
            hits = db.query(
                qs=query["qs"],
                method=query["method"],
                method_arg=query.get("method_arg", ""),
            )
            hits.finish()

            # Quick check: hit count
            if len(hits) != expected["total_hits"]:
                errors.append(
                    f"Test {test_id}: expected {expected['total_hits']} hits, got {len(hits)}"
                )
                continue

            # Full validation: compare hitlist files
            gold_file = GOLD_SETS_DIR / "eltec" / expected["hitlist_file"]
            test_file = Path(hits.filename)

            if not gold_file.exists():
                errors.append(f"Test {test_id}: gold hitlist file not found: {gold_file}")
                continue

            matches, error_info = compare_hitlists(gold_file, test_file)
            if not matches:
                errors.append(format_mismatch_error(test_id, error_info))

        if errors:
            pytest.fail("\n".join(errors))


@pytest.mark.gold_set
class TestGoldSetIntegrity:
    """Tests for gold set file integrity."""

    def test_gold_set_schema(self):
        """Test that gold set metadata files have correct schema."""
        for corpus in ["shakespeare", "eltec"]:
            available = get_available_gold_sets(corpus)
            for query_type in available:
                metadata = load_gold_set_metadata(corpus, query_type)

                # Check required top-level keys
                assert "corpus" in metadata, f"{corpus}/{query_type}: missing corpus"
                assert "test_cases" in metadata, f"{corpus}/{query_type}: missing test_cases"

                # Check test case structure
                for tc in metadata["test_cases"]:
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
                    assert "hitlist_file" in expected

    def test_hitlist_files_exist(self):
        """Test that all referenced hitlist files exist."""
        for corpus in ["shakespeare", "eltec"]:
            available = get_available_gold_sets(corpus)
            for query_type in available:
                metadata = load_gold_set_metadata(corpus, query_type)

                for tc in metadata["test_cases"]:
                    hitlist_path = GOLD_SETS_DIR / corpus / tc["expected"]["hitlist_file"]
                    assert hitlist_path.exists(), (
                        f"Missing hitlist file: {hitlist_path}"
                    )

                    # Verify size matches metadata
                    expected_size = tc["expected"].get("hitlist_size_bytes")
                    if expected_size:
                        actual_size = hitlist_path.stat().st_size
                        assert actual_size == expected_size, (
                            f"Hitlist size mismatch for {tc['id']}: "
                            f"expected {expected_size}, got {actual_size}"
                        )
