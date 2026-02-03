"""Performance benchmark tests.

These tests measure query performance using pytest-benchmark for statistical accuracy.
Each benchmark iteration runs with a fresh (uncached) hitlist to measure actual query time.

Usage:
    # Run benchmarks and show results
    pytest tests/benchmarks/ -v

    # Save current results as new baseline
    pytest tests/benchmarks/ --benchmark-save=baseline

    # Compare against saved baseline (shows colored diff)
    pytest tests/benchmarks/ --benchmark-compare=baseline

    # Check for regressions against committed baseline
    pytest tests/benchmarks/ --check-perf-regression

    # Update the committed baseline after verified improvements
    python tests/tools/update_perf_baseline.py

Workflow for performance improvement:
    1. Run: pytest tests/benchmarks/ --benchmark-save=before
    2. Make your changes
    3. Run: pytest tests/benchmarks/ --benchmark-compare=before
    4. If improved, update baseline: python tests/tools/update_perf_baseline.py
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

BASELINE_FILE = Path(__file__).parent / "baseline.json"

# Regression threshold: fail if current median > baseline_median * this factor
REGRESSION_THRESHOLD = 2.0

# Benchmark settings
BENCHMARK_ROUNDS = 10
BENCHMARK_WARMUP_ROUNDS = 5


def clear_hitlists(db_path):
    """Clear the hitlists directory to ensure fresh query results."""
    hitlists_dir = Path(db_path) / "hitlists"
    if hitlists_dir.exists():
        shutil.rmtree(hitlists_dir)
        hitlists_dir.mkdir()


def load_baseline() -> dict:
    """Load baseline performance data."""
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {}


def pytest_addoption_benchmark(parser):
    """Add custom benchmark options."""
    parser.addoption(
        "--check-perf-regression",
        action="store_true",
        default=False,
        help="Check for performance regressions against baseline",
    )


@pytest.mark.benchmark
@pytest.mark.integration
class TestQueryPerformance:
    """Performance benchmarks for query operations.

    Uses pytest-benchmark pedantic mode to clear hitlist cache before each
    timed iteration, ensuring we measure actual query execution time.
    """

    @pytest.fixture(scope="class")
    def db(self, eltec_db_path):
        """Database fixture for benchmarks (uses larger ELTeC corpus)."""
        from philologic.runtime.DB import DB
        return DB(str(eltec_db_path))

    @pytest.fixture(scope="class")
    def baseline(self):
        """Load baseline performance data."""
        return load_baseline()

    def _check_regression(self, benchmark, baseline: dict, test_name: str):
        """Check if current performance is a regression from baseline."""
        if not baseline or test_name not in baseline:
            return  # No baseline to compare against

        baseline_median = baseline[test_name].get("median_seconds")
        if baseline_median is None:
            return

        # Get current median from benchmark stats
        current_median = benchmark.stats.stats.median
        threshold = baseline_median * REGRESSION_THRESHOLD

        if current_median > threshold:
            pytest.fail(
                f"Performance regression detected!\n"
                f"  Test: {test_name}\n"
                f"  Baseline median: {baseline_median*1000:.2f}ms\n"
                f"  Current median:  {current_median*1000:.2f}ms\n"
                f"  Threshold ({REGRESSION_THRESHOLD}x): {threshold*1000:.2f}ms"
            )

    def test_single_term_common_word(self, db, benchmark, baseline, request):
        """Benchmark single term search for common word 'the'."""
        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query("the", method="single_term")
            hits.finish()
            return len(hits)

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result > 0, "Should find hits for 'the'"
        if request.config.getoption("--check-perf-regression", default=False):
            self._check_regression(benchmark, baseline, "single_term_common_word")

    def test_single_term_rare_word(self, db, benchmark, baseline, request):
        """Benchmark single term search for character name 'hamlet'."""
        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query("hamlet", method="single_term")
            hits.finish()
            return len(hits)

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result > 0, "Should find hits for 'hamlet'"
        if request.config.getoption("--check-perf-regression", default=False):
            self._check_regression(benchmark, baseline, "single_term_rare_word")

    def test_phrase_search(self, db, benchmark, baseline, request):
        """Benchmark phrase search for 'my lord'."""
        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query('"my lord"', method="phrase_ordered")
            hits.finish()
            return len(hits)

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result > 0, "Should find hits for phrase"
        if request.config.getoption("--check-perf-regression", default=False):
            self._check_regression(benchmark, baseline, "phrase_search")

    def test_proximity_search(self, db, benchmark, baseline, request):
        """Benchmark proximity search for 'love' and 'death' within 50 words."""
        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query("love death", method="proxy_unordered", method_arg="50")
            hits.finish()
            return len(hits)

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result >= 0, "Query should complete"
        if request.config.getoption("--check-perf-regression", default=False):
            self._check_regression(benchmark, baseline, "proximity_search")

    def test_sentence_search(self, db, benchmark, baseline, request):
        """Benchmark sentence-level search for 'to be'."""
        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query("to be", method="sentence_ordered")
            hits.finish()
            return len(hits)

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result > 0, "Should find hits for 'to be'"
        if request.config.getoption("--check-perf-regression", default=False):
            self._check_regression(benchmark, baseline, "sentence_search")

    def test_or_search(self, db, benchmark, baseline, request):
        """Benchmark OR search for multiple character names."""
        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query("hamlet | macbeth | othello", method="single_term")
            hits.finish()
            return len(hits)

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result > 0, "Should find hits for OR query"
        if request.config.getoption("--check-perf-regression", default=False):
            self._check_regression(benchmark, baseline, "or_search")


@pytest.mark.benchmark
@pytest.mark.integration
class TestIterationPerformance:
    """Benchmarks for result iteration.

    These tests measure query + iteration performance.
    Each benchmark clears the hitlist cache before each timed iteration.
    """

    @pytest.fixture(scope="class")
    def db(self, eltec_db_path):
        """Database fixture (uses larger ELTeC corpus)."""
        from philologic.runtime.DB import DB
        return DB(str(eltec_db_path))

    def test_iterate_first_100(self, db, benchmark):
        """Benchmark query + iterating over first 100 results."""
        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query("the", method="single_term")
            hits.finish()
            count = 0
            for i, hit in enumerate(hits):
                _ = hit.philo_id
                count += 1
                if i >= 99:
                    break
            return count

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result == 100

    def test_random_access(self, db, benchmark):
        """Benchmark query + random access to hits."""
        indices = [0, 10, 50, 99]

        def setup():
            clear_hitlists(db.path)

        def run():
            hits = db.query("the", method="single_term")
            hits.finish()
            accessed = 0
            for i in indices:
                if i < len(hits):
                    _ = hits[i]
                    accessed += 1
            return accessed

        result = benchmark.pedantic(
            run, setup=setup, rounds=BENCHMARK_ROUNDS, warmup_rounds=BENCHMARK_WARMUP_ROUNDS
        )
        assert result == 4
