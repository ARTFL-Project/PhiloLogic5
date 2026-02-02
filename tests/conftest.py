"""Pytest configuration and shared fixtures for PhiloLogic5 tests."""

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

# Add PhiloLogic and tests to path
REPO_ROOT = Path(__file__).parent.parent
TESTS_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures.corpus_manager import CorpusManager
from tests.fixtures.corpus_configs import SHAKESPEARE, ELTEC


# =============================================================================
# Command Line Options
# =============================================================================


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--force-rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of test corpora even if cached",
    )
    parser.addoption(
        "--check-perf-regression",
        action="store_true",
        default=False,
        help="Check for performance regressions against baseline",
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no corpus needed)")
    config.addinivalue_line("markers", "integration: Integration tests (require corpus)")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    config.addinivalue_line("markers", "gold_set: Gold set validation tests")


# =============================================================================
# Corpus Manager Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def corpus_manager():
    """Session-scoped corpus manager."""
    return CorpusManager()


@pytest.fixture(scope="session")
def force_rebuild(request):
    """Whether to force rebuild corpora."""
    return request.config.getoption("--force-rebuild")


# =============================================================================
# Corpus Path Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def shakespeare_db_path(corpus_manager, force_rebuild):
    """Path to Shakespeare corpus (38 plays)."""
    return corpus_manager.get_corpus(SHAKESPEARE, force_rebuild=force_rebuild)


@pytest.fixture(scope="session")
def eltec_db_path(corpus_manager, force_rebuild):
    """Path to ELTeC corpus (100 novels)."""
    return corpus_manager.get_corpus(ELTEC, force_rebuild=force_rebuild)


# =============================================================================
# Database Connection Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def shakespeare_db(shakespeare_db_path):
    """Shakespeare database connection."""
    from philologic.runtime.DB import DB
    return DB(str(shakespeare_db_path))


@pytest.fixture(scope="module")
def eltec_db(eltec_db_path):
    """ELTeC database connection."""
    from philologic.runtime.DB import DB
    return DB(str(eltec_db_path))


# =============================================================================
# Hitlist Cache Cleanup
# =============================================================================


def clear_hitlists(db_path):
    """Clear the hitlists directory to ensure fresh query results."""
    hitlists_dir = Path(db_path) / "hitlists"
    if hitlists_dir.exists():
        shutil.rmtree(hitlists_dir)
        hitlists_dir.mkdir()


@pytest.fixture(autouse=True, scope="function")
def clear_query_cache(request, corpus_manager):
    """Automatically clear hitlists before each test function."""
    for corpus_info in corpus_manager.list_cached():
        corpus_path = Path(corpus_info["path"]) / "data"
        clear_hitlists(corpus_path)
    yield


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def timer():
    """Simple timing context manager for benchmarks."""
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
            self.start_time = 0.0

        def __enter__(self):
            self.start_time = time.perf_counter()
            return self

        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start_time

    return Timer


@pytest.fixture
def gold_sets_dir():
    """Path to gold sets directory."""
    return Path(__file__).parent / "gold_sets"


@pytest.fixture
def reports_dir():
    """Path to reports directory."""
    reports = Path(__file__).parent / "reports"
    reports.mkdir(exist_ok=True)
    return reports


# =============================================================================
# Test Report Hooks
# =============================================================================


def _print_benchmark_comparison(terminalreporter, config):
    """Print comparison of benchmark results vs baseline."""
    baseline_file = Path(__file__).parent / "benchmarks" / "baseline.json"
    if not baseline_file.exists():
        return

    # Check if benchmark plugin is active and has results
    if not hasattr(config, "_benchmarksession"):
        return

    benchmarks = getattr(config._benchmarksession, "benchmarks", [])
    if not benchmarks:
        return

    baseline = json.loads(baseline_file.read_text())

    terminalreporter.write_line("")
    terminalreporter.write_line("=" * 70, bold=True)
    terminalreporter.write_line("BENCHMARK COMPARISON vs BASELINE", bold=True)
    terminalreporter.write_line("=" * 70, bold=True)
    terminalreporter.write_line(f"{'Test':<30} {'Baseline':>12} {'Current':>12} {'Status':>10}")
    terminalreporter.write_line("-" * 70)

    for bench in benchmarks:
        name = bench.name.split("::")[-1]
        if name.startswith("test_"):
            name = name[5:]

        current_median = bench.stats.median * 1000  # to ms
        baseline_data = baseline.get(name, {})
        baseline_median = baseline_data.get("median_seconds", 0) * 1000  # to ms

        if baseline_median > 0:
            ratio = current_median / baseline_median
            if ratio > 2.0:
                status = "REGRESSION"
                color = "red"
            elif ratio < 0.8:
                status = "FASTER"
                color = "green"
            else:
                status = "OK"
                color = None

            line = f"{name:<30} {baseline_median:>10.2f}ms {current_median:>10.2f}ms {status:>10}"
            if color:
                terminalreporter.write_line(line, **{color: True})
            else:
                terminalreporter.write_line(line)
        else:
            terminalreporter.write_line(f"{name:<30} {'N/A':>12} {current_median:>10.2f}ms {'NEW':>10}")

    terminalreporter.write_line("-" * 70)
    terminalreporter.write_line("Threshold: 2x baseline = REGRESSION")
    terminalreporter.write_line("")


@pytest.hookimpl(hookwrapper=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate custom test report after test run."""
    yield

    # Print benchmark comparison if benchmarks were run
    _print_benchmark_comparison(terminalreporter, config)

    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    passed = len(terminalreporter.stats.get("passed", []))
    failed = len(terminalreporter.stats.get("failed", []))
    skipped = len(terminalreporter.stats.get("skipped", []))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": passed + failed + skipped,
        "exit_status": exitstatus,
    }

    report_file = reports_dir / f"report_{timestamp}.json"
    report_file.write_text(json.dumps(summary, indent=2))
