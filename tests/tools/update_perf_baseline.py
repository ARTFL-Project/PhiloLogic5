#!/usr/bin/env python3
"""
Update the performance baseline file.

This script runs the benchmark tests and captures the median times
to create/update the baseline.json file used for regression detection.

Usage:
    python tests/tools/update_perf_baseline.py

The baseline is stored in tests/benchmarks/baseline.json and should be
committed to the repository after verified performance improvements.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
BASELINE_FILE = REPO_ROOT / "tests" / "benchmarks" / "baseline.json"
BENCHMARK_OUTPUT = REPO_ROOT / ".benchmarks"


def run_benchmarks():
    """Run benchmark tests and save results."""
    print("Running benchmarks on ELTeC corpus (10 rounds per test with cache clearing)...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/benchmarks/",
            "-v",
            "--benchmark-only",
            "--benchmark-json=benchmark_results.json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return REPO_ROOT / "benchmark_results.json"


def parse_benchmark_results(results_file: Path) -> dict:
    """Parse pytest-benchmark JSON output and extract medians."""
    data = json.loads(results_file.read_text())

    baseline = {
        "metadata": {
            "generated_date": datetime.now().isoformat(),
            "machine_info": data.get("machine_info", {}),
            "commit_info": data.get("commit_info", {}),
        },
    }

    for benchmark in data.get("benchmarks", []):
        # Extract test name from full name (e.g., "test_single_term_common_word")
        full_name = benchmark["name"]
        # Remove the class prefix if present
        test_name = full_name.split("::")[-1]
        # Remove "test_" prefix for cleaner names
        if test_name.startswith("test_"):
            test_name = test_name[5:]

        stats = benchmark["stats"]
        baseline[test_name] = {
            "median_seconds": stats["median"],
            "mean_seconds": stats["mean"],
            "min_seconds": stats["min"],
            "max_seconds": stats["max"],
            "stddev_seconds": stats["stddev"],
            "rounds": stats["rounds"],
        }

    return baseline


def main():
    print("=" * 60)
    print("Updating Performance Baseline")
    print("=" * 60)

    # Run benchmarks
    results_file = run_benchmarks()

    # Parse results
    baseline = parse_benchmark_results(results_file)

    # Show summary
    print("\nBaseline Summary:")
    print("-" * 60)
    for name, data in baseline.items():
        if name == "metadata":
            continue
        median_ms = data["median_seconds"] * 1000
        print(f"  {name}: {median_ms:.2f}ms (median)")

    # Save baseline
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2))
    print(f"\nBaseline saved to {BASELINE_FILE}")

    # Cleanup
    results_file.unlink()
    print("\nDone! Commit baseline.json to save the new baseline.")


if __name__ == "__main__":
    main()
