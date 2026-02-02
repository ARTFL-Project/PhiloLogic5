# PhiloLogic5 Test Suite

This directory contains the comprehensive test suite for PhiloLogic5, covering the Parser/Loader pipeline and Query runtime.

## Quick Start

```bash
# Install test dependencies
cd python && pip install -e ".[test]"

# Run all tests
pytest tests/ -v

# Run only unit tests (fast, no corpus build required)
pytest tests/unit/ -v
```

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── pytest.ini               # Pytest settings
├── collections/             # Test corpora (XML files)
│   ├── ELTeC-eng/          # 100 English novels
│   └── folger-shakespeare/ # 38 Shakespeare plays
├── fixtures/                # Test infrastructure
│   ├── corpus_manager.py   # On-demand corpus building with caching
│   └── corpus_configs.py   # Corpus configurations
├── load_configs/            # Corpus-specific load configurations
│   ├── shakespeare_config.py
│   └── eltec_config.py
├── unit/                    # Unit tests (no corpus required)
│   ├── test_parser.py
│   └── test_query_syntax.py
├── integration/             # Integration tests (require corpus)
│   ├── test_loader_pipeline.py
│   ├── test_query_methods.py
│   └── test_metadata_filtering.py
├── regression/              # Regression tests
│   └── test_gold_sets.py
├── benchmarks/              # Performance tests
│   ├── test_performance.py
│   └── baseline.json
├── gold_sets/               # Expected query results
│   ├── shakespeare/
│   └── eltec/
└── tools/                   # Test utilities
    ├── generate_gold_sets.py
    └── update_perf_baseline.py
```

## Running Tests

### All Tests

```bash
pytest tests/ -v
```

### By Category

```bash
# Unit tests only (fast, no corpus)
pytest tests/unit/ -v

# Integration tests (builds corpus on first run)
pytest tests/integration/ -v

# Gold set regression tests
pytest tests/regression/ -v

# Performance benchmarks
pytest tests/benchmarks/ -v
```

### Using Markers

```bash
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m benchmark      # Benchmark tests only
pytest -m gold_set       # Gold set validation only
```

## Test Corpora

Tests use full corpora for comprehensive coverage:

| Corpus | Documents | Description |
|--------|-----------|-------------|
| Shakespeare | 38 plays | Folger Shakespeare Library TEI |
| ELTeC | 100 novels | European Literary Text Collection |

### Corpus Caching

Corpora are built once and cached in the PhiloLogic database location
(defined in `/etc/philologic/philologic5.cfg`) under `test_dbs/`.

The cache is automatically invalidated when:
- Source XML files change
- Parser.py or Loader.py change
- Load config files change

### Manual Cache Management

```bash
# Force rebuild on next test run
pytest tests/integration/ --force-rebuild -v

# Clear all cached test corpora
rm -rf /var/www/html/philologic5/test_dbs/
```

## Gold Sets (Regression Testing)

Gold sets store expected query results for detecting regressions.

### Regenerating Gold Sets

After verifying PhiloLogic is working correctly:

```bash
# Regenerate all gold sets (auto-detects database paths)
python tests/tools/generate_gold_sets.py --all

# Regenerate for specific corpus
python tests/tools/generate_gold_sets.py --corpus shakespeare

# With explicit database path
python tests/tools/generate_gold_sets.py --corpus shakespeare --db-path /path/to/data
```

### Gold Set Format

```json
{
  "metadata": {
    "corpus": "shakespeare",
    "query_type": "single_term",
    "generated_date": "2026-02-02T10:18:20"
  },
  "test_cases": [
    {
      "id": "st_hamlet",
      "description": "Search for 'hamlet'",
      "query": {"qs": "hamlet", "method": "single_term"},
      "expected": {
        "total_hits": 565,
        "first_10_philo_ids": [[1, 1, 2, 1, 1, 5, 0], ...]
      }
    }
  ]
}
```

## Performance Benchmarks

### Running Benchmarks

```bash
# Basic run with statistics
pytest tests/benchmarks/ -v

# Save results for comparison
pytest tests/benchmarks/ --benchmark-save=before

# Compare against saved baseline
pytest tests/benchmarks/ --benchmark-compare=before

# Check for regressions against committed baseline
pytest tests/benchmarks/ --check-perf-regression
```

### Updating Performance Baseline

After verified performance improvements:

```bash
python tests/tools/update_perf_baseline.py
```

This runs benchmarks and updates `tests/benchmarks/baseline.json` with current median times.

### Regression Detection

Tests fail if current median exceeds 2x the baseline median. The baseline is stored in `tests/benchmarks/baseline.json`.

## Query Methods Tested

| Method | Description |
|--------|-------------|
| `single_term` | Single word search |
| `phrase_ordered` | Exact phrase (adjacent words in order) |
| `phrase_unordered` | Words within N positions, any order |
| `proxy_ordered` | Proximity search, maintaining order |
| `proxy_unordered` | Proximity search, any order |
| `exact_cooc_ordered` | Exact distance apart, in order |
| `exact_cooc_unordered` | Exact distance apart, any order |
| `sentence_ordered` | Within same sentence, in order |
| `sentence_unordered` | Within same sentence, any order |

## Writing New Tests

### Unit Test Example

```python
import pytest
from philologic.runtime.QuerySyntax import parse_query

@pytest.mark.unit
class TestMyFeature:
    def test_something(self):
        result = parse_query("hamlet")
        assert result == [("TERM", "hamlet")]
```

### Integration Test Example

```python
import pytest

@pytest.mark.integration
class TestMyQuery:
    def test_search(self, shakespeare_db):
        hits = shakespeare_db.query("hamlet", method="single_term")
        hits.finish()
        assert len(hits) > 0
```

### Available Fixtures

| Fixture | Scope | Description |
|---------|-------|-------------|
| `corpus_manager` | session | Manages corpus building/caching |
| `shakespeare_db_path` | session | Path to Shakespeare corpus data |
| `eltec_db_path` | session | Path to ELTeC corpus data |
| `shakespeare_db` | module | DB connection to Shakespeare corpus |
| `eltec_db` | module | DB connection to ELTeC corpus |
| `timer` | function | Timing context manager |

## Development Workflow

Tests import from local source code (no reinstall needed):

```bash
# Edit python/philologic/runtime/Query.py
# Run tests immediately
pytest tests/integration/test_query_methods.py -v
```

The `conftest.py` adds the local `python/` directory to `sys.path`, so changes are picked up immediately.

## Troubleshooting

### Tests fail with "No module named philologic"

Install PhiloLogic:
```bash
cd python && pip install -e .
```

### Cache seems stale

Force rebuild:
```bash
pytest tests/integration/ --force-rebuild -v
```

### Gold set tests fail after code changes

If query results changed intentionally, regenerate gold sets:
```bash
python tests/tools/generate_gold_sets.py --all
```

### Performance tests show regression

1. Verify it's a real regression (run multiple times)
2. If intentional (e.g., added feature), update baseline:
   ```bash
   python tests/tools/update_perf_baseline.py
   ```
