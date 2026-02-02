#!/usr/bin/env python3
"""
Script to generate gold set files from a known-good PhiloLogic installation.

Usage:
    # Regenerate gold sets for all corpora (auto-detects database paths)
    python generate_gold_sets.py --all

    # Regenerate for a specific corpus (auto-detects database path)
    python generate_gold_sets.py --corpus shakespeare

    # Regenerate with explicit database path
    python generate_gold_sets.py --corpus shakespeare --db-path /path/to/db/data

This script runs a set of predefined queries against a PhiloLogic database
and saves the expected results as JSON files for regression testing.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from philologic.runtime.DB import DB
from fixtures.corpus_manager import CorpusManager
from fixtures.corpus_configs import SHAKESPEARE, ELTEC


# Query definitions for Shakespeare corpus
SHAKESPEARE_QUERIES = {
    "single_term": [
        {"id": "st_hamlet", "qs": "hamlet", "method": "single_term", "description": "Search for 'hamlet'"},
        {"id": "st_love", "qs": "love", "method": "single_term", "description": "Search for 'love'"},
        {"id": "st_death", "qs": "death", "method": "single_term", "description": "Search for 'death'"},
        {"id": "st_king", "qs": "king", "method": "single_term", "description": "Search for 'king'"},
        {"id": "st_ghost", "qs": "ghost", "method": "single_term", "description": "Search for 'ghost'"},
        {"id": "st_the", "qs": "the", "method": "single_term", "description": "Search for common word 'the'"},
        {"id": "st_or", "qs": "hamlet | macbeth", "method": "single_term", "description": "OR search"},
    ],
    "phrase": [
        {"id": "po_mylord", "qs": '"my lord"', "method": "phrase_ordered", "description": "Exact phrase 'my lord'"},
        {"id": "po_tobe", "qs": '"to be"', "method": "phrase_ordered", "description": "Exact phrase 'to be'"},
        {"id": "po_goodnight", "qs": '"good night"', "method": "phrase_ordered", "description": "Exact phrase 'good night'"},
        {"id": "pu_lovedie", "qs": "love die", "method": "phrase_unordered", "method_arg": "5", "description": "Unordered within 5 words"},
    ],
    "proximity": [
        {"id": "prox_o_hamletghost", "qs": "hamlet ghost", "method": "proxy_ordered", "method_arg": "10", "description": "Proximity ordered within 10 words"},
        {"id": "prox_u_lovedeath", "qs": "love death", "method": "proxy_unordered", "method_arg": "20", "description": "Proximity unordered within 20 words"},
        {"id": "prox_u_kingqueen", "qs": "king queen", "method": "proxy_unordered", "method_arg": "15", "description": "Proximity unordered within 15 words"},
    ],
    "sentence": [
        {"id": "sent_o_tobe", "qs": "to be", "method": "sentence_ordered", "description": "Sentence ordered 'to be'"},
        {"id": "sent_u_kingdeath", "qs": "king death", "method": "sentence_unordered", "description": "Sentence unordered"},
    ],
}

# Query definitions for ELTeC corpus
ELTEC_QUERIES = {
    "single_term": [
        {"id": "st_love", "qs": "love", "method": "single_term", "description": "Search for 'love'"},
        {"id": "st_death", "qs": "death", "method": "single_term", "description": "Search for 'death'"},
        {"id": "st_house", "qs": "house", "method": "single_term", "description": "Search for 'house'"},
        {"id": "st_the", "qs": "the", "method": "single_term", "description": "Search for common word 'the'"},
    ],
    "phrase": [
        {"id": "po_said", "qs": '"he said"', "method": "phrase_ordered", "description": "Exact phrase 'he said'"},
        {"id": "po_young", "qs": '"young man"', "method": "phrase_ordered", "description": "Exact phrase 'young man'"},
    ],
    "proximity": [
        {"id": "prox_u_lovedeath", "qs": "love death", "method": "proxy_unordered", "method_arg": "20", "description": "Proximity unordered within 20 words"},
    ],
}

# Mapping of corpus names to configs and queries
CORPUS_MAP = {
    "shakespeare": (SHAKESPEARE, SHAKESPEARE_QUERIES),
    "eltec": (ELTEC, ELTEC_QUERIES),
}


def find_corpus_db(corpus_name: str) -> Path:
    """Find the database path for a corpus from the cache.

    Args:
        corpus_name: Name of the corpus (shakespeare or eltec)

    Returns:
        Path to the database data directory

    Raises:
        SystemExit: If corpus not found in cache
    """
    manager = CorpusManager()
    cached = manager.list_cached()

    for corpus in cached:
        if corpus.get("name") == corpus_name:
            return Path(corpus["path"]) / "data"

    print(f"Error: No cached database found for corpus '{corpus_name}'", file=sys.stderr)
    print("Run the test suite first to build the corpus, or specify --db-path explicitly.", file=sys.stderr)
    sys.exit(1)


def generate_gold_set(db: DB, queries: list, corpus_name: str, query_type: str) -> dict:
    """Generate gold set from query definitions.

    Args:
        db: PhiloLogic database connection
        queries: List of query definitions
        corpus_name: Name of the corpus
        query_type: Type of queries (single_term, phrase, etc.)

    Returns:
        Gold set dictionary
    """
    gold_set = {
        "metadata": {
            "corpus": corpus_name,
            "query_type": query_type,
            "generated_date": datetime.now().isoformat(),
            "philologic_version": "5.1.0.1",
            "description": f"Gold set for {query_type} queries on {corpus_name} corpus",
        },
        "test_cases": []
    }

    for query_def in queries:
        print(f"  Running query: {query_def['id']} - {query_def['qs']}")

        method_arg = query_def.get("method_arg", "")

        hits = db.query(
            qs=query_def["qs"],
            method=query_def["method"],
            method_arg=method_arg,
        )
        hits.finish()

        # Collect first N philo_ids
        first_hits = []
        for i, hit in enumerate(hits):
            if i >= 10:
                break
            first_hits.append(list(hit.philo_id))

        test_case = {
            "id": query_def["id"],
            "description": query_def.get("description", ""),
            "query": {
                "qs": query_def["qs"],
                "method": query_def["method"],
                "method_arg": method_arg,
            },
            "expected": {
                "total_hits": len(hits),
                "first_10_philo_ids": first_hits,
            }
        }
        gold_set["test_cases"].append(test_case)
        print(f"    Found {len(hits)} hits")

    return gold_set


def generate_for_corpus(corpus_name: str, db_path: Path = None, output_dir: Path = None):
    """Generate gold sets for a single corpus.

    Args:
        corpus_name: Name of the corpus
        db_path: Optional explicit database path
        output_dir: Optional output directory
    """
    if db_path is None:
        db_path = find_corpus_db(corpus_name)

    # Validate db path
    if not (db_path / "toms.db").exists():
        print(f"Error: {db_path}/toms.db not found", file=sys.stderr)
        sys.exit(1)

    # Set output directory
    if output_dir is None:
        output_dir = REPO_ROOT / "tests" / "gold_sets" / corpus_name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Connect to database
    print(f"Connecting to database at {db_path}")
    db = DB(str(db_path))

    # Get query definitions
    _, query_sets = CORPUS_MAP[corpus_name]

    # Generate gold sets for each query type
    for query_type, queries in query_sets.items():
        print(f"\nGenerating gold set for {query_type}...")
        gold_set = generate_gold_set(db, queries, corpus_name, query_type)

        output_file = output_dir / f"{query_type}.json"
        output_file.write_text(json.dumps(gold_set, indent=2))
        print(f"  Saved to {output_file}")

    print(f"\nGold sets generated successfully in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate gold set files for PhiloLogic regression testing"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate gold sets for all corpora"
    )
    parser.add_argument(
        "--corpus",
        choices=["shakespeare", "eltec"],
        help="Corpus to generate gold sets for"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to the PhiloLogic database data directory (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for gold set files (default: tests/gold_sets/<corpus>/)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.all and not args.corpus:
        parser.error("Either --all or --corpus is required")

    if args.all and args.db_path:
        parser.error("--db-path cannot be used with --all")

    db_path = Path(args.db_path) if args.db_path else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.all:
        # Generate for all corpora
        for corpus_name in CORPUS_MAP:
            print(f"\n{'='*60}")
            print(f"Generating gold sets for {corpus_name.upper()}")
            print(f"{'='*60}")
            generate_for_corpus(corpus_name)
    else:
        generate_for_corpus(args.corpus, db_path, output_dir)


if __name__ == "__main__":
    main()
