"""Regression test for the parser's output: parsing the test collections gives the same output, for each file, as
recorded in parser_output_hashes.json (md5 of the output). Parser changes which shouldn't change its output (such as
refactorings) can be checked with it. After an intended change of the output, record the new one with:
python tests/regression/test_parser_output.py --update"""

import hashlib
import io
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Parser import XMLParser

COLLECTIONS = REPO_ROOT / "tests" / "collections"
HASHES = Path(__file__).with_name("parser_output_hashes.json")


def collection_files():
    """All the Shakespeare plays, and one ELTeC novel out of four"""
    return (
        sorted((COLLECTIONS / "folger-shakespeare").glob("*.xml"))
        + sorted((COLLECTIONS / "ELTeC-eng").glob("*.xml"))[::4]
    )


def output_hash(path):
    """md5 of the parser's output for a file, parsed with the default parser options"""
    output = io.StringIO()
    parser = XMLParser(output, 1, path.stat().st_size, known_metadata={"filename": path.name}, metadata_sql_types={})
    with open(path, encoding="utf8", newline="") as input_file:
        parser.parse(input_file)
    return hashlib.md5(output.getvalue().encode("utf8")).hexdigest()


def output_hashes(paths):
    """{collection/file: output hash}, files parsed in parallel by forked processes where possible"""
    names = [f"{path.parent.name}/{path.name}" for path in paths]
    if "fork" not in multiprocessing.get_all_start_methods():
        return dict(zip(names, map(output_hash, paths)))
    with ProcessPoolExecutor(min(16, os.cpu_count() or 1), mp_context=multiprocessing.get_context("fork")) as pool:
        return dict(zip(names, pool.map(output_hash, paths)))


@pytest.mark.gold_set
def test_parser_output_unchanged():
    expected = json.loads(HASHES.read_text())
    actual = output_hashes(collection_files())
    assert sorted(actual) == sorted(expected), "the test collections changed: record their output again (--update)"
    changed = sorted(name for name in expected if actual[name] != expected[name])
    assert not changed, f"parser output changed for {len(changed)} of {len(expected)} files: {', '.join(changed[:10])}"


if __name__ == "__main__":
    if "--update" in sys.argv:
        hashes = output_hashes(collection_files())
        HASHES.write_text(json.dumps(hashes, indent=1, sort_keys=True) + "\n")
        print(f"recorded the output of {len(hashes)} files in {HASHES}")
