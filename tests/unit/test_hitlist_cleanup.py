"""The hitlist cleanup keeps the senses reports for the database's load, up to a limit of room, the least recently
used going first past it, and removes every other file after 10 minutes (www/middleware.py)."""

import datetime
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "www"))

from middleware import clean_hitlist_dir, is_senses_result  # noqa: E402
from philologic.runtime.reports import word_senses as ws  # noqa: E402


def write(directory, name, size, age):
    """A file of size bytes, last modified age seconds ago."""
    path = directory / name
    path.write_bytes(b"x" * size)
    when = time.time() - age
    os.utime(path, (when, when))
    return path


def cutoff():
    return datetime.datetime.now() - datetime.timedelta(minutes=10)


def test_the_senses_results_are_told_by_their_names():
    assert is_senses_result("ab12.senses.json")
    assert not is_senses_result("ab12.membership.npz")
    assert not is_senses_result("ab12.periods-3.json")
    assert not is_senses_result("ab12.hitlist")
    assert not is_senses_result("ab12.hitlist.done")
    assert not is_senses_result("ab12.colloc.json")
    assert not is_senses_result("tmpq1w2.tmp")


def test_old_files_go_but_the_senses_results_stay(tmp_path):
    old = 3600
    hitlist = write(tmp_path, "a.hitlist", 10, old)
    flag = write(tmp_path, "a.hitlist.done", 1, old)
    fresh = write(tmp_path, "b.hitlist", 10, 5)
    report = write(tmp_path, "c.senses.json", 10, old)
    membership = write(tmp_path, "c.membership.npz", 10, old)  # no longer written: an ordinary file
    periods = write(tmp_path, "c.periods-1.json", 10, old)
    clean_hitlist_dir(str(tmp_path), cutoff(), limit=1000)
    assert not hitlist.exists() and not flag.exists()
    assert fresh.exists()
    assert report.exists() and not membership.exists() and not periods.exists()


def test_past_the_limit_the_least_recently_used_results_go_first(tmp_path):
    used = [write(tmp_path, f"{i}.senses.json", 100, age) for i, age in enumerate((5000, 4000, 3000, 2000, 1000))]
    other = write(tmp_path, "x.hitlist", 100, 5)  # not counted against the limit, nor removed while fresh
    clean_hitlist_dir(str(tmp_path), cutoff(), limit=250)
    assert [p.exists() for p in used] == [False, False, False, True, True]  # 200 bytes left, the two used last
    assert other.exists()


def test_under_the_limit_nothing_of_the_senses_goes(tmp_path):
    used = [write(tmp_path, f"{i}.senses.json", 100, 10**6) for i in range(3)]
    clean_hitlist_dir(str(tmp_path), cutoff(), limit=300)
    assert all(p.exists() for p in used)


def test_a_report_read_from_the_cache_is_marked_as_just_used(tmp_path):
    db = SimpleNamespace(hitlist_dir=str(tmp_path))
    path = Path(ws._cache_path(db, "lemma:word", True, (), {}, None, None, []))
    path.write_text(json.dumps({"nodes": [], "cached": True}), encoding="utf-8")
    long_ago = time.time() - 10**6
    os.utime(path, (long_ago, long_ago))
    report = ws.word_senses(db, "lemma:word", True)
    assert report["cached"] is True
    assert path.stat().st_mtime > long_ago + 10**5
