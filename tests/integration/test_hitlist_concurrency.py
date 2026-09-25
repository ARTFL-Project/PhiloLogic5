"""Hitlists under concurrency: every answer must be the query's own, whatever else happens.

A hitlist is produced once and shared (see HitList.claim_hitlist): by the threads of a process, between processes,
by a request taking over from a producer that died, and while the hitlist cleanup removes files still in use. These
tests check that none of this changes a query's results or turns them into errors.
"""

import glob
import hashlib
import json
import multiprocessing
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import HitList, MetadataQuery, Query  # noqa: E402
from philologic.runtime.DB import DB  # noqa: E402
from tests.fixtures.hitlist_workers import fingerprint, orphan_producer, query_worker, run_query  # noqa: E402

# Year ranges covering every dated novel of the corpus: different metadata queries with the same answer
ALL_YEARS = "1000-3000"

QUERIES = [
    ("the", "single_term", "0", None, {}),
    ("and", "single_term", "0", None, {}),
    ("man", "single_term", "0", None, {}),
    ("house", "single_term", "0", None, {}),
    ("man | woman", "single_term", "0", None, {}),
    ("of the", "phrase_ordered", "0", None, {}),
    ("man house", "proxy_unordered", "5", None, {}),
    ("love heart", "sentence_unordered", "6", None, {}),
    ("the", "single_term", "0", None, {"year": "1850-1870"}),
    ("man", "single_term", "0", None, {"year": "1841-1860"}),
    ("love", "single_term", "0", None, {"year": "1861-1920"}),
    ("man", "single_term", "0", ["author", "title"], {}),
    ("love", "single_term", "0", ["year", "title"], {}),
    ("house", "single_term", "0", ["title"], {"year": "1850-1900"}),
    ("", "", "", ["author"], {"year": "1850-1870"}),
]


def corpus_file(db, **metadata):
    """Name of the metadata corpus hitlist DB.query uses for metadata."""
    h = hashlib.sha1()
    h.update(db.path.encode("utf8"))
    for key, value in metadata.items():
        h.update(f"{key}={value}".encode("utf8"))
    return f"{db.path}/hitlists/{h.hexdigest()}.hitlist"


def hitlist_files(db):
    return glob.glob(f"{db.path}/hitlists/*")


@pytest.fixture
def db(eltec_db_path):
    return DB(str(eltec_db_path))


@pytest.mark.integration
class TestFilesRemovedOrFailing:
    """Hitlist files removed while in use (as the hitlist cleanup does), and hitlists whose production fails."""

    def test_corpus_removed_during_the_search(self, db):
        """The search keeps the metadata corpus it filters against, even once the file is gone."""
        expected = {
            w: fingerprint(db.query(w, "single_term", "0", raw_results=True, year=ALL_YEARS))
            for w in ("the", "and", "man")
        }
        for i, word in enumerate(["the", "and", "man"] * 3):
            year = f"{1001 + i}-3000"  # a new corpus each time, with the same objects
            hits = db.query(word, "single_term", "0", raw_results=True, year=year)
            os.remove(corpus_file(db, year=year))
            assert fingerprint(hits) == expected[word]
            # and what is cached is right too
            assert fingerprint(db.query(word, "single_term", "0", raw_results=True, year=year)) == expected[word]

    def test_hitlist_removed_while_read(self, db):
        """Readers of a hitlist being produced, the producing request's and a waiting one's, read it to the end."""
        expected = fingerprint(db.query("the", "single_term", "0", raw_results=True))
        for f in hitlist_files(db):
            os.remove(f)
        hits = db.query("the", "single_term", "0", raw_results=True)
        waiter = db.query("the", "single_term", "0", raw_results=True)
        os.remove(hits.filename)
        assert fingerprint(hits) == expected
        assert fingerprint(waiter) == expected

    def test_hitlist_removed_and_produced_again_while_read(self, db):
        expected = fingerprint(db.query("and", "single_term", "0", raw_results=True))
        for f in hitlist_files(db):
            os.remove(f)
        hits = db.query("and", "single_term", "0", raw_results=True)
        os.remove(hits.filename)
        again = db.query("and", "single_term", "0", raw_results=True)
        assert fingerprint(hits) == expected
        assert fingerprint(again) == expected

    def test_sorted_copy_removed(self, db):
        expected = fingerprint(db.query("man", "single_term", "0", raw_results=True, sort_order=["title"]))
        for sorted_copy in glob.glob(f"{db.path}/hitlists/*.sorted.*"):
            os.remove(sorted_copy)
        assert fingerprint(db.query("man", "single_term", "0", raw_results=True, sort_order=["title"])) == expected

    @pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the crash is the point
    def test_failed_search_is_not_cached(self, db, monkeypatch):
        """Readers of a search that fails get an error, not what it wrote, and the next request searches again."""
        expected = fingerprint(db.query("house", "single_term", "0", raw_results=True))
        for f in hitlist_files(db):
            os.remove(f)

        def failing_search_word(db_path, hitlist_filename, overflow_words, **kwargs):
            with open(hitlist_filename, "wb") as out:
                out.write(np.zeros((50, 9), dtype=np.uint32).tobytes())
            time.sleep(0.3)  # while an identical request waits for it
            raise RuntimeError("search crashed")

        monkeypatch.setattr(Query, "search_word", failing_search_word)
        claimer = db.query("house", "single_term", "0", raw_results=True)
        waiter = db.query("house", "single_term", "0", raw_results=True)
        for hits in (claimer, waiter):
            with pytest.raises(HitList.HitlistFailed):
                hits.finish()
        monkeypatch.undo()
        assert fingerprint(db.query("house", "single_term", "0", raw_results=True)) == expected

    def test_failed_metadata_corpus_is_not_cached(self, db, monkeypatch):
        expected = fingerprint(db.query("man", "single_term", "0", raw_results=True, year=ALL_YEARS))
        for f in hitlist_files(db):
            os.remove(f)
        query_recursive = MetadataQuery.query_recursive

        def failing_query_recursive(*args, **kwargs):
            rows = query_recursive(*args, **kwargs)
            yield next(rows)
            raise RuntimeError("metadata query crashed")

        monkeypatch.setattr(MetadataQuery, "query_recursive", failing_query_recursive)
        db.query("man", "single_term", "0", raw_results=True, year=ALL_YEARS)
        monkeypatch.undo()
        assert fingerprint(db.query("man", "single_term", "0", raw_results=True, year=ALL_YEARS)) == expected


@pytest.mark.integration
class TestClaims:
    def test_orphaned_hitlist_taken_over_by_waiting_request(self, db):
        """A request waiting for a hitlist whose producer process dies produces it again, from scratch."""
        expected = fingerprint(db.query("house", "single_term", "0", raw_results=True))
        for f in hitlist_files(db):
            os.remove(f)
        ctx = multiprocessing.get_context("spawn")
        started, die = ctx.Event(), ctx.Event()
        producer = ctx.Process(target=orphan_producer, args=(db.path, "house", started, die))
        producer.start()
        try:
            assert started.wait(60), "the producer process did not start"
            waiter = db.query("house", "single_term", "0", raw_results=True)
            assert not waiter.done
            die.set()
            producer.join(30)
            assert fingerprint(waiter) == expected  # not the 10 bogus hits it left
        finally:
            producer.kill()

    def test_claim_released_by_two_threads_is_closed_once(self, tmp_path, monkeypatch):
        """The claimer and its producer thread can both release a claim. Closing its descriptor twice could close a
        file another thread has just opened under the same number."""
        closes = []
        real_close = os.close

        def counting_close(fd):
            if sys._getframe(1).f_code is HitList.HitlistClaim.release.__code__:
                closes.append(fd)
            real_close(fd)

        monkeypatch.setattr(os, "close", counting_close)
        for i in range(300):
            with HitList.claim_hitlist(str(tmp_path / f"{i}.hitlist")) as claim:
                claim.hand_over()
            number = claim.fd
            closes.clear()
            barrier = threading.Barrier(2)

            def release():
                barrier.wait()
                claim.release()

            threads = [threading.Thread(target=release) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert closes.count(number) == 1


def _stress(db, seconds, kill_every=None, remove_every=None, workers=6):
    """Run query workers against the same database for `seconds` and return their reports. Optionally kill a random
    worker every `kill_every` seconds (starting another), abandoning the hitlists it was producing, and every
    `remove_every` seconds remove the hitlist files that haven't changed for 2 seconds, in random order and a little
    at a time: the hitlist cleanup (see CleanupMiddleware) with its 10 minutes scaled down, so that it keeps removing
    files still in use."""
    truth = [fingerprint(run_query(db, query)) for query in QUERIES]  # sequential answers
    for f in hitlist_files(db):
        os.remove(f)
    ctx = multiprocessing.get_context("spawn")
    report_dir = Path(db.path) / "hitlists"
    reports, procs = [], []
    stop = time.time() + seconds

    def start(seed):
        report = report_dir / f"stress_report_{seed}.jsonl"
        reports.append(report)
        p = ctx.Process(target=query_worker, args=(db.path, QUERIES, truth, stop - time.time(), seed, str(report)))
        p.start()
        procs.append(p)

    def remove_files():
        rng = random.Random(0)
        while time.time() < stop:
            files = [f for f in hitlist_files(db) if "stress_report" not in f]
            rng.shuffle(files)
            for f in files:
                try:
                    if time.time() - os.path.getmtime(f) > 2:
                        os.remove(f)
                        time.sleep(0.001)
                except FileNotFoundError:
                    pass
            time.sleep(remove_every)

    remover = threading.Thread(target=remove_files) if remove_every else None
    for seed in range(workers):
        start(seed)
    time.sleep(3)  # workers are up (spawn imports philologic afresh)
    if remover:
        remover.start()
    rng, seed, killed = random.Random(1), workers, 0
    while time.time() < stop:
        time.sleep(kill_every or 0.2)
        if kill_every and time.time() < stop - 2:
            victims = [p for p in procs if p.is_alive()]
            if victims:
                os.kill(rng.choice(victims).pid, signal.SIGKILL)
                killed += 1
                start(seed)
                seed += 1
    for p in procs:
        p.join(120)
    if remover:
        remover.join()
    lines = []
    for report in reports:
        if report.exists():
            for line in report.read_text(encoding="utf8").splitlines():
                try:
                    lines.append(json.loads(line))
                except ValueError:  # the last line of a worker killed while writing it
                    pass
            report.unlink()
    return lines, killed


@pytest.mark.integration
class TestConcurrentQueries:
    """Many processes running identical and different queries at once: each answer must match the sequential one."""

    def test_while_workers_are_killed(self, db):
        lines, killed = _stress(db, seconds=12, kill_every=1)
        bad = [line for line in lines if not line.get("ok")]
        assert killed >= 5
        assert len(lines) > 100, "too few queries ran to test anything"
        assert not bad, bad[:3]

    def test_while_the_cleanup_removes_files_in_use(self, db):
        lines, _ = _stress(db, seconds=12, remove_every=0.05)
        bad = [line for line in lines if not line.get("ok")]
        assert len(lines) > 100, "too few queries ran to test anything"
        assert not bad, bad[:3]
