"""Processes for the hitlist concurrency tests (tests/integration/test_hitlist_concurrency.py).

They run under the spawn start method, so they live in an importable module. Each worker reports to its own file
rather than through a multiprocessing queue, which a worker killed mid-write could leave broken for the others.
"""

import hashlib
import json
import os
import random
import time
import traceback


def run_query(db, query):
    """Run a query spec: (q, method, method_arg, sort_order or None, metadata)."""
    q, method, arg, sort_order, metadata = query
    return db.query(q, method, arg, raw_results=True, sort_order=sort_order or ["rowid"], **metadata)


def fingerprint(hits):
    """What no concurrency may change: how many hits there are, and all of them, in order."""
    hits.finish()
    return [len(hits), hashlib.sha1(hits.read_data()).hexdigest()]


def query_worker(db_path, queries, truth, seconds, seed, report_path):
    """Run random queries for `seconds`, half of them from a small hot set so that identical queries overlap,
    and report every answer that differs from `truth` (the sequential answers) and every error."""
    # Thread pools kept small before numpy and numba start them (see Server in test_web_concurrency.py)
    os.environ["NUMBA_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from philologic.runtime.DB import DB

    db = DB(db_path)
    rng = random.Random(seed)
    hot = rng.sample(range(len(queries)), min(4, len(queries)))
    stop = time.time() + seconds
    with open(report_path, "a") as report:
        while time.time() < stop:
            i = rng.choice(hot) if rng.random() < 0.5 else rng.randrange(len(queries))
            try:
                got = fingerprint(run_query(db, queries[i]))
                line = {"ok": True} if got == truth[i] else {"wrong": queries[i], "expected": truth[i], "got": got}
            except Exception:  # noqa: BLE001
                line = {"error": queries[i], "traceback": traceback.format_exc(limit=6)}
            report.write(json.dumps(line) + "\n")
            report.flush()


def orphan_producer(db_path, q, started, die):
    """Start producing the hitlist for q, write a few (bogus) hits, and die without finishing it when told to."""
    from philologic.runtime import Query
    from philologic.runtime.DB import DB

    def stuck_search_word(db_path, hitlist_filename, overflow_words, **kwargs):
        with open(hitlist_filename, "wb") as out:
            out.write(b"\0" * 36 * 10)
            out.flush()
        started.set()
        time.sleep(3600)

    Query.search_word = stuck_search_word
    db = DB(db_path)
    db.query(q, "single_term", "0", raw_results=True)
    die.wait(60)
    os._exit(0)
