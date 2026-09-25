"""The web app under concurrent requests: every response must be its query's own.

Starts Gunicorn with the repository's web app (www/) on the test corpus, records each request's response one at a
time, then replays them concurrently while workers are killed (as timeouts and crashes do; their recycling after a
few requests abandons searches too) and while the hitlist cleanup removes files still in use. Every response must
match its sequential one: hitlist sharing between workers and threads must never show in the results.
"""

import collections
import http.client
import json
import os
import random
import re
import signal
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent

pytest.importorskip("gunicorn")
pytest.importorskip("falcon")

# Dropped by a worker we killed: expected, not a wrong answer
DROPPED = ("RemoteDisconnected", "ConnectionResetError", "BrokenPipeError", "IncompleteRead")


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.unix_path)


class Server:
    """A Gunicorn serving the web app on a private socket."""

    def __init__(self, db_root, db_name, workdir):
        self.db_name = db_name
        # A socket path can't be much longer than 100 characters, which a test's temporary directory may be
        self.socket_dir = tempfile.mkdtemp(prefix="philo-", dir="/tmp" if os.path.isdir("/tmp") else None)
        self.socket = os.path.join(self.socket_dir, "gunicorn.sock")
        self.log = workdir / "gunicorn.log"
        config = workdir / "gunicorn.conf.py"
        config.write_text(
            f"bind = 'unix:{self.socket}'\n"
            "workers = 4\n"
            "max_requests = 100  # recycled now and then, abandoning their background searches\n"
            "max_requests_jitter = 20\n"
            "timeout = 60\n"
            "graceful_timeout = 5\n"
            "preload_app = True\n"
            "control_socket_disable = True\n"
            f"errorlog = '{self.log}'\n"
            "loglevel = 'info'\n"
            "capture_output = True\n"
        )
        # Thread pools capped as in www/gunicorn.conf.py (which this test's config doesn't load)
        env = dict(
            os.environ,
            PHILOLOGIC_DB_ROOT=str(db_root),
            PYTHONPATH=str(REPO_ROOT / "python"),
            NUMBA_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "gunicorn",
                "--config",
                str(config),
                "--chdir",
                str(REPO_ROOT / "www"),
                "app:application",
            ],
            env=env,
        )
        deadline = time.time() + 60
        while not self._up():
            if self.process.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"gunicorn did not start:\n{self.log.read_text() if self.log.exists() else ''}")
            time.sleep(0.2)

    def _up(self):
        try:
            return self.get("scripts/get_total_results.py", {"q": "the"})[0] == 200
        except OSError:
            return False

    def get(self, path, params, timeout=120):
        conn = UnixHTTPConnection(self.socket, timeout)
        try:
            conn.request("GET", f"/philologic5/{self.db_name}/{path}?{urlencode(params, doseq=True)}")
            resp = conn.getresponse()
            body = resp.read()
        finally:
            conn.close()
        return resp.status, (json.loads(body) if resp.status == 200 else None)

    def workers(self):
        """Pids of the live workers, from the log (portable, unlike /proc)."""
        pids = {int(p) for p in re.findall(r"Booting worker with pid: (\d+)", self.log.read_text())}
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except OSError:
                pass
        return alive

    def errors(self):
        """Tracebacks logged by the app (a killed worker is only reported, not a traceback)."""
        return [line for line in self.log.read_text().splitlines() if "Traceback" in line]

    def stop(self):
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        shutil.rmtree(self.socket_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def server(eltec_db_path, tmp_path_factory):
    corpus = Path(eltec_db_path).parent
    s = Server(corpus.parent, corpus.name, tmp_path_factory.mktemp("web"))
    yield s
    s.stop()


BASE = {"colloc_filter_choice": "frequency", "filter_frequency": 100, "method_arg": 5}


def requests():
    """(name, path, params) of requests covering the reports and search methods."""
    r = []
    for w in ("the", "and", "man", "house", "love", "heart"):
        r.append((f"total {w}", "scripts/get_total_results.py", {"q": w}))
        r.append((f"concordance {w}", "reports/concordance.py", {"q": w, "start": 1, "end": 25}))
        r.append((f"concordance {w} p9", "reports/concordance.py", {"q": w, "start": 201, "end": 225}))
        r.append(
            (
                f"concordance {w} by author",
                "reports/concordance.py",
                {"q": w, "start": 1, "end": 25, "sort_by[]": ["author", "title"]},
            )
        )
        r.append((f"total {w} 1850-1870", "scripts/get_total_results.py", {"q": w, "year": "1850-1870"}))
    r += [
        ("kwic love", "reports/kwic.py", {"q": "love", "start": 1, "end": 25}),
        (
            "concordance heart by year",
            "reports/concordance.py",
            {"q": "heart", "start": 1, "end": 25, "sort_by[]": ["year", "title"]},
        ),
        ("concordance heart past the end", "reports/concordance.py", {"q": "heart", "start": 100000, "end": 100024}),
        ("total man | woman", "scripts/get_total_results.py", {"q": "man | woman"}),
        ("total of the (phrase)", "scripts/get_total_results.py", {"q": "of the", "method": "proxy", "arg": 0}),
        (
            "concordance man house within 5",
            "reports/concordance.py",
            {"q": "man house", "method": "proxy", "arg": 5, "start": 1, "end": 25},
        ),
        ("total love heart (sentence)", "scripts/get_total_results.py", {"q": "love heart", "method": "sentence"}),
        ("collocation man", "reports/collocation.py", {"q": "man", **BASE}),
        ("collocation love 1841-1880", "reports/collocation.py", {"q": "love", "year": "1841-1880", **BASE}),
        ("bibliography 1850-1870", "reports/bibliography.py", {"year": "1850-1870", "start": 1, "end": 50}),
        (
            "bibliography 1850-1870 by title",
            "reports/bibliography.py",
            {"year": "1850-1870", "start": 1, "end": 50, "sort_by[]": ["title", "author"]},
        ),
    ]
    return r


def fingerprint(path, body):
    """What concurrency must not change. A concordance can be answered before its search is done: then its count is
    marked partial (it can only be lower), while its page, which waits for its hits, must match."""
    if path.endswith("get_total_results.py"):
        return {"count": body}
    if path.endswith("collocation.py"):
        return {"count": body["results_length"], "filter_list": body["filter_list"], "collocates": body["collocates"]}
    fp = {
        "page": [[r.get("philo_id"), r.get("bytes") or []] for r in body["results"]],
        "count": body.get("results_length"),
    }
    if body.get("query_done") is False:
        fp["partial"] = True
    return fp


def matches(expected, got):
    if got.pop("partial", False):
        return got["page"] == expected["page"] and got["count"] <= expected["count"]
    return got == expected


def sequential_answers(server, reqs):
    answers = {}
    for name, path, params in reqs:
        for _ in range(240):  # until its search is done, so that the count is final
            status, body = server.get(path, params)
            assert status == 200, f"{name}: HTTP {status}"
            fp = fingerprint(path, body)
            if not fp.get("partial"):
                break
            time.sleep(0.25)
        answers[name] = fp
    return answers


def clear_hitlists(eltec_db_path):
    for f in (Path(eltec_db_path) / "hitlists").iterdir():
        try:
            f.unlink()
        except FileNotFoundError:
            pass


def chaos(server, stop, kill_every=None, age_every=None, eltec_db_path=None):
    """Kill a random worker every `kill_every` seconds, and backdate the hitlist files every `age_every` seconds so
    that the hitlist cleanup (CleanupMiddleware, on one request in ten) removes them, in use or not."""
    rng = random.Random(1)
    killed = 0
    next_kill = time.time() + (kill_every or 1e9)
    next_age = time.time() + (age_every or 1e9)
    while time.time() < stop:
        now = time.time()
        if kill_every and now >= next_kill:
            workers = server.workers()
            if workers:
                try:
                    os.kill(rng.choice(workers), signal.SIGKILL)
                    killed += 1
                except ProcessLookupError:
                    pass
            next_kill = now + kill_every
        if age_every and now >= next_age:
            old = now - 11 * 60
            for f in (Path(eltec_db_path) / "hitlists").iterdir():
                try:
                    os.utime(f, (old, old))
                except FileNotFoundError:
                    pass
            next_age = now + age_every
        time.sleep(0.05)
    return killed


@pytest.mark.integration
class TestConcurrentRequests:
    def test_mixed_requests_while_workers_die_and_files_are_cleaned_up(self, server, eltec_db_path):
        reqs = requests()
        expected = sequential_answers(server, reqs)
        clear_hitlists(eltec_db_path)
        stop = time.time() + 15
        outcomes = collections.Counter()
        bad = []
        lock = threading.Lock()

        def client(seed):
            rng = random.Random(seed)
            hot = rng.sample(reqs, 6)  # identical requests overlap often
            while time.time() < stop:
                name, path, params = rng.choice(hot) if rng.random() < 0.5 else rng.choice(reqs)
                try:
                    status, body = server.get(path, params)
                except Exception as e:  # noqa: BLE001
                    status, body = type(e).__name__, None
                with lock:
                    if status in DROPPED:
                        outcomes["dropped by a killed worker"] += 1
                    elif status != 200:
                        bad.append((name, f"HTTP {status}"))
                    elif matches(expected[name], fingerprint(path, body)):
                        outcomes["ok"] += 1
                    else:
                        bad.append((name, "wrong answer"))

        threads = [threading.Thread(target=client, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        killed = chaos(server, stop, kill_every=1, age_every=2, eltec_db_path=eltec_db_path)
        for t in threads:
            t.join()
        assert killed >= 5
        assert outcomes["ok"] > 200, outcomes
        assert not bad, bad[:5]
        assert not server.errors(), server.errors()[:3]

    def test_bursts_of_new_identical_requests_while_workers_die(self, server, eltec_db_path):
        """Bursts of identical requests never seen before (a new year range with every dated novel), so that they
        produce their hitlists together, while workers die: waiting requests take their searches over."""
        all_years = [r for r in requests() if r[0].startswith(("total ", "concordance ")) and "year" not in r[2]]
        reference = [(name, path, dict(params, year="1000-3000")) for name, path, params in all_years]
        expected = sequential_answers(server, reference)
        stop = time.time() + 12
        outcomes = collections.Counter()
        bad = []
        lock = threading.Lock()

        def runner(seed):
            rng = random.Random(100 + seed)
            while time.time() < stop:
                name, path, params = rng.choice(reference)
                fresh = dict(params, year=f"{rng.randint(1000, 1840)}-{rng.randint(1921, 9999)}")
                results = [None] * 6
                barrier = threading.Barrier(6)

                def one(j):
                    barrier.wait()
                    try:
                        results[j] = server.get(path, fresh)
                    except Exception as e:  # noqa: BLE001
                        results[j] = (type(e).__name__, None)

                burst = [threading.Thread(target=one, args=(j,)) for j in range(6)]
                for t in burst:
                    t.start()
                for t in burst:
                    t.join()
                with lock:
                    for status, body in results:
                        if status in DROPPED:
                            outcomes["dropped by a killed worker"] += 1
                        elif status != 200:
                            bad.append((name, fresh["year"], f"HTTP {status}"))
                        elif matches(expected[name], fingerprint(path, body)):
                            outcomes["ok"] += 1
                        else:
                            bad.append((name, fresh["year"], "wrong answer"))

        threads = [threading.Thread(target=runner, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        killed = chaos(server, stop, kill_every=0.7)
        for t in threads:
            t.join()
        assert killed >= 5
        assert outcomes["ok"] > 100, outcomes
        assert not bad, bad[:5]
        assert not server.errors(), server.errors()[:3]
