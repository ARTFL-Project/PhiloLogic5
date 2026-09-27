"""Unit tests for philologic.utils.processes: shell commands and pools of workers which fail loudly."""

import os
import resource
import subprocess
import sys
import time
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Loader import init_index_worker, report_progress
from philologic.utils.processes import process_pool, raise_open_files_limit, run_shell, shared_value, thread_pool

pytestmark = pytest.mark.unit


class TestRunShell:
    def test_failure_raises(self):
        with pytest.raises(RuntimeError, match="counting failed with exit status 3"):
            run_shell("exit 3", description="counting")

    def test_pipeline_failure_raises(self):
        """With pipefail: the pipeline fails when any of its commands fails, not only the last one"""
        with pytest.raises(RuntimeError, match="exit status 1"):
            run_shell("false | cat")

    def test_ok_statuses(self):
        assert run_shell("exit 1", ok_statuses=(0, 1)).returncode == 1

    def test_bash_syntax(self):
        """Commands run with bash: process substitution works"""
        assert run_shell("cat <(echo a)", capture_output=True).stdout == "a\n"

    def test_capture_output(self):
        with pytest.raises(RuntimeError, match="exit status 2: no such thing"):
            run_shell("echo no such thing >&2; exit 2", capture_output=True)


class TestProcessPool:
    def test_results(self):
        with process_pool(3) as pool:
            assert list(pool.map(abs, range(-50, 50), chunksize=7)) == [abs(n) for n in range(-50, 50)]

    def test_job_error_raised(self):
        with process_pool(2) as pool:
            with pytest.raises(ValueError):
                pool.submit(int, "not a number").result()

    def test_dying_worker_raises(self):
        """A worker which exits (or is killed) raises an error instead of leaving the pool waiting forever"""
        start = time.time()
        with pytest.raises(BrokenProcessPool):
            with process_pool(2) as pool:
                pool.submit(os._exit, 1).result(timeout=60)
        assert time.time() - start < 30

    def test_error_in_block_stops_workers(self):
        """Running jobs are stopped, and pending ones never run, when the block exits with an error"""
        start = time.time()
        with pytest.raises(KeyError):
            with process_pool(2) as pool:
                jobs = [pool.submit(time.sleep, 60) for _ in range(6)]
                time.sleep(1)  # let the first ones start
                workers = list(pool._processes.values())
                raise KeyError("stop")
        assert time.time() - start < 30
        while any(worker.is_alive() for worker in workers) and time.time() - start < 30:
            time.sleep(0.1)
        assert not any(worker.is_alive() for worker in workers)
        # None runs to its end: cancelled, or failed as the pool broke
        assert all(job.cancelled() or job.exception(timeout=30) is not None for job in jobs)

    def test_environment_and_working_directory(self, monkeypatch, tmp_path):
        """Workers get the environment and working directory of the process creating the pool"""
        monkeypatch.setenv("PHILOLOGIC_TEST_VARIABLE", "set after the worker server started")
        monkeypatch.chdir(tmp_path)
        with process_pool(2) as pool:
            assert pool.submit(os.getenv, "PHILOLOGIC_TEST_VARIABLE").result() == "set after the worker server started"
            assert pool.submit(os.getcwd).result() == str(tmp_path)

    def test_initializer_and_shared_value(self):
        progress = shared_value("q", 0)
        with process_pool(3, init_index_worker, (progress,)) as pool:
            for job in [pool.submit(report_progress, 5) for _ in range(10)]:
                job.result()
        assert progress.value == 50


class TestThreadPool:
    def test_error_cancels_pending_jobs(self):
        start = time.time()
        with pytest.raises(KeyError):
            with thread_pool(1) as executor:
                running = executor.submit(time.sleep, 0.5)
                pending = [executor.submit(time.sleep, 10) for _ in range(5)]
                raise KeyError("stop")
        assert time.time() - start < 5
        assert running.done() and all(job.cancelled() for job in pending)


def test_raise_open_files_limit():
    """A process started with a low limit on open files raises it as far as it may"""
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if hard != resource.RLIM_INFINITY and hard <= 256:
        pytest.skip("the hard limit is too low to lower the soft limit")
    code = (
        "import resource; from philologic.utils import raise_open_files_limit; "
        "print(raise_open_files_limit(), *resource.getrlimit(resource.RLIMIT_NOFILE))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "python")},
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard)),
    )
    returned, soft, new_hard = map(int, result.stdout.split())
    assert returned == soft and new_hard == hard
    assert soft == hard or (hard == resource.RLIM_INFINITY and soft >= 10240)
    assert raise_open_files_limit() == resource.getrlimit(resource.RLIMIT_NOFILE)[0]
