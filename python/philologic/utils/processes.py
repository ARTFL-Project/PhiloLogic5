"""Shell commands and pools of workers which fail loudly: an error stops the work and is raised, instead of being
ignored or leaving the caller waiting forever."""

import multiprocessing
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager


def run_shell(command, description=None, ok_statuses=(0,), capture_output=False):
    """Run a shell command with bash, raising a RuntimeError if its exit status is not in ok_statuses.
    With pipefail, a pipeline fails when any of its commands fails, not only the last one.
    Returns the CompletedProcess, with stdout and stderr as strings if capture_output is True."""
    process = subprocess.run(
        ["/bin/bash", "-o", "pipefail", "-c", command],
        capture_output=capture_output,
        text=capture_output,
        encoding="utf8" if capture_output else None,
    )
    if process.returncode not in ok_statuses:
        message = f"{description or command} failed with exit status {process.returncode}"
        if capture_output and process.stderr:
            message += f": {process.stderr.strip()}"
        raise RuntimeError(message)
    return process


@contextmanager
def process_pool(workers):
    """ProcessPoolExecutor whose workers are forked from this process, so they see its current state
    (class attributes set by the Loader, for instance): nothing but the jobs and their results is pickled.
    Unlike multiprocess(ing).Pool, a worker dying (killed, or calling exit()) raises an error instead of hanging.
    If the block exits with an error, pending jobs are cancelled and running ones stopped."""
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"))
    try:
        yield executor
    except BaseException:
        running_workers = list((executor._processes or {}).values())  # no public API before Python 3.14
        executor.shutdown(wait=False, cancel_futures=True)
        for worker in running_workers:
            worker.terminate()
        raise
    finally:
        executor.shutdown()


@contextmanager
def thread_pool(workers):
    """ThreadPoolExecutor whose pending jobs are cancelled if the block exits with an error
    (running jobs can't be stopped: they finish before the error is raised)."""
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            yield executor
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
