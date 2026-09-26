"""Shell commands and pools of workers which fail loudly: an error stops the work and is raised, instead of being
ignored or leaving the caller waiting forever."""

import multiprocessing
import multiprocessing.forkserver
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager

# Workers are forked from a server process started for that purpose, which runs no threads and holds no state:
# unlike forking this process, this is safe whatever it is doing.
START_METHOD = "forkserver" if "forkserver" in multiprocessing.get_all_start_methods() else "spawn"


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


def start_worker_server(preload=()):
    """Start the server from which workers are forked, importing the given modules in it: this is then done while
    this process carries on, rather than when workers are first needed. Only the first call has any effect."""
    if START_METHOD == "forkserver":
        multiprocessing.set_forkserver_preload(list(preload))
        multiprocessing.forkserver.ensure_running()


def shared_value(typecode, value):
    """multiprocessing.Value which can be given to the workers of a process_pool, as an argument of its initializer"""
    return multiprocessing.get_context(START_METHOD).Value(typecode, value)


def _init_worker(environment, initializer, initargs):
    """Give a worker the environment of the process which created its pool, then run the pool's initializer"""
    os.environ.clear()
    os.environ.update(environment)
    if initializer is not None:
        initializer(*initargs)


@contextmanager
def process_pool(workers, initializer=None, initargs=()):
    """ProcessPoolExecutor whose workers start from a clean process (see start_worker_server), with the environment
    and working directory of this one but none of its state: what they need beyond their jobs has to be set up by
    initializer(*initargs), run once in each worker. Functions are pickled by reference, so must be importable.
    Unlike multiprocess(ing).Pool, a worker dying (killed, or calling exit()) raises an error instead of hanging.
    If the block exits with an error, pending jobs are cancelled and running ones stopped."""
    executor = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context(START_METHOD),
        initializer=_init_worker,
        initargs=(dict(os.environ), initializer, initargs),
    )
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
