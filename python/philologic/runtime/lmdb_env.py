#!/var/lib/philologic5/philologic_env/bin/python3
"""Read-only LMDB environments shared by all the threads of a process.

An LMDB environment can only be opened once per process: py-lmdb raises "already open in this process"
when a thread opens a database that another thread has open. Searches opening the database they need
therefore go through lmdb_env(), which hands out the same environment to all concurrent users and
closes it once the last one is done, so that a reloaded database is opened afresh.
"""

import os
import threading
from contextlib import contextmanager

import lmdb

_lock = threading.Lock()
_open_envs = {}  # real path -> [environment, number of users]


@contextmanager
def lmdb_env(path):
    """Read-only environment of the LMDB database at path, shared with other threads using it"""
    path = os.path.realpath(path)
    with _lock:
        if path not in _open_envs:
            _open_envs[path] = [lmdb.open(path, readonly=True, lock=False, readahead=False), 0]
        env_users = _open_envs[path]
        env_users[1] += 1
    try:
        yield env_users[0]
    finally:
        with _lock:
            env_users[1] -= 1
            if env_users[1] == 0:
                del _open_envs[path]
                env_users[0].close()
