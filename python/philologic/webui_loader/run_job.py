"""Runs one load of philologic5-webui-loader: python -m philologic.webui_loader.run_job <job directory>

Started in a session of its own, so that the load goes on when the UI server, the browser or the SSH session go away.
philoload5 runs in a process group of its own (with its parse workers and sorts), which a cancellation (SIGTERM to
this process) terminates, then kills if it hasn't stopped after CANCEL_GRACE seconds. The exit code and whether the
load was cancelled end up in result.json, and the lock of the database is released."""

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

CANCEL_GRACE = 30


def write_json(path, data):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf8") as output:
        json.dump(data, output)
    os.replace(temporary, path)


def release_lock(lock_path, job_id):
    from philologic.webui_loader.jobs import read_lock

    lock = read_lock(lock_path)
    if isinstance(lock, dict) and lock.get("job") == job_id:
        try:
            os.remove(lock_path)
        except OSError:
            pass


def main(job_dir):
    with open(os.path.join(job_dir, "job.json"), encoding="utf8") as job_file:
        job = json.load(job_file)
    env = dict(os.environ)
    env.update(job["env"])
    cancelled = threading.Event()
    with open(os.path.join(job_dir, "load.log"), "ab") as log:
        load = subprocess.Popen(
            job["argv"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=job["cwd"],
            env=env,
            process_group=0,
        )

        def cancel(signum, frame):
            if cancelled.is_set():
                return
            cancelled.set()
            try:
                os.killpg(load.pid, signal.SIGTERM)
            except ProcessLookupError:
                return

            def kill():
                if load.poll() is None:
                    try:
                        os.killpg(load.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

            threading.Timer(CANCEL_GRACE, kill).start()

        signal.signal(signal.SIGTERM, cancel)
        signal.signal(signal.SIGINT, cancel)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        write_json(
            os.path.join(job_dir, "runner.json"),
            {"pid": os.getpid(), "load_pid": load.pid, "host": socket.gethostname(), "started": time.time()},
        )
        exit_code = load.wait()
    # Parse workers or sorts left behind by a cancelled or failed load
    try:
        os.killpg(load.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    write_json(
        os.path.join(job_dir, "result.json"),
        {"exit_code": exit_code, "cancelled": cancelled.is_set(), "ended": time.time()},
    )
    release_lock(job["lock"], job["id"])
    os._exit(0)  # without waiting for the kill timer


if __name__ == "__main__":
    main(sys.argv[1])
