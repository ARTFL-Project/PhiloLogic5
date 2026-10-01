"""Loads launched by philologic5-webui-loader. Each one runs philoload5 (python -m philologic.loadtime), the same command
as by hand, detached from the UI (see run_job.py). Its directory, in the loads directory of the state of the UI,
keeps its command, load config, file list, log and stages, so that it can be followed after the UI or the SSH session
went away, and run again by hand. A lock file in database_root prevents two loads of the same database at once."""

import datetime
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time

STAGES = ("copy_files", "metadata", "parse", "merge", "count_words", "index", "sql", "post_filters", "finish")
MAX_LOG_CHUNK = 256 * 1024
TQDM = re.compile(r"(?P<desc>[^\r\n|]*?):?\s*(?P<percent>\d+)%\|[^|\r\n]*\|\s*(?P<done>\d+)/(?P<total>\d+)")
REMOVED_FILES = re.compile(r"^File (?P<name>.+?): (?P<cause>invalid characters|no TEI header|invalid XML)$", re.M)
APPLICATION_URL = re.compile(r"^Application viewable at (?P<url>\S+)", re.M)
JOB_ID = re.compile(r"^[A-Za-z0-9._-]+-\d{8}-\d{6}-[0-9a-f]{6}$")


class JobError(Exception):
    """A load which can't be launched, found or cancelled"""


def read_json(path, default=None):
    try:
        with open(path, encoding="utf8") as json_file:
            return json.load(json_file)
    except (OSError, ValueError):
        return default


def write_json(path, data, mode=0o600):
    temporary = f"{path}.tmp"
    with open(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "w", encoding="utf8") as output:
        json.dump(data, output, indent=2)
    os.replace(temporary, path)


def process_alive(pid, job_dir=None):
    """Whether a process runs, and is the runner of this job when this can be checked (/proc)"""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    if job_dir is not None and os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as cmdline:
                return job_dir.encode() in cmdline.read()
        except OSError:
            return False
    return True


def read_lock(lock_path):
    """The content of a lock, read without following a link, from a regular file only, and at most 4 KB of it (a
    member of the group of database_root could put anything in its place), or None"""
    try:
        handle = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(handle).st_mode):
            return None
        return json.loads(os.read(handle, 4096))
    except ValueError:
        return None
    finally:
        os.close(handle)


def valid_lock(lock):
    return (
        isinstance(lock, dict)
        and isinstance(lock.get("pid"), int)
        and not isinstance(lock.get("pid"), bool)
        and 0 < lock["pid"] < 2**31
        and isinstance(lock.get("host"), str)
    )


def update_lock(lock_path, data):
    """Rewrite the lock taken by this process, without following a link put in its place (database_root is writable
    by a group)"""
    handle = os.open(lock_path, os.O_WRONLY | os.O_NOFOLLOW)
    with os.fdopen(handle, "w", encoding="utf8") as lock_file:
        lock_file.truncate()
        json.dump(data, lock_file)


def complete_utf8(data):
    """Length of data without a UTF-8 character cut at its end"""
    for back in range(1, min(4, len(data)) + 1):
        byte = data[-back]
        if byte & 0xC0 != 0x80:  # the first byte of a character
            length = 1 if byte < 0x80 else 2 if byte >> 5 == 0b110 else 3 if byte >> 4 == 0b1110 else 4
            return len(data) if length <= back else len(data) - back
    return len(data)


def compact_log(text, rewrites=False):
    """Log text with the progress bars which overwrite each other (carriage returns) reduced to their last state. With
    rewrites, that state keeps its carriage return: a part of the log can start in a line which the page already shows
    (a progress bar), which it must then rewrite rather than add to."""
    marker = "\r" if rewrites else ""
    return "\n".join(marker + line.rsplit("\r", 1)[1] if "\r" in line else line for line in text.split("\n"))


def load_arguments(dbname, config_path, files_path, cores, header, file_type, bibliography, overwrite):
    """Arguments of philoload5 for a load"""
    arguments = ["-c", str(cores), "-H", header, "-t", file_type]
    if bibliography:
        arguments += ["-b", bibliography]
    if overwrite:
        arguments.append("-D")
    return arguments + ["-l", config_path, "-F", dbname, files_path]


class Jobs:
    """The loads of one UI server: its loads directory, the database root and the settings to run philoload5"""

    def __init__(self, settings):
        self.settings = settings
        self.directory = settings.loads_dir
        os.makedirs(self.directory, mode=0o700, exist_ok=True)

    def lock_path(self, dbname):
        return os.path.join(self.settings.database_root, f".{dbname}.loading")

    def running_load(self, dbname):
        """The lock of a running load of a database, or None (a lock left by a dead load, or which isn't one, is
        removed)"""
        lock_path = self.lock_path(dbname)
        if not os.path.lexists(lock_path):
            return None
        lock = read_lock(lock_path)
        if not valid_lock(lock) and time.time() - os.lstat(lock_path).st_mtime < 5:
            return {"user": None, "job": None}  # being written
        if not valid_lock(lock) or (lock["host"] == socket.gethostname() and not process_alive(lock["pid"])):
            try:
                os.remove(self.lock_path(dbname))
            except OSError:
                pass
            return None
        return lock

    def launch(
        self,
        dbname,
        load_config_text,
        files,
        cores,
        header="tei",
        file_type="xml",
        bibliography=None,
        overwrite=False,
        cwd=None,
        user=None,
        source=None,
    ):
        """Launch a load: returns its id. Raises JobError if the database is being loaded."""
        job_id = f"{dbname}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        lock_path = self.lock_path(dbname)
        self.take_lock(dbname, job_id, user)  # before anything else: a refused load leaves nothing behind
        job_dir = os.path.join(self.directory, job_id)
        try:
            os.makedirs(job_dir, mode=0o700)
            config_path = os.path.join(job_dir, "load_config.py")
            with open(config_path, "w", encoding="utf8") as config_file:
                config_file.write(load_config_text)
            files_path = os.path.join(job_dir, "files.txt")
            with open(files_path, "w", encoding="utf8") as files_file:
                files_file.write("".join(f"{path}\n" for path in files))
            arguments = load_arguments(
                dbname, config_path, files_path, cores, header, file_type, bibliography, overwrite
            )
            job = {
                "id": job_id,
                "dbname": dbname,
                "user": user,
                "created": time.time(),
                # -P: no directory of the user (the working directory) first on sys.path
                "argv": [self.settings.python, "-P", "-m", "philologic.loadtime", *arguments],
                "command": shlex.join(["philoload5", *arguments]),
                # The directory of the load config (for its relative paths), on your own machine only
                "cwd": cwd if cwd and os.path.isdir(cwd) and not self.settings.service else job_dir,
                "env": {
                    "PHILOLOGIC_CONFIG": self.settings.global_config,
                    "PHILOLOGIC_PROGRESS_FILE": os.path.join(job_dir, "progress.jsonl"),
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONSAFEPATH": "1",
                },
                "lock": lock_path,
                "files": len(files),
                "cores": cores,
                "header": header,
                "file_type": file_type,
                "bibliography": bibliography,
                "overwrite": overwrite,
                "source": source,
                "url": self.settings.url_root.rstrip("/") + "/" + dbname,
            }
            write_json(os.path.join(job_dir, "job.json"), job)
            with open(os.path.join(job_dir, "runner.log"), "ab") as runner_log:
                runner = subprocess.Popen(
                    [self.settings.python, "-P", "-m", "philologic.webui_loader.run_job", job_dir],
                    stdin=subprocess.DEVNULL,
                    stdout=runner_log,
                    stderr=subprocess.STDOUT,
                    cwd=job_dir,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as error:
            os.remove(lock_path)
            shutil.rmtree(job_dir, ignore_errors=True)
            raise JobError(f"the load could not be started: {error}") from error
        update_lock(lock_path, self.lock_data(job_id, user, runner.pid))
        threading.Thread(target=runner.wait, daemon=True).start()  # reaped when it ends, if the server still runs
        return job_id

    @staticmethod
    def lock_data(job_id, user, pid):
        return {"job": job_id, "user": user, "pid": pid, "host": socket.gethostname(), "time": time.time()}

    def take_lock(self, dbname, job_id, user):
        lock_path = self.lock_path(dbname)
        for attempt in range(2):
            try:
                handle = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                lock = self.running_load(dbname)
                if lock is not None:
                    raise JobError(
                        f"{dbname} is already being loaded (by {lock.get('user') or 'someone'}, job {lock.get('job')})"
                    )
                continue  # the lock of a dead load was removed
            with os.fdopen(handle, "w", encoding="utf8") as lock_file:
                json.dump(self.lock_data(job_id, user, os.getpid()), lock_file)
            return
        raise JobError(f"the lock of {dbname} can't be taken ({lock_path})")

    def job_dir(self, job_id):
        if not JOB_ID.match(job_id or ""):
            raise JobError("no such load")
        job_dir = os.path.join(self.directory, job_id)
        if not os.path.isfile(os.path.join(job_dir, "job.json")):
            raise JobError("no such load")
        return job_dir

    def status(self, job_id, details=True):
        """State of a load: running, succeeded, failed, cancelled or interrupted (its runner died), with its stages
        and the progress of the current one, and once done the URL of the database and the files left out"""
        job_dir = self.job_dir(job_id)
        job = read_json(os.path.join(job_dir, "job.json"))
        runner = read_json(os.path.join(job_dir, "runner.json"))
        result = read_json(os.path.join(job_dir, "result.json"))
        if result is not None:
            if result["cancelled"]:
                state = "cancelled"
            else:
                state = "succeeded" if result["exit_code"] == 0 else "failed"
        elif (
            runner is not None
            and runner.get("host") == socket.gethostname()
            and not process_alive(runner["pid"], job_dir)
        ):
            state = "interrupted"
        elif runner is None and time.time() - job["created"] > 60:
            state = "interrupted"
        else:
            state = "running"
        status = {
            key: job.get(key)
            for key in (
                "id",
                "dbname",
                "user",
                "created",
                "command",
                "files",
                "cores",
                "header",
                "file_type",
                "bibliography",
                "overwrite",
                "source",
                "url",
            )
        }
        status.update(
            {
                "state": state,
                "started": runner.get("started") if runner else None,
                "ended": result.get("ended") if result else None,
                "exit_code": result.get("exit_code") if result else None,
                "cancel_requested": os.path.exists(os.path.join(job_dir, "cancel_requested")),
            }
        )
        if details:
            status["stages"] = self.stages(job_dir, state)
            tail = self.log_tail(job_dir)
            status["progress"] = self.progress(tail) if state == "running" else None
            if state != "running":
                log = self.read_log(job_dir)
                status["removed_files"] = [match.groupdict() for match in REMOVED_FILES.finditer(log)][:1000]
                url = APPLICATION_URL.search(log)
                status["application_url"] = url.group("url") if url and state == "succeeded" else None
        return status

    @staticmethod
    def stages(job_dir, state):
        """Each stage of the load: pending, running, done or failed, with its start and end times"""
        stages = {name: {"name": name, "state": "pending", "start": None, "end": None} for name in STAGES}
        try:
            with open(os.path.join(job_dir, "progress.jsonl"), encoding="utf8") as progress:
                for line in progress:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    stage = stages.get(event.get("stage"))
                    if stage is None:
                        continue
                    if event.get("event") == "start":
                        stage.update(state="running", start=event["time"])
                    elif event.get("event") == "end":
                        stage.update(state="done", end=event["time"])
        except OSError:
            pass
        if state != "running":
            for stage in stages.values():
                if stage["state"] == "running":
                    stage["state"] = "failed" if state != "cancelled" else "cancelled"
        return list(stages.values())

    @staticmethod
    def log_tail(job_dir, size=16384):
        try:
            with open(os.path.join(job_dir, "load.log"), "rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - size))
                return log.read().decode("utf8", errors="replace")
        except OSError:
            return ""

    @staticmethod
    def read_log(job_dir):
        try:
            with open(os.path.join(job_dir, "load.log"), "rb") as log:
                return compact_log(log.read().decode("utf8", errors="replace"))
        except OSError:
            return ""

    @staticmethod
    def progress(tail):
        """The last progress bar of the log, if it is the last thing written"""
        last = re.split(r"[\r\n]", tail.rstrip("\r\n"))[-1] if tail else ""
        match = TQDM.search(last)
        if match is None:
            return None
        return {
            "description": match.group("desc").strip(),
            "percent": int(match.group("percent")),
            "done": int(match.group("done")),
            "total": int(match.group("total")),
        }

    def log(self, job_id, offset=0):
        """Part of the log of a load from a byte offset, with the offset where the next part starts"""
        job_dir = self.job_dir(job_id)
        try:
            with open(os.path.join(job_dir, "load.log"), "rb") as log:
                log.seek(offset)
                data = log.read(MAX_LOG_CHUNK)
        except OSError:
            data = b""
        more = len(data) == MAX_LOG_CHUNK
        data = data[: complete_utf8(data)]
        text = compact_log(data.decode("utf8", errors="replace"), rewrites=True)
        return {"text": text, "offset": offset + len(data), "more": more}

    def list(self, user=None):
        """The loads, most recent first (those of one user if given)"""
        jobs = []
        for name in os.listdir(self.directory):
            if not JOB_ID.match(name):
                continue
            try:
                status = self.status(name, details=False)
            except JobError:
                continue
            if user is None or status["user"] == user:
                jobs.append(status)
        jobs.sort(key=lambda job: job["created"], reverse=True)
        return jobs

    def cancel(self, job_id):
        job_dir = self.job_dir(job_id)
        status = self.status(job_id, details=False)
        if status["state"] != "running":
            raise JobError("this load isn't running")
        runner = read_json(os.path.join(job_dir, "runner.json"))
        if runner is None:
            raise JobError("this load is starting: cancel it again in a moment")
        with open(os.path.join(job_dir, "cancel_requested"), "w", encoding="utf8"):
            pass
        try:
            os.kill(runner["pid"], signal.SIGTERM)
        except ProcessLookupError:
            # The runner is gone: stop the load itself
            try:
                os.killpg(runner["load_pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass

    def load_config(self, job_id):
        with open(os.path.join(self.job_dir(job_id), "load_config.py"), encoding="utf8") as config_file:
            return config_file.read()
