"""Uploads of the files of a load, for users of the service who have them on their own machine: a folder of files, or
one zip or tar archive (the easiest way for thousands of files), sent in chunks which can resume after a dropped
connection. Each upload goes in the upload area of its user, whose size is limited by a quota in the service (not on
your own machine, where the UI runs as you, and the upload area is in its state directory); archives are
extracted with checks (no absolute paths, no .., no links, and their real size counted as they are extracted)."""

import json
import os
import re
import secrets
import shutil
import stat
import tarfile
import threading
import time
import zipfile

MAX_CHUNK = 8 * 1024**2
MAX_PATH = 1024
# Bytes and files received by each upload being received, kept here rather than counted again at each chunk, and a
# lock per upload so that chunks sent at once can't exceed what the upload announced
_received = {}
_locks = {}
_locks_lock = threading.Lock()


def upload_lock(path):
    with _locks_lock:
        return _locks.setdefault(path, threading.Lock())


UPLOAD_ID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")
ARCHIVES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


class UploadError(Exception):
    """A refused upload: the message can be shown to the user"""


def safe_relative_path(path):
    """A relative path inside an upload, or raises UploadError"""
    if not isinstance(path, str) or not path or len(path) > MAX_PATH or "\x00" in path:
        raise UploadError("invalid file name")
    parts = path.replace("\\", "/").split("/")
    if path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        raise UploadError(f"invalid file name: {path}")
    if any(part != part.strip() or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in part) for part in parts):
        raise UploadError(f"file names with control characters or spaces around them can't be uploaded: {path!r}")
    if any(part.startswith(".") for part in parts):
        raise UploadError(f"hidden files are not uploaded: {path}")
    return os.path.join(*parts)


def skipped(path):
    """Hidden files and folders, and the metadata macOS adds to zip archives, which aren't extracted"""
    parts = path.replace("\\", "/").split("/")
    return parts[0] == "__MACOSX" or any(part.startswith(".") and part not in (".", "..") for part in parts)


def directory_size(path):
    total, count = 0, 0
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
                count += 1
            except OSError:
                pass
    return total, count


class Uploads:
    """The upload area of one user"""

    def __init__(self, settings, user):
        self.settings = settings
        self.directory = os.path.join(settings.upload_dir, user)
        os.makedirs(self.directory, mode=0o750, exist_ok=True)

    def usage(self):
        size, count = directory_size(self.directory)
        return {"size": size, "files": count, "quota": self.settings.upload_quota}

    def upload_dir(self, upload_id):
        if not UPLOAD_ID.match(upload_id or ""):
            raise UploadError("no such upload")
        path = os.path.join(self.directory, upload_id)
        if not os.path.isfile(os.path.join(path, "upload.json")):
            raise UploadError("no such upload")
        return path

    def meta(self, upload_id):
        with open(os.path.join(self.upload_dir(upload_id), "upload.json"), encoding="utf8") as meta_file:
            return json.load(meta_file)

    def write_meta(self, upload_id, meta):
        path = os.path.join(self.upload_dir(upload_id), "upload.json")
        with open(path + ".tmp", "w", encoding="utf8") as meta_file:
            json.dump(meta, meta_file)
        os.replace(path + ".tmp", path)

    def create(self, name, kind, size, file_count=1):
        """A new upload: kind "archive" (one archive file, named name) or "files" (a folder, named name, of
        file_count files). size is the total size announced, checked against the quota."""
        if kind not in ("archive", "files"):
            raise UploadError("an upload is an archive or files")
        if not isinstance(size, int) or size < 0 or not isinstance(file_count, int) or file_count < 1:
            raise UploadError("invalid size")
        name = os.path.basename(str(name or "upload"))[:200] or "upload"
        if kind == "archive" and not name.lower().endswith(ARCHIVES):
            raise UploadError(f"archives are {', '.join(ARCHIVES)} files")
        usage = self.usage()
        if self.settings.upload_quota is not None and usage["size"] + size > self.settings.upload_quota:
            raise UploadError(
                f"this upload would exceed your quota: {usage['size'] + size} of {self.settings.upload_quota} bytes"
            )
        if usage["files"] + file_count > self.settings.max_upload_files:
            raise UploadError(f"more than {self.settings.max_upload_files} files in your uploads")
        upload_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
        path = os.path.join(self.directory, upload_id)
        os.makedirs(os.path.join(path, "files"), mode=0o750)
        meta = {
            "id": upload_id,
            "name": name,
            "kind": kind,
            "size": size,
            "file_count": file_count,
            "state": "receiving",
            "created": time.time(),
            "error": None,
        }
        with open(os.path.join(path, "upload.json"), "w", encoding="utf8") as meta_file:
            json.dump(meta, meta_file)
        return meta

    def target(self, upload_id, path):
        meta = self.meta(upload_id)
        if meta["state"] != "receiving":
            raise UploadError("this upload is complete")
        if meta["kind"] == "archive":
            return meta, os.path.join(self.upload_dir(upload_id), "archive")
        return meta, os.path.join(self.upload_dir(upload_id), "files", safe_relative_path(path))

    def received(self, upload_id):
        """Bytes received of each file (of the archive), to resume an upload"""
        meta = self.meta(upload_id)
        upload_dir = self.upload_dir(upload_id)
        if meta["kind"] == "archive":
            path = os.path.join(upload_dir, "archive")
            return {meta["name"]: os.path.getsize(path) if os.path.exists(path) else 0}
        files_dir = os.path.join(upload_dir, "files")
        received = {}
        for dirpath, dirnames, filenames in os.walk(files_dir):
            for name in filenames:
                full = os.path.join(dirpath, name)
                received[os.path.relpath(full, files_dir)] = os.path.getsize(full)
        return received

    def write_chunk(self, upload_id, path, offset, stream, length):
        """Write a chunk of a file at an offset, which must be the size already received (so that a resumed upload
        starts where the last one stopped). An upload can't receive more bytes or files than it announced, which were
        checked against the quota when it was created."""
        if not isinstance(length, int) or length < 0 or length > MAX_CHUNK:
            raise UploadError(f"chunks are at most {MAX_CHUNK} bytes")
        meta, target = self.target(upload_id, path)
        upload_dir = self.upload_dir(upload_id)
        with upload_lock(upload_dir):
            if upload_dir not in _received:
                size, count = directory_size(os.path.join(upload_dir, "files"))
                archive = os.path.join(upload_dir, "archive")
                if os.path.exists(archive):
                    size, count = size + os.path.getsize(archive), count + 1
                _received[upload_dir] = [size, count]
            received = _received[upload_dir]
            os.makedirs(os.path.dirname(target), exist_ok=True)
            exists = os.path.exists(target)
            current = os.path.getsize(target) if exists else 0
            if offset != current:
                raise UploadError(f"expected offset {current}")
            if received[0] + length > meta["size"]:
                raise UploadError("more bytes than the upload announced")
            if not exists and received[1] + 1 > meta["file_count"]:
                raise UploadError("more files than the upload announced")
            written = 0
            with open(target, "ab") as output:
                while written < length:
                    data = stream.read(min(1024**2, length - written))
                    if not data:
                        break
                    output.write(data)
                    written += len(data)
            received[0] += written
            if not exists:
                received[1] += 1
        if written != length:
            raise UploadError("the chunk was cut short")
        return current + written

    def complete(self, upload_id):
        """End an upload: an archive is extracted (checked member by member)"""
        meta = self.meta(upload_id)
        if meta["state"] != "receiving":
            raise UploadError("this upload is complete")
        upload_dir = self.upload_dir(upload_id)
        if meta["kind"] == "archive":
            archive = os.path.join(upload_dir, "archive")
            if not os.path.exists(archive):
                raise UploadError("the archive wasn't received")
            try:
                self.extract(archive, meta["name"], os.path.join(upload_dir, "files"))
            except UploadError as error:
                shutil.rmtree(os.path.join(upload_dir, "files"), ignore_errors=True)
                os.makedirs(os.path.join(upload_dir, "files"))
                meta.update(state="error", error=str(error))
                self.write_meta(upload_id, meta)
                raise
            finally:
                if os.path.exists(archive):
                    os.remove(archive)
        meta["state"] = "ready"
        meta["size"], meta["file_count"] = directory_size(os.path.join(upload_dir, "files"))
        _received.pop(upload_dir, None)
        self.write_meta(upload_id, meta)
        return self.status(upload_id)

    def limits(self):
        usage = self.usage()
        quota = self.settings.upload_quota
        size_left = float("inf") if quota is None else quota - usage["size"]  # no quota on your own machine
        return size_left, self.settings.max_upload_files - usage["files"]

    def extract(self, archive, name, destination):
        size_left, files_left = self.limits()
        if name.lower().endswith(".zip"):
            self.extract_zip(archive, destination, size_left, files_left)
        else:
            self.extract_tar(archive, destination, size_left, files_left)

    @staticmethod
    def copy_limited(source, target, size_left):
        written = 0
        with open(target, "wb") as output:
            while True:
                data = source.read(1024**2)
                if not data:
                    break
                written += len(data)
                if written > size_left:
                    raise UploadError("the archive is larger than your quota once extracted")
                output.write(data)
        return written

    def extract_zip(self, archive, destination, size_left, files_left):
        try:
            with zipfile.ZipFile(archive) as zip_file:
                members = [member for member in zip_file.infolist() if not member.is_dir()]
                if len(members) > files_left:
                    raise UploadError("the archive has more files than your uploads can have")
                for member in members:
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise UploadError(f"the archive has a link: {member.filename}")
                    if skipped(member.filename):
                        continue
                    target = os.path.join(destination, safe_relative_path(member.filename))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zip_file.open(member) as source:
                        size_left -= self.copy_limited(source, target, size_left)
        except (zipfile.BadZipFile, NotImplementedError, EOFError, OSError) as error:
            raise UploadError(f"the archive can't be extracted: {error}") from error

    def extract_tar(self, archive, destination, size_left, files_left):
        try:
            with tarfile.open(archive) as tar_file:
                count = 0
                for member in tar_file:
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise UploadError(f"the archive has something else than files and folders: {member.name}")
                    if skipped(member.name):
                        continue
                    count += 1
                    if count > files_left:
                        raise UploadError("the archive has more files than your uploads can have")
                    target = os.path.join(destination, safe_relative_path(member.name))
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    source = tar_file.extractfile(member)
                    size_left -= self.copy_limited(source, target, size_left)
        except (tarfile.TarError, EOFError, OSError) as error:
            raise UploadError(f"the archive can't be extracted: {error}") from error

    def status(self, upload_id):
        meta = self.meta(upload_id)
        meta["path"] = os.path.join(self.upload_dir(upload_id), "files")
        if meta["state"] == "receiving":
            meta["received"] = sum(self.received(upload_id).values())
        return meta

    def list(self):
        uploads = []
        for name in sorted(os.listdir(self.directory), reverse=True):
            if UPLOAD_ID.match(name):
                try:
                    uploads.append(self.status(name))
                except (UploadError, OSError, ValueError):
                    continue
        return {"uploads": uploads, "usage": self.usage()}

    def delete(self, upload_id):
        shutil.rmtree(self.upload_dir(upload_id))
