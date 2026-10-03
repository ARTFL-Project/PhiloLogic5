"""Unit tests for the uploads of the philologic5-webui-loader service: chunks which resume, quotas, and archives which
can't write outside their upload (absolute paths, .., links) or grow beyond the quota once extracted."""

import io
import os
import stat
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.webui_loader.settings import Settings
from philologic.webui_loader.uploads import UploadError, Uploads, safe_relative_path

pytestmark = pytest.mark.unit


@pytest.fixture
def uploads(tmp_path):
    settings = Settings(
        mode="service",
        state_dir=str(tmp_path / "state"),
        database_root=str(tmp_path / "dbs"),
        upload_dir=str(tmp_path / "uploads"),
        upload_quota=10 * 1024**2,
        max_upload_files=100,
    )
    return Uploads(settings, "alice")


def send(uploads, upload_id, path, data, offset=0):
    return uploads.write_chunk(upload_id, path, offset, io.BytesIO(data), len(data))


def test_files_upload_resumes(uploads):
    meta = uploads.create("corpus", "files", 11, 2)
    send(uploads, meta["id"], "texts/a.xml", b"hello")
    assert uploads.received(meta["id"]) == {"texts/a.xml": 5}
    with pytest.raises(UploadError, match="expected offset 5"):
        send(uploads, meta["id"], "texts/a.xml", b" world")
    assert send(uploads, meta["id"], "texts/a.xml", b" world", offset=5) == 11
    status = uploads.complete(meta["id"])
    assert status["state"] == "ready" and status["file_count"] == 1 and status["size"] == 11
    assert Path(status["path"], "texts", "a.xml").read_bytes() == b"hello world"
    with pytest.raises(UploadError, match="complete"):
        send(uploads, meta["id"], "b.xml", b"x")


def test_paths_stay_inside(uploads):
    for path in ("../evil", "/etc/passwd", "a/../../b", "a//b", ".hidden", "a\x00b", ""):
        with pytest.raises(UploadError):
            safe_relative_path(path)
    meta = uploads.create("corpus", "files", 1, 1)
    with pytest.raises(UploadError):
        send(uploads, meta["id"], "../../outside.xml", b"x")


def test_quota(uploads):
    with pytest.raises(UploadError, match="quota"):
        uploads.create("big", "files", 11 * 1024**2, 1)
    with pytest.raises(UploadError, match="more than"):
        uploads.create("many", "files", 10, 101)
    meta = uploads.create("corpus", "files", 10, 1)
    with pytest.raises(UploadError, match="at most"):
        uploads.write_chunk(meta["id"], "a.xml", 0, io.BytesIO(b""), 9 * 1024**2)


def upload_archive(uploads, name, data):
    meta = uploads.create(name, "archive", len(data))
    send(uploads, meta["id"], name, data)
    return meta["id"]


def zip_bytes(members, symlink=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "/etc/passwd")
    return buffer.getvalue()


def tar_bytes(members, symlink=None, device=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            archive.addfile(info)
        if device:
            info = tarfile.TarInfo(device)
            info.type = tarfile.CHRTYPE
            archive.addfile(info)
    return buffer.getvalue()


def test_zip_archive(uploads):
    data = zip_bytes(
        {"corpus/a.xml": b"<TEI/>", "corpus/b.xml": b"<TEI/>", "__MACOSX/._a.xml": b"x", ".DS_Store": b"x"}
    )
    status = uploads.complete(upload_archive(uploads, "corpus.zip", data))
    assert status["state"] == "ready" and status["file_count"] == 2
    assert sorted(os.listdir(Path(status["path"], "corpus"))) == ["a.xml", "b.xml"]
    assert not Path(status["path"]).parent.joinpath("archive").exists()


@pytest.mark.parametrize(
    "data, message",
    [
        (zip_bytes({"../evil.xml": b"x"}), "invalid file name"),
        (zip_bytes({"/abs.xml": b"x"}), "invalid file name"),
        (zip_bytes({"a.xml": b"x"}, symlink="link"), "link"),
        (zip_bytes({"bomb.xml": b"\0" * (11 * 1024**2)}), "larger than your quota"),
    ],
)
def test_bad_zip_archives(uploads, data, message):
    upload_id = upload_archive(uploads, "bad.zip", data)
    with pytest.raises(UploadError, match=message):
        uploads.complete(upload_id)
    status = uploads.status(upload_id)
    assert status["state"] == "error" and os.listdir(status["path"]) == []


@pytest.mark.parametrize(
    "data, message",
    [
        (tar_bytes({"../evil.xml": b"x"}), "invalid file name"),
        (tar_bytes({"a.xml": b"x"}, symlink="link"), "something else than files"),
        (tar_bytes({"a.xml": b"x"}, device="dev"), "something else than files"),
    ],
)
def test_bad_tar_archives(uploads, data, message):
    upload_id = upload_archive(uploads, "bad.tar.gz", data)
    with pytest.raises(UploadError, match=message):
        uploads.complete(upload_id)


def test_tar_archive_and_list(uploads):
    status = uploads.complete(upload_archive(uploads, "corpus.tgz", tar_bytes({"a.xml": b"<TEI/>"})))
    assert status["file_count"] == 1
    listing = uploads.list()
    assert [upload["id"] for upload in listing["uploads"]] == [status["id"]]
    uploads.delete(status["id"])
    assert uploads.list()["uploads"] == []
