"""Unit tests for the files of the loads of philologic5-webui-loader: the allowed directories, and file names which
could be read differently by philoload5 (which gets the list of files one per line)."""

import os
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.webui_loader import files
from philologic.webui_loader.files import FilesError
from philologic.webui_loader.jobs import Jobs
from philologic.webui_loader.settings import Settings
from philologic.webui_loader.uploads import UploadError, safe_relative_path

pytestmark = pytest.mark.unit


def test_names_with_line_breaks_refused(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.xml").write_text("<TEI/>", encoding="utf8")
    # corpus/x\n/etc/passwd, which philoload5 would read as the line corpus/x and the line /etc/passwd
    (corpus / "x\n" / "etc").mkdir(parents=True)
    (corpus / "x\n" / "etc" / "passwd").write_text("<TEI/>", encoding="utf8")
    with pytest.raises(FilesError, match="control characters"):
        files.resolve({"directory": str(corpus), "pattern": "*", "recursive": True}, [])
    with pytest.raises(FilesError, match="control characters"):
        files.resolve({"paths": [str(corpus / "a.xml") + "\n/etc/passwd"]}, [])
    with pytest.raises(FilesError, match="spaces"):
        files.resolve({"paths": [" " + str(corpus / "a.xml")]}, [])
    for name in ("a\nb.xml", "a\rb.xml", " a.xml", "a.xml ", "dir/\tb.xml"):
        with pytest.raises(UploadError):
            safe_relative_path(name)


def test_allowed_roots(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.xml").write_text("x", encoding="utf8")
    (root / "link").symlink_to("/etc")
    assert files.resolve({"directory": str(root), "pattern": "*.xml"}, [str(root)]) == [str(root / "a.xml")]
    with pytest.raises(FilesError):
        files.list_directory(str(root / "link"), [str(root)])
    with pytest.raises(FilesError):
        files.resolve({"paths": [str(root / "link" / "passwd")]}, [str(root)])
    assert "link" not in files.list_directory(str(root), [str(root)])["directories"]


def test_load_command(tmp_path):
    """Loads run with python -P, so that no directory of the user comes first on sys.path; in the service, from their
    own directory"""
    settings = Settings(
        mode="service",
        state_dir=str(tmp_path / "state"),
        database_root=str(tmp_path),
        url_root="https://example.org/",
        python="/bin/true",  # the runner of the load doesn't run
    )
    jobs = Jobs(settings)
    job_id = jobs.launch("db", "x = 1\n", ["/data/a.xml"], 2, cwd=str(tmp_path))
    job = __import__("json").loads(Path(jobs.directory, job_id, "job.json").read_text(encoding="utf8"))
    assert job["argv"][1:4] == ["-P", "-m", "philologic.loadtime"]
    assert job["cwd"] == os.path.join(jobs.directory, job_id)
    assert job["env"]["PYTHONSAFEPATH"] == "1"


def test_previews_isolated_and_timed(tmp_path):
    from philologic.webui_loader import previews

    text = tmp_path / "a.xml"
    text.write_text(
        "<TEI><teiHeader><title>T</title></teiHeader><text><p>" + "a" * 40 + "!</p></text></TEI>", encoding="utf8"
    )
    result = previews.run_isolated(previews.header_preview, [str(text)], {"doc_xpaths": {"title": [".//title"]}}, 5)
    assert result["rows"][0]["metadata"]["title"] == "T"
    # A regular expression which backtracks for ever stops at the timeout
    start = __import__("time").time()
    with pytest.raises(previews.PreviewTimeout):
        previews.run_isolated(previews.tokens_preview, str(text), {"token_regex": "(a|a)+$"}, timeout=3)
    assert __import__("time").time() - start < 20
