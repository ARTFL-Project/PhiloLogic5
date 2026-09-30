"""Unit tests for the web config page of philologic5-webui-loader: saved edits must be read by the runtime as given,
change nothing else in web_config.cfg, and never touch access_control or access_file."""

import os
import stat
import sys
from pathlib import Path

import pytest
from black import FileMode, format_str

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.Config import MakeWebConfig
from philologic.webui_loader import web_config_io
from philologic.webui_loader.web_config_io import WebConfigError

pytestmark = pytest.mark.unit


@pytest.fixture
def db_path(tmp_path):
    """A database with the web_config.cfg the loader writes (Loader.write_web_config)"""
    data = tmp_path / "db" / "data"
    data.mkdir(parents=True)
    config_path = data / "web_config.cfg"
    web_config = MakeWebConfig(str(config_path), dbname="Test", metadata=["author", "title", "year"])
    config_path.write_text(format_str(str(web_config), mode=FileMode()), encoding="utf8")
    (data / "db.locals.py").write_text('metadata_fields = ["author", "title", "year", "head"]\n', encoding="utf8")
    return str(tmp_path / "db")


def runtime_config(db_path):
    return MakeWebConfig(os.path.join(db_path, "data", "web_config.cfg"))


def test_read(db_path):
    config = web_config_io.read(db_path)
    assert config["writable"] and config["error"] is None
    assert config["values"]["dbname"] == "Test"
    assert config["values"]["concordance_citation"] == [
        dict(citation) for citation in runtime_config(db_path)["concordance_citation"]
    ]
    assert config["metadata_fields"] == ["author", "title", "year", "head"]
    assert config["code"] == {} and config["other"] == {}


def test_save_changes_only_the_edited_assignments(db_path):
    config = web_config_io.read(db_path)
    path = Path(db_path) / "data" / "web_config.cfg"
    before = path.read_text(encoding="utf8").splitlines()
    concordance_citation = config["values"]["concordance_citation"][:3]
    changes = {
        "dbname": "Renamed",
        "facets": ["author", "year"],
        "concordance_citation": concordance_citation,
        "query_parser_regex": config["values"]["query_parser_regex"] + [["«", '"']],
        "metadata_input_style": {"year": "int"},
    }
    backup = web_config_io.save(db_path, changes, config["hash"])
    assert Path(backup).read_text(encoding="utf8").splitlines() == before
    after = path.read_text(encoding="utf8")
    # The citations are still written as references to the citations variable
    assert 'citations["author"]' in after.split("concordance_citation =")[1].split("\n\n")[0]
    runtime = runtime_config(db_path)
    assert runtime["dbname"] == "Renamed"
    assert runtime["facets"] == ["author", "year"]
    assert runtime["concordance_citation"] == concordance_citation
    assert runtime["query_parser_regex"][-1] == ("«", '"')  # tuples, as the runtime expects
    assert runtime["metadata_input_style"] == {"year": "int"}
    # Other lines are left as they were
    changed_lines = set(after.splitlines()) - set(before)
    for line in changed_lines:
        assert any(key in line for key in ("dbname", "facets", "citations[", '"', "]", "}", "(", "{")), line
    assert web_config_io.read(db_path)["values"]["bibliography_citation"] == config["values"]["bibliography_citation"]


def test_nothing_to_save(db_path):
    config = web_config_io.read(db_path)
    assert web_config_io.save(db_path, {"dbname": "Test"}, config["hash"]) is None


def test_access_options_never_change(db_path):
    config = web_config_io.read(db_path)
    with pytest.raises(WebConfigError, match="access_control"):
        web_config_io.save(db_path, {"access_control": True}, config["hash"])
    with pytest.raises(WebConfigError, match="access_file"):
        web_config_io.save(db_path, {"access_file": "/etc/passwd", "dbname": "x"}, config["hash"])
    assert runtime_config(db_path)["dbname"] == "Test"


def test_invalid_values(db_path):
    config = web_config_io.read(db_path)
    with pytest.raises(WebConfigError, match="search_reports"):
        web_config_io.save(db_path, {"search_reports": ["concordance", "nonsense"]}, config["hash"])
    with pytest.raises(WebConfigError, match="concordance_length"):
        web_config_io.save(db_path, {"concordance_length": "300"}, config["hash"])
    with pytest.raises(WebConfigError, match="unknown"):
        web_config_io.save(db_path, {"no_such_option": 1}, config["hash"])


def test_concurrent_change_detected(db_path):
    config = web_config_io.read(db_path)
    path = Path(db_path) / "data" / "web_config.cfg"
    path.write_text(path.read_text(encoding="utf8") + "\n# edited by hand\n", encoding="utf8")
    with pytest.raises(WebConfigError, match="changed since"):
        web_config_io.save(db_path, {"dbname": "x"}, config["hash"])


def test_code_is_kept_and_not_editable(db_path):
    path = Path(db_path) / "data" / "web_config.cfg"
    path.write_text(path.read_text(encoding="utf8") + '\nfacets = sorted(["title", "author"])\n', encoding="utf8")
    config = web_config_io.read(db_path)
    assert config["code"] == {"facets": 'sorted(["title", "author"])'}
    with pytest.raises(WebConfigError, match="code"):
        web_config_io.save(db_path, {"facets": ["year"]}, config["hash"])
    web_config_io.save(db_path, {"dbname": "Kept"}, config["hash"])
    assert runtime_config(db_path)["facets"] == ["author", "title"]


def test_option_missing_from_file_is_added(db_path):
    path = Path(db_path) / "data" / "web_config.cfg"
    text = path.read_text(encoding="utf8")
    start = text.index("concordance_citation = [")
    end = text.index("]\n", start) + 2
    path.write_text(text[:start] + text[end:], encoding="utf8")
    config = web_config_io.read(db_path)
    citation = config["values"]["concordance_citation"][:2]
    web_config_io.save(db_path, {"concordance_citation": citation, "logo": "logo.png"}, config["hash"])
    after = path.read_text(encoding="utf8")
    assert after.rstrip().endswith('concordance_citation = [citations["author"], citations["title"]]') or (
        'citations["author"]' in after.split("concordance_citation =")[-1]
    )
    assert runtime_config(db_path)["concordance_citation"] == citation
    assert runtime_config(db_path)["logo"] == "logo.png"


def test_file_without_citations_uses_the_defaults(db_path):
    path = Path(db_path) / "data" / "web_config.cfg"
    path.write_text('dbname = "Bare"\n', encoding="utf8")
    config = web_config_io.read(db_path)
    citation = config["values"]["bibliography_citation"][:1]
    web_config_io.save(db_path, {"bibliography_citation": citation}, config["hash"])
    assert 'citations["author"]' in path.read_text(encoding="utf8")
    assert runtime_config(db_path)["bibliography_citation"] == citation


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anything")
def test_not_writable(db_path):
    path = Path(db_path) / "data" / "web_config.cfg"
    config = web_config_io.read(db_path)
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP)
    try:
        read = web_config_io.read(db_path)
        assert not read["writable"]
        assert "web_config.cfg belongs to" in read["reason"] and "not group-writable" in read["reason"]
        with pytest.raises(WebConfigError, match="belongs to"):
            web_config_io.save(db_path, {"dbname": "x"}, config["hash"])
    finally:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def test_mode_kept(db_path):
    path = Path(db_path) / "data" / "web_config.cfg"
    os.chmod(path, 0o664)
    config = web_config_io.read(db_path)
    web_config_io.save(db_path, {"dbname": "x"}, config["hash"])
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o664
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".web_config")]


def test_code_of_the_file_never_changes(db_path):
    path = Path(db_path) / "data" / "web_config.cfg"
    path.write_text(path.read_text(encoding="utf8") + "\nimport os\n", encoding="utf8")
    config = web_config_io.read(db_path)
    web_config_io.save(db_path, {"dbname": "Kept"}, config["hash"])
    assert path.read_text(encoding="utf8").rstrip().endswith("import os")
    # The service doesn't edit web configs with code
    config = web_config_io.read(db_path, service=True)
    assert not config["writable"] and "code" in config["reason"]
    with pytest.raises(WebConfigError, match="code"):
        web_config_io.save(db_path, {"dbname": "x"}, config["hash"], service=True)


def test_restricted_users(db_path):
    config = web_config_io.read(db_path)
    citations = config["values"]["citations"]
    citations["author"]["prefix"] = "<img src=x onerror=alert(1)>"
    for changes, message in (
        ({"citations": citations}, "HTML"),
        ({"link_to_home_page": "javascript:alert(1)"}, "http"),
        ({"academic_citation": {"collection": "", "citation": [], "custom_url": " JavaScript:x"}}, "http"),
        ({"landing_page_browsing": "templates/landing.html"}, "templates"),
    ):
        with pytest.raises(WebConfigError, match=message):
            web_config_io.save(db_path, changes, config["hash"], service=True, restricted=True)
    web_config_io.save(
        db_path, {"link_to_home_page": "https://example.org"}, config["hash"], service=True, restricted=True
    )
    # HTML already there can be kept
    config = web_config_io.read(db_path)
    web_config_io.save(db_path, {"dbname": "<i>Admin's</i>"}, config["hash"], service=True)
    config = web_config_io.read(db_path)
    citations = config["values"]["citations"]
    web_config_io.save(
        db_path, {"dbname": "<i>Admin's</i>", "logo": "logo.png"}, config["hash"], service=True, restricted=True
    )
