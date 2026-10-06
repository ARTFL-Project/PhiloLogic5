"""Unit tests for the web config of requests (runtime.web_config.WebConfig): parsed once for each version of
web_config.cfg and db.locals.py, and copied for each request, which may change its copy."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import web_config
from philologic.runtime.web_config import WebConfig

pytestmark = pytest.mark.unit


@pytest.fixture
def database(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "web_config.cfg").write_text(
        'search_reports = ["concordance", "kwic", "time_series"]\nmetadata = ["author", "title"]\n'
    )
    (tmp_path / "data" / "db.locals.py").write_text('metadata_fields = ["author", "title"]\n')
    return tmp_path


@pytest.fixture
def parses(monkeypatch):
    """The paths MakeWebConfig parses"""
    parsed = []
    make = web_config.MakeWebConfig

    def counted(path):
        parsed.append(path)
        return make(path)

    monkeypatch.setattr(web_config, "MakeWebConfig", counted)
    return parsed


def test_parsed_once(database, parses):
    configs = [WebConfig(str(database)) for _ in range(3)]
    assert len(parses) == 1
    assert all(config["metadata"] == ["author", "title"] for config in configs)
    assert configs[0] is not configs[1]


def test_own_copy(database):
    """What a request changes stays in its copy: to_dict removed time_series from the config's own search_reports."""
    config = WebConfig(str(database))
    config["metadata"].append("year")
    config.db_locals["metadata_fields"].append("year")
    config.time_series_status = False
    assert "time_series" not in config.to_dict()["search_reports"]
    fresh = WebConfig(str(database))
    assert fresh["metadata"] == ["author", "title"] and fresh.db_locals["metadata_fields"] == ["author", "title"]
    assert fresh.time_series_status is True and "time_series" in fresh.to_dict()["search_reports"]


@pytest.mark.parametrize(
    "name, content, read",
    [
        ("web_config.cfg", 'metadata = ["author"]\n', lambda config: config["metadata"]),
        ("db.locals.py", 'metadata_fields = ["author"]\n', lambda config: config.db_locals["metadata_fields"]),
    ],
)
def test_changed_file(database, parses, name, content, read):
    WebConfig(str(database))
    (database / "data" / name).write_text(content)
    assert read(WebConfig(str(database))) == ["author"]
    assert len(parses) == 2


def test_broken_file_not_kept(database):
    (database / "data" / "web_config.cfg").write_text("metadata = [\n")
    assert WebConfig(str(database)).valid_config is False
    (database / "data" / "web_config.cfg").write_text('metadata = ["title"]\n')
    config = WebConfig(str(database))
    assert config.valid_config is True and config["metadata"] == ["title"]
