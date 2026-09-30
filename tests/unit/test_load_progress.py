"""Unit tests for the stages of a load recorded in PHILOLOGIC_PROGRESS_FILE (read by philologic5-webui-loader)"""

import sys
from pathlib import Path

import orjson
import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

import philologic.loadtime.__main__ as load_command

pytestmark = pytest.mark.unit


def test_stages_recorded(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.jsonl"
    monkeypatch.setattr(load_command, "PROGRESS_FILE", str(progress_file))
    with load_command.stage("parse"):
        pass
    with pytest.raises(ValueError):
        with load_command.stage("merge"):
            raise ValueError
    events = [orjson.loads(line) for line in progress_file.read_bytes().splitlines()]
    # A failed stage has no end
    assert [(event["stage"], event["event"]) for event in events] == [
        ("parse", "start"),
        ("parse", "end"),
        ("merge", "start"),
    ]
    assert events[0]["time"] <= events[1]["time"]


def test_nothing_recorded_without_progress_file(tmp_path, monkeypatch):
    monkeypatch.setattr(load_command, "PROGRESS_FILE", None)
    monkeypatch.chdir(tmp_path)
    with load_command.stage("parse"):
        pass
    assert list(tmp_path.iterdir()) == []
