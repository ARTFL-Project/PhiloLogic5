"""Integration tests for the loads of philologic5-webui-loader: a real philoload5 load, prepared, launched detached,
followed and done, and a load cancelled while it parses, which leaves no process and no lock behind."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.webui_loader import preflight
from philologic.webui_loader.jobs import JobError, Jobs
from philologic.webui_loader.settings import personal_settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.path.isdir(preflight.WEB_APP) and os.path.exists(preflight.NPM)), reason="PhiloLogic is not installed"
    ),
]

COLLECTIONS = REPO_ROOT / "tests" / "collections"


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    database_root = tmp_path / "dbs"
    database_root.mkdir()
    global_config = tmp_path / "philologic5.cfg"
    global_config.write_text(
        f'database_root = "{database_root}"\nhitlist_dir = "{tmp_path / "hitlists"}"\n',
        encoding="utf8",
    )
    # The loads run the code of this repository
    monkeypatch.setenv("PYTHONPATH", str(REPO_ROOT / "python"))
    settings = personal_settings(
        global_config=str(global_config), state_dir=str(tmp_path / "state"), web_app_config=str(tmp_path / "none.py")
    )
    return Jobs(settings)


def wait(jobs, job_id, condition, timeout=300):
    start = time.time()
    while time.time() - start < timeout:
        status = jobs.status(job_id)
        if condition(status):
            return status
        time.sleep(0.5)
    raise AssertionError(f"timed out: {jobs.status(job_id)['state']}")


def test_load(jobs):
    request = {
        "dbname": "folger3",
        "files": {"paths": [str(COLLECTIONS / "folger-shakespeare" / f"{name}.xml") for name in ("Ham", "Mac", "Oth")]},
        "options": {"break_apost": False},
        "cores": 2,
    }
    plan = preflight.prepare(jobs.settings, jobs, request, [])
    assert plan.errors == []
    job_id = jobs.launch("folger3", plan.config_text, plan.files, 2, cwd=plan.cwd, user="test")
    with pytest.raises(JobError, match="already being loaded"):
        jobs.launch("folger3", plan.config_text, plan.files, 2)
    status = wait(jobs, job_id, lambda status: status["state"] != "running")
    assert status["state"] == "succeeded", jobs.log(job_id)["text"][-2000:]
    assert [stage["state"] for stage in status["stages"]] == ["done"] * 9
    assert status["url"] is None  # on your own machine: the web server serves the databases under its own prefix
    assert not os.path.exists(jobs.lock_path("folger3"))
    saved = Path(jobs.settings.database_root, "folger3", "data", "load_config.py").read_text(encoding="utf8")
    assert "break_apost = False" in saved
    assert jobs.list()[0]["id"] == job_id
    # Loading it again needs its replacement to be confirmed
    plan = preflight.prepare(jobs.settings, jobs, request, [])
    assert [error["field"] for error in plan.errors] == ["dbname"]


def test_cancel(jobs):
    request = {
        "dbname": "eltec",
        "files": {"directory": str(COLLECTIONS / "ELTeC-eng"), "pattern": "*.xml"},
        "cores": 2,
    }
    plan = preflight.prepare(jobs.settings, jobs, request, [])
    assert plan.errors == [] and len(plan.files) == 100
    job_id = jobs.launch("eltec", plan.config_text, plan.files, 2, user="test")
    wait(jobs, job_id, lambda status: any(s["name"] == "parse" and s["state"] == "running" for s in status["stages"]))
    jobs.cancel(job_id)
    status = wait(jobs, job_id, lambda status: status["state"] != "running", timeout=60)
    assert status["state"] == "cancelled"
    assert {stage["name"]: stage["state"] for stage in status["stages"]}["parse"] == "cancelled"
    runner = json.loads(Path(jobs.directory, job_id, "runner.json").read_text(encoding="utf8"))
    left = subprocess.run(["pgrep", "-g", str(runner["load_pid"])], capture_output=True, text=True).stdout.split()
    assert left == []
    assert not os.path.exists(jobs.lock_path("eltec"))
