"""Unit tests for the log of a load as the job page reads it, in parts: progress bars, which rewrite their line with
carriage returns, must show their last state even when a part ends in the middle of their line"""

import json
import os
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.webui_loader import jobs as jobs_module
from philologic.webui_loader.jobs import Jobs, compact_log
from philologic.webui_loader.settings import Settings

pytestmark = pytest.mark.unit

JOB_ID = "mydb-20260101-000000-abcdef"


# The blocks of tqdm's progress bars
FULL, THREE_QUARTERS, HALF = "\u2588", "\u258a", "\u258c"


def bar(done, total=100):
    filled = FULL * (done * 10 // total)
    return f"\rParsing files: {done * 100 // total:3d}%|{filled:<10}| {done}/{total} [00:01<00:02, 42.0it/s]"


LOG = (
    "Copying files... done.\n"
    + "".join(bar(done) for done in range(0, 101, 7))
    + "\r"
    + " " * 60
    + "\rWed Sep 30 15:56:12 2026: done parsing\n"
    + "Merging... done.\n"
)


@pytest.fixture
def jobs(tmp_path):
    settings = Settings(
        mode="personal", state_dir=str(tmp_path), database_root=str(tmp_path)
    )
    jobs = Jobs(settings)
    os.makedirs(os.path.join(settings.loads_dir, JOB_ID))
    with open(os.path.join(settings.loads_dir, JOB_ID, "job.json"), "w", encoding="utf8") as job_file:
        json.dump({"id": JOB_ID}, job_file)
    return jobs


def write_log(jobs, text, mode="w"):
    with open(os.path.join(jobs.directory, JOB_ID, "load.log"), mode, encoding="utf8") as log:
        log.write(text)


def page_log(jobs, shown="", offset=0):
    """What the job page shows after reading the log from offset, as JobView.vue does: (text, next offset)"""
    more = True
    while more:
        part = jobs.log(JOB_ID, offset)
        lines = (shown + part["text"]).split("\n")
        shown = "\n".join(line[line.rfind("\r") + 1 :] for line in lines)
        offset, more = part["offset"], part["more"]
    return shown, offset


@pytest.mark.parametrize("chunk", [7, 37, 64, 1000])
def test_parts_show_the_last_state_of_progress_bars(jobs, monkeypatch, chunk):
    monkeypatch.setattr(jobs_module, "MAX_LOG_CHUNK", chunk)
    write_log(jobs, LOG)
    shown, _ = page_log(jobs)
    assert (
        shown
        == compact_log(LOG)
        == "Copying files... done.\nWed Sep 30 15:56:12 2026: done parsing\nMerging... done.\n"
    )


def test_a_running_progress_bar_rewrites_its_line(jobs):
    write_log(jobs, "Copying files... done.\n" + bar(7))
    shown, offset = page_log(jobs)
    assert shown.endswith("\nParsing files:   7%|          | 7/100 [00:01<00:02, 42.0it/s]")
    write_log(jobs, bar(50) + bar(64), mode="a")
    shown, offset = page_log(jobs, shown, offset)
    assert shown == f"Copying files... done.\nParsing files:  64%|{FULL * 6}    | 64/100 [00:01<00:02, 42.0it/s]"


@pytest.mark.parametrize(
    "tail, description",
    [
        (bar(42), "Parsing files"),
        # The time the loader starts some of its lines with is left out
        (
            f"\rWed Sep 30 16:17:11 2026: Parsing document level metadata:  98%|{FULL * 9}{THREE_QUARTERS}| 98/100 [00:01<00:00]",
            "Parsing document level metadata",
        ),
        (f"\rThu Oct  1 09:05:02 2026: Merging toms:  50%|{FULL * 5}     | 6/12 [00:03<00:03]", "Merging toms"),
        # A bar without a description
        (f"\r 25%|{FULL * 2}{HALF}       | 3/12 [00:01<00:03,  2.0it/s]", ""),
    ],
)
def test_progress_description(tail, description):
    assert Jobs.progress("Some line\n" + tail)["description"] == description
