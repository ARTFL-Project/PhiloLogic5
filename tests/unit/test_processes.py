"""Unit tests for philologic.utils.processes: the limits of the processes the loader starts."""

import os
import resource
import subprocess
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.utils.processes import raise_open_files_limit

pytestmark = pytest.mark.unit


def test_raise_open_files_limit():
    """A process started with a low limit on open files raises it as far as it may"""
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    if hard != resource.RLIM_INFINITY and hard <= 256:
        pytest.skip("the hard limit is too low to lower the soft limit")
    code = (
        "import resource; from philologic.utils import raise_open_files_limit; "
        "print(raise_open_files_limit(), *resource.getrlimit(resource.RLIMIT_NOFILE))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "python")},
        preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_NOFILE, (256, hard)),
    )
    returned, soft, new_hard = map(int, result.stdout.split())
    assert returned == soft and new_hard == hard
    assert soft == hard or (hard == resource.RLIM_INFINITY and soft >= 10240)
    assert raise_open_files_limit() == resource.getrlimit(resource.RLIMIT_NOFILE)[0]
