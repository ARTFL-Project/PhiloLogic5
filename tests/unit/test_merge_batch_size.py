"""Unit tests for merge_batch_size: the files the loader's merges of sorted files merge with a single sort."""

import resource
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime import Loader as loader_module
from philologic.loadtime.Loader import merge_batch_size

pytestmark = pytest.mark.unit


def test_merge_batch_size(monkeypatch):
    """Batches of files fit the limit on open files: sort has each file's pipe open, and opens it"""
    monkeypatch.setattr(sys, "platform", "linux")
    infinity = resource.RLIM_INFINITY
    for limits, batch_size in [((1 << 20, 1 << 20), 1000), ((infinity, infinity), 1000), ((1024, infinity), 480)]:
        monkeypatch.setattr(loader_module.resource, "getrlimit", lambda _, limits=limits: limits)
        assert merge_batch_size() == batch_size
        assert merge_batch_size(3) == 3
    monkeypatch.setattr(sys, "platform", "darwin")
    assert merge_batch_size() == 250
