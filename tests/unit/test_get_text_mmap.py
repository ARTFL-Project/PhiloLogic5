"""Unit tests for the mmap cache of get_text: a file replaced or rewritten in place (a database reloaded under a
running worker) is mapped again, and one left alone is served from the cache."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.runtime import get_text as get_text_module
from philologic.runtime.get_text import _get_mmap, get_text


@pytest.fixture
def text_file(tmp_path):
    """A database's data/TEXT/doc.txt, with an empty cache"""
    text_dir = tmp_path / "data" / "TEXT"
    text_dir.mkdir(parents=True)
    path = text_dir / "doc.txt"
    path.write_bytes(b"0123456789abcdefghij")
    get_text_module._mmap_cache.clear()
    yield path
    get_text_module._mmap_cache.clear()


def read(path, start=5, length=6):
    hit = SimpleNamespace(doc=SimpleNamespace(filename=path.name))
    return get_text(hit, start, length, str(path.parent.parent.parent))


def replace_by_rename(path, content):
    new = path.with_name("doc.new")
    new.write_bytes(content)
    os.replace(new, path)


@pytest.mark.unit
class TestMmapCache:
    def test_unchanged_file_is_served_from_the_cache(self, text_file):
        first = _get_mmap(str(text_file))
        assert read(text_file) == b"56789a"
        assert _get_mmap(str(text_file)) is first
        assert len(get_text_module._mmap_cache) == 1

    def test_file_replaced_by_rename_is_read_again(self, text_file):
        assert read(text_file) == b"56789a"
        replace_by_rename(text_file, b"ABCDEFGHIJKLMNOPQRST")  # same size, new inode
        assert read(text_file) == b"FGHIJK"

    def test_file_rewritten_with_another_size_is_read_again(self, text_file):
        assert read(text_file) == b"56789a"
        text_file.write_bytes(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # same inode, longer
        assert read(text_file) == b"FGHIJK"
        assert len(get_text_module._mmap_cache) == 1

    def test_file_rewritten_with_the_same_size_is_mapped_again(self, text_file):
        first = _get_mmap(str(text_file))
        text_file.write_bytes(b"ABCDEFGHIJKLMNOPQRST")
        os.utime(text_file, ns=(1, 1))
        assert _get_mmap(str(text_file)) is not first
        assert read(text_file) == b"FGHIJK"

    def test_replacement_with_the_same_size_and_mtime_is_read_again(self, text_file):
        """Only the inode tells: a reload that copies the mtimes, or a clock too coarse to change it"""
        mtime = text_file.stat().st_mtime_ns
        assert read(text_file) == b"56789a"
        new = text_file.with_name("doc.new")
        new.write_bytes(b"ABCDEFGHIJKLMNOPQRST")
        os.utime(new, ns=(mtime, mtime))
        os.replace(new, text_file)
        assert read(text_file) == b"FGHIJK"

    def test_file_grown_in_place_with_the_same_mtime_is_read_to_its_end(self, text_file):
        mtime = text_file.stat().st_mtime_ns
        assert read(text_file) == b"56789a"
        text_file.write_bytes(b"0123456789abcdefghijKLMNOPQRST")  # same inode: only the size changed
        os.utime(text_file, ns=(mtime, mtime))
        assert read(text_file, 20, 10) == b"KLMNOPQRST"

    def test_replaced_file_is_mapped_once(self, text_file):
        read(text_file)
        replace_by_rename(text_file, b"ABCDEFGHIJKLMNOPQRST")
        second = _get_mmap(str(text_file))
        assert _get_mmap(str(text_file)) is second

    def test_old_bytes_taken_before_the_replacement_stay_valid(self, text_file):
        old = _get_mmap(str(text_file))
        replace_by_rename(text_file, b"ABCDEFGHIJKLMNOPQRST")
        assert _get_mmap(str(text_file)) is not old
        assert old[5:11] == b"56789a"  # not closed under a reader still slicing it
