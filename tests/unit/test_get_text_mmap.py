"""Unit tests for the maps of the texts get_text keeps (get_text._get_mmap)"""

import os

import pytest

from philologic.runtime import get_text


@pytest.fixture(autouse=True)
def empty_cache():
    get_text._mmap_cache.clear()
    yield
    get_text._mmap_cache.clear()


def test_a_file_is_mapped_once(tmp_path):
    path = tmp_path / "text.xml"
    path.write_bytes(b"<p>Once</p>")
    first = get_text._get_mmap(str(path))
    assert first[:] == b"<p>Once</p>"
    assert get_text._get_mmap(str(path)) is first


def test_a_replaced_file_is_mapped_again(tmp_path):
    """A reload of the database replaces its texts: the new file is read, not the one mapped before."""
    path = tmp_path / "text.xml"
    path.write_bytes(b"<p>Old text</p>")
    assert get_text._get_mmap(str(path))[:] == b"<p>Old text</p>"
    replacement = tmp_path / "new.xml"
    replacement.write_bytes(b"<p>The new text</p>")
    os.replace(replacement, path)
    assert get_text._get_mmap(str(path))[:] == b"<p>The new text</p>"
    assert len(get_text._mmap_cache) == 1


def test_the_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(get_text, "_MMAP_CACHE_MAX", 2)
    maps = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.xml"
        path.write_bytes(name.encode() * 3)
        maps.append(get_text._get_mmap(str(path)))
    assert list(get_text._mmap_cache) == [str(tmp_path / "b.xml"), str(tmp_path / "c.xml")]
    assert maps[0].closed and not maps[2].closed
