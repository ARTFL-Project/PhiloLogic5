"""The code of philologic5-webui-loader is ASCII: other characters (invisible ones, look-alikes of ASCII, symbols...)
are written as escapes, or HTML entities in templates. Only its locales, which are text, have others."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

pytestmark = pytest.mark.unit

SOURCES = [
    *(REPO_ROOT / "python/philologic/webui_loader").rglob("*.py"),
    *(REPO_ROOT / "webui_loader/src").rglob("*.js"),
    *(REPO_ROOT / "webui_loader/src").rglob("*.vue"),
    *(REPO_ROOT / "webui_loader/src").rglob("*.scss"),
    *(REPO_ROOT / "webui_loader/tests").rglob("*.js"),
    REPO_ROOT / "webui_loader/index.html",
    REPO_ROOT / "webui_loader/vite.config.js",
    *(REPO_ROOT / "extras/webui_loader").iterdir(),
    *(REPO_ROOT / "tests/unit").glob("test_webui_*.py"),
]


def test_code_is_ascii():
    found = []
    for path in SOURCES:
        for number, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            characters = sorted({f"U+{ord(character):04X}" for character in line if ord(character) > 127})
            if characters:
                found.append(f"{path.relative_to(REPO_ROOT)}:{number}: {', '.join(characters)}")
    assert not found, "\n".join(found)
