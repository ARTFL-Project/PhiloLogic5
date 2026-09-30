"""Unit tests for the load option schema of philologic5-webui-loader: its defaults must be the loader's, and every
option of the loader must be known to it."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.Loader import RUN_OPTIONS
from philologic.loadtime.LoadOptions import LoadOptions
from philologic.webui_loader import load_schema

pytestmark = pytest.mark.unit

# Options whose default is written differently in a load config than the loader holds it
CONFIG_FORMS = {"words_to_index": ("", set()), "suppress_word_attributes": ([], set())}


def test_defaults_are_the_loaders():
    loader_values = LoadOptions().values
    for option in load_schema.OPTIONS:
        if option.kind == "code":
            continue
        if option.key in CONFIG_FORMS:
            assert (option.default, loader_values[option.key]) == CONFIG_FORMS[option.key], option.key
        else:
            assert option.default == loader_values[option.key], option.key
            assert type(option.default) is type(loader_values[option.key]), option.key


def test_every_loader_option_is_known():
    known = set(load_schema.OPTIONS_BY_KEY) | set(RUN_OPTIONS) | set(load_schema.UNUSED_OPTIONS)
    assert set(LoadOptions().values) - known == set()


def test_validate():
    assert load_schema.validate("token_regex", "[") is not None
    assert load_schema.validate("token_regex", r"[\p{L}]+") is None
    assert load_schema.validate("token_regex", "") is not None  # an empty value would leave the default
    assert load_schema.validate("tag_exceptions", []) is None  # but an empty list turns tag exceptions off
    assert load_schema.validate("tag_exceptions", ["<hi>", "("]) is not None
    assert load_schema.validate("navigable_objects", ["doc", "div1"]) is None
    assert load_schema.validate("navigable_objects", ["doc", "chapter"]) is not None
    assert load_schema.validate("metadata_sql_types", {"year": "int"}) is None
    assert load_schema.validate("metadata_sql_types", {"year": "number"}) is not None
    assert load_schema.validate("long_word_limit", 0) is not None
    assert load_schema.validate("long_word_limit", True) is not None
    assert load_schema.validate("parser_factory", "x") is not None
    assert load_schema.validate("nothing", 1) is not None


def test_is_default_and_config_value():
    assert load_schema.is_default("navigable_objects", ["doc", "div1", "div2", "div3", "para"])
    assert load_schema.to_config_value("navigable_objects", ["doc", "div1"]) == ("doc", "div1")
    assert load_schema.is_default("lemma_file", "")
    assert not load_schema.is_default("tag_exceptions", [])
