"""Unit tests for philoload5's options: the copy of the load config saved in a database has to load it again, with the
files and run options given on the command line, not those of the load it was saved by."""

import ast
import os
import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime import PlainTextParser
from philologic.loadtime.Loader import RUN_OPTIONS, Loader
from philologic.loadtime.LoadOptions import CONFIG_FILE, LoadOptions
from philologic.loadtime.Parser import XMLParser

pytestmark = pytest.mark.unit

# What a database's data/load_config.py looked like when it also saved the options of the run which loaded it
OLD_SAVED_CONFIG = """
database_root = '/somewhere/else'
url_root = 'https://example.org/philologic5/'
destination = './'
load_config = ''
token_regex = '[\\\\p{L}]+'
cores = 56
files = ['texts/old_1.xml', 'texts/old_2.xml']
sort_order = ['year', 'author', 'title', 'filename']
header = 'dc'
debug = False
force_delete = False
file_list = False
bibliography = ''
file_type = 'xml'
"""


def parse_options(*args):
    options = LoadOptions()
    options.parse(["philoload5", *args])
    return options


def saved_option_names(path):
    """Names of the options set in a load config"""
    tree = ast.parse(path.read_text(encoding="utf8"))
    return {target.id for node in tree.body if isinstance(node, ast.Assign) for target in node.targets}


def load_config_saved_by(options, destination):
    """The load_config.py which a load with these options saves in its database"""
    options["db_destination"] = str(destination)
    options["data_destination"] = str(destination / "data")
    Loader.set_class_attributes(options.values)
    return destination / "data" / "load_config.py"


@pytest.fixture
def files(tmp_path):
    paths = []
    for name in ("a.xml", "b.xml", "c.xml"):
        (tmp_path / name).write_text("<TEI/>", encoding="utf8")
        paths.append(str(tmp_path / name))
    return paths


def test_old_saved_config_does_not_override_command_line(tmp_path, files, capsys):
    config = tmp_path / "load_config.py"
    config.write_text(OLD_SAVED_CONFIG, encoding="utf8")
    options = parse_options("-c", "3", "-D", "-d", "-l", str(config), "newdb", files[0])
    assert options["files"] == [files[0]]
    assert options["cores"] == 3
    assert options["force_delete"] is True
    assert options["debug"] is True
    assert options["database_root"] == CONFIG_FILE.database_root
    assert options["url_root"] == CONFIG_FILE.url_root
    assert options["db_destination"] == os.path.join(CONFIG_FILE.database_root, "newdb")
    assert options["load_config"] == str(config)
    # The options of the database itself still come from the load config
    assert options["token_regex"] == r"[\p{L}]+"
    assert options["header"] == "dc"
    ignored = capsys.readouterr().err
    assert "Ignoring load config options" in ignored
    for option in ("files", "cores", "database_root", "force_delete", "debug"):
        assert option in ignored


def test_command_line_header_and_file_type_win_over_load_config(tmp_path, files):
    config = tmp_path / "load_config.py"
    config.write_text("header = 'dc'\nfile_type = 'plain_text'\n", encoding="utf8")
    options = parse_options("-l", str(config), "newdb", files[0])
    assert options["header"] == "dc"
    assert options["file_type"] == "plain_text"
    assert options["parser_factory"] is PlainTextParser.PlainTextParser
    options = parse_options("-H", "tei", "-t", "xml", "-l", str(config), "newdb", files[0])
    assert options["header"] == "tei"
    assert options["file_type"] == "xml"
    assert options["parser_factory"] is XMLParser


def test_defaults_without_load_config(files):
    options = parse_options("newdb", *files)
    assert options["header"] == "tei"
    assert options["file_type"] == "xml"
    assert options["parser_factory"] is XMLParser
    assert options["cores"] == 4


def test_saved_config_loads_database_again(tmp_path, files):
    saved = load_config_saved_by(parse_options("-c", "2", "-H", "dc", "firstdb", *files[:2]), tmp_path / "firstdb")
    names = saved_option_names(saved)
    assert not names & set(RUN_OPTIONS)
    assert {"header", "file_type", "token_regex", "doc_xpaths"} <= names

    options = parse_options("-c", "3", "-l", str(saved), "seconddb", files[2])
    assert options["files"] == [files[2]]
    assert options["cores"] == 3
    assert options["header"] == "dc"
    assert options["db_destination"] == os.path.join(CONFIG_FILE.database_root, "seconddb")


def test_saved_copy_of_a_load_config_loads_database_again(tmp_path, files):
    config = tmp_path / "my_config.py"
    config.write_text("token_regex = r'[\\p{L}]+'\n", encoding="utf8")
    saved = load_config_saved_by(
        parse_options("-c", "2", "-l", str(config), "firstdb", *files[:2]), tmp_path / "firstdb"
    )
    names = saved_option_names(saved)
    assert not names & set(RUN_OPTIONS)
    assert {"token_regex", "header"} <= names

    options = parse_options("-c", "3", "-l", str(saved), "seconddb", files[2])
    assert options["files"] == [files[2]]
    assert options["cores"] == 3
    assert options["token_regex"] == r"[\p{L}]+"


def test_load_config_with_its_own_filters_is_copied(tmp_path, files):
    config = tmp_path / "my_config.py"
    config.write_text(
        "from philologic.loadtime.LoadFilters import set_load_filters\n\nload_filters = set_load_filters()\n",
        encoding="utf8",
    )
    options = parse_options("-l", str(config), "firstdb", files[0])
    assert options["load_config"] == str(config)
    saved = load_config_saved_by(options, tmp_path / "firstdb")
    assert saved.read_text(encoding="utf8").startswith(config.read_text(encoding="utf8"))

    options = parse_options("-l", str(saved), "seconddb", files[1])
    assert [f.__name__ for f in options["load_filters"]] == [f.__name__ for f in LoadOptions()["load_filters"]]
