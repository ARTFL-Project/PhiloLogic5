"""Unit tests for how philologic5-webui-loader reads and writes Python config files without running them (pyconfig),
and load configs in particular (load_config_io)."""

import sys
from pathlib import Path

import pytest

# Add PhiloLogic to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

from philologic.loadtime.LoadOptions import LoadConfig
from philologic.webui_loader import load_config_io, load_schema
from philologic.webui_loader.pyconfig import Code, ConfigFile, assignment_source, to_source

pytestmark = pytest.mark.unit

CONFIG = '''"""A docstring"""
from philologic.loadtime import XMLParser  # the parser

citations = {"author": {"field": "author"}, "title": {"field": "title"}}
concordance_citation = [citations["author"], citations["title"]]  # a comment
words = set()
limit = -3
regex = r"[\\p{L}]+"
parser_factory = XMLParser
computed = sorted(["b", "a"])
a = 1; b = 2
'''


def test_values_without_running():
    config = ConfigFile(CONFIG)
    assert config.values["concordance_citation"] == [{"field": "author"}, {"field": "title"}]
    assert config.values["words"] == set()
    assert config.values["limit"] == -3
    assert config.values["regex"] == r"[\p{L}]+"
    assert config.entries["parser_factory"].value == Code("XMLParser")
    assert config.entries["computed"].value == Code('sorted(["b", "a"])')
    assert not config.entries["XMLParser"].assigned
    # Assignments sharing a line can't be edited in place
    assert config.entries["a"].statement is None and config.values["a"] == 1


def test_initial_names():
    config = ConfigFile('x = citations["author"]\n', {"citations": {"author": 1}})
    assert config.values["x"] == 1
    assert ConfigFile('x = citations["author"]\n').entries["x"].is_code


def test_edits_keep_the_rest():
    config = ConfigFile(CONFIG)
    text = config.edited(
        {
            "concordance_citation": assignment_source(
                "concordance_citation", [{"field": "title"}], [('citations["title"]', {"field": "title"})]
            ),
            "limit": assignment_source("limit", 5),
            "new": assignment_source("new", ("x",)),
        },
        comments={"new": "A new option"},
        removed=["words"],
    )
    assert 'concordance_citation = [citations["title"]]  # a comment\n' in text
    assert "limit = 5\n" in text and "words" not in text
    assert text.endswith('\n# A new option\nnew = ("x",)\n')
    assert text.startswith('"""A docstring"""\nfrom philologic.loadtime import XMLParser  # the parser\n')
    assert ConfigFile(text).values["new"] == ("x",)
    with pytest.raises(ValueError):
        config.edited({"computed": "computed = 1\n"})


def test_to_source():
    assert to_source(r"a\b") == 'r"a\\b"'
    assert to_source('a"\\') == repr('a"\\')
    assert to_source({"b", "a"}) == "{'a', 'b'}"
    assert to_source((1,)) == "(1,)"
    with pytest.raises(TypeError):
        to_source(object())


def test_read_load_config(tmp_path):
    text = """from philologic.loadtime import XMLParser
parser_factory = XMLParser
token_regex = r"[\\p{L}]+"
punctuation = ""
tag_exceptions = []
words_to_index = ""
cores = 56
files = ["a.xml", "b.xml"]
pos_tagger = ""
my_variable = 3
navigable_objects = ("doc", "chapter")
"""
    result = load_config_io.read(text)
    # As LoadConfig takes them: empty values leave the default, except for tag_exceptions
    assert result["options"] == {
        "token_regex": r"[\p{L}]+",
        "tag_exceptions": [],
        "navigable_objects": ["doc", "chapter"],
    }
    assert result["errors"] == {"navigable_objects": load_schema.validate("navigable_objects", ["doc", "chapter"])}
    assert result["run_options"] == {"cores": 56, "files": "2 items"}
    assert result["unused"] == ["pos_tagger"]
    assert result["unknown"] == {"my_variable": "3"}
    assert result["code"] == {} and result["custom_code"] == []
    config_file = tmp_path / "load_config.py"
    config_file.write_text(text, encoding="utf8")
    loader_config = LoadConfig()
    loader_config.parse(str(config_file))
    for key, value in result["options"].items():
        assert load_schema.json_value(loader_config.config[key]) == value


def test_custom_code_is_kept():
    text = """import sys
sys.path.append(".")
from my_parser import MyParser

parser_factory = MyParser
token_regex = r"[a-z]+"
cores = 8
"""
    result = load_config_io.read(text)
    assert result["code"] == {"parser_factory": "MyParser"}
    assert "from my_parser import MyParser" in result["custom_code"]
    rendered = load_config_io.render({"token_regex": r"[a-z]+", "break_apost": False}, text)
    # Edited in place: code kept, run options removed, new option added with its help
    assert rendered.startswith('import sys\nsys.path.append(".")\nfrom my_parser import MyParser\n')
    assert "parser_factory = MyParser" in rendered and "cores" not in rendered
    assert "break_apost = False" in rendered
    assert load_config_io.read(rendered)["options"] == {"token_regex": r"[a-z]+", "break_apost": False}


def test_render_without_code_only_sets_what_differs():
    options = {name: load_schema.json_value(load_schema.default(name)) for name in load_schema.OPTIONS_BY_KEY}
    options = {key: value for key, value in options.items() if load_schema.OPTIONS_BY_KEY[key].kind != "code"}
    options["break_apost"] = False
    options["navigable_objects"] = ["doc", "div1"]
    options["header"] = "dc"  # given on the command line instead
    text = load_config_io.render(options)
    assert text.startswith(load_config_io.MARKER)
    assert load_config_io.read(text)["options"] == {"break_apost": False, "navigable_objects": ["doc", "div1"]}
    assert load_config_io.read(text)["generated"]
    assert 'navigable_objects = ("doc", "div1")' in text


def test_render_from_a_saved_dump():
    dump = '''"""This is a dump of the default configuration used to load this database"""
database_root = '/var/www/html/philologic5'
cores = 56
files = ['texts/a.xml']
token_regex = '[\\\\p{L}]+'
break_apost = True
'''
    text = load_config_io.render(load_config_io.read(dump)["options"], dump)
    assert "files" not in text and "cores" not in text
    assert load_config_io.read(text)["options"] == {"token_regex": r"[\p{L}]+"}


def test_control_characters_dont_shift_lines():
    # Inside strings, form feeds and such are no line breaks for Python, but are for str.splitlines
    value = "a\x0cb\x85c\\d"
    text = assignment_source("first", value) + 'second = "x"\nthird = 1\n'
    config = ConfigFile(text)
    assert config.values["first"] == value
    edited = config.edited({"second": assignment_source("second", "y")})
    assert ConfigFile(edited).values == {"first": value, "second": "y", "third": 1}
    assert "\x0c" not in assignment_source("first", value)  # written escaped, not raw


def test_hostile_strings_stay_strings():
    for value in ('"; import os; os.system("x"); "', "\\\n\nimport os\n", "'''", 'a"""b', "\r\nimport os\r\n"):
        source = assignment_source("x", value) + assignment_source("y", {value: [value]})
        config = ConfigFile(source)
        assert config.code_statements == [] and config.values == {"x": value, "y": {value: [value]}}
