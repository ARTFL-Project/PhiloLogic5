"""The options of a database load: their kinds, defaults and help, for the load pages of philologic5-webui-loader and
the load configs it writes. Defaults are the loader's own (tests/unit/test_webui_load_schema.py checks that they
match LoadOptions). The help texts come from extras/load_config.py."""

import copy
from dataclasses import dataclass, field

import regex

from philologic.loadtime import Loader, Parser

OBJECT_TYPES = ("doc", "div1", "div2", "div3", "para")

# Types which tags can be mapped to in tag_to_obj_map
TAG_TYPES = ("div", "para", "page", "ref", "graphic", "line")

SQL_TYPES = ("text", "int", "date")

SPACY_MODEL = regex.compile(r"^[A-Za-z][A-Za-z0-9_]{1,100}$")

GROUPS = ("source", "structure", "metadata", "tokenization", "indexing", "nlp", "advanced")


@dataclass
class Option:
    """A load option. kind is one of bool, int, choice, multichoice (an ordered selection of choices), string, regex,
    path, list (of strings, or regexes with item_kind "regex"), dict (string values, from value_choices if given),
    dict_list (lists of strings as values), or code (set by Python code in a load config, not editable)."""

    key: str
    kind: str
    default: object
    group: str
    help: str
    basic: bool = False  # shown without "show advanced"
    choices: tuple = ()
    item_kind: str = "string"
    value_choices: tuple = ()
    allow_empty: bool = True  # False when an empty value in a load config would leave the default instead
    minimum: int = None
    cli_flag: str = None
    tuple_value: bool = False  # the loader's value is a tuple
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "key": self.key,
            "kind": self.kind,
            "default": json_value(self.default),
            "group": self.group,
            "help": self.help,
            "basic": self.basic,
            "choices": list(self.choices),
            "item_kind": self.item_kind,
            "value_choices": list(self.value_choices),
            "allow_empty": self.allow_empty,
            "minimum": self.minimum,
            "cli_flag": self.cli_flag,
        }


def json_value(value):
    """A value as JSON can represent it: tuples and sets as lists"""
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(json_value(item) for item in value)
    return value


OPTIONS = [
    # What the texts are: set in the source step, given to philoload5 as -H and -t
    Option(
        "header",
        "choice",
        "tei",
        "source",
        "Type of the header of the files: TEI, or Dublin Core.",
        basic=True,
        choices=("tei", "dc"),
        cli_flag="-H",
    ),
    Option(
        "file_type",
        "choice",
        "xml",
        "source",
        "Format of the files: XML (TEI), or plain text, which needs a bibliography file giving the metadata of each "
        "file. The plain text parser only finds paragraphs, separated by empty lines.",
        basic=True,
        choices=("xml", "plain_text"),
        cli_flag="-t",
    ),
    Option(
        "default_object_level",
        "choice",
        Loader.DEFAULT_OBJECT_LEVEL,
        "structure",
        "Type of object returned by most navigation reports: doc for most databases, div1 for dictionaries or "
        "encyclopedias.",
        basic=True,
        choices=OBJECT_TYPES,
    ),
    Option(
        "navigable_objects",
        "multichoice",
        Loader.NAVIGABLE_OBJECTS,
        "structure",
        "Object types stored in the database and available for searching, reporting and navigation. Add para if "
        "paragraphs carry interesting metadata, as in drama. Pages are handled separately.",
        basic=True,
        choices=OBJECT_TYPES,
        allow_empty=False,
        tuple_value=True,
    ),
    Option(
        "sort_order",
        "list",
        ["year", "author", "title", "filename"],
        "structure",
        "Metadata fields by which files are sorted in the database, which sets the order in which results are "
        "displayed. An empty list keeps the order of the files.",
        basic=True,
    ),
    Option(
        "doc_xpaths",
        "dict_list",
        Parser.DEFAULT_DOC_XPATHS,
        "metadata",
        "XPaths of the document metadata in the TEI header, for each field: the first XPath which finds a value is "
        "used. They are evaluated inside <teiHeader> and must apply to the whole document.",
        basic=True,
        allow_empty=False,
    ),
    Option(
        "metadata_sql_types",
        "dict",
        {},
        "metadata",
        "How metadata fields are stored and queried: text, int or date. Fields not listed are text.",
        value_choices=SQL_TYPES,
    ),
    Option(
        "tag_to_obj_map",
        "dict",
        Parser.DEFAULT_TAG_TO_OBJ_MAP,
        "metadata",
        "Maps tags to PhiloLogic's object types: div, para, page, ref, graphic or line.",
        value_choices=TAG_TYPES,
        allow_empty=False,
    ),
    Option(
        "metadata_to_parse",
        "dict_list",
        Parser.DEFAULT_METADATA_TO_PARSE,
        "metadata",
        "Metadata stored for each object type. They are attributes of its tag, except head and div_date, which are "
        "tags of their own.",
        allow_empty=False,
    ),
    Option(
        "token_regex",
        "regex",
        Parser.TOKEN_REGEX,
        "tokenization",
        "Regular expression of the words. For Asian scripts, try [\\p{L}\\p{M}\\p{N}\\p{Po}]+|[&\\p{L};]+",
        basic=True,
        allow_empty=False,
    ),
    Option(
        "break_apost",
        "bool",
        True,
        "tokenization",
        "Break words on apostrophes: probably off for English, on for French.",
        basic=True,
    ),
    Option(
        "chars_not_to_index",
        "regex",
        Parser.CHARS_NOT_TO_INDEX,
        "tokenization",
        "Characters which can be in words but are not indexed.",
        allow_empty=False,
    ),
    Option(
        "tag_exceptions",
        "list",
        Parser.TAG_EXCEPTIONS,
        "tokenization",
        "Tags which don't break words (tags normally do), as regular expressions. They are not indexed. An empty list "
        "turns this off.",
        item_kind="regex",
    ),
    Option(
        "punctuation",
        "regex",
        Parser.PUNCTUATION,
        "tokenization",
        "Regular expression of the punctuation recorded as such. It should not include the punctuation which ends "
        "sentences.",
        allow_empty=False,
    ),
    Option(
        "sentence_breakers",
        "list",
        [],
        "tokenization",
        'Strings which end a sentence, besides ".", "?" and "!".',
    ),
    Option(
        "break_sent_in_line_group",
        "bool",
        False,
        "tokenization",
        "In line groups, end sentences at </l>, instead of finding them automatically. Normally off.",
    ),
    Option(
        "join_hyphen_in_words",
        "bool",
        True,
        "tokenization",
        "Join words broken by soft hyphens (&shy;) at the end of lines.",
    ),
    Option(
        "abbrev_expand",
        "bool",
        True,
        "tokenization",
        'Index abbreviations under their expansion, as in <abbr expan="en">&emacr;</abbr>.',
    ),
    Option(
        "flatten_ligatures",
        "bool",
        True,
        "tokenization",
        "Index SGML ligatures as their base characters (&oelig; as oe). Leave this on.",
    ),
    Option(
        "long_word_limit",
        "int",
        200,
        "tokenization",
        "Longest word indexed, in bytes: longer words are left out of the index (over 235 bytes, words break the "
        "index).",
        allow_empty=False,
        minimum=1,
    ),
    Option(
        "ascii_conversion",
        "bool",
        Loader.ASCII_CONVERSION,
        "indexing",
        "Also search and autocomplete text and metadata by their ASCII form. Turn off for languages which don't "
        "convert well to ASCII (non-European languages in general).",
        basic=True,
    ),
    Option(
        "lowercase_index",
        "bool",
        True,
        "indexing",
        "Store words in lowercase in the index.",
        basic=True,
    ),
    Option(
        "suppress_tags",
        "list",
        [],
        "indexing",
        "Tags whose contents are not indexed, such as desc or fw. <gap> is always left out.",
    ),
    Option(
        "suppress_word_attributes",
        "list",
        [],
        "indexing",
        "Attributes of <w> tags which are not stored, such as type or id.",
    ),
    Option(
        "words_to_index",
        "path",
        "",
        "indexing",
        "File of the only words to index, one per line. Useful to leave dirty OCR out of the index.",
    ),
    Option(
        "pseudo_empty_tags",
        "list",
        [],
        "indexing",
        "Tags handled as empty, whatever their contents.",
    ),
    Option(
        "spacy_model",
        "string",
        None,
        "nlp",
        "spaCy model for lemmas, parts of speech and named entities (only official models). Unless it runs on the "
        "GPU, each parsing process (see cores) loads its own copy of the model: memory use grows with cores. On the "
        "GPU, files are tagged in a single process.",
        basic=True,
    ),
    Option(
        "lemma_file",
        "path",
        None,
        "nlp",
        "File mapping words to their lemma: one word per line, separated from its lemma by a tab.",
    ),
    Option(
        "parser_factory",
        "code",
        Parser.XMLParser,
        "advanced",
        "Parser class, with the same interface as philologic.loadtime.Parser.XMLParser.",
    ),
    Option(
        "load_filters",
        "code",
        None,
        "advanced",
        "Load filters replacing the default ones. This requires intimate knowledge of the parser and the filters.",
    ),
    Option(
        "post_filters",
        "code",
        None,
        "advanced",
        "Post filters replacing the default ones.",
    ),
]

OPTIONS_BY_KEY = {option.key: option for option in OPTIONS}

# Options which still exist in old load configs, but no longer have any effect
UNUSED_OPTIONS = ("pos", "pos_tagger", "plain_text_obj", "store_words_and_ids", "unicode_word_breakers")

# Options set by the source step of the UI and given to philoload5 on its command line, rather than in a load config
SOURCE_OPTIONS = ("header", "file_type")


def default(key):
    """Default value of an option, as it would be written in a load config (a copy, which can be changed)"""
    return copy.deepcopy(OPTIONS_BY_KEY[key].default)


def is_default(key, value):
    """Whether a value leaves the default: equal to it, or empty where the loader ignores empty values"""
    option = OPTIONS_BY_KEY[key]
    if value in (None, "") and option.kind in ("path", "string"):
        return option.default in (None, "")
    return json_value(value) == json_value(option.default)


def to_config_value(key, value):
    """Value of an option received as JSON, as the loader takes it (tuples where the loader uses tuples)"""
    option = OPTIONS_BY_KEY[key]
    if option.tuple_value and isinstance(value, list):
        return tuple(value)
    return value


def validate(key, value):
    """Error message if the value isn't valid for the option, else None"""
    option = OPTIONS_BY_KEY.get(key)
    if option is None:
        return f"unknown option {key}"
    if option.kind == "code":
        return "this option is set by code and can't be edited here"
    empty = value in (None, "", [], {}, ())
    if empty and not option.allow_empty:
        return "can't be empty (an empty value would leave the default)"
    if empty:
        return None
    kind = option.kind
    if kind == "bool":
        return None if isinstance(value, bool) else "must be true or false"
    if kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return "must be a whole number"
        if option.minimum is not None and value < option.minimum:
            return f"must be at least {option.minimum}"
        return None
    if kind == "choice":
        return None if value in option.choices else f"must be one of {', '.join(option.choices)}"
    if kind == "multichoice":
        if not isinstance(value, (list, tuple)) or any(item not in option.choices for item in value):
            return f"must be a selection of {', '.join(option.choices)}"
        if len(set(value)) != len(value):
            return "has duplicates"
        return None
    if key == "spacy_model":
        # a package name: spaCy imports it
        return None if isinstance(value, str) and SPACY_MODEL.match(value) else "must be the name of a spaCy model"
    if kind in ("string", "path"):
        return None if isinstance(value, str) else "must be a string"
    if kind == "regex":
        if not isinstance(value, str):
            return "must be a string"
        return regex_error(value)
    if kind == "list":
        if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
            return "must be a list of strings"
        if option.item_kind == "regex":
            for item in value:
                error = regex_error(item)
                if error:
                    return f"{item}: {error}"
        return None
    if kind == "dict":
        if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
            return "must map strings to strings"
        if option.value_choices:
            for item_key, item in value.items():
                if item not in option.value_choices:
                    return f"{item_key}: must be one of {', '.join(option.value_choices)}"
        return None
    if kind == "dict_list":
        if not isinstance(value, dict) or not all(
            isinstance(k, str) and isinstance(v, (list, tuple)) and all(isinstance(i, str) for i in v)
            for k, v in value.items()
        ):
            return "must map strings to lists of strings"
        return None
    return None


def regex_error(pattern):
    """Error message if the pattern isn't a valid regular expression (of the regex module, which the loader uses)"""
    try:
        regex.compile(pattern)
    except regex.error as error:
        return f"invalid regular expression: {error}"
    return None


def schema():
    """The options as JSON, for the UI"""
    return {
        "groups": list(GROUPS),
        "options": [option.to_dict() for option in OPTIONS if option.kind != "code"],
        "code_options": [option.key for option in OPTIONS if option.kind == "code"],
        "unused_options": list(UNUSED_OPTIONS),
    }
