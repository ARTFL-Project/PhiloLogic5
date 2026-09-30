"""Load configs in philologic5-webui-loader: reading a load_config.py, or the copy saved in a database, into the
options of the load pages (as LoadOptions would take them, without running the file), and writing the load config
of a load."""

import os
import re

from philologic.loadtime.Loader import RUN_OPTIONS
from philologic.webui_loader import load_schema
from philologic.webui_loader.load_schema import OPTIONS, OPTIONS_BY_KEY, SOURCE_OPTIONS, UNUSED_OPTIONS, json_value
from philologic.webui_loader.pyconfig import Code, ConfigFile, assignment_source

MARKER = "# Load config written by philologic5-webui-loader"

# LoadConfig takes empty values of these, rather than leaving the default
EMPTY_TAKEN = ("sort_order", "tag_exceptions")

DEFAULT_PARSER_IMPORT = re.compile(r"^from\s+philologic\.loadtime(\.Parser)?\s+import\s+.*\bXMLParser\b", re.S)


def is_default_parser(config):
    """Whether the parser_factory of a config is the default XMLParser, imported from philologic.loadtime"""
    parser = config.entries.get("parser_factory")
    xml_parser = config.entries.get("XMLParser")
    return (
        parser is not None
        and parser.value == Code("XMLParser")
        and xml_parser is not None
        and xml_parser.is_code
        and DEFAULT_PARSER_IMPORT.match(xml_parser.value.source) is not None
    )


def custom_code(config):
    """The code statements of a config, other than the import of the default parser"""
    statements = []
    default_parser = is_default_parser(config)
    for statement in config.code_statements:
        if default_parser and (DEFAULT_PARSER_IMPORT.match(statement) or statement == "parser_factory = XMLParser"):
            continue
        statements.append(statement)
    return statements


def read(text):
    """What a load config sets, as JSON for the load pages:
    options: the load options it sets (as LoadOptions takes them: an empty value leaves the default, except for
        sort_order and tag_exceptions),
    code: the options set by code, with its source (not editable),
    run_options: options of the load it was saved by, which only come from the command line,
    unused: options which no longer have any effect,
    unknown: other names it sets, with their source,
    custom_code: its code statements (imports, functions...),
    errors: invalid values of options,
    generated: whether it was written by philologic5-webui-loader.
    Raises SyntaxError if the text isn't valid Python."""
    config = ConfigFile(text)
    result = {
        "options": {},
        "code": {},
        "run_options": {},
        "unused": [],
        "unknown": {},
        "custom_code": custom_code(config),
        "errors": {},
        "generated": text.startswith(MARKER),
    }
    default_parser = is_default_parser(config)
    for name, entry in config.entries.items():
        if not entry.assigned or (default_parser and name in ("XMLParser", "parser_factory")):
            continue  # imports and definitions are in custom_code
        option = OPTIONS_BY_KEY.get(name)
        if name in RUN_OPTIONS:
            result["run_options"][name] = run_option_summary(entry.value)
        elif name in UNUSED_OPTIONS:
            result["unused"].append(name)
        elif option is None:
            result["unknown"][name] = source_of(entry)
        elif not entry.is_code and not (entry.value or entry.value is False or name in EMPTY_TAKEN):
            continue  # an empty value leaves the default
        elif entry.is_code or option.kind == "code":
            result["code"][name] = source_of(entry)
        else:
            value = sorted(entry.value) if isinstance(entry.value, (set, frozenset)) else entry.value
            error = load_schema.validate(name, json_value(value))
            if error:
                result["errors"][name] = error
            result["options"][name] = json_value(value)
    return result


def source_of(entry):
    """Source of the value of a name"""
    if entry.is_code:
        return entry.value.source
    return assignment_source("_", entry.value).split("=", 1)[1].strip()


def run_option_summary(value):
    """A run option as shown in the load pages: its value, or the number of items of a collection (such as the files
    of the load a config was saved by)"""
    if isinstance(value, Code):
        return value.source
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return f"{len(value)} items"
    return value


def read_file(path):
    """read() of a load config file, with its path"""
    with open(path, encoding="utf8") as config_file:
        result = read(config_file.read())
    result["path"] = os.path.abspath(path)
    return result


def database_config_path(db_path):
    """The copy of its load config which the loader saved in a database"""
    return os.path.join(db_path, "data", "load_config.py")


def validate(options):
    """Errors of the options sent by the load pages ({key: JSON value}): {key: message}"""
    errors = {}
    for key, value in options.items():
        error = load_schema.validate(key, value)
        if error:
            errors[key] = error
    return errors


def render(options, base_text=None):
    """render_unchecked(), checked: read back, the text must give the options as given, and keep the code of the base
    (if kept) and nothing else. Raises ValueError otherwise."""
    text = render_unchecked(options, base_text)
    try:
        written = ConfigFile(text)
    except SyntaxError as error:
        raise ValueError(f"the load config written isn't valid Python: {error}") from error
    base = ConfigFile(base_text) if base_text is not None else None
    expected_code = base.code_statements if base is not None and custom_code(base) else []
    if written.code_statements != expected_code:
        raise ValueError("the load config written has code it shouldn't")
    read_back = read(text)["options"]
    for key, value in options.items():
        if key in SOURCE_OPTIONS or key not in OPTIONS_BY_KEY or (base is not None and key in read(base_text)["code"]):
            continue
        if json_value(read_back.get(key, json_value(load_schema.default(key)))) != json_value(value) and not (
            load_schema.is_default(key, value) and key not in read_back
        ):
            raise ValueError(f"{key} could not be written as given")
    return text


def render_unchecked(options, base_text=None):
    """Text of a load config setting these options (JSON values of the load pages, already validated), apart from the
    source options, which go on philoload5's command line.
    Without a base, or with a base which has no code of its own (such as the copy saved in a database which was loaded
    without a load config), the config is written anew and only sets the options which differ from their default, so
    that later changes of the defaults apply to it. A base with code (its own parser or filters...) is edited in
    place, keeping its code and comments: only the assignments of changed options are replaced; its run options and
    unused options are removed."""
    values = {key: value for key, value in options.items() if key not in SOURCE_OPTIONS and key in OPTIONS_BY_KEY}
    base = ConfigFile(base_text) if base_text is not None else None
    if base is None or not custom_code(base):
        lines = [
            MARKER + ".",
            "# Only the options which differ from their default are set: see extras/load_config.py for all of them.",
        ]
        for option in OPTIONS:
            if option.key in values and not load_schema.is_default(option.key, values[option.key]):
                lines.append("")
                lines.extend(f"# {line}" for line in wrap(option.help))
                value = load_schema.to_config_value(option.key, values[option.key])
                lines.append(assignment_source(option.key, value).rstrip("\n"))
        return "\n".join(lines) + "\n"
    assignments = {}
    for key, value in values.items():
        entry = base.entries.get(key)
        value = load_schema.to_config_value(key, value)
        if entry is not None:
            if entry.is_code:
                continue
            current = entry.value if (entry.value or entry.value is False or key in EMPTY_TAKEN) else None
            if current is None:
                current = load_schema.default(key)
            if json_value(value) != json_value(current):
                assignments[key] = assignment_source(key, value)
        elif not load_schema.is_default(key, value):
            assignments[key] = assignment_source(key, value)
    comments = {key: "\n".join(wrap(OPTIONS_BY_KEY[key].help)) for key in assignments}
    removed = [name for name in base.entries if name in RUN_OPTIONS or name in UNUSED_OPTIONS]
    return base.edited(assignments, comments, removed)


def wrap(text, width=116):
    """Lines of a comment"""
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        lines.append(line)
    return lines
