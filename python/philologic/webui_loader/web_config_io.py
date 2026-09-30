"""Web configs (data/web_config.cfg) in philologic5-webui-loader: reading them without running them, and saving
edits, which only replace the assignments of the edited options (comments and the rest of the file are kept).
access_control and access_file are never changed. The runtime reads web_config.cfg again at each request, so a saved
edit applies from the next page load."""

import copy
import datetime
import grp
import hashlib
import os
import pwd
import re
import shutil
import stat
import tempfile

from philologic.Config import WEB_CONFIG_DEFAULTS
from philologic.webui_loader.load_schema import json_value
from philologic.webui_loader.pyconfig import Code, ConfigFile, assignment_source

REPORTS = ("concordance", "kwic", "aggregation", "collocation", "time_series")
INPUT_STYLES = ("text", "dropdown", "checkbox", "int", "date")
LANDING_PAGES = ("default", "dictionary", "simple", "toc")

# Never changed by the web config page
READ_ONLY = ("access_control", "access_file")

# Lists of tuples, which JSON turns into lists of lists
TUPLE_LISTS = (
    "concordance_biblio_sorting",
    "concordance_formatting_regex",
    "kwic_formatting_regex",
    "navigation_formatting_regex",
    "query_parser_regex",
)

GROUPS = (
    "general",
    "search",
    "results",
    "citations",
    "landing_page",
    "navigation",
    "time_series",
    "aggregation",
    "dictionary",
    "formatting",
    "access",
)

# key: (group, kind, choices). Kinds: bool, int, string, choice, reports (a selection of REPORTS), fields (metadata
# fields), field (one metadata field), field_map (metadata field -> string, value_choices if given), string_list, json
# (any other structure, edited as JSON).
KINDS = {
    "dbname": ("general", "string", ()),
    "link_to_home_page": ("general", "string", ()),
    "logo": ("general", "string", ()),
    "report_error_link": ("general", "string", ()),
    "academic_citation": ("general", "json", ()),
    "search_reports": ("search", "reports", REPORTS),
    "metadata": ("search", "fields", ()),
    "metadata_aliases": ("search", "field_map", ()),
    "metadata_input_style": ("search", "field_map", INPUT_STYLES),
    "metadata_choice_values": ("search", "json", ()),
    "word_property_aliases": ("search", "json", ()),
    "autocomplete": ("search", "fields", ()),
    "search_examples": ("search", "field_map", ()),
    "word_attributes": ("search", "json", ()),
    "query_parser_regex": ("search", "json", ()),
    "search_syntax_template": ("search", "string", ()),
    "results_summary": ("results", "json", ()),
    "concordance_length": ("results", "int", ()),
    "facets": ("results", "fields", ()),
    "words_facets": ("results", "string_list", ()),
    "kwic_bibliography_fields": ("results", "fields", ()),
    "concordance_biblio_sorting": ("results", "json", ()),
    "kwic_metadata_sorting_fields": ("results", "fields", ()),
    "collocation_fields_to_compare": ("results", "fields", ()),
    "stopwords": ("results", "string", ()),
    "citations": ("citations", "json", ()),
    "concordance_citation": ("citations", "json", ()),
    "bibliography_citation": ("citations", "json", ()),
    "table_of_contents_citation": ("citations", "json", ()),
    "navigation_citation": ("citations", "json", ()),
    "simple_landing_citation": ("citations", "json", ()),
    "landing_page_browsing": ("landing_page", "choice", LANDING_PAGES),
    "default_landing_page_browsing": ("landing_page", "json", ()),
    "default_landing_page_display": ("landing_page", "json", ()),
    "dico_letter_range": ("landing_page", "string_list", ()),
    "skip_table_of_contents": ("navigation", "bool", ()),
    "respect_text_line_breaks": ("navigation", "bool", ()),
    "header_in_toc": ("navigation", "bool", ()),
    "external_page_images": ("navigation", "bool", ()),
    "page_images_url_root": ("navigation", "string", ()),
    "page_image_extension": ("navigation", "string", ()),
    "time_series_year_field": ("time_series", "field", ()),
    "time_series_interval": ("time_series", "int", ()),
    "time_series_start_end_date": ("time_series", "json", ()),
    "aggregation_config": ("aggregation", "json", ()),
    "dictionary": ("dictionary", "bool", ()),
    "dictionary_bibliography": ("dictionary", "bool", ()),
    "dictionary_selection": ("dictionary", "bool", ()),
    "dictionary_selection_options": ("dictionary", "json", ()),
    "dictionary_lookup": ("dictionary", "json", ()),
    "dictionary_lookup_keywords": ("dictionary", "json", ()),
    "concordance_formatting_regex": ("formatting", "json", ()),
    "kwic_formatting_regex": ("formatting", "json", ()),
    "navigation_formatting_regex": ("formatting", "json", ()),
    "access_control": ("access", "bool", ()),
    "access_file": ("access", "string", ()),
}


# Options naming files which the runtime reads, with their values which aren't files: in the service, the files must
# be inside the database (a relative path without ..), so that its users can't make the runtime read other files
FILE_OPTIONS = {"stopwords": ("",), "landing_page_browsing": LANDING_PAGES, "search_syntax_template": ("default",)}


def service_file_problem(key, value):
    """Error message if a value names a file outside the database, else None"""
    if key not in FILE_OPTIONS or not isinstance(value, str) or value in FILE_OPTIONS[key]:
        return None
    if os.path.isabs(value) or ".." in value.replace("\\", "/").split("/"):
        return "must be a path inside the database, without .."
    return None


# Options whose strings the pages of a database use as links
URL_OPTIONS = (
    "link_to_home_page",
    "report_error_link",
    "logo",
    "page_images_url_root",
    "dictionary_lookup",
    "academic_citation",
)
URL_SCHEME = re.compile(r"^\s*([a-zA-Z][a-zA-Z0-9+.-]*):")


def strings(value):
    """All the strings of a value, keys included"""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from strings(key)
            yield from strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from strings(item)


def restricted_problem(key, value, current):
    """Error message if a value adds what only admins can set: HTML (citations and others are shown as HTML by the
    pages of the database), links which aren't http(s), or custom templates. Strings already in the current value can
    be kept."""
    existing = set(strings(current))
    new = [string for string in strings(value) if string not in existing]
    if any("<" in string or ">" in string for string in new):
        return "only admins can set HTML"
    if key in URL_OPTIONS:
        for string in new:
            scheme = URL_SCHEME.match(string)
            if scheme and scheme.group(1).lower() not in ("http", "https"):
                return "links must be http:// or https://"
    if (
        key in ("landing_page_browsing", "search_syntax_template")
        and value not in FILE_OPTIONS[key]
        and value != current
    ):
        return "only admins can set custom templates"
    return None


class WebConfigError(Exception):
    """A web config which can't be edited or saved"""


def defaults():
    """The default values of the web config options (a copy)"""
    return {key: copy.deepcopy(value["value"]) for key, value in WEB_CONFIG_DEFAULTS.items()}


def schema():
    """The web config options as JSON, for the web config page"""
    options = []
    for key, value in WEB_CONFIG_DEFAULTS.items():
        group, kind, choices = KINDS.get(key, ("general", "json", ()))
        options.append(
            {
                "key": key,
                "group": group,
                "kind": kind,
                "choices": list(choices),
                "default": json_value(value["value"]),
                "help": " ".join(line.lstrip("#").strip() for line in value["comment"].splitlines()).strip(),
                "read_only": key in READ_ONLY,
            }
        )
    return {"groups": list(GROUPS), "options": options}


def paths(db_path):
    data_dir = os.path.join(db_path, "data")
    return data_dir, os.path.join(data_dir, "web_config.cfg")


def user_name(uid):
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def group_name(gid):
    try:
        return grp.getgrgid(gid).gr_name
    except KeyError:
        return str(gid)


def write_permission(db_path):
    """(writable, reason): whether this process can save the web config of a database, and why not"""
    data_dir, config_path = paths(db_path)
    if not os.path.isfile(config_path):
        return False, "the database has no web_config.cfg"
    me = user_name(os.geteuid())
    for path, what in ((config_path, "web_config.cfg"), (data_dir, "its data directory")):
        if not os.access(path, os.W_OK):
            info = os.stat(path)
            group_mode = "group-writable" if info.st_mode & stat.S_IWGRP else "not group-writable"
            return False, (
                f"{what} belongs to {user_name(info.st_uid)} (group {group_name(info.st_gid)}, {group_mode}) "
                f"and can't be written by {me}"
            )
    return True, ""


def metadata_fields(db_path):
    """Metadata fields of a database, from its db.locals.py"""
    try:
        with open(os.path.join(db_path, "data", "db.locals.py"), encoding="utf8") as db_locals:
            values = ConfigFile(db_locals.read()).values
    except (OSError, SyntaxError):
        return []
    return [field for field in values.get("metadata_fields", []) if isinstance(field, str)]


def read(db_path, service=False):
    """The web config of a database as JSON for the web config page: values of all options (defaults for those it
    doesn't set), which it sets, those set by code (with their source, not editable), other names it sets, whether it
    can be saved (and why not), the metadata fields of the database, and a hash to detect concurrent changes. In the
    service, a web config with code (anything but assignments of literals) can't be saved."""
    data_dir, config_path = paths(db_path)
    try:
        with open(config_path, encoding="utf8") as config_file:
            text = config_file.read()
    except FileNotFoundError:
        text = None
    values = defaults()
    result = {"values": {}, "in_file": [], "code": {}, "other": {}, "error": None}
    if text is not None:
        try:
            config = ConfigFile(text, defaults())
        except SyntaxError as error:
            result["error"] = f"web_config.cfg isn't valid Python: {error}"
            config = None
        if config is not None:
            for name, entry in config.entries.items():
                if not entry.assigned:
                    result["code"][name] = entry.value.source
                elif entry.is_code:
                    result["code"][name] = entry.value.source
                elif name in values:
                    values[name] = entry.value
                    result["in_file"].append(name)
                else:
                    result["other"][name] = json_value(entry.value)
    writable, reason = write_permission(db_path) if text is not None else (False, "the database has no web_config.cfg")
    if result["error"]:
        writable, reason = False, result["error"]
    elif service and config is not None and config.code_statements:
        writable, reason = False, "web_config.cfg has code, which can only be edited from a shell"
    result.update(
        {
            "values": {key: json_value(value) for key, value in values.items()},
            "writable": writable,
            "reason": reason,
            "metadata_fields": metadata_fields(db_path),
            "hash": hashlib.sha256(text.encode("utf8")).hexdigest() if text is not None else None,
        }
    )
    return result


def from_json(key, value):
    """A value received as JSON, with the tuples of lists of tuples restored"""
    if key in TUPLE_LISTS and isinstance(value, list):
        return [tuple(item) if isinstance(item, list) else item for item in value]
    return value


def validate(key, value):
    """Error message if a value isn't valid for an option (checked against the type of its default), else None"""
    if key in READ_ONLY:
        return "this option can't be changed here"
    if key not in WEB_CONFIG_DEFAULTS:
        return "unknown option"
    group, kind, choices = KINDS.get(key, ("general", "json", ()))
    default = WEB_CONFIG_DEFAULTS[key]["value"]
    if kind == "bool":
        return None if isinstance(value, bool) else "must be true or false"
    if kind == "int":
        return None if isinstance(value, int) and not isinstance(value, bool) else "must be a whole number"
    if kind in ("string", "field"):
        return None if isinstance(value, str) else "must be a string"
    if kind == "choice":
        # a custom HTML template can also be given by its path
        return None if isinstance(value, str) and value else "must be a string"
    if kind in ("reports", "fields", "string_list"):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return "must be a list of strings"
        if kind == "reports" and any(item not in choices for item in value):
            return f"must be a selection of {', '.join(choices)}"
        return None
    if kind == "field_map":
        if not isinstance(value, dict) or not all(isinstance(v, str) for v in value.values()):
            return "must map fields to strings"
        if choices and any(v not in choices for v in value.values()):
            return f"values must be one of {', '.join(choices)}"
        return None
    # json: same type as the default at the top level
    if isinstance(default, (list, tuple)) and not isinstance(value, list):
        return "must be a list"
    if isinstance(default, dict) and not isinstance(value, dict):
        return "must be an object"
    return None


def citation_references(config, key, citations):
    """The citations which the value of an option can refer to as citations["name"]: those of a citations set before
    it in the file, or the default citations if the file doesn't set any (the runtime runs web configs with the defaults
    already set)"""
    if key == "citations" or not isinstance(citations, dict):
        return ()
    citations_entry = config.entries.get("citations")
    entry = config.entries.get(key)
    if citations_entry is not None:
        if citations_entry.statement is None:
            return ()
        if entry is not None and entry.statement.lineno < citations_entry.statement.lineno:
            return ()
    return [(f'citations["{name}"]', citation) for name, citation in citations.items()]


def save(db_path, changes, expected_hash, service=False, restricted=False):
    """Save edits of the web config of a database ({key: JSON value}), if it hasn't changed since it was read (its
    hash). Returns the path of the backup of the previous version. Raises WebConfigError. In the service, files read
    by the runtime must be in the database, and a web config with code can't be edited; restricted (users other than
    admins) can't add HTML, links other than http(s), or custom templates, which the pages of the database show."""
    writable, reason = write_permission(db_path)
    if not writable:
        raise WebConfigError(reason)
    data_dir, config_path = paths(db_path)
    with open(config_path, encoding="utf8") as config_file:
        text = config_file.read()
    if hashlib.sha256(text.encode("utf8")).hexdigest() != expected_hash:
        raise WebConfigError("web_config.cfg has changed since it was read: reload it before saving")
    config = ConfigFile(text, defaults())
    if service and config.code_statements:
        raise WebConfigError("web_config.cfg has code, which can only be edited from a shell")
    current = defaults()
    current.update({name: entry.value for name, entry in config.entries.items() if not entry.is_code})
    errors = {}
    values = {}
    for key, value in changes.items():
        error = validate(key, value) or (service_file_problem(key, value) if service else None)
        if error is None and restricted:
            error = restricted_problem(key, value, current.get(key))
        entry = config.entries.get(key)
        if error is None and entry is not None and (entry.is_code or entry.statement is None):
            error = "set by code in web_config.cfg, which can't be edited here"
        if error:
            errors[key] = error
        else:
            values[key] = from_json(key, value)
    if errors:
        raise WebConfigError("; ".join(f"{key}: {error}" for key, error in errors.items()))
    values = {key: value for key, value in values.items() if json_value(value) != json_value(current.get(key))}
    if not values:
        return None
    citations = values.get("citations", current.get("citations", {}))
    citations_entry = config.entries.get("citations")
    comments = {
        key: " ".join(line.lstrip("#").strip() for line in WEB_CONFIG_DEFAULTS[key]["comment"].splitlines())
        for key in values
        if key not in config.entries
    }
    to_write = dict(values)
    if "citations" in to_write and citations_entry is None:
        # Added at the end first, so that the other options added after it can refer to it
        added = config.edited({"citations": assignment_source("citations", to_write.pop("citations"))}, comments)
        config = ConfigFile(added, defaults())
    assignments = {
        key: assignment_source(key, value, citation_references(config, key, citations))
        for key, value in to_write.items()
    }
    new_text = config.edited(assignments, comments)
    check = ConfigFile(new_text, defaults())
    compile(new_text, config_path, "exec")
    if check.code_statements != config.code_statements:
        raise WebConfigError("the edit would change the code of web_config.cfg")
    for key, value in values.items():
        if check.values.get(key) != value:
            raise WebConfigError(f"{key} could not be written as given")
    effective = defaults()
    effective.update(check.values)
    for key in READ_ONLY:
        if effective[key] != current[key]:
            raise WebConfigError(f"{key} would have changed")
    backup = f"{config_path}.{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.bak"
    shutil.copy2(config_path, backup)
    mode = stat.S_IMODE(os.stat(config_path).st_mode)
    handle, temporary = tempfile.mkstemp(dir=data_dir, prefix=".web_config.cfg.")
    try:
        with os.fdopen(handle, "w", encoding="utf8") as temporary_file:
            temporary_file.write(new_text)
        os.chmod(temporary, mode)
        os.replace(temporary, config_path)
    except BaseException:
        if os.path.exists(temporary):
            os.remove(temporary)
        raise
    return backup
