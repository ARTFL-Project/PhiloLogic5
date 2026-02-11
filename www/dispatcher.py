#!/var/lib/philologic5/philologic_env/bin/python3
"""Central WSGI dispatcher for PhiloLogic5.

Routes all requests for all databases through a single Gunicorn instance.
Extracts the database name from the URL path and dispatches to the
appropriate report, script, or static file handler.

URL structure: /prefix/dbname/action
    /philologic5/mydb/reports/concordance.py?q=test
    /philologic5/mydb/scripts/get_web_config.py
    /philologic5/mydb/assets/app.js
    /philologic5/mydb/                          → SPA index.html

Usage:
    gunicorn --config gunicorn.conf.py dispatcher:application
"""


import datetime
import mimetypes
import os
import sys
from random import randint

import reports
import scripts
from webApp import start_web_app

# Read central config to find the database root directory.
# /etc/philologic/philologic5.cfg uses Python syntax (e.g. database_root = "/var/www/philologic")
_CONFIG_FILE = "/etc/philologic/philologic5.cfg"
if "PHILOLOGIC_DB_ROOT" not in os.environ and os.path.exists(_CONFIG_FILE):
    _config = {}
    with open(_CONFIG_FILE) as _f:
        exec(_f.read(), _config)
    _db_root = _config.get("database_root")
    if _db_root and _db_root != "None":
        os.environ["PHILOLOGIC_DB_ROOT"] = _db_root

PHILOLOGIC_DB_ROOT = os.environ["PHILOLOGIC_DB_ROOT"]

# Whitelist of allowed report names
ALLOWED_REPORTS = {
    "concordance",
    "kwic",
    "bibliography",
    "collocation",
    "time_series",
    "navigation",
    "table_of_contents",
    "aggregation",
}

# Whitelist of allowed script names (filename without .py)
ALLOWED_SCRIPTS = {
    "access_request",
    "alignment_to_text",
    "autocomplete_metadata",
    "autocomplete_term",
    "collocation_time_series",
    "comparative_collocations",
    "export_results",
    "get_academic_citation",
    "get_bibliography",
    "get_collocate_distribution",
    "get_filter_list",
    "get_frequency",
    "get_header",
    "get_hitlist_stats",
    "get_landing_page_content",
    "get_more_context",
    "get_neighboring_words",
    "get_notes",
    "get_query_terms",
    "get_similar_collocate_distributions",
    "get_sorted_frequency",
    "get_sorted_kwic",
    "get_table_of_contents",
    "get_term_groups",
    "get_text_object",
    "get_total_results",
    "get_web_config",
    "get_word_frequency",
    "get_word_property_count",
    "resolve_cite",
}

# Static file URL prefixes → filesystem path prefixes (relative to db_path)
_STATIC_ROUTES = {
    "assets/": "app/dist/assets/",
    "img/": "app/dist/img/",
}


def _get_database_path(environ):
    """Extract the database name from the URL and return its filesystem path.

    Checks each URL path component against PHILOLOGIC_DB_ROOT to find
    the first one that corresponds to an actual database directory.
    """
    path_info = environ.get("PATH_INFO", "")
    for part in path_info.split("/"):
        if part:
            db_path = os.path.join(PHILOLOGIC_DB_ROOT, part)
            if os.path.isdir(db_path):
                return db_path
    return None


def _serve_static(db_path, relative_path, environ, start_response):
    """Serve static files from the database directory.

    Maps URL paths to filesystem paths:
        assets/* → app/dist/assets/*
        img/*    → app/dist/img/*
        favicon.ico → favicon.ico

    Returns response bytes, or None if not a static file request.
    """
    file_path = None
    for url_prefix, fs_prefix in _STATIC_ROUTES.items():
        if relative_path.startswith(url_prefix):
            file_path = os.path.join(db_path, fs_prefix, relative_path[len(url_prefix):])
            break

    if relative_path == "favicon.ico":
        file_path = os.path.join(db_path, "favicon.ico")

    if file_path is None:
        return None

    # Security: prevent path traversal
    file_path = os.path.realpath(file_path)
    if not file_path.startswith(os.path.realpath(db_path)):
        start_response("403 Forbidden", [("Content-type", "text/plain")])
        return b"Forbidden"

    if not os.path.isfile(file_path):
        start_response("404 Not Found", [("Content-type", "text/plain")])
        return b"Not Found"

    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    headers = [("Content-type", content_type)]

    # Serve Brotli-compressed version if client supports it
    accept_encoding = environ.get("HTTP_ACCEPT_ENCODING", "")
    if "br" in accept_encoding and os.path.isfile(file_path + ".br"):
        file_path = file_path + ".br"
        headers.append(("Content-Encoding", "br"))
        headers.append(("Vary", "Accept-Encoding"))

    with open(file_path, "rb") as f:
        data = f.read()

    headers.append(("Content-Length", str(len(data))))
    headers.append(("Cache-Control", "public, max-age=31536000, immutable"))
    start_response("200 OK", headers)
    return data


def philo_dispatcher(environ, start_response):
    """Central WSGI dispatcher for all PhiloLogic5 databases."""
    db_path = _get_database_path(environ)
    if db_path is None:
        start_response("404 Not Found", [("Content-type", "text/plain")])
        yield b"Database not found"
        return

    environ["PHILOLOGIC_DBPATH"] = db_path

    # Set CGI-compatible environ vars that WSGIHandler expects.
    # Apache CGI sets these automatically; Gunicorn does not.
    environ.setdefault("SCRIPT_FILENAME", os.path.join(db_path, "dispatcher.py"))
    db_name = os.path.basename(db_path)
    path_parts = environ.get("PATH_INFO", "").split("/")
    if db_name in path_parts:
        idx = path_parts.index(db_name)
        if not environ.get("SCRIPT_NAME"):
            environ["SCRIPT_NAME"] = "/".join(path_parts[: idx + 1]) + "/dispatcher.py"
        relative_path = "/".join(path_parts[idx + 1:])
    else:
        relative_path = ""

    # Static files (assets, img, favicon)
    static_response = _serve_static(db_path, relative_path, environ, start_response)
    if static_response is not None:
        yield static_response
        return

    # Scripts: /dbname/scripts/get_web_config.py
    path_info = environ.get("PATH_INFO", "")
    if "/scripts/" in path_info:
        name = path_info.split("/scripts/")[-1].replace(".py", "")
        if name in ALLOWED_SCRIPTS:
            yield from getattr(scripts, name)(environ, start_response)
            return
        print(f"SECURITY WARNING: Invalid script: {name!r} from {environ.get('REMOTE_ADDR', 'unknown')}",
              file=sys.stderr)
        start_response("400 Bad Request", [("Content-type", "text/plain")])
        yield b"Invalid script name"
        return

    # Reports: /dbname/reports/concordance.py
    if "/reports/" in path_info:
        name = path_info.split("/reports/")[-1].replace(".py", "")
        if name in ALLOWED_REPORTS:
            yield from getattr(reports, name)(environ, start_response)
            return
        print(f"SECURITY WARNING: Invalid report: {name!r} from {environ.get('REMOTE_ADDR', 'unknown')}",
              file=sys.stderr)
        start_response("400 Bad Request", [("Content-type", "text/plain")])
        yield b"Invalid report name"
        return

    # SPA fallback — serve the web app index.html
    yield start_web_app(environ, start_response)

    # Opportunistic hitlist cleanup
    if randint(0, 10) == 1:
        hitlist_dir = os.path.join(db_path, "data/hitlists")
        if os.path.isdir(hitlist_dir):
            try:
                for file in os.scandir(hitlist_dir):
                    if file.is_file():
                        file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file.path))
                        if datetime.datetime.now() - file_modified > datetime.timedelta(minutes=10):
                            try:
                                os.remove(file.path)
                            except OSError:
                                pass
            except OSError:
                pass


# WSGI entry point for Gunicorn
application = philo_dispatcher
