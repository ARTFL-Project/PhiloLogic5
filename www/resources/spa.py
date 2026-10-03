"""Falcon sink for SPA fallback — serves index.html for unmatched routes.

Handles access control (IP/domain checking, cookie setting) and
Brotli pre-compression, replicating the logic from webApp.py.
"""

import html
import os
import re

import falcon

from philologic.runtime import WebConfig, WSGIHandler, access_control
from middleware import PHILOLOGIC_DB_ROOT, is_database_name


def _build_misconfig_page(traceback, config_file):
    """Return bad config HTML page."""
    template_path = os.path.join(os.path.dirname(__file__), "..", "app", "misconfiguration.html")
    if not os.path.exists(template_path):
        return f"<pre>Configuration error in {config_file}:\n{traceback}</pre>"
    with open(template_path, encoding="utf8") as f:
        html_page = f.read()
    html_page = html_page.replace("$TRACEBACK", traceback)
    html_page = html_page.replace("$config_FILE", config_file)
    return html_page


def spa_handler(req, resp):
    """Serve the SPA index.html for any unmatched route under a database prefix."""
    # Extract db_path from URL
    db_path = None
    for part in req.path.split("/"):
        if is_database_name(part):
            db_path = os.path.join(PHILOLOGIC_DB_ROOT, part)
            break

    if db_path is None:
        resp.status = "404 Not Found"
        resp.content_type = "text/plain"
        resp.text = "Database not found"
        return

    config = WebConfig(db_path)

    if not config.valid_config:
        resp.content_type = "text/html; charset=UTF-8"
        resp.text = _build_misconfig_page(config.traceback, "webconfig.cfg")
        return

    # Set environ keys for WSGIHandler
    req.env["PHILOLOGIC_DBPATH"] = db_path
    db_name = os.path.basename(db_path)
    parts = req.path.split("/")
    if db_name in parts:
        idx = parts.index(db_name)
        req.env["PHILOLOGIC_DBURL"] = "/".join(parts[: idx + 1])
    else:
        req.env["PHILOLOGIC_DBURL"] = ""

    request = WSGIHandler(req.env, config)

    resp.content_type = "text/html; charset=UTF-8"

    # Access control: check IP/domain and set auth cookies if needed
    if config.access_control and not request.authenticated:
        cookie = access_control.check_access(req.env, config)
        if cookie:
            resp.append_header("Set-Cookie", cookie)

    # index.html with the database's own path as its base: the client is built with paths relative to it, so a
    # database can be served under any prefix, or copied elsewhere, without rebuilding it (and no url_root setting)
    index_path = os.path.join(config.db_path, "app", "dist", "index.html")
    with open(index_path, "rb") as f:
        resp.data = with_base(f.read(), f"{req.root_path}/{db_name}/")


_BASE_TAG = re.compile(rb"<base\b[^>]*>", re.I)


def with_base(index_html, base):
    """index_html with base as its <base href>: replacing the one it has, or else first in its <head>."""
    tag = f'<base href="{html.escape(base)}" />'.encode()
    if _BASE_TAG.search(index_html):
        return _BASE_TAG.sub(lambda _: tag, index_html, count=1)
    return re.sub(rb"(<head\b[^>]*>)", lambda m: m.group(1) + tag, index_html, count=1, flags=re.I)
