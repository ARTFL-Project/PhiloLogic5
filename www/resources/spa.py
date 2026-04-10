"""Falcon sink for SPA fallback — serves index.html for unmatched routes.

Handles access control (IP/domain checking, cookie setting) and
Brotli pre-compression, replicating the logic from webApp.py.
"""

import os

import falcon

from philologic.runtime import WebConfig, WSGIHandler, access_control
from middleware import PHILOLOGIC_DB_ROOT


def _build_misconfig_page(traceback, config_file):
    """Return bad config HTML page."""
    template_path = os.path.join(os.path.dirname(__file__), "..", "app", "misconfiguration.html")
    if not os.path.exists(template_path):
        return f"<pre>Configuration error in {config_file}:\n{traceback}</pre>"
    with open(template_path) as f:
        html_page = f.read()
    html_page = html_page.replace("$TRACEBACK", traceback)
    html_page = html_page.replace("$config_FILE", config_file)
    return html_page


def spa_handler(req, resp):
    """Serve the SPA index.html for any unmatched route under a database prefix."""
    # Extract db_path from URL
    db_path = None
    for part in req.path.split("/"):
        if part:
            candidate = os.path.join(PHILOLOGIC_DB_ROOT, part)
            if os.path.isdir(candidate):
                db_path = candidate
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
    if config.access_control:
        if not request.authenticated:
            token = access_control.check_access(req.env, config)
            if token:
                h, ts = token
                resp.append_header("Set-Cookie", "hash=%s; Path=/" % h)
                resp.append_header("Set-Cookie", "timestamp=%s; Path=/" % ts)

    # Serve Brotli-compressed index.html if client supports it
    accept_encoding = req.get_header("Accept-Encoding") or ""
    if "br" in accept_encoding:
        index_file = "index.html.br"
        resp.set_header("Content-Encoding", "br")
    else:
        index_file = "index.html"

    index_path = os.path.join(config.db_path, "app", "dist", index_file)
    with open(index_path, "rb") as f:
        resp.data = f.read()
