"""Falcon middleware for PhiloLogic5.

Replaces the boilerplate previously handled by @json_endpoint / @html_endpoint
decorators and the dispatcher.py routing logic.

Three middleware components:
    PhiloDBMiddleware  - database path extraction, config/request resolution
    CORSMiddleware     - Access-Control headers on every response
    CleanupMiddleware  - opportunistic hitlist file cleanup
"""

import datetime
import os
from random import randint

import falcon

from philologic.runtime import WebConfig, WSGIHandler, access_control
from philologic.runtime.hitlist_dir import get_hitlist_dir
from philologic.runtime.HitWrapper import SHARED_CACHE
from wsgi_helpers import resolve

# Read central config to find the database root directory.
_CONFIG_FILE = os.environ.get("PHILOLOGIC_CONFIG", "/etc/philologic/philologic5.cfg")
if "PHILOLOGIC_DB_ROOT" not in os.environ and os.path.exists(_CONFIG_FILE):
    _config = {}
    with open(_CONFIG_FILE, encoding="utf8") as _f:
        exec(_f.read(), _config)
    _db_root = _config.get("database_root")
    if _db_root and _db_root != "None":
        os.environ["PHILOLOGIC_DB_ROOT"] = _db_root

PHILOLOGIC_DB_ROOT = os.environ["PHILOLOGIC_DB_ROOT"]

# Scripts an access-controlled database answers whoever asks: the client needs them to show its login screen. So does
# every resource with a true "public" attribute.
PUBLIC_SCRIPTS = {"get_web_config"}


def is_database_name(part):
    """Whether part, a component of a URL path, names a database: a directory of PHILOLOGIC_DB_ROOT. "." and ".." are
    directories too, but they would make the database root, or its parent, pass for a database."""
    return part not in ("", ".", "..") and os.path.isdir(os.path.join(PHILOLOGIC_DB_ROOT, part))


def _get_database_path(path_info):
    """Extract the database filesystem path from the URL.

    Checks each URL path component against PHILOLOGIC_DB_ROOT to find
    the first one that corresponds to an actual database directory.
    Returns (db_path, db_name) or (None, None).
    """
    for part in path_info.split("/"):
        if is_database_name(part):
            return os.path.join(PHILOLOGIC_DB_ROOT, part), part
    return None, None


class PhiloDBMiddleware:
    """Extract database path from URL, resolve per-database config and request objects.

    Sets on req.context:
        db_path  - filesystem path to the database (e.g. /var/www/philologic/mydb)
        db_name  - database directory name (e.g. mydb)
        config   - WebConfig instance (possibly custom per-database)
        request  - WSGIHandler instance (possibly custom per-database)
    """

    def process_request(self, req, resp):
        """Clear shared cache and extract database path."""
        SHARED_CACHE.clear()

        db_path, db_name = _get_database_path(req.path)
        if db_path is None:
            raise falcon.HTTPNotFound(description="Database not found")

        req.context.db_path = db_path
        req.context.db_name = db_name

        # Set environ keys that WSGIHandler reads
        req.env["PHILOLOGIC_DBPATH"] = db_path
        parts = req.path.split("/")
        if db_name in parts:
            idx = parts.index(db_name)
            req.env["PHILOLOGIC_DBURL"] = "/".join(parts[: idx + 1])
        else:
            req.env["PHILOLOGIC_DBURL"] = ""

    def process_resource(self, req, resp, resource, params):
        """Resolve per-database WebConfig and WSGIHandler, attach to req.context. Refuse the clients an
        access-controlled database doesn't let in: those with no auth cookie whose address the access file doesn't
        allow."""
        if resource is None:
            return
        db_path = req.context.db_path
        _WebConfig = resolve(db_path, "WebConfig", WebConfig)
        _WSGIHandler = resolve(db_path, "WSGIHandler", WSGIHandler)
        config = req.context.config = _WebConfig(db_path)
        request = req.context.request = _WSGIHandler(req.env, config)
        if (
            config["access_control"]
            and not getattr(resource, "public", False)
            and params.get("script_name") not in PUBLIC_SCRIPTS
            and not request.authenticated
            and not access_control.is_allowed(req.env, config)
        ):
            raise falcon.HTTPForbidden(description="Access to this database is restricted")


class CORSMiddleware:
    """Add CORS headers to every response and handle OPTIONS preflight.

    Must be listed first in middleware so process_request runs before
    PhiloDBMiddleware — an OPTIONS preflight short-circuits via HTTPStatus
    before database resolution is attempted.
    """

    def process_request(self, req, resp):
        if req.method == "OPTIONS":
            raise falcon.HTTPStatus(falcon.HTTP_200, headers={
                "Access-Control-Allow-Origin": req.get_header("Origin") or "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
                "Access-Control-Max-Age": "86400",
            })

    def process_response(self, req, resp, resource, req_succeeded):
        # Not for access-controlled databases: any site could read them through the browsers of the clients they let in
        # by their address
        config = getattr(req.context, "config", None)
        if config is not None and config["access_control"]:
            return
        origin = req.get_header("Origin") or "*"
        resp.set_header("Access-Control-Allow-Origin", origin)


class CleanupMiddleware:
    """Opportunistic hitlist cleanup (1-in-10 chance per request)."""

    def process_response(self, req, resp, resource, req_succeeded):
        if randint(0, 10) != 1:
            return
        db_path = getattr(req.context, "db_path", None)
        if db_path is None:
            return
        hitlist_dir = get_hitlist_dir(os.path.join(db_path, "data"))
        if not os.path.isdir(hitlist_dir):
            return
        now = datetime.datetime.now()
        cutoff = datetime.timedelta(minutes=10)
        try:
            for entry in os.scandir(hitlist_dir):
                if entry.is_file():
                    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(entry.path))
                    if now - mtime > cutoff:
                        try:
                            os.remove(entry.path)
                        except OSError:
                            pass
        except OSError:
            pass
