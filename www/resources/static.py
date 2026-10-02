"""Falcon resource for serving static files from database directories.

Handles assets, images, and favicon with Brotli pre-compression support
and path traversal protection.
"""

import mimetypes
import os

import falcon

from middleware import PHILOLOGIC_DB_ROOT, is_database_name

_ROUTE_MAP = {
    "assets": "app/dist/assets/",
    "img": "app/dist/img/",
}


class StaticResource:
    """Serve static files from a database's app/dist/ directory."""

    public = True  # the client's code, which shows the login screen of access-controlled databases

    def __init__(self, route_type):
        self.route_type = route_type

    def on_get(self, req, resp, db_name, filepath=None):
        if not is_database_name(db_name):
            raise falcon.HTTPNotFound()
        db_path = os.path.join(PHILOLOGIC_DB_ROOT, db_name)

        if self.route_type == "favicon":
            file_path = os.path.join(db_path, "favicon.ico")
        else:
            # Path traversal protection: filepath may hold "..", as is or %-encoded, and the file must stay in the
            # route's own directory (being in the database's isn't enough: data/ holds logins.txt, the texts...)
            base = os.path.realpath(os.path.join(db_path, _ROUTE_MAP[self.route_type]))
            file_path = os.path.realpath(os.path.join(base, filepath))
            if os.path.commonpath([base, file_path]) != base:
                raise falcon.HTTPForbidden()

        if not os.path.isfile(file_path):
            raise falcon.HTTPNotFound()

        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        resp.content_type = content_type

        # Serve Brotli-compressed version if client supports it
        accept_encoding = req.get_header("Accept-Encoding") or ""
        if "br" in accept_encoding and os.path.isfile(file_path + ".br"):
            file_path = file_path + ".br"
            resp.set_header("Content-Encoding", "br")
            resp.set_header("Vary", "Accept-Encoding")

        resp.set_header("Cache-Control", "public, max-age=31536000, immutable")

        with open(file_path, "rb") as f:
            resp.data = f.read()
