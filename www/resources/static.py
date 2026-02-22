"""Falcon resource for serving static files from database directories.

Handles assets, images, and favicon with Brotli pre-compression support
and path traversal protection.
"""

import mimetypes
import os

import falcon

from middleware import PHILOLOGIC_DB_ROOT

_ROUTE_MAP = {
    "assets": "app/dist/assets/",
    "img": "app/dist/img/",
}


class StaticResource:
    """Serve static files from a database's app/dist/ directory."""

    def __init__(self, route_type):
        self.route_type = route_type

    def on_get(self, req, resp, db_name, filepath=None):
        db_path = os.path.join(PHILOLOGIC_DB_ROOT, db_name)
        if not os.path.isdir(db_path):
            raise falcon.HTTPNotFound()

        if self.route_type == "favicon":
            file_path = os.path.join(db_path, "favicon.ico")
        else:
            fs_prefix = _ROUTE_MAP.get(self.route_type, "")
            file_path = os.path.join(db_path, fs_prefix, filepath)

        # Path traversal protection
        file_path = os.path.realpath(file_path)
        if not file_path.startswith(os.path.realpath(db_path)):
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
