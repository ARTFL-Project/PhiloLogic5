"""Falcon application for PhiloLogic5.

Replaces dispatcher.py as the Gunicorn WSGI entry point.

Usage:
    gunicorn --config gunicorn.conf.py app:application
"""

import os

import falcon

from middleware import (
    PHILOLOGIC_DB_ROOT,
    CORSMiddleware,
    CleanupMiddleware,
    PhiloDBMiddleware,
)
from resources.reports import ReportResource
from resources.scripts import ScriptResource
from resources.spa import spa_handler
from resources.static import StaticResource
from resources.streaming import (
    AccessRequestResource,
    ExportResultsResource,
    ResolveCiteResource,
    SortedKWICResource,
)


class StripURLPrefix:
    """WSGI middleware that strips the deployment URL prefix from PATH_INFO.

    PhiloLogic5 databases live at URLs like /philologic5/mydb/scripts/...
    where /philologic5 is a deployment prefix (determined by the web server).
    Falcon routes expect /{db_name}/scripts/... so we need to strip the prefix.

    This middleware finds the database name in the URL path (by checking each
    component against PHILOLOGIC_DB_ROOT) and strips everything before it,
    adjusting SCRIPT_NAME and PATH_INFO per the WSGI spec.
    """

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")
        parts = path.split("/")
        for i, part in enumerate(parts):
            if part and os.path.isdir(os.path.join(PHILOLOGIC_DB_ROOT, part)):
                prefix = "/".join(parts[:i])
                environ["SCRIPT_NAME"] = environ.get("SCRIPT_NAME", "") + prefix
                environ["PATH_INFO"] = "/" + "/".join(parts[i:])
                break
        return self.app(environ, start_response)


def create_app():
    falcon_app = falcon.App(middleware=[
        CORSMiddleware(),
        PhiloDBMiddleware(),
        CleanupMiddleware(),
    ])

    # Reports: /{db_name}/reports/{report_name}.py
    falcon_app.add_route("/{db_name}/reports/{report_name}.py", ReportResource())

    # Special scripts (streaming, auth, redirect)
    falcon_app.add_route("/{db_name}/scripts/get_sorted_kwic.py", SortedKWICResource())
    falcon_app.add_route("/{db_name}/scripts/export_results.py", ExportResultsResource())
    falcon_app.add_route("/{db_name}/scripts/access_request.py", AccessRequestResource())
    falcon_app.add_route("/{db_name}/scripts/resolve_cite.py", ResolveCiteResource())

    # Standard JSON/HTML scripts: /{db_name}/scripts/{script_name}.py
    falcon_app.add_route("/{db_name}/scripts/{script_name}.py", ScriptResource())

    # Static files
    falcon_app.add_route("/{db_name}/assets/{filepath:path}", StaticResource("assets"))
    falcon_app.add_route("/{db_name}/img/{filepath:path}", StaticResource("img"))
    falcon_app.add_route("/{db_name}/favicon.ico", StaticResource("favicon"))

    # SPA fallback — catches all unmatched routes
    falcon_app.add_sink(spa_handler, prefix="/")

    return StripURLPrefix(falcon_app)


application = create_app()
