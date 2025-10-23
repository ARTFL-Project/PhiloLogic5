#!/var/lib/philologic5/philologic_env/bin/python3
"""Routing for PhiloLogic5."""


import datetime
import os
import sys
from random import randint
from typing import Callable
from urllib.parse import parse_qs, urlparse
from wsgiref.handlers import CGIHandler

import reports
from philologic.runtime import WebConfig, WSGIHandler
from webApp import start_web_app

path = os.path.abspath(os.path.dirname(__file__))

# Whitelist of allowed report names to prevent code injection
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


def philo_dispatcher(environ, start_response):
    """Dispatcher function."""
    config = WebConfig(path)
    request = WSGIHandler(environ, config)
    if request.content_type == "application/json" or request.format == "json":
        try:
            path_components = [c for c in environ["PATH_INFO"].split("/") if c]
        except Exception:
            path_components = []
        if path_components:
            if path_components[-1] == "table-of-contents":
                yield b"".join(reports.table_of_contents(environ, start_response))
            else:
                yield b"".join(reports.navigation(environ, start_response))
        else:
            try:
                report_name: str = parse_qs(environ["QUERY_STRING"])["report"][0]
            except KeyError:
                report_name = urlparse(environ["REQUEST_URI"]).path.split("/")[-1]

            # Security: Validate report name against whitelist
            if report_name not in ALLOWED_REPORTS:
                # Log the security violation
                print(f"SECURITY WARNING: Invalid report name attempted: {report_name!r} from {environ.get('REMOTE_ADDR', 'unknown')}",
                      file=sys.stderr)
                start_response("400 Bad Request", [("Content-type", "text/plain")])
                yield b"Invalid report name"
                return

            report: Callable = getattr(reports, report_name)
            yield b"".join(report(environ, start_response))
    elif request.full_bibliography is True:
        yield b"".join(reports.bibliography(environ, start_response))
    else:
        yield start_web_app(environ, start_response)

    # clean-up hitlist every now and then
    if randint(0, 10) == 1:
        for file in os.scandir(os.path.join(path, "data/hitlists/*")):
            file_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file.path))
            if datetime.datetime.now() - file_modified > datetime.timedelta(minutes=10):
                os.remove(file.path)


if __name__ == "__main__":
    CGIHandler().run(philo_dispatcher)
