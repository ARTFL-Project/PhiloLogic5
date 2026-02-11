#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

from philologic.runtime import get_tei_header
from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def get_header(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "text/html; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    header = get_tei_header(request, config)
    yield header.encode("utf8")


if __name__ == "__main__":
    CGIHandler().run(get_header)
