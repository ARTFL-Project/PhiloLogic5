#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import bibliography_results, WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def bibliography(environ, start_response):
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _bibliography_results = get_custom(db_path, "bibliography_results", bibliography_results)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    bibliography_object, _ = _bibliography_results(request, config)
    yield orjson.dumps(bibliography_object)


if __name__ == "__main__":
    CGIHandler().run(bibliography)
