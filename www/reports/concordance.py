#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import concordance_results, WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def concordance(environ, start_response):
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _concordance_results = get_custom(db_path, "concordance_results", concordance_results)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    concordance_object = _concordance_results(request, config)
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    yield orjson.dumps(concordance_object)


if __name__ == "__main__":
    CGIHandler().run(concordance)
