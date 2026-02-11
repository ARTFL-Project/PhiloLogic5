#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson

from philologic.runtime import WebConfig, WSGIHandler, frequency_results

from custom_functions_loader import get_custom


def get_frequency(environ, start_response):
    """reads through a hitlist. looks up q.frequency_field in each hit, and builds up a list of
    unique values and their frequencies."""
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _frequency_results = get_custom(db_path, "frequency_results", frequency_results)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    results = _frequency_results(request, config)
    yield orjson.dumps(results)


if __name__ == "__main__":
    CGIHandler().run(get_frequency)
