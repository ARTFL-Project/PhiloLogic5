#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import generate_time_series, WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def time_series(environ, start_response):
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _generate_time_series = get_custom(db_path, "generate_time_series", generate_time_series)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    time_series_object = _generate_time_series(request, config)
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    yield orjson.dumps(time_series_object)


if __name__ == "__main__":
    CGIHandler().run(time_series)
