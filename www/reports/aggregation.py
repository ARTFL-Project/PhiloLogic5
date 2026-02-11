#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import aggregation_by_field, WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def aggregation(environ, start_response):
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _aggregation_by_field = get_custom(db_path, "aggregation_by_field", aggregation_by_field)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    aggregation_object = _aggregation_by_field(request, config)
    headers = [
        ("Content-type", "application/json; charset=UTF-8"),
        ("Access-Control-Allow-Origin", "*"),
    ]
    start_response("200 OK", headers)
    yield orjson.dumps(aggregation_object)


if __name__ == "__main__":
    CGIHandler().run(aggregation)
