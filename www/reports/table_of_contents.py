#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import generate_toc_object, WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def table_of_contents(environ, start_response):
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _generate_toc_object = get_custom(db_path, "generate_toc_object", generate_toc_object)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)

    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    toc_object = _generate_toc_object(request, config)
    yield orjson.dumps(toc_object)


if __name__ == "__main__":
    CGIHandler().run(table_of_contents)
