#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import concordance_results

from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def get_word_frequency(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    word_frequency_object = generate_word_frequency(request, config)
    yield orjson.dumps(word_frequency_object)


if __name__ == "__main__":
    CGIHandler().run(get_word_frequency)
