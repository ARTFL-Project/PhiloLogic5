#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
from wsgiref.handlers import CGIHandler

import orjson

sys.path.append("..")
import custom_functions

try:
    from custom_functions import generate_time_series
except ImportError:
    from philologic.runtime import generate_time_series
try:
    from custom_functions import WebConfig
except ImportError:
    from philologic.runtime import WebConfig
try:
    from custom_functions import WSGIHandler
except ImportError:
    from philologic.runtime import WSGIHandler


def time_series(environ, start_response):
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    request = WSGIHandler(environ, config)
    time_series_object = generate_time_series(request, config)
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    yield orjson.dumps(time_series_object)


if __name__ == "__main__":
    CGIHandler().run(time_series)
