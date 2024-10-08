#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import build_filter_list

sys.path.append("..")
import custom_functions

try:
    from custom_functions import WebConfig
except ImportError:
    from philologic.runtime import WebConfig
try:
    from custom_functions import WSGIHandler
except ImportError:
    from philologic.runtime import WSGIHandler


def get_filter_list(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)
    filter_list = build_filter_list(request, config)
    yield orjson.dumps(filter_list)


if __name__ == "__main__":
    CGIHandler().run(get_filter_list)
