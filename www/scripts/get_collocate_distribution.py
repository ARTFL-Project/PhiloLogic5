#! /usr/bin/env python3

import os
import pickle
from wsgiref.handlers import CGIHandler
import sys

import orjson


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


def get_collocate_distribution(environ, start_response):
    """Get collocate distribution"""
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)
    start_response(
        "200 OK",
        [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )

    with open(request.file_path, "rb") as f:
        collocations_per_field = pickle.load(f)
    collocates = sorted(collocations_per_field[request.field].items(), key=lambda x: x[1], reverse=True)

    yield orjson.dumps({"collocates": collocates})


if __name__ == "__main__":
    CGIHandler().run(get_collocate_distribution)
