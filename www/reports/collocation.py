#!/usr/bin/env python3

import os
from wsgiref.handlers import CGIHandler

import orjson

import sys

sys.path.append("..")
import custom_functions

try:
    from custom_functions import collocation_results
except ImportError:
    from philologic.runtime import collocation_results
try:
    from custom_functions import WebConfig
except ImportError:
    from philologic.runtime import WebConfig
try:
    from custom_functions import WSGIHandler
except ImportError:
    from philologic.runtime import WSGIHandler


def collocation(environ, start_response):
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    request = WSGIHandler(environ, config)
    if environ["REQUEST_METHOD"] == "OPTIONS":
        # Handle preflight request
        start_response(
            "200 OK",
            [
                ("Content-Type", "text/plain"),
                ("Access-Control-Allow-Origin", environ["HTTP_ORIGIN"]),  # Replace with your client domain
                ("Access-Control-Allow-Methods", "POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "Content-Type"),  # Adjust if needed for your headers
            ],
        )
        return [b""]  # Empty response body for OPTIONS
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    post_data = environ["wsgi.input"].read()
    current_collocates = orjson.loads(post_data)["current_collocates"]
    collocation_object = collocation_results(request, config, current_collocates)
    yield orjson.dumps(collocation_object)


if __name__ == "__main__":
    CGIHandler().run(collocation)
