#!/usr/bin/env python3
"""Calculate mutual information between two words."""

import math
import os
import pickle
import sys
from wsgiref.handlers import CGIHandler
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


def get_mutual_information(environ, start_response):
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
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)
    all_collocates = orjson.loads(environ["wsgi.input"].read())["all_collocates"]
    collocate_distance = request.collocate_distance
    if "lemma:" in request["q"]:
        word = "lemma"
    else:
        word = "word"
    frequency_file = (
        config.db_path + f"/data/frequencies/expected_bigram_frequencies_{word}_{collocate_distance}.pickle"
    )
    with open(frequency_file, "rb") as input_file:
        bigram_expected_frequency = pickle.load(input_file)
    for collocate in all_collocates:
        bigram = (request["q"], collocate)
        bigram_frequency = all_collocates[collocate]["count"]
        expected_frequency = bigram_expected_frequency[bigram]
        all_collocates[collocate] = math.log2(bigram_frequency / expected_frequency)
    yield orjson.dumps(all_collocates)


if __name__ == "__main__":
    CGIHandler().run(get_mutual_information)
