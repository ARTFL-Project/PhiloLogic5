#!/usr/bin/env python3
"""Calculate mutual information between two words."""

import math
import os
import pickle
import sys
from wsgiref.handlers import CGIHandler
import orjson
from philologic.runtime.DB import DB


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
    """Calculate normalized pointwise mutual information between two words."""
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
    db = DB(config.db_path + "/data/")
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
    results = []
    with open(frequency_file, "rb") as input_file:
        bigram_expected_frequency = pickle.load(input_file)
    for collocate, value in all_collocates.items():
        bigram = tuple((sorted([request["q"], collocate])))
        expected_frequency = bigram_expected_frequency[bigram]
        pmi = math.log2(value["count"] / expected_frequency)
        npmi = pmi / -math.log2(expected_frequency)
        results.append({"collocate": collocate, "count": npmi})
    results.sort(key=lambda x: x["count"], reverse=True)
    yield orjson.dumps(results[:100])


def get_word_prob(word, db):
    hits = db.query(word, raw_bytes=True, raw_results=True)
    hits.finish()
    return len(hits) / db.total_word_count


if __name__ == "__main__":
    CGIHandler().run(get_mutual_information)
