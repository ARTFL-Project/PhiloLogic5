#!/usr/bin/env python3
"""Calculate mutual information between two words."""

import math
import os
import pickle
import sys
from wsgiref.handlers import CGIHandler
import orjson
from philologic.runtime.DB import DB
from philologic.runtime.Query import get_expanded_query

import numpy as np
import lmdb
from sklearn.preprocessing import RobustScaler


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


def get_collocation_relative_proportions(environ, start_response):
    """Calculate relative proportion of each collocate."""
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

    with open(f"{config.db_path}/data/frequencies/idf.pickle", "rb") as f:
        idf = pickle.load(f)

    # Reweigh collocates using tf-idf
    relative_frequencies = []
    for collocate, value in all_collocates.items():
        sub_linear_tf = 1 + math.log(value["count"])
        tf_idf_score = sub_linear_tf * float(value["count"] * idf[collocate])
        relative_frequencies.append({"collocate": collocate, "count": tf_idf_score})

    relative_frequencies.sort(key=lambda x: x["count"], reverse=True)
    yield orjson.dumps(relative_frequencies[:100])


if __name__ == "__main__":
    CGIHandler().run(get_collocation_relative_proportions)
