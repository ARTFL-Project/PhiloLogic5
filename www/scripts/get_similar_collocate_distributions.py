#!/usr/bin/env python3

"""Get similar collocate distributions"""

import hashlib
import os
import pickle
from wsgiref.handlers import CGIHandler
import sys

from sklearn.metrics.pairwise import cosine_similarity
import orjson
import pandas as pd


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


def get_similar_collocate_distributions(environ, start_response):
    """Get similar collocate distributions"""
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
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)
    start_response(
        "200 OK",
        [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )
    post_data = orjson.loads(environ["wsgi.input"].read())
    reference_collocates = dict(post_data["collocates"])

    with open(request.file_path, "rb") as f:
        collocations_per_field = pickle.load(f)
    collocations_per_field["reference"] = reference_collocates

    colloc_df = pd.DataFrame(collocations_per_field.values(), index=collocations_per_field.keys())
    # move the reference distribution to the first row
    colloc_df = colloc_df.reindex(["reference"] + [field for field in colloc_df.index if field != "reference"])
    colloc_df.fillna(0, inplace=True)
    first_row = colloc_df.iloc[0].to_numpy().reshape(1, -1)
    rest_of_data = colloc_df.iloc[1:].values
    similarities = cosine_similarity(first_row, rest_of_data)[0]
    similarity_series = pd.Series(similarities, index=colloc_df.index[1:]).astype(float)
    similarity_series.sort_values(ascending=False, inplace=True)
    most_similar_distributions = [(k, v) for k, v in similarity_series.items()]
    yield orjson.dumps(
        {
            "most_similar_distributions": most_similar_distributions,
        }
    )


def create_file_path(request, field, field_value, path):
    hash = hashlib.sha1()
    hash.update(request["q"].encode("utf-8"))
    hash.update(request["method"].encode("utf-8"))
    hash.update(str(request["arg"]).encode("utf-8"))
    hash.update(f"{field}={field_value}".encode("utf-8"))
    return f"{path}/hitlists/{hash.hexdigest()}.pickle"


def get_distributions(reference_collocates, field_values, request, field, path):
    yield reference_collocates
    for field_value in field_values:
        file_path = create_file_path(request, field, field_value, path)
        with open(file_path, "rb") as f:
            collocates = pickle.load(f)
        yield collocates


if __name__ == "__main__":
    CGIHandler().run(get_similar_collocate_distributions)
