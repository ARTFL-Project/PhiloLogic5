#!/usr/bin/env python3
"""Calculate mutual information between two words."""

import math
import os
import pickle
import sys
from wsgiref.handlers import CGIHandler
import orjson
from philologic.runtime.DB import DB
from philologic.runtime.reports import collocation_results

import numpy as np
import lmdb


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

    post_data = environ["wsgi.input"].read()

    all_collocates = orjson.loads(post_data)["all_collocates"]
    other_corpus_metadata = orjson.loads(post_data)["other_corpus_metadata"]

    relative_proportions = []
    total_words_in_window = sum(v["count"] for v in all_collocates.values())

    whole_corpus = True
    if other_corpus_metadata:
        request.metadata = other_corpus_metadata
        whole_corpus = False
    else:
        # Run collocation against whole corpus
        request.metadata = {}  # Clear metadata to run against whole corpus
    request.max_time = None  # fetch all results
    other_collocates = collocation_results(request, config)["collocates"]
    other_total_words_in_window = sum(v["count"] for v in other_collocates.values())

    relative_proportions = []
    for collocate, value in all_collocates.items():
        collocate_count = value["count"] + 1  # Add 1 for Laplace smoothing
        sub_corpus_proportion = (1 + math.log(collocate_count)) / total_words_in_window
        if collocate not in other_collocates:
            other_corpus_collocate_count = 1  # Add 1 for Laplace smoothing
        else:
            other_corpus_collocate_count = other_collocates[collocate]["count"] + 1  # Add 1 for Laplace smoothing
        if whole_corpus is True:
            other_corpus_collocate_count -= collocate_count  # subtract sub corpus count to get rest of the corpus count
        if other_corpus_collocate_count == 0:
            other_corpus_collocate_count = 1
        other_corpus_proportion = (1 + math.log(other_corpus_collocate_count)) / other_total_words_in_window
        relative_proportion = (sub_corpus_proportion + 1) / (other_corpus_proportion + 1)
        relative_proportions.append({"label": collocate, "count": relative_proportion})

    relative_proportions = sorted(relative_proportions, key=lambda x: x["count"], reverse=True)

    # Get top relative proportions
    normalized_proportions = normalize_proportions(relative_proportions)
    top_relative_proportions = [p for p in normalized_proportions[:100] if p["count"] > 0]

    # Get bottom relative proportions
    normalized_proportions.sort(key=lambda x: x["count"])
    low_relative_proportions = [
        {"label": p["label"], "count": float(p["count"])} for p in normalized_proportions[:100] if p["count"] < 0
    ]

    yield orjson.dumps({"top": top_relative_proportions, "bottom": low_relative_proportions})


def normalize_proportions(relative_proportions) -> list:
    """Normalize relative proportions with:
    - L1 normalization
    - Scale proportions to enhance differences"""
    proportions_array = np.array([x["count"] for x in relative_proportions])
    sum_of_proportions = proportions_array.sum()
    normalized_scores = proportions_array / sum_of_proportions

    # Calculate average distinctiveness score
    average_score = sum(normalized_scores) / len(normalized_scores)

    # Calculate relative difference for each collocate
    for i, score in enumerate(normalized_scores):
        relative_difference = ((score - average_score) / average_score) * 100  # we are enhancing the difference
        relative_proportions[i]["count"] = float(relative_difference)

    return relative_proportions


if __name__ == "__main__":
    CGIHandler().run(get_collocation_relative_proportions)
