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

    all_collocates = orjson.loads(environ["wsgi.input"].read())["all_collocates"]

    # Create a contingency table for each collocate
    relative_proportions = []
    attrib = None
    attrib_value = None
    token_type = "word"
    if "lemma:" in request.q:
        if request.q.count(":") == 1:
            token_type = "lemma"
        else:
            attrib = request.q.split(":")[2]
            attrib_value = request.q.split(":")[3]
            token_type = f"lemma_{attrib}_{attrib_value}"
    elif request.q.count(":") == 2:
        attrib = request.q.split(":")[1]
        attrib_value = request.q.split(":")[2]
        token_type = f"word_{attrib}_{attrib_value}"
    total_corpus_words = db.get_total_word_count(token_type)
    with open(config.db_path + f"/data/frequencies/{token_type}_idf.pickle", "rb") as f:
        idf = pickle.load(f)
    total_words_in_window = sum(v["count"] for v in all_collocates.values())
    total_corpus_words -= total_words_in_window

    # Run collocation against whole corpus
    request.metadata = {}  # Clear metadata to run against whole corpus
    request.max_time = None  # fetch all results
    other_collocates = collocation_results(request, config)["collocates"]

    relative_proportions = []
    for collocate, value in all_collocates.items():
        collocate_count = value["count"]
        if collocate not in other_collocates:
            collocate_count_in_corpus = 0
        else:
            collocate_count_in_corpus = other_collocates[collocate]["count"] - collocate_count + 1
        sub_corpus_proportion = (1 + math.log(collocate_count)) / total_words_in_window
        corpus_proportion = (1 + math.log(collocate_count_in_corpus)) / total_corpus_words
        relative_proportion = (sub_corpus_proportion + 1) / (corpus_proportion + 1)
        relative_proportions.append({"collocate": collocate, "count": relative_proportion})

    relative_proportions = sorted(relative_proportions, key=lambda x: x["count"], reverse=True)

    top_relative_proportions = normalize_proportions(relative_proportions[:100])
    top_relative_proportions = [p for p in top_relative_proportions if p["count"] > 0]

    # add missing collocates from other_collocates
    relative_proportions = {p["collocate"]: p["count"] for p in relative_proportions}
    for collocate, value in other_collocates.items():
        if collocate not in relative_proportions:
            collocate_count_in_corpus = value["count"]
            sub_corpus_proportion = (1 + math.log(1)) / total_words_in_window
            corpus_proportion = (1 + math.log(collocate_count_in_corpus)) / total_corpus_words
            relative_proportion = (sub_corpus_proportion + 1) / (corpus_proportion + 1) * idf[collocate]
            relative_proportions[collocate] = relative_proportion
        else:
            relative_proportions[collocate] *= idf[collocate] * relative_proportions[collocate] / total_words_in_window

    relative_proportions = sorted(relative_proportions.items(), key=lambda x: x[1])[:100]
    relative_proportions = [{"collocate": p[0], "count": p[1]} for p in relative_proportions]
    low_relative_proportions = normalize_proportions(relative_proportions[:100])
    low_relative_proportions = [
        {"collocate": p["collocate"], "count": float(p["count"])} for p in low_relative_proportions if p["count"] < 0
    ]

    yield orjson.dumps({"top": top_relative_proportions, "bottom": low_relative_proportions})


def normalize_proportions(relative_proportions):
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
