#!/usr/bin/env python3
"""Calculate mutual information between two words."""

import math
import os
import sys
from wsgiref.handlers import CGIHandler
import orjson
from philologic.runtime.reports import collocation_results

import numpy as np
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

    top_relative_proportions, low_relative_proportions = get_relative_proportions(
        all_collocates, other_collocates, total_words_in_window, other_total_words_in_window, whole_corpus
    )

    yield orjson.dumps(
        {"top": top_relative_proportions, "bottom": low_relative_proportions, "other_collocates": other_collocates}
    )


def get_relative_proportions(
    all_collocates, other_collocates, total_words_in_window, other_total_words_in_window, whole_corpus
):
    # Create DataFrames
    df_sub = pd.DataFrame.from_dict(
        {k: v["count"] for k, v in all_collocates.items()}, orient="index", columns=["sub_corpus_count"]
    )
    df_other = pd.DataFrame.from_dict(
        {k: v["count"] for k, v in other_collocates.items()}, orient="index", columns=["other_corpus_count"]
    )

    # Outer Join (Preserves all collocates)
    df_combined = df_sub.join(df_other, how="outer").fillna(0)

    df_combined.to_csv("/tmp/combined.csv")

    # Adjust counts if comparing against the whole corpus
    if whole_corpus:
        df_combined["other_corpus_count"] = df_combined["other_corpus_count"] - df_combined["sub_corpus_count"]
        df_combined.loc[df_combined["other_corpus_count"] == 0, "other_corpus_count"] = 1

    # Calculate Proportions (Replace with your total words count variables)
    df_combined["sub_corpus_proportion"] = (df_combined["sub_corpus_count"] + 1).apply(np.log) / total_words_in_window
    df_combined["other_corpus_proportion"] = (df_combined["other_corpus_count"] + 1).apply(
        np.log
    ) / other_total_words_in_window

    # Relative proportion
    df_combined["relative_proportion"] = (df_combined["sub_corpus_proportion"] + 1) / (
        df_combined["other_corpus_proportion"] + 1
    )

    # Calculate normalized relative proportions (z-score)
    relative_diff = (df_combined["relative_proportion"] - df_combined["relative_proportion"].mean()) / df_combined[
        "relative_proportion"
    ].std()
    # Multiply by 100 to get percentage
    df_combined["normalized_relative_proportion"] = relative_diff * 100

    top_relative_proportions = [
        {"label": word, "count": value}
        for word, value in df_combined[df_combined["normalized_relative_proportion"] > 0][
            "normalized_relative_proportion"
        ]
        .sort_values(ascending=False)
        .head(100)
        .items()
    ]

    bottom_relative_proportions = [
        {"label": word, "count": value}
        for word, value in df_combined[df_combined["normalized_relative_proportion"] < 0][
            "normalized_relative_proportion"
        ]
        .sort_values()
        .head(100)
        .items()
    ]

    return top_relative_proportions, bottom_relative_proportions


if __name__ == "__main__":
    CGIHandler().run(get_collocation_relative_proportions)
