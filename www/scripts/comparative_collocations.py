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

    post_data = orjson.loads(environ["wsgi.input"].read())
    all_collocates = post_data["all_collocates"]
    other_collocates = post_data["other_collocates"]
    whole_corpus = post_data["whole_corpus"]

    top_relative_proportions, low_relative_proportions = get_relative_proportions(
        all_collocates, other_collocates, whole_corpus
    )

    yield orjson.dumps(
        {
            "top": top_relative_proportions,
            "bottom": low_relative_proportions,
        }
    )


def get_relative_proportions(all_collocates, other_collocates, whole_corpus):
    # Create DataFrames
    df_sub = pd.DataFrame.from_dict(dict(all_collocates), orient="index", columns=["sub_corpus_count"])
    df_other = pd.DataFrame.from_dict(dict(other_collocates), orient="index", columns=["other_corpus_count"])

    # Outer Join (Preserves all collocates)
    df_combined = df_sub.join(df_other, how="outer").infer_objects(copy=False).fillna(0)
    # Adjust counts if comparing against the whole corpus
    if whole_corpus:
        df_combined["other_corpus_count"] = df_combined["other_corpus_count"] - df_combined["sub_corpus_count"]
        df_combined.loc[df_combined["other_corpus_count"] == 0, "other_corpus_count"] = 1

    # Calculate Proportions
    df_combined["sub_corpus_proportion"] = (
        np.log(df_combined["sub_corpus_count"] + 1) / df_combined["sub_corpus_count"].sum()
    )
    df_combined["other_corpus_proportion"] = (
        np.log(df_combined["other_corpus_count"] + 1) / df_combined["other_corpus_count"].sum()
    )

    # Calculate Z-scores for each corpus
    all_proportions = pd.concat([df_combined["sub_corpus_proportion"], df_combined["other_corpus_proportion"]])
    mean_proportion = all_proportions.mean()
    std_proportion = all_proportions.std()
    df_combined["sub_corpus_zscore"] = (df_combined["sub_corpus_proportion"] - mean_proportion) / std_proportion
    df_combined["other_corpus_zscore"] = (df_combined["other_corpus_proportion"] - mean_proportion) / std_proportion

    # Over-representation score
    df_combined["over_representation_score"] = df_combined["sub_corpus_zscore"] - df_combined["other_corpus_zscore"]

    top_relative_proportions = [
        (word, value)
        for word, value in df_combined[df_combined["over_representation_score"] > 0]["over_representation_score"]
        .sort_values(ascending=False)
        .head(100)
        .items()
    ]

    bottom_relative_proportions = [
        (word, abs(value))
        for word, value in df_combined[df_combined["over_representation_score"] < 0]["over_representation_score"]
        .sort_values()
        .head(100)
        .items()
    ]

    return top_relative_proportions, bottom_relative_proportions


if __name__ == "__main__":
    CGIHandler().run(get_collocation_relative_proportions)
