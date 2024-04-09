#!/usr/bin/env python3

"""Time series of collocations: each period is compared to the previous to get a sense of the shift between each period. """

import os
import sys
from wsgiref.handlers import CGIHandler
import pickle
import orjson

import numpy as np
import pandas as pd
import numba as nb


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


nb.config.CACHE_DIR = "/tmp/numba"


def collocation_time_series(environ, start_response):
    """Reads in a pickled file containing collocations for each year. Then groups years by range
    and then compares the difference from one period to the next."""
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
    with open(request.file_path, "rb") as f:
        collocates_per_year = pickle.load(f)
    # We create a dataframe where rows are years and columns are collocates.
    collocates_per_year_df = pd.DataFrame.from_dict(collocates_per_year, orient="index").fillna(0).astype(int)

    # We group the years by ranges
    period = int(request.year_interval)
    collocates_per_year_df["period_group"] = (collocates_per_year_df.index // period) * period
    collocates_per_period = collocates_per_year_df.groupby("period_group").sum() + 1  # + 1 for Laplace smoothing
    collocates_per_period.sort_index(inplace=True)

    # Initialize an empty array to hold the cumulative sum
    cumulative_sum = np.zeros_like(collocates_per_period.iloc[0].to_numpy())

    # We calculate the difference between each period as the cosine similarity between each period based on the z-score of the collocates
    consecutive_similarities = []
    for i in range(len(collocates_per_period) - 1):
        # Update the cumulative sum to include the current period
        cumulative_sum += collocates_per_period.iloc[i].to_numpy()
        second_period = collocates_per_period.iloc[i + 1].to_numpy()
        similarity = zscore_sim(cumulative_sum, second_period)
        consecutive_similarities.append(
            {
                "range": f"{collocates_per_period.index[i]}-{collocates_per_period.index[i + 1]}",
                "similarity": similarity,
            }
        )

    # Get the mean similarity across all periods:
    mean_similarity = np.mean([item["similarity"] for item in consecutive_similarities])

    # Adjusted similarity scores relative to the mean
    adjusted_similarities = [
        {"range": item["range"], "similarity": float(item["similarity"] - mean_similarity)}
        for item in consecutive_similarities
    ]

    yield orjson.dumps({"consecutive_similarities": adjusted_similarities})


@nb.jit(nopython=True, cache=True, fastmath=True)
def zscore_sim(first_period, second_period):
    first_period = np.log(first_period) / first_period.sum()
    second_period = np.log(second_period) / second_period.sum()

    # Get the mean values for each collocate
    combined_array = np.empty(first_period.size + second_period.size, dtype=np.float64)
    combined_array[: first_period.size] = first_period
    combined_array[first_period.size :] = second_period
    mean_count = np.mean(combined_array)
    std_count = np.std(combined_array)

    # Calculate the z-score for each period
    first_period_z = (first_period - mean_count) / std_count
    second_period_z = (second_period - mean_count) / std_count

    similarity = cosine_similarity(first_period_z, second_period_z)
    return similarity


@nb.jit(nopython=True, cache=True, fastmath=True)
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


if __name__ == "__main__":
    CGIHandler().run(collocation_time_series)
