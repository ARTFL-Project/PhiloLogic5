#!/usr/bin/env python3

"""Time series of collocations: each period is compared to the previous to get a sense of the shift between each period. """

import os
import sys
from wsgiref.handlers import CGIHandler
import pickle
import orjson

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


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


def collocation_time_series(environ, start_response):
    """Reads in a pickled file containing collocations for each year. Then groups years by range
    and then compares the difference from one period to the next."""
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)

    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)

    with open(request.file_path, "rb") as f:
        collocates_per_year = pickle.load(f)
    # We create a dataframe where rows are years and columns are collocates.
    vectorizer = DictVectorizer(dtype=int, sparse=True)
    arrays = vectorizer.fit_transform(collocates_per_year.values())
    collocates_per_year_df = pd.DataFrame(
        arrays.toarray(), index=collocates_per_year.keys(), columns=vectorizer.feature_names_
    )

    # We group the years by ranges
    period = int(request.year_interval)
    collocates_per_year_df["period_group"] = (collocates_per_year_df.index // period) * period
    collocates_per_period = collocates_per_year_df.groupby("period_group").sum() + 1  # + 1 for Laplace smoothing
    collocates_per_period.sort_index(inplace=True)

    # We calculate the difference between each period as the cosine similarity between each period based on the z-score of the collocates
    period_number = int(request.period_number)
    first_period = collocates_per_period.iloc[period_number].to_numpy()
    second_period = collocates_per_period.iloc[period_number + 1].to_numpy()
    first_period_diff, second_period_diff, similarity = zscore_diff(first_period, second_period)
    first_period_series = (
        pd.Series(first_period_diff, index=collocates_per_period.columns).sort_values(ascending=False).round(4)
    )
    first_period_over_representation = list(first_period_series[first_period_series > 0].head(100).items())
    second_period_series = (
        pd.Series(second_period_diff, index=collocates_per_period.columns).sort_values(ascending=False).round(4)
    )
    second_period_over_representation = list(second_period_series[second_period_series > 0].head(100).items())

    if collocates_per_period.iloc[period_number + 1].name == collocates_per_period.iloc[-1].name:
        done = True
    else:
        done = False

    yield orjson.dumps(
        {
            "first_period": {
                "year": int(collocates_per_period.index[period_number]),
                "collocates": first_period_over_representation,
            },
            "second_period": {
                "year": int(collocates_per_period.index[period_number + 1]),
                "collocates": second_period_over_representation,
            },
            "similarity": float(similarity),
            "done": done,
        }
    )


def zscore_diff(first_period, second_period):
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

    # Get the difference between the two periods
    first_period_diff = first_period_z - second_period_z
    second_period_diff = second_period_z - first_period_z

    # Get similarity between the two periods
    similarity = cosine_similarity(first_period_z, second_period_z)

    return first_period_diff, second_period_diff, similarity


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


if __name__ == "__main__":
    CGIHandler().run(collocation_time_series)
