#!/var/lib/philologic5/philologic_env/bin/python3

"""Time series of collocations: each period is compared to the previous to get a sense of the shift between each period. """

import os
import sys
from wsgiref.handlers import CGIHandler

import numpy as np
import orjson
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
from philologic.runtime.reports.collocation import safe_pickle_load


def calculate_distinctive_collocates(current_period, prev_period, next_period, collocates_per_period):
    """Calculate distinctive collocates using z-scores"""
    if prev_period is None and next_period is None:
        return []

    # Create combined neighbor period
    if prev_period is None:
        neighbor_period = next_period
    elif next_period is None:
        neighbor_period = prev_period
    else:
        # Simple sum of neighbors - zscore_diff will handle normalization
        neighbor_period = prev_period + next_period

    # Calculate distinctiveness using zscore_diff
    diff, _, _ = zscore_diff(current_period, neighbor_period)

    # Convert to series and sort
    diff_series = pd.Series(diff, index=collocates_per_period.columns)
    diff_series = diff_series[diff_series > 0].sort_values(ascending=False).round(4)

    distinctive = [(word, float(score)) for word, score in diff_series.head(100).items()]
    return distinctive


def collocation_time_series(environ, start_response):
    """Reads in a pickle file containing collocations for each year."""
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)

    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)

    collocates_per_year = safe_pickle_load(request.file_path)
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

    period_number = int(request.period_number)
    current_year = int(collocates_per_period.index[period_number])
    current_period = collocates_per_period.iloc[period_number].to_numpy()

    # Get frequent collocates for current period
    current_freq = np.log(current_period) / current_period.sum()
    frequent_collocates = pd.Series(current_freq, index=collocates_per_period.columns).sort_values(ascending=False)
    frequent_collocates = [(word, float(score)) for word, score in frequent_collocates.head(100).items()]

    # Get neighboring periods
    prev_period = collocates_per_period.iloc[period_number - 1].to_numpy() if period_number > 0 else None
    next_period = (
        collocates_per_period.iloc[period_number + 1].to_numpy()
        if period_number < len(collocates_per_period) - 1
        else None
    )

    # Calculate distinctive collocates compared to both neighbors
    distinctive = calculate_distinctive_collocates(current_period, prev_period, next_period, collocates_per_period)

    done = period_number == len(collocates_per_period) - 1

    yield orjson.dumps(
        {
            "period": {
                "year": current_year,
                "collocates": {"frequent": frequent_collocates, "distinctive": distinctive},
            },
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
