#!/var/lib/philologic5/philologic_env/bin/python3
"""Compare collocations between two corpora."""

from wsgiref.handlers import CGIHandler

import numpy as np
import orjson
import pandas as pd
from scipy import stats


def comparative_collocations(environ, start_response):
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
        df_combined.loc[df_combined["other_corpus_count"] == 0, "other_corpus_count"] = 1  # Avoid division by zero

    # Median Absolute Deviation filtering
    df_combined["total_freq"] = df_combined["sub_corpus_count"] + df_combined["other_corpus_count"]  # Total frequency
    median = np.median(df_combined["total_freq"])  # Median frequency
    mad = np.median(np.abs(df_combined["total_freq"] - median))  # Median Absolute Deviation
    lower_bound = max(1, median * mad)  # ensure lower bound is at least 1
    df_combined = df_combined[df_combined["total_freq"] > lower_bound]  # Filter out low frequency collocates

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
    CGIHandler().run(comparative_collocations)
