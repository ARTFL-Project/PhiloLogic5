#!/usr/bin/env python3

"""Get similar collocate distributions"""

import os
import pickle
from wsgiref.handlers import CGIHandler
import sys

from sklearn.metrics.pairwise import cosine_similarity
import orjson
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
    author_word_matrix = create_word_matrix(collocations_per_field, reference_collocates)

    first_row = author_word_matrix[0].reshape(1, -1)
    similarities = cosine_similarity(first_row, author_word_matrix[1:])[0]
    similarity_series = pd.Series(similarities, index=collocations_per_field.keys()).astype(float).round(3)
    similarity_series.sort_values(ascending=False, inplace=True)
    most_similar_distributions = [(k, v) for k, v in similarity_series.items() if v < 1.0][:50]
    most_dissimilar_distributions = [(k, v) for k, v in similarity_series.iloc[::-1].items() if v < 1.0][:50]
    yield orjson.dumps(
        {
            "most_similar_distributions": most_similar_distributions,
            "most_dissimilar_distributions": most_dissimilar_distributions,
        }
    )


def create_word_matrix(collocations_per_field, reference_collocates):
    """Creates a NumPy matrix from a dictionary of field names associated with word counts."""
    words = set([w for f in collocations_per_field for w in collocations_per_field[f]])
    words.update(reference_collocates.keys())
    word_to_index = {word: i for i, word in enumerate(words)}
    author_word_matrix = np.zeros((len(collocations_per_field) + 1, len(words)))

    for word, count in reference_collocates.items():
        index = word_to_index[word]
        author_word_matrix[0, index] = count

    for i, author in enumerate(collocations_per_field, start=1):
        for word, count in collocations_per_field[author].items():
            index = word_to_index[word]
            author_word_matrix[i, index] = count

    return author_word_matrix


if __name__ == "__main__":
    CGIHandler().run(get_similar_collocate_distributions)
