#!/var/lib/philologic5/philologic_env/bin/python3
"""Compare collocations between two corpora."""

from wsgiref.handlers import CGIHandler

import numpy as np
import orjson
from philologic.runtime.reports.collocation import fightin_words_zscores, safe_pickle_load


def comparative_collocations(environ, start_response):
    """Calculate relative proportion of each collocate."""
    if environ["REQUEST_METHOD"] == "OPTIONS":
        # Handle preflight request
        start_response(
            "200 OK",
            [
                ("Content-Type", "text/plain"),
                ("Access-Control-Allow-Origin", environ["HTTP_ORIGIN"]),
                ("Access-Control-Allow-Methods", "POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "Content-Type"),
            ],
        )
        return [b""]
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)

    post_data = orjson.loads(environ["wsgi.input"].read())
    all_collocates = safe_pickle_load(post_data["primary_file_path"])
    other_collocates = safe_pickle_load(post_data["other_file_path"])
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
    """Compare two Counter dicts using Fightin' Words z-scores."""
    # Collect the full vocabulary and build aligned count arrays
    words = sorted(set(all_collocates) | set(other_collocates))
    y_sub = np.array([all_collocates.get(w, 0) for w in words], dtype=np.float64)
    y_other = np.array([other_collocates.get(w, 0) for w in words], dtype=np.float64)

    # When comparing against the whole corpus, subtract the sub-corpus counts
    if whole_corpus:
        y_other = np.maximum(y_other - y_sub, 0.0)

    zscores = fightin_words_zscores(y_sub, y_other)

    # Split into over- and under-represented, sorted by magnitude
    order = np.argsort(zscores)[::-1]

    top = []
    bottom = []
    for idx in order:
        z = round(float(zscores[idx]), 4)
        if z > 0:
            top.append((words[idx], z))
        else:
            bottom.append((words[idx], abs(z)))
    bottom.reverse()

    return top[:100], bottom[:100]


if __name__ == "__main__":
    CGIHandler().run(comparative_collocations)
