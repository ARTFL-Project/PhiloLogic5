#!/usr/bin/env python3
"""Calculate mutual information between two words."""

import math
import os
import pickle
import sys
from wsgiref.handlers import CGIHandler
import orjson
from philologic.runtime.DB import DB
from philologic.runtime.Query import get_expanded_query

import numpy as np
import lmdb
from sklearn.preprocessing import MinMaxScaler


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
    lemma = False
    attrib = None
    attrib_value = None
    if ":" not in request.q:
        total_corpus_words = db.get_total_word_count("word")
    elif "lemma:" in request.q:
        if request.q.count(":") == 1:
            total_corpus_words = db.get_total_word_count("lemma")
            lemma = True
        else:
            attrib = request.q.split(":")[2]
            attrib_value = request.q.split(":")[3]
            total_corpus_words = db.get_total_word_count(f"lemma_{attrib}_{attrib_value}")
    else:
        attrib = request.q.split(":")[1]
        attrib_value = request.q.split(":")[2]
        total_corpus_words = db.get_total_word_count(f"word_{attrib}_{attrib_value}")
    total_words_in_window = sum(v["count"] for v in all_collocates.values())
    total_corpus_words -= total_words_in_window
    relative_proportions = []
    for collocate, value in all_collocates.items():
        collocate_count = value["count"]
        collocate_count_in_corpus = (
            get_number_of_occurrences(collocate, config.db_path, lemma=lemma, attrib=attrib, attrib_value=attrib_value)
            - collocate_count
        )
        sub_corpus_proportion = collocate_count / total_words_in_window
        corpus_proportion = collocate_count_in_corpus / total_corpus_words
        relative_proportions.append(
            {"collocate": collocate, "count": (sub_corpus_proportion + 1) / (corpus_proportion + 1)}
        )
    relative_proportions = sorted(relative_proportions, key=lambda x: x["count"], reverse=True)[:100]

    proportions_array = np.array([x["count"] for x in relative_proportions]).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_scores = scaler.fit_transform(proportions_array)
    for i, score in enumerate(scaled_scores):
        relative_proportions[i]["count"] = float(score[0])

    yield orjson.dumps(relative_proportions)


def get_number_of_occurrences(word, db_path, lemma=False, attrib=None, attrib_value=None):
    """Get the number of occurrences of a word in the corpus."""
    if lemma:
        word = f"lemma:{word}".lower()
    if attrib is not None:
        word = f"{word}:{attrib}:{attrib_value}"
    env = lmdb.open(f"{db_path}/data/words.lmdb", readonly=True, lock=False, readahead=False)
    with env.begin(buffers=True) as txn:
        occurrences = txn.get(word.encode("utf-8"))
        if occurrences is None and lemma is True:  # no lemma form in index
            occurrences = txn.get(word[6:].encode("utf-8"))
    return len(occurrences) / 36  # 36 bytes per occurrence


if __name__ == "__main__":
    CGIHandler().run(get_collocation_relative_proportions)
