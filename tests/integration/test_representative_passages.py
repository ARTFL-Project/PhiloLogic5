"""Integration tests for the representative passages of a distinctive group: the sentences of the group's hits with
signature collocates around them, each scored by the z-scores of the distinct signature collocates it has, for queries
of one word, of several, and with phrases."""

import os
import sys
import urllib.parse
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "www"))

from philologic.runtime import WebConfig, WSGIHandler  # noqa: E402
from philologic.runtime.Query import get_word_groups  # noqa: E402
from philologic.runtime.reports.collocation import (  # noqa: E402
    collocation_results,
    collocation_search_method,
    phrase_words,
)
from scripts.get_outlier_groups import get_outlier_groups  # noqa: E402
from scripts.get_representative_passages import get_representative_passages  # noqa: E402

pytestmark = pytest.mark.integration


def call(fn, db, **params):
    root = os.path.dirname(os.path.normpath(db.path))
    config = WebConfig(root)
    return fn(WSGIHandler({"QUERY_STRING": urllib.parse.urlencode(params), "PHILOLOGIC_DBPATH": root}, config), config)


def vocab_ids(db, words):
    """The ids of words in the word vocabulary"""
    col = os.path.join(db.path, "collocations")
    offsets = np.load(os.path.join(col, "vocab_offsets.npy"))
    with open(os.path.join(col, "vocab_strings.bin"), "rb") as f:
        blob = f.read()
    names = {blob[offsets[i] : offsets[i + 1]].decode("utf-8"): i for i in range(len(offsets) - 1)}
    return {word: names[word] for word in words if word in names}


def expected_scores(db, q, distance, field, value, weights):
    """The score of each sentence of the group's hits of q (the collocation report's search) with a signature word
    (weights: its z-score by word) at no position of a query word, within distance words of the hit's query words
    (a phrase being one word), read from the collocation arrays"""
    col = os.path.join(db.path, "collocations")
    sent_keys = np.load(os.path.join(col, "sent_keys_s24.npy"))
    sent_offsets = np.load(os.path.join(col, "sent_offsets.npy"))
    token_ids = np.load(os.path.join(col, "token_ids.npy"))
    method, method_arg = collocation_search_method(q, db.locals.query_patterns, distance)
    hits = db.query(q, method, method_arg, **{field: value})
    hits.finish()
    main = set(vocab_ids(db, {word for group in get_word_groups(f"{hits.filename}.terms") for word in group}).values())
    weight_of = {tid: weights[word] for word, tid in vocab_ids(db, weights).items()}
    max_dist = None if distance is None else distance + phrase_words(q, db.locals.query_patterns)
    found = {}
    for row in hits.read_array():
        key = np.ascontiguousarray(row[:6].astype(">u4")).view("S24")[0]
        sent = int(np.searchsorted(sent_keys, key))
        positions = row[7::2]
        first, last = int(positions.min()), int(positions.max())
        words = found.setdefault(key, set())
        for j in range(int(sent_offsets[sent]), int(sent_offsets[sent + 1])):
            position = j - int(sent_offsets[sent]) + 1
            if max_dist is not None and max(last, position) - min(first, position) > max_dist:
                continue
            tid = int(token_ids[j])
            if tid not in main and tid in weight_of:
                words.add(tid)
    return sorted((sum(weight_of[tid] for tid in words) for words in found.values() if words), reverse=True)


@pytest.mark.parametrize("q", ["lord", "my lord", '"my lord"', '"my lord" king'])
@pytest.mark.parametrize("distance", [None, 3])
def test_passages_of_each_hit_sentence(shakespeare_db, q, distance):
    """Every sentence with signature collocates is a passage, scored by their z-scores, whatever the number of words
    of the query: the hit rows of multi-word queries have a word position and byte offset for each of their words"""
    params = {"q": q, "colloc_filter_choice": "nofilter", "report": "collocation"}
    if distance is None:
        params.update(method="sentence", colloc_within="sent")
    else:
        params.update(method="proxy", colloc_within="n", method_arg=str(distance))
    by_title = call(collocation_results, shakespeare_db, map_field="title", **params)
    outliers = call(get_outlier_groups, shakespeare_db, file_path=by_title["file_path"], min_hits=1)["outliers"]
    assert outliers
    title, _, words, z_scores = outliers[0]
    assert len(words) == len(z_scores) and z_scores == sorted(z_scores, reverse=True)
    value = f'"{title}"'

    scores, offset = [], 0
    while True:
        response = call(
            get_representative_passages,
            shakespeare_db,
            restrict_to_field="title",
            restrict_to_value=value,
            signature_tokens=",".join(words),
            signature_weights=",".join(map(str, z_scores)),
            top_n=100,
            offset=offset,
            **params,
        )
        scores += [passage["score"] for passage in response["passages"]]
        if not response["has_more"]:
            break
        offset += 100

    expected = expected_scores(shakespeare_db, q, distance, "title", value, dict(zip(words, z_scores)))
    assert expected
    assert scores == pytest.approx(expected, abs=1e-3)
