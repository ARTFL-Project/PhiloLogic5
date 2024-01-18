#!/usr/bin/env python3
"""Collocation results"""

import os
import timeit
import struct
from typing import Any

import msgpack
import lmdb
from philologic.runtime.DB import DB
from philologic.runtime.Query import get_expanded_query


def collocation_results(request, config):
    """Fetch collocation results"""
    db = DB(config.db_path + "/data/")
    collocation_object: dict[str, Any] = {"query": dict([i for i in request])}

    # We turn on lemma counting if the query word is a lemma search
    if "lemma:" in request["q"]:
        count_lemmas = True
    else:
        count_lemmas = False

    try:
        collocate_distance = int(request["arg_proxy"])
    except ValueError:  # Getting an empty string since the keyword is not specificed in the URL
        collocate_distance = None

    if request.colloc_filter_choice == "nofilter":
        filter_list = []
    else:
        filter_list = build_filter_list(request, config, count_lemmas)
    collocation_object["filter_list"] = filter_list
    filter_list = set(filter_list)

    hits = db.query(
        request["q"],
        "proxy",
        request["arg"],
        raw_results=True,
        raw_bytes=True,
        **request.metadata,
    )

    # Build list of search terms to filter out
    query_words = []
    for group in get_expanded_query(hits):
        for word in group:
            word = word.replace('"', "")
            query_words.append(word)
    query_words = set(query_words)
    filter_list = filter_list.union(query_words)

    hits_done = request.start or 0
    max_time = request.max_time or 2
    all_collocates = {}
    start_time = timeit.default_timer()

    env = lmdb.open(
        os.path.join(db.path, "sentences.lmdb"),
        readonly=True,
        lock=False,
    )

    with env.begin() as txn:
        cursor = txn.cursor()
        for hit in hits[hits_done:]:
            parent_sentence = hit[:24]  # 24 bytes for the first 6 integers
            q_word_position = struct.unpack("1I", hit[24:28])  # 4 bytes for the 7th integer
            sentence = cursor.get(parent_sentence)
            word_objects = msgpack.loads(sentence)
            if count_lemmas is False:
                words = [(word, position) for word, _, _, position in word_objects]
            else:
                words = [(lemma, position) for _, lemma, _, position in word_objects]
            for collocate, position in words:
                if collocate not in filter_list:
                    if collocate_distance is None:
                        if collocate not in all_collocates:
                            all_collocates[collocate] = {"count": 1}
                        else:
                            all_collocates[collocate]["count"] += 1
                    else:
                        if abs(position - q_word_position[0]) <= collocate_distance:
                            if collocate not in all_collocates:
                                all_collocates[collocate] = {"count": 1}
                            else:
                                all_collocates[collocate]["count"] += 1
            hits_done += 1

            elapsed = timeit.default_timer() - start_time
            # split the query if more than request.max_time has been spent in the loop
            if elapsed > int(max_time):
                break
    env.close()
    hits.finish()

    collocation_object["collocates"] = all_collocates
    collocation_object["results_length"] = len(hits)
    if hits_done < collocation_object["results_length"]:
        collocation_object["more_results"] = True
        collocation_object["hits_done"] = hits_done
    else:
        collocation_object["more_results"] = False
        collocation_object["hits_done"] = collocation_object["results_length"]
    collocation_object["distance"] = collocate_distance

    return collocation_object


def build_filter_list(request, config, count_lemmas):
    """set up filtering with stopwords or most frequent terms."""
    if config.stopwords and request.colloc_filter_choice == "stopwords":
        if config.stopwords and "/" not in config.stopwords:
            filter_file = os.path.join(config.db_path, "data", config.stopwords)
        elif os.path.isabs(config.stopwords):
            filter_file = config.stopwords
        else:
            return ["stopwords list not found"]
        if not os.path.exists(filter_file):
            return ["stopwords list not found"]
        filter_num = float("inf")
    else:
        filter_file = config.db_path + "/data/frequencies/word_frequencies"
        if request.filter_frequency:
            filter_num = int(request.filter_frequency)
        else:
            filter_num = 100  # default value in case it's not defined
    filter_list = [request["q"]]
    with open(filter_file, encoding="utf8") as filehandle:
        for line_count, line in enumerate(filehandle):
            if line_count == filter_num:
                break
            try:
                word = line.split()[0]
            except IndexError:
                continue
            if count_lemmas is False:
                filter_list.append(word)
            else:
                filter_list.append("lemma:" + word)
    return filter_list
