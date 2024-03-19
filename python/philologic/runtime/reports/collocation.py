#!/usr/bin/env python3
"""Collocation results"""

import time
import os
import timeit
import struct
from typing import Any

import msgpack
import lmdb
from philologic.runtime.DB import DB
from philologic.runtime.Query import get_word_groups
from orjson import dumps


def collocation_results(request, config, current_collocates):
    """Fetch collocation results"""
    collocation_object: dict[str, Any] = {"query": dict([i for i in request])}
    db = DB(config.db_path + "/data/")

    hits = db.query(
        request["q"],
        "proxy",
        request["arg"],
        raw_results=True,
        raw_bytes=True,
        **request.metadata,
    )

    try:
        collocate_distance = int(request["arg_proxy"])
    except ValueError:  # Getting an empty string since the keyword is not specificed in the URL
        collocate_distance = None

    # We turn on lemma counting if the query word is a lemma search
    if "lemma:" in request["q"]:
        count_lemmas = True
    else:
        count_lemmas = False

    if '"' in request["q"]:
        exact_match = True
    else:
        exact_match = False

    # Attribute filtering
    if request.colloc_filter_choice == "attribute":
        attribute = request.q_attribute
        attribute_value = request.q_attribute_value
    else:
        attribute = None
        attribute_value = None

    # Build list of search terms to filter out
    query_words = []
    while not os.path.exists(f"{hits.filename}.terms"):
        time.sleep(0.1)
    for group in get_word_groups(f"{hits.filename}.terms"):
        for word in group:
            title_word = word.title()
            upper_word = word.upper()
            query_words.extend([word, title_word, upper_word])

    if request.colloc_filter_choice == "nofilter":
        filter_list = set(query_words)
    elif request.colloc_filter_choice == "attribute":
        if f"{attribute}:{attribute_value}" not in request.q:
            filter_list = {f"{word}:{attribute}:{attribute_value}" for word in query_words}
        else:
            filter_list = set(query_words)
            filter_list = filter_list.union(set(query_words))
        filter_list.add(f"{request['q']}:{attribute}:{attribute_value}")
    else:
        filter_list = set(build_filter_list(request, config, count_lemmas))
        filter_list = filter_list.union(set(query_words))
    collocation_object["filter_list"] = list(filter_list)

    hits_done = request.start or 0
    if request.max_time is None:
        max_time = None
    else:
        max_time = request.max_time or 2

    if current_collocates:
        all_collocates = dict(current_collocates)
    else:
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
            if collocate_distance is not None:
                q_word_position = struct.unpack("1I", hit[28:32])  # 4 bytes for the 8th integer
            sentence = cursor.get(parent_sentence)
            word_objects = msgpack.loads(sentence)

            # If not attribute filter set, we just get the words/lemmas
            if attribute is None:
                if count_lemmas is False:
                    words = [(word, position) for word, _, position, _ in word_objects if word not in filter_list]
                else:
                    words = [
                        (attr.get("lemma"), position)
                        for word, _, position, attr in word_objects
                        if attr.get("lemma") not in filter_list
                    ]

            # If attribute filter is set, we get the words/lemmas that match the filter
            else:
                if count_lemmas is False:
                    words = [
                        (f"{word.lower()}:{attribute}:{attribute_value}", position)
                        for word, _, position, attr in word_objects
                        if attr.get(attribute) == attribute_value
                    ]
                else:
                    words = [
                        (f"{attr['lemma']}:{attribute}:{attribute_value}", position)
                        for _, _, position, attr in word_objects
                        if attr.get(attribute) == attribute_value
                    ]

            for collocate, position in words:
                if collocate is not None:  # in the event lemma is None
                    if collocate_distance is None:
                        if collocate not in all_collocates:
                            all_collocates[collocate] = 1
                        else:
                            all_collocates[collocate] += 1
                    else:
                        if abs(position - q_word_position[0]) <= collocate_distance:  # type: ignore
                            if collocate not in all_collocates:
                                all_collocates[collocate] = 1
                            else:
                                all_collocates[collocate] += 1

            hits_done += 1
            elapsed = timeit.default_timer() - start_time
            # split the query if more than request.max_time has been spent in the loop
            if max_time is not None:
                if elapsed > int(max_time):
                    break
    env.close()
    hits.finish()

    all_collocates = sorted(all_collocates.items(), key=lambda item: item[1], reverse=True)
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
    elif count_lemmas is True:
        filter_file = config.db_path + "/data/frequencies/lemmas"
        if request.filter_frequency:
            filter_num = int(request.filter_frequency)
        else:
            filter_num = 100
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
            if count_lemmas is True:
                word = word.replace("lemma:", "")
            filter_list.append(word)
    filter_list.append(request["q"].replace("lemma:", ""))
    return filter_list
