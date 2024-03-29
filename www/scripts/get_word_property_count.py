#!/usr/bin/env python3

import os
import sys
from wsgiref.handlers import CGIHandler

import lmdb
import orjson
from philologic.runtime.DB import DB
from philologic.runtime.Query import get_word_groups


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


OBJECT_LEVEL = {"doc": 6, "div1": 5, "div2": 4, "div3": 3, "para": 2, "sent": 1}
OBJ_DICT = {"doc": 1, "div1": 2, "div2": 3, "div3": 4, "para": 5, "sent": 6, "word": 7}


def get_word_property_count(environ, start_response):
    """Get word property count"""
    status = "200 OK"
    headers = [
        ("Content-type", "application/json; charset=UTF-8"),
        ("Access-Control-Allow-Origin", "*"),
    ]
    start_response(status, headers)
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)
    db = DB(config.db_path + "/data/")

    word_property_count = []
    if request.word_property != "lemma":
        # Get all word properties from config
        possible_word_properties = config.word_attributes[request.word_property]

        for word_property in possible_word_properties:
            query = f"{request.q}:{request.word_property}:{word_property}"
            hits = db.query(
                query,
                request["method"],
                request["arg"],
                raw_results=True,
                raw_bytes=True,
                **request.metadata,
            )
            hits.finish()
            word_property_count.append({"label": word_property, "count": len(hits), "q": query})
    else:
        # Get all lemmas
        hits = db.query(
            request.q,
            request["method"],
            request["arg"],
            raw_results=True,
            raw_bytes=True,
            **request.metadata,
        )
        lemma_db_env = lmdb.open(f"{config.db_path}/data/lemma_lookup.lmdb", readonly=True, lock=False)
        lemma_count = {}
        with lemma_db_env.begin() as txn:
            for hit in hits:
                lemma = txn.get(hit)
                if lemma is not None:  # some hits may not have corresponding lemmas
                    lemma = lemma.decode("utf8")
                    if lemma in lemma_count:
                        lemma_count[lemma] += 1
                    else:
                        lemma_count[lemma] = 1
        word_property_count = [{"label": k.replace("lemma:", ""), "count": v, "q": k} for k, v in lemma_count.items()]

    word_property_count.sort(key=lambda x: x["count"], reverse=True)

    results = {"query": dict([i for i in request]), "results": word_property_count}
    yield orjson.dumps(results)


if __name__ == "__main__":
    CGIHandler().run(get_word_property_count)
