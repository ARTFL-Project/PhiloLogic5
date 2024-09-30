#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from wsgiref.handlers import CGIHandler

import lmdb
import orjson
from philologic.runtime.DB import DB

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


def query_word_property(db, query, request):
    hits = db.query(
        query,
        request["method"],
        request["arg"],
        raw_results=True,
        raw_bytes=True,
        **request.metadata,
    )
    hits.finish()
    result = {"label": query.split(":")[-1], "count": len(hits), "q": query}
    os.remove(hits.filename)  # we don't want to clog up the server with hitlist files
    os.remove(hits.filename + ".done")
    os.remove(hits.filename + ".terms")
    return result


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

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_property = {
                executor.submit(
                    query_word_property, db, f"{request.q}:{request.word_property}:{word_property}", request
                ): word_property
                for word_property in possible_word_properties
            }
            for future in as_completed(future_to_property):
                word_property = future_to_property[future]
                try:
                    if future.result()["count"] > 0:
                        word_property_count.append(future.result())
                except Exception as e:
                    print(f"Exception occurred during processing {word_property}: {e}", file=sys.stderr)
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
        lemma_db_env = lmdb.open(f"{config.db_path}/data/lemmas.lmdb", readonly=True, lock=False)
        lemma_count = {}
        total_count_per_lemma = {}
        with lemma_db_env.begin() as txn:
            for hit in hits:
                lemma = txn.get(hit)
                if lemma is not None:  # some hits may not have corresponding lemmas
                    lemma = lemma.decode("utf8")
                    if lemma in lemma_count:
                        lemma_count[lemma] += 1
                    else:
                        lemma_count[lemma] = 1
                        # local_hits = db.query(
                        #     lemma,
                        #     request["method"],
                        #     request["arg"],
                        #     raw_results=True,
                        #     raw_bytes=True,
                        #     **request.metadata,
                        # )
                        # local_hits.finish()
                        # total_count_per_lemma[lemma] = len(local_hits)
        # word_property_count = [
        #     {"label": k.replace("lemma:", ""), "count": v, "overall_count": total_count_per_lemma[k], "q": k}
        #     for k, v in lemma_count.items()
        # ]
        word_property_count = [{"label": k.replace("lemma:", ""), "count": v, "q": k} for k, v in lemma_count.items()]

    word_property_count.sort(key=lambda x: x["count"], reverse=True)

    results = {"query": dict([i for i in request]), "results": word_property_count}
    yield orjson.dumps(results)


if __name__ == "__main__":
    CGIHandler().run(get_word_property_count)
