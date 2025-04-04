#!/var/lib/philologic5/philologic_env/bin/python3

import hashlib
import os
import struct
import sys
import timeit
from wsgiref.handlers import CGIHandler

import lmdb
import msgspec
import orjson
import regex as re
from philologic.runtime.DB import DB
from unidecode import unidecode

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

NUMBER = re.compile(r"\d")


def get_neighboring_words(environ, start_response):
    """Get words in the same sentence as query"""
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    db = DB(config.db_path + "/data/")
    request = WSGIHandler(environ, config)
    try:
        index = int(request.hits_done)
    except:
        index = 0
    max_time = int(request.max_time)
    start_time = timeit.default_timer()

    fields = ["index", "left", "right", "q"]
    metadata_search = False
    if (
        request.first_kwic_sorting_option in ("left", "right", "q", "")
        and request.second_kwic_sorting_option in ("left", "right", "q", "")
        and request.third_kwic_sorting_option in ("left", "right", "q", "")
    ):  # fast path
        hits = db.query(
            request["q"], request["method"], request["arg"], raw_results=True, raw_bytes=True, **request.metadata
        )
    else:
        metadata_search = True
        hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
        fields.extend(config.kwic_metadata_sorting_fields)
    cache_path = get_cache_path(request, db)
    if os.path.exists(f"{cache_path}.sorted"):
        yield orjson.dumps({"hits_done": len(hits), "cache_path": cache_path})
    else:
        if not os.path.exists(cache_path):
            with open(cache_path, "w") as cache_file:
                print("\t".join(fields), file=cache_file)

        env = lmdb.open(
            os.path.join(db.path, "sentences.lmdb"),
            readonly=True,
            lock=False,
        )

        Word = msgspec.defstruct(
            "Word", [("token", str), ("position", int)] + [(k, str) for k in db.locals.word_attributes], array_like=True
        )
        Sentence = msgspec.defstruct("Sentence", [("words", list[Word])], array_like=True)
        decoder = msgspec.msgpack.Decoder(type=Sentence)

        with open(cache_path, "a") as cache_file, env.begin() as txn:
            cursor = txn.cursor()
            for hit in hits[index:]:
                if metadata_search is False:
                    remaining = hit[28:]  # 28 bytes for the first 7 integers
                    positions = []
                    while remaining:
                        # First integer in the pair is the word position
                        position = int.from_bytes(remaining[:4], "little", signed=False)
                        positions.append(position)  # Store the position
                        remaining = remaining[4:]

                        # Skip the second integer (the offset)
                        if remaining:
                            remaining = remaining[4:]

                    positions.sort()
                    sentence = hit[:24]  # 24 bytes for the first 6 integers
                else:
                    positions = [word.philo_id[-1] for word in hit.words]
                    sentence = struct.pack("6I", *hit.hit[:6])

                words = decoder.decode(cursor.get(sentence)).words
                left_side_text = []
                right_side_text = []
                query_words = []

                for word_obj in words:
                    word = word_obj.token
                    position = word_obj.position

                    if NUMBER.search(word):
                        continue
                    if db.locals.ascii_conversion is True:
                        word = unidecode(word)

                    # Compare using position
                    if positions[0] > position:
                        left_side_text.append(word)
                    elif position > positions[-1]:
                        right_side_text.append(word)
                    else:
                        query_words.append(word)
                left_side_text = left_side_text[-10:]
                left_side_text.reverse()
                if not left_side_text:
                    left_side_text = ["zzzzzzz" for _ in range(10)]  # make sure we sort last if no words before
                if not right_side_text:
                    right_side_text = ["zzzzzzz" for _ in range(10)]  # make sure we sort last if no words after
                result_obj = {
                    "right": ",".join(right_side_text[:10]),
                    "left": ",".join(left_side_text),
                    "q": ",".join(query_words),
                    "index": index,
                }
                if metadata_search is True:
                    for metadata in config.kwic_metadata_sorting_fields:
                        result_obj[metadata] = ",".join(f"{hit[metadata]}".lower().split())
                print("\t".join(str(result_obj[field]) for field in fields), file=cache_file)
                index += 1
                elapsed = timeit.default_timer() - start_time
                if (
                    elapsed > max_time
                ):  # avoid timeouts by splitting the query if more than 5 seconds has been spent in the loop
                    break
        yield orjson.dumps({"hits_done": index, "cache_path": cache_path})


def get_cache_path(request, db):
    """Retrieve the file path for the cache"""
    hash = hashlib.sha1()
    hash.update(request["q"].encode("utf-8"))
    hash.update(request["method"].encode("utf-8"))
    hash.update(str(request["arg"]).encode("utf-8"))
    hash.update(request.first_kwic_sorting_option.encode("utf-8"))
    hash.update(request.second_kwic_sorting_option.encode("utf-8"))
    hash.update(request.third_kwic_sorting_option.encode("utf-8"))
    for field, metadata in sorted(request.metadata.items(), key=lambda x: x[0]):
        hash.update(f"{field}: {metadata}".encode("utf-8"))
    cache_path = os.path.join(db.path, "hitlists", f"{hash.hexdigest()}.kwic")
    return cache_path


if __name__ == "__main__":
    CGIHandler().run(get_neighboring_words)
