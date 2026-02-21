import hashlib
import os
import orjson
from philologic.runtime.reports.collocation import (
    atomic_pickle_dump,
    decode_group_collocates,
    load_map_field_cache,
)
from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def get_collocate_distribution(environ, start_response):
    """Get collocate distribution for a single field value from a map_field numpy cache."""
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    start_response(
        "200 OK",
        [
            ("Content-Type", "application/json"),
            ("Access-Control-Allow-Origin", "*"),
        ],
    )

    tids, counts, group_bounds, group_names, count_lemmas, attribute, attribute_value = load_map_field_cache(
        request.file_path
    )

    # Find the requested group
    group_index = group_names.index(request.field)
    field_counter = decode_group_collocates(
        tids, counts, group_bounds, group_index,
        db_path, count_lemmas, attribute, attribute_value,
    )

    collocates = sorted(field_counter.items(), key=lambda x: x[1], reverse=True)

    # Cache the field's Counter to disk for downstream use (e.g. comparative_collocations)
    h = hashlib.sha1(f"{request.file_path}:{request.field}".encode("utf-8")).hexdigest()
    field_file_path = os.path.join(db_path, "data", "hitlists", f"{h}.pickle")
    atomic_pickle_dump(field_counter, field_file_path)

    yield orjson.dumps({"collocates": collocates[:100], "file_path": field_file_path})

