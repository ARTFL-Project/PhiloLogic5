#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime.reports.collocation import safe_pickle_load
from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def get_collocate_distribution(environ, start_response):
    """Get collocate distribution"""
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

    collocations_per_field = safe_pickle_load(request.file_path)
    collocates = sorted(collocations_per_field[request.field].items(), key=lambda x: x[1], reverse=True)

    yield orjson.dumps({"collocates": collocates})


if __name__ == "__main__":
    CGIHandler().run(get_collocate_distribution)
