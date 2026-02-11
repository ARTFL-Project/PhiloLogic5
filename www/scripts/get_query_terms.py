#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime.DB import DB
from philologic.runtime.Query import get_expanded_query

from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def get_query_terms(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    db = DB(config.db_path + "/data/")
    request = _WSGIHandler(environ, config)
    hits = db.query(request["q"], request["method"], request["arg"], **request.metadata)
    hits.finish()
    expanded_terms = get_expanded_query(hits)
    yield orjson.dumps(expanded_terms[0])


if __name__ == "__main__":
    CGIHandler().run(get_query_terms)
