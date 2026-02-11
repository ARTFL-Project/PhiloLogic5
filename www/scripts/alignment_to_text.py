#!/var/lib/philologic5/philologic_env/bin/python3

import os
from json import dumps
from wsgiref.handlers import CGIHandler

from philologic.runtime.DB import DB
from philologic.runtime.link import byte_range_to_link
from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def alignment_to_text(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    db = DB(config.db_path + "/data/")
    request = _WSGIHandler(environ, config)
    link = byte_range_to_link(db, config, request)
    yield dumps({"link": link}).encode("utf-8")


if __name__ == "__main__":
    CGIHandler().run(alignment_to_text)
