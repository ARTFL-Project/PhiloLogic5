#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import generate_text_object
from philologic.runtime.DB import DB
from philologic.runtime.HitWrapper import ObjectWrapper

from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def get_text_object(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _generate_text_object = get_custom(db_path, "generate_text_object", generate_text_object)
    config = _WebConfig(db_path)
    db = DB(config.db_path + "/data/")
    request = _WSGIHandler(environ, config)
    path = config.db_path
    zeros = 7 - len(request.philo_id)
    if zeros:
        request.philo_id += zeros * " 0"
    obj = ObjectWrapper(request["philo_id"].split(), db)
    text_object = _generate_text_object(request, config)
    yield orjson.dumps(text_object)


if __name__ == "__main__":
    CGIHandler().run(get_text_object)
