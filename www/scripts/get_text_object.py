#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import generate_text_object
from philologic.runtime.DB import DB
from philologic.runtime.HitWrapper import ObjectWrapper

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


def get_text_object(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    db = DB(config.db_path + "/data/")
    request = WSGIHandler(environ, config)
    path = config.db_path
    zeros = 7 - len(request.philo_id)
    if zeros:
        request.philo_id += zeros * " 0"
    obj = ObjectWrapper(request["philo_id"].split(), db)
    text_object = generate_text_object(request, config)
    yield orjson.dumps(text_object)


if __name__ == "__main__":
    CGIHandler().run(get_text_object)
