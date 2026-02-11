#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import access_control, login_access
from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


default_reports = ["concordance", "kwic", "collocation", "time_series", "navigation"]


def access_request(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    access, headers = login_access(environ, request, config, headers)
    start_response(status, headers)
    if access:
        yield orjson.dumps({"access": True})
    else:
        incoming_address, domain_name = access_control.get_client_info(environ)
        yield orjson.dumps({"access": False, "incoming_address": incoming_address, "domain_name": domain_name})


if __name__ == "__main__":
    CGIHandler().run(access_request)
