#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import access_control, login_access

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


default_reports = ["concordance", "kwic", "collocation", "time_series", "navigation"]


def access_request(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    request = WSGIHandler(environ, config)
    access, headers = login_access(environ, request, config, headers)
    start_response(status, headers)
    if access:
        yield orjson.dumps({"access": True})
    else:
        incoming_address, domain_name = access_control.get_client_info(environ)
        yield orjson.dumps({"access": False, "incoming_address": incoming_address, "domain_name": domain_name})


if __name__ == "__main__":
    CGIHandler().run(access_request)
