import os
import orjson
from philologic.runtime import access_control, login_access
from philologic.runtime import WebConfig, WSGIHandler

from wsgi_helpers import resolve


default_reports = ["concordance", "kwic", "collocation", "time_series", "navigation"]


def access_request(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = resolve(db_path, "WebConfig", WebConfig)
    _WSGIHandler = resolve(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    access, headers = login_access(environ, request, config, headers)
    start_response(status, headers)
    if access:
        yield orjson.dumps({"access": True})
    else:
        incoming_address, domain_name = access_control.get_client_info(environ)
        yield orjson.dumps({"access": False, "incoming_address": incoming_address, "domain_name": domain_name})

