#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime import collocation_results, WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def collocation(environ, start_response):
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("reports", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    _collocation_results = get_custom(db_path, "collocation_results", collocation_results)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    if environ["REQUEST_METHOD"] == "OPTIONS":
        # Handle preflight request
        start_response(
            "200 OK",
            [
                ("Content-Type", "text/plain"),
                ("Access-Control-Allow-Origin", environ["HTTP_ORIGIN"]),  # Replace with your client domain
                ("Access-Control-Allow-Methods", "POST, OPTIONS"),
                ("Access-Control-Allow-Headers", "Content-Type"),  # Adjust if needed for your headers
            ],
        )
        return [b""]  # Empty response body for OPTIONS
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response("200 OK", headers)
    post_data = environ["wsgi.input"].read()
    current_collocates = orjson.loads(post_data)["current_collocates"]
    collocation_object = _collocation_results(request, config, current_collocates)
    yield orjson.dumps(collocation_object)


if __name__ == "__main__":
    CGIHandler().run(collocation)
