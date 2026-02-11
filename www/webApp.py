#!/var/lib/philologic5/philologic_env/bin/python3
"""Bootstrap Web app"""


import os.path

from philologic.runtime import WebConfig, WSGIHandler, access_control

PATH = os.path.abspath(os.path.dirname(__file__))


def start_web_app(environ, start_response):
    """Return index.html to start web app"""
    config = WebConfig(environ.get("PHILOLOGIC_DBPATH", os.path.abspath(PATH)))
    headers = [("Content-type", "text/html; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    if not config.valid_config:  # This means we have an error in the webconfig file
        return build_misconfig_page(config.traceback, "webconfig.cfg")
    request = WSGIHandler(environ, config)
    if config.access_control:
        if not request.authenticated:
            token = access_control.check_access(environ, config)
            if token:
                h, ts = token
                headers.append(("Set-Cookie", "hash=%s" % h))
                headers.append(("Set-Cookie", "timestamp=%s" % ts))

    # Check if the client supports Brotli
    accept_encoding = environ.get("HTTP_ACCEPT_ENCODING", "")
    brotli_supported = "br" in accept_encoding
    if brotli_supported:
        index_file = "index.html.br"
        headers.append(("Content-Encoding", "br"))
    else:
        index_file = "index.html"
    with open(f"{config.db_path}/app/dist/{index_file}", "rb") as index_page:
        html_page = index_page.read()
    start_response("200 OK", headers)
    return html_page


def build_misconfig_page(traceback, config_file):
    """Return bad config HTML page"""
    with open("%s/app/misconfiguration.html" % PATH) as input:
        html_page = input.read()
    html_page = html_page.replace("$TRACEBACK", traceback)
    html_page = html_page.replace("$config_FILE", config_file)
    return html_page.encode("utf-8")
