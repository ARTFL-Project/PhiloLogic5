#!/var/lib/philologic5/philologic_env/bin/python3

import os
import sys
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime.DB import DB
from philologic.runtime.Query import split_terms
from philologic.runtime.QuerySyntax import group_terms, parse_query

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


def term_group(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    config = WebConfig(os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    db = DB(config.db_path + "/data/")
    request = WSGIHandler(environ, config)
    if not request["q"]:
        dump = orjson.dumps({"original_query": "", "term_groups": []})
    else:
        parsed = parse_query(request.q)
        group = group_terms(parsed)
        all_groups = split_terms(group)
        term_groups = []
        for g in all_groups:
            term_group = ""
            not_started = False
            for kind, term in g:
                if kind == "NOT":
                    if not_started is False:
                        not_started = True
                        term_group += " NOT "
                elif kind == "OR":
                    term_group += "|"
                elif kind in ("TERM", "QUOTE", "LEMMA", "ATTR", "LEMMA_ATTR"):
                    term_group += f" {term} "
            term_group = term_group.strip()
            term_groups.append(term_group)
        dump = orjson.dumps({"term_groups": term_groups, "original_query": request.original_q})
    yield dump


if __name__ == "__main__":
    CGIHandler().run(term_group)
