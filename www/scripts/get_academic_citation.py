#!/var/lib/philologic5/philologic_env/bin/python3

import os
from wsgiref.handlers import CGIHandler

import orjson
from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB
from philologic.runtime import WebConfig, WSGIHandler

from custom_functions_loader import get_custom


def get_academic_citation(environ, start_response):
    status = "200 OK"
    headers = [("Content-type", "application/json; charset=UTF-8"), ("Access-Control-Allow-Origin", "*")]
    start_response(status, headers)
    db_path = environ.get("PHILOLOGIC_DBPATH", os.path.abspath(os.path.dirname(__file__)).replace("scripts", ""))
    _WebConfig = get_custom(db_path, "WebConfig", WebConfig)
    _WSGIHandler = get_custom(db_path, "WSGIHandler", WSGIHandler)
    config = _WebConfig(db_path)
    request = _WSGIHandler(environ, config)
    db = DB(config.db_path + "/data/")
    text_obj = db[request.philo_id]
    citation_hrefs = citation_links(db, config, text_obj)
    citation = citations(text_obj, citation_hrefs, config, citation_type=config.academic_citation["citation"])
    if os.path.exists(os.path.join(config.db_path, "filenames_to_permalinks.json")):
        with open(os.path.join(config.db_path, "filenames_to_permalinks.json"), encoding="utf8") as f:
            filenames_to_permalinks = orjson.loads(f.read())
        permalink = filenames_to_permalinks[text_obj.filename]

        def update_link(field_citation):
            if field_citation["object_type"] == "doc":
                field_citation["href"] = ""
            return field_citation

        citation = [update_link(field_citation) for field_citation in citation]
        yield orjson.dumps({"citation": citation, "link": permalink})
    else:
        yield orjson.dumps({"citation": citation, "link": ""})


if __name__ == "__main__":
    CGIHandler().run(get_academic_citation)
