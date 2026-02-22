import os

import orjson
from philologic.runtime.citations import citation_links, citations
from philologic.runtime.DB import DB

def get_academic_citation(request, config):
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
        return {"citation": citation, "link": permalink}
    else:
        return {"citation": citation, "link": ""}
